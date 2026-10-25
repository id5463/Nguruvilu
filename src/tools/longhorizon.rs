//! Missions: work that outlives a context window.
//!
//! A conversation is a working set; a mission is state on disk. This module is
//! the second one, so a task measured in days does not have to fit in a window
//! measured in tokens:
//!
//! * **State lives in the session workspace**, at `<workspace>/missions/<id>/`,
//!   which means the agent's own `read`/`write`/`edit` tools work on it, the
//!   web interface can serve it, and a workspace snapshot covers it.
//! * **The log is the memory.** `log.jsonl` is append-only and holds every
//!   advance, checkpoint, mail and tick. Nothing here keeps history in RAM: the
//!   file is the record, and `recall` searches it. That is what lets the context
//!   stay small while the mission grows without bound.
//! * **The process is the truth.** State is read from disk on every call, so a
//!   restart loses nothing and a second process sees the same mission.
//! * **`mission_tick` is one step.** It runs a fresh worker (a subagent with an
//!   empty history) on the mission's current step and writes back what it did.
//!   A long task is many short lives, not one long one.
//!
//! # Robustness rules this module keeps
//!
//! * Every write is a temp-file + rename, so a crash mid-write cannot leave a
//!   half-written `mission.json` behind.
//! * Appends are single `write_all` calls on a handle opened in append mode.
//! * Every read parses into `serde_json::Value` and reports what it could not
//!   understand instead of panicking; a missing mission is guidance, a corrupt
//!   one is a named error.
//! * Names that become path segments are sanitised, so a mailbox address can
//!   never name a directory outside the mission.
//! * Nothing blocks: every tool is a plain async function over std file calls.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use serde_json::{json, Value};

use crate::tools::{ToolDef, ToolFuture};

/// The plugin name a pack asks for with `builtin:long-horizon`.
pub const PLUGIN: &str = "long-horizon";

/// Tool names.
pub const START: &str = "mission_start";
pub const STATUS: &str = "mission_status";
pub const ADVANCE: &str = "mission_advance";
pub const CHECKPOINT: &str = "mission_checkpoint";
pub const RECALL: &str = "recall";
pub const TICK: &str = "mission_tick";
pub const POST: &str = "agent_post";
pub const INBOX: &str = "agent_inbox";

/// Most steps one tick's worker may take.
const TICK_STEPS: usize = 12;

/// How long one tick's worker may run before it is abandoned.
///
/// Step count bounds *how many* things the worker may do, never how long it
/// spends doing them: a single `bash` call can hold a step for minutes and a
/// retry loop for hours. Thirty minutes is generous for one mission step and
/// short enough that a wedged worker does not hang a whole turn.
const TICK_WALL_CLOCK_MS: u64 = 30 * 60 * 1000;

/// Floor for a caller-supplied `timeout_ms`: below this the worker could not
/// even report what it found, and any timeout would be indistinguishable from
/// a failure of the step itself.
const TICK_MIN_WALL_CLOCK_MS: u64 = 10 * 1000;

/// Ceiling for `timeout_ms` — six hours, the same order as a mission's own
/// useful lifetime.
const TICK_MAX_WALL_CLOCK_MS: u64 = 6 * 60 * 60 * 1000;

/// Most lines `recall` returns in one call.
const RECALL_LIMIT: usize = 40;

/// Bytes read from one file while searching. Longer files are searched from
/// the end: the recent past is what a long task usually needs.
const RECALL_FILE_CAP: u64 = 4 * 1024 * 1024;

/// How much of `STATE.md` a worker is handed.
const STATE_CAP: usize = 200 * 1024;

/// Log text kept per entry, so one runaway report cannot bloat the log.
const LOG_TEXT_CAP: usize = 8 * 1024;

/// The fences around the block `mission_advance` maintains inside `STATE.md`.
/// Everything outside them belongs to the model and is never rewritten.
const FENCE_START: &str = "<!-- mission:generated -->";
const FENCE_END: &str = "<!-- /mission:generated -->";

/// Register the plugin. Defining is not loading: nothing appears in a tool
/// table until a pack's assembly asks for `builtin:long-horizon`.
pub fn define(kernel: &mut crate::plugin::Kernel) {
    kernel.define(Arc::new(LongHorizon));
}

/// The plugin behind the mission tools.
pub struct LongHorizon;

impl crate::plugin::Plugin for LongHorizon {
    fn name(&self) -> &str {
        PLUGIN
    }

    /// The route publishes as a service and `mission_tick` runs a worker on it,
    /// so the tool waits for the route rather than capturing a client.
    fn inject(&self) -> Vec<String> {
        vec![crate::model::SERVICE.to_string()]
    }

    fn apply(&self, ctx: &crate::plugin::PluginCtx) -> Result<crate::plugin::Contributions> {
        let access = crate::model::ModelHandle::from_view(&ctx.services)
            .ok_or_else(|| anyhow!("the session has not published a route"))?;
        Ok(crate::plugin::Contributions::new()
            .tool(start_tool())
            .tool(status_tool())
            .tool(advance_tool())
            .tool(checkpoint_tool())
            .tool(recall_tool())
            .tool(tick_tool(access))
            .tool(post_tool())
            .tool(inbox_tool())
            .ui(panel()))
    }
}

// ---------------------------------------------------------------- the board

/// The mission bar: the task card, drawn to match the harness's own todo strip.
///
/// It repaints whenever a tool result carries a `<!--mission-state …-->`
/// marker, which is how the mission tools report themselves. That alone leaves the
/// bar idle until the next mission call, so it also listens to the page's `status`
/// event: every panel is handed the status the page already holds the moment its
/// frame loads, and that status carries the board of the missions on disk (the
/// desktop shell reads the workspace's `missions/` directories for it). Without
/// both halves a reader who starts a mission and looks at the bar sees "no mission
/// in progress".
///
/// The script is a bare document of its own sandboxed iframe: it has no access
/// to the page's bus, only to `window.ngu` from the panel SDK, so everything here
/// goes through that and through `document`, never through page globals.
///
/// The look is copied from the harness's own todo strip (its `TodoPanel`): one
/// card, and a header row that reads as a single line while collapsed — leading
/// checklist glyph, the label, a counts summary, a chevron — which opens into the
/// mission's id and status pill, its objective and current step, the board of
/// every mission on disk, and the two buttons. Each board row carries one of the
/// strip's three status glyphs (completed check, spinning fraction ring, dashed
/// unstarted ring), so a mission list and a todo list read as one component, and
/// the summary in the header is worded the same way ("1 进行中 · 2 待办 · 1 已完成",
/// en-space separated, zero-count segments dropped as noise).
///
/// It carries its own `<style>` block — the iframe inherits no CSS variables from
/// the page — so the harness's design tokens are transcribed into local `--mp-*`
/// properties, light block and dark block, switched by `prefers-color-scheme`; a
/// reader whose engine knows neither still gets the light block. It defines no
/// body height, because the panel SDK measures `body` and reports it through
/// `panel:resize`, so the content has to keep sizing the document.
fn panel() -> crate::ui::UiPanel {
    let html = r##"<style>
  /* Palette: the harness's design-platform tokens transcribed into local
     properties. The panel document cannot see `--dsw-*` (opaque origin, no host
     CSS), so every colour it needs is restated here from the same source values:
     light first, then the dark block in the media query. */
  :root {
    color-scheme: light dark;
    --mp-surface: rgb(245, 246, 247);
    --mp-border: rgba(0, 0, 0, 0.04);
    --mp-border-strong: rgba(0, 0, 0, 0.1);
    --mp-hover: rgba(38, 49, 72, 0.06);
    --mp-label-primary: rgb(15, 17, 21);
    --mp-label-secondary: rgb(97, 102, 107);
    --mp-label-tertiary: rgb(129, 133, 140);
    --mp-label-caption: rgb(173, 178, 184);
    --mp-business: rgb(65, 118, 230);
    --mp-business-soft: rgba(65, 118, 230, 0.1);
    --mp-success: rgb(34, 197, 94);
    --mp-success-soft: rgba(34, 197, 94, 0.14);
    --mp-warn: rgb(221, 134, 41);
    --mp-warn-soft: rgba(221, 134, 41, 0.14);
    --mp-scrollbar: rgb(229, 229, 229);
    --mp-mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --mp-surface: rgb(53, 54, 56);
      --mp-border: rgba(255, 255, 255, 0.06);
      --mp-border-strong: rgba(255, 255, 255, 0.12);
      --mp-hover: rgba(255, 255, 255, 0.08);
      --mp-label-primary: rgb(249, 250, 251);
      --mp-label-secondary: rgb(207, 211, 214);
      --mp-label-tertiary: rgb(173, 178, 184);
      --mp-label-caption: rgb(129, 133, 140);
      --mp-business: rgb(103, 158, 254);
      --mp-business-soft: rgba(103, 158, 254, 0.16);
      --mp-success: rgb(34, 197, 94);
      --mp-success-soft: rgba(34, 197, 94, 0.18);
      --mp-warn: rgb(247, 173, 49);
      --mp-warn-soft: rgba(247, 173, 49, 0.16);
      --mp-scrollbar: rgb(84, 85, 87);
    }
  }
  /* The card: the todo strip's surface — tip fill, hairline border, 12px radius,
     clipped so the header highlight never bleeds past the corner. No fixed
     height: the panel SDK measures this document and sizes the frame to it. */
  #mission-bar {
    box-sizing: border-box;
    width: 100%;
    overflow: hidden;
    border: 1px solid var(--mp-border);
    border-radius: 12px;
    background: var(--mp-surface);
    color: var(--mp-label-primary);
    font: 13px/1.45 system-ui, -apple-system, 'Segoe UI', sans-serif;
  }
  #mission-bar .mp-body {
    display: flex;
    flex-direction: column;
    gap: 8px;
    padding: 6px 12px;
  }
  /* The resting state: one row. Everything else lives in .mp-detail. */
  #mission-bar .mp-head {
    display: flex;
    align-items: center;
    gap: 10px;
    width: 100%;
    padding: 0;
    border: none;
    background: transparent;
    color: inherit;
    font: inherit;
    text-align: left;
    cursor: pointer;
    -webkit-tap-highlight-color: transparent;
  }
  #mission-bar .mp-head:focus-visible {
    border-radius: 6px;
    outline: 2px solid var(--mp-business);
    outline-offset: 2px;
  }
  #mission-bar .mp-lead,
  #mission-bar .mp-chevron {
    display: grid;
    flex: none;
    place-items: center;
    color: var(--mp-label-tertiary);
  }
  #mission-bar .mp-lead svg,
  #mission-bar .mp-chevron svg {
    display: block;
  }
  #mission-bar .mp-title {
    flex: none;
    color: var(--mp-label-primary);
    font-size: 13px;
    line-height: 24px;
    font-weight: 500;
  }
  /* The counts line: DSH's .progress — takes the slack, ellipsizes at the end. */
  #mission-bar .mp-progress {
    flex: 1 1 auto;
    min-width: 0;
    overflow: hidden;
    color: var(--mp-label-tertiary);
    font-size: 13px;
    line-height: 20px;
    font-weight: 400;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  /* Collapsed shows the up chevron, expanded the down one — same choice the
     todo strip makes, so the two cards flip the same way. */
  #mission-bar .mp-chevron .mp-chev-down {
    display: none;
  }
  #mission-bar .mp-head[aria-expanded="true"] .mp-chevron .mp-chev-up {
    display: none;
  }
  #mission-bar .mp-head[aria-expanded="true"] .mp-chevron .mp-chev-down {
    display: block;
  }
  #mission-bar .mp-detail {
    display: flex;
    flex-direction: column;
    gap: 8px;
    min-width: 0;
  }
  /* Explicit, because a `display` on this class would outrank the UA's
     `[hidden] { display: none }` and the panel would never collapse. */
  #mission-bar .mp-detail[hidden] {
    display: none;
  }
  #mission-bar .mp-current {
    display: flex;
    align-items: center;
    gap: 8px;
    min-width: 0;
  }
  #mission-bar .mp-id {
    min-width: 0;
    overflow: hidden;
    color: var(--mp-label-secondary);
    font-family: var(--mp-mono);
    font-size: 12px;
    line-height: 20px;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  #mission-bar .mp-pill {
    display: inline-block;
    flex: none;
    padding: 0 7px;
    border-radius: 999px;
    background: var(--mp-hover);
    color: var(--mp-label-secondary);
    font-size: 11px;
    line-height: 18px;
    white-space: nowrap;
  }
  #mission-bar .mp-pill[data-s="running"] {
    background: var(--mp-business-soft);
    color: var(--mp-business);
  }
  #mission-bar .mp-pill[data-s="done"] {
    background: var(--mp-success-soft);
    color: var(--mp-success);
  }
  #mission-bar .mp-pill[data-s="blocked"] {
    background: var(--mp-warn-soft);
    color: var(--mp-warn);
  }
  /* The loudest line of the card; clamped to two lines so one long objective
     cannot push the board and the buttons out of the frame. */
  #mission-bar .mp-objective {
    display: -webkit-box;
    -webkit-box-orient: vertical;
    -webkit-line-clamp: 2;
    margin: 0;
    overflow: hidden;
    overflow-wrap: anywhere;
    color: var(--mp-label-primary);
    font-size: 13px;
    line-height: 20px;
  }
  #mission-bar .mp-step {
    margin: 0;
    min-width: 0;
    overflow: hidden;
    color: var(--mp-label-secondary);
    font-size: 13px;
    line-height: 20px;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  /* The board: the todo strip's .list, one glyph + one line per mission. */
  #mission-bar .mp-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
    max-height: 180px;
    margin: 0;
    padding: 0;
    overflow-y: auto;
    overscroll-behavior: contain;
    list-style: none;
  }
  #mission-bar .mp-list:empty {
    display: none;
  }
  #mission-bar .mp-item {
    display: flex;
    align-items: center;
    gap: 10px;
    min-width: 0;
    color: var(--mp-label-secondary);
    font-size: 13px;
    line-height: 20px;
  }
  #mission-bar .mp-glyph {
    display: grid;
    flex: none;
    place-items: center;
    width: 16px;
    height: 16px;
  }
  #mission-bar .mp-glyph svg {
    display: block;
  }
  /* The three status glyphs, coloured the way the todo strip colours them. */
  #mission-bar .mp-glyph-completed {
    color: var(--mp-success);
  }
  #mission-bar .mp-glyph-progress {
    color: var(--mp-business);
    animation: mp-spin 1s linear infinite;
  }
  #mission-bar .mp-glyph-pending {
    color: var(--mp-label-caption);
  }
  @keyframes mp-spin {
    to {
      transform: rotate(360deg);
    }
  }
  /* One copy line per mission: ellipsized, never wrapped. */
  #mission-bar .mp-text {
    flex: 1 1 auto;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  /* The mission the card is describing reads in primary ink; the strip has no
     other marker for "this one", and a filled row would be louder than DSH. */
  #mission-bar .mp-item[data-current="1"] .mp-text {
    color: var(--mp-label-primary);
  }
  /* The empty-board line is a sentence, not a mission: it wraps instead of
     ellipsizing, so the whole hint survives a narrow panel. */
  #mission-bar .mp-empty {
    color: var(--mp-label-caption);
    overflow-wrap: anywhere;
    white-space: normal;
  }
  /* The two buttons: quiet, hairline, secondary ink — they sit under the list
     rather than competing with the header row. */
  #mission-bar .mp-actions {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
  }
  #mission-bar .mp-action {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    box-sizing: border-box;
    height: 24px;
    padding: 0 10px;
    border: 1px solid var(--mp-border-strong);
    border-radius: 6px;
    background: transparent;
    color: var(--mp-label-secondary);
    font: inherit;
    font-size: 12px;
    line-height: 20px;
    white-space: nowrap;
    cursor: pointer;
  }
  #mission-bar .mp-action:hover {
    background: var(--mp-hover);
    color: var(--mp-label-primary);
  }
  #mission-bar .mp-action:active {
    background: var(--mp-hover);
  }
  #mission-bar .mp-action:focus-visible {
    outline: 2px solid var(--mp-business);
    outline-offset: 1px;
  }
  /* Scrollbar skin, following the harness: an 8px thumb on a transparent track,
     with the standard properties only where the pseudo-elements are missing. */
  ::-webkit-scrollbar {
    width: 8px;
    height: 8px;
  }
  ::-webkit-scrollbar-track {
    background: transparent;
  }
  ::-webkit-scrollbar-thumb {
    border-radius: 4px;
    background: var(--mp-scrollbar);
  }
  @supports not selector(::-webkit-scrollbar) {
    #mission-bar * {
      scrollbar-width: thin;
      scrollbar-color: var(--mp-scrollbar) transparent;
    }
  }
</style>
<section id="mission-bar" class="mp-root" aria-label="任务">
  <div class="mp-body">
    <button class="mp-head" type="button" aria-expanded="false" aria-controls="mission-detail">
      <span class="mp-lead" aria-hidden="true">
        <svg width="14" height="14" viewBox="0 0 14 14" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M13.3277 9.69629V10.976H7.28086V9.69629H13.3277Z" fill="currentColor"/>
          <path d="M13.3277 2.97256V4.25225H7.28086V2.97256H13.3277Z" fill="currentColor"/>
          <path d="M4.64512 10.336C4.64505 9.62755 4.07081 9.05322 3.3623 9.05322C2.65386 9.05329 2.07956 9.62759 2.07949 10.336C2.07949 11.0445 2.65382 11.6188 3.3623 11.6188C4.07085 11.6188 4.64512 11.0446 4.64512 10.336ZM5.92559 10.336C5.92559 11.7515 4.77777 12.8993 3.3623 12.8993C1.94689 12.8993 0.799805 11.7515 0.799805 10.336C0.799871 8.92066 1.94693 7.7736 3.3623 7.77354C4.77773 7.77354 5.92552 8.92062 5.92559 10.336Z" fill="currentColor"/>
          <path d="M4.64531 3.6123C4.6453 2.90382 4.07098 2.32949 3.3625 2.32949C2.65403 2.32951 2.0797 2.90383 2.07969 3.6123C2.07969 4.32079 2.65402 4.8951 3.3625 4.89512C4.07099 4.89512 4.64531 4.3208 4.64531 3.6123ZM5.925 3.6123C5.925 5.02772 4.77792 6.1748 3.3625 6.1748C1.9471 6.17479 0.8 5.02771 0.8 3.6123C0.800013 2.19691 1.9471 1.04982 3.3625 1.0498C4.77791 1.0498 5.92499 2.1969 5.925 3.6123Z" fill="currentColor"/>
        </svg>
      </span>
      <span class="mp-title">任务</span>
      <span class="mp-progress" data-f="counts">还没有任务</span>
      <span class="mp-chevron" aria-hidden="true">
        <svg class="mp-chev-up" width="14" height="14" viewBox="0 0 14 14" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M2.15137 8.5L2.57617 8.07617L5.30273 5.34863C5.55843 5.09294 5.78438 4.86618 5.98828 4.70215C6.20088 4.53117 6.44405 4.38244 6.75 4.33398C6.91565 4.30778 7.08435 4.30778 7.25 4.33398C7.55595 4.38244 7.79912 4.53117 8.01172 4.70215C8.21561 4.86618 8.44157 5.09294 8.69727 5.34863L11.4238 8.07617L11.8486 8.5L11 9.34863L10.5762 8.92383L7.84863 6.19727C7.57405 5.92269 7.40124 5.75152 7.25977 5.6377C7.12709 5.53096 7.07728 5.52187 7.0625 5.51953C7.02105 5.51297 6.97895 5.51297 6.9375 5.51953C6.92272 5.52187 6.87291 5.53096 6.74023 5.6377C6.59876 5.75152 6.42595 5.92268 6.15137 6.19727L3.42383 8.92383L3 9.34863L2.15137 8.5Z" fill="currentColor"/>
        </svg>
        <svg class="mp-chev-down" width="14" height="14" viewBox="0 0 14 14" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M11.8486 5.5L11.4238 5.92383L8.69727 8.65137C8.44157 8.90706 8.21562 9.13382 8.01172 9.29785C7.79912 9.46883 7.55595 9.61756 7.25 9.66602C7.08435 9.69222 6.91565 9.69222 6.75 9.66602C6.44405 9.61756 6.20088 9.46883 5.98828 9.29785C5.78438 9.13382 5.55843 8.90706 5.30273 8.65137L2.57617 5.92383L2.15137 5.5L3 4.65137L3.42383 5.07617L6.15137 7.80273C6.42595 8.07732 6.59876 8.24849 6.74023 8.3623C6.87291 8.46904 6.92272 8.47813 6.9375 8.48047C6.97895 8.48703 7.02105 8.48703 7.0625 8.48047C7.07728 8.47813 7.12709 8.46904 7.25977 8.3623C7.40124 8.24849 7.57405 8.07732 7.84863 7.80273L10.5762 5.07617L11 4.65137L11.8486 5.5Z" fill="currentColor"/>
        </svg>
      </span>
    </button>
    <div class="mp-detail" id="mission-detail" hidden>
      <div class="mp-current">
        <span class="mp-id" data-f="id">—</span>
        <span class="mp-pill" data-f="status" data-s="idle">idle</span>
      </div>
      <p class="mp-objective" data-f="objective">还没有进行中的任务</p>
      <p class="mp-step" data-f="step">没有进行中的任务</p>
      <ul id="mission-list" class="mp-list"><li class="mp-item mp-empty">还没有任务（让模型调用 mission_start 建一个）</li></ul>
      <div class="mp-actions">
        <button class="mp-action" data-act="status" type="button">刷新</button>
        <button class="mp-action" data-act="tick" type="button">推进一步</button>
      </div>
    </div>
  </div>
</section>"##;

    let script = r##"
const bar = document.getElementById("mission-bar");
const list = document.getElementById("mission-list");
const head = bar ? bar.querySelector(".mp-head") : null;
const detail = document.getElementById("mission-detail");
const ngu = window.ngu;
// Whether a tool answer has painted the bar. The board can arrive first (the page
// feeds the cached status on load), and a board alone must not look like a report
// of the active mission.
let sawMarker = false;
// The mission id the bar is showing. The board marks that row, and paintBoard()
// runs after whichever reporter set the id, so the id is kept here rather than
// read back out of the DOM.
let shownId = "";
// The board of the missions on disk: an array once one has been reported, null
// while it is still unknown. The summary in the header is drawn from it, and the
// panel has to tell "no mission yet" apart from "nobody has said yet".
let board = null;
// The active mission's own counters, as the marker reported them. Only used
// while the board is still unknown.
let counters = "";
const EMPTY = "还没有任务";
// En spaces (U+2002) separate the counts: HTML collapses runs of ASCII spaces, so
// widening the separator's breathing room needs a literal wide space.
const SEP = "\u2002\u00b7\u2002";
function put(field, text) {
  if (!bar) return;
  const node = bar.querySelector('[data-f="' + field + '"]');
  if (node) node.textContent = text;
}
function setId(id) {
  const value = id ? String(id) : "—";
  shownId = value === "—" ? "" : value;
  put("id", value);
}
// The status is written twice on purpose: as the pill's text, and as its data-s
// variant, which is what the stylesheet colours. One writer keeps the two in step.
function setStatus(status) {
  const value = status ? String(status) : "?";
  if (!bar) return;
  const node = bar.querySelector('[data-f="status"]');
  if (!node) return;
  node.textContent = value;
  node.setAttribute("data-s", value.toLowerCase());
}
// The objective is the line a reader came for. The marker carries it; a marker
// from an older build does not, so fall back to the board row for the same id.
function objectiveOf(state) {
  if (state && typeof state.objective === "string" && state.objective.trim()) {
    return state.objective;
  }
  if (state && Array.isArray(state.missions)) {
    for (const row of state.missions) {
      if (row && row.id === state.id && typeof row.objective === "string" && row.objective.trim()) {
        return row.objective;
      }
    }
  }
  return "";
}
// The board row a mission status maps onto: the todo strip's three glyphs are
// completed / in progress / pending, and everything else on disk (blocked, idle,
// a status from a newer build) draws as not started.
function variant(status) {
  const value = String(status || "").toLowerCase();
  if (value === "done" || value === "completed") return "completed";
  if (value === "running" || value === "in_progress") return "in_progress";
  return "pending";
}
// Header summary, worded like the todo strip's: counts joined by an en space and
// a middle dot, with zero-count segments dropped as noise.
function boardSummary(missions) {
  let done = 0;
  let active = 0;
  for (const row of missions) {
    const kind = variant(row && row.status);
    if (kind === "completed") done += 1;
    else if (kind === "in_progress") active += 1;
  }
  const pending = missions.length - done - active;
  const bits = [];
  if (active > 0) bits.push(active + " 进行中");
  if (pending > 0) bits.push(pending + " 待办");
  if (done > 0) bits.push(done + " 已完成");
  return bits.join(SEP);
}
// The fallback summary, for a shell whose status carries no board: the active
// mission's own counters, exactly as the marker reported them.
function missionCounts(state) {
  const bits = [];
  bits.push((state.checkpoints || 0) + " 检查点");
  if (state.advances) bits.push(state.advances + " 推进");
  if (state.unread) bits.push(state.unread + " 未读信件");
  return bits.join(" \u00b7 ");
}
// One writer for the header summary, so the board and the counters cannot
// overwrite each other out of order: a known board always wins, an empty one
// says so, and only an unknown board falls back to the mission's counters.
function renderCounts() {
  if (Array.isArray(board)) {
    put("counts", board.length ? boardSummary(board) : EMPTY);
    return;
  }
  put("counts", counters || EMPTY);
}
function paint(state) {
  if (!state || typeof state !== "object") return;
  sawMarker = true;
  setId(state.id);
  setStatus(state.status);
  put("objective", objectiveOf(state) || "（未写目标）");
  put("step", state.step || (state.done_when ? "完成判据: " + state.done_when : "（未设下一步）"));
  counters = missionCounts(state);
  paintBoard(state.missions);
  renderCounts();
}
function scan(text) {
  if (typeof text !== "string") return;
  const found = /<!--mission-state ([\s\S]*?)-->/.exec(text);
  if (!found) return;
  try { paint(JSON.parse(found[1])); } catch (error) { /* a marker we cannot read is ignored */ }
}
// The status glyphs, drawn as inline SVG on the todo strip's 14×14 artboard and
// centered by the 16×16 .mp-glyph cell. Same art as TodoPanel, built here because
// the panel is a script and has no component library.
const SVG_NS = "http://www.w3.org/2000/svg";
// The completed glyph's check, copied from the strip's CompletedGlyph path.
const CHECK_PATH = "M10.9631 5.71411L7.70154 8.97571C7.48011 9.19714 7.27736 9.40099 7.09229 9.54993C6.89742 9.70669 6.66314 9.85279 6.3634 9.90027C6.2049 9.92534 6.04339 9.92534 5.88489 9.90027C5.58515 9.85279 5.35087 9.70669 5.15601 9.54993C4.97093 9.40099 4.76818 9.19714 4.54675 8.97571L3.03516 7.46411L3.96313 6.53613L5.47473 8.04773C5.7169 8.28989 5.86196 8.43389 5.97888 8.52795C6.08597 8.61409 6.10875 8.60701 6.08997 8.604C6.11259 8.60758 6.13571 8.60758 6.15833 8.604C6.13954 8.60701 6.16232 8.61409 6.26941 8.52795C6.38633 8.43389 6.53139 8.28989 6.77356 8.04773L10.0352 4.78613L10.9631 5.71411Z";
// Every in-progress ring needs a gradient, and a gradient reference is an id
// lookup in this document: two rows sharing one id would draw two identical
// rings off whichever <linearGradient> the engine found first. So each glyph
// takes the next number instead of a fixed name, and a repaint starts a fresh
// set after the old rows are gone.
let gradientSeq = 0;
function svgEl(name, attrs) {
  const node = document.createElementNS(SVG_NS, name);
  for (const key in attrs) {
    if (Object.prototype.hasOwnProperty.call(attrs, key)) node.setAttribute(key, attrs[key]);
  }
  return node;
}
function glyphShell(cls) {
  return svgEl("svg", {
    width: "14",
    height: "14",
    viewBox: "0 0 14 14",
    fill: "none",
    "class": cls,
    "aria-hidden": "true",
    "focusable": "false",
  });
}
// Completed: filled check inside the ring, in the success colour.
function completedGlyph() {
  const svg = glyphShell("mp-glyph-completed");
  svg.append(svgEl("circle", { cx: "7", cy: "7", r: "6.4", stroke: "currentColor", "stroke-width": "1.2" }));
  svg.append(svgEl("path", { d: CHECK_PATH, fill: "currentColor" }));
  return svg;
}
// In progress: the ring fades from the business colour to transparent along a
// diagonal, and the stylesheet spins the whole svg.
function progressGlyph() {
  gradientSeq += 1;
  const id = "mp-g" + gradientSeq;
  const svg = glyphShell("mp-glyph-progress");
  const defs = svgEl("defs");
  const gradient = svgEl("linearGradient", {
    id: id,
    x1: "2.5",
    y1: "12",
    x2: "10.5",
    y2: "3.5",
    gradientUnits: "userSpaceOnUse",
  });
  gradient.append(svgEl("stop", { "stop-color": "currentColor" }));
  gradient.append(svgEl("stop", { offset: "1", "stop-color": "currentColor", "stop-opacity": "0" }));
  defs.append(gradient);
  svg.append(defs);
  svg.append(svgEl("circle", { cx: "7", cy: "7", r: "6.4", stroke: "url(#" + id + ")", "stroke-width": "1.2" }));
  return svg;
}
// Pending: the dashed, unstarted ring in the caption colour.
function pendingGlyph() {
  const svg = glyphShell("mp-glyph-pending");
  svg.append(svgEl("circle", {
    cx: "7",
    cy: "7",
    r: "6.4",
    stroke: "currentColor",
    "stroke-width": "1.2",
    "stroke-dasharray": "2.4 2.4",
  }));
  return svg;
}
function glyphFor(status) {
  const kind = variant(status);
  if (kind === "completed") return completedGlyph();
  if (kind === "in_progress") return progressGlyph();
  return pendingGlyph();
}
// The board: every mission on disk, not just the one this tool just touched.
// Without it the card can only ever say "here is the current step", and a reader
// looking for the task they started sees nothing.
//
// A missing list and an empty one are different facts. Missing (an older shell's
// status, a marker without a board) leaves whatever is drawn alone; only an empty
// list says "no missions yet". Printing that over a board that exists is the
// complaint this panel was reported for.
function paintBoard(missions) {
  if (!Array.isArray(missions)) return;
  board = missions.slice();
  if (!list) return;
  list.replaceChildren();
  if (!board.length) {
    const empty = document.createElement("li");
    empty.className = "mp-item mp-empty";
    empty.textContent = "还没有任务（让模型调用 mission_start 建一个）";
    list.append(empty);
    return;
  }
  // Newest first is the order the board hands over. The mission the card is
  // describing is the one a reader looks for first, and it is not always the
  // newest — a tick on an older mission is exactly that case — so it is hoisted.
  const rows = board.slice();
  const at = shownId ? rows.findIndex((row) => row && row.id === shownId) : -1;
  if (at > 0) rows.unshift(rows.splice(at, 1)[0]);
  for (const row of rows) {
    const status = (row && row.status) || "?";
    const id = (row && row.id) || "";
    const objective = (row && row.objective) || "";
    const item = document.createElement("li");
    item.className = "mp-item";
    item.setAttribute("data-status", variant(status));
    if (id) item.setAttribute("data-id", String(id));
    if (shownId && id === shownId) item.setAttribute("data-current", "1");
    // The row is one line of text, so the id it belongs to rides in the tooltip
    // instead of taking a column of its own.
    item.title = id ? id + " — " + (objective || "（未写目标）") : objective;
    const glyph = document.createElement("span");
    glyph.className = "mp-glyph";
    glyph.setAttribute("aria-hidden", "true");
    glyph.append(glyphFor(status));
    const text = document.createElement("span");
    text.className = "mp-text";
    text.textContent = objective || id || "?";
    item.append(glyph, text);
    list.append(item);
  }
}
// The page hands every panel the status it already holds the moment the frame
// loads, so the list can be painted before any tool has answered. On its own that
// would put a list of missions under a card still reading "idle / no mission in
// progress", which is the same complaint in another shape — so until a marker has
// been seen the newest mission stands in for the card. A tool answer wins over it:
// that one is the more specific report.
function paintStatus(event) {
  const status = (event && event.status) || {};
  if (!Array.isArray(status.missions)) return;
  // The card's id is set before the board is drawn: paintBoard() marks the row
  // whose id is showing, and reading it back out of the DOM here would only ever
  // describe the previous paint.
  if (!sawMarker && status.missions.length) {
    const newest = status.missions[0] || {};
    setId(newest.id);
    setStatus(newest.status);
    put("objective", newest.objective || "（未写目标）");
    // The board carries no step, and the objective now has a line of its own, so
    // this one says the step is unknown instead of repeating the objective.
    put("step", "（未设下一步）");
  }
  paintBoard(status.missions);
  renderCounts();
}
// Collapsed is the resting state — the same choice the todo strip makes, so the
// dock holds one 36px row until a reader asks for the rest.
function setExpanded(open) {
  if (head) head.setAttribute("aria-expanded", open ? "true" : "false");
  if (detail) detail.hidden = !open;
}
function send(text) {
  if (!ngu || typeof ngu.call !== "function") return;
  // The window path has no reply channel (ipc is one-way), so the promise may
  // never settle: sending is the point, and it must not raise inside the panel.
  const pending = ngu.call("prompt", { text: text });
  if (pending && typeof pending.catch === "function") pending.catch(() => {});
}
// Every mission tool answer carries the marker, so no event needs to name the tool.
if (ngu && typeof ngu.on === "function") {
  ngu.on("tool_end", (event) => scan(event && event.result));
  ngu.on("status", paintStatus);
}
if (head) {
  head.addEventListener("click", () => {
    setExpanded(head.getAttribute("aria-expanded") !== "true");
  });
}
if (bar) {
  const refresh = bar.querySelector('[data-act="status"]');
  if (refresh) refresh.addEventListener("click", () => send("调用 mission_status，把当前长程任务的状态原样报告给我。"));
  const tick = bar.querySelector('[data-act="tick"]');
  if (tick) tick.addEventListener("click", () => send("调用 mission_tick 推进当前长程任务一步，然后汇报这一步的结果和下一步。"));
}
"##;

    crate::ui::UiPanel::new("mission-bar", crate::ui::UiSlot::AboveComposer, html)
        .with_script(script)
        .ordered(-10)
}

// ------------------------------------------------------------------- tools

fn start_tool() -> ToolDef {
    ToolDef::new(
        START,
        "Start a long-horizon mission. A mission is state on disk \
         (<workspace>/missions/<id>/): an objective, a done-when, a current step, \
         an append-only log, checkpoints, and mailboxes. Use it for work that will \
         not fit one conversation — it survives restarts and context compaction, \
         and mission_tick runs one step of it in a fresh worker.",
        json!({
            "type": "object",
            "properties": {
                "objective": { "type": "string", "description": "What the mission is for, in one paragraph. This is the standing answer to 'why am I doing this'." },
                "done_when": { "type": "string", "description": "How anyone can tell it is finished. Something checkable beats a feeling: a test that passes, a file that exists, a page that renders." },
                "first_step": { "type": "string", "description": "Optional first step. A mission with a next step can be ticked immediately." }
            },
            "required": ["objective"]
        }),
        PLUGIN,
        |args| Box::pin(async move { start(args).map(Into::into) }) as ToolFuture,
    )
}

fn status_tool() -> ToolDef {
    ToolDef::new(
        STATUS,
        "Report the active mission: objective, status, current step, checkpoints, \
         advances, unread mail, and the last few log entries. Cheap on purpose — \
         call it at the start of a turn instead of reading the whole history.",
        json!({ "type": "object", "properties": {} }),
        PLUGIN,
        |args| Box::pin(async move { status(args).map(Into::into) }) as ToolFuture,
    )
}

fn advance_tool() -> ToolDef {
    ToolDef::new(
        ADVANCE,
        "Record progress on the active mission: what was just done, what the next \
         step is, and optionally a new status (running, blocked, done). Appends to \
         the mission log and rewrites only the generated block of STATE.md.",
        json!({
            "type": "object",
            "properties": {
                "summary": { "type": "string", "description": "What changed in this step. One or two lines." },
                "next": { "type": "string", "description": "The next step, written so a worker with no memory of this conversation could start it." },
                "status": { "type": "string", "enum": ["running", "blocked", "done"], "description": "Mission status after this step." },
                "note": { "type": "string", "description": "Optional longer note kept in the log only." }
            },
            "required": ["summary"]
        }),
        PLUGIN,
        |args| Box::pin(async move { advance(args).map(Into::into) }) as ToolFuture,
    )
}

fn checkpoint_tool() -> ToolDef {
    ToolDef::new(
        CHECKPOINT,
        "Take a mission checkpoint: a named snapshot of STATE.md and the mission \
         record under checkpoints/. This is the rollback point a long task needs — \
         cheap, greppable, and independent of any VCS.",
        json!({
            "type": "object",
            "properties": {
                "label": { "type": "string", "description": "Short name for the point reached, e.g. 'milestone-1-crawler-done'." },
                "note": { "type": "string", "description": "Optional note stored with the checkpoint." }
            },
            "required": ["label"]
        }),
        PLUGIN,
        |args| Box::pin(async move { checkpoint(args).map(Into::into) }) as ToolFuture,
    )
}

fn recall_tool() -> ToolDef {
    ToolDef::new(
        RECALL,
        "Search the mission memory on disk — logs, STATE.md, checkpoints, mailboxes — \
         and return matching lines with file and line number. This is how to recover a \
         detail that fell out of the context window: search it, read the few lines that \
         matter, and keep the window small.",
        json!({
            "type": "object",
            "properties": {
                "query": { "type": "string", "description": "Words to look for. All of them must appear on the same line (case-insensitive)." },
                "mission": { "type": "string", "description": "Mission id to search. Defaults to every mission in this workspace." },
                "limit": { "type": "integer", "description": "Maximum matching lines to return (default 40, max 200)." }
            },
            "required": ["query"]
        }),
        PLUGIN,
        |args| Box::pin(async move { recall(args).map(Into::into) }) as ToolFuture,
    )
}

fn tick_tool(access: Arc<dyn crate::model::ModelAccess>) -> ToolDef {
    ToolDef::new(
        TICK,
        "Run one step of the active mission in a fresh worker — a subagent with no \
         memory of this conversation, handed the objective, the done-when, STATE.md \
         and the current step. The worker's report is logged, and any 'Next step:' it \
         names becomes the mission's next step. A long task is many of these, not one \
         long conversation.",
        json!({
            "type": "object",
            "properties": {
                "step": { "type": "string", "description": "Step to run. Defaults to the mission's current step." },
                "model": { "type": "string", "description": "Optional model for the worker. A cheaper model is usually right for mechanical steps." },
                "max_steps": { "type": "integer", "description": "Steps the worker may take before giving up (default 12, max 40)." },
                "timeout_ms": { "type": "integer", "description": "Wall-clock budget for the worker in milliseconds (default 1800000 = 30 minutes; max 6 hours). The worker is abandoned when it runs out." }
            }
        }),
        PLUGIN,
        move |args| {
            let access = Arc::clone(&access);
            Box::pin(async move { tick(access, args).await.map(Into::into) }) as ToolFuture
        },
    )
}

fn post_tool() -> ToolDef {
    ToolDef::new(
        POST,
        format!(
            "Send a letter to another agent working on the active mission. Mail is written \
             to that agent's mailbox on disk, so it is delivered whether or not the \
             recipient is running — it is read at the start of the recipient's next turn. \
             Use it to hand over findings, ask for something, or warn about a dead end. \
             A body longer than {LETTER_BODY_CAP} characters is cut, and this tool's reply \
             says so: write anything longer to a file and name the path in the letter."
        ),
        json!({
            "type": "object",
            "properties": {
                "to": { "type": "string", "description": "Recipient agent path, e.g. 'root' or 'worker-crawler'." },
                "body": { "type": "string", "description": "The letter. Short and specific: what the recipient should know or do." },
                "from": { "type": "string", "description": "Your own agent path. Defaults to 'root'." }
            },
            "required": ["to", "body"]
        }),
        PLUGIN,
        |args| Box::pin(async move { post(args).map(Into::into) }) as ToolFuture,
    )
}

fn inbox_tool() -> ToolDef {
    ToolDef::new(
        INBOX,
        format!(
            "Read your mailbox for the active mission. Unread letters are returned and \
             marked read, so the same letter is not delivered twice; pass mark_read=false \
             to peek. Recipients that are not running still receive mail — it waits here. \
             One read renders at most {INBOX_LETTER_CAP} letters or about \
             {INBOX_RENDER_CAP} characters and reports how many are still unread; call it \
             again to render the rest, which stay unread until they are shown."
        ),
        json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "Whose mailbox to read. Defaults to 'root'." },
                "mark_read": { "type": "boolean", "description": "Mark the returned letters read (default true)." }
            }
        }),
        PLUGIN,
        |args| Box::pin(async move { inbox(args).map(Into::into) }) as ToolFuture,
    )
}

// ------------------------------------------------------------ path plumbing

/// `<workspace>/missions`, created on demand.
fn missions_root() -> Result<PathBuf> {
    let workspace = crate::tools::current_workspace().ok_or_else(|| {
        anyhow!(
            "no session workspace is pinned, and a mission lives in one \
             (<workspace>/missions). Start a mission from a session that has a \
             working directory."
        )
    })?;
    let root = workspace.join("missions");
    std::fs::create_dir_all(&root)
        .with_context(|| format!("creating {}", root.display()))?;
    Ok(root)
}

/// Where the pointer to the current mission lives.
fn pointer_path(root: &Path) -> PathBuf {
    root.join("active.json")
}

/// The id of the active mission, if any.
fn active_id(root: &Path) -> Result<Option<String>> {
    let path = pointer_path(root);
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(error).with_context(|| format!("reading {}", path.display()))
        }
    };
    let value: Value = serde_json::from_str(&text)
        .with_context(|| format!("{} is not valid JSON", path.display()))?;
    let id = value
        .get("mission")
        .and_then(|m| m.as_str())
        .map(str::to_string);
    Ok(id.filter(|id| !id.trim().is_empty()))
}

/// A mission's directory, checked for existence.
fn mission_dir(root: &Path, id: &str) -> PathBuf {
    root.join(id)
}

/// The active mission's directory plus its record, or a message that says how
/// to get one. Missing is guidance; unparseable is an error naming the file.
fn active_mission() -> Result<(PathBuf, Value)> {
    let root = missions_root()?;
    let Some(id) = active_id(&root)? else {
        return Err(anyhow!(
            "no active mission. Start one with {START} (objective, and ideally \
             done_when and first_step); missions live in {}/missions/<id>/.",
            root.display()
        ));
    };
    let dir = mission_dir(&root, &id);
    let record = read_json(&dir.join("mission.json"))?;
    Ok((dir, record))
}

/// Read one JSON file, naming it when it cannot be understood.
fn read_json(path: &Path) -> Result<Value> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("{} is not valid JSON", path.display()))
}

/// Write a file so that a crash mid-write cannot leave it half-written.
fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let tmp = path.with_extension("tmp-write");
    std::fs::write(&tmp, bytes).with_context(|| format!("writing {}", tmp.display()))?;
    match std::fs::rename(&tmp, path) {
        Ok(()) => Ok(()),
        Err(_) => {
            // Windows refuses to rename onto an existing file. Removing first
            // costs the atomicity of this one replacement, not the file's
            // integrity: the new bytes are already written and flushed.
            let _ = std::fs::remove_file(path);
            std::fs::rename(&tmp, path)
                .with_context(|| format!("replacing {}", path.display()))
        }
    }
}

/// Append one line to a file, creating it if needed.
fn append_line(path: &Path, line: &str) -> Result<()> {
    use std::io::Write;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("opening {} for append", path.display()))?;
    file.write_all(line.as_bytes())
        .with_context(|| format!("appending to {}", path.display()))?;
    file.write_all(b"\n")
        .with_context(|| format!("appending to {}", path.display()))?;
    file.flush().ok();
    Ok(())
}

/// A path segment that cannot escape its parent.
fn safe_segment(name: &str) -> String {
    let cleaned: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' {
                c
            } else {
                '_'
            }
        })
        .collect();
    let trimmed = cleaned.trim_matches('.').to_string();
    if trimmed.is_empty() {
        "unnamed".to_string()
    } else {
        trimmed
    }
}

/// Milliseconds since the epoch, the stamp every record carries.
fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// `2026-10-19 14:03:22` in local time, for humans.
fn stamp(ms: u64) -> String {
    let secs = (ms / 1000) as i64;
    match chrono::DateTime::from_timestamp(secs, 0) {
        Some(time) => time
            .with_timezone(&chrono::Local)
            .format("%Y-%m-%d %H:%M:%S")
            .to_string(),
        None => ms.to_string(),
    }
}

/// Every mission on the board, newest first: what the panel lists beside the bar.
///
/// The bar can only describe the mission a tool just talked about, so without
/// this a reader has no way to see that other missions exist — which reads
/// exactly like "the task I started is gone". A missing or unpinned workspace
/// yields an empty list rather than an error: the marker decorates a tool
/// answer, and decoration must not turn a successful call into a failure.
fn board_json() -> Value {
    let Ok(root) = missions_root() else {
        return json!([]);
    };
    let mut rows = Vec::new();
    for id in list_missions(&root) {
        let path = root.join(&id).join("mission.json");
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        let Ok(record) = serde_json::from_str::<Value>(&text) else {
            continue;
        };
        rows.push(json!({
            "id": record.get("id").cloned().unwrap_or(json!(id)),
            "objective": record.get("objective").cloned().unwrap_or(Value::Null),
            "status": record.get("status").cloned().unwrap_or(Value::Null),
        }));
    }
    Value::Array(rows)
}

// ------------------------------------------- the mission a turn must not leave

/// How much of each field [`running_mission`] hands to the agent loop.
///
/// The brief is read out into an injected message, not into a report: 160
/// characters is the budget the panel and the worker prompt already give one
/// line, and it is enough to say what the mission is without turning every
/// batch of a long turn into a copy of `STATE.md`.
const BRIEF_CLIP: usize = 160;

/// What the agent loop needs to keep a turn pointed at a running mission.
///
/// A copy of the record's small text fields, not the record: the loop holds
/// this only long enough to write a message, and handing it `Value`s of live
/// mission state would invite someone to treat a stale copy as the truth.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MissionBrief {
    /// `m-20260101-000000`.
    pub id: String,
    /// What the mission is for.
    pub objective: String,
    /// The status as recorded. Only `running` produces a brief at all.
    pub status: String,
    /// The step the mission says comes next; empty when none was recorded.
    pub step: String,
    /// How anyone can tell it is finished; empty when none was given.
    pub done_when: String,
}

/// The mission a turn must not walk away from, if there is one.
///
/// The agent loop asks this between steps and after every answer. While the
/// active mission is `running`, ending the turn abandons work the model has
/// already taken responsibility for, and only the model may decide a mission is
/// over (`mission_advance` with `done` or `blocked`). So the question this
/// answers is narrow — "is there work on disk that still says it is going?" —
/// and it is answered from disk every time, because the same turn can change
/// the answer.
///
/// Cheap and quiet by construction: nothing is created, nothing is cached, and
/// every unreadable step (no workspace, no pointer, a corrupt pointer, a
/// missing or half-written record) is "no mission" rather than an error. This
/// is called from inside a turn, so a reader that could fail loudly would be
/// able to end one.
pub fn running_mission(workspace: &Path) -> Option<MissionBrief> {
    let root = workspace.join("missions");
    let id = active_id(&root).ok().flatten()?;
    let record = read_json(&mission_dir(&root, &id).join("mission.json")).ok()?;
    let field = |key: &str| -> String {
        record
            .get(key)
            .and_then(|value| value.as_str())
            .map(|text| clip(text.trim(), BRIEF_CLIP))
            .unwrap_or_default()
    };
    // Only `running` keeps a turn open. `blocked` and `done` are the model
    // saying it is no longer going; a shell that went round again anyway would
    // be overruling the one party allowed to end a mission.
    if field("status") != "running" {
        return None;
    }
    let recorded_id = field("id");
    Some(MissionBrief {
        // The record's own id when it has one, the pointer's otherwise: both
        // name the same mission, and a mission named wrongly in the injected
        // message is worse than a slightly stale name.
        id: if recorded_id.is_empty() { id } else { recorded_id },
        objective: field("objective"),
        status: field("status"),
        step: field("step"),
        done_when: field("done_when"),
    })
}

/// The marker every mission answer ends with.
///
/// The interface reads it to paint the mission bar; it costs a line and saves
/// the panel from being told which tool produced it.
fn marker(record: &Value, extra: Value) -> String {
    let mut state = json!({
        "id": record.get("id").cloned().unwrap_or(Value::Null),
        // The bar's largest line is the objective, so the marker carries it: the
        // board's copy belongs to a row, and a marker can be painted before any
        // board arrives (a tool answer out of the gate).
        "objective": record.get("objective").cloned().unwrap_or(Value::Null),
        "status": record.get("status").cloned().unwrap_or(Value::Null),
        "step": record.get("step").cloned().unwrap_or(Value::Null),
        "done_when": record.get("done_when").cloned().unwrap_or(Value::Null),
        "advances": record.get("advances").cloned().unwrap_or(json!(0)),
        "checkpoints": record
            .get("checkpoints")
            .and_then(|c| c.as_array())
            .map(|list| list.len())
            .unwrap_or(0),
    });
    if let (Some(map), Some(extra)) = (state.as_object_mut(), extra.as_object()) {
        for (key, value) in extra {
            map.insert(key.clone(), value.clone());
        }
    }
    if let Some(map) = state.as_object_mut() {
        map.insert("missions".to_string(), board_json());
    }
    format!("\n<!--mission-state {}-->", state)
}

/// One log entry: `{at, kind, actor, text, next}`.
fn log_entry(dir: &Path, kind: &str, actor: &str, text: &str, next: Option<&str>) -> Result<()> {
    let mut entry = json!({
        "at": now_ms(),
        "when": stamp(now_ms()),
        "kind": kind,
        "actor": actor,
        "text": clip(text, LOG_TEXT_CAP),
    });
    if let Some(next) = next {
        entry["next"] = json!(clip(next, LOG_TEXT_CAP));
    }
    append_line(&dir.join("log.jsonl"), &entry.to_string())
}

/// Shorten text without cutting a character in half.
fn clip(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_string();
    }
    let mut out: String = text.chars().take(max).collect();
    out.push_str("…");
    out
}

// -------------------------------------------------------------- mission_start

fn start(args: Value) -> Result<String> {
    let objective = args
        .get("objective")
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .ok_or_else(|| anyhow!("missing required argument: objective"))?
        .to_string();
    let done_when = args
        .get("done_when")
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .map(str::to_string);
    let first_step = args
        .get("first_step")
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .map(str::to_string);

    let root = missions_root()?;
    let id = format!("m-{}", chrono::Utc::now().format("%Y%m%d-%H%M%S"));
    let dir = mission_dir(&root, &id);
    std::fs::create_dir_all(dir.join("mailbox"))
        .with_context(|| format!("creating {}", dir.display()))?;
    std::fs::create_dir_all(dir.join("checkpoints"))
        .with_context(|| format!("creating {}", dir.display()))?;

    let record = json!({
        "id": id,
        "objective": objective,
        "done_when": done_when,
        "status": "running",
        "step": first_step,
        "created_ms": now_ms(),
        "updated_ms": now_ms(),
        "advances": 0,
        "ticks": 0,
        "checkpoints": [],
        "agents": {},
    });
    write_atomic(&dir.join("mission.json"), record.to_string().as_bytes())?;
    write_atomic(
        &dir.join("STATE.md"),
        state_shell(&record).as_bytes(),
    )?;
    log_entry(&dir, "start", "root", &objective, first_step.as_deref())?;
    write_atomic(
        &pointer_path(&root),
        json!({ "mission": id, "at": now_ms() }).to_string().as_bytes(),
    )?;

    let mut text = format!(
        "Mission {id} started.\n  directory: {}\n  objective: {}\n",
        dir.display(),
        record["objective"].as_str().unwrap_or("")
    );
    if let Some(done) = record.get("done_when").and_then(|v| v.as_str()) {
        text.push_str(&format!("  done when: {done}\n"));
    }
    match record.get("step").and_then(|v| v.as_str()) {
        Some(step) => text.push_str(&format!("  next step: {step}\n\n{TICK} runs it.\n")),
        None => text.push_str(&format!(
            "\nNo step yet. Record one with {ADVANCE} (summary + next), then {TICK} \
             runs it — or pass a step straight to {TICK}.\n"
        )),
    }
    text.push_str(
        "STATE.md and log.jsonl are in that directory: edit the parts of STATE.md \
         outside the generated block freely, and search the log with recall instead \
         of re-reading it.",
    );
    Ok(text + &marker(&record, json!({})))
}

/// The starting `STATE.md`: a generated block, plus sections the model owns.
fn state_shell(record: &Value) -> String {
    format!(
        "# 任务 {id}\n\n{fence}\n{block}\n{end}\n\n\
         ## 目标\n\n{objective}\n\n\
         ## 完成判据\n\n{done}\n\n\
         ## 决策记录\n\n（做过的关键选择写这里：为什么这样、放弃了什么）\n\n\
         ## 阻塞与风险\n\n（卡住的地方、已知的坑）\n\n\
         ## 备注\n\n（给下一个接管者的话）\n",
        id = record["id"].as_str().unwrap_or("?"),
        fence = FENCE_START,
        block = generated_block(record, &[]),
        end = FENCE_END,
        objective = record["objective"].as_str().unwrap_or(""),
        done = record
            .get("done_when")
            .and_then(|v| v.as_str())
            .unwrap_or("（未设：补一个可检查的判据）"),
    )
}

/// The block `mission_advance` maintains between the fences.
fn generated_block(record: &Value, recent: &[String]) -> String {
    let mut out = String::from("## 当前（自动维护，别手改这一段）\n\n");
    out.push_str(&format!(
        "- 状态：{}\n",
        record.get("status").and_then(|v| v.as_str()).unwrap_or("?")
    ));
    if let Some(step) = record.get("step").and_then(|v| v.as_str()) {
        out.push_str(&format!("- 下一步：{step}\n"));
    } else {
        out.push_str("- 下一步：（未设）\n");
    }
    out.push_str(&format!(
        "- 推进 {} 次 · 检查点 {} 个 · 更新于 {}\n",
        record.get("advances").and_then(|v| v.as_u64()).unwrap_or(0),
        record
            .get("checkpoints")
            .and_then(|v| v.as_array())
            .map(|l| l.len())
            .unwrap_or(0),
        stamp(record.get("updated_ms").and_then(|v| v.as_u64()).unwrap_or(0)),
    ));
    if !recent.is_empty() {
        out.push_str("- 最近：\n");
        for line in recent {
            out.push_str(&format!("  - {line}\n"));
        }
    }
    out
}

/// Replace the generated block, or add one when the file lost its fences.
fn replace_generated(state: &str, block: &str) -> String {
    match (state.find(FENCE_START), state.find(FENCE_END)) {
        (Some(start), Some(end)) if end > start => {
            let mut out = String::with_capacity(state.len() + block.len());
            out.push_str(&state[..start]);
            out.push_str(FENCE_START);
            out.push('\n');
            out.push_str(block);
            out.push('\n');
            out.push_str(&state[end..]);
            out
        }
        _ => {
            let mut out = state.trim_end().to_string();
            out.push_str(&format!("\n\n{FENCE_START}\n{block}\n{FENCE_END}\n"));
            out
        }
    }
}

/// The last few advances, newest first, for the generated block.
fn recent_advances(dir: &Path, count: usize) -> Vec<String> {
    let Ok(text) = std::fs::read_to_string(dir.join("log.jsonl")) else {
        return Vec::new();
    };
    let mut lines: Vec<String> = text
        .lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .filter(|entry| {
            matches!(
                entry.get("kind").and_then(|k| k.as_str()),
                Some("advance") | Some("tick") | Some("checkpoint")
            )
        })
        .map(|entry| {
            format!(
                "{} {}",
                entry.get("when").and_then(|w| w.as_str()).unwrap_or(""),
                clip(entry.get("text").and_then(|t| t.as_str()).unwrap_or(""), 160)
            )
        })
        .collect();
    lines.reverse();
    lines.truncate(count);
    lines
}

// ------------------------------------------------------------- mission_status

fn status(_args: Value) -> Result<String> {
    let root = missions_root()?;
    let Some(id) = active_id(&root)? else {
        return Ok(format!(
            "No active mission. {START} creates one in {}/missions/.\n\
             Missions already on disk: {}",
            root.display(),
            list_missions(&root).join(", ")
        ));
    };
    let dir = mission_dir(&root, &id);
    let record = read_json(&dir.join("mission.json"))?;
    let unread = total_unread(&dir);

    let mut out = format!(
        "{id}  [{}]\n  objective: {}\n",
        record.get("status").and_then(|v| v.as_str()).unwrap_or("?"),
        record.get("objective").and_then(|v| v.as_str()).unwrap_or("")
    );
    if let Some(done) = record.get("done_when").and_then(|v| v.as_str()) {
        out.push_str(&format!("  done when: {done}\n"));
    }
    match record.get("step").and_then(|v| v.as_str()) {
        Some(step) => out.push_str(&format!("  next step: {step}\n")),
        None => out.push_str("  next step: (none recorded)\n"),
    }
    let checkpoints = record
        .get("checkpoints")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    out.push_str(&format!(
        "  advances {} · ticks {} · checkpoints {}",
        record.get("advances").and_then(|v| v.as_u64()).unwrap_or(0),
        record.get("ticks").and_then(|v| v.as_u64()).unwrap_or(0),
        checkpoints.len()
    ));
    if let Some(last) = checkpoints.last() {
        out.push_str(&format!(
            " (last: {} @ {})",
            last.get("label").and_then(|v| v.as_str()).unwrap_or("?"),
            last.get("when").and_then(|v| v.as_str()).unwrap_or("")
        ));
    }
    out.push('\n');
    out.push_str(&format!(
        "  updated {} · directory {}\n",
        stamp(record.get("updated_ms").and_then(|v| v.as_u64()).unwrap_or(0)),
        dir.display()
    ));
    if unread > 0 {
        out.push_str(&format!("  unread mail: {unread} (agent_inbox reads it)\n"));
    }
    let recent = recent_advances(&dir, 5);
    if recent.is_empty() {
        out.push_str("  (no advances logged yet)\n");
    } else {
        out.push_str("  recent:\n");
        for line in recent {
            out.push_str(&format!("    - {line}\n"));
        }
    }
    Ok(out + &marker(&record, json!({ "unread": unread })))
}

/// Every mission directory on disk, newest name first.
fn list_missions(root: &Path) -> Vec<String> {
    let Ok(entries) = std::fs::read_dir(root) else {
        return Vec::new();
    };
    let mut ids: Vec<String> = entries
        .flatten()
        .filter(|entry| entry.path().is_dir())
        .filter_map(|entry| entry.file_name().to_str().map(str::to_string))
        .collect();
    ids.sort();
    ids.reverse();
    ids
}

/// Unread letters across every mailbox of a mission.
fn total_unread(dir: &Path) -> usize {
    let maildir = dir.join("mailbox");
    let Ok(entries) = std::fs::read_dir(&maildir) else {
        return 0;
    };
    entries
        .flatten()
        .filter(|entry| entry.path().extension().and_then(|e| e.to_str()) == Some("jsonl"))
        .map(|entry| {
            let name = entry
                .file_name()
                .to_string_lossy()
                .trim_end_matches(".jsonl")
                .to_string();
            // `mailbox_unread` takes the *mission* directory: it resolves the
            // mailbox itself (`<dir>/mailbox/<who>.jsonl`) and the watermark
            // beside it. Handing it the mailbox directory made both paths one
            // level too deep, so every mailbox read as empty and `status`
            // reported no unread mail at all.
            mailbox_unread(dir, &name)
        })
        .sum()
}

// ------------------------------------------------------------ mission_advance

fn advance(args: Value) -> Result<String> {
    let summary = args
        .get("summary")
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .ok_or_else(|| anyhow!("missing required argument: summary"))?
        .to_string();
    let next = args
        .get("next")
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .map(str::to_string);
    let note = args.get("note").and_then(|v| v.as_str()).map(str::to_string);
    // Optional, but a value that is present must be one of the three: a typo
    // silently keeping the old status is how a mission reads "running" forever.
    let status = match args.get("status").and_then(|v| v.as_str()) {
        None => None,
        Some(value) if matches!(value, "running" | "blocked" | "done") => Some(value.to_string()),
        Some(other) => {
            return Err(anyhow!(
                "status must be one of: running, blocked, done (got '{other}')"
            ))
        }
    };

    let (dir, mut record) = active_mission()?;
    let id = record
        .get("id")
        .and_then(|v| v.as_str())
        .unwrap_or("?")
        .to_string();

    if let Some(note) = note {
        log_entry(&dir, "note", "root", &note, None)?;
    }
    log_entry(&dir, "advance", "root", &summary, next.as_deref())?;

    record["advances"] = json!(record.get("advances").and_then(|v| v.as_u64()).unwrap_or(0) + 1);
    record["updated_ms"] = json!(now_ms());
    if let Some(next) = &next {
        record["step"] = json!(next);
    }
    if let Some(status) = &status {
        record["status"] = json!(status);
    }
    write_atomic(
        &dir.join("mission.json"),
        serde_json::to_string_pretty(&record)?.as_bytes(),
    )?;

    // Only the generated block is rewritten; the model's own prose survives.
    let state_path = dir.join("STATE.md");
    let current = std::fs::read_to_string(&state_path).unwrap_or_default();
    let recent = recent_advances(&dir, 3);
    let updated = replace_generated(&current, &generated_block(&record, &recent));
    write_atomic(&state_path, updated.as_bytes())?;

    let mut out = format!("{id} advanced.\n  {summary}\n");
    if let Some(next) = &next {
        out.push_str(&format!("  next step: {next}\n"));
    }
    if let Some(status) = &status {
        out.push_str(&format!("  status: {status}\n"));
    }
    out.push_str("  log.jsonl appended · STATE.md's generated block rewritten\n");
    Ok(out + &marker(&record, json!({})))
}

// ---------------------------------------------------------- mission_checkpoint

fn checkpoint(args: Value) -> Result<String> {
    let label = args
        .get("label")
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .ok_or_else(|| anyhow!("missing required argument: label"))?
        .to_string();
    let note = args.get("note").and_then(|v| v.as_str()).map(str::to_string);

    let (dir, mut record) = active_mission()?;
    let id = record
        .get("id")
        .and_then(|v| v.as_str())
        .unwrap_or("?")
        .to_string();
    let checkpoints = record
        .get("checkpoints")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    let number = checkpoints.len() + 1;
    let file_name = format!("{:03}-{}.md", number, safe_segment(&label));
    let file = dir.join("checkpoints").join(&file_name);

    let state = std::fs::read_to_string(dir.join("STATE.md")).unwrap_or_default();
    let mut body = format!(
        "# 检查点 {number} · {label}\n\n- 时间：{}\n- 任务：{id}\n- 状态：{}\n",
        stamp(now_ms()),
        record.get("status").and_then(|v| v.as_str()).unwrap_or("?"),
    );
    if let Some(note) = &note {
        body.push_str(&format!("- 备注：{note}\n"));
    }
    body.push_str(&format!(
        "\n---\n\n## 当时的 STATE.md\n\n{state}\n\n## 当时的任务记录\n\n```json\n{}\n```\n",
        serde_json::to_string_pretty(&record)?
    ));
    write_atomic(&file, body.as_bytes())?;

    let entry = json!({
        "n": number,
        "label": label,
        "when": stamp(now_ms()),
        "at": now_ms(),
        "file": format!("checkpoints/{file_name}"),
        "note": note,
    });
    let mut list = checkpoints;
    list.push(entry.clone());
    record["checkpoints"] = json!(list);
    record["updated_ms"] = json!(now_ms());
    write_atomic(
        &dir.join("mission.json"),
        serde_json::to_string_pretty(&record)?.as_bytes(),
    )?;
    log_entry(&dir, "checkpoint", "root", &label, None)?;

    let state_path = dir.join("STATE.md");
    let current = std::fs::read_to_string(&state_path).unwrap_or_default();
    let recent = recent_advances(&dir, 3);
    write_atomic(
        &state_path,
        replace_generated(&current, &generated_block(&record, &recent)).as_bytes(),
    )?;

    Ok(format!(
        "{id} checkpoint {number} “{label}” saved to {}\n  (STATE.md + the mission \
         record as they are now; recall can read it later)\n",
        file.display()
    ) + &marker(&record, json!({})))
}

// ------------------------------------------------------------------- recall

fn recall(args: Value) -> Result<String> {
    let query = args
        .get("query")
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .ok_or_else(|| anyhow!("missing required argument: query"))?
        .to_lowercase();
    let terms: Vec<&str> = query.split_whitespace().collect();
    let limit = args
        .get("limit")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(RECALL_LIMIT)
        .clamp(1, 200);
    let only = args
        .get("mission")
        .and_then(|v| v.as_str())
        .map(str::to_string);

    let root = missions_root()?;
    let mut dirs: Vec<PathBuf> = match &only {
        Some(id) => vec![mission_dir(&root, &safe_segment(id))],
        None => list_missions(&root)
            .into_iter()
            .map(|id| mission_dir(&root, &id))
            .collect(),
    };
    dirs.retain(|dir| dir.is_dir());
    if dirs.is_empty() {
        return Ok(format!(
            "Nothing to search yet: {}/missions/ holds no mission directories.\n",
            root.display()
        ));
    }

    let mut hits: Vec<String> = Vec::new();
    let mut searched = 0usize;
    'outer: for dir in &dirs {
        let mission = dir
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        for file in memory_files(dir) {
            searched += 1;
            let Ok(text) = read_tail(&file, RECALL_FILE_CAP) else {
                continue;
            };
            for (number, line) in text.lines().enumerate() {
                let lowered = line.to_lowercase();
                if !terms.iter().all(|term| lowered.contains(term)) {
                    continue;
                }
                let shown = clip(line.trim(), 240);
                hits.push(format!("{mission}/{}:{}: {shown}", rel(&file, dir), number + 1));
                if hits.len() >= limit {
                    break 'outer;
                }
            }
        }
    }

    let mut out = format!(
        "recall “{}”: {} hit(s) in {} file(s) across {} mission(s).\n",
        query,
        hits.len(),
        searched,
        dirs.len()
    );
    if hits.is_empty() {
        out.push_str("(nothing matched — try fewer words, or a word you know is in a log entry)\n");
    } else {
        for hit in &hits {
            out.push_str(&format!("  {hit}\n"));
        }
        if hits.len() >= limit {
            out.push_str(&format!("(stopped at the {limit}-line limit)\n"));
        }
    }
    Ok(out)
}

/// The files that make up a mission's memory, logs first.
fn memory_files(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    for name in ["log.jsonl", "STATE.md", "mission.json"] {
        let path = dir.join(name);
        if path.is_file() {
            files.push(path);
        }
    }
    for sub in ["checkpoints", "mailbox"] {
        let Ok(entries) = std::fs::read_dir(dir.join(sub)) else {
            continue;
        };
        let mut found: Vec<PathBuf> = entries
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| path.is_file())
            .collect();
        found.sort();
        files.extend(found);
    }
    files
}

/// A file's path relative to its mission directory, for a readable hit.
fn rel(file: &Path, dir: &Path) -> String {
    file.strip_prefix(dir)
        .map(|p| p.to_string_lossy().replace('\\', "/"))
        .unwrap_or_else(|_| file.to_string_lossy().to_string())
}

/// Read a file, or its last `cap` bytes when it is longer than that.
fn read_tail(path: &Path, cap: u64) -> Result<String> {
    use std::io::{Read, Seek, SeekFrom};
    let mut file = std::fs::File::open(path)
        .with_context(|| format!("opening {}", path.display()))?;
    let len = file.metadata()?.len();
    if len > cap {
        file.seek(SeekFrom::Start(len - cap))?;
    }
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

// --------------------------------------------------------------- mission_tick

/// The prompt a tick's worker is run with. Pure, so it can be tested.
fn worker_prompt(record: &Value, state: &str, step: &str) -> String {
    let objective = record
        .get("objective")
        .and_then(|v| v.as_str())
        .unwrap_or("(none recorded)");
    let done_when = record
        .get("done_when")
        .and_then(|v| v.as_str())
        .unwrap_or("(no done-when recorded — judge sensibly and say what you assumed)");
    format!(
        "You are the worker for one step of a long mission. You have no memory of the \
         conversation that set this up; everything you need is below.\n\n\
         ## Mission objective\n{objective}\n\n\
         ## Done when\n{done_when}\n\n\
         ## State so far (STATE.md)\n{state}\n\n\
         ## Your step\n{step}\n\n\
         Do this step and only this step. Work with the tools you have, verify what \
         you can (run it, test it, read it back), and do not start the next step. \
         When you are done, reply with:\n\
         1. what you changed or produced (paths),\n\
         2. how you checked it,\n\
         3. a line starting `Next step:` naming the single next thing to do.\n\
         If the step cannot be done, say so plainly and name what is blocking it."
    )
}

/// `30 minutes`, `2 hours` — a budget as the error message states it.
fn format_wall_clock(budget: std::time::Duration) -> String {
    let secs = budget.as_secs();
    if secs >= 3600 && secs % 3600 == 0 {
        format!("{} hours", secs / 3600)
    } else if secs >= 60 {
        format!("{} minutes", secs / 60)
    } else {
        format!("{secs} seconds")
    }
}

async fn tick(access: Arc<dyn crate::model::ModelAccess>, args: Value) -> Result<String> {
    let (dir, mut record) = active_mission()?;
    let id = record
        .get("id")
        .and_then(|v| v.as_str())
        .unwrap_or("?")
        .to_string();
    let step = args
        .get("step")
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .map(str::to_string)
        .or_else(|| {
            record
                .get("step")
                .and_then(|v| v.as_str())
                .map(str::to_string)
        })
        .ok_or_else(|| {
            anyhow!(
                "this mission has no current step. Pass one (`step`) or record one \
                 with {ADVANCE} (summary + next)."
            )
        })?;
    let state = std::fs::read_to_string(dir.join("STATE.md"))
        .map(|text| clip(&text, STATE_CAP))
        .unwrap_or_default();

    let client = match args.get("model").and_then(|m| m.as_str()) {
        Some(model) => access.client_for(model)?,
        None => access.client()?,
    };
    let max_steps = args
        .get("max_steps")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(TICK_STEPS)
        .clamp(1, 40);
    // A step count is not a time bound: one step can sit inside a single `bash`
    // call for an hour (the default bash timeout alone is two minutes). The
    // wall clock is the bound that actually protects the turn.
    let wall_clock = std::time::Duration::from_millis(
        args.get("timeout_ms")
            .and_then(|v| v.as_u64())
            .unwrap_or(TICK_WALL_CLOCK_MS)
            .clamp(TICK_MIN_WALL_CLOCK_MS, TICK_MAX_WALL_CLOCK_MS),
    );

    let config: Arc<dyn crate::agent::TurnConfig> = Arc::new(
        crate::agent::StaticConfig::new(access.system_prompt(), access.tools(), client.model())
            .with_policy(access.context_policy()),
    );
    let mut agent = crate::agent::Agent::new(client, config, Vec::new())
        .with_max_steps(max_steps);
    let prompt = worker_prompt(&record, &state, &step);
    let outcome = match tokio::time::timeout(wall_clock, agent.run(&prompt)).await {
        Ok(outcome) => outcome,
        // The worker future is dropped here, which cancels it: a mission that
        // hangs must not hold the turn open indefinitely. Saying exactly that
        // is better than an empty reply, and the log records it as the tick's
        // error so the next tick sees what happened.
        Err(_) => Err(anyhow!(
            "the worker ran past its wall-clock budget of {} and was abandoned. Nothing \
             it may have done is known to be undone — check the working tree before \
             retrying, or raise timeout_ms (max {} s) for a genuinely long step.",
            format_wall_clock(wall_clock),
            TICK_MAX_WALL_CLOCK_MS / 1000
        )),
    };

    let (text, failed) = match outcome {
        Ok(outcome) => (outcome.text.clone(), outcome.error.clone()),
        Err(error) => (format!("the worker could not run: {error:#}"), Some(format!("{error:#}"))),
    };

    // The next step, when the worker named one. A missing line is fine: the
    // mission keeps its current step and the orchestrator decides.
    let next = text
        .lines()
        .find_map(|line| {
            let trimmed = line.trim().trim_start_matches(['-', '*', ' ']).trim();
            for prefix in ["Next step:", "NEXT STEP:", "next step:", "下一步：", "下一步:"] {
                if let Some(rest) = trimmed.strip_prefix(prefix) {
                    let rest = rest.trim();
                    if !rest.is_empty() {
                        return Some(rest.to_string());
                    }
                }
            }
            None
        });

    log_entry(&dir, "tick", "worker", &text, next.as_deref())?;
    if let Some(error) = &failed {
        log_entry(&dir, "tick-error", "worker", error, None)?;
    }
    record["ticks"] = json!(record.get("ticks").and_then(|v| v.as_u64()).unwrap_or(0) + 1);
    record["updated_ms"] = json!(now_ms());
    if let Some(next) = &next {
        record["step"] = json!(next);
    }
    write_atomic(
        &dir.join("mission.json"),
        serde_json::to_string_pretty(&record)?.as_bytes(),
    )?;
    let state_path = dir.join("STATE.md");
    let current = std::fs::read_to_string(&state_path).unwrap_or_default();
    let recent = recent_advances(&dir, 3);
    write_atomic(
        &state_path,
        replace_generated(&current, &generated_block(&record, &recent)).as_bytes(),
    )?;

    let mut out = format!("{id} · step run: {step}\n\n{text}\n");
    if let Some(error) = failed {
        out.push_str(&format!("\nThe worker did not finish cleanly: {error}\n"));
    }
    match next {
        Some(next) => out.push_str(&format!("\nNext step recorded: {next}\n")),
        None => out.push_str(
            "\nThe worker named no next step. Record one with mission_advance \
             (summary + next), or pass one to the next mission_tick.\n",
        ),
    }
    Ok(out + &marker(&record, json!({})))
}

// ------------------------------------------------------------------- mailbox

/// Most characters one letter may store.
///
/// A letter is rendered at the start of the recipient's next turn, so an
/// oversized one is charged to that turn's context, and to every later read
/// that renders it again. A sender with more to say writes a file instead and
/// puts the path in the letter.
const LETTER_BODY_CAP: usize = 16_000;

/// Most characters one `inbox` read renders into the transcript.
///
/// A mailbox is an append-only file that can hold thousands of letters, so
/// "read the unread ones" is unbounded unless a read bounds itself.
const INBOX_RENDER_CAP: usize = 64 * 1024;

/// Most letters one `inbox` read renders, so a mailbox of many tiny letters is
/// bounded as well rather than only by the character cap.
const INBOX_LETTER_CAP: usize = 200;

/// One mailbox file per recipient.
fn mailbox_file(dir: &Path, who: &str) -> PathBuf {
    dir.join("mailbox").join(format!("{}.jsonl", safe_segment(who)))
}

/// How many letters a mailbox holds, and how many were already read.
///
/// The "read" half goes through the same check [`read_watermark`] applies to a
/// read: a bare count clamped to a *rewritten* file would report "nothing
/// unread" for letters that are still on disk and are about to be delivered
/// again, so `status` and `inbox` would disagree about the same mailbox.
fn mailbox_counts(dir: &Path, who: &str) -> (usize, usize) {
    let file = mailbox_file(dir, who);
    let text = std::fs::read_to_string(&file).unwrap_or_default();
    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    let (read, _rotated) = read_watermark(&lines, dir, who);
    (lines.len(), read)
}

fn offset_path(dir: &Path, who: &str) -> PathBuf {
    dir.join("mailbox").join(format!("{}.read", safe_segment(who)))
}

/// The watermark as it is stored on disk: how many letters were consumed, and
/// — in the new format — a fingerprint of the prefix that was consumed,
/// `12@0f3a…`. A bare number (the old format) is still accepted, but it cannot
/// be verified, so it reads as "rotated" once: the letters come back, and the
/// next mark rewrites it in the verifiable format.
fn read_mark(dir: &Path, who: &str) -> Option<(usize, Option<String>)> {
    let raw = std::fs::read_to_string(offset_path(dir, who)).ok()?;
    let raw = raw.trim();
    let (count, fingerprint) = match raw.split_once('@') {
        Some((count, fingerprint)) => (count.parse::<usize>().ok()?, Some(fingerprint.to_string())),
        None => (raw.parse::<usize>().ok()?, None),
    };
    Some((count, fingerprint))
}

/// Where "already read" sits in this mailbox, and whether the mailbox no
/// longer matches what was read.
///
/// A watermark that is only a count cannot survive a rewrite: when the file is
/// truncated or rotated, the stale count either points past the end — or,
/// clamped to the new length, silently marks letters that are **still on disk**
/// as already read, and they become unreachable forever, with no error and no
/// hint. The fingerprint is the tripwire: if the prefix the count covered is
/// not the prefix on disk any more, the mailbox was rewritten.
fn read_watermark(lines: &[&str], dir: &Path, who: &str) -> (usize, bool) {
    let (read, fingerprint) = match read_mark(dir, who) {
        Some(mark) => mark,
        None => return (0, false),
    };
    if read == 0 {
        return (0, false);
    }
    // The file shrank under the watermark: those letters are gone.
    if read > lines.len() {
        return (0, true);
    }
    match fingerprint {
        // The prefix that was read is still the prefix on disk.
        Some(fingerprint) if fingerprint == prefix_fingerprint(&lines[..read]) => (read, false),
        // Everything else — a prefix that changed, and a watermark written
        // before fingerprints existed, which nothing can verify — reads again
        // from the top: a duplicate delivery costs a re-read; a lost letter
        // costs the work it carried. The first mark rewrites the watermark in
        // the verifiable format, so the re-read happens once, not forever.
        _ => (0, true),
    }
}

/// Persist the watermark after a read consumed `lines` in full.
fn write_watermark(dir: &Path, who: &str, lines: &[&str]) -> Result<()> {
    let mark = format!("{}@{}", lines.len(), prefix_fingerprint(lines));
    write_atomic(&offset_path(dir, who), mark.as_bytes())
}

/// A short, stable fingerprint of the letters a watermark covered.
///
/// FNV-1a, by hand: a persisted watermark must compare equal across processes,
/// so the digest has to be stable forever, which a dependency's hash (or the
/// std one, whose algorithm is unspecified) does not promise. The newline each
/// line lost to `lines()` goes back in, so a line that splits in two never
/// reads as the same prefix.
fn prefix_fingerprint(lines: &[&str]) -> String {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = FNV_OFFSET;
    for line in lines {
        for byte in line.as_bytes() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        hash ^= u64::from(b'\n');
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    format!("{hash:016x}")
}

fn mailbox_unread(dir: &Path, who: &str) -> usize {
    let (total, read) = mailbox_counts(dir, who);
    total.saturating_sub(read)
}

fn post(args: Value) -> Result<String> {
    let to = args
        .get("to")
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .ok_or_else(|| anyhow!("missing required argument: to"))?
        .to_string();
    let body = args
        .get("body")
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .ok_or_else(|| anyhow!("missing required argument: body"))?;
    // The letter is capped at the door, not at the reader: whoever opens the
    // mailbox pays for whatever is in it. The cut is reported to the sender,
    // so a truncated letter is never mistaken for a whole one.
    let body_chars = body.chars().count();
    let body = clip(body, LETTER_BODY_CAP);
    let from = args
        .get("from")
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .unwrap_or("root")
        .to_string();

    let (dir, record) = active_mission()?;
    let id = record
        .get("id")
        .and_then(|v| v.as_str())
        .unwrap_or("?")
        .to_string();
    let file = mailbox_file(&dir, &to);
    let entry = json!({
        "at": now_ms(),
        "when": stamp(now_ms()),
        "from": from,
        "to": to,
        "body": body,
    });
    append_line(&file, &entry.to_string())?;
    log_entry(&dir, "mail", &from, &format!("→ {to}: {}", clip(&body, 400)), None)?;

    let (total, read) = mailbox_counts(&dir, &to);
    let mut out = format!(
        "Letter from {from} to {to} written to {}\n  {to} has {} unread letter(s) \
         (it sees them with {INBOX} at the start of its next turn).\n",
        file.display(),
        total.saturating_sub(read)
    );
    if body_chars > LETTER_BODY_CAP {
        out.push_str(&format!(
            "  the body was cut to {LETTER_BODY_CAP} characters (sent {body_chars}); \
             {to} will not see the rest. Write the whole text to a file and put its \
             path in the letter when it matters.\n"
        ));
    }
    Ok(out + &marker(&record, json!({ "unread": total_unread(&dir), "id": id })))
}

fn inbox(args: Value) -> Result<String> {
    let who = args
        .get("path")
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .unwrap_or("root")
        .to_string();
    let mark = args
        .get("mark_read")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);

    let (dir, record) = active_mission()?;
    let id = record
        .get("id")
        .and_then(|v| v.as_str())
        .unwrap_or("?")
        .to_string();
    let file = mailbox_file(&dir, &who);
    let text = std::fs::read_to_string(&file).unwrap_or_default();
    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    let total = lines.len();
    // The watermark may no longer match the file: a truncated or rotated
    // mailbox reads again from the top instead of silently skipping letters
    // that are still on disk.
    let (read, rotated) = read_watermark(&lines, &dir, &who);
    let fresh: Vec<&str> = lines.iter().skip(read).copied().collect();

    let mut out = format!("{who}@…{}: {} letter(s), {} unread\n", id, total, fresh.len());
    if rotated {
        out.push_str("  (mailbox rotated, re-read from the top)\n");
    }
    if fresh.is_empty() {
        out.push_str("(no new mail)\n");
    }
    // Rendering is what this turn pays for, so the read bounds itself instead
    // of trusting the mailbox to stay small. A letter that is not rendered is
    // not consumed either: the watermark covers only the prefix this read
    // actually showed, so the rest is delivered on the next call.
    let mut shown = 0usize;
    let mut rendered_chars = 0usize;
    for line in &fresh {
        if shown >= INBOX_LETTER_CAP || rendered_chars >= INBOX_RENDER_CAP {
            break;
        }
        let entry = match serde_json::from_str::<Value>(line) {
            Ok(entry) => format!(
                "  [{} · {}]\n    {}\n",
                entry.get("from").and_then(|v| v.as_str()).unwrap_or("?"),
                entry.get("when").and_then(|v| v.as_str()).unwrap_or(""),
                entry.get("body").and_then(|v| v.as_str()).unwrap_or("")
            ),
            Err(_) => format!("  (unreadable letter) {}\n", clip(line, 160)),
        };
        rendered_chars += entry.chars().count();
        out.push_str(&entry);
        shown += 1;
    }
    let withheld = fresh.len().saturating_sub(shown);
    if withheld > 0 {
        out.push_str(&format!(
            "  ({withheld} more unread letter(s) not shown by this read — they stay \
             unread; run {INBOX} again to render the next batch)\n"
        ));
    }
    if mark && shown > 0 {
        write_watermark(&dir, &who, &lines[..read + shown])?;
        out.push_str(&format!(
            "  (marked read; {} of {total} letters)\n",
            read + shown
        ));
    }
    Ok(out + &marker(&record, json!({ "unread": total_unread(&dir) })))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A scratch workspace with the session's workspace pinned to it.
    fn in_workspace<R>(tag: &str, f: impl FnOnce(&Path) -> R) -> R {
        let dir = std::env::temp_dir().join(format!(
            "ngu-mission-{tag}-{}",
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        crate::tools::CURRENT_WORKSPACE.with(|cell| *cell.borrow_mut() = Some(dir.clone()));
        let out = f(&dir);
        crate::tools::CURRENT_WORKSPACE.with(|cell| *cell.borrow_mut() = None);
        let _ = std::fs::remove_dir_all(&dir);
        out
    }

    fn start_mission(dir: &Path, objective: &str) -> String {
        let _ = dir;
        start(json!({
            "objective": objective,
            "done_when": "the tests pass",
            "first_step": "write the failing test"
        }))
        .expect("start")
    }

    #[test]
    fn a_mission_writes_state_log_and_pointer_to_disk() {
        in_workspace("start", |dir| {
            let text = start_mission(dir, "ship the crawler");
            assert!(text.contains("m-"), "{text}");

            let missions = dir.join("missions");
            assert!(missions.join("active.json").is_file(), "pointer written");
            let id = active_id(&missions).unwrap().expect("an active mission");
            let mdir = missions.join(&id);
            assert!(mdir.join("mission.json").is_file());
            assert!(mdir.join("STATE.md").is_file());
            assert!(mdir.join("log.jsonl").is_file());
            assert!(mdir.join("mailbox").is_dir());
            assert!(mdir.join("checkpoints").is_dir());

            let state = std::fs::read_to_string(mdir.join("STATE.md")).unwrap();
            assert!(state.contains("ship the crawler"), "objective in STATE.md");
            assert!(state.contains(FENCE_START), "generated block present");
            let log = std::fs::read_to_string(mdir.join("log.jsonl")).unwrap();
            assert!(log.contains("\"kind\":\"start\""), "{log}");
        });
    }

    #[test]
    fn status_reports_what_advance_and_checkpoint_recorded() {
        in_workspace("status", |dir| {
            start_mission(dir, "long job");
            advance(json!({
                "summary": "scaffold done",
                "next": "crawl the first 100 pages",
                "status": "running"
            }))
            .expect("advance");
            checkpoint(json!({ "label": "scaffold", "note": "green build" })).expect("checkpoint");

            let text = status(json!({})).expect("status");
            assert!(text.contains("scaffold done"), "{text}");
            assert!(text.contains("crawl the first 100 pages"), "{text}");
            assert!(text.contains("advances 1"), "{text}");
            assert!(text.contains("checkpoints 1"), "{text}");
            assert!(text.contains("scaffold @"), "{text}");
            // The marker is what the mission bar reads.
            let found = text
                .split("<!--mission-state ")
                .nth(1)
                .and_then(|rest| rest.split("-->").next())
                .expect("marker present");
            let state: Value = serde_json::from_str(found).expect("marker is JSON");
            assert_eq!(state["advances"], 1);
            assert_eq!(state["checkpoints"], 1);
            assert_eq!(state["step"], "crawl the first 100 pages");

            let missions = dir.join("missions");
            let id = active_id(&missions).unwrap().unwrap();
            let files: Vec<String> = std::fs::read_dir(missions.join(&id).join("checkpoints"))
                .unwrap()
                .flatten()
                .map(|e| e.file_name().to_string_lossy().to_string())
                .collect();
            assert_eq!(files.len(), 1, "{files:?}");
            assert!(files[0].contains("scaffold"), "{files:?}");
        });
    }

    /// The marker carries the board, not only the mission that just answered.
    ///
    /// The panel lists `missions` straight from the marker, and the shell's status
    /// push reads the same directories. A marker that described only its own
    /// mission would leave every other task on the machine invisible — which is
    /// precisely "I cannot see the task it created". The shape and the fields are
    /// what this asserts; what a given machine holds is not the test's business.
    #[test]
    fn the_marker_carries_every_mission_on_the_board() {
        in_workspace("board", |dir| {
            let record = json!({
                "id": "m-20260101-000000",
                "objective": "ship the crawler",
                "status": "running",
                "step": "write the test",
                "done_when": "the tests pass",
            });
            let mdir = dir.join("missions").join("m-20260101-000000");
            std::fs::create_dir_all(&mdir).unwrap();
            std::fs::write(mdir.join("mission.json"), record.to_string()).unwrap();

            let text = marker(&record, json!({}));
            let body = text
                .split("<!--mission-state ")
                .nth(1)
                .and_then(|rest| rest.split("-->").next())
                .expect("marker present");
            let state: Value = serde_json::from_str(body).expect("marker is JSON");
            assert_eq!(state["id"], "m-20260101-000000");
            let missions = state["missions"].as_array().expect("missions is an array");
            assert_eq!(missions.len(), 1, "{state}");
            assert_eq!(missions[0]["id"], "m-20260101-000000");
            assert_eq!(missions[0]["objective"], "ship the crawler");
            assert_eq!(missions[0]["status"], "running");
        });
    }

    #[test]
    fn advance_rewrites_only_the_generated_block() {
        in_workspace("fences", |dir| {
            start_mission(dir, "keep my prose");
            let missions = dir.join("missions");
            let id = active_id(&missions).unwrap().unwrap();
            let state_path = missions.join(&id).join("STATE.md");

            // The model owns everything outside the fences.
            let mine = "# 任务\n\n## 决策记录\n\n我决定不用队列。\n";
            std::fs::write(&state_path, format!("{mine}\n{FENCE_START}\nstale\n{FENCE_END}\n")).unwrap();

            advance(json!({ "summary": "changed the plan", "next": "redo it" })).expect("advance");
            let state = std::fs::read_to_string(&state_path).unwrap();
            assert!(state.contains("我决定不用队列"), "model prose survives:\n{state}");
            assert!(!state.contains("stale"), "generated block replaced:\n{state}");
            assert!(state.contains("redo it"), "{state}");

            // A STATE.md that lost its fences still gets a maintained block.
            std::fs::write(&state_path, "# 任务\n\n手写内容\n").unwrap();
            advance(json!({ "summary": "second", "next": "third" })).expect("advance");
            let state = std::fs::read_to_string(&state_path).unwrap();
            assert!(state.contains("手写内容"), "{state}");
            assert!(state.contains("third"), "{state}");
        });
    }

    #[test]
    fn recall_finds_lines_in_log_state_and_checkpoints() {
        in_workspace("recall", |dir| {
            start_mission(dir, "search me");
            advance(json!({
                "summary": "wired the zanzibar token through",
                "next": "test the zanzibar path"
            }))
            .expect("advance");
            checkpoint(json!({ "label": "zanzibar-done", "note": "zanzibar verified" }))
                .expect("checkpoint");

            let text = recall(json!({ "query": "zanzibar" })).expect("recall");
            assert!(text.contains("log.jsonl"), "{text}");
            assert!(text.contains("checkpoints/001-zanzibar-done.md"), "{text}");
            assert!(text.contains("hit(s)"), "{text}");

            // Two words must appear on the same line.
            let narrow = recall(json!({ "query": "zanzibar nonexistentword" })).expect("recall");
            assert!(narrow.contains("0 hit(s)"), "{narrow}");

            // An unknown mission directory is searched and simply matches nothing.
            let miss = recall(json!({ "query": "zanzibar", "mission": "m-nope" })).expect("recall");
            assert!(miss.contains("Nothing to search") || miss.contains("0 hit(s)"), "{miss}");
        });
    }

    #[test]
    fn mail_is_delivered_once_and_waits_for_a_recipient_that_is_not_running() {
        in_workspace("mail", |dir| {
            start_mission(dir, "coordinate");
            post(json!({ "to": "worker-a", "body": "the schema lives in db.rs" })).expect("post");
            post(json!({
                "to": "worker-a",
                "body": "do not touch migrations",
                "from": "worker-b"
            }))
            .expect("post");

            let first = inbox(json!({ "path": "worker-a" })).expect("inbox");
            assert!(first.contains("the schema lives in db.rs"), "{first}");
            assert!(first.contains("do not touch migrations"), "{first}");
            assert!(first.contains("worker-b"), "{first}");

            // Read once, not twice.
            let second = inbox(json!({ "path": "worker-a" })).expect("inbox");
            assert!(second.contains("no new mail"), "{second}");

            // Peeking does not consume.
            post(json!({ "to": "worker-a", "body": "third letter" })).expect("post");
            let peek = inbox(json!({ "path": "worker-a", "mark_read": false })).expect("inbox");
            assert!(peek.contains("third letter"), "{peek}");
            let again = inbox(json!({ "path": "worker-a" })).expect("inbox");
            assert!(again.contains("third letter"), "peek left it unread: {again}");

            // A mailbox address cannot name a path outside the mission.
            post(json!({ "to": "../escape", "body": "nope" })).expect("post");
            let escaped = dir.join("missions").join("escape.jsonl");
            assert!(!escaped.exists(), "mail stayed inside the mission");
        });
    }

    /// A mailbox that is truncated or rotated keeps delivering the letters that
    /// are still on disk.
    ///
    /// The watermark used to be a bare line count. After a rewrite the stale
    /// count either pointed past the new end or — clamped to it — marked
    /// letters that were *still there* as already read: they became unreachable
    /// forever, with no error and no hint. The read now notices that the prefix
    /// it remembers is not the prefix on disk, and starts again from the top.
    #[test]
    fn a_truncated_mailbox_still_shows_the_letters_that_are_left() {
        in_workspace("mail-truncate", |dir| {
            start_mission(dir, "rotate");
            for body in ["letter one", "letter two", "letter three"] {
                post(json!({ "to": "worker-a", "body": body })).expect("post");
            }
            let first = inbox(json!({ "path": "worker-a" })).expect("inbox");
            assert!(first.contains("letter three"), "{first}");
            assert!(!first.contains("rotated"), "a first read is not a rotation: {first}");

            // The mailbox is rewritten, as a cleanup or a rotation would: the
            // one letter left is a letter that was already read.
            let missions = dir.join("missions");
            let id = active_id(&missions).unwrap().expect("an active mission");
            let mailbox = mailbox_file(&missions.join(&id), "worker-a");
            let kept = std::fs::read_to_string(&mailbox).unwrap();
            let last = kept.lines().last().unwrap().to_string() + "\n";
            std::fs::write(&mailbox, last).unwrap();

            // `status` and `inbox` must agree about the same mailbox: one
            // letter on disk and an unverifiable mark is one unread letter,
            // not "0" from a stale count clamped to the new length.
            let status_text = status(json!({})).expect("status");
            assert!(status_text.contains("unread mail: 1"), "{status_text}");

            let second = inbox(json!({ "path": "worker-a" })).expect("inbox");
            assert!(
                second.contains("mailbox rotated, re-read from the top"),
                "the rewrite is reported, not swallowed: {second}"
            );
            assert!(
                second.contains("letter three"),
                "a letter still on disk is delivered again instead of vanishing: {second}"
            );
            assert!(
                !second.contains("letter one"),
                "a letter no longer on disk is not invented: {second}"
            );

            // The watermark caught up with the rewritten file, so the next
            // letter is an ordinary increment.
            post(json!({ "to": "worker-a", "body": "letter four" })).expect("post");
            let third = inbox(json!({ "path": "worker-a" })).expect("inbox");
            assert!(third.contains("letter four"), "{third}");
            assert!(!third.contains("rotated"), "it matches again: {third}");
            assert!(!third.contains("letter three"), "already read: {third}");
        });
    }

    /// Reading twice returns the increment only — the property the fingerprint
    /// must preserve rather than trade away for rotation detection.
    #[test]
    fn an_appended_mailbox_returns_only_the_increment() {
        in_workspace("mail-append", |dir| {
            start_mission(dir, "append");
            post(json!({ "to": "worker-a", "body": "first" })).expect("post");
            post(json!({ "to": "worker-a", "body": "second" })).expect("post");
            let first = inbox(json!({ "path": "worker-a" })).expect("inbox");
            assert!(first.contains("first") && first.contains("second"), "{first}");

            let again = inbox(json!({ "path": "worker-a" })).expect("inbox");
            assert!(again.contains("no new mail"), "{again}");
            assert!(
                !again.contains("rotated"),
                "an unchanged mailbox is not a rotation: {again}"
            );

            post(json!({ "to": "worker-a", "body": "third" })).expect("post");
            let increment = inbox(json!({ "path": "worker-a" })).expect("inbox");
            assert!(increment.contains("third"), "{increment}");
            assert!(!increment.contains("first"), "only the increment: {increment}");
            assert!(!increment.contains("second"), "only the increment: {increment}");
            assert!(increment.contains("1 unread"), "{increment}");
        });
    }

    /// A letter that arrives oversized is stored cut, and the sender is told so:
    /// a letter is charged to the recipient's context, so the door is where the
    /// bound has to be enforced.
    #[test]
    fn a_letter_longer_than_the_cap_is_stored_cut_and_says_so() {
        in_workspace("mail-cap", |dir| {
            start_mission(dir, "coordinate");
            let sent = LETTER_BODY_CAP * 3;
            let reply = post(json!({ "to": "worker-a", "body": "y".repeat(sent) }))
                .expect("post");
            assert!(
                reply.contains("the body was cut to"),
                "the sender learns about the cut: {reply}"
            );
            assert!(reply.contains(&sent.to_string()), "with the size sent: {reply}");

            let missions = dir.join("missions");
            let id = active_id(&missions).unwrap().expect("an active mission");
            let mailbox = mailbox_file(&missions.join(&id), "worker-a");
            let text = std::fs::read_to_string(&mailbox).unwrap();
            let letter: Value =
                serde_json::from_str(text.lines().next().expect("one letter")).unwrap();
            let stored = letter.get("body").and_then(|v| v.as_str()).expect("a body");
            assert!(
                stored.chars().count() <= LETTER_BODY_CAP + 1,
                "the stored body is bounded: {} chars",
                stored.chars().count()
            );
            assert!(stored.ends_with('…'), "the cut is visible in the letter");

            // A letter under the cap keeps its exact text and no warning.
            let short = post(json!({ "to": "worker-a", "body": "short one" })).expect("post");
            assert!(!short.contains("cut to"), "no warning when nothing was cut: {short}");
        });
    }

    /// One read renders a bounded batch, names what is left, and leaves exactly
    /// that remainder unread — a letter that was not shown must not be consumed.
    #[test]
    fn a_full_mailbox_renders_a_bounded_reply_and_leaves_the_rest_unread() {
        in_workspace("mail-full", |dir| {
            start_mission(dir, "many letters");
            let total = INBOX_LETTER_CAP + 50;
            for index in 0..total {
                post(json!({ "to": "worker-a", "body": format!("letter {index}") })).expect("post");
            }

            let first = inbox(json!({ "path": "worker-a" })).expect("inbox");
            assert_eq!(
                first.matches('·').count(),
                INBOX_LETTER_CAP,
                "the batch is bounded by the letter cap"
            );
            assert!(first.contains("letter 0"), "the oldest letter leads the batch");
            assert!(
                first.contains("more unread letter(s) not shown"),
                "the remainder is named: {first}"
            );
            assert!(
                !first.contains(&format!("letter {}", total - 1)),
                "the last letter is withheld, not rendered"
            );

            // The unread count matches what this read did not show.
            let status_text = status(json!({})).expect("status");
            assert!(
                status_text.contains("unread mail: 50"),
                "only the withheld letters stay unread: {status_text}"
            );

            let second = inbox(json!({ "path": "worker-a" })).expect("inbox");
            assert!(
                second.contains(&format!("letter {}", total - 1)),
                "the withheld letters arrive on the next read: {second}"
            );
            assert!(
                !second.contains("letter 0"),
                "a rendered letter is not delivered twice: {second}"
            );
            assert!(
                !second.contains("more unread letter(s)"),
                "the tail fits in one read: {second}"
            );
            let third = inbox(json!({ "path": "worker-a" })).expect("inbox");
            assert!(third.contains("no new mail"), "{third}");
        });
    }

    /// The character cap bounds a read even when few letters carry a lot of
    /// text: eight full-sized letters would otherwise render in one reply.
    #[test]
    fn a_few_huge_letters_are_rendered_only_up_to_the_character_cap() {
        in_workspace("mail-huge", |dir| {
            start_mission(dir, "big letters");
            for _ in 0..8 {
                post(json!({ "to": "worker-a", "body": "z".repeat(LETTER_BODY_CAP) }))
                    .expect("post");
            }
            let reply = inbox(json!({ "path": "worker-a" })).expect("inbox");
            assert!(
                reply.contains("more unread letter(s)"),
                "the read stops at the cap and says so"
            );
            assert!(
                reply.chars().count() < INBOX_RENDER_CAP + LETTER_BODY_CAP,
                "the reply is bounded by the cap plus the one letter that crossed it: \
                 {} chars",
                reply.chars().count()
            );
        });
    }

    /// A watermark written by an older build is a bare line count, which
    /// nothing can verify. It is not trusted: the letters come back once, and
    /// the mark is rewritten in the verifiable format.
    #[test]
    fn an_old_bare_count_watermark_migrates_instead_of_hiding_letters() {
        in_workspace("mail-legacy-mark", |dir| {
            start_mission(dir, "legacy mark");
            post(json!({ "to": "worker-a", "body": "first" })).expect("post");
            post(json!({ "to": "worker-a", "body": "second" })).expect("post");
            let first = inbox(json!({ "path": "worker-a" })).expect("inbox");
            assert!(first.contains("first") && first.contains("second"), "{first}");

            let missions = dir.join("missions");
            let id = active_id(&missions).unwrap().expect("an active mission");
            let mdir = missions.join(&id);
            let mark_file = offset_path(&mdir, "worker-a");
            let fresh = std::fs::read_to_string(&mark_file).expect("the new-format mark");
            assert!(fresh.contains('@'), "count@fingerprint is the new format: {fresh}");

            // What an older build left behind: the count, and nothing to check
            // it against.
            std::fs::write(&mark_file, "2").expect("downgrade the watermark");

            // Unverifiable, so it reads as a rewrite: the letters that are
            // still on disk are delivered again rather than hidden behind a
            // mark nothing can check.
            let again = inbox(json!({ "path": "worker-a" })).expect("inbox");
            assert!(again.contains("first") && again.contains("second"), "{again}");
            let upgraded = std::fs::read_to_string(&mark_file).expect("the rewritten mark");
            assert!(
                upgraded.contains('@'),
                "the old mark is migrated, not kept as a bare number: {upgraded}"
            );

            // From here it is an ordinary watermark again: only the increment.
            post(json!({ "to": "worker-a", "body": "third" })).expect("post");
            let third = inbox(json!({ "path": "worker-a" })).expect("inbox");
            assert!(third.contains("third"), "{third}");
            assert!(
                !third.contains("first") && !third.contains("second"),
                "only the increment: {third}"
            );
        });
    }

    #[test]
    fn a_missing_mission_is_guidance_and_a_corrupt_one_is_a_named_error() {
        in_workspace("robust", |dir| {
            let hint = status(json!({})).expect("status without a mission is fine");
            assert!(hint.contains("No active mission"), "{hint}");
            assert!(advance(json!({ "summary": "x" })).is_err(), "advance needs a mission");
            assert!(post(json!({ "to": "a", "body": "b" })).is_err(), "mail needs a mission");

            // A corrupt pointer names the file instead of panicking.
            let missions = dir.join("missions");
            std::fs::create_dir_all(&missions).unwrap();
            std::fs::write(missions.join("active.json"), "{ not json").unwrap();
            let error = status(json!({})).expect_err("corrupt pointer is an error");
            assert!(format!("{error:#}").contains("active.json"), "{error:#}");

            // A corrupt mission record names itself too.
            std::fs::write(missions.join("active.json"), "{\"mission\":\"m-x\"}").unwrap();
            std::fs::create_dir_all(missions.join("m-x")).unwrap();
            std::fs::write(missions.join("m-x").join("mission.json"), "oops").unwrap();
            let error = status(json!({})).expect_err("corrupt record is an error");
            assert!(
                format!("{error:#}").contains("mission.json"),
                "{error:#}"
            );

            // A mission whose files are gone still lists, rather than vanishing.
            let listed = recall(json!({ "query": "anything" })).expect("recall is tolerant");
            assert!(listed.contains("hit(s)"), "{listed}");
        });
    }

    /// The reader the agent loop keeps a turn open with.
    ///
    /// Everything except a running mission has to read as "no mission": the loop
    /// consults this between steps, and a reader that could not tell "finished"
    /// from "unreadable" would either abandon a live task or keep a dead one
    /// alive forever.
    #[test]
    fn running_mission_reports_only_a_running_mission() {
        in_workspace("brief-running", |dir| {
            start_mission(dir, "ship the crawler");
            let brief = running_mission(dir).expect("a running mission is reported");
            assert!(brief.id.starts_with("m-"), "{brief:?}");
            assert_eq!(brief.objective, "ship the crawler");
            assert_eq!(brief.status, "running");
            assert_eq!(brief.step, "write the failing test");
            assert_eq!(brief.done_when, "the tests pass");

            // The model blocking or finishing the mission is what lets a turn
            // end, so the reader follows a status change in the same workspace
            // rather than remembering the first answer it saw.
            advance(json!({ "summary": "the site is down", "status": "blocked" }))
                .expect("advance");
            assert!(running_mission(dir).is_none(), "blocked is not running");

            advance(json!({ "summary": "shipped", "status": "done" })).expect("advance");
            assert!(running_mission(dir).is_none(), "done is not running");
        });

        // Nothing on disk at all.
        let empty = std::env::temp_dir().join(format!(
            "ngu-mission-brief-empty-{}",
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&empty).unwrap();
        assert!(running_mission(&empty).is_none(), "no missions directory");
        let _ = std::fs::remove_dir_all(&empty);

        // A pointer nothing can parse, a record nothing can parse, and a record
        // that does not say what its status is: all "no mission", because this
        // runs inside a turn and must never be able to end one.
        in_workspace("brief-corrupt", |dir| {
            let missions = dir.join("missions");
            std::fs::create_dir_all(&missions).unwrap();
            std::fs::write(missions.join("active.json"), "{ not json").unwrap();
            assert!(running_mission(dir).is_none(), "a corrupt pointer is not a mission");

            std::fs::write(missions.join("active.json"), "{\"mission\":\"m-x\"}").unwrap();
            std::fs::create_dir_all(missions.join("m-x")).unwrap();
            std::fs::write(missions.join("m-x").join("mission.json"), "oops").unwrap();
            assert!(running_mission(dir).is_none(), "an unreadable record is not a mission");

            std::fs::write(
                missions.join("m-x").join("mission.json"),
                "{\"id\":\"m-x\",\"objective\":\"?\"}",
            )
            .unwrap();
            assert!(running_mission(dir).is_none(), "a missing status is not running");
        });
    }

    #[test]
    fn the_worker_prompt_carries_everything_a_memoryless_worker_needs() {
        let record = json!({
            "objective": "crawl the site",
            "done_when": "every page is fetched",
        });
        let prompt = worker_prompt(&record, "# 任务\n进度：两页已抓", "抓第三页");
        assert!(prompt.contains("crawl the site"));
        assert!(prompt.contains("every page is fetched"));
        assert!(prompt.contains("两页已抓"), "state travels with the step");
        assert!(prompt.contains("抓第三页"));
        assert!(prompt.contains("Next step:"), "the worker is asked for the next step");
        assert!(prompt.contains("no memory"), "the worker is told it has no history");
    }

    #[test]
    fn safe_segment_strips_anything_that_could_be_a_path() {
        assert_eq!(safe_segment("worker-a"), "worker-a");
        assert_eq!(safe_segment("a b"), "a_b");
        // A name that is only dots, or nothing at all, is not a name.
        assert_eq!(safe_segment(".."), "unnamed");
        assert_eq!(safe_segment("."), "unnamed");
        assert_eq!(safe_segment(""), "unnamed");
        // What matters is the property, not the exact spelling: no separators
        // survive, so the result can only ever be one path segment.
        for hostile in [
            "../../etc/passwd",
            "..\\..\\windows\\system32",
            "/absolute/path",
            "a/../b",
            "~/.ssh/id_rsa",
        ] {
            let cleaned = safe_segment(hostile);
            assert!(!cleaned.contains('/'), "{hostile} → {cleaned}");
            assert!(!cleaned.contains('\\'), "{hostile} → {cleaned}");
            assert_ne!(cleaned, "..", "{hostile} → {cleaned}");
            assert_ne!(cleaned, ".");
            assert!(!cleaned.is_empty(), "{hostile} → {cleaned}");
        }
    }
}
