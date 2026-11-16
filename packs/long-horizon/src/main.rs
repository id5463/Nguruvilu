//! Long-horizon missions, as a process.
//!
//! This was `src/tools/longhorizon.rs` — 1,979 lines compiled into the kernel,
//! plus 26,000 characters of panel HTML in a Rust string. Both are gone from the
//! kernel: the tools are this executable, and the panel is
//! `packs/long-horizon/look.json`.
//!
//! # What changed in the move
//!
//! Almost nothing. The mission logic — the record on disk, the generated block in
//! `STATE.md`, the append-only log, checkpoints, mailboxes, `recall` — is the
//! code that was there, moved. Three things had to become questions:
//!
//! * **The workspace.** It came from a thread-local the kernel pinned per tool
//!   call; a plugin is a separate process and has no such thread. It asks the
//!   kernel: `{"ask":{"state":"workspace"}}`.
//! * **Tool definitions.** `ToolDef`/`ToolFuture` were kernel types. Here a tool
//!   is a name, a description, a schema, and a function — which is all the
//!   protocol ever needed.
//! * **The panel.** It sent a prompt through the page; it still does, and that is
//!   why the panel needed no rewrite: a panel talks to the page, and the page
//!   talks to the agent.
//!
//! # Why `state` and not an environment variable
//!
//! A workspace fixed at start-up is wrong for a long task: the session can move,
//! and a mission that keeps writing into the old directory is a mission nobody
//! can find. Asking each time costs one round trip (0.25 ms) and is correct.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use serde_json::{json, Value};

/// One tool this plugin provides: the shape `describe` returns and `call` runs.
struct Tool {
    name: &'static str,
    description: &'static str,
    parameters: Value,
    run: fn(Value) -> Result<String>,
}

/// Ask the kernel for one of its state keys.
fn ask_state(question: &Value) -> Option<Value> {
    let body = serde_json::to_vec(&json!({ "ask": question })).ok()?;
    let mut out = std::io::stdout();
    out.write_all(&(body.len() as u32).to_be_bytes()).ok()?;
    out.write_all(&body).ok()?;
    out.flush().ok()?;
    let mut stdin = std::io::stdin();
    let frame = read_frame(&mut stdin)?;
    let answer: Value = serde_json::from_slice(&frame).ok()?;
    answer.get("value").cloned()
}

/// The session workspace the kernel reports, or `None` when it has none.
fn workspace() -> Option<PathBuf> {
    ask_state(&json!({ "state": "workspace" }))
        .and_then(|value| value.as_str().map(PathBuf::from))
}

fn main() {
    let tools = tools();
    let mut stdin = std::io::stdin();
    loop {
        let frame = match read_frame(&mut stdin) {
            Some(frame) => frame,
            None => return,
        };
        let message: Value = match serde_json::from_slice(&frame) {
            Ok(value) => value,
            Err(error) => {
                write_frame(&json!({ "error": format!("not JSON: {error}") }));
                continue;
            }
        };
        let answer = match message.get("op").and_then(|op| op.as_str()) {
            Some("describe") => json!({
                "tools": tools
                    .iter()
                    .map(|tool| json!({
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    }))
                    .collect::<Vec<_>>(),
                // The panel is data in the pack; this plugin does not build it.
                "settings_key": Value::Null,
            }),
            Some("call") => {
                let name = message.get("tool").and_then(|t| t.as_str()).unwrap_or("");
                let args = message.get("args").cloned().unwrap_or_else(|| json!({}));
                match tools.iter().find(|tool| tool.name == name) {
                    Some(tool) => match (tool.run)(args) {
                        Ok(text) => json!({ "result": text }),
                        Err(error) => json!({ "error": format!("{error:#}") }),
                    },
                    None => json!({ "error": format!("no such tool: {name}") }),
                }
            }
            Some(other) => json!({ "error": format!("unsupported op: {other}") }),
            None => json!({ "error": "missing op" }),
        };
        write_frame(&answer);
    }
}

fn read_frame(input: &mut impl Read) -> Option<Vec<u8>> {
    let mut header = [0u8; 4];
    input.read_exact(&mut header).ok()?;
    let len = u32::from_be_bytes(header) as usize;
    let mut body = vec![0u8; len];
    input.read_exact(&mut body).ok()?;
    Some(body)
}

fn write_frame(value: &Value) {
    let body = match serde_json::to_vec(value) {
        Ok(body) => body,
        Err(_) => return,
    };
    let mut out = std::io::stdout();
    out.write_all(&(body.len() as u32).to_be_bytes()).ok();
    out.write_all(&body).ok();
    out.flush().ok();
}

/// The seven tools, with the same names and arguments the kernel version had.
fn tools() -> Vec<Tool> {
    vec![
        Tool {
            name: START,
            description: "Start a long-horizon mission: a task entity on disk (objective, done-when, next step), a STATE.md the model owns, and an append-only log. While it is `running` the turn does not end; each step is recorded and the next one begins.",
            parameters: json!({
                "type": "object",
                "properties": {
                    "objective": { "type": "string", "description": "What the mission is for." },
                    "done_when": { "type": "string", "description": "How to tell it is finished." },
                    "first_step": { "type": "string", "description": "The first concrete action." }
                },
                "required": ["objective"]
            }),
            run: start,
        },
        Tool {
            name: STATUS,
            description: "Report the active mission's state, its recent advances, and the board of every mission on disk.",
            parameters: json!({ "type": "object", "properties": {} }),
            run: status,
        },
        Tool {
            name: ADVANCE,
            description: "Record one completed step: what was done (summary) and what comes next (next). Mark completion or a block with `status`. This is how a mission makes progress — do it every step, and the turn continues while it is `running`.",
            parameters: json!({
                "type": "object",
                "properties": {
                    "summary": { "type": "string", "description": "What this step did." },
                    "next": { "type": "string", "description": "The next concrete step." },
                    "status": { "type": "string", "description": "running (default), done, or blocked." }
                },
                "required": ["summary"]
            }),
            run: advance,
        },
        Tool {
            name: CHECKPOINT,
            description: "Snapshot the mission's directory so a bad turn can be undone. Without `restore` it makes one; with it, that checkpoint is put back.",
            parameters: json!({
                "type": "object",
                "properties": {
                    "label": { "type": "string", "description": "Name for this checkpoint." },
                    "restore": { "type": "string", "description": "Restore this checkpoint instead of making one." }
                }
            }),
            run: checkpoint,
        },
        Tool {
            name: RECALL,
            description: "Search this mission's own files by line — the log, STATE.md, anything under the mission directory. Use it instead of re-reading a long log: the recent past is what a long task usually needs.",
            parameters: json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "Text to find (case-insensitive)." },
                    "file": { "type": "string", "description": "Limit to one file under the mission." },
                    "limit": { "type": "integer", "description": "Maximum lines to return." }
                },
                "required": ["query"]
            }),
            run: recall,
        },
        Tool {
            name: POST,
            description: "Send a message to another agent working on this mission. Messages live in a mailbox under the mission directory, so they survive compaction and restarts.",
            parameters: json!({
                "type": "object",
                "properties": {
                    "to": { "type": "string", "description": "Recipient name." },
                    "text": { "type": "string", "description": "The message." }
                },
                "required": ["to", "text"]
            }),
            run: post,
        },
        Tool {
            name: INBOX,
            description: "Read messages other agents sent this agent about the mission. Unread messages are marked read once reported.",
            parameters: json!({
                "type": "object",
                "properties": {
                    "name": { "type": "string", "description": "Whose inbox; defaults to `root`." }
                }
            }),
            run: inbox,
        },
    ]
}

// ---------------------------------------------------------------- mission logic
// Everything below is the code that lived in the kernel, with `ToolDef` gone and
// the workspace resolved through `workspace()`.

/// The plugin name a pack asks for with `builtin:long-horizon`.
pub const PLUGIN: &str = "long-horizon";

/// Tool names.
pub const START: &str = "mission_start";
pub const STATUS: &str = "mission_status";
pub const ADVANCE: &str = "mission_advance";
pub const CHECKPOINT: &str = "mission_checkpoint";
pub const RECALL: &str = "recall";
pub const POST: &str = "agent_post";
pub const INBOX: &str = "agent_inbox";

/// Most lines `recall` returns in one call.
const RECALL_LIMIT: usize = 40;

/// Bytes read from one file while searching. Longer files are searched from
/// the end: the recent past is what a long task usually needs.
const RECALL_FILE_CAP: u64 = 4 * 1024 * 1024;

/// Log text kept per entry, so one runaway report cannot bloat the log.
const LOG_TEXT_CAP: usize = 8 * 1024;

/// The fences around the block `mission_advance` maintains inside `STATE.md`.
/// Everything outside them belongs to the model and is never rewritten.
const FENCE_START: &str = "<!-- mission:generated -->";
const FENCE_END: &str = "<!-- /mission:generated -->";

/// Register the plugin. Defining is not loading: nothing appears in a tool
/// table until a pack's assembly asks for `builtin:long-horizon`.

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
/// The mission bar and its script live in the pack now, not here.
///
/// They were `fn panel()` — 26,000 characters of HTML and JavaScript inside a
/// Rust string, which meant a pack could not change its own interface, and the
/// only way to move a button was to rebuild the kernel. `packs/long-horizon/
/// look.json` carries them as data; the shell reads that file for every pack, so
/// this one is not special.
///
/// What remains in this file is the mission logic. The panel asks for it by
/// sending a prompt ("调用 mission_status …"), which is what a panel can do
/// without the kernel knowing what a mission is.


// ------------------------------------------------------------------- tools








// ------------------------------------------------------------ path plumbing

/// `<workspace>/missions`, created on demand.
fn missions_root() -> Result<PathBuf> {
    let workspace = workspace().ok_or_else(|| {
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
/// characters is the budget the panel already gives one line, and it is enough
/// to say what the mission is without turning every batch of a long turn into a
/// copy of `STATE.md`.
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
        Some(step) => text.push_str(&format!("  next step: {step}\n")),
        None => text.push_str(&format!(
            "\nNo step yet. Record one with {ADVANCE} (summary + next).\n"
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
                Some("advance") | Some("checkpoint")
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
        "  advances {} · checkpoints {}",
        record.get("advances").and_then(|v| v.as_u64()).unwrap_or(0),
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

