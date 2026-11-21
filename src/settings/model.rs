//! What a settings file holds, and the defaults its missing keys take.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::io::mask;

/// Default model when nothing is configured.
pub const DEFAULT_MODEL: &str = "deepseek-v4.1-flash";

/// Settings as stored on disk.
///
/// Unknown keys are **kept**, not rejected. `deny_unknown_fields` used to make
/// one unrecognised key — a hand-typed one, a plugin's, or one written by a
/// newer build — fail the whole parse, and [`Settings::load`] answered a failed
/// parse with [`Settings::default`]. The next `ngu config set` then wrote that
/// default back over the file, so a single typo cost the user their `api_key`,
/// model, network tuning, judge credentials at once. The unknown keys
/// land in [`Settings::unknown`] and are written back out unchanged.
#[derive(Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    /// API base URL including the version segment, e.g. `https://host/v1`.
    #[serde(default)]
    pub base_url: String,
    /// API key.
    #[serde(default)]
    pub api_key: String,
    /// Model id.
    #[serde(default)]
    pub model: String,
    /// Reasoning effort: `minimal`, `low`, `medium`, `high`, or empty to let
    /// the provider decide.
    ///
    /// This is the main cost lever on a reasoning model. Measured against a
    /// real endpoint, `minimal` cut one answer from 137 output tokens to 69.
    #[serde(default)]
    pub reasoning_effort: String,
    /// How the answer is delivered: `stream`, `buffered`, or empty for `stream`.
    ///
    /// `buffered` exists for gateways that accept a streaming request, open the
    /// response, and then never write the answer — see the module docs in
    /// [`crate::llm`]. On such an endpoint the only alternative is a turn that
    /// hangs until the whole-request timeout.
    #[serde(default)]
    pub delivery: String,
    /// Whether a stalled stream switches this route to buffered delivery.
    ///
    /// Absent means yes: one stalled step is enough evidence to stop paying the
    /// stall bound on every later step. Set to `false` to keep asking the stream
    /// regardless.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub degrade_on_stall: Option<bool>,
    /// Proxy for all requests, e.g. `http://127.0.0.1:7890`.
    ///
    /// Empty means direct, and — more importantly — it means the process's
    /// `HTTP_PROXY`/`HTTPS_PROXY` variables are ignored. Leaving those to be
    /// picked up implicitly is how a proxy configured for some other tool
    /// silently reroutes model traffic and produces a `tls handshake eof` that
    /// mentions nothing about proxies.
    #[serde(default)]
    pub proxy: String,
    /// Context window in tokens.
    ///
    /// Empty means "work it out": the provider's own number when it reports one,
    /// a built-in table for well-known models, otherwise a conservative floor.
    /// Setting it explicitly always wins.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_window: Option<usize>,
    /// Compact once this percentage of the window is in use.
    #[serde(default = "default_compact_percent")]
    pub compact_percent: u32,
    /// Recent messages kept verbatim when compacting.
    #[serde(default = "default_keep_recent")]
    pub compact_keep_recent: usize,
    /// Most model steps one turn may take before the loop stops and reports.
    ///
    /// Defaults to the loop's own 50 rather than to `0`: a file that never
    /// mentions this key must not describe a turn that is not allowed to run.
    #[serde(default = "default_max_steps")]
    pub max_steps: u32,
    /// Tool deadline in seconds — the ceiling on one foreground `bash` call.
    ///
    /// It is both what a call gets when the model asks for nothing, and the most
    /// a model may ask for: an argument above it is cut to this value and the
    /// answer says so. It used to be only the default, and a measured session
    /// showed requests of 300000, 600000 and 1500000 ms all honoured — one call
    /// holding a turn for twenty-five minutes because a number was typed.
    ///
    /// Not a ceiling on everything: `background: true` jobs live until
    /// `job_kill`, and a subagent or a pack tool may run long on purpose — that
    /// work is a deliberate exception, not a leak.
    ///
    /// Seconds rather than milliseconds because the field is written by hand and
    /// by the settings panel, and both count in seconds.
    #[serde(default = "default_tool_timeout_secs")]
    pub tool_timeout_secs: u64,

    /// How long one MCP request may take — the deadline that used to be a
    /// hardcoded 60 seconds inside the loader, invisible in every interface.
    #[serde(default = "default_mcp_timeout_secs")]
    pub mcp_timeout_secs: u64,

    /// Model rounds a subagent (`delegate`) may take. Was `MAX_STEPS = 12`.
    #[serde(default = "default_subagent_steps")]
    pub subagent_max_steps: usize,

    /// Wall clock a subagent may run. Was `WALL_BUDGET_SECS = 15 * 60`.
    #[serde(default = "default_subagent_wall_secs")]
    pub subagent_wall_secs: u64,

    /// How deep subagents may nest. Was `MAX_DEPTH = 2`.
    #[serde(default = "default_subagent_depth")]
    pub subagent_max_depth: usize,

    /// The largest `wait_ms` a job read may block for. Was `MAX_WAIT_MS = 30_000`.
    #[serde(default = "default_job_wait_secs")]
    pub job_wait_secs: u64,

    /// Cloud relay URL the local app dials out to for remote control.
    ///
    /// Empty leaves remote control off. No default host is assumed.
    #[serde(default)]
    pub relay_url: String,
    /// Bearer token presented to the relay. Empty leaves remote control off.
    #[serde(default)]
    pub relay_token: String,

    /// Output token ceiling, when the provider accepts one.
    ///
    /// Sent as max_completion_tokens. Reasoning tokens count against it, so
    /// a value that looks generous for the answer can still be tight for a
    /// model that thinks first.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<usize>,

    /// How requests are sent: timeouts, pooling, retries.
    ///
    /// A pack's `models.json` may set these, and a plugin publishing the
    /// `network` service overrides both.
    #[serde(default)]
    pub network: crate::network::NetworkSettings,

    /// How much of a tool's output stays in the request, and for how long.
    #[serde(default)]
    pub tool_output: ToolOutputSettings,

    /// Settings a pack owns, keyed by pack name.
    ///
    /// # Why this replaced four named fields
    ///
    /// `skill_mode`, `mcp_proxy_mode`, `mcp_proxy_targets` and `judge` used to
    /// live here by name. Each one was the kernel knowing about one specific
    /// pack: a third-party pack that wanted a setting had to either add a field
    /// to this struct or keep its own file. The first is editing the kernel, and
    /// the second loses the settings panel.
    ///
    /// So the kernel stores an arbitrary JSON object per pack name and
    /// **interprets none of it**. A pack reads and writes `packs["its-name"]`
    /// itself, through `PluginCtx::config` at load time or whatever API it
    /// chooses at runtime. The kernel's only job is to round-trip the bytes.
    ///
    /// The old keys still work: `Settings::migrate_legacy_pack_keys` folds them
    /// in on load, so an existing settings file loses nothing.
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub packs: std::collections::BTreeMap<String, serde_json::Value>,

    /// Extra fields merged into every request body.
    ///
    /// This is how a provider difference stays a settings change: an endpoint
    /// that wants enable_thinking, top_k, or any other field gets it here.
    /// A null value removes a field the kernel would otherwise send.
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub extra_body: serde_json::Map<String, serde_json::Value>,

    /// Keys this build does not know about, kept verbatim.
    ///
    /// Round-trip rather than validation: a field written by a newer build, a
    /// plugin, or a hand edit must survive being read and written by an older
    /// one. This is also what replaces `deny_unknown_fields` — an unknown key is
    /// stored here instead of being a parse error.
    #[serde(flatten, skip_serializing_if = "BTreeMap::is_empty")]
    pub unknown: BTreeMap<String, serde_json::Value>,
}

impl std::fmt::Debug for Settings {
    /// Written by hand so a secret never reaches a log.
    ///
    /// `#[derive(Debug)]` would print `api_key` and `relay_token` in full, and
    /// the first `eprintln!("{settings:?}")` added while chasing a route problem
    /// would put both into a terminal scrollback and every bug report pasted
    /// from it. Everything a diagnostic actually needs is here; the credentials
    /// are not. Nested settings print as presence only, so this impl never
    /// depends on how those types choose to render themselves.
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Settings")
            .field("base_url", &self.base_url)
            .field("api_key", &mask(&self.api_key))
            .field("model", &self.model)
            .field("reasoning_effort", &self.reasoning_effort)
            .field("delivery", &self.delivery)
            .field("degrade_on_stall", &self.degrade_on_stall)
            .field("proxy", &self.proxy)
            .field("context_window", &self.context_window)
            .field("compact_percent", &self.compact_percent)
            .field("compact_keep_recent", &self.compact_keep_recent)
            .field("max_steps", &self.max_steps)
            .field("relay_url", &self.relay_url)
            .field("relay_token", &mask(&self.relay_token))
            .field("max_output_tokens", &self.max_output_tokens)
            .field("packs", &self.packs.keys().collect::<Vec<_>>())
            .field("extra_body", &self.extra_body.keys().collect::<Vec<_>>())
            .field("unknown", &self.unknown.keys().collect::<Vec<_>>())
            .finish()
    }
}

fn default_compact_percent() -> u32 {
    75
}

/// How much of a tool's output stays in the request, and for how long.
///
/// Two dials for one measured problem: a long session re-sends every tool result
/// it ever produced, so the per-step prompt grows to the size of that history.
/// A measured session of 2,749 steps carried ~360k tokens per step, and its 3,245
/// tool results added up to almost exactly that figure.
///
/// `head_chars`/`tail_chars` bound one *old* result; `keep_recent` decides how
/// many of the newest stay exact. Both are view-time only — the session file
/// keeps every original byte, so raising the dials restores detail immediately.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolOutputSettings {
    /// Whether old tool results are thinned at all.
    #[serde(default = "tool_output_enabled")]
    pub enabled: bool,
    /// Characters of the head of an old result that stay.
    #[serde(default = "default_prune_head")]
    pub head_chars: usize,
    /// Characters of the tail of an old result that stay.
    #[serde(default = "default_prune_tail")]
    pub tail_chars: usize,
    /// How many of the newest results stay untouched.
    #[serde(default = "default_prune_keep")]
    pub keep_recent: usize,
    /// How many of the newest inline images stay in the request.
    #[serde(default = "default_prune_images")]
    pub keep_images: usize,
}

impl Default for ToolOutputSettings {
    fn default() -> Self {
        Self {
            enabled: tool_output_enabled(),
            head_chars: default_prune_head(),
            tail_chars: default_prune_tail(),
            keep_recent: default_prune_keep(),
            keep_images: default_prune_images(),
        }
    }
}

impl ToolOutputSettings {
    /// The policy these settings describe.
    ///
    /// The two *counts* treat `0` as "not written" and fall back to the shipped
    /// values: a panel box left empty, or a file written by an older build, must
    /// not silently mean "thin every result in the history" — that turns a
    /// context lever into a memory hole. `head_chars`/`tail_chars` keep `0` as a
    /// real choice, because head 0 + tail 0 is the deliberate "pointer only"
    /// preset.
    pub fn policy(&self) -> crate::compaction::PrunePolicy {
        crate::compaction::PrunePolicy {
            enabled: self.enabled,
            head_chars: self.head_chars,
            tail_chars: self.tail_chars,
            // 0 就是 0。以前这里把 0 当成"没写"、回落到 8/2，那是为了绕开面板把空
            // 输入框写成 0 的老毛病；面板现在不再那么干了，所以"我手动写 0"必须
            // 被照字面执行。缺失的字段仍然由 serde 的 default 补，两件事分开。
            keep_recent: self.keep_recent,
            keep_images: self.keep_images,
        }
    }
}

fn tool_output_enabled() -> bool {
    true
}

pub(super) fn default_prune_images() -> usize {
    2
}

fn default_prune_head() -> usize {
    4_000
}

fn default_prune_tail() -> usize {
    1_000
}

pub(super) fn default_prune_keep() -> usize {
    8
}

fn default_keep_recent() -> usize {
    8
}

fn default_max_steps() -> u32 {
    50
}

fn default_tool_timeout_secs() -> u64 {
    120
}

fn default_mcp_timeout_secs() -> u64 {
    60
}

fn default_subagent_steps() -> usize {
    12
}

fn default_subagent_wall_secs() -> u64 {
    15 * 60
}

fn default_subagent_depth() -> usize {
    2
}

fn default_job_wait_secs() -> u64 {
    30
}

impl Settings {
    /// One pack's stored settings, or `Null` when it has never written any.
    ///
    /// The kernel's only accessor over [`Settings::packs`]: it hands back
    /// whatever is there and asks nothing about the shape, because the shape is
    /// the pack's business.
    pub fn pack(&self, name: &str) -> serde_json::Value {
        self.packs.get(name).cloned().unwrap_or(serde_json::Value::Null)
    }

    /// Store one pack's settings.
    pub fn set_pack(&mut self, name: impl Into<String>, value: serde_json::Value) {
        self.packs.insert(name.into(), value);
    }

    /// Move a pre-`packs` settings file's top-level pack keys into [`Settings::packs`].
    ///
    /// This struct used to carry four fields that belonged to three packs:
    /// `skill_mode`, `mcp_proxy_mode`, `mcp_proxy_targets` and `judge`. An
    /// existing settings file has them at the top level, where `serde` now files
    /// them under [`Settings::unknown`], and dropping them would lose a key the
    /// user typed.
    ///
    /// # Why this is a table of strings and not three pack names
    ///
    /// It has to name them: `skill_mode` is not derivable from `skill-router`,
    /// and the whole point is to move values written by a build that knew both
    /// names. This is the one place left in the kernel that says a pack called
    /// `judge` exists — and it is a **migration**, not a dependency: it runs on
    /// a file written before the change and does nothing on any file after.
    ///
    /// The honest fix is to delete this once the old files are gone; a
    /// version-numbered settings file would have made it unnecessary from the
    /// start. It is kept, bounded and commented, rather than spread through
    /// three host call sites the way it was.
    ///
    /// Idempotent: a file already migrated has nothing to move, and a pack that
    /// has since written its own entry is not overwritten by an old key.
    pub fn migrate_legacy_pack_keys(&mut self) {
        use serde_json::{json, Value};

        // `<old top-level key>` → `<pack>` and, when the old key's name is not
        // the new one, `<new key inside that pack>`.
        const MIGRATIONS: &[(&str, &str, &str)] = &[
            ("skill_mode", "skill-router", "skill_mode"),
            ("mcp_proxy_mode", "mcp-proxy", "mcp_proxy_mode"),
            ("mcp_proxy_targets", "mcp-proxy", "mcp_proxy_targets"),
            ("judge", "judge", "judge"),
        ];

        for (old_key, pack, new_key) in MIGRATIONS {
            let Some(value) = self.unknown.get(*old_key).cloned() else {
                continue;
            };
            // Never overwrite: a pack that has run since the upgrade owns its
            // own entry, and its value is newer than the leftover.
            let entry = self
                .packs
                .entry((*pack).to_string())
                .or_insert_with(|| Value::Object(Default::default()));
            match entry.as_object_mut() {
                Some(map) => {
                    map.entry((*new_key).to_string()).or_insert(value);
                }
                None => *entry = json!({ *new_key: value }),
            }
        }

        // The old keys are moved, not copied: leaving them would make every
        // later load re-migrate from a stale copy.
        for (old_key, _, _) in MIGRATIONS {
            self.unknown.remove(*old_key);
        }
    }

    /// The runtime budgets these settings describe.
    ///
    /// Every accessor treats `0` as "not written" and falls back to the shipped
    /// value: a settings file that predates a field, or one where somebody
    /// cleared a box, must not hand out instant deadlines.
    pub fn budgets(&self) -> crate::tools::Budgets {
        // `0` means "not written" — a cleared box or a file that predates the
        // field falls back to the shipped number rather than an instant budget.
        let seconds = |value: u64, fallback: u64| -> u64 {
            if value == 0 {
                fallback
            } else {
                value.min(24 * 3600)
            }
        };
        let count = |value: usize, fallback: usize| -> usize {
            if value == 0 {
                fallback
            } else {
                value.min(1_000_000)
            }
        };
        crate::tools::Budgets {
            tool_deadline_ms: seconds(self.tool_timeout_secs(), default_tool_timeout_secs()) * 1000,
            mcp_timeout_ms: seconds(self.mcp_timeout_secs, default_mcp_timeout_secs()) * 1000,
            subagent_max_steps: count(self.subagent_max_steps, default_subagent_steps()),
            subagent_wall_ms: seconds(self.subagent_wall_secs, default_subagent_wall_secs()) * 1000,
            subagent_max_depth: count(self.subagent_max_depth, default_subagent_depth()),
            job_wait_max_ms: seconds(self.job_wait_secs, default_job_wait_secs()) * 1000,
        }
    }

    /// Where one budget's setting lives, for the note that names it when it bites.
    pub fn budget_keys() -> [(&'static str, &'static str); 6] {
        [
            ("tool deadline", "tool-timeout"),
            ("MCP request", "mcp-timeout"),
            ("subagent steps", "subagent-steps"),
            ("subagent wall clock", "subagent-wall"),
            ("subagent depth", "subagent-depth"),
            ("job wait", "job-wait"),
        ]
    }

    /// Whether the settings are complete enough to make a request.
    pub fn is_configured(&self) -> bool {
        !self.base_url.trim().is_empty() && !self.api_key.trim().is_empty()
    }

    /// What is missing, in the words a settings panel can show.
    pub fn missing(&self) -> Vec<&'static str> {
        let mut out = Vec::new();
        if self.base_url.trim().is_empty() {
            out.push("base_url");
        }
        if self.api_key.trim().is_empty() {
            out.push("api_key");
        }
        if self.model.trim().is_empty() {
            out.push("model");
        }
        out
    }

    /// The model to use, falling back to the default.
    pub fn model_or_default(&self) -> String {
        if self.model.trim().is_empty() {
            DEFAULT_MODEL.to_string()
        } else {
            self.model.clone()
        }
    }

    /// Build the context policy these settings describe.
    ///
    /// This is the built-in policy; a plugin that provides `context.policy`
    /// replaces it at runtime without touching these settings.
    pub fn context_policy(&self, provider_window: Option<usize>) -> crate::window::DefaultContextPolicy {
        crate::window::DefaultContextPolicy {
            configured: self.context_window.filter(|t| *t > 0),
            // The threshold is clamped here as well as in the policy: a
            // hand-edited `600` would otherwise mean "never compact" (and a `0`
            // "compact on every step"), and both are written by hand in a file
            // this program does not prompt for.
            threshold_percent: self.compact_percent.clamp(1, 95),
            keep_recent: self.compact_keep_recent.max(1),
            provider_window: provider_window.filter(|t| *t > 0),
            // The output ceiling is part of the window: a prompt that fills the
            // whole window leaves no room for the answer, and the provider
            // rejects the request instead of compacting.
            reserve: self.max_output_tokens.unwrap_or(0),
            prune: self.tool_output.policy(),
        }
    }

    /// The step cap one turn may use, in the range the loop can honour.
    ///
    /// `0` reads as "not set" and means the default, not "no steps": this type
    /// derives `Default`, so every numeric field starts at zero and the field
    /// defaults only exist inside serde — a fresh install with no file would
    /// otherwise get a cap of one step, i.e. a turn that cannot finish a single
    /// tool round. Above the ceiling the value is clamped, because a turn bills
    /// for every step it takes and the loop's continuation behaviour already
    /// multiplies whatever is asked for here.
    pub fn max_steps(&self) -> usize {
        match self.max_steps {
            0 => default_max_steps() as usize,
            steps => (steps as usize).min(200),
        }
    }

    /// The default tool deadline, in seconds, in the range a call can honour.
    ///
    /// `0` reads as "not set" and means the built-in default, not "no deadline":
    /// this type derives `Default`, so a fresh install with no file would
    /// otherwise get a zero-second deadline, i.e. a tool call that cannot even
    /// start. The model's own per-call `timeout_ms` still wins over this.
    pub fn tool_timeout_secs(&self) -> u64 {
        match self.tool_timeout_secs {
            0 => default_tool_timeout_secs(),
            secs => secs.min(6 * 3600),
        }
    }

    /// The key with everything but its first and last few characters hidden.
    ///
    /// The settings panel has to show *something* so a user can tell whether a
    /// key is present, without putting the whole secret on screen.
    pub fn masked_key(&self) -> String {
        mask(&self.api_key)
    }
}
