//! The `judge` tool: a decision model in the loop, not a gate in the way.
//!
//! Jev (TypeSafe AI's "System One" model) answers typed questions about a
//! block of state: a choice, a score, or a yes/no probability — with a
//! confidence derived from the answer distribution. It does not chat; a
//! full request runs in 70–500 ms, which is what makes "ask before deciding"
//! affordable inside a turn.
//!
//! # The red line
//!
//! The judge is an **adviser, never a gate**: its answer arrives in the
//! transcript as data, and the model (and the user) decide what it means.
//! A `0` is reported, not enforced — this kernel refuses no action on a
//! judge's say-so, and the tool's own description says so. That is the same
//! "no gates, report honestly" rule the rest of the system runs under.
//!
//! # Contract
//!
//! `POST <endpoint>/v1/systemone` with `Authorization: Bearer <key>`:
//!
//! ```json
//! { "model": "jev-latest",
//!   "state": "…anything to judge…",
//!   "questions": { "q": { "type": "choice", "instructions": "…",
//!                          "choices": ["a", "b"] } } }
//! ```
//!
//! Answer shapes vary per question type (`choice` carries `choice`,
//! `noul` a boolean, `score` a number — each with `confidence` and usually
//! `probabilities`), so answers are flattened generically: a verdict field is
//! picked when present, and anything unrecognised is passed through as raw
//! JSON rather than guessed at.

use anyhow::{anyhow, Context, Result};
use serde_json::{json, Value};

/// The tool name.
pub const TOOL: &str = "judge";

/// The plugin's name: what a pack asks for as `builtin:judge`.
pub const PLUGIN: &str = "judge";

/// The evaluate endpoint's path under the endpoint host.
const PATH: &str = "/v1/systemone";

/// How the judge is reached and which model answers.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct JudgeSettings {
    /// API base, e.g. `https://api.typesafe.ai` — no path.
    pub endpoint: String,
    /// Bearer key. Never enters a pack; the panel only ever sees "(set)".
    pub api_key: String,
    /// Model alias; `jev-latest` tracks whatever the provider calls current.
    pub model: String,
}

impl Default for JudgeSettings {
    fn default() -> Self {
        Self {
            endpoint: "https://api.typesafe.ai".into(),
            api_key: String::new(),
            model: "jev-latest".into(),
        }
    }
}

impl JudgeSettings {
    /// Usable only with a key: an unauthenticated call is a guaranteed 401,
    /// and a tool that always fails costs a turn every time it is tried.
    pub fn is_configured(&self) -> bool {
        !self.api_key.trim().is_empty()
    }

    /// The full evaluate URL.
    fn url(&self) -> String {
        format!(
            "{}{}",
            self.endpoint.trim_end_matches('/'),
            PATH
        )
    }
}

/// Environment fallbacks, mirroring how the model key and search arrive:
/// a user who has never opened a panel can still export three variables.
pub fn from_env() -> Option<JudgeSettings> {
    let api_key = std::env::var("NGU_JUDGE_API_KEY").unwrap_or_default();
    if api_key.trim().is_empty() {
        return None;
    }
    let mut settings = JudgeSettings::default();
    settings.api_key = api_key;
    if let Ok(endpoint) = std::env::var("NGU_JUDGE_ENDPOINT") {
        if !endpoint.trim().is_empty() {
            settings.endpoint = endpoint.trim().to_string();
        }
    }
    if let Ok(model) = std::env::var("NGU_JUDGE_MODEL") {
        if !model.trim().is_empty() {
            settings.model = model.trim().to_string();
        }
    }
    Some(settings)
}

/// Where the judge's settings live in the pack table.
///
/// The same string the plugin uses. Kept in the kernel because the settings file
/// is the kernel's: a panel that edits it needs to know the key, and the plugin
/// that reads it needs to agree. The kernel never runs a judge.
pub const PACK: &str = "judge";

/// Read the judge's settings out of the settings file.
pub fn from_settings(settings: &crate::settings::Settings) -> Option<JudgeSettings> {
    let entry = settings.pack(PACK);
    if entry.is_null() {
        return None;
    }
    serde_json::from_value(entry).ok()
}

/// Write the judge's settings back into the settings file's pack table.
pub fn into_settings(
    settings: &mut crate::settings::Settings,
    judge: Option<JudgeSettings>,
) -> anyhow::Result<()> {
    match judge {
        Some(judge) => settings.set_pack(PACK, serde_json::to_value(judge)?),
        None => {
            settings.packs.remove(PACK);
        }
    }
    Ok(())
}
