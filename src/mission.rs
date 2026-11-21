//! Reading the mission record on disk, for the loop's own decision.
//!
//! The mission tools are a plugin now (`packs/long-horizon`), and this module is
//! the one thing that could not go with them: a mission whose status is
//! `running` keeps the turn open, and "does this turn end" is the loop's
//! question, not a tool's.
//!
//! So what remains is a reader, not a capability. It knows two things:
//!
//! * a mission is a directory `<workspace>/missions/<id>/` with a `mission.json`,
//! * `missions/active.json` names the current one.
//!
//! Both are the plugin's file format, published in `packs/long-horizon/README.md`.
//! The kernel parsing it is not the kernel knowing what a mission *is* — it is
//! the kernel able to answer "may I stop?", which it must answer whether or not
//! the plugin is installed. A workspace with no `missions/` directory answers
//! `None`, which is what an uninstalled pack looks like.

use std::path::Path;

use serde_json::Value;

/// The clip applied to text lifted from the record for the injected message.
///
/// Generous on purpose: this text goes into the context of every step of a long
/// task, so a runaway objective could cost more than the mission is worth, and a
/// one-line objective must not be cut.
const BRIEF_CLIP: usize = 400;

/// What the loop needs to know about a running mission.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MissionBrief {
    /// Mission id.
    pub id: String,
    /// What it is for.
    pub objective: String,
    /// `running`, `blocked`, or `done`.
    pub status: String,
    /// The next step, when one is recorded.
    pub step: String,
    /// How to tell it is finished.
    pub done_when: String,
}

/// The mission whose record says `running`, if there is one.
///
/// Only `running` keeps a turn open. `blocked` and `done` are the model saying it
/// is no longer going; a shell that went round again anyway would be overruling
/// the one party allowed to end a mission.
pub fn running_mission(workspace: &Path) -> Option<MissionBrief> {
    let root = workspace.join("missions");
    let id = active_id(&root)?;
    let text = std::fs::read_to_string(root.join(&id).join("mission.json")).ok()?;
    let record: Value = serde_json::from_str(&text).ok()?;
    let field = |key: &str| -> String {
        record
            .get(key)
            .and_then(|value| value.as_str())
            .map(|text| clip(text.trim(), BRIEF_CLIP))
            .unwrap_or_default()
    };
    if field("status") != "running" {
        return None;
    }
    let recorded_id = field("id");
    Some(MissionBrief {
        // The record's own id when it has one, the pointer's otherwise: both name
        // the same mission, and a mission named wrongly in the injected message
        // is worse than a slightly stale name.
        id: if recorded_id.is_empty() { id } else { recorded_id },
        objective: field("objective"),
        status: field("status"),
        step: field("step"),
        done_when: field("done_when"),
    })
}

/// The id in `missions/active.json`, if that file exists and parses.
fn active_id(root: &Path) -> Option<String> {
    let text = std::fs::read_to_string(root.join("active.json")).ok()?;
    let value: Value = serde_json::from_str(&text).ok()?;
    value
        .get("mission")
        .and_then(|m| m.as_str())
        .map(str::to_string)
        .filter(|id| !id.trim().is_empty())
}

/// Cut to `max` characters, counting characters rather than bytes so a Chinese
/// objective is not cut mid-character.
fn clip(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_string();
    }
    let mut out: String = text.chars().take(max).collect();
    out.push('…');
    out
}

