//! Which window a model accepts, and where that number came from.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, OnceLock};

use serde::{Deserialize, Serialize};

/// Where a window size came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum WindowSource {
    /// Explicitly configured by the user.
    Configured,
    /// Reported by the provider's model catalog.
    Provider,
    /// Matched a built-in table entry.
    ///
    /// Not a provider statement and not verified against one: the table is a
    /// conservative estimate written by hand. Reported separately from
    /// `Assumed` only because a table hit is a better guess than the floor,
    /// not because it is a fact.
    Table,
    /// Nothing matched; a conservative floor was used.
    Assumed,
}

impl WindowSource {
    /// A short label for display.
    pub fn as_str(&self) -> &'static str {
        match self {
            WindowSource::Configured => "configured",
            WindowSource::Provider => "provider",
            WindowSource::Table => "table (estimate, not from the provider)",
            WindowSource::Assumed => "assumed",
        }
    }

    /// Whether the number is a guess rather than a fact.
    ///
    /// Both the floor and the table are guesses. A table entry is a claim about
    /// someone else's deployment, written by hand from documentation that the
    /// provider is free to change, and the same id can serve different windows
    /// on different gateways — so presenting it as known would tell the user
    /// something this program cannot verify. `Configured` and `Provider` are the
    /// only sources that are statements rather than estimates.
    pub fn is_guess(&self) -> bool {
        matches!(self, WindowSource::Table | WindowSource::Assumed)
    }
}

/// A resolved context window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextWindow {
    /// Tokens the model accepts in one request, prompt plus completion.
    pub tokens: usize,
    /// Where the number came from.
    pub source: WindowSource,
}

impl ContextWindow {
    /// Build a window from an explicit number.
    pub fn new(tokens: usize, source: WindowSource) -> Self {
        Self { tokens, source }
    }
}

/// The floor used when nothing is known about a model.
///
/// Low on purpose. Guessing high means a long conversation fails with a
/// provider error; guessing low means it compacts a little earlier than
/// necessary, which costs a summary and nothing else.
pub const ASSUMED_WINDOW: usize = 32_768;

/// The smallest window worth believing.
///
/// Shared with [`crate::llm::reported_window`], which reads windows out of
/// provider error messages, so one definition decides what counts as plausible.
/// No served model accepts fewer than a thousand tokens, and a smaller number in
/// a provider message is something else — a token count from the rejected
/// request, a message id, or an HTTP status — that must not be mistaken for a
/// window.
pub const MIN_PLAUSIBLE_WINDOW: usize = 1_000;

/// The largest window worth believing.
///
/// Above the largest window any provider sells, so a bigger number in an error
/// message is a byte count or a billing figure rather than a context limit.
/// Storing one would also switch compaction off in practice, because the
/// threshold against such a window is a prompt size nothing reaches.
pub const MAX_PLAUSIBLE_WINDOW: usize = 10_000_000;

/// Windows that providers have reported at runtime, keyed by model id.
///
/// Interior mutability rather than a plain map, because the reader is a policy
/// that was already built and is held behind `Arc<dyn ContextPolicy>`; see the
/// module docs for why the store is separate from that handle.
///
/// A clone shares the same map, and that is the point of cloning it: the copy
/// the agent loop records into and the copy a caller resolves against have to be
/// one store, or a limit learned during a turn would never reach the decision
/// made on the next one.
#[derive(Debug, Clone, Default)]
pub struct ReportedWindows {
    /// Model id to accepted tokens. Ordered rather than hashed so the contents
    /// are stable to read in a diagnostic, and because a route table holds
    /// tens of entries at most.
    windows: Arc<Mutex<BTreeMap<String, usize>>>,
}

impl ReportedWindows {
    /// Record that `model` accepts `tokens` in one request.
    ///
    /// A value outside [`MIN_PLAUSIBLE_WINDOW`]..=[`MAX_PLAUSIBLE_WINDOW`] is
    /// dropped rather than stored. A reported zero would be worse than no report
    /// at all — [`crate::window::ContextPolicy::should_compact`] measures against the window
    /// and returns false for a zero one, which is compaction switched off
    /// — and a reported absurdity would keep the loop waiting for room that does
    /// not exist. The newest plausible report for a model wins: a route that
    /// changed its limit is described by the limit it complained about last.
    pub fn record(&self, model: &str, tokens: usize) {
        let model = model.trim();
        if model.is_empty() || !(MIN_PLAUSIBLE_WINDOW..=MAX_PLAUSIBLE_WINDOW).contains(&tokens) {
            return;
        }
        // A poisoned lock means another thread died in the middle of an update.
        // The map is a `BTreeMap<String, usize>` with no invariant a torn write
        // could break, and losing a hint about a window is not a reason to stop
        // the program, so the map is recovered and the report goes in anyway.
        // Panicking here would turn one unrelated panic into a failed turn.
        let mut windows = self
            .windows
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        windows.insert(model.to_string(), tokens);
    }

    /// The most recent window recorded for `model`, if one was recorded.
    pub fn get(&self, model: &str) -> Option<usize> {
        let windows = self
            .windows
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        windows.get(model.trim()).copied()
    }
}

/// The process-wide store [`crate::window::DefaultContextPolicy::resolve`] consults.
///
/// A global instead of a field on the policy, for two reasons. The mechanical
/// one decides it: `Settings::context_policy` builds the policy with a struct
/// literal that names every field and no `..Default::default()`, and adding a
/// field would make that literal a missing-field error in a module this change
/// is not allowed to touch — so the number would have to be carried somewhere
/// that existing construction sites already reach, which is a static.
///
/// The design reason is separate and would stand on its own: what a provider
/// said about a model's window is a fact about this process's route, not about
/// one policy value. The loop that learns it and the policy that needs it are
/// not the same object and do not know each other, so a store reachable without
/// either holding the other is the honest shape.
static REPORTED_WINDOWS: OnceLock<ReportedWindows> = OnceLock::new();

/// The shared store of windows that providers have reported at runtime.
///
/// Every caller gets the same map: the initialiser runs once per process and its
/// empty map is the one handed out from then on, so a report from one turn is
/// visible to the policy in the next. Cloning the reference is enough to record
/// from anywhere; there is no per-session copy to keep in step.
pub fn reported_windows() -> &'static ReportedWindows {
    REPORTED_WINDOWS.get_or_init(ReportedWindows::default)
}

/// Well-known model families and the context they accept.
///
/// Matched against the model id on whole boundaries — the pattern must sit
/// between non-alphanumeric characters, or at either end of the id — with the
/// longest matching pattern winning. Values are deliberately conservative: when
/// a family ships several sizes, the smallest is used so compaction errs early.
const KNOWN_WINDOWS: &[(&str, usize)] = &[
    ("gemini-3", 1_000_000),
    ("gemini-2", 1_000_000),
    ("gemini-1", 1_000_000),
    ("claude", 200_000),
    ("kimi", 256_000),
    ("moonshot", 128_000),
    ("deepseek", 128_000),
    ("qwen", 128_000),
    ("glm", 128_000),
    ("minimax", 200_000),
    ("gpt-4", 128_000),
    ("gpt-5", 128_000),
    ("gpt-6", 128_000),
    ("o1", 128_000),
    ("o3", 128_000),
    ("o4", 128_000),
    ("llama", 128_000),
    ("mistral", 128_000),
    ("mimo", 128_000),
];

/// Whether `pattern` appears in `id` as a whole token rather than as a fragment.
///
/// Both ends of the occurrence have to sit on a boundary: the start or end of
/// the id, or a non-alphanumeric character. A plain `contains` treated any
/// occurrence as a family match, so `mimo` claimed `mimosa-7b`, `o1` claimed
/// every id carrying those two characters, and `glm` claimed anything with them
/// — each of those picked a window far larger than the model accepts, which is
/// the one direction this table exists to avoid. Dropping an unverified match
/// instead falls back to the floor, so the cost of a miss is erring small.
fn matches_token(id: &str, pattern: &str) -> bool {
    // An empty pattern would report a match at every offset while advancing the
    // cursor by nothing, so it is refused rather than guessed at.
    if pattern.is_empty() {
        return false;
    }
    let is_boundary = |byte: u8| !byte.is_ascii_alphanumeric();
    let mut from = 0;
    while let Some(offset) = id[from..].find(pattern) {
        let start = from + offset;
        let end = start + pattern.len();
        let starts_clean = start == 0 || is_boundary(id.as_bytes()[start - 1]);
        // A trailing non-ASCII byte is the lead byte of another character, and
        // never an ASCII letter or digit, so it counts as a boundary too.
        let ends_clean = end == id.len() || is_boundary(id.as_bytes()[end]);
        if starts_clean && ends_clean {
            return true;
        }
        // `find` only ever returns offsets that are character boundaries, so
        // resuming at the end of the failed occurrence cannot cut a character.
        from = end;
    }
    false
}

/// Look a model up in the known table.
pub fn known_window(model: &str) -> Option<usize> {
    let lowered = model.to_ascii_lowercase();
    // Longest pattern first, so `gemini-3` beats a hypothetical shorter
    // `gemini`. `sort_by` is stable, so table order breaks ties.
    let mut matches: Vec<(&str, usize)> = KNOWN_WINDOWS
        .iter()
        .filter(|(pattern, _)| matches_token(&lowered, pattern))
        .copied()
        .collect();
    matches.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
    matches.first().map(|(_, tokens)| *tokens)
}
