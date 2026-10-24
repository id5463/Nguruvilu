//! How large a model's context window is, and where that number came from.
//!
//! The kernel needs this number to know when history is about to overflow. It
//! is resolved from the most trustworthy source available, in order:
//!
//! 1. **Configured** — an explicit setting always wins.
//! 2. **Provider** — some gateways report `context_length` / `max_model_len`
//!    per model; when they do, that is the truth for that route. This covers
//!    two ways of learning it: a number a caller passed in when it built the
//!    policy, and a number a provider stated at runtime, in an error that
//!    named the limit it was enforcing.
//! 3. **Known table** — a small table of well-known model families.
//! 4. **Assumed** — a deliberately low floor, so an unknown model compacts
//!    early rather than failing late.
//!
//! The source is part of the answer, not a hidden detail: a window that was
//! *assumed*, or that only matched the built-in table, is a guess and the UI
//! should be able to say so.
//!
//! # Interface
//!
//! [`ContextPolicy`] is the seam. The kernel ships a default implementation and
//! consults it every turn; a plugin that provides the `context.policy` service
//! replaces it wholesale — different window, different threshold, different
//! number of messages kept verbatim.
//!
//! The seam is also why [`ReportedWindows`] exists. A policy is built once and
//! then held behind `Arc<dyn ContextPolicy>`, which is a read-only handle, while
//! the thing that learns a real limit is the agent loop — from a provider
//! rejecting an oversized prompt with a message that states the limit. The loop
//! has no way to hand that number to the policy it is using, so both sides
//! instead share one store: the loop records into it, and [`DefaultContextPolicy`]
//! reads it when it resolves a window. The next compaction decision then
//! measures against what this deployment actually enforces rather than against
//! the table's guess about the model id.

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
    /// at all — [`ContextPolicy::should_compact`] measures against the window
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

/// The process-wide store [`DefaultContextPolicy::resolve`] consults.
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

/// The seam a plugin replaces to control context behaviour.
pub trait ContextPolicy: Send + Sync {
    /// The window available for a model on this route.
    fn window(&self, model: &str) -> ContextWindow;

    /// Whether history should be compacted now.
    ///
    /// `prompt_tokens` is what the provider last reported, which is the only
    /// accurate signal available — estimating from characters drifts badly once
    /// tool output is involved.
    fn should_compact(&self, prompt_tokens: usize, window: ContextWindow) -> bool;

    /// How many recent messages to keep verbatim when compacting.
    fn keep_recent(&self) -> usize;
}

/// The kernel's default policy: configured window, or provider, or table, or floor.
#[derive(Debug, Clone)]
pub struct DefaultContextPolicy {
    /// Explicit override; wins over everything.
    pub configured: Option<usize>,
    /// Compact once this percentage of the window is in use.
    pub threshold_percent: u32,
    /// Recent messages kept verbatim across a compaction.
    pub keep_recent: usize,
    /// Window reported by the provider's catalog, when it reported one.
    pub provider_window: Option<usize>,
    /// Room set aside for the answer, so a full prompt can still be answered.
    ///
    /// [`ContextWindow::tokens`] covers prompt *and* completion, so a threshold
    /// measured against the whole window leaves the output ceiling unaccounted
    /// for: a 32k prompt budget on a 32k window admits no answer at all.
    pub reserve: usize,
}

impl Default for DefaultContextPolicy {
    fn default() -> Self {
        Self {
            configured: None,
            threshold_percent: 75,
            keep_recent: 8,
            provider_window: None,
            reserve: 0,
        }
    }
}

impl DefaultContextPolicy {
    /// Resolve the window, recording which source supplied it.
    pub fn resolve(&self, model: &str) -> ContextWindow {
        if let Some(tokens) = self.configured.filter(|t| *t > 0) {
            return ContextWindow::new(tokens, WindowSource::Configured);
        }
        if let Some(tokens) = self.provider_window.filter(|t| *t > 0) {
            return ContextWindow::new(tokens, WindowSource::Provider);
        }
        // A runtime report sits above the table and below anything explicit. The
        // table describes what a model id usually means somewhere else; a
        // provider saying "8192" for an id the table thinks is 32_768 is telling
        // us about *this* deployment, which is exactly the case the table gets
        // wrong. So a report resolves as `Provider`, not as a guess.
        if let Some(tokens) = reported_windows().get(model) {
            return ContextWindow::new(tokens, WindowSource::Provider);
        }
        if let Some(tokens) = known_window(model) {
            return ContextWindow::new(tokens, WindowSource::Table);
        }
        ContextWindow::new(ASSUMED_WINDOW, WindowSource::Assumed)
    }
}

impl ContextPolicy for DefaultContextPolicy {
    fn window(&self, model: &str) -> ContextWindow {
        self.resolve(model)
    }

    fn should_compact(&self, prompt_tokens: usize, window: ContextWindow) -> bool {
        if prompt_tokens == 0 || window.tokens == 0 {
            return false;
        }
        // The window is prompt *plus* completion, so the output ceiling is taken
        // out before measuring: a prompt at 75% of a 32k window with a 32k
        // output ceiling is 1.75x the window, and the provider rejects that
        // request instead of the loop compacting ahead of it.
        //
        // A reserve bigger than half the window is a misconfiguration rather
        // than an instruction, so it is capped: without the cap a nonsense
        // `max_output_tokens` would make `usable` zero and switch compaction off
        // entirely.
        let reserve = self.reserve.min(window.tokens / 2);
        let usable = window.tokens - reserve;
        // Capped at 95, not 100. At 100 the threshold is the whole window, which
        // nothing can reach before the request is already oversized, so
        // "compact at 100%" silently meant "never compact" — and the percentage
        // is written by hand in a file this program does not prompt for.
        let percent = self.threshold_percent.clamp(1, 95) as u64;
        let threshold = usable as u64 * percent / 100;
        prompt_tokens as u64 >= threshold
    }

    fn keep_recent(&self) -> usize {
        self.keep_recent
    }
}

/// The same ruler as [`crate::context::estimate_tokens`], re-exported here.
///
/// One estimate, called from both sides: compaction asks "should this history
/// be trimmed?" and the context budget asks "how much of it fits?", and two
/// rulers made those answers disagree — a plain four-characters-per-token count
/// undercounts CJK by three to four times in exactly the module that decides
/// when to compact.
pub use crate::context::estimate_tokens;

/// Estimate the size of a whole history, for the first request of a session.
pub fn estimate_history(messages: &[crate::message::Message]) -> usize {
    messages
        .iter()
        .map(|message| {
            let text = estimate_tokens(message.text());
            let calls: usize = message
                .tool_calls
                .iter()
                .map(|call| estimate_tokens(&call.name) + estimate_tokens(&call.arguments))
                .sum();
            // Per-message overhead: role, delimiters, and ids.
            text + calls + 4
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::Message;

    #[test]
    fn a_configured_window_wins_over_everything() {
        let policy = DefaultContextPolicy {
            configured: Some(8000),
            provider_window: Some(200_000),
            ..Default::default()
        };
        let window = policy.window("claude-opus-5");
        assert_eq!(window.tokens, 8000);
        assert_eq!(window.source, WindowSource::Configured);
    }

    #[test]
    fn a_provider_window_beats_the_table() {
        let policy = DefaultContextPolicy {
            provider_window: Some(64_000),
            ..Default::default()
        };
        let window = policy.window("deepseek-v4.1-flash");
        assert_eq!(window.tokens, 64_000);
        assert_eq!(window.source, WindowSource::Provider);
    }

    #[test]
    fn a_known_model_uses_the_table() {
        let policy = DefaultContextPolicy::default();
        let window = policy.window("claude-opus-5");
        assert_eq!(window.tokens, 200_000);
        assert_eq!(window.source, WindowSource::Table);
    }

    #[test]
    fn an_unknown_model_falls_back_to_the_floor_and_says_so() {
        let policy = DefaultContextPolicy::default();
        let window = policy.window("some-new-model-9000");
        assert_eq!(window.tokens, ASSUMED_WINDOW);
        assert_eq!(window.source, WindowSource::Assumed);
        assert!(window.source.is_guess(), "the UI must be able to flag a guess");
    }

    #[test]
    fn the_longest_pattern_wins() {
        // `gemini-3` must beat a hypothetical shorter `gemini` entry.
        assert_eq!(known_window("gemini-3.1-pro"), Some(1_000_000));
        assert_eq!(known_window("gemini-3-flash"), Some(1_000_000));
        assert_eq!(known_window("nothing-matches"), None);
    }

    #[test]
    fn the_table_is_case_insensitive() {
        assert_eq!(known_window("CLAUDE-OPUS-5"), Some(200_000));
    }

    #[test]
    fn a_family_name_matches_at_a_boundary_but_not_inside_a_word() {
        // `o1-preview` carries the family name as its first token, so the entry
        // applies. `mimosa-7b` only happens to contain those four letters: it is
        // a different model whose window this table does not know, and the floor
        // is the honest answer — claiming a `mimo` entry here would pick a
        // window one family too large.
        assert_eq!(known_window("o1-preview"), Some(128_000));
        assert_eq!(known_window("mimo-v2"), Some(128_000));
        assert_eq!(known_window("mimosa-7b"), None);
        // A boundary on one side only is a fragment, not a family name.
        assert_eq!(known_window("ko1la"), None);
    }

    #[test]
    fn an_exact_id_still_resolves() {
        // A bare family name is the whole id, so it matches at both ends.
        assert_eq!(known_window("o1"), Some(128_000));
        assert_eq!(known_window("deepseek"), Some(128_000));
        assert_eq!(known_window("gemini-3"), Some(1_000_000));
    }

    #[test]
    fn a_table_hit_is_reported_as_a_guess() {
        // The table is written by hand and describes someone else's deployment,
        // so a hit is an estimate. Only a configured or provider-supplied window
        // is a statement of fact, and the caller is entitled to know the
        // difference before it decides how much history to trust.
        let policy = DefaultContextPolicy::default();
        let window = policy.window("claude-opus-5");
        assert_eq!(window.source, WindowSource::Table);
        assert!(window.source.is_guess(), "a table hit must be flaggable as a guess");

        assert!(!WindowSource::Configured.is_guess());
        assert!(!WindowSource::Provider.is_guess());
        assert!(WindowSource::Assumed.is_guess());
    }

    #[test]
    fn a_zero_configured_window_is_ignored_rather_than_breaking_everything() {
        let policy = DefaultContextPolicy {
            configured: Some(0),
            ..Default::default()
        };
        assert_eq!(policy.window("claude-opus-5").source, WindowSource::Table);
    }

    #[test]
    fn compaction_triggers_at_the_threshold() {
        let policy = DefaultContextPolicy {
            threshold_percent: 75,
            ..Default::default()
        };
        let window = ContextWindow::new(100_000, WindowSource::Table);

        assert!(!policy.should_compact(74_000, window));
        assert!(policy.should_compact(75_000, window));
        assert!(policy.should_compact(99_000, window));
    }

    #[test]
    fn compaction_does_not_trigger_without_a_measurement() {
        let policy = DefaultContextPolicy::default();
        let window = ContextWindow::new(100_000, WindowSource::Table);
        assert!(!policy.should_compact(0, window));
    }

    #[test]
    fn a_threshold_above_the_cap_still_leaves_room_to_answer() {
        // 100% would mean "compact only once the window is already full", which
        // no request reaches before the provider rejects it — so the cap keeps
        // the last five percent as the place for the compaction to happen.
        let policy = DefaultContextPolicy {
            threshold_percent: 500,
            ..Default::default()
        };
        let window = ContextWindow::new(1000, WindowSource::Table);
        assert!(!policy.should_compact(949, window));
        assert!(policy.should_compact(950, window));
        assert!(policy.should_compact(1000, window));
    }

    #[test]
    fn a_zero_threshold_still_compacts_instead_of_never() {
        // 0 would compare the prompt against a zero threshold, which is true
        // after every single step: a summariser call per turn. The floor is 1%.
        let policy = DefaultContextPolicy {
            threshold_percent: 0,
            ..Default::default()
        };
        let window = ContextWindow::new(100_000, WindowSource::Table);
        assert!(!policy.should_compact(0, window));
        assert!(policy.should_compact(1_000, window));
    }

    #[test]
    fn the_output_reserve_is_taken_out_of_the_window() {
        // The window covers prompt plus completion, so with an 8k output ceiling
        // on a 100k window the prompt threshold is 75% of the 92k that is left.
        let policy = DefaultContextPolicy {
            threshold_percent: 75,
            reserve: 8_000,
            ..Default::default()
        };
        let window = ContextWindow::new(100_000, WindowSource::Table);
        assert!(!policy.should_compact(68_900, window));
        assert!(policy.should_compact(69_000, window));
    }

    #[test]
    fn a_nonsense_reserve_does_not_switch_compaction_off() {
        let policy = DefaultContextPolicy {
            threshold_percent: 75,
            reserve: 1_000_000,
            ..Default::default()
        };
        let window = ContextWindow::new(100_000, WindowSource::Table);
        // Capped at half the window: 75% of 50_000 is 37_500.
        assert!(!policy.should_compact(37_000, window));
        assert!(policy.should_compact(37_500, window));
    }

    #[test]
    fn history_estimation_counts_text_and_tool_arguments() {
        let messages = vec![
            Message::user("a".repeat(400)),
            Message::assistant_tools(
                vec![crate::message::ToolCall {
                    id: "c1".into(),
                    name: "read".into(),
                    arguments: "{\"path\":\"x\"}".into(),
                }],
                None,
            ),
        ];
        let estimate = estimate_history(&messages);
        // 400 ASCII chars ≈ 100 tokens, plus the call and per-message overhead.
        assert!(estimate > 100, "{estimate}");
        assert!(estimate < 200, "{estimate}");
    }

    #[test]
    fn the_same_chinese_text_costs_the_same_in_both_modules() {
        // The two rulers used to disagree on exactly the scripts that matter:
        // 4 characters per token here, CJK-weighted in `context`. A history
        // could then be "small enough to keep" and "too big to fit" at once.
        for text in ["说明：很好。", "模型上下文窗口", "", "plain ascii text", "éé привет"] {
            assert_eq!(
                estimate_tokens(text),
                crate::context::estimate_tokens(text),
                "{text:?} is measured with two different rulers"
            );
        }
        assert_eq!(estimate_tokens("说明：很好。"), 6);
    }

    /// A model id nothing else in this process reports on.
    ///
    /// The runtime report store is shared by every test in this binary, so a
    /// name another test recorded a window for would make these tests depend on
    /// the order they happened to run in. A fresh id per run keeps each test on
    /// its own entry.
    fn unique_model(label: &str) -> String {
        format!("{label}-{}", uuid::Uuid::new_v4().simple())
    }

    #[test]
    fn a_runtime_report_teaches_the_policy_the_window_it_enforces() {
        // The round trip the store exists for. Before a report the policy knows
        // nothing about this id and falls back to the floor; after one it
        // measures against the number the provider said it enforces — for that
        // model, and for no other.
        let model = unique_model("runtime-probe");
        let unrelated = unique_model("unrelated-probe");
        let policy = DefaultContextPolicy::default();

        let before = policy.window(&model);
        assert_eq!(before.tokens, ASSUMED_WINDOW);
        assert_eq!(before.source, WindowSource::Assumed);

        reported_windows().record(&model, 8192);

        let after = policy.window(&model);
        assert_eq!(after.tokens, 8192, "the reported number is the window");
        assert_eq!(after.source, WindowSource::Provider);
        assert!(
            !after.source.is_guess(),
            "a provider's statement about this deployment is not a guess"
        );
        assert!(
            !policy.should_compact(6000, after),
            "6000 of 8192 is below the default threshold"
        );
        assert!(
            policy.should_compact(6500, after),
            "and 6500 of 8192 is above it, which the old floor of 32768 hid"
        );

        let untouched = policy.window(&unrelated);
        assert_eq!(
            untouched.source,
            WindowSource::Assumed,
            "a report describes one model, not every model"
        );
        assert_eq!(untouched.tokens, ASSUMED_WINDOW);
    }

    #[test]
    fn a_runtime_report_beats_the_table_for_the_model_it_names() {
        // A gateway serving a smaller window than the family name implies is
        // exactly the case the table gets wrong: the table says a million for
        // any `gemini-3` id, and this deployment takes 8192. The report is the
        // provider talking about the route in front of us, so it wins.
        let model = unique_model("gemini-3-runtime-probe");
        assert_eq!(known_window(&model), Some(1_000_000));

        let policy = DefaultContextPolicy::default();
        assert_eq!(
            policy.window(&model).source,
            WindowSource::Table,
            "the table is the first answer, before anything is reported"
        );

        reported_windows().record(&model, 8192);

        let window = policy.window(&model);
        assert_eq!(window.tokens, 8192);
        assert_eq!(window.source, WindowSource::Provider);
    }

    #[test]
    fn a_clone_of_the_report_store_shares_its_reports() {
        // The agent loop records through a handle it holds while the policy
        // reads the process-wide one. If cloning copied the map instead of the
        // handle, the loop would be writing into a store nobody reads.
        let store = ReportedWindows::default();
        let clone = store.clone();
        clone.record("shared-clone-probe", 4096);

        assert_eq!(store.get("shared-clone-probe"), Some(4096));
        assert_eq!(clone.get("shared-clone-probe"), Some(4096));
        assert_eq!(store.get("never-recorded-probe"), None);
    }

    #[test]
    fn an_implausible_report_is_dropped_rather_than_believed() {
        // A reported zero would switch compaction off — nothing reaches the
        // threshold of a zero window — and an absurd one would leave the loop
        // waiting for room that does not exist, so neither is stored.
        let store = ReportedWindows::default();
        let model = unique_model("implausible-probe");

        for tokens in [0, 1, MIN_PLAUSIBLE_WINDOW - 1, MAX_PLAUSIBLE_WINDOW + 1] {
            store.record(&model, tokens);
            assert_eq!(store.get(&model), None, "{tokens} must not be stored");
        }

        store.record(&model, MIN_PLAUSIBLE_WINDOW);
        assert_eq!(store.get(&model), Some(MIN_PLAUSIBLE_WINDOW));

        // An unnamed model has nothing to key the report on.
        store.record("   ", 4096);
        assert_eq!(store.get(""), None);
    }
}
