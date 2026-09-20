//! How large a model's context window is, and where that number came from.
//!
//! The kernel needs this number to know when history is about to overflow. It
//! is resolved from the most trustworthy source available, in order:
//!
//! 1. **Configured** — an explicit setting always wins.
//! 2. **Provider** — some gateways report `context_length` / `max_model_len`
//!    per model; when they do, that is the truth for that route.
//! 3. **Known table** — a small table of well-known model families.
//! 4. **Assumed** — a deliberately low floor, so an unknown model compacts
//!    early rather than failing late.
//!
//! The source is part of the answer, not a hidden detail: a window that was
//! *assumed* is a guess and the UI should be able to say so.
//!
//! # Interface
//!
//! [`ContextPolicy`] is the seam. The kernel ships a default implementation and
//! consults it every turn; a plugin that provides the `context.policy` service
//! replaces it wholesale — different window, different threshold, different
//! number of messages kept verbatim.

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
    Known,
    /// Nothing matched; a conservative floor was used.
    Assumed,
}

impl WindowSource {
    /// A short label for display.
    pub fn as_str(&self) -> &'static str {
        match self {
            WindowSource::Configured => "configured",
            WindowSource::Provider => "provider",
            WindowSource::Known => "known",
            WindowSource::Assumed => "assumed",
        }
    }

    /// Whether the number is a guess rather than a fact.
    pub fn is_guess(&self) -> bool {
        matches!(self, WindowSource::Assumed)
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

/// Well-known model families and the context they accept.
///
/// Matched by substring against the model id, longest pattern first. Values are
/// deliberately conservative: when a family ships several sizes, the smallest
/// is used so compaction errs early.
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

/// Look a model up in the known table.
pub fn known_window(model: &str) -> Option<usize> {
    let lowered = model.to_ascii_lowercase();
    // Longest pattern first, so `gemini-3` beats `gemini`.
    let mut matches: Vec<(&str, usize)> = KNOWN_WINDOWS
        .iter()
        .filter(|(pattern, _)| lowered.contains(pattern))
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
}

impl Default for DefaultContextPolicy {
    fn default() -> Self {
        Self {
            configured: None,
            threshold_percent: 75,
            keep_recent: 8,
            provider_window: None,
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
        if let Some(tokens) = known_window(model) {
            return ContextWindow::new(tokens, WindowSource::Known);
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
        let threshold = window.tokens as u64 * self.threshold_percent.min(100) as u64 / 100;
        prompt_tokens as u64 >= threshold
    }

    fn keep_recent(&self) -> usize {
        self.keep_recent
    }
}

/// A rough token count for text, used only before the provider reports one.
///
/// Four characters per token is the usual rule of thumb for mixed prose and
/// code. It is not accurate enough to make a compaction decision on when real
/// usage is available, which is why the loop prefers `prompt_tokens`.
pub fn estimate_tokens(text: &str) -> usize {
    text.chars().count().div_ceil(4)
}

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
        assert_eq!(window.source, WindowSource::Known);
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
    fn a_zero_configured_window_is_ignored_rather_than_breaking_everything() {
        let policy = DefaultContextPolicy {
            configured: Some(0),
            ..Default::default()
        };
        assert_eq!(policy.window("claude-opus-5").source, WindowSource::Known);
    }

    #[test]
    fn compaction_triggers_at_the_threshold() {
        let policy = DefaultContextPolicy {
            threshold_percent: 75,
            ..Default::default()
        };
        let window = ContextWindow::new(100_000, WindowSource::Known);

        assert!(!policy.should_compact(74_000, window));
        assert!(policy.should_compact(75_000, window));
        assert!(policy.should_compact(99_000, window));
    }

    #[test]
    fn compaction_does_not_trigger_without_a_measurement() {
        let policy = DefaultContextPolicy::default();
        let window = ContextWindow::new(100_000, WindowSource::Known);
        assert!(!policy.should_compact(0, window));
    }

    #[test]
    fn the_threshold_is_capped_at_one_hundred_percent() {
        let policy = DefaultContextPolicy {
            threshold_percent: 500,
            ..Default::default()
        };
        let window = ContextWindow::new(1000, WindowSource::Known);
        assert!(policy.should_compact(1000, window));
        assert!(!policy.should_compact(999, window));
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
        // 400 chars ≈ 100 tokens, plus the call and per-message overhead.
        assert!(estimate > 100, "{estimate}");
        assert!(estimate < 200, "{estimate}");
    }
}
