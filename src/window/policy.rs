//! The seam a plugin replaces to control context behaviour, and the kernel's own.

use super::estimate_tokens;
use super::resolve::{known_window, reported_windows, ContextWindow, WindowSource, ASSUMED_WINDOW};

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

    /// How much of an old tool result stays in the request.
    ///
    /// Separate from compaction on purpose: compaction rewrites the front of a
    /// conversation once it passes a share of the window, while pruning thins
    /// the middle on every request and consults no window at all. A session can
    /// sit far below its compaction threshold and still spend a large prompt on
    /// history it has finished with.
    fn prune(&self) -> crate::compaction::PrunePolicy {
        crate::compaction::PrunePolicy::default()
    }

    /// The share of the window at which a summary replaces the front of the log.
    ///
    /// Exposed so the request builder can thin history *before* it gets there.
    /// The two are different decisions on one ladder; a builder that guessed this
    /// number would drift from the compactor's own.
    fn compact_at_percent(&self) -> u32 {
        75
    }
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
    /// How much of an old tool result survives into the next request.
    pub prune: crate::compaction::PrunePolicy,
}

impl Default for DefaultContextPolicy {
    fn default() -> Self {
        Self {
            configured: None,
            threshold_percent: 75,
            keep_recent: 8,
            provider_window: None,
            reserve: 0,
            prune: crate::compaction::PrunePolicy::default(),
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

    fn compact_at_percent(&self) -> u32 {
        self.threshold_percent
    }

    fn prune(&self) -> crate::compaction::PrunePolicy {
        self.prune
    }
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
