//! Receives progress from the loop.
//!
//! Hosts implement [`AgentObserver`] to render a turn as it happens: text
//! fragments, tool boundaries, retries, compactions. Every method has a default,
//! so an implementation writes only what it actually shows.

use crate::compaction::Compaction;
use crate::message::Message;
use crate::window::ContextWindow;

/// Receives progress from the loop. Implementations decide how to render it.
pub trait AgentObserver: Send + Sync {
    /// Assistant text fragment.
    fn on_text(&self, _delta: &str) {}
    /// Reasoning fragment, when the provider emits one.
    fn on_reasoning(&self, _delta: &str) {}
    /// A model step is starting (1-based).
    fn on_step(&self, _step: usize) {}
    /// A tool call is about to run.
    fn on_tool_start(&self, _name: &str, _arguments: &str) {}
    /// A tool call finished.
    fn on_tool_end(&self, _name: &str, _ok: bool, _result: &str) {}
    /// The current step is being retried: the provider failed or answered
    /// with nothing, and whatever it had produced for this step is rewound.
    ///
    /// `attempt` is the attempt that just failed (1-based), `reason` says why.
    fn on_step_retry(&self, _attempt: usize, _reason: &str) {}
    /// The answer hit the provider's output ceiling; the loop is asking the
    /// model to continue it in the next step.
    fn on_continuation(&self, _reason: &str) {}
    /// A message was committed to the history and will be part of the next
    /// request.
    ///
    /// Hosts that persist the conversation as it happens write it here: a
    /// process that dies mid-turn then loses only what never completed,
    /// never everything since the last turn boundary.
    fn on_message(&self, _message: &Message) {}
    /// History is being compacted, before the summary is written.
    fn on_compaction_start(&self, _messages: usize, _window: ContextWindow) {}
    /// A compaction finished.
    fn on_compaction(&self, _compaction: &Compaction) {}
    /// Fragments were injected into the request about to be sent.
    fn on_injection(&self, _injection: &crate::context::Injection) {}
}

/// An observer that discards everything.
pub struct SilentObserver;

impl AgentObserver for SilentObserver {}
