//! The data a turn carries: its state, its outcome, and one step's result.
//!
//! [`LoopState`] goes into a loop, [`AgentOutcome`] comes back out, and the rest
//! of this file is what one streaming attempt produced before it was committed.

use crate::compaction::{Compaction, Summarizer};
use crate::llm::{LlmClient, PartialCallView, TokenUsage};
use crate::message::Message;
use crate::tools::ToolRegistry;
use std::collections::VecDeque;
use std::sync::Arc;

use super::config::TurnSettings;
use super::observer::AgentObserver;
use crate::llm;
use crate::window;

/// Per-turn timing breakdown, so overhead is measurable rather than guessed.
#[derive(Debug, Clone, Default)]
pub struct TurnTiming {
    /// Time spent waiting on the model (time to first token plus generation).
    pub model_ms: u128,
    /// Time spent executing tools.
    pub tools_ms: u128,
}

/// What one `run` produced.
#[derive(Debug, Clone)]
pub struct AgentOutcome {
    /// Final assistant text.
    pub text: String,
    /// Number of model steps taken.
    pub steps: usize,
    /// Number of tool calls executed.
    pub tool_calls: usize,
    /// Token usage summed over the turn.
    pub usage: TokenUsage,
    /// Messages appended during this turn, in order.
    pub new_messages: Vec<Message>,
    /// Timing breakdown.
    pub timing: TurnTiming,
    /// Configuration version the turn ran under.
    pub config_version: u64,
    /// Injections performed during this turn, in order.
    ///
    /// Recorded so the session log can explain what the model was actually
    /// sent; an injected fragment is otherwise invisible after the fact.
    pub injections: Vec<crate::session::InjectionRecord>,
    /// Compactions performed during this turn, in order.
    ///
    /// The caller persists these so a reloaded session matches what the model
    /// actually saw.
    pub compactions: Vec<Compaction>,
    /// Set when the loop stopped on a failure.
    ///
    /// The messages produced before the failure are kept and reported in
    /// `new_messages`: a turn that dies half-done must not also erase what it
    /// already did. This says why the turn did not finish.
    pub error: Option<String>,
    /// How many step attempts were retried during this turn.
    pub retries: u32,
}

/// Everything a turn's loop works on, handed over wholesale.
///
/// The fields are public on purpose. This is the price of a replaceable loop:
/// a pack that writes its own has to see the conversation, the tools, the
/// client, the cancel signal and the budget, because a loop that cannot see
/// those cannot run a step. Keeping them private and passing a curated view
/// would mean the view, not the loop, decides what is possible — which is the
/// shape this file just spent its history growing out of.
///
/// What is *not* here is as deliberate: no kernel handle, no plugin registry.
/// A loop asks the tool registry for a call and the client for a step; anything
/// more would make the loop depend on the machinery that chose it.
pub struct LoopState {
    /// The conversation, including anything appended by this turn so far.
    pub messages: Vec<Message>,
    /// The model client.
    pub client: LlmClient,
    /// The tools this turn may call.
    pub tools: Arc<ToolRegistry>,
    /// The turn's frozen settings.
    pub settings: TurnSettings,
    /// Streaming callbacks, so a custom loop still paints the transcript.
    pub observer: Arc<dyn AgentObserver>,
    /// What a summary is written by.
    pub summarizer: Arc<dyn Summarizer>,
    /// The session this turn belongs to, for tool attribution.
    pub identity: Option<crate::tools::ToolIdentity>,
    /// A stop request from the host.
    pub cancel: Option<tokio::sync::watch::Receiver<bool>>,
    /// Messages the host wants delivered mid-turn.
    pub inbox: Option<Arc<std::sync::Mutex<VecDeque<String>>>>,
    /// Step cap for this turn.
    pub max_steps: usize,
    /// Whether this turn is somebody's subtask.
    pub subtask: bool,
    /// What a stop reports as the error.
    pub stop_reason: String,
    /// Injection engine override, for tests and embedding.
    pub injection_override: Option<Arc<dyn crate::context::InjectionSource>>,
    /// The user's message that started this turn.
    ///
    /// Carried rather than read back off `messages`: a loop decides where the
    /// turn's history begins, and the first message of the turn is not
    /// necessarily the last user message once a loop has appended its own.
    pub start_prompt: String,
}

/// What a stopped turn reports as its error.
pub(crate) const STOPPED: &str = "stopped by the user";

/// Extra batches of steps granted after a batch spends `max_steps`, each as long
/// as the cap itself.
///
/// Hitting the cap does not mean the turn failed: a genuinely long task can
/// need more rounds than one batch holds, and ending there abandons the work
/// half-done with nothing to answer. So a spent batch asks the model whether a
/// round count this long is normal and earns one more batch. The number is
/// finite because the renewal is the easiest place for a runaway to hide — a
/// model that never concludes would act and bill forever — and two renewals are
/// enough for a real task to either finish or say that it cannot.
pub const CONTINUATION_BATCHES: usize = 2;

/// What the shell says while a mission is still running.
///
/// Two characters, on purpose. The mission's objective, its next step and its
/// done-when are already in the system prompt of every request, and
/// `mission_status` returns them again on demand; restating them here spent two
/// hundred characters of every context window on text the model already had, in
/// the one place a long task cannot afford it.
///
/// It says nothing about ending: nothing in this file can end a mission. The
/// only ways out are the model recording `done` or `blocked`, or the user
/// stopping the turn.
pub(crate) const MISSION_NUDGE: &str = "继续";

/// The longest pause between one retry and the next while a mission holds the
/// turn open.
///
/// The network backoff doubles per attempt and reaches 200 s at its cap; a
/// mission waiting on a flaky route deserves one ask per half minute rather
/// than a silence the user reads as a hang.
pub(crate) const MISSION_RETRY_CAP: std::time::Duration = std::time::Duration::from_secs(30);

/// How much of a tool's failure is sent back to the model.
///
/// `{error:#}` is the whole anyhow cause chain, written for a developer sitting
/// at the terminal: nested absolute paths, internal hostnames, a token pasted
/// into a failed URL, a stack of "while doing X" contexts. All of it would go
/// out to the provider and stay in the history for the rest of the session —
/// and a model that can read the plumbing tends to imitate it. The model needs
/// to know *which call failed and for what reason*; that fits in a line.
pub(crate) const MAX_TOOL_ERROR_BYTES: usize = 500;

/// A bounded, single-line summary of a tool failure, for the transcript.
///
/// The top-level message is kept (it names the failure), the cause chain is
/// dropped, newlines are collapsed so one error cannot impersonate several
/// turns of text, and the tail is cut at a character boundary with the cut
/// marked. A synthetic "tool panicked" line is short and passes through here
/// unchanged.
pub(crate) fn summarize_tool_error(error: &anyhow::Error) -> String {
    let one_line = error.to_string().trim().replace(['\n', '\r'], " ");
    if one_line.len() <= MAX_TOOL_ERROR_BYTES {
        return one_line;
    }
    let mut end = MAX_TOOL_ERROR_BYTES;
    while end > 0 && !one_line.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}… [truncated]", &one_line[..end])
}

/// Waits until a stop is requested for the turn this receiver belongs to.
///
/// `false` means the requester went away without asking — nobody will ever
/// ask, so the caller stops selecting on this signal and waits on the stream
/// alone.
pub(crate) async fn stop_flag(rx: &mut tokio::sync::watch::Receiver<bool>) -> bool {
    loop {
        if *rx.borrow_and_update() {
            return true;
        }
        if rx.changed().await.is_err() {
            return false;
        }
    }
}

/// What one streaming attempt produced, including how it failed.
///
/// A failed attempt still carries the text that arrived before the failure:
/// the caller either discards it for a retry or commits it, and losing it in
/// transit would make those two indistinguishable.
#[derive(Default)]
pub(crate) struct StepOutput {
    /// Assistant text streamed so far.
    pub(crate) text: String,
    /// Tool-call fragments assembled so far.
    pub(crate) calls: Vec<PartialCallView>,
    /// Usage the provider reported, when it got that far.
    pub(crate) usage: TokenUsage,
    /// The provider's finish reason, when it sent one.
    pub(crate) finish_reason: Option<String>,
    /// Whether any reasoning fragment arrived.
    pub(crate) saw_reasoning: bool,
    /// The failure that ended this attempt, if any.
    pub(crate) error: Option<anyhow::Error>,
}

impl StepOutput {
    /// The provider put out nothing at all: no text, no tool call, no thinking.
    pub(crate) fn is_empty(&self) -> bool {
        self.text.is_empty() && self.calls.is_empty() && !self.saw_reasoning
    }
}

/// Learn this route's real context window from a provider error.
///
/// The built-in table is a guess about a model id, and it is wrong on exactly
/// the routes that matter — a gateway serving an id whose window is smaller than
/// the family name implies. The rejection message is the provider's own
/// statement about the route in front of us, so the number is recorded in the
/// store the context policy consults, and the next `should_compact` measures
/// against the real limit instead of the table's estimate.
///
/// Nothing about the failure changes here: the error is neither consumed nor
/// reworded, the retry classification is untouched, and the turn still ends the
/// way it did before. Only a number is taken from it.
pub(crate) fn record_reported_window(error: &anyhow::Error, model: &str) {
    // The whole chain is flattened because the number usually sits one or more
    // layers down: the streaming open labels the failure with the URL it posted
    // to, above the request error whose message carries the provider's words.
    let message = format!("{error:#}");
    if let Some(tokens) = llm::reported_window(&message) {
        window::reported_windows().record(model, tokens);
    }
}
