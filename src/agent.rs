//! The agent loop.
//!
//! One turn is: send the history, stream the reply, dispatch any requested
//! tools **in parallel**, append their results, and repeat until the model
//! stops asking for tools.
//!
//! Three rules from the spec live here:
//!
//! * **Independent tool calls run concurrently.** Serial dispatch is the single
//!   largest avoidable cost in an agent turn — tool execution accounts for
//!   roughly a third to two thirds of total request time.
//! * **Each turn takes one snapshot.** Settings are read once at the start of a
//!   turn, so a hot reload landing mid-turn cannot change a request already in
//!   flight. A turn sees changes; a turn never sees half of one.
//! * **Timing is measured, not guessed.** Every turn reports how long the model
//!   took versus how long tools took, because overhead that cannot be measured
//!   cannot be fixed.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use tokio::task::JoinSet;

use crate::compaction::{self, Compaction, Summarizer};
use crate::llm::{self, LlmClient, LlmEvent, PartialCallView, TokenUsage};
use crate::message::{Message, ToolCall};
use crate::tools::longhorizon::MissionBrief;
use crate::tools::ToolRegistry;
use crate::window::{self, ContextPolicy, ContextWindow};

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

/// Everything one turn needs, frozen at the start of that turn.
#[derive(Clone)]
pub struct TurnSettings {
    /// System prompt for this turn.
    pub system_prompt: String,
    /// Tool table for this turn.
    pub tools: Arc<ToolRegistry>,
    /// Model id for this turn.
    pub model: String,
    /// Configuration version this snapshot came from.
    pub version: u64,
    /// Context window, compaction threshold, and how much survives a compaction.
    ///
    /// It travels with the rest of the turn's settings so that changing the
    /// window in a settings panel takes effect on the next turn rather than
    /// requiring a restart.
    pub policy: Arc<dyn ContextPolicy>,
    /// Extra fragments to place in this request, and where.
    pub injection: Arc<crate::context::InjectionEngine>,
    /// Cache policy, which decides whether a prefix injection is allowed.
    pub cache_policy: crate::hotreload::CachePolicy,
}

/// Supplies turn settings.
///
/// The runtime implements this so a change lands on the *next* turn. Nothing
/// in the loop reads mutable configuration directly.
pub trait TurnConfig: Send + Sync {
    /// Freeze the settings for one turn.
    fn settings(&self) -> TurnSettings;
}

/// A configuration that never changes.
///
/// Useful for tests, for embedding the kernel without a runtime, and as the
/// fallback when no hot-reload runtime is in play.
pub struct StaticConfig {
    /// System prompt.
    pub system_prompt: String,
    /// Tool table.
    pub tools: Arc<ToolRegistry>,
    /// Model id.
    pub model: String,
    /// Context window, threshold, and how much survives a compaction.
    pub policy: Arc<dyn ContextPolicy>,
    /// Extra fragments to place in each request.
    pub injection: Arc<crate::context::InjectionEngine>,
    /// Cache policy, which decides whether a prefix injection is allowed.
    pub cache_policy: crate::hotreload::CachePolicy,
}

impl StaticConfig {
    /// Build a static configuration.
    pub fn new(
        system_prompt: impl Into<String>,
        tools: Arc<ToolRegistry>,
        model: impl Into<String>,
    ) -> Self {
        Self {
            system_prompt: system_prompt.into(),
            tools,
            model: model.into(),
            policy: Arc::new(window::DefaultContextPolicy::default()),
            injection: Arc::new(crate::context::InjectionEngine::new()),
            cache_policy: crate::hotreload::CachePolicy::default(),
        }
    }

    /// Use a specific context policy.
    pub fn with_policy(mut self, policy: Arc<dyn ContextPolicy>) -> Self {
        self.policy = policy;
        self
    }
}

impl TurnConfig for StaticConfig {
    fn settings(&self) -> TurnSettings {
        TurnSettings {
            system_prompt: self.system_prompt.clone(),
            tools: Arc::clone(&self.tools),
            model: self.model.clone(),
            version: 0,
            policy: Arc::clone(&self.policy),
            injection: Arc::clone(&self.injection),
            cache_policy: self.cache_policy,
        }
    }
}

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

/// How many times one answer may be continued after hitting the provider's
/// output ceiling, before the loop stops extending it.
const MAX_CONTINUATIONS: usize = 3;

/// What a stopped turn reports as its error.
const STOPPED: &str = "stopped by the user";

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

/// How many batches one turn may take while a long-horizon mission is running.
///
/// A running mission lifts [`CONTINUATION_BATCHES`]: the mission, not the shell,
/// says when the work is over, so the turn keeps being granted batches for as
/// long as the mission's status says `running` and the user has not stopped it.
///
/// This ceiling exists only so a wedged model cannot run forever: it is
/// `MISSION_BATCH_CEILING × max_steps` steps of absolute limit, and reaching it
/// is a bug report, not a normal outcome — a mission that is genuinely too large
/// for one turn is what `mission_tick` is for, and a model that cannot decide
/// whether to finish or block after this many rounds is not going to. The turn
/// says so, naming the mission, and stops; the mission itself is untouched.
const MISSION_BATCH_CEILING: usize = 64;

/// How many continuation asks one turn may hand out while a running mission has
/// produced **no** `mission_*` tool call since the previous ask.
///
/// The ask is an invitation, not a life sentence: a model that answers 「收到」
/// to every nudge without ever touching a mission tool is telling us it will
/// not engage, and holding the turn for [`MISSION_BATCH_CEILING`] batches turns
/// that into a freeze the user reads as a hang. Any `mission_*` call resets the
/// budget — engagement is what keeps the turn open, not the status line on disk.
const MISSION_IDLE_NUDGES: usize = 2;

/// The message the shell injects while a mission is still running.
///
/// Chinese, like the other text this loop inserts into a turn: the model that
/// started the mission reads that message, and this one is a continuation of it.
///
/// It names the mission, its objective, the recorded next step and the done-when
/// so the model can act on it without re-reading the mission files, and it names
/// the only way out that is not "keep working": blocking the mission with a
/// reason. Nothing here can end the mission — the shell has no opinion about
/// whether the work is finished, and saying "give me your final answer" is
/// exactly what made a turn end mid-task.
fn mission_nudge(mission: &MissionBrief, total: usize) -> String {
    format!(
        "本回合已经连续 {total} 轮，任务 {id} 仍是 running（目标：{objective}；下一步：{step}；\
         完成判据：{done_when}）。不要结束回合：继续推进任务（mission_tick / mission_advance）。\
         如果你确实无法继续，调用 mission_advance 把状态改成 blocked 并说明原因。",
        total = total,
        id = mission.id,
        objective = or_unset(&mission.objective),
        step = or_unset(&mission.step),
        done_when = or_unset(&mission.done_when),
    )
}

/// `value`, or a placeholder for a field the mission never recorded.
///
/// A mission may be started with no `done_when` and no first step, and an
/// injected message saying "完成判据：" and nothing reads as a field the model
/// failed to fill in rather than one that was never asked for.
fn or_unset(value: &str) -> &str {
    if value.is_empty() {
        "（未设）"
    } else {
        value
    }
}

/// How much of a tool's failure is sent back to the model.
///
/// `{error:#}` is the whole anyhow cause chain, written for a developer sitting
/// at the terminal: nested absolute paths, internal hostnames, a token pasted
/// into a failed URL, a stack of "while doing X" contexts. All of it would go
/// out to the provider and stay in the history for the rest of the session —
/// and a model that can read the plumbing tends to imitate it. The model needs
/// to know *which call failed and for what reason*; that fits in a line.
const MAX_TOOL_ERROR_BYTES: usize = 500;

/// A bounded, single-line summary of a tool failure, for the transcript.
///
/// The top-level message is kept (it names the failure), the cause chain is
/// dropped, newlines are collapsed so one error cannot impersonate several
/// turns of text, and the tail is cut at a character boundary with the cut
/// marked. A synthetic "tool panicked" line is short and passes through here
/// unchanged.
fn summarize_tool_error(error: &anyhow::Error) -> String {
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
async fn stop_flag(rx: &mut tokio::sync::watch::Receiver<bool>) -> bool {
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
struct StepOutput {
    /// Assistant text streamed so far.
    text: String,
    /// Tool-call fragments assembled so far.
    calls: Vec<PartialCallView>,
    /// Usage the provider reported, when it got that far.
    usage: TokenUsage,
    /// The provider's finish reason, when it sent one.
    finish_reason: Option<String>,
    /// Whether any reasoning fragment arrived.
    saw_reasoning: bool,
    /// The failure that ended this attempt, if any.
    error: Option<anyhow::Error>,
}

impl StepOutput {
    /// The provider put out nothing at all: no text, no tool call, no thinking.
    fn is_empty(&self) -> bool {
        self.text.is_empty() && self.calls.is_empty() && !self.saw_reasoning
    }
}

/// Drives one conversation.
pub struct Agent {
    client: LlmClient,
    config: Arc<dyn TurnConfig>,
    messages: Vec<Message>,
    max_steps: usize,
    observer: Arc<dyn AgentObserver>,
    /// Decides how a summary is written.
    ///
    /// The window and threshold come from the turn's settings instead, so a
    /// change made in a settings panel lands on the next turn.
    summarizer: Arc<dyn Summarizer>,
    /// Injection engine override, for tests and embedding.
    injection_override: Option<Arc<crate::context::InjectionEngine>>,
    /// A stop request for this turn, when the host can cancel it.
    ///
    /// Checked between steps, between stream events, and between tool joins —
    /// a stop lands at the next safe point, keeps everything already
    /// committed, and ends the turn with [`STOPPED`] as its error.
    cancel: Option<tokio::sync::watch::Receiver<bool>>,
    /// Pending user messages the host wants delivered while this turn runs.
    ///
    /// Drained at the top of every step: what the user typed mid-turn joins
    /// the history exactly where a claim is legal — before the next request
    /// is built — so the model reads it on its next step. `None` for hosts
    /// without steering (CLI, subagents).
    inbox: Option<Arc<std::sync::Mutex<VecDeque<String>>>>,
    /// The session this turn belongs to, when the host knows it.
    ///
    /// Resolved once at the start of [`Agent::run`] and handed to every tool
    /// call, so a host that re-homes its "current session" mid-turn (the
    /// desktop does, on every switch) cannot move a running turn's workspace
    /// or its job attribution. `None` means "ask the registry at turn start".
    identity: Option<crate::tools::ToolIdentity>,
}

impl Agent {
    /// Build an agent over a configuration source and an existing history.
    pub fn new(client: LlmClient, config: Arc<dyn TurnConfig>, messages: Vec<Message>) -> Self {
        Self {
            client,
            config,
            messages,
            max_steps: 50,
            observer: Arc::new(SilentObserver),
            summarizer: Arc::new(compaction::ModelSummarizer::default()),
            injection_override: None,
            cancel: None,
            inbox: None,
            identity: None,
        }
    }

    /// Bind this turn to one session and its workspace.
    ///
    /// Hosts that serve more than one conversation (the desktop, an embedded
    /// server) pass the session they are running for. Without it the turn
    /// falls back to whatever the registry has pinned when the turn starts —
    /// still a snapshot, never a per-call read.
    pub fn with_identity(mut self, identity: crate::tools::ToolIdentity) -> Self {
        self.identity = Some(identity);
        self
    }

    /// Build an agent over a fixed configuration.
    pub fn with_static(
        client: LlmClient,
        tools: Arc<ToolRegistry>,
        system_prompt: impl Into<String>,
        messages: Vec<Message>,
    ) -> Self {
        let model = client.model().to_string();
        Self::new(
            client,
            Arc::new(StaticConfig::new(system_prompt, tools, model)),
            messages,
        )
    }

    /// Cap the number of model steps in one batch.
    ///
    /// A turn may run up to `1 + CONTINUATION_BATCHES` batches of this length:
    /// a batch that spends its steps asks the model whether to continue instead
    /// of ending the turn, so `steps` bounds one batch and not the whole turn.
    /// While a mission is running this is still the size of one batch, but the
    /// number of batches is decided by the mission rather than by that constant.
    pub fn with_max_steps(mut self, steps: usize) -> Self {
        self.max_steps = steps.max(1);
        self
    }

    /// Attach an observer.
    pub fn with_observer(mut self, observer: Arc<dyn AgentObserver>) -> Self {
        self.observer = observer;
        self
    }

    /// Make this turn stoppable: when the flag flips, the loop finishes at
    /// its next safe point, keeps every message it committed, and reports
    /// "stopped by the user" as the turn's error.
    pub fn with_cancel(mut self, cancel: tokio::sync::watch::Receiver<bool>) -> Self {
        self.cancel = Some(cancel);
        self
    }

    /// Deliver host-queued user messages at step boundaries.
    ///
    /// The queue is drained, not sampled: everything the user said while the
    /// turn ran is claimed before the next request is built.
    pub fn with_inbox(mut self, inbox: Arc<std::sync::Mutex<VecDeque<String>>>) -> Self {
        self.inbox = Some(inbox);
        self
    }

    /// Whether a stop has already been requested for this turn.
    fn stop_requested(&self) -> bool {
        self.cancel.as_ref().is_some_and(|rx| *rx.borrow())
    }

    /// The mission this turn may not walk away from, if one is running.
    ///
    /// Read from the turn's own identity, not from the process: a mission is a
    /// directory in one session's workspace, and a turn that was never bound to
    /// a workspace is a nested worker (a subagent, a `mission_tick` worker) —
    /// it already *is* one step of somebody else's mission, and holding it open
    /// would mean a mission that cannot run its own steps.
    ///
    /// Re-read at every decision point rather than remembered for the turn: the
    /// batch that just ran may have blocked or finished the mission, and that
    /// status — not the step count — is what decides whether the turn may end.
    fn running_mission(&self) -> Option<MissionBrief> {
        let workspace = self.identity.as_ref()?.workspace.as_deref()?;
        crate::tools::longhorizon::running_mission(workspace)
    }


    /// Replace the summarizer: how a compaction's summary is produced.
    pub fn with_summarizer(mut self, summarizer: Arc<dyn Summarizer>) -> Self {
        self.summarizer = summarizer;
        self
    }

    /// Set the injection engine used for the next turn.
    pub fn set_injection(&mut self, injection: Arc<crate::context::InjectionEngine>) {
        self.injection_override = Some(injection);
    }


    /// The conversation history.
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    /// Replace the conversation history.
    pub fn set_messages(&mut self, messages: Vec<Message>) {
        self.messages = messages;
    }

    /// The configuration version the next turn will run under.
    pub fn config_version(&self) -> u64 {
        self.config.settings().version
    }

    /// Run one turn: append `prompt` and loop until the model stops calling
    /// tools, a stop is requested, or the step budget is spent.
    ///
    /// The budget is `max_steps` per batch for `1 + CONTINUATION_BATCHES`
    /// batches: a spent batch asks the model whether to continue rather than
    /// ending the turn.
    ///
    /// One thing outranks all of that: a long-horizon mission whose record still
    /// says `running`. Its turn does not end on a final answer and is not held to
    /// `1 + CONTINUATION_BATCHES` — see [`MISSION_BATCH_CEILING`] for the only
    /// bound that remains. The shell never decides that such a mission is
    /// finished; it only refuses to walk away from it.
    pub async fn run(&mut self, prompt: &str) -> Result<AgentOutcome> {
        let user_message = Message::user(prompt);
        self.messages.push(user_message.clone());
        // The prompt is committed before anything is attempted: asking the
        // model is already a thing that happened.
        self.observer.on_message(&user_message);

        // Frozen for the whole turn: a reload landing now is next turn's news.
        let settings = self.config.settings();

        // The tool table is frozen with them, for a reason of its own: the
        // desktop releases a conversation's plugins when that conversation is
        // left, and a background turn can still be running. Reading the live
        // table per step would let a tool the model has already seen vanish
        // between two steps, and the turn would end on "no tool named …"
        // halfway through its work. Copying the table is cheap — every handler
        // is an `Arc` — and the copy is what keeps the code it names alive
        // until this turn returns.
        let tools = Arc::new((*settings.tools).clone());

        // The session identity is frozen the same way, and for a sharper
        // reason: the desktop pins "the current session" onto one shared tool
        // table, so reading it per call would let a switch mid-turn re-home
        // this turn's `bash` cwd, relative paths and job ownership. One read,
        // at turn start, then every tool call in this turn carries it.
        let identity = self
            .identity
            .clone()
            .unwrap_or_else(|| tools.default_identity());

        let mut outcome = AgentOutcome {
            text: String::new(),
            steps: 0,
            tool_calls: 0,
            usage: TokenUsage::default(),
            new_messages: vec![user_message],
            timing: TurnTiming::default(),
            config_version: settings.version,
            injections: Vec::new(),
            compactions: Vec::new(),
            error: None,
            retries: 0,
        };

        let schemas = tools.schemas();

        // What the provider last reported for the prompt. This is the only
        // accurate size signal available: estimating from characters drifts
        // badly once tool output is involved.
        let mut last_prompt_tokens: Option<usize> = None;

        // How much output was cut off by the provider's ceiling and continued
        // so far. One answer, possibly delivered in several chunks.
        let mut continuations: usize = 0;

        // One place decides how hard a failing request is retried: the same
        // network settings the send-phase retries read.
        let network = self.client.network().clone();
        let max_attempts = 1 + network.retry_attempts;

        // The cap bounds one batch of steps, not the whole turn. Spending a
        // batch without a final answer asks the model whether a round count
        // this long is normal, tells it to keep working, and earns another
        // batch — see [`CONTINUATION_BATCHES`] for why that is bounded, and
        // [`MISSION_BATCH_CEILING`] for why a running mission outranks it.
        //
        // One counter drives all of it. `step` is the global index across every
        // batch, so `outcome.steps` and the observer report the work the turn
        // actually did instead of restarting at 1 with every batch and hiding
        // two of them from the ledger.
        //
        // The steps are counted by hand rather than by a range, because the
        // allowance is no longer known when the turn starts: a mission that is
        // still running earns batches the count cannot predict.
        //
        // `saturating_mul` rather than `*`: a caller passing a nonsensical cap
        // must get a very long turn, not a panic in the multiplication before
        // the first step. `max(1)` likewise guards a future caller that writes
        // the field directly: a batch that runs zero steps would spin.
        let batch_size = self.max_steps.max(1);
        let step_budget = batch_size.saturating_mul(1 + CONTINUATION_BATCHES);
        let mut step: usize = 0;
        let mut renewals: usize = 0;
        // Batches granted because a mission was running. Bounded by
        // [`MISSION_BATCH_CEILING`], and counted separately from `renewals` so
        // the two kinds of continuation never pay for each other's allowance.
        let mut mission_batches: usize = 0;
        // `mission_*` tool calls so far. They are what the idle budget below is
        // spent against: the nudge is an ask, and a model that answers it without
        // ever touching a mission tool is telling us it will not engage.
        let mut mission_calls: usize = 0;
        // Continuation asks handed out since the last `mission_*` call.
        let mut idle_nudges: usize = 0;

        // One outer loop, one decision: whether the turn may end here. The inner
        // loop is one batch and the code below it is the boundary between
        // batches, so granting another batch — for a spent cap or for a mission
        // — is a `continue` rather than a second copy of the step body.
        'batches: loop {
            // Whether this batch stopped early because the model produced a
            // final answer, rather than because it spent its last step. The
            // boundary says something different in each case: a model that just
            // answered must not be asked whether a round count this long is
            // normal.
            let mut answered = false;

            for _ in 0..batch_size {
                step += 1;
                // A stop between steps ends the turn here: everything committed
                // so far is already in the history and is reported with it.
                if self.stop_requested() {
                    outcome.error = Some(STOPPED.to_string());
                    return Ok(outcome);
                }

                // Steering claimed at the boundary: what the user typed while this
                // turn ran joins the history before the next request is built, so
                // the model reads it on this very step. Claimed messages are
                // committed like any other — they are model-visible from here on.
                let inbox = self.inbox.clone();
                if let Some(inbox) = inbox {
                    // A panic in another thread while the inbox was locked poisons
                    // the mutex. The queue itself is intact — it is a plain
                    // `VecDeque` with no invariant a torn write could break, and
                    // `drain` leaves it empty either way — so the step continues
                    // with what is there. Panicking here instead would fail every
                    // remaining step of the turn for a lock that has nothing left
                    // to corrupt.
                    let texts: Vec<String> = match inbox.lock() {
                        Ok(mut queue) => queue.drain(..).collect(),
                        Err(poisoned) => poisoned.into_inner().drain(..).collect(),
                    };
                    for text in texts {
                        let message = Message::user(text);
                        self.messages.push(message.clone());
                        outcome.new_messages.push(message.clone());
                        self.observer.on_message(&message);
                    }
                }

                // The global step index, not the batch's: a caller reading
                // `steps` must see the turn's whole work.
                outcome.steps = step;
                self.observer.on_step(step);

                // Compact before the request, not after a failure. A provider that
                // rejects an oversized prompt has already cost the round trip.
                if let Some(tokens) = last_prompt_tokens {
                    let window = settings.policy.window(&settings.model);
                    if settings.policy.should_compact(tokens, window) {
                        let client = self.summary_client(&settings.model);
                        match self.compact(&client, window, settings.policy.as_ref()).await {
                            Ok(Some(record)) => {
                                outcome.compactions.push(record);
                                // The prompt just changed size; the old measurement
                                // no longer describes it.
                                last_prompt_tokens = None;
                            }
                            Ok(None) => {}
                            Err(error) => {
                                // A failed summary must not end the turn: the
                                // conversation can still continue, and the next step
                                // will try again.
                                eprintln!("[compaction] failed: {error:#}");
                            }
                        }
                    }
                }

                let (request, injection) = self.build_request(&settings);
                if !injection.is_empty() {
                    self.observer.on_injection(&injection);
                    let mut texts = injection.prefix.clone();
                    texts.extend(injection.tail.iter().cloned());
                    texts.extend(injection.at_depth.iter().map(|(_, _, t)| t.clone()));
                    outcome.injections.push(crate::session::InjectionRecord {
                        activated: injection.activated.clone(),
                        relocated: injection.relocated.clone(),
                        budget_used: injection.budget_used,
                        texts,
                    });
                }
                // The route may name a different model than the client was built
                // with; switching is a clone, not a reconnect.
                let client = self.client.with_model(&settings.model);

                // One step is one or more streaming attempts. A transient
                // failure or an empty response is retried with backoff; only a
                // permanent failure or the attempt budget ends the step in error,
                // and even then the turn's progress so far is kept.
                let mut attempt: u32 = 0;
                let mut out = loop {
                    // A stop during the backoff ends the attempt loop without
                    // another request; the error branch below reports it.
                    if self.stop_requested() {
                        let mut stopped = StepOutput::default();
                        stopped.error = Some(anyhow::anyhow!(STOPPED));
                        break stopped;
                    }
                    attempt += 1;
                    let started = Instant::now();
                    let produced = self.stream_once(&client, &request, &schemas).await;
                    outcome.timing.model_ms += started.elapsed().as_millis();

                    match &produced.error {
                        Some(error) => {
                            // A provider that refuses an oversized prompt names the
                            // window it enforces, and that is a fact about this
                            // route rather than about the model id. It is recorded
                            // here, before the retry decision, so the next
                            // compaction check measures against it — the failure
                            // itself still travels exactly as far as it did before.
                            record_reported_window(error, &settings.model);
                            let retry = llm::retryable(error);
                            if retry && attempt < max_attempts {
                                let message = format!("{error:#}");
                                outcome.retries += 1;
                                self.observer.on_step_retry(attempt as usize, &message);
                                tokio::time::sleep(network.backoff(attempt)).await;
                                continue;
                            }
                        }
                        None if produced.is_empty() && attempt < max_attempts => {
                            outcome.retries += 1;
                            self.observer.on_step_retry(
                                attempt as usize,
                                "the provider returned an empty response",
                            );
                            tokio::time::sleep(network.backoff(attempt)).await;
                            continue;
                        }
                        None => {}
                    }
                    break produced;
                };

                outcome.usage.input += out.usage.input;
                outcome.usage.output += out.usage.output;
                outcome.usage.cached += out.usage.cached;
                // Cache *writes* are billed at a premium and reasoning tokens are
                // billed as output, so dropping either under-reports what the turn
                // cost. Both are summed here for the same reason as the other
                // counters: the ledger is the only place a user sees the bill.
                outcome.usage.cache_created += out.usage.cache_created;
                outcome.usage.reasoning += out.usage.reasoning;

                // Remember the prompt size the provider just reported, so the next
                // step can decide whether history needs reducing.
                if out.usage.input > 0 {
                    last_prompt_tokens = Some(out.usage.input as usize);
                }

                // The step ended in failure. Whatever text arrived before it is
                // committed — the transcript must show what the model actually
                // said — and the error says why the loop stopped.
                if let Some(error) = out.error.take() {
                    if !out.text.is_empty() {
                        let partial = Message::assistant_tools(Vec::new(), Some(out.text.clone()));
                        self.messages.push(partial.clone());
                        outcome.new_messages.push(partial);
                        self.observer.on_message(self.messages.last().expect("just pushed"));
                    }
                    outcome.error = Some(format!("{error:#}"));
                    return Ok(outcome);
                }

                // The attempt budget ran out on a response with nothing in it.
                // Reporting this as an error matters: a silent end reads as a
                // finished turn when the task never even started.
                if out.is_empty() {
                    outcome.error = Some(format!(
                        "the model returned an empty response after {attempt} attempt(s)"
                    ));
                    return Ok(outcome);
                }

                let indexed: Vec<(usize, PartialCallView)> = out
                    .calls
                    .iter()
                    .enumerate()
                    .map(|(i, c)| (i, c.clone()))
                    .collect();
                let mut tool_calls: Vec<ToolCall> = llm::assemble_calls(&indexed);
                let text = out.text;

                // Stopped between the answer and its tool calls: the calls never
                // ran, so they are dropped rather than left unanswered — a tool
                // call with no result is a request no provider accepts next turn.
                let stopped_before_tools = self.stop_requested() && !tool_calls.is_empty();
                if stopped_before_tools {
                    tool_calls.clear();
                }

                let assistant = Message::assistant_tools(
                    tool_calls.clone(),
                    if text.is_empty() { None } else { Some(text.clone()) },
                );
                self.messages.push(assistant.clone());
                outcome.new_messages.push(assistant);
                self.observer.on_message(self.messages.last().expect("just pushed"));
                // One string across a continuation chain: each cut-off segment
                // appends to the answer it belongs to, a plain step replaces it.
                if continuations > 0 {
                    outcome.text.push_str(&text);
                } else {
                    outcome.text = text;
                }

                // The answer stops at the provider's output ceiling with no tool
                // call to execute. Without a continuation the turn ends looking
                // complete while the task is half-done — an interruption exactly
                // where small models produce it most.
                if out.finish_reason.as_deref() == Some("length")
                    && tool_calls.is_empty()
                    && continuations < MAX_CONTINUATIONS
                    && !stopped_before_tools
                {
                    continuations += 1;
                    let nudge = Message::user(
                        "Your previous reply was cut off by the output token limit. \
                         Continue exactly where you left off, without repeating \
                         anything you already wrote.",
                    );
                    self.messages.push(nudge.clone());
                    outcome.new_messages.push(nudge);
                    self.observer.on_message(self.messages.last().expect("just pushed"));
                    self.observer
                        .on_continuation(&format!("step {step} hit the output limit"));
                    continue;
                }

                if tool_calls.is_empty() {
                    if stopped_before_tools {
                        outcome.error = Some(STOPPED.to_string());
                        return Ok(outcome);
                    }
                    // The model answered. Whether that ends the turn is not decided
                    // here: a running mission says the work is not over, and the
                    // boundary below is the one place that weighs the two. Breaking
                    // out of the batch keeps this decision single instead of growing
                    // a second copy of it under an `if`.
                    answered = true;
                    break;
                }

                // Dispatch every call concurrently, then restore call order.
                // A `mission_*` call is the only progress that justifies holding
                // the turn open, so it resets the idle budget the boundary
                // spends: engagement is the difference between a mission being
                // worked and a mission being acknowledged.
                let mission_calls_before = mission_calls;
                mission_calls += tool_calls
                    .iter()
                    .filter(|call| call.name.starts_with("mission_"))
                    .count();
                if mission_calls != mission_calls_before {
                    idle_nudges = 0;
                }
                let tools_started = Instant::now();
                let mut set: JoinSet<(usize, ToolCall, Result<crate::tools::ToolOutput>)> = JoinSet::new();
                // A stop that landed while the answer was being committed is seen
                // *before* the batch starts. The user has already walked away, and
                // starting a whole set of tools anyway spends side effects nobody
                // will read. The calls are answered below regardless, so the
                // history stays complete either way.
                let mut stopped = self.stop_requested();
                if !stopped {
                    for (position, call) in tool_calls.iter().enumerate() {
                        let registry = Arc::clone(&tools);
                        let call = call.clone();
                        let identity = identity.clone();
                        self.observer.on_tool_start(&call.name, &call.arguments);
                        set.spawn(async move {
                            let result = registry
                                .execute_as(identity, &call.name, &call.arguments)
                                .await;
                            (position, call, result)
                        });
                    }
                }

                let mut results: Vec<Option<(ToolCall, Result<crate::tools::ToolOutput>)>> =
                    (0..tool_calls.len()).map(|_| None).collect();
                // Collecting is cancellable: a bash call that runs for minutes
                // must not outlive the user's decision to stop watching it.
                let mut stop = self.cancel.clone();
                let mut stop_closed = false;
                'collect: loop {
                    if stop_closed {
                        stop = None;
                        stop_closed = false;
                    }
                    let joined = if let Some(stop_rx) = &mut stop {
                        tokio::select! {
                            biased;
                            requested = stop_flag(stop_rx) => {
                                if requested {
                                    stopped = true;
                                    break 'collect;
                                }
                                stop_closed = true;
                                continue 'collect;
                            }
                            joined = set.join_next() => joined,
                        }
                    } else {
                        set.join_next().await
                    };
                    match joined {
                        Some(Ok((position, call, result))) => results[position] = Some((call, result)),
                        Some(Err(failed)) => {
                            // The tool task died: a panic inside the handler, or an
                            // abort from the runtime. Its slot stays empty and the
                            // loop below answers that call explicitly.
                            //
                            // Escaping the turn with `failed` here (what this arm
                            // used to do) skipped that fill-in with the assistant
                            // message already committed: a `tool_call` with no
                            // result. Every later request built from that history
                            // is refused by the provider with a 400, so one
                            // panicking tool cost the whole conversation instead of
                            // one call.
                            eprintln!("[tools] task failed: {failed}");
                        }
                        // Every task finished: nothing left to wait for.
                        None => break 'collect,
                    }
                }
                outcome.timing.tools_ms += tools_started.elapsed().as_millis();
                if stopped {
                    // The in-flight calls are killed; the answers below fill in
                    // for them either way.
                    set.abort_all();
                }

                // Images a tool produced cannot travel in its tool message: the
                // format restricts that content to text. They are collected here and
                // attached to one user message after all the tool results, which is
                // the only place the format allows them.
                let mut attached: Vec<(String, crate::message::ImageAttachment)> = Vec::new();

                for (position, call) in tool_calls.iter().enumerate() {
                    let result = match results[position].take() {
                        Some((_, result)) => result,
                        // Killed mid-call: the call still gets an answer, because
                        // an unanswered tool call is a history the next request
                        // cannot be built from.
                        None if stopped => Err(anyhow::anyhow!(
                            "stopped by the user before this tool call finished"
                        )),
                        // A task that died without a stop request: it panicked or
                        // was aborted, so nothing was written into its slot. The
                        // call is answered explicitly anyway — the same rule as
                        // above, and this arm *is* reachable: a panicking tool is
                        // exactly what lands here.
                        None => Err(anyhow::anyhow!("tool panicked: {}", call.name)),
                    };
                    outcome.tool_calls += 1;
                    let (ok, body) = match result {
                        Ok(output) => {
                            for image in output.images {
                                attached.push((call.name.clone(), image));
                            }
                            (true, output.text)
                        }
                        Err(error) => (false, format!("error: {}", summarize_tool_error(&error))),
                    };
                    self.observer.on_tool_end(&call.name, ok, &body);
                    let message = Message::tool_result(call.id.clone(), call.name.clone(), body);
                    self.messages.push(message.clone());
                    outcome.new_messages.push(message);
                    self.observer.on_message(self.messages.last().expect("just pushed"));
                }

                if !attached.is_empty() {
                    let mut text = String::from("Image(s) produced by the tool call above:");
                    for (tool, image) in &attached {
                        text.push_str(&format!(
                            "\n- from {tool}: {}",
                            image.label.as_deref().unwrap_or("image")
                        ));
                    }
                    let message =
                        Message::user_with_images(text, attached.into_iter().map(|(_, i)| i).collect());
                    self.messages.push(message.clone());
                    outcome.new_messages.push(message);
                    self.observer.on_message(self.messages.last().expect("just pushed"));
                }

                // Every dispatched call now has an answer and the turn was asked
                // to stop: report it instead of taking another step.
                if stopped {
                    outcome.error = Some(STOPPED.to_string());
                    return Ok(outcome);
                }
            }

            // A batch boundary, and the one place in this loop that decides
            // whether the turn may end. Reached either because the batch spent
            // its last step with tool calls still coming, or because the model
            // produced a final answer and broke out above.
            //
            // The mission is re-read here rather than remembered from the top of
            // the batch: a `mission_advance` inside the batch may have blocked or
            // finished it, and that status — never a step count — is what makes a
            // turn endable.
            let mission = self.running_mission();

            match mission {
                // No mission: exactly what this loop did before, unchanged. An
                // answer ends the turn, a spent allowance ends it, and a batch
                // that ends without an answer asks the model whether a round
                // count this long is normal.
                None if answered => return Ok(outcome),
                None if step >= step_budget => break 'batches,
                None => {
                    renewals += 1;
                    let ask = format!(
                        "本回合已经连续 {total} 轮（单批上限 {max}）没有给出最终答复。\
                         对话轮数过长是否正常？如果还需要继续，就继续工作；如果已经可以收尾，请现在直接给出结论。",
                        total = step,
                        max = self.max_steps,
                    );
                    // Also on stderr: without it, a turn that keeps running past its
                    // cap is indistinguishable from a stuck one in a terminal or a
                    // desktop log.
                    eprintln!(
                        "[steps] {step} steps without a final answer; asking the model whether to continue"
                    );
                    // Committed exactly like steering: this is a user-role message,
                    // so the model reads it on the next request and the transcript
                    // shows it rather than leaving the turn inexplicably longer.
                    let message = Message::user(ask);
                    self.messages.push(message.clone());
                    outcome.new_messages.push(message.clone());
                    self.observer.on_message(&message);
                    continue 'batches;
                }
                // The mission is still running, so the turn may not end — not
                // because the model asked for more time, but because the work it
                // took on is unfinished on disk. Nor may the shell end the
                // mission: nothing in this file writes `mission.json`, and
                // nothing here decides that work is done. It is named as still
                // running, with the only two ways out — keep going, or block it
                // with a reason — and the turn goes round again.
                Some(mission) => {
                    // A stop outranks the mission. The user walking away ends
                    // the turn, and nothing is injected after it: a message
                    // asking a model to keep working on a turn nobody is
                    // watching is exactly the run-away this loop must not do.
                    if self.stop_requested() {
                        outcome.error = Some(STOPPED.to_string());
                        return Ok(outcome);
                    }
                    // The idle guard outranks the ceiling: a model that answers
                    // every nudge with words and no `mission_*` call has said,
                    // in the only language this loop can measure, that it will
                    // not engage. MISSION_IDLE_NUDGES asks is the whole allowance
                    // for that; ending the turn here is what keeps 「卡死」 from
                    // being the product's answer to a polite model. The mission
                    // is untouched — the next turn starts from the same disk.
                    if idle_nudges >= MISSION_IDLE_NUDGES {
                        outcome.text = format!(
                            "任务 {id} 仍是 running，但模型已经 {nudges} 次没有用 mission_* 工具回应推进请求，\
                             本回合到此为止以免看起来像卡死。任务本身没有被改动；\
                             继续它请明确要求模型调用 mission_tick / mission_advance。",
                            id = mission.id,
                            nudges = idle_nudges,
                        );
                        return Ok(outcome);
                    }
                    mission_batches += 1;
                    if mission_batches > MISSION_BATCH_CEILING {
                        // Reaching this is a bug report, not an outcome: the
                        // mission is still running after the ceiling worth of
                        // batches, so no continuation count will ever satisfy
                        // it. Say what happened, name the mission, and leave the
                        // mission exactly as it is — the next turn (or
                        // `mission_tick`) continues from disk.
                        outcome.text = format!(
                            "任务 {id} 仍是 running，但本回合已被任务批次上限（{ceiling} 批 × {max} 轮，\
                             共 {limit} 轮）停下：只有模型能结束任务，而它既没有完成也没有说明阻塞。\
                             任务本身没有被改动：继续它请用 mission_tick，或用 mission_advance 记录进度。",
                            id = mission.id,
                            ceiling = MISSION_BATCH_CEILING,
                            max = batch_size,
                            limit = MISSION_BATCH_CEILING.saturating_mul(batch_size),
                        );
                        return Ok(outcome);
                    }
                    let nudge = mission_nudge(&mission, step);
                    idle_nudges += 1;
                    // On stderr for the same reason as the cap above: a turn
                    // held open by a mission is otherwise indistinguishable from
                    // a stuck one in a terminal or a desktop log.
                    eprintln!(
                        "[mission] {step} steps in, mission {} is still running; the turn continues",
                        mission.id
                    );
                    // Committed like steering, because that is what it is: a
                    // message the model reads on the next request, that the
                    // transcript shows, and that survives the turn.
                    let message = Message::user(nudge);
                    self.messages.push(message.clone());
                    outcome.new_messages.push(message.clone());
                    self.observer.on_message(&message);
                    continue 'batches;
                }
            }
        }

        // Every batch is spent and no mission was holding the turn open. The
        // turn stops — that bound is what keeps a model that never concludes
        // from acting and billing forever — and it says how many steps it really
        // took and how often it was asked, so a capped turn is distinguishable
        // from a finished one and greppable in a log.
        //
        // `step` rather than `step_budget`: they are the same number on this
        // path, and the count keeps being honest if a mission held the turn open
        // for a while before the model blocked it.
        outcome.text = format!(
            "stopped after {} steps without a final answer ({} continuation(s) asked)",
            step, renewals
        );
        Ok(outcome)
    }

    /// One streaming attempt for the current step.
    ///
    /// It does not fail: a transport error before the response, or partway
    /// through it, lands in `error` carrying whatever text already arrived, so
    /// the caller can choose between retrying and committing the partial.
    async fn stream_once(
        &self,
        client: &LlmClient,
        request: &[Message],
        schemas: &[serde_json::Value],
    ) -> StepOutput {
        let mut out = StepOutput::default();

        // Waiting for the response's first byte is part of the turn: a slow
        // gateway can hold it for seconds, and a stop arriving there has to
        // land just as it does during streaming.
        let mut stop = self.cancel.clone();
        let mut stop_closed = false;
        let opened = loop {
            // Dropping the signal here would need the borrow this loop holds;
            // the flag is one-shot, so clearing it once and taking the plain
            // path below is enough.
            if stop_closed {
                stop = None;
            }
            let attempt = if let Some(stop_rx) = &mut stop {
                tokio::select! {
                    biased;
                    requested = stop_flag(stop_rx) => {
                        if requested {
                            out.error = Some(anyhow::anyhow!(STOPPED));
                            return out;
                        }
                        // Nobody is asking anymore; open the stream plainly.
                        stop_closed = true;
                        continue;
                    }
                    opened = client.chat_stream(request, schemas) => opened,
                }
            } else {
                client.chat_stream(request, schemas).await
            };
            break attempt;
        };
        let mut rx = match opened {
            Ok(rx) => rx,
            Err(error) => {
                out.error = Some(error);
                return out;
            }
        };

        // The stream is cancellable: text already shown is kept (the caller
        // commits it), and the stop ends this attempt instead of waiting for
        // a model that may keep generating for another minute.
        let mut stop = self.cancel.clone();
        let mut stop_closed = false;
        'stream: loop {
            if stop_closed {
                stop = None;
                stop_closed = false;
            }
            let event = if let Some(stop_rx) = &mut stop {
                tokio::select! {
                    biased;
                    requested = stop_flag(stop_rx) => {
                        if requested {
                            out.error = Some(anyhow::anyhow!(STOPPED));
                            break 'stream;
                        }
                        // Nobody is asking anymore; wait on the stream alone.
                        stop_closed = true;
                        continue 'stream;
                    }
                    event = rx.recv() => event,
                }
            } else {
                rx.recv().await
            };
            let Some(event) = event else { break 'stream; };
            match event {
                Ok(LlmEvent::TextDelta(delta)) => {
                    self.observer.on_text(&delta);
                    out.text.push_str(&delta);
                }
                Ok(LlmEvent::ReasoningDelta(delta)) => {
                    self.observer.on_reasoning(&delta);
                    out.saw_reasoning = true;
                }
                Ok(LlmEvent::ToolCallDelta { index, id, name, arguments }) => {
                    while out.calls.len() <= index {
                        out.calls.push(PartialCallView::default());
                    }
                    let slot = &mut out.calls[index];
                    if let Some(id) = id {
                        slot.id = id;
                    }
                    if let Some(name) = name {
                        slot.name = name;
                    }
                    if let Some(args) = arguments {
                        slot.arguments.push_str(&args);
                    }
                }
                Ok(LlmEvent::Usage(usage)) => out.usage = usage,
                Ok(LlmEvent::Finished { reason }) => out.finish_reason = reason,
                Err(error) => {
                    out.error = Some(error);
                    break 'stream;
                }
            }
        }
        out
    }

    /// The client to use for summarization.
    ///
    /// A summarizer may name its own model — summaries are a good place to spend
    /// less — and otherwise the session's model is used.
    fn summary_client(&self, session_model: &str) -> LlmClient {
        match self.summarizer.model() {
            Some(model) if !model.trim().is_empty() => self.client.with_model(model),
            _ => self.client.with_model(session_model),
        }
    }

    /// Reduce the history: summarize the oldest messages and stand them in.
    ///
    /// Returns the record when a compaction happened, so the caller can persist
    /// it. `None` means there was nothing that could safely be removed.
    async fn compact(
        &mut self,
        client: &LlmClient,
        window: ContextWindow,
        policy: &dyn ContextPolicy,
    ) -> Result<Option<Compaction>> {
        let Some(plan) = compaction::plan(&self.messages, policy.keep_recent()) else {
            return Ok(None);
        };

        self.observer.on_compaction_start(plan.summarize, window);
        let history = &self.messages[..plan.summarize];
        let summary = compaction::summarize(client, self.summarizer.as_ref(), history).await?;

        self.messages = compaction::apply(&self.messages, plan, &summary);
        let record = Compaction {
            replaced: plan.summarize,
            summary,
        };
        self.observer.on_compaction(&record);
        Ok(Some(record))
    }

    /// Build the request: system prompt, injected fragments, then the history.
    ///
    /// The system prompt is prepended at request time rather than stored in the
    /// history, so a prompt change does not rewrite recorded messages — which
    /// is also what keeps the cached prefix intact across turns.
    ///
    /// Injected fragments never enter `self.messages`. They are recomputed for
    /// every request, so a fragment that stops matching simply stops appearing,
    /// and the recorded conversation stays exactly what was said.
    fn build_request(&self, settings: &TurnSettings) -> (Vec<Message>, crate::context::Injection) {
        let window = settings.policy.window(&settings.model);
        let engine = self.injection_override.as_ref().unwrap_or(&settings.injection);
        let injection = engine.inject(&self.messages, window.tokens, settings.cache_policy);

        let mut request = Vec::with_capacity(self.messages.len() + 4);

        if !settings.system_prompt.is_empty() {
            request.push(Message::system(&settings.system_prompt));
        }
        // A prefix fragment is a standing rule: it belongs above the history,
        // and under CacheFirst the engine has already moved it elsewhere rather
        // than dropping it.
        for text in &injection.prefix {
            request.push(Message::system(text.clone()));
        }

        // Depth is counted from the end, so depth 1 means "immediately before
        // the newest message" and the numbering stays stable as history grows.
        //
        // An insertion must never land between an assistant's tool call and the
        // tool messages answering it: providers reject that outright. So each
        // depth is resolved to the nearest earlier position that is not a `tool`
        // message.
        let len = self.messages.len();
        let mut insert_at: std::collections::BTreeMap<usize, Vec<(crate::message::Role, String)>> =
            std::collections::BTreeMap::new();
        for (depth, role, text) in &injection.at_depth {
            let mut index = len.saturating_sub(*depth).min(len.saturating_sub(1));
            while index > 0 && self.messages[index].role == crate::message::Role::Tool {
                index -= 1;
            }
            insert_at
                .entry(index)
                .or_default()
                .push((*role, text.clone()));
        }

        for (index, message) in self.messages.iter().enumerate() {
            if let Some(fragments) = insert_at.get(&index) {
                for (role, text) in fragments {
                    request.push(Message::of_role(*role, text.clone()));
                }
            }
            request.push(message.clone());
        }

        // Tail fragments go after everything: the model reads them last, which
        // is where a reminder about the current request is most useful.
        for text in &injection.tail {
            request.push(Message::user(text.clone()));
        }

        (request, injection)
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
fn record_reported_window(error: &anyhow::Error, model: &str) {
    // The whole chain is flattened because the number usually sits one or more
    // layers down: the streaming open labels the failure with the URL it posted
    // to, above the request error whose message carries the provider's words.
    let message = format!("{error:#}");
    if let Some(tokens) = llm::reported_window(&message) {
        window::reported_windows().record(model, tokens);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_static_config_reports_its_settings() {
        let tools = Arc::new(ToolRegistry::with_base_tools().unwrap());
        let config = StaticConfig::new("prompt", Arc::clone(&tools), "test-model");
        let settings = config.settings();
        assert_eq!(settings.system_prompt, "prompt");
        assert_eq!(settings.model, "test-model");
        assert_eq!(settings.version, 0);
        assert!(settings.tools.get("read").is_some());
    }

    #[test]
    fn the_request_prepends_the_system_prompt_without_storing_it() {
        let tools = Arc::new(ToolRegistry::with_base_tools().unwrap());
        let config: Arc<dyn TurnConfig> = Arc::new(StaticConfig::new("SYS", Arc::clone(&tools), "m"));
        let client = LlmClient::new(crate::llm::LlmConfig::new("http://localhost:1", "k", "m"))
            .expect("client builds");
        let agent = Agent::new(client, Arc::clone(&config), vec![Message::user("hi")]);

        let request = agent.build_request(&config.settings()).0;
        assert_eq!(request.len(), 2);
        assert_eq!(request[0].role, crate::message::Role::System);
        assert_eq!(request[0].text(), "SYS");
        assert_eq!(request[1].text(), "hi");

        // The history itself is untouched, so recording stays clean.
        assert_eq!(agent.messages().len(), 1);

        // An empty prompt contributes no message at all.
        let bare: Arc<dyn TurnConfig> = Arc::new(StaticConfig::new("", Arc::clone(&tools), "m"));
        assert_eq!(agent.build_request(&bare.settings()).0.len(), 1);
    }

    #[test]
    fn an_injected_prefix_fragment_sits_above_the_history() {
        let tools = Arc::new(ToolRegistry::with_base_tools().unwrap());
        let mut engine = crate::context::InjectionEngine::new();
        engine.add(
            crate::context::InjectionEntry::constant("rule", "never edit vendor/")
                .at(crate::context::PositionKind::Prefix),
        );

        let config: Arc<dyn TurnConfig> = Arc::new(
            StaticConfig::new("SYS", tools, "m")
                .with_policy(Arc::new(window::DefaultContextPolicy::default())),
        );
        let client = LlmClient::new(crate::llm::LlmConfig::new("http://localhost:1", "k", "m"))
            .expect("client builds");
        let mut agent = Agent::new(client, Arc::clone(&config), vec![Message::user("hi")]);
        agent.set_injection(Arc::new(engine));

        let (request, injection) = agent.build_request(&config.settings());
        assert_eq!(injection.activated, vec!["rule"]);
        assert_eq!(request.len(), 3, "system, injected rule, history");
        assert_eq!(request[0].text(), "SYS");
        assert_eq!(request[1].text(), "never edit vendor/");
        assert_eq!(request[2].text(), "hi");

        // Injected text never enters the recorded history.
        assert_eq!(agent.messages().len(), 1);
    }

    #[test]
    fn an_injected_tail_fragment_comes_last() {
        let tools = Arc::new(ToolRegistry::with_base_tools().unwrap());
        let mut engine = crate::context::InjectionEngine::new();
        engine.add(
            crate::context::InjectionEntry::constant("note", "remember the deadline")
                .at(crate::context::PositionKind::HistoryTail),
        );

        let config: Arc<dyn TurnConfig> = Arc::new(StaticConfig::new("SYS", tools, "m"));
        let client = LlmClient::new(crate::llm::LlmConfig::new("http://localhost:1", "k", "m"))
            .expect("client builds");
        let mut agent = Agent::new(client, Arc::clone(&config), vec![Message::user("hi")]);
        agent.set_injection(Arc::new(engine));

        let (request, _) = agent.build_request(&config.settings());
        assert_eq!(request.len(), 3);
        assert_eq!(request.last().unwrap().text(), "remember the deadline");
    }

    #[test]
    fn an_injected_depth_fragment_never_splits_a_tool_pair() {
        // A provider rejects a request where an assistant's tool_calls message
        // is not immediately followed by its tool results, so an insertion that
        // would land there has to move earlier.
        let tools = Arc::new(ToolRegistry::with_base_tools().unwrap());
        let mut engine = crate::context::InjectionEngine::new();
        engine.add(crate::context::InjectionEntry::constant("deep", "inserted").at_depth(1));

        let config: Arc<dyn TurnConfig> = Arc::new(StaticConfig::new("SYS", tools, "m"));
        let client = LlmClient::new(crate::llm::LlmConfig::new("http://localhost:1", "k", "m"))
            .expect("client builds");

        let calling = Message::assistant_tools(
            vec![ToolCall {
                id: "c1".into(),
                name: "read".into(),
                arguments: "{}".into(),
            }],
            None,
        );
        let result = Message::tool_result("c1", "read", "contents");
        let mut agent = Agent::new(
            client,
            Arc::clone(&config),
            vec![Message::user("please read"), calling, result],
        );
        agent.set_injection(Arc::new(engine));

        let (request, _) = agent.build_request(&config.settings());

        // Find the assistant call and assert the very next message is its result.
        let call_index = request
            .iter()
            .position(|m| !m.tool_calls.is_empty())
            .expect("the call survives");
        assert_eq!(
            request[call_index + 1].role,
            crate::message::Role::Tool,
            "the call must be immediately followed by its result: {request:#?}"
        );
    }

    #[test]
    fn an_injected_depth_fragment_lands_that_many_messages_back() {
        let tools = Arc::new(ToolRegistry::with_base_tools().unwrap());
        let mut engine = crate::context::InjectionEngine::new();
        engine.add(
            crate::context::InjectionEntry::constant("deep", "inserted here").at_depth(1),
        );

        let config: Arc<dyn TurnConfig> = Arc::new(StaticConfig::new("SYS", tools, "m"));
        let client = LlmClient::new(crate::llm::LlmConfig::new("http://localhost:1", "k", "m"))
            .expect("client builds");
        let mut agent = Agent::new(
            client,
            Arc::clone(&config),
            vec![Message::user("first"), Message::user("second")],
        );
        agent.set_injection(Arc::new(engine));

        let (request, _) = agent.build_request(&config.settings());
        // system, first, injected (depth 1 = before the newest), second.
        assert_eq!(request.len(), 4);
        assert_eq!(request[1].text(), "first");
        assert_eq!(request[2].text(), "inserted here");
        assert_eq!(request[3].text(), "second");
    }

    #[test]
    fn settings_are_frozen_per_turn() {
        use std::sync::atomic::{AtomicU64, Ordering};

        struct Counting {
            calls: AtomicU64,
            tools: Arc<ToolRegistry>,
        }

        impl TurnConfig for Counting {
            fn settings(&self) -> TurnSettings {
                let n = self.calls.fetch_add(1, Ordering::Relaxed);
                TurnSettings {
                    system_prompt: format!("v{n}"),
                    tools: Arc::clone(&self.tools),
                    model: "m".into(),
                    version: n,
                    policy: Arc::new(window::DefaultContextPolicy::default()),
                    injection: Arc::new(crate::context::InjectionEngine::new()),
                    cache_policy: crate::hotreload::CachePolicy::default(),
                }
            }
        }

        let tools = Arc::new(ToolRegistry::with_base_tools().unwrap());
        let config: Arc<dyn TurnConfig> = Arc::new(Counting { calls: AtomicU64::new(0), tools });
        let client = LlmClient::new(crate::llm::LlmConfig::new("http://localhost:1", "k", "m"))
            .expect("client builds");
        let agent = Agent::new(client, Arc::clone(&config), Vec::new());

        // Each read advances the counter, which is what makes "one snapshot per
        // turn" observable rather than assumed.
        assert_eq!(agent.config.settings().version, 0);
        assert_eq!(agent.config.settings().version, 1);
        assert_eq!(agent.config_version(), 2);
    }

    #[test]
    fn an_empty_step_is_nothing_at_all() {
        let empty = StepOutput::default();
        assert!(empty.is_empty());

        let mut spoken = StepOutput::default();
        spoken.text = "hi".into();
        assert!(!spoken.is_empty());

        let mut thought = StepOutput::default();
        thought.saw_reasoning = true;
        assert!(!thought.is_empty());

        let mut called = StepOutput::default();
        called.calls.push(PartialCallView::default());
        assert!(!called.is_empty());
    }

    /// A provider answering canned HTTP responses in order, repeating the
    /// last one, one response per connection.
    ///
    /// Enough of the wire for the loop under test: the loop's decisions are
    /// made from statuses, SSE frames, and connection endings, not from
    /// anything else the request carries — so the request head and body are
    /// drained and ignored.
    async fn provider(
        responses: Vec<String>,
    ) -> (String, std::sync::Arc<std::sync::atomic::AtomicUsize>) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let hits = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let served = std::sync::Arc::clone(&hits);

        tokio::spawn(async move {
            while let Ok((mut socket, _)) = listener.accept().await {
                // Drain the request head byte-wise, then the declared body, so
                // a close after the response cannot race a client still
                // writing and turn every test into a flake.
                let mut head = Vec::new();
                loop {
                    let mut byte = [0u8; 1];
                    match socket.read(&mut byte).await {
                        Ok(0) | Err(_) => break,
                        Ok(_) => {
                            head.push(byte[0]);
                            if head.ends_with(b"\r\n\r\n") {
                                break;
                            }
                        }
                    }
                }
                let head_text = String::from_utf8_lossy(&head);
                let declared = head_text
                    .lines()
                    .filter_map(|line| {
                        let (key, value) = line.split_once(':')?;
                        key.eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse::<usize>().ok())
                            .flatten()
                    })
                    .next()
                    .unwrap_or(0);
                let mut remaining = declared;
                let mut buf = [0u8; 4096];
                while remaining > 0 {
                    match socket.read(&mut buf).await {
                        Ok(0) | Err(_) => break,
                        Ok(read) => remaining = remaining.saturating_sub(read),
                    }
                }

                let index = served.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let body = responses
                    .get(index)
                    .or_else(|| responses.last())
                    .cloned()
                    .unwrap_or_default();
                let _ = socket.write_all(body.as_bytes()).await;
                let _ = socket.flush().await;
            }
        });
        (format!("http://{addr}"), hits)
    }

    /// One SSE response carrying a single content frame with `finish`.
    fn sse_answer(text: &str, finish: &str) -> String {
        let frame = serde_json::json!({
            "choices": [{ "delta": { "content": text }, "finish_reason": finish }]
        });
        format!(
            "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\nconnection: close\r\n\r\ndata: {frame}\n\ndata: [DONE]\n\n"
        )
    }

    fn agent_against(base: String) -> Agent {
        let tools = Arc::new(ToolRegistry::with_base_tools().unwrap());
        let config: Arc<dyn TurnConfig> = Arc::new(StaticConfig::new("SYS", tools, "test-model"));
        let client =
            LlmClient::new(crate::llm::LlmConfig::new(base, "k", "test-model")).unwrap();
        Agent::new(client, config, Vec::new())
    }

    #[tokio::test]
    async fn a_truncated_answer_is_continued_rather_than_taken_as_finished() {
        // A small model hits the output ceiling mid-answer. The turn must go
        // on asking for the rest instead of ending half-done and looking
        // complete.
        let (base, hits) = provider(vec![
            sse_answer("part one ", "length"),
            sse_answer("part two", "stop"),
        ])
        .await;
        let mut agent = agent_against(base);

        let outcome = agent.run("go").await.expect("the loop finishes");

        assert!(outcome.error.is_none(), "{:?}", outcome.error);
        assert_eq!(outcome.steps, 2, "the cut-off step runs again");
        assert_eq!(outcome.text, "part one part two");
        assert_eq!(
            hits.load(std::sync::atomic::Ordering::Relaxed),
            2,
            "one request per step"
        );

        // The history keeps both segments and the nudge between them: the
        // model must see where it was cut off to continue from there.
        let messages = agent.messages();
        assert_eq!(messages.len(), 4);
        assert_eq!(messages[0].text(), "go");
        assert_eq!(messages[1].text(), "part one ");
        assert!(messages[2].text().contains("cut off"), "{:?}", messages[2].text());
        assert_eq!(messages[3].text(), "part two");
    }

    #[tokio::test]
    async fn a_transient_provider_failure_is_retried_within_the_step() {
        let (base, hits) = provider(vec![
            "HTTP/1.1 503 Service Unavailable\r\nconnection: close\r\n\r\noverloaded"
                .to_string(),
            sse_answer("recovered", "stop"),
        ])
        .await;
        let mut agent = agent_against(base);

        let outcome = agent.run("go").await.expect("the loop finishes");

        assert!(outcome.error.is_none(), "{:?}", outcome.error);
        assert_eq!(outcome.retries, 1);
        assert_eq!(hits.load(std::sync::atomic::Ordering::Relaxed), 2);
        assert_eq!(outcome.text, "recovered");
        // Only the final attempt's answer enters the history — the failed one
        // produced nothing worth keeping.
        assert_eq!(agent.messages().len(), 2);
    }

    #[tokio::test]
    async fn a_permanent_failure_reports_an_error_without_losing_the_turn() {
        let (base, hits) = provider(vec![
            "HTTP/1.1 400 Bad Request\r\nconnection: close\r\n\r\nmalformed".to_string(),
        ])
        .await;
        let mut agent = agent_against(base);

        let outcome = agent.run("go").await.expect("the loop still answers");

        let error = outcome.error.expect("a 400 must surface as the turn's error");
        assert!(error.contains("400"), "{error}");
        assert_eq!(outcome.retries, 0, "a rejected request is not retried");
        assert_eq!(hits.load(std::sync::atomic::Ordering::Relaxed), 1);
        // The prompt itself survives: the next turn continues from what was
        // said, not from a conversation that pretends the ask never happened.
        assert_eq!(outcome.new_messages.len(), 1);
        assert_eq!(outcome.new_messages[0].text(), "go");
    }

    #[tokio::test]
    async fn a_rejected_prompt_teaches_the_route_its_real_window() {
        // A gateway whose model accepts far less than the family table assumes
        // answers an oversized prompt with a 400 that names the limit. The turn
        // must still fail exactly as it did before, and the next compaction
        // decision must measure against the number the provider stated instead
        // of against the table's estimate for the model id.
        //
        // The id is unique because the report is stored process-wide: recording
        // one for a name another test resolves would make these tests depend on
        // the order they ran in.
        let model = format!("window-probe-{}", uuid::Uuid::new_v4().simple());
        let body = serde_json::json!({
            "error": { "message": "This model's maximum context length is 8192 tokens" }
        })
        .to_string();
        let (base, hits) = provider(vec![format!(
            "HTTP/1.1 400 Bad Request\r\ncontent-type: application/json\r\nconnection: close\r\n\r\n{body}"
        )])
        .await;

        let tools = Arc::new(ToolRegistry::with_base_tools().unwrap());
        let config: Arc<dyn TurnConfig> = Arc::new(StaticConfig::new("SYS", tools, model.clone()));
        let client = LlmClient::new(crate::llm::LlmConfig::new(base, "k", model.clone())).unwrap();
        let mut agent = Agent::new(client, config, Vec::new());

        let outcome = agent.run("go").await.expect("the loop still answers");

        assert!(outcome.error.is_some(), "a rejected request still fails the turn");
        assert_eq!(
            hits.load(std::sync::atomic::Ordering::Relaxed),
            1,
            "and it is still not retried"
        );
        assert_eq!(
            window::reported_windows().get(&model),
            Some(8192),
            "the provider's own number is learned from its refusal"
        );
        let resolved = window::DefaultContextPolicy::default().window(&model);
        assert_eq!(resolved.tokens, 8192);
        assert_eq!(resolved.source, window::WindowSource::Provider);
    }

    #[tokio::test]
    async fn an_empty_response_is_retried_and_then_reported() {
        let (base, hits) = provider(vec!["HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\nconnection: close\r\n\r\ndata: [DONE]\n\n".to_string()])
            .await;
        let mut agent = agent_against(base);

        let outcome = agent.run("go").await.expect("the loop still answers");

        let error = outcome.error.expect("silence must not read as success");
        assert!(error.contains("empty response"), "{error}");
        // Default budget: one attempt plus three retries.
        assert_eq!(outcome.retries, 3);
        assert_eq!(hits.load(std::sync::atomic::Ordering::Relaxed), 4);
    }

    #[tokio::test]
    async fn a_poisoned_inbox_does_not_end_the_turn() {
        // A panic in another thread while the inbox was locked leaves the mutex
        // poisoned. `expect("inbox")` then panics on *every* remaining step of
        // the turn, so one unrelated panic silently breaks the whole loop. The
        // queue is a plain `VecDeque` and is still intact, so the step goes on
        // with what is there.
        //
        // The panic printed while this test sets that up is the point: poisoning
        // can only happen by panicking, so it cannot be arranged quietly.
        let inbox: Arc<std::sync::Mutex<VecDeque<String>>> =
            Arc::new(std::sync::Mutex::new(VecDeque::new()));
        let doomed = Arc::clone(&inbox);
        let _ = std::thread::spawn(move || {
            let mut queue = doomed.lock().expect("a fresh lock");
            queue.push_back("said while the thread was dying".to_string());
            panic!("this panic is what poisons the inbox");
        })
        .join();
        assert!(inbox.lock().is_err(), "the mutex is poisoned");

        let (base, hits) = provider(vec![sse_answer("answered", "stop")]).await;
        let mut agent = agent_against(base).with_inbox(Arc::clone(&inbox));

        let outcome = agent.run("go").await.expect("the turn must not panic");

        assert!(outcome.error.is_none(), "{:?}", outcome.error);
        assert_eq!(outcome.text, "answered");
        assert_eq!(hits.load(std::sync::atomic::Ordering::Relaxed), 1);

        // What was queued before the poison still reached the model, and the
        // queue was drained rather than left to be re-read on every step.
        let texts: Vec<String> = agent
            .messages()
            .iter()
            .map(|message| message.text().to_string())
            .collect();
        assert!(
            texts.iter().any(|text| text == "said while the thread was dying"),
            "the steering text survived: {texts:?}"
        );
        let drained = match inbox.lock() {
            Ok(queue) => queue.is_empty(),
            Err(poisoned) => poisoned.into_inner().is_empty(),
        };
        assert!(drained, "claimed once, not on every step");
    }

    /// A registry holding exactly one tool, built from a handler.
    fn registry_with(
        name: &str,
        handler: impl Fn(serde_json::Value) -> crate::tools::ToolFuture + Send + Sync + 'static,
    ) -> Arc<ToolRegistry> {
        let mut registry = ToolRegistry::new();
        registry
            .register(
                crate::tools::ToolDef::new(
                    name,
                    "a tool that does exactly what this test needs",
                    serde_json::json!({ "type": "object", "properties": {} }),
                    "test",
                    handler,
                ),
                crate::tools::ConflictPolicy::Error,
            )
            .expect("the tool registers");
        Arc::new(registry)
    }

    /// One SSE response asking for a single tool call.
    fn sse_tool_call(id: &str, name: &str, arguments: &str) -> String {
        let frame = serde_json::json!({
            "choices": [{
                "delta": { "tool_calls": [{
                    "index": 0,
                    "id": id,
                    "function": { "name": name, "arguments": arguments }
                }]},
                "finish_reason": "tool_calls"
            }]
        });
        format!(
            "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\nconnection: close\r\n\r\ndata: {frame}\n\ndata: [DONE]\n\n"
        )
    }

    /// An agent against `base`, with `tools` and a cap of one step per batch.
    ///
    /// The mock provider repeats its last response, so a script that ends on a
    /// tool call keeps the model asking for tools until the whole step budget
    /// is spent. A test that wants one tool round and then an end scripts a
    /// final answer as the last response.
    fn agent_with_tools(base: String, tools: Arc<ToolRegistry>) -> Agent {
        let config: Arc<dyn TurnConfig> = Arc::new(StaticConfig::new("SYS", tools, "test-model"));
        let client = LlmClient::new(crate::llm::LlmConfig::new(base, "k", "test-model"))
            .expect("client builds");
        Agent::new(client, config, Vec::new()).with_max_steps(1)
    }

    /// The tool message answering call `id`, if the history holds one.
    fn answer_to(agent: &Agent, id: &str) -> Option<String> {
        agent.messages().iter().find_map(|message| {
            (message.role == crate::message::Role::Tool
                && message.tool_call_id.as_deref() == Some(id))
            .then(|| message.text().to_string())
        })
    }

    #[tokio::test]
    async fn a_panicking_tool_is_answered_instead_of_ending_the_turn() {
        // A panic inside a tool makes its task return a `JoinError`. The
        // assistant message that asked for the call is already in the history,
        // so letting that error escape leaves a call with no result — a history
        // no provider accepts, refused with a 400 on every later request, which
        // ends the conversation rather than the call. The call is answered
        // instead, and the turn goes on.
        let tools = registry_with("boom", |_arguments| -> crate::tools::ToolFuture {
            Box::pin(async move {
                panic!("the tool went down");
            })
        });
        // The script ends on a final answer. The provider repeats its last
        // response, and the cap is one step per batch, so a script that ended on
        // the tool call would keep the model calling it until the whole step
        // budget was spent — this test is about the panic, not the budget.
        let (base, hits) = provider(vec![
            sse_tool_call("call_1", "boom", "{}"),
            sse_answer("done", "stop"),
        ])
        .await;
        let mut agent = agent_with_tools(base, tools);

        let outcome = agent
            .run("go")
            .await
            .expect("a panicking tool must not end the turn");

        assert!(outcome.error.is_none(), "{:?}", outcome.error);
        assert_eq!(outcome.tool_calls, 1);
        assert_eq!(
            hits.load(std::sync::atomic::Ordering::Relaxed),
            2,
            "the tool round, then the answer that follows the ask to continue"
        );

        // The pairing, asserted where a request builder looks for it: the call
        // is immediately followed by its result, and the result names the call.
        let messages = agent.messages();
        let call_at = messages
            .iter()
            .position(|message| !message.tool_calls.is_empty())
            .expect("the call was committed");
        assert_eq!(messages[call_at].tool_calls[0].id, "call_1");
        assert_eq!(messages[call_at + 1].role, crate::message::Role::Tool);
        assert_eq!(
            messages[call_at + 1].tool_call_id.as_deref(),
            Some("call_1")
        );

        // The model reads a failure it can act on, not a missing answer.
        let answer = answer_to(&agent, "call_1").expect("the call has an answer");
        assert!(answer.starts_with("error:"), "{answer}");
        assert!(answer.contains("tool panicked: boom"), "{answer}");
    }

    #[tokio::test]
    async fn a_tool_failure_reaches_the_model_as_a_bounded_summary() {
        // The whole cause chain is a developer artifact — nested contexts,
        // absolute paths, internal hostnames, a token pasted into a failed URL —
        // and it enters the history and travels to the provider on every request
        // after it. The model gets the top-level reason, bounded and on one line;
        // the chain underneath stays in the log.
        let tools = registry_with("explode", |_arguments| -> crate::tools::ToolFuture {
            Box::pin(async move {
                let long = "C:\\Users\\someone\\project\\".to_string() + &"nested\\".repeat(200);
                let failed: Result<crate::tools::ToolOutput> = Err(anyhow::anyhow!(
                    "no such device (os error 6)"
                )
                .context(format!("copy failed for {long}artifact.bin")));
                failed
            })
        });
        // One tool round, then a final answer: the failure summary is what this
        // test measures, so the turn is allowed to end after the call instead of
        // running the whole step budget.
        let (base, _) = provider(vec![
            sse_tool_call("call_2", "explode", "{}"),
            sse_answer("done", "stop"),
        ])
        .await;
        let mut agent = agent_with_tools(base, tools);

        let outcome = agent.run("go").await.expect("the turn finishes");
        assert!(outcome.error.is_none(), "{:?}", outcome.error);

        let answer = answer_to(&agent, "call_2").expect("the call has an answer");
        assert!(answer.starts_with("error: copy failed"), "{answer}");
        assert!(
            answer.len() <= "error: ".len() + MAX_TOOL_ERROR_BYTES + "… [truncated]".len(),
            "the failure is bounded; it was {} bytes",
            answer.len()
        );
        assert!(answer.ends_with("[truncated]"), "{answer}");
        assert!(
            !answer.contains("no such device"),
            "the cause chain stays out of the transcript: {answer}"
        );
        assert!(!answer.contains('\n'), "one line, not several: {answer}");
    }

    #[test]
    fn a_long_tool_error_is_cut_at_a_character_boundary() {
        // Cutting by byte would split a multi-byte character and put U+FFFD in
        // the transcript. Chinese text is the easy way to reach the limit.
        let error = anyhow::anyhow!("读取失败：{}", "路径".repeat(400));
        let summary = summarize_tool_error(&error);
        assert!(summary.ends_with("[truncated]"), "{summary}");
        assert!(summary.len() <= MAX_TOOL_ERROR_BYTES + "… [truncated]".len());
        assert!(!summary.contains('\u{FFFD}'), "no torn characters: {summary}");
        assert!(summary.starts_with("读取失败："), "the reason survives");

        // A short failure passes through untouched, newlines collapsed.
        let short = summarize_tool_error(&anyhow::anyhow!("line one\nline two"));
        assert_eq!(short, "line one line two");
    }

    /// A tool that only returns, so a script that keeps calling it spends steps
    /// instead of measuring a tool's behaviour.
    fn tick_tools() -> Arc<ToolRegistry> {
        registry_with("tick", |_arguments| -> crate::tools::ToolFuture {
            Box::pin(async move {
                let answered: Result<crate::tools::ToolOutput> =
                    Ok(crate::tools::ToolOutput::text("tick"));
                answered
            })
        })
    }

    /// The part of the continuation ask a test matches on.
    ///
    /// A fragment rather than the whole sentence, so rewording the question the
    /// model is asked does not break the tests that only care that it was asked.
    const CONTINUATION_FRAGMENT: &str = "对话轮数过长是否正常";

    /// How many of `messages` carry the continuation ask.
    fn continuation_asks<'a>(messages: impl IntoIterator<Item = &'a Message>) -> usize {
        messages
            .into_iter()
            .filter(|message| message.text().contains(CONTINUATION_FRAGMENT))
            .count()
    }

    #[tokio::test]
    async fn the_step_cap_asks_the_model_to_continue_instead_of_ending_the_turn() {
        // A model that only ever asks for tools never produces a final answer.
        // The cap used to end the turn there, abandoning a long task mid-work
        // with nothing but a note saying it had been abandoned. Reaching it now
        // asks whether a round count this long is normal, tells the model to
        // keep working, and grants a bounded number of batches of the same
        // length.
        let (base, hits) = provider(vec![sse_tool_call("call_1", "tick", "{}")]).await;
        let mut agent = agent_with_tools(base, tick_tools()).with_max_steps(2);

        let outcome = agent.run("go").await.expect("the loop finishes");

        assert!(outcome.error.is_none(), "{:?}", outcome.error);
        assert_eq!(
            continuation_asks(&outcome.new_messages),
            CONTINUATION_BATCHES,
            "one ask per spent batch, and no more: {:?}",
            outcome
                .new_messages
                .iter()
                .map(Message::text)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            outcome.steps,
            2 * (1 + CONTINUATION_BATCHES),
            "each renewal grants another full cap of steps"
        );
        assert_eq!(
            hits.load(std::sync::atomic::Ordering::Relaxed),
            outcome.steps,
            "every one of those steps was a real request"
        );
        assert_eq!(
            outcome.text,
            format!(
                "stopped after {} steps without a final answer ({} continuation(s) asked)",
                2 * (1 + CONTINUATION_BATCHES),
                CONTINUATION_BATCHES
            ),
            "the report is honest about the real total and what was asked"
        );
        // The ask is a request, not a remark: it has to say what to do next.
        assert!(
            outcome
                .new_messages
                .iter()
                .any(|message| message.text().contains("如果已经可以收尾，请现在直接给出结论")),
            "the ask carries the instruction to keep working"
        );

        // Committed to the conversation, not merely reported: a message the
        // model never reads is not a request to continue.
        assert_eq!(continuation_asks(agent.messages()), CONTINUATION_BATCHES);
    }

    #[tokio::test]
    async fn the_continuation_message_reaches_the_observer() {
        // Hosts persist the conversation from `on_message`, and that is what the
        // user reads as the transcript. An ask the model receives but the
        // transcript never shows makes a turn that overran its cap look like one
        // that hung.
        #[derive(Default)]
        struct Recorder {
            seen: std::sync::Mutex<Vec<String>>,
        }

        impl AgentObserver for Recorder {
            fn on_message(&self, message: &Message) {
                self.seen
                    .lock()
                    .expect("the recorder lock is not poisoned")
                    .push(message.text().to_string());
            }
        }

        let (base, _) = provider(vec![sse_tool_call("call_1", "tick", "{}")]).await;
        let recorder = Arc::new(Recorder::default());
        // Cloned through the binding, not `Arc::clone(&recorder)`: the turbofish
        // is inferred from the argument, so the unsizing to `dyn AgentObserver`
        // has to happen where the expected type is known.
        let observing: Arc<dyn AgentObserver> = recorder.clone();
        let mut agent = agent_with_tools(base, tick_tools())
            .with_max_steps(1)
            .with_observer(observing);

        let outcome = agent.run("go").await.expect("the loop finishes");

        let seen = recorder
            .seen
            .lock()
            .expect("the recorder lock is not poisoned")
            .clone();
        assert_eq!(
            seen.iter()
                .filter(|text| text.contains(CONTINUATION_FRAGMENT))
                .count(),
            CONTINUATION_BATCHES,
            "the transcript shows every ask: {seen:?}"
        );
        // The prompt is observed too, so the count above is not simply "every
        // committed message".
        assert_eq!(seen.first().map(String::as_str), Some("go"));
        assert_eq!(outcome.steps, 1 + CONTINUATION_BATCHES);
    }

    #[tokio::test]
    async fn a_turn_that_finishes_early_never_asks_for_a_continuation() {
        // The renewal must be invisible to a turn that ends on its own: asking a
        // finished model whether to keep working would invite it to invent more
        // work.
        let (base, hits) = provider(vec![sse_answer("all done", "stop")]).await;
        let mut agent = agent_against(base).with_max_steps(2);

        let outcome = agent.run("go").await.expect("the loop finishes");

        assert!(outcome.error.is_none(), "{:?}", outcome.error);
        assert_eq!(outcome.text, "all done");
        assert!(
            outcome.steps < 2,
            "a finished turn leaves the rest of its batch unused"
        );
        assert_eq!(continuation_asks(&outcome.new_messages), 0, "nothing to renew");
        assert_eq!(continuation_asks(agent.messages()), 0);
        assert_eq!(hits.load(std::sync::atomic::Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn a_stop_during_the_second_batch_ends_the_turn_without_another_ask() {
        // A renewal must not outlive the user's decision to stop. The stop lands
        // while the second batch is running, so the ask that authorized that
        // batch is already committed — and nothing may be appended after it.
        struct StopAtStep {
            at: usize,
            cancel: tokio::sync::watch::Sender<bool>,
        }

        impl AgentObserver for StopAtStep {
            fn on_step(&self, step: usize) {
                if step == self.at {
                    let _ = self.cancel.send(true);
                }
            }
        }

        let (base, hits) = provider(vec![sse_tool_call("call_1", "tick", "{}")]).await;
        let (cancel, cancelled) = tokio::sync::watch::channel(false);
        // Step 3 is the first step of the second batch, so batch one is spent
        // and its ask is in the history when the stop is requested.
        let stopper: Arc<dyn AgentObserver> = Arc::new(StopAtStep { at: 3, cancel });
        let mut agent = agent_with_tools(base, tick_tools())
            .with_max_steps(2)
            .with_observer(stopper)
            .with_cancel(cancelled);

        let outcome = agent.run("go").await.expect("the loop still answers");

        assert_eq!(outcome.error.as_deref(), Some(STOPPED), "{:?}", outcome.error);
        assert_eq!(outcome.steps, 3, "the stop lands before that step's request");
        assert_eq!(
            hits.load(std::sync::atomic::Ordering::Relaxed),
            2,
            "no request is made for the step the user stopped"
        );
        assert_eq!(
            continuation_asks(&outcome.new_messages),
            1,
            "the ask that started batch two, and none after the stop"
        );
        assert_eq!(continuation_asks(agent.messages()), 1);
    }

    // ------------------------------------------- a mission outlives the answer

    /// The id every mission test in this file uses.
    const MISSION_TEST_ID: &str = "m-20260101-000000";

    /// The part of the mission message a test matches on.
    ///
    /// A fragment, like [`CONTINUATION_FRAGMENT`], so rewording the sentence the
    /// model is asked does not break the tests that only care that it was asked.
    const MISSION_FRAGMENT: &str = "仍是 running";

    /// A scratch workspace holding one mission record and the pointer to it.
    ///
    /// Pinned through [`crate::tools::ToolIdentity`], never through a
    /// process-global: the loop reads the mission from the identity the turn was
    /// built with, so a test that rewrote a global would pass or fail on
    /// whatever else happened to be running.
    fn workspace_with_mission(tag: &str, status: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "ngu-agent-mission-{tag}-{}",
            uuid::Uuid::new_v4().simple()
        ));
        let mdir = dir.join("missions").join(MISSION_TEST_ID);
        std::fs::create_dir_all(&mdir).expect("the mission directory");
        std::fs::write(
            mdir.join("mission.json"),
            serde_json::json!({
                "id": MISSION_TEST_ID,
                "objective": "ship the crawler",
                "status": status,
                "step": "write the failing test",
                "done_when": "the tests pass",
            })
            .to_string(),
        )
        .expect("the mission record");
        std::fs::write(
            dir.join("missions").join("active.json"),
            serde_json::json!({ "mission": MISSION_TEST_ID }).to_string(),
        )
        .expect("the active pointer");
        dir
    }

    /// How many of `messages` carry the mission message.
    fn mission_nudges<'a>(messages: impl IntoIterator<Item = &'a Message>) -> usize {
        messages
            .into_iter()
            .filter(|message| message.text().contains(MISSION_FRAGMENT))
            .count()
    }

    #[tokio::test]
    async fn a_running_mission_keeps_the_turn_alive_after_a_final_answer() {
        // The model answered; the mission on disk says the work is not over. The
        // shell may not end that turn, and it may not end the mission either —
        // only the model can — so it names the mission that is still running and
        // goes round again. The ceiling is the only thing that stops it, and
        // reaching the ceiling is the bug this test is looking for.
        #[derive(Default)]
        struct Counting {
            seen: std::sync::atomic::AtomicUsize,
        }

        impl AgentObserver for Counting {
            fn on_message(&self, message: &Message) {
                if message.text().contains(MISSION_FRAGMENT) {
                    self.seen.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
            }
        }

        let dir = workspace_with_mission("running", "running");
        // The provider repeats its last response and the cap is one step per
        // batch, so every request is one batch and the batch count and the
        // request count are the same number.
        let (base, hits) = provider(vec![sse_answer("all done", "stop")]).await;
        let seen = Arc::new(Counting::default());
        let observing: Arc<dyn AgentObserver> = seen.clone();
        let mut agent = agent_with_tools(base, tick_tools())
            .with_max_steps(1)
            .with_observer(observing)
            .with_identity(crate::tools::ToolIdentity::for_session(
                "test-session",
                Some(dir.clone()),
            ));

        let outcome = agent.run("go").await.expect("the loop finishes");

        // The first batch, plus every batch the mission bought, plus the one the
        // ceiling refused. `steps` is the true total across all of them.
        assert_eq!(
            outcome.steps,
            1 + MISSION_IDLE_NUDGES,
            "a running mission lifts the continuation cap; the idle budget ends the turn"
        );
        assert!(
            outcome.steps < 1 + MISSION_BATCH_CEILING,
            "the ceiling is the backstop for an engaged model, not for this one"
        );
        assert_eq!(
            hits.load(std::sync::atomic::Ordering::Relaxed),
            outcome.steps,
            "every one of those steps was a real request"
        );
        assert!(outcome.error.is_none(), "{:?}", outcome.error);

        // The message is committed, reported with the turn, and shown to the
        // host that persists the transcript — a push-back the model never reads
        // is not a push-back.
        assert_eq!(
            mission_nudges(&outcome.new_messages),
            MISSION_IDLE_NUDGES,
            "one push-back per granted batch"
        );
        assert_eq!(mission_nudges(agent.messages()), MISSION_IDLE_NUDGES);
        assert_eq!(
            seen.seen.load(std::sync::atomic::Ordering::Relaxed),
            MISSION_IDLE_NUDGES,
            "the observer saw every one"
        );

        // It says what the model needs to act on, not just that time passed.
        let first = outcome
            .new_messages
            .iter()
            .find(|message| message.text().contains(MISSION_FRAGMENT))
            .map(Message::text)
            .expect("a push-back");
        assert!(first.contains(MISSION_TEST_ID), "{first}");
        assert!(first.contains("ship the crawler"), "the objective travels: {first}");
        assert!(first.contains("write the failing test"), "the next step travels: {first}");
        assert!(first.contains("the tests pass"), "the done-when travels: {first}");
        assert!(
            first.contains("mission_advance"),
            "the only other way out is named: {first}"
        );

        // The turn ends naming the mission it could not finish, and nothing in
        // this file touched the mission's own record on the way there.
        assert!(outcome.text.contains(MISSION_TEST_ID), "{}", outcome.text);
        assert!(outcome.text.contains("仍是 running"), "{}", outcome.text);
        assert!(outcome.text.contains("mission_"), "{}", outcome.text);
        let record = std::fs::read_to_string(
            dir.join("missions").join(MISSION_TEST_ID).join("mission.json"),
        )
        .expect("the record is still there");
        assert!(
            record.contains("\"running\""),
            "the shell never ends a mission: {record}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn a_mission_that_is_not_running_lets_the_turn_end_immediately() {
        // A finished mission is the model having decided the work is over — the
        // one decision the shell must not second-guess — so the answer ends the
        // turn exactly as it did before missions existed.
        let dir = workspace_with_mission("done", "done");
        let (base, hits) = provider(vec![sse_answer("all done", "stop")]).await;
        let mut agent = agent_with_tools(base, tick_tools())
            .with_max_steps(1)
            .with_identity(crate::tools::ToolIdentity::for_session(
                "test-session",
                Some(dir.clone()),
            ));

        let outcome = agent.run("go").await.expect("the loop finishes");

        assert!(outcome.error.is_none(), "{:?}", outcome.error);
        assert_eq!(outcome.text, "all done", "the model's answer is the turn's answer");
        assert_eq!(outcome.steps, 1);
        assert_eq!(
            hits.load(std::sync::atomic::Ordering::Relaxed),
            1,
            "one request, not another batch"
        );
        assert_eq!(mission_nudges(&outcome.new_messages), 0);
        assert_eq!(mission_nudges(agent.messages()), 0);
        let _ = std::fs::remove_dir_all(&dir);

        // A workspace with no mission at all is the ordinary case, and it must
        // stay exactly as cheap: one answer, one turn.
        let empty = std::env::temp_dir().join(format!(
            "ngu-agent-mission-none-{}",
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&empty).expect("a scratch workspace");
        let (base, hits) = provider(vec![sse_answer("all done", "stop")]).await;
        let mut agent = agent_with_tools(base, tick_tools())
            .with_max_steps(1)
            .with_identity(crate::tools::ToolIdentity::for_session(
                "test-session",
                Some(empty.clone()),
            ));

        let outcome = agent.run("go").await.expect("the loop finishes");

        assert_eq!(outcome.text, "all done");
        assert_eq!(hits.load(std::sync::atomic::Ordering::Relaxed), 1);
        assert_eq!(mission_nudges(&outcome.new_messages), 0);
        let _ = std::fs::remove_dir_all(&empty);
    }

    #[tokio::test]
    async fn a_stop_request_still_ends_a_turn_with_a_running_mission() {
        // A running mission must not outlive the user's decision to stop. The
        // stop lands at the top of the second step, after the message that kept
        // the turn open is already committed — and nothing may be appended after
        // it.
        struct StopAtStep {
            at: usize,
            cancel: tokio::sync::watch::Sender<bool>,
        }

        impl AgentObserver for StopAtStep {
            fn on_step(&self, step: usize) {
                if step == self.at {
                    let _ = self.cancel.send(true);
                }
            }
        }

        let dir = workspace_with_mission("stop", "running");
        let (base, hits) = provider(vec![sse_answer("all done", "stop")]).await;
        let (cancel, cancelled) = tokio::sync::watch::channel(false);
        let stopper: Arc<dyn AgentObserver> = Arc::new(StopAtStep { at: 2, cancel });
        let mut agent = agent_with_tools(base, tick_tools())
            .with_max_steps(1)
            .with_observer(stopper)
            .with_cancel(cancelled)
            .with_identity(crate::tools::ToolIdentity::for_session(
                "test-session",
                Some(dir.clone()),
            ));

        let outcome = agent.run("go").await.expect("the loop still answers");

        assert_eq!(
            outcome.error.as_deref(),
            Some(STOPPED),
            "a stop outranks the mission: {:?}",
            outcome.error
        );
        assert_eq!(outcome.steps, 2, "the stop lands before that step's request");
        assert_eq!(
            hits.load(std::sync::atomic::Ordering::Relaxed),
            1,
            "no request is made for the step the user stopped"
        );
        assert_eq!(
            mission_nudges(&outcome.new_messages),
            1,
            "the message that kept the turn open, and none after the stop"
        );
        assert!(
            outcome
                .new_messages
                .last()
                .expect("the turn committed messages")
                .text()
                .contains(MISSION_FRAGMENT),
            "nothing is injected after the stop"
        );
        assert_eq!(mission_nudges(agent.messages()), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
