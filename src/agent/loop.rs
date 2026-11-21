//! The loop a turn runs on: the trait, the default implementation, and the
//! process-wide slot a pack installs its own into.
//!
//! [`Agent`] is the loop every build has; [`AgentLoop`] is what makes it
//! replaceable, so a pack can run a different one without being second-class.

use crate::compaction::{Compaction, Summarizer};
use crate::llm::{LlmClient, LlmEvent, PartialCallView, TokenUsage};
use crate::message::{Message, ToolCall};
use crate::mission::MissionBrief;
use crate::tools::ToolRegistry;
use crate::window::{ContextPolicy, ContextWindow};
use anyhow::Result;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Instant;
use tokio::task::JoinSet;

use super::config::{StaticConfig, StaticStateConfig, TurnConfig, TurnSettings};
use super::observer::{AgentObserver, SilentObserver};
use super::state::{
    record_reported_window, stop_flag, summarize_tool_error, AgentOutcome, LoopState, StepOutput,
    TurnTiming, CONTINUATION_BATCHES, MISSION_NUDGE, MISSION_RETRY_CAP, STOPPED,
};
use crate::compaction;
use crate::llm;

/// The loop a turn runs on.
///
/// One method, and the whole turn's state goes in and comes out. The default is
/// [`Agent`]'s own loop, which is what every existing caller gets; a pack that
/// wants a different shape — a tree search, a vote across routes, a loop that
/// stops only when a person says so — supplies its own and is not second-class:
/// nothing in the kernel asks whether the loop is "the real one".
///
/// Being replaceable does not mean being replaced. A build that never installs
/// another loop runs exactly the code it ran before.
pub trait AgentLoop: Send + Sync {
    /// Run one turn to its end.
    ///
    /// `state` is owned, not borrowed: a loop that spawns work with the
    /// conversation in hand cannot do that through a borrow that must outlive
    /// the spawn. Whatever the loop changes is left in `state` when it returns,
    /// and the caller reads it back out.
    fn run<'a>(
        &'a self,
        state: LoopState,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = (LoopState, Result<AgentOutcome>)> + Send + 'a>,
    >;
}

/// How many times one answer may be continued after hitting the provider's
/// output ceiling, before the loop stops extending it.
const MAX_CONTINUATIONS: usize = 3;

/// The loop every build has, wrapped so it can be chosen like any other.
///
/// It does not reimplement anything: it builds an [`Agent`] from the state,
/// runs the same `run` the kernel always had, and hands the conversation back.
/// That is deliberate — a "built-in loop" that had drifted from the one every
/// caller gets would mean two behaviours to keep correct, and the point of the
/// trait is that replacing the loop is ordinary, not that the default is
/// special.
pub struct BuiltinLoop;

impl AgentLoop for BuiltinLoop {
    fn run<'a>(
        &'a self,
        mut state: LoopState,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = (LoopState, Result<AgentOutcome>)> + Send + 'a>,
    > {
        Box::pin(async move {
            let mut agent = Agent::new(
                state.client.clone(),
                Arc::new(StaticStateConfig {
                    settings: state.settings.clone(),
                }),
                std::mem::take(&mut state.messages),
            )
            .as_builtin();
            agent.max_steps = state.max_steps;
            agent.observer = Arc::clone(&state.observer);
            agent.summarizer = Arc::clone(&state.summarizer);
            agent.identity = state.identity.clone();
            agent.cancel = state.cancel.clone();
            agent.inbox = state.inbox.clone();
            agent.subtask = state.subtask;
            agent.stop_reason = state.stop_reason.clone();
            agent.injection_override = state.injection_override.clone();
            let prompt = state.start_prompt.clone();
            let result = agent.run(&prompt).await;
            state.messages = std::mem::take(&mut agent.messages);
            (state, result)
        })
    }
}

/// Drives one conversation.
pub struct Agent {
    client: LlmClient,
    pub(crate) config: Arc<dyn TurnConfig>,
    pub(crate) messages: Vec<Message>,
    max_steps: usize,
    observer: Arc<dyn AgentObserver>,
    /// Decides how a summary is written.
    ///
    /// The window and threshold come from the turn's settings instead, so a
    /// change made in a settings panel lands on the next turn.
    summarizer: Arc<dyn Summarizer>,
    /// Injection engine override, for tests and embedding.
    injection_override: Option<Arc<dyn crate::context::InjectionSource>>,
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
    /// Whether this turn is somebody's subtask rather than a conversation.
    ///
    /// A subagent inherits the session's tool table, its route, and its
    /// standing instructions — that is the point of [`crate::tools::subagent`]
    /// — but not the *privileges of a conversation*. Two things must not
    /// travel into it:
    ///
    /// * **The batch renewal.** A top-level turn that spends its steps asks the
    ///   model whether the round count is normal and earns another batch, which
    ///   is right for a long task someone is watching. A subtask was given a
    ///   budget by its caller — the tool description states it — and quietly
    ///   spending three times that is how a `delegate` call that promised
    ///   twelve steps took half an hour and answered nothing.
    /// * **The mission hold.** A mission keeps *its own* turn open until the
    ///   record stops running. That record lives in the parent's workspace, and
    ///   a subtask that is held open by it — or worse, advances it — is doing
    ///   work nobody gave it, invisibly, for as long as the parent's mission
    ///   lasts. [`Agent::as_subtask`] turns both off.
    subtask: bool,
    /// What a stop reports as this turn's error.
    ///
    /// [`STOPPED`] is the truth for a turn a person cancelled and a lie for one
    /// the shell cut off at its own deadline: the parent model reads this string
    /// as the reason its subtask did not finish, and "stopped by the user" sends
    /// it looking for a user who did no such thing.
    stop_reason: String,
    /// The loop this agent runs on, when it is not the built-in one.
    ///
    /// `None` means [`Agent`]'s own loop, which is what every caller had before
    /// this field existed. A replaced loop is handed the state wholesale; see
    /// [`AgentLoop`] for why that is the shape rather than a curated view.
    replaced_loop: Option<Arc<dyn AgentLoop>>,
    /// Whether this agent *is* the built-in loop.
    ///
    /// A guard, not a setting: the built-in loop builds an [`Agent`] to do its
    /// work, and that inner agent must not pick up an installed loop or the two
    /// would call each other for ever.
    builtin: bool,
}

/// The loop installed by whoever assembled this process.
///
/// A pack that ships a loop cannot hand it to every caller — the callers are the
/// CLI, the desktop shell, and a subagent, and none of them knows the pack
/// exists. So the loop is published process-wide the same way the tool budgets
/// are (`tools::set_budgets`), and each caller asks for it with
/// [`Agent::with_installed_loop`].
///
/// Process-wide rather than per-session on purpose: a loop changes how a turn
/// thinks, and a machine that has installed one expects every turn to think that
/// way. A caller that wants the built-in loop simply does not ask.
static INSTALLED_LOOP: std::sync::RwLock<Option<Arc<dyn AgentLoop>>> = std::sync::RwLock::new(None);

/// Install the loop every subsequent turn runs on.
///
/// Replacing an installed loop is allowed: a second pack's loop wins, and the
/// loser's turns run the winner's code until the process restarts. That is the
/// same "last writer wins" rule the service table uses, and for the same reason
/// — refusing would make the kernel decide which of two packs is legitimate.
pub fn install_loop(loop_impl: Arc<dyn AgentLoop>) {
    let mut slot = INSTALLED_LOOP.write().unwrap_or_else(|p| p.into_inner());
    *slot = Some(loop_impl);
}

/// Remove the installed loop, returning it. Turns go back to the built-in one.
pub fn uninstall_loop() -> Option<Arc<dyn AgentLoop>> {
    let mut slot = INSTALLED_LOOP.write().unwrap_or_else(|p| p.into_inner());
    slot.take()
}

/// The installed loop, if any.
pub fn installed_loop() -> Option<Arc<dyn AgentLoop>> {
    INSTALLED_LOOP
        .read()
        .unwrap_or_else(|p| p.into_inner())
        .clone()
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
            subtask: false,
            stop_reason: STOPPED.to_string(),
            replaced_loop: None,
            builtin: false,
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
    /// A subtask ([`Agent::as_subtask`]) gets exactly this many steps and no
    /// renewal at all.
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
    /// [`Agent::with_stop_reason`] (by default "stopped by the user") as the
    /// turn's error.
    pub fn with_cancel(mut self, cancel: tokio::sync::watch::Receiver<bool>) -> Self {
        self.cancel = Some(cancel);
        self
    }

    /// Say why this turn stopped, in the words its reader needs.
    ///
    /// A host that cuts a turn off at its own deadline is not the user walking
    /// away, and the error string is read by whoever asked for the turn — for a
    /// subtask, that is the parent model deciding what to do next. "stopped by
    /// the user" would send it looking for a user who did nothing.
    pub fn with_stop_reason(mut self, reason: impl Into<String>) -> Self {
        self.stop_reason = reason.into();
        self
    }

    /// Run this turn as somebody's subtask: an exact step budget, and no hold
    /// from the parent's mission.
    ///
    /// See the field documentation on [`Agent::subtask`] for why both are the
    /// same switch rather than two knobs a caller can get wrong one at a time.
    pub fn as_subtask(mut self) -> Self {
        self.subtask = true;
        self
    }

    /// Run this agent's turns on a different loop.
    ///
    /// Nothing here decides whether the loop is trustworthy: a host that
    /// installs one has already decided that, the same way loading a pack is the
    /// decision to trust the pack. The default is the built-in loop, and a build
    /// that never calls this runs exactly the code it always ran.
    pub fn with_loop(mut self, loop_impl: Arc<dyn AgentLoop>) -> Self {
        self.replaced_loop = Some(loop_impl);
        self
    }

    /// Use the loop installed process-wide, when one was installed.
    ///
    /// This is the line that makes a pack-supplied loop reach every caller
    /// without any of them naming the pack: they ask "is there a loop?" and the
    /// answer comes from [`installed_loop`].
    ///
    /// It does not apply to the built-in loop's own agent — that agent is what
    /// runs when nobody installed anything, and letting it pick up an installed
    /// loop would make it call itself for ever.
    pub fn with_installed_loop(mut self) -> Self {
        if !self.builtin && self.replaced_loop.is_none() {
            if let Some(loop_impl) = installed_loop() {
                self.replaced_loop = Some(loop_impl);
            }
        }
        self
    }

    /// Mark this agent as the built-in loop running.
    ///
    /// Set by [`BuiltinLoop`] on the agent it builds, so the recursion guard in
    /// [`Agent::with_installed_loop`] has something to check.
    pub fn as_builtin(mut self) -> Self {
        self.builtin = true;
        self
    }

    /// Hand this turn's state to a replaced loop.
    ///
    /// The conversation moves rather than being copied: a loop is free to append
    /// thousands of messages and copying them out and back would double the peak
    /// memory of every turn for a feature most builds never use.
    fn take_state(&mut self, prompt: &str) -> LoopState {
        let settings = self.config.settings();
        LoopState {
            messages: std::mem::take(&mut self.messages),
            client: self.client.clone(),
            tools: Arc::clone(&settings.tools),
            settings,
            observer: Arc::clone(&self.observer),
            summarizer: Arc::clone(&self.summarizer),
            identity: self.identity.clone(),
            cancel: self.cancel.clone(),
            inbox: self.inbox.clone(),
            max_steps: self.max_steps,
            subtask: self.subtask,
            stop_reason: self.stop_reason.clone(),
            injection_override: self.injection_override.clone(),
            start_prompt: prompt.to_string(),
        }
    }

    /// Take the conversation back from a replaced loop.
    ///
    /// Only the conversation comes back. A loop that changed the tool table or
    /// the settings changed them for its own turn; the next turn reads those
    /// from the configuration source like every turn did.
    fn give_back(&mut self, state: LoopState) {
        self.messages = state.messages;
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
    /// a workspace is a nested worker (a subagent) —
    /// it already *is* one step of somebody else's mission, and holding it open
    /// would mean a mission that cannot run its own steps.
    ///
    /// Re-read at every decision point rather than remembered for the turn: the
    /// batch that just ran may have blocked or finished the mission, and that
    /// status — not the step count — is what decides whether the turn may end.
    fn running_mission(&self) -> Option<MissionBrief> {
        // A subtask is never held open by one. It inherits the parent's tools
        // and may even be handed the parent's identity, so a mission found here
        // would be work the parent accepted — and a subtask that advances or
        // outlives it is exactly the runaway `delegate` must not produce. The
        // parent's own turn is the one that owes the mission its time.
        if self.subtask {
            return None;
        }
        let workspace = self.identity.as_ref()?.workspace.as_deref()?;
        crate::mission::running_mission(workspace)
    }

    /// Replace the summarizer: how a compaction's summary is produced.
    pub fn with_summarizer(mut self, summarizer: Arc<dyn Summarizer>) -> Self {
        self.summarizer = summarizer;
        self
    }

    /// Set the injection engine used for the next turn.
    pub fn set_injection(&mut self, injection: Arc<dyn crate::context::InjectionSource>) {
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
    /// says `running`. That turn has no batch count, no step ceiling and no wall
    /// clock at all — it ends when the mission is `done`, when it is `blocked`,
    /// or when the user stops it. The shell never decides that such a mission is
    /// finished; it only refuses to walk away from it.
    pub async fn run(&mut self, prompt: &str) -> Result<AgentOutcome> {
        // A replaced loop takes the whole turn, including committing the user's
        // message: it owns the conversation while it runs, and splitting that
        // between the caller and the loop would make the two disagree about
        // where the turn's history begins.
        if let Some(loop_impl) = self.replaced_loop.clone() {
            let state = self.take_state(prompt);
            let (state, result) = loop_impl.run(state).await;
            self.give_back(state);
            return result;
        }

        let user_message = Message::user(prompt);
        self.messages.push(user_message.clone());
        // The prompt is committed before anything is attempted: asking the
        // model is already a thing that happened.
        self.observer.on_message(&user_message);

        // Frozen for the whole turn: a reload landing now is next turn's news.
        let settings = self.config.settings();

        // The turn is announced before anything else happens, so a plugin that
        // wants to seed the conversation does it once and at the top. Emitted
        // rather than returned: a listener here may inject, and nothing about
        // startup depends on its answer.
        if let Some(hooks) = &settings.plugin_hooks {
            hooks.intercept(
                prompt,
                &settings.model,
                crate::plugin::Seam {
                    name: "session-start".into(),
                    body: serde_json::json!({ "input": prompt }),
                },
            );
        }

        // The tool table is frozen with them, for a reason of its own: the
        // desktop releases a conversation's plugins when that conversation is
        // left, and a background turn can still be running. Reading the live
        // table per step would let a tool the model has already seen vanish
        // between two steps, and the turn would end on "no tool named …"
        // halfway through its work. Copying the table is cheap — every handler
        // is an `Arc` — and the copy is what keeps the code it names alive
        // until this turn returns.
        // The plugin seam narrows the table for this turn. Skipped entirely
        // when no plugin is loaded, so the common case pays nothing.
        let mut snapshot_tools = (*settings.tools).clone();
        if let Some(hooks) = &settings.plugin_hooks {
            let all = snapshot_tools.names();
            // The old per-seam method, kept because it answers the common case
            // in one call; a plugin that returns `None` here can still narrow
            // the table through `intercept(Slot::Tools)`.
            let filtered = hooks
                .filter_tools(prompt, &settings.model, all.clone())
                .or_else(|| {
                    let seam = hooks.intercept(
                        prompt,
                        &settings.model,
                        crate::plugin::Seam::names("tools", all.clone()),
                    );
                    seam.as_names()
                });
            if let Some(keep) = filtered {
                let keep: std::collections::HashSet<String> = keep.into_iter().collect();
                for name in all {
                    if !keep.contains(&name) {
                        snapshot_tools.unregister(&name);
                    }
                }
            }
        }
        let tools = Arc::new(snapshot_tools);

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
        // batch — see [`CONTINUATION_BATCHES`]. A running mission is the one
        // exception: it is not capped at all, see [`MISSION_NUDGE`]. A subtask
        // is the other: it was given a budget by its caller and does not renew
        // it — see [`Agent::as_subtask`].
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
        let renewal_batches = if self.subtask {
            0
        } else {
            CONTINUATION_BATCHES
        };
        let step_budget = batch_size.saturating_mul(1 + renewal_batches);
        let mut step: usize = 0;
        let mut renewals: usize = 0;

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
                // The step is announced before its request is assembled, so a
                // plugin can see which step this is and what the history holds.
                // The answer is read: a listener may add text to the request,
                // which is the one thing a per-step point is for.
                let step_note = match &settings.plugin_hooks {
                    Some(hooks) => {
                        let seam = hooks.intercept(
                            prompt,
                            &settings.model,
                            crate::plugin::Seam {
                                name: "pre-step".into(),
                                body: serde_json::json!({
                                    "step": step,
                                    "messages": self.messages.len(),
                                }),
                            },
                        );
                        seam.body
                            .get("prompt")
                            .and_then(serde_json::Value::as_str)
                            .map(str::to_string)
                    }
                    None => None,
                };
                // A stop between steps ends the turn here: everything committed
                // so far is already in the history and is reported with it.
                if self.stop_requested() {
                    outcome.error = Some(self.stop_reason.clone());
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
                        match self
                            .compact(&client, window, settings.policy.as_ref())
                            .await
                        {
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

                let (request, injection) = self.build_request(&settings, step_note.as_deref());
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
                        stopped.error = Some(anyhow::anyhow!(self.stop_reason.clone()));
                        break stopped;
                    }
                    attempt += 1;
                    let started = Instant::now();
                    let produced = self.stream_once(&client, &request, &schemas).await;
                    outcome.timing.model_ms += started.elapsed().as_millis();

                    // A running mission outranks the attempt budget. The turn is
                    // holding work someone asked for, and a provider that is
                    // briefly unreachable — a dropped connection, a gateway
                    // restart — must not be able to end it: the mission is not
                    // finished, so the turn is not finishable. It keeps asking,
                    // with the network backoff (capped, so a long outage is one
                    // ask per half minute rather than a tight loop), until the
                    // route answers, the model blocks the mission, or the user
                    // stops it. A mission that is no longer running falls back
                    // to the ordinary budget.
                    let mission_open = self.running_mission().is_some();

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
                            if retry && (attempt < max_attempts || mission_open) {
                                let message = format!("{error:#}");
                                outcome.retries += 1;
                                self.observer.on_step_retry(attempt as usize, &message);
                                let pause = if mission_open {
                                    network.backoff(attempt).min(MISSION_RETRY_CAP)
                                } else {
                                    network.backoff(attempt)
                                };
                                tokio::time::sleep(pause).await;
                                continue;
                            }
                        }
                        None if produced.is_empty() && (attempt < max_attempts || mission_open) => {
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
                        self.observer
                            .on_message(self.messages.last().expect("just pushed"));
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
                // The reply passes through the plugins before anything else
                // sees it. Deliberately not during streaming: buffering deltas
                // to catch a word split across two of them would add latency to
                // every answer, and the user would rather see the correction at
                // the end than wait for certainty on each character.
                let text = {
                    let original = out.text;
                    match &settings.plugin_hooks {
                        Some(hooks) => {
                            let seam = hooks.intercept(
                                prompt,
                                &settings.model,
                                crate::plugin::Seam::text("reply", original.clone()),
                            );
                            seam.as_text().map(str::to_string).unwrap_or(original)
                        }
                        None => original,
                    }
                };

                // Stopped between the answer and its tool calls: the calls never
                // ran, so they are dropped rather than left unanswered — a tool
                // call with no result is a request no provider accepts next turn.
                let stopped_before_tools = self.stop_requested() && !tool_calls.is_empty();
                if stopped_before_tools {
                    tool_calls.clear();
                }

                let assistant = Message::assistant_tools(
                    tool_calls.clone(),
                    if text.is_empty() {
                        None
                    } else {
                        Some(text.clone())
                    },
                );
                self.messages.push(assistant.clone());
                outcome.new_messages.push(assistant);
                self.observer
                    .on_message(self.messages.last().expect("just pushed"));
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
                    self.observer
                        .on_message(self.messages.last().expect("just pushed"));
                    self.observer
                        .on_continuation(&format!("step {step} hit the output limit"));
                    continue;
                }

                if tool_calls.is_empty() {
                    if stopped_before_tools {
                        outcome.error = Some(self.stop_reason.clone());
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
                let tools_started = Instant::now();
                let mut set: JoinSet<(usize, ToolCall, Result<crate::tools::ToolOutput>)> =
                    JoinSet::new();
                // A stop that landed while the answer was being committed is seen
                // *before* the batch starts. The user has already walked away, and
                // starting a whole set of tools anyway spends side effects nobody
                // will read. The calls are answered below regardless, so the
                // history stays complete either way.
                // Refusals decided before spawning, folded in below so the batch
                // keeps one shape.
                let mut results_early: Vec<(usize, ToolCall, Result<crate::tools::ToolOutput>)> =
                    Vec::new();
                let mut stopped = self.stop_requested();
                if !stopped {
                    for (position, call) in tool_calls.iter().enumerate() {
                        // Asked before the call runs, once per call. A refusal is
                        // handed back as the tool result, which is the only way to
                        // tell the model why — the call already exists in the
                        // history and dropping it would leave a request no
                        // provider accepts.
                        let refusal = settings.plugin_hooks.as_ref().and_then(|hooks| {
                            let seam = hooks.intercept(
                                prompt,
                                &settings.model,
                                crate::plugin::Seam {
                                    name: "pre-tool".into(),
                                    body: serde_json::json!({
                                        "name": call.name,
                                        "arguments": call.arguments,
                                    }),
                                },
                            );
                            seam.body
                                .get("refuse")
                                .and_then(serde_json::Value::as_str)
                                .filter(|reason| !reason.is_empty())
                                .map(str::to_string)
                        });
                        if let Some(reason) = refusal {
                            self.observer.on_tool_start(&call.name, &call.arguments);
                            results_early.push((
                                position,
                                call.clone(),
                                Err(anyhow::anyhow!(reason)),
                            ));
                            continue;
                        }

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
                for (position, call, result) in results_early {
                    if let Some(slot) = results.get_mut(position) {
                        *slot = Some((call, result));
                    }
                }
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
                        Some(Ok((position, call, result))) => {
                            results[position] = Some((call, result))
                        }
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

                // Decided once for the batch: whether this route can carry an
                // image inside the tool result itself, from the measured table.
                let tool_images_ok = crate::llm::model_caps(&settings.model)
                    .image_in_tool_result
                    .unwrap_or(true);

                for (position, call) in tool_calls.iter().enumerate() {
                    // Told after each call returns, before its result is shaped
                    // into the history. Observe-only: the answer is not read, so a
                    // listener can count, log or learn without being able to change
                    // what the model sees.
                    if let Some(hooks) = &settings.plugin_hooks {
                        hooks.intercept(
                            prompt,
                            &settings.model,
                            crate::plugin::Seam {
                                name: "post-tool".into(),
                                body: serde_json::json!({
                                    "name": call.name,
                                    "ok": results[position].as_ref().map(|(_, r)| r.is_ok()),
                                }),
                            },
                        );
                    }
                    let result = match results[position].take() {
                        Some((_, result)) => result,
                        // Killed mid-call: the call still gets an answer, because
                        // an unanswered tool call is a history the next request
                        // cannot be built from.
                        None if stopped => Err(anyhow::anyhow!(
                            "{} before this tool call finished",
                            self.stop_reason
                        )),
                        // A task that died without a stop request: it panicked or
                        // was aborted, so nothing was written into its slot. The
                        // call is answered explicitly anyway — the same rule as
                        // above, and this arm *is* reachable: a panicking tool is
                        // exactly what lands here.
                        None => Err(anyhow::anyhow!("tool panicked: {}", call.name)),
                    };
                    outcome.tool_calls += 1;
                    let mut images_for_this_call: Vec<crate::message::ImageAttachment> = Vec::new();
                    let (ok, body) = match result {
                        Ok(output) => {
                            for image in output.images {
                                if tool_images_ok {
                                    images_for_this_call.push(image);
                                } else {
                                    attached.push((call.name.clone(), image));
                                }
                            }
                            (true, output.text)
                        }
                        Err(error) => (false, format!("error: {}", summarize_tool_error(&error))),
                    };
                    self.observer.on_tool_end(&call.name, ok, &body);
                    let mut message =
                        Message::tool_result(call.id.clone(), call.name.clone(), body);
                    // Images stay on the tool result when the route can carry
                    // them; otherwise they are collected below and re-attached
                    // as a user message, the shape every provider accepts.
                    if tool_images_ok {
                        message.images = images_for_this_call;
                    }
                    self.messages.push(message.clone());
                    outcome.new_messages.push(message);
                    self.observer
                        .on_message(self.messages.last().expect("just pushed"));
                }

                // When the route accepts images inside a tool result, they are
                // already on the message that carried them and this extra user
                // turn is pure noise: a second "user said something" the model
                // never had, costing a turn of history per picture.
                if !attached.is_empty() && !tool_images_ok {
                    let mut text = String::from("Image(s) produced by the tool call above:");
                    for (tool, image) in &attached {
                        text.push_str(&format!(
                            "\n- from {tool}: {}",
                            image.label.as_deref().unwrap_or("image")
                        ));
                    }
                    let message = Message::user_with_images(
                        text,
                        attached.into_iter().map(|(_, i)| i).collect(),
                    );
                    self.messages.push(message.clone());
                    outcome.new_messages.push(message);
                    self.observer
                        .on_message(self.messages.last().expect("just pushed"));
                }

                // Every dispatched call now has an answer and the turn was asked
                // to stop: report it instead of taking another step.
                if stopped {
                    outcome.error = Some(self.stop_reason.clone());
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

            // The batch boundary is where a turn would end, so it is the point
            // worth asking about. A listener that answers `continue` keeps the
            // loop going — that is the one way a plugin can extend a turn, and it
            // is why this is asked before any `return` below.
            let asked_to_continue = match &settings.plugin_hooks {
                Some(hooks) => {
                    let seam = hooks.intercept(
                        prompt,
                        &settings.model,
                        crate::plugin::Seam {
                            name: "turn-stopping".into(),
                            body: serde_json::json!({
                                "step": step,
                                "answered": answered,
                                "budget_spent": step >= step_budget,
                            }),
                        },
                    );
                    seam.body
                        .get("continue")
                        .and_then(serde_json::Value::as_bool)
                        .unwrap_or(false)
                }
                None => false,
            };
            if asked_to_continue && step < step_budget.saturating_mul(4) {
                // Bounded: a listener that always says continue must not be able
                // to run the loop forever. Four times the normal allowance is
                // enough room for the case this exists for — a plugin that needs
                // one more round trip — and finite for a plugin that is wrong.
                renewals += 1;
                continue 'batches;
            }

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
                // nothing here decides that work is done.
                Some(mission) => {
                    // A stop outranks the mission. The user walking away ends
                    // the turn, and nothing is injected after it: a message
                    // asking a model to keep working on a turn nobody is
                    // watching is exactly the run-away this loop must not do.
                    if self.stop_requested() {
                        outcome.error = Some(self.stop_reason.clone());
                        return Ok(outcome);
                    }
                    // No ceiling, no idle budget, no wall clock. A mission that
                    // is still running is a job the model accepted, and the only
                    // question this loop is allowed to ask is "is it running?" —
                    // a step cap here is exactly what turned a task into
                    // something that stops and waits for the user.
                    //
                    // The nudge is 「继续」 and nothing else: the objective, the
                    // next step and the done-when are already in the system
                    // prompt of every request, so restating them bought nothing
                    // and cost context in the one place a long task cannot
                    // afford it.
                    //
                    // On stderr too, for the same reason as the cap above: a
                    // turn held open by a mission is otherwise indistinguishable
                    // from a stuck one in a terminal or a desktop log.
                    eprintln!(
                        "[mission] {step} steps in, mission {} is still running; the turn continues",
                        mission.id
                    );
                    // Committed like steering, because that is what it is: a
                    // message the model reads on the next request, that the
                    // transcript shows, and that survives the turn.
                    let message = Message::user(MISSION_NUDGE);
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
        //
        // A subtask reports this as a **failure**, not as a result. "stopped
        // after 12 steps without a final answer" is a useful thing to print at
        // the end of a conversation and a trap at the end of a subtask: the
        // parent model asked a question, and reading a stop marker as the answer
        // is how an actor-critic round ends up trusting a critic that never
        // critiqued anything.
        if self.subtask {
            outcome.error = Some(format!(
                "the subtask spent its {step}-step budget without a final answer \
                 ({} renewal(s) asked)",
                renewals
            ));
        } else {
            outcome.text = format!(
                "stopped after {} steps without a final answer ({} continuation(s) asked)",
                step, renewals
            );
        }
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
                            out.error = Some(anyhow::anyhow!(self.stop_reason.clone()));
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
                            out.error = Some(anyhow::anyhow!(self.stop_reason.clone()));
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
            let Some(event) = event else {
                break 'stream;
            };
            match event {
                Ok(LlmEvent::TextDelta(delta)) => {
                    self.observer.on_text(&delta);
                    out.text.push_str(&delta);
                }
                Ok(LlmEvent::ReasoningDelta(delta)) => {
                    self.observer.on_reasoning(&delta);
                    out.saw_reasoning = true;
                }
                Ok(LlmEvent::ToolCallDelta {
                    index,
                    id,
                    name,
                    arguments,
                }) => {
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
    pub(crate) fn build_request(
        &self,
        settings: &TurnSettings,
        step_note: Option<&str>,
    ) -> (Vec<Message>, crate::context::Injection) {
        let window = settings.policy.window(&settings.model);
        let engine = self
            .injection_override
            .as_ref()
            .unwrap_or(&settings.injection);
        let injection = engine.inject(&self.messages, window.tokens, settings.cache_policy);

        let mut request = Vec::with_capacity(self.messages.len() + 4);

        // The history is thinned for this request only. The session file keeps
        // every original byte, so raising the limit later brings the detail back
        // and an audit still sees what happened; nothing here mutates state.
        //
        // Thinning is not free, so it is not unconditional. Under half a window
        // the model gets the conversation exactly as it happened — that is the
        // state where a long tool output is *useful*, and replacing it with
        // "[old tool output thinned]" throws away the thing the model asked for
        // while there is still room to keep it. The ladder:
        //
        //   below `quiet_at`      — nothing is touched at all
        //   below `compact_percent` — old tool text and images are thinned
        //   at or above it         — the preset becomes aggressive
        //
        // Compaction (a summary replacing the front of the log) stays a separate
        // decision made by `compact`, because it is the only one that is
        // irreversible in the request.
        let used = crate::window::estimate_history(&self.messages);
        let share = if window.tokens == 0 {
            0
        } else {
            (used.saturating_mul(100) / window.tokens) as u32
        };
        // Where thinning starts. Three quarters of the way to compaction leaves
        // room for the tool results a turn is actually using while still being
        // early enough that the prompt does not jump straight to a summary.
        const QUIET_AT_PERCENT: u32 = 35;
        let policy = if share < QUIET_AT_PERCENT {
            crate::compaction::PrunePolicy::loose()
        } else if share < settings.policy.compact_at_percent() {
            settings.policy.prune()
        } else {
            crate::compaction::PrunePolicy::aggressive()
        };
        let thinned = crate::compaction::prune_tool_results(&self.messages, &policy);
        let history = crate::compaction::prune_images(&thinned, &policy);

        // The leading block is ONE message, not several.
        //
        // Measured against a real route: a vLLM-shaped OpenAI-compatible gateway
        // answers `400 System message must be at the beginning` when a `system`
        // message sits at any index after 0 — and a persona plus two standing
        // injection rules is exactly that. The check is about the *count*, not
        // the order, so sending [system, system, system, user] fails even though
        // every system is at the front. Tolerant gateways never cared; strict ones
        // refuse the turn outright, which looks like a broken key rather than a
        // shape the provider dislikes.
        //
        // Joined with a blank line, the bytes of each fragment stay verbatim, so a
        // pack's rule text is still exactly what its author wrote.
        let mut added: Vec<String> = Vec::new();
        let mut head: Vec<String> = Vec::new();
        // The system prompt goes through the plugins before it is sent.
        //
        // `Slot::Prompt` was declared with this exact comment — "the text about
        // to be added to the system prompt" — and never created anywhere, so a
        // plugin could rewrite a finished reply but not the standing text the
        // model reads first. What that cost: anything that has to say something
        // in the prompt had to change `TurnSettings` instead, and a field on
        // `TurnSettings` is a field every front end has to fill.
        //
        // Only this block is offered. A plugin rewriting the conversation would
        // be rewriting the record of what the user said, which the session file
        // keeps verbatim on purpose.
        let system_prompt = match &settings.plugin_hooks {
            Some(hooks) => {
                let input = self
                    .messages
                    .iter()
                    .rev()
                    .find(|m| m.role == crate::message::Role::User)
                    .map(|m| m.text())
                    .unwrap_or_default();
                let seam = hooks.intercept(
                    &input,
                    &settings.model,
                    crate::plugin::Seam::text("prompt", settings.system_prompt.clone()),
                );
                seam.as_text()
                    .map(str::to_string)
                    .unwrap_or_else(|| settings.system_prompt.clone())
            }
            None => settings.system_prompt.clone(),
        };
        if !system_prompt.is_empty() {
            head.push(system_prompt);
        }
        // What the `pre-step` listener asked to add for this step alone. Appended
        // after the standing prompt so it cannot change the cached prefix, and
        // bounded because a plugin that returns a novel every step would spend
        // the window it was trying to help with.
        if let Some(note) = step_note {
            if !note.trim().is_empty() {
                added.push(note.chars().take(4096).collect());
            }
        }
        // What the plugins want added, requested here rather than baked into
        // `system_prompt`: only this point knows the user's message, which is
        // the whole input of a retriever.
        if let Some(hooks) = &settings.plugin_hooks {
            // The newest user message is this turn's input; the loop has already
            // pushed it by the time a request is built, so reading it back off
            // the history is exact rather than a remembered copy.
            let input = self
                .messages
                .iter()
                .rev()
                .find(|m| m.role == crate::message::Role::User)
                .map(|m| m.text())
                .unwrap_or_default();
            for text in hooks.prompt_additions(&input, &settings.model) {
                added.push(text);
            }
        }
        // A prefix fragment is a standing rule: it belongs above the history,
        // and under CacheFirst the engine has already moved it elsewhere rather
        // than dropping it.
        for text in &injection.prefix {
            head.push(text.clone());
        }
        if !head.is_empty() {
            request.push(Message::system(head.join("\n\n")));
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

        for (index, message) in history.iter().enumerate() {
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
