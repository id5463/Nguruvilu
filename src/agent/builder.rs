//! `Agent` 本身，以及怎么把它装配起来。
//!
//! 二十个 `with_*` 方法，每一个给一轮对话加一个东西：观察者、取消信号、
//! 最大步数、摘要器、注入源、任务收件箱……它们都是同一种形状，所以在一处。
//!
//! 装配与运行分开放：装配是启动时的事，运行是每一轮的事。

use std::sync::Arc;

use anyhow::Result;

use crate::agent::config::{TurnConfig, TurnSettings};
use crate::agent::hooks::PluginHooks;
use crate::agent::observer::AgentObserver;
use crate::agent::state::{AgentOutcome, LoopState};
use crate::compaction::Summarizer;
use crate::llm::LlmClient;
use crate::message::Message;
use crate::window::ContextPolicy;


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
}
