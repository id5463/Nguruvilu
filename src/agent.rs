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
        }
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

    /// Cap the number of model steps in one turn.
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

    /// Run one turn: append `prompt` and loop until the model stops calling tools.
    pub async fn run(&mut self, prompt: &str) -> Result<AgentOutcome> {
        let user_message = Message::user(prompt);
        self.messages.push(user_message.clone());
        // The prompt is committed before anything is attempted: asking the
        // model is already a thing that happened.
        self.observer.on_message(&user_message);

        // Frozen for the whole turn: a reload landing now is next turn's news.
        let settings = self.config.settings();

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

        let schemas = settings.tools.schemas();

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

        for step in 1..=self.max_steps {
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
                let texts: Vec<String> = inbox.lock().expect("inbox").drain(..).collect();
                for text in texts {
                    let message = Message::user(text);
                    self.messages.push(message.clone());
                    outcome.new_messages.push(message.clone());
                    self.observer.on_message(&message);
                }
            }

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
                }
                return Ok(outcome);
            }

            // Dispatch every call concurrently, then restore call order.
            let tools_started = Instant::now();
            let mut set: JoinSet<(usize, ToolCall, Result<crate::tools::ToolOutput>)> = JoinSet::new();
            for (position, call) in tool_calls.iter().enumerate() {
                let registry = Arc::clone(&settings.tools);
                let call = call.clone();
                self.observer.on_tool_start(&call.name, &call.arguments);
                set.spawn(async move {
                    let result = registry.execute(&call.name, &call.arguments).await;
                    (position, call, result)
                });
            }

            let mut results: Vec<Option<(ToolCall, Result<crate::tools::ToolOutput>)>> =
                (0..tool_calls.len()).map(|_| None).collect();
            // Collecting is cancellable: a bash call that runs for minutes
            // must not outlive the user's decision to stop watching it.
            let mut stopped = false;
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
                    Some(Err(error)) => return Err(error.into()),
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
                    // Unreachable without a stop: every task joined above.
                    None => continue,
                };
                outcome.tool_calls += 1;
                let (ok, body) = match result {
                    Ok(output) => {
                        for image in output.images {
                            attached.push((call.name.clone(), image));
                        }
                        (true, output.text)
                    }
                    Err(error) => (false, format!("error: {error:#}")),
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

        outcome.text = format!(
            "stopped after {} steps without a final answer",
            self.max_steps
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
}
