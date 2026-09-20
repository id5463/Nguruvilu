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
    /// Compactions performed during this turn, in order.
    ///
    /// The caller persists these so a reloaded session matches what the model
    /// actually saw.
    pub compactions: Vec<Compaction>,
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
            compactions: Vec::new(),
        };

        let schemas = settings.tools.schemas();

        // What the provider last reported for the prompt. This is the only
        // accurate size signal available: estimating from characters drifts
        // badly once tool output is involved.
        let mut last_prompt_tokens: Option<usize> = None;

        for step in 1..=self.max_steps {
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
            }
            // The route may name a different model than the client was built
            // with; switching is a clone, not a reconnect.
            let client = self.client.with_model(&settings.model);

            let model_started = Instant::now();
            let mut rx = client.chat_stream(&request, &schemas).await?;

            let mut text = String::new();
            let mut calls: Vec<PartialCallView> = Vec::new();
            let mut step_usage = TokenUsage::default();

            while let Some(event) = rx.recv().await {
                match event? {
                    LlmEvent::TextDelta(delta) => {
                        self.observer.on_text(&delta);
                        text.push_str(&delta);
                    }
                    LlmEvent::ReasoningDelta(delta) => self.observer.on_reasoning(&delta),
                    LlmEvent::ToolCallDelta { index, id, name, arguments } => {
                        while calls.len() <= index {
                            calls.push(PartialCallView::default());
                        }
                        let slot = &mut calls[index];
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
                    LlmEvent::Usage(usage) => step_usage = usage,
                    LlmEvent::Finished { .. } => {}
                }
            }

            outcome.timing.model_ms += model_started.elapsed().as_millis();
            outcome.usage.input += step_usage.input;
            outcome.usage.output += step_usage.output;
            outcome.usage.cached += step_usage.cached;

            // Remember the prompt size the provider just reported, so the next
            // step can decide whether history needs reducing.
            if step_usage.input > 0 {
                last_prompt_tokens = Some(step_usage.input as usize);
            }

            let indexed: Vec<(usize, PartialCallView)> = calls
                .iter()
                .enumerate()
                .map(|(i, c)| (i, c.clone()))
                .collect();
            let tool_calls: Vec<ToolCall> = llm::assemble_calls(&indexed);

            let assistant = Message::assistant_tools(
                tool_calls.clone(),
                if text.is_empty() { None } else { Some(text.clone()) },
            );
            self.messages.push(assistant.clone());
            outcome.new_messages.push(assistant);
            outcome.text = text;

            if tool_calls.is_empty() {
                return Ok(outcome);
            }

            // Dispatch every call concurrently, then restore call order.
            let tools_started = Instant::now();
            let mut set: JoinSet<(usize, ToolCall, Result<String>)> = JoinSet::new();
            for (position, call) in tool_calls.iter().enumerate() {
                let registry = Arc::clone(&settings.tools);
                let call = call.clone();
                self.observer.on_tool_start(&call.name, &call.arguments);
                set.spawn(async move {
                    let result = registry.execute(&call.name, &call.arguments).await;
                    (position, call, result)
                });
            }

            let mut results: Vec<Option<(ToolCall, Result<String>)>> =
                (0..tool_calls.len()).map(|_| None).collect();
            while let Some(joined) = set.join_next().await {
                let (position, call, result) = joined?;
                results[position] = Some((call, result));
            }
            outcome.timing.tools_ms += tools_started.elapsed().as_millis();

            for entry in results.into_iter().flatten() {
                let (call, result) = entry;
                outcome.tool_calls += 1;
                let (ok, body) = match result {
                    Ok(output) => (true, output),
                    Err(error) => (false, format!("error: {error:#}")),
                };
                self.observer.on_tool_end(&call.name, ok, &body);
                let message = Message::tool_result(call.id.clone(), call.name.clone(), body);
                self.messages.push(message.clone());
                outcome.new_messages.push(message);
            }
        }

        outcome.text = format!(
            "stopped after {} steps without a final answer",
            self.max_steps
        );
        Ok(outcome)
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
}
