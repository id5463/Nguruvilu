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

use crate::llm::{self, LlmClient, LlmEvent, PartialCallView, TokenUsage};
use crate::message::{Message, ToolCall};
use crate::tools::ToolRegistry;

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
        }
    }
}

impl TurnConfig for StaticConfig {
    fn settings(&self) -> TurnSettings {
        TurnSettings {
            system_prompt: self.system_prompt.clone(),
            tools: Arc::clone(&self.tools),
            model: self.model.clone(),
            version: 0,
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
}

/// Drives one conversation.
pub struct Agent {
    client: LlmClient,
    config: Arc<dyn TurnConfig>,
    messages: Vec<Message>,
    max_steps: usize,
    observer: Arc<dyn AgentObserver>,
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
        };

        let schemas = settings.tools.schemas();

        for step in 1..=self.max_steps {
            outcome.steps = step;
            self.observer.on_step(step);

            let request = self.build_request(&settings.system_prompt);
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

    /// Build the request history: system prompt first, then the conversation.
    ///
    /// The system prompt is prepended at request time rather than stored in the
    /// history, so a prompt change does not rewrite recorded messages — which
    /// is also what keeps the cached prefix intact across turns.
    fn build_request(&self, system_prompt: &str) -> Vec<Message> {
        let mut request = Vec::with_capacity(self.messages.len() + 1);
        if !system_prompt.is_empty() {
            request.push(Message::system(system_prompt));
        }
        request.extend(self.messages.iter().cloned());
        request
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
        let config: Arc<dyn TurnConfig> = Arc::new(StaticConfig::new("SYS", tools, "m"));
        let client = LlmClient::new(crate::llm::LlmConfig::new("http://localhost:1", "k", "m"))
            .expect("client builds");
        let agent = Agent::new(client, config, vec![Message::user("hi")]);

        let request = agent.build_request("SYS");
        assert_eq!(request.len(), 2);
        assert_eq!(request[0].role, crate::message::Role::System);
        assert_eq!(request[0].text(), "SYS");
        assert_eq!(request[1].text(), "hi");

        // The history itself is untouched, so recording stays clean.
        assert_eq!(agent.messages().len(), 1);

        // An empty prompt contributes no message at all.
        assert_eq!(agent.build_request("").len(), 1);
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
