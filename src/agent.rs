//! The agent loop.
//!
//! One turn is: send the history, stream the reply, dispatch any requested
//! tools **in parallel**, append their results, and repeat until the model
//! stops asking for tools.
//!
//! Two performance rules from the spec live here:
//!
//! * Independent tool calls run concurrently. Serial dispatch is the single
//!   largest avoidable cost in an agent turn — tool execution accounts for
//!   roughly a third to two thirds of total request time.
//! * The loop takes one snapshot of the tool table and system prompt at the
//!   start of a turn, so a hot reload landing mid-turn cannot change the
//!   request that is already in flight.

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

/// Per-turn timing breakdown, so overhead is measurable rather than guessed.
#[derive(Debug, Clone, Default)]
pub struct TurnTiming {
    /// Time spent waiting on the model (TTFT plus generation).
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
}

/// Drives one conversation against one model route.
pub struct Agent {
    client: LlmClient,
    tools: Arc<ToolRegistry>,
    messages: Vec<Message>,
    system_prompt: String,
    max_steps: usize,
    observer: Arc<dyn AgentObserver>,
}

impl Agent {
    /// Build an agent over an existing history.
    pub fn new(
        client: LlmClient,
        tools: Arc<ToolRegistry>,
        system_prompt: impl Into<String>,
        messages: Vec<Message>,
    ) -> Self {
        Self {
            client,
            tools,
            messages,
            system_prompt: system_prompt.into(),
            max_steps: 50,
            observer: Arc::new(SilentObserver),
        }
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

    /// Replace the system prompt.
    pub fn set_system_prompt(&mut self, prompt: impl Into<String>) {
        self.system_prompt = prompt.into();
    }

    /// Run one turn: append `prompt` and loop until the model stops calling tools.
    pub async fn run(&mut self, prompt: &str) -> Result<AgentOutcome> {
        let user_message = Message::user(prompt);
        self.messages.push(user_message.clone());

        let mut outcome = AgentOutcome {
            text: String::new(),
            steps: 0,
            tool_calls: 0,
            usage: TokenUsage::default(),
            new_messages: vec![user_message],
            timing: TurnTiming::default(),
        };

        let schemas = self.tools.schemas();

        for step in 1..=self.max_steps {
            outcome.steps = step;
            self.observer.on_step(step);

            let request = self.build_request();
            let model_started = Instant::now();
            let mut rx = self.client.chat_stream(&request, &schemas).await?;

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
                let registry = Arc::clone(&self.tools);
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
    /// The system prompt is prepended at request time rather than stored in
    /// the history, so a prompt change does not rewrite recorded messages.
    fn build_request(&self) -> Vec<Message> {
        let mut request = Vec::with_capacity(self.messages.len() + 1);
        if !self.system_prompt.is_empty() {
            request.push(Message::system(self.system_prompt.clone()));
        }
        request.extend(self.messages.iter().cloned());
        request
    }
}
