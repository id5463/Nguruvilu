//! OpenAI-compatible model client with streaming and tool calls.
//!
//! Only the OpenAI wire format is supported by design: multi-provider
//! adaptation is the largest source of accidental complexity in agent
//! frameworks, and it buys nothing here.
//!
//! Streaming is the only request mode. Tool-call arguments arrive in
//! fragments keyed by `index`; this module accumulates them so the loop sees
//! one complete call per index.

use anyhow::{anyhow, Context, Result};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::sync::mpsc;

use crate::message::{Message, ToolCall};

/// Connection settings for one model route.
#[derive(Debug, Clone)]
pub struct LlmConfig {
    /// Base URL including the version segment, e.g. `https://api.example.com/v1`.
    pub base_url: String,
    /// Bearer token.
    pub api_key: String,
    /// Model id.
    pub model: String,
    /// Sampling temperature; omitted when `None`.
    pub temperature: Option<f32>,
    /// Output token ceiling; omitted when `None`.
    pub max_tokens: Option<u32>,
    /// Request timeout in seconds.
    pub timeout_secs: u64,
}

impl LlmConfig {
    /// Build a config with defaults for the optional fields.
    pub fn new(base_url: impl Into<String>, api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            api_key: api_key.into(),
            model: model.into(),
            temperature: None,
            max_tokens: None,
            timeout_secs: 300,
        }
    }
}

/// Token accounting reported by the provider.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TokenUsage {
    /// Uncached prompt tokens.
    pub input: u64,
    /// Completion tokens.
    pub output: u64,
    /// Prompt tokens served from the provider's cache.
    pub cached: u64,
}

/// One incremental event from a streaming completion.
#[derive(Debug, Clone)]
pub enum LlmEvent {
    /// A text fragment.
    TextDelta(String),
    /// A reasoning/thinking fragment, when the provider emits one.
    ReasoningDelta(String),
    /// A tool-call fragment. `index` identifies the call being assembled.
    ToolCallDelta {
        /// Position of the call in this assistant turn.
        index: usize,
        /// Present on the first fragment of a call.
        id: Option<String>,
        /// Present on the first fragment of a call.
        name: Option<String>,
        /// Arguments fragment; concatenated in arrival order.
        arguments: Option<String>,
    },
    /// Usage report, usually on the final chunk.
    Usage(TokenUsage),
    /// Stream finished. `reason` is the provider's finish reason.
    Finished { reason: Option<String> },
}

/// A tool call being assembled from fragments.
#[derive(Debug, Default, Clone)]
struct PartialCall {
    id: String,
    name: String,
    arguments: String,
}

/// Streaming client for one model route.
#[derive(Clone)]
pub struct LlmClient {
    http: reqwest::Client,
    config: LlmConfig,
}

impl LlmClient {
    /// Build a client. `http` should be reused across requests so connections
    /// are pooled (a hard performance requirement: no per-request handshakes).
    pub fn new(config: LlmConfig) -> Result<Self> {
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(config.timeout_secs))
            .pool_idle_timeout(std::time::Duration::from_secs(90))
            .build()
            .context("building HTTP client")?;
        Ok(Self { http, config })
    }

    /// The configured model id.
    pub fn model(&self) -> &str {
        &self.config.model
    }

    /// A clone using a different model id.
    ///
    /// The HTTP client is shared, so switching models reuses the connection
    /// pool instead of reconnecting: a route change must not cost a handshake.
    pub fn with_model(&self, model: impl Into<String>) -> Self {
        let mut next = self.clone();
        next.config.model = model.into();
        next
    }

    /// Send a streaming completion request.
    ///
    /// Returns a receiver of incremental events. The request itself has
    /// already been dispatched, so an error here means the provider refused
    /// or the connection failed.
    pub async fn chat_stream(&self, messages: &[Message], tools: &[Value]) -> Result<mpsc::Receiver<Result<LlmEvent>>> {
        let body = self.build_body(messages, tools);
        let url = format!("{}/chat/completions", self.config.base_url.trim_end_matches('/'));

        let response = self
            .http
            .post(&url)
            .bearer_auth(&self.config.api_key)
            .header("content-type", "application/json")
            .header("accept", "text/event-stream")
            .json(&body)
            .send()
            .await
            .with_context(|| format!("POST {url}"))?;

        let status = response.status();
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            return Err(anyhow!("provider returned {status}: {}", truncate(&text, 2000)));
        }

        let (tx, rx) = mpsc::channel::<Result<LlmEvent>>(256);
        let mut stream = response.bytes_stream();

        tokio::spawn(async move {
            let mut buffer = String::new();
            let mut calls: Vec<PartialCall> = Vec::new();
            let mut finish_reason: Option<String> = None;

            while let Some(chunk) = stream.next().await {
                let chunk = match chunk {
                    Ok(c) => c,
                    Err(e) => {
                        let _ = tx.send(Err(anyhow!("stream error: {e}"))).await;
                        return;
                    }
                };
                buffer.push_str(&String::from_utf8_lossy(&chunk));

                // SSE frames are newline-delimited; keep the trailing partial line.
                while let Some(pos) = buffer.find('\n') {
                    let line = buffer[..pos].trim_end_matches('\r').to_string();
                    buffer.drain(..=pos);

                    let Some(data) = line.strip_prefix("data:") else { continue };
                    let data = data.trim();
                    if data.is_empty() {
                        continue;
                    }
                    if data == "[DONE]" {
                        let _ = tx.send(Ok(LlmEvent::Finished { reason: finish_reason.clone() })).await;
                        return;
                    }

                    let parsed: Value = match serde_json::from_str(data) {
                        Ok(v) => v,
                        Err(_) => continue,
                    };

                    if let Some(usage) = parsed.get("usage").filter(|u| !u.is_null()) {
                        let _ = tx.send(Ok(LlmEvent::Usage(extract_usage(usage)))).await;
                    }

                    let Some(choice) = parsed.get("choices").and_then(|c| c.get(0)) else { continue };

                    if let Some(reason) = choice.get("finish_reason").and_then(|r| r.as_str()) {
                        finish_reason = Some(reason.to_string());
                    }

                    let Some(delta) = choice.get("delta") else { continue };

                    if let Some(text) = delta.get("content").and_then(|c| c.as_str()) {
                        if !text.is_empty() {
                            let _ = tx.send(Ok(LlmEvent::TextDelta(text.to_string()))).await;
                        }
                    }

                    // Providers spell the reasoning channel differently; accept the common ones.
                    for key in ["reasoning_content", "reasoning"] {
                        if let Some(text) = delta.get(key).and_then(|c| c.as_str()) {
                            if !text.is_empty() {
                                let _ = tx.send(Ok(LlmEvent::ReasoningDelta(text.to_string()))).await;
                            }
                        }
                    }

                    if let Some(tool_calls) = delta.get("tool_calls").and_then(|t| t.as_array()) {
                        for call in tool_calls {
                            let index = call.get("index").and_then(|i| i.as_u64()).unwrap_or(0) as usize;
                            while calls.len() <= index {
                                calls.push(PartialCall::default());
                            }
                            let slot = &mut calls[index];

                            let id = call.get("id").and_then(|i| i.as_str()).map(str::to_string);
                            let name = call
                                .get("function")
                                .and_then(|f| f.get("name"))
                                .and_then(|n| n.as_str())
                                .map(str::to_string);
                            let args = call
                                .get("function")
                                .and_then(|f| f.get("arguments"))
                                .and_then(|a| a.as_str())
                                .map(str::to_string);

                            if let Some(id) = &id {
                                slot.id = id.clone();
                            }
                            if let Some(name) = &name {
                                slot.name = name.clone();
                            }
                            if let Some(args) = &args {
                                slot.arguments.push_str(args);
                            }

                            let _ = tx
                                .send(Ok(LlmEvent::ToolCallDelta { index, id, name, arguments: args }))
                                .await;
                        }
                    }
                }
            }

            // Stream ended without an explicit [DONE].
            let _ = tx.send(Ok(LlmEvent::Finished { reason: finish_reason })).await;
        });

        Ok(rx)
    }

    fn build_body(&self, messages: &[Message], tools: &[Value]) -> Value {
        let wire: Vec<Value> = messages.iter().map(to_wire).collect();
        let mut body = json!({
            "model": self.config.model,
            "messages": wire,
            "stream": true,
        });
        if !tools.is_empty() {
            body["tools"] = json!(tools);
            body["tool_choice"] = json!("auto");
        }
        if let Some(t) = self.config.temperature {
            body["temperature"] = json!(t);
        }
        if let Some(m) = self.config.max_tokens {
            body["max_tokens"] = json!(m);
        }
        body
    }

    /// Fetch the model catalog. Used to verify a route and discover ids.
    pub async fn list_models(&self) -> Result<Vec<String>> {
        let url = format!("{}/models", self.config.base_url.trim_end_matches('/'));
        let response = self
            .http
            .get(&url)
            .bearer_auth(&self.config.api_key)
            .send()
            .await
            .with_context(|| format!("GET {url}"))?;
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        if !status.is_success() {
            return Err(anyhow!("provider returned {status}: {}", truncate(&text, 500)));
        }
        let parsed: Value = serde_json::from_str(&text).context("parsing model list")?;
        let ids = parsed
            .get("data")
            .and_then(|d| d.as_array())
            .map(|items| {
                items
                    .iter()
                    .filter_map(|m| m.get("id").and_then(|i| i.as_str()).map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();
        Ok(ids)
    }
}

/// Convert one neutral message to the OpenAI wire shape.
pub fn to_wire(message: &Message) -> Value {
    let mut obj = json!({ "role": message.role.as_wire() });

    match message.role {
        crate::message::Role::Assistant if !message.tool_calls.is_empty() => {
            obj["content"] = match &message.content {
                Some(text) if !text.is_empty() => json!(text),
                _ => Value::Null,
            };
            let calls: Vec<Value> = message
                .tool_calls
                .iter()
                .map(|c| {
                    json!({
                        "id": c.id,
                        "type": "function",
                        "function": { "name": c.name, "arguments": c.arguments },
                    })
                })
                .collect();
            obj["tool_calls"] = json!(calls);
        }
        crate::message::Role::Tool => {
            obj["content"] = json!(message.text());
            if let Some(id) = &message.tool_call_id {
                obj["tool_call_id"] = json!(id);
            }
        }
        _ => {
            obj["content"] = json!(message.text());
        }
    }

    obj
}

/// Assemble complete tool calls from their accumulated fragments.
pub fn assemble_calls(parts: &[(usize, PartialCallView)]) -> Vec<ToolCall> {
    let mut ordered: Vec<(usize, PartialCallView)> = parts.to_vec();
    ordered.sort_by_key(|(i, _)| *i);
    ordered
        .into_iter()
        .filter(|(_, p)| !p.name.is_empty())
        .map(|(_, p)| ToolCall { id: p.id, name: p.name, arguments: p.arguments })
        .collect()
}

/// Read-only view of an assembled call, used by [`assemble_calls`].
#[derive(Debug, Clone, Default)]
pub struct PartialCallView {
    /// Provider call id.
    pub id: String,
    /// Tool name.
    pub name: String,
    /// Raw JSON arguments.
    pub arguments: String,
}

fn extract_usage(usage: &Value) -> TokenUsage {
    let input = usage.get("prompt_tokens").and_then(|v| v.as_u64()).unwrap_or(0);
    let output = usage.get("completion_tokens").and_then(|v| v.as_u64()).unwrap_or(0);
    let cached = usage
        .get("prompt_tokens_details")
        .and_then(|d| d.get("cached_tokens"))
        .and_then(|v| v.as_u64())
        .or_else(|| usage.get("prompt_cache_hit_tokens").and_then(|v| v.as_u64()))
        .unwrap_or(0);
    TokenUsage { input: input.saturating_sub(cached), output, cached }
}

fn truncate(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_string();
    }
    let mut out: String = text.chars().take(max).collect();
    out.push_str("… (truncated)");
    out
}
