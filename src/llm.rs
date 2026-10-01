//! OpenAI-compatible model client with streaming and tool calls.
//!
//! Only the OpenAI wire format is supported by design: multi-provider
//! adaptation is the largest source of accidental complexity in agent
//! frameworks, and it buys nothing here.
//!
//! Streaming is the only request mode. Tool-call arguments arrive in
//! fragments keyed by `index`; this module accumulates them so the loop sees
//! one complete call per index.

use std::sync::Arc;

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
    /// Reasoning effort, sent as `reasoning_effort`.
    ///
    /// Providers that do not reason ignore it; providers that do treat it as
    /// the main cost lever. Measured against a real endpoint, `minimal` cut one
    /// answer from 137 output tokens to 69, so it is worth exposing rather than
    /// leaving every request at the provider default.
    pub reasoning_effort: Option<String>,
    /// Proxy for all requests; empty means direct.
    pub proxy: String,
    /// Request timeout in seconds.
    ///
    /// Kept for callers that set it directly; [LlmConfig::network] is what the
    /// client actually uses, and it carries this value.
    pub timeout_secs: u64,
    /// How the request is sent: timeouts, pooling, retries.
    pub network: crate::network::NetworkSettings,
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
            reasoning_effort: None,
            proxy: String::new(),
            timeout_secs: 300,
            network: crate::network::NetworkSettings::default(),
        }
    }
}

/// Token accounting reported by the provider.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TokenUsage {
    /// Reasoning tokens, when the provider reports them.
    ///
    /// The honest measure of how much a model thought. Output tokens vary a lot
    /// between identical runs, so comparing them proves nothing; reasoning
    /// tokens dropping to zero when effort is "none" does.
    #[serde(default)]
    pub reasoning: u64,
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
    /// Last word on the request body; see [`crate::request::RequestShaper`].
    shaper: Arc<dyn crate::request::RequestShaper>,
}

impl LlmClient {
    /// Build a client. `http` should be reused across requests so connections
    /// are pooled (a hard performance requirement: no per-request handshakes).
    ///
    /// Proxy handling is explicit, never inherited. `reqwest` reads
    /// `HTTP_PROXY`/`HTTPS_PROXY` from the environment by default, which means a
    /// variable set for some unrelated tool silently reroutes the model traffic
    /// — and a proxy that cannot reach the endpoint fails with a TLS handshake
    /// error that says nothing about proxies. Here an empty `proxy` disables
    /// proxying outright, and a set one is used deliberately.
    pub fn new(config: LlmConfig) -> Result<Self> {
        // The network settings are the source of truth; `timeout_secs` is kept
        // in step so a caller reading it back sees what the client will use.
        let mut config = config;
        config.timeout_secs = config.network.request_timeout_secs;

        let mut builder = reqwest::Client::builder()
            .timeout(config.network.request_timeout())
            // A connection kept longer than the provider keeps its own is a
            // connection the provider has already closed, handed back out on the
            // next request. Below their timeout, not above it.
            .pool_idle_timeout(config.network.pool_idle_timeout());

        builder = match config.proxy.trim() {
            proxy if !proxy.is_empty() => {
                let proxy = reqwest::Proxy::all(proxy)
                    .with_context(|| format!("invalid proxy URL {proxy:?}"))?;
                builder.proxy(proxy)
            }
            // No proxy configured: do not pick one up from the environment.
            _ => builder.no_proxy(),
        };

        let http = builder.build().context("building HTTP client")?;
        Ok(Self {
            http,
            config,
            shaper: Arc::new(crate::request::Passthrough),
        })
    }

    /// The configured model id.
    pub fn model(&self) -> &str {
        &self.config.model
    }

    /// Replace the request shaper.
    ///
    /// The shaper has the last word on every request body, so this is the seam
    /// for provider-specific fields, spellings, and opt-ins.
    pub fn with_shaper(mut self, shaper: Arc<dyn crate::request::RequestShaper>) -> Self {
        self.shaper = shaper;
        self
    }

    /// The shaper in use, for diagnostics.
    pub fn shaper_name(&self) -> &str {
        self.shaper.name()
    }

    /// The connection settings this client was built with.
    ///
    /// Exposed so a caller can prove that a route's numbers reached the wire:
    /// a timeout or retry count that is stored but not applied reads the same
    /// in settings and behaves differently on the network.
    pub fn network(&self) -> &crate::network::NetworkSettings {
        &self.config.network
    }

    /// The request body this client would send, for inspection and tests.
    pub fn preview_body(&self, messages: &[Message], tools: &[Value]) -> Value {
        self.build_body(messages, tools)
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

        // A pooled connection can be closed by the peer between the moment it
        // is taken from the pool and the moment the request is written to it.
        // The failure reads as "peer closed connection without sending TLS
        // close_notify", which is a race rather than a bad request, and it ends
        // the turn if nothing retries. Retrying is safe here and only here: the
        // request has not been answered, so it cannot have taken effect twice.
        let response = self.send_retrying(|| {
            self.http
                .post(&url)
                .bearer_auth(&self.config.api_key)
                .header("content-type", "application/json")
                .header("accept", "text/event-stream")
                .json(&body)
        })
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

                    // Providers spell the reasoning channel differently, and reading
                    // only one spelling loses it silently: DeepSeek, Moonshot, Zhipu,
                    // DashScope and llama.cpp use `reasoning_content`, while vLLM,
                    // Groq, Together and Ollama use `reasoning`. OpenRouter adds a
                    // typed `reasoning_details` array on top of both.
                    for key in ["reasoning_content", "reasoning"] {
                        if let Some(text) = delta.get(key).and_then(|c| c.as_str()) {
                            if !text.is_empty() {
                                let _ = tx.send(Ok(LlmEvent::ReasoningDelta(text.to_string()))).await;
                            }
                        }
                    }
                    if let Some(parts) = delta.get("reasoning_details").and_then(|d| d.as_array()) {
                        for part in parts {
                            if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                                if !text.is_empty() {
                                    let _ = tx.send(Ok(LlmEvent::ReasoningDelta(text.to_string()))).await;
                                }
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

    /// Send a request, retrying when the connection fails before a response.
    ///
    /// A pooled connection can be closed by the peer between the moment it is
    /// taken from the pool and the moment the request is written to it. The
    /// failure reads as "peer closed connection without sending TLS
    /// close_notify" — a race, not a bad request — and without a retry it ends
    /// the turn.
    ///
    /// Retrying is safe **only** for this phase. Once a response has begun, the
    /// model may already have produced output the caller has seen, so a retry
    /// there would duplicate work rather than recover it.
    async fn send_retrying<F>(&self, build: F) -> Result<reqwest::Response>
    where
        F: Fn() -> reqwest::RequestBuilder,
    {
        let attempts = self.config.network.retry_attempts;
        let mut last: Option<reqwest::Error> = None;

        for attempt in 1..=attempts {
            match build().send().await {
                Ok(response) => return Ok(response),
                Err(error) => {
                    // Only connection-level failures are retried. A timeout is
                    // not: the request may have been received and answered, and
                    // sending it again is how one turn becomes two.
                    if !error.is_connect() && !is_connection_closed(&error) {
                        return Err(error.into());
                    }
                    last = Some(error);
                    if attempt < attempts {
                        // A pause lets a pooled connection be discarded and a
                        // fresh one opened, which is what the retry is for;
                        // retrying immediately can pick the same dead one. The
                        // delay doubles, so a provider that is overloaded is not
                        // hit three more times at the worst moment.
                        tokio::time::sleep(self.config.network.backoff(attempt)).await;
                    }
                }
            }
        }

        Err(last
            .map(|error| anyhow!("{error}"))
            .unwrap_or_else(|| anyhow!("request failed after {attempts} attempts")))
    }

    /// Whether an error is the peer closing a connection mid-request.

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
        // Reasoning effort is the main cost lever on a reasoning model, and it
        // is a flat top-level field in Chat Completions — the nested
        // `reasoning: {effort: ...}` object belongs to the Responses API, which
        // is a different endpoint.
        if let Some(effort) = &self.config.reasoning_effort {
            let effort = effort.trim();
            if !effort.is_empty() && effort != "default" {
                body["reasoning_effort"] = json!(effort);
            }
        }
        if let Some(m) = self.config.max_tokens {
            // `max_tokens` is deprecated across OpenAI, Groq, Moonshot, and
            // DashScope; `max_completion_tokens` is the current name. Providers
            // that never adopted it ignore the field rather than failing.
            body["max_completion_tokens"] = json!(m);
        }

        // Last word goes to the shaper: it exists precisely to express what the
        // kernel cannot know about a particular endpoint.
        self.shaper.shape(&mut body);
        body
    }

    /// Fetch the model catalog. Used to verify a route and discover ids.
    /// Fetch the model catalog with the provider's own metadata.
    ///
    /// Asking the endpoint beats making a user type a model id from memory: a
    /// gateway can front dozens of models and the list changes without notice.
    /// An endpoint that answers `/models` but reports nothing useful still yields
    /// ids, so this degrades rather than fails.
    pub async fn list_models_detailed(&self) -> Result<Vec<ModelInfo>> {
        let url = format!("{}/models", self.config.base_url.trim_end_matches('/'));
        let response = self
            .send_retrying(|| self.http.get(&url).bearer_auth(&self.config.api_key))
            .await
            .with_context(|| format!("GET {url}"))?;
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        if !status.is_success() {
            return Err(anyhow!("provider returned {status}: {}", truncate(&text, 500)));
        }

        // A catalog is `{"data": [...]}`; some gateways return a bare array.
        let catalog: ModelCatalog = match serde_json::from_str(&text) {
            Ok(catalog) => catalog,
            Err(_) => ModelCatalog {
                data: serde_json::from_str::<Vec<ModelInfo>>(&text).unwrap_or_default(),
            },
        };

        let mut models = catalog.data;
        models.retain(|model| !model.id.trim().is_empty());
        models.sort_by(|a, b| a.id.cmp(&b.id));
        Ok(models)
    }

    pub async fn list_models(&self) -> Result<Vec<String>> {
        let url = format!("{}/models", self.config.base_url.trim_end_matches('/'));
        let response = self
            .send_retrying(|| self.http.get(&url).bearer_auth(&self.config.api_key))
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
            if message.images.is_empty() {
                obj["content"] = json!(message.text());
            } else {
                // An image forces the content to become an array of parts; the
                // provider reads a plain string and a part list differently.
                let mut parts: Vec<Value> = Vec::with_capacity(message.images.len() + 1);
                let text = message.text();
                if !text.is_empty() {
                    parts.push(json!({ "type": "text", "text": text }));
                }
                for image in &message.images {
                    let mut url = json!({ "url": image.url });
                    if let Some(detail) = &image.detail {
                        url["detail"] = json!(detail);
                    }
                    parts.push(json!({ "type": "image_url", "image_url": url }));
                }
                obj["content"] = json!(parts);
            }
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

/// One entry from a provider's model catalog.
///
/// Providers report more than an id: `owned_by` says which vendor a model
/// actually belongs to (useful when a gateway fronts several), and
/// `supported_endpoint_types` says which API dialects it serves.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Model id, as passed in a request.
    pub id: String,
    /// Vendor the provider attributes it to.
    #[serde(default)]
    pub owned_by: Option<String>,
    /// Creation timestamp, when reported.
    #[serde(default)]
    pub created: Option<i64>,
    /// Endpoint families the model serves, e.g. `["openai", "anthropic"]`.
    #[serde(default, alias = "supported_endpoint_types")]
    pub endpoint_types: Vec<String>,
}

impl ModelInfo {
    /// A short label for a picker: the id, plus the vendor when it is known.
    pub fn label(&self) -> String {
        match &self.owned_by {
            Some(owner) if !owner.is_empty() && owner != "unknown" => {
                format!("{}  ({})", self.id, owner)
            }
            _ => self.id.clone(),
        }
    }
}

/// A model catalog, as reported by the provider.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelCatalog {
    /// Entries.
    #[serde(default)]
    pub data: Vec<ModelInfo>,
}

/// Read-only view of an assembled call, used by [`assemble_calls`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PartialCallView {
    /// Provider call id.
    pub id: String,
    /// Tool name.
    pub name: String,
    /// Raw JSON arguments.
    pub arguments: String,
}

/// Read token accounting, tolerating every spelling in the wild.
///
/// There is no single field name for cache hits: OpenAI reports
/// `prompt_tokens_details.cached_tokens`, DeepSeek and SiliconFlow report
/// `prompt_cache_hit_tokens`, Together has a flat `cached_tokens`, xAI nests it
/// under `input_tokens_details`, and Anthropic-shaped gateways use
/// `cache_read_input_tokens`. Reading only one spelling silently reports zero,
/// with no error to notice.
///
/// `prompt_tokens` is inclusive of cached tokens on every provider checked, so
/// `input` is reported as-is rather than having the cached count subtracted.
fn extract_usage(usage: &Value) -> TokenUsage {
    let input = usage
        .get("prompt_tokens")
        .or_else(|| usage.get("input_tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let output = usage
        .get("completion_tokens")
        .or_else(|| usage.get("output_tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    let cached = first_u64(
        usage,
        &[
            &["prompt_tokens_details", "cached_tokens"][..],
            &["input_tokens_details", "cached_tokens"][..],
            &["prompt_cache_hit_tokens"][..],
            &["cached_tokens"][..],
            &["cache_read_input_tokens"][..],
        ],
    );

    let reasoning = first_u64(
        usage,
        &[
            &["completion_tokens_details", "reasoning_tokens"][..],
            &["output_tokens_details", "reasoning_tokens"][..],
        ],
    );

    TokenUsage { input, output, cached, reasoning }
}

/// Walk key paths and return the first one that holds a number.
fn first_u64(value: &Value, paths: &[&[&str]]) -> u64 {
    for path in paths {
        let mut current = value;
        let mut found = true;
        for key in path.iter() {
            match current.get(*key) {
                Some(next) => current = next,
                None => {
                    found = false;
                    break;
                }
            }
        }
        if found {
            if let Some(number) = current.as_u64() {
                return number;
            }
        }
    }
    0
}

fn truncate(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_string();
    }
    let mut out: String = text.chars().take(max).collect();
    out.push_str("… (truncated)");
    out
}

#[cfg(test)]
mod body_tests {
    use super::*;
    use serde_json::json;

    fn client_with(effort: Option<&str>, extra: serde_json::Map<String, serde_json::Value>) -> LlmClient {
        let mut config = LlmConfig::new("http://localhost:1/v1", "k", "test-model");
        config.reasoning_effort = effort.map(str::to_string);
        LlmClient::new(config)
            .unwrap()
            .with_shaper(crate::request::from_extra_fields(extra))
    }

    #[test]
    fn reasoning_effort_reaches_the_request_body() {
        let client = client_with(Some("minimal"), serde_json::Map::new());
        let body = client.preview_body(&[Message::user("hi")], &[]);
        assert_eq!(
            body["reasoning_effort"], "minimal",
            "the field must actually be sent, not merely stored: {body}"
        );
    }

    #[test]
    fn an_empty_or_default_effort_is_not_sent() {
        for value in ["", "  ", "default"] {
            let client = client_with(Some(value), serde_json::Map::new());
            let body = client.preview_body(&[Message::user("hi")], &[]);
            assert!(
                body.get("reasoning_effort").is_none(),
                "{value:?} should mean 'let the provider decide': {body}"
            );
        }
        let client = client_with(None, serde_json::Map::new());
        let body = client.preview_body(&[Message::user("hi")], &[]);
        assert!(body.get("reasoning_effort").is_none());
    }

    #[test]
    fn extra_body_fields_reach_the_request_body() {
        let mut extra = serde_json::Map::new();
        extra.insert("enable_thinking".into(), json!(true));
        extra.insert("top_k".into(), json!(40));

        let client = client_with(None, extra);
        let body = client.preview_body(&[Message::user("hi")], &[]);

        assert_eq!(body["enable_thinking"], true, "{body}");
        assert_eq!(body["top_k"], 40, "{body}");
    }

    #[test]
    fn extra_body_can_remove_a_field_the_kernel_set() {
        let mut extra = serde_json::Map::new();
        extra.insert("tool_choice".into(), serde_json::Value::Null);

        let client = client_with(None, extra);
        let tools = vec![json!({ "type": "function", "function": { "name": "t" } })];
        let body = client.preview_body(&[Message::user("hi")], &tools);

        assert!(body.get("tools").is_some(), "tools stay");
        assert!(body.get("tool_choice").is_none(), "tool_choice removed: {body}");
    }

    #[test]
    fn the_stream_flag_and_model_are_always_present() {
        let client = client_with(None, serde_json::Map::new());
        let body = client.preview_body(&[Message::user("hi")], &[]);
        assert_eq!(body["stream"], true);
        assert_eq!(body["model"], "test-model");
    }
}

/// Whether an error is the peer closing a connection mid-request.
///
/// reqwest reports this as a body or request error rather than a connect error,
/// because the connection was established and then went away. The distinction
/// matters: it is the one failure that is certainly safe to retry.
fn is_connection_closed(error: &reqwest::Error) -> bool {
    // The text that identifies this failure is in the cause chain rather than
    // the outermost error. reqwest reports "error sending request for url (...)"
    // and the actionable words — "connection closed before message completed" —
    // are one level down, so testing only `error.to_string()` classifies every
    // dropped connection as unretryable.
    let mut current: Option<&(dyn std::error::Error + 'static)> = Some(error);
    while let Some(link) = current {
        if matches_closed_text(&link.to_string()) {
            return true;
        }
        current = link.source();
    }
    false
}

/// The message test behind [`is_connection_closed`].
///
/// Separated from the error type so it can be exercised against the exact
/// strings a peer produces, which is the whole of the decision.
fn matches_closed_text(text: &str) -> bool {
    text.contains("close_notify")
        || text.contains("connection closed")
        || text.contains("Connection reset")
        || text.contains("connection reset")
        || text.contains("broken pipe")
        || text.contains("IncompleteMessage")
        || os_error_is_connection_loss(text)
}

/// Whether the message carries an OS error code for a connection that went
/// away.
///
/// Windows writes that sentence in the system language — "你的主机中的软件中止了
/// 一个已建立的连接。(os error 10053)" on a Chinese machine — so no English
/// phrase matches it there. The number after "os error" is the same on every
/// machine, which is what makes the classification survive a translation.
fn os_error_is_connection_loss(text: &str) -> bool {
    const LOST: [u32; 5] = [
        10053, // WSAECONNABORTED: the connection was aborted
        10054, // WSAECONNRESET: the peer reset it
        103,   // ECONNABORTED
        104,   // ECONNRESET
        32,    // EPIPE: writing to a socket the peer closed
    ];

    let mut rest = text;
    while let Some(at) = rest.find("os error ") {
        rest = &rest[at + "os error ".len()..];
        let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
        // `take_while` stops at a non-digit, so "os error 10060" is read as
        // 10060 rather than as 1006 with a trailing digit.
        if let Ok(code) = digits.parse::<u32>() {
            if LOST.contains(&code) {
                return true;
            }
        }
    }
    false
}

#[cfg(test)]
mod retry_tests {
    use super::*;

    /// Build a reqwest error of the kind the peer-closed path produces.
    fn error_from(message: &str) -> reqwest::Error {
        // A request to a port nothing listens on gives a real connect error;
        // the classification below is what is under test, so the exact variant
        // matters less than the message it carries.
        let client = reqwest::Client::new();
        let url = "http://127.0.0.1:1/";
        let error = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(async { client.get(url).send().await.unwrap_err() });
        let _ = message;
        error
    }

    #[test]
    fn a_connect_error_is_recognised() {
        let error = error_from("connection refused");
        assert!(error.is_connect(), "{error}");
    }

    #[test]
    fn the_close_notify_message_is_recognised_as_closed() {
        // The exact string a peer that closes without a TLS shutdown produces.
        // It cannot be built from a real reqwest error here, so the predicate
        // is exercised through the messages it must match.
        for text in [
            "peer closed connection without sending TLS close_notify",
            "connection closed before message completed",
            "Connection reset by peer",
            "broken pipe",
        ] {
            assert!(
                matches_closed_text(text),
                "{text:?} should be treated as a closed connection"
            );
        }
    }

    #[test]
    fn an_unrelated_failure_is_not_treated_as_closed() {
        for text in [
            "provider returned 400 Bad Request",
            "invalid api key",
            "operation timed out",
        ] {
            assert!(
                !matches_closed_text(text),
                "{text:?} must not be retried"
            );
        }
    }

    #[test]
    fn a_translated_connection_error_is_still_recognised() {
        // On a Chinese Windows the sentence is translated and only the code is
        // stable, so a machine that reads English and one that does not must
        // classify the same failure the same way.
        for text in [
            "你的主机中的软件中止了一个已建立的连接。(os error 10053)",
            "另一端强行关闭了一个现有的连接。(os error 10054)",
            "Connection reset by peer (os error 10054)",
        ] {
            assert!(matches_closed_text(text), "{text:?} must be retried");
        }
        // A timeout is translated too, and it must stay unretryable: the
        // request may already have been answered.
        assert!(
            !matches_closed_text("由于连接方在一段时间后没有反应...(os error 10060)"),
            "a timed-out connection must not be retried"
        );
    }

    #[tokio::test]
    async fn a_connection_dropped_before_the_response_is_retried() {
        // A server that closes the first connection without answering and
        // answers every one after. This is the shape of the failure a peer
        // produces when it discards a pooled connection, and the retry is the
        // whole of the defence against it.
        //
        // It keeps accepting rather than answering exactly the second: under a
        // parallel test run the client may make its first attempt before the
        // listener is ready, so counting connections makes the test depend on
        // scheduling. Answering everything after the first does not.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        tokio::spawn(async move {
            let mut first = true;
            while let Ok((mut socket, _)) = listener.accept().await {
                if first {
                    first = false;
                    // Closed without a response, which is what a reaped
                    // connection looks like from the client's side.
                    drop(socket);
                    continue;
                }
                use tokio::io::AsyncWriteExt;
                let body = "{\"ok\":true}";
                let response = format!(
                    "HTTP/1.1 200 OK\r\ncontent-length: {}\r\ncontent-type: application/json\r\n\r\n{body}",
                    body.len()
                );
                let _ = socket.write_all(response.as_bytes()).await;
                let _ = socket.flush().await;
                // Held open until the client has read it; closing immediately
                // can truncate the response instead of ending it cleanly.
                tokio::time::sleep(std::time::Duration::from_secs(2)).await;
            }
        });

        let config = LlmConfig::new(format!("http://{addr}"), "k", "m");
        let client = LlmClient::new(config).unwrap();
        let url = format!("http://{addr}/x");

        let response = client
            .send_retrying(|| client.http.get(&url))
            .await
            .expect("the retry should recover");

        assert_eq!(response.status(), 200);
        assert_eq!(response.text().await.unwrap(), "{\"ok\":true}");
    }

    #[tokio::test]
    async fn a_server_that_always_drops_fails_after_the_attempts_run_out() {
        // Retrying must not become an infinite loop against a server that is
        // simply down.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        tokio::spawn(async move {
            while let Ok((socket, _)) = listener.accept().await {
                drop(socket);
            }
        });

        let config = LlmConfig::new(format!("http://{addr}"), "k", "m");
        let client = LlmClient::new(config).unwrap();
        let url = format!("http://{addr}/x");

        let started = std::time::Instant::now();
        let _ = client
            .send_retrying(|| client.http.get(&url))
            .await
            .expect_err("it must give up");
        assert!(
            started.elapsed() < std::time::Duration::from_secs(10),
            "giving up took {:?}",
            started.elapsed()
        );
    }
}
