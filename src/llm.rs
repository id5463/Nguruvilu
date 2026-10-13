//! OpenAI-compatible model client with streaming and tool calls.
//!
//! Only the OpenAI wire format is supported by design: multi-provider
//! adaptation is the largest source of accidental complexity in agent
//! frameworks, and it buys nothing here.
//!
//! Streaming is the default request mode. Tool-call arguments arrive in
//! fragments keyed by `index`; this module accumulates them so the loop sees
//! one complete call per index.
//!
//! # Why a non-streaming mode exists at all
//!
//! A gateway that answers `stream: true` by opening the response, writing one
//! empty frame, and then never writing again is not a gateway the loop can use:
//! the turn hangs until the whole-request timeout, and the retry reopens the
//! same broken stream. Measured against one such endpoint, ten consecutive
//! streaming requests produced no answer and no `[DONE]` — while the same body
//! with `stream: false` returned the complete answer, including tool calls, in
//! under six seconds.
//!
//! So the mode is a choice rather than an assumption, and both modes feed the
//! loop the same [`LlmEvent`] sequence: the difference is on the wire, never in
//! `agent.rs`.

use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::sync::mpsc;

use crate::message::{Message, ToolCall};

/// How one turn's answer is delivered.
///
/// A choice rather than an assumption, because streaming is not universally
/// available: a gateway that opens an SSE response and then stops writing
/// leaves a streaming-only client with no answer and no error until the
/// whole-request timeout. Measured against one such endpoint, no streaming
/// request ever produced `[DONE]`, while the same body with `stream: false`
/// returned the complete answer — tool calls included — in under six seconds.
///
/// The mode is chosen per route. Both modes feed the loop the same [`LlmEvent`]
/// sequence, so the difference lives on the wire and never in `agent.rs`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum Delivery {
    /// Server-sent events: the answer arrives in fragments.
    #[default]
    Stream,
    /// One JSON body: the answer arrives whole, when it is finished.
    Buffered,
}

impl Delivery {
    /// The settings and `models.json` word for this mode.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Stream => "stream",
            Self::Buffered => "buffered",
        }
    }

    /// Every accepted spelling, for a usage message.
    pub fn names() -> &'static [&'static str] {
        &["stream", "buffered"]
    }

    /// Parse what a user typed. `None` means it is not one of the two modes.
    ///
    /// `true`/`false` are accepted because the setting used to be a boolean, and
    /// `sse`/`json` because those are the names on the wire.
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "stream" | "sse" | "true" => Some(Self::Stream),
            "buffered" | "json" | "false" => Some(Self::Buffered),
            _ => None,
        }
    }
}

/// Connection settings for one model route.
#[derive(Debug, Clone)]
pub struct LlmConfig {
    /// Base URL including the version segment, e.g. `https://api.example.com/v1`.
    pub base_url: String,
    /// Bearer token.
    pub api_key: String,
    /// Model id.
    pub model: String,
    /// How the answer is delivered.
    pub delivery: Delivery,
    /// Whether a stalled stream may switch this client to [`Delivery::Buffered`].
    ///
    /// The auto-degrade half of the setting: on by default, because a route
    /// whose stream never completes is not a route worth re-trying every step.
    pub degrade_on_stall: bool,
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
            delivery: Delivery::default(),
            degrade_on_stall: true,
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

/// Why a chat request failed, tagged with whether another attempt can succeed.
///
/// The loop retries a failed step only when this says it can: a retried 401 is
/// a wasted round trip, and a retried dropped stream is a turn that survives.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestError {
    /// HTTP status when the provider answered; `None` for transport failures.
    pub status: Option<u16>,
    /// Whether another attempt may succeed.
    pub retryable: bool,
    /// Human-readable cause.
    pub message: String,
}

impl std::fmt::Display for RequestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for RequestError {}

/// Whether a chat failure is worth another attempt.
///
/// Walks the whole cause chain, because a classification placed on the root
/// error must still be visible after a `POST <url>` context is layered on top.
pub fn retryable(error: &anyhow::Error) -> bool {
    error.chain().any(|cause| {
        if let Some(failure) = cause.downcast_ref::<RequestError>() {
            return failure.retryable;
        }
        // A request timeout seen at the transport layer means no response
        // arrived, so repeating the request cannot duplicate a seen answer.
        if let Some(transport) = cause.downcast_ref::<reqwest::Error>() {
            return transport.is_timeout();
        }
        false
    })
}

/// Streaming client for one model route.
#[derive(Clone)]
pub struct LlmClient {
    http: reqwest::Client,
    config: LlmConfig,
    /// Last word on the request body; see [`crate::request::RequestShaper`].
    shaper: Arc<dyn crate::request::RequestShaper>,
    /// Set once a stream on this route has stalled with nothing delivered.
    ///
    /// Shared across clones, because a route that cannot stream once cannot
    /// stream on step nine either: without this, every step of every turn pays
    /// the whole `stream_idle` bound before discovering the same fact again.
    /// A route change builds a fresh client and so re-arms the question.
    degraded: Arc<std::sync::atomic::AtomicBool>,
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
            degraded: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
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

    /// The delivery mode this client will actually use.
    ///
    /// Differs from [`LlmConfig::delivery`] once a stream has stalled: the
    /// configured mode says what the user asked for, this says what the next
    /// request is sent as.
    pub fn effective_delivery(&self) -> Delivery {
        if self.stall_degrades() && self.degraded.load(std::sync::atomic::Ordering::Relaxed) {
            return Delivery::Buffered;
        }
        self.config.delivery
    }

    /// Whether this client has switched itself to [`Delivery::Buffered`]
    /// because a stream stalled.
    ///
    /// Exposed so a host can say so on screen: an agent that answers instantly
    /// after a first slow step looks like a different agent, and the user
    /// deserves the one sentence that explains it.
    pub fn is_degraded(&self) -> bool {
        self.stall_degrades() && self.degraded.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Whether a stalled stream is allowed to change this client's mode.
    fn stall_degrades(&self) -> bool {
        self.config.degrade_on_stall && matches!(self.config.delivery, Delivery::Stream)
    }

    /// Switch back to the configured mode.
    pub fn reset_delivery(&self) {
        self.degraded
            .store(false, std::sync::atomic::Ordering::Relaxed);
    }

    /// A clone using a different model id.
    ///
    /// The HTTP client is shared, so switching models reuses the connection
    /// pool instead of reconnecting: a route change must not cost a handshake.
    /// A model switch keeps the delivery mode, because the mode belongs to the
    /// gateway rather than to the model behind it.
    pub fn with_model(&self, model: impl Into<String>) -> Self {
        let mut next = self.clone();
        next.config.model = model.into();
        next
    }

    /// Send a completion request and return its incremental events.
    ///
    /// Both delivery modes arrive here, and both yield the same event sequence,
    /// so a caller never branches on how the answer was carried. The request
    /// itself has already been dispatched, so an error here means the provider
    /// refused or the connection failed; an error on the receiver means the
    /// answer itself did not complete.
    pub async fn chat_stream(&self, messages: &[Message], tools: &[Value]) -> Result<mpsc::Receiver<Result<LlmEvent>>> {
        let body = self.build_body(messages, tools);
        // The mode is read back off the finished body rather than taken from the
        // config, because the shaper has the last word on it: an `extraBody`
        // that says `"stream": false` means a single JSON answer, and reading
        // that answer as SSE would lose every word of it without an error.
        if asks_for_stream(&body) {
            self.open_stream(body).await
        } else {
            self.open_buffered(body).await
        }
    }

    /// Send a streaming completion request.
    async fn open_stream(&self, body: Value) -> Result<mpsc::Receiver<Result<LlmEvent>>> {
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
            let code = status.as_u16();
            return Err(RequestError {
                status: Some(code),
                retryable: code == 429 || (500..=599).contains(&code),
                message: format!("provider returned {status}: {}", truncate(&text, 2000)),
            }
            .into());
        }

        let (tx, rx) = mpsc::channel::<Result<LlmEvent>>(256);
        let mut stream = response.bytes_stream();
        let idle = self.config.network.stream_idle();
        let degraded = Arc::clone(&self.degraded);

        tokio::spawn(async move {
            let mut buffer = String::new();
            let mut calls: Vec<PartialCall> = Vec::new();
            let mut finish_reason: Option<String> = None;
            // Whether the provider has delivered anything of the answer yet.
            // The auto-degrade decision turns on this: past the first fragment,
            // a second request would show the answer twice rather than finish it.
            let mut saw_output = false;

            loop {
                // Bounded wait for the next frame. An unbounded `next().await`
                // lets a provider that answers the request and then stops
                // writing hold the turn for the whole request timeout, which is
                // indistinguishable from a hung agent; this turns that into a
                // failed request that the step-level retry can see.
                let next = match tokio::time::timeout(idle, stream.next()).await {
                    Ok(next) => next,
                    Err(_) => {
                        // Nothing arrived, so nothing was lost. On a route where
                        // this is how every stream ends, asking again as a stream
                        // rediscovers the same fact one step at a time forever.
                        if !saw_output {
                            degraded.store(true, std::sync::atomic::Ordering::Relaxed);
                        }
                        let failure = RequestError {
                            status: None,
                            retryable: true,
                            message: format!("provider sent no stream frame within {}s", idle.as_secs()),
                        };
                        let _ = tx.send(Err(failure.into())).await;
                        return;
                    }
                };
                let Some(chunk) = next else { break };
                let chunk = match chunk {
                    Ok(c) => c,
                    Err(e) => {
                        // Transport loss mid-answer: the one failure the loop
                        // must retry, because output already shown can be
                        // rewound while an unfinishable turn cannot.
                        if !saw_output {
                            degraded.store(true, std::sync::atomic::Ordering::Relaxed);
                        }
                        let failure = RequestError {
                            status: None,
                            retryable: true,
                            message: format!("stream error: {e}"),
                        };
                        let _ = tx.send(Err(failure.into())).await;
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

                    // A provider that fails mid-answer reports it as an `error`
                    // object inside an otherwise normal frame. Reading only
                    // `choices` would drop that on the floor: the stream ends,
                    // the turn looks complete, and half an answer reads as a
                    // finished one.
                    // A provider that fails mid-answer reports it as an `error`
                    // object inside an otherwise normal frame. Reading only
                    // `choices` would drop that on the floor: the stream ends,
                    // the turn looks complete, and half an answer reads as a
                    // finished one.
                    if let Some(error) = parsed.get("error").filter(|value| !value.is_null()) {
                        let _ = tx.send(Err(error_request(error).into())).await;
                        return;
                    }

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
                            saw_output = true;
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
                                saw_output = true;
                                let _ = tx.send(Ok(LlmEvent::ReasoningDelta(text.to_string()))).await;
                            }
                        }
                    }
                    if let Some(parts) = delta.get("reasoning_details").and_then(|d| d.as_array()) {
                        for part in parts {
                            if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                                if !text.is_empty() {
                                    saw_output = true;
                                    let _ = tx.send(Ok(LlmEvent::ReasoningDelta(text.to_string()))).await;
                                }
                            }
                        }
                    }

                    if let Some(tool_calls) = delta.get("tool_calls").and_then(|t| t.as_array()) {
                        for call in tool_calls {
                            saw_output = true;
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

    /// Send one completion request and deliver the whole answer as events.
    ///
    /// The cost of this mode is that nothing can be shown while the model is
    /// writing: the events arrive together, once the provider has finished. On
    /// a gateway whose stream never completes, it is the price of a turn that
    /// ends at all. Tool calls come back in `message` rather than in fragments,
    /// so they are emitted as one complete call per position.
    async fn open_buffered(&self, body: Value) -> Result<mpsc::Receiver<Result<LlmEvent>>> {
        let url = format!("{}/chat/completions", self.config.base_url.trim_end_matches('/'));
        let response = self
            .send_retrying(|| {
                self.http
                    .post(&url)
                    .bearer_auth(&self.config.api_key)
                    .header("content-type", "application/json")
                    .header("accept", "application/json")
                    .json(&body)
            })
            .await
            .with_context(|| format!("POST {url}"))?;

        let status = response.status();
        // Read the body before branching on it: a gateway that answers 200 with
        // an `error` object in it is a failure, and dropping that on the floor
        // would hand the loop an empty turn that looks like a finished one.
        let text = response
            .text()
            .await
            .with_context(|| format!("reading the answer from {url}"))?;
        if !status.is_success() {
            let code = status.as_u16();
            return Err(RequestError {
                status: Some(code),
                retryable: code == 429 || (500..=599).contains(&code),
                message: format!("provider returned {status}: {}", truncate(&text, 2000)),
            }
            .into());
        }

        let parsed: Value =
            serde_json::from_str(&text).with_context(|| format!("parsing the answer from {url}"))?;
        if let Some(error) = parsed.get("error").filter(|value| !value.is_null()) {
            return Err(error_request(error).into());
        }

        let (tx, rx) = mpsc::channel::<Result<LlmEvent>>(256);
        for event in buffered_events(&parsed) {
            let _ = tx.send(Ok(event)).await;
        }
        // Dropping the sender ends the sequence: the loop sees the channel close
        // once every event has been taken, exactly as it does at `[DONE]`.
        drop(tx);
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
                    //
                    // TLS damage is: rustls reports it after a successful
                    // handshake, so `is_connect` misses it, yet it fires before
                    // a single response byte is read and the poisoned
                    // connection is dropped behind it.
                    if !error.is_connect()
                        && !is_connection_closed(&error)
                        && !is_tls_stream_damage(&error)
                    {
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

        // Every attempt failed at the connection level: retried here as far as
        // the settings allow, and classified retryable so a step-level retry
        // gives the whole send phase another run after a longer backoff.
        Err(RequestError {
            status: None,
            retryable: true,
            message: match &last {
                Some(error) => format!("connection failed after {attempts} attempts: {error}"),
                None => format!("request failed after {attempts} attempts"),
            },
        }
        .into())
    }

    /// Whether an error is the peer closing a connection mid-request.

    fn build_body(&self, messages: &[Message], tools: &[Value]) -> Value {
        let wire: Vec<Value> = messages.iter().map(to_wire).collect();
        let mut body = json!({
            "model": self.config.model,
            "messages": wire,
            // Sent rather than omitted, and sent as the mode the client will
            // actually use: a provider that defaults to streaming would
            // otherwise hand a buffered client an SSE body it cannot parse.
            "stream": matches!(self.effective_delivery(), Delivery::Stream),
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
            .send_retrying(|| {
                self.http
                    .get(&url)
                    .bearer_auth(&self.config.api_key)
                    .timeout(self.config.network.catalog_timeout())
            })
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
            .send_retrying(|| {
                self.http
                    .get(&url)
                    .bearer_auth(&self.config.api_key)
                    .timeout(self.config.network.catalog_timeout())
            })
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

/// Whether the finished body asks for a stream.
///
/// Absent or `null` means it does not: the OpenAI default for `stream` is
/// `false`, so a shaper that removes the field is asking for a JSON answer and
/// must be read as one.
fn asks_for_stream(body: &Value) -> bool {
    body.get("stream").and_then(Value::as_bool).unwrap_or(false)
}

/// The events a single JSON answer carries, in the order a stream would have.
///
/// Reasoning before text, then the calls, then usage, then the close: the loop
/// assembles an assistant turn from these, and an answer that reads differently
/// because it arrived whole instead of in fragments would put the delivery mode
/// in the conversation.
fn buffered_events(answer: &Value) -> Vec<LlmEvent> {
    let mut events = Vec::new();
    let choice = answer.get("choices").and_then(|c| c.get(0));

    if let Some(message) = choice.and_then(|c| c.get("message")) {
        // Same three spellings of the reasoning channel as the stream reader.
        for key in ["reasoning_content", "reasoning"] {
            if let Some(text) = message.get(key).and_then(|t| t.as_str()) {
                if !text.is_empty() {
                    events.push(LlmEvent::ReasoningDelta(text.to_string()));
                }
            }
        }
        if let Some(parts) = message.get("reasoning_details").and_then(|d| d.as_array()) {
            for part in parts {
                if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                    if !text.is_empty() {
                        events.push(LlmEvent::ReasoningDelta(text.to_string()));
                    }
                }
            }
        }
        if let Some(text) = message.get("content").and_then(content_text) {
            if !text.is_empty() {
                events.push(LlmEvent::TextDelta(text));
            }
        }
        if let Some(calls) = message.get("tool_calls").and_then(|t| t.as_array()) {
            for (index, call) in calls.iter().enumerate() {
                let function = call.get("function");
                events.push(LlmEvent::ToolCallDelta {
                    index,
                    id: call.get("id").and_then(|i| i.as_str()).map(str::to_string),
                    name: function.and_then(|f| f.get("name")).and_then(|n| n.as_str()).map(str::to_string),
                    // A provider that hands back arguments as an object rather
                    // than as a JSON string still means the same call, and the
                    // fragment reader concatenates strings anyway.
                    arguments: function
                        .and_then(|f| f.get("arguments"))
                        .map(|a| match a {
                            Value::String(text) => text.clone(),
                            other => other.to_string(),
                        }),
                });
            }
        }
    }

    if let Some(usage) = answer.get("usage").filter(|u| !u.is_null()) {
        events.push(LlmEvent::Usage(extract_usage(usage)));
    }

    let reason = choice
        .and_then(|c| c.get("finish_reason"))
        .and_then(|r| r.as_str())
        .map(str::to_string);
    events.push(LlmEvent::Finished { reason });
    events
}

/// The text of a `content` field, which arrives as a string or as a part list.
fn content_text(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::Array(parts) => {
            let mut out = String::new();
            for part in parts {
                if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                    out.push_str(text);
                }
            }
            Some(out)
        }
        _ => None,
    }
}

/// A failure a provider reported inside an otherwise successful answer.
///
/// Shared by both modes: a stream carries it in a frame, a buffered answer
/// carries it as the whole body, and the retry decision cannot be allowed to
/// depend on which one the route happens to use.
fn error_request(error: &Value) -> RequestError {
    let message = error
        .get("message")
        .and_then(|m| m.as_str())
        .unwrap_or("provider reported an error mid-stream")
        .to_string();
    let status = error
        .get("code")
        .or_else(|| error.get("status"))
        .and_then(|code| code.as_u64())
        .map(|code| code as u16);
    let retryable = match status {
        Some(code) => code == 429 || (500..=599).contains(&code),
        None => !looks_fatal(&message),
    };
    RequestError {
        status,
        retryable,
        message,
    }
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
    fn the_default_route_asks_for_a_stream() {
        let client = client_with(None, serde_json::Map::new());
        let body = client.preview_body(&[Message::user("hi")], &[]);
        assert_eq!(body["stream"], true, "{body}");
    }

    #[test]
    fn a_buffered_route_asks_for_one_json_body() {
        // The word `stream` rather than omitting it: a provider that defaults to
        // streaming would otherwise hand this client an SSE body to parse.
        let mut config = LlmConfig::new("http://localhost:1/v1", "k", "test-model");
        config.delivery = Delivery::Buffered;
        let client = LlmClient::new(config).unwrap();
        let body = client.preview_body(&[Message::user("hi")], &[]);
        assert_eq!(body["stream"], false, "{body}");
    }

    #[test]
    fn an_extra_body_can_turn_the_stream_off() {
        // The setting is a route-level one, but `extraBody` still has the last
        // word — and the reader has to follow it, which is what
        // `asks_for_stream` is for.
        let mut extra = serde_json::Map::new();
        extra.insert("stream".into(), json!(false));
        let client = client_with(None, extra);
        let body = client.preview_body(&[Message::user("hi")], &[]);
        assert!(!asks_for_stream(&body), "the shaped body decides: {body}");
    }

    #[test]
    fn the_stream_flag_and_model_are_always_present() {
        let client = client_with(None, serde_json::Map::new());
        let body = client.preview_body(&[Message::user("hi")], &[]);
        assert_eq!(body["stream"], true);
        assert_eq!(body["model"], "test-model");
    }
}

/// Whether a provider error message describes a failure no retry can fix.
///
/// Used when the provider reports an error without a numeric status: retrying
/// a request the API rejected on its merits only repeats the rejection.
fn looks_fatal(message: &str) -> bool {
    let lower = message.to_lowercase();
    [
        "context_length",
        "context length",
        "invalid_request",
        "invalid request",
        "unauthorized",
        "authentication",
        "not_found",
        "insufficient_quota",
        "quota exceeded",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
}

/// Whether an error is the peer closing a connection mid-request.
///
/// reqwest reports this as a body or request error rather than a connect error,
/// because the connection was established and then went away. The distinction
/// matters: it is the one failure that is certainly safe to retry.
fn is_connection_closed(error: &reqwest::Error) -> bool {
    error_chain_matches(error, matches_closed_text)
}

/// Whether TLS records arrived that this session's keys cannot open.
///
/// rustls raises these after a completed handshake, so reqwest reports a request
/// failure rather than a connect error. Nothing of the response has been read
/// when they surface and the connection is unusable behind them, which puts the
/// next attempt on a fresh session: retryable on the same terms as
/// [`is_connection_closed`].
fn is_tls_stream_damage(error: &reqwest::Error) -> bool {
    error_chain_matches(error, matches_tls_damage_text)
}

/// Test `pred` against every link of a reqwest error's cause chain.
///
/// The text that identifies these failures is below the outermost error:
/// reqwest reports "error sending request for url (...)" and the actionable
/// words — "connection closed before message completed", "cannot decrypt peer's
/// message" — arrive one or more levels down, so testing only
/// `error.to_string()` classifies every transport loss as unretryable.
fn error_chain_matches(error: &reqwest::Error, pred: impl Fn(&str) -> bool) -> bool {
    let mut current: Option<&(dyn std::error::Error + 'static)> = Some(error);
    while let Some(link) = current {
        if pred(&link.to_string()) {
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

/// The message test behind [`is_tls_stream_damage`].
///
/// These are rustls's own words for a record whose integrity check fails or
/// whose declared length does not describe a record. They come from an endpoint
/// that tears a session down mid-stream, or from an intermediary that writes
/// another session's bytes onto this one — the two ways a proxy chain in front
/// of a model API breaks TLS without breaking TCP.
///
/// Matching is case-insensitive because the same failure reaches the chain as
/// both a rustls `Display` string and a lower-cased `source` sentence.
fn matches_tls_damage_text(text: &str) -> bool {
    const DAMAGED: [&str; 7] = [
        "cannot decrypt peer's message",
        "couldn't decrypt",
        "decrypt failed",
        "bad record mac",
        "corrupt message",
        "record overflow",
        "illegal packet length",
    ];
    let lower = text.to_lowercase();
    DAMAGED.iter().any(|needle| lower.contains(needle))
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
        //
        // `no_proxy`, or the machine's `HTTP_PROXY` would route this loopback
        // address somewhere that answers — turning a refused connection into a
        // successful response and making the test depend on the environment.
        let client = reqwest::Client::builder().no_proxy().build().unwrap();
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
    fn a_tls_record_the_session_cannot_open_is_recognised_as_damage() {
        // The sentences rustls puts on the chain when bytes arrive that its
        // keys cannot open, including the one a proxy chain in front of the
        // provider produces when it tears a session down mid-stream.
        for text in [
            "connection error: cannot decrypt peer's message",
            "received corrupt message",
            "Failed to create packet: record overflow",
            "illegal packet length",
            "couldn't decrypt message",
        ] {
            assert!(
                matches_tls_damage_text(text),
                "{text:?} should be treated as a lost connection"
            );
        }
    }

    #[test]
    fn a_request_the_provider_rejected_is_not_tls_damage() {
        // Retrying these repeats the rejection. A bare "decrypt" appearing in
        // unrelated prose must not pull them into the retryable set, which is
        // why the needles above are whole rustls sentences.
        for text in [
            "provider returned 400 Bad Request",
            "invalid api key",
            "operation timed out",
            "unable to decrypt the local session store",
            "connection closed before message completed",
        ] {
            assert!(
                !matches_tls_damage_text(text),
                "{text:?} must not be classified as TLS damage"
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

    #[test]
    fn a_retry_classification_survives_added_context() {
        // chat_stream labels its failures with the URL it posted to; the loop
        // must still read the classification the failure carries underneath.
        let busy = RequestError {
            status: Some(503),
            retryable: true,
            message: "overloaded".into(),
        };
        let error =
            Result::<(), RequestError>::Err(busy)
                .context("POST http://x/chat/completions")
                .unwrap_err();
        assert!(retryable(&error));

        let rejected = RequestError {
            status: Some(401),
            retryable: false,
            message: "bad key".into(),
        };
        let error =
            Result::<(), RequestError>::Err(rejected)
                .context("POST http://x/chat/completions")
                .unwrap_err();
        assert!(!retryable(&error));
    }

    #[test]
    fn an_unclassified_error_is_not_retried() {
        // Unknown failures are not given the benefit of the doubt: a retry
        // loop that invents retryable failures turns a bug into a hang.
        let error = anyhow::anyhow!("unexpected failure");
        assert!(!retryable(&error));
    }

    #[test]
    fn a_fatal_provider_message_ends_the_step_instead_of_retrying() {
        // Mid-stream provider errors often arrive without a numeric status;
        // the message is the only thing left to classify on.
        assert!(looks_fatal("context_length_exceeded for this model"));
        assert!(looks_fatal("invalid request: tools schema"));
        assert!(looks_fatal("unauthorized"));
        assert!(looks_fatal("insufficient_quota"));
        assert!(!looks_fatal("The model is overloaded, please try again"));
        assert!(!looks_fatal("rate_limit_exceeded"));
    }
}

#[cfg(test)]
mod delivery_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn the_two_modes_are_named_and_parsed() {
        assert_eq!(Delivery::default(), Delivery::Stream);
        for word in ["stream", "Stream", " sse ", "true"] {
            assert_eq!(Delivery::parse(word), Some(Delivery::Stream), "{word}");
        }
        for word in ["buffered", "BUFFERED", "json", "false"] {
            assert_eq!(Delivery::parse(word), Some(Delivery::Buffered), "{word}");
        }
        assert_eq!(Delivery::parse("streaming"), None);
        assert_eq!(Delivery::parse(""), None, "empty is 'not set', not a mode");
        assert_eq!(Delivery::Buffered.as_str(), "buffered");
    }

    /// The body one `chat/completions` answer returns, in the shape a real
    /// gateway sends it: content and calls together rather than in fragments.
    fn answer() -> Value {
        json!({
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "I'll run that now.",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": { "name": "bash", "arguments": "{\"command\":\"ipconfig\"}" }
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": { "prompt_tokens": 1200, "completion_tokens": 30, "total_tokens": 1230 }
        })
    }

    #[test]
    fn a_whole_answer_becomes_the_events_a_stream_would_have_sent() {
        let events = buffered_events(&answer());
        let text: String = events
            .iter()
            .filter_map(|e| match e {
                LlmEvent::TextDelta(t) => Some(t.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(text, "I'll run that now.");

        let calls: Vec<&LlmEvent> = events
            .iter()
            .filter(|e| matches!(e, LlmEvent::ToolCallDelta { .. }))
            .collect();
        assert_eq!(calls.len(), 1, "one complete call, not fragments");
        match calls[0] {
            LlmEvent::ToolCallDelta { index, id, name, arguments } => {
                assert_eq!(*index, 0);
                assert_eq!(id.as_deref(), Some("call_1"));
                assert_eq!(name.as_deref(), Some("bash"));
                assert_eq!(arguments.as_deref(), Some("{\"command\":\"ipconfig\"}"));
            }
            _ => unreachable!(),
        }

        let usage = events.iter().find_map(|e| match e {
            LlmEvent::Usage(u) => Some(u.clone()),
            _ => None,
        });
        assert_eq!(usage.unwrap().input, 1200);

        let reason = events.iter().find_map(|e| match e {
            LlmEvent::Finished { reason } => reason.clone(),
            _ => None,
        });
        assert_eq!(reason.as_deref(), Some("tool_calls"));
    }

    #[test]
    fn a_part_list_content_is_joined() {
        let answer = json!({
            "choices": [{ "message": { "content": [
                { "type": "text", "text": "one " },
                { "type": "text", "text": "two" }
            ] }, "finish_reason": "stop" }]
        });
        let text: String = buffered_events(&answer)
            .iter()
            .filter_map(|e| match e {
                LlmEvent::TextDelta(t) => Some(t.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(text, "one two");
    }

    #[test]
    fn tool_arguments_sent_as_an_object_are_still_one_call() {
        // A gateway that hands the arguments back as a JSON object rather than a
        // string means the same call; the reader concatenates strings anyway.
        let answer = json!({
            "choices": [{ "message": { "tool_calls": [
                { "id": "c", "function": { "name": "n", "arguments": { "a": 1 } } }
            ] }, "finish_reason": "tool_calls" }]
        });
        let events = buffered_events(&answer);
        match events.iter().find(|e| matches!(e, LlmEvent::ToolCallDelta { .. })) {
            Some(LlmEvent::ToolCallDelta { arguments, name, .. }) => {
                assert_eq!(name.as_deref(), Some("n"));
                let parsed: Value = serde_json::from_str(arguments.as_deref().unwrap())
                    .expect("the arguments are valid JSON");
                assert_eq!(parsed["a"], 1);
            }
            other => panic!("no tool call: {other:?}"),
        }
    }

    #[test]
    fn a_missing_answer_still_closes_the_stream() {
        // An empty `choices` is a strange provider, but the loop must still see
        // the end of the turn rather than wait for an event that never comes.
        let events = buffered_events(&json!({ "choices": [] }));
        assert!(matches!(
            events.last(),
            Some(LlmEvent::Finished { reason: None })
        ));
    }

    #[test]
    fn an_error_inside_a_two_hundred_body_is_classified_like_a_stream_one() {
        // Both modes carry the same object, so both must reach the same verdict.
        let busy = error_request(&json!({ "message": "overloaded", "code": 503 }));
        assert_eq!(busy.status, Some(503));
        assert!(busy.retryable);

        // "invalid api key" alone is not one of the fatal phrases: an endpoint
        // that rejects on its merits says so in words the list carries.
        let rejected = error_request(&json!({ "message": "unauthorized: invalid api key" }));
        assert!(!rejected.retryable);
    }

    #[tokio::test]
    async fn a_stream_that_stalls_with_nothing_delivered_degrades_the_route() {
        // The shape measured against a real gateway: 200, `text/event-stream`,
        // one empty frame, then silence until the connection dies. A client that
        // only ever asks again as a stream spends the stall bound on every step
        // of every turn and never gets an answer.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        tokio::spawn(async move {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            // One task per connection: the stalled branch holds its socket open
            // on purpose, and holding the accept loop with it would make the
            // follow-up request time out for a reason that has nothing to do
            // with what is under test.
            while let Ok((mut socket, _)) = listener.accept().await {
                tokio::spawn(async move {
                    let mut collected: Vec<u8> = Vec::new();
                    let mut chunk = [0u8; 8192];
                    // Read until the body has arrived: a request split across two
                    // writes would otherwise be judged on its headers alone.
                    for _ in 0..8 {
                        match tokio::time::timeout(
                            std::time::Duration::from_millis(500),
                            socket.read(&mut chunk),
                        )
                        .await
                        {
                            Ok(Ok(0)) | Err(_) => break,
                            Ok(Ok(n)) => {
                                collected.extend_from_slice(&chunk[..n]);
                                if let Ok(text) = std::str::from_utf8(&collected) {
                                    if text.contains("\"stream\":") {
                                        break;
                                    }
                                }
                            }
                            Ok(Err(_)) => break,
                        }
                    }
                    let text = String::from_utf8_lossy(&collected).to_string();
                    if text.contains("\"stream\":true") {
                        // Headers, one empty frame, then nothing for the rest of the
                        // request timeout.
                        let _ = socket
                            .write_all(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n\r\n")
                            .await;
                        let _ = socket
                            .write_all(b"data: {\"choices\":[{\"delta\":{\"content\":\"\"}}]}\n\n")
                            .await;
                        let _ = socket.flush().await;
                        tokio::time::sleep(std::time::Duration::from_secs(30)).await;
                        return;
                    }
                    let body = "{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"buffered answer\"},\"finish_reason\":\"stop\"}]}";
                    let response = format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{}",
                        body.len(),
                        body
                    );
                    let _ = socket.write_all(response.as_bytes()).await;
                    let _ = socket.flush().await;
                    tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                });
            }
        });

        let mut config = LlmConfig::new(format!("http://{addr}"), "k", "m");
        config.network.stream_idle_secs = 1;
        config.network.request_timeout_secs = 20;
        let client = LlmClient::new(config).unwrap();

        // First step: the stream stalls, and the failure reaches the caller.
        let mut rx = client.chat_stream(&[Message::user("go")], &[]).await.unwrap();
        let mut saw_failure = false;
        while let Some(event) = rx.recv().await {
            if event.is_err() {
                saw_failure = true;
                break;
            }
        }
        assert!(saw_failure, "a stalled stream must not look like a finished turn");
        assert!(client.is_degraded(), "nothing was delivered, so the mode may change");
        assert_eq!(client.effective_delivery(), Delivery::Buffered);

        // Second step: asked as one body, and answered.
        let mut rx = client.chat_stream(&[Message::user("go")], &[]).await.unwrap();
        let mut text = String::new();
        let mut reason: Option<String> = None;
        while let Some(event) = rx.recv().await {
            match event.expect("the buffered answer completes") {
                LlmEvent::TextDelta(delta) => text.push_str(&delta),
                LlmEvent::Finished { reason: r } => reason = r,
                _ => {}
            }
        }
        assert_eq!(text, "buffered answer");
        assert_eq!(reason.as_deref(), Some("stop"));

        // The mode can be re-armed: a route change or a deliberate retry asks the
        // stream again rather than being stuck buffered forever.
        client.reset_delivery();
        assert!(!client.is_degraded());
    }

    #[tokio::test]
    async fn a_degraded_client_sends_no_stream_request() {
        // The body on the wire is what proves the mode reached it: a client that
        // reads a JSON answer while still asking for a stream would hang the same
        // way the original bug did.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let seen = Arc::new(std::sync::Mutex::new(String::new()));
        let recorder = Arc::clone(&seen);

        tokio::spawn(async move {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            while let Ok((mut socket, _)) = listener.accept().await {
                let mut collected = Vec::new();
                let mut chunk = [0u8; 4096];
                for _ in 0..8 {
                    match socket.read(&mut chunk).await {
                        Ok(0) | Err(_) => break,
                        Ok(n) => {
                            collected.extend_from_slice(&chunk[..n]);
                            let text = String::from_utf8_lossy(&collected);
                            if text.contains("\"stream\":") {
                                break;
                            }
                        }
                    }
                }
                let _ = recorder
                    .lock()
                    .unwrap()
                    .push_str(&String::from_utf8_lossy(&collected));
                let body = "{\"choices\":[{\"message\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}";
                let response = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{}",
                    body.len(),
                    body
                );
                let _ = socket.write_all(response.as_bytes()).await;
            }
        });

        let mut config = LlmConfig::new(format!("http://{addr}"), "k", "m");
        config.delivery = Delivery::Buffered;
        let client = LlmClient::new(config).unwrap();
        let mut rx = client.chat_stream(&[Message::user("go")], &[]).await.unwrap();
        while rx.recv().await.is_some() {}

        let sent = seen.lock().unwrap().clone();
        assert!(sent.contains("\"stream\":false"), "the wire asked for a body: {sent}");
        assert!(!sent.contains("text/event-stream"), "and not an SSE response: {sent}");
    }
}
