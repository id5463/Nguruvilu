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

/// Ceiling on the wall-clock time of one streamed answer.
///
/// [`crate::network::NetworkSettings::stream_idle`] bounds only the *gap*
/// between frames, so a gateway that dribbles one fragment just inside that gap
/// holds a turn open indefinitely — every step, for as long as the user waits.
/// Thirty minutes is generous for one answer and finite.
const STREAM_TOTAL_BUDGET_SECS: u64 = 30 * 60;

/// The total budget for one streamed answer, from `NGU_STREAM_MAX_SECS`.
///
/// Zero and unparsable values fall back to the default rather than to "no
/// budget": a variable set to 0 must not mean "hang forever".
fn stream_total_budget() -> std::time::Duration {
    let secs = std::env::var("NGU_STREAM_MAX_SECS")
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .filter(|secs| *secs > 0)
        .unwrap_or(STREAM_TOTAL_BUDGET_SECS);
    std::time::Duration::from_secs(secs)
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
    /// Prompt tokens written into the provider's cache by this turn.
    ///
    /// Anthropic-shaped gateways bill a cache write above the normal prompt
    /// price, so a turn that paid for one and records nothing understates its
    /// own cost by the largest line in it. This is the mirror image of `cached`,
    /// which counts tokens *read* from the cache rather than written to it.
    #[serde(default)]
    pub cache_created: u64,
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
    /// Total time one streamed answer may take, whatever arrives in between.
    ///
    /// A field rather than a constant read inside the reader task so a test can
    /// spend 400 ms instead of half an hour; [`LlmClient::new`] fills it from
    /// [`stream_total_budget`].
    stream_budget: std::time::Duration,
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

        // No proxy configured: do not pick one up from the environment.
        builder = match proxy_rules(&config.proxy)? {
            Some(proxy) => builder.proxy(proxy),
            None => builder.no_proxy(),
        };

        let http = builder.build().context("building HTTP client")?;
        Ok(Self {
            http,
            config,
            shaper: Arc::new(crate::request::Passthrough),
            degraded: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            stream_budget: stream_total_budget(),
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

    /// Replace the total budget for one streamed answer.
    ///
    /// Exists for tests, which cannot wait thirty minutes to prove the bound
    /// fires; production takes it from [`stream_total_budget`].
    pub fn with_stream_budget(mut self, budget: std::time::Duration) -> Self {
        self.stream_budget = budget;
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
                retryable: retryable_status(code),
                message: format!("provider returned {status}: {}", truncate(&text, 2000)),
            }
            .into());
        }

        let (tx, rx) = mpsc::channel::<Result<LlmEvent>>(256);
        let mut stream = response.bytes_stream();
        let idle = self.config.network.stream_idle();
        let budget = self.stream_budget;
        let degraded = Arc::clone(&self.degraded);

        tokio::spawn(async move {
            let mut decoder = SseDecoder::new();
            let mut calls: Vec<PartialCall> = Vec::new();
            let mut finish_reason: Option<String> = None;
            // Whether the provider has delivered anything of the answer yet.
            // The auto-degrade decision turns on this: past the first fragment,
            // a second request would show the answer twice rather than finish it.
            let mut saw_output = false;
            // Where the total budget runs out. A deadline rather than a running
            // clock, so nothing has to be updated per frame.
            let deadline = tokio::time::Instant::now() + budget;

            loop {
                // Bounded wait for the next frame. An unbounded `next().await`
                // lets a provider that answers the request and then stops
                // writing hold the turn for the whole request timeout, which is
                // indistinguishable from a hung agent; this turns that into a
                // failed request that the step-level retry can see.
                //
                // The bound for this one wait is the sooner of the idle gap and
                // whatever is left of the total budget: the idle bound alone
                // never fires on a stream that keeps arriving just inside it.
                let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
                let next = match tokio::time::timeout(idle.min(remaining), stream.next()).await {
                    Ok(next) => next,
                    Err(_) => {
                        // Past the deadline it was the total budget that fired,
                        // not a gap in the output. End the answer where it
                        // stands: the deltas already sent are real output the
                        // caller has been shown, and a second request would show
                        // them twice — so a truncated answer with a reason beats
                        // a turn that never ends.
                        if tokio::time::Instant::now() >= deadline {
                            if !saw_output {
                                degraded.store(true, std::sync::atomic::Ordering::Relaxed);
                                let failure = RequestError {
                                    status: None,
                                    retryable: true,
                                    message: format!(
                                        "stream sent nothing within its {}s total budget",
                                        budget.as_secs()
                                    ),
                                };
                                let _ = tx.send(Err(failure.into())).await;
                            } else {
                                let _ = tx
                                    .send(Ok(LlmEvent::Finished {
                                        reason: Some(format!(
                                            "stream cut off at its {}s total budget",
                                            budget.as_secs()
                                        )),
                                    }))
                                    .await;
                            }
                            return;
                        }
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
                // A line is decoded only once all of its bytes have arrived;
                // see `SseDecoder` for why decoding per network chunk is not
                // the same thing.
                let lines = match decoder.push(&chunk) {
                    Ok(lines) => lines,
                    Err(overflow) => {
                        // A frame that never ends is not an answer, and asking
                        // again would receive the same bytes: this one is final.
                        let failure = RequestError {
                            status: None,
                            retryable: false,
                            message: overflow,
                        };
                        let _ = tx.send(Err(failure.into())).await;
                        return;
                    }
                };

                for line in lines {
                    match handle_line(&line, &tx, &mut calls, &mut finish_reason, &mut saw_output).await {
                        // `[DONE]` was read: `Finished` has been sent.
                        Step::End | Step::Stop => return,
                        Step::Continue => {}
                    }
                }
            }

            // The stream ended without a newline after its last frame. That
            // frame is still a frame — `None` from the stream means the bytes
            // are over, not that there was nothing left — and dropping it loses
            // the last tool-call fragment, the usage report, and the finish
            // reason of an otherwise complete answer.
            if let Some(line) = decoder.finish() {
                if handle_line(&line, &tx, &mut calls, &mut finish_reason, &mut saw_output).await
                    != Step::Continue
                {
                    return;
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
                retryable: retryable_status(code),
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
            // A closed receiver means the turn was abandoned: there is nobody
            // left to deliver the rest of the answer to.
            if !deliver(&tx, Ok(event)).await {
                break;
            }
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
        // The scheme is checked here as well as in `crate::settings`, because a
        // route does not only come from the settings file: `NGU_BASE_URL`, a
        // pack-injected route, and the `set_model` command all reach this client
        // without passing through that write. Plain `http://` to anything but
        // loopback puts the bearer token on the wire for every hop to read. The
        // error names the endpoint and never the key.
        crate::model::check_base_url(&self.config.base_url).with_context(|| {
            format!(
                "refusing to send a request to '{}'",
                self.config.base_url.trim()
            )
        })?;

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

    /// The catalog as `(id, the context window the provider reports)` pairs.
    ///
    /// Read through [`serde_json::Value`] rather than through [`ModelInfo`] on
    /// purpose: the id list is what every command needs, and one unrecognised
    /// field — a gateway that reports a string where the struct wants an array —
    /// must not cost the caller the whole catalog. An entry without an id is
    /// dropped; nothing else about an entry is read except the window.
    ///
    /// The window is only believed when it is plausible for a context window
    /// (see [`crate::window::MIN_PLAUSIBLE_WINDOW`]); a provider that reports
    /// `0` or a few hundred is describing something else, and a wrong small
    /// number would make the loop compact constantly.
    pub async fn list_models_with_windows(&self) -> Result<Vec<(String, Option<usize>)>> {
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
        let mut models: Vec<(String, Option<usize>)> = parsed
            .get("data")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(|item| {
                        let id = item.get("id").and_then(Value::as_str)?.trim();
                        if id.is_empty() {
                            return None;
                        }
                        // Three spellings are in the wild, and a provider that
                        // quotes the number as a string is as easy to read as one
                        // that does not.
                        let window = ["context_length", "context_window", "max_model_len"]
                            .iter()
                            .find_map(|key| item.get(*key).and_then(plausible_window));
                        Some((id.to_string(), window))
                    })
                    .collect()
            })
            .unwrap_or_default();
        models.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(models)
    }
}

/// A token count a provider wrote as a number or a string, when it is plausible.
///
/// Anything else — an object, a negative, a number far outside what a context
/// window can be — reads as "not reported" rather than as an error: the catalog
/// is still useful without it.
fn plausible_window(value: &Value) -> Option<usize> {
    let tokens = match value {
        Value::Number(number) => number.as_u64().map(|n| n as usize)?,
        Value::String(text) => text.trim().replace(['_', ','], "").parse::<usize>().ok()?,
        _ => return None,
    };
    (crate::window::MIN_PLAUSIBLE_WINDOW..=crate::window::MAX_PLAUSIBLE_WINDOW)
        .contains(&tokens)
        .then_some(tokens)
}

/// The largest partial frame the reader will hold before calling the stream
/// broken.
///
/// A frame is a small JSON object — the largest one any provider sends is a
/// tool-call fragment of a few kilobytes. Eight mebibytes is orders of
/// magnitude above that and orders of magnitude below what it takes to stop
/// this process: without a ceiling, an endpoint that writes bytes forever
/// without a newline grows the buffer until the machine refuses.
const MAX_FRAME_BYTES: usize = 8 * 1024 * 1024;

/// What the reader should do after one line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Step {
    /// Keep reading frames.
    Continue,
    /// The sequence is complete: `Finished` has been sent.
    End,
    /// Stop reading and send nothing more.
    Stop,
}

/// Split a byte stream into complete SSE lines.
///
/// Two properties matter here, and neither is free:
///
/// * A line is decoded only when all of its bytes have arrived. Decoding each
///   network chunk on its own replaces a UTF-8 character that straddles two
///   chunks with U+FFFD and reports nothing, so a Chinese or emoji answer has a
///   silent chance of arriving mangled — and the mangling is then written to the
///   session file as the answer.
/// * The last line of a stream is not newline-terminated. `next()` returning
///   `None` says the bytes are over, not that the buffer was empty, so whatever
///   is left has to be read as a line rather than dropped.
///
/// It also bounds what it holds: a peer that never sends a newline must not be
/// able to grow the buffer without limit.
struct SseDecoder {
    /// Bytes of a line that has not been terminated yet.
    pending: Vec<u8>,
}

impl SseDecoder {
    fn new() -> Self {
        Self { pending: Vec::new() }
    }

    /// The complete lines in `chunk`, keeping any unterminated tail for later.
    ///
    /// `Err` means the unterminated tail has outgrown [`MAX_FRAME_BYTES`]; the
    /// message says so in the words a user needs to understand the stream was
    /// abandoned rather than finished.
    fn push(&mut self, chunk: &[u8]) -> std::result::Result<Vec<String>, String> {
        self.pending.extend_from_slice(chunk);

        let mut lines = Vec::new();
        let mut start = 0usize;
        while let Some(at) = self.pending[start..].iter().position(|byte| *byte == b'\n') {
            let end = start + at;
            lines.push(decode_line(&self.pending[start..end]));
            start = end + 1;
        }
        if start > 0 {
            self.pending.drain(..start);
        }

        if self.pending.len() > MAX_FRAME_BYTES {
            return Err(format!(
                "the provider sent more than {} MiB without ending a stream frame; \
                 the stream is abandoned rather than buffered",
                MAX_FRAME_BYTES / (1024 * 1024)
            ));
        }
        Ok(lines)
    }

    /// The last line of a stream that ended without a newline after it.
    fn finish(&mut self) -> Option<String> {
        if self.pending.is_empty() {
            return None;
        }
        let line = decode_line(&self.pending);
        self.pending.clear();
        Some(line)
    }
}

/// Decode one frame line, dropping the carriage return of a CRLF ending.
///
/// Lossy is right here and only here: the bytes of one line may still contain
/// invalid UTF-8, and a frame that cannot be decoded is a frame to skip rather
/// than a stream to fail. The point is that a *valid* character is never split
/// across the decode.
fn decode_line(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).trim_end_matches('\r').to_string()
}

/// Send one event, reporting whether anyone is still listening.
///
/// A closed receiver means the turn was abandoned — the user cancelled, or the
/// loop moved on. Ignoring that and reading on is how a cancelled turn keeps
/// pulling frames and paying for tokens nobody will see.
async fn deliver(tx: &mpsc::Sender<Result<LlmEvent>>, event: Result<LlmEvent>) -> bool {
    tx.send(event).await.is_ok()
}

/// Read one complete SSE line and deliver what it carries.
///
/// The parsing lives here rather than in the read loop because the final line
/// arrives from [`SseDecoder::finish`] instead of from the chunk loop: both
/// paths must read a line the same way, or the last frame of every answer would
/// be read by a second, weaker parser.
async fn handle_line(
    line: &str,
    tx: &mpsc::Sender<Result<LlmEvent>>,
    calls: &mut Vec<PartialCall>,
    finish_reason: &mut Option<String>,
    saw_output: &mut bool,
) -> Step {
    let Some(data) = line.strip_prefix("data:") else { return Step::Continue };
    let data = data.trim();
    if data.is_empty() {
        return Step::Continue;
    }
    if data == "[DONE]" {
        // The terminal event of the sequence: whether or not anyone is still
        // listening, this task has nothing left to send.
        let _ = deliver(tx, Ok(LlmEvent::Finished { reason: finish_reason.clone() })).await;
        return Step::End;
    }

    let parsed: Value = match serde_json::from_str(data) {
        Ok(v) => v,
        Err(_) => return Step::Continue,
    };

    // A provider that fails mid-answer reports it as an `error` object inside an
    // otherwise normal frame. Reading only `choices` would drop that on the
    // floor: the stream ends, the turn looks complete, and half an answer reads
    // as a finished one.
    if let Some(error) = parsed.get("error").filter(|value| !value.is_null()) {
        let _ = deliver(tx, Err(error_request(error).into())).await;
        return Step::Stop;
    }

    if let Some(usage) = parsed.get("usage").filter(|u| !u.is_null()) {
        if !deliver(tx, Ok(LlmEvent::Usage(extract_usage(usage)))).await {
            return Step::Stop;
        }
    }

    let Some(choice) = parsed.get("choices").and_then(|c| c.get(0)) else { return Step::Continue };

    if let Some(reason) = choice.get("finish_reason").and_then(|r| r.as_str()) {
        *finish_reason = Some(reason.to_string());
    }

    let Some(delta) = choice.get("delta") else { return Step::Continue };

    if let Some(text) = delta.get("content").and_then(|c| c.as_str()) {
        if !text.is_empty() {
            *saw_output = true;
            if !deliver(tx, Ok(LlmEvent::TextDelta(text.to_string()))).await {
                return Step::Stop;
            }
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
                *saw_output = true;
                if !deliver(tx, Ok(LlmEvent::ReasoningDelta(text.to_string()))).await {
                    return Step::Stop;
                }
            }
        }
    }
    if let Some(parts) = delta.get("reasoning_details").and_then(|d| d.as_array()) {
        for part in parts {
            if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                if !text.is_empty() {
                    *saw_output = true;
                    if !deliver(tx, Ok(LlmEvent::ReasoningDelta(text.to_string()))).await {
                        return Step::Stop;
                    }
                }
            }
        }
    }

    if let Some(tool_calls) = delta.get("tool_calls").and_then(|t| t.as_array()) {
        for call in tool_calls {
            *saw_output = true;
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

            if !deliver(tx, Ok(LlmEvent::ToolCallDelta { index, id, name, arguments: args })).await {
                return Step::Stop;
            }
        }
    }

    Step::Continue
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

/// The proxy rules both HTTP clients in this program build from.
///
/// The two clients take their proxy from different configuration channels on
/// purpose — the model route's `proxy` setting, and `NGU_PROXY` for pack
/// downloads — but they must not *behave* differently, and they did: neither
/// read `NO_PROXY`, so a host the rest of the machine sends direct was still
/// sent through the proxy here, and the exclusion list everyone else honours
/// was silently ignored around credentials.
///
/// One function now answers for both: an empty configuration means a direct
/// connection (the environment's `HTTP(S)_PROXY` is never inherited), a
/// configured one is used for every request except the hosts `NO_PROXY` names,
/// matched by suffix the way every other tool means it.
pub(crate) fn proxy_rules(configured: &str) -> Result<Option<reqwest::Proxy>> {
    let configured = configured.trim();
    if configured.is_empty() {
        return Ok(None);
    }
    // A bare `host:port` is a common way to write this; without a scheme it is
    // not a URL, and refusing it here would read as "the proxy is broken".
    let absolute = if configured.contains("://") {
        configured.to_string()
    } else {
        format!("http://{configured}")
    };
    let upstream = reqwest::Url::parse(&absolute)
        .with_context(|| format!("invalid proxy URL {configured:?}"))?;

    let excluded = std::env::var("NO_PROXY")
        .or_else(|_| std::env::var("no_proxy"))
        .ok();

    Ok(Some(reqwest::Proxy::custom(move |target| {
        let host = target.host_str().unwrap_or_default();
        if bypasses_proxy(excluded.as_deref(), host) {
            None
        } else {
            Some(upstream.clone())
        }
    })))
}

/// Whether `NO_PROXY`'s list excludes `host`.
///
/// The list is comma- or space-separated. An entry means everything under it,
/// with or without the leading dot (`example.com` and `.example.com` both
/// exclude `api.example.com` and `example.com` itself), so a suffix only
/// matches at a dot boundary — `example.com` does not exclude
/// `notexample.com`. A `*` entry excludes everything, a `:port` on the entry is
/// ignored (the exclusion is about the host, not the port), and the comparison
/// is case-insensitive because host names are.
pub(crate) fn bypasses_proxy(list: Option<&str>, host: &str) -> bool {
    let Some(list) = list else { return false };
    let host = host.trim().to_ascii_lowercase();
    if host.is_empty() {
        return false;
    }

    for entry in list.split([',', ' ', ';']).map(str::trim).filter(|entry| !entry.is_empty()) {
        if entry == "*" {
            return true;
        }
        let mut name = entry.to_ascii_lowercase();
        // Bracketed IPv6 (`[::1]`) keeps its colons; anything else with a
        // single colon is `host:port`.
        if name.starts_with('[') {
            name = name.trim_matches(['[', ']']).to_string();
        } else if name.matches(':').count() == 1 {
            name = name.split(':').next().unwrap_or_default().to_string();
        }
        let name = name.trim_start_matches('.').to_string();
        if name.is_empty() {
            continue;
        }
        if host == name || host.ends_with(&format!(".{name}")) {
            return true;
        }
    }
    false
}

/// Whether an HTTP status is worth sending the same request again.
///
/// Rate limiting and a broken upstream are the two families every gateway
/// retries on its own, and the third is a request the gateway itself could not
/// take: 408 Request Timeout and 425 Too Early both mean the server declined
/// before the answer began, so repeating the request cannot duplicate an answer
/// the caller has already seen. One function because three paths classify the
/// same response — the streaming open, the buffered open, and an error object
/// carried inside a successful one — and they used to hold three copies of
/// `429 || 5xx`: a 408 was fatal on all of them, and fixing that in three
/// places is how the copies drift apart again.
pub(crate) fn retryable_status(code: u16) -> bool {
    matches!(code, 408 | 425 | 429) || (500..=599).contains(&code)
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

    // Providers label the same failure three ways: `{"code": 429}`,
    // `{"code": "429"}` (gateways behind a JSON string), or a name such as
    // `{"type": "rate_limit_error"}`. Reading only the number loses the first
    // case to the message heuristic and every name entirely.
    let status = error
        .get("code")
        .or_else(|| error.get("status"))
        .or_else(|| error.get("type"))
        .and_then(status_of);

    // Everything the error says about itself, lower-cased: the type, the raw
    // code, and the message. Classification has to read the words, because
    // that is all a provider that never sends a number gives us.
    let words = ["type", "code", "message", "status"]
        .iter()
        .filter_map(|key| error.get(*key))
        .filter_map(|value| value.as_str())
        .collect::<Vec<&str>>()
        .join(" ")
        .to_lowercase();

    let retryable = match status {
        Some(code) => retryable_status(code),
        // With no status to go on, the word list decides — and the default is
        // "do not retry". It used to be the opposite (anything that did not
        // read as one of the fatal phrases was retried), which turned a
        // provider's invalid parameter into three more identical round trips
        // and, on a metered route, three more bills.
        None => !looks_fatal(&words) && says_retryable(&words),
    };
    RequestError {
        status,
        retryable,
        message,
    }
}

/// The HTTP status an error field carries, however it is spelled.
///
/// A number is the normal form; a quoted number (`"429"`) is what a gateway
/// built on stringly-typed errors sends; anything else — `rate_limit_error`,
/// `overloaded` — is a name, answered by [`says_retryable`] instead.
fn status_of(value: &Value) -> Option<u16> {
    let code = match value {
        Value::Number(number) => number.as_u64(),
        Value::String(text) => text.trim().parse::<u64>().ok(),
        _ => None,
    }?;
    // A "status" that is not a status (a name, a 0, a 200) is not usable:
    // treating 200 as a retryable-worthy code would be worse than ignoring it.
    (100..=599).contains(&code).then_some(code as u16)
}

/// The retryable families a provider names in words instead of numbers.
///
/// Anthropic types the error `rate_limit_error` or `overloaded_error`; OpenAI
/// uses `rate_limit_exceeded`; the gateways in between invent `server_error`,
/// `upstream_error`, `capacity_error` and plain "please try again". Matching is
/// a lowercase substring because the spellings multiply and an exact list ages
/// badly — and because a false negative here costs one failed turn, while a
/// false positive costs a retry loop.
fn says_retryable(words: &str) -> bool {
    const WORDS: [&str; 18] = [
        "rate_limit",
        "rate limit",
        "ratelimit",
        "too many requests",
        "overload",
        "server_error",
        "server error",
        "internal error",
        "internal_error",
        "bad gateway",
        "service unavailable",
        "gateway timeout",
        "timed out",
        "temporarily",
        "try again",
        "capacity",
        "upstream",
        "busy",
    ];
    WORDS.iter().any(|word| words.contains(word))
}

/// The context window a provider stated in an error message, when it stated one.
///
/// An error that rejects an oversized prompt is the provider stating the limit
/// it actually enforces, and for the route in front of us that is the most
/// trustworthy source there is: the built-in table is a guess about a model id,
/// while this is a statement about this deployment. Real messages read like
///
/// * `This model's maximum context length is 8192 tokens`
/// * `context length is 128000`
/// * `maximum context window: 65536`
/// * `max_model_len=32768`
///
/// So the rule is a phrase about a context length or window followed, within a
/// few words, by a number — not "the first large number in the message". The
/// same message usually also names the prompt size that was rejected, and
/// measuring against that number would compact history down to a size the
/// provider refuses just as firmly.
pub fn reported_window(message: &str) -> Option<usize> {
    if message.trim().is_empty() {
        return None;
    }
    // ASCII-lowercased rather than `to_lowercase`: the byte offsets found in the
    // lowered copy are used to slice that same copy, and a full lowercase can
    // change the byte length of a non-ASCII character, which would shift every
    // offset after it and could cut a character in half.
    let lowered = message.to_ascii_lowercase();

    // The earliest phrase that is followed by a number answers the question,
    // because that is the first place in the message that states a window; a
    // later mention is a restatement or a different limit.
    let mut found: Option<(usize, usize)> = None;
    for &phrase in CONTEXT_WINDOW_PHRASES {
        let mut from = 0;
        while let Some(offset) = lowered[from..].find(phrase) {
            let at = from + offset;
            // Resume past this occurrence, so a phrase that is not followed by a
            // number does not stop the search for a later one.
            from = at + phrase.len();
            if let Some(tokens) = number_near(&lowered[from..]) {
                if found.map_or(true, |(earliest, _)| at < earliest) {
                    found = Some((at, tokens));
                }
            }
        }
    }
    found.map(|(_, tokens)| tokens)
}

/// The words providers use to introduce a context window in an error message.
///
/// Both spacings are listed because both appear on the wire: Chat Completions
/// writes prose ("maximum context length is …"), while vLLM and the servers
/// around it name their own field ("max_model_len=…"). Everything is lowercase
/// because it is matched against a lowercased message.
const CONTEXT_WINDOW_PHRASES: &[&str] = &[
    "context length",
    "context_length",
    "context window",
    "context_window",
    "context limit",
    "context_limit",
    "max_model_len",
];

/// The first plausible window written within a few words after a phrase.
///
/// The number follows the phrase but rarely immediately: a verb, a colon, or an
/// `=` sits in between ("is 8192", ": 65536", "=32768"). Four words covers every
/// shape seen in the wild and is short enough that a number belonging to the
/// next clause — the rejected prompt's `tokens: 123456` — is left to that clause
/// instead of being read as the limit.
fn number_near(tail: &str) -> Option<usize> {
    const NEARBY_WORDS: usize = 4;
    let plausible =
        crate::window::MIN_PLAUSIBLE_WINDOW..=crate::window::MAX_PLAUSIBLE_WINDOW;
    tail.split_whitespace()
        .take(NEARBY_WORDS)
        .filter_map(number_in_word)
        .find(|tokens| plausible.contains(tokens))
}

/// The number one word holds, if it is a plain count with optional separators.
///
/// Surrounding punctuation is trimmed because the number sits inside prose and
/// inside `key=value`: `(8192`, `8192,`, `:65536` and `=32768` all mean the same
/// number. A word that still holds anything except digits and underscores after
/// that trimming — `4.5`, `1.2e6`, `v1.8192` — is refused rather than parsed
/// loosely, because a decimal, an exponent, or a version is not a window.
fn number_in_word(word: &str) -> Option<usize> {
    let digits = word
        .trim_matches(|character: char| !character.is_ascii_digit() && character != '_');
    let plain = !digits.is_empty()
        && digits
            .chars()
            .all(|character| character.is_ascii_digit() || character == '_');
    if !plain {
        return None;
    }
    // Underscores are the thousands separators a provider writes when it prints
    // a number it read from its own configuration (`max_model_len=32_768`).
    digits.replace('_', "").parse::<usize>().ok()
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
///
/// A cache *write* is a different number from a cache read and is billed
/// differently: Anthropic reports `cache_creation_input_tokens` and prices it
/// above a normal prompt token, so a turn that paid for one and recorded
/// nothing was also the most expensive turn in the accounting.
fn extract_usage(usage: &Value) -> TokenUsage {
    let input = first_count(usage, &[&["prompt_tokens"][..], &["input_tokens"][..]]).unwrap_or(0);
    let output = first_count(usage, &[&["completion_tokens"][..], &["output_tokens"][..]])
        .unwrap_or(0);

    let cached = first_count(
        usage,
        &[
            &["prompt_tokens_details", "cached_tokens"][..],
            &["input_tokens_details", "cached_tokens"][..],
            &["prompt_cache_hit_tokens"][..],
            &["cached_tokens"][..],
            &["cache_read_input_tokens"][..],
        ],
    )
    .unwrap_or(0);

    let cache_created = first_count(
        usage,
        &[
            &["cache_creation_input_tokens"][..],
            &["cache_creation_tokens"][..],
        ],
    )
    .unwrap_or(0);

    let reasoning = first_count(
        usage,
        &[
            &["completion_tokens_details", "reasoning_tokens"][..],
            &["output_tokens_details", "reasoning_tokens"][..],
        ],
    )
    .unwrap_or(0);

    TokenUsage { input, output, cached, cache_created, reasoning }
}

/// The count a single usage field holds, however the gateway wrote it.
///
/// The normal form is a JSON integer, but the same API is fronted by gateways
/// that re-encode numbers: one sends `"123"` because its own struct declared a
/// string, another sends `0.0` because an upstream database column is a float.
/// Both are the number they look like, so both are read. Anything else is
/// reported as absent rather than as zero, because a count of zero and a count
/// nobody could read are different facts and a provider sending nonsense must
/// not look like a provider sending nothing.
fn count_of(value: &Value) -> Option<u64> {
    match value {
        Value::Number(number) => number.as_u64().or_else(|| {
            // A float only counts if it is a whole, non-negative one: `45.0` is
            // the integer a float column could not keep, while `-1` and `4.5`
            // are not token counts and rounding them would invent a number.
            number
                .as_f64()
                .filter(|float| float.is_finite() && *float >= 0.0 && float.fract() == 0.0)
                .map(|float| float as u64)
        }),
        Value::String(text) => text.trim().parse::<u64>().ok(),
        _ => None,
    }
}

/// Walk key paths and return the first one that holds a readable count.
fn first_count(value: &Value, paths: &[&[&str]]) -> Option<u64> {
    paths.iter().find_map(|path| count_of(at_path(value, *path)?))
}

/// Follow a key path, answering only when every key on it exists.
fn at_path<'a>(value: &'a Value, path: &[&str]) -> Option<&'a Value> {
    let mut current = value;
    for key in path {
        current = current.get(*key)?;
    }
    Some(current)
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

    #[test]
    fn a_reported_window_is_read_from_a_number_a_string_or_not_at_all() {
        // The three shapes providers actually send, and the ones that mean
        // nothing: an implausibly small number is a different field being read,
        // not a window, and believing it would make the loop compact constantly.
        assert_eq!(super::plausible_window(&json!(128_000)), Some(128_000));
        assert_eq!(super::plausible_window(&json!("131072")), Some(131_072));
        assert_eq!(super::plausible_window(&json!("128_000")), Some(128_000));
        assert_eq!(super::plausible_window(&json!(null)), None);
        assert_eq!(super::plausible_window(&json!(0)), None);
        assert_eq!(super::plausible_window(&json!(512)), None);
        assert_eq!(super::plausible_window(&json!(-8192)), None);
        assert_eq!(super::plausible_window(&json!({"tokens": 8192})), None);
        assert_eq!(super::plausible_window(&json!(99_999_999)), None);
    }

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
///
/// Shared with [`crate::fetch`], which faces the same pooled-connection race
/// while downloading and must classify it the same way.
pub(crate) fn is_connection_closed(error: &reqwest::Error) -> bool {
    error_chain_matches(error, matches_closed_text)
}

/// Whether TLS records arrived that this session's keys cannot open.
///
/// rustls raises these after a completed handshake, so reqwest reports a request
/// failure rather than a connect error. Nothing of the response has been read
/// when they surface and the connection is unusable behind them, which puts the
/// next attempt on a fresh session: retryable on the same terms as
/// [`is_connection_closed`].
pub(crate) fn is_tls_stream_damage(error: &reqwest::Error) -> bool {
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

    #[test]
    fn a_window_is_read_from_every_shape_a_provider_reports() {
        // One sentence per wire dialect: the prose OpenAI and Anthropic-shaped
        // gateways send, the terse field names vLLM and llama.cpp send, and the
        // same number written with the underscore separators a provider uses
        // when it prints a value from its own configuration.
        for (message, tokens) in [
            ("This model's maximum context length is 8192 tokens", 8192),
            ("context length is 128000", 128_000),
            ("maximum context window: 65536", 65_536),
            ("max_model_len=32768", 32_768),
            ("this model's max_model_len = 32_768", 32_768),
            ("Context Window: 1_000_000 exceeded", 1_000_000),
            (
                "This model's maximum context length is 8192 tokens. However, you requested 20000 tokens",
                8192,
            ),
        ] {
            assert_eq!(reported_window(message), Some(tokens), "{message:?}");
        }
    }

    #[test]
    fn a_number_that_is_not_about_the_context_window_is_not_a_window() {
        // The rejected request's own token count sits in the same message, is
        // often larger than the window, and is the number a naive "first big
        // number" parser would take. Sizing later prompts to it would keep the
        // loop building requests the provider still refuses.
        assert_eq!(reported_window("tokens: 123456"), None);
        assert_eq!(reported_window("you sent 200000 tokens in one request"), None);
        assert_eq!(
            reported_window("maximum context length is 8192 tokens; you sent 200000 tokens"),
            Some(8192),
            "the number after the phrase wins, not the larger one later on"
        );
    }

    #[test]
    fn an_implausible_or_absent_window_reads_as_no_window() {
        // Every refusal here matters: a wrong report silently resizes every
        // later prompt, so anything short of a plausible number is `None`.
        assert_eq!(reported_window(""), None);
        assert_eq!(reported_window("   "), None);
        assert_eq!(reported_window("context length exceeded"), None);
        assert_eq!(
            reported_window("context window: 42"),
            None,
            "a two-digit number is a token count or an id, not a window"
        );
        assert_eq!(
            reported_window("context window: 99999999999"),
            None,
            "a window larger than any provider sells is a byte count"
        );
        assert_eq!(reported_window("invalid request: tools schema"), None);
        assert_eq!(reported_window("上下文长度超出限制"), None);
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
    fn a_cache_write_is_recorded_not_dropped() {
        // Anthropic bills a cache write above the ordinary prompt price and
        // reports it in a field of its own; nothing else in the object repeats
        // the number, so a reader that only knows the read side cannot recover
        // it, and the turn that paid the most looks like the cheapest one.
        let usage = extract_usage(&json!({
            "input_tokens": 12,
            "output_tokens": 7,
            "cache_read_input_tokens": 400,
            "cache_creation_input_tokens": 900
        }));
        assert_eq!(usage.cache_created, 900, "{usage:?}");
        assert_eq!(usage.cached, 400, "a write is not a read: {usage:?}");
        assert_eq!(usage.input, 12);
        assert_eq!(usage.output, 7);
    }

    #[test]
    fn a_count_written_as_a_string_or_a_float_is_still_read() {
        // One gateway is stringly typed and the next is float typed, depending
        // on which upstream struct it copied. Both are the integers they look
        // like, and reading neither would report a whole turn as free.
        let usage = extract_usage(&json!({
            "input_tokens": "123",
            "output_tokens": 45.0,
            "cache_creation_input_tokens": "6"
        }));
        assert_eq!(usage.input, 123, "{usage:?}");
        assert_eq!(usage.output, 45, "{usage:?}");
        assert_eq!(usage.cache_created, 6, "{usage:?}");
    }

    #[test]
    fn a_count_that_is_not_a_count_is_not_read() {
        // A negative count, a fraction and a word are not "zero tokens".
        // Reporting them as zero hides a provider answering nonsense, which is
        // the failure a usage number exists to make visible.
        let below_zero = Value::from(-5i64);
        for nonsense in [below_zero, json!("many"), json!(4.5), json!(null), json!(true)] {
            assert_eq!(count_of(&nonsense), None, "{nonsense} is not a count");
        }
        // What reaches the accounting is "not reported" rather than an invented
        // number, and a field that cannot be read leaves the next spelling to
        // answer instead of ending the search.
        let mut unreadable = json!({ "output_tokens": "n/a" });
        unreadable["input_tokens"] = Value::from(-5i64);
        let usage = extract_usage(&unreadable);
        assert_eq!(usage.input, 0, "{usage:?}");
        assert_eq!(usage.output, 0, "{usage:?}");
        let fallback = extract_usage(&json!({ "prompt_tokens": "many", "input_tokens": 30 }));
        assert_eq!(fallback.input, 30, "{fallback:?}");
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

    #[test]
    fn a_quoted_status_code_is_read_like_a_numeric_one() {
        // Gateways built on stringly-typed errors send `"code": "429"`. Only
        // numbers used to be read, so a quoted 429 fell through to the message
        // heuristic and was classified by its prose instead of its status.
        let quoted = error_request(&json!({ "code": "429", "message": "slow down" }));
        assert_eq!(quoted.status, Some(429), "{quoted:?}");
        assert!(quoted.retryable, "{quoted:?}");

        let padded = error_request(&json!({ "status": " 503 ", "message": "upstream is unhappy" }));
        assert_eq!(padded.status, Some(503), "{padded:?}");
        assert!(padded.retryable);

        // A "code" that is a name is not a status, and must not be read as one.
        let named = error_request(&json!({ "code": "rate_limit_error", "message": "slow down" }));
        assert_eq!(named.status, None, "{named:?}");
        assert!(named.retryable, "the name still classifies it: {named:?}");
    }

    #[test]
    fn a_named_error_kind_classifies_without_a_number() {
        // Anthropic types the error, OpenAI codes it, the gateways between them
        // invent the rest. With no number to look at, the words are the whole
        // of the evidence.
        for kind in [
            "rate_limit_error",
            "overloaded_error",
            "server_error",
            "rate_limit_exceeded",
        ] {
            let error = error_request(&json!({ "type": kind, "message": "cannot take it now" }));
            assert_eq!(error.status, None, "{kind}");
            assert!(error.retryable, "{kind} is worth another attempt");
        }
    }

    #[test]
    fn a_status_less_error_is_not_retried_without_a_reason_to_be() {
        // The old default was the optimistic one: anything that failed to read
        // as a fatal phrase was retried, so a request the provider rejected on
        // its merits cost three more identical round trips.
        let unsupported = error_request(&json!({
            "message": "the model does not support this parameter combination"
        }));
        assert_eq!(unsupported.status, None);
        assert!(
            !unsupported.retryable,
            "nothing here says another attempt would go differently: {unsupported:?}"
        );

        // ...while a provider that does say "again" still gets one.
        let busy = error_request(&json!({ "message": "The engine is busy, please try again" }));
        assert!(busy.retryable, "{busy:?}");

        // And a status-less message that reads fatal stays refused even when it
        // happens to contain a retryable word.
        let contradictory = error_request(&json!({
            "message": "invalid_request_error: rate limit is not applicable here"
        }));
        assert!(!contradictory.retryable, "{contradictory:?}");
    }

    #[test]
    fn the_statuses_worth_another_attempt_are_recognised() {
        // 408 and 425 both mean the server declined before the answer started,
        // so a repeat cannot duplicate anything the caller has seen; the rest
        // are the rate-limit and broken-upstream families as before.
        for code in [408, 425, 429, 500, 503] {
            assert!(retryable_status(code), "{code} is worth another attempt");
        }
        // A request the provider rejected on its merits fails the same way the
        // second time, and on a metered route it fails for money.
        for code in [400, 401, 403, 404, 422] {
            assert!(!retryable_status(code), "{code} must not be retried");
        }

        // The verdict has to reach the classification, not just the predicate:
        // a 408 carried inside a successful body is the same decline.
        let timed_out = error_request(&json!({ "code": 408, "message": "the upstream took too long" }));
        assert!(timed_out.retryable, "{timed_out:?}");
        let too_early = error_request(&json!({ "status": 425, "message": "too early" }));
        assert!(too_early.retryable, "{too_early:?}");
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
    async fn a_clear_text_endpoint_is_refused_before_a_request_is_built() {
        // The route can arrive from `NGU_BASE_URL`, a pack, or `set_model`
        // without passing through the settings write that checks the scheme, so
        // the request path needs its own gate. Refusing here means no request is
        // ever built, which is the only thing that keeps the key off the wire.
        let config = LlmConfig::new("http://api.example.test/v1", "secret-key-value", "m");
        let client = LlmClient::new(config).unwrap();

        let error = match client.chat_stream(&[Message::user("go")], &[]).await {
            Ok(_) => panic!("a clear-text remote endpoint must be refused"),
            Err(error) => error,
        };
        let message = format!("{error:#}");
        assert!(message.contains("http://api.example.test/v1"), "{message}");
        assert!(
            !message.contains("secret-key-value"),
            "the refusal must not print the key: {message}"
        );
    }

    #[tokio::test]
    async fn a_loopback_clear_text_endpoint_is_allowed() {
        // A gateway on this machine never leaves it, so the loopback exemption
        // has to survive the request-path check too: refusing it would break
        // every local llama.cpp / Ollama / vLLM setup.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            while let Ok((mut socket, _)) = listener.accept().await {
                tokio::spawn(async move {
                    let mut chunk = [0u8; 8192];
                    let _ = socket.read(&mut chunk).await;
                    let body = "{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"local answer\"},\"finish_reason\":\"stop\"}]}";
                    let response = format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{}",
                        body.len(),
                        body
                    );
                    let _ = socket.write_all(response.as_bytes()).await;
                    let _ = socket.flush().await;
                    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
                });
            }
        });

        let mut config = LlmConfig::new(format!("http://{addr}/v1"), "k", "m");
        config.delivery = Delivery::Buffered;
        let client = LlmClient::new(config).unwrap();

        let mut rx = client.chat_stream(&[Message::user("go")], &[]).await.unwrap();
        let mut text = String::new();
        while let Some(event) = rx.recv().await {
            if let LlmEvent::TextDelta(delta) = event.expect("the local answer completes") {
                text.push_str(&delta);
            }
        }
        assert_eq!(text, "local answer");
    }

    #[tokio::test]
    async fn a_stream_that_never_ends_is_cut_off_by_the_total_budget() {
        // The shape that motivated the bound: 200, `text/event-stream`, then one
        // frame every 20 ms, for ever. Every frame lands well inside the idle
        // bound, so the idle timeout never fires — only a total budget ends it.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            while let Ok((mut socket, _)) = listener.accept().await {
                tokio::spawn(async move {
                    let mut chunk = [0u8; 8192];
                    let _ = socket.read(&mut chunk).await;
                    let _ = socket
                        .write_all(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n\r\n")
                        .await;
                    let _ = socket.flush().await;
                    for _ in 0..10_000 {
                        if socket
                            .write_all(b"data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n\n")
                            .await
                            .is_err()
                        {
                            return;
                        }
                        let _ = socket.flush().await;
                        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
                    }
                });
            }
        });

        let mut config = LlmConfig::new(format!("http://{addr}/v1"), "k", "m");
        config.network.stream_idle_secs = 30;
        config.network.request_timeout_secs = 60;
        let client = LlmClient::new(config)
            .unwrap()
            .with_stream_budget(std::time::Duration::from_millis(400));

        let started = std::time::Instant::now();
        let mut rx = client.chat_stream(&[Message::user("go")], &[]).await.unwrap();
        let mut text = String::new();
        let mut reason: Option<String> = None;
        while let Some(event) = rx.recv().await {
            match event.expect("a cut-off stream still delivers what it received") {
                LlmEvent::TextDelta(delta) => text.push_str(&delta),
                LlmEvent::Finished { reason: r } => reason = r,
                _ => {}
            }
        }

        assert!(
            !text.is_empty(),
            "what arrived before the cut is kept, not dropped"
        );
        let reason = reason.expect("the cut says why the answer stopped");
        assert!(reason.contains("total budget"), "{reason}");
        assert!(
            started.elapsed() < std::time::Duration::from_secs(20),
            "the total budget, not the 30 s idle bound, must end this stream"
        );
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

#[cfg(test)]
mod stream_tests {
    use super::*;
    use serde_json::json;
    use std::time::Duration;

    /// Read a request off a socket so the client is never writing into a closed
    /// pipe: head first, then the declared body.
    async fn drain_request(socket: &mut tokio::net::TcpStream) -> String {
        use tokio::io::AsyncReadExt;
        let mut head = Vec::new();
        loop {
            let mut byte = [0u8; 1];
            match socket.read(&mut byte).await {
                Ok(0) | Err(_) => return String::from_utf8_lossy(&head).to_string(),
                Ok(_) => {
                    head.push(byte[0]);
                    if head.ends_with(b"\r\n\r\n") {
                        break;
                    }
                }
            }
        }
        let text = String::from_utf8_lossy(&head).to_string();
        let declared = text
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
        text
    }

    /// The lines produced by feeding `wire` in `step`-byte pieces, as the read
    /// loop does: complete lines as they finish, then whatever tail is left.
    fn decode_in_pieces(wire: &[u8], step: usize) -> Vec<String> {
        let mut decoder = SseDecoder::new();
        let mut lines = Vec::new();
        for piece in wire.chunks(step.max(1)) {
            lines.extend(decoder.push(piece).expect("no frame outgrows the ceiling here"));
        }
        lines.extend(decoder.finish());
        lines
    }

    /// The JSON a `data:` line carries.
    fn payload(line: &str) -> Value {
        let data = line.strip_prefix("data:").expect("a data line");
        serde_json::from_str(data.trim()).expect("a JSON frame")
    }

    #[test]
    fn a_frame_that_ends_without_a_newline_is_still_a_frame() {
        // A provider closes the connection instead of writing `[DONE]`, and its
        // last frame has no trailing newline. The bytes are over, not empty:
        // dropping that frame loses the last tool-call fragment, the usage
        // report, and the finish reason together.
        let first = "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n";
        let last = "data: {\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":1}}";
        let mut decoder = SseDecoder::new();
        let lines = decoder.push(format!("{first}{last}").as_bytes()).unwrap();

        let frames: Vec<&String> = lines.iter().filter(|line| line.starts_with("data:")).collect();
        assert_eq!(frames.len(), 1, "the terminated frame, and not the tail");
        assert_eq!(
            decoder.finish().as_deref(),
            Some(last),
            "the unterminated tail is the frame the stream ended on"
        );
        assert!(decoder.finish().is_none(), "and it is only read once");
    }

    #[test]
    fn a_character_split_across_two_chunks_is_not_damaged() {
        // The failure this pins down: decoding each network chunk on its own
        // turns a UTF-8 character that straddles the boundary into U+FFFD, with
        // no error anywhere, and that mangled text becomes the answer.
        let text = "模型回答：天气很好，参数 {\"命令\":\"运行\"} 🌤 结束";
        let wire = format!(
            "data: {}\n\n",
            json!({ "choices": [{ "delta": { "content": text } }] })
        );
        let bytes = wire.as_bytes();

        // Cut inside the three-byte first character of the Chinese text.
        let at = bytes
            .iter()
            .position(|byte| *byte >= 0xE0)
            .expect("the frame carries a multi-byte character")
            + 1;

        let mut decoder = SseDecoder::new();
        let mut lines = decoder.push(&bytes[..at]).unwrap();
        assert!(lines.is_empty(), "half a character is not a line");
        lines.extend(decoder.push(&bytes[at..]).unwrap());
        assert!(decoder.finish().is_none());

        let line = lines
            .iter()
            .find(|line| line.starts_with("data:"))
            .expect("the frame is a data line");
        assert!(
            !line.contains('\u{fffd}'),
            "no character may be replaced: {line:?}"
        );
        assert_eq!(
            payload(line)["choices"][0]["delta"]["content"],
            json!(text),
            "the text must survive byte for byte"
        );
    }

    #[test]
    fn the_worst_case_chunking_keeps_every_character() {
        // One byte per read is the adversarial version of the same question: no
        // character ever arrives whole, and all of them must still come out
        // whole.
        let text = "中文流式输出：适配器跨越数据块边界 🚀";
        let wire = format!(
            "data: {}\n\ndata: [DONE]\n\n",
            json!({ "choices": [{ "delta": { "content": text } }] })
        );

        let lines: Vec<String> = decode_in_pieces(wire.as_bytes(), 1)
            .into_iter()
            .filter(|line| !line.trim().is_empty())
            .collect();
        assert_eq!(lines.len(), 2);
        assert_eq!(payload(&lines[0])["choices"][0]["delta"]["content"], json!(text));
        assert_eq!(lines[1].trim(), "data: [DONE]");
    }

    #[test]
    fn a_frame_that_never_ends_is_refused_rather_than_buffered() {
        // An endpoint that writes bytes forever without a newline must not be
        // able to grow this process's memory without limit.
        let mut decoder = SseDecoder::new();
        let chunk = vec![b'x'; 1024 * 1024];
        let mut refusal = None;
        for _ in 0..=(MAX_FRAME_BYTES / chunk.len()) + 1 {
            match decoder.push(&chunk) {
                Ok(lines) => assert!(lines.is_empty(), "not a single line was terminated"),
                Err(reason) => {
                    refusal = Some(reason);
                    break;
                }
            }
        }
        let refusal = refusal.expect("an endless frame must be refused");
        assert!(refusal.contains("MiB"), "the reason names the ceiling: {refusal}");
    }

    #[tokio::test]
    async fn the_last_unterminated_frame_reaches_the_caller() {
        // The whole fix, end to end: an answer whose final frame carries the
        // tool call's arguments, the usage report, and the finish reason, and
        // whose connection then closes without a newline or a `[DONE]`.
        let final_frame = json!({
            "choices": [{
                "delta": { "tool_calls": [{
                    "index": 0,
                    "id": "call_1",
                    "function": { "name": "bash", "arguments": "{\"command\":\"dir\"}" }
                }] },
                "finish_reason": "tool_calls"
            }],
            "usage": { "prompt_tokens": 11, "completion_tokens": 5, "total_tokens": 16 }
        });
        let body = format!(
            "data: {}\n\ndata: {}",
            json!({ "choices": [{ "delta": { "content": "好的" } }] }),
            final_frame
        );

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            use tokio::io::AsyncWriteExt;
            while let Ok((mut socket, _)) = listener.accept().await {
                let body = body.clone();
                tokio::spawn(async move {
                    drain_request(&mut socket).await;
                    let response = format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\nconnection: close\r\n\r\n{body}"
                    );
                    let _ = socket.write_all(response.as_bytes()).await;
                    let _ = socket.flush().await;
                    // Hold the socket open for a moment so the close the client
                    // sees is an orderly end of body rather than a reset.
                    tokio::time::sleep(Duration::from_millis(50)).await;
                });
            }
        });

        let client = LlmClient::new(LlmConfig::new(format!("http://{addr}"), "k", "m")).unwrap();
        let mut rx = client.chat_stream(&[Message::user("go")], &[]).await.unwrap();

        let mut text = String::new();
        let mut arguments: Option<String> = None;
        let mut usage: Option<TokenUsage> = None;
        let mut reason: Option<String> = None;
        while let Some(event) = rx.recv().await {
            match event.expect("the answer completes") {
                LlmEvent::TextDelta(delta) => text.push_str(&delta),
                LlmEvent::ToolCallDelta { arguments: args, .. } => {
                    arguments.get_or_insert_with(String::new).push_str(args.as_deref().unwrap_or(""))
                }
                LlmEvent::Usage(reported) => usage = Some(reported),
                LlmEvent::Finished { reason: why } => reason = why,
                LlmEvent::ReasoningDelta(_) => {}
            }
        }

        assert_eq!(text, "好的");
        let arguments = arguments.expect("the tool call in the final frame arrived");
        let parsed: Value = serde_json::from_str(&arguments)
            .expect("the arguments are complete JSON, not a truncated fragment");
        assert_eq!(parsed["command"], "dir");

        let usage = usage.expect("the usage report in the final frame arrived");
        assert_eq!(usage.input, 11);
        assert_eq!(usage.output, 5);
        assert_eq!(reason.as_deref(), Some("tool_calls"));
    }

    #[tokio::test]
    async fn an_endless_frame_failure_reaches_the_caller_as_an_error() {
        // The same ceiling as seen from outside: the read loop must report the
        // abandonment, not hand the turn a truncated answer.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            use tokio::io::AsyncWriteExt;
            let mut socket = match listener.accept().await {
                Ok((socket, _)) => socket,
                Err(_) => return,
            };
            drain_request(&mut socket).await;
            let _ = socket
                .write_all(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n\r\n")
                .await;
            // No newline, ever.
            let block = vec![b' '; 1024 * 1024];
            for _ in 0..(MAX_FRAME_BYTES / block.len() + 2) {
                if socket.write_all(&block).await.is_err() {
                    return;
                }
            }
            tokio::time::sleep(Duration::from_secs(1)).await;
        });

        let mut config = LlmConfig::new(format!("http://{addr}"), "k", "m");
        config.network.stream_idle_secs = 30;
        let client = LlmClient::new(config).unwrap();
        let mut rx = client.chat_stream(&[Message::user("go")], &[]).await.unwrap();

        let mut failure: Option<anyhow::Error> = None;
        while let Some(event) = rx.recv().await {
            if let Err(error) = event {
                failure = Some(error);
                break;
            }
        }
        let failure = failure.expect("an endless frame is an error, not an answer");
        assert!(
            format!("{failure:#}").contains("MiB"),
            "the reason explains the abandonment: {failure:#}"
        );
    }

    #[tokio::test]
    async fn a_cancelled_turn_stops_reading_the_provider_stream() {
        // Dropping the receiver is how a turn is cancelled. If the send failure
        // is ignored, the read loop runs on for as long as the provider keeps
        // writing, spending tokens on frames nobody will ever see. The server
        // notices only when the client's task has actually stopped.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (gone, mut gone_rx) = tokio::sync::watch::channel(false);

        tokio::spawn(async move {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            let mut socket = match listener.accept().await {
                Ok((socket, _)) => socket,
                Err(_) => return,
            };
            drain_request(&mut socket).await;

            let (mut reader, mut writer) = socket.into_split();
            let write_gone = gone.clone();
            tokio::spawn(async move {
                if writer
                    .write_all(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n\r\n")
                    .await
                    .is_err()
                {
                    let _ = write_gone.send(true);
                    return;
                }
                // Frames forever: the only thing that can stop this is the
                // client hanging up, and that is what is under test.
                for _ in 0..1000 {
                    if writer
                        .write_all(b"data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n\n")
                        .await
                        .is_err()
                    {
                        break;
                    }
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
                let _ = write_gone.send(true);
            });

            let mut buf = [0u8; 4096];
            loop {
                match reader.read(&mut buf).await {
                    Ok(0) | Err(_) => break,
                    Ok(_) => {}
                }
            }
            let _ = gone.send(true);
        });

        let mut config = LlmConfig::new(format!("http://{addr}"), "k", "m");
        config.network.stream_idle_secs = 30;
        let client = LlmClient::new(config).unwrap();
        let mut rx = client.chat_stream(&[Message::user("go")], &[]).await.unwrap();

        // Let the stream begin, then cancel the turn.
        let first = rx.recv().await.expect("a first frame");
        assert!(first.is_ok());
        drop(rx);

        tokio::time::timeout(Duration::from_secs(10), async {
            while !*gone_rx.borrow() {
                if gone_rx.changed().await.is_err() {
                    return;
                }
            }
        })
        .await
        .expect("the client must stop reading the stream once nobody is listening");
    }
}

#[cfg(test)]
mod proxy_tests {
    use super::*;

    #[test]
    fn no_proxy_excludes_a_host_and_everything_below_it() {
        // The list everyone else on the machine honours; neither client read it
        // before, so a host sent direct everywhere else went through the proxy
        // here — credentials and all.
        assert!(bypasses_proxy(Some("example.com"), "example.com"));
        assert!(bypasses_proxy(Some("example.com"), "api.example.com"));
        assert!(bypasses_proxy(Some(".example.com"), "api.example.com"));

        // A suffix only matches at a dot boundary.
        assert!(!bypasses_proxy(Some("example.com"), "notexample.com"));
        assert!(!bypasses_proxy(Some(".example.com"), "example.com.evil.test"));

        // Lists are comma- or space-separated and may be padded or empty.
        assert!(bypasses_proxy(Some("a.test, b.test ,c.test"), "c.test"));
        assert!(bypasses_proxy(Some("a.test b.test"), "b.test"));
        assert!(!bypasses_proxy(Some("  "), "example.com"));
        assert!(!bypasses_proxy(None, "example.com"));

        // `*` is everything; a port on the entry is not part of the host.
        assert!(bypasses_proxy(Some("*"), "anywhere.test"));
        assert!(bypasses_proxy(Some("host.test:8080"), "host.test"));

        // Host names are case-insensitive, and an empty host matches nothing.
        assert!(bypasses_proxy(Some("API.TEST"), "api.test"));
        assert!(!bypasses_proxy(Some("example.com"), ""));
    }

    #[test]
    fn an_empty_proxy_setting_means_a_direct_connection() {
        // Both clients ask this one function, so "direct by default" cannot
        // drift into "direct for one of them and inherited for the other".
        assert!(proxy_rules("").unwrap().is_none());
        assert!(proxy_rules("   ").unwrap().is_none());
        assert!(proxy_rules("http://127.0.0.1:7890").unwrap().is_some());
        // A bare `host:port` is a common way to write it and must not read as
        // a broken proxy.
        assert!(proxy_rules("127.0.0.1:7890").unwrap().is_some());
        let broken = proxy_rules("://broken").expect_err("not a URL");
        assert!(
            format!("{broken:#}").contains("invalid proxy"),
            "{broken:#}"
        );
    }
}
