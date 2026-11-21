use super::*;

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
pub const STREAM_TOTAL_BUDGET_SECS: u64 = 30 * 60;

/// The total budget for one streamed answer, from `NGU_STREAM_MAX_SECS`.
///
/// Zero and unparsable values fall back to the default rather than to "no
/// budget": a variable set to 0 must not mean "hang forever".
pub fn stream_total_budget() -> std::time::Duration {
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

/// Streaming client for one model route.
#[derive(Clone)]
pub struct LlmClient {
    pub http: reqwest::Client,
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
    /// An effort spelling this endpoint has already refused, if any.
    ///
    /// Effort levels are not a standard vocabulary — the same request means
    /// `minimal` to one gateway and `off` to another — and a gateway that does
    /// not know a word rejects the *whole* request with 400 rather than ignoring
    /// the field. Discovering that on every turn costs a turn each time, so the
    /// first refusal is remembered and later bodies are built without the field.
    /// Shared across clones for the same reason as `degraded`: a subagent on this
    /// route would otherwise re-learn it, one wasted turn at a time.
    ///
    /// A clone is not the only way this client is rebuilt: the desktop builds a
    /// fresh client for every turn, so a per-client slot alone would relearn the
    /// same refusal once per turn — one 400 round trip per turn, forever. The
    /// learned refusal therefore also lives in REFUSED_EFFORTS, keyed by
    /// endpoint, and a new client starts from what this process already knows.
    refused_effort: Arc<std::sync::Mutex<Option<String>>>,
}

/// Effort spellings this process has already seen an endpoint refuse.
///
/// Keyed by endpoint plus effort, because the rejection is a property of the
/// endpoint's vocabulary and not of the client object: one gateway refusing
/// `minimal` says nothing about another gateway, and a user may have both
/// configured as profiles. A process-lifetime table rather than a file: the fact
/// is cheap to rediscover once per run and worthless to persist, and an endpoint
/// that gains a word in an upgrade must not inherit an old refusal.
pub static REFUSED_EFFORTS: std::sync::Mutex<Option<std::collections::HashSet<String>>> =
    std::sync::Mutex::new(None);

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
        // Seed from what this process already learned about this endpoint: the
        // desktop builds a client per turn, and a refusal rediscovered every
        // turn would cost a 400 on every turn.
        let wanted = config.reasoning_effort.clone().unwrap_or_default();
        let seed = if !wanted.trim().is_empty()
            && effort_already_refused(&config.base_url, wanted.trim())
        {
            Some(wanted.trim().to_string())
        } else {
            None
        };
        Ok(Self {
            http,
            config,
            shaper: Arc::new(crate::request::Passthrough),
            degraded: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            stream_budget: stream_total_budget(),
            refused_effort: Arc::new(std::sync::Mutex::new(seed)),
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
        // One retry without the effort field, and only for the failure that
        // names it. A gateway whose vocabulary for effort differs from ours
        // rejects the whole request over one word; asking again with the
        // field absent is the difference between a turn the user can use and
        // a 400 they cannot act on. The refusal is remembered, so this costs
        // one extra round trip per route rather than one per turn.
        let refused_word = body
            .get("reasoning_effort")
            .and_then(Value::as_str)
            .map(str::to_string);
        match (self.dispatch(body.clone()).await, refused_word) {
            // An image inside a tool result is the second shape a provider may
            // refuse. The fallback is the same idea: remember the refusal on
            // this route and ask again with the shape every provider accepts
            // (the image travels as a user message instead).
            (Err(error), _) if self.endpoint_refused_tool_image(&error) => {
                remember_tool_image_refused(&self.config.base_url, &self.config.model);
                eprintln!(
                    "[llm] this route rejects images inside tool results;                      sending them as user messages from now on"
                );
                self.dispatch(self.build_body(messages, tools)).await
            }
            (Err(error), Some(word)) if self.endpoint_refused_effort(&error) => {
                // The endpoint names `reasoning_effort` in its rejection, so the
                // field is the problem and not the turn. Ask once more without
                // it -- and say so, because a setting that silently stops being
                // sent is one the user will report as broken.
                self.note_refused_effort(&word);
                eprintln!(
                    "[llm] this endpoint does not accept reasoning_effort = {word:?}; sending this route without it from now on"
                );
                self.dispatch(self.build_body(messages, tools)).await
            }
            (outcome, _) => outcome,
        }
    }

    /// Send a body that has already been built, by the mode it asks for.
    ///
    /// The mode is read back off the finished body rather than taken from the
    /// config, because the shaper has the last word on it: an `extraBody` that
    /// says `"stream": false` means a single JSON answer, and reading that
    /// answer as SSE would lose every word of it without an error.
    async fn dispatch(&self, body: Value) -> Result<mpsc::Receiver<Result<LlmEvent>>> {
        if asks_for_stream(&body) {
            self.open_stream(body).await
        } else {
            self.open_buffered(body).await
        }
    }

    /// Whether this failure is the endpoint rejecting our effort spelling.
    ///
    /// Narrow on purpose: a 400 that names `reasoning_effort` says the word is
    /// wrong, while any other 400 is about the request as a whole and must
    /// reach the caller unchanged.
    /// Whether this failure is the endpoint rejecting an image in a tool result.
    ///
    /// Narrow like `endpoint_refused_effort`: the rejection has to name the
    /// field we added, otherwise a 400 about something else would silently cost
    /// this route its images for good.
    fn endpoint_refused_tool_image(&self, error: &anyhow::Error) -> bool {
        error.chain().any(|cause| {
            cause
                .downcast_ref::<RequestError>()
                .map(|failure| {
                    failure.status == Some(400)
                        && (failure.message.contains("image_url")
                            || failure.message.contains("tool")
                            || failure.message.contains("content"))
                })
                .unwrap_or(false)
        })
    }

    fn endpoint_refused_effort(&self, error: &anyhow::Error) -> bool {
        error.chain().any(|cause| {
            cause
                .downcast_ref::<RequestError>()
                .map(|failure| {
                    failure.status == Some(400)
                        && failure.message.contains("reasoning_effort")
                })
                .unwrap_or(false)
        })
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
    pub async fn send_retrying<F>(&self, build: F) -> Result<reqwest::Response>
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

    /// The effort spelling this endpoint has already refused, if any.
    pub fn refused_effort(&self) -> Option<String> {
        self.refused_effort
            .lock()
            .expect("refused lock")
            .clone()
    }

    /// Remember that this endpoint does not accept this effort spelling.
    ///
    /// Returns whether it was news. The caller that saw the 400 asks again once
    /// without the field; every later request omits it for free rather than
    /// earning the same rejection on each turn. Recorded both on this client
    /// and in the process-wide table, because the next turn gets a different
    /// client object.
    pub fn note_refused_effort(&self, effort: &str) -> bool {
        remember_refused_effort(&self.config.base_url, effort);
        let mut slot = self.refused_effort.lock().expect("refused lock");
        if slot.as_deref() == Some(effort) {
            return false;
        }
        *slot = Some(effort.to_string());
        true
    }
    fn build_body(&self, messages: &[Message], tools: &[Value]) -> Value {
        let tool_images = tool_image_allowed(&self.config.base_url, &self.config.model);
        let wire: Vec<Value> = messages
            .iter()
            .map(|m| to_wire_with(m, tool_images))
            .collect();
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
            // A spelling this endpoint has already refused is not sent
            // again: the rejection is about the word, not about the turn,
            // so repeating it would fail every request in the session.
            let refused = self
                .refused_effort
                .lock()
                .expect("refused lock")
                .clone()
                .unwrap_or_default();
            if !effort.is_empty() && effort != "default" && effort != refused {
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

/// Whether the finished body asks for a stream.
///
/// Absent or `null` means it does not: the OpenAI default for `stream` is
/// `false`, so a shaper that removes the field is asking for a JSON answer and
/// must be read as one.
pub fn asks_for_stream(body: &Value) -> bool {
    body.get("stream").and_then(Value::as_bool).unwrap_or(false)
}

/// The events a single JSON answer carries, in the order a stream would have.
///
/// Reasoning before text, then the calls, then usage, then the close: the loop
/// assembles an assistant turn from these, and an answer that reads differently
/// because it arrived whole instead of in fragments would put the delivery mode
/// in the conversation.
pub fn buffered_events(answer: &Value) -> Vec<LlmEvent> {
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
        // A loopback address never goes through a proxy. Routing the machine's
        // own request to its own service through a proxy lands at whatever that
        // proxy thinks the address means — and for `127.0.0.1` that is the
        // proxy's own loopback, not this machine's. Nobody sets NO_PROXY to
        // exclude localhost for this reason: it is not a policy, it is the only
        // answer that works.
        if is_loopback(host) || bypasses_proxy(excluded.as_deref(), host) {
            None
        } else {
            Some(upstream.clone())
        }
    })))
}

/// Whether `host` names this machine.
///
/// Only the literal forms: a hostname that resolves to loopback ("mybox.local")
/// cannot be told apart from one that does not without a DNS lookup, and this
/// runs on every request.
pub fn is_loopback(host: &str) -> bool {
    matches!(host, "127.0.0.1" | "localhost" | "::1" | "[::1]")
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




