//! How the kernel talks to a provider over the network.
//!
//! Four numbers decide whether a turn survives a flaky connection: how long a
//! request may take, how long an idle connection is kept for reuse, how many
//! times a dropped connection is retried, and how long to wait between tries.
//!
//! They are not constants. The right values depend on the provider and on the
//! network — a local model wants no retries and a two-second timeout; a gateway
//! behind a load balancer that reaps idle connections at thirty seconds wants a
//! shorter pool timeout and more retries; a slow reasoning model wants a long
//! request timeout and a short pool timeout, because its requests are long but
//! its connections are not reused often.
//!
//! # Three ways to set them
//!
//! | | where | who |
//! |---|---|---|
//! | settings | `ngu config set --request-timeout 600` | the person |
//! | pack | `models.json` in a pack | a pack author |
//! | plugin | the `network` service | a plugin |
//!
//! The plugin path wins when it is present, because a plugin that publishes this
//! service is making a deliberate claim about how the connection should work —
//! telling it to stand down because a settings file says otherwise would make
//! the service useless.

use std::time::Duration;

use serde::{Deserialize, Serialize};

/// The name a plugin publishes this service under.
pub const SERVICE: &str = "network";

/// The numbers that decide how a request is sent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NetworkSettings {
    /// Whole-request timeout, in seconds.
    #[serde(default = "default_request_timeout")]
    pub request_timeout_secs: u64,
    /// How long an idle connection is kept for reuse, in seconds.
    ///
    /// This should sit below the provider's own idle timeout. If it is above,
    /// the client hands out connections the server has already closed, and the
    /// symptom is a request that fails with the peer closing the connection.
    #[serde(default = "default_pool_idle")]
    pub pool_idle_timeout_secs: u64,
    /// How many times a connection-level failure is retried.
    #[serde(default = "default_retry_attempts")]
    pub retry_attempts: u32,
    /// Milliseconds before the first retry. Each further retry doubles it.
    #[serde(default = "default_retry_backoff")]
    pub retry_backoff_ms: u64,
    /// How long a streaming answer may go without a single SSE frame before the
    /// turn is abandoned, in seconds.
    ///
    /// Without it, a provider that accepts the request and then stops writing
    /// holds the turn for the whole request timeout, which reads to the user as
    /// a hung agent rather than a failed request. Measured against a real
    /// endpoint that emits the opening frame and then goes silent, the first
    /// frame arrives in under a second, so this only has to outpace a slow
    /// start, not a long answer.
    #[serde(default = "default_stream_idle")]
    pub stream_idle_secs: u64,
    /// How long a catalog request may go unanswered, in seconds.
    ///
    /// The model list is fetched to fill a picker, so the alternative to a bound
    /// is not a later answer but a settings panel that sits showing nothing.
    /// Measured against a real gateway that answers `chat/completions` and never
    /// answers `GET /models`, the request held the whole request timeout — three
    /// hundred seconds of a frozen picker for a list that was never coming.
    #[serde(default = "default_catalog_timeout")]
    pub catalog_timeout_secs: u64,
}

fn default_catalog_timeout() -> u64 {
    20
}

fn default_request_timeout() -> u64 {
    300
}

fn default_pool_idle() -> u64 {
    30
}

fn default_retry_attempts() -> u32 {
    3
}

fn default_retry_backoff() -> u64 {
    200
}

fn default_stream_idle() -> u64 {
    45
}

impl Default for NetworkSettings {
    fn default() -> Self {
        Self {
            request_timeout_secs: default_request_timeout(),
            pool_idle_timeout_secs: default_pool_idle(),
            retry_attempts: default_retry_attempts(),
            retry_backoff_ms: default_retry_backoff(),
            stream_idle_secs: default_stream_idle(),
            catalog_timeout_secs: default_catalog_timeout(),
        }
    }
}

impl NetworkSettings {
    /// The whole-request timeout.
    pub fn request_timeout(&self) -> Duration {
        Duration::from_secs(self.request_timeout_secs.max(1))
    }

    /// How long the model catalog may go unanswered.
    ///
    /// Clamped below the whole-request timeout for the same reason as
    /// [`NetworkSettings::stream_idle`]: a bound that outlives the request it
    /// guards is not a bound.
    pub fn catalog_timeout(&self) -> Duration {
        Duration::from_secs(self.catalog_timeout_secs.clamp(1, self.request_timeout_secs.max(1)))
    }

    /// How long an idle connection is kept.
    pub fn pool_idle_timeout(&self) -> Duration {
        Duration::from_secs(self.pool_idle_timeout_secs.max(1))
    }

    /// How long a streaming answer may stay silent before the turn is abandoned.
    ///
    /// Clamped below the whole-request timeout: an idle bound that outlives the
    /// request it guards can never fire, so it would restore the hang this
    /// exists to remove.
    pub fn stream_idle(&self) -> Duration {
        Duration::from_secs(self.stream_idle_secs.clamp(1, self.request_timeout_secs.max(1)))
    }

    /// How long to wait before retry number `attempt` (1-based).
    ///
    /// Doubling rather than a fixed pause: a provider that is briefly
    /// overloaded needs the client to back off, and three quick retries are
    /// three more requests arriving at the worst moment.
    pub fn backoff(&self, attempt: u32) -> Duration {
        let shift = attempt.saturating_sub(1).min(10);
        Duration::from_millis(self.retry_backoff_ms.saturating_mul(1u64 << shift))
    }

    /// The values a plausible provider actually wants, rather than the midpoint
    /// of nothing. Used by `ngu config set --network <name>`.
    pub fn preset(name: &str) -> Option<Self> {
        match name {
            // Long requests, connections rarely reused: a reasoning model holds
            // one request open for minutes.
            "reasoning" => Some(Self {
                request_timeout_secs: 900,
                pool_idle_timeout_secs: 10,
                retry_attempts: 2,
                retry_backoff_ms: 500,
                // A reasoning model can spend minutes deciding before it writes
                // anything, so the silence bound is generous rather than tight.
                stream_idle_secs: 180,
                // The catalog is one small JSON body either way; a reasoning
                // model's patience says nothing about how long a list takes.
                catalog_timeout_secs: 20,
            }),
            // Nothing to retry and nothing to wait for.
            "local" => Some(Self {
                request_timeout_secs: 120,
                pool_idle_timeout_secs: 60,
                retry_attempts: 1,
                retry_backoff_ms: 50,
                stream_idle_secs: 15,
                catalog_timeout_secs: 5,
            }),
            // A gateway that reaps idle connections aggressively, behind a
            // network that drops them.
            "flaky" => Some(Self {
                request_timeout_secs: 300,
                pool_idle_timeout_secs: 10,
                retry_attempts: 5,
                retry_backoff_ms: 300,
                stream_idle_secs: 30,
                catalog_timeout_secs: 15,
            }),
            "default" => Some(Self::default()),
            _ => None,
        }
    }

    /// Every preset name.
    pub fn presets() -> &'static [&'static str] {
        &["default", "reasoning", "local", "flaky"]
    }

    /// Problems that would make these values behave unlike their names.
    pub fn problems(&self) -> Vec<String> {
        let mut problems = Vec::new();
        if self.request_timeout_secs == 0 {
            problems.push("requestTimeoutSecs is 0; a request would never be sent".into());
        }
        if self.pool_idle_timeout_secs == 0 {
            problems.push("poolIdleTimeoutSecs is 0; no connection could ever be reused".into());
        }
        if self.retry_attempts == 0 {
            problems.push(
                "retryAttempts is 0; a connection dropped before the response ends the turn"
                    .into(),
            );
        }
        if self.retry_attempts > 10 {
            problems.push(format!(
                "retryAttempts is {}; against a provider that is down this is a long wait",
                self.retry_attempts
            ));
        }
        if self.stream_idle_secs >= self.request_timeout_secs {
            problems.push(format!(
                "streamIdleSecs is {} but requestTimeoutSecs is {}; the stall detector would only ever fire after the request itself is gone",
                self.stream_idle_secs, self.request_timeout_secs
            ));
        }
        if self.catalog_timeout_secs >= self.request_timeout_secs {
            problems.push(format!(
                "catalogTimeoutSecs is {} but requestTimeoutSecs is {}; a model list that never arrives would still wait for the whole request timeout, which is the hang this bound exists to remove",
                self.catalog_timeout_secs, self.request_timeout_secs
            ));
        }
        problems
    }
}

/// How a plugin overrides the numbers.
///
/// Published as the [`SERVICE`]. A host that finds it uses it in place of the
/// settings, so a plugin that knows a provider's behaviour can fix it for
/// everyone using that provider.
pub trait NetworkPolicy: Send + Sync {
    /// The numbers to use.
    fn settings(&self) -> NetworkSettings;
}

/// The published handle, and the type a plugin downcasts to.
///
/// A wrapper rather than `Arc<dyn NetworkPolicy>` directly, because a service
/// table stores `Arc<dyn Any>` and Rust cannot coerce one trait object into
/// another.
pub struct NetworkHandle(pub std::sync::Arc<dyn NetworkPolicy>);

impl NetworkHandle {
    /// Look the service up in a realm's view.
    pub fn from_view(view: &crate::plugin::ServiceView) -> Option<NetworkSettings> {
        view.get::<NetworkHandle>(SERVICE)
            .map(|handle| handle.0.settings())
    }
}

/// A fixed policy, for tests and for a host with nothing to override.
pub struct Fixed(pub NetworkSettings);

impl NetworkPolicy for Fixed {
    fn settings(&self) -> NetworkSettings {
        self.0.clone()
    }
}

/// A built-in plugin that publishes a network policy.
pub struct NetworkService {
    settings: NetworkSettings,
}

impl NetworkService {
    /// Publish these numbers.
    pub fn new(settings: NetworkSettings) -> Self {
        Self { settings }
    }

    /// The plugin name, used to load it.
    pub const NAME: &'static str = "network";
}

impl crate::plugin::Plugin for NetworkService {
    fn name(&self) -> &str {
        Self::NAME
    }

    fn provide(&self) -> Vec<String> {
        vec![SERVICE.to_string()]
    }

    fn apply(&self, _ctx: &crate::plugin::PluginCtx) -> anyhow::Result<crate::plugin::Contributions> {
        let handle: std::sync::Arc<dyn std::any::Any + Send + Sync> =
            std::sync::Arc::new(NetworkHandle(std::sync::Arc::new(Fixed(
                self.settings.clone(),
            ))));
        Ok(crate::plugin::Contributions::new().service(SERVICE, handle))
    }
}

/// Register a network policy in a kernel, unless one is already provided.
pub fn install(
    kernel: &mut crate::plugin::Kernel,
    settings: NetworkSettings,
) -> anyhow::Result<()> {
    // A plugin may have loaded first and published its own; the kernel's
    // conflict policy would fail the fiber, which would leave the built-in
    // service absent rather than the plugin's.
    if kernel.service_view(crate::plugin::RealmMap::new()).has(SERVICE) {
        return Ok(());
    }
    kernel.define(std::sync::Arc::new(NetworkService::new(settings)));
    let _ = kernel.load(NetworkService::NAME, crate::plugin::RealmMap::new(), serde_json::Value::Null);
    Ok(())
}

