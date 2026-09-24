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
#[serde(rename_all = "camelCase", deny_unknown_fields)]
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

impl Default for NetworkSettings {
    fn default() -> Self {
        Self {
            request_timeout_secs: default_request_timeout(),
            pool_idle_timeout_secs: default_pool_idle(),
            retry_attempts: default_retry_attempts(),
            retry_backoff_ms: default_retry_backoff(),
        }
    }
}

impl NetworkSettings {
    /// The whole-request timeout.
    pub fn request_timeout(&self) -> Duration {
        Duration::from_secs(self.request_timeout_secs.max(1))
    }

    /// How long an idle connection is kept.
    pub fn pool_idle_timeout(&self) -> Duration {
        Duration::from_secs(self.pool_idle_timeout_secs.max(1))
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
            }),
            // Nothing to retry and nothing to wait for.
            "local" => Some(Self {
                request_timeout_secs: 120,
                pool_idle_timeout_secs: 60,
                retry_attempts: 1,
                retry_backoff_ms: 50,
            }),
            // A gateway that reaps idle connections aggressively, behind a
            // network that drops them.
            "flaky" => Some(Self {
                request_timeout_secs: 300,
                pool_idle_timeout_secs: 10,
                retry_attempts: 5,
                retry_backoff_ms: 300,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plugin::{Kernel, RealmMap};

    #[test]
    fn the_defaults_are_the_numbers_that_were_hardcoded() {
        let settings = NetworkSettings::default();
        assert_eq!(settings.request_timeout_secs, 300);
        assert_eq!(settings.pool_idle_timeout_secs, 30);
        assert_eq!(settings.retry_attempts, 3);
        assert_eq!(settings.retry_backoff_ms, 200);
    }

    #[test]
    fn the_pool_timeout_sits_below_a_typical_provider_idle_timeout() {
        // The whole point of the value: a client that keeps connections longer
        // than the server hands out dead ones.
        assert!(
            NetworkSettings::default().pool_idle_timeout_secs < 60,
            "most gateways reap idle connections at 30 to 60 seconds"
        );
    }

    #[test]
    fn backoff_doubles_and_then_stops_growing() {
        let settings = NetworkSettings {
            retry_backoff_ms: 100,
            ..Default::default()
        };
        assert_eq!(settings.backoff(1), Duration::from_millis(100));
        assert_eq!(settings.backoff(2), Duration::from_millis(200));
        assert_eq!(settings.backoff(3), Duration::from_millis(400));
        // Clamped rather than overflowing into a wait measured in centuries.
        assert!(settings.backoff(100) < Duration::from_secs(60 * 60));
    }

    #[test]
    fn a_zero_delay_is_kept_as_zero() {
        // A caller that asks for no pause means it.
        let settings = NetworkSettings {
            retry_backoff_ms: 0,
            ..Default::default()
        };
        assert_eq!(settings.backoff(1), Duration::ZERO);
    }

    #[test]
    fn the_values_round_trip_through_json() {
        let settings = NetworkSettings {
            request_timeout_secs: 900,
            pool_idle_timeout_secs: 5,
            retry_attempts: 7,
            retry_backoff_ms: 50,
        };
        let text = serde_json::to_string(&settings).unwrap();
        let back: NetworkSettings = serde_json::from_str(&text).unwrap();
        assert_eq!(back, settings);
    }

    #[test]
    fn an_absent_field_takes_its_default() {
        let text = "{\"retryAttempts\": 9}";
        let settings: NetworkSettings = serde_json::from_str(text).unwrap();
        assert_eq!(settings.retry_attempts, 9);
        assert_eq!(settings.request_timeout_secs, 300, "the rest default");
    }

    #[test]
    fn an_unknown_field_is_refused() {
        // A typo would otherwise be a setting that silently does nothing.
        let text = "{\"retryAttemps\": 9}";
        let error = serde_json::from_str::<NetworkSettings>(text).expect_err("must refuse");
        assert!(format!("{error}").contains("retryAttemps"), "{error}");
    }

    #[test]
    fn a_zero_or_absurd_value_is_reported() {
        let mut settings = NetworkSettings::default();
        assert!(settings.problems().is_empty());

        settings.retry_attempts = 0;
        assert!(settings.problems().iter().any(|p| p.contains("retryAttempts is 0")));

        settings.retry_attempts = 99;
        assert!(settings.problems().iter().any(|p| p.contains("long wait")));

        let mut settings = NetworkSettings::default();
        settings.request_timeout_secs = 0;
        assert!(settings.problems().iter().any(|p| p.contains("never be sent")));
    }

    #[test]
    fn a_preset_exists_for_each_name_it_offers() {
        for name in NetworkSettings::presets() {
            assert!(
                NetworkSettings::preset(name).is_some(),
                "{name} is listed but not defined"
            );
        }
        assert!(NetworkSettings::preset("nonsense").is_none());
    }

    #[test]
    fn the_local_preset_does_not_retry_much() {
        // A local model that is not running is not going to start.
        let local = NetworkSettings::preset("local").unwrap();
        assert_eq!(local.retry_attempts, 1);
        assert!(local.request_timeout_secs < 300);
    }

    #[test]
    fn the_flaky_preset_retries_more_and_holds_connections_less() {
        let flaky = NetworkSettings::preset("flaky").unwrap();
        let default = NetworkSettings::default();
        assert!(flaky.retry_attempts > default.retry_attempts);
        assert!(flaky.pool_idle_timeout_secs < default.pool_idle_timeout_secs);
    }

    #[test]
    fn the_reasoning_preset_allows_a_long_request() {
        let reasoning = NetworkSettings::preset("reasoning").unwrap();
        assert!(reasoning.request_timeout_secs >= 900);
    }

    #[test]
    fn a_published_policy_is_found_through_the_service() {
        let policy = NetworkSettings {
            retry_attempts: 11,
            ..Default::default()
        };
        let mut kernel = Kernel::new();
        install(&mut kernel, policy.clone()).unwrap();

        let found = NetworkHandle::from_view(&kernel.service_view(RealmMap::new()));
        assert_eq!(found.unwrap().retry_attempts, 11);
    }

    #[test]
    fn installing_twice_keeps_the_first_policy() {
        // A plugin that published its own first must not be replaced by the
        // built-in, which would silently undo the plugin.
        let mut kernel = Kernel::new();
        install(&mut kernel, NetworkSettings { retry_attempts: 2, ..Default::default() }).unwrap();
        install(&mut kernel, NetworkSettings { retry_attempts: 9, ..Default::default() }).unwrap();

        let found = NetworkHandle::from_view(&kernel.service_view(RealmMap::new())).unwrap();
        assert_eq!(found.retry_attempts, 2);
    }

    #[test]
    fn an_absent_policy_leaves_the_service_empty() {
        let kernel = Kernel::new();
        assert!(NetworkHandle::from_view(&kernel.service_view(RealmMap::new())).is_none());
    }
}
