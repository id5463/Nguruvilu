//! What a plugin needs to run its own agent.
//!
//! A subagent is not special machinery — it is another [`crate::agent::Agent`]
//! with its own history. What a plugin cannot supply itself is the *session's*
//! configuration: which endpoint, which key, which model, which tools, and what
//! standing instructions are in force. Duplicating those into plugin config
//! would mean a second set of credentials to keep in step, and a subagent
//! running on a different model than the user chose without saying so.
//!
//! [`ModelAccess`] publishes exactly that, and nothing else. A plugin asks for
//! the current route and builds an agent on it.
//!
//! # Why a trait rather than the concrete types
//!
//! The kernel does not own the route — the host does, and it changes as the user
//! edits settings. The host implements this over whatever it already has, and
//! the values are read at call time so a settings change is picked up by the
//! next subagent rather than the next restart.

use std::sync::Arc;

use anyhow::Result;

use crate::hotreload::ModelRoute;
use crate::llm::LlmClient;
use crate::tools::ToolRegistry;
use crate::window::ContextPolicy;

/// The session's model route and the context a nested agent needs.
pub trait ModelAccess: Send + Sync {
    /// The route in force right now.
    fn route(&self) -> ModelRoute;

    /// A client for the current route.
    fn client(&self) -> Result<LlmClient>;

    /// A client for a different model on the same route.
    ///
    /// A subagent doing a mechanical subtask usually wants a cheaper model than
    /// the one the conversation is using, and it should not need different
    /// credentials to get one.
    fn client_for(&self, model: &str) -> Result<LlmClient>;

    /// The tool table a subagent may use.
    ///
    /// The same table the session uses, so a subagent is bounded by exactly the
    /// permissions the session has — it cannot reach a tool the user disabled.
    fn tools(&self) -> Arc<ToolRegistry>;

    /// Standing instructions for a nested agent.
    fn system_prompt(&self) -> String;

    /// The context policy in force, so a subagent compacts like its parent.
    fn context_policy(&self) -> Arc<dyn ContextPolicy>;
}

/// Take a lock without panicking on poisoning.
///
/// A `std::sync::Mutex` is poisoned when a thread panicked while holding it, and
/// `.expect("…lock")` turns that one panic into a second panic on *every* later
/// read — for the route, tools, prompt and policy locks that means every
/// subsequent turn fails, not the one step that panicked. None of those locks
/// guards an invariant a panic could have broken (they hold plain snapshots), so
/// recovering the guard is the right answer. Use this at every host call site
/// instead of `.lock().expect(…)`.
pub fn lock_recovering<T>(mutex: &std::sync::Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Reject a base URL that would put the credentials on the wire in clear text.
///
/// The route reaches the provider through [`crate::llm::LlmConfig`], which sends
/// the key as a `bearer_auth` header. Nothing between the settings file and that
/// header looks at the scheme, so `http://api.example/v1` — a typo, or a URL
/// copied from a gateway's docs — ships the key in clear text to every hop on the
/// path. Loopback is exempt: a gateway on the same machine (`127.0.0.1`,
/// `localhost`, `[::1]`) never leaves it. An empty URL is not an error, because
/// "not configured" is reported by [`crate::settings::Settings::missing`].
pub fn check_base_url(base_url: &str) -> Result<()> {
    let trimmed = base_url.trim();
    if trimmed.is_empty() {
        return Ok(());
    }
    let Some((scheme, rest)) = trimmed.split_once("://") else {
        return Err(anyhow::anyhow!(
            "base URL '{trimmed}' has no scheme; expected https://host/path"
        ));
    };
    let scheme = scheme.trim().to_ascii_lowercase();
    if scheme == "https" {
        return Ok(());
    }
    if scheme != "http" {
        return Err(anyhow::anyhow!(
            "base URL scheme '{scheme}' is not supported; use https://"
        ));
    }

    // `http://user:pass@host/` — strip any userinfo *before* looking at the host,
    // or a password could make a remote endpoint look like `localhost`.
    let authority = rest.split(['/', '?', '#']).next().unwrap_or_default();
    let authority = authority.rsplit('@').next().unwrap_or(authority);
    let host = match authority.strip_prefix('[') {
        Some(bracketed) => bracketed.split(']').next().unwrap_or_default(),
        None => authority.split(':').next().unwrap_or_default(),
    };
    let host = host.trim_end_matches('.').to_ascii_lowercase();
    let loopback = host == "localhost"
        || host == "::1"
        || host
            .parse::<std::net::Ipv4Addr>()
            .map(|address| address.is_loopback())
            .unwrap_or(false);
    if loopback {
        return Ok(());
    }

    Err(anyhow::anyhow!(
        "base URL '{trimmed}' is http:// but not loopback, so the API key would travel in \
         clear text. Use https://, or a gateway on localhost"
    ))
}

/// The name a plugin looks the service up under.
pub const SERVICE: &str = "llm";

/// The published handle, and the type a plugin downcasts to.
///
/// A wrapper rather than `Arc<dyn ModelAccess>` directly, because a service
/// table stores `Arc<dyn Any>` and Rust cannot coerce one trait object into
/// another. The wrapper is a concrete type, so the coercion is a normal
/// unsizing.
pub struct ModelHandle(pub Arc<dyn ModelAccess>);

impl ModelHandle {
    /// Look the service up in a realm's view.
    pub fn from_view(view: &crate::plugin::ServiceView) -> Option<Arc<dyn ModelAccess>> {
        view.get::<ModelHandle>(SERVICE).map(|handle| Arc::clone(&handle.0))
    }
}

/// A built-in plugin that publishes the session's model access.
///
/// Registered by the host rather than by a plugin: it is how the kernel tells
/// plugins about the session, so it cannot itself come from one.
pub struct ModelService {
    access: Arc<dyn ModelAccess>,
}

impl ModelService {
    /// Wrap a host's access implementation.
    pub fn new(access: Arc<dyn ModelAccess>) -> Self {
        Self { access }
    }

    /// The plugin name, used to load it.
    pub const NAME: &'static str = "model";
}

impl crate::plugin::Plugin for ModelService {
    fn name(&self) -> &str {
        Self::NAME
    }

    fn provide(&self) -> Vec<String> {
        vec![SERVICE.to_string()]
    }

    fn apply(&self, _ctx: &crate::plugin::PluginCtx) -> Result<crate::plugin::Contributions> {
        let handle: Arc<dyn std::any::Any + Send + Sync> =
            Arc::new(ModelHandle(Arc::clone(&self.access)));
        Ok(crate::plugin::Contributions::new().service(SERVICE, handle))
    }
}

/// Register the model service in a kernel.
///
/// Called by the host once at startup. Loading it is not an error when it is
/// already present, so a host may call this unconditionally.
pub fn install(
    kernel: &mut crate::plugin::Kernel,
    access: Arc<dyn ModelAccess>,
) -> Result<u64> {
    kernel.define(Arc::new(ModelService::new(access)));
    let realm = crate::plugin::RealmMap::new();
    kernel.load(ModelService::NAME, realm, serde_json::Value::Null)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plugin::{Kernel, PluginCtx, RealmMap};

    /// A minimal access implementation, standing in for a host's.
    struct Fixed {
        route: ModelRoute,
        tools: Arc<ToolRegistry>,
    }

    impl ModelAccess for Fixed {
        fn route(&self) -> ModelRoute {
            self.route.clone()
        }

        fn client(&self) -> Result<LlmClient> {
            self.client_for(&self.route.model)
        }

        fn client_for(&self, model: &str) -> Result<LlmClient> {
            let mut config = crate::llm::LlmConfig::new(
                self.route.base_url.clone(),
                self.route.api_key.clone(),
                model,
            );
            config.proxy = self.route.proxy.clone().unwrap_or_default();
            LlmClient::new(config)
        }

        fn tools(&self) -> Arc<ToolRegistry> {
            Arc::clone(&self.tools)
        }

        fn system_prompt(&self) -> String {
            "subagent".to_string()
        }

        fn context_policy(&self) -> Arc<dyn ContextPolicy> {
            Arc::new(crate::window::DefaultContextPolicy::default())
        }
    }

    fn fixed() -> Arc<dyn ModelAccess> {
        Arc::new(Fixed {
            route: ModelRoute {
                provider: "openai".into(),
                base_url: "http://localhost:1/v1".into(),
                api_key: "k".into(),
                model: "m".into(),
                ..Default::default()
            },
            tools: Arc::new(ToolRegistry::with_base_tools().unwrap()),
        })
    }

    #[test]
    fn installing_publishes_the_service() {
        let mut kernel = Kernel::new();
        install(&mut kernel, fixed()).unwrap();

        let view = kernel.service_view(RealmMap::new());
        assert!(view.has(SERVICE), "a plugin can find it");
        assert!(ModelHandle::from_view(&view).is_some(), "and downcast it");
    }

    #[test]
    fn a_plugin_can_reach_the_sessions_route_through_it() {
        let mut kernel = Kernel::new();
        install(&mut kernel, fixed()).unwrap();

        let access = ModelHandle::from_view(&kernel.service_view(RealmMap::new()))
            .expect("published");

        assert_eq!(access.route().model, "m");
        assert!(access.client().is_ok());
        assert!(access.tools().get("read").is_some());
        assert_eq!(access.system_prompt(), "subagent");
    }

    #[test]
    fn a_client_for_another_model_keeps_the_same_route() {
        let access = fixed();
        let cheaper = access.client_for("small-model").unwrap();
        assert_eq!(cheaper.model(), "small-model");

        let default = access.client().unwrap();
        assert_eq!(default.model(), "m");
    }

    #[test]
    fn a_clear_text_base_url_is_refused_unless_it_stays_on_the_machine() {
        assert!(check_base_url("").is_ok(), "not configured is not an error");
        assert!(check_base_url("https://api.example.test/v1").is_ok());
        assert!(check_base_url("https://api.example.test:8443/v1").is_ok());
        assert!(check_base_url("http://127.0.0.1:11434/v1").is_ok());
        assert!(check_base_url("http://localhost:8080/v1").is_ok());
        assert!(check_base_url("http://[::1]:8080/v1").is_ok());
        assert!(check_base_url("http://127.0.0.5/v1").is_ok());

        for refused in [
            "http://api.example.test/v1",
            "http://192.168.1.5:8080/v1",
            "http://api.example.test:80/v1",
            // Userinfo must not be able to disguise a remote host as loopback.
            "http://user:pass@api.example.test/v1",
            "ftp://api.example.test/v1",
            "api.example.test/v1",
        ] {
            let error = check_base_url(refused)
                .expect_err(&format!("{refused} must be refused"));
            assert!(
                !format!("{error:#}").is_empty(),
                "the refusal explains itself: {error:#}"
            );
        }
        let error = check_base_url("http://api.example.test/v1").unwrap_err();
        assert!(
            format!("{error:#}").contains("clear text"),
            "the reason names the risk: {error:#}"
        );
    }

    #[test]
    fn a_poisoned_lock_is_recovered_instead_of_panicking_again() {
        let mutex = std::sync::Mutex::new(7_usize);
        let poisoned = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = mutex.lock().unwrap();
            panic!("a holder panicked");
        }));
        assert!(poisoned.is_err(), "the lock is poisoned now");
        assert!(mutex.lock().is_err(), "which .expect() would turn into a panic");

        let value = *lock_recovering(&mutex);
        assert_eq!(value, 7, "the guard is still usable");
    }

    #[test]
    fn the_service_is_named_for_lookup() {
        assert_eq!(SERVICE, "llm");
        assert_eq!(ModelService::NAME, "model");
    }

    #[test]
    fn the_plugin_declares_what_it_provides() {
        let service = ModelService::new(fixed());
        assert_eq!(
            crate::plugin::Plugin::provide(&service),
            vec!["llm".to_string()]
        );
    }

    #[test]
    fn applying_the_plugin_contributes_the_service() {
        let service = ModelService::new(fixed());
        let ctx = PluginCtx {
            plugin: "model".into(),
            fiber: 1,
            realm: RealmMap::new(),
            services: crate::plugin::ServiceView::default(),
            config: serde_json::Value::Null,
            events: crate::events::EventBus::default(),
        };
        let contributions = crate::plugin::Plugin::apply(&service, &ctx).unwrap();
        assert_eq!(contributions.services.len(), 1);
        assert_eq!(contributions.services[0].0, SERVICE);
    }

    #[test]
    fn installing_twice_is_reported_rather_than_silently_stacking() {
        let mut kernel = Kernel::new();
        install(&mut kernel, fixed()).unwrap();
        // The second load finds the service already provided and fails the
        // fiber, which is the kernel's ordinary conflict behaviour.
        let _ = install(&mut kernel, fixed());
        assert_eq!(kernel.service_list().len(), 1, "still exactly one provider");
    }
}
