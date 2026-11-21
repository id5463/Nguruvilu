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
    /// Standing instructions for a nested agent.
    fn system_prompt(&self) -> String;

    /// Standing instructions for a nested agent — the *base* rules only.
    ///
    /// A subagent needs the operating rules (its working directory convention,
    /// how to report results) and must not inherit the session's persona. The
    /// persona is written for the agent the user is talking to: it says "you are
    /// the planner, delegate what can be delegated", and a subtask that reads
    /// that as its own job will spend its budget handing work to further
    /// subagents instead of doing the task it was given. The framing paragraph
    /// in `tools::subagent` cannot undo it either, because it arrives after the
    /// system prompt rather than instead of it.
    ///
    /// Defaults to the full prompt so an embedder that has no separate base gets
    /// today's behaviour rather than an empty one.
    fn subagent_prompt(&self) -> String {
        self.system_prompt()
    }


    /// Model ids the route's endpoint offers, for a caller that must choose one.
    ///
    /// `delegate` used to describe its `model` argument as "optional, defaults to
    /// the session's" and nothing more: the model had to *guess* an id, and a
    /// guess that the endpoint does not know fails the subtask with a 404 that
    /// reads like a broken tool. The list is the difference between choosing and
    /// guessing.
    ///
    /// Defaults to empty, because an implementor that has no catalog to offer
    /// must still compile — and because "I do not know" is honest here, while a
    /// made-up list is not.
    fn available_models(&self) -> Vec<String> {
        Vec::new()
    }

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

/// A [`ModelAccess`] over a fixed route and tool table.
///
/// For a caller that is not running a turn: `ngu apply` loads a pack so the
/// caller can see what it contributes, and `delegate` injects the `llm` service,
/// so the service has to exist or the plugin loads and contributes nothing. This
/// publishes the route from the settings file and the kernel's own tool table —
/// enough for a plugin to apply, and nothing that could call a model.
pub struct StaticAccess {
    route: ModelRoute,
    tools: Arc<ToolRegistry>,
    prompt: String,
}

impl StaticAccess {
    /// Publish `route` and `tools` as they are.
    pub fn new(route: ModelRoute, tools: Arc<ToolRegistry>) -> Self {
        Self { route, tools, prompt: String::new() }
    }

    /// Override the standing instructions a nested agent would get.
    pub fn with_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt = prompt.into();
        self
    }
}

impl ModelAccess for StaticAccess {
    fn route(&self) -> ModelRoute {
        self.route.clone()
    }

    fn client(&self) -> Result<LlmClient> {
        self.route.client_for(&self.route.model)
    }

    fn client_for(&self, model: &str) -> Result<LlmClient> {
        self.route.client_for(model)
    }

    fn tools(&self) -> Arc<ToolRegistry> {
        Arc::clone(&self.tools)
    }

    fn system_prompt(&self) -> String {
        self.prompt.clone()
    }

    fn context_policy(&self) -> Arc<dyn crate::window::ContextPolicy> {
        Arc::new(crate::window::DefaultContextPolicy::default())
    }
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

