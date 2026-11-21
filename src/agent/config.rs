//! How a turn is configured.
//!
//! [`TurnSettings`] is one turn's frozen snapshot, [`TurnConfig`] is the source
//! it is frozen from, and [`StaticConfig`] is that source for a build with no
//! hot-reload runtime behind it.

use crate::tools::ToolRegistry;
use crate::window::ContextPolicy;
use std::sync::Arc;

use super::hooks::PluginHooks;
use crate::window;

/// Everything one turn needs, frozen at the start of that turn.
#[derive(Clone)]
pub struct TurnSettings {
    /// System prompt for this turn.
    pub system_prompt: String,
    /// Tool table for this turn.
    pub tools: Arc<ToolRegistry>,
    /// Model id for this turn.
    pub model: String,
    /// Configuration version this snapshot came from.
    pub version: u64,
    /// Context window, compaction threshold, and how much survives a compaction.
    ///
    /// It travels with the rest of the turn's settings so that changing the
    /// window in a settings panel takes effect on the next turn rather than
    /// requiring a restart.
    pub policy: Arc<dyn ContextPolicy>,
    /// Extra fragments to place in this request, and where.
    pub injection: Arc<dyn crate::context::InjectionSource>,
    /// Cache policy, which decides whether a prefix injection is allowed.
    pub cache_policy: crate::hotreload::CachePolicy,
    /// Ask the loaded plugins what they want done for this turn.
    ///
    /// A trait object rather than a kernel reference: the loop must not hold
    /// the kernel while a turn runs, and a test can substitute one with no
    /// kernel at all. `None` means "no plugins", which is the headless case.
    pub plugin_hooks: Option<Arc<dyn PluginHooks>>,
}

/// Supplies turn settings.
///
/// The runtime implements this so a change lands on the *next* turn. Nothing
/// in the loop reads mutable configuration directly.
pub trait TurnConfig: Send + Sync {
    /// Freeze the settings for one turn.
    fn settings(&self) -> TurnSettings;
}

/// A configuration that never changes.
///
/// Useful for tests, for embedding the kernel without a runtime, and as the
/// fallback when no hot-reload runtime is in play.
pub struct StaticConfig {
    /// System prompt.
    pub system_prompt: String,
    /// Tool table.
    pub tools: Arc<ToolRegistry>,
    /// Model id.
    pub model: String,
    /// Context window, threshold, and how much survives a compaction.
    pub policy: Arc<dyn ContextPolicy>,
    /// Extra fragments to place in each request.
    pub injection: Arc<dyn crate::context::InjectionSource>,
    /// Cache policy, which decides whether a prefix injection is allowed.
    pub cache_policy: crate::hotreload::CachePolicy,
}

impl StaticConfig {
    /// Build a static configuration.
    pub fn new(
        system_prompt: impl Into<String>,
        tools: Arc<ToolRegistry>,
        model: impl Into<String>,
    ) -> Self {
        Self {
            system_prompt: system_prompt.into(),
            tools,
            model: model.into(),
            policy: Arc::new(window::DefaultContextPolicy::default()),
            injection: Arc::new(crate::context::InjectionEngine::new()),
            cache_policy: crate::hotreload::CachePolicy::default(),
        }
    }

    /// Use a specific context policy.
    pub fn with_policy(mut self, policy: Arc<dyn ContextPolicy>) -> Self {
        self.policy = policy;
        self
    }
}

impl TurnConfig for StaticConfig {
    fn settings(&self) -> TurnSettings {
        TurnSettings {
            system_prompt: self.system_prompt.clone(),
            tools: Arc::clone(&self.tools),
            model: self.model.clone(),
            version: 0,
            policy: Arc::clone(&self.policy),
            injection: Arc::clone(&self.injection),
            cache_policy: self.cache_policy,
            // A static config carries no kernel, so nothing is asked. Tests and
            // embeddings get the loop without plugins, which is the point of
            // the static shape.
            plugin_hooks: None,
        }
    }
}

/// A turn's settings held still, for a loop that has already been given them.
pub(crate) struct StaticStateConfig {
    pub(crate) settings: TurnSettings,
}

impl TurnConfig for StaticStateConfig {
    fn settings(&self) -> TurnSettings {
        self.settings.clone()
    }
}
