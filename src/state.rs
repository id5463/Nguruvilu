//! The kernel's own state, published where a pack can change it.
//!
//! The kernel keeps several things a conversation actually runs on: the settings
//! (route, budgets, persona-adjacent knobs), the turn's model route, the persona
//! text, and the loop. Before this module those lived in structs the host owned
//! and a pack could not reach — a pack could register a tool and nothing else.
//! "A pack can modify the kernel" was not true, and the workaround was to edit
//! the kernel.
//!
//! # The shape
//!
//! Each piece is published as a **service** under a fixed name, behind the
//! kernel's own `RwLock`. The service table already existed and already stores
//! `Arc<dyn Any>`; the only thing missing was that it was read-only for plugins.
//!
//! So a pack does:
//!
//! ```ignore
//! ctx.change::<Settings>(state::SETTINGS, |s| s.reasoning_effort = "high".into());
//! ctx.change::<ModelRoute>(state::ROUTE, |r| r.model = "other".into());
//! ctx.state.set(state::PERSONA, RwLock::new("new persona".to_string()));
//! ```
//!
//! and the kernel reads the new value on its next turn without knowing which
//! pack wrote it. Nothing here names a capability: the names below are the
//! kernel's *own* state, not a catalog of what packs may do.
//!
//! # Why names rather than types
//!
//! A service is looked up by string. That is deliberate — it is the same
//! mechanism `llm` and `network` already use, so a pack learns one rule. A pack
//! that wants to publish its own state picks its own name and uses the identical
//! call.

use std::sync::{Arc, RwLock};

use crate::hotreload::ModelRoute;
use crate::plugin::Kernel;
use crate::settings::Settings;

/// The live settings, behind one lock.
pub const SETTINGS: &str = "settings";

/// The turn's model route. Derived from the settings, but published separately
/// because a pack that only wants to change the endpoint should not have to
/// rebuild the whole settings struct.
pub const ROUTE: &str = "route";

/// The persona text the system prompt is built from.
pub const PERSONA: &str = "persona";

/// The model route a turn runs on.
///
/// `ModelRoute` comes from `hotreload` because that is where the hot-reload path
/// already keeps it; publishing the same type means a pack's change and a
/// settings save cannot disagree about the shape.
pub type Route = ModelRoute;

/// A persona the kernel reads on every turn.
///
/// Wrapped because the text is replaced wholesale rather than edited: a pack
/// either is the persona or is not, and a partial write would leave the prompt
/// half of each.
#[derive(Debug, Clone, Default)]
pub struct Persona(pub String);

impl Persona {
    /// A persona from its text.
    pub fn new(text: impl Into<String>) -> Self {
        Self(text.into())
    }

    /// The text.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// What the kernel publishes, and the handles a host keeps.
///
/// The handles are `Arc<RwLock<_>>` rather than the values: the kernel's own
/// readers take them every turn, and a pack's write must be visible to that
/// same lock rather than to a copy.
#[derive(Clone)]
pub struct Published {
    /// Live settings.
    pub settings: Arc<RwLock<Settings>>,
    /// The route a turn runs on.
    pub route: Arc<RwLock<Route>>,
    /// The persona text.
    pub persona: Arc<RwLock<Persona>>,
}

impl Published {
    /// Wrap the host's values and publish them into `kernel`.
    ///
    /// Also returns the handles, because the host keeps reading them directly:
    /// routing every settings read through the service table would make the
    /// kernel's hot path depend on a string lookup.
    pub fn install(kernel: &mut Kernel, settings: Settings) -> Self {
        let published = Self {
            settings: Arc::new(RwLock::new(settings.clone())),
            route: Arc::new(RwLock::new(Route::from_settings(&settings))),
            persona: Arc::new(RwLock::new(Persona::default())),
        };
        published.publish(kernel);
        published
    }

    /// Push the current values into the kernel's service table.
    ///
    /// Called at startup and again after anything the host itself changed, so a
    /// pack always finds a live value rather than a stale copy.
    pub fn publish(&self, kernel: &mut Kernel) {
        // `_arc`, not the plain call: the host already holds these behind an
        // `Arc` and publishes *that same allocation*, so a pack's write lands in
        // the memory the host reads. `publish_state` would wrap it again
        // (`Arc<Arc<RwLock<_>>>`) and no reader would find it.
        kernel.publish_state_arc(SETTINGS, Arc::clone(&self.settings) as Arc<dyn std::any::Any + Send + Sync>);
        kernel.publish_state_arc(ROUTE, Arc::clone(&self.route) as Arc<dyn std::any::Any + Send + Sync>);
        kernel.publish_state_arc(PERSONA, Arc::clone(&self.persona) as Arc<dyn std::any::Any + Send + Sync>);
    }

    /// The settings as they are right now.
    pub fn settings(&self) -> Settings {
        self.settings
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plugin::{PluginCtx, RealmMap};

    fn kernel() -> (Kernel, Published) {
        let mut kernel = Kernel::new();
        let published = Published::install(&mut kernel, Settings::default());
        (kernel, published)
    }

    #[test]
    fn a_pack_changes_the_settings_through_the_service_table() {
        let (mut kernel, published) = kernel();
        kernel.define(Arc::new(Changer));
        kernel.load("changer", RealmMap::new(), serde_json::Value::Null).unwrap();

        // The pack wrote through the same lock the host holds.
        assert_eq!(published.settings().reasoning_effort, "high");
    }

    #[test]
    fn a_pack_changes_the_route() {
        let (mut kernel, published) = kernel();
        kernel.define(Arc::new(Changer));
        kernel.load("changer", RealmMap::new(), serde_json::Value::Null).unwrap();

        let route = published.route.read().unwrap();
        assert_eq!(route.model, "pack-chosen");
    }

    #[test]
    fn a_write_to_a_name_nobody_published_is_reported_not_swallowed() {
        let (mut kernel, _published) = kernel();
        kernel.define(Arc::new(Changer));
        kernel.load("changer", RealmMap::new(), serde_json::Value::Null).unwrap();
        // `Changer::apply` asserted the missing name came back `false`. If
        // `change` had claimed success for a name nobody published, that
        // assertion would have failed the load.
    }

    /// A pack that reaches into the kernel and changes it.
    struct Changer;

    impl crate::plugin::Plugin for Changer {
        fn name(&self) -> &str {
            "changer"
        }

        fn apply(&self, ctx: &PluginCtx) -> anyhow::Result<crate::plugin::Contributions> {
            assert!(ctx.change::<Settings>(SETTINGS, |s| s.reasoning_effort = "high".into()));
            assert!(ctx.change::<Route>(ROUTE, |r| r.model = "pack-chosen".into()));
            assert!(ctx.state.update::<Persona>(PERSONA, |p| p.0 = "pack persona".into()));
            assert!(!ctx.change::<Settings>("nothing-here", |_| {}));
            Ok(crate::plugin::Contributions::new())
        }
    }
}
