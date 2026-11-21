//! What a plugin sees and may change while it loads.
//!
//! [`PluginCtx`] carries the realm map, a read-only [`ServiceView`], a writable
//! [`StateHandle`] over the same table, and the event bus. Applying a plugin
//! returns a [`Contributions`] value that the kernel installs, which is what
//! makes rollback possible.

use crate::tools::ToolDef;
use serde_json::Value;
use std::any::Any;
use std::sync::atomic::Ordering;
use std::sync::{Arc, RwLock};

use super::realm::RealmMap;
use super::service::{ServiceEntry, ServiceTable, ServiceView};

/// A reversible side effect: calling it undoes whatever was registered.
pub type Disposer = Box<dyn FnOnce() + Send>;

/// What a plugin contributes when it loads.
#[derive(Default)]
pub struct Contributions {
    /// Interface panels this plugin contributes.
    pub ui: Vec<crate::ui::UiPanel>,
    /// Theme layers this plugin contributes.
    pub themes: Vec<crate::theme::Theme>,
    /// Interface labels this plugin contributes, by element id.
    ///
    /// Text rather than colour: a pack that translates the shell says what each
    /// element reads, and the shell applies it without knowing any language.
    pub strings: std::collections::BTreeMap<String, String>,
    /// Stylesheet rules this plugin contributes, appended after the interface's
    /// own sheet.
    ///
    /// Text rather than a file: a pack says which rules it wants and the shell
    /// appends them, so a small restyle never has to ship a whole page. See
    /// `LookFile::css` for why this exists.
    pub css: Vec<String>,
    /// Named interface points this plugin hangs code on.
    ///
    /// Keyed by point name, holding the script text; the page evaluates it once
    /// after load. Empty for a plugin that only restyles. See `LookFile::hooks`.
    pub hooks: std::collections::BTreeMap<String, String>,
    /// Panel kinds this plugin defines, by name. See `LookFile::types`.
    pub panel_types: std::collections::BTreeMap<String, crate::pack::PanelType>,
    /// Services this plugin provides, by name.
    pub services: Vec<(String, Arc<dyn Any + Send + Sync>)>,
    /// Tools this plugin registers.
    pub tools: Vec<ToolDef>,
    /// Readers this plugin contributes: byte-sniffers that teach `read` a
    /// format the kernel does not parse — 读不了就装包, the pack provides
    /// the format, the kernel only provides the asking.
    pub readers: Vec<crate::tools::readers::ReaderSpec>,
    /// Cleanup for anything else the plugin started.
    pub effects: Vec<Disposer>,
}

impl Contributions {
    /// An empty contribution set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a service.
    pub fn service(mut self, name: impl Into<String>, value: Arc<dyn Any + Send + Sync>) -> Self {
        self.services.push((name.into(), value));
        self
    }

    /// Add a tool.
    pub fn tool(mut self, def: ToolDef) -> Self {
        self.tools.push(def);
        self
    }

    /// Contribute a reader for one file format.
    ///
    /// The kernel stamps the owner (this plugin's name) and keeps the
    /// registration in the fiber's effects, so unloading the pack unregisters
    /// the reader.
    pub fn reader(mut self, spec: crate::tools::readers::ReaderSpec) -> Self {
        self.readers.push(spec);
        self
    }

    /// Contribute a theme layer.
    ///
    /// A plugin may contribute several: a pack ships a light and a dark
    /// variant, and which one is active is the reader's choice, not the
    /// author's.
    pub fn theme(mut self, theme: crate::theme::Theme) -> Self {
        self.themes.push(theme);
        self
    }

    /// Contribute an interface panel.
    pub fn ui(mut self, panel: crate::ui::UiPanel) -> Self {
        self.ui.push(panel);
        self
    }

    /// Contribute interface labels.
    ///
    /// Merged rather than replaced, so a plugin may contribute two maps in one
    /// `apply` and the later entries win over the earlier ones.
    pub fn strings<I: IntoIterator<Item = (String, String)>>(mut self, map: I) -> Self {
        self.strings.extend(map);
        self
    }

    /// Add a stylesheet fragment, appended after the interface's own sheet.
    pub fn css(mut self, rules: impl Into<String>) -> Self {
        self.css.push(rules.into());
        self
    }

    /// Define panel kinds other panels can ask for by name.
    pub fn panel_types(
        mut self,
        map: std::collections::BTreeMap<String, crate::pack::PanelType>,
    ) -> Self {
        self.panel_types.extend(map);
        self
    }

    /// Hang scripts on named interface points.
    pub fn hooks(mut self, map: std::collections::BTreeMap<String, String>) -> Self {
        self.hooks.extend(map);
        self
    }

    /// Add a disposer.
    pub fn effect(mut self, disposer: impl FnOnce() + Send + 'static) -> Self {
        self.effects.push(Box::new(disposer));
        self
    }
}

/// A plugin's write access to the kernel's shared state.
///
/// The service table was already the one open slot in the kernel — a name and an
/// `Arc<dyn Any>` — but it was readable only. That made the kernel's own state
/// (settings, route, persona, the loop) visible to a pack and unreachable: a pack
/// could *see* the model route and not change it, and the only way to change it
/// was to edit the kernel.
///
/// This is the same table with its lock opened for writing. It is deliberately
/// not a new abstraction: [`ServiceTable`] already had `write()`, used by the
/// kernel itself when a fiber provides a service. Handing the same lock to a
/// plugin is the difference between "the kernel lets you look" and "the kernel
/// lets you change it".
///
/// There is no permission check here and none anywhere else in the kernel: a
/// pack runs with the same authority the kernel does, which is the whole premise
/// of a pack making the kernel do something the kernel did not know about.
#[derive(Clone)]
pub struct StateHandle {
    table: Arc<ServiceTable>,
    /// The plugin doing the writing, recorded on every entry it installs.
    pub owner: String,
    /// Realm new entries land in.
    realm: RealmMap,
}

impl StateHandle {
    pub(crate) fn of(table: Arc<ServiceTable>, owner: impl Into<String>, realm: RealmMap) -> Self {
        Self {
            table,
            owner: owner.into(),
            realm,
        }
    }

    /// A handle over a brand-new table, for a plugin applied outside a kernel.
    ///
    /// `ngu plugin load` and the tests apply a plugin without assembling one,
    /// and a plugin that writes state still needs somewhere to write it. The
    /// table is fresh and empty, so the writes are visible to nothing — which
    /// is the honest answer for a plugin applied in isolation.
    pub fn detached(owner: impl Into<String>) -> Self {
        Self::of(Arc::new(ServiceTable::default()), owner, RealmMap::new())
    }

    /// Install or replace a service under `name`.
    ///
    /// Replacing is not an error: the later writer wins, whether that writer is
    /// a second pack or the same pack reloading. Two packs that both claim
    /// `settings` is a real collision, and the honest answer is "the last one
    /// loaded owns it" rather than a kernel that refuses to start.
    pub fn set<T: Any + Send + Sync>(&self, name: &str, value: T) {
        self.insert_arc(name, Arc::new(value));
    }

    /// Install or replace a service that is already behind an `Arc`.
    pub fn insert_arc(&self, name: &str, value: Arc<dyn Any + Send + Sync>) {
        let real = self.realm.realm_of(name);
        let mut guard = self.table.write();
        let serial = guard
            .get(&(name.to_string(), real))
            .map(|entry| entry.serial + 1)
            .unwrap_or(1);
        guard.insert(
            (name.to_string(), real),
            ServiceEntry {
                value,
                fiber: 0,
                owner: self.owner.clone(),
                serial,
            },
        );
        self.table.generation.fetch_add(1, Ordering::Relaxed);
    }

    /// Remove a service by name. Reports whether it was there.
    pub fn remove(&self, name: &str) -> bool {
        let real = self.realm.realm_of(name);
        let existed = self
            .table
            .write()
            .remove(&(name.to_string(), real))
            .is_some();
        if existed {
            self.table.generation.fetch_add(1, Ordering::Relaxed);
        }
        existed
    }

    /// Replace whatever `T` is stored under `name`, in place.
    ///
    /// The ordinary shape of "a pack changes the kernel's state": the value is
    /// behind an `RwLock` the kernel itself reads, so a pack mutates it without
    /// the kernel having to know which pack did.
    ///
    /// `false` when nothing is stored under that name, or when it holds a
    /// different type — both are reported rather than papered over, because a
    /// pack that expected to change the settings and silently did not is the
    /// failure this method exists to make visible.
    pub fn update<T: Any + Send + Sync>(&self, name: &str, change: impl FnOnce(&mut T)) -> bool {
        // Two shapes, because both are ordinary. A host that keeps its own
        // `Arc<RwLock<T>>` and publishes the same allocation (so a pack's write
        // lands in the memory the host reads) stores the `Arc`; a plugin that
        // simply calls `set` stores the bare `RwLock`. Accepting either is what
        // lets a pack change the kernel's state without knowing which way the
        // host published it.
        // The host's shape: it kept its own `Arc<RwLock<T>>` and published the
        // same allocation, so this write lands in the memory the host reads.
        if let Some(lock) = self.get::<Arc<RwLock<T>>>(name) {
            let mut guard = lock
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            change(&mut guard);
            return true;
        }
        // A plugin's shape: it called `set` with a bare `RwLock`.
        if let Some(lock) = self.get::<RwLock<T>>(name) {
            let mut guard = lock
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            change(&mut guard);
            return true;
        }
        false
    }

    /// Read a service the kernel installed.
    /// Read a service the kernel installed.
    ///
    /// Two shapes, because the host publishes one and the table stores another:
    /// a host that keeps its own handle stores `Arc<RwLock<T>>`, while a plugin
    /// that simply called `set` stores the value it was given. Asking for `T`
    /// finds either — the `Arc` wrapper is unwrapped for you — so a pack does not
    /// have to know which way the kernel published the thing it wants to read.
    pub fn get<T: Any + Send + Sync>(&self, name: &str) -> Option<Arc<T>> {
        let real = self.realm.realm_of(name);
        let entry = self
            .table
            .read()
            .get(&(name.to_string(), real))
            .map(|entry| Arc::clone(&entry.value))?;
        if let Ok(theirs) = Arc::clone(&entry).downcast::<T>() {
            return Some(theirs);
        }
        // The value behind one more `Arc`: what `publish_state` makes from a
        // host's own `Arc<RwLock<T>>`.
        entry
            .downcast::<Arc<T>>()
            .ok()
            .map(|inner| Arc::clone(inner.as_ref()))
    }

    /// Every name currently in the table.
    pub fn names(&self) -> Vec<String> {
        self.table
            .read()
            .keys()
            .map(|(name, _)| name.clone())
            .collect()
    }
}

/// What a plugin can see while it loads.
pub struct PluginCtx {
    /// Plugin name.
    pub plugin: String,
    /// Fiber id of this instance.
    pub fiber: u64,
    /// The isolation map in force for this plugin.
    pub realm: RealmMap,
    /// Read-only service lookup, already scoped to `realm`.
    pub services: ServiceView,
    /// **Write** access to the same table.
    ///
    /// `services` answers "what is there"; this answers "make it different".
    /// Every kernel state worth changing — the settings, the model route, the
    /// loop, the tool table — is stored as a service behind an `RwLock`, so a
    /// pack changes it here and the kernel reads the new value on its next turn
    /// without knowing which pack did it.
    ///
    /// Adding this field is the whole of "a pack can modify the kernel": the
    /// table, the lock, and the generation counter already existed.
    pub state: StateHandle,
    /// Plugin configuration.
    pub config: Value,
    /// The event bus: observational subscription only (see `events.rs`) —
    /// a plugin watches facts, it cannot veto them.
    pub events: crate::events::EventBus,
}

impl PluginCtx {
    /// A context for a plugin applied outside a kernel.
    ///
    /// One fresh table shared by `services` and `state`, so a plugin that writes
    /// and then reads sees its own write — which is what applying it in
    /// isolation is supposed to mean.
    pub fn detached(plugin: impl Into<String>) -> Self {
        let plugin = plugin.into();
        let table = Arc::new(ServiceTable::default());
        let realm = RealmMap::new();
        Self {
            services: ServiceView::of(Arc::clone(&table), realm.clone()),
            state: StateHandle::of(table, plugin.clone(), realm.clone()),
            plugin,
            fiber: 0,
            realm,
            config: Value::Null,
            events: crate::events::EventBus::default(),
        }
    }

    /// Look up a service visible to this plugin.
    pub fn get<T: Any + Send + Sync>(&self, name: &str) -> Option<Arc<T>> {
        self.services.get::<T>(name)
    }

    /// Change a service the kernel installed, in place.
    ///
    /// The one call a pack needs to retune the kernel: the value is behind the
    /// kernel's own `RwLock`, so this is the same mutation the kernel would make
    /// itself. `false` when the name holds nothing of that type.
    pub fn change<T: Any + Send + Sync>(&self, name: &str, f: impl FnOnce(&mut T)) -> bool {
        self.state.update::<T>(name, f)
    }

    /// Whether a service is visible.
    pub fn has(&self, name: &str) -> bool {
        self.services.has(name)
    }

    /// Subscribe to one event; put the disposer in `Contributions::effect`
    /// so unloading actually unsubscribes (effects are undone, not dropped).
    pub fn observe(
        &self,
        event: &str,
        observer: impl Fn(&str, &Value) + Send + Sync + 'static,
    ) -> crate::events::Disposer {
        self.events.subscribe(event, observer)
    }
}
