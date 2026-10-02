//! The plugin mechanism: isolation, dependency tracking, reversible effects.
//!
//! Design follows Cordis (`vendor/cordis/src/`, MIT) but is rewritten for
//! Rust's ownership model, which changes the shape of the mechanism:
//!
//! * **Isolation (realm).** A context carries a `RealmMap` naming which realm
//!   each service lives in. A child context clones the map and overrides one
//!   name, so two sessions can each own an independent instance of the same
//!   service. Service lookup keys on `(name, realm)`.
//! * **Epoch.** A plugin is never reloaded by hand. Its epoch is the string of
//!   its dependencies' fiber ids; when a dependency is replaced the string
//!   changes, and [`Kernel::refresh`] unloads and reloads the dependent.
//! * **Effect.** Every side effect is registered and returns a disposer.
//!   Unloading runs disposers in reverse order, which is the only thing that
//!   keeps hot reload from leaking.
//!
//! A plugin does not mutate the kernel directly: `apply` returns a
//! [`Contributions`] value that the kernel installs. That sidesteps the borrow
//! conflicts an in-place API would create, and it makes rollback possible — a
//! contribution set either installs completely or not at all.
//!
//! A fiber whose dependency disappears goes back to `Pending` rather than being
//! destroyed, so it reloads by itself when the dependency returns.

use std::any::Any;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use anyhow::{anyhow, Result};
use serde_json::Value;

use crate::tools::{ConflictPolicy, ToolDef, ToolRegistry};

/// Identifies one isolation realm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RealmId(u64);

impl std::fmt::Display for RealmId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "realm-{}", self.0)
    }
}

impl RealmId {
    /// The realm every context starts in.
    pub fn root() -> Self {
        RealmId(0)
    }
}

/// Service name → realm, plus the realm used for names not listed.
///
/// The default realm is what makes session isolation work without enumerating
/// every service a plugin will provide: a session-scoped context simply
/// defaults to its own realm, so everything that plugin contributes is
/// isolated without the loader having to guess.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RealmMap {
    named: BTreeMap<String, RealmId>,
    default: RealmId,
}

impl Default for RealmMap {
    fn default() -> Self {
        Self { named: BTreeMap::new(), default: RealmId::root() }
    }
}

impl RealmMap {
    /// An empty map: every name resolves in the root realm.
    pub fn new() -> Self {
        Self::default()
    }

    /// The realm a service name resolves to.
    pub fn realm_of(&self, name: &str) -> RealmId {
        self.named.get(name).copied().unwrap_or(self.default)
    }

    /// A copy with `name` moved into `realm`.
    pub fn with_isolated(&self, name: &str, realm: RealmId) -> Self {
        let mut next = self.clone();
        next.named.insert(name.to_string(), realm);
        next
    }

    /// A copy where every unlisted name resolves in `realm`.
    pub fn with_default(&self, realm: RealmId) -> Self {
        let mut next = self.clone();
        next.default = realm;
        next
    }

    /// The realm unlisted names resolve to.
    pub fn default_realm(&self) -> RealmId {
        self.default
    }

    /// Names explicitly placed outside the default realm.
    pub fn isolated_names(&self) -> Vec<&str> {
        self.named
            .iter()
            .filter(|(_, realm)| **realm != self.default)
            .map(|(name, _)| name.as_str())
            .collect()
    }
}

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
    /// Services this plugin provides, by name.
    pub services: Vec<(String, Arc<dyn Any + Send + Sync>)>,
    /// Tools this plugin registers.
    pub tools: Vec<ToolDef>,
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

    /// Add a disposer.
    pub fn effect(mut self, disposer: impl FnOnce() + Send + 'static) -> Self {
        self.effects.push(Box::new(disposer));
        self
    }
}

/// A read-only, realm-scoped view of the service table.
#[derive(Clone, Default)]
pub struct ServiceView {
    entries: Arc<BTreeMap<(String, RealmId), ServiceEntry>>,
    realm: RealmMap,
}

impl ServiceView {
    /// Look up a service in this view's realm.
    pub fn get<T: Any + Send + Sync>(&self, name: &str) -> Option<Arc<T>> {
        let key = (name.to_string(), self.realm.realm_of(name));
        self.entries
            .get(&key)
            .and_then(|entry| entry.value.clone().downcast::<T>().ok())
    }

    /// Whether a service exists in this view's realm.
    pub fn has(&self, name: &str) -> bool {
        let key = (name.to_string(), self.realm.realm_of(name));
        self.entries.contains_key(&key)
    }

    /// Names visible in this view's realm.
    pub fn names(&self) -> Vec<String> {
        self.entries
            .keys()
            .filter(|(name, realm)| self.realm.realm_of(name) == *realm)
            .map(|(name, _)| name.clone())
            .collect()
    }
}

/// One registered service instance.
#[derive(Clone)]
struct ServiceEntry {
    value: Arc<dyn Any + Send + Sync>,
    /// Fiber that provided it; the epoch input for dependents.
    fiber: u64,
    /// Plugin that provided it.
    owner: String,
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
    /// Plugin configuration.
    pub config: Value,
}

impl PluginCtx {
    /// Look up a service visible to this plugin.
    pub fn get<T: Any + Send + Sync>(&self, name: &str) -> Option<Arc<T>> {
        self.services.get::<T>(name)
    }

    /// Whether a service is visible.
    pub fn has(&self, name: &str) -> bool {
        self.services.has(name)
    }
}

/// Lifecycle state of one plugin instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FiberState {
    /// Waiting for a required service.
    Pending,
    /// Loaded and contributing.
    Active,
    /// Load failed; the fiber stays for diagnostics.
    Failed,
}

/// Epoch of a fiber: the identity of everything it depends on.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Epoch {
    /// At least one dependency is missing.
    Inactive,
    /// All dependencies present; the string changes when any is replaced.
    Active(String),
}

/// One plugin instance.
pub struct Fiber {
    /// Stable id.
    pub id: u64,
    /// Plugin name.
    pub plugin: String,
    /// Services this fiber needs.
    pub inject: Vec<String>,
    /// Services this fiber declared it provides.
    pub provide: Vec<String>,
    /// Isolation map in force.
    pub realm: RealmMap,
    /// Current state.
    pub state: FiberState,
    /// Failure detail, when `state` is `Failed`.
    pub error: Option<String>,
    /// Interface panels this fiber contributes.
    pub ui: Vec<crate::ui::UiPanel>,
    /// Theme layers this fiber contributes.
    pub themes: Vec<crate::theme::Theme>,
    /// Interface labels this fiber contributes, by element id.
    pub strings: std::collections::BTreeMap<String, String>,
    effects: Vec<Disposer>,
    epoch: Epoch,
}

impl Fiber {
    /// The current epoch, for diagnostics.
    pub fn epoch_label(&self) -> String {
        match &self.epoch {
            Epoch::Inactive => "inactive".to_string(),
            Epoch::Active(s) => s.clone(),
        }
    }
}

/// A plugin.
pub trait Plugin: Send + Sync {
    /// Plugin name, unique per kernel.
    fn name(&self) -> &str;

    /// Services that must exist before this plugin loads.
    fn inject(&self) -> Vec<String> {
        Vec::new()
    }

    /// Services this plugin will provide. Advisory: it lets the kernel reject a
    /// clash before `apply` runs, but the authoritative check happens at
    /// install time, where a plugin may provide something it did not declare.
    fn provide(&self) -> Vec<String> {
        Vec::new()
    }

    /// Produce contributions. Called only when every dependency is present.
    fn apply(&self, ctx: &PluginCtx) -> Result<Contributions>;
}

/// A permission decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Decision {
    /// No opinion.
    Abstain,
    /// Allow the action.
    Allow,
    /// Refuse the action, with a reason.
    Deny(String),
}

/// An action a plugin may be asked about.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Action {
    /// Category, e.g. `tool`, `fs.write`, `bash`, `net`, `hotreload`.
    pub kind: String,
    /// Human-readable detail.
    pub detail: String,
}

impl Action {
    /// Build an action.
    pub fn new(kind: impl Into<String>, detail: impl Into<String>) -> Self {
        Self { kind: kind.into(), detail: detail.into() }
    }
}

/// A permission provider contributed by a plugin.
///
/// Providers are consulted as a **collection**, never as a pipeline: the result
/// must not depend on registration order. Any denial wins.
pub trait PermissionProvider: Send + Sync {
    /// Name for diagnostics.
    fn name(&self) -> &str;
    /// The decision for one action.
    fn check(&self, action: &Action) -> Decision;
}

/// The permission stack: collection semantics with deny-wins.
#[derive(Default, Clone)]
pub struct PermissionStack {
    providers: Vec<Arc<dyn PermissionProvider>>,
}

impl PermissionStack {
    /// An empty stack: everything is allowed.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a provider. The stack is a collection, so order carries no
    /// meaning; providers are sorted by name purely for reproducibility.
    pub fn register(&mut self, provider: Arc<dyn PermissionProvider>) {
        self.providers.push(provider);
        self.providers.sort_by(|a, b| a.name().cmp(b.name()));
    }

    /// Remove a provider by name.
    pub fn unregister(&mut self, name: &str) -> bool {
        let before = self.providers.len();
        self.providers.retain(|p| p.name() != name);
        self.providers.len() != before
    }

    /// Consult every provider. Order-independent by construction.
    pub fn check(&self, action: &Action) -> Decision {
        let mut denials = Vec::new();
        for provider in &self.providers {
            if let Decision::Deny(reason) = provider.check(action) {
                denials.push(format!("{}: {reason}", provider.name()));
            }
        }
        if denials.is_empty() {
            Decision::Allow
        } else {
            Decision::Deny(denials.join("; "))
        }
    }

    /// Names of registered providers.
    pub fn providers(&self) -> Vec<&str> {
        self.providers.iter().map(|p| p.name()).collect()
    }
}

/// What the kernel did during one refresh.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReloadEvent {
    /// Fiber affected.
    pub fiber: u64,
    /// Plugin name.
    pub plugin: String,
    /// `activated`, `reloaded`, or `deactivated`.
    pub action: String,
    /// Why it happened.
    pub reason: String,
}

/// The plugin kernel: fibers, services, permissions.
pub struct Kernel {
    plugins: BTreeMap<String, Arc<dyn Plugin>>,
    fibers: BTreeMap<u64, Fiber>,
    services: BTreeMap<(String, RealmId), ServiceEntry>,
    /// Tools contributed by each fiber, so unload removes exactly its own.
    fiber_tools: BTreeMap<u64, Vec<String>>,
    permissions: PermissionStack,
    tools: ToolRegistry,
    counter: Arc<AtomicU64>,
    realm_counter: Arc<AtomicU64>,
    /// Bumped on every change, so callers can tell snapshots apart.
    pub version: u64,
}

impl Default for Kernel {
    fn default() -> Self {
        Self::new()
    }
}

impl Kernel {
    /// An empty kernel with the four base tools registered.
    pub fn new() -> Self {
        Self {
            plugins: BTreeMap::new(),
            fibers: BTreeMap::new(),
            services: BTreeMap::new(),
            fiber_tools: BTreeMap::new(),
            permissions: PermissionStack::new(),
            tools: ToolRegistry::with_base_tools().expect("base tools register"),
            counter: Arc::new(AtomicU64::new(1)),
            realm_counter: Arc::new(AtomicU64::new(1)),
            version: 0,
        }
    }

    /// The tool table.
    pub fn tools(&self) -> &ToolRegistry {
        &self.tools
    }

    /// The tool table, mutably.
    pub fn tools_mut(&mut self) -> &mut ToolRegistry {
        &mut self.tools
    }

    /// The permission stack.
    pub fn permissions(&self) -> &PermissionStack {
        &self.permissions
    }

    /// The permission stack, mutably.
    pub fn permissions_mut(&mut self) -> &mut PermissionStack {
        &mut self.permissions
    }

    /// Create a fresh realm id.
    pub fn new_realm(&self) -> RealmId {
        RealmId(self.realm_counter.fetch_add(1, Ordering::Relaxed))
    }

    /// All fibers, ordered by id.
    pub fn fibers(&self) -> Vec<&Fiber> {
        self.fibers.values().collect()
    }

    /// Look up a fiber.
    pub fn fiber(&self, id: u64) -> Option<&Fiber> {
        self.fibers.get(&id)
    }

    /// Look up a defined plugin by name.
    pub fn plugin(&self, name: &str) -> Option<&Arc<dyn Plugin>> {
        self.plugins.get(name)
    }

    /// Names of defined plugins.
    pub fn plugin_names(&self) -> Vec<String> {
        self.plugins.keys().cloned().collect()
    }

    /// A realm-scoped service view.
    pub fn service_view(&self, realm: RealmMap) -> ServiceView {
        ServiceView { entries: Arc::new(self.services.clone()), realm }
    }

    /// Every service currently registered, as `(name, realm, owner)`.
    pub fn service_list(&self) -> Vec<(String, RealmId, String)> {
        self.services
            .iter()
            .map(|((name, realm), entry)| (name.clone(), *realm, entry.owner.clone()))
            .collect()
    }

    /// Tools contributed by one fiber.
    pub fn tools_of(&self, fiber: u64) -> Vec<String> {
        self.fiber_tools.get(&fiber).cloned().unwrap_or_default()
    }

    /// Interface panels contributed by every active plugin.
    ///
    /// A panel from a plugin that is not active is not sent: a shell showing a
    /// panel whose plugin failed to load would be presenting a control that does
    /// nothing.
    pub fn ui_panels(&self) -> Vec<crate::ui::UiPanel> {
        let mut registry = crate::ui::UiRegistry::new();
        for fiber in self.fibers.values() {
            if fiber.state != FiberState::Active {
                continue;
            }
            registry.extend(fiber.ui.iter().cloned());
        }
        registry.panels().into_iter().cloned().collect()
    }
    /// The effective theme: every active plugin's layer applied in order.
    ///
    /// Resolved here rather than in the shell, so the page receives one token
    /// set and does not have to know how many plugins produced it.
    pub fn resolved_theme(&self) -> crate::theme::Theme {
        let mut registry = crate::theme::ThemeRegistry::new();
        let mut layers: Vec<&crate::theme::Theme> = self
            .fibers
            .values()
            .filter(|fiber| fiber.state == FiberState::Active)
            .flat_map(|fiber| fiber.themes.iter())
            .collect();
        // Sorted by name so the layering is deterministic: a map's iteration
        // order is not, and two plugins setting the same token must resolve the
        // same way on every run.
        layers.sort_by(|a, b| a.name.cmp(&b.name));
        for layer in layers {
            registry.add(layer.clone());
        }
        registry.resolve()
    }

    /// The interface labels every active plugin contributes, merged.
    ///
    /// Resolved here for the same reason as the theme: the shell receives one
    /// map rather than one per plugin, and a key set by two plugins resolves
    /// the same way on every run — the later-loaded one wins, because a pack
    /// installed after another is the more specific choice.
    pub fn resolved_strings(&self) -> std::collections::BTreeMap<String, String> {
        let mut fibers: Vec<&Fiber> = self
            .fibers
            .values()
            .filter(|fiber| fiber.state == FiberState::Active)
            .collect();
        fibers.sort_by_key(|fiber| fiber.id);
        let mut merged = std::collections::BTreeMap::new();
        for fiber in fibers {
            merged.extend(fiber.strings.clone());
        }
        merged
    }

    /// Register a plugin definition without loading an instance.
    pub fn define(&mut self, plugin: Arc<dyn Plugin>) {
        self.plugins.insert(plugin.name().to_string(), plugin);
    }

    /// Load one instance of a plugin into `realm`.
    ///
    /// Returns the fiber id. The fiber may be `Pending` when dependencies are
    /// missing; [`Kernel::refresh`] activates it once they appear.
    pub fn load(&mut self, name: &str, realm: RealmMap, config: Value) -> Result<u64> {
        let plugin = self
            .plugins
            .get(name)
            .cloned()
            .ok_or_else(|| anyhow!("plugin '{name}' is not defined"))?;

        let id = self.counter.fetch_add(1, Ordering::Relaxed);
        let inject = plugin.inject();
        let provide = plugin.provide();

        // A declared provider name may only be claimed once per realm. Catching
        // it here gives a clear error before `apply` runs.
        for service in &provide {
            let key = (service.clone(), realm.realm_of(service));
            if let Some(existing) = self.services.get(&key) {
                return Err(anyhow!(
                    "service '{service}' is already provided by '{}' in this realm",
                    existing.owner
                ));
            }
        }

        self.fibers.insert(
            id,
            Fiber {
                id,
                plugin: name.to_string(),
                inject,
                provide,
                realm: realm.clone(),
                state: FiberState::Pending,
                error: None,
                ui: Vec::new(),
                themes: Vec::new(),
                strings: std::collections::BTreeMap::new(),
                effects: Vec::new(),
                epoch: Epoch::Inactive,
            },
        );

        self.try_activate(id, &config)?;
        Ok(id)
    }

    /// Unload a fiber entirely.
    pub fn unload(&mut self, id: u64) -> Result<()> {
        if !self.fibers.contains_key(&id) {
            return Ok(());
        }
        self.deactivate(id);
        self.fibers.remove(&id);
        self.version += 1;
        self.refresh()?;
        Ok(())
    }

    /// Reload a fiber in place, keeping its realm.
    pub fn reload(&mut self, id: u64) -> Result<()> {
        let Some(fiber) = self.fibers.get(&id) else {
            return Err(anyhow!("fiber {id} is not loaded"));
        };
        let name = fiber.plugin.clone();
        let realm = fiber.realm.clone();

        self.unload(id)?;
        self.load(&name, realm, Value::Null)?;
        Ok(())
    }

    /// Re-apply every fiber of one plugin.
    ///
    /// [`Kernel::refresh`] reacts to *service* changes through epochs, which
    /// never fire for a plugin whose configuration lives in a cell the host
    /// owns. This is the other trigger: the host changed that cell and asked
    /// for the plugin's contributions to be rebuilt from it. Mechanics are
    /// [`Kernel::reload`]'s — unload then load — scoped by plugin name, and
    /// fibers that are pending or failed are re-armed exactly like active
    /// ones.
    ///
    /// Returns how many fibers were re-applied.
    pub fn reload_plugin(&mut self, name: &str) -> Result<usize> {
        let targets: Vec<(u64, String, RealmMap)> = self
            .fibers
            .iter()
            .filter(|(_, fiber)| fiber.plugin == name)
            .map(|(id, fiber)| (*id, fiber.plugin.clone(), fiber.realm.clone()))
            .collect();
        let mut reloaded = 0;
        for (id, plugin, realm) in targets {
            self.unload(id)?;
            self.load(&plugin, realm, Value::Null)?;
            reloaded += 1;
        }
        Ok(reloaded)
    }

    /// Recompute every fiber's epoch and reconcile the ones that changed.
    ///
    /// This is the whole of hot reload: nothing calls `reload` by hand. A fiber
    /// whose dependency disappears is deactivated but kept, so it activates
    /// again by itself when the dependency returns.
    pub fn refresh(&mut self) -> Result<Vec<ReloadEvent>> {
        let mut events = Vec::new();

        // Each pass may change other fibers' epochs, so iterate to a fixed
        // point. The cap guards against a pathological cycle, not a limit
        // anyone should reach in practice.
        for _ in 0..64 {
            let mut progressed = false;

            let ids: Vec<u64> = self.fibers.keys().copied().collect();
            for id in ids {
                let Some(fiber) = self.fibers.get(&id) else { continue };
                let desired = self.compute_epoch(&fiber.inject, &fiber.realm);
                if fiber.epoch == desired {
                    continue;
                }

                match (&fiber.epoch, &desired) {
                    (Epoch::Active(_), Epoch::Active(_)) => {
                        events.push(ReloadEvent {
                            fiber: id,
                            plugin: fiber.plugin.clone(),
                            action: "reloaded".into(),
                            reason: "a dependency was replaced".into(),
                        });
                        self.deactivate(id);
                        if let Some(fiber) = self.fibers.get_mut(&id) {
                            fiber.epoch = desired;
                        }
                        self.try_activate(id, &Value::Null)?;
                    }
                    (Epoch::Active(_), Epoch::Inactive) => {
                        events.push(ReloadEvent {
                            fiber: id,
                            plugin: fiber.plugin.clone(),
                            action: "deactivated".into(),
                            reason: "a dependency disappeared".into(),
                        });
                        self.deactivate(id);
                        if let Some(fiber) = self.fibers.get_mut(&id) {
                            fiber.epoch = Epoch::Inactive;
                        }
                    }
                    (Epoch::Inactive, Epoch::Active(_)) => {
                        events.push(ReloadEvent {
                            fiber: id,
                            plugin: fiber.plugin.clone(),
                            action: "activated".into(),
                            reason: "dependencies became available".into(),
                        });
                        if let Some(fiber) = self.fibers.get_mut(&id) {
                            fiber.epoch = desired;
                        }
                        self.try_activate(id, &Value::Null)?;
                    }
                    (Epoch::Inactive, Epoch::Inactive) => {
                        if let Some(fiber) = self.fibers.get_mut(&id) {
                            fiber.epoch = desired;
                        }
                    }
                }

                progressed = true;
                break;
            }

            if !progressed {
                break;
            }
        }

        Ok(events)
    }

    /// Tear down one fiber's contributions while keeping the fiber itself.
    fn deactivate(&mut self, id: u64) {
        let Some(mut fiber) = self.fibers.remove(&id) else {
            return;
        };

        // Reverse order: later effects may depend on earlier ones.
        while let Some(disposer) = fiber.effects.pop() {
            disposer();
        }

        self.services.retain(|_, entry| entry.fiber != id);

        if let Some(tools) = self.fiber_tools.remove(&id) {
            for tool in tools {
                self.tools.unregister(&tool);
            }
        }

        fiber.state = FiberState::Pending;
        self.fibers.insert(id, fiber);
    }

    fn compute_epoch(&self, inject: &[String], realm: &RealmMap) -> Epoch {
        let mut parts = Vec::with_capacity(inject.len());
        for name in inject {
            let key = (name.clone(), realm.realm_of(name));
            match self.services.get(&key) {
                Some(entry) => parts.push(format!("{name}:{}", entry.fiber)),
                // A missing dependency keeps the fiber inactive.
                None => return Epoch::Inactive,
            }
        }
        Epoch::Active(parts.join(","))
    }

    /// Run a plugin's `apply` and install its contributions atomically.
    fn try_activate(&mut self, id: u64, config: &Value) -> Result<()> {
        let Some(fiber) = self.fibers.get(&id) else {
            return Ok(());
        };
        let desired = self.compute_epoch(&fiber.inject, &fiber.realm);
        if desired == Epoch::Inactive {
            return Ok(());
        }

        let Some(plugin) = self.plugins.get(&fiber.plugin).cloned() else {
            return Ok(());
        };
        let realm = fiber.realm.clone();
        let plugin_name = fiber.plugin.clone();

        let ctx = PluginCtx {
            plugin: plugin_name.clone(),
            fiber: id,
            realm: realm.clone(),
            services: self.service_view(realm.clone()),
            config: config.clone(),
        };

        let contributions = match plugin.apply(&ctx) {
            Ok(c) => c,
            Err(error) => {
                if let Some(fiber) = self.fibers.get_mut(&id) {
                    fiber.state = FiberState::Failed;
                    fiber.error = Some(format!("{error:#}"));
                }
                return Ok(());
            }
        };

        // Validate every service binding before installing any of them, so a
        // clash cannot leave the kernel half-configured.
        for (name, _) in &contributions.services {
            let key = (name.clone(), realm.realm_of(name));
            if let Some(existing) = self.services.get(&key) {
                if let Some(fiber) = self.fibers.get_mut(&id) {
                    fiber.state = FiberState::Failed;
                    fiber.error = Some(format!(
                        "service '{name}' is already provided by '{}' in this realm",
                        existing.owner
                    ));
                }
                return Ok(());
            }
        }

        let mut registered_tools: Vec<String> = Vec::new();
        for def in contributions.tools {
            let tool_name = def.name.clone();
            // The kernel owns the tool table, so it stamps ownership itself: a
            // plugin does not get to claim someone else's tool.
            let stamped = ToolDef { owner: plugin_name.clone(), ..def };
            if let Err(error) = self.tools.register(stamped, ConflictPolicy::Error) {
                for tool in &registered_tools {
                    self.tools.unregister(tool);
                }
                if let Some(fiber) = self.fibers.get_mut(&id) {
                    fiber.state = FiberState::Failed;
                    fiber.error = Some(format!("{error:#}"));
                }
                return Ok(());
            }
            registered_tools.push(tool_name);
        }

        for (name, value) in contributions.services {
            let key = (name.clone(), realm.realm_of(&name));
            self.services.insert(
                key,
                ServiceEntry { value, fiber: id, owner: plugin_name.clone() },
            );
        }

        self.fiber_tools.insert(id, registered_tools);
        if let Some(fiber) = self.fibers.get_mut(&id) {
            fiber.ui = contributions.ui;
            fiber.themes = contributions.themes;
            fiber.strings = contributions.strings;
            fiber.effects = contributions.effects;
            fiber.epoch = desired;
            fiber.state = FiberState::Active;
            fiber.error = None;
        }
        self.version += 1;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Named {
        name: String,
        inject: Vec<String>,
        provide: Vec<String>,
        body: Arc<dyn Fn(&PluginCtx) -> Result<Contributions> + Send + Sync>,
    }

    impl Named {
        fn new(
            name: &str,
            body: impl Fn(&PluginCtx) -> Result<Contributions> + Send + Sync + 'static,
        ) -> Arc<Self> {
            Arc::new(Self {
                name: name.to_string(),
                inject: Vec::new(),
                provide: Vec::new(),
                body: Arc::new(body),
            })
        }

        fn with_deps(name: &str, inject: &[&str], provide: &[&str]) -> Arc<Self> {
            Arc::new(Self {
                name: name.to_string(),
                inject: inject.iter().map(|s| s.to_string()).collect(),
                provide: provide.iter().map(|s| s.to_string()).collect(),
                body: Arc::new(|_| Ok(Contributions::new())),
            })
        }
    }

    impl Plugin for Named {
        fn name(&self) -> &str {
            &self.name
        }
        fn inject(&self) -> Vec<String> {
            self.inject.clone()
        }
        fn provide(&self) -> Vec<String> {
            self.provide.clone()
        }
        fn apply(&self, ctx: &PluginCtx) -> Result<Contributions> {
            (self.body)(ctx)
        }
    }

    fn empty_tool(name: &str) -> ToolDef {
        ToolDef::new(
            name,
            "d",
            serde_json::json!({ "type": "object" }),
            "ignored-by-kernel",
            |_| Box::pin(async { Ok(crate::tools::ToolOutput::default()) }) as crate::tools::ToolFuture,
        )
    }

    #[test]
    fn isolated_realms_keep_same_named_services_apart() {
        let mut kernel = Kernel::new();
        kernel.define(Named::new("provider", |_| {
            Ok(Contributions::new().service("counter", Arc::new(1u32)))
        }));

        let realm_a = RealmMap::new().with_isolated("counter", kernel.new_realm());
        let realm_b = RealmMap::new().with_isolated("counter", kernel.new_realm());

        kernel.load("provider", realm_a.clone(), Value::Null).unwrap();
        kernel.load("provider", realm_b.clone(), Value::Null).unwrap();

        // Same service name, two realms, two independent instances.
        assert!(kernel.service_view(realm_a).get::<u32>("counter").is_some());
        assert!(kernel.service_view(realm_b).get::<u32>("counter").is_some());
        assert_eq!(kernel.service_list().len(), 2);

        // The root realm sees neither.
        assert!(!kernel.service_view(RealmMap::new()).has("counter"));
    }

    #[test]
    fn duplicate_service_in_one_realm_fails_loud() {
        let mut kernel = Kernel::new();
        kernel.define(Named::new("first", |_| {
            Ok(Contributions::new().service("thing", Arc::new(1u8)))
        }));
        kernel.define(Named::new("second", |_| {
            Ok(Contributions::new().service("thing", Arc::new(2u8)))
        }));

        kernel.load("first", RealmMap::new(), Value::Null).unwrap();
        let second = kernel.load("second", RealmMap::new(), Value::Null).unwrap();

        // The clash is caught at install time even though `second` never
        // declared the service it provides.
        let fiber = kernel.fiber(second).unwrap();
        assert_eq!(fiber.state, FiberState::Failed);
        let error = fiber.error.clone().unwrap();
        assert!(error.contains("already provided"), "{error}");
        assert!(error.contains("first"), "the existing owner is named: {error}");

        // The first provider is untouched.
        assert_eq!(
            *kernel.service_view(RealmMap::new()).get::<u8>("thing").unwrap(),
            1
        );
    }

    #[test]
    fn a_declared_provider_clash_is_rejected_before_apply() {
        let mut kernel = Kernel::new();
        kernel.define(Named::with_deps("declarer", &[], &["shared"]));
        kernel.define(Named::new("provider", |_| {
            Ok(Contributions::new().service("shared", Arc::new(1u8)))
        }));

        kernel.load("provider", RealmMap::new(), Value::Null).unwrap();
        let error = kernel
            .load("declarer", RealmMap::new(), Value::Null)
            .expect_err("declared clash is refused up front");
        assert!(format!("{error:#}").contains("already provided"));
    }

    #[test]
    fn a_plugin_waits_for_its_dependency_then_activates() {
        let mut kernel = Kernel::new();
        kernel.define(Named::with_deps("consumer", &["thing"], &[]));
        kernel.define(Named::new("provider", |_| {
            Ok(Contributions::new().service("thing", Arc::new(7u8)))
        }));

        let consumer = kernel.load("consumer", RealmMap::new(), Value::Null).unwrap();
        assert_eq!(kernel.fiber(consumer).unwrap().state, FiberState::Pending);

        kernel.load("provider", RealmMap::new(), Value::Null).unwrap();
        let events = kernel.refresh().unwrap();

        assert!(events.iter().any(|e| e.action == "activated"), "{events:?}");
        assert_eq!(kernel.fiber(consumer).unwrap().state, FiberState::Active);
    }

    #[test]
    fn replacing_a_dependency_reloads_the_dependent_automatically() {
        let mut kernel = Kernel::new();
        kernel.define(Named::with_deps("consumer", &["thing"], &[]));
        kernel.define(Named::new("provider", |_| {
            Ok(Contributions::new().service("thing", Arc::new(1u8)))
        }));

        let consumer = kernel.load("consumer", RealmMap::new(), Value::Null).unwrap();
        let provider = kernel.load("provider", RealmMap::new(), Value::Null).unwrap();
        kernel.refresh().unwrap();
        assert_eq!(kernel.fiber(consumer).unwrap().state, FiberState::Active);
        let epoch_before = kernel.fiber(consumer).unwrap().epoch_label();

        // Replace the provider: the consumer's epoch must change and the kernel
        // must reconcile it without being asked.
        kernel.unload(provider).unwrap();
        kernel.load("provider", RealmMap::new(), Value::Null).unwrap();
        let events = kernel.refresh().unwrap();

        assert!(
            events.iter().any(|e| e.fiber == consumer),
            "the dependent was reconciled: {events:?}"
        );
        let epoch_after = kernel.fiber(consumer).unwrap().epoch_label();
        assert_ne!(epoch_before, epoch_after, "epoch tracks dependency identity");
        assert_eq!(
            kernel.fiber(consumer).unwrap().state,
            FiberState::Active,
            "and it is active again"
        );
    }

    #[test]
    fn a_fiber_survives_its_dependency_disappearing() {
        let mut kernel = Kernel::new();
        kernel.define(Named::with_deps("consumer", &["thing"], &[]));
        kernel.define(Named::new("provider", |_| {
            Ok(Contributions::new().service("thing", Arc::new(1u8)))
        }));

        let consumer = kernel.load("consumer", RealmMap::new(), Value::Null).unwrap();
        let provider = kernel.load("provider", RealmMap::new(), Value::Null).unwrap();
        kernel.refresh().unwrap();
        assert_eq!(kernel.fiber(consumer).unwrap().state, FiberState::Active);

        // The dependency goes away: the consumer is deactivated, not destroyed.
        kernel.unload(provider).unwrap();
        let fiber = kernel.fiber(consumer).expect("fiber still exists");
        assert_eq!(fiber.state, FiberState::Pending);
        assert_eq!(fiber.epoch_label(), "inactive");

        // ...and it comes back on its own when the dependency returns.
        kernel.load("provider", RealmMap::new(), Value::Null).unwrap();
        let events = kernel.refresh().unwrap();
        assert!(events.iter().any(|e| e.fiber == consumer && e.action == "activated"));
        assert_eq!(kernel.fiber(consumer).unwrap().state, FiberState::Active);
    }

    #[test]
    fn unloading_runs_effects_in_reverse_order() {
        let mut kernel = Kernel::new();
        let order = Arc::new(std::sync::Mutex::new(Vec::new()));
        let order_in = Arc::clone(&order);

        kernel.define(Named::new("resourceful", move |_| {
            let a = Arc::clone(&order_in);
            let b = Arc::clone(&order_in);
            let c = Arc::clone(&order_in);
            Ok(Contributions::new()
                .effect(move || a.lock().unwrap().push("first"))
                .effect(move || b.lock().unwrap().push("second"))
                .effect(move || c.lock().unwrap().push("third")))
        }));

        let fiber = kernel.load("resourceful", RealmMap::new(), Value::Null).unwrap();
        kernel.unload(fiber).unwrap();

        assert_eq!(
            *order.lock().unwrap(),
            vec!["third", "second", "first"],
            "disposers run last-registered-first"
        );
    }

    #[test]
    fn unloading_removes_services_and_tools() {
        let mut kernel = Kernel::new();
        kernel.define(Named::new("contributor", |_| {
            Ok(Contributions::new()
                .service("svc", Arc::new(1u8))
                .tool(empty_tool("plugin_tool")))
        }));

        let fiber = kernel.load("contributor", RealmMap::new(), Value::Null).unwrap();
        assert!(kernel.service_view(RealmMap::new()).has("svc"));
        assert!(kernel.tools().get("plugin_tool").is_some());
        assert_eq!(kernel.tools_of(fiber), vec!["plugin_tool".to_string()]);
        // The kernel stamps ownership itself.
        assert_eq!(kernel.tools().owner("plugin_tool"), Some("contributor"));

        kernel.unload(fiber).unwrap();
        assert!(!kernel.service_view(RealmMap::new()).has("svc"));
        assert!(kernel.tools().get("plugin_tool").is_none());
        // Base tools are untouched.
        assert!(kernel.tools().get("read").is_some());
    }

    #[test]
    fn two_instances_of_one_plugin_do_not_remove_each_others_tools() {
        let mut kernel = Kernel::new();
        kernel.define(Named::new("multi", |ctx| {
            Ok(Contributions::new().tool(empty_tool(&format!("tool_of_{}", ctx.fiber))))
        }));

        let first = kernel.load("multi", RealmMap::new(), Value::Null).unwrap();
        let second = kernel.load("multi", RealmMap::new(), Value::Null).unwrap();
        let first_tool = format!("tool_of_{first}");
        let second_tool = format!("tool_of_{second}");
        assert!(kernel.tools().get(&first_tool).is_some());
        assert!(kernel.tools().get(&second_tool).is_some());

        // Unloading one instance must not touch the other's tool.
        kernel.unload(first).unwrap();
        assert!(kernel.tools().get(&first_tool).is_none());
        assert!(
            kernel.tools().get(&second_tool).is_some(),
            "the sibling instance keeps its tool"
        );
    }

    #[test]
    fn a_failing_load_rolls_back_partial_installation() {
        let mut kernel = Kernel::new();
        kernel.define(Named::new("ok", |_| {
            Ok(Contributions::new().tool(empty_tool("shared")))
        }));
        kernel.define(Named::new("clash", |_| {
            Ok(Contributions::new()
                .service("side", Arc::new(9u8))
                .tool(empty_tool("shared")))
        }));

        let first = kernel.load("ok", RealmMap::new(), Value::Null).unwrap();
        assert!(kernel.tools().get("shared").is_some());

        let second = kernel.load("clash", RealmMap::new(), Value::Null).unwrap();
        assert_eq!(kernel.fiber(second).unwrap().state, FiberState::Failed);
        // The failed plugin left nothing behind.
        assert!(!kernel.service_view(RealmMap::new()).has("side"));
        assert_eq!(kernel.fiber(first).unwrap().state, FiberState::Active);
        assert_eq!(kernel.tools().owner("shared"), Some("ok"));
    }

    #[test]
    fn a_service_clash_rolls_back_the_whole_contribution_set() {
        let mut kernel = Kernel::new();
        kernel.define(Named::new("owner", |_| {
            Ok(Contributions::new().service("taken", Arc::new(1u8)))
        }));
        kernel.define(Named::new("latecomer", |_| {
            Ok(Contributions::new()
                .tool(empty_tool("latecomer_tool"))
                .service("taken", Arc::new(2u8)))
        }));

        kernel.load("owner", RealmMap::new(), Value::Null).unwrap();
        let late = kernel.load("latecomer", RealmMap::new(), Value::Null).unwrap();

        assert_eq!(kernel.fiber(late).unwrap().state, FiberState::Failed);
        // Validation happens before installation, so the tool never appeared.
        assert!(kernel.tools().get("latecomer_tool").is_none());
    }

    #[test]
    fn permission_denials_win_regardless_of_registration_order() {
        struct Fixed(&'static str, Decision);
        impl PermissionProvider for Fixed {
            fn name(&self) -> &str {
                self.0
            }
            fn check(&self, _action: &Action) -> Decision {
                self.1.clone()
            }
        }

        let action = Action::new("bash", "rm -rf /");

        let mut stack = PermissionStack::new();
        stack.register(Arc::new(Fixed("allower", Decision::Allow)));
        stack.register(Arc::new(Fixed("denier", Decision::Deny("nope".into()))));
        let first = stack.check(&action);

        // Same providers registered the other way round must agree.
        let mut reversed = PermissionStack::new();
        reversed.register(Arc::new(Fixed("denier", Decision::Deny("nope".into()))));
        reversed.register(Arc::new(Fixed("allower", Decision::Allow)));
        let second = reversed.check(&action);

        assert_eq!(first, second, "decision must not depend on order");
        assert!(matches!(first, Decision::Deny(_)));

        // With no denials the stack allows.
        let mut permissive = PermissionStack::new();
        permissive.register(Arc::new(Fixed("allower", Decision::Allow)));
        permissive.register(Arc::new(Fixed("silent", Decision::Abstain)));
        assert_eq!(permissive.check(&action), Decision::Allow);

        // Unregistering the denier flips the decision.
        stack.unregister("denier");
        assert_eq!(stack.check(&action), Decision::Allow);
    }

    #[test]
    fn realm_map_reports_isolated_names() {
        let realm = RealmId(42);
        let map = RealmMap::new().with_isolated("compaction", realm);
        assert_eq!(map.realm_of("compaction"), realm);
        assert_eq!(map.realm_of("other"), RealmId::root());
        assert_eq!(map.isolated_names(), vec!["compaction"]);
    }

    #[test]
    fn version_advances_on_change() {
        let mut kernel = Kernel::new();
        kernel.define(Named::new("p", |_| Ok(Contributions::new().service("s", Arc::new(1u8)))));
        let before = kernel.version;
        let fiber = kernel.load("p", RealmMap::new(), Value::Null).unwrap();
        assert!(kernel.version > before, "loading bumps the version");
        let after_load = kernel.version;
        kernel.unload(fiber).unwrap();
        assert!(kernel.version > after_load, "unloading bumps the version");
    }

    #[test]
    fn a_failing_plugin_records_its_error() {
        let mut kernel = Kernel::new();
        kernel.define(Named::new("broken", |_| Err(anyhow!("cannot start"))));

        let fiber = kernel.load("broken", RealmMap::new(), Value::Null).unwrap();
        let fiber = kernel.fiber(fiber).unwrap();
        assert_eq!(fiber.state, FiberState::Failed);
        assert!(fiber.error.as_deref().unwrap().contains("cannot start"));
    }

    #[test]
    fn a_later_plugins_label_wins_and_unloading_puts_the_earlier_back() {
        let mut kernel = Kernel::new();
        kernel.define(Named::new("first", |_| {
            Ok(Contributions::new().strings([("send".into(), "Send".into())]))
        }));
        kernel.define(Named::new("second", |_| {
            Ok(Contributions::new().strings([
                ("send".into(), "发送".into()),
                ("save".into(), "保存".into()),
            ]))
        }));
        kernel.load("first", RealmMap::new(), Value::Null).unwrap();
        let second = kernel.load("second", RealmMap::new(), Value::Null).unwrap();

        let strings = kernel.resolved_strings();
        assert_eq!(
            strings.get("send").map(String::as_str),
            Some("发送"),
            "the pack installed later decides"
        );
        assert_eq!(strings.get("save").map(String::as_str), Some("保存"));

        kernel.unload(second).unwrap();
        let strings = kernel.resolved_strings();
        assert_eq!(
            strings.get("send").map(String::as_str),
            Some("Send"),
            "unloading a pack must take its words with it"
        );
        assert!(!strings.contains_key("save"));
    }
}
