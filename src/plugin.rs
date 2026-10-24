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
//!   its dependencies' fiber ids — plus the serial of each instance it depends
//!   on, so a provider re-applied in place still counts as a replacement. When
//!   that string changes, [`Kernel::refresh`] rebuilds the dependent.
//! * **Effect.** Every side effect is registered and returns a disposer.
//!   Unloading runs disposers in reverse order, which is the only thing that
//!   keeps hot reload from leaking.
//! * **Users.** A fiber records which sessions asked for it. Teardown through
//!   [`Kernel::release`] drops one session's claim and leaves the instance
//!   running while anyone else still holds it, so two sessions sharing an
//!   isolated instance cannot switch each other's capabilities off. Unloading
//!   outright ([`Kernel::unload`]) is the kernel-and-pack path, not the
//!   session path.
//! * **Staging.** Activation is split into [`stage`](Kernel) — run `apply` and
//!   check every claim without touching a kernel table — and install, which
//!   cannot fail. That split is what lets a reload build the replacement
//!   *before* it tears the running version down, so a replacement that does not
//!   build leaves the old one running instead of leaving the kernel in a mixed
//!   state.
//!
//! A plugin does not mutate the kernel directly: `apply` returns a
//! [`Contributions`] value that the kernel installs. That sidesteps the borrow
//! conflicts an in-place API would create, and it makes rollback possible — a
//! contribution set either installs completely or not at all.
//!
//! A fiber whose dependency disappears goes back to `Pending` rather than being
//! destroyed, so it reloads by itself when the dependency returns.

use std::any::Any;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

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

    /// Add a disposer.
    pub fn effect(mut self, disposer: impl FnOnce() + Send + 'static) -> Self {
        self.effects.push(Box::new(disposer));
        self
    }
}

/// Why a service lookup through a [`ServiceView`] did not produce a service.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ServiceViewError {
    /// The kernel's service table moved on after this view was taken.
    ///
    /// The view is a reference into a table that is still live, so its numbers
    /// are honest: what it can no longer promise is that the value it would
    /// return is still registered. Answering anyway would be the bug — a plugin
    /// whose service was unregistered must not keep serving from a snapshot.
    Stale {
        /// Generation the view was taken at.
        view: u64,
        /// Generation the kernel is at now.
        current: u64,
    },
    /// Nothing of that name is registered in this view's realm.
    Missing {
        /// Service name asked for.
        name: String,
        /// Realm the view resolves it in.
        realm: RealmId,
    },
    /// The name is registered, but holds something other than the requested
    /// type.
    WrongType {
        /// Service name asked for.
        name: String,
        /// Rust type the caller asked for.
        expected: &'static str,
    },
}

impl std::fmt::Display for ServiceViewError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Stale { view, current } => write!(
                f,
                "this service view was taken at table generation {view} and the kernel is now at \
                 {current}; the service it names may have been unregistered since"
            ),
            Self::Missing { name, realm } => {
                write!(f, "no service named '{name}' is registered in {realm}")
            }
            Self::WrongType { name, expected } => {
                write!(f, "service '{name}' is not a {expected}")
            }
        }
    }
}

impl std::error::Error for ServiceViewError {}

/// A read-only, realm-scoped view of the service table.
///
/// The view holds the kernel's live table behind an `Arc` plus the generation it
/// was taken at — it is not a copy. Two consequences, both of them the point:
///
/// * Handing a plugin a view is O(1). The previous implementation cloned the
///   whole table on every activation *and* on every lookup of a view.
/// * A view never keeps an unregistered service alive. When the kernel changes
///   its table the generation moves, and [`ServiceView::try_get`] reports
///   [`ServiceViewError::Stale`] instead of serving a value that is no longer
///   registered.
#[derive(Clone)]
pub struct ServiceView {
    table: Arc<ServiceTable>,
    /// Table generation this view was taken at.
    generation: u64,
    realm: RealmMap,
}

impl Default for ServiceView {
    fn default() -> Self {
        let table = Arc::new(ServiceTable::default());
        let generation = table.generation.load(Ordering::Relaxed);
        Self { table, generation, realm: RealmMap::new() }
    }
}

impl ServiceView {
    fn of(table: Arc<ServiceTable>, realm: RealmMap) -> Self {
        let generation = table.generation.load(Ordering::Relaxed);
        Self { table, generation, realm }
    }

    /// Look up a service in this view's realm, or fail with a reason.
    pub fn try_get<T: Any + Send + Sync>(
        &self,
        name: &str,
    ) -> std::result::Result<Arc<T>, ServiceViewError> {
        let current = self.table.generation.load(Ordering::Relaxed);
        if current != self.generation {
            return Err(ServiceViewError::Stale { view: self.generation, current });
        }
        let realm = self.realm.realm_of(name);
        let key = (name.to_string(), realm);
        let found = self.table.read().get(&key).map(|entry| Arc::clone(&entry.value));
        match found {
            Some(value) => value.downcast::<T>().map_err(|_| ServiceViewError::WrongType {
                name: name.to_string(),
                expected: std::any::type_name::<T>(),
            }),
            None => Err(ServiceViewError::Missing { name: name.to_string(), realm }),
        }
    }

    /// Look up a service in this view's realm.
    ///
    /// `None` covers all three ways that can fail — stale view, unregistered
    /// name, wrong type — because the callers that want a value only care that
    /// they did not get one. Callers that need to tell the cases apart use
    /// [`ServiceView::try_get`].
    pub fn get<T: Any + Send + Sync>(&self, name: &str) -> Option<Arc<T>> {
        self.try_get::<T>(name).ok()
    }

    /// Whether a service exists in this view's realm.
    ///
    /// A stale view reports nothing: it cannot say what is there any more.
    pub fn has(&self, name: &str) -> bool {
        if !self.is_current() {
            return false;
        }
        let key = (name.to_string(), self.realm.realm_of(name));
        self.table.read().contains_key(&key)
    }

    /// Names visible in this view's realm.
    pub fn names(&self) -> Vec<String> {
        if !self.is_current() {
            return Vec::new();
        }
        self.table
            .read()
            .keys()
            .filter(|(name, realm)| self.realm.realm_of(name) == *realm)
            .map(|(name, _)| name.clone())
            .collect()
    }

    /// The generation this view was taken at.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Whether the kernel's table has changed since this view was taken.
    pub fn is_current(&self) -> bool {
        self.table.generation.load(Ordering::Relaxed) == self.generation
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
    /// Serial of this exact instance.
    ///
    /// Part of the epoch, so re-applying a provider — same fiber, new value —
    /// still invalidates its dependents. Without it a reloaded provider would
    /// keep its fiber id and its dependents would never be rebuilt against the
    /// new instance.
    serial: u64,
}

/// The kernel's live service table, shared with every [`ServiceView`] by `Arc`.
#[derive(Default)]
struct ServiceTable {
    entries: RwLock<BTreeMap<(String, RealmId), ServiceEntry>>,
    /// Bumped on every change; a view taken at an older value is stale.
    generation: AtomicU64,
}

impl ServiceTable {
    /// The table for reading.
    ///
    /// A poisoned lock is treated as usable: the map holds only whole entries
    /// and has no invariant a half-finished write could break, so refusing to
    /// read it would turn one plugin's panic into a kernel-wide outage. The
    /// panic itself is still reported where it happened.
    fn read(&self) -> RwLockReadGuard<'_, BTreeMap<(String, RealmId), ServiceEntry>> {
        self.entries.read().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// The table for writing.
    ///
    /// Poison policy is [`ServiceTable::read`]'s.
    fn write(&self) -> RwLockWriteGuard<'_, BTreeMap<(String, RealmId), ServiceEntry>> {
        self.entries.write().unwrap_or_else(|poisoned| poisoned.into_inner())
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
    /// Plugin configuration.
    pub config: Value,
    /// The event bus: observational subscription only (see `events.rs`) —
    /// a plugin watches facts, it cannot veto them.
    pub events: crate::events::EventBus,
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
    /// Which pack activated this fiber, when a pack did.
    ///
    /// Display provenance: [`Kernel::plugin_origins`] pairs it with
    /// [`Kernel::plugin_names`] so the panel can read a capability back to
    /// the pack that brought it. A load with no pack behind it (kernel code,
    /// a test) keeps `None`.
    pub origin: Option<String>,
    /// Services this fiber needs.
    pub inject: Vec<String>,
    /// Services this fiber declared it provides.
    pub provide: Vec<String>,
    /// Isolation map in force.
    pub realm: RealmMap,
    /// Current state.
    pub state: FiberState,
    /// Failure detail, when `state` is `Failed`, or the reason the last rebuild
    /// was refused.
    pub error: Option<String>,
    /// Configuration the plugin is applied with.
    ///
    /// Kept on the fiber rather than passed in once, because the kernel rebuilds
    /// fibers on its own — [`Kernel::refresh`] re-applies a dependent when its
    /// dependency is replaced. Carrying the configuration here means that
    /// rebuild re-applies with what the host asked for instead of falling back
    /// to `Null` and quietly dropping it.
    pub config: Value,
    /// Interface panels this fiber contributes.
    pub ui: Vec<crate::ui::UiPanel>,
    /// Theme layers this fiber contributes.
    pub themes: Vec<crate::theme::Theme>,
    /// Interface labels this fiber contributes, by element id.
    pub strings: std::collections::BTreeMap<String, String>,
    /// Sessions that have asked for this instance.
    ///
    /// The reference count that makes session isolation real: an instance is
    /// torn down when its last user releases it, not when the first one asks.
    users: BTreeSet<String>,
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

    /// Which sessions hold this instance, sorted.
    ///
    /// The session plane needs this to answer "who would lose this capability if
    /// I unloaded it" before doing so.
    pub fn users(&self) -> Vec<String> {
        self.users.iter().cloned().collect()
    }

    /// How many sessions hold this instance.
    pub fn user_count(&self) -> usize {
        self.users.len()
    }

    /// Whether `session` holds this instance.
    pub fn is_used_by(&self, session: &str) -> bool {
        self.users.contains(session)
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
    /// `activated`, `reloaded`, `deactivated`, `failed`, `rebuild-failed`, or
    /// `truncated`.
    pub action: String,
    /// Why it happened.
    pub reason: String,
}

/// What a [`Kernel::release`] did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReleaseOutcome {
    /// No session still holds the instance: it was torn down and forgotten.
    Unloaded,
    /// The caller's claim was dropped, and this many sessions still hold it.
    StillUsed(usize),
    /// No such fiber: already unloaded.
    Absent,
}

/// One contribution set that passed every check the kernel can make before it
/// touches its own tables.
///
/// This is the whole content of "build the replacement before tearing down the
/// running version": a `Staged` value means `apply` succeeded and nothing it
/// claims clashes with the kernel, so installing it cannot fail.
struct Staged {
    fiber: u64,
    plugin: String,
    epoch: Epoch,
    ui: Vec<crate::ui::UiPanel>,
    themes: Vec<crate::theme::Theme>,
    strings: std::collections::BTreeMap<String, String>,
    services: Vec<((String, RealmId), Arc<dyn Any + Send + Sync>)>,
    tools: Vec<ToolDef>,
    readers: Vec<crate::tools::readers::ReaderSpec>,
    effects: Vec<Disposer>,
}

/// Names already claimed by fibers staged in the same batch.
///
/// Staging validates against the kernel's live tables, which do not yet contain
/// the batch's own earlier entries; without this, two fibers reloaded together
/// could each look free and then collide at install time — past the point where
/// a rollback is possible.
#[derive(Default)]
struct StagedClaims {
    services: BTreeMap<(String, RealmId), String>,
    tools: BTreeMap<String, String>,
}

/// The plugin kernel: fibers, services, permissions.
pub struct Kernel {
    plugins: BTreeMap<String, Arc<dyn Plugin>>,
    fibers: BTreeMap<u64, Fiber>,
    /// Shared with every live [`ServiceView`]; see [`ServiceTable`].
    services: Arc<ServiceTable>,
    /// Tools contributed by each fiber, so unload removes exactly its own.
    fiber_tools: BTreeMap<u64, Vec<String>>,
    permissions: PermissionStack,
    tools: ToolRegistry,
    counter: Arc<AtomicU64>,
    realm_counter: Arc<AtomicU64>,
    /// Serial for service instances, so an epoch changes when a provider is
    /// re-applied even though its fiber id stays the same.
    instance_counter: u64,
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
            services: Arc::new(ServiceTable::default()),
            fiber_tools: BTreeMap::new(),
            permissions: PermissionStack::new(),
            tools: ToolRegistry::with_base_tools().expect("base tools register"),
            counter: Arc::new(AtomicU64::new(1)),
            realm_counter: Arc::new(AtomicU64::new(1)),
            instance_counter: 0,
            version: 0,
        }
    }

    /// The tool table.
    pub fn tools(&self) -> &ToolRegistry {
        &self.tools
    }

    /// Pin the session whose turn will call into this kernel's tool table.
    ///
    /// The desktop swaps sessions in place - one kernel, one registry - so the
    /// id moves here rather than being threaded through every tool signature.
    /// Jobs and session-scoped pack state read it via the thread-local that
    /// `ToolRegistry::execute` pins for the duration of each handler call.
    pub fn set_session(&self, session: Option<String>, workspace: Option<std::path::PathBuf>) {
        self.tools.set_session(session, workspace);
    }

    /// The kernel's event bus (shared with every clone of the tool table).
    ///
    /// Observational only: subscribers watch facts; vetoes belong to the
    /// hooks milestone, not to this bus.
    pub fn events(&self) -> crate::events::EventBus {
        self.tools.events()
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
    ///
    /// O(1): the view shares the kernel's table instead of copying it. See
    /// [`ServiceView`] for what it does when the table changes underneath.
    pub fn service_view(&self, realm: RealmMap) -> ServiceView {
        ServiceView::of(Arc::clone(&self.services), realm)
    }

    /// Generation of the service table.
    ///
    /// A [`ServiceView`] taken at an older generation is stale; this is how a
    /// caller holding one can tell without asking for a service.
    pub fn service_generation(&self) -> u64 {
        self.services.generation.load(Ordering::Relaxed)
    }

    /// Every service currently registered, as `(name, realm, owner)`.
    pub fn service_list(&self) -> Vec<(String, RealmId, String)> {
        self.services
            .read()
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
        self.load_owned(name, realm, config, None)
    }

    /// Load, recording which pack asked for the plugin.
    ///
    /// The same as [`Kernel::load`] with provenance attached — see
    /// [`Fiber::origin`].
    ///
    /// # Failure
    ///
    /// A plugin whose `apply` fails makes this return `Err`. That is the point:
    /// the previous version reported `Ok(id)` for a plugin that never ran, so
    /// no caller could tell a live capability from a dead one. Nothing is left
    /// behind either — activation stages its contributions before installing
    /// them, so a failure means the kernel's tables were never touched and the
    /// effects `apply` did start are undone. The fiber stays in `Failed` for
    /// diagnostics (a panel can show why) but is never reported as a success.
    pub fn load_owned(
        &mut self,
        name: &str,
        realm: RealmMap,
        config: Value,
        origin: Option<String>,
    ) -> Result<u64> {
        let plugin = self
            .plugins
            .get(name)
            .cloned()
            .ok_or_else(|| anyhow!("plugin '{name}' is not defined"))?;

        // A declared provider name may only be claimed once per realm. Catching
        // it here gives a clear error before `apply` runs.
        let provide = plugin.provide();
        for service in &provide {
            let key = (service.clone(), realm.realm_of(service));
            let held = self.services.read().get(&key).map(|entry| entry.owner.clone());
            if let Some(owner) = held {
                return Err(anyhow!(
                    "service '{service}' is already provided by '{owner}' in this realm"
                ));
            }
        }

        // A previous attempt at this same instance is replaced rather than
        // accumulated: the panel should show the latest reason, not a graveyard
        // of failures.
        if let Some(stale) = self.stale_failure(name, &realm) {
            self.unload(stale)?;
        }

        let id = self.counter.fetch_add(1, Ordering::Relaxed);
        self.fibers.insert(
            id,
            Fiber {
                id,
                plugin: name.to_string(),
                origin,
                inject: plugin.inject(),
                provide,
                realm: realm.clone(),
                state: FiberState::Pending,
                error: None,
                config,
                ui: Vec::new(),
                themes: Vec::new(),
                strings: std::collections::BTreeMap::new(),
                users: BTreeSet::new(),
                effects: Vec::new(),
                epoch: Epoch::Inactive,
            },
        );

        self.activate(id)?;
        Ok(id)
    }

    /// Load `name` for one session, sharing a single instance per
    /// `(plugin, realm)`.
    ///
    /// This is the session plane's load. Two sessions asking for the same
    /// isolated instance get the *same* fiber and each becomes one of its
    /// [`Fiber::users`]; they never get two fibers that cannot see each other,
    /// which is what made the "who unloaded last" bug possible.
    ///
    /// The fiber is not torn down by [`Kernel::release`] while any other user
    /// remains — see [`ReleaseOutcome`].
    pub fn load_for_session(
        &mut self,
        session: &str,
        name: &str,
        realm: RealmMap,
        config: Value,
    ) -> Result<u64> {
        // A failed instance is not reused: handing a session a claim on a fiber
        // that never applied would report a capability the kernel does not have.
        // `load_owned` below replaces the failed fiber instead.
        let reusable = self.fiber_of(name, &realm).filter(|id| {
            self.fiber(*id)
                .map(|fiber| fiber.state != FiberState::Failed)
                .unwrap_or(false)
        });
        if let Some(id) = reusable {
            if let Some(fiber) = self.fibers.get_mut(&id) {
                fiber.users.insert(session.to_string());
            }
            self.version += 1;
            return Ok(id);
        }

        let id = self.load_owned(name, realm, config, None)?;
        if let Some(fiber) = self.fibers.get_mut(&id) {
            fiber.users.insert(session.to_string());
        }
        self.version += 1;
        Ok(id)
    }

    /// The fiber already serving `(plugin, realm)`, when there is one.
    pub fn fiber_of(&self, name: &str, realm: &RealmMap) -> Option<u64> {
        self.fibers
            .values()
            .find(|fiber| fiber.plugin == name && &fiber.realm == realm)
            .map(|fiber| fiber.id)
    }

    /// Which sessions hold one fiber.
    pub fn users_of(&self, id: u64) -> Vec<String> {
        self.fibers.get(&id).map(Fiber::users).unwrap_or_default()
    }

    /// The failed fiber already standing for `(plugin, realm)`, if any.
    fn stale_failure(&self, name: &str, realm: &RealmMap) -> Option<u64> {
        self.fibers
            .values()
            .find(|fiber| {
                fiber.plugin == name && &fiber.realm == realm && fiber.state == FiberState::Failed
            })
            .map(|fiber| fiber.id)
    }

    /// Unload a fiber entirely, whatever users it has.
    ///
    /// This is the kernel-and-pack path — a manifest entry removed, the panel's
    /// disable control — and it is deliberately not the session path. A session
    /// dropping its own interest goes through [`Kernel::release`], which only
    /// removes that session's claim and leaves the instance running for
    /// everyone else.
    ///
    /// The refresh that follows is best effort and not part of this call's
    /// contract: the unload either happened or it did not, and a dependent that
    /// failed to re-apply carries its own error on its own fiber. Reporting that
    /// through this return value would say "unload failed" about an unload that
    /// plainly succeeded.
    pub fn unload(&mut self, id: u64) -> Result<()> {
        if !self.fibers.contains_key(&id) {
            return Ok(());
        }
        self.deactivate(id);
        self.fibers.remove(&id);
        self.version += 1;
        let _ = self.refresh();
        Ok(())
    }

    /// Give up one session's claim on a fiber.
    ///
    /// The instance is torn down only when the last user lets go, which is the
    /// rule that makes session isolation real: the first session to unload a
    /// shared pack must not remove the capability from the second one. A fiber
    /// nobody claims (loaded by the kernel or a pack, with no session) has no
    /// users to outlive it, so releasing it unloads it.
    pub fn release(&mut self, session: &str, id: u64) -> Result<ReleaseOutcome> {
        let Some(fiber) = self.fibers.get_mut(&id) else {
            return Ok(ReleaseOutcome::Absent);
        };
        let dropped = fiber.users.remove(session);
        let remaining = fiber.users.len();
        if remaining > 0 {
            if dropped {
                self.version += 1;
            }
            return Ok(ReleaseOutcome::StillUsed(remaining));
        }
        self.unload(id)?;
        Ok(ReleaseOutcome::Unloaded)
    }

    /// Release every claim `session` holds.
    ///
    /// Returns the fibers that were torn down as a result; instances other
    /// sessions still hold survive with this session's claim removed.
    pub fn unload_session(&mut self, session: &str) -> Result<Vec<u64>> {
        let held: Vec<u64> = self
            .fibers
            .values()
            .filter(|fiber| fiber.is_used_by(session))
            .map(|fiber| fiber.id)
            .collect();
        let mut dropped = Vec::new();
        for id in held {
            if matches!(self.release(session, id)?, ReleaseOutcome::Unloaded) {
                dropped.push(id);
            }
        }
        Ok(dropped)
    }

    /// Release every claim held by any session except `keep`.
    ///
    /// This is the desktop's "switch session" teardown: the session being
    /// switched to keeps its capabilities, every other session's claims go, and
    /// a fiber that only another session held is unloaded. Fibers that `keep`
    /// also holds stay running — the difference from [`Kernel::unload`], which
    /// would take them away from the session that is still using them.
    pub fn unload_all_except(&mut self, keep: &str) -> Result<Vec<u64>> {
        let mut sessions: BTreeSet<String> = BTreeSet::new();
        for fiber in self.fibers.values() {
            for user in fiber.users() {
                if user != keep {
                    sessions.insert(user);
                }
            }
        }
        let mut dropped = Vec::new();
        for session in sessions {
            dropped.extend(self.unload_session(&session)?);
        }
        Ok(dropped)
    }

    /// Reload a fiber in place, keeping its id, realm and users.
    ///
    /// Build first, swap second: the replacement is staged — `apply` runs and
    /// every claim is checked — and the running instance is only torn down once
    /// that succeeded. If it did not, this returns the error and the old version
    /// keeps running untouched.
    pub fn reload(&mut self, id: u64) -> Result<()> {
        let Some(fiber) = self.fibers.get(&id) else {
            return Err(anyhow!("fiber {id} is not loaded"));
        };
        let name = fiber.plugin.clone();
        let realm = fiber.realm.clone();
        let config = fiber.config.clone();
        let desired = self.compute_epoch(&fiber.inject, &fiber.realm);

        if desired == Epoch::Inactive {
            // Nothing can be applied while a dependency is missing: take the old
            // contributions down and leave the fiber armed for when it returns.
            self.deactivate(id);
            if let Some(fiber) = self.fibers.get_mut(&id) {
                fiber.epoch = Epoch::Inactive;
                fiber.error = None;
            }
            let _ = self.refresh();
            return Ok(());
        }

        let mut claims = StagedClaims::default();
        // The `?` is the rollback: nothing has been torn down yet.
        self.rebuild(id, &name, &realm, &config, desired, Some(id), &mut claims)?;
        if let Some(fiber) = self.fibers.get_mut(&id) {
            fiber.error = None;
        }
        let _ = self.refresh();
        Ok(())
    }

    /// Re-apply every fiber of one plugin.
    ///
    /// [`Kernel::refresh`] reacts to *service* changes through epochs, which
    /// never fire for a plugin whose configuration lives in a cell the host
    /// owns. This is the other trigger: the host changed that cell and asked
    /// for the plugin's contributions to be rebuilt from it.
    ///
    /// All or nothing. Every target is staged before any target is touched, so a
    /// fiber that refuses to re-apply leaves the whole plugin on its previous
    /// version — the previous implementation unloaded and reloaded one fiber at
    /// a time and let a `?` in the middle leave the table half old and half new.
    ///
    /// Returns how many fibers were re-applied.
    pub fn reload_plugin(&mut self, name: &str) -> Result<usize> {
        let mut targets: Vec<(u64, RealmMap, Value, Epoch)> = Vec::new();
        for fiber in self.fibers.values().filter(|fiber| fiber.plugin == name) {
            let desired = self.compute_epoch(&fiber.inject, &fiber.realm);
            targets.push((fiber.id, fiber.realm.clone(), fiber.config.clone(), desired));
        }

        // Phase one — build every replacement. Nothing is torn down yet.
        let mut claims = StagedClaims::default();
        let mut plan: Vec<(u64, Option<Staged>)> = Vec::with_capacity(targets.len());
        for (id, realm, config, desired) in targets {
            if desired == Epoch::Inactive {
                plan.push((id, None));
                continue;
            }
            let staged = self.stage(id, name, &realm, &config, desired, Some(id), &mut claims)?;
            plan.push((id, Some(staged)));
        }

        // Phase two — swap. Installation cannot fail, so the batch lands
        // together.
        let mut reloaded = 0;
        for (id, staged) in plan {
            self.deactivate(id);
            match staged {
                Some(staged) => {
                    self.install(staged);
                    reloaded += 1;
                }
                None => {
                    if let Some(fiber) = self.fibers.get_mut(&id) {
                        fiber.epoch = Epoch::Inactive;
                        fiber.error = None;
                    }
                }
            }
        }
        let _ = self.refresh();
        Ok(reloaded)
    }

    /// Which pack activated each plugin, for display.
    ///
    /// Pairs with [`Kernel::plugin_names`]: the panel reads `delegate` back as
    /// `delegate ← subagent`, so a capability has a pack to point at. Plugins
    /// with no pack behind them are absent — those belong to the kernel
    /// itself. If two packs ever load the same plugin, the first one is kept.
    pub fn plugin_origins(&self) -> std::collections::BTreeMap<String, String> {
        let mut origins = std::collections::BTreeMap::new();
        for fiber in self.fibers.values() {
            if let Some(origin) = &fiber.origin {
                origins
                    .entry(fiber.plugin.clone())
                    .or_insert_with(|| origin.clone());
            }
        }
        origins
    }

    /// Recompute every fiber's epoch and reconcile the ones that changed.
    ///
    /// This is the whole of hot reload: nothing calls `reload` by hand. A fiber
    /// whose dependency disappears is deactivated but kept, so it activates
    /// again by itself when the dependency returns.
    ///
    /// Nothing in here fails the whole pass. A fiber that cannot be rebuilt is
    /// reported as a [`ReloadEvent`] with `action` `failed` or `rebuild-failed`
    /// and carries its reason on itself; returning `Err` from the middle of the
    /// pass is what used to leave the other fibers unreconciled.
    pub fn refresh(&mut self) -> Result<Vec<ReloadEvent>> {
        /// Passes before the fixed-point iteration gives up.
        ///
        /// Only a pathological cycle can reach this, and reaching it is now
        /// *reported*: the previous version capped at 64 passes and returned as
        /// if everything were reconciled, leaving stale epochs behind silently.
        const MAX_PASSES: usize = 64;

        let mut events = Vec::new();
        for pass in 1..=MAX_PASSES {
            let mut progressed = false;
            let ids: Vec<u64> = self.fibers.keys().copied().collect();
            for id in ids {
                if self.reconcile(id, &mut events) {
                    progressed = true;
                }
            }
            if !progressed {
                return Ok(events);
            }
            if pass == MAX_PASSES {
                eprintln!(
                    "[plugin] refresh stopped after {MAX_PASSES} passes with epochs still changing"
                );
                events.push(ReloadEvent {
                    fiber: 0,
                    plugin: "<kernel>".to_string(),
                    action: "truncated".to_string(),
                    reason: format!(
                        "epochs were still changing after {MAX_PASSES} passes; some fibers keep \
                         a stale epoch"
                    ),
                });
            }
        }
        Ok(events)
    }

    /// Move one fiber toward its epoch. Returns whether anything was attempted.
    fn reconcile(&mut self, id: u64, events: &mut Vec<ReloadEvent>) -> bool {
        let Some(fiber) = self.fibers.get(&id) else {
            return false;
        };
        let current = fiber.epoch.clone();
        let desired = self.compute_epoch(&fiber.inject, &fiber.realm);
        if current == desired {
            return false;
        }
        // A fiber that already failed to apply is not retried here: a
        // deterministic failure would spin for ever and never converge.
        // `reload` is the deliberate retry.
        if fiber.state == FiberState::Failed {
            return false;
        }
        let name = fiber.plugin.clone();
        let realm = fiber.realm.clone();
        let config = fiber.config.clone();
        let was_active = matches!(current, Epoch::Active(_));
        let will_be_active = matches!(desired, Epoch::Active(_));

        match (was_active, will_be_active) {
            (true, false) => {
                events.push(ReloadEvent {
                    fiber: id,
                    plugin: name,
                    action: "deactivated".to_string(),
                    reason: "a dependency disappeared".to_string(),
                });
                self.deactivate(id);
                if let Some(fiber) = self.fibers.get_mut(&id) {
                    fiber.epoch = Epoch::Inactive;
                }
            }
            (false, true) => {
                let outcome = self.activate_at(id, desired);
                events.push(ReloadEvent {
                    fiber: id,
                    plugin: name,
                    action: (if outcome.is_ok() { "activated" } else { "failed" }).to_string(),
                    reason: match &outcome {
                        Ok(()) => "dependencies became available".to_string(),
                        Err(error) => format!("{error:#}"),
                    },
                });
            }
            (true, true) => {
                // Build the replacement first: if it does not build, the version
                // that is running keeps its contributions instead of the kernel
                // being left with a torn-down fiber and nothing to put back.
                let mut claims = StagedClaims::default();
                let outcome =
                    self.rebuild(id, &name, &realm, &config, desired.clone(), Some(id), &mut claims);
                match outcome {
                    Ok(()) => events.push(ReloadEvent {
                        fiber: id,
                        plugin: name,
                        action: "reloaded".to_string(),
                        reason: "a dependency was replaced".to_string(),
                    }),
                    Err(error) => {
                        // The old version is still installed and still correct,
                        // so the fiber stays `Active`; accepting the epoch stops
                        // the failure from being retried on every pass, and the
                        // reason stays on the fiber for a panel to show.
                        if let Some(fiber) = self.fibers.get_mut(&id) {
                            fiber.epoch = desired;
                            fiber.error = Some(format!("{error:#}"));
                        }
                        events.push(ReloadEvent {
                            fiber: id,
                            plugin: name,
                            action: "rebuild-failed".to_string(),
                            reason: format!("{error:#}"),
                        });
                    }
                }
            }
            (false, false) => {
                if let Some(fiber) = self.fibers.get_mut(&id) {
                    fiber.epoch = Epoch::Inactive;
                }
            }
        }
        true
    }

    /// Tear down one fiber's contributions while keeping the fiber itself.
    ///
    /// Disposers run first, and the fiber is never removed from the map while
    /// they run: an effect that looks its own fiber or service up during
    /// teardown still finds them, and one that panics cannot take the remaining
    /// disposers — or the fiber — down with it.
    fn deactivate(&mut self, id: u64) {
        let mut effects = match self.fibers.get_mut(&id) {
            Some(fiber) => std::mem::take(&mut fiber.effects),
            None => return,
        };

        // Reverse order: later effects may depend on earlier ones.
        while let Some(disposer) = effects.pop() {
            run_disposer(disposer);
        }

        let dropped_services = {
            let mut table = self.services.write();
            let before = table.len();
            table.retain(|_, entry| entry.fiber != id);
            before != table.len()
        };
        if dropped_services {
            self.services.generation.fetch_add(1, Ordering::Relaxed);
        }

        if let Some(tools) = self.fiber_tools.remove(&id) {
            for tool in tools {
                self.tools.unregister(&tool);
            }
        }

        if let Some(fiber) = self.fibers.get_mut(&id) {
            fiber.state = FiberState::Pending;
            fiber.ui.clear();
            fiber.themes.clear();
            fiber.strings.clear();
        }
        self.version += 1;
    }

    fn compute_epoch(&self, inject: &[String], realm: &RealmMap) -> Epoch {
        let table = self.services.read();
        let mut parts = Vec::with_capacity(inject.len());
        for name in inject {
            let key = (name.clone(), realm.realm_of(name));
            match table.get(&key) {
                // The instance serial makes a re-applied provider — same fiber,
                // new value — a different dependency, so dependents rebuild.
                Some(entry) => parts.push(format!("{name}:{}:{}", entry.fiber, entry.serial)),
                // A missing dependency keeps the fiber inactive.
                None => return Epoch::Inactive,
            }
        }
        Epoch::Active(parts.join(","))
    }

    fn next_serial(&mut self) -> u64 {
        self.instance_counter += 1;
        self.instance_counter
    }

    /// Install a fiber's contributions, if its dependencies are present.
    ///
    /// On failure the fiber is marked `Failed` with the reason and keeps its old
    /// epoch, so the automatic pass leaves it alone until something asks for a
    /// retry through [`Kernel::reload`].
    fn activate(&mut self, id: u64) -> Result<()> {
        let Some(fiber) = self.fibers.get(&id) else {
            return Ok(());
        };
        let desired = self.compute_epoch(&fiber.inject, &fiber.realm);
        if desired == Epoch::Inactive {
            return Ok(());
        }
        self.activate_at(id, desired)
    }

    fn activate_at(&mut self, id: u64, desired: Epoch) -> Result<()> {
        let Some(fiber) = self.fibers.get(&id) else {
            return Ok(());
        };
        let name = fiber.plugin.clone();
        let realm = fiber.realm.clone();
        let config = fiber.config.clone();

        let mut claims = StagedClaims::default();
        match self.stage(id, &name, &realm, &config, desired, Some(id), &mut claims) {
            Ok(staged) => {
                self.install(staged);
                Ok(())
            }
            Err(error) => {
                if let Some(fiber) = self.fibers.get_mut(&id) {
                    fiber.state = FiberState::Failed;
                    fiber.error = Some(format!("{error:#}"));
                }
                Err(error)
            }
        }
    }

    /// Replace one fiber's contributions, tearing the old ones down only after
    /// the new ones exist.
    fn rebuild(
        &mut self,
        id: u64,
        name: &str,
        realm: &RealmMap,
        config: &Value,
        desired: Epoch,
        ignore: Option<u64>,
        claims: &mut StagedClaims,
    ) -> Result<()> {
        let staged = self.stage(id, name, realm, config, desired, ignore, claims)?;
        self.deactivate(id);
        self.install(staged);
        Ok(())
    }

    /// Run a plugin's `apply` and check every claim it makes, without changing
    /// anything the kernel owns.
    ///
    /// This is the half of activation that can fail. Everything decidable before
    /// the kernel's tables change is decided here, which is what makes
    /// [`Kernel::install`] infallible — and therefore what lets a reload leave
    /// the running version untouched when its replacement does not build.
    ///
    /// `ignore` names the fiber whose own entries are being replaced: a reload
    /// re-takes exactly the names its predecessor holds, which is not a clash.
    fn stage(
        &self,
        fiber: u64,
        plugin_name: &str,
        realm: &RealmMap,
        config: &Value,
        epoch: Epoch,
        ignore: Option<u64>,
        claims: &mut StagedClaims,
    ) -> Result<Staged> {
        let plugin = self
            .plugins
            .get(plugin_name)
            .cloned()
            .ok_or_else(|| anyhow!("plugin '{plugin_name}' is not defined"))?;

        let ctx = PluginCtx {
            plugin: plugin_name.to_string(),
            fiber,
            realm: realm.clone(),
            services: self.service_view(realm.clone()),
            config: config.clone(),
            events: self.tools.events(),
        };

        // A plugin that panics is a plugin that failed, not a kernel that dies:
        // the panic is stopped at this boundary and reported like any other
        // `apply` failure. (`AssertUnwindSafe` is honest here — the kernel's own
        // state is untouched by `apply`, which only reads the service view.)
        let applied = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| plugin.apply(&ctx)));
        let contributions = match applied {
            Ok(Ok(contributions)) => contributions,
            Ok(Err(error)) => {
                return Err(anyhow!("plugin '{plugin_name}' failed to apply: {error:#}"));
            }
            Err(payload) => {
                return Err(anyhow!(
                    "plugin '{plugin_name}' panicked while applying: {}",
                    panic_text(payload)
                ));
            }
        };

        // From here on `apply` has already run, so every rejection has to undo
        // what it started rather than abandon it.
        let Contributions {
            ui,
            themes,
            strings,
            services: provided,
            tools: provided_tools,
            readers,
            effects,
        } = contributions;

        let mut staged_services: Vec<((String, RealmId), Arc<dyn Any + Send + Sync>)> =
            Vec::with_capacity(provided.len());
        let mut staged_tools: Vec<ToolDef> = Vec::with_capacity(provided_tools.len());
        let mut problem: Option<String> = None;

        for (name, value) in provided {
            let key = (name.clone(), realm.realm_of(&name));
            let held = self.services.read().get(&key).map(|entry| (entry.fiber, entry.owner.clone()));
            if let Some((holder, owner)) = held {
                if Some(holder) != ignore {
                    problem = Some(format!(
                        "service '{name}' is already provided by '{owner}' in this realm"
                    ));
                    break;
                }
            }
            if let Some(owner) = claims.services.get(&key) {
                problem = Some(format!(
                    "service '{name}' is claimed by both '{owner}' and '{plugin_name}' in this realm"
                ));
                break;
            }
            claims.services.insert(key.clone(), plugin_name.to_string());
            staged_services.push((key, value));
        }

        if problem.is_none() {
            for def in provided_tools {
                let name = def.name.clone();
                // The kernel owns the tool table, so it stamps ownership itself:
                // a plugin does not get to claim someone else's tool.
                let replacing_own = ignore
                    .and_then(|fiber| {
                        self.fiber_tools
                            .get(&fiber)
                            .map(|tools| tools.iter().any(|held| held == &name))
                    })
                    .unwrap_or(false);
                if !replacing_own {
                    if let Some(owner) = self.tools.owner(&name) {
                        problem =
                            Some(format!("tool '{name}' is already registered by '{owner}'"));
                        break;
                    }
                }
                if let Some(owner) = claims.tools.get(&name) {
                    problem = Some(format!(
                        "tool '{name}' is claimed by both '{owner}' and '{plugin_name}'"
                    ));
                    break;
                }
                claims.tools.insert(name, plugin_name.to_string());
                staged_tools.push(ToolDef { owner: plugin_name.to_string(), ..def });
            }
        }

        if let Some(problem) = problem {
            unwind_effects(effects);
            return Err(anyhow!("{problem}"));
        }

        Ok(Staged {
            fiber,
            plugin: plugin_name.to_string(),
            epoch,
            ui,
            themes,
            strings,
            services: staged_services,
            tools: staged_tools,
            readers,
            effects,
        })
    }

    /// Install a [`Staged`] contribution set.
    ///
    /// Infallible by construction: every check that can fail happened in
    /// [`Kernel::stage`]. Where the tool registry still refuses a registration
    /// the kernel says so once and carries on rather than pretending the tool
    /// exists.
    fn install(&mut self, staged: Staged) {
        let Staged {
            fiber,
            plugin,
            epoch,
            ui,
            themes,
            strings,
            services,
            tools,
            readers,
            effects,
        } = staged;

        let mut registered: Vec<String> = Vec::with_capacity(tools.len());
        for def in tools {
            let name = def.name.clone();
            match self.tools.register(def, ConflictPolicy::Error) {
                Ok(()) => registered.push(name),
                Err(error) => {
                    eprintln!("[plugin] '{plugin}' could not register tool '{name}': {error:#}");
                }
            }
        }

        let mut incoming: Vec<((String, RealmId), ServiceEntry)> = Vec::with_capacity(services.len());
        for (key, value) in services {
            let serial = self.next_serial();
            incoming.push((key, ServiceEntry { value, fiber, owner: plugin.clone(), serial }));
        }
        if !incoming.is_empty() {
            {
                let mut table = self.services.write();
                for (key, entry) in incoming {
                    table.insert(key, entry);
                }
            }
            self.services.generation.fetch_add(1, Ordering::Relaxed);
        }

        // Readers join the fiber's effects: unload unregisters them
        // (registrations are undone, not dropped), with ownership stamped
        // here — a plugin does not get to claim provenance.
        let mut effects = effects;
        for spec in readers {
            let dispose = crate::tools::readers::register(spec, plugin.clone());
            effects.push(Box::new(dispose));
        }

        self.fiber_tools.insert(fiber, registered);
        if let Some(target) = self.fibers.get_mut(&fiber) {
            target.ui = ui;
            target.themes = themes;
            target.strings = strings;
            target.effects = effects;
            target.epoch = epoch;
            target.state = FiberState::Active;
            target.error = None;
        }
        self.version += 1;
    }
}

/// Run one disposer, isolating a panic so the rest still run.
fn run_disposer(disposer: Disposer) {
    if std::panic::catch_unwind(std::panic::AssertUnwindSafe(disposer)).is_err() {
        eprintln!("[plugin] a disposer panicked; the remaining ones still run");
    }
}

/// Undo the effects of a contribution set that will not be installed.
///
/// Dropping a disposer does not undo its effect, so a rejected `apply` has to be
/// unwound explicitly — otherwise a plugin that was refused after starting
/// something would leak exactly what this module promises to reverse.
fn unwind_effects(effects: Vec<Disposer>) {
    for disposer in effects.into_iter().rev() {
        run_disposer(disposer);
    }
}

/// The message from a panic payload, for error reporting.
fn panic_text(payload: Box<dyn Any + Send>) -> String {
    if let Some(text) = payload.downcast_ref::<&str>() {
        (*text).to_string()
    } else if let Some(text) = payload.downcast_ref::<String>() {
        text.clone()
    } else {
        "a panic with a non-string payload".to_string()
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

        // The clash is caught at install time even though `second` never
        // declared the service it provides — and it is reported as a failed
        // load, not as a load that succeeded and then did nothing.
        let error = kernel
            .load("second", RealmMap::new(), Value::Null)
            .expect_err("a service clash fails the load");
        let text = format!("{error:#}");
        assert!(text.contains("already provided"), "{text}");
        assert!(text.contains("first"), "the existing owner is named: {text}");

        // The reason stays on the fiber, so a panel can show why.
        let state = kernel
            .fibers()
            .into_iter()
            .find(|fiber| fiber.plugin == "second")
            .map(|fiber| fiber.state);
        assert_eq!(state, Some(FiberState::Failed));

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

        kernel
            .load("clash", RealmMap::new(), Value::Null)
            .expect_err("the clashing load fails");
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
        kernel
            .load("latecomer", RealmMap::new(), Value::Null)
            .expect_err("the service clash fails the load");

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

        // The load reports the failure — a plugin that never ran is not a
        // success — and the reason is readable both ways.
        let error = kernel
            .load("broken", RealmMap::new(), Value::Null)
            .expect_err("a plugin that cannot apply fails its load");
        assert!(format!("{error:#}").contains("cannot start"), "{error:#}");

        let fiber = kernel
            .fibers()
            .into_iter()
            .find(|fiber| fiber.plugin == "broken")
            .expect("the failed fiber stays for diagnostics");
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

    /// A contributed reader exists exactly while its fiber does: registration
    /// rides the fiber's effects, so unload unregisters (the registry's
    /// promise — 读不了就装包 means the pack's arrival *and* departure both
    /// show in what `read` can open).
    #[test]
    fn a_plugin_reader_is_registered_on_load_and_removed_on_unload() {
        use crate::tools::readers::{ReadAnswer, ReaderSpec};
        use std::path::Path;

        let plugin = Named::new("format", |_| {
            Ok(Contributions::new().reader(ReaderSpec {
                id: "test-fmt".into(),
                claims: Arc::new(|bytes: &[u8], _path: &Path| bytes.starts_with(b"FMT")),
                convert: Arc::new(|bytes: &[u8], _path: &Path| {
                    crate::tools::readers::ReadOutcome::Text(
                        String::from_utf8_lossy(&bytes[3..]).into_owned(),
                    )
                }),
            }))
        });

        let mut kernel = Kernel::new();
        kernel.define(plugin);
        let fiber = kernel.load("format", RealmMap::new(), Value::Null).unwrap();

        let claimed = crate::tools::readers::convert(b"FMT readable", Path::new("x.bin"));
        assert!(
            matches!(claimed, ReadAnswer::Converted(ref text) if text == " readable"),
            "the pack taught read its format"
        );
        assert!(
            crate::tools::readers::installed().iter().any(|(id, owner)| id == "test-fmt" && owner == "format"),
            "with ownership stamped by the kernel, not the plugin"
        );

        kernel.unload(fiber).unwrap();
        assert!(
            matches!(
                crate::tools::readers::convert(b"FMT readable", Path::new("x.bin")),
                ReadAnswer::Unclaimed
            ),
            "unload took the reader: registrations are undone, not dropped"
        );
    }

    #[test]
    fn an_instance_is_shared_by_its_sessions_and_dies_with_the_last_one() {
        let mut kernel = Kernel::new();
        kernel.define(Named::new("provider", |_| {
            Ok(Contributions::new().service("thing", Arc::new(1u8)))
        }));

        let realm = RealmMap::new().with_default(kernel.new_realm());
        let first = kernel
            .load_for_session("alice", "provider", realm.clone(), Value::Null)
            .unwrap();
        let second = kernel
            .load_for_session("bob", "provider", realm.clone(), Value::Null)
            .unwrap();
        assert_eq!(first, second, "one instance per (plugin, realm), not one per session");
        assert_eq!(
            kernel.users_of(first),
            vec!["alice".to_string(), "bob".to_string()],
            "the instance knows which sessions hold it"
        );

        // The session that leaves first must not take the capability with it.
        assert_eq!(
            kernel.release("alice", first).unwrap(),
            ReleaseOutcome::StillUsed(1)
        );
        assert!(kernel.service_view(realm.clone()).has("thing"));
        assert_eq!(kernel.fiber(first).unwrap().state, FiberState::Active);

        // The last one does.
        assert_eq!(
            kernel.release("bob", first).unwrap(),
            ReleaseOutcome::Unloaded
        );
        assert!(!kernel.service_view(realm).has("thing"));
        assert!(kernel.fiber(first).is_none());
        assert_eq!(
            kernel.release("bob", first).unwrap(),
            ReleaseOutcome::Absent,
            "a second release is not an error"
        );
    }

    #[test]
    fn switching_sessions_releases_only_the_other_sessions_claims() {
        let mut kernel = Kernel::new();
        kernel.define(Named::new("mine", |_| {
            Ok(Contributions::new().service("mine", Arc::new(1u8)))
        }));
        kernel.define(Named::new("theirs", |_| {
            Ok(Contributions::new().service("theirs", Arc::new(2u8)))
        }));

        let mine = kernel
            .load_for_session("alice", "mine", RealmMap::new(), Value::Null)
            .unwrap();
        kernel
            .load_for_session("bob", "theirs", RealmMap::new(), Value::Null)
            .unwrap();

        let dropped = kernel.unload_all_except("alice").unwrap();
        assert_eq!(dropped.len(), 1, "only the other session's instance went: {dropped:?}");
        assert!(kernel.service_view(RealmMap::new()).has("mine"));
        assert!(!kernel.service_view(RealmMap::new()).has("theirs"));
        assert_eq!(kernel.fiber(mine).unwrap().state, FiberState::Active);
    }

    #[test]
    fn a_reload_that_cannot_apply_keeps_the_running_version() {
        let mut kernel = Kernel::new();
        let healthy = Arc::new(std::sync::atomic::AtomicBool::new(true));
        let flag = Arc::clone(&healthy);
        kernel.define(Named::new("flaky", move |_| {
            if flag.load(Ordering::SeqCst) {
                Ok(Contributions::new().tool(empty_tool("flaky_tool")))
            } else {
                Err(anyhow!("the replacement does not build"))
            }
        }));

        let fiber = kernel.load("flaky", RealmMap::new(), Value::Null).unwrap();
        assert!(kernel.tools().get("flaky_tool").is_some());

        healthy.store(false, Ordering::SeqCst);
        let error = kernel.reload(fiber).expect_err("the replacement fails");
        assert!(format!("{error:#}").contains("does not build"), "{error:#}");

        // Building first and swapping second is what keeps this true: the old
        // version is still installed, still owned, and still callable.
        assert!(kernel.tools().get("flaky_tool").is_some());
        assert_eq!(kernel.tools().owner("flaky_tool"), Some("flaky"));
        assert_eq!(kernel.tools_of(fiber), vec!["flaky_tool".to_string()]);
        assert_eq!(kernel.fiber(fiber).unwrap().state, FiberState::Active);
    }

    #[test]
    fn a_failed_reload_in_a_batch_rolls_back_every_fiber() {
        let mut kernel = Kernel::new();
        let stubborn = kernel.new_realm();
        let armed = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let flag = Arc::clone(&armed);
        kernel.define(Named::new("plugin", move |ctx| {
            if flag.load(Ordering::SeqCst) && ctx.realm.default_realm() == stubborn {
                return Err(anyhow!("this realm refuses to re-apply"));
            }
            Ok(Contributions::new().tool(empty_tool(&format!("tool_{}", ctx.fiber))))
        }));

        let realm_a = RealmMap::new().with_default(kernel.new_realm());
        let realm_b = RealmMap::new().with_default(stubborn);
        let a = kernel.load("plugin", realm_a, Value::Null).unwrap();
        let b = kernel.load("plugin", realm_b, Value::Null).unwrap();

        armed.store(true, Ordering::SeqCst);
        let error = kernel
            .reload_plugin("plugin")
            .expect_err("the batch cannot complete");
        assert!(format!("{error:#}").contains("refuses to re-apply"), "{error:#}");

        // Every target was staged before any was touched, so neither fiber lost
        // anything: no half-old, half-new table.
        assert!(kernel.tools().get(&format!("tool_{a}")).is_some());
        assert!(kernel.tools().get(&format!("tool_{b}")).is_some());
        assert_eq!(kernel.fiber(a).unwrap().state, FiberState::Active);
        assert_eq!(kernel.fiber(b).unwrap().state, FiberState::Active);
    }

    #[test]
    fn a_panicking_disposer_does_not_stop_the_others() {
        let ran: Arc<AtomicU64> = Arc::new(AtomicU64::new(0));
        let tally = Arc::clone(&ran);

        let mut kernel = Kernel::new();
        kernel.define(Named::new("messy", move |_| {
            // Cloned per apply: a `Fn` plugin body may not move its own captures
            // into the disposers it hands back.
            let first = Arc::clone(&tally);
            let second = Arc::clone(&tally);
            Ok(Contributions::new()
                // Registered first, so it runs last.
                .effect(move || {
                    first.fetch_add(1, Ordering::SeqCst);
                })
                .effect(move || {
                    second.fetch_add(1, Ordering::SeqCst);
                    panic!("this disposer is deliberately broken");
                }))
        }));

        let fiber = kernel.load("messy", RealmMap::new(), Value::Null).unwrap();
        kernel.unload(fiber).unwrap();

        assert_eq!(
            ran.load(Ordering::SeqCst),
            2,
            "both disposers ran despite one of them panicking"
        );
        assert!(
            kernel.fiber(fiber).is_none(),
            "teardown finished: the fiber is gone, not stranded half-torn-down"
        );
    }

    #[test]
    fn a_service_view_goes_stale_when_the_table_changes() {
        let mut kernel = Kernel::new();
        kernel.define(Named::new("provider", |_| {
            Ok(Contributions::new().service("thing", Arc::new(7u8)))
        }));

        let fiber = kernel.load("provider", RealmMap::new(), Value::Null).unwrap();
        let view = kernel.service_view(RealmMap::new());
        assert_eq!(*view.get::<u8>("thing").unwrap(), 7);
        assert!(view.is_current());

        // Unloading changes the table. The view taken before must not keep
        // serving a service that is no longer registered, and must say why.
        kernel.unload(fiber).unwrap();
        assert!(!view.is_current());
        match view.try_get::<u8>("thing") {
            Err(ServiceViewError::Stale { .. }) => {}
            other => panic!("expected a stale view, got {other:?}"),
        }
        assert!(view.get::<u8>("thing").is_none());
        assert!(!view.has("thing"));

        // A fresh view is current again, and reports the truth.
        let fresh = kernel.service_view(RealmMap::new());
        assert!(fresh.is_current());
        assert!(!fresh.has("thing"));
    }

    #[test]
    fn a_dependency_reload_reapplies_its_dependent_with_the_same_configuration() {
        struct Recorder {
            seen: Arc<std::sync::Mutex<Vec<String>>>,
        }

        impl Plugin for Recorder {
            fn name(&self) -> &str {
                "consumer"
            }
            fn inject(&self) -> Vec<String> {
                vec!["thing".to_string()]
            }
            fn apply(&self, ctx: &PluginCtx) -> Result<Contributions> {
                self.seen.lock().unwrap().push(ctx.config.to_string());
                Ok(Contributions::new())
            }
        }

        let seen: Arc<std::sync::Mutex<Vec<String>>> = Arc::new(std::sync::Mutex::new(Vec::new()));

        let mut kernel = Kernel::new();
        kernel.define(Arc::new(Recorder { seen: Arc::clone(&seen) }));
        kernel.define(Named::new("provider", |_| {
            Ok(Contributions::new().service("thing", Arc::new(1u8)))
        }));

        let realm = RealmMap::new();
        let consumer = kernel
            .load("consumer", realm.clone(), serde_json::json!({ "answer": 42 }))
            .unwrap();
        assert_eq!(kernel.fiber(consumer).unwrap().state, FiberState::Pending);

        kernel.load("provider", realm.clone(), Value::Null).unwrap();
        kernel.refresh().unwrap();
        assert_eq!(kernel.fiber(consumer).unwrap().state, FiberState::Active);

        // Replace the dependency: the kernel rebuilds the dependent on its own,
        // and must rebuild it with the configuration the host gave it rather
        // than with `Null`.
        let provider = kernel.fiber_of("provider", &realm).unwrap();
        kernel.unload(provider).unwrap();
        kernel.load("provider", realm.clone(), Value::Null).unwrap();
        kernel.refresh().unwrap();

        let configs = seen.lock().unwrap().clone();
        assert!(configs.len() >= 2, "the consumer was re-applied: {configs:?}");
        assert!(
            configs.iter().all(|config| config.contains("42")),
            "configuration survived the automatic reload: {configs:?}"
        );
    }
}
