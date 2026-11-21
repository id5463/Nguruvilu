//! The plugin kernel: the [`Plugin`] contract and the machinery that runs it.
//!
//! [`Kernel`] holds the fibers, the service table, the tool table and the
//! serials. Activation is split into stage — run `apply` and check every claim
//! — and install, which cannot fail, so a reload builds the replacement before
//! it tears the running version down.

use crate::tools::{ConflictPolicy, ToolDef, ToolRegistry};
use anyhow::{anyhow, Result};
use serde_json::Value;
use std::any::Any;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use super::ctx::{Contributions, Disposer, PluginCtx, StateHandle};
use super::fiber::{Epoch, Fiber, FiberState, ReleaseOutcome, Staged, StagedClaims};
use super::realm::{RealmId, RealmMap};
use super::seam::{Seam, ToolCallView, TurnContext};
use super::service::{ServiceEntry, ServiceTable, ServiceView};

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

    // ---- what a plugin may do while the agent runs ----
    //
    // These are methods, not a registry of named points. The difference
    // matters: a registry means the kernel publishes a fixed list of moments
    // and a plugin hopes one matches; a method means the kernel asks every
    // plugin the question it already knows how to ask. Adding a moment is
    // adding a method here — the plugin says what it can do by implementing
    // it, and there is no name to misspell and no table to look up.

    /// Text to append to this turn's system prompt.
    ///
    /// Called once per turn, after the base prompt and the skill catalog are
    /// assembled, with the user's message in hand. Returning nothing is the
    /// common case; returning text is how a retriever says "you will probably
    /// need these skills" without the kernel knowing what a retriever is.
    fn build_prompt(&self, _ctx: &TurnContext) -> Option<String> {
        None
    }

    /// Which of these tool names survive for one request.
    ///
    /// Runs after the registry is snapshotted and before the body is built, so
    /// a server advertising fifty-five primitives can be cut to the six a task
    /// needs. Names rather than definitions: the registry owns the handlers, and
    /// handing them out to be rebuilt invites a plugin to drop one by accident.
    /// `None` leaves the table whole, and nothing is unregistered either way —
    /// the next turn starts from the full set again.
    fn filter_tools(&self, _ctx: &TurnContext, _names: Vec<String>) -> Option<Vec<String>> {
        None
    }

    /// A last word before a tool call runs.
    ///
    /// Returning `Some(reason)` refuses the call and hands the reason back to
    /// the model as the tool result. This is advice with teeth the other
    /// directions do not have: it is the only place a plugin can say no.
    fn before_tool_call(&self, _ctx: &TurnContext, _call: &ToolCallView) -> Option<String> {
        None
    }

    /// News after a tool call returned.
    ///
    /// Observational, like the event bus, but typed: the output is here rather
    /// than in a JSON blob, and a plugin that wants to count calls does not
    /// have to parse anything.
    fn after_tool_call(&self, _ctx: &TurnContext, _call: &ToolCallView, _ok: bool) {}

    /// Change one piece of data as it passes through the loop.
    ///
    /// One method for every seam instead of one method per seam. A seam that
    /// needs its own method means the kernel grows a method each time someone
    /// wants to touch something new; a seam that is a `Slot` variant means the
    /// plugin says which data it cares about and does not care how many kinds
    /// exist. Adding a kind is a variant here plus one call site, not a new
    /// trait method that every plugin has to keep implementing.
    ///
    /// The default returns the value untouched, so a plugin that does not care
    /// pays one match.
    fn intercept(&self, _ctx: &TurnContext, seam: Seam) -> Seam {
        seam
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

/// The plugin kernel: fibers, services, permissions.
pub struct Kernel {
    plugins: BTreeMap<String, Arc<dyn Plugin>>,
    fibers: BTreeMap<u64, Fiber>,
    /// Shared with every live [`ServiceView`]; see [`ServiceTable`].
    services: Arc<ServiceTable>,
    /// Tools contributed by each fiber, so unload removes exactly its own.
    fiber_tools: BTreeMap<u64, Vec<String>>,
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
    /// An empty kernel.
    ///
    /// **No tools.** `read`, `write`, `edit`, `bash` and the job tools used to be
    /// registered here, before any pack could load — which meant a pack could
    /// never replace one, and the only way to change `bash` was to edit this
    /// file. They are now published by [`Kernel::install_bootstrap`], which a
    /// pack is free to override because every registration goes through
    /// [`crate::tools::ConflictPolicy::Override`].
    ///
    /// A test that wants the bootstrap set calls that method; a caller that
    /// wants none calls nothing.
    pub fn new() -> Self {
        Self {
            plugins: BTreeMap::new(),
            fibers: BTreeMap::new(),
            services: Arc::new(ServiceTable::default()),
            fiber_tools: BTreeMap::new(),
            tools: ToolRegistry::new(),
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

    /// Register the tools a kernel cannot start without.
    ///
    /// Four file and shell tools plus the job tools behind `bash background`.
    /// Called by the host before anything else loads.
    ///
    /// # Why this is not `Kernel::new`
    ///
    /// A pack that wants a different `bash` — one that logs every command, or
    /// refuses a working directory, or runs in a container — had no way to say
    /// so while the base tools were installed inside `new`: by the time a pack
    /// loaded, the name was taken and the policy was `Error`. Splitting the call
    /// out is what makes the four tools ordinary contributions like any other.
    ///
    /// The names still appear here, and that is honest rather than avoidable:
    /// *something* has to install the tools a conversation cannot function
    /// without. What changed is that installing them is a choice a host makes,
    /// not a fact about the kernel.
    pub fn install_bootstrap(&mut self) -> Result<()> {
        crate::tools::base::register_all(&mut self.tools)
    }

    /// Define every `builtin:` plugin this binary carries.
    ///
    /// # Why this exists as one call
    ///
    /// The list used to be written out three times — in `ngu`'s `run`, in `ngu
    /// apply`, and in the desktop shell — and each copy drifted. `ngu apply` was
    /// the one that fell behind: it never defined `delegate`, `long-horizon` or
    /// `judge`, so `ngu apply packs/starter/assembly.yaml` failed with
    /// "plugin 'delegate' is not defined in this kernel" while the desktop
    /// loaded the same pack without complaint. That is the exact failure mode a
    /// pack system is supposed to make impossible: the same pack working or not
    /// depending on which entrance you came through.
    ///
    /// # You still have to name the binaries you ship
    ///
    /// `builtin:` means "compiled into this binary", so this function is where
    /// the binary admits what it contains. A pack still decides whether any of it
    /// loads — defining is not enabling — and a pack can still replace any of
    /// these tools, because registration is `Override`.
    pub fn define_builtins(&mut self) {
        // Empty on purpose.
        //
        // Every capability this binary used to carry — `restart`, `judge`,
        // `long-horizon`, `delegate` — is a pack now. What is left here are the
        // four base tools, the job tools, and `pack`, all of which a kernel must
        // have to be usable at all: without them a first pack cannot be built,
        // installed, or run.
        //
        // Keeping this function rather than deleting it is deliberate: it is the
        // place a future capability would have to argue its way into, and an
        // empty function is an easier thing to keep empty than a missing one is
        // to stay missing.
    }

    /// The base tools without a kernel, for a caller that only wants a registry.
    ///
    /// [`Kernel::install_bootstrap`] installs them into a kernel; this is the
    /// same set as a free-standing [`crate::tools::ToolRegistry`], which is what
    /// a subagent, a test, or an embedded agent wants.
    pub fn bootstrap_registry() -> Result<crate::tools::ToolRegistry> {
        crate::tools::ToolRegistry::with_base_tools()
    }

    /// Publish a value under a name a plugin can find and change.
    ///
    /// This is how the kernel makes *its own* state reachable: the settings, the
    /// route, the persona, and anything else worth a pack's attention. It is
    /// `StateHandle::set` reached from the kernel side, with the kernel recorded
    /// as the owner so an unload of a pack cannot remove what the kernel put
    /// there.
    pub fn publish_state<T: Any + Send + Sync>(&mut self, name: &str, value: T) {
        self.publish_state_arc(name, Arc::new(value));
    }

    /// [`Kernel::publish_state`] for a value that is already behind an `Arc`.
    ///
    /// The ordinary case: the host keeps its own `Arc<RwLock<Settings>>` so it
    /// does not pay a string lookup on every turn, and publishes the same
    /// allocation so a pack's write lands in the memory the host reads.
    pub fn publish_state_arc(&mut self, name: &str, value: Arc<dyn Any + Send + Sync>) {
        StateHandle::of(Arc::clone(&self.services), "kernel", RealmMap::new())
            .insert_arc(name, value);
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

    /// Ask every active plugin for prompt text, in load order.
    ///
    /// No registry and no point names: the kernel asks the question it knows
    /// how to ask, and a plugin answers by having implemented the method. A
    /// plugin that does not care returns `None` for free.
    pub fn prompt_additions(&self, ctx: &TurnContext) -> Vec<String> {
        let mut out = Vec::new();
        for plugin in self.active_plugins() {
            if let Some(text) = plugin.build_prompt(ctx) {
                if !text.trim().is_empty() {
                    out.push(text);
                }
            }
        }
        out
    }

    /// Run the tool-name list past every active plugin, in load order.
    ///
    /// Each filter sees the previous one's output, so two packs can narrow the
    /// same table without either knowing about the other. The result is for one
    /// request only — nothing is unregistered, so the next turn starts whole.
    pub fn filter_tools(&self, ctx: &TurnContext, mut names: Vec<String>) -> Vec<String> {
        for plugin in self.active_plugins() {
            if let Some(narrowed) = plugin.filter_tools(ctx, names.clone()) {
                names = narrowed;
            }
        }
        names
    }

    /// The first refusal for a tool call, if any plugin has one.
    ///
    /// First rather than all: a refusal is an answer to the model, and two
    /// answers to one call would need a format the tool result does not have.
    /// Load order decides, which is also the order the panel lists packs in.
    pub fn refuse_tool_call(&self, ctx: &TurnContext, call: &ToolCallView) -> Option<String> {
        self.active_plugins()
            .into_iter()
            .find_map(|plugin| plugin.before_tool_call(ctx, call))
    }

    /// Tell every active plugin that a tool call finished.
    pub fn note_tool_call(&self, ctx: &TurnContext, call: &ToolCallView, ok: bool) {
        for plugin in self.active_plugins() {
            plugin.after_tool_call(ctx, call, ok);
        }
    }

    /// A [`crate::agent::PluginHooks`] view over this kernel.
    ///
    /// The loop asks through this rather than holding the kernel: a turn can
    /// run for minutes, and pinning the kernel for its duration would block
    /// every reload. The view upgrades a weak handle per call, so a kernel that
    /// is gone simply answers nothing.
    pub fn hook_view(kernel: &Arc<std::sync::Mutex<Self>>) -> Arc<dyn crate::agent::PluginHooks> {
        Arc::new(KernelHooks {
            kernel: Arc::downgrade(kernel),
        })
    }

    /// The plugins behind active fibers, in fiber order.
    ///
    /// Fiber order rather than name order: it is load order, which is the order
    /// a user sees in the pack panel and the order a later pack's override is
    /// expected to win in.
    fn active_plugins(&self) -> Vec<Arc<dyn Plugin>> {
        let mut fibers: Vec<&Fiber> = self
            .fibers
            .values()
            .filter(|fiber| fiber.state == FiberState::Active)
            .collect();
        fibers.sort_by_key(|fiber| fiber.id);
        fibers
            .into_iter()
            .filter_map(|fiber| self.plugins.get(&fiber.plugin).cloned())
            .collect()
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

    /// The stylesheet rules every active plugin contributes, in load order.
    ///
    /// Returned as one string the shell appends to a single `<style>` element,
    /// so unloading a pack removes exactly the rules that pack added — the
    /// element is rewritten whole on every update rather than appended to. A
    /// pack's rules therefore cannot outlive the pack, which is the same
    /// guarantee `resolved_theme` gives its tokens.
    pub fn resolved_css(&self) -> String {
        let mut fibers: Vec<&Fiber> = self
            .fibers
            .values()
            .filter(|fiber| fiber.state == FiberState::Active)
            .collect();
        // Load order, so a later pack can override an earlier one's selector
        // without depending on a hash map's iteration order.
        fibers.sort_by_key(|fiber| fiber.id);
        let mut out = String::new();
        for fiber in fibers {
            for rules in &fiber.css {
                if out.is_empty() {
                    out.push_str("/* pack stylesheets */\n");
                }
                out.push_str(rules);
                if !rules.ends_with('\n') {
                    out.push('\n');
                }
            }
        }
        out
    }

    /// The scripts every active plugin hangs on interface points, merged.
    ///
    /// Merged rather than picked, because several packs may extend the same
    /// point — that is the whole point of named hooks. Sorted by point name and
    /// then by fiber id, so the order two packs run in is the same on every
    /// start rather than whatever the map happened to iterate first.
    pub fn resolved_hooks(&self) -> std::collections::BTreeMap<String, String> {
        let mut fibers: Vec<&Fiber> = self
            .fibers
            .values()
            .filter(|fiber| fiber.state == FiberState::Active)
            .collect();
        fibers.sort_by_key(|fiber| fiber.id);
        let mut merged: std::collections::BTreeMap<String, String> =
            std::collections::BTreeMap::new();
        for fiber in fibers {
            for (point, script) in &fiber.hooks {
                // Two scripts on one point are concatenated rather than one
                // replacing the other: both packs asked to run there, and the
                // later one is the more specific choice only for *values*, not
                // for code.
                let entry = merged.entry(point.clone()).or_default();
                if !entry.is_empty() {
                    entry.push_str(
                        "
;
",
                    );
                }
                entry.push_str(script);
            }
        }
        merged
    }

    /// The panel kinds every active plugin defines, merged.
    ///
    /// A name defined twice keeps the later fiber's version — the pack loaded
    /// after another is the more specific choice, the same rule labels follow.
    pub fn resolved_panel_types(
        &self,
    ) -> std::collections::BTreeMap<String, crate::pack::PanelType> {
        let mut fibers: Vec<&Fiber> = self
            .fibers
            .values()
            .filter(|fiber| fiber.state == FiberState::Active)
            .collect();
        fibers.sort_by_key(|fiber| fiber.id);
        let mut merged = std::collections::BTreeMap::new();
        for fiber in fibers {
            merged.extend(fiber.panel_types.clone());
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
            let held = self
                .services
                .read()
                .get(&key)
                .map(|entry| entry.owner.clone());
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
                css: Vec::new(),
                hooks: std::collections::BTreeMap::new(),
                panel_types: std::collections::BTreeMap::new(),
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
                    action: (if outcome.is_ok() {
                        "activated"
                    } else {
                        "failed"
                    })
                    .to_string(),
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
                let outcome = self.rebuild(
                    id,
                    &name,
                    &realm,
                    &config,
                    desired.clone(),
                    Some(id),
                    &mut claims,
                );
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
            fiber.css.clear();
            fiber.hooks.clear();
            fiber.panel_types.clear();
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
            state: StateHandle::of(
                Arc::clone(&self.services),
                plugin_name.to_string(),
                realm.clone(),
            ),
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
            css,
            hooks,
            panel_types,
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
            let held = self
                .services
                .read()
                .get(&key)
                .map(|entry| (entry.fiber, entry.owner.clone()));
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
                        problem = Some(format!("tool '{name}' is already registered by '{owner}'"));
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
                staged_tools.push(ToolDef {
                    owner: plugin_name.to_string(),
                    ..def
                });
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
            css,
            hooks,
            panel_types,
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
            css,
            hooks,
            panel_types,
            services,
            tools,
            readers,
            effects,
        } = staged;

        let mut registered: Vec<String> = Vec::with_capacity(tools.len());
        for def in tools {
            let name = def.name.clone();
            match self.tools.register(def, ConflictPolicy::Override) {
                Ok(()) => registered.push(name),
                Err(error) => {
                    eprintln!("[plugin] '{plugin}' could not register tool '{name}': {error:#}");
                }
            }
        }

        let mut incoming: Vec<((String, RealmId), ServiceEntry)> =
            Vec::with_capacity(services.len());
        for (key, value) in services {
            let serial = self.next_serial();
            incoming.push((
                key,
                ServiceEntry {
                    value,
                    fiber,
                    owner: plugin.clone(),
                    serial,
                },
            ));
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
            target.css = css;
            target.hooks = hooks;
            target.panel_types = panel_types;
            target.effects = effects;
            target.epoch = epoch;
            target.state = FiberState::Active;
            target.error = None;
        }
        self.version += 1;
    }
}

/// Run one disposer, isolating a panic so the rest still run.
pub(crate) fn run_disposer(disposer: Disposer) {
    if std::panic::catch_unwind(std::panic::AssertUnwindSafe(disposer)).is_err() {
        eprintln!("[plugin] a disposer panicked; the remaining ones still run");
    }
}

/// Undo the effects of a contribution set that will not be installed.
///
/// Dropping a disposer does not undo its effect, so a rejected `apply` has to be
/// unwound explicitly — otherwise a plugin that was refused after starting
/// something would leak exactly what this module promises to reverse.
pub(crate) fn unwind_effects(effects: Vec<Disposer>) {
    for disposer in effects.into_iter().rev() {
        run_disposer(disposer);
    }
}

/// The message from a panic payload, for error reporting.
pub(crate) fn panic_text(payload: Box<dyn Any + Send>) -> String {
    if let Some(text) = payload.downcast_ref::<&str>() {
        (*text).to_string()
    } else if let Some(text) = payload.downcast_ref::<String>() {
        text.clone()
    } else {
        "a panic with a non-string payload".to_string()
    }
}

/// [`crate::agent::PluginHooks`] implemented over a live kernel.
pub(crate) struct KernelHooks {
    kernel: std::sync::Weak<std::sync::Mutex<Kernel>>,
}

impl KernelHooks {
    /// Run `f` against the kernel if it is still alive.
    fn with<R>(&self, f: impl FnOnce(&Kernel) -> R) -> Option<R> {
        let strong = self.kernel.upgrade()?;
        let guard = strong.lock().ok()?;
        Some(f(&guard))
    }

    fn turn_context(&self, input: &str, model: &str) -> TurnContext {
        TurnContext {
            session: String::new(),
            input: input.to_string(),
            model: model.to_string(),
            skills: Vec::new(),
        }
    }
}

impl crate::agent::PluginHooks for KernelHooks {
    fn prompt_additions(&self, input: &str, model: &str) -> Vec<String> {
        let ctx = self.turn_context(input, model);
        self.with(|kernel| kernel.prompt_additions(&ctx))
            .unwrap_or_default()
    }

    fn filter_tools(&self, input: &str, model: &str, names: Vec<String>) -> Option<Vec<String>> {
        let ctx = self.turn_context(input, model);
        let mut keep: Option<Vec<String>> = None;
        self.with(|kernel| {
            let mut current = names;
            for plugin in kernel.active_plugins() {
                if let Some(next) = plugin.filter_tools(&ctx, current.clone()) {
                    current = next;
                }
            }
            keep = Some(current);
        })?;
        keep
    }

    fn refuse_call(&self, input: &str, model: &str, name: &str, arguments: &str) -> Option<String> {
        let ctx = self.turn_context(input, model);
        let call = ToolCallView {
            name: name.to_string(),
            arguments: arguments.to_string(),
        };
        self.with(|kernel| kernel.refuse_tool_call(&ctx, &call))
            .flatten()
    }

    fn note_call(&self, input: &str, model: &str, name: &str, ok: bool) {
        let ctx = self.turn_context(input, model);
        let call = ToolCallView {
            name: name.to_string(),
            arguments: String::new(),
        };
        self.with(|kernel| kernel.note_tool_call(&ctx, &call, ok));
    }

    fn intercept(
        &self,
        input: &str,
        model: &str,
        seam: crate::plugin::Seam,
    ) -> crate::plugin::Seam {
        let ctx = self.turn_context(input, model);
        self.with(|kernel| {
            let mut current = seam;
            for plugin in kernel.active_plugins() {
                current = plugin.intercept(&ctx, current);
            }
            current
        })
        .unwrap_or_else(|| crate::plugin::Seam::text("reply", ""))
    }
}
