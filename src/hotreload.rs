//! Hot reload: one entry point, per-turn snapshots, scoped consent.
//!
//! Three rules shape this module:
//!
//! 1. **One entry point.** Every change goes through [`Runtime::apply`], which
//!    validates, applies, and rolls back on failure. There is no second path
//!    that could leave the runtime half-changed.
//! 2. **Per-turn snapshots.** The loop reads one [`Snapshot`] at the start of a
//!    turn, so a change landing mid-turn cannot alter a request already in
//!    flight. Turns see changes; a turn never sees half of one.
//! 3. **Capability is never gated on cost.** A cache policy decides *how* a
//!    change is written into the prompt, never *whether* it may happen. Consent
//!    gates scope, not capability.
//!
//! Consent is decided once, where a change takes effect, and for every scope:
//! a `Deny` is the user saying "not this", and a session-scoped change the user
//! denied is still one they denied. What keeps a session's own change from
//! needing approval is the *default* rule — nobody decided otherwise — not a
//! separate path that skips the check.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::plugin::Kernel;
use crate::tools::ToolRegistry;

/// Where a model request goes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelRoute {
    /// Provider label, for diagnostics.
    pub provider: String,
    /// API base URL including the version segment.
    pub base_url: String,
    /// Bearer token.
    #[serde(default)]
    pub api_key: String,
    /// Model id.
    pub model: String,
    /// Sampling temperature.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Output ceiling.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Reasoning effort for this route.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    /// How the answer is delivered: `stream` or `buffered`.
    ///
    /// Carried on the route like `extra_body` and `network`, because a subagent
    /// that streams while its session buffers is a difference the user never
    /// chose — and on a gateway whose stream never completes, it is a
    /// subagent that cannot answer at all.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub delivery: Option<String>,
    /// Whether a stalled stream may switch this route to buffered delivery.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub degrade_on_stall: Option<bool>,
    /// Proxy for requests on this route; empty means direct.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proxy: Option<String>,
    /// Extra request fields merged into the body.
    ///
    /// Carried on the route rather than read from the settings at call time so
    /// that every client built on this route — the session's and a subagent's —
    /// sends the same body.
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub extra_body: serde_json::Map<String, serde_json::Value>,
    /// How requests on this route are sent: timeouts, pooling, retries.
    ///
    /// The same reason as `extra_body`: connection behaviour that differs
    /// between a session and its subagent is a difference the user never chose.
    #[serde(default)]
    pub network: crate::network::NetworkSettings,
}

impl Default for ModelRoute {
    fn default() -> Self {
        Self {
            provider: "openai".into(),
            base_url: "https://api.openai.com/v1".into(),
            api_key: String::new(),
            model: "gpt-4o-mini".into(),
            temperature: None,
            max_tokens: None,
            reasoning_effort: None,
            delivery: None,
            degrade_on_stall: None,
            proxy: None,
            extra_body: serde_json::Map::new(),
            network: crate::network::NetworkSettings::default(),
        }
    }
}

impl ModelRoute {
    /// Whether two routes differ in a way that changes the model endpoint.
    pub fn same_endpoint(&self, other: &Self) -> bool {
        self.base_url == other.base_url && self.model == other.model
    }

    /// The route these settings describe.
    ///
    /// One place where settings become a route, so a host cannot update one
    /// field and forget another — which is how a session and its subagent come
    /// to disagree about the proxy.
    pub fn from_settings(settings: &crate::settings::Settings) -> Self {
        let effort = settings.reasoning_effort.trim();
        Self {
            provider: "openai".into(),
            base_url: settings.base_url.clone(),
            api_key: settings.api_key.clone(),
            model: settings.model_or_default(),
            temperature: None,
            max_tokens: settings.max_output_tokens.map(|tokens| tokens as u32),
            reasoning_effort: if effort.is_empty() || effort == "default" {
                None
            } else {
                Some(effort.to_string())
            },
            delivery: if settings.delivery.trim().is_empty() {
                None
            } else {
                Some(settings.delivery.trim().to_string())
            },
            degrade_on_stall: settings.degrade_on_stall,
            proxy: Some(settings.proxy.clone()),
            extra_body: settings.extra_body.clone(),
            network: settings.network.clone(),
        }
    }

    /// The client this route describes.
    pub fn client(&self) -> Result<crate::llm::LlmClient> {
        let model = self.model.clone();
        self.client_for(&model)
    }

    /// The client this route describes, on another model of the same route.
    ///
    /// The only builder, so a subagent inherits the endpoint, key, proxy,
    /// effort, output ceiling, extra fields, and connection behaviour its
    /// session runs under rather than a subset of them.
    pub fn client_for(&self, model: &str) -> Result<crate::llm::LlmClient> {
        let mut config =
            crate::llm::LlmConfig::new(self.base_url.clone(), self.api_key.clone(), model);
        config.proxy = self.proxy.clone().unwrap_or_default();
        // `default` asks the provider for its own behaviour, so it is sent as
        // no effort at all rather than as a value it would have to parse.
        let effort = self.reasoning_effort.as_deref().unwrap_or_default().trim();
        config.reasoning_effort = if effort.is_empty() || effort == "default" {
            None
        } else {
            Some(effort.to_string())
        };
        config.max_tokens = self.max_tokens;
        // An unknown mode is refused rather than defaulted: silently streaming
        // when the user wrote a word meaning the opposite is how a route that
        // cannot stream keeps being asked to.
        let word = self.delivery.as_deref().unwrap_or_default().trim();
        config.delivery = if word.is_empty() {
            crate::llm::Delivery::default()
        } else {
            crate::llm::Delivery::parse(word).ok_or_else(|| {
                anyhow::anyhow!(
                    "unknown delivery mode {word:?}; expected one of {}",
                    crate::llm::Delivery::names().join(", ")
                )
            })?
        };
        config.degrade_on_stall = self.degrade_on_stall.unwrap_or(true);
        config.network = self.network.clone();
        Ok(crate::llm::LlmClient::new(config)?
            .with_shaper(crate::request::from_extra_fields(self.extra_body.clone())))
    }
}

/// How much prefix stability to trade for freshness.
///
/// This governs *presentation* of a change, never its permissibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum CachePolicy {
    /// Rewrite the prompt immediately; accept losing the cached prefix.
    Freshness,
    /// Append where possible; rewrite only when structure demands it.
    #[default]
    Balanced,
    /// Append everything; defer anything that would rewrite the prefix.
    CacheFirst,
}

/// What a change alters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ChangeKind {
    /// The model route (provider, base URL, model).
    ModelRoute,
    /// The persona text.
    Persona,
    /// The skill search roots.
    SkillRoots,
    /// An MCP server.
    McpServer,
    /// A tool registration.
    Tool,
    /// A plugin load or unload.
    Plugin,
    /// The default cache policy itself.
    CachePolicy,
}

impl ChangeKind {
    /// Stable name for diagnostics and policy keys.
    pub fn as_str(self) -> &'static str {
        match self {
            ChangeKind::ModelRoute => "model-route",
            ChangeKind::Persona => "persona",
            ChangeKind::SkillRoots => "skill-roots",
            ChangeKind::McpServer => "mcp-server",
            ChangeKind::Tool => "tool",
            ChangeKind::Plugin => "plugin",
            ChangeKind::CachePolicy => "cache-policy",
        }
    }

    /// Every kind, for policy reporting.
    pub fn all() -> [ChangeKind; 7] {
        [
            ChangeKind::ModelRoute,
            ChangeKind::Persona,
            ChangeKind::SkillRoots,
            ChangeKind::McpServer,
            ChangeKind::Tool,
            ChangeKind::Plugin,
            ChangeKind::CachePolicy,
        ]
    }
}

/// Which sessions a change reaches.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChangeScope {
    /// Only the session that asked.
    Session(String),
    /// Every session, including later ones.
    Global,
}

impl ChangeScope {
    /// Whether this scope needs consent.
    pub fn is_global(&self) -> bool {
        matches!(self, ChangeScope::Global)
    }
}

/// What to do when a change arrives.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Consent {
    /// Apply without asking.
    AutoAllow,
    /// Ask the user.
    Ask,
    /// Refuse. Never offered as a "remember this" option.
    Deny,
}

/// The payload of a change.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum ChangePayload {
    /// Replace the model route.
    ModelRoute(ModelRoute),
    /// Replace the persona text.
    Persona(String),
    /// Add a skill search root.
    AddSkillRoot(String),
    /// Remove a skill search root.
    RemoveSkillRoot(String),
    /// Set the default cache policy.
    CachePolicy(CachePolicy),
    /// The tool table was swapped by the kernel (a plugin load or unload).
    ///
    /// Carries a count rather than the table: a change is serialized and
    /// persisted, and a live registry is not data. The swap itself happens in
    /// [`Runtime::set_tools`]; this payload exists so that swap is a
    /// first-class, authorized and audited change rather than a second door
    /// into the same state.
    Tools {
        /// How many tools the new table holds.
        count: usize,
    },
}

impl ChangePayload {
    /// The kind this payload carries.
    pub fn kind(&self) -> ChangeKind {
        match self {
            ChangePayload::ModelRoute(_) => ChangeKind::ModelRoute,
            ChangePayload::Persona(_) => ChangeKind::Persona,
            ChangePayload::AddSkillRoot(_) | ChangePayload::RemoveSkillRoot(_) => {
                ChangeKind::SkillRoots
            }
            ChangePayload::CachePolicy(_) => ChangeKind::CachePolicy,
            ChangePayload::Tools { .. } => ChangeKind::Tool,
        }
    }

    /// A one-line description for the consent prompt.
    pub fn describe(&self) -> String {
        match self {
            ChangePayload::ModelRoute(route) => {
                format!("model route → {} ({})", route.model, route.base_url)
            }
            ChangePayload::Persona(text) => {
                let preview: String = text.chars().take(60).collect();
                format!("persona → {preview:?}")
            }
            ChangePayload::AddSkillRoot(path) => format!("add skill root {path}"),
            ChangePayload::RemoveSkillRoot(path) => format!("remove skill root {path}"),
            ChangePayload::CachePolicy(policy) => format!("cache policy → {policy:?}"),
            ChangePayload::Tools { count } => format!("tool table → {count} tools"),
        }
    }
}

/// One requested change.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Change {
    /// What to change.
    pub payload: ChangePayload,
    /// Which sessions it reaches.
    pub scope: ChangeScope,
    /// Whether to write it to durable settings.
    #[serde(default)]
    pub persist: bool,
    /// Presentation policy for this change specifically.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_policy: Option<CachePolicy>,
}

impl Change {
    /// A session-scoped change.
    pub fn session(session: impl Into<String>, payload: ChangePayload) -> Self {
        Self {
            payload,
            scope: ChangeScope::Session(session.into()),
            persist: false,
            cache_policy: None,
        }
    }

    /// A global change.
    pub fn global(payload: ChangePayload) -> Self {
        Self {
            payload,
            scope: ChangeScope::Global,
            persist: true,
            cache_policy: None,
        }
    }

    /// Mark the change durable.
    pub fn persisting(mut self) -> Self {
        self.persist = true;
        self
    }

    /// Force a presentation policy for this change.
    pub fn with_cache_policy(mut self, policy: CachePolicy) -> Self {
        self.cache_policy = Some(policy);
        self
    }
}

/// The result of asking the runtime to apply something.
#[derive(Debug, Clone, PartialEq)]
pub enum ApplyOutcome {
    /// Applied immediately.
    Applied(AppliedChange),
    /// Waiting for consent; nothing changed yet.
    NeedsConsent(PendingChange),
    /// Refused by policy.
    Denied(String),
}

/// A change waiting for consent.
#[derive(Debug, Clone, PartialEq)]
pub struct PendingChange {
    /// Token used to approve or reject it.
    pub id: String,
    /// The change itself.
    pub change: Change,
    /// Why consent is required.
    pub reason: String,
}

/// A change that was applied.
#[derive(Debug, Clone, PartialEq)]
pub struct AppliedChange {
    /// Sequence number within this runtime.
    pub seq: u64,
    /// The change.
    pub change: Change,
    /// Runtime version after applying.
    pub version: u64,
    /// How it was decided.
    pub decided_by: &'static str,
}

/// The mutable runtime state.
#[derive(Debug, Clone)]
pub struct RuntimeState {
    /// Where model requests go.
    pub model_route: ModelRoute,
    /// Persona text.
    pub persona: String,
    /// Skill search roots, in scan order.
    pub skill_roots: Vec<String>,
    /// Default presentation policy.
    pub cache_policy: CachePolicy,
}

impl Default for RuntimeState {
    fn default() -> Self {
        Self {
            model_route: ModelRoute::default(),
            persona: String::new(),
            skill_roots: Vec::new(),
            cache_policy: CachePolicy::default(),
        }
    }
}

/// An immutable view of the runtime for one turn.
///
/// The loop holds this for the whole turn, so a change that lands while a
/// request is in flight is visible to the next turn, never to this one.
#[derive(Clone)]
pub struct Snapshot {
    /// Runtime version this snapshot was taken at.
    pub version: u64,
    /// Model route to use.
    pub model_route: ModelRoute,
    /// Persona text.
    pub persona: String,
    /// Skill roots in force.
    pub skill_roots: Vec<String>,
    /// Presentation policy in force.
    pub cache_policy: CachePolicy,
    /// Tool table at snapshot time.
    pub tools: Arc<ToolRegistry>,
}

impl Snapshot {
    /// The system prompt this snapshot implies, given a base prompt.
    ///
    /// Order is a cache decision, not a stylistic one. A prompt cache matches on
    /// a **prefix**, so the bytes most likely to change must come last: the base
    /// prompt is compiled into the binary and identical across every turn and
    /// every session, while the persona is a join over the packs that happen to
    /// be loaded and changes whenever one is loaded, unloaded, or edited.
    ///
    /// Written persona-first, every pack toggle invalidated the cache for the
    /// entire prompt — base included — on the very first token.
    pub fn system_prompt(&self, base: &str) -> String {
        if self.persona.trim().is_empty() {
            base.to_string()
        } else {
            format!("{}\n\n{}", base, self.persona.trim())
        }
    }
}

/// The runtime: state, policy, and the audit trail of what changed.
pub struct Runtime {
    state: RuntimeState,
    /// Decisions that cover a whole kind, in every scope.
    ///
    /// Empty means nothing has been decided, which is not the same as "ask
    /// about everything": [`Runtime::rule_for`] fills in the default for a
    /// scope where no decision has been made.
    policy: HashMap<ChangeKind, Consent>,
    /// Decisions the user asked to be remembered, keyed by the *value* they
    /// were made about.
    ///
    /// Keyed by kind alone, one approval of a local endpoint allowed every
    /// later route — including one pointing somewhere else — in every session.
    /// A remembered decision now covers exactly what was approved.
    remembered: BTreeSet<(ChangeKind, String)>,
    history: Vec<AppliedChange>,
    /// Waiting changes, in the order they arrived.
    ///
    /// Keyed by a counter of its own: the id used to be `seq + 1`, and `seq`
    /// only advances on commit, so two queued changes shared an id and the
    /// second silently replaced the first — which then could be neither
    /// approved nor audited.
    pending: BTreeMap<u64, PendingChange>,
    next_pending: u64,
    tools: Arc<ToolRegistry>,
    version: u64,
    seq: u64,
}

impl Runtime {
    /// Build a runtime over a tool table, with no decisions taken yet.
    ///
    /// Nothing decided means a session-scoped change is applied, because it
    /// cannot reach anyone else, and a global one asks, because it reaches
    /// every session. Nothing is denied.
    pub fn new(tools: Arc<ToolRegistry>) -> Self {
        Self {
            state: RuntimeState::default(),
            policy: HashMap::new(),
            remembered: BTreeSet::new(),
            history: Vec::new(),
            pending: BTreeMap::new(),
            next_pending: 0,
            tools,
            version: 1,
            seq: 0,
        }
    }

    /// Seed the initial route and persona.
    pub fn with_route(mut self, route: ModelRoute) -> Self {
        self.state.model_route = route;
        self
    }

    /// Seed the initial persona.
    pub fn with_persona(mut self, persona: impl Into<String>) -> Self {
        self.state.persona = persona.into();
        self
    }

    /// Seed the initial skill roots.
    pub fn with_skill_roots(mut self, roots: Vec<String>) -> Self {
        self.state.skill_roots = roots;
        self
    }

    /// The current version.
    pub fn version(&self) -> u64 {
        self.version
    }

    /// The current state.
    pub fn state(&self) -> &RuntimeState {
        &self.state
    }

    /// Take a snapshot for one turn.
    pub fn snapshot(&self) -> Snapshot {
        Snapshot {
            version: self.version,
            model_route: self.state.model_route.clone(),
            persona: self.state.persona.clone(),
            skill_roots: self.state.skill_roots.clone(),
            cache_policy: self.state.cache_policy,
            tools: Arc::clone(&self.tools),
        }
    }

    /// Replace the tool table (a plugin load or unload changed it).
    ///
    /// A tool table is capability, so it takes the same door as every other
    /// change: the rule for [`ChangeKind::Tool`] can refuse it, and the swap
    /// leaves an entry in [`Runtime::history`] naming the kernel as the decider.
    /// It used to be a second door that skipped both, which meant a turn could
    /// be handed a table no rule had ever seen — and nothing recorded it.
    ///
    /// The outcome is returned rather than swallowed so a caller can report a
    /// refusal; ignoring it still applies the swap whenever the policy allows.
    pub fn set_tools(&mut self, tools: Arc<ToolRegistry>) -> ApplyOutcome {
        if self.rule_for(ChangeKind::Tool) == Consent::Deny {
            return ApplyOutcome::Denied("policy denies tool-table changes".into());
        }
        let count = tools.len();
        self.tools = tools;
        self.version += 1;
        self.seq += 1;
        let applied = AppliedChange {
            seq: self.seq,
            change: Change {
                payload: ChangePayload::Tools { count },
                scope: ChangeScope::Global,
                persist: false,
                cache_policy: None,
            },
            version: self.version,
            decided_by: "kernel",
        };
        self.history.push(applied.clone());
        ApplyOutcome::Applied(applied)
    }

    /// Apply a tool-table change coming from the kernel.
    pub fn adopt_kernel_tools(&mut self, kernel: &Kernel) -> ApplyOutcome {
        self.set_tools(Arc::new(kernel.tools().clone()))
    }

    /// Set the consent rule for one kind of change, in every scope.
    pub fn set_consent(&mut self, kind: ChangeKind, consent: Consent) {
        self.policy.insert(kind, consent);
    }

    /// The rule recorded for one kind of change.
    ///
    /// `Ask` when nothing has been decided. This is the table as the user wrote
    /// it; [`Runtime::rule_for`] is what applies a rule to an actual change.
    pub fn consent(&self, kind: ChangeKind) -> Consent {
        self.policy.get(&kind).cloned().unwrap_or(Consent::Ask)
    }

    /// Every rule, for display.
    pub fn policy_table(&self) -> Vec<(ChangeKind, Consent)> {
        ChangeKind::all()
            .into_iter()
            .map(|kind| (kind, self.consent(kind)))
            .collect()
    }

    /// The audit trail.
    pub fn history(&self) -> &[AppliedChange] {
        &self.history
    }

    /// Changes awaiting consent, in the order they arrived.
    pub fn pending(&self) -> Vec<&PendingChange> {
        self.pending.values().collect()
    }

    /// The rule that applies to one change, in its own scope.
    ///
    /// An explicit decision covers every scope — a `Deny` is the user saying
    /// "not this", and a session-scoped change they denied is still one they
    /// denied. With nothing decided, a global change asks, because it reaches
    /// every session, and a session change does not, because it cannot reach
    /// anyone else. A remembered approval covers exactly the value it was made
    /// about.
    fn rule_for(&self, kind: ChangeKind) -> Consent {
        self.policy.get(&kind).cloned().unwrap_or(Consent::Ask)
    }

    fn rule_for_change(&self, change: &Change) -> Consent {
        if let Some(rule) = self.policy.get(&change.payload.kind()) {
            return rule.clone();
        }
        if self.remembered.contains(&authorization(&change.payload)) {
            return Consent::AutoAllow;
        }
        if change.scope.is_global() {
            Consent::Ask
        } else {
            Consent::AutoAllow
        }
    }

    /// Request a change.
    ///
    /// This is the only path that mutates runtime state.
    pub fn apply(&mut self, change: Change) -> Result<ApplyOutcome> {
        let kind = change.payload.kind();

        // Validate before anything is touched, so a rejected change cannot
        // leave a trace.
        validate(&change)?;

        // The tool table is not part of this state: the kernel swaps it through
        // `set_tools`, which consults the same rule and writes the same history.
        if matches!(&change.payload, ChangePayload::Tools { .. }) {
            return Err(anyhow!(
                "the tool table is swapped by the kernel, not by a change"
            ));
        }

        // One decision point, every scope. A `Deny` stops here whatever the
        // scope, and an `Ask` queues here whatever the scope; nothing reaches
        // `commit` without passing this.
        match self.rule_for_change(&change) {
            Consent::Deny => {
                return Ok(ApplyOutcome::Denied(format!(
                    "policy denies {} changes{}",
                    kind.as_str(),
                    if change.scope.is_global() {
                        " in every session"
                    } else {
                        " in this session"
                    }
                )));
            }
            Consent::Ask => {
                self.next_pending += 1;
                let token = self.next_pending;
                let pending = PendingChange {
                    id: format!("chg-{token}"),
                    change: change.clone(),
                    reason: if change.scope.is_global() {
                        format!("global {} change affects every session", kind.as_str())
                    } else {
                        format!("a session-scoped {} change", kind.as_str())
                    },
                };
                // Never overwrite: two waiting changes are two decisions.
                debug_assert!(!self.pending.contains_key(&token));
                self.pending.insert(token, pending.clone());
                return Ok(ApplyOutcome::NeedsConsent(pending));
            }
            Consent::AutoAllow => {}
        }

        let decided_by = if change.scope.is_global() { "auto-allowed" } else { "session-scope" };
        self.commit(change, decided_by)
    }

    /// The pending entry with this id.
    fn take_pending(&mut self, id: &str) -> Result<PendingChange> {
        let key = id
            .strip_prefix("chg-")
            .unwrap_or(id)
            .parse::<u64>()
            .ok()
            .filter(|key| self.pending.contains_key(key))
            .ok_or_else(|| anyhow!("no pending change '{id}'"))?;
        Ok(self
            .pending
            .remove(&key)
            .expect("filtered on contains_key"))
    }

    /// Approve a pending change.
    pub fn approve(&mut self, id: &str) -> Result<ApplyOutcome> {
        let pending = self.take_pending(id)?;
        self.commit(pending.change, "approved")
    }

    /// Approve a pending change and let *this value* through automatically from
    /// now on.
    ///
    /// The value, not the kind: approving a route to a local endpoint says
    /// nothing about an endpoint on the internet, and approving one skill root
    /// says nothing about another directory. A remembered decision that covered
    /// the whole kind turned a single answer into a standing permission.
    pub fn approve_and_remember(&mut self, id: &str) -> Result<ApplyOutcome> {
        let pending = self.take_pending(id)?;
        let key = authorization(&pending.change.payload);
        self.remembered.insert(key);
        self.commit(pending.change, "approved-and-remembered")
    }

    /// Reject a pending change. Nothing was applied.
    pub fn reject(&mut self, id: &str) -> Result<()> {
        self.take_pending(id)?;
        Ok(())
    }

    /// Apply a change and record it.
    fn commit(&mut self, change: Change, decided_by: &'static str) -> Result<ApplyOutcome> {
        let backup = self.state.clone();

        if let Err(error) = self.mutate(&change) {
            // Restore exactly what was there, so a failed change is invisible.
            self.state = backup;
            return Err(error);
        }

        self.version += 1;
        self.seq += 1;
        let applied = AppliedChange {
            seq: self.seq,
            change,
            version: self.version,
            decided_by,
        };
        self.history.push(applied.clone());
        Ok(ApplyOutcome::Applied(applied))
    }

    fn mutate(&mut self, change: &Change) -> Result<()> {
        match &change.payload {
            ChangePayload::ModelRoute(route) => {
                if route.base_url.trim().is_empty() {
                    return Err(anyhow!("model route needs a base URL"));
                }
                if route.model.trim().is_empty() {
                    return Err(anyhow!("model route needs a model id"));
                }
                self.state.model_route = route.clone();
            }
            ChangePayload::Persona(text) => {
                self.state.persona = text.clone();
            }
            ChangePayload::AddSkillRoot(path) => {
                if !self.state.skill_roots.iter().any(|r| r == path) {
                    self.state.skill_roots.push(path.clone());
                }
            }
            ChangePayload::RemoveSkillRoot(path) => {
                let before = self.state.skill_roots.len();
                self.state.skill_roots.retain(|r| r != path);
                if self.state.skill_roots.len() == before {
                    return Err(anyhow!("skill root '{path}' is not registered"));
                }
            }
            ChangePayload::CachePolicy(policy) => {
                self.state.cache_policy = *policy;
            }
            // The table lives on the runtime, not in this state, and a live
            // registry cannot be reconstructed from a payload: the kernel swaps
            // it in `set_tools`.
            ChangePayload::Tools { .. } => {
                return Err(anyhow!(
                    "the tool table is swapped by the kernel, not by a change"
                ));
            }
        }
        Ok(())
    }
}

/// What a remembered decision was about: the kind *and* the value.
///
/// Two approvals of different values are two decisions. Normalisation is
/// deliberately conservative — surrounding whitespace and a trailing slash —
/// because anything more would let a different endpoint pass as the same one.
fn authorization(payload: &ChangePayload) -> (ChangeKind, String) {
    let kind = payload.kind();
    let value = match payload {
        ChangePayload::ModelRoute(route) => format!(
            "{}|{}",
            route.base_url.trim().trim_end_matches('/'),
            route.model.trim()
        ),
        // The persona is remembered by digest: it can be sixty kilobytes, and
        // what matters is that the next one is the same one.
        ChangePayload::Persona(text) => format!("#{}", hex_digest(text.trim())),
        ChangePayload::AddSkillRoot(path) => format!("+{}", path.trim()),
        ChangePayload::RemoveSkillRoot(path) => format!("-{}", path.trim()),
        ChangePayload::CachePolicy(policy) => format!("{policy:?}"),
        ChangePayload::Tools { count } => format!("{count}"),
    };
    (kind, value)
}

fn hex_digest(text: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// Reject a change that could never be valid, before any state moves.
fn validate(change: &Change) -> Result<()> {
    match &change.payload {
        ChangePayload::ModelRoute(route) => {
            if route.model.trim().is_empty() {
                return Err(anyhow!("model route requires a model id"));
            }
        }
        ChangePayload::Persona(text) => {
            if text.len() > 64 * 1024 {
                return Err(anyhow!("persona is too large ({} bytes)", text.len()));
            }
        }
        ChangePayload::AddSkillRoot(path) | ChangePayload::RemoveSkillRoot(path) => {
            if path.trim().is_empty() {
                return Err(anyhow!("skill root path is empty"));
            }
        }
        ChangePayload::CachePolicy(_) => {}
        // Nothing to check: the table is swapped by the kernel, never applied
        // as a change (see `Runtime::apply`).
        ChangePayload::Tools { .. } => {}
    }
    Ok(())
}

