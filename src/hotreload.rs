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
    pub fn system_prompt(&self, base: &str) -> String {
        if self.persona.trim().is_empty() {
            base.to_string()
        } else {
            format!("{}\n\n{}", self.persona.trim(), base)
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

#[cfg(test)]
mod tests {
    use super::*;

    fn fresh_runtime() -> Runtime {
        Runtime::new(Arc::new(ToolRegistry::with_base_tools().unwrap()))
    }

    #[test]
    fn a_session_change_applies_without_consent() {
        let mut runtime = fresh_runtime();
        let change = Change::session("s1", ChangePayload::Persona("be terse".into()));

        let outcome = runtime.apply(change).unwrap();
        match outcome {
            ApplyOutcome::Applied(applied) => {
                assert_eq!(applied.decided_by, "session-scope");
                assert_eq!(applied.version, 2);
            }
            other => panic!("expected an immediate apply, got {other:?}"),
        }
        assert_eq!(runtime.state().persona, "be terse");
        assert_eq!(runtime.history().len(), 1);
    }

    #[test]
    fn a_global_change_waits_for_consent() {
        let mut runtime = fresh_runtime();
        let change = Change::global(ChangePayload::Persona("global voice".into()));

        let outcome = runtime.apply(change).unwrap();
        let pending = match outcome {
            ApplyOutcome::NeedsConsent(pending) => pending,
            other => panic!("expected consent to be required, got {other:?}"),
        };

        // Nothing changed while it waits.
        assert!(runtime.state().persona.is_empty());
        assert_eq!(runtime.pending().len(), 1);
        assert_eq!(runtime.history().len(), 0);

        // Approving applies it.
        let applied = runtime.approve(&pending.id).unwrap();
        assert!(matches!(applied, ApplyOutcome::Applied(_)));
        assert_eq!(runtime.state().persona, "global voice");
        assert!(runtime.pending().is_empty());
    }

    #[test]
    fn rejecting_leaves_the_runtime_untouched() {
        let mut runtime = fresh_runtime();
        let before = runtime.state().clone();

        let pending = match runtime
            .apply(Change::global(ChangePayload::Persona("no".into())))
            .unwrap()
        {
            ApplyOutcome::NeedsConsent(p) => p,
            other => panic!("expected consent, got {other:?}"),
        };

        runtime.reject(&pending.id).unwrap();
        assert_eq!(runtime.state().persona, before.persona);
        assert_eq!(runtime.version(), 1);
        assert!(runtime.history().is_empty());
        assert!(runtime.pending().is_empty());
    }

    #[test]
    fn denying_a_kind_refuses_it_outright() {
        let mut runtime = fresh_runtime();
        runtime.set_consent(ChangeKind::ModelRoute, Consent::Deny);

        let outcome = runtime
            .apply(Change::global(ChangePayload::ModelRoute(ModelRoute {
                model: "other".into(),
                ..Default::default()
            })))
            .unwrap();

        match outcome {
            ApplyOutcome::Denied(reason) => assert!(reason.contains("denies"), "{reason}"),
            other => panic!("expected a denial, got {other:?}"),
        }
        // A denial is not a pending request.
        assert!(runtime.pending().is_empty());
        assert_eq!(runtime.state().model_route.model, ModelRoute::default().model);
    }

    #[test]
    fn auto_allow_skips_the_prompt_for_that_kind_only() {
        let mut runtime = fresh_runtime();
        runtime.set_consent(ChangeKind::ModelRoute, Consent::AutoAllow);

        // The auto-allowed kind applies at once...
        let applied = runtime
            .apply(Change::global(ChangePayload::ModelRoute(ModelRoute {
                model: "fast-model".into(),
                ..Default::default()
            })))
            .unwrap();
        assert!(matches!(applied, ApplyOutcome::Applied(_)));
        assert_eq!(runtime.state().model_route.model, "fast-model");

        // ...while every other kind still asks.
        let asked = runtime
            .apply(Change::global(ChangePayload::Persona("x".into())))
            .unwrap();
        assert!(matches!(asked, ApplyOutcome::NeedsConsent(_)));
    }

    #[test]
    fn remembering_a_decision_covers_the_value_that_was_approved() {
        let mut runtime = fresh_runtime();
        let pending = match runtime
            .apply(Change::global(ChangePayload::AddSkillRoot("/skills".into())))
            .unwrap()
        {
            ApplyOutcome::NeedsConsent(p) => p,
            other => panic!("expected consent, got {other:?}"),
        };

        runtime.approve_and_remember(&pending.id).unwrap();
        assert_eq!(runtime.state().skill_roots, vec!["/skills".to_string()]);

        // The same value does not ask again...
        let again = runtime
            .apply(Change::global(ChangePayload::AddSkillRoot("/skills".into())))
            .unwrap();
        assert!(matches!(again, ApplyOutcome::Applied(_)), "{again:?}");

        // ...and a different value is a different decision. Remembering the
        // *kind* would have turned one answer into a standing permission.
        let other = runtime
            .apply(Change::global(ChangePayload::AddSkillRoot(
                "/somewhere-else".into(),
            )))
            .unwrap();
        assert!(
            matches!(other, ApplyOutcome::NeedsConsent(_)),
            "one approval is not a standing permission for the kind: {other:?}"
        );
        assert_eq!(runtime.state().skill_roots, vec!["/skills".to_string()]);
        assert_eq!(
            runtime.consent(ChangeKind::SkillRoots),
            Consent::Ask,
            "nothing was decided about the kind as a whole"
        );
    }

    #[test]
    fn remembering_one_route_does_not_allow_another_endpoint() {
        let mut runtime = fresh_runtime();
        let local = ModelRoute {
            provider: "ollama".into(),
            base_url: "http://127.0.0.1:11434/v1".into(),
            model: "local-model".into(),
            ..Default::default()
        };
        let pending = match runtime
            .apply(Change::global(ChangePayload::ModelRoute(local.clone())))
            .unwrap()
        {
            ApplyOutcome::NeedsConsent(p) => p,
            other => panic!("expected consent, got {other:?}"),
        };
        runtime.approve_and_remember(&pending.id).unwrap();

        // The same route, spelled with a trailing slash, is the same decision.
        let same = ModelRoute {
            base_url: "http://127.0.0.1:11434/v1/".into(),
            ..local.clone()
        };
        assert!(
            matches!(
                runtime
                    .apply(Change::global(ChangePayload::ModelRoute(same)))
                    .unwrap(),
                ApplyOutcome::Applied(_)
            ),
            "the approved route is approved again"
        );

        // Another endpoint is another decision — approving a local ollama must
        // not become permission to send traffic anywhere.
        let elsewhere = ModelRoute {
            base_url: "https://api.example.com/v1".into(),
            ..local.clone()
        };
        assert!(matches!(
            runtime
                .apply(Change::global(ChangePayload::ModelRoute(elsewhere)))
                .unwrap(),
            ApplyOutcome::NeedsConsent(_)
        ));

        // And so is another model on the endpoint that was approved.
        let remodelled = ModelRoute {
            model: "another-model".into(),
            ..local.clone()
        };
        assert!(matches!(
            runtime
                .apply(Change::global(ChangePayload::ModelRoute(remodelled)))
                .unwrap(),
            ApplyOutcome::NeedsConsent(_)
        ));
        assert_eq!(
            runtime.state().model_route.base_url,
            "http://127.0.0.1:11434/v1/"
        );
    }

    #[test]
    fn two_waiting_changes_keep_separate_ids_and_both_stay_answerable() {
        let mut runtime = fresh_runtime();
        let first = match runtime
            .apply(Change::global(ChangePayload::Persona("one".into())))
            .unwrap()
        {
            ApplyOutcome::NeedsConsent(p) => p,
            other => panic!("expected consent, got {other:?}"),
        };
        let second = match runtime
            .apply(Change::global(ChangePayload::AddSkillRoot("/two".into())))
            .unwrap()
        {
            ApplyOutcome::NeedsConsent(p) => p,
            other => panic!("expected consent, got {other:?}"),
        };

        assert_ne!(
            first.id, second.id,
            "two queued changes must not share an id: the second used to replace the first"
        );
        let ids: Vec<String> = runtime.pending().iter().map(|p| p.id.clone()).collect();
        assert_eq!(
            ids,
            vec![first.id.clone(), second.id.clone()],
            "both wait, in arrival order, and neither is lost"
        );

        // Denying one is denying that one.
        runtime.reject(&second.id).unwrap();
        let left: Vec<String> = runtime.pending().iter().map(|p| p.id.clone()).collect();
        assert_eq!(left, vec![first.id.clone()]);

        // The one that was not denied is still approvable by its own id.
        assert!(matches!(
            runtime.approve(&first.id).unwrap(),
            ApplyOutcome::Applied(_)
        ));
        assert!(runtime.pending().is_empty());
        assert_eq!(runtime.state().persona, "one");
        assert!(runtime.state().skill_roots.is_empty());
    }

    #[test]
    fn a_kernel_tool_swap_is_consented_and_audited() {
        let mut runtime = fresh_runtime();
        let before = runtime.snapshot().tools;
        let empty = Arc::new(ToolRegistry::new());

        match runtime.set_tools(Arc::clone(&empty)) {
            ApplyOutcome::Applied(applied) => {
                assert_eq!(applied.decided_by, "kernel");
                assert_eq!(applied.change.payload.kind(), ChangeKind::Tool);
            }
            other => panic!("a kernel swap applies, got {other:?}"),
        }
        assert_eq!(runtime.history().len(), 1, "the swap leaves a record");
        assert!(
            !Arc::ptr_eq(&runtime.snapshot().tools, &before),
            "the table really changed"
        );

        // A refusal leaves the table where it was, and is not an apply.
        runtime.set_consent(ChangeKind::Tool, Consent::Deny);
        let denied = runtime.set_tools(Arc::clone(&before));
        assert!(matches!(denied, ApplyOutcome::Denied(_)), "{denied:?}");
        assert!(
            Arc::ptr_eq(&runtime.snapshot().tools, &empty),
            "a refused swap changes nothing"
        );
        assert_eq!(runtime.history().len(), 1, "a refusal is not an apply");
    }

    #[test]
    fn a_snapshot_does_not_see_later_changes() {
        let mut runtime = fresh_runtime();
        runtime
            .apply(Change::session("s1", ChangePayload::Persona("first".into())))
            .unwrap();

        let snapshot = runtime.snapshot();
        assert_eq!(snapshot.persona, "first");
        let version = snapshot.version;

        // A change landing now must not rewrite a snapshot already taken.
        runtime
            .apply(Change::session("s1", ChangePayload::Persona("second".into())))
            .unwrap();

        assert_eq!(snapshot.persona, "first", "the snapshot is frozen");
        assert_eq!(snapshot.version, version);
        assert_eq!(runtime.snapshot().persona, "second", "a new turn sees it");
    }

    #[test]
    fn an_invalid_change_leaves_no_trace() {
        let mut runtime = fresh_runtime();
        let before = runtime.state().clone();

        let error = runtime
            .apply(Change::session(
                "s1",
                ChangePayload::ModelRoute(ModelRoute {
                    model: "   ".into(),
                    ..Default::default()
                }),
            ))
            .expect_err("empty model id must be refused");
        assert!(format!("{error:#}").contains("model id"));

        assert_eq!(runtime.state().model_route, before.model_route);
        assert_eq!(runtime.version(), 1);
        assert!(runtime.history().is_empty());
    }

    #[test]
    fn removing_an_unknown_skill_root_is_an_error() {
        let mut runtime = fresh_runtime();
        let error = runtime
            .apply(Change::session(
                "s1",
                ChangePayload::RemoveSkillRoot("/nope".into()),
            ))
            .expect_err("unknown root");
        assert!(format!("{error:#}").contains("not registered"));
        assert_eq!(runtime.version(), 1);
    }

    #[test]
    fn adding_the_same_skill_root_twice_is_idempotent() {
        let mut runtime = fresh_runtime();
        let change = || Change::session("s1", ChangePayload::AddSkillRoot("/skills".into()));
        runtime.apply(change()).unwrap();
        runtime.apply(change()).unwrap();
        assert_eq!(runtime.state().skill_roots.len(), 1);
    }

    #[test]
    fn cache_policy_is_presentation_not_permission() {
        let mut runtime = fresh_runtime();
        // CacheFirst is the most conservative presentation, and it still does
        // not stop a change from applying.
        runtime
            .apply(Change::session(
                "s1",
                ChangePayload::CachePolicy(CachePolicy::CacheFirst),
            ))
            .unwrap();
        assert_eq!(runtime.state().cache_policy, CachePolicy::CacheFirst);

        // A persona change under CacheFirst still lands immediately.
        let applied = runtime
            .apply(
                Change::session("s1", ChangePayload::Persona("still applies".into()))
                    .with_cache_policy(CachePolicy::CacheFirst),
            )
            .unwrap();
        assert!(matches!(applied, ApplyOutcome::Applied(_)));
        assert_eq!(runtime.state().persona, "still applies");
    }

    #[test]
    fn the_default_policy_asks_for_global_and_allows_session() {
        let runtime = fresh_runtime();
        for (kind, consent) in runtime.policy_table() {
            assert_eq!(consent, Consent::Ask, "{kind:?} records no decision");
        }

        // With nothing decided: a session's own change is applied, and a global
        // one asks. The difference is the default rule, not a skipped check.
        let mut runtime = fresh_runtime();
        let applied = runtime
            .apply(Change::session("s1", ChangePayload::Persona("ok".into())))
            .unwrap();
        assert!(matches!(applied, ApplyOutcome::Applied(_)));
        let asked = runtime
            .apply(Change::global(ChangePayload::Persona("everyone".into())))
            .unwrap();
        assert!(matches!(asked, ApplyOutcome::NeedsConsent(_)));
    }

    #[test]
    fn a_denial_reaches_every_scope_not_only_global() {
        // A `Deny` is the user saying "not this". A session-scoped persona
        // change is still a persona change, and used to slip past the denial
        // entirely because consent was only consulted for global scope.
        let mut runtime = fresh_runtime();
        runtime.set_consent(ChangeKind::Persona, Consent::Deny);

        let outcome = runtime
            .apply(Change::session("s1", ChangePayload::Persona("no".into())))
            .unwrap();

        match outcome {
            ApplyOutcome::Denied(reason) => assert!(reason.contains("denies"), "{reason}"),
            other => panic!("a denied kind must refuse a session change too, got {other:?}"),
        }
        assert!(runtime.state().persona.is_empty());
        assert!(runtime.history().is_empty());
        assert!(runtime.pending().is_empty());
    }

    #[test]
    fn an_ask_for_a_kind_queues_a_session_change_too() {
        let mut runtime = fresh_runtime();
        runtime.set_consent(ChangeKind::Persona, Consent::Ask);

        let pending = match runtime
            .apply(Change::session("s1", ChangePayload::Persona("ask me".into())))
            .unwrap()
        {
            ApplyOutcome::NeedsConsent(p) => p,
            other => panic!("expected consent, got {other:?}"),
        };
        assert!(pending.reason.contains("session"), "{:?}", pending.reason);
        assert!(
            runtime.state().persona.is_empty(),
            "nothing changes while it waits"
        );

        assert!(matches!(
            runtime.approve(&pending.id).unwrap(),
            ApplyOutcome::Applied(_)
        ));
        assert_eq!(runtime.state().persona, "ask me");
    }

    #[test]
    fn history_records_how_each_change_was_decided() {
        let mut runtime = fresh_runtime();
        runtime
            .apply(Change::session("s1", ChangePayload::Persona("a".into())))
            .unwrap();

        let pending = match runtime
            .apply(Change::global(ChangePayload::AddSkillRoot("/x".into())))
            .unwrap()
        {
            ApplyOutcome::NeedsConsent(p) => p,
            other => panic!("expected consent, got {other:?}"),
        };
        runtime.approve(&pending.id).unwrap();

        let history = runtime.history();
        assert_eq!(history.len(), 2);
        assert_eq!(history[0].decided_by, "session-scope");
        assert_eq!(history[1].decided_by, "approved");
        assert_eq!(history[0].seq, 1);
        assert_eq!(history[1].seq, 2);
        // Versions advance monotonically.
        assert!(history[1].version > history[0].version);
    }

    #[test]
    fn the_system_prompt_combines_persona_and_base() {
        let mut runtime = fresh_runtime();
        runtime
            .apply(Change::session("s1", ChangePayload::Persona("Be terse.".into())))
            .unwrap();
        let snapshot = runtime.snapshot();
        let prompt = snapshot.system_prompt("You are an agent.");
        assert!(prompt.starts_with("Be terse."));
        assert!(prompt.contains("You are an agent."));

        // With no persona the base prompt is used unchanged.
        let plain = fresh_runtime();
        assert_eq!(plain.snapshot().system_prompt("base"), "base");
    }

    #[test]
    fn unknown_pending_ids_are_reported() {
        let mut runtime = fresh_runtime();
        assert!(runtime.approve("nope").is_err());
        assert!(runtime.reject("nope").is_err());
    }

    #[test]
    fn a_model_route_change_switches_the_endpoint() {
        let mut runtime = fresh_runtime();
        runtime
            .apply(Change::session(
                "s1",
                ChangePayload::ModelRoute(ModelRoute {
                    provider: "test".into(),
                    base_url: "https://api.example.com/v1".into(),
                    api_key: "k".into(),
                    model: "deepseek-v4.1-flash".into(),
                    temperature: Some(0.2),
                    max_tokens: Some(4096),
                    reasoning_effort: None,
                    proxy: None,
                    ..Default::default()
                }),
            ))
            .unwrap();

        let route = &runtime.snapshot().model_route;
        assert_eq!(route.model, "deepseek-v4.1-flash");
        assert_eq!(route.base_url, "https://api.example.com/v1");
        assert!(!route.same_endpoint(&ModelRoute::default()));
    }

    /// Settings that differ from every default, so nothing can pass by
    /// agreeing with a default it never read.
    fn distinctive_settings() -> crate::settings::Settings {
        let mut settings = crate::settings::Settings::default();
        settings.base_url = "https://api.example.com/v1".into();
        settings.api_key = "sk-secret".into();
        settings.model = "session-model".into();
        settings.max_output_tokens = Some(4096);
        settings.reasoning_effort = "high".into();
        settings.delivery = "buffered".into();
        settings.degrade_on_stall = Some(false);
        settings.proxy = "http://127.0.0.1:7890".into();
        settings
            .extra_body
            .insert("thinking".into(), serde_json::json!({ "type": "enabled" }));
        settings.network.retry_attempts = 5;
        settings
    }

    #[test]
    fn a_route_from_settings_carries_everything_a_client_needs() {
        let route = ModelRoute::from_settings(&distinctive_settings());

        assert_eq!(route.api_key, "sk-secret");
        assert_eq!(route.max_tokens, Some(4096));
        assert_eq!(route.reasoning_effort.as_deref(), Some("high"));
        assert_eq!(route.delivery.as_deref(), Some("buffered"));
        assert_eq!(route.degrade_on_stall, Some(false));
        assert_eq!(route.network.retry_attempts, 5);
        assert_eq!(route.extra_body["thinking"]["type"], "enabled");
    }

    #[test]
    fn a_buffered_route_reaches_the_wire_as_one_body() {
        // The mode is only real if the request stops asking for a stream. A route
        // that stores "buffered" while still sending `"stream": true` would keep
        // the original failure and report it as a provider problem.
        let route = ModelRoute::from_settings(&distinctive_settings());
        let body = route.client().unwrap().preview_body(
            &[crate::message::Message::user("hi")],
            &[],
        );
        assert_eq!(body["stream"], false, "{body}");
    }

    #[test]
    fn a_subagent_inherits_its_sessions_delivery_mode() {
        // A subtask that streams while its parent buffers is a difference the
        // user never chose, and on a gateway whose stream never completes it is a
        // subtask that cannot answer.
        let route = ModelRoute::from_settings(&distinctive_settings());
        let child = route.client_for("cheaper-model").unwrap();
        let body = child.preview_body(&[crate::message::Message::user("hi")], &[]);
        assert_eq!(body["stream"], false, "{body}");
    }

    #[test]
    fn an_unknown_delivery_word_is_refused_rather_than_streamed() {
        // Defaulting here would silently do the opposite of what was written.
        let mut route = ModelRoute::from_settings(&distinctive_settings());
        route.delivery = Some("streaming".into());
        let error = match route.client() {
            Err(error) => error,
            Ok(_) => panic!("an unreadable mode must not quietly become a stream"),
        };
        assert!(
            format!("{error:#}").contains("streaming"),
            "the refusal names what was wrong: {error}"
        );
        assert!(
            format!("{error:#}").contains("buffered"),
            "and offers the words that work: {error}"
        );
    }

    #[test]
    fn a_client_built_from_a_route_uses_that_routes_numbers() {
        // Stored numbers that never reach the wire read the same in settings
        // and behave differently on the network, so the claim is checked on
        // the client rather than on the route.
        let route = ModelRoute::from_settings(&distinctive_settings());
        let client = route.client().unwrap();

        assert_eq!(client.model(), "session-model");
        assert_eq!(client.network().retry_attempts, 5);
        let body = client.preview_body(&[crate::message::Message::user("hi")], &[]);
        assert_eq!(body["thinking"]["type"], "enabled");
    }

    #[test]
    fn a_subagents_client_differs_only_in_the_model() {
        // A subtask that ran on different connection settings or without the
        // session's extra fields would be a difference the user never chose.
        let route = ModelRoute::from_settings(&distinctive_settings());
        let parent = route.client().unwrap();
        let child = route.client_for("cheaper-model").unwrap();

        assert_eq!(child.model(), "cheaper-model");
        assert_eq!(child.network(), parent.network());
        assert_eq!(child.shaper_name(), parent.shaper_name());
        let body = child.preview_body(&[crate::message::Message::user("hi")], &[]);
        assert_eq!(body["thinking"]["type"], "enabled");
    }

    #[test]
    fn a_route_written_before_the_new_fields_still_reads() {
        // Change payloads are persisted; an older one must load with the
        // defaults rather than refuse to deserialize.
        let written = r#"{"provider":"openai","base_url":"https://api.example.com/v1","model":"m","api_key":"k"}"#;
        let route: ModelRoute = serde_json::from_str(written).unwrap();

        assert_eq!(route.api_key, "k");
        assert_eq!(route.network, crate::network::NetworkSettings::default());
        assert!(route.extra_body.is_empty());
    }

    #[test]
    fn a_default_effort_is_sent_as_no_effort() {
        // `default` asks the provider for its own behaviour, so sending it as
        // a value would override exactly what the name promises.
        let mut settings = crate::settings::Settings::default();
        settings.reasoning_effort = "default".into();
        assert_eq!(ModelRoute::from_settings(&settings).reasoning_effort, None);

        let mut route = ModelRoute::default();
        route.reasoning_effort = Some("default".into());
        let client = route.client().unwrap();
        let body = client.preview_body(&[crate::message::Message::user("hi")], &[]);
        assert!(body.get("reasoning_effort").is_none(), "{body}");
    }
}
