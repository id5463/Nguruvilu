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
//! Scope is what consent is about: a session-scoped change is the default and
//! needs no approval, because it cannot affect anyone else. A global change
//! touches every session, so the policy table decides whether to ask.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

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
        }
    }
}

impl ModelRoute {
    /// Whether two routes differ in a way that changes the model endpoint.
    pub fn same_endpoint(&self, other: &Self) -> bool {
        self.base_url == other.base_url && self.model == other.model
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
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
    policy: HashMap<ChangeKind, Consent>,
    history: Vec<AppliedChange>,
    pending: HashMap<String, PendingChange>,
    tools: Arc<ToolRegistry>,
    version: u64,
    seq: u64,
}

impl Runtime {
    /// Build a runtime over a tool table, with the default policy.
    ///
    /// Session-scoped changes are allowed without asking; global ones ask;
    /// nothing is denied by default.
    pub fn new(tools: Arc<ToolRegistry>) -> Self {
        let mut policy = HashMap::new();
        for kind in ChangeKind::all() {
            policy.insert(kind, Consent::Ask);
        }
        Self {
            state: RuntimeState::default(),
            policy,
            history: Vec::new(),
            pending: HashMap::new(),
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
    pub fn set_tools(&mut self, tools: Arc<ToolRegistry>) {
        self.tools = tools;
        self.version += 1;
    }

    /// Apply a tool-table change coming from the kernel.
    pub fn adopt_kernel_tools(&mut self, kernel: &Kernel) {
        self.set_tools(Arc::new(kernel.tools().clone()));
    }

    /// Set the consent rule for one kind of change.
    pub fn set_consent(&mut self, kind: ChangeKind, consent: Consent) {
        self.policy.insert(kind, consent);
    }

    /// The consent rule for one kind of change.
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

    /// Changes awaiting consent.
    pub fn pending(&self) -> Vec<&PendingChange> {
        self.pending.values().collect()
    }

    /// Request a change.
    ///
    /// This is the only path that mutates runtime state.
    pub fn apply(&mut self, change: Change) -> Result<ApplyOutcome> {
        let kind = change.payload.kind();

        // Validate before anything is touched, so a rejected change cannot
        // leave a trace.
        validate(&change)?;

        let consent = self.consent(kind);

        if change.scope.is_global() {
            match consent {
                Consent::Deny => {
                    return Ok(ApplyOutcome::Denied(format!(
                        "policy denies global {} changes",
                        kind.as_str()
                    )));
                }
                Consent::Ask => {
                    let id = format!("chg-{}", self.seq + 1);
                    let pending = PendingChange {
                        id: id.clone(),
                        change: change.clone(),
                        reason: format!(
                            "global {} change affects every session",
                            kind.as_str()
                        ),
                    };
                    self.pending.insert(id, pending.clone());
                    return Ok(ApplyOutcome::NeedsConsent(pending));
                }
                Consent::AutoAllow => {}
            }
        }

        let decided_by = if change.scope.is_global() { "auto-allowed" } else { "session-scope" };
        self.commit(change, decided_by)
    }

    /// Approve a pending change.
    pub fn approve(&mut self, id: &str) -> Result<ApplyOutcome> {
        let pending = self
            .pending
            .remove(id)
            .ok_or_else(|| anyhow!("no pending change '{id}'"))?;
        self.commit(pending.change, "approved")
    }

    /// Approve a pending change and allow this kind automatically from now on.
    pub fn approve_and_remember(&mut self, id: &str) -> Result<ApplyOutcome> {
        let pending = self
            .pending
            .remove(id)
            .ok_or_else(|| anyhow!("no pending change '{id}'"))?;
        let kind = pending.change.payload.kind();
        self.policy.insert(kind, Consent::AutoAllow);
        self.commit(pending.change, "approved-and-remembered")
    }

    /// Reject a pending change. Nothing was applied.
    pub fn reject(&mut self, id: &str) -> Result<()> {
        self.pending
            .remove(id)
            .ok_or_else(|| anyhow!("no pending change '{id}'"))?;
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
        }
        Ok(())
    }
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
    fn remembering_a_decision_updates_the_policy() {
        let mut runtime = fresh_runtime();
        let pending = match runtime
            .apply(Change::global(ChangePayload::AddSkillRoot("/skills".into())))
            .unwrap()
        {
            ApplyOutcome::NeedsConsent(p) => p,
            other => panic!("expected consent, got {other:?}"),
        };

        runtime.approve_and_remember(&pending.id).unwrap();
        assert_eq!(runtime.consent(ChangeKind::SkillRoots), Consent::AutoAllow);

        // The next one of the same kind does not ask.
        let next = runtime
            .apply(Change::global(ChangePayload::AddSkillRoot("/more".into())))
            .unwrap();
        assert!(matches!(next, ApplyOutcome::Applied(_)));
        assert_eq!(runtime.state().skill_roots.len(), 2);
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
            assert_eq!(consent, Consent::Ask, "{kind:?} defaults to asking");
        }

        // Session scope never consults the table.
        let mut runtime = fresh_runtime();
        runtime.set_consent(ChangeKind::Persona, Consent::Deny);
        let outcome = runtime
            .apply(Change::session("s1", ChangePayload::Persona("ok".into())))
            .unwrap();
        assert!(
            matches!(outcome, ApplyOutcome::Applied(_)),
            "a denial governs global scope; a session cannot reach anyone else"
        );
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
                }),
            ))
            .unwrap();

        let route = &runtime.snapshot().model_route;
        assert_eq!(route.model, "deepseek-v4.1-flash");
        assert_eq!(route.base_url, "https://api.example.com/v1");
        assert!(!route.same_endpoint(&ModelRoute::default()));
    }
}
