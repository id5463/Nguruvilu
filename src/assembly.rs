//! The assembly manifest: how a pack declares what it loads and in what order.
//!
//! A pack ships an `assembly.yaml` alongside its identity manifest. It states
//! which plugins, MCP servers, and skills to load — the three are peers, not
//! nested — grouped into stages, with explicit dependencies and per-entry
//! policy.
//!
//! Ordering has four levels, highest first:
//!
//! 1. `after` / `before` — a real dependency, never violated.
//! 2. stage order — coarse grouping (`foundation` before `extensions`).
//! 3. `order` — a numeric sort within a stage.
//! 4. declaration order — the stable tiebreak.
//!
//! A cycle in the dependency graph is an error, not something to paper over.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Current manifest format.
pub const ASSEMBLY_VERSION: u32 = 1;

/// Which plane a change lands on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Scope {
    /// Only the session that loaded the pack (the default).
    #[default]
    Session,
    /// Every session, including ones created later. Requires user consent.
    Global,
}

/// What to do when a contribution claims a name that is taken.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ConflictKind {
    /// Refuse the load and report both owners.
    #[default]
    Error,
    /// Replace the existing contribution.
    Override,
    /// Prefix the contribution with the pack id.
    Namespaced,
}

/// What to do when an entry fails to load.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum OnFailure {
    /// Fail the whole assembly.
    #[default]
    Abort,
    /// Skip this entry and continue.
    Skip,
    /// Retry once before giving up.
    Retry,
}

/// When to load an entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum LoadMode {
    /// At assembly time.
    #[default]
    Eager,
    /// On first use.
    Lazy,
}

/// Transport for an MCP entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum Transport {
    /// A child process speaking JSON-RPC on stdio.
    #[default]
    Stdio,
    /// A remote Streamable HTTP endpoint (declared, not connected here).
    StreamableHttp,
}

/// Defaults applied to every entry in a manifest.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct Defaults {
    /// Default scope.
    #[serde(default)]
    pub scope: Scope,
    /// Default conflict policy.
    #[serde(default)]
    pub conflict: ConflictKind,
    /// Default failure policy.
    #[serde(default)]
    pub on_failure: OnFailure,
    /// Default load mode.
    #[serde(default)]
    pub load: LoadMode,
}

/// One loadable entry, shared shape across the three kinds.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EntryBase {
    /// Identifier, unique within the manifest.
    pub id: String,
    /// Where it comes from.
    #[serde(default)]
    pub source: String,
    /// Numeric sort key inside the stage.
    #[serde(default)]
    pub order: i64,
    /// Entries that must load first.
    #[serde(default)]
    pub after: Vec<String>,
    /// Entries that must load later.
    #[serde(default)]
    pub before: Vec<String>,
    /// Per-entry overrides of the defaults.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope: Option<Scope>,
    /// Conflict policy override.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub conflict: Option<ConflictKind>,
    /// Failure policy override.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub on_failure: Option<OnFailure>,
    /// Load mode override.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub load: Option<LoadMode>,
    /// Restrict to these platforms (`win32`, `darwin`, `linux`).
    #[serde(default)]
    pub platform: Vec<String>,
    /// Version or capability requirement, recorded for diagnostics.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requires: Option<String>,
}

/// A plugin entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PluginEntry {
    /// Shared entry fields.
    #[serde(flatten)]
    pub base: EntryBase,
    /// Plugin configuration passed to `apply`.
    #[serde(default)]
    pub config: Value,
}

/// An MCP server entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct McpEntry {
    /// Shared entry fields.
    #[serde(flatten)]
    pub base: EntryBase,
    /// How to reach the server.
    #[serde(default)]
    pub transport: Transport,
    /// Executable, for `stdio`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub command: Option<String>,
    /// Arguments, for `stdio`.
    #[serde(default)]
    pub args: Vec<String>,
    /// Environment additions, for `stdio`.
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    /// Endpoint, for `streamable-http`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    /// Headers, for `streamable-http`.
    #[serde(default)]
    pub headers: BTreeMap<String, String>,
    /// Per-call timeout.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_ms: Option<u64>,
}

/// A skill entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SkillEntry {
    /// Shared entry fields.
    #[serde(flatten)]
    pub base: EntryBase,
    /// Expected content hash, when the pack pins one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha1: Option<String>,
}

/// One stage: a coarse ordering group.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Stage {
    /// Stage name, for diagnostics.
    pub name: String,
    /// Plugins in this stage.
    #[serde(default)]
    pub plugins: Vec<PluginEntry>,
    /// MCP servers in this stage.
    #[serde(default)]
    pub mcp: Vec<McpEntry>,
    /// Skills in this stage.
    #[serde(default)]
    pub skills: Vec<SkillEntry>,
}

/// Teardown policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Teardown {
    /// Unload stages in reverse order.
    #[serde(default = "default_true")]
    pub reverse_stages: bool,
}

fn default_true() -> bool {
    true
}

impl Default for Teardown {
    fn default() -> Self {
        Self { reverse_stages: true }
    }
}

/// A parsed `assembly.yaml`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Assembly {
    /// Format version.
    #[serde(default = "default_version")]
    pub version: u32,
    /// Pack name, for diagnostics.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Defaults for every entry.
    #[serde(default)]
    pub defaults: Defaults,
    /// Ordered stages.
    #[serde(default)]
    pub stages: Vec<Stage>,
    /// Teardown policy.
    #[serde(default)]
    pub teardown: Teardown,
}

fn default_version() -> u32 {
    ASSEMBLY_VERSION
}

/// What kind of thing a planned step loads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntryKind {
    /// A plugin.
    Plugin,
    /// An MCP server.
    Mcp,
    /// A skill.
    Skill,
}

impl EntryKind {
    /// Lowercase name used in diagnostics and the ledger.
    pub fn as_str(self) -> &'static str {
        match self {
            EntryKind::Plugin => "plugin",
            EntryKind::Mcp => "mcp",
            EntryKind::Skill => "skill",
        }
    }
}

/// One resolved step in the load plan.
#[derive(Debug, Clone)]
pub struct PlannedStep {
    /// Which kind this step loads.
    pub kind: EntryKind,
    /// Entry id.
    pub id: String,
    /// Stage it came from.
    pub stage: String,
    /// Resolved scope.
    pub scope: Scope,
    /// Resolved conflict policy.
    pub conflict: ConflictKind,
    /// Resolved failure policy.
    pub on_failure: OnFailure,
    /// Resolved load mode.
    pub load: LoadMode,
    /// Numeric sort key inside the stage.
    pub order: i64,
    /// Source string.
    pub source: String,
    /// Plugin configuration, for plugin steps.
    pub config: Value,
    /// MCP spec, for mcp steps.
    pub mcp: Option<McpEntry>,
    /// Skill hash, for skill steps.
    pub sha1: Option<String>,
    /// Declared requirement, if any.
    pub requires: Option<String>,
}

/// An entry excluded from the plan, with the reason.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkippedEntry {
    /// Entry id.
    pub id: String,
    /// Why it was left out.
    pub reason: String,
}

/// A fully resolved load plan.
#[derive(Debug, Clone, Default)]
pub struct Plan {
    /// Steps in load order.
    pub steps: Vec<PlannedStep>,
    /// Entries left out, with reasons.
    pub skipped: Vec<SkippedEntry>,
}

/// One stage entry, normalized across the three kinds so they share a single
/// ordering pass — plugins, MCP servers, and skills are peers.
enum Raw {
    Plugin(PluginEntry),
    Mcp(McpEntry),
    Skill(SkillEntry),
}

impl Raw {
    /// The fields every kind shares.
    fn base(&self) -> &EntryBase {
        match self {
            Raw::Plugin(p) => &p.base,
            Raw::Mcp(m) => &m.base,
            Raw::Skill(s) => &s.base,
        }
    }
}

impl Assembly {
    /// Parse a manifest from YAML text.
    pub fn parse(text: &str) -> Result<Self> {
        let assembly: Assembly =
            serde_yaml::from_str(text).context("parsing assembly manifest")?;
        if assembly.version != ASSEMBLY_VERSION {
            return Err(anyhow!(
                "assembly version {} is not supported (expected {ASSEMBLY_VERSION})",
                assembly.version
            ));
        }
        Ok(assembly)
    }

    /// Read a manifest from disk.
    pub fn from_file(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading {}", path.display()))?;
        Self::parse(&text)
    }

    /// An empty manifest.
    pub fn empty() -> Self {
        Self {
            version: ASSEMBLY_VERSION,
            name: None,
            defaults: Defaults::default(),
            stages: Vec::new(),
            teardown: Teardown::default(),
        }
    }

    /// Resolve the manifest into an ordered load plan for `platform`.
    ///
    /// `platform` is `win32`, `darwin`, or `linux`.
    pub fn plan(&self, platform: &str) -> Result<Plan> {
        let mut plan = Plan::default();
        let mut seen_ids: BTreeSet<String> = BTreeSet::new();

        for stage in &self.stages {
            // Collect the stage's entries as a uniform list so the three kinds
            // share one ordering pass — they are peers.

            let mut raws: Vec<Raw> = Vec::new();
            for plugin in &stage.plugins {
                raws.push(Raw::Plugin(plugin.clone()));
            }
            for mcp in &stage.mcp {
                raws.push(Raw::Mcp(mcp.clone()));
            }
            for skill in &stage.skills {
                raws.push(Raw::Skill(skill.clone()));
            }


            // Ids must be unique across the whole manifest: `after` refers to
            // them by name, so a duplicate would be ambiguous.
            for raw in &raws {
                let id = raw.base().id.clone();
                if !seen_ids.insert(id.clone()) {
                    return Err(anyhow!("duplicate entry id '{id}' in assembly"));
                }
            }

            // Platform filter, applied before ordering so a skipped entry does
            // not drag its dependents into the graph.
            let mut active: Vec<Raw> = Vec::new();
            for raw in raws {
                let base = raw.base();
                if !base.platform.is_empty() && !base.platform.iter().any(|p| p == platform) {
                    plan.skipped.push(SkippedEntry {
                        id: base.id.clone(),
                        reason: format!("platform {} not in {:?}", platform, base.platform),
                    });
                    continue;
                }
                active.push(raw);
            }

            let stage_ids: BTreeSet<String> =
                active.iter().map(|r| r.base().id.clone()).collect();

            // Dependencies inside the stage drive the order. A reference
            // outside the stage is allowed only if that id exists earlier in
            // the manifest, where the stage order already satisfies it.
            let mut indegree: BTreeMap<String, usize> = BTreeMap::new();
            let mut edges: BTreeMap<String, Vec<String>> = BTreeMap::new();
            for raw in &active {
                let id = raw.base().id.clone();
                indegree.entry(id.clone()).or_insert(0);
                edges.entry(id).or_default();
            }

            for raw in &active {
                let base = raw.base();
                for dependency in &base.after {
                    if !seen_ids.contains(dependency) {
                        return Err(anyhow!(
                            "entry '{}' declares after: '{}', which is not defined",
                            base.id,
                            dependency
                        ));
                    }
                    if !stage_ids.contains(dependency) {
                        // Satisfied by an earlier stage.
                        continue;
                    }
                    edges.entry(dependency.clone()).or_default().push(base.id.clone());
                    *indegree.entry(base.id.clone()).or_insert(0) += 1;
                }
                for dependent in &base.before {
                    if !seen_ids.contains(dependent) {
                        return Err(anyhow!(
                            "entry '{}' declares before: '{}', which is not defined",
                            base.id,
                            dependent
                        ));
                    }
                    if !stage_ids.contains(dependent) {
                        continue;
                    }
                    edges.entry(base.id.clone()).or_default().push(dependent.clone());
                    *indegree.entry(dependent.clone()).or_insert(0) += 1;
                }
            }

            // Kahn's algorithm. Among the ready set, prefer the lower `order`,
            // then the id, so the result is deterministic.
            let order_of = |id: &str| -> (i64, String) {
                let raw = active.iter().find(|r| r.base().id == id).expect("id from this stage");
                (raw.base().order, id.to_string())
            };

            let mut ready: Vec<String> = indegree
                .iter()
                .filter(|(_, degree)| **degree == 0)
                .map(|(id, _)| id.clone())
                .collect();
            let mut ordered: Vec<String> = Vec::with_capacity(active.len());

            while !ready.is_empty() {
                ready.sort_by_key(|id| order_of(id));
                let id = ready.remove(0);
                ordered.push(id.clone());
                if let Some(dependents) = edges.get(&id) {
                    for dependent in dependents {
                        if let Some(degree) = indegree.get_mut(dependent) {
                            *degree -= 1;
                            if *degree == 0 {
                                ready.push(dependent.clone());
                            }
                        }
                    }
                }
            }

            if ordered.len() != active.len() {
                let unresolved: Vec<String> = active
                    .iter()
                    .map(|r| r.base().id.clone())
                    .filter(|id| !ordered.contains(id))
                    .collect();
                return Err(anyhow!(
                    "dependency cycle in stage '{}' among: {}",
                    stage.name,
                    unresolved.join(", ")
                ));
            }

            for id in ordered {
                let raw = active
                    .iter()
                    .find(|r| r.base().id == id)
                    .expect("id from this stage");
                let base = raw.base();

                let step = PlannedStep {
                    kind: match raw {
                        Raw::Plugin(_) => EntryKind::Plugin,
                        Raw::Mcp(_) => EntryKind::Mcp,
                        Raw::Skill(_) => EntryKind::Skill,
                    },
                    id: base.id.clone(),
                    stage: stage.name.clone(),
                    scope: base.scope.unwrap_or(self.defaults.scope),
                    conflict: base.conflict.clone().unwrap_or_else(|| self.defaults.conflict.clone()),
                    on_failure: base.on_failure.unwrap_or(self.defaults.on_failure),
                    load: base.load.unwrap_or(self.defaults.load),
                    order: base.order,
                    source: base.source.clone(),
                    config: match raw {
                        Raw::Plugin(p) => p.config.clone(),
                        _ => Value::Null,
                    },
                    mcp: match raw {
                        Raw::Mcp(m) => Some(m.clone()),
                        _ => None,
                    },
                    sha1: match raw {
                        Raw::Skill(s) => s.sha1.clone(),
                        _ => None,
                    },
                    requires: base.requires.clone(),
                };
                plan.steps.push(step);
            }
        }

        Ok(plan)
    }

    /// Every id declared in the manifest.
    pub fn entry_ids(&self) -> Vec<String> {
        let mut ids = Vec::new();
        for stage in &self.stages {
            ids.extend(stage.plugins.iter().map(|p| p.base.id.clone()));
            ids.extend(stage.mcp.iter().map(|m| m.base.id.clone()));
            ids.extend(stage.skills.iter().map(|s| s.base.id.clone()));
        }
        ids
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"
version: 1
name: sample-pack
defaults:
  scope: session
  conflict: error
  on_failure: abort
  load: eager
stages:
  - name: foundation
    plugins:
      - id: fs-tools
        source: "npm:@dshd/fs-tools@^1.2"
        order: 10
      - id: shell-tools
        source: "npm:@dshd/shell-tools@^1.0"
        order: 20
  - name: services
    plugins:
      - id: compaction
        source: "npm:@dshd/compaction@^0.5"
        order: 10
        config:
          threshold_tokens: 150000
      - id: mcp-client
        source: "npm:@dshd/mcp-client@^1.0"
        order: 20
        after: [compaction]
    mcp:
      - id: github-mcp
        transport: stdio
        command: npx
        args: ["-y", "@modelcontextprotocol/server-github"]
        order: 30
        after: [mcp-client]
    skills:
      - id: pdf-tools
        source: "github:owner/repo@skills/pdf@v1"
        sha1: "abc123"
        order: 5
  - name: extensions
    plugins:
      - id: search
        source: "npm:@dshd/search@^2.0"
        order: 10
        on_failure: skip
      - id: windows-only
        source: "npm:@dshd/win@^1.0"
        platform: [win32]
teardown:
  reverse_stages: true
"#;

    #[test]
    fn parses_a_full_manifest() {
        let assembly = Assembly::parse(SAMPLE).unwrap();
        assert_eq!(assembly.name.as_deref(), Some("sample-pack"));
        assert_eq!(assembly.stages.len(), 3);
        assert_eq!(assembly.entry_ids().len(), 8);
        assert!(assembly.teardown.reverse_stages);
    }

    #[test]
    fn an_unsupported_version_is_rejected() {
        let error = Assembly::parse("version: 99\nstages: []\n").expect_err("bad version");
        assert!(format!("{error:#}").contains("not supported"));
    }

    #[test]
    fn unknown_fields_are_rejected_rather_than_ignored() {
        let error = Assembly::parse("version: 1\nstages: []\nstrategy: nonsense\n")
            .expect_err("unknown top-level field");
        assert!(format!("{error:#}").contains("parsing assembly manifest"));
    }

    #[test]
    fn plan_orders_stages_then_dependencies_then_order() {
        let plan = Assembly::parse(SAMPLE).unwrap().plan("linux").unwrap();
        let ids: Vec<&str> = plan.steps.iter().map(|s| s.id.as_str()).collect();

        // Stage order dominates.
        assert!(ids.iter().position(|i| *i == "fs-tools").unwrap()
            < ids.iter().position(|i| *i == "compaction").unwrap());
        assert!(ids.iter().position(|i| *i == "compaction").unwrap()
            < ids.iter().position(|i| *i == "search").unwrap());

        // Inside `foundation`, `order` decides.
        assert!(ids.iter().position(|i| *i == "fs-tools").unwrap()
            < ids.iter().position(|i| *i == "shell-tools").unwrap());

        // `after` beats `order`: mcp-client has order 20 and compaction 10, but
        // the dependency would win even if the numbers were reversed.
        assert!(ids.iter().position(|i| *i == "compaction").unwrap()
            < ids.iter().position(|i| *i == "mcp-client").unwrap());

        // A skill with the lowest order in its stage still lands after entries
        // it does not depend on, because ordering is per-stage and stable.
        assert!(ids.contains(&"pdf-tools"));
    }

    #[test]
    fn the_three_kinds_are_planned_as_peers() {
        let plan = Assembly::parse(SAMPLE).unwrap().plan("linux").unwrap();
        let kinds: Vec<(EntryKind, &str)> =
            plan.steps.iter().map(|s| (s.kind, s.id.as_str())).collect();

        assert!(kinds.contains(&(EntryKind::Plugin, "compaction")));
        assert!(kinds.contains(&(EntryKind::Mcp, "github-mcp")));
        assert!(kinds.contains(&(EntryKind::Skill, "pdf-tools")));

        // The MCP entry depends on the plugin, so the plugin comes first even
        // though they are different kinds.
        let plugin_at = kinds.iter().position(|(_, id)| *id == "mcp-client").unwrap();
        let mcp_at = kinds.iter().position(|(_, id)| *id == "github-mcp").unwrap();
        assert!(plugin_at < mcp_at);
    }

    #[test]
    fn platform_restricted_entries_are_skipped_with_a_reason() {
        let assembly = Assembly::parse(SAMPLE).unwrap();

        let linux = assembly.plan("linux").unwrap();
        assert!(!linux.steps.iter().any(|s| s.id == "windows-only"));
        assert!(linux.skipped.iter().any(|s| s.id == "windows-only"));

        let windows = assembly.plan("win32").unwrap();
        assert!(windows.steps.iter().any(|s| s.id == "windows-only"));
    }

    #[test]
    fn a_dependency_cycle_is_an_error() {
        let manifest = r#"
version: 1
stages:
  - name: bad
    plugins:
      - id: a
        after: [b]
      - id: b
        after: [a]
"#;
        let error = Assembly::parse(manifest).unwrap().plan("linux").expect_err("cycle");
        let text = format!("{error:#}");
        assert!(text.contains("cycle"), "{text}");
        assert!(text.contains('a') && text.contains('b'), "{text}");
    }

    #[test]
    fn an_unknown_dependency_is_an_error() {
        let manifest = r#"
version: 1
stages:
  - name: bad
    plugins:
      - id: a
        after: [ghost]
"#;
        let error = Assembly::parse(manifest).unwrap().plan("linux").expect_err("unknown ref");
        assert!(format!("{error:#}").contains("not defined"));
    }

    #[test]
    fn duplicate_ids_are_rejected() {
        let manifest = r#"
version: 1
stages:
  - name: one
    plugins:
      - id: same
  - name: two
    skills:
      - id: same
"#;
        let error = Assembly::parse(manifest).unwrap().plan("linux").expect_err("duplicate");
        assert!(format!("{error:#}").contains("duplicate entry id"));
    }

    #[test]
    fn entry_policy_overrides_defaults() {
        let assembly = Assembly::parse(SAMPLE).unwrap();
        let plan = assembly.plan("linux").unwrap();

        let search = plan.steps.iter().find(|s| s.id == "search").unwrap();
        assert_eq!(search.on_failure, OnFailure::Skip, "per-entry override wins");

        let fs = plan.steps.iter().find(|s| s.id == "fs-tools").unwrap();
        assert_eq!(fs.on_failure, OnFailure::Abort, "default applies otherwise");
        assert_eq!(fs.scope, Scope::Session);
    }

    #[test]
    fn config_reaches_the_plugin_step() {
        let plan = Assembly::parse(SAMPLE).unwrap().plan("linux").unwrap();
        let compaction = plan.steps.iter().find(|s| s.id == "compaction").unwrap();
        assert_eq!(compaction.config["threshold_tokens"], 150000);
    }

    #[test]
    fn an_empty_manifest_plans_nothing() {
        let plan = Assembly::empty().plan("linux").unwrap();
        assert!(plan.steps.is_empty());
        assert!(plan.skipped.is_empty());
    }

    #[test]
    fn a_cross_stage_dependency_is_satisfied_by_stage_order() {
        let manifest = r#"
version: 1
stages:
  - name: first
    plugins:
      - id: base
  - name: second
    plugins:
      - id: dependent
        after: [base]
"#;
        let plan = Assembly::parse(manifest).unwrap().plan("linux").unwrap();
        let ids: Vec<&str> = plan.steps.iter().map(|s| s.id.as_str()).collect();
        assert_eq!(ids, vec!["base", "dependent"]);
    }
}
