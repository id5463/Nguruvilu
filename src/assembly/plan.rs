//! The load plan: manifest entries resolved into an ordered list of steps.
//!
//! Ordering is computed here — dependencies first, then stage order, then
//! `order`, then declaration order. The types live here too, because they
//! describe the plan and nothing else.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{anyhow, Result};
use serde_json::Value;

use super::manifest::{
    Assembly, BuildSpec, ConflictKind, EntryBase, LoadMode, McpEntry, OnFailure, PluginEntry,
    Scope, SkillEntry,
};

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
    /// How to build the executable, for plugin steps that ship source.
    pub build: Option<BuildSpec>,
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
                    build: match raw {
                        Raw::Plugin(p) => p.build.clone(),
                        _ => None,
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
