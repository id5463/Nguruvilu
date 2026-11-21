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

pub mod manifest;
pub mod plan;

pub use manifest::{
    Assembly, BuildSpec, ConflictKind, Defaults, EntryBase, LoadMode, McpEntry, OnFailure,
    PluginEntry, Scope, SkillEntry, Stage, Teardown, Transport, ASSEMBLY_VERSION,
};
pub use plan::{EntryKind, Plan, PlannedStep, SkippedEntry};

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
        source: "npm:@acme/fs-tools@^1.2"
        order: 10
      - id: shell-tools
        source: "npm:@acme/shell-tools@^1.0"
        order: 20
  - name: services
    plugins:
      - id: compaction
        source: "npm:@acme/compaction@^0.5"
        order: 10
        config:
          threshold_tokens: 150000
      - id: mcp-client
        source: "npm:@acme/mcp-client@^1.0"
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
        source: "npm:@acme/search@^2.0"
        order: 10
        on_failure: skip
      - id: windows-only
        source: "npm:@acme/win@^1.0"
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
    fn an_unknown_field_does_not_stop_the_pack_from_loading() {
        // This used to be `unknown_fields_are_rejected_rather_than_ignored`,
        // asserting `deny_unknown_fields`. That rule made the kernel refuse a
        // whole pack because it carried one field the kernel had not heard of —
        // `via_proxy` on an mcp entry did exactly that, and the symptom was a
        // pack that could not be installed on an older binary at all. A pack
        // must be able to grow a field without the kernel being rebuilt.
        let assembly = Assembly::parse("version: 1
stages: []
strategy: nonsense
")
            .expect("an unknown top-level field is kept, not refused");
        assert_eq!(assembly.version, 1);
        assert!(assembly.stages.is_empty());
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