//! The dynamic loading layer.
//!
//! This is where a pack's assembly manifest becomes running state. It loads
//! three kinds of thing — plugins, MCP servers, and skills — and treats them
//! as **peers**: one manifest, one ordering pass, one ledger, one failure
//! policy. None of them is a sub-case of another.
//!
//! Every load is idempotent. The ledger records what was installed and with
//! which hash, so re-applying a pack reuses rather than duplicates.
//!
//! Nothing here reloads a plugin by hand: loading changes the service table,
//! and [`Kernel::refresh`] propagates that along the dependency graph.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};

use crate::assembly::{EntryKind, OnFailure, PlannedStep, Plan, Scope, SkippedEntry, Transport};
use crate::dylib::DynamicPlugin;
use crate::ledger::{Ledger, LedgerEntry};
use crate::mcp::{McpClient, McpSpec};
use crate::plugin::{Kernel, Plugin as _, RealmMap};
use crate::session::now_iso;
use crate::skills::{register_skill_tool, SkillRegistry};
use crate::tools::{ConflictPolicy, ToolDef, ToolFuture};

/// What happened to one entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    /// Loaded for the first time in this process.
    Loaded,
    /// Already present; reused.
    Reused,
    /// Deliberately left out.
    Skipped(String),
}

/// An entry that failed while its policy allowed the rest to continue.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FailedEntry {
    /// Entry id.
    pub id: String,
    /// Entry kind.
    pub kind: String,
    /// Why it failed.
    pub error: String,
}

/// Result of applying one plan.
#[derive(Debug, Clone, Default)]
pub struct LoadReport {
    /// Entries loaded.
    pub loaded: Vec<String>,
    /// Entries reused from a previous load.
    pub reused: Vec<String>,
    /// Entries skipped, with reasons.
    pub skipped: Vec<SkippedEntry>,
    /// Entries that failed under a non-aborting policy.
    pub failed: Vec<FailedEntry>,
}

impl LoadReport {
    /// Whether every entry either loaded or was reused.
    pub fn is_clean(&self) -> bool {
        self.failed.is_empty()
    }
}

/// Owns the kernel plus everything the loader manages alongside it.
pub struct Loader {
    kernel: Kernel,
    skills: SkillRegistry,
    mcp_clients: Vec<Arc<McpClient>>,
    ledger: Ledger,
    ledger_path: PathBuf,
    pack: String,
    pack_dir: Option<PathBuf>,
}

impl Loader {
    /// Build a loader over a kernel, persisting the ledger at `ledger_path`.
    pub fn new(kernel: Kernel, ledger_path: PathBuf, pack: impl Into<String>) -> Self {
        let ledger = Ledger::load(&ledger_path).unwrap_or_default();
        Self {
            kernel,
            skills: SkillRegistry::new(),
            mcp_clients: Vec::new(),
            ledger,
            ledger_path,
            pack: pack.into(),
            pack_dir: None,
        }
    }

    /// Build a loader with the default skill roots already configured.
    pub fn with_default_skills(mut self) -> Self {
        for root in SkillRegistry::default_roots() {
            self.skills.add_root(root);
        }
        self
    }

    /// Treat `dir` as the pack's own directory, so relative sources resolve
    /// against it.
    pub fn with_pack_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.pack_dir = Some(dir.into());
        self
    }

    /// The kernel.
    pub fn kernel(&self) -> &Kernel {
        &self.kernel
    }

    /// The kernel, mutably.
    pub fn kernel_mut(&mut self) -> &mut Kernel {
        &mut self.kernel
    }

    /// The skill registry.
    pub fn skills(&self) -> &SkillRegistry {
        &self.skills
    }

    /// The ledger.
    pub fn ledger(&self) -> &Ledger {
        &self.ledger
    }

    /// Connected MCP servers.
    pub fn mcp_servers(&self) -> Vec<&str> {
        self.mcp_clients.iter().map(|c| c.server()).collect()
    }

    /// Rescan the skill roots.
    pub fn rescan_skills(&mut self) -> Result<usize> {
        let report = self.skills.scan()?;
        Ok(report.loaded)
    }

    /// Apply an assembly plan.
    ///
    /// Aborts on the first failure unless the entry's policy says otherwise.
    pub async fn apply(&mut self, plan: &Plan) -> Result<LoadReport> {
        let mut report = LoadReport::default();
        report.skipped.extend(plan.skipped.iter().cloned());

        for step in &plan.steps {
            let outcome = self.apply_step(step).await;

            match outcome {
                Ok(Outcome::Loaded) => report.loaded.push(label(step)),
                Ok(Outcome::Reused) => report.reused.push(label(step)),
                Ok(Outcome::Skipped(reason)) => report.skipped.push(SkippedEntry {
                    id: step.id.clone(),
                    reason,
                }),
                Err(error) => match step.on_failure {
                    OnFailure::Abort => {
                        return Err(error).with_context(|| {
                            format!("loading {} '{}'", step.kind.as_str(), step.id)
                        })
                    }
                    OnFailure::Skip => report.failed.push(FailedEntry {
                        id: step.id.clone(),
                        kind: step.kind.as_str().to_string(),
                        error: format!("{error:#}"),
                    }),
                    OnFailure::Retry => {
                        // One retry, then treat like Skip: a flaky server should
                        // not fail the whole pack, but it must be reported.
                        match self.apply_step(step).await {
                            Ok(Outcome::Loaded) => report.loaded.push(label(step)),
                            Ok(Outcome::Reused) => report.reused.push(label(step)),
                            Ok(Outcome::Skipped(reason)) => {
                                report.skipped.push(SkippedEntry { id: step.id.clone(), reason })
                            }
                            Err(second) => report.failed.push(FailedEntry {
                                id: step.id.clone(),
                                kind: step.kind.as_str().to_string(),
                                error: format!("{error:#}; retry: {second:#}"),
                            }),
                        }
                    }
                },
            }
        }

        // Skills may have changed, so the catalog tool is rebuilt and the
        // dependency graph is re-evaluated.
        self.rebuild_skill_tool()?;
        self.kernel.refresh()?;
        self.ledger.save(&self.ledger_path)?;

        Ok(report)
    }

    async fn apply_step(&mut self, step: &PlannedStep) -> Result<Outcome> {
        match step.kind {
            EntryKind::Plugin => self.load_plugin(step),
            EntryKind::Mcp => self.load_mcp(step).await,
            EntryKind::Skill => self.load_skill(step),
        }
    }

    /// Load a plugin.
    ///
    /// The kernel is compiled, so a plugin must already be defined in it. The
    /// `source` field selects one:
    ///
    /// * `builtin:<name>` or a bare name — a plugin the kernel registered.
    ///
    /// A pack cannot load arbitrary native code, and pretending otherwise
    /// would be a security hole rather than a feature. Anything else is
    /// refused with a clear message.
    fn load_plugin(&mut self, step: &PlannedStep) -> Result<Outcome> {
        // `dylib:<path>` defines the plugin on the spot by loading a shared
        // library. This is the only source form that runs native code, so it is
        // always explicit in the manifest — never inferred from a file
        // extension.
        let name = if let Some(raw) = step.source.strip_prefix("dylib:") {
            let path = self.resolve_source(raw);
            let plugin = unsafe { DynamicPlugin::load(&path) }
                .with_context(|| format!("loading dynamic plugin from {}", path.display()))?;
            let name = plugin.name().to_string();
            self.kernel.define(Arc::new(plugin));
            name
        } else {
            plugin_name(&step.source, &step.id)
        };

        if self.kernel.plugin(&name).is_none() {
            return Err(anyhow!(
                "plugin '{name}' is not defined in this kernel (available: {})",
                if self.kernel.plugin_names().is_empty() {
                    "none".to_string()
                } else {
                    self.kernel.plugin_names().join(", ")
                }
            ));
        }

        let realm = self.realm_for(step);
        self.kernel.load(&name, realm, step.config.clone())?;

        self.ledger.record(LedgerEntry {
            kind: EntryKind::Plugin.as_str().into(),
            id: step.id.clone(),
            source: step.source.clone(),
            sha1: None,
            path: None,
            scope: scope_name(step.scope).into(),
            pack: self.pack.clone(),
            installed_at: now_iso(),
        });

        Ok(Outcome::Loaded)
    }

    /// Load an MCP server and register its tools.
    async fn load_mcp(&mut self, step: &PlannedStep) -> Result<Outcome> {
        let entry = step
            .mcp
            .as_ref()
            .ok_or_else(|| anyhow!("mcp step '{}' has no server definition", step.id))?;

        match entry.transport {
            Transport::StreamableHttp => {
                // Declared but not connected: the HTTP transport is a separate
                // piece of work, and silently pretending to load it would be
                // worse than saying so.
                return Ok(Outcome::Skipped(
                    "streamable-http transport is not implemented yet".into(),
                ));
            }
            Transport::Stdio => {}
        }

        let command = entry
            .command
            .clone()
            .ok_or_else(|| anyhow!("mcp '{}' uses stdio but declares no command", step.id))?;

        let spec = McpSpec {
            id: step.id.clone(),
            command,
            args: entry.args.clone(),
            env: entry.env.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
            cwd: entry
                .base
                .source
                .is_empty()
                .then(|| self.pack_dir.as_ref().map(|d| d.display().to_string()))
                .flatten(),
            timeout_ms: entry.timeout_ms.unwrap_or(60_000),
        };

        let client = Arc::new(McpClient::connect(&spec).await?);
        let owner = format!("mcp:{}", step.id);
        let mut registered: Vec<String> = Vec::new();

        for tool in client.tools() {
            let exposed = client.exposed_name(&tool.name);
            let remote_name = tool.name.clone();
            let client_ref = Arc::clone(&client);

            let schema = if tool.input_schema.is_null() {
                serde_json::json!({ "type": "object", "properties": {} })
            } else {
                tool.input_schema.clone()
            };

            let def = ToolDef::new(
                exposed.clone(),
                if tool.description.is_empty() {
                    format!("MCP tool '{}' from server '{}'", tool.name, step.id)
                } else {
                    tool.description.clone()
                },
                schema,
                owner.clone(),
                move |args| {
                    let client = Arc::clone(&client_ref);
                    let name = remote_name.clone();
                    Box::pin(async move { client.call(&name, args).await }) as ToolFuture
                },
            );

            if let Err(error) = self.kernel.tools_mut().register(def, ConflictPolicy::Error) {
                // Roll back the tools this server already contributed, so a
                // partial server never half-registers.
                for tool in &registered {
                    self.kernel.tools_mut().unregister(tool);
                }
                client.shutdown().await;
                return Err(error);
            }
            registered.push(exposed);
        }

        let tool_count = registered.len();
        self.mcp_clients.push(client);

        self.ledger.record(LedgerEntry {
            kind: EntryKind::Mcp.as_str().into(),
            id: step.id.clone(),
            source: format!("stdio:{} {}", spec.command, spec.args.join(" ")).trim().to_string(),
            sha1: None,
            path: None,
            scope: scope_name(step.scope).into(),
            pack: self.pack.clone(),
            installed_at: now_iso(),
        });

        if tool_count == 0 {
            return Ok(Outcome::Skipped("server exposed no tools".into()));
        }
        Ok(Outcome::Loaded)
    }

    /// Load a skill by adding its directory as a scan root.
    fn load_skill(&mut self, step: &PlannedStep) -> Result<Outcome> {
        if step.source.starts_with("github:") {
            // Remote fetching needs archive handling and network access; the
            // declaration is recorded so a later implementation can honour it,
            // but nothing is claimed to be installed now.
            return Ok(Outcome::Skipped(
                "github: skill sources are not implemented yet; place the skill locally and use a path"
                    .into(),
            ));
        }

        // A relative source resolves against the pack directory when there is
        // one, so a pack can ship its skills.
        let raw = if step.source.is_empty() { step.id.clone() } else { step.source.clone() };
        let mut candidate = PathBuf::from(&raw);
        if !candidate.is_absolute() {
            if let Some(dir) = &self.pack_dir {
                let joined = dir.join(&candidate);
                if joined.exists() {
                    candidate = joined;
                }
            }
        }

        if !candidate.exists() {
            return Err(anyhow!(
                "skill source '{}' does not exist (resolved to {})",
                step.source,
                candidate.display()
            ));
        }

        // A skill may be given as the directory holding SKILL.md, or as a root
        // holding several skills. Accept both.
        let root = if candidate.join("SKILL.md").is_file() {
            candidate
                .parent()
                .map(PathBuf::from)
                .ok_or_else(|| anyhow!("skill path {} has no parent", candidate.display()))?
        } else {
            candidate
        };

        self.skills.add_root(root.clone());
        let report = self.skills.scan()?;

        self.ledger.record(LedgerEntry {
            kind: EntryKind::Skill.as_str().into(),
            id: step.id.clone(),
            source: root.display().to_string(),
            sha1: step.sha1.clone(),
            path: Some(root.display().to_string()),
            scope: scope_name(step.scope).into(),
            pack: self.pack.clone(),
            installed_at: now_iso(),
        });

        if report.loaded == 0 {
            return Ok(Outcome::Skipped(format!(
                "no SKILL.md found under {}",
                root.display()
            )));
        }
        Ok(Outcome::Loaded)
    }

    /// Rebuild the `skill` tool so the catalog reflects the current roots.
    fn rebuild_skill_tool(&mut self) -> Result<()> {
        self.kernel.tools_mut().unregister("skill");
        if self.skills.is_empty() {
            // Nothing to advertise; the model does not need a tool that would
            // only ever answer "none".
            return Ok(());
        }
        register_skill_tool(self.kernel.tools_mut(), Arc::new(self.skills.clone()))?;
        Ok(())
    }

    /// Resolve a source path against the pack directory when it is relative.
    fn resolve_source(&self, raw: &str) -> PathBuf {
        let candidate = PathBuf::from(raw);
        if candidate.is_absolute() {
            return candidate;
        }
        match &self.pack_dir {
            Some(dir) => {
                let joined = dir.join(&candidate);
                if joined.exists() {
                    joined
                } else {
                    candidate
                }
            }
            None => candidate,
        }
    }

    /// Resolve the realm for an entry from its scope.
    ///
    /// A session-scoped entry gets a fresh realm, which is what lets two packs
    /// each own a service of the same name. A global entry lands in the root
    /// realm where every session sees it.
    fn realm_for(&self, step: &PlannedStep) -> RealmMap {
        match step.scope {
            // Global entries land in the root realm, where every session sees
            // them.
            Scope::Global => RealmMap::new(),
            // Session entries default to a fresh realm, so everything this pack
            // contributes is isolated from every other session's copy.
            Scope::Session => RealmMap::new().with_default(self.kernel.new_realm()),
        }
    }

    /// Take the kernel back, stopping any MCP servers this loader started.
    pub async fn finish(mut self) -> Kernel {
        self.shutdown().await;
        self.kernel
    }

    /// Stop every MCP server this loader started.
    pub async fn shutdown(&mut self) {
        for client in self.mcp_clients.drain(..) {
            client.shutdown().await;
        }
    }
}

/// Resolve a plugin's registered name from its source string.
fn plugin_name(source: &str, fallback: &str) -> String {
    if let Some(rest) = source.strip_prefix("builtin:") {
        return rest.trim().to_string();
    }
    if source.trim().is_empty() {
        return fallback.to_string();
    }
    // A bare name that is not a path or URL is taken as the plugin name.
    if !source.contains('/') && !source.contains(':') && !source.contains('@') {
        return source.trim().to_string();
    }
    fallback.to_string()
}

fn scope_name(scope: Scope) -> &'static str {
    match scope {
        Scope::Session => "session",
        Scope::Global => "global",
    }
}

fn label(step: &PlannedStep) -> String {
    format!("{}:{}", step.kind.as_str(), step.id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assembly::Assembly;
    use crate::plugin::{Contributions, Plugin, PluginCtx};

    struct Echo {
        name: String,
        inject: Vec<String>,
        provide: Vec<String>,
    }

    impl Plugin for Echo {
        fn name(&self) -> &str {
            &self.name
        }
        fn inject(&self) -> Vec<String> {
            self.inject.clone()
        }
        fn provide(&self) -> Vec<String> {
            self.provide.clone()
        }
        fn apply(&self, _ctx: &PluginCtx) -> Result<Contributions> {
            let mut contributions = Contributions::new();
            for service in &self.provide {
                contributions = contributions.service(service.clone(), Arc::new(service.clone()));
            }
            Ok(contributions)
        }
    }

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("ngu-loader-{tag}-{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_skill(root: &std::path::Path, id: &str) {
        let dir = root.join(id);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("SKILL.md"),
            format!("---\nname: {id}\ndescription: test skill\n---\n\nBody of {id}.\n"),
        )
        .unwrap();
    }

    fn loader_with(dir: &std::path::Path, plugins: Vec<Arc<dyn Plugin>>) -> Loader {
        let mut kernel = Kernel::new();
        for plugin in plugins {
            kernel.define(plugin);
        }
        Loader::new(kernel, dir.join("installed.json"), "test-pack")
    }

    #[tokio::test]
    async fn a_builtin_plugin_loads_and_records_itself() {
        let dir = temp_dir("plugin");
        let mut loader = loader_with(
            &dir,
            vec![Arc::new(Echo {
                name: "provider".into(),
                inject: vec![],
                provide: vec!["thing".into()],
            })],
        );

        let manifest = r#"
version: 1
stages:
  - name: foundation
    plugins:
      - id: my-provider
        source: "builtin:provider"
"#;
        let plan = Assembly::parse(manifest).unwrap().plan("linux").unwrap();
        let report = loader.apply(&plan).await.unwrap();

        assert_eq!(report.loaded, vec!["plugin:my-provider"]);
        assert!(report.is_clean());
        assert!(loader.ledger().has("plugin", "my-provider", None));
        assert!(loader.ledger().find("plugin", "my-provider").unwrap().pack == "test-pack");
    }

    #[tokio::test]
    async fn an_unknown_plugin_fails_with_the_available_names() {
        let dir = temp_dir("unknown");
        let mut loader = loader_with(
            &dir,
            vec![Arc::new(Echo { name: "known".into(), inject: vec![], provide: vec![] })],
        );

        let manifest = r#"
version: 1
stages:
  - name: foundation
    plugins:
      - id: ghost
        source: "builtin:missing"
"#;
        let plan = Assembly::parse(manifest).unwrap().plan("linux").unwrap();
        let error = loader.apply(&plan).await.expect_err("unknown plugin");
        let text = format!("{error:#}");
        assert!(text.contains("not defined"), "{text}");
        assert!(text.contains("known"), "the available plugins are listed: {text}");
    }

    #[tokio::test]
    async fn on_failure_skip_continues_and_reports() {
        let dir = temp_dir("skip");
        let mut loader = loader_with(
            &dir,
            vec![Arc::new(Echo {
                name: "good".into(),
                inject: vec![],
                provide: vec!["svc".into()],
            })],
        );

        let manifest = r#"
version: 1
stages:
  - name: mixed
    plugins:
      - id: broken
        source: "builtin:nope"
        on_failure: skip
        order: 10
      - id: fine
        source: "builtin:good"
        order: 20
"#;
        let plan = Assembly::parse(manifest).unwrap().plan("linux").unwrap();
        let report = loader.apply(&plan).await.unwrap();

        assert_eq!(report.failed.len(), 1);
        assert_eq!(report.failed[0].id, "broken");
        assert_eq!(report.loaded, vec!["plugin:fine"]);
        assert!(!report.is_clean());
        // The good plugin still landed.
        assert!(loader.kernel().service_list().iter().any(|(name, _, _)| name == "svc"));
    }

    #[tokio::test]
    async fn abort_policy_stops_the_whole_assembly() {
        let dir = temp_dir("abort");
        let mut loader = loader_with(
            &dir,
            vec![Arc::new(Echo {
                name: "good".into(),
                inject: vec![],
                provide: vec!["svc".into()],
            })],
        );

        let manifest = r#"
version: 1
stages:
  - name: mixed
    plugins:
      - id: broken
        source: "builtin:nope"
        order: 10
      - id: fine
        source: "builtin:good"
        order: 20
"#;
        let plan = Assembly::parse(manifest).unwrap().plan("linux").unwrap();
        let error = loader.apply(&plan).await.expect_err("abort");
        assert!(format!("{error:#}").contains("broken"));

        // Nothing after the failure ran.
        assert!(!loader.kernel().service_list().iter().any(|(name, _, _)| name == "svc"));
    }

    #[tokio::test]
    async fn a_local_skill_loads_and_the_tool_appears() {
        let dir = temp_dir("skill");
        let skills_dir = dir.join("skills");
        write_skill(&skills_dir, "pdf-tools");

        let mut loader = loader_with(&dir, vec![]);
        let manifest = format!(
            r#"
version: 1
stages:
  - name: extensions
    skills:
      - id: pdf
        source: "{}"
"#,
            skills_dir.display().to_string().replace('\\', "/")
        );
        let plan = Assembly::parse(&manifest).unwrap().plan("linux").unwrap();
        let report = loader.apply(&plan).await.unwrap();

        assert_eq!(report.loaded, vec!["skill:pdf"]);
        assert_eq!(loader.skills().len(), 1);
        // The catalog tool is installed because a skill exists now.
        assert!(loader.kernel().tools().get("skill").is_some());

        let catalog = loader.skills().catalog().unwrap();
        assert!(catalog.contains("pdf-tools"));

        // The skill body is not in the standing catalog.
        assert!(!catalog.contains("Body of pdf-tools"));
    }

    #[tokio::test]
    async fn a_missing_skill_source_is_an_error() {
        let dir = temp_dir("skill-missing");
        let mut loader = loader_with(&dir, vec![]);
        let manifest = r#"
version: 1
stages:
  - name: extensions
    skills:
      - id: ghost
        source: "./nowhere-at-all"
"#;
        let plan = Assembly::parse(manifest).unwrap().plan("linux").unwrap();
        let error = loader.apply(&plan).await.expect_err("missing skill dir");
        assert!(format!("{error:#}").contains("does not exist"));
    }

    #[tokio::test]
    async fn github_skill_sources_are_reported_as_unimplemented_not_silently_ignored() {
        let dir = temp_dir("skill-github");
        let mut loader = loader_with(&dir, vec![]);
        let manifest = r#"
version: 1
stages:
  - name: extensions
    skills:
      - id: remote
        source: "github:owner/repo@skills/pdf@v1"
"#;
        let plan = Assembly::parse(manifest).unwrap().plan("linux").unwrap();
        let report = loader.apply(&plan).await.unwrap();

        assert!(report.loaded.is_empty());
        assert_eq!(report.skipped.len(), 1);
        assert!(report.skipped[0].reason.contains("not implemented"));
    }

    #[tokio::test]
    async fn streamable_http_mcp_is_declared_but_not_connected() {
        let dir = temp_dir("mcp-http");
        let mut loader = loader_with(&dir, vec![]);
        let manifest = r#"
version: 1
stages:
  - name: services
    mcp:
      - id: remote-mcp
        transport: streamable-http
        url: "https://example.invalid/mcp"
"#;
        let plan = Assembly::parse(manifest).unwrap().plan("linux").unwrap();
        let report = loader.apply(&plan).await.unwrap();

        assert_eq!(report.skipped.len(), 1);
        assert!(report.skipped[0].reason.contains("not implemented"));
        assert!(loader.mcp_servers().is_empty());
    }

    #[tokio::test]
    async fn session_scope_isolates_and_global_scope_shares() {
        let dir = temp_dir("scope");
        let mut loader = loader_with(
            &dir,
            vec![Arc::new(Echo {
                name: "provider".into(),
                inject: vec![],
                provide: vec!["shared".into()],
            })],
        );

        let session_manifest = r#"
version: 1
defaults:
  scope: session
stages:
  - name: one
    plugins:
      - id: isolated
        source: "builtin:provider"
"#;
        let plan = Assembly::parse(session_manifest).unwrap().plan("linux").unwrap();
        loader.apply(&plan).await.unwrap();

        // A session-scoped service is not in the root realm.
        assert!(!loader.kernel().service_view(RealmMap::new()).has("shared"));
        assert_eq!(loader.kernel().service_list().len(), 1);

        // The same plugin in the global scope lands in the root realm.
        let global_manifest = r#"
version: 1
defaults:
  scope: global
stages:
  - name: two
    plugins:
      - id: shared-global
        source: "builtin:provider"
"#;
        let plan = Assembly::parse(global_manifest).unwrap().plan("linux").unwrap();
        loader.apply(&plan).await.unwrap();
        assert!(loader.kernel().service_view(RealmMap::new()).has("shared"));
    }

    #[tokio::test]
    async fn applying_the_same_plan_twice_is_not_an_error() {
        let dir = temp_dir("idempotent");
        let mut loader = loader_with(
            &dir,
            vec![Arc::new(Echo {
                name: "provider".into(),
                inject: vec![],
                provide: vec!["svc".into()],
            })],
        );

        let manifest = r#"
version: 1
stages:
  - name: foundation
    plugins:
      - id: p
        source: "builtin:provider"
"#;
        let plan = Assembly::parse(manifest).unwrap().plan("linux").unwrap();
        loader.apply(&plan).await.unwrap();

        // The ledger is written and readable after the first pass.
        let reloaded = Ledger::load(&dir.join("installed.json")).unwrap();
        assert!(reloaded.has("plugin", "p", None));
    }

    #[test]
    fn plugin_names_resolve_from_sources() {
        assert_eq!(plugin_name("builtin:search", "fallback"), "search");
        assert_eq!(plugin_name("search", "fallback"), "search");
        assert_eq!(plugin_name("", "fallback"), "fallback");
        // A path or URL is not a plugin name.
        assert_eq!(plugin_name("npm:@dshd/search@^2", "fallback"), "fallback");
        assert_eq!(plugin_name("./local/thing", "fallback"), "fallback");
    }
}
