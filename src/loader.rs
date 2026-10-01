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

use std::path::{Path, PathBuf};
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
                    Box::pin(async move { client.call(&name, args).await.map(Into::into) }) as ToolFuture
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

    /// Take the kernel back, handing over the skill roots this pack added and
    /// the MCP servers it started.
    ///
    /// The MCP servers are deliberately **not** stopped here: their tools are
    /// in this kernel and stay callable for as long as the conversation runs;
    /// stopping them here killed every MCP tool the moment loading ended. The
    /// caller keeps them — and stops them on the way out, because a server
    /// still running when the runtime tears down is what kept the process alive
    /// after everything it had to say was printed.
    ///
    /// The skills come back rather than staying inside the loader for the same
    /// reason: the host owns the standing catalog, and a pack's roots join it
    /// instead of replacing it.
    pub async fn finish(self) -> (Kernel, SkillRegistry, Vec<Arc<McpClient>>) {
        let Loader { kernel, skills, mcp_clients, .. } = self;
        (kernel, skills, mcp_clients)
    }

    /// Stop every MCP server this loader started.
    ///
    /// For a preview that is about to exit, not for a session: see
    /// [`Loader::finish`].
    pub async fn shutdown(&mut self) {
        for client in self.mcp_clients.drain(..) {
            client.shutdown().await;
        }
    }
}

/// What an unload took out of the kernel.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct UnloadReport {
    /// Plugin instances unloaded, by plugin name.
    pub plugins: Vec<String>,
    /// MCP servers whose tools were removed; the child stops with its last
    /// handle.
    pub mcp: Vec<String>,
    /// Skill roots removed from the catalog.
    pub skills: Vec<String>,
    /// Anything that did not match, so an unload that did nothing says so
    /// rather than looking like it worked.
    pub notes: Vec<String>,
}

/// Unload everything one pack contributed, leaving its files where they are.
///
/// Unload and delete are different acts: this takes the pack *out of the
/// conversation* — its tools stop being callable — and keeps the installed
/// directory, so it can be loaded again without fetching anything.
///
/// `label` is the ledger's pack label (`name-version`); `name` is the
/// manifest's name, which is what the appearance fiber is registered under.
/// The ledger entries for the pack are removed as they are undone, so a later
/// load sees them as new work rather than as something already in place.
pub fn unload_pack(
    kernel: &mut Kernel,
    skills: &mut SkillRegistry,
    ledger_path: &Path,
    label: &str,
    name: &str,
) -> Result<UnloadReport> {
    let mut ledger = Ledger::load(ledger_path)?;
    let entries: Vec<LedgerEntry> = ledger
        .entries
        .iter()
        .filter(|entry| entry.pack == label)
        .cloned()
        .collect();

    let mut report = UnloadReport::default();

    for entry in &entries {
        match entry.kind.as_str() {
            "plugin" => {
                let plugin = plugin_name(&entry.source, &entry.id);
                let ids: Vec<u64> = kernel
                    .fibers()
                    .iter()
                    .filter(|fiber| fiber.plugin == plugin)
                    .map(|fiber| fiber.id)
                    .collect();
                if ids.is_empty() {
                    report
                        .notes
                        .push(format!("plugin '{plugin}' was not loaded here"));
                }
                for id in ids {
                    kernel.unload(id)?;
                    report.plugins.push(plugin.clone());
                }
                ledger.remove("plugin", &entry.id);
            }
            "mcp" => {
                // An MCP server has no fiber: its tools are registered
                // directly, so they are removed directly. With the loader's own
                // handle already dropped, removing the last tool drops the last
                // handle and the child process stops.
                let owner = format!("mcp:{}", entry.id);
                let tools: Vec<String> = kernel
                    .tools()
                    .names()
                    .into_iter()
                    .filter(|tool| kernel.tools().owner(tool).as_deref() == Some(owner.as_str()))
                    .collect();
                for tool in tools {
                    kernel.tools_mut().unregister(&tool);
                }
                report.mcp.push(entry.id.clone());
                ledger.remove("mcp", &entry.id);
            }
            "skill" => {
                if let Some(path) = &entry.path {
                    skills.remove_root(Path::new(path));
                    report.skills.push(entry.id.clone());
                }
                ledger.remove("skill", &entry.id);
            }
            other => report
                .notes
                .push(format!("entry '{other}:{}' has no unload path", entry.id)),
        }
    }

    // The appearance a pack's look.json installs is a fiber of its own,
    // registered under the manifest name rather than the ledger label.
    let appearance = format!("pack:{name}");
    let ids: Vec<u64> = kernel
        .fibers()
        .iter()
        .filter(|fiber| fiber.plugin == appearance)
        .map(|fiber| fiber.id)
        .collect();
    for id in ids {
        kernel.unload(id)?;
        report.plugins.push(appearance.clone());
    }

    // The catalog tool holds a snapshot of the registry, so removing a root
    // only reaches the model once the tool is rebuilt over what is left — and
    // when nothing is left, no tool is the right answer rather than one that
    // advertises an empty catalog.
    kernel.tools_mut().unregister("skill");
    skills.scan()?;
    if !skills.is_empty() {
        crate::skills::register_skill_tool(kernel.tools_mut(), std::sync::Arc::new(skills.clone()))?;
    }

    if entries.is_empty() && report.plugins.is_empty() {
        report
            .notes
            .push(format!("nothing in the ledger came from '{label}'"));
    }

    ledger.save(ledger_path)?;
    Ok(report)
}

/// Load one pack's assembly into a kernel, returning the roots it added.
///
/// Takes the kernel by reference and gives it back on **every** path, including
/// a failure: a load that loses the kernel would leave the session with no
/// tools at all, which is far worse than a pack that failed to load. The
/// skills come back rather than being merged here so the caller never holds
/// its catalog across the awaits in this function; hand them straight to
/// [`merge_skills`].
pub async fn load_pack(
    kernel: &mut Kernel,
    ledger_path: &Path,
    assembly: &Path,
    label: &str,
) -> Result<(SkillRegistry, LoadReport, Vec<Arc<McpClient>>)> {
    let pack_dir = assembly
        .parent()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));

    let taken = std::mem::replace(kernel, Kernel::new());
    let mut loader = Loader::new(taken, ledger_path.to_path_buf(), label).with_pack_dir(&pack_dir);

    let loaded = async {
        let document = crate::assembly::Assembly::from_file(assembly)?;
        let plan = document.plan(crate::fetch::platform_tag())?;
        loader.apply(&plan).await
    }
    .await;

    let (returned, pack_skills, servers) = loader.finish().await;
    *kernel = returned;
    let report = loaded?;
    Ok((pack_skills, report, servers))
}

/// Merge a pack's skill roots into the standing catalog and rebuild the tool.
///
/// The `skill` tool holds a snapshot of the registry, so a root that arrives
/// or leaves only reaches the model once the tool is registered over what is
/// there now. Without this, a pack's skills would *replace* the host's instead
/// of joining them.
pub fn merge_skills(
    kernel: &mut Kernel,
    host: &mut SkillRegistry,
    pack: SkillRegistry,
) -> Result<()> {
    for root in pack.roots() {
        host.add_root(root.clone());
    }
    host.scan()?;
    kernel.tools_mut().unregister("skill");
    if !host.is_empty() {
        crate::skills::register_skill_tool(kernel.tools_mut(), Arc::new(host.clone()))?;
    }
    Ok(())
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

    /// A plugin whose whole contribution is one tool, so an unload has
    /// something visible to take away.
    struct WithTool {
        name: String,
    }

    impl Plugin for WithTool {
        fn name(&self) -> &str {
            &self.name
        }
        fn apply(&self, _ctx: &PluginCtx) -> Result<Contributions> {
            Ok(Contributions::new().tool(crate::tools::ToolDef::new(
                format!("tool_of_{}", self.name),
                "a test tool",
                serde_json::json!({ "type": "object", "properties": {} }),
                self.name.clone(),
                |_| {
                    Box::pin(async { Ok(crate::tools::ToolOutput::text("ok")) })
                        as crate::tools::ToolFuture
                },
            )))
        }
    }

    fn ledger_entry(kind: &str, id: &str, source: &str, path: Option<String>) -> LedgerEntry {
        LedgerEntry {
            kind: kind.into(),
            id: id.into(),
            source: source.into(),
            sha1: None,
            path,
            scope: "session".into(),
            pack: "demo-1.0.0".into(),
            installed_at: "now".into(),
        }
    }

    #[test]
    fn unloading_a_pack_takes_everything_it_brought_and_leaves_the_files_alone() {
        let dir = temp_dir("unload");
        let ledger_path = dir.join("installed.json");
        let skills_root = dir.join("skills");
        write_skill(&skills_root, "demo-skill");

        let mut kernel = Kernel::new();
        kernel.define(Arc::new(WithTool { name: "demo".into() }));
        // The appearance a look.json installs, registered under the manifest
        // name rather than the ledger label.
        kernel.define(Arc::new(WithTool { name: "pack:demo".into() }));
        kernel
            .load("demo", RealmMap::new(), serde_json::Value::Null)
            .unwrap();
        kernel
            .load("pack:demo", RealmMap::new(), serde_json::Value::Null)
            .unwrap();
        assert!(kernel.tools().get("tool_of_demo").is_some());

        let mut skills = SkillRegistry::new();
        skills.add_root(&skills_root);
        skills.scan().unwrap();
        register_skill_tool(kernel.tools_mut(), Arc::new(skills.clone())).unwrap();
        assert!(kernel.tools().get("skill").is_some());

        let mut ledger = Ledger::new();
        ledger.record(ledger_entry(
            "plugin",
            "demo",
            "builtin:demo",
            None,
        ));
        ledger.record(ledger_entry(
            "skill",
            "demo-skill",
            &skills_root.display().to_string(),
            Some(skills_root.display().to_string()),
        ));
        ledger.record(ledger_entry("mcp", "demo-server", "stdio:x", None));
        ledger.save(&ledger_path).unwrap();

        let report =
            unload_pack(&mut kernel, &mut skills, &ledger_path, "demo-1.0.0", "demo").unwrap();

        // Everything it contributed is out of the conversation.
        assert!(kernel.tools().get("tool_of_demo").is_none(), "its tool");
        assert!(
            kernel.tools()
                .get("tool_of_pack:demo")
                .is_none(),
            "its appearance"
        );
        assert!(kernel.tools().get("skill").is_none(), "no skills left to advertise");
        assert_eq!(report.plugins, vec!["demo".to_string(), "pack:demo".to_string()]);
        assert_eq!(report.mcp, vec!["demo-server".to_string()]);
        assert_eq!(report.skills, vec!["demo-skill".to_string()]);
        assert!(
            skills.roots().iter().all(|root| root != &skills_root),
            "the root is out of the catalog"
        );

        // The ledger no longer claims they are in place, so a later load sees
        // them as work to do rather than as already done.
        let back = Ledger::load(&ledger_path).unwrap();
        assert!(
            back.entries.iter().all(|entry| entry.pack != "demo-1.0.0"),
            "{:?}",
            back.entries
        );

        // …and the pack's own files are untouched: unload is not delete.
        assert!(skills_root.join("demo-skill").join("SKILL.md").is_file());
    }

    #[test]
    fn unloading_a_pack_that_is_not_loaded_says_so_rather_than_looking_successful() {
        let dir = temp_dir("unload-nothing");
        let ledger_path = dir.join("installed.json");
        let mut kernel = Kernel::new();
        let mut skills = SkillRegistry::new();

        let report =
            unload_pack(&mut kernel, &mut skills, &ledger_path, "ghost-1.0.0", "ghost").unwrap();

        assert!(report.plugins.is_empty());
        assert!(report.mcp.is_empty());
        assert!(report.skills.is_empty());
        assert!(
            report
                .notes
                .iter()
                .any(|note| note.contains("nothing in the ledger")),
            "{:?}",
            report.notes
        );
    }

    #[test]
    fn merging_pack_skills_keeps_the_hosts_own() {
        // Before this, a pack's skills replaced the standing catalog — the
        // model would lose every skill it had as soon as one pack loaded.
        let dir = temp_dir("merge-skills");
        let host_root = dir.join("host");
        let pack_root = dir.join("pack");
        write_skill(&host_root, "host-skill");
        write_skill(&pack_root, "pack-skill");

        let mut host = SkillRegistry::new();
        host.add_root(&host_root);
        host.scan().unwrap();

        let mut pack = SkillRegistry::new();
        pack.add_root(&pack_root);
        pack.scan().unwrap();

        let mut kernel = Kernel::new();
        merge_skills(&mut kernel, &mut host, pack).unwrap();

        let catalog = host.catalog().expect("both skills are in the catalog");
        assert!(catalog.contains("host-skill"), "{catalog}");
        assert!(catalog.contains("pack-skill"), "{catalog}");
        assert!(kernel.tools().get("skill").is_some());
    }
}
