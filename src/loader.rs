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

    /// The pack these steps come from, by manifest name.
    ///
    /// `None` when the directory has no manifest (a bare assembly): nothing
    /// points at a pack because there is none.
    fn origin(&self) -> Option<String> {
        self.pack_dir
            .as_ref()
            .and_then(|dir| crate::pack::read_manifest(dir).ok())
            .map(|manifest| manifest.name)
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
        // A rescan is usually triggered because a skill appeared or vanished,
        // so the paths it could not use are exactly what the user needs to see;
        // the return value alone would report "0 loaded" with no reason.
        if !report.skipped.is_empty() || !report.collisions.is_empty() {
            eprintln!("[skills] {}", report.summary());
        }
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
        // Which pack asked, for the panel: the manifest beside the assembly
        // is the same source `pack:` naming uses — not the label, which would
        // have to be split on a dash.
        self.kernel
            .load_owned(&name, realm, step.config.clone(), self.origin())?;

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

        let spec = match entry.transport {
            Transport::StreamableHttp => {
                // One URL carries every message; a declaration without one
                // cannot be connected and says so instead of being skipped.
                let url = entry.url.clone().ok_or_else(|| {
                    anyhow!("mcp '{}' uses streamable-http but declares no url", step.id)
                })?;
                McpSpec {
                    id: step.id.clone(),
                    command: String::new(),
                    args: Vec::new(),
                    env: std::collections::HashMap::new(),
                    cwd: None,
                    timeout_ms: entry.timeout_ms.unwrap_or(60_000),
                    url: Some(url),
                    headers: entry.headers.clone(),
                }
            }
            Transport::Stdio => {
                let command = entry.command.clone().ok_or_else(|| {
                    anyhow!("mcp '{}' uses stdio but declares no command", step.id)
                })?;
                McpSpec {
                    id: step.id.clone(),
                    command,
                    args: entry.args.clone(),
                    env: entry
                        .env
                        .iter()
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .collect(),
                    cwd: entry
                        .base
                        .source
                        .is_empty()
                        .then(|| self.pack_dir.as_ref().map(|d| d.display().to_string()))
                        .flatten(),
                    timeout_ms: entry.timeout_ms.unwrap_or(60_000),
                    url: None,
                    headers: std::collections::BTreeMap::new(),
                }
            }
        };

        let client = Arc::new(McpClient::connect(&spec).await?);
        let owner = format!("mcp:{}", step.id);

        // The server is already registered under this owner (another
        // conversation loaded the same pack). Its tools are live, so this new
        // process is redundant: drop it and reuse the existing registration.
        let already: Vec<String> = client.tools().iter().map(|t| client.exposed_name(&t.name)).collect();
        if !already.is_empty()
            && already
                .iter()
                .all(|name| self.kernel.tools().owner(name) == Some(owner.as_str()))
        {
            client.shutdown().await;
            return Ok(Outcome::Reused);
        }

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
        // Keep a way back to this process that does not depend on the tools: an
        // unload has the ledger, not the handles, and this is what lets it stop
        // a server the ledger named instead of trusting the last drop to do it.
        crate::mcp::track(&client);
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

        // A manifest that says `./skills` means the same directory as `skills`,
        // but the two spellings are different strings — and the ledger stores the
        // string. Left alone, a diagnostic reads `...\packs\x\./skills`, which
        // looks like a bug in the path rather than a style in the manifest, and
        // the same root can be added twice under two spellings.
        let candidate = tidy(candidate);

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
    /// MCP servers taken out of the conversation: their tools are gone, and
    /// their processes were stopped here rather than left to whatever drops
    /// last. A server that could not be reached is in [`Self::notes`], with the
    /// reason for it.
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
                    // Pack-plane unload is deliberately global and
                    // ledger-driven: a pack's resources are not per-session
                    // references, so this tears the fiber down for every session
                    // at once. `Kernel::release` is the other shape — a teardown
                    // counted against the sessions still holding a reference —
                    // and it is not what a ledger row means.
                    kernel.unload(id)?;
                    report.plugins.push(plugin.clone());
                }
                ledger.remove("plugin", &entry.id);
            }
            "mcp" => {
                // An MCP server has no fiber: its tools are registered
                // directly, so they are removed directly. Removing them drops
                // *a* handle on the client — not the last one, whatever the
                // old comment here assumed: whoever started the server keeps
                // one, and a call in flight keeps another. So the process is
                // stopped here, explicitly, through the bounded kill path, and
                // what it could not reach is reported rather than hoped about.
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
                let closed = crate::mcp::close_server(&entry.id);
                for failure in &closed.failed {
                    // A process this could not stop is said out loud, where the
                    // user sees the unload report, rather than left to be found
                    // later in a process list.
                    report
                        .notes
                        .push(format!("an MCP server was not stopped — {failure}"));
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
        // Same rule as the ledger rows above, for the appearance fiber: pack
        // unload is global and ledger-driven, not reference-counted per session,
        // so this one teardown takes the look away from every session at once.
        // (`Kernel::release` is the per-session form, and a ledger row is not a
        // session's claim.)
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
/// A path with its `.` components dropped.
///
/// Only `.` is removed. `..` is left alone on purpose: resolving it is a decision
/// about where a pack is allowed to point, and that decision is not this
/// function's to make — a `..` that escapes the pack directory has to stay
/// visible for whatever checks it rather than being silently folded here.
fn tidy(path: PathBuf) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            std::path::Component::CurDir => {}
            other => out.push(other.as_os_str()),
        }
    }
    out
}

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
    async fn steps_record_the_pack_they_came_from() {
        // The panel pairs a plugin with its pack (`delegate ← subagent`), and
        // the manifest beside the assembly is the same file `pack:` naming
        // uses — not the `name-version` label, which would have to be split.
        let dir = temp_dir("origin");
        let mut loader = loader_with(
            &dir,
            vec![Arc::new(Echo {
                name: "provider".into(),
                inject: vec![],
                provide: vec!["thing".into()],
            })],
        )
        .with_pack_dir(&dir);
        std::fs::write(
            dir.join("dsh.index.json"),
            r#"{"formatVersion":1,"game":"nguruvilu","name":"origin-pack","versionId":"1.0.0","license":"MIT","kernelVersion":"0.1.0","dependencies":{"nguruvilu":">=0.1.0"}}"#,
        )
        .unwrap();

        let manifest = r#"
version: 1
stages:
  - name: foundation
    plugins:
      - id: my-provider
        source: "builtin:provider"
"#;
        let plan = Assembly::parse(manifest).unwrap().plan("linux").unwrap();
        loader.apply(&plan).await.unwrap();

        assert_eq!(
            loader
                .kernel()
                .plugin_origins()
                .get("provider")
                .map(String::as_str),
            Some("origin-pack"),
            "the plugin is pointed at its pack"
        );
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
    async fn a_manifest_path_with_a_dot_component_is_recorded_without_it() {
        // A pack ships `./skills`; the ledger must not end up holding
        // `...\pack\./skills`, both because the string is what every diagnostic
        // prints and because the same root added twice under two spellings would
        // scan the same skills twice.
        let dir = temp_dir("skill-dot");
        let pack = dir.join("pack");
        write_skill(&pack.join("skills"), "pdf-tools");
        let mut loader = loader_with(&dir, vec![]).with_pack_dir(&pack);
        let manifest = r#"
version: 1
stages:
  - name: extensions
    skills:
      - id: pdf
        source: "./skills"
"#;
        let plan = Assembly::parse(manifest).unwrap().plan("linux").unwrap();
        loader.apply(&plan).await.unwrap();

        let recorded = loader
            .ledger()
            .find("skill", "pdf")
            .and_then(|entry| entry.path.clone())
            .expect("the skill was recorded with its path");
        assert!(
            !recorded.contains("./") && !recorded.contains(".\\"),
            "the recorded path keeps the manifest's dot component: {recorded}"
        );
        assert_eq!(
            recorded,
            pack.join("skills").display().to_string(),
            "the recorded path is the one the scan actually used"
        );
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
    async fn streamable_http_without_a_url_fails_with_the_reason() {
        let dir = temp_dir("mcp-http-no-url");
        let mut loader = loader_with(&dir, vec![]);
        let manifest = r#"
version: 1
defaults:
  on_failure: skip
stages:
  - name: services
    mcp:
      - id: remote-mcp
        transport: streamable-http
"#;
        let plan = Assembly::parse(manifest).unwrap().plan("linux").unwrap();
        let report = loader.apply(&plan).await.unwrap();

        assert_eq!(report.failed.len(), 1);
        assert!(
            report.failed[0].error.contains("declares no url"),
            "{}",
            report.failed[0].error
        );
        assert!(loader.mcp_servers().is_empty());
    }

    #[tokio::test]
    async fn streamable_http_with_a_url_is_connected_not_skipped() {
        let dir = temp_dir("mcp-http");
        let mut loader = loader_with(&dir, vec![]);
        // Port 9 on loopback: nothing listens there, so the connection fails
        // fast and locally — the point is *which* failure: a connection
        // attempt, never the old "not implemented" skip.
        let manifest = r#"
version: 1
defaults:
  on_failure: skip
stages:
  - name: services
    mcp:
      - id: remote-mcp
        transport: streamable-http
        url: "http://127.0.0.1:9/mcp"
"#;
        let plan = Assembly::parse(manifest).unwrap().plan("linux").unwrap();
        let report = loader.apply(&plan).await.unwrap();

        assert_eq!(report.failed.len(), 1, "{:?}", report.failed);
        assert!(!report.failed[0].error.contains("not implemented"));
        assert!(
            report.failed[0].error.contains("MCP server"),
            "{}",
            report.failed[0].error
        );
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

    /// A server process that sits there for thirty seconds, the shape of the
    /// one that used to outlive every unload.
    #[cfg(windows)]
    fn stub_server() -> (tokio::process::Child, u32) {
        let child = tokio::process::Command::new("cmd")
            .args(["/c", "ping -n 30 127.0.0.1"])
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .spawn()
            .expect("spawning the stub server");
        let pid = child.id().expect("a running child has a process id");
        (child, pid)
    }

    /// Whether the operating system lists this process right now.
    ///
    /// `None` when the question could not be asked at all. Under load `tasklist`
    /// can fail to start, and "the tool that answers failed" is not the same
    /// answer as "the process is gone": a caller waiting for a process to appear
    /// must not read a broken query as a completed disappearance, or this test
    /// fails for a reason that has nothing to do with the code it checks.
    #[cfg(windows)]
    fn listed(pid: u32) -> Option<bool> {
        let output = std::process::Command::new("tasklist")
            .args(["/NH", "/FI", &format!("PID eq {pid}")])
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        Some(String::from_utf8_lossy(&output.stdout).contains(&pid.to_string()))
    }

    /// Wait for the process list to agree with what was asked of it: a spawn is
    /// not instant, and neither is a kill.
    ///
    /// The window is wide because the machine running the suite may be running
    /// four tests at once, each spawning its own children; a tight bound here
    /// measures the test harness rather than the behaviour under test.
    #[cfg(windows)]
    fn wait_for(pid: u32, want_running: bool) -> bool {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        loop {
            if listed(pid) == Some(want_running) {
                return true;
            }
            if std::time::Instant::now() >= deadline {
                return false;
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
    }

    #[cfg(windows)]
    #[tokio::test]
    async fn unloading_a_pack_stops_the_server_it_named_rather_than_the_last_handle() {
        // The old unload removed the tools and called that enough, on the
        // assumption that the tools were the only handles. They are not: the
        // caller that started the server keeps one, so the process kept running
        // and every load/unload cycle added another. This is that leak, pinned.
        let dir = temp_dir("unload-mcp");
        let ledger_path = dir.join("installed.json");
        let mut kernel = Kernel::new();
        let mut skills = SkillRegistry::new();

        let (mut child, pid) = stub_server();
        // Live, not listed: this test needs a server that is running, and asking
        // the child is the direct answer. Whether the operating system's process
        // table shows it is a second question that `tasklist` answers slowly
        // under a parallel suite — measuring that would make the test about the
        // machine instead of about the unload.
        assert!(
            child.try_wait().expect("asking about the stub").is_none(),
            "the stub server is up before the unload"
        );
        let client = Arc::new(McpClient::from_child_for_tests(child, "demo-server"));

        // One tool, holding one handle, registered exactly the way the loader
        // registers a server's tools.
        let tool_client = Arc::clone(&client);
        kernel
            .tools_mut()
            .register(
                ToolDef::new(
                    "mcp__demo_server__ping",
                    "a tool from the stub server",
                    serde_json::json!({ "type": "object", "properties": {} }),
                    "mcp:demo-server",
                    move |_| {
                        let client = Arc::clone(&tool_client);
                        Box::pin(async move { client.call("ping", serde_json::json!({})).await })
                            as ToolFuture
                    },
                ),
                ConflictPolicy::Error,
            )
            .unwrap();
        crate::mcp::track(&client);

        let mut ledger = Ledger::new();
        ledger.record(ledger_entry("mcp", "demo-server", "stdio:x", None));
        ledger.save(&ledger_path).unwrap();

        // The handle the caller keeps. With this alive, no drop of the tool can
        // be what ends the process — so the unload has to end it by name.
        let held = Arc::clone(&client);

        let report =
            unload_pack(&mut kernel, &mut skills, &ledger_path, "demo-1.0.0", "demo").unwrap();

        assert_eq!(report.mcp, vec!["demo-server".to_string()]);
        assert!(kernel.tools().get("mcp__demo_server__ping").is_none(), "its tool");
        assert!(report.notes.is_empty(), "{:?}", report.notes);
        assert!(
            wait_for(pid, false),
            "the server stopped with the unload, not after it"
        );
        assert_eq!(Arc::strong_count(&held), 2, "and it stopped while a handle was still held");

        // The second unload is the idempotent case: the entry is out of the
        // ledger and the client is out of the registry, so there is nothing to
        // wait on and it must say so at once.
        let started = std::time::Instant::now();
        let again =
            unload_pack(&mut kernel, &mut skills, &ledger_path, "demo-1.0.0", "demo").unwrap();
        assert!(again.mcp.is_empty(), "{:?}", again.mcp);
        assert!(
            started.elapsed() < std::time::Duration::from_secs(1),
            "a repeat unload waited {:?}",
            started.elapsed()
        );

        drop(held);
        drop(client);
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
