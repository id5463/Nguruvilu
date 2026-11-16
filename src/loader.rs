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
    /// Plugin processes this loader started, kept alive for as long as it is.
    ///
    /// Held rather than dropped because dropping one stops it: the process is
    /// the capability, and a plugin whose process ended contributes nothing.
    processes: Vec<Arc<crate::protocol::Process>>,
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
            processes: Vec::new(),
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
        // `plugin:<path>` is a process. This is the ordinary way a capability
        // arrives now: the pack ships an executable (or the source for one), the
        // kernel starts it once, and `src/protocol.rs` defines the four
        // operations they exchange.
        //
        // It is checked before anything else because a process plugin has no
        // entry in `kernel.define(...)`: the whole point is that the kernel was
        // not rebuilt to know about it.
        if let Some(raw) = step.source.strip_prefix("plugin:") {
            return self.load_process_plugin(step, raw);
        }
        // `dylib:<path>` defines the plugin on the spot by loading a shared
        // library. Kept for compatibility; `plugin:` is the recommended form and
        // the one every shipped pack uses.
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
        let fiber = self
            .kernel
            .load_owned(&name, realm, step.config.clone(), self.origin())?;

        // A plugin whose dependencies are not present loads without error and
        // contributes nothing — `activate` returns early on `Epoch::Inactive`.
        // That is a silent failure with a misleading report: the caller is told
        // "loaded", the pack looks installed, and the tool the pack promised
        // never appears. It cost an afternoon once: `ngu apply packs/starter`
        // reported `plugin:delegate` loaded while the tool table held no
        // `delegate`, because that entrance had not published the `llm` service
        // every `delegate` fiber injects.
        //
        // Reporting `Pending` with the missing names is the honest outcome: the
        // pack is not wrong, and the entrance that forgot a dependency is named.
        if let Some(fiber) = self.kernel.fiber(fiber) {
            if fiber.state == crate::plugin::FiberState::Pending {
                let view = self.kernel.service_view(fiber.realm.clone());
                let missing: Vec<&str> = fiber
                    .inject
                    .iter()
                    .filter(|dependency| !view.has(dependency))
                    .map(|dependency| dependency.as_str())
                    .collect();
                return Ok(Outcome::Skipped(format!(
                    "plugin '{name}' is defined but waiting for {} — nothing of it is                      active until that is published",
                    if missing.is_empty() {
                        "a service this host did not install".to_string()
                    } else {
                        format!("service(s) {}", missing.join(", "))
                    }
                )));
            }
        }

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

    /// Where a stdio MCP server should run from.
///
/// A server that ships inside the pack names its own files by relative path
/// (`args: ["files/ticket-mcp.mjs"]`), and the child process inherits the *host's*
/// working directory unless told otherwise — which is wherever `ngu` happened to
/// start, so the script is simply not there and the server dies on contact with
/// "closed the connection". So: when the command or one of the arguments resolves
/// to a file inside the pack, the pack directory is the working directory.
///
/// A server pulled from npm or git is left exactly as it was: its command is
/// resolved elsewhere, and moving its cwd into the pack could silently change
/// where it writes profiles and temp files.
fn pack_cwd(pack_dir: Option<&std::path::Path>, command: &str, args: &[String]) -> Option<String> {
    let directory = pack_dir?;
    let inside = |candidate: &str| {
        let path = std::path::Path::new(candidate);
        // An absolute path already says where it lives; only a relative one that
        // actually resolves inside the pack means "this server came with the pack".
        !path.is_absolute() && directory.join(path).is_file()
    };
    if inside(command) || args.iter().any(|arg| inside(arg)) {
        Some(directory.display().to_string())
    } else {
        None
    }
}

/// Load an MCP server and register its tools.
    async fn load_mcp(&mut self, step: &PlannedStep) -> Result<Outcome> {        let entry = step
            .mcp
            .as_ref()
            .ok_or_else(|| anyhow!("mcp step '{}' has no server definition", step.id))?;

        // Handed to a proxy: this server is still declared here, it just is not
        // started here. The pack keeps saying what it provides, the proxy
        // decides how much of it the model sees.
        //
        // Nothing is registered, so the ledger gets no `mcp` entry either — an
        // entry would make unload try to unregister tools that were never
        // registered. What the pack *does* get is a line in the proxy's target
        // list, written below, and that is what has to be withdrawn on unload.
        if let Some(proxy_id) = entry.via_proxy.as_deref() {
            let line = proxy_target_line(step, entry, self.pack_dir.as_deref())?;
            crate::loader::register_proxy_target(proxy_id, line);
            return Ok(Outcome::Skipped(format!(
                "handed to MCP proxy '{proxy_id}'"
            )));
        }

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
                    timeout_ms: entry.timeout_ms,
                    url: Some(url),
                    headers: entry.headers.clone(),
                }
            }
            Transport::Stdio => {
                let command = entry.command.clone().ok_or_else(|| {
                    anyhow!("mcp '{}' uses stdio but declares no command", step.id)
                })?;
                // Decided before the struct literal: `command` is moved into the
                // spec below, and field initialisers run in written order.
                let cwd = Self::pack_cwd(self.pack_dir.as_deref(), &command, &entry.args);
                // A pack that re-invokes the CLI cannot know where the host
                // binary lives: its MCP child inherits the *host's* PATH, which
                // need not contain `ngu` at all (a desktop build run straight
                // from a target directory is the normal case). Measured: the
                // ticket pack's worker launch burned six steps hunting for the
                // executable and never started. So every stdio MCP server gets
                // `NGU_EXE` — the host's own path — unless it declared its own.
                let mut env = mcp_env(&entry.env);
                // A proxy starts the servers its packs handed to it. The list is
                // built during the same load pass that reaches this line, so it
                // is complete by the time the proxy's own entry is loaded — packs
                // sort alphabetically and `mcp-proxy` sorts before the packs that
                // hand it work only by luck, which is why the list is also
                // re-read at spawn time by the proxy itself.
                let handed = proxy_targets_for(&step.id);
                if !handed.is_empty() {
                    env.insert("NGU_MCP_TARGETS".to_string(), handed.join("
"));
                }
                McpSpec {
                    id: step.id.clone(),
                    command,
                    args: entry.args.clone(),
                    env,
                    cwd,
                    timeout_ms: entry.timeout_ms,
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

            // `Override`, not `Error`: a pack is allowed to replace a kernel
            // tool. Refusing made the four base tools and the job tools
            // unreachable property — a pack that wanted a stricter `bash` had no
            // way to say so, and the answer was "edit the kernel".
            if let Err(error) = self.kernel.tools_mut().register(def, ConflictPolicy::Override) {
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
    /// Load a plugin that runs as a process.
    ///
    /// Three steps, in this order, because each makes the next cheaper:
    ///
    /// 1. **Build it if the pack ships source and the binary is missing.**
    /// 2. **Start it.** One spawn; the process stays for the life of the loader.
    ///    Measured: 45 ms to start against 0.25 ms per call, so reconnecting per
    ///    call would make the protocol 180x slower than the transport it uses.
    /// 3. **Register what it declared.** Each tool becomes an ordinary entry in
    ///    the tool table, owned by `plugin:<id>`, so unloading removes exactly
    ///    what this plugin added.
    fn load_process_plugin(&mut self, step: &PlannedStep, raw: &str) -> Result<Outcome> {
        let pack_dir = self.pack_dir.clone();
        let path = self.resolve_process_path(raw, pack_dir.as_deref());

        let build_spec = step.build.clone();
        let path = match path {
            Some(path) => path,
            None => match build_spec {
                Some(build) => self.build_plugin(step, &build)?,
                None => {
                    return Err(anyhow!(
                        "plugin '{}' names '{}', which is not there, and the pack \
                         declares no `build` to produce it",
                        step.id,
                        raw
                    ))
                }
            },
        };

        let mut spec = crate::protocol::Spec::new(step.id.clone(), path.display().to_string());
        if let Some(dir) = &pack_dir {
            // A plugin that came with a pack runs inside it, the same rule MCP
            // servers already follow: a relative path in its own arguments then
            // means what the pack author meant.
            spec = spec.in_dir(dir.display().to_string());
        }
        let process = start_process_with_state_for(&spec)
            .with_context(|| format!("loading process plugin '{}'", step.id))?;
        let process = Arc::new(process);

        let owner = format!("plugin:{}", step.id);
        let mut registered: Vec<String> = Vec::new();
        for decl in &process.description().tools {
            let handler_process = Arc::clone(&process);
            let tool_name = decl.name.clone();
            let def = crate::tools::ToolDef::new(
                decl.name.clone(),
                decl.description.clone(),
                decl.parameters.clone(),
                owner.clone(),
                move |args| {
                    let process = Arc::clone(&handler_process);
                    let name = tool_name.clone();
                    Box::pin(async move {
                        let value = process.call(&name, args).await?;
                        // A plugin answers with JSON. Text is what the model
                        // reads, so a string answer is passed through and
                        // anything else is serialised compactly.
                        Ok(match value {
                            serde_json::Value::String(text) => {
                                crate::tools::ToolOutput::text(text)
                            }
                            other => crate::tools::ToolOutput::text(
                                serde_json::to_string(&other).unwrap_or_default(),
                            ),
                        })
                    }) as crate::tools::ToolFuture
                },
            );
            if let Err(error) = self
                .kernel
                .tools_mut()
                .register(def, ConflictPolicy::Override)
            {
                for tool in &registered {
                    self.kernel.tools_mut().unregister(tool);
                }
                return Err(error);
            }
            registered.push(decl.name.clone());
        }

        let tool_count = registered.len();
        let panel_count = process.description().panels.len();
        self.processes.push(Arc::clone(&process));

        self.ledger.record(LedgerEntry {
            kind: EntryKind::Plugin.as_str().into(),
            id: step.id.clone(),
            source: step.source.clone(),
            sha1: None,
            path: Some(path.display().to_string()),
            scope: scope_name(step.scope).into(),
            pack: self.pack.clone(),
            installed_at: now_iso(),
        });

        if tool_count == 0 && panel_count == 0 {
            return Ok(Outcome::Skipped(
                "plugin described neither a tool nor a panel".into(),
            ));
        }
        Ok(Outcome::Loaded)
    }

    /// Where a `plugin:` source points, if it is there.
    ///
    /// Relative paths resolve inside the pack; a bare name is handed to the
    /// operating system's `PATH` lookup, exactly like every other `command` in
    /// this loader.
    fn resolve_process_path(
        &self,
        raw: &str,
        pack_dir: Option<&std::path::Path>,
    ) -> Option<PathBuf> {
        let candidate = PathBuf::from(raw);
        if candidate.is_absolute() {
            return candidate.is_file().then_some(candidate);
        }
        if !raw.contains('/') && !raw.contains('\\') {
            // A bare name: let `Command` resolve it. Claiming a file we have not
            // checked would be worse than letting the spawn say it is missing.
            return Some(candidate);
        }
        let dir = pack_dir?;
        let joined = dir.join(&candidate);
        joined.is_file().then_some(joined)
    }

    /// Build a plugin from the source a pack ships.
    ///
    /// The toolchain is the pack's choice (`build.command`), not an enum here:
    /// the field names a directory and a binary, which is everything a build
    /// needs to know and everything the kernel should ever learn about one.
    ///
    /// A build that fails reports the command it ran and the compiler's own
    /// words. That is the whole value of building on the user's machine: the
    /// error names the real problem (a missing toolchain, a crates index the
    /// machine cannot reach) rather than "the binary is missing from the
    /// archive".
    fn build_plugin(
        &self,
        step: &PlannedStep,
        build: &crate::assembly::BuildSpec,
    ) -> Result<PathBuf> {
        let pack_dir = self.pack_dir.clone().ok_or_else(|| {
            anyhow!("plugin '{}' needs a build but has no pack directory", step.id)
        })?;
        let dir = pack_dir.join(&build.dir);
        if !dir.is_dir() {
            return Err(anyhow!(
                "plugin '{}' declares `build` with dir '{}', which is not a directory \
                 under {}",
                step.id,
                build.dir,
                pack_dir.display()
            ));
        }
        let command = build.command.clone().unwrap_or_else(|| "cargo".to_string());
        let args: Vec<String> = if build.args.is_empty() {
            vec![
                "build".into(),
                "--release".into(),
                // `--offline` unless the pack overrides: a build that reaches
                // the network while a turn is starting is a stall with no
                // explanation, and the crates a pack needs are in its lockfile
                // by the time it ships.
                "--offline".into(),
            ]
        } else {
            build.args.clone()
        };

        // Resolve the program through the *extended* PATH rather than letting
        // `Command` use the inherited one.
        //
        // On Windows the program lookup happens against the parent's PATH before
        // any `env` override applies, so `Command::new("cargo")` fails even with
        // a corrected child PATH. Resolving the absolute path first is the only
        // order that works.
        let program = resolve_program(&command).unwrap_or_else(|| std::path::PathBuf::from(&command));
        let mut spawn = std::process::Command::new(&program);
        spawn.args(&args).current_dir(&dir);
        // The toolchain has to be findable even when this process was started
        // with a narrower PATH than the user's shell has.
        //
        // Measured cause: `cargo` lives on `I:\cargoin` and was added to the
        // user's PATH, but the running `ngu` inherited the PATH it was launched
        // with, so a pack that built fine by hand failed here with "program not
        // found". A build step that only works when the parent happens to have
        // the right environment is a build step that fails for the user who
        // never opened a terminal.
        if let Some(path) = toolchain_path() {
            spawn.env("PATH", path);
        }
        // `CARGO_HOME` is where the crates index and the download cache live. A
        // build with the wrong one is not a build that fails to find `cargo` —
        // it is a build that finds `cargo` and then cannot find a single crate,
        // which is a much more confusing message. When the caller has not set
        // it, take the one the toolchain on `PATH` implies.
        if std::env::var_os("CARGO_HOME").is_none() {
            if let Some(home) = cargo_home_from_path() {
                // `RUSTUP_HOME` is the toolchains; `CARGO_HOME` is the crates.
                // They are usually siblings, and a machine that moved one moved
                // the other — but only claim it when the directory is there, so
                // a plain cargo install (no rustup) is left alone.
                let toolchains = home.join("..").join("rustup");
                if toolchains.join("toolchains").is_dir() {
                    if std::env::var_os("RUSTUP_HOME").is_none() {
                        spawn.env("RUSTUP_HOME", &toolchains);
                    }
                }
                spawn.env("CARGO_HOME", home);
            }
        }
        let output = spawn
            .output()
            .with_context(|| {
                format!(
                    "building plugin '{}' with `{}` in {} — is the toolchain installed \
                     and on PATH?",
                    step.id,
                    command,
                    dir.display()
                )
            })?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow!(
                "building plugin '{}' failed: `{} {}` in {} exited {}.\n{}",
                step.id,
                command,
                args.join(" "),
                dir.display(),
                output.status,
                stderr.trim()
            ));
        }

        let candidates = [
            dir.join("target").join("release").join(&build.binary),
            dir.join("target")
                .join("release")
                .join(format!("{}.exe", build.binary)),
            dir.join(&build.binary),
        ];
        for candidate in candidates {
            if candidate.is_file() {
                return Ok(candidate);
            }
        }
        Err(anyhow!(
            "plugin '{}' built without error but produced no '{}' under {}",
            step.id,
            build.binary,
            dir.join("target/release").display()
        ))
    }

    /// Every pack's settings, as the table a plugin reads.
    ///
    /// Serialised here rather than in `protocol` because the loader is what has
    /// a process to hand it to; the protocol module never reads the settings
    /// file.
    fn pack_settings(&self) -> std::collections::BTreeMap<String, serde_json::Value> {
        crate::settings::Settings::load()
            .unwrap_or_default()
            .packs
    }



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

    // The pack's appearance is not in the ledger — `content::apply` loads it as
    // a built-in plugin named `pack:<name>`, which is provenance the ledger
    // never records. Unloading it here is what makes a pack that was copied
    // into the directory by hand (no install, so no ledger entry) still lose
    // its stylesheet, labels and hooks: without this the interface kept them
    // forever, because the ledger loop above had nothing to iterate over.
    //
    // Done last so the ledger's own unloads keep their order in the report.
    let appearance = format!("pack:{name}");
    let appearance_fibers: Vec<u64> = kernel
        .fibers()
        .iter()
        .filter(|fiber| fiber.plugin == appearance)
        .map(|fiber| fiber.id)
        .collect();
    for id in appearance_fibers {
        if kernel.unload(id).is_ok() {
            report.plugins.push(appearance.clone());
        }
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

/// The environment a stdio MCP server starts with.
///
/// Declared entries win; `NGU_EXE` is added when the pack did not name one,
/// because the host's own path is not otherwise discoverable from inside a
/// plugin — and a pack that shells out to `ngu` (starting a worker, checking a
/// flag) needs exactly that. `${VAR}` references are left untouched on purpose:
/// substitution happens where the values are known.
/// Lines handed to each proxy, keyed by proxy id.
///
/// Process-wide because a proxy's target list is not per-session: the proxy is
/// one child process however many conversations exist, so a per-conversation list
/// would have to be merged back into one anyway.
static PROXY_TARGETS: std::sync::Mutex<
    Option<std::collections::BTreeMap<String, Vec<String>>>,
> = std::sync::Mutex::new(None);

/// The command line a proxy should run for one declared server.
///
/// The shape is the proxy's own, not the kernel's: `name=command args`, with a
/// `cwd=<dir>::` prefix when the server is a script named relative to its pack.
/// A pack that declares `node bin/x.mjs` means *its* `bin`, and the proxy starts
/// its children elsewhere — the prefix is what carries that.
fn proxy_target_line(
    step: &PlannedStep,
    entry: &crate::assembly::McpEntry,
    pack_dir: Option<&std::path::Path>,
) -> Result<String> {
    let command = entry
        .command
        .clone()
        .ok_or_else(|| anyhow!("mcp '{}' is handed to a proxy but declares no command", step.id))?;
    // The pack's own directory, when the server is a script that came with it.
    // Same rule the direct-spawn path uses: an argument that resolves to a file
    // inside the pack means the pack's directory is where the process belongs.
    let cwd = {
        let inside = |candidate: &str| {
            pack_dir
                .map(|dir| {
                    let path = std::path::Path::new(candidate);
                    !path.is_absolute() && dir.join(path).is_file()
                })
                .unwrap_or(false)
        };
        if inside(&command) || entry.args.iter().any(|arg| inside(arg)) {
            pack_dir
                .map(|dir| format!("cwd={}::", dir.display()))
                .unwrap_or_default()
        } else {
            String::new()
        }
    };
    let mut line = format!("{}={cwd}{command}", step.id);
    for arg in &entry.args {
        line.push(' ');
        line.push_str(arg);
    }
    Ok(line)
}

/// Record one server for a proxy to start.
fn register_proxy_target(proxy_id: &str, line: String) {
    let mut guard = PROXY_TARGETS.lock().expect("proxy target table");
    let table = guard.get_or_insert_with(Default::default);
    let entry = table.entry(proxy_id.to_string()).or_default();
    if !entry.contains(&line) {
        entry.push(line);
    }
}

/// The targets a proxy should start this process, one per line.
///
/// Read when the proxy itself is launched. Settings' own `mcp_proxy_targets` are
/// not consulted here: those are the user's hand-written list for servers no pack
/// declares, and mixing the two would make it impossible to tell which line came
/// from where when one of them fails to start.
fn proxy_targets_for(proxy_id: &str) -> Vec<String> {
    PROXY_TARGETS
        .lock()
        .expect("proxy target table")
        .as_ref()
        .and_then(|table| table.get(proxy_id).cloned())
        .unwrap_or_default()
}

fn mcp_env(declared: &std::collections::BTreeMap<String, String>) -> std::collections::HashMap<String, String> {
    let mut env: std::collections::HashMap<String, String> =
        declared.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
    if !env.contains_key("NGU_EXE") {
        if let Ok(exe) = std::env::current_exe() {
            env.insert("NGU_EXE".to_string(), exe.display().to_string());
        }
    }
    // Every pack's settings, as one JSON object, under one name.
    //
    // Before this, each pack that wanted a setting the user had chosen needed a
    // kernel-side reader: `mcp-proxy` had `NGU_MCP_MODE` and `NGU_MCP_TARGETS`
    // written out here by name, which made the kernel know that a pack called
    // `mcp-proxy` exists and that it has settings called `mode` and `targets`.
    // That is the shape this whole change is trying to remove — a pack cannot be
    // added without editing this function.
    //
    // One variable instead: the whole `packs` table, verbatim. A server that
    // wants its own settings reads `NGU_PACK_SETTINGS` and takes its own key;
    // one that wants nothing ignores it. The kernel's involvement ends at
    // "serialize the map".
    //
    // Declared entries still win, for the ordinary reason: a pack that hardcodes
    // an environment variable is saying it does not want this control.
    if !env.contains_key("NGU_PACK_SETTINGS") {
        let settings = crate::settings::Settings::load().unwrap_or_default();
        if !settings.packs.is_empty() {
            if let Ok(text) = serde_json::to_string(&settings.packs) {
                env.insert("NGU_PACK_SETTINGS".to_string(), text);
            }
        }
    }
    env
}

/// Start a plugin process from a synchronous caller.
///
/// `Loader` runs inside the host's async runtime most of the time, but
/// `load_plugin` is a synchronous method reached from `apply`, and the process
/// start is a `tokio` operation. Building a current-thread runtime here rather
/// than making every caller async keeps one boundary: everything above this line
/// is sync, everything below is not.
/// The cargo home implied by wherever `cargo` was found.
///
/// `cargo` lives in `<CARGO_HOME>/bin`, so the parent of its directory is the
/// home — which is the directory holding `registry/`. A machine that installed
/// Rust anywhere but the default has its cache there and nowhere else, and the
/// default (`~/.cargo`) does not exist at all.
fn cargo_home_from_path() -> Option<std::path::PathBuf> {
    let cargo = resolve_program("cargo")?;
    let home = cargo.parent()?.parent()?;
    // Only when it actually looks like a cargo home: a `cargo` on a plain
    // `PATH` with no registry beside it is not one, and setting `CARGO_HOME` to
    // nonsense replaces a clear error with a confusing one.
    home.join("registry").is_dir().then(|| home.to_path_buf())
}

/// Find an executable by name, looking where a shell would.
///
/// `None` when it is not on any of those paths, which leaves the caller to try a
/// bare name and report the operating system's own error.
fn resolve_program(command: &str) -> Option<std::path::PathBuf> {
    let direct = std::path::Path::new(command);
    if direct.components().count() > 1 {
        return direct.is_file().then(|| direct.to_path_buf());
    }
    let path = toolchain_path()?;
    let exts: Vec<&str> = if cfg!(windows) {
        vec!["", ".exe", ".cmd", ".bat"]
    } else {
        vec![""]
    };
    for dir in std::env::split_paths(&path) {
        for ext in &exts {
            let candidate = dir.join(format!("{command}{ext}"));
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    None
}

/// A `PATH` that includes the usual toolchain locations, whether or not the
/// process was started with one.
///
/// The cheap part is the process's own `PATH`, extended with `CARGO_HOME`/`.cargo`
/// when they are set. The expensive part — asking the OS for the *user's*
/// configured `PATH` rather than the inherited one — is only worth it on Windows,
/// where a program started from Explorer or a shortcut gets a stale environment
/// and a developer's toolchain is on a drive the default `PATH` has never heard
/// of.
///
/// Returns `None` when there is nothing to add, so the caller can leave the
/// environment alone: copying a `PATH` that is already right is a way to be wrong
/// about it later.
fn toolchain_path() -> Option<std::ffi::OsString> {
    let mut parts: Vec<String> = std::env::var("PATH")
        .unwrap_or_default()
        .split(if cfg!(windows) { ';' } else { ':' })
        .filter(|part| !part.trim().is_empty())
        .map(str::to_string)
        .collect();
    let before = parts.len();

    // The user's own environment, which on Windows is what their shell would
    // have had. Read only here, and only when a build is about to run.
    #[cfg(windows)]
    {
        for (var, scope) in [("Path", "User"), ("Path", "Machine")] {
            let script = format!("[Environment]::GetEnvironmentVariable('{var}','{scope}')");
            // Absolute path, not a bare name: this runs precisely when the
            // inherited `PATH` is the broken thing, so a lookup through it would
            // fail for the reason we are here to fix.
            let shell = std::path::Path::new(&std::env::var("SystemRoot").unwrap_or_else(|_| r"C:\Windows".into()))
                .join("System32")
                .join("WindowsPowerShell")
                .join("v1.0")
                .join("powershell.exe");
            if let Ok(output) = std::process::Command::new(&shell)
                .args(["-NoProfile", "-NonInteractive", "-Command", &script])
                .output()
            {
                if output.status.success() {
                    let text = String::from_utf8_lossy(&output.stdout);
                    for part in text.split(';') {
                        let part = part.trim();
                        if !part.is_empty() && !parts.iter().any(|known| known == part) {
                            parts.push(part.to_string());
                        }
                    }
                }
            }
        }
    }

    // `CARGO_HOME` names where cargo keeps its binaries; a machine that set it
    // did so for a reason and `cargo` is almost certainly in `<it>/bin`.
    if let Ok(home) = std::env::var("CARGO_HOME") {
        let candidate = std::path::Path::new(&home).join("bin");
        let text = candidate.display().to_string();
        if !parts.iter().any(|known| known == &text) {
            parts.push(text);
        }
    }

    // On Windows the user's `PATH` is consulted even when nothing was added:
    // the inherited value is not the one a developer's shell has, and the whole
    // point of this function is to bridge that gap rather than to be frugal.
    if parts.len() == before && !cfg!(windows) {
        return None;
    }
    if parts.is_empty() {
        return None;
    }
    Some(std::ffi::OsString::from(parts.join(if cfg!(windows) { ";" } else { ":" })))
}

fn start_process_with_state_for(spec: &crate::protocol::Spec) -> Result<crate::protocol::Process> {
    // The provider captures nothing: it reads the settings file and the thread's
    // workspace at the moment the plugin asks, so a plugin that asks twice sees
    // a settings save that landed between the two.
    let provider: std::sync::Arc<dyn Fn(&str) -> Option<serde_json::Value> + Send + Sync> =
        std::sync::Arc::new(|key: &str| plugin_state_of(key));
    match tokio::runtime::Handle::try_current() {
        Ok(_) => std::thread::scope(|scope| {
            scope
                .spawn(|| {
                    tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                        .map_err(|error| anyhow!("building a runtime: {error}"))
                        .and_then(|runtime| {
                            runtime.block_on(crate::protocol::Process::start_with_state(
                                spec,
                                Some(provider),
                            ))
                        })
                })
                .join()
                .map_err(|_| anyhow!("the plugin start thread panicked"))?
        }),
        Err(_) => tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|error| anyhow!("building a runtime: {error}"))
            .and_then(|runtime| {
                runtime.block_on(crate::protocol::Process::start_with_state(
                    spec,
                    Some(provider),
                ))
            }),
    }
}

/// The state keys, answered from module-level state rather than from a loader.
///
/// Split out of `Loader::plugin_state` because the provider handed to a process
/// must not borrow the loader: the process outlives the load call, and a borrow
/// would tie its lifetime to a stack frame that is gone by the time a plugin
/// asks.
fn plugin_state_of(key: &str) -> Option<serde_json::Value> {
    use crate::protocol::*;
    match key {
        KERNEL_PID => Some(serde_json::json!(crate::protocol::kernel_pid())),
        KERNEL_EXE => std::env::current_exe()
            .ok()
            .map(|exe| serde_json::json!(exe.display().to_string())),
        SETTINGS => serde_json::to_value(
            crate::settings::Settings::load()
                .unwrap_or_default()
                .packs,
        )
        .ok(),
        WORKSPACE => crate::tools::current_workspace()
            .map(|dir| serde_json::json!(dir.display().to_string())),
        SESSION => crate::tools::current_session()
            .map(|id| serde_json::json!(id)),
        _ => None,
    }
}

fn start_process(spec: &crate::protocol::Spec) -> Result<crate::protocol::Process> {
    match tokio::runtime::Handle::try_current() {
        // Already inside a runtime: `block_on` there panics. A dedicated thread
        // with its own runtime is the only correct way to wait.
        Ok(_) => std::thread::scope(|scope| {
            scope
                .spawn(|| {
                    tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                        .map_err(|error| anyhow!("building a runtime: {error}"))
                        .and_then(|runtime| runtime.block_on(crate::protocol::Process::start(spec)))
                })
                .join()
                .map_err(|_| anyhow!("the plugin start thread panicked"))?
        }),
        Err(_) => tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|error| anyhow!("building a runtime: {error}"))
            .and_then(|runtime| runtime.block_on(crate::protocol::Process::start(spec))),
    }
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
    #[test]
    fn a_pack_carried_script_runs_from_the_pack() {
        // The bug this pins: the rendered assembly writes `source: "stdio:node
        // files/x.mjs"`, so an emptiness test on `source` was never true and the
        // child got the host's cwd instead — the script was not there, and the
        // pack's MCP server died with "closed the connection".
        let pack = temp_pack_dir();
        std::fs::create_dir_all(pack.join("files")).unwrap();
        std::fs::write(pack.join("files").join("server.mjs"), "//x").unwrap();
        let cwd = super::Loader::pack_cwd(
            Some(&pack),
            "node",
            &["files/server.mjs".to_string(), "--verbose".to_string()],
        );
        assert_eq!(cwd, Some(pack.display().to_string()), "a relative file inside the pack decides");
        let _ = std::fs::remove_dir_all(&pack);
    }

    #[test]
    fn a_server_from_the_registry_keeps_its_own_working_directory() {
        // npm/git servers resolve their command elsewhere, and moving their cwd
        // into the pack could change where they write profiles and temp files.
        let pack = temp_pack_dir();
        let cwd = super::Loader::pack_cwd(Some(&pack), "chrome-devtools-mcp", &["--headless".to_string()]);
        assert_eq!(cwd, None, "nothing here names a file inside the pack");
        let _ = std::fs::remove_dir_all(&pack);
    }

    #[test]
    fn an_absolute_path_is_not_a_pack_claim() {
        let pack = temp_pack_dir();
        std::fs::create_dir_all(pack.join("files")).unwrap();
        std::fs::write(pack.join("files").join("x.mjs"), "//x").unwrap();
        let outside = pack.join("files").join("x.mjs");
        assert_eq!(
            super::Loader::pack_cwd(Some(&pack), "node", &[outside.to_string_lossy().to_string()]),
            None,
            "an absolute path already says where it lives"
        );
        let _ = std::fs::remove_dir_all(&pack);
    }

    #[test]
    fn no_pack_directory_means_no_opinion() {
        assert_eq!(super::Loader::pack_cwd(None, "node", &["files/x.mjs".to_string()]), None);
    }

    /// The ticket pack's worker launch failed in a real run because `ngu` was not
    /// on the shell's PATH and the pack had no way to learn where the host lives.
    #[test]
    fn every_stdio_server_is_told_where_the_host_is() {
        let env = super::mcp_env(&std::collections::BTreeMap::new());
        let exe = env.get("NGU_EXE").expect("NGU_EXE is published");
        assert!(std::path::Path::new(exe).is_absolute(), "an absolute path: {exe}");

        // A pack that declared its own value keeps it: a wrapper script or a
        // pinned older CLI is the author's call, not the loader's.
        let mut declared = std::collections::BTreeMap::new();
        declared.insert("NGU_EXE".to_string(), "/opt/pinned/ngu".to_string());
        let env = super::mcp_env(&declared);
        assert_eq!(env.get("NGU_EXE").map(String::as_str), Some("/opt/pinned/ngu"));
    }

    fn temp_pack_dir() -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!("ngu-mcp-cwd-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&path).unwrap();
        path
    }
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
            dir.join(crate::pack::MANIFEST_NAME),
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

    /// The line a proxy is given for one declared server.
    ///
    /// Two properties, both learned from failures: a pack script carries its
    /// directory (the proxy starts children elsewhere), and a Windows drive
    /// letter survives — the single-colon form silently truncated
    /// `C:\Users\...` to `C`, and every target then reported ENOENT for the node
    /// executable rather than for the path that was actually wrong.
    #[test]
    fn a_proxied_server_carries_its_pack_directory_and_its_drive_letter() {
        let dir = std::env::temp_dir().join(format!("ngu-proxy-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let pack = dir.join("a-pack");
        std::fs::create_dir_all(pack.join("bin")).unwrap();
        std::fs::write(pack.join("bin").join("srv.mjs"), "//").unwrap();

        let mut entry = crate::assembly::McpEntry::default();
        entry.command = Some("node".into());
        entry.args = vec!["bin/srv.mjs".into()];
        let step = crate::assembly::PlannedStep {
            kind: crate::assembly::EntryKind::Mcp,
            id: "ticket".into(),
            stage: "pack".into(),
            scope: crate::assembly::Scope::Session,
            conflict: crate::assembly::ConflictKind::Error,
            on_failure: crate::assembly::OnFailure::Skip,
            load: crate::assembly::LoadMode::Eager,
            order: 0,
            source: "stdio:node bin/srv.mjs".into(),
            config: serde_json::Value::Null,
            build: None,
            mcp: Some(entry.clone()),
            sha1: None,
            requires: None,
        };

        let line = proxy_target_line(&step, &entry, Some(pack.as_path())).unwrap();
        assert!(line.starts_with("ticket=cwd="), "{line}");
        assert!(line.contains("::node bin/srv.mjs"), "{line}");
        // The drive letter must survive the separator split: `cwd=C:` would be
        // the old bug in miniature.
        let cwd_part = line
            .trim_start_matches("ticket=cwd=")
            .split("::")
            .next()
            .unwrap();
        assert_eq!(cwd_part, pack.as_path().display().to_string(), "{line}");

        // A server that came from somewhere else keeps no cwd: moving it into
        // the pack would change where it writes its own state.
        let mut entry = crate::assembly::McpEntry::default();
        entry.command = Some("npx".into());
        entry.args = vec!["-y".into(), "some-server".into()];
        let step = crate::assembly::PlannedStep {
            kind: crate::assembly::EntryKind::Mcp,
            id: "vk".into(),
            stage: "pack".into(),
            scope: crate::assembly::Scope::Session,
            conflict: crate::assembly::ConflictKind::Error,
            on_failure: crate::assembly::OnFailure::Skip,
            load: crate::assembly::LoadMode::Eager,
            order: 0,
            source: "stdio:npx -y some-server".into(),
            config: serde_json::Value::Null,
            build: None,
            mcp: Some(entry.clone()),
            sha1: None,
            requires: None,
        };
        let line = proxy_target_line(&step, &entry, Some(pack.as_path())).unwrap();
        assert_eq!(line, "vk=npx -y some-server");
    }
}