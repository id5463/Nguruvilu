//! 加载器的行为：建它、用它跑一份计划、以及三类条目各自的加载动作。
//!
//! `apply` 是唯一的装配入口：按顺序走计划的每一步，把失败交给该步的
//! `on_failure` 策略，最后交回一份 `LoadReport`。插件、MCP server、技能
//! 三类条目共用一个顺序、一份策略和一个账本，所以谁都不是特例。

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use serde_json::Value;

use crate::assembly::{OnFailure, Plan, PlannedStep, Scope};
use crate::ledger::{Ledger, LedgerEntry};
use crate::mcp::{McpClient, McpSpec};
use crate::plugin::{Kernel, Plugin, RealmMap};
use crate::protocol::{self, Spec};
use crate::skills::{Skill, SkillRegistry};

use super::model::{FailedEntry, LazyMcp, LoadReport, Loader, Outcome};

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
            lazy_mcp: Vec::new(),
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
    ///
    /// Always stored absolute. `ngu apply assembly.yaml` passes `file.parent()`,
    /// which is `""` for a bare file name — and `absolute("")` is this process's
    /// cwd, so a plugin's binary became `<cwd>/bin/x` while `Spec::cwd` was set
    /// to the same directory. The child then resolved `<cwd>/bin/x` a second
    /// time *after* the `chdir`, and `bin/x` inside `bin/` does not exist:
    /// ENOENT, from a file that is plainly there.
    ///
    /// Measured: `cd packs/delegate && ngu apply assembly.yaml` failed while
    /// `ngu apply packs/delegate/assembly.yaml` from the parent succeeded.
    pub fn with_pack_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.pack_dir = Some(absolute(&dir.into()));
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
        //
        // Only when the proxy is actually here. A pack that says `via_proxy:
        // mcp-proxy` is asking for the proxy *pack* to be installed, and nothing
        // in the manifest can make that true — so a machine with the server but
        // not the proxy used to lose the server's tools entirely, and report
        // `skipped`, which reads as success. Measured on a server with
        // browser-use, mail and moodle installed and no mcp-proxy: 46 tools
        // vanished and no line said so.
        //
        // Falling through to a direct start is the honest answer: the tools the
        // pack declares appear, unproxied, instead of not appearing at all.
        if let Some(proxy_id) = entry.via_proxy.as_deref() {
            if self.kernel.tools().has_owner(&format!("plugin:{proxy_id}"))
                || self.kernel.tools().has_owner(&format!("mcp:{proxy_id}"))
            {
                let line = proxy_target_line(step, entry, self.pack_dir.as_deref())?;
                crate::loader::register_proxy_target(proxy_id, line);
                return Ok(Outcome::Skipped(format!(
                    "handed to MCP proxy '{proxy_id}'"
                )));
            }
            // No proxy: say so once, then start the server directly.
            eprintln!(
                "[pack] {} asked for MCP proxy '{}', which is not installed;                  starting it directly so its tools are reachable",
                step.id, proxy_id
            );
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

        let owner = format!("mcp:{}", step.id);

        // Declared tools: register them and start the server on first use.
        //
        // Measured on this machine, thirteen servers cost 6.2 s on a cold start
        // against 47 ms for a process that loads nothing — and a subagent
        // answering one question paid the whole bill. With the list declared,
        // the cost is a file read at load and the start happens on the call that
        // needs it.
        if !entry.tools.is_empty() {
            let lazy = Arc::new(LazyMcp {
                spec: spec.clone(),
                client: tokio::sync::Mutex::new(None),
            });
            let mut registered: Vec<String> = Vec::new();
            for decl in &entry.tools {
                let exposed = format!("mcp__{}__{}", step.id, decl.name);
                let handler = Arc::clone(&lazy);
                let remote = decl.name.clone();
                let name_for_closure = exposed.clone();
                let def = ToolDef::new(
                    exposed.clone(),
                    if decl.description.is_empty() {
                        format!("MCP tool '{}' from server '{}'", decl.name, step.id)
                    } else {
                        decl.description.clone()
                    },
                    decl.input_schema
                        .clone()
                        .unwrap_or_else(|| serde_json::json!({ "type": "object", "properties": {} })),
                    owner.clone(),
                    move |args| {
                        let handler = Arc::clone(&handler);
                        let tool = remote.clone();
                        Box::pin(async move {
                            let client = handler.get().await?;
                            client.call(&tool, args).await
                        }) as ToolFuture
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
                    return Err(error).with_context(|| {
                        format!("registering '{}' from '{}'", name_for_closure, step.id)
                    });
                }
                registered.push(exposed);
            }
            self.lazy_mcp.push(Arc::clone(&lazy));
            self.ledger.record(LedgerEntry {
                kind: EntryKind::Mcp.as_str().into(),
                id: step.id.clone(),
                source: format!("stdio:{} {}", spec.command, spec.args.join(" "))
                    .trim()
                    .to_string(),
                sha1: None,
                path: None,
                scope: scope_name(step.scope).into(),
                pack: self.pack.clone(),
                installed_at: now_iso(),
            });
            if registered.is_empty() {
                return Ok(Outcome::Skipped("server declared no tools".into()));
            }
            return Ok(Outcome::Loaded);
        }

        let client = Arc::new(McpClient::connect(&spec).await?);

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

        // Already provided by another pack: reuse it, do not start a second copy.
        //
        // The same capability legitimately appears in more than one pack —
        // `delegate` is wanted by the ticket board, the research pack, and a
        // plain starter install alike, and each of them declares it because a
        // pack that silently depends on another pack being installed is a pack
        // that breaks when it is installed alone.
        //
        // Before this, the second declaration *overwrote* the first
        // (`ConflictPolicy::Override`), so which copy ran depended on load
        // order, and unloading either pack removed the tool for both. Starting a
        // second process was the other half of that: two pipes, one table.
        //
        // The identity is the entry id — `delegate` means the same capability no
        // matter which pack names it. A plugin whose id is already serving tools
        // is reported as reused, and the loader keeps no handle to it.
        // A process plugin's tools are owned by `plugin:<id>`, so a live entry
        // for any of them means this capability is already running.
        let owner = format!("plugin:{}", step.id);
        if self.kernel.tools().has_owner(&owner) {
            return Ok(Outcome::Reused);
        }

        let mut spec = crate::protocol::Spec::new(step.id.clone(), path.display().to_string());
        if let Some(dir) = &pack_dir {
            // A plugin that came with a pack runs inside it, the same rule MCP
            // servers already follow: a relative path in its own arguments then
            // means what the pack author meant.
            spec = spec.in_dir(dir.display().to_string());
        }
        // `NGU_EXE` — the host's own binary — so a plugin can start another
        // copy of the kernel.
        //
        // MCP servers have had this since the pack protocol was written; process
        // plugins did not, and the gap is exactly the one a subagent hits: it is
        // a plugin whose whole job is to run `ngu -p`, and without this it has
        // to guess where the kernel is. Measured on the server, where every
        // guess missed and `delegate` answered "cannot find the kernel binary".
        //
        // A pack that names its own `NGU_EXE` wins: it may want a specific build.
        if !spec.env.contains_key("NGU_EXE") {
            if let Ok(exe) = std::env::current_exe() {
                spec.env
                    .insert("NGU_EXE".to_string(), exe.display().to_string());
            }
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

        // Readers: a format this plugin teaches the kernel's own `read`.
        //
        // Registered into the same table the kernel version used, with a process
        // where the closure was — see `readers::register_process`. The disposers
        // go on the ledger's effects so an unload takes them with the tools.
        let mut reader_count = 0;
        for decl in &process.description().readers {
            crate::tools::readers::register_process(crate::tools::readers::ProcessReader {
                id: decl.id.clone(),
                process: Arc::clone(&process),
                magic: decl.magic.clone(),
                extensions: decl.extensions.clone(),
            });
            reader_count += 1;
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

        if tool_count == 0 && panel_count == 0 && reader_count == 0 {
            return Ok(Outcome::Skipped(
                "plugin described no tool, panel, or reader".into(),
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
        // Absolute, because the returned path travels through `Spec::command`
        // and is then executed *after* `Spec::cwd` has moved the child. A
        // relative path resolves twice — once here against this process's cwd,
        // once in the child against `cwd` — and the second resolution is the one
        // that fails with ENOENT the moment the two differ. Measured: loading
        // `delegate` from its own directory failed while the same manifest from
        // the parent directory succeeded, which is the fingerprint of exactly
        // this.
        let dir = absolute(pack_dir?);
        let joined = absolute(&dir.join(&candidate));
        if joined.is_file() {
            return Some(joined);
        }
        // The platform's own suffix, when the name does not carry one.
        //
        // A pack says `plugin:bin/ngu-restart` and means the binary on whatever
        // machine this is. Windows needs `.exe`; Unix does not. Resolving both
        // here is what lets one `assembly.yaml` work on either — measured on a
        // Linux server, where the first version looked for `ngu-restart` with no
        // suffix, did not find it, and reported "the file is missing" for a file
        // that had compiled a moment earlier.
        for suffix in platform_suffixes() {
            let with_suffix = absolute(&dir.join(format!("{}{suffix}", candidate.display())));
            if with_suffix.is_file() {
                return Some(with_suffix);
            }
        }
        None
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
        // Absolute, always. The pack directory a caller hands over may be
        // relative (`packs/restart`, as `ngu apply` passes it), and a relative
        // path survives neither a `current_dir` change nor the spawn below —
        // measured on a Linux server, where the build succeeded, the binary
        // existed, and starting it failed with "No such file or directory"
        // because the path was resolved against the *pack* once and the
        // *process's* directory twice.
        let dir = if pack_dir.is_absolute() {
            pack_dir.join(&build.dir)
        } else {
            std::env::current_dir()
                .map(|cwd| cwd.join(&pack_dir).join(&build.dir))
                .unwrap_or_else(|_| pack_dir.join(&build.dir))
        };
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
