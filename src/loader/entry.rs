//! 按计划装配：走一步、装一个条目。
//!
//! `apply` 是唯一入口。它按顺序走计划的每一步，把失败交给该步的 `on_failure`
//! 策略。三类条目（插件、MCP server、技能）共用这一次遍历、这一份策略、这一个
//! 账本 —— 所以谁都不是特例。
//!
//! # 为什么 `apply` 不吞错误
//!
//! 装配失败时最坏的结果是“半装”：一半的包在跑，另一半不在。所以 `on_failure`
//! 是包作者声明的，不是内核猜的：`abort` 让整次装配失败，`skip` 记一笔继续。

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};

use crate::assembly::{OnFailure, Plan, PlannedStep, Scope};
use crate::ledger::LedgerEntry;
use crate::mcp::McpSpec;
use crate::plugin::{Kernel, Plugin};
use crate::skills::Skill;

use super::model::{FailedEntry, LazyMcp, LoadReport, Loader, Outcome};

impl Loader {
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

}
