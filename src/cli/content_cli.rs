//! 把一次包加载的内容折进当前会话：设置、路由、人格、外观。
//!
//! 启动时调一次，回合之间热加载包时再调一次 —— 两条路径应用完全相同的东西。
//! 包的设置会替换原有的，因为包是用户选择的单位。
//!
//! 外观那一半已经在内核里落地了（见 `nguruvilu::content::apply`），这里只做
//! 会话侧的折叠。

use std::sync::Arc;

use anyhow::Result;

use nguruvilu::plugin::Kernel;
use nguruvilu::settings::Settings;

/// Fold one pack's content into this session.
///
/// Used at startup and again when a pack is hot-loaded between turns, so both
/// paths apply exactly the same things: a pack's settings replace the standing
/// ones, because the pack is the unit the user chose. The content's appearance
/// part has already landed on the kernel by the time this runs — see
/// [`nguruvilu::content::apply`].
fn adopt_content(
    content: nguruvilu::content::PackContent,
    settings: &mut Settings,
    runtime: &mut Runtime,
    injection: &Arc<Mutex<Arc<dyn nguruvilu::context::InjectionSource>>>,
) -> Result<()> {
    for line in content.summary() {
        eprintln!("pack content: {line}");
    }
    // The persona is the joined soul of every installed pack — never this
    // pack's alone: applied one by one, load order would decide who the
    // model is. A user's `--persona` outranks it and is re-applied after
    // adoption (and after a drain) by the callers that hold the flag.
    if let Some(persona) =
        nguruvilu::content::installed_persona(&nguruvilu::pack::default_packs_dir())?
    {
        runtime.apply(Change::session("pack", ChangePayload::Persona(persona)))?;
    }
    if let Some(models) = &content.models {
        if let Some(url) = &models.base_url {
            settings.base_url = url.clone();
        }
        if let Some(name) = &models.model {
            settings.model = name.clone();
        }
        if let Some(effort) = &models.reasoning_effort {
            settings.reasoning_effort = effort.clone();
        }
        // The delivery mode belongs to the gateway, so the pack that names the
        // gateway names the mode too: an endpoint whose stream never completes
        // is not a preference a user can be assumed to have discovered.
        if let Some(delivery) = &models.delivery {
            settings.delivery = delivery.clone();
        }
        if let Some(degrade) = models.degrade_on_stall {
            settings.degrade_on_stall = Some(degrade);
        }
        if let Some(proxy) = &models.proxy {
            settings.proxy = proxy.clone();
        }
        if let Some(ceiling) = &models.max_output_tokens {
            settings.max_output_tokens = Some(nguruvilu::size::parse_size(ceiling)?);
        }
        for (key, value) in &models.extra_body {
            settings.extra_body.insert(key.clone(), value.clone());
        }
        // Connection behaviour travels with the rest: a pack that names a
        // flaky provider should also be able to say how hard to retry it.
        if let Some(network) = &models.network {
            settings.network = network.clone();
        }
        // The key itself is never in a pack; only the name of the variable
        // holding it.
        if let Some(var) = &models.api_key_env {
            if let Ok(value) = std::env::var(var) {
                if !value.trim().is_empty() {
                    settings.api_key = value;
                }
            }
        }
    }
    if let Some(context) = &content.context {
        if let Some(window) = &context.window {
            settings.context_window = Some(nguruvilu::size::parse_size(window)?);
        }
        if let Some(percent) = context.compact_percent {
            settings.compact_percent = percent.min(100);
        }
        if let Some(keep) = context.compact_keep_recent {
            settings.compact_keep_recent = keep.max(1);
        }
    }
    if let Some(rules) = content.injections {
        *injection
            .lock()
            .expect("injection lock") = Arc::new(rules);
    }
    Ok(())
}

/// Applies the pack load and unload a turn asked for.
///
/// The tool runs while the turn owns the kernel, so it records what it wants
/// in a queue; this drains the queue between turns, which is the first moment
/// the tool table is read again anyway. Nothing here restarts or pauses the
/// conversation — it continues with a different table, and a tool that was
/// unloaded answers "no tool named …" from then on.
struct PackHost<'a> {
    cli: &'a Cli,
    kernel: &'a mut Kernel,
    skills: &'a Arc<Mutex<SkillRegistry>>,
    runtime: &'a Arc<Mutex<Runtime>>,
    settings: &'a mut Settings,
    injection: &'a Arc<Mutex<Arc<dyn nguruvilu::context::InjectionSource>>>,
    queue: &'a nguruvilu::tools::pack::PendingQueue,
    /// The session's MCP servers: joined when a pack loads, stopped when it
    /// unloads, and stopped for good on the way out.
    mcp: &'a mut Vec<Arc<nguruvilu::mcp::McpClient>>,
    /// An explicit `--persona` from this run: adoption joins the installed
    /// packs' souls, and this outranks them, so it goes back on top after
    /// every drain.
    persona_override: Option<String>,
    /// The user's own settings before any pack folded over them.
    ///
    /// A drain recomputes the adopted state from this baseline plus the
    /// enabled set, so an unload takes back its pack's route and window
    /// while the stored values come back with it.
    baseline: Settings,
}

impl PackHost<'_> {
    /// Apply every request recorded during the turn that just ended.
    async fn drain(&mut self) -> Result<()> {
        use nguruvilu::tools::pack::Pending;

        let mut changed = false;
        // Kept because the tool-table adoption happens once, after the loop,
        // and by then no per-pack label is in scope — yet a refusal has to be
        // able to say which packs the table it would not take belonged to.
        let mut touched: Vec<String> = Vec::new();
        for action in self.queue.drain() {
            let packs = nguruvilu::pack::installed(&nguruvilu::pack::default_packs_dir())?;
            match action {
                // The files already landed from the tool call itself; the CLI
                // has no page to refresh, so the log line is the report.
                Pending::Installed { name, version } => {
                    eprintln!("[pack] installed {name} {version}");
                }
                Pending::Unload { name, session: _ } => {
                    let mut found = false;
                    for pack in packs.iter().filter(|pack| pack.manifest.name == name) {
                        let label = format!("{}-{}", pack.manifest.name, pack.manifest.version_id);
                        let report = {
                            let mut guard = self.skills.lock().expect("skills lock");
                            nguruvilu::loader::unload_pack(
                                self.kernel,
                                &mut *guard,
                                &default_ledger_path(),
                                &label,
                                &pack.manifest.name,
                            )?
                        };
                        found = true;
                        changed = true;
                        touched.push(label.clone());
                        eprintln!(
                            "[pack] unloaded {label}: {} plugin instance(s), {} mcp server(s), {} skill root(s)",
                            report.plugins.len(),
                            report.mcp.len(),
                            report.skills.len()
                        );
                        for note in &report.notes {
                            eprintln!("[pack] {note}");
                        }
                        // The tools are gone, so the servers they belonged to
                        // have no reason to keep running.
                        for client in self
                            .mcp
                            .iter()
                            .filter(|client| report.mcp.iter().any(|id| id == client.server()))
                            .cloned()
                            .collect::<Vec<_>>()
                        {
                            client.shutdown().await;
                            self.mcp.retain(|kept| !Arc::ptr_eq(kept, &client));
                        }
                    }
                    if !found {
                        eprintln!("[pack] {name} is not installed; nothing was loaded to unload");
                    }
                }
                Pending::Load { name, session: _ } => {
                    for pack in packs
                        .iter()
                        .filter(|pack| pack.manifest.name == name && pack.enabled)
                    {
                        let label = format!("{}-{}", pack.manifest.name, pack.manifest.version_id);
                        // Same repair as boot: a pack with no assembly gets
                        // one rendered rather than loading nothing.
                        let assembly = match nguruvilu::pack::ensure_assembly(&pack.path) {
                            Ok(assembly) => assembly,
                            Err(error) => {
                                eprintln!("[pack] {} assembly: {error:#}", pack.manifest.name);
                                None
                            }
                        };
                        if let Some(assembly) = &assembly {
                            let (pack_skills, report, servers) = nguruvilu::loader::load_pack(
                                self.kernel,
                                &default_ledger_path(),
                                assembly,
                                &label,
                            )
                            .await?;
                            self.mcp.extend(servers);
                            {
                                let mut guard = self.skills.lock().expect("skills lock");
                                nguruvilu::loader::merge_skills(
                                    self.kernel,
                                    &mut *guard,
                                    pack_skills,
                                )?;
                            }
                            eprintln!(
                                "[pack] loaded {label}: {} entr(ies), {} failed",
                                report.loaded.len(),
                                report.failed.len()
                            );
                            // Failures name themselves, the same way boot
                            // prints them: a count alone is how a pack ends
                            // up "loaded but nothing works".
                            for failed in &report.failed {
                                eprintln!(
                                    "[pack] failed {} '{}': {}",
                                    failed.kind, failed.id, failed.error
                                );
                            }
                            for entry in &report.skipped {
                                eprintln!("[pack] skipped {}: {}", entry.id, entry.reason);
                            }
                            changed = true;
                            touched.push(label.clone());
                        }
                        match nguruvilu::pack::read_manifest(&pack.path) {
                            Ok(manifest) => {
                                match nguruvilu::content::apply(self.kernel, &pack.path, &manifest)
                                {
                                    // Appearance has landed on the kernel; the
                                    // settings-side content is recomputed from
                                    // the whole enabled set below, once the
                                    // drain knows what changed.
                                    Ok(content) => {
                                        for line in content.summary() {
                                            eprintln!("pack content: {line}");
                                        }
                                    }
                                    Err(error) => eprintln!(
                                        "[pack] {} content: {error:#}",
                                        pack.manifest.name
                                    ),
                                }
                            }
                            Err(error) => {
                                eprintln!("[pack] {}: {error:#}", pack.manifest.name)
                            }
                        }
                    }
                }
            }
        }

        if changed {
            // Set-shaped adoption: every enabled pack, recomputed from the
            // user's own settings at once. A load and an unload converge on
            // the state a fresh boot would produce — an unload has an inverse
            // this way, where folding one pack at a time had none.
            let adopted = nguruvilu::content::AdoptedContent::recompute(
                &self.baseline,
                &nguruvilu::pack::default_packs_dir(),
            )?;
            *self.settings = adopted.settings;
            *self.injection
                .lock()
                .expect("injection lock") = Arc::new(adopted.injections);
            // The route may have moved with the pack's content, and the tool
            // table certainly did; both are published together so the next
            // turn reads one consistent view.
            let route = model_route(self.cli, self.settings, Some(self.kernel));
            let mut guard = self.runtime.lock().expect("runtime lock");
            // The joined persona of every enabled pack — cleared when none
            // carries one, so an unloaded pack's voice leaves with it. A run
            // started with `--persona` outranks packs and goes on top right
            // after.
            guard.apply(Change::session(
                "pack",
                ChangePayload::Persona(adopted.persona.unwrap_or_default()),
            ))?;
            guard.apply(Change::session(
                "pack",
                ChangePayload::ModelRoute(route),
            ))?;
            if let Some(persona) = self.persona_override.as_ref() {
                guard.apply(Change::session(
                    "cli",
                    ChangePayload::Persona(persona.clone()),
                ))?;
            }
            // A denial leaves the previous table in place, so the pack reads as
            // loaded while its tools are not callable. That is the whole trace
            // a refused swap leaves, so it says which packs it was for and
            // repeats the policy's own words. It is not an error: this returns
            // `Ok` and the turn loop carries on with the table it already had.
            // `changed` is only ever set beside a push, so `touched` is not
            // empty here.
            if let nguruvilu::hotreload::ApplyOutcome::Denied(reason) =
                guard.adopt_kernel_tools(self.kernel)
            {
                eprintln!(
                    "[pack] the tool table was not swapped for {}: {reason}",
                    touched.join(", ")
                );
            } else {
                eprintln!("[pack] the next turn sees the new tool table");
            }
        }
        Ok(())
    }
}

