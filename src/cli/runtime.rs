//! `ngu runtime` 一族：内核自己现在是什么样、装配清单长什么样、有哪些技能、
//! 以及 git 快照。
//!
//! 这四件事都是"读出来给人看"，不改变任何状态，所以它们在一处。共同的判据是：
//! 想知道内核现在装了什么的人，和想换掉"怎么报告"的人，是两种人 —— 所以每一件
//! 都可以单独被覆盖。

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};

use nguruvilu::agent::Agent;
use nguruvilu::plugin::Kernel;
use nguruvilu::skills::SkillRegistry;

use super::args::*;

fn list_skills(skills: &SkillRegistry, verbose: bool, as_json: bool) -> Result<()> {
    let entries: Vec<serde_json::Value> = skills
        .list()
        .iter()
        .map(|skill| {
            serde_json::json!({
                "id": skill.id,
                "description": skill.description,
                "path": skill.path.display().to_string(),
                "source": skill.source.display().to_string(),
            })
        })
        .collect();

    if as_json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "roots": skills.roots().iter().map(|r| r.display().to_string()).collect::<Vec<_>>(),
                "skills": entries,
            }))?
        );
        return Ok(());
    }

    if skills.is_empty() {
        println!("no skills found under:");
        for root in skills.roots() {
            println!("  {}", root.display());
        }
        return Ok(());
    }

    for skill in skills.list() {
        println!("{:<24} {}", skill.id, skill.description);
    }
    if verbose {
        for skill in skills.list() {
            println!("\n=== {} ===\n{}", skill.id, skill.body);
        }
    }
    Ok(())
}

fn show_assembly(file: &PathBuf, platform: &str, as_json: bool) -> Result<()> {
    let assembly = Assembly::from_file(file)?;
    let plan = assembly.plan(platform)?;

    if as_json {
        let steps: Vec<serde_json::Value> = plan
            .steps
            .iter()
            .map(|step| {
                serde_json::json!({
                    "kind": step.kind.as_str(),
                    "id": step.id,
                    "stage": step.stage,
                    "scope": match step.scope { nguruvilu::assembly::Scope::Session => "session", nguruvilu::assembly::Scope::Global => "global" },
                    "source": step.source,
                    "order": step.order,
                })
            })
            .collect();
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "file": file.display().to_string(),
                "platform": platform,
                "steps": steps,
                "skipped": plan.skipped.iter().map(|s| serde_json::json!({"id": s.id, "reason": s.reason})).collect::<Vec<_>>(),
            }))?
        );
        return Ok(());
    }

    println!("{} (platform {platform})", file.display());
    let mut stage = String::new();
    for step in &plan.steps {
        if step.stage != stage {
            stage = step.stage.clone();
            println!("\n[{stage}]");
        }
        println!(
            "  {:<8} {:<24} {}",
            step.kind.as_str(),
            step.id,
            if step.source.is_empty() { "-" } else { &step.source }
        );
    }
    for skipped in &plan.skipped {
        println!("\nskipped {}: {}", skipped.id, skipped.reason);
    }
    Ok(())
}

async fn apply_assembly(
    file: &PathBuf,
    platform: &str,
    skills: &SkillRegistry,
    as_json: bool,
) -> Result<()> {
    let assembly = Assembly::from_file(file)?;
    let plan = assembly.plan(platform)?;

    // Prefer the pack's own identity manifest over the file name, so the ledger
    // records which pack a contribution came from rather than which file
    // happened to describe it.
    let pack = file
        .parent()
        .and_then(|dir| nguruvilu::pack::read_manifest(dir).ok())
        .map(|manifest| format!("{}-{}", manifest.name, manifest.version_id))
        .unwrap_or_else(|| {
            file.file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| "pack".into())
        });

    let mut kernel = Kernel::new();
    // The base tools, and every `builtin:` plugin this binary carries — the same
    // two calls `run` makes. Without them `ngu apply` could not load a pack whose
    // assembly names a builtin, which is how it drifted before.
    kernel.install_bootstrap()?;
    kernel.define_builtins();
    nguruvilu::kernel_routes::route_everything(&kernel);
    // The agent can build and install packs itself.
    nguruvilu::tools::pack::register(kernel.tools_mut())?;
    // The model access service. `delegate` injects it, so without this a pack
    // naming `builtin:delegate` loads and contributes nothing — the silent
    // failure this command used to have. Applying a pack is not the same as
    // running a turn, so the route comes from the settings file and needs no
    // key: nothing here will call the model.
    let plan_settings = Settings::resolve();
    let access: Arc<dyn nguruvilu::model::ModelAccess> =
        Arc::new(nguruvilu::model::StaticAccess::new(
            nguruvilu::hotreload::ModelRoute::from_settings(&plan_settings),
            Arc::new(kernel.tools().clone()),
        ));
    nguruvilu::model::install(&mut kernel, access)?;
    if !skills.is_empty() {
        register_skill_tool(kernel.tools_mut(), Arc::new(skills.clone()))?;
    }

    let mut loader = Loader::new(kernel, default_ledger_path(), pack)
        .with_pack_dir(file.parent().map(PathBuf::from).unwrap_or_else(|| PathBuf::from(".")));

    let report = loader.apply(&plan).await?;

    if as_json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "loaded": report.loaded,
                "reused": report.reused,
                "failed": report.failed.iter().map(|f| serde_json::json!({"id": f.id, "kind": f.kind, "error": f.error})).collect::<Vec<_>>(),
                "skipped": report.skipped.iter().map(|s| serde_json::json!({"id": s.id, "reason": s.reason})).collect::<Vec<_>>(),
                "tools": loader.kernel().tools().names(),
                "services": loader.kernel().service_list().iter().map(|(name, realm, owner)| serde_json::json!({"name": name, "realm": realm.to_string(), "owner": owner})).collect::<Vec<_>>(),
                "mcp_servers": loader.mcp_servers(),
                "skills": loader.skills().list().iter().map(|s| s.id.clone()).collect::<Vec<_>>(),
            }))?
        );
    } else {
        for entry in &report.loaded {
            println!("loaded  {entry}");
        }
        for entry in &report.reused {
            println!("reused  {entry}");
        }
        for entry in &report.failed {
            println!("failed  {}:{} — {}", entry.kind, entry.id, entry.error);
        }
        for entry in &report.skipped {
            println!("skipped {} — {}", entry.id, entry.reason);
        }
        println!("\ntools: {}", loader.kernel().tools().names().join(", "));
        let servers = loader.mcp_servers();
        if !servers.is_empty() {
            println!("mcp:   {}", servers.join(", "));
        }
        let loaded_skills: Vec<String> =
            loader.skills().list().iter().map(|s| s.id.clone()).collect();
        if !loaded_skills.is_empty() {
            println!("skills: {}", loaded_skills.join(", "));
        }
    }

    loader.shutdown().await;
    Ok(())
}

fn snapshot_command(
    cli: &Cli,
    action: &str,
    commit: Option<&str>,
    limit: usize,
    as_json: bool,
) -> Result<()> {
    let workspace = cli
        .cwd
        .clone()
        .unwrap_or(std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
    let git = GitSnapshot::open(&workspace)?;

    if !git.is_available() {
        return Err(anyhow!("git is not available on this machine"));
    }

    match action {
        "list" => {
            let records = git.history(limit)?;
            if as_json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({
                        "workspace": workspace.display().to_string(),
                        "snapshots": records.iter().map(|r| serde_json::json!({"commit": r.commit, "subject": r.subject})).collect::<Vec<_>>(),
                    }))?
                );
                return Ok(());
            }
            if records.is_empty() {
                println!("no snapshots in {}", workspace.display());
                return Ok(());
            }
            for record in records {
                println!("{}  {}", record.short(), record.subject);
            }
            Ok(())
        }
        "restore" => {
            let commit = commit.ok_or_else(|| anyhow!("restore needs a commit id"))?;
            let record = git.restore(commit)?;
            println!("restored to {} ({})", record.short(), record.subject);
            Ok(())
        }
        other => Err(anyhow!("unknown snapshot action '{other}'; expected list or restore")),
    }
}

/// Describe the session that would run.
///
/// Answered from the assembled kernel and the runtime a turn reads, so the
/// tools and services reported here are the ones a model would actually be
/// handed — a fresh table with only the base tools would describe a session
/// that does not exist.
fn show_runtime(
    route: &ModelRoute,
    runtime: &Runtime,
    cache_policy: CachePolicy,
    skills: &SkillRegistry,
    as_json: bool,
    kernel: &Kernel,
) -> Result<()> {
    let tools = kernel.tools().names();
    let services: Vec<serde_json::Value> = kernel
        .service_list()
        .iter()
        .map(|(name, realm, owner)| {
            serde_json::json!({ "name": name, "realm": realm.to_string(), "owner": owner })
        })
        .collect();

    let policy: Vec<serde_json::Value> = runtime
        .policy_table()
        .into_iter()
        .map(|(kind, consent)| {
            serde_json::json!({
                "kind": kind.as_str(),
                "consent": match consent {
                    Consent::AutoAllow => "auto-allow",
                    Consent::Ask => "ask",
                    Consent::Deny => "deny",
                },
            })
        })
        .collect();

    let network = serde_json::json!({
        "request_timeout_secs": route.network.request_timeout_secs,
        "pool_idle_timeout_secs": route.network.pool_idle_timeout_secs,
        "retry_attempts": route.network.retry_attempts,
        "retry_backoff_ms": route.network.retry_backoff_ms,
    });

    if as_json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "model": route.model,
                "base_url": route.base_url,
                "api_key": if route.api_key.is_empty() { "(not set)" } else { "(set)" },
                "cache_policy": format!("{cache_policy:?}"),
                "network": network,
                "tools": tools,
                "services": services,
                "policy": policy,
                "skill_roots": skills.roots().iter().map(|r| r.display().to_string()).collect::<Vec<_>>(),
                "skill_count": skills.len(),
                "change_kinds": ChangeKind::all().iter().map(|k| k.as_str()).collect::<Vec<_>>(),
            }))?
        );
        return Ok(());
    }

    println!("model:        {}", route.model);
    println!("base url:     {}", route.base_url);
    println!(
        "api key:      {}",
        if route.api_key.is_empty() { "(not set)" } else { "(set)" }
    );
    println!("cache policy: {cache_policy:?}");
    println!(
        "network:      request {}s, pool idle {}s, {} retry(ies) from {}ms",
        route.network.request_timeout_secs,
        route.network.pool_idle_timeout_secs,
        route.network.retry_attempts,
        route.network.retry_backoff_ms,
    );
    println!("\ntools ({}):", tools.len());
    println!("  {}", tools.join(", "));
    println!("\nservices:");
    if services.is_empty() {
        println!("  (none)");
    } else {
        for service in &services {
            println!(
                "  {:<12} {} [{}]",
                service["name"].as_str().unwrap_or(""),
                service["owner"].as_str().unwrap_or(""),
                service["realm"].as_str().unwrap_or(""),
            );
        }
    }
    println!("\nconsent policy (global changes):");
    for entry in &policy {
        println!(
            "  {:<14} {}",
            entry["kind"].as_str().unwrap_or(""),
            entry["consent"].as_str().unwrap_or("")
        );
    }
    println!("\nskill roots:");
    if skills.roots().is_empty() {
        println!("  (none configured)");
    } else {
        for root in skills.roots() {
            println!("  {}", root.display());
        }
    }
    println!("\nskills found: {}", skills.len());
    Ok(())
}


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

/// The route this session runs on, built from the resolved settings.
///
/// Command-line flags win over the stored settings, and a plugin that
/// published a network policy wins over both: it is the more specific choice,
/// made by the pack the user installed for this provider. Everything a client
/// needs is on the route, so the session, a one-off command, and a subagent
/// built later all read the same values.
fn model_route(cli: &Cli, settings: &Settings, kernel: Option<&Kernel>) -> ModelRoute {
    let mut route = ModelRoute::from_settings(settings);
    // The proxy stays explicit: when it is empty, the process's `HTTP_PROXY`
    // and `HTTPS_PROXY` are deliberately ignored. Picking those up silently is
    // how a proxy set for another tool reroutes model traffic and fails with a
    // TLS handshake error that never mentions proxies.
    if let Some(url) = &cli.base_url {
        route.base_url = url.clone();
    }
    if let Some(key) = &cli.api_key {
        route.api_key = key.clone();
    }
    if let Some(name) = &cli.model {
        route.model = name.clone();
    }
    if let Some(effort) = &cli.reasoning_effort {
        route.reasoning_effort = if effort.trim().is_empty() {
            None
        } else {
            Some(effort.clone())
        };
    }
    if let Some(delivery) = &cli.delivery {
        // Checked here rather than passed through as a string: a mistyped mode
        // on the command line should say so before the first request, not after
        // a turn has hung for the whole request timeout.
        let word = delivery.trim();
        if !word.is_empty() && nguruvilu::llm::Delivery::parse(word).is_none() {
            eprintln!(
                "warning: unknown delivery mode {word:?}; expected one of {}. Keeping the configured mode.",
                nguruvilu::llm::Delivery::names().join(", ")
            );
        } else {
            route.delivery = if word.is_empty() { None } else { Some(word.to_string()) };
        }
    }
    if let Some(kernel) = kernel {
        let view = kernel.service_view(nguruvilu::plugin::RealmMap::new());
        if let Some(network) = nguruvilu::network::NetworkHandle::from_view(&view) {
            route.network = network;
        }
    }
    route
}


/// `ngu routes` —— 列出路由表，或调一条。
///
/// 这是"一切皆调用"的可验证入口：表里有什么、能不能调通，一条命令看清。
fn routes_command(call: Option<&str>, raw_args: &str, json: bool) -> Result<()> {
    let dispatch = nguruvilu::kernel_routes::dispatch();

    let Some(name) = call else {
        let names = dispatch.names();
        if json {
            println!("{}", serde_json::to_string_pretty(&names)?);
        } else {
            for line in &names {
                println!("{line}");
            }
            println!();
            println!("{} 条路由。加 --call <名字> --args '<JSON>' 真的调一条。", names.len());
        }
        return Ok(());
    };

    let args: serde_json::Value =
        serde_json::from_str(raw_args).with_context(|| format!("--args 不是 JSON：{raw_args}"))?;

    // 已经在 tokio 的 runtime 里了（`run` 整个跑在 block_on 之内），
    // 再 `block_on` 一次就是 "Cannot start a runtime from within a runtime"。
    // 所以这里用 `block_in_place` + `Handle::block_on`，让出当前 worker 去等，
    // 而不是另起一个 runtime。
    let call = tokio::task::block_in_place(|| {
        tokio::runtime::Handle::current().block_on(dispatch.call(name, args))
    });

    match call {
        Ok(value) => {
            if json {
                println!("{}", serde_json::to_string_pretty(&serde_json::json!({
                    "route": name,
                    "result": value,
                }))?);
            } else {
                println!("{}", serde_json::to_string_pretty(&value)?);
            }
            Ok(())
        }
        Err(e) => {
            if json {
                println!("{}", serde_json::to_string_pretty(&serde_json::json!({
                    "route": name,
                    "error": e,
                }))?);
                std::process::exit(1);
            }
            anyhow::bail!("路由 {name} 失败：{e}")
        }
    }
}

// ---------------------------------------------------------------- the routes

/// 注册 `ngu skills` 与 `ngu assembly` 这两条能自成一体的命令行路径。
///
/// # 为什么只有两条
///
/// `show_runtime` 要拿一个已经装好的 `Runtime` 和 `ModelRoute` —— 那是启动
/// 过程中才有的东西，不是一条路由的入参能表达的。把它压成 JSON 参数会让
/// 「换掉内核运行时报告」变成「模拟启动一遍」，那比直接改源码更难。
///
/// 所以留在这里直接调用；能被替换的是这两条：技能目录怎么列、装配清单怎么读。
pub fn register(dispatch: &nguruvilu::dispatch::Dispatch) {
    use nguruvilu::dispatch::{Args, Chain, Fut};
    use std::sync::Arc;

    let json_flag = |args: &Args| args.get("json").and_then(|v| v.as_bool()).unwrap_or(false);

    dispatch.register(
        "cli.runtime.skills",
        Arc::new(|args: Args, _next: Chain| -> Fut {
            Box::pin(async move {
                let verbose = args.get("verbose").and_then(|v| v.as_bool()).unwrap_or(false);
                let mut skills = SkillRegistry::default();
                skills.scan().map_err(|e| format!("{e:#}"))?;
                list_skills(&skills, verbose, json_flag(&args)).map_err(|e| format!("{e:#}"))?;
                Ok(serde_json::json!({ "ok": true }))
            })
        }),
    );

    dispatch.register(
        "cli.runtime.assembly",
        Arc::new(|args: Args, _next: Chain| -> Fut {
            Box::pin(async move {
                let file = args
                    .get("file")
                    .and_then(|v| v.as_str())
                    .map(PathBuf::from)
                    .ok_or_else(|| "cli.runtime.assembly needs a file".to_string())?;
                let platform = args
                    .get("platform")
                    .and_then(|v| v.as_str())
                    .unwrap_or("linux-x64")
                    .to_string();
                show_assembly(&file, &platform, json_flag(&args)).map_err(|e| format!("{e:#}"))?;
                Ok(serde_json::json!({ "ok": true }))
            })
        }),
    );
}
