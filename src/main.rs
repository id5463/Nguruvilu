//! `ngu` — the Nguruvilu command-line interface.
//!
//! Three run modes over the same kernel:
//!
//! * **interactive** — a terminal REPL (default when stdin is a terminal).
//! * **print** — one prompt in, final text out (`-p`), for scripts and pipes.
//! * **json** — the same run with a structured result (`--json`), which is the
//!   mode other agents drive: it reports the session id, step count, tool
//!   calls, token usage, and the timing breakdown.
//!
//! Plus management subcommands for the parts that are not the loop: skills,
//! assembly manifests, workspace snapshots, and the model catalog.
//!
//! `stdout` carries results only. Progress, tool traces, and diagnostics go to
//! `stderr`, so the output stays pipeable.

use std::io::{IsTerminal, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Context, Result};
use clap::Parser;

use nguruvilu::agent::{Agent, AgentObserver, TurnConfig, TurnSettings};
use nguruvilu::assembly::Assembly;
use nguruvilu::git::GitSnapshot;
use nguruvilu::hotreload::{
    CachePolicy, Change, ChangeKind, ChangePayload, Consent, ModelRoute, Runtime,
};
use nguruvilu::ledger::default_ledger_path;
use nguruvilu::llm::LlmClient;
use nguruvilu::loader::Loader;
use nguruvilu::source::PackSource;
use nguruvilu::plugin::{Kernel, Plugin as PluginTrait};
use nguruvilu::session::{totals_of, JsonlStore, Session, SessionStore, SessionTotals, TurnStats};
use nguruvilu::settings::Settings;
use nguruvilu::window::ContextPolicy;
use nguruvilu::skills::{register_skill_tool, SkillRegistry};

mod cli;

use cli::args::*;
use cli::observer::{PrettyObserver, SessionEnd};
use cli::{events_command, mcp_import_command, plugin_command, show_injections, ui_command};
use cli::config::require_key;

const DEFAULT_SYSTEM_PROMPT: &str = "You are a coding agent. Your working directory is the \
process working directory. Use the tools to inspect and change files, and to run commands. \
Prefer `read` over shelling out to view files, and `edit` for surgical changes. \
Prefer a dedicated tool over a shell command whenever one exists: an `mcp__*` tool is a \
working integration with a real service (its auth, its endpoints, its parsing), and \
re-deriving that work with `bash` and `curl` usually fails in a way that looks like bad \
data rather than a bug. When the task names a service — mail, a course platform, a \
browser, an issue tracker — look for its tools before reaching for the shell. \
Keep answers short and concrete. When you tell the user about a file, give its absolute \
path — a bare or relative name is not something they can open.";


/// Supplies turn settings from a runtime, adding the skill catalog.
///
/// The runtime owns persona, route, and tools; the catalog is composed here
/// because it belongs to the CLI's view of the prompt rather than to the
/// runtime's state. The mission block is here for the same reason: it describes
/// this session's workspace, which the runtime knows nothing about.
/// Whether a pack has taken responsibility for choosing skills.
///
/// The kernel knows one thing here: a pack called `skill-router` said it would
/// answer, and the answer is a mode name in that pack's own settings block. Any
/// mode other than `all`/`full` means the catalog is that pack's business, so
/// the kernel stops writing every skill into the prompt.
///
/// `false` for everything else — no setting, an empty string, `all`, `full` —
/// which is the behaviour every machine had before this existed.
fn skills_are_routed(packs: &std::collections::BTreeMap<String, serde_json::Value>) -> bool {
    packs
        .get("skill-router")
        .and_then(|block| block.get("skill_mode"))
        .and_then(serde_json::Value::as_str)
        .map(|mode| !matches!(mode, "" | "all" | "full"))
        .unwrap_or(false)
}

struct CliConfig {
    runtime: Arc<Mutex<Runtime>>,
    base_prompt: String,
    skills: Arc<Mutex<SkillRegistry>>,
    /// Context window, threshold, and how much survives a compaction.
    policy: Arc<Mutex<Arc<dyn ContextPolicy>>>,
    /// Extra fragments to place in each request.
    ///
    /// Behind a lock and shared with the pack host, so a pack hot-loaded
    /// between turns replaces them for the next turn rather than the next run.
    injection: Arc<Mutex<Arc<dyn nguruvilu::context::InjectionSource>>>,
    /// The workspace whose mission board the prompt should name, once known.
    ///
    /// A cell rather than a plain path because the turn settings are assembled
    /// before the session is resolved: the workspace does not exist until then,
    /// and a resumed session's mission must be named from its very first turn.
    session_workspace: Arc<Mutex<Option<PathBuf>>>,
    /// Whether this process was started minimal. See [`Cli::bare`].
    bare: bool,
    /// Per-pack settings, straight from the settings file.
    ///
    /// Here rather than read off `Runtime` because `Runtime` is the hot-reload
    /// view (tools, route, persona) and pack settings are not part of it. The
    /// kernel does not interpret the block: it looks up `skill-router` because
    /// that is the pack that said it would answer, and reads one string out of
    /// whatever the user put there.
    packs: std::collections::BTreeMap<String, serde_json::Value>,
}

/// The session's route, published so a plugin can build a subagent on it.
///
/// Reads through to the same `Runtime` the turn uses, so a plugin sees the
/// model and proxy in force now rather than the ones at startup.
struct CliModelAccess {
    runtime: Arc<Mutex<Runtime>>,
    skills: Arc<Mutex<SkillRegistry>>,
    base_prompt: String,
    policy: Arc<Mutex<Arc<dyn ContextPolicy>>>,
}

impl nguruvilu::model::ModelAccess for CliModelAccess {
    fn route(&self) -> nguruvilu::hotreload::ModelRoute {
        self.runtime.lock().expect("runtime lock").snapshot().model_route
    }

    fn client(&self) -> anyhow::Result<LlmClient> {
        self.client_for(&self.route().model)
    }

    fn client_for(&self, model: &str) -> anyhow::Result<LlmClient> {
        // The route's own builder, so a subtask is sent with the same proxy,
        // effort, output ceiling, extra fields, and connection behaviour the
        // session itself is using right now.
        self.route().client_for(model)
    }

    fn tools(&self) -> Arc<nguruvilu::tools::ToolRegistry> {
        self.runtime.lock().expect("runtime lock").snapshot().tools
    }

    fn system_prompt(&self) -> String {
        let mut prompt = self
            .runtime
            .lock()
            .expect("runtime lock")
            .snapshot()
            .system_prompt(&self.base_prompt);
        let tool_names = self.tools().names();
        if let Some(catalog) = self.skills.lock().expect("skills lock").catalog_with_tools(&tool_names, None) {
            prompt.push_str("\n\n");
            prompt.push_str(&catalog);
        }
        prompt
    }

    /// The base rules for a nested agent — deliberately without the persona.
    ///
    /// The persona is written for the agent the user talks to. When it says
    /// "you are the planner, delegate what can be delegated", a subtask that
    /// inherits it treats that as its own job and spends its budget spawning
    /// more subtasks instead of doing the work it was handed. The operating
    /// rules (working-directory convention, how to report) and the skill
    /// catalog are what a subtask actually needs, so those stay.
    fn subagent_prompt(&self) -> String {
        let mut prompt = self.base_prompt.clone();
        let tool_names = self.tools().names();
        if let Some(catalog) = self.skills.lock().expect("skills lock").catalog_with_tools(&tool_names, None) {
            prompt.push_str("\n\n");
            prompt.push_str(&catalog);
        }
        prompt
    }

    /// The endpoint's catalog, for `delegate`.
    ///
    /// The CLI has no long-lived catalog to read, so this asks the endpoint and
    /// gives up quietly on failure: a subtask without a model argument works
    /// fine, and blocking registration on the network would make an offline
    /// start impossible.
    fn available_models(&self) -> Vec<String> {
        let client = match self.client() {
            Ok(client) => client,
            Err(_) => return Vec::new(),
        };
        // 这个方法在 turn 内部被调到，而 turn 本身跑在 `main` 的 runtime 里。
        // 另建一个 runtime 再 block_on 会 panic（"Cannot start a runtime from
        // within a runtime"），这就是 `ngu -p` 每次退出都报错的原因。
        // 复用当前 runtime：block_in_place 让出 worker，Handle::block_on 等结果。
        let fetch = async { client.list_models().await.unwrap_or_default() };
        match tokio::runtime::Handle::try_current() {
            Ok(handle) => tokio::task::block_in_place(|| handle.block_on(fetch)),
            // 没有 runtime 时（比如单元测试直接调）才自己建一个。
            Err(_) => {
                let Ok(runtime) = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                else {
                    return Vec::new();
                };
                runtime.block_on(fetch)
            }
        }
    }

    fn context_policy(&self) -> Arc<dyn ContextPolicy> {
        Arc::clone(&self.policy.lock().expect("policy lock"))
    }
}

impl TurnConfig for CliConfig {
    fn settings(&self) -> TurnSettings {
        let snapshot = self.runtime.lock().expect("runtime lock").snapshot();
        let mut system_prompt = snapshot.system_prompt(&self.base_prompt);

        // A mission lives on disk, and nothing else in the request says one
        // exists: without this block a turn starts with the model remembering
        // only what the transcript says, and a long-horizon mission reads as
        // dead until something happens to call `mission_status`. The path is
        // cloned out before the scan, so a slow disk holds no lock.
        let workspace = self
            .session_workspace
            .lock()
            .expect("workspace lock")
            .clone();
        if let Some(workspace) = workspace {
            if let Some(block) = active_mission_block(&workspace) {
                system_prompt.push_str(&block);
            }
        }

        // A bare start has no skills to advertise. The registry is empty in that
        // case so `catalog_with_tools` would answer `None` anyway; the check is
        // here so a future registry that answers `Some("")` cannot leave a stray
        // blank line in a subagent's prompt.
        // A pack that routes skills says so in its own settings block; the kernel
        // reads the answer and does not know what a router is.
        //
        // This is the seam the skill table was published for. Without it the
        // prompt carried every skill on the machine and a router beside them
        // would only add weight: the model reads all fifty *and* a tool that can
        // pick from them. With it, a pack that has taken responsibility for
        // selection turns the automatic catalog off, and the model asks for the
        // two skills that matter.
        //
        // Absent or `all` keeps today's behaviour, so a machine that never
        // configures a router is unchanged.
        let skills_routed = skills_are_routed(&self.packs);
        if !self.bare && !skills_routed {
            let tool_names = snapshot.tools.names();
            if let Some(catalog) = self.skills.lock().expect("skills lock").catalog_with_tools(&tool_names, None) {
                system_prompt.push_str("

");
                system_prompt.push_str(&catalog);
            }
        }

        TurnSettings {
            system_prompt,
            tools: snapshot.tools,
            model: snapshot.model_route.model,
            version: snapshot.version,
            policy: Arc::clone(&self.policy.lock().expect("policy lock")),
            injection: Arc::clone(&self.injection.lock().expect("injection lock")),
            cache_policy: snapshot.cache_policy,
            // The command-line path already assembled its prompt above; the
            // plugin seam exists for the desktop shell, which has a kernel to
            // ask.
            plugin_hooks: None,
        }
    }
}

/// Build the async runtime, run one session, and leave.
///
/// The runtime is deliberately **not** dropped: tearing it down waits on state
/// a spawned MCP server left registered with it, and the process then never
/// exits — after everything it had to say has already been printed. Everything
/// the runtime owned that matters is stopped first (`stop_mcp` on the success
/// path; `McpClient`'s own drop on every other path), so exiting without
/// dropping it is the ending this process chooses rather than an oversight.
fn main() {
    let runtime = match tokio::runtime::Builder::new_multi_thread().enable_all().build() {
        Ok(runtime) => runtime,
        Err(error) => {
            eprintln!("ngu: starting the async runtime: {error}");
            std::process::exit(1);
        }
    };

    let code = match runtime.block_on(run()) {
        Ok(()) => 0,
        Err(error) => {
            eprintln!("ngu: {error:#}");
            1
        }
    };
    std::process::exit(code);
}

fn platform_tag() -> &'static str {
    if cfg!(windows) {
        "win32"
    } else if cfg!(target_os = "macos") {
        "darwin"
    } else {
        "linux"
    }
}

fn parse_cache_policy(raw: &str) -> Result<CachePolicy> {
    match raw.to_ascii_lowercase().as_str() {
        "freshness" | "fresh" => Ok(CachePolicy::Freshness),
        "balanced" | "balance" => Ok(CachePolicy::Balanced),
        "cache-first" | "cachefirst" | "cache" => Ok(CachePolicy::CacheFirst),
        other => Err(anyhow!(
            "unknown cache policy '{other}'; expected freshness, balanced, or cache-first"
        )),
    }
}

/// 走一条命令行的路由，把结果折成 `Result<()>`。
///
/// 命令行这一层的每个子命令都是"做一件事、打印、返回"，所以路由失败时
/// 只需要把消息带上去。集中在这里是因为 8 条路由的折法完全一样。
async fn cli_call(route: &str, args: serde_json::Value) -> Result<()> {
    let dispatch = nguruvilu::kernel_routes::dispatch();
    match dispatch.call(route, args).await {
        Ok(_) => Ok(()),
        Err(message) => Err(anyhow!("{message}")),
    }
}

async fn run() -> Result<()> {
    let cli = Cli::parse();

    let store = JsonlStore::open(cli.home.clone().unwrap_or_else(JsonlStore::default_root))?;

    // Precedence: command-line flag, then stored settings, then environment.
    // `Settings::resolve` already folds the environment in, so a machine-level
    // variable still wins over the file.
    let mut settings = Settings::resolve();
    // The budgets are process globals read when tools are *registered*, so they
    // go in before the kernel builds its table: a tool description written after
    // this point states the numbers the machine actually uses.
    nguruvilu::tools::set_budgets(settings.budgets());
    let api_key = cli
        .api_key
        .clone()
        .unwrap_or_else(|| settings.api_key.clone());
    let cache_policy = parse_cache_policy(&cli.cache_policy)?;

    // Skills are scanned once up front so every mode sees the same catalog.
    //
    // `--bare` starts with no roots at all: a subagent gets its task in the
    // prompt, and a catalog of skills it was never told to use is prompt weight
    // that changes its answers for no reason. The scan is skipped rather than
    // filtered afterwards so the cost goes with it.
    let mut skills = if cli.bare {
        SkillRegistry::with_roots(Vec::new())
    } else {
        SkillRegistry::with_roots(SkillRegistry::default_roots())
    };
    if !cli.bare {
        for dir in &cli.skill_dirs {
            skills.add_root(dir.clone());
        }
    }
    match skills.scan() {
        Ok(report) => {
            // A scan that skipped something is the only trace of a skill the
            // user expects to exist, so it is printed; a clean scan says
            // nothing, because a silent start is the normal start.
            if !report.skipped.is_empty() || !report.collisions.is_empty() {
                eprintln!("[skills] {}", report.summary());
            }
        }
        // The scan is hard to fail by design, but swallowing an Err here would
        // leave the user with no catalog and no reason for it.
        Err(error) => eprintln!("[skills] scan failed: {error:#}"),
    }

    // 路由命令要等 Kernel 建好才有表可列，先记下来。
    let mut pending_routes: Option<(Option<String>, String)> = None;
    match &cli.command {
        Some(Command::Sessions) => return list_sessions(&store, cli.json),
        Some(Command::Show { id }) => return show_session(&store, id),
        Some(Command::Delete { id }) => {
            store.delete(id)?;
            println!("deleted session {id}");
            return Ok(());
        }
        Some(Command::Stats { id, all, cost, prices }) => {
            // 走路由而不是直接调 `stats_command`：一个包覆盖 `cli.stats` 之后，
            // 这一行打印的就是包的那份，内核那份留在链尾当回退。
            let dispatch = nguruvilu::kernel_routes::dispatch();
            let args = serde_json::json!({
                "id": id,
                "all": all,
                "cost": cost,
                "prices": prices.as_ref().map(|p| p.display().to_string()),
                "json": cli.json,
                "home": cli.home.as_ref().map(|p| p.display().to_string()),
            });
            return match dispatch.call("cli.stats", args).await {
                Ok(_) => Ok(()),
                Err(message) => Err(anyhow!("{message}")),
            };
        }
        Some(Command::Export { id, out, format }) => {
            let dispatch = nguruvilu::kernel_routes::dispatch();
            let args = serde_json::json!({
                "id": id,
                "out": out.as_ref().map(|p: &std::path::PathBuf| p.display().to_string()),
                "format": format,
                "json": cli.json,
                "home": cli.home.as_ref().map(|p| p.display().to_string()),
            });
            return match dispatch.call("cli.export", args).await {
                Ok(_) => Ok(()),
                Err(message) => Err(anyhow!("{message}")),
            };
        }
        Some(Command::Doctor) => {
            // The command decides its own exit code, because a report with
            // warnings is a successful run and a report with a failure is not.
            let options = nguruvilu::doctor::Options {
                sessions: store.root().to_path_buf(),
                // The cap this invocation would actually use, so the report
                // answers "why did my turn stop?" with the real number.
                max_steps: Some(cli.max_steps.unwrap_or_else(|| settings.max_steps())),
                skill_dirs: cli.skill_dirs.clone(),
                ..nguruvilu::doctor::Options::default()
            };
            let code = nguruvilu::doctor::print_report(&options, cli.json)?;
            if code != 0 {
                std::process::exit(code);
            }
            return Ok(());
        }
        Some(Command::Models { windows }) => {
            require_key(&api_key)?;
            let client = model_route(&cli, &settings, None).client()?;
            // The plain list keeps its own tolerant reader: it is what scripts
            // and the picker use, and it must not start failing because a
            // gateway's catalog gained a field. Only `--windows` reads the wider
            // shape.
            let mut models: Vec<(String, Option<usize>)> = if *windows {
                client.list_models_with_windows().await?
            } else {
                client
                    .list_models()
                    .await?
                    .into_iter()
                    .map(|id| (id, None))
                    .collect()
            };
            models.sort_by(|a, b| a.0.cmp(&b.0));
            for (id, window) in &models {
                if let Some(tokens) = window {
                    nguruvilu::window::reported_windows().record(id, *tokens);
                }
            }
            if cli.json {
                let entries: Vec<serde_json::Value> = models
                    .iter()
                    .map(|(id, window)| {
                        serde_json::json!({ "id": id, "context_length": window })
                    })
                    .collect();
                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({ "models": entries }))?
                );
            } else if *windows {
                // The id column is wide enough for a gateway's long names, and a
                // model the provider did not price in tokens says so instead of
                // showing a zero that looks like a window of nothing.
                let width = models
                    .iter()
                    .map(|(id, _)| id.chars().count())
                    .max()
                    .unwrap_or(8)
                    .clamp(8, 48);
                for (id, window) in &models {
                    match window {
                        Some(tokens) => println!("{id:<width$}  {tokens:>9}"),
                        None => println!("{id:<width$}  {:>9}", "-"),
                    }
                }
            } else {
                for (id, _) in &models {
                    println!("{id}");
                }
            }
            return Ok(());
        }
        Some(Command::Skills { verbose }) => {
            let args = serde_json::json!({ "verbose": verbose, "json": cli.json });
            return cli_call("cli.runtime.skills", args).await;
        }
        Some(Command::Assembly { file, platform }) => {
            let platform = platform.clone().unwrap_or_else(|| platform_tag().to_string());
            let args = serde_json::json!({
                "file": file.display().to_string(),
                "platform": platform,
                "json": cli.json,
            });
            return cli_call("cli.runtime.assembly", args).await;
        }
        Some(Command::Apply { file, platform }) => {
            let platform = platform.clone().unwrap_or_else(|| platform_tag().to_string());
            return apply_assembly(file, &platform, &skills, cli.json).await;
        }
        Some(Command::Snapshot { action, commit, limit }) => {
            return snapshot_command(&cli, action, commit.as_deref(), *limit, cli.json)
        }
        Some(Command::Pack { dir, out, pin, offline }) => {
            let args = serde_json::json!({
                "dir": dir.display().to_string(),
                "out": out.as_ref().map(|p: &PathBuf| p.display().to_string()),
                "pin": pin,
                "offline": offline,
                "json": cli.json,
            });
            return cli_call("cli.pack.pack", args).await;
        }
        Some(Command::Install { file, into }) => {
            let args = serde_json::json!({
                "file": file.display().to_string(),
                "into": into.as_ref().map(|p: &PathBuf| p.display().to_string()),
                "json": cli.json,
            });
            return cli_call("cli.pack.install", args).await;
        }
        Some(Command::Mcp {
            action: McpAction::Import { file, out, name, license },
        }) => return mcp_import_command(&file, out.as_deref(), name.as_deref(), &license),
        Some(Command::Events { last }) => return events_command(*last, cli.json),
        Some(Command::Packs) => {
            return cli_call("cli.pack.list", serde_json::json!({ "json": cli.json })).await;
        }
        Some(Command::Uninstall { name, all, delete, into }) => {
            let args = serde_json::json!({
                "name": name,
                "all": all,
                "delete": delete,
                "into": into.as_ref().map(|p: &PathBuf| p.display().to_string()),
                "json": cli.json,
            });
            return cli_call("cli.pack.uninstall", args).await;
        }
        Some(Command::Ui { dir }) => return ui_command(dir.as_ref(), cli.json),
        Some(Command::Verify { file }) => {
            let args = serde_json::json!({
                "file": file.display().to_string(),
                "json": cli.json,
            });
            return cli_call("cli.pack.verify", args).await;
        }
        Some(Command::Plugin { action }) => return plugin_command(action, cli.json),
        Some(Command::Config { action }) => {
            // 三个只读动作走路由（换配置源、换展示），写入仍直接调：
            // `Set` 带着十几个 flag，压成 JSON 只是换一种拼 CLI 的方式。
            let routed = match action {
                Some(ConfigAction::Path) => Some("path"),
                Some(ConfigAction::Routes) => Some("routes"),
                Some(ConfigAction::Show) => Some("show"),
                None => Some("show"),
                _ => None,
            };
            if let Some(action_name) = routed {
                let dispatch = nguruvilu::kernel_routes::dispatch();
                let args = serde_json::json!({ "action": action_name, "json": cli.json });
                return match dispatch.call("cli.config", args).await {
                    Ok(_) => Ok(()),
                    Err(message) => Err(anyhow!("{message}")),
                };
            }
            return config_command(action.as_ref(), cli.json);
        }
        Some(Command::Injections { r#for }) => {
            return show_injections(r#for.as_deref(), cli.json)
        }
        Some(Command::Runtime) => {
            // Described below, once the kernel is assembled: the tool table
            // and the published services do not exist until then.
        }
        Some(Command::Routes { call, args }) => {
            // 内核默认路由在建 Kernel 之后才装好，所以这里不能提前 return。
            // 记下来，装配完再执行。
            pending_routes = Some((call.clone(), args.clone()));
        }
        None => {}
    }

    if let Some(dir) = &cli.cwd {
        std::env::set_current_dir(dir)
            .with_context(|| format!("changing directory to {}", dir.display()))?;
    }

    // Describing the session that would run needs no key; running one does.
    let describe_runtime = matches!(&cli.command, Some(Command::Runtime));
    if !describe_runtime {
        require_key(&api_key)?;
    }

    // The kernel owns the tool table; skills add one tool to it.
    let mut kernel = Kernel::new();
    // The tools a conversation cannot work without. Installed as a call rather
    // than inside `Kernel::new`, so a pack can replace any of them: every
    // registration below goes through `ConflictPolicy::Override`.
    kernel.install_bootstrap()?;
    // The agent can build, install, load, and unload packs itself; the host
    // applies the load and unload between turns.
    let pending_packs = Arc::new(nguruvilu::tools::pack::PendingQueue::new());
    nguruvilu::tools::pack::register_with(kernel.tools_mut(), Some(Arc::clone(&pending_packs)))?;
    // Every `builtin:` plugin this binary carries, defined in one place so the
    // CLI, `ngu apply`, and the desktop cannot disagree about what exists.
    // Defining is not enabling: a pack's assembly still decides what loads.
    kernel.define_builtins();
    // 一切皆调用的接线：内核默认实现进路由表，工具表进活对象槽。
    // 这里装一次，之后内核和包走同一条路 —— 否则"暴露"只是对外的假象。
    nguruvilu::kernel_routes::route_everything(&kernel);
    // 命令行自己的默认实现也进表：`ngu stats` 打印什么由 `cli.stats` 这条路由
    // 决定，所以一个包覆盖同名路由就能换掉那份报告。放进来的位置在内核接线之后，
    // 因为它进的是同一张表。
    cli::stats::register(nguruvilu::kernel_routes::dispatch());
    cli::export::register(nguruvilu::kernel_routes::dispatch());
    cli::config::register(nguruvilu::kernel_routes::dispatch());
    cli::pack::register(nguruvilu::kernel_routes::dispatch());
    cli::runtime::register(nguruvilu::kernel_routes::dispatch());
    // The judge keeps its settings in the host's cell rather than in the plugin,
    // so a settings save can re-apply the fibers without rebuilding anything.
    // The kernel's own state, published where a pack can reach and change it:
    // settings, route, persona. Reading stays direct (`published.settings()`),
    // because routing the hot path through a string lookup would make every turn
    // pay for a feature it may not use.
    let published = nguruvilu::state::Published::install(&mut kernel, settings.clone());
    settings = published.settings();
    if cli.bare {
        // A subagent is not the conversation that started it. The persona is
        // the one prompt input that changes *how* the model talks, and a worker
        // answering one question has no business speaking in somebody else's
        // voice. Cleared after `install` so a pack that sets one later is still
        // overridden — the flag means "this process", not "before packs".
        *published.persona.write().unwrap_or_else(|poisoned| poisoned.into_inner()) =
            nguruvilu::state::Persona::default();
    }
    if !skills.is_empty() {
        register_skill_tool(kernel.tools_mut(), Arc::new(skills.clone()))?;
    }

    // An assembly manifest loads before the session starts, so its plugins,
    // MCP servers, and skills are in place for the first turn.
    let mut contents: Vec<nguruvilu::content::PackContent> = Vec::new();
    // The MCP servers the session started, kept alive for as long as it runs
    // and stopped on the way out — see `stop_mcp`.
    let mut mcp_clients: Vec<Arc<nguruvilu::mcp::McpClient>> = Vec::new();
    if let Some(file) = &cli.assembly {
        let platform = platform_tag();
        let assembly = Assembly::from_file(file)?;
        let plan = assembly.plan(platform)?;

        // Prefer the pack's own identity manifest over the file name, so the
        // ledger records which pack a contribution came from rather than which
        // file happened to describe it.
        let pack = file
            .parent()
            .and_then(|dir| nguruvilu::pack::read_manifest(dir).ok())
            .map(|manifest| format!("{}-{}", manifest.name, manifest.version_id))
            .unwrap_or_else(|| {
                file.file_stem()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_else(|| "pack".into())
            });

        let mut loader = Loader::new(kernel, default_ledger_path(), pack).with_pack_dir(
            file.parent().map(PathBuf::from).unwrap_or_else(|| PathBuf::from(".")),
        );
        let report = loader.apply(&plan).await?;

        for entry in &report.loaded {
            eprintln!("loaded {entry}");
        }
        for entry in &report.failed {
            eprintln!("failed {}:{}: {}", entry.kind, entry.id, entry.error);
        }
        for entry in &report.skipped {
            eprintln!("skipped {}: {}", entry.id, entry.reason);
        }

        // Take the kernel back. The MCP servers it started stay up — their
        // tools are in this kernel and are meant to be callable for as long as
        // this conversation runs — and its skill roots join ours rather than
        // replacing them.
        let (loaded, pack_skills, servers) = loader.finish().await;
        kernel = loaded;
        mcp_clients.extend(servers);
        nguruvilu::loader::merge_skills(&mut kernel, &mut skills, pack_skills)?;

        // Apply what the pack carries beyond its entries. Appearance lands on
        // the kernel now; the rest describes this session, and is applied below
        // once the settings it overrides exist.
        let pack_dir = file
            .parent()
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("."));
        // A bare start skips pack content: persona, themes, look, and the
        // session defaults a pack overrides. Those describe how a *conversation*
        // should look; a subagent answering one question has no page to draw and
        // no defaults worth inheriting.
        if !cli.bare {
            if let Ok(manifest) = nguruvilu::pack::read_manifest(&pack_dir) {
                contents.push(nguruvilu::content::apply(&mut kernel, &pack_dir, &manifest)?);
            }
        }
    }

    // Installed packs load the same way the window loads them: every enabled
    // pack, in name order, so the command line and the desktop agree on what a
    // conversation has. An unloaded pack keeps its files and is skipped — it
    // can be brought back with `pack` action `load` without reinstalling.
    //
    // Before any of that, the packs this build ships are placed: a program
    // that has to be told to install its own features is not finished. A
    // failure here is reported rather than fatal — a read-only or full disk
    // must not stop a session from starting.
    match nguruvilu::preinstall::seed(&nguruvilu::pack::default_packs_dir()).await {
        Ok(placed) if !placed.is_empty() => eprintln!("[preinstall] {}", placed.join(", ")),
        Ok(_) => {}
        Err(error) => eprintln!("[preinstall] {error:#}"),
    }
    // `--packs none` (or `no`) means the base tools and nothing else. A list is
    // matched by manifest name, so a caller names what it wants rather than
    // which directory happens to hold it.
    let wanted: Option<Vec<String>> = match cli.packs.trim() {
        "all" | "" => None,
        "none" | "no" => Some(Vec::new()),
        list => Some(
            list.split(',')
                .map(str::trim)
                .filter(|name| !name.is_empty())
                .map(str::to_string)
                .collect(),
        ),
    };
    for pack in nguruvilu::pack::installed(&nguruvilu::pack::default_packs_dir())? {
        if !pack.enabled {
            eprintln!("[pack] {} is unloaded; files kept", pack.manifest.name);
            continue;
        }
        if let Some(list) = &wanted {
            if !list.iter().any(|name| name == &pack.manifest.name) {
                continue;
            }
        }
        // A hand-placed pack carries no assembly and would silently load
        // nothing; render it from the manifest (reported, not fatal — one
        // pack must not stop the session from starting).
        let assembly = match nguruvilu::pack::ensure_assembly(&pack.path) {
            Ok(assembly) => assembly,
            Err(error) => {
                eprintln!("[pack] {} assembly: {error:#}", pack.manifest.name);
                None
            }
        };
        if let Some(assembly) = &assembly {
            let label = format!("{}-{}", pack.manifest.name, pack.manifest.version_id);
            eprintln!("[pack] loading {label}");
            let (pack_skills, report, servers) = nguruvilu::loader::load_pack(
                &mut kernel,
                &default_ledger_path(),
                assembly,
                &label,
            )
            .await?;
            mcp_clients.extend(servers);
            nguruvilu::loader::merge_skills(&mut kernel, &mut skills, pack_skills)?;
            for entry in &report.loaded {
                eprintln!("[pack] loaded {entry}");
            }
            // Reuse is the normal outcome for a capability two packs both
            // declare — delegate is wanted by the ticket board, the research
            // pack, and a plain starter install. Saying so is what makes "only
            // one process is running" visible rather than inferred.
            for entry in &report.reused {
                eprintln!("[pack] reused {entry} (already provided by another pack)");
            }
            for entry in &report.failed {
                eprintln!("[pack] failed {}:{}: {}", entry.kind, entry.id, entry.error);
            }
            // Skips are decisions too: a silently skipped entry is how a pack
            // ends up "installed but nothing happened".
            for entry in &report.skipped {
                eprintln!("[pack] skipped {}: {}", entry.id, entry.reason);
            }
        }
        match nguruvilu::pack::read_manifest(&pack.path) {
            Err(error) if cli.bare => {
                // Nothing to apply, so nothing to report: a bare start does not
                // read content and a missing manifest is not news to it.
                let _ = error;
            }
            Ok(manifest) if cli.bare => {
                let _ = manifest;
            }
            Ok(manifest) => match nguruvilu::content::apply(&mut kernel, &pack.path, &manifest) {
                Ok(content) => contents.push(content),
                // Reported rather than fatal: one pack's content must not stop
                // the session from starting.
                Err(error) => eprintln!("[pack] {} content: {error:#}", pack.manifest.name),
            },
            Err(error) => eprintln!("[pack] {}: {error:#}", pack.manifest.name),
        }
    }

    // `ngu routes` 要等 assembly 和已装包都进内核之后才答，
    // 否则它看不到包的工具 —— 那正是验证包的时候要看的。
    nguruvilu::kernel_routes::route_everything(&kernel);
    if let Some((call, raw_args)) = pending_routes.take() {
        return routes_command(call.as_deref(), &raw_args, cli.json);
    }

    let tools = Arc::new(kernel.tools().clone());

    // One route for the session, built from the resolved settings: the flags
    // win over the stored ones, and a plugin that published a network policy
    // wins over both. It is rebuilt after the pack's content lands, because a
    // pack may replace any of these values.
    let route = model_route(&cli, &settings, Some(&kernel));

    let mut runtime = Runtime::new(Arc::clone(&tools))
        .with_route(route)
        .with_skill_roots(
            skills
                .roots()
                .iter()
                .map(|r| r.display().to_string())
                .collect(),
        );

    // Cache policy is presentation, so it is a runtime setting rather than a
    // flag the loop reads directly.
    runtime.apply(Change::session(
        "cli",
        ChangePayload::CachePolicy(cache_policy),
    ))?;

    let base_prompt = cli.system.clone().unwrap_or_else(|| DEFAULT_SYSTEM_PROMPT.to_string());

    // The context policy comes from settings, so `--context-window` and the
    // settings file both land here. A provider-reported window would be folded
    // in the same way once the catalog has been fetched.
    //
    // Each pack's content replaces the standing one: the pack is the unit the
    // user chose, so it decides. They are folded in load order, so the last
    // pack to speak is the one in force — the same rule the window follows.
    let injection: Arc<Mutex<Arc<dyn nguruvilu::context::InjectionSource>>> =
        Arc::new(Mutex::new(Arc::new(
            nguruvilu::context::InjectionEngine::load_default(),
        )));
    // The user's own settings, before any pack folds over them: a drain's
    // set-shaped recompute starts here, so unloading a pack takes back what
    // it brought while the stored values come back with it.
    let pack_baseline = settings.clone();
    let pack_ran = !contents.is_empty();
    for content in contents {
        adopt_content(content, &mut settings, &mut runtime, &injection)?;
    }
    // `--persona` outranks every pack: applied *after* adoption, or a pack's
    // soul would silently replace what the user asked for on this run.
    if let Some(persona) = &cli.persona {
        runtime.apply(Change::session(
            "cli",
            ChangePayload::Persona(persona.clone()),
        ))?;
    }

    // A pack's numbers replace the standing ones, so the route is rebuilt from
    // what the settings now say — and the client is built from that route
    // rather than from the values captured before the pack loaded. This is
    // also where a plugin's network policy is folded in: assembly plugins are
    // loaded by now, and the built-in service below publishes the settings
    // only when no plugin provided one.
    let route = model_route(&cli, &settings, Some(&kernel));
    if pack_ran {
        runtime.apply(Change::session(
            "pack",
            ChangePayload::ModelRoute(route.clone()),
        ))?;
    }
    // Published for plugins that want to read the numbers in force. A pack
    // that brought its own provider keeps it: install() yields rather than
    // failing the fiber over an occupied service.
    nguruvilu::network::install(&mut kernel, settings.network.clone())?;
    let client = route.client()?;

    let policy: Arc<dyn ContextPolicy> = Arc::new(settings.context_policy(None));

    // The tool deadline is a process global: bash and every other deadline-less
    // tool fall back to it, so one settings key covers the whole tool surface.
    nguruvilu::tools::set_budgets(settings.budgets());

    let runtime = Arc::new(Mutex::new(runtime));
    let skills = Arc::new(Mutex::new(skills));
    let policy = Arc::new(Mutex::new(policy));

    // The rest of the kernel's own state, published beside `settings`, `route`
    // and `persona`. Not a special case for any one pack: the rule is that
    // kernel state is reachable, and these three were the ones still held in a
    // local the kernel did not share.
    //
    // `skills` is the one that was actually missed. A prompt is assembled from
    // every skill on the machine, and a pack that wants to choose *which* had no
    // way to say so — it could add a skill, never narrow the set. The same
    // `Arc<Mutex<_>>` the kernel reads is what gets published, so a pack's write
    // lands in the memory the next turn reads rather than in a copy.
    kernel.publish_state_arc(
        nguruvilu::state::SKILLS,
        Arc::clone(&skills) as Arc<dyn std::any::Any + Send + Sync>,
    );
    kernel.publish_state_arc(
        nguruvilu::state::POLICY,
        Arc::clone(&policy) as Arc<dyn std::any::Any + Send + Sync>,
    );
    kernel.publish_state_arc(
        nguruvilu::state::TOOLS,
        Arc::clone(&tools) as Arc<dyn std::any::Any + Send + Sync>,
    );

    // Published before anything can ask for it: a plugin loaded from a pack
    // needs the service to exist already.
    let access: Arc<dyn nguruvilu::model::ModelAccess> = Arc::new(CliModelAccess {
        runtime: Arc::clone(&runtime),
        skills: Arc::clone(&skills),
        base_prompt: base_prompt.clone(),
        policy: Arc::clone(&policy),
    });
    nguruvilu::model::install(&mut kernel, Arc::clone(&access))?;

    // A pack may have been waiting on the route above; now that it is published
    // the kernel activates it, and the turn's tool snapshot — taken long before
    // any of this — is taken again so the session sees whatever just arrived.
    kernel.refresh()?;
    // A refusal is not an error for this run: the session keeps the table it
    // already had and nothing downstream changes. It is still worth a line,
    // because the only other symptom of a denied swap is that a pack's tools
    // never appeared, and the reason the policy wrote would be read by nobody.
    if let nguruvilu::hotreload::ApplyOutcome::Denied(reason) = runtime
        .lock()
        .expect("runtime lock")
        .adopt_kernel_tools(&kernel)
    {
        eprintln!("[pack] the adopted tool table was refused: {reason}");
    }

    // `runtime` describes the session that would run, so it is answered from
    // the assembled kernel: the tools a turn would actually see, the services
    // a plugin could find, and the route in force after any pack loaded.
    if describe_runtime {
        let guard = runtime.lock().expect("runtime lock");
        let skills = skills.lock().expect("skills lock");
        let result = show_runtime(&route, &guard, cache_policy, &skills, cli.json, &kernel);
        drop(skills);
        drop(guard);
        stop_mcp(&mcp_clients).await;
        return result;
    }

    // Filled once the session's workspace is resolved, below: the turn settings
    // are assembled before that, so this cell is how the workspace reaches them.
    let session_workspace: Arc<Mutex<Option<PathBuf>>> = Arc::new(Mutex::new(None));

    // What each pack put in its own settings block. Copied once here so a later
    // settings save in the same process does not move a prompt out from under a
    // running turn — the same rule the settings snapshot follows.
    let pack_blocks = settings.packs.clone();

    let config: Arc<dyn TurnConfig> = Arc::new(CliConfig {
        runtime: Arc::clone(&runtime),
        base_prompt,
        skills: Arc::clone(&skills),
        policy: Arc::clone(&policy),
        injection: Arc::clone(&injection),
        session_workspace: Arc::clone(&session_workspace),
        bare: cli.bare,
        packs: pack_blocks.clone(),
    });

    let snapshotter = if cli.git_snapshot {
        let workspace = std::env::current_dir()?;
        let git = GitSnapshot::open(&workspace)?;
        if git.is_available() {
            eprintln!("git snapshots enabled for {}", workspace.display());
            Some(git)
        } else {
            eprintln!("git is not available; snapshots disabled");
            None
        }
    } else {
        None
    };

    // Resolve the session: explicit id, else a fresh one.
    let mut session = if cli.new_session {
        Session::new(Some(route.model.clone()))
    } else if let Some(id) = &cli.session {
        store
            .load(id)?
            .ok_or_else(|| anyhow!("session not found: {id}"))?
    } else {
        Session::new(Some(route.model.clone()))
    };
    let is_new = !store.root().join(format!("{}.jsonl", session.id)).exists();

    // Pin this session and its workspace to the tool table — the same thing
    // the desktop does on open_session. Path-taking tools (missions, and any
    // workspace-relative resolution) ask `current_workspace()` at call time;
    // without this pin a one-shot CLI turn has no workspace at all and every
    // workspace tool refuses to run.
    let workspace =
        nguruvilu::session::ensure_workspace(&nguruvilu::settings::data_dir(), &session.id)?;
    session.cwd = Some(workspace.display().to_string());
    // The workspace the tools resolve against is also the board the prompt
    // reads; set from one path so the model can never be told about a mission
    // this turn cannot advance. `/new` starts a fresh session but does not move
    // this pin, so the cell is written once per process.
    *session_workspace.lock().expect("workspace lock") = Some(workspace.clone());
    kernel.set_session(Some(session.id.clone()), Some(workspace));

    // The session file is created on first write, never here. Creating it up
    // front leaves an empty file behind whenever a run produces nothing — a
    // failed request, a bare `--new-session` — and empty sessions are noise the
    // user has to clean up.
    let ensure_session = |session: &Session| -> Result<()> {
        if is_new {
            store.create(session)?;
        }
        Ok(())
    };

    let prompt = match cli.prompt.clone() {
        Some(p) => Some(p),
        None => {
            if std::io::stdin().is_terminal() {
                None
            } else {
                let mut buffer = String::new();
                std::io::stdin().read_to_string(&mut buffer)?;
                let trimmed = buffer.trim().to_string();
                if trimmed.is_empty() {
                    None
                } else {
                    Some(trimmed)
                }
            }
        }
    };

    let observer = Arc::new(PrettyObserver { quiet: cli.quiet || cli.json });
    // Audit + session boundary: this process runs one session (or switches it
    // with /new), and the log joins its turns through this pair. The disposer
    // is bound to `_audit` on purpose — a disposer unsubscribes only when
    // called, so the log outlives this scope without holding anything open.
    let events = kernel.events();
    let _audit = nguruvilu::events::install_audit(&events, nguruvilu::events::audit_path());
    events.emit(
        nguruvilu::events::SESSION_START,
        &serde_json::json!({ "session": &session.id }),
    );

    match prompt {
        Some(text) => {
            // One end per run: this guard fires when the arm — the whole
            // one-shot path — finishes, early `?` returns included (Drop
            // does not care about `?`).
            let _session_end = SessionEnd::new(events.clone(), session.id.clone());
            if let Some(git) = &snapshotter {
                git.snapshot("before turn")?;
            }

            events.emit(
                nguruvilu::events::PROMPT_SUBMIT,
                &serde_json::json!({
                    "session": &session.id,
                    "text": nguruvilu::events::truncate(&text, 4000),
                }),
            );
            events.emit(
                nguruvilu::events::TURN_START,
                &serde_json::json!({ "session": &session.id }),
            );
            let turn_started = std::time::Instant::now();
            let original_len = session.messages.len();
            let mut agent = Agent::new(client, Arc::clone(&config), session.messages.clone())
                .with_max_steps(cli.max_steps.unwrap_or_else(|| settings.max_steps()))
                // A pack may have installed its own loop; without this line that
                // pack's code shipped and never ran — which was the state of the
                // `with_loop` chain before: written, tested, unreachable.
                .with_installed_loop()
                .with_observer(observer)
                // This CLI turn's session, fixed for the turn: a plugin that
                // re-pins the registry mid-turn (or a second process) cannot
                // move this turn's workspace or job attribution.
                .with_identity(nguruvilu::tools::ToolIdentity::for_session(
                    session.id.clone(),
                    session.cwd.as_deref().map(std::path::PathBuf::from),
                ));
            let outcome = agent.run(&text).await;
            let turn_ms = turn_started.elapsed().as_millis() as u64;
            match &outcome {
                Ok(done) if done.error.is_none() => events.emit(
                    nguruvilu::events::TURN_COMPLETE,
                    &serde_json::json!({
                        "session": &session.id,
                        "ok": true,
                        "duration_ms": turn_ms,
                        "steps": done.steps,
                        "tool_calls": done.tool_calls,
                        "retries": done.retries,
                    }),
                ),
                Ok(done) => events.emit(
                    nguruvilu::events::TURN_COMPLETE,
                    &serde_json::json!({
                        "session": &session.id,
                        "ok": false,
                        "duration_ms": turn_ms,
                        "steps": done.steps,
                        "retries": done.retries,
                        "error": nguruvilu::events::truncate(
                            done.error.as_deref().unwrap_or("turn failed"),
                            1000
                        ),
                    }),
                ),
                Err(error) => events.emit(
                    nguruvilu::events::TURN_COMPLETE,
                    &serde_json::json!({
                        "session": &session.id,
                        "ok": false,
                        "duration_ms": turn_ms,
                        "error": nguruvilu::events::truncate(&format!("{error:#}"), 1000),
                    }),
                ),
            }
            // On a hard failure the outcome itself is gone, but the agent
            // still holds every message it committed before dying: those are
            // persisted below like any other partial turn.
            let outcome = match outcome {
                Ok(outcome) => outcome,
                Err(error) => {
                    ensure_session(&session)?;
                    // A compaction mid-turn rewrites the history, so it can be
                    // *shorter* than it was when the turn started. Indexing at
                    // `original_len` then panics — and a panic here swallows the
                    // real error, reporting a crash where the honest answer is
                    // the provider failure. A saturated slice keeps the failure
                    // a failure: when the history was compacted there is no
                    // new-message run to salvage, and the earlier turns are
                    // already recorded.
                    let salvaged = agent
                        .messages()
                        .get(original_len..)
                        .unwrap_or(&[])
                        .to_vec();
                    session.messages.extend(salvaged.clone());
                    store.append(&session.id, &salvaged)?;
                    // The turn happened even though nothing about its cost was
                    // reported: one row keeps the turn count honest.
                    record_turn_stats(&store, &session.id, &TurnStats::unreported());
                    return Err(error);
                }
            };

            ensure_session(&session)?;
            session.messages.extend(outcome.new_messages.clone());
            store.append(&session.id, &outcome.new_messages)?;

            // Record what was injected, so the transcript explains what the model
            // was actually sent.
            for record in &outcome.injections {
                store.append_injection(&session.id, record)?;
                if !cli.quiet {
                    eprintln!(
                        "[injected {}: {} tokens]",
                        record.activated.join(", "),
                        record.budget_used
                    );
                }
            }

            // Record compactions after the messages they replace, so a reload
            // reduces to the same history the model saw.
            for compaction in &outcome.compactions {
                store.append_compaction(&session.id, compaction)?;
                if !cli.quiet {
                    eprintln!(
                        "[compacted {} messages into a summary of {} chars]",
                        compaction.replaced,
                        compaction.summary.chars().count()
                    );
                }
            }
            // The live history must match what was recorded, or the next turn
            // would send messages that are no longer part of the conversation.
            if !outcome.compactions.is_empty() {
                session.messages = agent.messages().to_vec();
            }

            // What the turn cost, recorded before the failure return below: a
            // turn that stopped on an error still spent its tokens, and a
            // ledger that only counts happy turns is a ledger that lies.
            record_turn_stats(&store, &session.id, &TurnStats::from_outcome(&outcome));

            // The turn stopped on a failure — everything it produced is on
            // disk by now, so leave with a non-zero status instead of letting
            // a half-finished run pass as a finished one.
            if let Some(error) = &outcome.error {
                return Err(anyhow::anyhow!(error.clone()));
            }

            // Apply any pack load or unload the turn asked for, before the
            // report goes out: from here on the tool table is a different one,
            // and this is the moment it is read again.
            PackHost {
                cli: &cli,
                kernel: &mut kernel,
                skills: &skills,
                runtime: &runtime,
                settings: &mut settings,
                injection: &injection,
                queue: &pending_packs,
                mcp: &mut mcp_clients,
                persona_override: cli.persona.clone(),
                baseline: pack_baseline,
            }
            .drain()
            .await?;

            let snapshot = if let Some(git) = &snapshotter {
                git.snapshot("after turn")?
            } else {
                None
            };

            if cli.json {
                let payload = serde_json::json!({
                    "ok": true,
                    "session_id": session.id,
                    "text": outcome.text,
                    "steps": outcome.steps,
                    "tool_calls": outcome.tool_calls,
                    "usage": {
                        "input": outcome.usage.input,
                        "output": outcome.usage.output,
                        "cached": outcome.usage.cached,
                        "cache_created": outcome.usage.cache_created,
                        "reasoning": outcome.usage.reasoning,
                    },
                    "timing": {
                        "model_ms": outcome.timing.model_ms,
                        "tools_ms": outcome.timing.tools_ms,
                    },
                    "config_version": outcome.config_version,
                    "snapshot": snapshot.as_ref().map(|s| s.commit.clone()),
                    "messages": outcome.new_messages,
                });
                println!("{}", serde_json::to_string_pretty(&payload)?);
            } else {
                if cli.quiet {
                    println!("{}", outcome.text);
                } else {
                    println!();
                }
                if !cli.quiet {
                    eprintln!(
                        "\n[session {} | {} steps | {} tool calls | {} in / {} out ({} cached, {} written to cache) | model {} ms, tools {} ms]",
                        session.id,
                        outcome.steps,
                        outcome.tool_calls,
                        outcome.usage.input,
                        outcome.usage.output,
                        outcome.usage.cached,
                        outcome.usage.cache_created,
                        outcome.timing.model_ms,
                        outcome.timing.tools_ms
                    );
                    if let Some(snapshot) = &snapshot {
                        eprintln!("[snapshot {}]", snapshot.short());
                    }
                }
            }
        }
        None => {
            interactive(
                client,
                config,
                session,
                &store,
                cli.max_steps.unwrap_or_else(|| settings.max_steps()),
                snapshotter,
                PackHost {
                    cli: &cli,
                    kernel: &mut kernel,
                    skills: &skills,
                    runtime: &runtime,
                    settings: &mut settings,
                    injection: &injection,
                    queue: &pending_packs,
                    mcp: &mut mcp_clients,
                    persona_override: cli.persona.clone(),
                    baseline: pack_baseline,
                },
            )
            .await?;
        }
    }

    stop_mcp(&mcp_clients).await;
    Ok(())
}

/// Stop every MCP server this session started.
///
/// Before the runtime tears down rather than at the moment loading finishes:
/// a server still registered with a runtime that is shutting down is what
/// keeps the process alive after everything it had to say has been printed.
/// Loading keeps them running; this is the other half of that decision.
async fn stop_mcp(clients: &[Arc<nguruvilu::mcp::McpClient>]) {
    for client in clients {
        // Bounded: a server that will not die must not keep the CLI alive
        // after everything it has printed. `shutdown_within` kills on timeout.
        client
            .shutdown_within(std::time::Duration::from_secs(3))
            .await;
    }
}

/// Record one turn's usage in its session file.
///
/// One helper for every recording site, so the three callers cannot drift
/// apart. A failure is reported rather than fatal: the turn itself is already
/// on disk, and losing the accounting must not turn a finished turn into a
/// failed one in the caller's eyes.
fn record_turn_stats(store: &JsonlStore, id: &str, stats: &TurnStats) {
    if let Err(error) = store.append_stats(id, stats) {
        eprintln!("[stats] {error:#}");
    }
}



