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
use clap::{Parser, Subcommand};

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
use nguruvilu::session::{JsonlStore, Session, SessionStore};
use nguruvilu::settings::Settings;
use nguruvilu::window::ContextPolicy;
use nguruvilu::skills::{register_skill_tool, SkillRegistry};

const DEFAULT_SYSTEM_PROMPT: &str = "You are a coding agent. Your working directory is the \
process working directory. Use the tools to inspect and change files, and to run commands. \
Prefer `read` over shelling out to view files, and `edit` for surgical changes. Keep answers \
short and concrete.";

#[derive(Parser, Debug)]
#[command(
    name = "ngu",
    version,
    about = "Nguruvilu — an independent agent kernel (four tools, a loop, everything else is loaded)"
)]
struct Cli {
    /// One-shot prompt. When omitted, stdin is read if it is not a terminal.
    #[arg(short, long)]
    prompt: Option<String>,

    /// Model id.
    #[arg(short, long, env = "NGU_MODEL")]
    model: Option<String>,

    /// API base URL, including the version segment.
    #[arg(long, env = "NGU_BASE_URL")]
    base_url: Option<String>,

    /// API key.
    #[arg(long, env = "NGU_API_KEY")]
    api_key: Option<String>,

    /// Continue an existing session by id.
    #[arg(short, long)]
    session: Option<String>,

    /// Start a new session even when `--session` would resolve one.
    #[arg(long)]
    new_session: bool,

    /// Emit a structured JSON result instead of plain text.
    #[arg(long, global = true)]
    json: bool,

    /// Suppress progress output.
    #[arg(short, long)]
    quiet: bool,

    /// Maximum model steps in one turn.
    #[arg(long, default_value_t = 50)]
    max_steps: usize,

    /// System prompt override.
    #[arg(long)]
    system: Option<String>,

    /// Working directory for the session.
    #[arg(long, global = true)]
    cwd: Option<PathBuf>,

    /// Session storage root.
    #[arg(long, env = "NGU_HOME", global = true)]
    home: Option<PathBuf>,

    /// Extra skill search root (repeatable).
    #[arg(long = "skill-dir", global = true)]
    skill_dirs: Vec<PathBuf>,

    /// Reasoning effort: minimal, low, medium, or high.
    #[arg(long, env = "NGU_REASONING_EFFORT", global = true)]
    reasoning_effort: Option<String>,

    /// Persona text, applied as a session-scoped hot change.
    #[arg(long)]
    persona: Option<String>,

    /// Cache policy: freshness, balanced, or cache-first.
    #[arg(long, default_value = "balanced")]
    cache_policy: String,

    /// Snapshot the workspace with git before and after each turn.
    #[arg(long)]
    git_snapshot: bool,

    /// Load an assembly manifest before running.
    #[arg(long)]
    assembly: Option<PathBuf>,

    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// List stored sessions.
    Sessions,
    /// Print a stored session's transcript.
    Show {
        /// Session id.
        id: String,
    },
    /// Delete a stored session.
    Delete {
        /// Session id.
        id: String,
    },
    /// List models available on the configured route.
    Models,
    /// List skills found under the configured roots.
    Skills {
        /// Also print each skill's full instructions.
        #[arg(long)]
        verbose: bool,
    },
    /// Validate an assembly manifest and print its load plan.
    Assembly {
        /// Path to `assembly.yaml`.
        file: PathBuf,
        /// Platform to plan for (win32, darwin, linux).
        #[arg(long)]
        platform: Option<String>,
    },
    /// Apply an assembly manifest: load its plugins, MCP servers, and skills.
    Apply {
        /// Path to `assembly.yaml`.
        file: PathBuf,
        /// Platform to plan for.
        #[arg(long)]
        platform: Option<String>,
    },
    /// Inspect or restore workspace snapshots.
    Snapshot {
        /// `list` or `restore`.
        #[arg(default_value = "list")]
        action: String,
        /// Commit to restore, for `restore`.
        commit: Option<String>,
        /// Maximum snapshots to list.
        #[arg(long, default_value_t = 20)]
        limit: usize,
    },
    /// Build a `.dshpack` archive from a pack directory.
    Pack {
        /// Pack directory (must contain dsh.index.json).
        dir: PathBuf,
        /// Output archive path. Defaults to <name>-<version>.dshpack.
        #[arg(short, long)]
        out: Option<PathBuf>,
        /// Fetch every reference and write its hash into the manifest.
        ///
        /// Without a hash an install cannot use the cache — there is nothing to
        /// look the content up by — so every install re-downloads. Pinning also
        /// makes the pack reproducible: the bytes are fixed rather than
        /// whatever the branch happens to hold today.
        #[arg(long)]
        pin: bool,
        /// Embed the referenced content, for an air-gapped install.
        #[arg(long)]
        offline: bool,
    },
    /// Install a `.dshpack` archive.
    Install {
        /// What to install: a local `.dshpack` path, `github:owner/repo[@path][@ref]`,
        /// or an `https://` URL to an archive.
        file: PathBuf,
        /// Directory to install into.
        #[arg(long)]
        into: Option<PathBuf>,
    },
    /// MCP helpers (`ngu mcp import`, matching `config set` / `plugin load`).
    Mcp {
        #[command(subcommand)]
        action: McpAction,
    },
    /// Show the plugin event log (the audit trail on disk).
    Events {
        /// Lines from the end to show.
        #[arg(default_value_t = 20)]
        last: usize,
    },
    /// List installed packs.
    Packs,

    /// Take an installed pack out of every conversation, keeping its files.
    Uninstall {
        /// Pack name, or `name@version` to pick one of several.
        name: String,
        /// Remove every installed version of that name.
        #[arg(long)]
        all: bool,
        /// Delete the files instead of only unloading them.
        #[arg(long)]
        delete: bool,
        /// Directory to unload from (or delete from, with `--delete`).
        #[arg(long)]
        into: Option<PathBuf>,
    },
    /// Write the built-in interface into a directory, to start a pack from.
    ///
    /// Without this, replacing the interface means rewriting every event
    /// handler from nothing, which nobody does. The built-in one is the
    /// worked example.
    Ui {
        /// Where to write it. Defaults to ./ui.
        dir: Option<PathBuf>,
    },
    /// Verify a `.dshpack` archive without installing it.
    Verify {
        /// Archive to verify.
        file: PathBuf,
    },
    /// Load a dynamically linked plugin and report what it contributes.
    Plugin {
        #[command(subcommand)]
        action: PluginAction,
    },
    /// Show or change the stored API settings (endpoint, key, model).
    Config {
        #[command(subcommand)]
        action: Option<ConfigAction>,
    },
    /// Print the runtime's policy table and effective configuration.
    /// Show the injection entries and what they would match.
    Injections {
        /// Show which entries would activate for this text.
        #[arg(long)]
        r#for: Option<String>,
    },
    Runtime,
}

#[derive(Subcommand, Debug)]
enum ConfigAction {
    /// Print the effective settings and where they are stored.
    Show,
    /// Write settings to the settings file.
    Set {
        /// API endpoint, including the version segment, e.g. https://host/v1.
        #[arg(long)]
        base_url: Option<String>,
        /// API key.
        #[arg(long)]
        api_key: Option<String>,
        /// Model id.
        #[arg(long)]
        model: Option<String>,
        /// Reasoning effort: none, minimal, low, medium, high, xhigh, or max.
        #[arg(long)]
        reasoning_effort: Option<String>,
        /// Proxy URL, e.g. http://127.0.0.1:7890. Pass an empty string to clear
        /// it and go direct.
        #[arg(long)]
        proxy: Option<String>,
        /// Context window: 128K, 1M, or a plain token count. Pass 0 for automatic.
        #[arg(long)]
        context_window: Option<String>,
        /// Compact once this percentage of the window is in use.
        #[arg(long)]
        compact_percent: Option<u32>,
        /// Recent messages kept verbatim when compacting.
        #[arg(long)]
        compact_keep_recent: Option<usize>,
        /// Output token ceiling: 8K, 32K, or a plain count. Pass 0 to clear.
        #[arg(long)]
        max_output_tokens: Option<String>,
        /// Network tuning preset: default, reasoning, local, or flaky.
        #[arg(long)]
        network: Option<String>,
        /// Whole-request timeout, in seconds.
        #[arg(long)]
        request_timeout: Option<u64>,
        /// How long an idle connection is kept for reuse, in seconds.
        #[arg(long)]
        pool_idle_timeout: Option<u64>,
        /// How many times a dropped connection is retried.
        #[arg(long)]
        retry_attempts: Option<u32>,
        /// Milliseconds before the first retry; doubles each attempt.
        #[arg(long)]
        retry_backoff_ms: Option<u64>,
        /// Search provider: tavily, brave, or exa.
        #[arg(long)]
        search_provider: Option<String>,
        /// Search API key. The tool appears only when this is set.
        #[arg(long)]
        search_api_key: Option<String>,
        /// Judge endpoint, e.g. https://api.typesafe.ai (no path).
        #[arg(long)]
        judge_endpoint: Option<String>,
        /// Judge API key. The `judge` tool appears only when this is set.
        #[arg(long)]
        judge_key: Option<String>,
        /// Judge model alias (default: jev-latest).
        #[arg(long)]
        judge_model: Option<String>,
    },
    /// Print the settings file path.
    Path,
}

#[derive(Subcommand, Debug)]
enum PluginAction {
    /// Load a plugin library and list the tools it registers.
    Load {
        /// Path to the shared library (.dll / .so / .dylib).
        path: PathBuf,
    },
    /// Load a plugin library and call one of its tools.
    Call {
        /// Path to the shared library.
        path: PathBuf,
        /// Tool name.
        tool: String,
        /// Arguments as a JSON object.
        #[arg(default_value = "{}")]
        args: String,
    },
}

#[derive(Subcommand, Debug)]
enum McpAction {
    /// Import MCP servers from a Claude Code / Claude Desktop config into a pack.
    Import {
        /// The `.mcp.json` (or claude_desktop_config.json) to read.
        file: PathBuf,
        /// Pack directory to create (default: ./<name>).
        #[arg(long)]
        out: Option<PathBuf>,
        /// Pack name (default: derived from the file name).
        #[arg(long)]
        name: Option<String>,
        /// License for the generated pack — the servers' own licenses stay yours to check.
        #[arg(long, default_value = "MIT")]
        license: String,
    },
}

/// Prints progress to stderr, keeping stdout clean for the answer.
struct PrettyObserver {
    quiet: bool,
}

/// Emits `session.end` when the run finishes — including early `?` returns,
/// because Drop does not care about `?`.
struct SessionEnd {
    events: nguruvilu::events::EventBus,
    session: String,
}

impl Drop for SessionEnd {
    fn drop(&mut self) {
        self.events.emit(
            nguruvilu::events::SESSION_END,
            &serde_json::json!({ "session": &self.session }),
        );
    }
}

impl AgentObserver for PrettyObserver {
    fn on_step(&self, step: usize) {
        if !self.quiet && step > 1 {
            eprintln!("\n[step {step}]");
        }
    }

    fn on_text(&self, delta: &str) {
        if !self.quiet {
            print!("{delta}");
            let _ = std::io::stdout().flush();
        }
    }

    fn on_tool_start(&self, name: &str, arguments: &str) {
        if !self.quiet {
            let preview: String = arguments.chars().take(160).collect();
            eprintln!("\n→ {name} {preview}");
        }
    }

    fn on_tool_end(&self, name: &str, ok: bool, result: &str) {
        if !self.quiet {
            let first_line = result.lines().next().unwrap_or("");
            let preview: String = first_line.chars().take(140).collect();
            let mark = if ok { "✓" } else { "✗" };
            eprintln!("{mark} {name}: {preview}");
        }
    }
}

/// Supplies turn settings from a runtime, adding the skill catalog.
///
/// The runtime owns persona, route, and tools; the catalog is composed here
/// because it belongs to the CLI's view of the prompt rather than to the
/// runtime's state.
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
    injection: Arc<Mutex<Arc<nguruvilu::context::InjectionEngine>>>,
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
        if let Some(catalog) = self.skills.lock().expect("skills lock").catalog() {
            prompt.push_str("\n\n");
            prompt.push_str(&catalog);
        }
        prompt
    }

    fn context_policy(&self) -> Arc<dyn ContextPolicy> {
        Arc::clone(&self.policy.lock().expect("policy lock"))
    }
}

impl TurnConfig for CliConfig {
    fn settings(&self) -> TurnSettings {
        let snapshot = self.runtime.lock().expect("runtime lock").snapshot();
        let mut system_prompt = snapshot.system_prompt(&self.base_prompt);

        if let Some(catalog) = self.skills.lock().expect("skills lock").catalog() {
            system_prompt.push_str("\n\n");
            system_prompt.push_str(&catalog);
        }

        TurnSettings {
            system_prompt,
            tools: snapshot.tools,
            model: snapshot.model_route.model,
            version: snapshot.version,
            policy: Arc::clone(&self.policy.lock().expect("policy lock")),
            injection: Arc::clone(&self.injection.lock().expect("injection lock")),
            cache_policy: snapshot.cache_policy,
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

async fn run() -> Result<()> {
    let cli = Cli::parse();

    let store = JsonlStore::open(cli.home.clone().unwrap_or_else(JsonlStore::default_root))?;

    // Precedence: command-line flag, then stored settings, then environment.
    // `Settings::resolve` already folds the environment in, so a machine-level
    // variable still wins over the file.
    let mut settings = Settings::resolve();
    let api_key = cli
        .api_key
        .clone()
        .unwrap_or_else(|| settings.api_key.clone());
    let cache_policy = parse_cache_policy(&cli.cache_policy)?;

    // Skills are scanned once up front so every mode sees the same catalog.
    let mut skills = SkillRegistry::with_roots(SkillRegistry::default_roots());
    for dir in &cli.skill_dirs {
        skills.add_root(dir.clone());
    }
    skills.scan().ok();

    match &cli.command {
        Some(Command::Sessions) => return list_sessions(&store, cli.json),
        Some(Command::Show { id }) => return show_session(&store, id),
        Some(Command::Delete { id }) => {
            store.delete(id)?;
            println!("deleted session {id}");
            return Ok(());
        }
        Some(Command::Models) => {
            require_key(&api_key)?;
            let client = model_route(&cli, &settings, None).client()?;
            let mut ids = client.list_models().await?;
            ids.sort();
            if cli.json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({ "models": ids }))?
                );
            } else {
                for id in ids {
                    println!("{id}");
                }
            }
            return Ok(());
        }
        Some(Command::Skills { verbose }) => return list_skills(&skills, *verbose, cli.json),
        Some(Command::Assembly { file, platform }) => {
            let platform = platform.clone().unwrap_or_else(|| platform_tag().to_string());
            return show_assembly(file, &platform, cli.json);
        }
        Some(Command::Apply { file, platform }) => {
            let platform = platform.clone().unwrap_or_else(|| platform_tag().to_string());
            return apply_assembly(file, &platform, &skills, cli.json).await;
        }
        Some(Command::Snapshot { action, commit, limit }) => {
            return snapshot_command(&cli, action, commit.as_deref(), *limit, cli.json)
        }
        Some(Command::Pack { dir, out, pin, offline }) => {
            return pack_command(dir, out.as_ref(), *pin, *offline, cli.json).await
        }
        Some(Command::Install { file, into }) => return install_command(file, into.as_ref(), cli.json).await,
        Some(Command::Mcp {
            action: McpAction::Import { file, out, name, license },
        }) => return mcp_import_command(&file, out.as_deref(), name.as_deref(), &license),
        Some(Command::Events { last }) => return events_command(*last, cli.json),
        Some(Command::Packs) => return list_packs(cli.json),
        Some(Command::Uninstall { name, all, delete, into }) => {
            return uninstall_command(name, *all, *delete, into.as_ref(), cli.json)
        }
        Some(Command::Ui { dir }) => return ui_command(dir.as_ref(), cli.json),
        Some(Command::Verify { file }) => return verify_command(file, cli.json),
        Some(Command::Plugin { action }) => return plugin_command(action, cli.json),
        Some(Command::Config { action }) => return config_command(action.as_ref(), cli.json),
        Some(Command::Injections { r#for }) => {
            return show_injections(r#for.as_deref(), cli.json)
        }
        Some(Command::Runtime) => {
            // Described below, once the kernel is assembled: the tool table
            // and the published services do not exist until then.
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
    // The agent can build, install, load, and unload packs itself; the host
    // applies the load and unload between turns.
    let pending_packs = Arc::new(nguruvilu::tools::pack::PendingQueue::new());
    nguruvilu::tools::pack::register_with(kernel.tools_mut(), Some(Arc::clone(&pending_packs)))?;
    // The subagent's code is available to a pack, not loaded by the kernel:
    // `packs/starter` asks for `builtin:delegate` and it appears, and without
    // that pack the kernel has no such tool.
    nguruvilu::tools::subagent::define(&mut kernel);
    // Same for document reading (读不了就装包): `packs/documents` asks for
    // `builtin:documents` and `read` learns PDF/Office; without the pack,
    // `read` says so and names what is installed.
    nguruvilu::tools::documents::define(&mut kernel);
    // Same split for search: the code is available, the pack's assembly is
    // what makes a conversation have it. The cell is the host's copy of the
    // settings — filled now, refilled after pack content lands, and again
    // whenever something changes them.
    let search_cell = nguruvilu::tools::search::cell(settings.search.clone());
    nguruvilu::tools::search::install(&mut kernel, search_cell.clone());
    // And the same split for the judge: code in the kernel, permission to
    // exist from the pack, settings from the host's cell.
    let judge_cell = nguruvilu::tools::judge::cell(settings.judge.clone());
    nguruvilu::tools::judge::install(&mut kernel, judge_cell.clone());
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
        if let Ok(manifest) = nguruvilu::pack::read_manifest(&pack_dir) {
            contents.push(nguruvilu::content::apply(&mut kernel, &pack_dir, &manifest)?);
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
    for pack in nguruvilu::pack::installed(&nguruvilu::pack::default_packs_dir())? {
        if !pack.enabled {
            eprintln!("[pack] {} is unloaded; files kept", pack.manifest.name);
            continue;
        }
        if let Some(assembly) = &pack.assembly {
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
            Ok(manifest) => match nguruvilu::content::apply(&mut kernel, &pack.path, &manifest) {
                Ok(content) => contents.push(content),
                // Reported rather than fatal: one pack's content must not stop
                // the session from starting.
                Err(error) => eprintln!("[pack] {} content: {error:#}", pack.manifest.name),
            },
            Err(error) => eprintln!("[pack] {}: {error:#}", pack.manifest.name),
        }
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
    let injection = Arc::new(Mutex::new(Arc::new(
        nguruvilu::context::InjectionEngine::load_default(),
    )));
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
    // A pack may be what configured search, so the settings go back into the
    // plugin after the content lands: the fibers re-apply, and the tool table
    // says the truth about whether search is usable right now.
    nguruvilu::tools::search::configure(&mut kernel, &search_cell, settings.search.clone())?;
    if kernel.tools().get(nguruvilu::tools::search::TOOL).is_some() {
        eprintln!("[search] enabled");
    }

    // The judge keeps its own cell: settings first, fibers re-apply, and the
    // table says whether a verdict is available this turn.
    nguruvilu::tools::judge::configure(&mut kernel, &judge_cell, settings.judge.clone())?;
    if kernel.tools().get(nguruvilu::tools::judge::TOOL).is_some() {
        eprintln!("[judge] enabled");
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

    let runtime = Arc::new(Mutex::new(runtime));
    let skills = Arc::new(Mutex::new(skills));
    let policy = Arc::new(Mutex::new(policy));

    // Published before anything can ask for it: a plugin loaded from a pack
    // needs the service to exist already.
    let access: Arc<dyn nguruvilu::model::ModelAccess> = Arc::new(CliModelAccess {
        runtime: Arc::clone(&runtime),
        skills: Arc::clone(&skills),
        base_prompt: base_prompt.clone(),
        policy: Arc::clone(&policy),
    });
    nguruvilu::model::install(&mut kernel, Arc::clone(&access))?;

    // A pack that asked for `builtin:delegate` was waiting on the route above;
    // now that it is published the kernel activates it, and the turn's tool
    // snapshot — taken long before any of this — is taken again so the session
    // sees whatever just arrived.
    kernel.refresh()?;
    if kernel.tools().get(nguruvilu::tools::subagent::TOOL).is_some() {
        eprintln!("[subagent] delegate loaded by the subagent pack");
    }
    runtime
        .lock()
        .expect("runtime lock")
        .adopt_kernel_tools(&kernel);

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

    let config: Arc<dyn TurnConfig> = Arc::new(CliConfig {
        runtime: Arc::clone(&runtime),
        base_prompt,
        skills: Arc::clone(&skills),
        policy: Arc::clone(&policy),
        injection: Arc::clone(&injection),
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
            let _session_end = SessionEnd {
                events: events.clone(),
                session: session.id.clone(),
            };
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
            let mut agent = Agent::new(client, Arc::clone(&config), session.messages.clone())
                .with_max_steps(cli.max_steps)
                .with_observer(observer);
            let outcome = agent.run(&text).await;
            let turn_ms = turn_started.elapsed().as_millis() as u64;
            match &outcome {
                Ok(done) => events.emit(
                    nguruvilu::events::TURN_COMPLETE,
                    &serde_json::json!({
                        "session": &session.id,
                        "ok": true,
                        "duration_ms": turn_ms,
                        "steps": done.steps,
                        "tool_calls": done.tool_calls,
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
            let outcome = outcome?;

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
                search: &search_cell,
                judge: &judge_cell,
                persona_override: cli.persona.clone(),
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
                        "\n[session {} | {} steps | {} tool calls | {} in / {} out ({} cached) | model {} ms, tools {} ms]",
                        session.id,
                        outcome.steps,
                        outcome.tool_calls,
                        outcome.usage.input,
                        outcome.usage.output,
                        outcome.usage.cached,
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
                cli.max_steps,
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
                    search: &search_cell,
                    judge: &judge_cell,
                    persona_override: cli.persona.clone(),
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

async fn interactive(
    client: LlmClient,
    config: Arc<dyn TurnConfig>,
    mut session: Session,
    store: &JsonlStore,
    max_steps: usize,
    snapshotter: Option<GitSnapshot>,
    mut pack_host: PackHost<'_>,
) -> Result<()> {
    println!(
        "Nguruvilu {} — session {}",
        env!("CARGO_PKG_VERSION"),
        session.id
    );
    println!("Type a prompt, or /exit to quit, /new for a new session.\n");

    // The run that called this REPL already installed the audit log and
    // emitted `session.start`; this function joins the same bus for turn
    // events and owns the matching `session.end`.
    let events = pack_host.kernel.events();
    let mut session_end = SessionEnd {
        events: events.clone(),
        session: session.id.clone(),
    };

    let mut agent = Agent::new(client, Arc::clone(&config), session.messages.clone())
        .with_max_steps(max_steps)
        .with_observer(Arc::new(PrettyObserver { quiet: false }));

    loop {
        print!("› ");
        std::io::stdout().flush()?;

        let mut line = String::new();
        let read = std::io::stdin().read_line(&mut line)?;
        if read == 0 {
            break;
        }
        let input = line.trim();
        if input.is_empty() {
            continue;
        }
        if input == "/exit" || input == "/quit" {
            break;
        }
        if input == "/new" {
            // The old session ends here, the new one starts: a process can
            // switch sessions any number of times, and the audit log joins
            // each turn to its own session through this pair.
            events.emit(
                nguruvilu::events::SESSION_END,
                &serde_json::json!({ "session": &session.id }),
            );
            session = Session::new(session.model.clone());
            store.create(&session)?;
            events.emit(
                nguruvilu::events::SESSION_START,
                &serde_json::json!({ "session": &session.id }),
            );
            session_end.session = session.id.clone();
            agent.set_messages(Vec::new());
            println!("new session {}\n", session.id);
            continue;
        }

        if let Some(git) = &snapshotter {
            if let Some(record) = git.snapshot("before turn")? {
                eprintln!("[snapshot {}]", record.short());
            }
        }

        events.emit(
            nguruvilu::events::PROMPT_SUBMIT,
            &serde_json::json!({
                "session": &session.id,
                "text": nguruvilu::events::truncate(input, 4000),
            }),
        );
        events.emit(
            nguruvilu::events::TURN_START,
            &serde_json::json!({ "session": &session.id }),
        );
        let turn_started = std::time::Instant::now();
        let turn = agent.run(input).await;
        let turn_ms = turn_started.elapsed().as_millis() as u64;
        match &turn {
            Ok(outcome) => events.emit(
                nguruvilu::events::TURN_COMPLETE,
                &serde_json::json!({
                    "session": &session.id,
                    "ok": true,
                    "duration_ms": turn_ms,
                    "steps": outcome.steps,
                    "tool_calls": outcome.tool_calls,
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
        match turn {
            Ok(outcome) => {
                store.append(&session.id, &outcome.new_messages)?;
                session.messages.extend(outcome.new_messages);
                println!(
                    "\n[{} steps | {} tool calls | {} in / {} out ({} cached) | model {} ms, tools {} ms]\n",
                    outcome.steps,
                    outcome.tool_calls,
                    outcome.usage.input,
                    outcome.usage.output,
                    outcome.usage.cached,
                    outcome.timing.model_ms,
                    outcome.timing.tools_ms
                );
                if let Some(git) = &snapshotter {
                    if let Some(record) = git.snapshot("after turn")? {
                        eprintln!("[snapshot {}]\n", record.short());
                    }
                }
                // A pack the turn loaded or unloaded is applied here, between
                // turns: the conversation keeps going, with a new tool table.
                pack_host.drain().await?;
            }
            Err(error) => {
                eprintln!("error: {error:#}\n");
            }
        }
    }

    Ok(())
}

fn list_sessions(store: &JsonlStore, as_json: bool) -> Result<()> {
    let sessions = store.list()?;
    if as_json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({ "sessions": sessions }))?
        );
        return Ok(());
    }
    if sessions.is_empty() {
        println!("no sessions in {}", store.root().display());
        return Ok(());
    }
    println!("{:<24} {:>8}  {}", "id", "messages", "title");
    for s in sessions {
        println!(
            "{:<24} {:>8}  {}",
            s.id,
            s.message_count,
            s.title.unwrap_or_else(|| "(untitled)".into())
        );
    }
    Ok(())
}

fn show_session(store: &JsonlStore, id: &str) -> Result<()> {
    let session = store
        .load(id)?
        .ok_or_else(|| anyhow!("session not found: {id}"))?;
    println!(
        "session {} ({} messages, created {})",
        session.id,
        session.messages.len(),
        session.created_at
    );
    if let Some(cwd) = &session.cwd {
        println!("cwd: {cwd}");
    }
    println!();
    for message in &session.messages {
        let label = match message.role {
            nguruvilu::message::Role::System => "system",
            nguruvilu::message::Role::User => "user",
            nguruvilu::message::Role::Assistant => "assistant",
            nguruvilu::message::Role::Tool => "tool",
        };
        for call in &message.tool_calls {
            println!("[{label}] → {}({})", call.name, call.arguments);
        }
        let text = message.text();
        if !text.is_empty() {
            println!("[{label}] {text}");
        }
    }
    Ok(())
}

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
    // The agent can build and install packs itself.
    nguruvilu::tools::pack::register(kernel.tools_mut())?;
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

async fn pack_command(
    dir: &PathBuf,
    out: Option<&PathBuf>,
    pin: bool,
    offline: bool,
    as_json: bool,
) -> Result<()> {
    let mut manifest = nguruvilu::pack::read_manifest(dir)?;
    let archive = match out {
        Some(path) => path.clone(),
        None => PathBuf::from(format!("{}-{}.dshpack", manifest.name, manifest.version_id)),
    };

    if pin {
        let fetcher = nguruvilu::fetch::Fetcher::new()?;
        // Collected first: pinning mutates the manifest, so the borrow that
        // produced these references cannot still be live.
        let todo: Vec<(String, String, nguruvilu::fetch::Source, Option<String>)> = manifest
            .references()?
            .into_iter()
            .filter(|reference| reference.source.is_remote())
            .map(|reference| {
                (
                    reference.kind.to_string(),
                    reference.id.to_string(),
                    reference.source.clone(),
                    reference.sha256.map(str::to_string),
                )
            })
            .collect();

        let mut pinned = 0;
        for (kind, id, source, expected) in todo {
            let fetched = fetcher
                .fetch(&source, expected.as_deref())
                .await
                .with_context(|| format!("pinning {kind} '{id}'"))?;
            manifest.set_hash(&kind, &id, &fetched.sha256)?;
            pinned += 1;
            if !as_json {
                eprintln!(
                    "pinned {kind} {id} → {}",
                    &fetched.sha256[..12.min(fetched.sha256.len())]
                );
            }
        }
        // Written back so the directory and the archive agree; a pinned pack
        // that only existed inside the archive would be re-pinned on every
        // build, which is the opposite of pinning.
        nguruvilu::pack::write_manifest(dir, &manifest)?;
        if !as_json && pinned == 0 {
            eprintln!("nothing to pin: every reference is local or built in");
        }
    }

    let packed = if offline {
        nguruvilu::pack::pack_offline(dir, &archive).await?
    } else {
        nguruvilu::pack::pack(dir, &archive)?
    };
    // The same exclusion the archive used, so the count is what was packed
    // rather than what happens to be in the directory.
    let contents = nguruvilu::pack::packable_files(dir, &archive)?;

    if as_json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "ok": true,
                "archive": archive.display().to_string(),
                "name": packed.name,
                "version": packed.version_id,
                "license": packed.license,
                "files": contents.files.len(),
                "has_assembly": contents.has_assembly,
            }))?
        );
    } else {
        println!(
            "packed {} {} → {}",
            packed.name,
            packed.version_id,
            archive.display()
        );
        println!("{} files", contents.files.len());
        if !contents.has_assembly {
            println!("note: no assembly.yaml — the pack loads nothing yet");
        }
    }
    Ok(())
}

async fn install_command(file: &PathBuf, into: Option<&PathBuf>, as_json: bool) -> Result<()> {
    let packs_dir = into.cloned().unwrap_or_else(nguruvilu::pack::default_packs_dir);

    // Two ways in, both the same command: a spec downloads the archive first,
    // a path installs what is already on disk — the offline half. The content
    // inside has always had both forms (`Carried` vs `Fetched`); this is the
    // pack itself catching up.
    let raw = file.to_string_lossy();
    let raw = raw.trim();
    let local = if is_remote_spec(raw) {
        resolve_pack_source(raw).await?
    } else {
        file.clone()
    };

    let placed = nguruvilu::pack::install(&local, &packs_dir).await?;

    if as_json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "ok": true,
                "name": placed.manifest.name,
                "version": placed.manifest.version_id,
                "path": placed.path.display().to_string(),
                "assembly": placed.assembly.as_ref().map(|p| p.display().to_string()),
            }))?
        );
    } else {
        println!(
            "installed {} {} → {}",
            placed.manifest.name,
            placed.manifest.version_id,
            placed.path.display()
        );
        match &placed.assembly {
            Some(assembly) => println!("assembly: {}", assembly.display()),
            None => println!("no assembly.yaml in this pack"),
        }
    }
    Ok(())
}

/// A download spec rather than a path: `github:…`, `https://…`, `http://…`.
///
/// A prefix check rather than `Source::parse` on purpose: a Windows path
/// (`C:\…`) contains a colon too, and a typo in a path must fail as a path
/// instead of being reinterpreted as a source.
fn is_remote_spec(raw: &str) -> bool {
    raw.starts_with("github:") || raw.starts_with("https://") || raw.starts_with("http://")
}

/// Resolve a download spec to a local archive to install.
///
/// A URL that *is* the archive lands as a file and goes straight through; a
/// `github:` spec lands as a directory (the pack's own directory, or a repo),
/// and the archive inside it is picked.
async fn resolve_pack_source(spec: &str) -> Result<PathBuf> {
    let source = nguruvilu::fetch::Source::parse(spec)
        .with_context(|| format!("reading '{spec}' as a download source"))?;
    let fetcher = nguruvilu::fetch::Fetcher::new()?;
    let fetched = fetcher
        .fetch(&source, None)
        .await
        .with_context(|| format!("fetching '{spec}'"))?;
    if fetched.path.is_file() {
        return Ok(fetched.path);
    }
    pack_archive_in(&fetched.path)
}

/// The one `.dshpack` inside a fetched directory.
///
/// A pack's own directory contains one archive; a repository contains many,
/// and guessing which was meant would install the wrong pack — so it lists
/// them instead of choosing.
fn pack_archive_in(dir: &std::path::Path) -> Result<PathBuf> {
    let mut found = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(current) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&current) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().and_then(|ext| ext.to_str()) == Some("dshpack") {
                found.push(path);
            }
        }
    }
    match found.len() {
        0 => Err(anyhow::anyhow!("no .dshpack archive under {}", dir.display())),
        1 => Ok(found.pop().expect("exactly one match")),
        _ => {
            found.sort();
            let names = found
                .iter()
                .map(|path| {
                    path.strip_prefix(dir)
                        .unwrap_or(path)
                        .display()
                        .to_string()
                })
                .collect::<Vec<_>>()
                .join(", ");
            Err(anyhow::anyhow!(
                "{} archives under {}; point at the pack's own directory, one of: {names}",
                found.len(),
                dir.display()
            ))
        }
    }
}

/// Import MCP servers from a Claude Code config file into a new pack directory.
///
/// The classic config shape is `{"mcpServers": {name: {command, args, env}}}`
/// (Claude Code's `.mcp.json`, Claude Desktop's config file); our own
/// `{"servers": [...]}` passes through the strict reader. Entries without a
/// command — the url-based transports — are reported and *not* written to the
/// pack: this kernel connects stdio servers today, and a pack claiming
/// otherwise would fail at load instead of at import.
fn mcp_import_command(
    file: &Path,
    out: Option<&Path>,
    name: Option<&str>,
    license: &str,
) -> Result<()> {
    let raw =
        std::fs::read_to_string(file).with_context(|| format!("reading {}", file.display()))?;
    let doc: serde_json::Value = serde_json::from_str(&raw)
        .with_context(|| format!("parsing {} as JSON", file.display()))?;

    let (servers, skipped) = convert_mcp_servers(&doc)?;

    let pack_name = match name {
        Some(name) => name.trim().to_string(),
        None => sanitize_pack_name(
            &file
                .file_stem()
                .map(|stem| stem.to_string_lossy().to_string())
                .unwrap_or_else(|| "mcp".into()),
        ),
    };
    if pack_name.is_empty() {
        anyhow::bail!("the pack name is empty after sanitizing; pass --name");
    }

    let dir = out.map(Path::to_path_buf).unwrap_or_else(|| PathBuf::from(&pack_name));
    if dir.exists() {
        let non_empty = std::fs::read_dir(&dir)?.next().transpose()?.is_some();
        if non_empty {
            anyhow::bail!(
                "{} exists and is not empty — refusing to overwrite; pass --out for another directory",
                dir.display()
            );
        }
    }
    std::fs::create_dir_all(&dir)?;

    let manifest = serde_json::json!({
        "formatVersion": 1,
        "game": "nguruvilu",
        "name": pack_name,
        "versionId": "1.0.0",
        "license": license,
        "kernelVersion": "0.1.0",
        "dependencies": { "nguruvilu": ">=0.1.0" },
        "mcp": "mcp.json",
    });
    std::fs::write(
        dir.join("dsh.index.json"),
        serde_json::to_string_pretty(&manifest)?,
    )?;
    std::fs::write(
        dir.join("mcp.json"),
        serde_json::to_string_pretty(&nguruvilu::pack::McpFile {
            servers: servers.clone(),
        })?,
    )?;
    std::fs::write(
        dir.join("README.md"),
        generated_pack_readme(&pack_name, file, license, &servers, &skipped),
    )?;

    println!("imported {} server(s) into {}", servers.len(), dir.display());
    for server in &servers {
        println!(
            "  + {}  ({})",
            server.id,
            server.command.as_deref().unwrap_or("(no command)")
        );
    }
    for (id, reason) in &skipped {
        println!("  - {id}: {reason}");
    }
    println!();
    println!(
        "next: ngu pack {dir} --out {name}-1.0.0.dshpack   then   ngu install <the archive>",
        dir = dir.display(),
        name = pack_name
    );
    println!("note: each server's runtime (node, …) and its own license are yours to provide.");
    Ok(())
}

/// Convert a config document into our servers plus what had to be left out.
fn convert_mcp_servers(
    doc: &serde_json::Value,
) -> Result<(Vec<nguruvilu::pack::McpServerDecl>, Vec<(String, String)>)> {
    let mut servers = Vec::new();
    let mut skipped: Vec<(String, String)> = Vec::new();

    if let Some(map) = doc.get("mcpServers").and_then(|v| v.as_object()) {
        for (id, entry) in map {
            match classify_server(id, entry) {
                Ok(server) => servers.push(server),
                Err(reason) => skipped.push((id.clone(), reason)),
            }
        }
        return Ok((servers, skipped));
    }
    if let Some(list) = doc.get("servers").and_then(|v| v.as_array()) {
        // Our own format: pass it through the same strict reader an install
        // would use, so an invalid file fails here rather than at load.
        let ours: nguruvilu::pack::McpFile = serde_json::from_value(
            serde_json::json!({ "servers": list.clone() }),
        )
        .context("the servers array does not match our mcp.json shape")?;
        return Ok((ours.servers, skipped));
    }
    anyhow::bail!(
        "no servers found: expected an object with 'mcpServers' (Claude Code / \
         Claude Desktop) or 'servers' (our mcp.json)"
    );
}

/// One entry → one declaration, or the reason it was left out.
fn classify_server(
    id: &str,
    entry: &serde_json::Value,
) -> Result<nguruvilu::pack::McpServerDecl, String> {
    let obj = entry
        .as_object()
        .ok_or_else(|| "entry is not an object".to_string())?;
    let command = obj
        .get("command")
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|c| !c.is_empty());
    let Some(command) = command else {
        let kind = obj.get("type").and_then(serde_json::Value::as_str).unwrap_or_else(|| {
            if obj.contains_key("url") { "http" } else { "unknown" }
        });
        return Err(format!(
            "'{kind}' transport is not supported yet (stdio only) — add it by hand when http lands"
        ));
    };
    let args = obj
        .get("args")
        .and_then(serde_json::Value::as_array)
        .map(|list| {
            list.iter()
                .map(|value| match value {
                    serde_json::Value::String(text) => text.clone(),
                    other => other.to_string(),
                })
                .collect()
        })
        .unwrap_or_default();
    let env = obj
        .get("env")
        .and_then(serde_json::Value::as_object)
        .map(|map| {
            map.iter()
                .map(|(key, value)| match value {
                    serde_json::Value::String(text) => (key.clone(), text.clone()),
                    other => (key.clone(), other.to_string()),
                })
                .collect::<std::collections::BTreeMap<String, String>>()
        })
        .unwrap_or_default();
    Ok(nguruvilu::pack::McpServerDecl {
        id: id.to_string(),
        transport: "stdio".into(),
        source: None,
        sha256: None,
        command: Some(command.to_string()),
        args,
        env,
        scope: Some("session".into()),
        platforms: std::collections::BTreeMap::new(),
    })
}

/// A file-derived pack name: lowercase, `[a-z0-9-]`, no edge dashes.
fn sanitize_pack_name(raw: &str) -> String {
    let lowered = raw.trim().to_ascii_lowercase();
    let mut out = String::new();
    let mut dash = false;
    for ch in lowered.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch);
            dash = false;
        } else if !dash && !out.is_empty() {
            out.push('-');
            dash = true;
        }
    }
    out.trim_end_matches('-').to_string()
}

/// The generated pack's README: what was imported, what was skipped, how to
/// pack and install it — and that the runtime and licenses stay with the user.
fn generated_pack_readme(
    pack_name: &str,
    source_file: &Path,
    license: &str,
    servers: &[nguruvilu::pack::McpServerDecl],
    skipped: &[(String, String)],
) -> String {
    let mut out = String::new();
    out.push_str(&format!("# {pack_name} — imported MCP servers\n\n"));
    out.push_str(&format!(
        "由 `ngu mcp import {}` 生成:身份清单 + `mcp.json` + 这份说明。\n\n",
        source_file.display()
    ));
    out.push_str(
        "**只搬运声明,不搬运程序** —— 每个服务器的运行时(npx 需要 Node、各自的\n\
         npm 包或可执行文件)与它自己的许可证,由你提供与核对。环境变量的**值**\n\
         不会出现在这份文件里。\n\n",
    );
    out.push_str("## 服务器\n\n| id | 命令 | 参数 | 环境变量名 |\n|---|---|---|---|\n");
    for server in servers {
        let command = server.command.as_deref().unwrap_or("");
        let args = server
            .args
            .iter()
            .map(|arg| format!("`{arg}`"))
            .collect::<Vec<_>>()
            .join(" ");
        let args = if args.is_empty() { "—".into() } else { args };
        let keys = server.env.keys().cloned().collect::<Vec<_>>().join(", ");
        let keys = if keys.is_empty() { "—".into() } else { keys };
        out.push_str(&format!("| {} | `{command}` | {args} | {keys} |\n", server.id));
    }
    if !skipped.is_empty() {
        out.push_str("\n## 未导入(本内核暂不支持)\n\n");
        for (id, reason) in skipped {
            out.push_str(&format!("- `{id}`:{reason}\n"));
        }
    }
    out.push_str(&format!(
        "\n## 用法\n\n```bash\nngu pack {name} --out {name}-1.0.0.dshpack\n\
         ngu install {name}-1.0.0.dshpack\nngu packs\n```\n\n\
         装载后每个服务器的工具以 `mcp__<id>__*` 出现;卸载留文件,`--delete` 才删。\n\n\
         ## 许可\n\n本包清单声明 **{license}**(这是这个*包*的许可);各 MCP 服务器\n\
         本身的许可以其上游为准。\n",
        name = pack_name
    ));
    out
}

/// `ngu events` — read back the audit trail the hosts append to.
///
/// The file is the source of truth (the bus keeps no history), so this is a
/// file read with a shape, not a query into a running process.
fn events_command(last: usize, as_json: bool) -> Result<()> {
    let path = nguruvilu::events::audit_path();
    let Ok(raw) = std::fs::read_to_string(&path) else {
        if as_json {
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "ok": true,
                    "path": path.display().to_string(),
                    "lines": [],
                }))?
            );
        } else {
            println!("no events yet — the log is {}", path.display());
        }
        return Ok(());
    };
    let lines: Vec<&str> = raw.lines().filter(|l| !l.trim().is_empty()).collect();
    let start = lines.len().saturating_sub(last.max(1));
    if as_json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "ok": true,
                "path": path.display().to_string(),
                "count": lines.len(),
                "lines": &lines[start..],
            }))?
        );
        return Ok(());
    }
    println!(
        "{} — {} event(s), showing the last {}",
        path.display(),
        lines.len(),
        lines.len() - start
    );
    for line in &lines[start..] {
        match serde_json::from_str::<serde_json::Value>(line) {
            Ok(value) => println!(
                "{:>13}  {:<15}  {}",
                value["ts"].as_u64().unwrap_or(0),
                value["event"].as_str().unwrap_or("?"),
                nguruvilu::events::truncate(&value["payload"].to_string(), 200)
            ),
            Err(_) => println!("  (unparseable) {line}"),
        }
    }
    Ok(())
}

fn list_packs(as_json: bool) -> Result<()> {
    let packs_dir = nguruvilu::pack::default_packs_dir();
    let packs = nguruvilu::pack::installed(&packs_dir)?;

    if as_json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "packs_dir": packs_dir.display().to_string(),
                "packs": packs.iter().map(|p| serde_json::json!({
                    "name": p.manifest.name,
                    "version": p.manifest.version_id,
                    "license": p.manifest.license,
                    "path": p.path.display().to_string(),
                    "has_assembly": p.assembly.is_some(),
                    "loaded": p.enabled,
                })).collect::<Vec<_>>(),
            }))?
        );
        return Ok(());
    }

    if packs.is_empty() {
        println!("no packs installed in {}", packs_dir.display());
        return Ok(());
    }
    println!(
        "{:<24} {:<12} {:<10} {:<8} {}",
        "name", "version", "license", "loaded", "assembly"
    );
    for pack in packs {
        println!(
            "{:<24} {:<12} {:<10} {:<8} {}",
            pack.manifest.name,
            pack.manifest.version_id,
            pack.manifest.license,
            if pack.enabled { "yes" } else { "no" },
            if pack.assembly.is_some() { "yes" } else { "no" }
        );
    }
    Ok(())
}

/// Write a copy of the built-in interface to a directory.
fn ui_command(dir: Option<&PathBuf>, as_json: bool) -> Result<()> {
    let target = dir.cloned().unwrap_or_else(|| PathBuf::from("ui"));
    std::fs::create_dir_all(&target)
        .with_context(|| format!("creating {}", target.display()))?;

    let page = target.join("index.html");
    std::fs::write(&page, nguruvilu::ui::BUILTIN_HTML)
        .with_context(|| format!("writing {}", page.display()))?;

    // The declaration that makes the directory an interface, so `eject` gives
    // something a pack can point at rather than a loose file.
    let manifest = nguruvilu::pack::PackManifest {
        ui: Some(nguruvilu::pack::UiDecl {
            id: "my-ui".into(),
            title: "我的界面".into(),
            entry: "index.html".into(),
            source: None,
            sha256: None,
        }),
        ..nguruvilu::pack::PackManifest::new("my-ui", "1.0.0")
    };
    nguruvilu::pack::write_manifest(&target, &manifest)?;

    if as_json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "ok": true,
                "dir": target.display().to_string(),
                "entry": "index.html",
                "bytes": nguruvilu::ui::BUILTIN_HTML.len(),
            }))?
        );
    } else {
        println!("wrote the built-in interface to {}", target.display());
        println!("edit index.html, then `ngu pack {}`", target.display());
        println!("the declaration in dsh.index.json points at it");
    }
    Ok(())
}

/// Take an installed pack out of every conversation, or delete its files.
///
/// Two different acts with one command on purpose: the ordinary one is
/// *unload* — the pack stops being loaded into new conversations, and its
/// files stay, so it can come back without downloading anything. `--delete` is
/// the one that removes files, because deleting is not how you stop using
/// something.
///
/// Named rather than pathed: the user knows which pack they installed, and
/// asking them to find the directory is asking them to know how install
/// arranges things. `name@version` picks one when several are installed.
fn uninstall_command(
    name: &str,
    all: bool,
    delete: bool,
    into: Option<&PathBuf>,
    as_json: bool,
) -> Result<()> {
    let root = into
        .cloned()
        .unwrap_or_else(nguruvilu::pack::default_packs_dir);
    let source = nguruvilu::source::FilesystemSource::new(&root);

    let (wanted_name, wanted_version) = match name.split_once('@') {
        Some((pack, version)) => (pack, Some(version)),
        None => (name, None),
    };

    let matches: Vec<nguruvilu::pack::InstalledPack> = source
        .list()?
        .into_iter()
        .filter(|pack| pack.manifest.name == wanted_name)
        .filter(|pack| match wanted_version {
            Some(version) => pack.manifest.version_id == version,
            None => true,
        })
        .collect();

    if matches.is_empty() {
        // Naming what is installed turns a dead end into a next step.
        let installed = source.list()?;
        let names: Vec<String> = installed
            .iter()
            .map(|pack| format!("{}@{}", pack.manifest.name, pack.manifest.version_id))
            .collect();
        anyhow::bail!(
            "no installed pack matches '{name}'.{}",
            if names.is_empty() {
                format!(" Nothing is installed in {}.", root.display())
            } else {
                format!(" Installed: {}", names.join(", "))
            }
        );
    }

    // Several versions at once is what `--all` is for; without it, ask rather
    // than guessing which one was meant.
    if matches.len() > 1 && !all {
        let versions: Vec<String> = matches
            .iter()
            .map(|pack| pack.manifest.version_id.clone())
            .collect();
        anyhow::bail!(
            "'{wanted_name}' has {} versions installed: {}. \
             Name one as '{wanted_name}@<version>', or pass --all.",
            versions.len(),
            versions.join(", ")
        );
    }

    let mut affected = Vec::new();
    for pack in &matches {
        let label = format!("{}@{}", pack.manifest.name, pack.manifest.version_id);
        if delete {
            source.remove(&pack.path)?;
        } else {
            nguruvilu::pack::set_enabled(&pack.path, false)?;
        }
        affected.push(label);
    }

    if as_json {
        let payload = if delete {
            serde_json::json!({
                "ok": true,
                "removed": affected,
                "files_kept": false,
                "dir": root.display().to_string(),
            })
        } else {
            serde_json::json!({
                "ok": true,
                "unloaded": affected,
                "files_kept": true,
                "dir": root.display().to_string(),
            })
        };
        println!("{}", serde_json::to_string_pretty(&payload)?);
    } else if delete {
        for entry in &affected {
            println!("deleted {entry}");
        }
        println!("Its files are gone; reinstall to use it again.");
    } else {
        for entry in &affected {
            println!("unloaded {entry} (files kept)");
        }
        println!(
            "New conversations will not load it. To bring it back: ngu install <archive>, \
             or load it from inside a conversation with the pack tool."
        );
        println!(
            "A session already running keeps what it loaded; ask it to unload, or restart it."
        );
    }
    Ok(())
}

fn verify_command(file: &PathBuf, as_json: bool) -> Result<()> {
    let report = nguruvilu::pack::verify(file)?;

    if as_json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "ok": report.warnings.is_empty(),
                "manifest": {
                    "name": report.manifest.name,
                    "version": report.manifest.version_id,
                    "license": report.manifest.license,
                    "kernel_version": report.manifest.kernel_version,
                    "requires": report.manifest.dependencies.nguruvilu,
                },
                "files": report.contents.files.len(),
                "has_assembly": report.contents.has_assembly,
                "warnings": report.warnings,
            }))?
        );
        return Ok(());
    }

    println!(
        "{} {} (license {}, built against kernel {})",
        report.manifest.name,
        report.manifest.version_id,
        report.manifest.license,
        report.manifest.kernel_version
    );
    println!("requires nguruvilu {}", report.manifest.dependencies.nguruvilu);


    println!("{} files", report.contents.files.len());
    if report.warnings.is_empty() {
        println!("no problems found");
    } else {
        for warning in &report.warnings {
            println!("warning: {warning}");
        }
    }
    Ok(())
}

fn plugin_command(action: &PluginAction, as_json: bool) -> Result<()> {
    match action {
        PluginAction::Load { path } => plugin_load(path, as_json),
        PluginAction::Call { path, tool, args } => plugin_call(path, tool, args, as_json),
    }
}

/// Load a plugin and report what it contributes.
///
/// Loading runs native code in this process, so this command is the explicit
/// opt-in: nothing loads a library implicitly.
fn plugin_load(path: &PathBuf, as_json: bool) -> Result<()> {
    let plugin = unsafe { nguruvilu::dylib::DynamicPlugin::load(path)? };

    // Tools are registered through a callback during `apply`, so the plugin has
    // to be applied once before its contributions are visible.
    let ctx = nguruvilu::plugin::PluginCtx {
        plugin: plugin.name().to_string(),
        fiber: 0,
        realm: nguruvilu::plugin::RealmMap::new(),
        services: nguruvilu::plugin::ServiceView::default(),
        config: serde_json::Value::Null,
        events: nguruvilu::events::EventBus::default(),
    };
    let contributions = PluginTrait::apply(&plugin, &ctx)?;
    let tools: Vec<String> = contributions.tools.iter().map(|t| t.name.clone()).collect();
    let logs = plugin.log_lines();

    if as_json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "ok": true,
                "path": plugin.path().display().to_string(),
                "name": plugin.name(),
                "version": plugin.meta().version,
                "description": plugin.meta().description,
                "abi_version": plugin.meta().abi_version,
                "inject": plugin.inject(),
                "provide": plugin.provide(),
                "tools": tools,
                "log": logs.iter().map(|(level, text)| serde_json::json!({"level": level, "text": text})).collect::<Vec<_>>(),
            }))?
        );
        return Ok(());
    }

    println!("loaded {} from {}", plugin.name(), plugin.path().display());
    if let Some(version) = &plugin.meta().version {
        println!("version:     {version}");
    }
    if let Some(description) = &plugin.meta().description {
        println!("description: {description}");
    }
    if let Some(abi) = plugin.meta().abi_version {
        println!("abi:         {abi}");
    }
    if !plugin.inject().is_empty() {
        println!("injects:     {}", plugin.inject().join(", "));
    }
    if !plugin.provide().is_empty() {
        println!("provides:    {}", plugin.provide().join(", "));
    }
    println!("tools ({}):", tools.len());
    for tool in &tools {
        println!("  {tool}");
    }
    for (level, text) in logs {
        let name = match level {
            0 => "debug",
            1 => "info",
            2 => "warn",
            _ => "error",
        };
        println!("[{name}] {text}");
    }
    Ok(())
}

/// Load a plugin and call one of its tools, for verification.
fn plugin_call(path: &PathBuf, tool: &str, args: &str, as_json: bool) -> Result<()> {
    let plugin = unsafe { nguruvilu::dylib::DynamicPlugin::load(path)? };

    let ctx = nguruvilu::plugin::PluginCtx {
        plugin: plugin.name().to_string(),
        fiber: 0,
        realm: nguruvilu::plugin::RealmMap::new(),
        services: nguruvilu::plugin::ServiceView::default(),
        config: serde_json::Value::Null,
        events: nguruvilu::events::EventBus::default(),
    };
    PluginTrait::apply(&plugin, &ctx)?;

    let arguments: serde_json::Value =
        serde_json::from_str(args).with_context(|| format!("parsing arguments {args:?}"))?;
    let result = plugin.call(tool, arguments);

    if as_json {
        let payload = match &result {
            Ok(content) => serde_json::json!({"ok": true, "tool": tool, "content": content}),
            Err(error) => serde_json::json!({"ok": false, "tool": tool, "error": format!("{error:#}")}),
        };
        println!("{}", serde_json::to_string_pretty(&payload)?);
        // A plugin-reported failure is a failed call, and the exit code says so.
        if result.is_err() {
            std::process::exit(1);
        }
        return Ok(());
    }

    match result {
        Ok(content) => {
            println!("{content}");
            Ok(())
        }
        Err(error) => Err(error),
    }
}

/// Show or change the stored settings.
fn config_command(action: Option<&ConfigAction>, as_json: bool) -> Result<()> {
    match action {
        Some(ConfigAction::Path) => {
            println!("{}", Settings::path().display());
            Ok(())
        }
        Some(ConfigAction::Set {
            base_url,
            api_key,
            model,
            reasoning_effort,
            proxy,
            context_window,
            compact_percent,
            compact_keep_recent,
            max_output_tokens,
            network,
            request_timeout,
            pool_idle_timeout,
            retry_attempts,
            retry_backoff_ms,
            search_provider,
            search_api_key,
            judge_endpoint,
            judge_key,
            judge_model,
        }) => {
            let mut settings = Settings::load()?;
            if let Some(value) = base_url {
                settings.base_url = value.trim().to_string();
            }
            if let Some(value) = api_key {
                settings.api_key = value.trim().to_string();
            }
            if let Some(value) = model {
                settings.model = value.trim().to_string();
            }
            if let Some(value) = reasoning_effort {
                settings.reasoning_effort = value.trim().to_string();
            }
            if let Some(value) = proxy {
                settings.proxy = value.trim().to_string();
            }
            if let Some(value) = context_window {
                // Accepts `128K`, `1M`, or a plain count. Zero means "go back
                // to working it out".
                let tokens = nguruvilu::size::parse_size(value)?;
                settings.context_window = if tokens == 0 { None } else { Some(tokens) };
            }
            if let Some(value) = compact_percent {
                settings.compact_percent = (*value).min(100);
            }
            if let Some(value) = compact_keep_recent {
                settings.compact_keep_recent = (*value).max(1);
            }
            if let Some(value) = max_output_tokens {
                let tokens = nguruvilu::size::parse_size(value)?;
                settings.max_output_tokens = if tokens == 0 { None } else { Some(tokens) };
            }
            if let Some(name) = network {
                settings.network = nguruvilu::network::NetworkSettings::preset(name)
                    .ok_or_else(|| {
                        anyhow!(
                            "unknown network preset '{name}'; expected one of {}",
                            nguruvilu::network::NetworkSettings::presets().join(", ")
                        )
                    })?;
            }
            if let Some(value) = request_timeout {
                settings.network.request_timeout_secs = *value;
            }
            if let Some(value) = pool_idle_timeout {
                settings.network.pool_idle_timeout_secs = *value;
            }
            if let Some(value) = retry_attempts {
                settings.network.retry_attempts = *value;
            }
            if let Some(value) = retry_backoff_ms {
                settings.network.retry_backoff_ms = *value;
            }
            // A value that would behave unlike its name is reported here rather
            // than at the first failed request.
            if search_provider.is_some() || search_api_key.is_some() {
                let mut search = settings
                    .search
                    .clone()
                    .unwrap_or_else(|| nguruvilu::tools::search::SearchSettings {
                        provider: nguruvilu::tools::search::Dialect::Tavily,
                        api_key: String::new(),
                        endpoint: None,
                        max_results: 5,
                    });
                if let Some(name) = search_provider {
                    search.provider = nguruvilu::tools::search::Dialect::parse(name).ok_or_else(
                        || {
                            anyhow!(
                                "unknown search provider '{name}'; expected one of {}",
                                nguruvilu::tools::search::Dialect::all()
                                    .iter()
                                    .map(|d| d.as_str())
                                    .collect::<Vec<_>>()
                                    .join(", ")
                            )
                        },
                    )?;
                }
                if let Some(key) = search_api_key {
                    search.api_key = key.clone();
                }
                settings.search = Some(search);
            }

            // The judge mirrors search: one of three flags means "touch it",
            // the key alone decides whether the tool exists, and an endpoint
            // must be a URL because a typo here becomes a failed call later.
            if judge_endpoint.is_some() || judge_key.is_some() || judge_model.is_some() {
                let mut judge = settings
                    .judge
                    .clone()
                    .unwrap_or_else(nguruvilu::tools::judge::JudgeSettings::default);
                if let Some(endpoint) = judge_endpoint {
                    let endpoint = endpoint.trim().trim_end_matches('/').to_string();
                    if !(endpoint.starts_with("http://") || endpoint.starts_with("https://")) {
                        anyhow::bail!("judge endpoint must be an http(s) URL, got '{endpoint}'");
                    }
                    judge.endpoint = endpoint;
                }
                if let Some(key) = judge_key {
                    judge.api_key = key.clone();
                }
                if let Some(model) = judge_model {
                    judge.model = model.trim().to_string();
                }
                settings.judge = Some(judge);
            }

            let network_problems = settings.network.problems();
            if !network_problems.is_empty() {
                anyhow::bail!("{}", network_problems.join("; "));
            }
            settings.save()?;

            if as_json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({
                        "ok": true,
                        "path": Settings::path().display().to_string(),
                        "base_url": settings.base_url,
                        "api_key": settings.masked_key(),
                        "model": settings.model_or_default(),
                        "configured": settings.is_configured(),
                    }))?
                );
            } else {
                println!("saved to {}", Settings::path().display());
                println!("  endpoint: {}", if settings.base_url.is_empty() { "(not set)" } else { &settings.base_url });
                println!("  key:      {}", if settings.api_key.is_empty() { "(not set)" } else { "(stored)" });
                println!("  model:    {}", settings.model_or_default());
                println!(
                    "  thinking: {}",
                    if settings.reasoning_effort.trim().is_empty() {
                        "(provider default)"
                    } else {
                        &settings.reasoning_effort
                    }
                );
                if let Some(judge) = &settings.judge {
                    println!(
                        "  judge:    {} ({})",
                        judge.endpoint,
                        if judge.is_configured() {
                            "key set — judge tool on"
                        } else {
                            "no key — judge tool off"
                        }
                    );
                }
                if !settings.is_configured() {
                    println!("\nstill missing: {}", settings.missing().join(", "));
                }
            }
            Ok(())
        }
        None | Some(ConfigAction::Show) => {
            // `resolve` is what a request would actually use, so this is the
            // honest answer to "what is configured right now".
            let effective = Settings::resolve();
            let stored = Settings::load()?;

            if as_json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({
                        "path": Settings::path().display().to_string(),
                        "base_url": effective.base_url,
                        "api_key": effective.masked_key(),
                        "model": effective.model_or_default(),
                        "configured": effective.is_configured(),
                        "missing": effective.missing(),
                        "stored_base_url": stored.base_url,
                        "stored_model": stored.model,
                    }))?
                );
                return Ok(());
            }

            println!("settings file: {}", Settings::path().display());
            println!(
                "endpoint:      {}",
                if effective.base_url.is_empty() { "(not set)" } else { &effective.base_url }
            );
            println!(
                "api key:       {}",
                if effective.api_key.is_empty() { "(not set)".to_string() } else { effective.masked_key() }
            );
            println!("model:         {}", effective.model_or_default());
            println!(
                "thinking:      {}",
                if effective.reasoning_effort.trim().is_empty() {
                    "(provider default)"
                } else {
                    &effective.reasoning_effort
                }
            );
            println!(
                "proxy:         {}",
                if effective.proxy.trim().is_empty() {
                    "(direct; HTTP_PROXY/HTTPS_PROXY ignored)"
                } else {
                    &effective.proxy
                }
            );
            let window = effective
                .context_policy(None)
                .resolve(&effective.model_or_default());
            println!(
                "context:       {} tokens ({}){}",
                nguruvilu::size::format_size(window.tokens),
                window.source.as_str(),
                if window.source.is_guess() {
                    "  <- a guess; set --context-window if you know better"
                } else {
                    ""
                }
            );
            println!(
                "compaction:    at {}% of the window, keeping {} recent messages",
                effective.compact_percent, effective.compact_keep_recent
            );
            println!(
                "network:       request {}s, pool idle {}s, {} retry(ies) from {}ms",
                effective.network.request_timeout_secs,
                effective.network.pool_idle_timeout_secs,
                effective.network.retry_attempts,
                effective.network.retry_backoff_ms
            );
            println!(
                "max output:    {}",
                match effective.max_output_tokens {
                    Some(tokens) => format!("{} tokens", nguruvilu::size::format_size(tokens)),
                    None => "(provider default)".to_string(),
                }
            );
            if !effective.is_configured() {
                println!("\nnot configured — set it with:");
                println!("  ngu config set --base-url https://your-host/v1 --api-key sk-...");
            } else if effective.base_url != stored.base_url || effective.api_key != stored.api_key {
                println!("\n(an environment variable is overriding the stored file)");
            }
            Ok(())
        }
    }
}

/// Show the injection entries, and optionally what a piece of text activates.
fn show_injections(for_text: Option<&str>, as_json: bool) -> Result<()> {
    let path = nguruvilu::context::InjectionEngine::default_path();
    let engine = nguruvilu::context::InjectionEngine::load_default();

    // What would activate for the given text, if any.
    let activated: Vec<String> = match for_text {
        Some(text) => {
            let probe = vec![nguruvilu::message::Message::user(text)];
            engine
                .inject(&probe, 128_000, nguruvilu::hotreload::CachePolicy::Balanced)
                .activated
        }
        None => Vec::new(),
    };

    if as_json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "path": path.display().to_string(),
                "exists": path.is_file(),
                "budget_percent": engine.budget_percent,
                "budget_cap": engine.budget_cap,
                "max_recursion": engine.max_recursion,
                "entries": engine.entries.iter().map(|entry| serde_json::json!({
                    "id": entry.id,
                    "constant": entry.constant,
                    "triggers": entry.triggers,
                    "position": format!("{:?}", entry.position),
                    "depth": entry.depth,
                    "order": entry.order,
                    "group": entry.group,
                    "ignore_budget": entry.ignore_budget,
                    "enabled": entry.enabled,
                })).collect::<Vec<_>>(),
                "activated": activated,
            }))?
        );
        return Ok(());
    }

    println!("file: {}", path.display());
    if !path.is_file() {
        println!("(no file yet — write one to add standing rules or triggered hints)");
        return Ok(());
    }
    println!(
        "budget: {}% of the window{}, up to {} recursion passes",
        engine.budget_percent,
        match engine.budget_cap {
            Some(cap) => format!(" capped at {cap} tokens"),
            None => String::new(),
        },
        engine.max_recursion
    );
    println!("entries: {}", engine.len());
    for entry in &engine.entries {
        let when = if entry.constant {
            "always".to_string()
        } else if entry.triggers.is_empty() {
            "never (no triggers)".to_string()
        } else {
            format!("on {}", entry.triggers.join(", "))
        };
        let where_to = match entry.position {
            nguruvilu::context::PositionKind::AtDepth => {
                format!("at-depth {}", entry.depth.unwrap_or(1))
            }
            other => other.as_str().to_string(),
        };
        let mark = if activated.contains(&entry.id) { " *" } else { "  " };
        println!("{mark} {:<20} {:<28} {when}", entry.id, where_to);
    }
    if for_text.is_some() {
        println!("\n* = activates for the given text");
    }
    Ok(())
}

fn require_key(api_key: &str) -> Result<()> {
    if api_key.trim().is_empty() {
        return Err(anyhow!(
            "no API key: pass --api-key or set NGU_API_KEY (or OPENAI_API_KEY)"
        ));
    }
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
    injection: &Arc<Mutex<Arc<nguruvilu::context::InjectionEngine>>>,
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
    if let Some(search) = &content.search {
        search.apply(&mut settings.search);
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
    injection: &'a Arc<Mutex<Arc<nguruvilu::context::InjectionEngine>>>,
    queue: &'a nguruvilu::tools::pack::PendingQueue,
    /// The session's MCP servers: joined when a pack loads, stopped when it
    /// unloads, and stopped for good on the way out.
    mcp: &'a mut Vec<Arc<nguruvilu::mcp::McpClient>>,
    /// The search plugin's settings cell: a hot load may have configured
    /// search, and the plugin re-reads it through `search::configure`.
    search: &'a nguruvilu::tools::search::SettingsCell,
    /// The judge plugin's settings cell — same rule, `judge::configure`.
    judge: &'a nguruvilu::tools::judge::SettingsCell,
    /// An explicit `--persona` from this run: adoption joins the installed
    /// packs' souls, and this outranks them, so it goes back on top after
    /// every drain.
    persona_override: Option<String>,
}

impl PackHost<'_> {
    /// Apply every request recorded during the turn that just ended.
    async fn drain(&mut self) -> Result<()> {
        use nguruvilu::tools::pack::Pending;

        let mut changed = false;
        for action in self.queue.drain() {
            let packs = nguruvilu::pack::installed(&nguruvilu::pack::default_packs_dir())?;
            match action {
                Pending::Unload(name) => {
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
                Pending::Load(name) => {
                    for pack in packs
                        .iter()
                        .filter(|pack| pack.manifest.name == name && pack.enabled)
                    {
                        let label = format!("{}-{}", pack.manifest.name, pack.manifest.version_id);
                        if let Some(assembly) = &pack.assembly {
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
                            for entry in &report.skipped {
                                eprintln!("[pack] skipped {}: {}", entry.id, entry.reason);
                            }
                            changed = true;
                        }
                        match nguruvilu::pack::read_manifest(&pack.path) {
                            Ok(manifest) => {
                                match nguruvilu::content::apply(self.kernel, &pack.path, &manifest)
                                {
                                    Ok(content) => {
                                        let mut guard =
                                            self.runtime.lock().expect("runtime lock");
                                        adopt_content(
                                            content,
                                            self.settings,
                                            &mut *guard,
                                            self.injection,
                                        )?;
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
            // A hot load may have configured search as well: the settings go
            // back into the plugin's cell and its fibers re-apply, so the tool
            // table is in step before it is published.
            nguruvilu::tools::search::configure(
                self.kernel,
                self.search,
                self.settings.search.clone(),
            )?;
            nguruvilu::tools::judge::configure(
                self.kernel,
                self.judge,
                self.settings.judge.clone(),
            )?;
            // The route may have moved with the pack's content, and the tool
            // table certainly did; both are published together so the next
            // turn reads one consistent view.
            let route = model_route(self.cli, self.settings, Some(self.kernel));
            let mut guard = self.runtime.lock().expect("runtime lock");
            guard.apply(Change::session(
                "pack",
                ChangePayload::ModelRoute(route),
            ))?;
            // The join inside adoption just rewrote the persona; a run started
            // with `--persona` outranks packs, so it goes back on top here.
            if let Some(persona) = self.persona_override.as_ref() {
                guard.apply(Change::session(
                    "cli",
                    ChangePayload::Persona(persona.clone()),
                ))?;
            }
            guard.adopt_kernel_tools(self.kernel);
            eprintln!("[pack] the next turn sees the new tool table");
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
    if let Some(kernel) = kernel {
        let view = kernel.service_view(nguruvilu::plugin::RealmMap::new());
        if let Some(network) = nguruvilu::network::NetworkHandle::from_view(&view) {
            route.network = network;
        }
    }
    route
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remote_specs_are_recognised_and_paths_are_not() {
        // The split decides whether the argument is downloaded or installed
        // from disk, so a path must never be reinterpreted as a spec.
        assert!(is_remote_spec("github:id5463/Nguruvilu@packs/browser-use@main"));
        assert!(is_remote_spec("https://example.com/packs/demo-1.0.0.dshpack"));
        assert!(is_remote_spec("http://example.com/demo-1.0.0.dshpack"));
        assert!(!is_remote_spec(r"C:\packs\demo-1.0.0.dshpack"));
        assert!(!is_remote_spec("packs/demo-1.0.0.dshpack"));
        assert!(!is_remote_spec("./github:looks-like-a-spec"));
    }

    #[test]
    fn one_archive_is_picked_and_many_list_themselves() {
        let root = std::env::temp_dir().join(format!(
            "ngu-install-spec-{}",
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(root.join("sub")).unwrap();
        assert!(
            pack_archive_in(&root).is_err(),
            "an empty directory yields nothing, by name"
        );

        std::fs::write(root.join("sub/demo-1.0.0.dshpack"), b"one").unwrap();
        let one = pack_archive_in(&root).unwrap();
        assert!(one.ends_with("demo-1.0.0.dshpack"));

        // A repository holds several packs: refuse and say which, rather than
        // guess and install the wrong one.
        std::fs::write(root.join("other-2.0.0.dshpack"), b"two").unwrap();
        let error = format!("{:#}", pack_archive_in(&root).expect_err("two archives"));
        assert!(error.contains("2 archives"), "{error}");
        assert!(error.contains("demo-1.0.0.dshpack"), "{error}");
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn claude_config_converts_stdio_and_reports_the_rest() {
        let doc = serde_json::json!({
            "mcpServers": {
                "files": { "command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "."] },
                "brave": { "type": "stdio", "command": "node", "args": ["brave.js"],
                           "env": { "BRAVE_API_KEY": "sekret" } },
                "remote": { "type": "http", "url": "https://example.com/mcp" }
            }
        });
        let (servers, skipped) = convert_mcp_servers(&doc).expect("converts");
        assert_eq!(servers.len(), 2);
        let files = servers.iter().find(|s| s.id == "files").expect("files");
        assert_eq!(files.command.as_deref(), Some("npx"));
        assert_eq!(files.transport, "stdio");
        assert_eq!(files.scope.as_deref(), Some("session"));
        assert_eq!(files.args.len(), 3);
        let brave = servers.iter().find(|s| s.id == "brave").expect("brave");
        assert_eq!(brave.env.get("BRAVE_API_KEY").unwrap(), "sekret");

        // The url transport is reported, not smuggled into the pack.
        assert_eq!(skipped.len(), 1);
        assert_eq!(skipped[0].0, "remote");
        assert!(skipped[0].1.contains("stdio only"), "{}", skipped[0].1);

        // And the generated file must pass our own strict reader.
        let text = serde_json::to_string_pretty(&nguruvilu::pack::McpFile { servers }).unwrap();
        let back: nguruvilu::pack::McpFile = serde_json::from_str(&text).unwrap();
        assert_eq!(back.servers.len(), 2);
    }

    #[test]
    fn our_own_servers_shape_passes_through() {
        let doc = serde_json::json!({
            "servers": [{ "id": "browser", "transport": "stdio", "command": "x",
                          "args": ["a"], "scope": "session" }]
        });
        let (servers, skipped) = convert_mcp_servers(&doc).expect("passes");
        assert_eq!(servers.len(), 1);
        assert_eq!(servers[0].id, "browser");
        assert!(skipped.is_empty());
    }

    #[test]
    fn an_entry_without_a_command_is_skipped_and_names_are_derived() {
        let doc = serde_json::json!({ "mcpServers": { "broken": {} } });
        let (servers, skipped) = convert_mcp_servers(&doc).expect("converts around it");
        assert!(servers.is_empty());
        assert_eq!(skipped.len(), 1);
        assert!(skipped[0].1.contains("unknown"), "{}", skipped[0].1);

        assert_eq!(sanitize_pack_name(".mcp"), "mcp");
        assert_eq!(sanitize_pack_name("Claude_Desktop Config"), "claude-desktop-config");
        assert_eq!(sanitize_pack_name("--weird--name--"), "weird-name");
    }
}
