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
use std::path::PathBuf;
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
use nguruvilu::llm::{LlmClient, LlmConfig};
use nguruvilu::loader::Loader;
use nguruvilu::plugin::{Kernel, Plugin as PluginTrait};
use nguruvilu::session::{JsonlStore, Session, SessionStore};
use nguruvilu::settings::Settings;
use nguruvilu::window::{ContextPolicy, DefaultContextPolicy};
use nguruvilu::skills::{register_skill_tool, SkillRegistry};
use nguruvilu::tools::ToolRegistry;

/// Default model when neither `--model` nor `NGU_MODEL` is set.
const DEFAULT_MODEL: &str = "deepseek-v4.1-flash";

/// Default API base when neither `--base-url` nor the environment supplies one.
const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

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
    },
    /// Install a `.dshpack` archive.
    Install {
        /// Archive to install.
        file: PathBuf,
        /// Directory to install into.
        #[arg(long)]
        into: Option<PathBuf>,
    },
    /// List installed packs.
    Packs,
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
        /// Context window in tokens. Pass 0 to go back to automatic.
        #[arg(long)]
        context_window: Option<usize>,
        /// Compact once this percentage of the window is in use.
        #[arg(long)]
        compact_percent: Option<u32>,
        /// Recent messages kept verbatim when compacting.
        #[arg(long)]
        compact_keep_recent: Option<usize>,
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

/// Prints progress to stderr, keeping stdout clean for the answer.
struct PrettyObserver {
    quiet: bool,
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
    runtime: Mutex<Runtime>,
    base_prompt: String,
    skills: Mutex<SkillRegistry>,
    /// Context window, threshold, and how much survives a compaction.
    policy: Mutex<Arc<dyn ContextPolicy>>,
}

impl TurnConfig for CliConfig {
    fn settings(&self) -> TurnSettings {
        let runtime = self.runtime.lock().expect("runtime lock");
        let snapshot = runtime.snapshot();
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
        }
    }
}

#[tokio::main]
async fn main() {
    if let Err(error) = run().await {
        eprintln!("ngu: {error:#}");
        std::process::exit(1);
    }
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
    let settings = Settings::resolve();
    let base_url = cli
        .base_url
        .clone()
        .unwrap_or_else(|| settings.base_url.clone());
    let api_key = cli
        .api_key
        .clone()
        .unwrap_or_else(|| settings.api_key.clone());
    let model = cli
        .model
        .clone()
        .unwrap_or_else(|| settings.model_or_default());
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
            let client = client(&base_url, &api_key, &model, &settings)?;
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
        Some(Command::Pack { dir, out }) => return pack_command(dir, out.as_ref(), cli.json),
        Some(Command::Install { file, into }) => return install_command(file, into.as_ref(), cli.json),
        Some(Command::Packs) => return list_packs(cli.json),
        Some(Command::Verify { file }) => return verify_command(file, cli.json),
        Some(Command::Plugin { action }) => return plugin_command(action, cli.json),
        Some(Command::Config { action }) => return config_command(action.as_ref(), cli.json),
        Some(Command::Runtime) => {
            return show_runtime(&model, &base_url, cache_policy, &skills, cli.json)
        }
        None => {}
    }

    if let Some(dir) = &cli.cwd {
        std::env::set_current_dir(dir)
            .with_context(|| format!("changing directory to {}", dir.display()))?;
    }

    require_key(&api_key)?;

    let client = client(&base_url, &api_key, &model, &settings)?;

    // The kernel owns the tool table; skills add one tool to it.
    let mut kernel = Kernel::new();
    if !skills.is_empty() {
        register_skill_tool(kernel.tools_mut(), Arc::new(skills.clone()))?;
    }

    // An assembly manifest loads before the session starts, so its plugins,
    // MCP servers, and skills are in place for the first turn.
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

        // Take the kernel back; the loader stops any MCP servers it started.
        kernel = loader.finish().await;
    }

    let tools = Arc::new(kernel.tools().clone());

    // The flag wins over the stored setting.

    let effective_effort = cli
        .reasoning_effort
        .clone()
        .unwrap_or_else(|| settings.reasoning_effort.clone());

    let mut runtime = Runtime::new(Arc::clone(&tools))
        .with_route(ModelRoute {
            provider: "openai".into(),
            base_url: base_url.clone(),
            api_key: String::new(),
            model: model.clone(),
            temperature: None,
            max_tokens: None,
            reasoning_effort: if effective_effort.trim().is_empty() {
                None
            } else {
                Some(effective_effort.clone())
            },
        })
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
    if let Some(persona) = &cli.persona {
        runtime.apply(Change::session(
            "cli",
            ChangePayload::Persona(persona.clone()),
        ))?;
    }

    // The context policy comes from settings, so `--context-window` and the
    // settings file both land here. A provider-reported window would be folded
    // in the same way once the catalog has been fetched.
    let policy: Arc<dyn ContextPolicy> = Arc::new(settings.context_policy(None));

    let config: Arc<dyn TurnConfig> = Arc::new(CliConfig {
        runtime: Mutex::new(runtime),
        base_prompt,
        skills: Mutex::new(skills),
        policy: Mutex::new(policy),
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
        Session::new(Some(model.clone()))
    } else if let Some(id) = &cli.session {
        store
            .load(id)?
            .ok_or_else(|| anyhow!("session not found: {id}"))?
    } else {
        Session::new(Some(model.clone()))
    };
    let is_new = !store.root().join(format!("{}.jsonl", session.id)).exists();
    if is_new {
        store.create(&session)?;
    }

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

    match prompt {
        Some(text) => {
            if let Some(git) = &snapshotter {
                git.snapshot("before turn")?;
            }

            let mut agent = Agent::new(client, Arc::clone(&config), session.messages.clone())
                .with_max_steps(cli.max_steps)
                .with_observer(observer);
            let outcome = agent.run(&text).await?;

            session.messages.extend(outcome.new_messages.clone());
            store.append(&session.id, &outcome.new_messages)?;

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
            interactive(client, config, session, &store, cli.max_steps, snapshotter).await?;
        }
    }

    Ok(())
}

async fn interactive(
    client: LlmClient,
    config: Arc<dyn TurnConfig>,
    mut session: Session,
    store: &JsonlStore,
    max_steps: usize,
    snapshotter: Option<GitSnapshot>,
) -> Result<()> {
    println!(
        "Nguruvilu {} — session {}",
        env!("CARGO_PKG_VERSION"),
        session.id
    );
    println!("Type a prompt, or /exit to quit, /new for a new session.\n");

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
            session = Session::new(session.model.clone());
            store.create(&session)?;
            agent.set_messages(Vec::new());
            println!("new session {}\n", session.id);
            continue;
        }

        if let Some(git) = &snapshotter {
            if let Some(record) = git.snapshot("before turn")? {
                eprintln!("[snapshot {}]", record.short());
            }
        }

        match agent.run(input).await {
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

fn show_runtime(
    model: &str,
    base_url: &str,
    cache_policy: CachePolicy,
    skills: &SkillRegistry,
    as_json: bool,
) -> Result<()> {
    let tools = Arc::new(ToolRegistry::with_base_tools()?);
    let runtime = Runtime::new(tools);

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

    if as_json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "model": model,
                "base_url": base_url,
                "cache_policy": format!("{cache_policy:?}"),
                "policy": policy,
                "skill_roots": skills.roots().iter().map(|r| r.display().to_string()).collect::<Vec<_>>(),
                "skill_count": skills.len(),
                "change_kinds": ChangeKind::all().iter().map(|k| k.as_str()).collect::<Vec<_>>(),
            }))?
        );
        return Ok(());
    }

    println!("model:        {model}");
    println!("base url:     {base_url}");
    println!("cache policy: {cache_policy:?}");
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

fn pack_command(dir: &PathBuf, out: Option<&PathBuf>, as_json: bool) -> Result<()> {
    let manifest = nguruvilu::pack::read_manifest(dir)?;
    let archive = match out {
        Some(path) => path.clone(),
        None => PathBuf::from(format!("{}-{}.dshpack", manifest.name, manifest.version_id)),
    };

    let packed = nguruvilu::pack::pack(dir, &archive)?;
    let contents = nguruvilu::pack::inspect(dir)?;

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

fn install_command(file: &PathBuf, into: Option<&PathBuf>, as_json: bool) -> Result<()> {
    let packs_dir = into.cloned().unwrap_or_else(nguruvilu::pack::default_packs_dir);
    let placed = nguruvilu::pack::install(file, &packs_dir)?;

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
                })).collect::<Vec<_>>(),
            }))?
        );
        return Ok(());
    }

    if packs.is_empty() {
        println!("no packs installed in {}", packs_dir.display());
        return Ok(());
    }
    println!("{:<24} {:<12} {:<10} {}", "name", "version", "license", "assembly");
    for pack in packs {
        println!(
            "{:<24} {:<12} {:<10} {}",
            pack.manifest.name,
            pack.manifest.version_id,
            pack.manifest.license,
            if pack.assembly.is_some() { "yes" } else { "no" }
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
                    "requires": report.manifest.dependencies.kernel,
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
    if let Some(range) = &report.manifest.dependencies.kernel {
        println!("requires kernel {range}");
    }
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
                // Zero means "go back to working it out".
                settings.context_window = if *value == 0 { None } else { Some(*value) };
            }
            if let Some(value) = compact_percent {
                settings.compact_percent = (*value).min(100);
            }
            if let Some(value) = compact_keep_recent {
                settings.compact_keep_recent = (*value).max(1);
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
                window.tokens,
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

fn require_key(api_key: &str) -> Result<()> {
    if api_key.trim().is_empty() {
        return Err(anyhow!(
            "no API key: pass --api-key or set NGU_API_KEY (or OPENAI_API_KEY)"
        ));
    }
    Ok(())
}

/// Build a model client, applying the configured proxy.
///
/// The proxy is explicit: when it is empty, the process's `HTTP_PROXY` and
/// `HTTPS_PROXY` are deliberately ignored. Picking those up silently is how a
/// proxy set for another tool reroutes model traffic and fails with a TLS
/// handshake error that never mentions proxies.
fn client(base_url: &str, api_key: &str, model: &str, settings: &Settings) -> Result<LlmClient> {
    let mut config = LlmConfig::new(base_url, api_key, model);
    config.proxy = settings.proxy.clone();
    LlmClient::new(config).context("building the model client")
}
