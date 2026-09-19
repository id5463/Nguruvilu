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
use nguruvilu::plugin::Kernel;
use nguruvilu::session::{JsonlStore, Session, SessionStore};
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
    /// Print the runtime's policy table and effective configuration.
    Runtime,
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

    let base_url = cli
        .base_url
        .clone()
        .or_else(|| std::env::var("OPENAI_BASE_URL").ok())
        .unwrap_or_else(|| DEFAULT_BASE_URL.to_string());
    let api_key = cli
        .api_key
        .clone()
        .or_else(|| std::env::var("OPENAI_API_KEY").ok())
        .unwrap_or_default();
    let model = cli.model.clone().unwrap_or_else(|| DEFAULT_MODEL.to_string());
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
            let client = LlmClient::new(LlmConfig::new(base_url, api_key, model))?;
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

    let client = LlmClient::new(LlmConfig::new(base_url.clone(), api_key, model.clone()))?;

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

        let pack = file
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "pack".into());

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

    let mut runtime = Runtime::new(Arc::clone(&tools))
        .with_route(ModelRoute {
            provider: "openai".into(),
            base_url: base_url.clone(),
            api_key: String::new(),
            model: model.clone(),
            temperature: None,
            max_tokens: None,
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

    let config: Arc<dyn TurnConfig> = Arc::new(CliConfig {
        runtime: Mutex::new(runtime),
        base_prompt,
        skills: Mutex::new(skills),
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

    let pack = file
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "pack".into());

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

fn require_key(api_key: &str) -> Result<()> {
    if api_key.trim().is_empty() {
        return Err(anyhow!(
            "no API key: pass --api-key or set NGU_API_KEY (or OPENAI_API_KEY)"
        ));
    }
    Ok(())
}
