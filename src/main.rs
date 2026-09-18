//! `ngu` — the Nguruvilu command-line interface.
//!
//! Three modes, all over the same kernel:
//!
//! * **interactive** — a terminal REPL (default when stdin is a terminal).
//! * **print** — one prompt in, final text out (`-p`), for scripts and pipes.
//! * **json** — the same run with a structured result (`--json`), which is the
//!   mode other agents drive: it reports the session id, step count, tool
//!   calls, token usage, and the timing breakdown.

use std::io::{IsTerminal, Read, Write};
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use clap::{Parser, Subcommand};

use nguruvilu::agent::{Agent, AgentObserver};
use nguruvilu::llm::{LlmClient, LlmConfig};
use nguruvilu::session::{JsonlStore, Session, SessionStore};
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
    about = "Nguruvilu — an independent agent kernel (four tools, a loop, everything else is a plugin)"
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
    #[arg(long)]
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
    #[arg(long)]
    cwd: Option<PathBuf>,

    /// Session storage root.
    #[arg(long, env = "NGU_HOME")]
    home: Option<PathBuf>,

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

#[tokio::main]
async fn main() {
    if let Err(error) = run().await {
        eprintln!("ngu: {error:#}");
        std::process::exit(1);
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

    // Subcommands that do not need a model route.
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
                println!("{}", serde_json::to_string_pretty(&serde_json::json!({ "models": ids }))?);
            } else {
                for id in ids {
                    println!("{id}");
                }
            }
            return Ok(());
        }
        None => {}
    }

    if let Some(dir) = &cli.cwd {
        std::env::set_current_dir(dir).with_context(|| format!("changing directory to {}", dir.display()))?;
    }

    require_key(&api_key)?;

    let client = LlmClient::new(LlmConfig::new(base_url, api_key, model.clone()))?;
    let tools = Arc::new(ToolRegistry::with_base_tools()?);

    // Resolve the session: explicit id, else the latest, else a fresh one.
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

    let system_prompt = cli.system.clone().unwrap_or_else(|| DEFAULT_SYSTEM_PROMPT.to_string());
    let observer = Arc::new(PrettyObserver { quiet: cli.quiet || cli.json });

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

    match prompt {
        Some(text) => {
            let outcome = run_once(
                client,
                tools,
                system_prompt,
                session.clone(),
                &text,
                cli.max_steps,
                observer,
            )
            .await?;

            session.messages.extend(outcome.new_messages.clone());
            store.append(&session.id, &outcome.new_messages)?;

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
                    "messages": outcome.new_messages,
                });
                println!("{}", serde_json::to_string_pretty(&payload)?);
            } else {
                // In pretty mode the text already streamed to stdout; in quiet
                // mode nothing was printed yet, so emit the final text.
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
                }
            }
        }
        None => {
            interactive(client, tools, system_prompt, session, &store, cli.max_steps).await?;
        }
    }

    Ok(())
}

async fn run_once(
    client: LlmClient,
    tools: Arc<ToolRegistry>,
    system_prompt: String,
    session: Session,
    prompt: &str,
    max_steps: usize,
    observer: Arc<PrettyObserver>,
) -> Result<nguruvilu::agent::AgentOutcome> {
    let mut agent = Agent::new(client, tools, system_prompt, session.messages.clone())
        .with_max_steps(max_steps)
        .with_observer(observer);
    agent.run(prompt).await
}

async fn interactive(
    client: LlmClient,
    tools: Arc<ToolRegistry>,
    system_prompt: String,
    mut session: Session,
    store: &JsonlStore,
    max_steps: usize,
) -> Result<()> {
    println!("Nguruvilu {} — session {}", env!("CARGO_PKG_VERSION"), session.id);
    println!("Type a prompt, or /exit to quit, /new for a new session.\n");

    let mut agent = Agent::new(client, tools, system_prompt, session.messages.clone())
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
        println!("{}", serde_json::to_string_pretty(&serde_json::json!({ "sessions": sessions }))?);
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
    println!("session {} ({} messages, created {})", session.id, session.messages.len(), session.created_at);
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
        if !message.tool_calls.is_empty() {
            for call in &message.tool_calls {
                println!("[{label}] → {}({})", call.name, call.arguments);
            }
        }
        let text = message.text();
        if !text.is_empty() {
            println!("[{label}] {text}");
        }
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
