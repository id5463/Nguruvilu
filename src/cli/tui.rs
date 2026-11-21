//! 交互式模式：一行一行读输入，跑一轮，把答案打出来。
//!
//! 这不是一个 REPL 的实现细节，而是"人在这里说话"的那条路径：它读 stdin、
//! 显示进度、处理 Ctrl-C、在退出时留下会话。换掉它就换掉了人跟它对话的方式
//! （换行编辑、换历史、换多行输入），而不必碰下面跑回合的那部分。

use std::io::{IsTerminal, Read, Write};
use std::sync::Arc;

use anyhow::{anyhow, Result};

use nguruvilu::agent::Agent;
use nguruvilu::session::{JsonlStore, SessionStore};

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
        .with_installed_loop()
        .with_observer(Arc::new(PrettyObserver { quiet: false }))
        // The interactive session this loop serves, fixed for every turn it
        // runs (the id and workspace do not change while the loop is alive).
        .with_identity(nguruvilu::tools::ToolIdentity::for_session(
            session.id.clone(),
            session.cwd.as_deref().map(std::path::PathBuf::from),
        ));

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
        let original_len = session.messages.len();
        let turn = agent.run(input).await;
        let turn_ms = turn_started.elapsed().as_millis() as u64;
        match &turn {
            Ok(outcome) if outcome.error.is_none() => events.emit(
                nguruvilu::events::TURN_COMPLETE,
                &serde_json::json!({
                    "session": &session.id,
                    "ok": true,
                    "duration_ms": turn_ms,
                    "steps": outcome.steps,
                    "tool_calls": outcome.tool_calls,
                    "retries": outcome.retries,
                }),
            ),
            Ok(outcome) => events.emit(
                nguruvilu::events::TURN_COMPLETE,
                &serde_json::json!({
                    "session": &session.id,
                    "ok": false,
                    "duration_ms": turn_ms,
                    "steps": outcome.steps,
                    "retries": outcome.retries,
                    "error": nguruvilu::events::truncate(
                        outcome.error.as_deref().unwrap_or("turn failed"),
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
        match turn {
            Ok(outcome) => {
                // Read before the messages are moved out of the outcome: the
                // stats are a snapshot of the turn, taken here.
                let stats = TurnStats::from_outcome(&outcome);
                // Recorded even when the turn failed: the agent kept what it
                // produced, and a REPL that forgets half a turn is a REPL that
                // lies about the conversation on the next input.
                store.append(&session.id, &outcome.new_messages)?;
                session.messages.extend(outcome.new_messages);

                // Record what was injected and what was compacted, the same way
                // the one-shot path does. Without the compaction line a reload
                // would replay messages the model never saw again, and the REPL
                // would diverge from the file as it accumulates turns.
                for record in &outcome.injections {
                    store.append_injection(&session.id, record)?;
                }
                for compaction in &outcome.compactions {
                    store.append_compaction(&session.id, compaction)?;
                    eprintln!(
                        "[compacted {} messages into a summary of {} chars]",
                        compaction.replaced,
                        compaction.summary.chars().count()
                    );
                }
                // The live history must match what was recorded; a compaction
                // shrinks it, and the next turn's split point is read from here.
                if !outcome.compactions.is_empty() {
                    session.messages = agent.messages().to_vec();
                }
                record_turn_stats(&store, &session.id, &stats);

                let retries = if outcome.retries > 0 {
                    format!(" | {} retries", outcome.retries)
                } else {
                    String::new()
                };
                println!(
                    "\n[{} steps | {} tool calls{} | {} in / {} out ({} cached, {} written to cache) | model {} ms, tools {} ms]\n",
                    outcome.steps,
                    outcome.tool_calls,
                    retries,
                    outcome.usage.input,
                    outcome.usage.output,
                    outcome.usage.cached,
                    outcome.usage.cache_created,
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
                if let Some(error) = outcome.error {
                    eprintln!("turn interrupted: {error}\n");
                }
            }
            Err(error) => {
                // The agent still holds everything it committed before dying;
                // record it so the failure costs the error message, not the
                // progress that was already made.
                //
                // A mid-turn compaction can leave the history shorter than it
                // was when the turn started, which is why this is a `get`: the
                // slice index would panic there and hide the real error behind
                // a crash.
                let salvaged = agent
                    .messages()
                    .get(original_len..)
                    .unwrap_or(&[])
                    .to_vec();
                if !salvaged.is_empty() {
                    store.append(&session.id, &salvaged)?;
                    session.messages.extend(salvaged);
                }
                record_turn_stats(&store, &session.id, &TurnStats::unreported());
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

/// One session's totals as the JSON payload's `totals` object.
///
/// The field names are the contract: the desktop builds the same object from


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

