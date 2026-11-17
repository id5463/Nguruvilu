//! The `delegate` plugin: run a subtask in its own kernel process.
//!
//! # Why a process, and not a function in the kernel
//!
//! This used to be `src/tools/subagent.rs` — 962 lines that reached for twelve
//! kernel types (`ModelAccess`, `TurnConfig`, `Message`, `ModelHandle` …) in
//! order to build an `Agent` inside the process. But the kernel already *is* an
//! agent runner: `ngu -p "…"` runs a complete one. Assembling a second one by
//! hand was duplication, and the copy could only ever be the one kind of
//! subagent that copy knew how to build.
//!
//! As a process:
//!
//!   * a subtask has its own session, its own failure, its own exit — a stuck
//!     subtask is killed and the conversation that started it is untouched;
//!   * `--bare --packs none` starts it in 162 ms with no persona, no injection
//!     rules, and none of the MCP servers this conversation happens to have;
//!   * a different kind of subagent is a different pack, not a kernel change.
//!
//! # The context is empty, and that is the point
//!
//! `ngu -p` starts a new session, so the subtask cannot see what was said here.
//! A task has to stand on its own: what to do, where, and what counts as
//! finished. That is the whole value — a large side investigation stays outside
//! this transcript, and only the answer comes back.
//!
//! # Recursion
//!
//! A subtask can call `delegate` itself, so the depth travels in the
//! environment: each process knows which layer it is, and refuses past the
//! limit. Without it one instruction can spawn an unbounded tree and nothing in
//! the transcript shows where the budget went.

use std::io::{Read, Write};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

/// The tool's name, as the model sees it.
const TOOL: &str = "delegate";

/// How many steps a subtask gets. The kernel's own default used to be 12; this
/// is the same number, and a caller can move it with `NGU_SUBAGENT_STEPS`.
const DEFAULT_STEPS: u64 = 12;

/// Wall clock a subtask may run before it is stopped.
const DEFAULT_MINUTES: u64 = 15;

/// How deep subagents may nest.
const DEFAULT_DEPTH: u64 = 2;

fn main() {
    let mut stdin = std::io::stdin();
    loop {
        let Some(body) = read_frame(&mut stdin) else {
            return;
        };
        let Ok(request) = serde_json::from_slice::<Value>(&body) else {
            continue;
        };
        match request.get("op").and_then(Value::as_str).unwrap_or("") {
            "describe" => write_frame(&json!({ "tools": [tool()] })),
            "call" => {
                let args = request.get("args").cloned().unwrap_or_else(|| json!({}));
                match run(&args) {
                    Ok(text) => write_frame(&json!({ "result": text })),
                    Err(error) => write_frame(&json!({ "error": error })),
                }
            }
            // The kernel asks these of every plugin. This one has nothing to
            // register and no state to answer, but an unanswered frame would
            // leave the kernel waiting, so it says so rather than staying quiet.
            "state" | "register" | "next" | "ask" => write_frame(&json!({ "ok": true })),
            _ => write_frame(&json!({ "error": "unsupported op" })),
        }
    }
}

/// The tool as the model sees it.
///
/// The description states the budgets as numbers because they are the numbers in
/// force: a caller that is told "12 steps" can size its task to fit, and one told
/// "some steps" cannot.
fn tool() -> Value {
    let steps = std::env::var("NGU_SUBAGENT_STEPS")
        .ok()
        .and_then(|raw| raw.parse().ok())
        .unwrap_or(DEFAULT_STEPS);
    let minutes = std::env::var("NGU_SUBAGENT_MINUTES")
        .ok()
        .and_then(|raw| raw.parse().ok())
        .unwrap_or(DEFAULT_MINUTES);
    json!({
        "name": TOOL,
        "description": format!(
            "Run a self-contained subtask in a fresh kernel process with no memory of this \
             conversation, and get back only its answer. Use it to keep a large side \
             investigation out of this transcript. The subtask sees the same tools and the \
             same model you do, but not what was said here — so state everything it needs, \
             including paths and what 'done' looks like. It cannot ask you anything. It stops \
             after {steps} steps or {minutes} minutes. It is one process: it can be killed, it \
             can fail on its own, and it does not touch this conversation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The whole subtask, standing on its own: what to do, where, \
                                    and what counts as finished."
                },
                "model": {
                    "type": "string",
                    "description": "Optional model for the subtask. Defaults to the session's. \
                                    A cheaper model is usually right for mechanical work."
                },
                "packs": {
                    "type": "string",
                    "description": "Which packs the subtask loads: `none` (base tools only, \
                                    121 ms), or `all` (this machine's packs, 339 ms). Pass `none` \
                                    unless the task needs a tool this conversation has."
                }
            },
            "required": ["task"]
        }
    })
}

fn run(args: &Value) -> Result<String, String> {
    let task = args
        .get("task")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|task| !task.is_empty())
        .ok_or_else(|| {
            "missing required argument: task. Say what to do, where, and what counts as \
             finished."
                .to_string()
        })?;

    // The depth this process was started at, plus the one this call would add.
    let depth: u64 = std::env::var("NGU_DEPTH")
        .ok()
        .and_then(|raw| raw.parse().ok())
        .unwrap_or(0);
    let limit: u64 = std::env::var("NGU_DEPTH_LIMIT")
        .ok()
        .and_then(|raw| raw.parse().ok())
        .unwrap_or(DEFAULT_DEPTH);
    if depth >= limit {
        return Err(format!(
            "subagents are already {depth} deep; this one may not delegate further. Do the \
             work here instead, or raise NGU_DEPTH_LIMIT."
        ));
    }

    let binary = find_kernel()?;

    // `none` means the four base tools and nothing else, which is right for
    // most subtasks and starts in 121 ms against 339 ms. The default is `all`:
    // a caller that did not think about it should get what this machine has
    // rather than a subtask that mysteriously cannot read a ticket.
    let packs = args
        .get("packs")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|packs| !packs.is_empty())
        .unwrap_or("all");

    let mut argv: Vec<String> = vec![
        "--bare".into(),
        "--packs".into(),
        packs.into(),
        "--json".into(),
        "-p".into(),
        task.into(),
    ];
    if let Some(model) = args.get("model").and_then(Value::as_str) {
        if !model.trim().is_empty() {
            argv.push("--model".into());
            argv.push(model.trim().into());
        }
    }
    if let Ok(steps) = std::env::var("NGU_SUBAGENT_STEPS") {
        if !steps.trim().is_empty() {
            argv.push("--max-steps".into());
            argv.push(steps.trim().into());
        }
    }

    let mut child = Command::new(&binary)
        .args(&argv)
        .env("NGU_DEPTH", (depth + 1).to_string())
        // A subtask that inherited this process's stdin would read the parent's
        // request frames as its prompt when `-p` was empty. `-p` is always given
        // here, but closing it is the honest statement of intent.
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("starting a subagent ({}): {error}", binary.display()))?;

    // The wall clock. A subtask that hangs must not hold this call open for
    // ever: the caller cannot tell "slow" from "wedged" and has no way to ask.
    let deadline = Duration::from_secs(
        std::env::var("NGU_SUBAGENT_MINUTES")
            .ok()
            .and_then(|raw| raw.parse::<u64>().ok())
            .unwrap_or(DEFAULT_MINUTES)
            * 60,
    );
    let started = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) => {
                if started.elapsed() > deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(format!(
                        "the subagent ran past {} minutes and was stopped. Split the task rather \
                         than making one subtask do everything.",
                        deadline.as_secs() / 60
                    ));
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(error) => return Err(format!("waiting on the subagent: {error}")),
        }
    }

    let output = child
        .wait_with_output()
        .map_err(|error| format!("collecting the subagent's output: {error}"))?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    if !output.status.success() {
        return Err(format!(
            "the subagent failed ({}):\n{}",
            output.status,
            truncate(stderr.trim(), 800)
        ));
    }
    read_result(&stdout, &stderr)
}

/// The answer out of `ngu --json`.
///
/// `ngu` writes its progress to stderr and only the payload to stdout, so a
/// non-empty stdout is the JSON object. Anything else is reported as text rather
/// than discarded — a subtask that answered in prose is still an answer.
fn read_result(stdout: &str, stderr: &str) -> Result<String, String> {
    let trimmed = stdout.trim();
    if trimmed.is_empty() {
        return Err(format!(
            "the subagent said nothing.\n{}",
            truncate(stderr.trim(), 800)
        ));
    }
    let Ok(payload) = serde_json::from_str::<Value>(trimmed) else {
        return Ok(truncate(trimmed, 8000));
    };
    if payload.get("ok").and_then(Value::as_bool) == Some(false) {
        let reason = payload
            .get("error")
            .and_then(Value::as_str)
            .unwrap_or("the subagent reported failure without saying why");
        return Err(reason.to_string());
    }
    let text = payload
        .get("text")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    if text.is_empty() {
        return Ok("(the subtask finished without saying anything)".into());
    }
    // The step count goes into the answer because it tells the caller how much
    // to trust it: a subtask that spent its whole budget is usually a truncated
    // answer, and the parent should split the work rather than build on it.
    match payload.get("steps").and_then(Value::as_u64) {
        Some(steps) if steps > 0 => Ok(format!("{text}\n\n（子代理用了 {steps} 步）")),
        _ => Ok(text),
    }
}

/// Find the kernel binary to start.
///
/// `NGU_EXE` first, because a pack can live anywhere and the host knows where it
/// installed the kernel. Then `PATH`, so a normal install needs no configuration.
/// Then next to this plugin and up a few directories, which is where a checkout
/// keeps it — that is what makes `packs/delegate` work when run from a clone.
fn find_kernel() -> Result<std::path::PathBuf, String> {
    if let Ok(configured) = std::env::var("NGU_EXE") {
        let path = std::path::PathBuf::from(configured);
        if path.is_file() {
            return Ok(path);
        }
    }
    if let Some(found) = search_path() {
        return Ok(found);
    }
    let mut dir = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(std::path::Path::to_path_buf))
        .ok_or_else(|| "cannot locate this plugin's own directory".to_string())?;
    for _ in 0..6 {
        for suffix in ["", ".exe"] {
            for sub in ["", "target/release/", "target/debug/"] {
                let candidate = dir.join(format!("{sub}ngu{suffix}"));
                if candidate.is_file() {
                    return Ok(candidate);
                }
            }
        }
        match dir.parent() {
            Some(parent) => dir = parent.to_path_buf(),
            None => break,
        }
    }
    Err("cannot find the kernel binary (`ngu`): set NGU_EXE to its path".into())
}

fn search_path() -> Option<std::path::PathBuf> {
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        for name in ["ngu.exe", "ngu"] {
            let candidate = dir.join(name);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    None
}

fn truncate(text: &str, max: usize) -> String {
    let trimmed = text.trim();
    if trimmed.chars().count() <= max {
        return trimmed.to_string();
    }
    let mut out: String = trimmed.chars().take(max).collect();
    out.push('…');
    out
}

/* ---------------------------------------------------------------- framing */

/// Read one `[u32 big-endian length][JSON]` frame. `None` is end of file.
fn read_frame(input: &mut impl Read) -> Option<Vec<u8>> {
    let mut header = [0u8; 4];
    // A clean EOF at the start of a frame is the kernel closing the pipe, which
    // is how a plugin is told to stop.
    if input.read_exact(&mut header).is_err() {
        return None;
    }
    let len = u32::from_be_bytes(header) as usize;
    let mut body = vec![0u8; len];
    input.read_exact(&mut body).ok()?;
    Some(body)
}

/// Write one frame and flush it.
///
/// Flushed every time: a plugin that buffered its answer would hang the kernel
/// until the buffer filled, and the kernel has no way to ask for a partial frame.
fn write_frame(value: &Value) {
    let body = match serde_json::to_vec(value) {
        Ok(body) => body,
        Err(_) => return,
    };
    let mut out = std::io::stdout();
    let mut header = [0u8; 4];
    header.copy_from_slice(&(body.len() as u32).to_be_bytes());
    let _ = out.write_all(&header);
    let _ = out.write_all(&body);
    let _ = out.flush();
}
