//! The `restart` plugin: let the agent restart the kernel it runs in.
//!
//! This used to live in the kernel (`src/tools/restart.rs`, 386 lines), and the
//! move is the point of the exercise: the kernel no longer contains a single line
//! about restarting, and the capability arrived as a pack.
//!
//! # What a plugin is
//!
//! A process. It reads one length-prefixed JSON frame from stdin and writes one
//! back to stdout. The kernel starts it once and keeps it — measured at 45 ms to
//! spawn against 0.25 ms per round trip, which is why nothing here reconnects.
//!
//! # How it reaches the kernel
//!
//! It does not need to. It asks the kernel for `kernel.pid` over the protocol,
//! and `kill(pid)` is a restart request — measured to work on Windows and Linux.
//! That is why restarting does not have to be kernel business: any process
//! running as the user can signal any other.
//!
//! # Why it refuses
//!
//! Signalling the kernel is the easy half. The hard half is knowing whether
//! anything will start it again: under `systemd` with `Restart=always`, or a
//! container, exiting *is* the restart; started by hand in a terminal, it is just
//! death. So this checks first and says what to do instead — the same three
//! decisions the kernel version made, which were right:
//!
//! * delayed, so the tool's answer reaches the model before the process goes;
//! * refused when nothing will restart it, with the alternative named;
//! * never on a wrong guess: `reason` is required and logged first, so an
//!   unattended restart has a cause on disk.

use std::io::{Read, Write};

use serde_json::{json, Value};

/// The tool's name.
const TOOL: &str = "restart";

/// How long to wait before signalling, so the tool's answer lands first.
const GRACE_MS: u64 = 700;

fn main() {
    // stdin holds frames until the kernel closes it, which is the signal to
    // leave. No threads and no async: a plugin this small does not need them,
    // and a panic here would take down only this process.
    let mut stdin = std::io::stdin();
    loop {
        let frame = match read_frame(&mut stdin) {
            Some(frame) => frame,
            None => return,
        };
        let message: Value = match serde_json::from_slice(&frame) {
            Ok(value) => value,
            Err(error) => {
                write_frame(&json!({ "error": format!("not JSON: {error}") }));
                continue;
            }
        };

        let answer = match message.get("op").and_then(|op| op.as_str()) {
            Some("describe") => json!({
                "tools": [{
                    "name": TOOL,
                    "description": DESCRIPTION,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reason": {
                                "type": "string",
                                "description": "One line: what changed that needs a restart. Logged."
                            },
                            "delay_ms": {
                                "type": "integer",
                                "description": "Wait before signalling (default 700). Raise it if the caller needs longer."
                            }
                        },
                        "required": ["reason"]
                    }
                }]
            }),
            Some("call") if message.get("tool").and_then(|t| t.as_str()) == Some(TOOL) => {
                run(message.get("args").cloned().unwrap_or(Value::Null))
            }
            Some("call") => json!({ "error": format!("no such tool: {}", message.get("tool").unwrap_or(&Value::Null)) }),
            Some(other) => json!({ "error": format!("unsupported op: {other}") }),
            None => json!({ "error": "missing op" }),
        };
        write_frame(&answer);
    }
}

const DESCRIPTION: &str = "Restart the process this agent runs in. Use it after changing something \
     that is only read at startup: an installed or unloaded pack, a settings \
     file, a rebuilt binary. The current turn finishes and its answer is \
     delivered first; the process is then signalled and its supervisor starts it \
     again. Refused when nothing is supervising it — then `systemctl restart` \
     from `bash` is the safe alternative. Say why, in one line; it is logged \
     before the exit so an unattended restart has a cause on disk.";

/// One `call`. Returns the frame to write back.
fn run(args: Value) -> Value {
    let reason = match args.get("reason").and_then(|r| r.as_str()).map(str::trim) {
        Some(text) if !text.is_empty() => text.to_string(),
        _ => {
            return json!({
                "error": "missing required argument: reason. An unattended restart \
                          with no stated cause is not debuggable afterwards."
            })
        }
    };

    let plan = match supervision() {
        Ok(plan) => plan,
        Err(why) => return json!({ "result": why }),
    };

    // The kernel's pid, asked for over the protocol rather than read from the
    // environment. Same value, and it is part of the vocabulary `describe`
    // exposes instead of a start-up name nobody would find.
    let pid = match ask_state(&json!({ "state": "kernel.pid" })).and_then(|v| v.as_u64()) {
        Some(pid) => pid as u32,
        None => {
            return json!({
                "error": "the kernel did not answer `kernel.pid`, so this plugin cannot \n                          name the process to restart."
            })
        }
    };

    let delay = args
        .get("delay_ms")
        .and_then(|v| v.as_u64())
        .unwrap_or(GRACE_MS);

    // On stderr before the signal: under systemd this lands in the journal, which
    // is where anyone asking "why did it restart at 3am" will look.
    eprintln!("[restart] requested: {reason}");

    let how = match plan {
        Plan::Supervised => "the systemd unit will start it again",
        Plan::Adopted => "it is PID-1's child, so its supervisor starts it again",
        Plan::Declared => "the supervisor that started it was declared, so it starts it again",
    };

    // A detached thread so this answer is written before the signal lands.
    std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(delay));
        eprintln!("[restart] signalling kernel pid {pid}; {how}");
        signal(pid);
        // If the signal did not take, exit anyway: a plugin that survives its own
        // restart request is a pack that cannot be reloaded.
        std::thread::sleep(std::time::Duration::from_millis(1000));
        std::process::exit(0);
    });

    json!({
        "result": format!(
            "Restarting in {} ms ({how}).\nReason logged: {reason}\n\
             This reply is delivered first; the kernel is then signalled. Anything \
             in flight — a running turn, a background job — will not survive.",
            delay
        )
    })
}

/// Why the kernel will come back, if it will.
enum Plan {
    /// A systemd unit owns it; signalling is the restart.
    Supervised,
    /// Parent is PID 1, which is what a container looks like from inside.
    Adopted,
    /// Something set `NGU_SUPERVISED`; the deployment said so itself.
    Declared,
}

/// Whether signalling the kernel leads to a running kernel.
///
/// The checks are the ones that were in the kernel, moved here: they are about
/// the *deployment*, and a plugin is as good a place to know a deployment as the
/// kernel is. `NGU_SUPERVISED` is checked first because it is the only signal
/// that works when the supervisor is something nobody recognises.
fn supervision() -> Result<Plan, String> {
    if std::env::var_os("NGU_SUPERVISED").is_some() {
        return Ok(Plan::Declared);
    }
    if std::env::var_os("INVOCATION_ID").is_some() || std::env::var_os("JOURNAL_STREAM").is_some() {
        return Ok(Plan::Supervised);
    }
    let parent = parent_pid();
    if parent == Some(1) {
        return Ok(Plan::Adopted);
    }
    #[cfg(windows)]
    if let Some(plan) = windows_supervision(parent) {
        return Ok(plan);
    }
    Err(format!(
        "this process is not supervised (parent pid {parent:?}), so signalling the kernel \
         would stop it for good. Start it under a supervisor — on Linux `systemctl restart \
         <unit>`, on Windows a service or a scheduled task — or restart it from `bash`."
    ))
}

/// Whether a Windows parent image is a person's terminal rather than a supervisor.
#[cfg(windows)]
fn windows_supervision(parent: Option<u32>) -> Option<Plan> {
    if std::env::var("SESSIONNAME")
        .map(|name| name.eq_ignore_ascii_case("Services"))
        .unwrap_or(false)
    {
        return Some(Plan::Supervised);
    }
    let name = windows_parent_name(parent?)?;
    if is_interactive_parent(&name) {
        None
    } else {
        Some(Plan::Supervised)
    }
}

/// A shell parent means a person is sitting in front of it.
#[cfg(windows)]
fn is_interactive_parent(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    [
        "bash.exe",
        "sh.exe",
        "cmd.exe",
        "powershell.exe",
        "pwsh.exe",
        "windowsterminal.exe",
        "alacritty.exe",
        "mintty.exe",
        "wezterm-gui.exe",
        "explorer.exe",
        "code.exe",
    ]
    .iter()
    .any(|shell| lower == *shell)
}

/// The image name of a Windows process, asked of the OS rather than guessed.
#[cfg(windows)]
fn windows_parent_name(pid: u32) -> Option<String> {
    let script = format!("(Get-CimInstance Win32_Process -Filter \"ProcessId={pid}\").Name");
    let output = std::process::Command::new("powershell.exe")
        .args(["-NoProfile", "-NonInteractive", "-Command", &script])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let name = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if name.is_empty() {
        None
    } else {
        Some(name)
    }
}

#[cfg(not(windows))]
#[allow(dead_code)]
fn windows_supervision(_parent: Option<u32>) -> Option<Plan> {
    None
}

#[cfg(unix)]
fn parent_pid() -> Option<u32> {
    let stat = std::fs::read_to_string("/proc/self/stat").ok()?;
    let after = stat.rsplit_once(')')?.1;
    after.split_whitespace().nth(1)?.parse().ok()
}

#[cfg(not(unix))]
fn parent_pid() -> Option<u32> {
    // Windows has no `/proc`, and the parent's image name is what actually
    // decides (see `windows_supervision`), so this stays `None` rather than
    // pretending a pid number means something it does not.
    None
}

/// Ask another process to end, in the way that platform supports.
///
/// Split out so the one platform-specific line is together and a reader can see
/// there is nothing else to it: no `/proc`, no job objects, no signal numbers
/// leaking into the rest of the file.
#[cfg(unix)]
fn signal(pid: u32) {
    // SIGTERM, by number, without pulling in `libc` for one constant.
    let _ = std::process::Command::new("kill")
        .arg("-TERM")
        .arg(pid.to_string())
        .status();
}

#[cfg(windows)]
fn signal(pid: u32) {
    // `taskkill` is present on every Windows; `/PID` names the process and no
    // shell is involved, so a pid cannot be reinterpreted as anything else.
    let _ = std::process::Command::new("taskkill")
        .args(["/PID", &pid.to_string()])
        .status();
}

/// Ask the kernel for one of its state keys.
///
/// This is the protocol's other direction, and it is why nothing is passed in
/// the environment: a plugin writes `{"ask":{"state":"kernel.pid"}}` and reads
/// the answer. The kernel is waiting for the reply to the `call` it made, and
/// handles the question before continuing to wait — see the read loop in
/// `protocol::Process::request`.
///
/// `None` when the kernel answered with an error, which the caller turns into a
/// message. Silence is not an option: a plugin blocked on a frame that never
/// comes would hang the tool call.
fn ask_state(question: &serde_json::Value) -> Option<serde_json::Value> {
    let body = serde_json::to_vec(&json!({ "ask": question })).ok()?;
    let mut out = std::io::stdout();
    let mut header = [0u8; 4];
    header.copy_from_slice(&(body.len() as u32).to_be_bytes());
    out.write_all(&header).ok()?;
    out.write_all(&body).ok()?;
    out.flush().ok()?;

    let mut stdin = std::io::stdin();
    let frame = read_frame(&mut stdin)?;
    let answer: serde_json::Value = serde_json::from_slice(&frame).ok()?;
    answer.get("value").cloned()
}

/* ---------------------------------------------------------------- framing */

/// Read one `[u32 big-endian length][JSON]` frame. `None` is end of file.
fn read_frame(input: &mut impl Read) -> Option<Vec<u8>> {
    let mut header = [0u8; 4];
    // `read_exact` at the very start of a frame: a clean EOF here is the kernel
    // closing the pipe, which is how a plugin is told to stop.
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
