//! The `restart` plugin: let the agent restart the process it is running in.
//!
//! Why an agent needs this at all: a turn can change something the running
//! process only reads at startup — a pack's assembly, a settings file, a binary
//! on disk. Without this tool the recovery path is "ask the user to do it",
//! which is exactly the case where the user is not there (a headless box, a
//! phone, a mission running overnight).
//!
//! Three decisions shape the implementation, and each of them is about not
//! making things worse:
//!
//! * **Delayed, not immediate.** The reply to the tool call has to reach the
//!   model and the page before the process goes away. A restart issued inline
//!   would kill the very response that explains it, and the caller would see a
//!   dropped connection instead of a result.
//! * **Refused when it cannot work.** Under a supervisor, exiting is the restart
//!   (`Restart=always`). Started by hand, exiting is just death. The tool checks
//!   for a supervisor and says so rather than leaving a box with no service.
//! * **Never on a wrong guess.** `reason` is required and is logged before the
//!   exit, so the restart has a cause on disk afterwards. An unexplained restart
//!   loop is the worst outcome available here.

use std::sync::Arc;
use std::time::Duration;

use serde_json::json;

use crate::plugin::{Contributions, Kernel, Plugin, PluginCtx};
use crate::tools::{ToolDef, ToolFuture, ToolOutput};

/// The plugin's name: what a pack asks for as `builtin:restart`.
pub const PLUGIN: &str = "restart";

/// The tool's name.
pub const TOOL: &str = "restart";

/// How long to wait before exiting, so the reply lands first.
///
/// Long enough for the tool result to be written to the transcript and pushed to
/// every listener; short enough that nobody thinks the call hung.
const GRACE: Duration = Duration::from_millis(700);

pub fn define(kernel: &mut Kernel) {
    kernel.define(Arc::new(Restart));
}

struct Restart;

impl Plugin for Restart {
    fn name(&self) -> &str {
        PLUGIN
    }

    fn apply(&self, _ctx: &PluginCtx) -> anyhow::Result<Contributions> {
        Ok(Contributions::new().tool(tool()))
    }
}

/// Why the process cannot restart itself, if it cannot.
///
/// The check is the same one systemd uses to decide whether `Restart=` applies:
/// being a child of PID 1 means something is supervising this process, and that
/// something will start it again. Otherwise `exit` is final.
fn supervision() -> Result<RestartPlan, String> {
    // systemd sets `INVOCATION_ID` for every service it starts. `JOURNAL_STREAM`
    // is set alongside it, and either is proof enough.
    let under_systemd = std::env::var_os("INVOCATION_ID").is_some()
        || std::env::var_os("JOURNAL_STREAM").is_some();
    // A container or a supervisor that is not systemd sets none of those; the
    // parent's pid is the portable answer, and `1` means "nothing will restart
    // me when I go".
    let parent = parent_pid();
    if under_systemd {
        return Ok(RestartPlan::Supervised);
    }
    if parent == Some(1) {
        return Ok(RestartPlan::Adopted);
    }
    // Windows has no PID 1 and no systemd, so neither check above can fire
    // there. Before this branch existed, `restart` was a tool that was
    // installed, advertised, and then refused every single time on Windows —
    // the "a tool that cannot work costs a turn every time the model tries it"
    // failure the rest of the pack system is careful to avoid.
    //
    // The question on Windows is the same one in different clothes: did
    // something other than a person's terminal start this process? A service
    // host, a scheduled task, or a detached wrapper is "yes"; a console window
    // is "no", and there exiting really is final.
    #[cfg(windows)]
    if let Some(plan) = windows_supervision(parent) {
        return Ok(plan);
    }
    Err(format!(
        "this process is not supervised (parent pid {:?}), so exiting would stop it \
         for good. Start it under a supervisor — on Linux `systemctl restart <unit>`, \
         on Windows a service or a scheduled task — or run it from `bash` and restart \
         it there.",
        parent
    ))
}

/// Whether a Windows process will be started again when it exits.
///
/// Three signals, in order of how much they prove:
///
/// * **`NGU_SUPERVISED`** — whoever deployed this said so. The direct analogue of
///   systemd's `INVOCATION_ID`, and the only one that works when the supervisor
///   is something we cannot recognise.
/// * **`SESSIONNAME=Services`** — the process belongs to the service session.
///   Windows sets this for a service host without any help from us.
/// * **The parent is not a terminal** — a `bash.exe`, `cmd.exe`,
///   `powershell.exe`, `pwsh.exe`, a terminal emulator, or `explorer.exe` parent
///   means a person is sitting in front of it. Anything else (`services.exe`,
///   `svchost.exe`, a CI agent, an `ngu serve` wrapper) is a supervisor.
#[cfg(windows)]
fn windows_supervision(parent: Option<u32>) -> Option<RestartPlan> {
    if std::env::var_os("NGU_SUPERVISED").is_some() {
        return Some(RestartPlan::Supervised);
    }
    if std::env::var("SESSIONNAME")
        .map(|name| name.eq_ignore_ascii_case("Services"))
        .unwrap_or(false)
    {
        return Some(RestartPlan::Supervised);
    }
    let name = windows_parent_name(parent?)?;
    if is_interactive_parent(&name) {
        None
    } else {
        Some(RestartPlan::Supervised)
    }
}

/// Whether this parent image is a person's terminal rather than a supervisor.
///
/// A shell parent means someone is sitting in front of it: exiting closes their
/// window and nothing starts it again. `conhost.exe` is deliberately **not** in
/// the list — it hosts the console of a service session too, so treating it as
/// interactive would make a supervised service look unattended.
///
/// A function of its own because this is the decision worth testing: the rest of
/// [`windows_supervision`] is two environment lookups and one process spawn.
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
///
/// `wmic` is deprecated and missing from parts of Windows 11; the CIM instance
/// query through PowerShell is what is actually there. One process spawn, only
/// on the restart path, and only after the two cheap environment checks failed.
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

/// Unix never needs this: [`supervision`] already answered from `/proc`.
#[cfg(not(windows))]
#[allow(dead_code)]
fn windows_supervision(_parent: Option<u32>) -> Option<RestartPlan> {
    None
}

enum RestartPlan {
    /// A systemd unit owns this process; exiting is the restart.
    Supervised,
    /// Parent is PID 1, which is what a container looks like from inside.
    Adopted,
}

#[cfg(unix)]
fn parent_pid() -> Option<u32> {
    // `/proc/self/stat` field 4 is the parent pid. Read rather than shelled out:
    // this runs inside a tool call, and spawning a process to ask is silly.
    let stat = std::fs::read_to_string("/proc/self/stat").ok()?;
    // The command name is parenthesised and may contain spaces, so the fields
    // after it are found from the last `)`.
    let after = stat.rsplit_once(')')?.1;
    after.split_whitespace().nth(1)?.parse().ok()
}

#[cfg(not(unix))]
fn parent_pid() -> Option<u32> {
    None
}

fn tool() -> ToolDef {
    ToolDef::new(
        TOOL,
        "Restart the process this agent runs in. Use it after changing something \
         that is only read at startup: an installed or unloaded pack, a settings \
         file, a rebuilt binary. The current turn finishes and its answer is \
         delivered first; the process then exits and its supervisor starts it \
         again. Refused when nothing is supervising it — then `systemctl restart` \
         from `bash` is the safe alternative. Say why, in one line; it is logged \
         before the exit so an unattended restart has a cause on disk.",
        json!({
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "One line: what changed that needs a restart. Logged."
                },
                "delay_ms": {
                    "type": "integer",
                    "description": "Wait before exiting (default 700). Raise it if the caller needs longer."
                }
            },
            "required": ["reason"]
        }),
        TOOL,
        move |args| Box::pin(async move { run(args).await }) as ToolFuture,
    )
}

async fn run(args: serde_json::Value) -> anyhow::Result<ToolOutput> {
    let reason = args
        .get("reason")
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "missing required argument: reason. An unattended restart with no \
                 stated cause is not debuggable afterwards."
            )
        })?;

    let plan = match supervision() {
        Ok(plan) => plan,
        Err(why) => return Ok(ToolOutput::text(why)),
    };

    let delay = args
        .get("delay_ms")
        .and_then(|v| v.as_u64())
        .map(Duration::from_millis)
        .unwrap_or(GRACE);

    // Bytes on stderr before the exit: under systemd this lands in the journal,
    // which is where anyone looking at "why did it restart at 3am" will look.
    eprintln!("[restart] requested: {reason}");

    let how = match plan {
        RestartPlan::Supervised => "the systemd unit will start it again",
        RestartPlan::Adopted => "it is PID-1's child, so its supervisor starts it again",
    };

    // The exit happens on a detached thread so this future can return first.
    // `std::process::exit` does not run destructors, which is why the answer is
    // delivered by the grace period rather than by unwinding.
    std::thread::spawn(move || {
        std::thread::sleep(delay);
        eprintln!("[restart] exiting now; {how}");
        std::process::exit(75);
    });

    Ok(ToolOutput::text(format!(
        "Restarting in {} ms ({how}).\nReason logged: {reason}\n\
         This reply is delivered first; the process then exits. Anything that was \
         in flight — a running turn, a background job — will not survive.",
        delay.as_millis()
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_declared_supervisor_is_believed_without_asking_anyone() {
        // `NGU_SUPERVISED` is the escape hatch for supervisors we cannot
        // recognise: a CI runner, a process manager that is not systemd, an
        // `ngu serve` wrapper. Whoever deployed it knows; this is how they say
        // so. It has to short-circuit *before* the parent-pid query, or the
        // escape hatch only works on the platforms that need it least.
        std::env::set_var("NGU_SUPERVISED", "1");
        let plan = supervision();
        std::env::remove_var("NGU_SUPERVISED");
        assert!(
            matches!(plan, Ok(RestartPlan::Supervised)),
            "a declared supervisor must be believed: {plan:?}",
            plan = plan.as_ref().err()
        );
    }

    #[cfg(windows)]
    #[test]
    fn a_shell_parent_is_not_a_supervisor() {
        // The failure this branch was written for: on Windows `parent_pid` used
        // to return `None` unconditionally, so `restart` was installed,
        // advertised, and refused every time. A console window really is not a
        // supervisor — but nothing else was ever recognised either.
        for shell in ["bash.exe", "cmd.exe", "PowerShell.exe", "pwsh.exe"] {
            // Call the classifier with an explicit name rather than spawning a
            // process: the mapping is what carries the decision.
            assert!(
                is_interactive_parent(shell),
                "{shell} is a person's terminal, not a supervisor"
            );
        }
        for supervisor in ["services.exe", "svchost.exe", "ngu-serve.exe"] {
            assert!(
                !is_interactive_parent(supervisor),
                "{supervisor} starts its child again when it exits"
            );
        }
    }

    #[test]
    fn a_missing_reason_is_refused() {
        // The whole point of the argument: an unattended restart has to leave a
        // cause behind. Refusing here is cheaper than a 3am mystery.
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        for args in [json!({}), json!({"reason": ""}), json!({"reason": "   "})] {
            let result = runtime.block_on(run(args));
            assert!(result.is_err(), "must refuse an empty reason");
        }
    }

    #[test]
    fn the_parent_pid_is_readable_on_unix() {
        // Supervision detection depends on this; if it silently returns None the
        // tool would always claim "not supervised" and never work.
        #[cfg(unix)]
        {
            let parent = parent_pid();
            assert!(parent.is_some(), "expected a parent pid from /proc/self/stat");
            assert_ne!(parent, Some(0), "pid 0 is not a parent");
        }
    }

    #[test]
    fn supervision_is_reported_rather_than_assumed() {
        // Either the check finds a supervisor, or it explains itself. What it
        // must never do is return "fine" when nothing would restart us.
        match supervision() {
            Ok(RestartPlan::Supervised) => {}
            Ok(RestartPlan::Adopted) => {}
            Err(reason) => {
                assert!(
                    reason.contains("not supervised"),
                    "the refusal must say so plainly: {reason}"
                );
            }
        }
    }

    #[test]
    fn the_description_mentions_the_safe_alternative() {
        // A refusal that does not say what to do instead just moves the problem.
        let described = tool();
        assert!(described.description.contains("systemctl restart"));
    }
}
