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
    Err(format!(
        "this process is not supervised (parent pid {:?}), so exiting would stop it \
         for good. Start it under a supervisor — `systemctl restart <unit>` from \
         `bash` is the safe way to restart from here, or wrap it in one.",
        parent
    ))
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
