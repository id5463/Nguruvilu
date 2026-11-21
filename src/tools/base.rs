//! The four base tools: `read`, `write`, `edit`, `bash`.
//!
//! These are the kernel's whole model-facing surface. They follow the
//! Agent-Computer Interface principle: the interface itself decides how well
//! the model performs, so each tool bounds its own output rather than dumping
//! unbounded text into the context.
//!
//! * `read` returns a bounded window with line numbers and tells the model how
//!   to continue — far cheaper than `cat` for a large file.
//! * `edit` replaces an exact string and refuses ambiguous matches, which is
//!   more reliable than line-number patching.
//! * `bash` is the escape hatch and bounds its own output.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use serde_json::{json, Value};
use tokio::io::AsyncReadExt;
use tokio::process::Command;

use super::{ConflictPolicy, ToolDef, ToolFuture, ToolRegistry};

/// Lines returned by `read` when the caller does not specify a range.
const DEFAULT_READ_WINDOW: usize = 200;

/// Hard ceiling on lines one `read` call may return.
const MAX_READ_WINDOW: usize = 2000;

/// Bytes one `read` call may return, whatever the line window says.
///
/// The window counts lines, and a line has no ceiling of its own: a minified
/// bundle, a base64 blob, or a log with everything on one line would each let
/// "2000 lines" carry megabytes into the next prompt. This bound is what keeps
/// the answer to a size the model can actually work with, and the reply says
/// which line to continue from when it cuts.
const MAX_READ_BYTES: usize = 192 * 1024;

/// Characters one line may contribute before it is cut.
///
/// Beyond this a single line is not something a model reads anyway; keeping the
/// head is enough to recognise what the line is, and the reply says it was cut
/// so nobody mistakes it for the whole line.
const MAX_READ_LINE_CHARS: usize = 4_000;

/// Output ceiling for `bash` before head/tail truncation kicks in.
const MAX_BASH_OUTPUT: usize = 30_000;

/// Bytes one `bash` call may accumulate per pipe while the command runs.
///
/// [`MAX_BASH_OUTPUT`] bounds what the *model* sees, but truncation used to
/// happen only after everything had been read into memory: a command printing
/// gigabytes became gigabytes of `Vec<u8>` first. The cap is generous enough
/// that the head/tail view still shows real content on both ends, and the
/// reader keeps draining past it (a child whose pipe is never read blocks
/// forever) while recording that the output was cut.
const MAX_BASH_BUFFER: usize = 8 * 1024 * 1024;

/// Default `bash` timeout, out of the box. The settings panel's tool deadline
/// replaces it at boot: this is only the value the global starts with.
pub(crate) const DEFAULT_BASH_TIMEOUT_MS: u64 = 120_000;

/// Largest image the kernel will attach without warning.
///
/// Providers cap what they accept, and a silently rejected request is worse
/// than a warning the model can pass on to the user.
const MAX_IMAGE_BYTES: usize = 4 * 1024 * 1024;

/// Read width and height from an image header.
///
/// Deliberately minimal: only the formats the kernel recognises, and only the
/// header. Dimensions are a courtesy to the model, not a reason to decode a
/// multi-megabyte image.
fn image_dimensions(bytes: &[u8], mime: &str) -> Option<(u32, u32)> {
    match mime {
        // PNG: IHDR always begins at byte 16.
        "image/png" if bytes.len() >= 24 => {
            let w = u32::from_be_bytes(bytes[16..20].try_into().ok()?);
            let h = u32::from_be_bytes(bytes[20..24].try_into().ok()?);
            Some((w, h))
        }
        // GIF: little-endian, immediately after the signature.
        "image/gif" if bytes.len() >= 10 => {
            let w = u16::from_le_bytes(bytes[6..8].try_into().ok()?) as u32;
            let h = u16::from_le_bytes(bytes[8..10].try_into().ok()?) as u32;
            Some((w, h))
        }
        // BMP: little-endian, at a fixed offset. Height may be negative for a
        // top-down bitmap, so take the magnitude.
        "image/bmp" if bytes.len() >= 26 => {
            let w = i32::from_le_bytes(bytes[18..22].try_into().ok()?);
            let h = i32::from_le_bytes(bytes[22..26].try_into().ok()?);
            Some((w.unsigned_abs(), h.unsigned_abs()))
        }
        // JPEG and WebP need real parsing; not worth it for a courtesy.
        _ => None,
    }
}
/// Register all four base tools under the `kernel` owner.
pub fn register_all(registry: &mut ToolRegistry) -> Result<()> {
    registry.register(
        ToolDef::new(
            "read",
            "Read a file, or a line range of one. Returns line-numbered content. \
             Large files are returned in windows; use start_line/end_line to continue.",
            json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "File path: absolute (C:\\\\dir\\\\file or /c/dir/file), or relative to the working directory" },
                    "start_line": { "type": "integer", "description": "First line to return, 1-based. Defaults to 1." },
                    "end_line": { "type": "integer", "description": "Last line to return, inclusive. Defaults to start_line + 199." }
                },
                "required": ["path"]
            }),
            "kernel",
            |args| Box::pin(read_any(args)) as ToolFuture,
        ),
        ConflictPolicy::Error,
    )?;

    registry.register(
        ToolDef::new(
            "write",
            "Write a file, replacing its contents entirely. Creates parent directories as needed.",
            json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "File path: absolute (C:\\\\dir\\\\file or /c/dir/file), or relative to the working directory" },
                    "content": { "type": "string", "description": "Full file contents" }
                },
                "required": ["path", "content"]
            }),
            "kernel",
            |args| Box::pin(async move { write_tool(args).await.map(Into::into) }) as ToolFuture,
        ),
        ConflictPolicy::Error,
    )?;

    registry.register(
        ToolDef::new(
            "edit",
            "Replace an exact string in a file. The old string must appear exactly once, \
             otherwise the edit is refused and nothing is written.",
            json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "File path: absolute (C:\\\\dir\\\\file or /c/dir/file), or relative to the working directory" },
                    "old_str": { "type": "string", "description": "Exact text to replace; must occur exactly once" },
                    "new_str": { "type": "string", "description": "Replacement text" }
                },
                "required": ["path", "old_str", "new_str"]
            }),
            "kernel",
            |args| Box::pin(async move { edit_tool(args).await.map(Into::into) }) as ToolFuture,
        ),
        ConflictPolicy::Error,
    )?;

    let shell = detect_shell();
    registry.register(
        ToolDef::new(
            "bash",
            format!(
                "Run a shell command and return its combined output.\n\
                 The command runs in {} — {} \n\
                 Set timeout_ms to bound long-running commands — the settings' tool deadline is a ceiling, so a longer request is cut and the answer says so. Output is truncated when very large.\n\
                 Pass background: true to start the command as a job and return immediately; \
                 collect it later with job_output / job_list / job_kill (timeout_ms does not apply). \
                 With notify: true, the shell messages you when the job finishes — even after this turn ends.",
                shell.label, shell.syntax_hint
            ),
            json!({
                "type": "object",
                "properties": {
                    "command": { "type": "string", "description": "Command to run" },
                    "timeout_ms": { "type": "integer", "description": "Timeout in milliseconds, capped by the tool deadline in the settings (120000 out of the box): ask for more and the call is bounded to the ceiling and the answer names it. Use background: true for work that must outlive the step" },
                    "cwd": { "type": "string", "description": "Working directory for the command" },
                    "background": { "type": "boolean", "description": "Run as a background job and return a job id immediately (default false)" },
                    "notify": { "type": "boolean", "description": "With background: wake you with a message when the job finishes, even if this turn has ended (default false)" }
                },
                "required": ["command"]
            }),
            "kernel",
            |args| Box::pin(async move { bash_tool(args).await.map(Into::into) }) as ToolFuture,
        ),
        ConflictPolicy::Error,
    )?;

    // The job tools extend `bash` (`background: true` hands a command to the
    // registry), so they belong to the base set every host loads.
    super::jobs::install(registry)?;

    Ok(())
}

/// Resolve a caller-supplied path against the session working directory.
///
/// Relative paths resolve under the current session's workspace when one is
/// bound, else under the process directory (CLI use). Absolute paths are
/// returned unchanged.
///
/// On Windows this also accepts the POSIX form a shell reports. Git Bash mounts
/// drives as `/c/...`, `/i/...`, and `pwd` prints exactly that — so a model that
/// reads the working directory from `bash` and then hands the same string to
/// `read` or `write` is holding a path these tools would otherwise resolve
/// relative to the working directory, quietly landing the file somewhere else.
/// That happened in practice: a model wrote to `I:/i/tmp-test/x` intending
/// `I:\tmp-test\x`, could not find the file afterwards, and concluded the write
/// tool was lying.
///
/// The translation applies only on Windows, where `/i/...` cannot be a real
/// absolute path. On Unix a leading `/i` is a genuine directory and is left
/// alone.
fn resolve_path(raw: &str) -> PathBuf {
    resolve_path_in(raw, default_base_dir())
}

/// The directory relative paths resolve against for the current tool call.
fn default_base_dir() -> PathBuf {
    super::current_workspace()
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")))
}

/// Resolve `raw` against `base`, applying the POSIX drive translation first.
fn resolve_path_in(raw: &str, base: PathBuf) -> PathBuf {
    let path = Path::new(raw);
    if path.is_absolute() {
        return path.to_path_buf();
    }

    #[cfg(windows)]
    if let Some(drive) = posix_drive_path(raw) {
        return drive;
    }

    base.join(path)
}


/// Translate `/c/Users/x` into `C:\Users\x`, when that is what it means.
///
/// Only a single leading letter followed by a separator is treated as a drive
/// mount, which is exactly the Git Bash convention.
#[cfg(windows)]
fn posix_drive_path(raw: &str) -> Option<PathBuf> {
    let rest = raw.strip_prefix('/')?;
    let mut chars = rest.chars();
    let letter = chars.next()?;
    if !letter.is_ascii_alphabetic() {
        return None;
    }
    let after = chars.as_str();
    // `/c` alone, or `/c/...`; `/code/...` is a real relative directory name.
    let tail = match after.strip_prefix('/') {
        Some(tail) => tail,
        None if after.is_empty() => "",
        None => return None,
    };
    Some(PathBuf::from(format!(
        "{}:\\{}",
        letter.to_ascii_uppercase(),
        tail.replace('/', "\\")
    )))
}

/// Read a file, attaching it as an image when it is one.
///
/// Text is read as lines. An image cannot be read as lines at all — decoding it
/// as UTF-8 fails outright — so it is attached instead and the model is told
/// what was attached. Anything else binary is reported as such rather than
/// handed to the model as mojibake.
async fn read_any(args: Value) -> Result<crate::tools::ToolOutput> {
    let raw_path = args
        .get("path")
        .and_then(|p| p.as_str())
        .ok_or_else(|| anyhow!("missing required argument: path"))?;
    let path = resolve_path(raw_path);

    let bytes = tokio::fs::read(&path)
        .await
        .with_context(|| format!("reading {}", path.display()))?;

    // An image is attached rather than decoded. The check is on the bytes, not
    // the name: a file called .png that is not one should be reported, not sent
    // to a provider as a broken image.
    if let Some(mime) = crate::message::image_mime(&bytes, &path.display().to_string()) {
        let (width, height) = image_dimensions(&bytes, mime).unwrap_or((0, 0));
        let size = bytes.len();
        let dimensions = if width > 0 {
            format!(", {width}x{height}")
        } else {
            String::new()
        };

        let mut text = format!(
            "{} is an image ({mime}{dimensions}, {} bytes). It is attached below; \
             describe what you see rather than trying to read it as text.",
            path.display(),
            size
        );

        // A huge image is worth flagging: providers cap what they will accept.
        if size > MAX_IMAGE_BYTES {
            text.push_str(&format!(
                "\nWarning: this image is {} MB, which many providers reject.",
                size / (1024 * 1024)
            ));
        }

        let image = crate::message::ImageAttachment::url(crate::message::data_uri(mime, &bytes))
            .labelled(format!("{} ({mime})", path.display()));
        return Ok(crate::tools::ToolOutput::text(text).with_image(image));
    }

    // The pack-provided readers teach `read` its formats (读不了就装包 —
    // see `readers.rs`). Unclaimed falls through to plain text; Unreadable
    // carries the reasons, which is where a machine missing its toolchain
    // gets the install-command card back. The whole ask runs on a blocking
    // thread: readers spawn processes and poll with sleeps, and parking an
    // executor worker for thirty seconds is not a trade the loop agreed to.
    let content = {
        let probe = bytes.clone();
        let probe_path = path.clone();
        let answer = match tokio::task::spawn_blocking(move || {
            crate::tools::readers::convert(&probe, &probe_path)
        })
        .await
        {
            Ok(answer) => answer,
            Err(join) => {
                // A panicking reader is contained the way the event bus
                // contains a panicking observer: reported, then out of the way.
                eprintln!("[read] a reader panicked: {join}");
                crate::tools::readers::ReadAnswer::Unclaimed
            }
        };
        match answer {
            crate::tools::readers::ReadAnswer::Converted(text) => text,
            crate::tools::readers::ReadAnswer::Unclaimed => {
                String::from_utf8(bytes).map_err(|_| {
                    let installed = crate::tools::readers::installed();
                    let list = if installed.is_empty() {
                        "no readers are installed — a pack can provide one for this format"
                            .to_string()
                    } else {
                        format!(
                            "installed readers: {}",
                            installed
                                .iter()
                                .map(|(id, owner)| format!("{id} (from {owner})"))
                                .collect::<Vec<_>>()
                                .join(", ")
                        )
                    };
                    anyhow!(
                        "{} is binary and not a recognised image format; no reader claims it, \
                         and {list}",
                        path.display()
                    )
                })?
            }
            crate::tools::readers::ReadAnswer::Unreadable(reasons) => {
                return Err(anyhow!(
                    "{} is claimed by a reader but no text came out — {}",
                    path.display(),
                    reasons.join(" | ")
                ));
            }
        }
    };

    let lines: Vec<&str> = content.lines().collect();
    let total = lines.len();

    let start = args.get("start_line").and_then(|v| v.as_u64()).unwrap_or(1).max(1) as usize;
    let end = match args.get("end_line").and_then(|v| v.as_u64()) {
        Some(v) => (v as usize).max(start),
        None => start + DEFAULT_READ_WINDOW - 1,
    };
    let end = end.min(start + MAX_READ_WINDOW - 1).min(total.max(1));

    if total == 0 {
        return Ok(format!("{} (empty file)", path.display()).into());
    }
    if start > total {
        return Ok(format!(
            "{} has {total} lines; start_line {start} is past the end",
            path.display()
        )
        .into());
    }

    let width = end.to_string().len();
    let mut out = String::with_capacity((end - start + 1) * 80 + 128);
    out.push_str(&format!("{} ({} lines total)\n", path.display(), total));
    // The window counts lines; bytes are what the next prompt pays. Both are
    // checked as the answer is built: a line is cut when it is enormous on its
    // own, and the whole reply stops when it has reached [`MAX_READ_BYTES`].
    let mut shown_upto = start - 1;
    let mut cut_lines = 0usize;
    for (i, line) in lines[start - 1..end].iter().enumerate() {
        let number = start + i;
        let rendered;
        if line.chars().count() > MAX_READ_LINE_CHARS {
            cut_lines += 1;
            rendered = line
                .chars()
                .take(MAX_READ_LINE_CHARS)
                .collect::<String>()
                + " …(line continues)";
        } else {
            rendered = (*line).to_string();
        }
        let piece = format!("{:>width$}| {}\n", number, rendered, width = width);
        if out.len() + piece.len() > MAX_READ_BYTES {
            break;
        }
        out.push_str(&piece);
        shown_upto = number;
    }
    if cut_lines > 0 {
        out.push_str(&format!(
            "… {cut_lines} line(s) cut at {MAX_READ_LINE_CHARS} characters; a line that long \
             is not readable through this tool.\n"
        ));
    }
    if shown_upto < end {
        // The reply stopped early, so the honest hint is the first line that
        // was *not* shown — not `end`, which may never have been reached.
        out.push_str(&format!(
            "… stopped after {} line(s) to keep this reply bounded. \
             Continue with start_line={}.\n",
            shown_upto + 1 - start,
            shown_upto + 1
        ));
    } else if end < total {
        out.push_str(&format!(
            "… {} more lines. Continue with start_line={}.\n",
            total - end,
            end + 1
        ));
    }
    Ok(out.into())
}

async fn write_tool(args: Value) -> Result<String> {
    let raw_path = args
        .get("path")
        .and_then(|p| p.as_str())
        .ok_or_else(|| anyhow!("missing required argument: path"))?;
    let content = args
        .get("content")
        .and_then(|c| c.as_str())
        .ok_or_else(|| anyhow!("missing required argument: content"))?;
    let path = resolve_path(raw_path);

    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            tokio::fs::create_dir_all(parent)
                .await
                .with_context(|| format!("creating {}", parent.display()))?;
        }
    }

    tokio::fs::write(&path, content)
        .await
        .with_context(|| format!("writing {}", path.display()))?;

    Ok(format!("wrote {} bytes to {}", content.len(), path.display()))
}

async fn edit_tool(args: Value) -> Result<String> {
    let raw_path = args
        .get("path")
        .and_then(|p| p.as_str())
        .ok_or_else(|| anyhow!("missing required argument: path"))?;
    let old_str = args
        .get("old_str")
        .and_then(|c| c.as_str())
        .ok_or_else(|| anyhow!("missing required argument: old_str"))?;
    let new_str = args.get("new_str").and_then(|c| c.as_str()).unwrap_or("");
    let path = resolve_path(raw_path);

    let content = tokio::fs::read_to_string(&path)
        .await
        .with_context(|| format!("reading {}", path.display()))?;

    let occurrences = content.matches(old_str).count();
    match occurrences {
        0 => return Err(anyhow!("old_str not found in {}", path.display())),
        1 => {}
        n => {
            return Err(anyhow!(
                "old_str occurs {n} times in {}; include more surrounding context to make it unique",
                path.display()
            ))
        }
    }

    let updated = content.replacen(old_str, new_str, 1);
    tokio::fs::write(&path, &updated)
        .await
        .with_context(|| format!("writing {}", path.display()))?;

    Ok(format!("replaced 1 occurrence in {}", path.display()))
}

/// One command's stdout and stderr, formatted as the model reads them.
///
/// Shared by both exits of `bash_tool`: the normal one and the timed-out one,
/// which must not throw away what the command printed before the deadline.
fn render_bash_output(stdout: &[u8], stderr: &[u8], empty_placeholder: bool) -> String {
    let mut out = String::new();
    let stdout_text = String::from_utf8_lossy(stdout);
    let stderr_text = String::from_utf8_lossy(stderr);

    if !stdout_text.trim().is_empty() {
        out.push_str(&truncate_output(&stdout_text, MAX_BASH_OUTPUT));
    }
    if !stderr_text.trim().is_empty() {
        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str("--- stderr ---\n");
        out.push_str(&truncate_output(&stderr_text, MAX_BASH_OUTPUT));
    }
    if out.trim().is_empty() && empty_placeholder {
        out.push_str("(no output)");
    }
    out
}

/// Drain one output pipe into a shared buffer until EOF.
///
/// Chunks land in the buffer as they arrive, so a wait that runs out of time
/// can still return what the command printed instead of losing it with the
/// reader. Generic because `ChildStdout` and `ChildStderr` are distinct types.
///
/// The buffer stops growing at [`MAX_BASH_BUFFER`] and `capped` is set; the
/// pipe keeps being drained, because a child blocked on a full pipe never
/// exits and the deadline would then hide the hang instead of the memory.
fn pump<R>(
    pipe: Option<R>,
    buf: Arc<Mutex<Vec<u8>>>,
    capped: Arc<std::sync::atomic::AtomicBool>,
) -> tokio::task::JoinHandle<()>
where
    R: tokio::io::AsyncRead + Send + Unpin + 'static,
{
    use std::sync::atomic::Ordering;

    tokio::spawn(async move {
        let Some(mut pipe) = pipe else { return };
        let mut chunk = [0u8; 8192];
        loop {
            match pipe.read(&mut chunk).await {
                Ok(0) | Err(_) => break,
                Ok(read) => {
                    // Recovering the data from a poisoned mutex beats turning
                    // one panic into a permanently panicking `bash`.
                    let mut guard = buf.lock().unwrap_or_else(|error| error.into_inner());
                    let room = MAX_BASH_BUFFER.saturating_sub(guard.len());
                    if read > room {
                        guard.extend_from_slice(&chunk[..room]);
                        drop(guard);
                        capped.store(true, Ordering::Relaxed);
                    } else {
                        guard.extend_from_slice(&chunk[..read]);
                    }
                }
            }
        }
    })
}

/// Wait for one pump task to finish, but no later than `deadline`.
///
/// Returns `true` when the task is done, and `false` when the deadline came
/// first. The distinction matters: a `JoinHandle` that already returned `Ready`
/// panics when polled again ("JoinHandle polled after completion"), so a handle
/// whose wait timed out is the only one that may be waited on a second time.
async fn pump_finished(
    task: &mut tokio::task::JoinHandle<()>,
    deadline: tokio::time::Instant,
) -> bool {
    tokio::time::timeout_at(deadline, &mut *task).await.is_ok()
}

async fn bash_tool(args: Value) -> Result<String> {
    let command = args
        .get("command")
        .and_then(|c| c.as_str())
        .ok_or_else(|| anyhow!("missing required argument: command"))?;
    // The settings deadline is a ceiling, not only a default. A measured session
    // showed `timeout_ms` requests of 300000, 600000 and 1500000 ms all honoured,
    // because the argument was accepted whenever it was merely positive — twenty
    // five minutes of one call holding a turn. The cut is named to the model
    // below: silently running short is how a caller keeps waiting for output that
    // was never going to arrive.
    let (timeout_ms, cut_from) = super::effective_timeout(
        args.get("timeout_ms").and_then(|v| v.as_u64()),
        super::default_tool_timeout_ms(),
    );
    let timeout = Duration::from_millis(timeout_ms);
    let cut_note = cut_from.map(|asked| {
        format!(
            "--- timeout_ms {asked} is above this machine's foreground budget \
             ({timeout_ms} ms, set by `tool_timeout_secs` — editable in the settings \
             panel or with `ngu config set --tool-timeout <sec>`, and it applies to the \
             next call). A command that must outlive the step belongs in bash with \
             background: true — a job's lifetime is job_kill's, not this budget's ---"
        )
    });
    let cwd = match args.get("cwd").and_then(|c| c.as_str()) {
        Some(explicit) => Some(resolve_path(explicit)),
        None => super::current_workspace(),
    };

    let shell = detect_shell();
    let (program, program_args) = shell.invocation(command);

    let mut cmd = Command::new(&program);
    cmd.args(&program_args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if let Some(dir) = &cwd {
        cmd.current_dir(dir);
    }

    // A command that should outlive this step: hand it to the job registry
    // instead of pinning the turn until it finishes. timeout_ms is a
    // foreground concept — a job's lifetime belongs to job_kill.
    if args.get("background").and_then(|v| v.as_bool()).unwrap_or(false) {
        let notify = args.get("notify").and_then(|v| v.as_bool()).unwrap_or(false);
        let job = super::jobs::spawn_background(command, cmd, notify)?;
        return Ok(format!(
            "started {} in the background: {command}\n\
             Read its output with job_output (wait_ms blocks until it finishes), \
             list jobs with job_list, stop it with job_kill.{}",
            job.id,
            if notify {
                "\nWhen it finishes, you will receive a message about it — you may end this turn now."
            } else {
                ""
            }
        ));
    }

    // Same process-group rule as jobs: a timeout must stop what the command
    // forked, not just the shell. What that means for a command that started
    // something on purpose: work that *redirects its own stdio*
    // (`nohup my-server > /dev/null 2>&1 &`) never holds this call's pipes, so
    // it is never signalled and outlives the call as intended. Work that
    // inherits the pipes and keeps them after the shell is gone is
    // indistinguishable from the leak that stalled a turn for 105 minutes, so
    // it is stopped with a note saying so. Anything that should genuinely
    // outlive the step belongs in `bash { background: true }`, which is a job
    // the caller owns and `job_kill` can stop.
    #[cfg(unix)]
    cmd.process_group(0);

    let mut child = cmd
        .spawn()
        .with_context(|| format!("spawning {program} (shell: {})", shell.label))?;

    // The whole tree as one unit, adopted before anything is read: descendants
    // inherit the job as they are created (see `jobs::ProcessTree`). This is
    // what makes the two exits below able to *close the write ends* of the
    // output pipes, which is the only thing that lets a pump's in-flight read
    // finish. On Unix this is `None` — the process group from the spawn is the
    // same mechanism, and the signal path already stops it.
    let tree = super::jobs::ProcessTree::adopt(child.id());

    // One deadline for the whole call: waiting for the shell, then waiting
    // for its pipes to close. The old code bounded only the wait on the
    // shell, so a background process that inherited the pipes (`nohup ... &`)
    // kept the readers alive forever after the shell itself had exited --
    // one such call stalled a turn for 105 minutes with no timeout anywhere.
    let deadline = tokio::time::Instant::now() + timeout;

    let stdout_buf: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(Vec::new()));
    let stderr_buf: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(Vec::new()));
    let stdout_capped = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let stderr_capped = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let mut stdout_task = pump(
        child.stdout.take(),
        Arc::clone(&stdout_buf),
        Arc::clone(&stdout_capped),
    );
    let mut stderr_task = pump(
        child.stderr.take(),
        Arc::clone(&stderr_buf),
        Arc::clone(&stderr_capped),
    );

    let status = match tokio::time::timeout_at(deadline, child.wait()).await {
        Ok(result) => result.context("waiting for command")?,
        Err(_) => {
            // Stop the tree as one unit when the platform can name it as one:
            // one call, every descendant gone at the same instant. Otherwise
            // fall back to the signal sweep, which is what Unix has always used.
            let stopped_as_tree = tree.as_ref().map(|tree| tree.terminate()).unwrap_or(false);
            if stopped_as_tree {
                let _ = child.start_kill();
            } else {
                super::jobs::kill_tree(&mut child);
            }
            // kill_tree only sends the signal. A descendant (a `ping` under
            // `cmd`) outlives the shell on Windows, so this reap must have its
            // own bound too: an unbounded wait here is the 60-second stall.
            let _ = tokio::time::timeout(Duration::from_millis(500), child.wait()).await;
            // Now that the writers are gone, the readers owe an EOF. Give them
            // the moment it takes to arrive, *then* cut them loose: aborting a
            // task whose overlapped read the OS already accepted parks the
            // handle until that read completes on its own — the tail this
            // whole arrangement exists to remove.
            let _ = tokio::time::timeout(Duration::from_millis(500), async {
                let _ = (&mut stdout_task).await;
                let _ = (&mut stderr_task).await;
            })
            .await;
            stdout_task.abort();
            stderr_task.abort();
            let stdout = std::mem::take(&mut *stdout_buf.lock().unwrap_or_else(|e| e.into_inner()));
            let stderr = std::mem::take(&mut *stderr_buf.lock().unwrap_or_else(|e| e.into_inner()));
            let mut message = format!(
                "command timed out after {} ms and was killed: {command}\n({})",
                timeout.as_millis(),
                super::budget_hint("this budget", "tool-timeout")
            );
            // Whatever was printed before the deadline still answers part of
            // the question -- dropping it would make a timed-out call
            // strictly less informative than one that finished.
            if !stdout.is_empty() || !stderr.is_empty() {
                message.push('\n');
                message.push_str(&render_bash_output(&stdout, &stderr, false));
            }
            note_capped_output(
                &mut message,
                stdout_capped.load(std::sync::atomic::Ordering::Relaxed)
                    || stderr_capped.load(std::sync::atomic::Ordering::Relaxed),
            );
            if let Some(note) = &cut_note {
                message.push('\n');
                message.push_str(note);
            }
            // `tree` is dropped here with kill-on-close still set: a last,
            // free guarantee that nothing of this command outlives its timeout.
            return Ok(message);
        }
    };

    // The shell exited, but its pipes only owe an EOF once every writer is
    // gone -- and a background descendant may hold them open indefinitely.
    // This wait runs to the same deadline: close enough is good enough, and
    // what was already read is returned with a note instead of the turn
    // hanging on a writer that is never going away.
    //
    // Each handle is awaited at most once and its completion is remembered: a
    // `JoinHandle` that already returned `Ready` panics if it is polled again
    // ("JoinHandle polled after completion"), and the second bounded wait below
    // exists precisely for the handle that did *not* finish the first time.
    let mut stdout_finished = pump_finished(&mut stdout_task, deadline).await;
    let mut stderr_finished = pump_finished(&mut stderr_task, deadline).await;
    let mut pipes_closed = stdout_finished && stderr_finished;
    // Still open: something the shell forked is holding the pipes. On Windows
    // the job can close those write ends as a group — the difference between a
    // pump that ends here and one whose read stays pending for as long as the
    // descendant feels like living.
    let mut stopped_holder = false;
    if !pipes_closed && tree.as_ref().map(|tree| tree.terminate()).unwrap_or(false) {
        stopped_holder = true;
        let grace = tokio::time::Instant::now() + Duration::from_millis(500);
        if !stdout_finished {
            stdout_finished = pump_finished(&mut stdout_task, grace).await;
        }
        if !stderr_finished {
            stderr_finished = pump_finished(&mut stderr_task, grace).await;
        }
        pipes_closed = stdout_finished && stderr_finished;
    }
    stdout_task.abort();
    stderr_task.abort();
    let stdout = std::mem::take(&mut *stdout_buf.lock().unwrap_or_else(|e| e.into_inner()));
    let stderr = std::mem::take(&mut *stderr_buf.lock().unwrap_or_else(|e| e.into_inner()));

    let mut out = render_bash_output(&stdout, &stderr, true);
    if !pipes_closed {
        out.push('\n');
        out.push_str(&format!(
            "--- the command exited, but a background process kept its output \
             pipes open past {} ms; output ends here ---",
            timeout.as_millis()
        ));
    } else if stopped_holder {
        out.push('\n');
        out.push_str(&format!(
            "--- the command exited, but a background process was still holding \
             its output pipes past {} ms; that process was stopped so the output \
             above could end ---",
            timeout.as_millis()
        ));
    }
    note_capped_output(
        &mut out,
        stdout_capped.load(std::sync::atomic::Ordering::Relaxed)
            || stderr_capped.load(std::sync::atomic::Ordering::Relaxed),
    );
    if let Some(note) = &cut_note {
        out.push('\n');
        out.push_str(note);
    }

    // Hand the tree back. Two cases, and the same call does both: if nothing
    // had to be stopped, clearing kill-on-close is what lets a stdio-redirected
    // `nohup`/server outlive the command (members are only killed when this
    // handle closes *with* the flag set); if the pipe-holder above was
    // terminated, it is already gone and this is a formality.
    if let Some(tree) = tree {
        tree.release();
    }

    let code = status.code();
    // On failure, name the shell so the model can correct its syntax on the
    // next attempt instead of retrying the same wrong command.
    let mut result = format!(
        "exit code: {}\n{out}",
        code.map(|c| c.to_string()).unwrap_or_else(|| "signal".into())
    );
    if code != Some(0) {
        result.push_str(&format!("\n(shell: {})", shell.label));
    }
    Ok(result)
}

/// The shell a `bash` command runs in.
#[derive(Debug, Clone)]
pub struct ShellSpec {
    /// Executable path or name.
    pub program: String,
    /// Arguments placed before the command string.
    pub args: Vec<String>,
    /// Human-readable label shown to the model.
    pub label: String,
    /// One-line syntax hint shown to the model.
    pub syntax_hint: String,
}

impl ShellSpec {
    /// The full argv for one command.
    fn invocation(&self, command: &str) -> (String, Vec<String>) {
        let mut args = self.args.clone();
        args.push(command.to_string());
        (self.program.clone(), args)
    }
}

/// Detect the shell once and reuse it.
///
/// Resolution order:
///
/// 1. `NGU_SHELL` — an explicit override, e.g. `NGU_SHELL=cmd /C`.
/// 2. On Windows, Git Bash when installed. Models overwhelmingly write POSIX
///    commands, and Git Bash both understands them and resolves Windows paths,
///    so it is the better default than PowerShell. WSL's `bash.exe` is
///    deliberately not selected: its filesystem is isolated from the Windows
///    working directory.
/// 3. PowerShell on Windows, `bash`/`sh` elsewhere.
pub fn detect_shell() -> ShellSpec {
    use std::sync::OnceLock;
    static SHELL: OnceLock<ShellSpec> = OnceLock::new();
    SHELL.get_or_init(resolve_shell).clone()
}

fn resolve_shell() -> ShellSpec {
    if let Ok(custom) = std::env::var("NGU_SHELL") {
        let mut parts = custom.split_whitespace();
        if let Some(program) = parts.next() {
            return ShellSpec {
                program: program.to_string(),
                args: parts.map(str::to_string).collect(),
                label: format!("`{custom}` (from NGU_SHELL)"),
                syntax_hint: "use the syntax that shell expects".to_string(),
            };
        }
    }

    if cfg!(windows) {
        if let Some(git_bash) = find_git_bash() {
            return ShellSpec {
                program: git_bash,
                args: vec!["-c".to_string()],
                label: "Git Bash (POSIX shell)".to_string(),
                syntax_hint: "write POSIX commands such as `ls -la`, `grep -rn`, `find . -name`; \
                              Windows paths like `C:\\dir` also work, and `pwd` reports `/c/dir` form"
                    .to_string(),
            };
        }
        return ShellSpec {
            program: "powershell".to_string(),
            args: vec![
                "-NoProfile".to_string(),
                "-NonInteractive".to_string(),
                "-Command".to_string(),
            ],
            label: "Windows PowerShell (not a POSIX shell)".to_string(),
            syntax_hint: "write PowerShell: `Get-ChildItem` not `ls -la`, \
                          `Select-String` not `grep`, `$env:NAME` not `$NAME`"
                .to_string(),
        };
    }

    for candidate in ["bash", "sh"] {
        if which(candidate).is_some() {
            return ShellSpec {
                program: candidate.to_string(),
                args: vec!["-c".to_string()],
                label: format!("`{candidate}` (POSIX shell)"),
                syntax_hint: "write POSIX commands".to_string(),
            };
        }
    }

    ShellSpec {
        program: "sh".to_string(),
        args: vec!["-c".to_string()],
        label: "`sh` (POSIX shell)".to_string(),
        syntax_hint: "write POSIX commands".to_string(),
    }
}

/// Locate Git Bash on Windows, skipping WSL's `bash.exe` under System32.
fn find_git_bash() -> Option<String> {
    const CANDIDATES: [&str; 4] = [
        r"C:\Program Files\Git\bin\bash.exe",
        r"C:\Program Files (x86)\Git\bin\bash.exe",
        r"C:\Program Files\Git\usr\bin\bash.exe",
        r"C:\Program Files (x86)\Git\usr\bin\bash.exe",
    ];
    for candidate in CANDIDATES {
        if Path::new(candidate).exists() {
            return Some(candidate.to_string());
        }
    }
    // Fall back to PATH, but never to the WSL shim.
    let found = which("bash")?;
    let lowered = found.to_lowercase();
    if lowered.contains("system32") || lowered.contains("syswow64") {
        return None;
    }
    Some(found)
}

/// Minimal `which`: look the program up on `PATH`.
fn which(program: &str) -> Option<String> {
    let path = std::env::var_os("PATH")?;
    let extensions: Vec<String> = if cfg!(windows) {
        std::env::var("PATHEXT")
            .unwrap_or_else(|_| ".EXE;.CMD;.BAT".to_string())
            .split(';')
            .map(|e| e.to_lowercase())
            .collect()
    } else {
        vec![String::new()]
    };

    for dir in std::env::split_paths(&path) {
        for ext in &extensions {
            let candidate = dir.join(format!("{program}{ext}"));
            if candidate.is_file() {
                return Some(candidate.display().to_string());
            }
        }
    }
    None
}

/// Say so when the *buffer* was cut, not just the view.
///
/// [`truncate_output`] trims the two ends of whatever was collected; this note
/// distinguishes "that is all it printed" from "that is all we kept", which
/// otherwise look identical in a head/tail rendering.
fn note_capped_output(out: &mut String, capped: bool) {
    if !capped {
        return;
    }
    out.push('\n');
    out.push_str(&format!(
        "--- buffering stopped at {} MB, so the tail above is not the end of what \
         the command printed ---",
        MAX_BASH_BUFFER / (1024 * 1024)
    ));
}

/// Keep the head and tail of an oversized output.
fn truncate_output(text: &str, max: usize) -> String {
    if text.len() <= max {
        return text.to_string();
    }
    let head = max / 2;
    let tail = max / 2;
    let start: String = text.chars().take(head).collect();
    let end: String = text.chars().rev().take(tail).collect::<String>().chars().rev().collect();
    format!("{start}\n… ({} bytes omitted) …\n{end}", text.len() - head - tail)
}
