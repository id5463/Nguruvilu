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

/// Output ceiling for `bash` before head/tail truncation kicks in.
const MAX_BASH_OUTPUT: usize = 30_000;

/// Default `bash` timeout.
const DEFAULT_BASH_TIMEOUT_MS: u64 = 120_000;

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
                 Set timeout_ms to bound long-running commands. Output is truncated when very large.\n\
                 Pass background: true to start the command as a job and return immediately; \
                 collect it later with job_output / job_list / job_kill (timeout_ms does not apply).",
                shell.label, shell.syntax_hint
            ),
            json!({
                "type": "object",
                "properties": {
                    "command": { "type": "string", "description": "Command to run" },
                    "timeout_ms": { "type": "integer", "description": "Timeout in milliseconds (default 120000)" },
                    "cwd": { "type": "string", "description": "Working directory for the command" },
                    "background": { "type": "boolean", "description": "Run as a background job and return a job id immediately (default false)" }
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

/// Resolve a caller-supplied path against the process working directory.
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
    let path = Path::new(raw);
    if path.is_absolute() {
        return path.to_path_buf();
    }

    #[cfg(windows)]
    if let Some(drive) = posix_drive_path(raw) {
        return drive;
    }

    std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")).join(path)
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

async fn read_tool(args: Value) -> Result<String> {
    read_any(args).await.map(|output| output.text)
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
    for (i, line) in lines[start - 1..end].iter().enumerate() {
        let number = start + i;
        out.push_str(&format!("{:>width$}| {}\n", number, line, width = width));
    }
    if end < total {
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

async fn bash_tool(args: Value) -> Result<String> {
    let command = args
        .get("command")
        .and_then(|c| c.as_str())
        .ok_or_else(|| anyhow!("missing required argument: command"))?;
    let timeout = Duration::from_millis(
        args.get("timeout_ms")
            .and_then(|v| v.as_u64())
            .unwrap_or(DEFAULT_BASH_TIMEOUT_MS),
    );
    let cwd = args.get("cwd").and_then(|c| c.as_str()).map(resolve_path);

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
        let job = super::jobs::spawn_background(command, cmd)?;
        return Ok(format!(
            "started {} in the background: {command}\n\
             Read its output with job_output (wait_ms blocks until it finishes), \
             list jobs with job_list, stop it with job_kill.",
            job.id
        ));
    }

    let mut child = cmd
        .spawn()
        .with_context(|| format!("spawning {program} (shell: {})", shell.label))?;

    let mut stdout_pipe = child.stdout.take();
    let mut stderr_pipe = child.stderr.take();

    // Read both pipes concurrently: a command that fills one pipe while we
    // wait on the other would otherwise deadlock.
    let stdout_task = tokio::spawn(async move {
        let mut buf = Vec::new();
        if let Some(pipe) = stdout_pipe.as_mut() {
            let _ = pipe.read_to_end(&mut buf).await;
        }
        buf
    });
    let stderr_task = tokio::spawn(async move {
        let mut buf = Vec::new();
        if let Some(pipe) = stderr_pipe.as_mut() {
            let _ = pipe.read_to_end(&mut buf).await;
        }
        buf
    });

    let status = match tokio::time::timeout(timeout, child.wait()).await {
        Ok(result) => result.context("waiting for command")?,
        Err(_) => {
            let _ = child.kill().await;
            let _ = child.wait().await;
            return Ok(format!(
                "command timed out after {} ms and was killed: {command}",
                timeout.as_millis()
            ));
        }
    };

    let stdout = stdout_task.await.unwrap_or_default();
    let stderr = stderr_task.await.unwrap_or_default();

    let mut out = String::new();
    let stdout_text = String::from_utf8_lossy(&stdout);
    let stderr_text = String::from_utf8_lossy(&stderr);

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
    if out.trim().is_empty() {
        out.push_str("(no output)");
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

#[cfg(test)]
mod image_tests {
    use super::*;
    use crate::message::image_mime;
    use serde_json::json;

    /// A minimal valid 2x1 PNG, so the magic-byte path is exercised for real.
    fn tiny_png() -> Vec<u8> {
        let mut bytes = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
        bytes.extend_from_slice(&[0, 0, 0, 13]);
        bytes.extend_from_slice(b"IHDR");
        bytes.extend_from_slice(&2u32.to_be_bytes()); // width
        bytes.extend_from_slice(&1u32.to_be_bytes()); // height
        bytes.extend_from_slice(&[8, 6, 0, 0, 0]);
        bytes
    }

    fn scratch(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("ngu-img-{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    #[tokio::test]
    async fn reading_an_image_attaches_it_instead_of_failing() {
        let path = scratch("pic.png");
        std::fs::write(&path, tiny_png()).unwrap();

        let output = read_any(json!({ "path": path })).await.expect("read succeeds");

        assert_eq!(output.images.len(), 1, "the image is attached");
        let image = &output.images[0];
        assert!(image.url.starts_with("data:image/png;base64,"), "{}", image.url);
        assert!(output.text.contains("image/png"));
        assert!(output.text.contains("2x1"), "dimensions are reported: {}", output.text);
    }

    #[tokio::test]
    async fn a_png_reports_its_real_dimensions() {
        assert_eq!(image_dimensions(&tiny_png(), "image/png"), Some((2, 1)));
    }

    #[tokio::test]
    async fn text_files_still_read_as_text_with_no_attachment() {
        let path = scratch("note.txt");
        std::fs::write(&path, "alpha\nbeta\n").unwrap();

        let output = read_any(json!({ "path": path })).await.expect("read succeeds");
        assert!(output.images.is_empty(), "a text file attaches nothing");
        assert!(output.text.contains("alpha"));
        assert!(output.text.contains("beta"));
    }

    #[tokio::test]
    async fn binary_that_is_not_an_image_is_refused_clearly() {
        let path = scratch("blob.bin");
        std::fs::write(&path, [0x00u8, 0x01, 0x02, 0xFF, 0xFE]).unwrap();

        let error = read_any(json!({ "path": path })).await.expect_err("refused");
        let text = format!("{error:#}");
        assert!(text.contains("binary"), "{text}");
        assert!(text.contains("not a recognised image format"), "{text}");
    }

    #[test]
    fn formats_are_detected_by_magic_bytes_not_by_name() {
        // A file named .png that is not one must not be sent as a broken image.
        assert_eq!(image_mime(b"not a png at all", "picture.png"), None);
        assert_eq!(image_mime(&[0xFF, 0xD8, 0xFF, 0xE0], "x.jpg"), Some("image/jpeg"));
        assert_eq!(image_mime(b"GIF89a....", "x.gif"), Some("image/gif"));
        assert_eq!(image_mime(b"BM......", "x.bmp"), Some("image/bmp"));
        assert_eq!(image_mime(b"RIFF....WEBPVP8 ", "x.webp"), Some("image/webp"));
        // SVG has no magic bytes, so the extension is the only signal.
        assert_eq!(image_mime(b"<svg/>", "x.svg"), Some("image/svg+xml"));
    }

    #[test]
    fn a_data_uri_carries_the_bytes() {
        let uri = crate::message::data_uri("image/png", &[1, 2, 3]);
        assert!(uri.starts_with("data:image/png;base64,"));
        assert!(uri.ends_with("AQID"), "{uri}");
    }
}

#[cfg(all(test, windows))]
mod path_tests {
    use super::*;

    #[test]
    fn a_shell_style_drive_path_is_translated() {
        // What `pwd` prints in Git Bash, and what a model then hands back.
        assert_eq!(
            posix_drive_path("/i/tmp-test/x.txt"),
            Some(PathBuf::from("I:\\tmp-test\\x.txt"))
        );
        assert_eq!(
            posix_drive_path("/c/Users/a/Desktop/a.txt"),
            Some(PathBuf::from("C:\\Users\\a\\Desktop\\a.txt"))
        );
        assert_eq!(posix_drive_path("/i"), Some(PathBuf::from("I:\\")));
    }

    #[test]
    fn an_ordinary_relative_name_is_not_mistaken_for_a_drive() {
        // `/code/x` is a real directory name, not drive `C:`.
        assert_eq!(posix_drive_path("/code/x"), None);
        assert_eq!(posix_drive_path("/tmp/x"), None);
        assert_eq!(posix_drive_path("relative/x"), None);
        assert_eq!(posix_drive_path(""), None);
    }

    #[test]
    fn resolve_path_handles_every_form_a_model_might_use() {
        // Windows absolute.
        assert_eq!(
            resolve_path("I:\\tmp-test\\a.txt"),
            PathBuf::from("I:\\tmp-test\\a.txt")
        );
        // Git Bash absolute, which is what `pwd` shows.
        assert_eq!(
            resolve_path("/i/tmp-test/a.txt"),
            PathBuf::from("I:\\tmp-test\\a.txt")
        );
        // Relative stays relative to the working directory.
        let relative = resolve_path("a.txt");
        assert!(relative.ends_with("a.txt"));
        assert!(relative.is_absolute(), "joined onto the working directory");
    }
}
