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
                    "path": { "type": "string", "description": "File path (absolute, or relative to the working directory)" },
                    "start_line": { "type": "integer", "description": "First line to return, 1-based. Defaults to 1." },
                    "end_line": { "type": "integer", "description": "Last line to return, inclusive. Defaults to start_line + 199." }
                },
                "required": ["path"]
            }),
            "kernel",
            |args| Box::pin(read_tool(args)) as ToolFuture,
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
                    "path": { "type": "string", "description": "File path" },
                    "content": { "type": "string", "description": "Full file contents" }
                },
                "required": ["path", "content"]
            }),
            "kernel",
            |args| Box::pin(write_tool(args)) as ToolFuture,
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
                    "path": { "type": "string", "description": "File path" },
                    "old_str": { "type": "string", "description": "Exact text to replace; must occur exactly once" },
                    "new_str": { "type": "string", "description": "Replacement text" }
                },
                "required": ["path", "old_str", "new_str"]
            }),
            "kernel",
            |args| Box::pin(edit_tool(args)) as ToolFuture,
        ),
        ConflictPolicy::Error,
    )?;

    registry.register(
        ToolDef::new(
            "bash",
            "Run a shell command and return its combined output. \
             Set timeout_ms to bound long-running commands. \
             Output is truncated when very large.",
            json!({
                "type": "object",
                "properties": {
                    "command": { "type": "string", "description": "Command to run" },
                    "timeout_ms": { "type": "integer", "description": "Timeout in milliseconds (default 120000)" },
                    "cwd": { "type": "string", "description": "Working directory for the command" }
                },
                "required": ["command"]
            }),
            "kernel",
            |args| Box::pin(bash_tool(args)) as ToolFuture,
        ),
        ConflictPolicy::Error,
    )?;

    Ok(())
}

/// Resolve a caller-supplied path against the process working directory.
fn resolve_path(raw: &str) -> PathBuf {
    let path = Path::new(raw);
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")).join(path)
    }
}

async fn read_tool(args: Value) -> Result<String> {
    let raw_path = args
        .get("path")
        .and_then(|p| p.as_str())
        .ok_or_else(|| anyhow!("missing required argument: path"))?;
    let path = resolve_path(raw_path);

    let content = tokio::fs::read_to_string(&path)
        .await
        .with_context(|| format!("reading {}", path.display()))?;

    let lines: Vec<&str> = content.lines().collect();
    let total = lines.len();

    let start = args.get("start_line").and_then(|v| v.as_u64()).unwrap_or(1).max(1) as usize;
    let end = match args.get("end_line").and_then(|v| v.as_u64()) {
        Some(v) => (v as usize).max(start),
        None => start + DEFAULT_READ_WINDOW - 1,
    };
    let end = end.min(start + MAX_READ_WINDOW - 1).min(total.max(1));

    if total == 0 {
        return Ok(format!("{} (empty file)", path.display()));
    }
    if start > total {
        return Ok(format!(
            "{} has {total} lines; start_line {start} is past the end",
            path.display()
        ));
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
    Ok(out)
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

    let (program, program_args) = shell_invocation(command);

    let mut cmd = Command::new(&program);
    cmd.args(&program_args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if let Some(dir) = &cwd {
        cmd.current_dir(dir);
    }

    let mut child = cmd
        .spawn()
        .with_context(|| format!("spawning {program}"))?;

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

    let code = status.code().map(|c| c.to_string()).unwrap_or_else(|| "signal".into());
    Ok(format!("exit code: {code}\n{out}"))
}

/// Build the shell invocation for a command string.
///
/// Windows defaults to PowerShell because that is the platform's real shell
/// surface; `NGU_SHELL` overrides it (for example `NGU_SHELL=cmd /C` or
/// `NGU_SHELL=bash -c`).
fn shell_invocation(command: &str) -> (String, Vec<String>) {
    if let Ok(custom) = std::env::var("NGU_SHELL") {
        let mut parts = custom.split_whitespace();
        if let Some(program) = parts.next() {
            let mut args: Vec<String> = parts.map(str::to_string).collect();
            args.push(command.to_string());
            return (program.to_string(), args);
        }
    }

    if cfg!(windows) {
        (
            "powershell".to_string(),
            vec![
                "-NoProfile".to_string(),
                "-NonInteractive".to_string(),
                "-Command".to_string(),
                command.to_string(),
            ],
        )
    } else {
        ("sh".to_string(), vec!["-c".to_string(), command.to_string()])
    }
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
