//! 命令行把一次运行的过程打到 stderr 上。
//!
//! stdout 只留答案：`--json` 模式的调用方要解析它，任何进度文字混进去都会
//! 让那份 JSON 不再是 JSON。所以进度走 stderr，答案走 stdout。

use std::sync::Arc;

use nguruvilu::agent::AgentObserver;
use nguruvilu::events::{self, EventBus};

pub struct PrettyObserver {
    pub quiet: bool,
}

/// Emits `session.end` when the run finishes — including early `?` returns,
/// because Drop does not care about `?`.
pub struct SessionEnd {
    events: nguruvilu::events::EventBus,
    session: String,
}

impl SessionEnd {
    /// 记下这一轮属于哪个会话；`session.end` 在它被丢弃时发出。
    ///
    /// 构造函数而不是公开字段：事件总线和会话 id 是一对，分开给会让人以
    /// 为可以只给其中一个。
    pub fn new(events: nguruvilu::events::EventBus, session: impl Into<String>) -> Self {
        Self { events, session: session.into() }
    }
}

impl Drop for SessionEnd {
    fn drop(&mut self) {
        self.events.emit(
            nguruvilu::events::SESSION_END,
            &serde_json::json!({ "session": &self.session }),
        );
    }
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

    fn on_step_retry(&self, attempt: usize, reason: &str) {
        if !self.quiet {
            eprintln!("\n[retry {attempt}: {reason} — output so far is discarded]");
        }
    }

    fn on_continuation(&self, reason: &str) {
        if !self.quiet {
            eprintln!("\n[{reason}; asking the model to continue]");
        }
    }
}
