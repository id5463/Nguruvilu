//! MCP 响应的解析与整理：把服务端返回的形状变成内核能用的形状。
//!
//! 三个关注点，各自独立：一条工具结果压成 `ToolOutput`（[`flatten_result`]）、
//! 一条文本做长度与不可打印字符的清洗（[`sanitize`]）、SSE 流切成一条条消息
//! （[`sse_message`] / [`sse_messages`]）。

use std::sync::atomic::Ordering;

use serde_json::Value;

use super::model::{IdSpace, KillReport, McpClient, McpSpec, Pending, PROTOCOL_VERSION};

pub(crate) fn sanitize(value: &str) -> String {
    value
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '_' })
        .collect()
}

/// The answer to `id` inside an SSE body.
///
/// The stream may also carry server-initiated messages; the id is what makes
/// one of them *the* response rather than a side channel event.
pub(crate) fn sse_message(body: &str, id: u64) -> Option<Value> {
    sse_messages(body)
        .into_iter()
        .find(|message| message.get("id").and_then(|i| i.as_u64()) == Some(id))
}

/// Every JSON-RPC message an SSE body carries.
///
/// One event's `data:` lines hold one message (several join with newlines);
/// an event ends at the blank line, and a stream that ends without one still
/// flushes its last payload.
pub(crate) fn sse_messages(body: &str) -> Vec<Value> {
    let mut messages = Vec::new();
    let mut data: Vec<&str> = Vec::new();
    for line in body.lines() {
        if let Some(rest) = line.strip_prefix("data:") {
            data.push(rest.trim_start());
            continue;
        }
        if line.trim().is_empty() && !data.is_empty() {
            let joined = data.join("\n");
            data.clear();
            if let Ok(value) = serde_json::from_str::<Value>(&joined) {
                messages.push(value);
            }
        }
    }
    if !data.is_empty() {
        if let Ok(value) = serde_json::from_str::<Value>(&data.join("\n")) {
            messages.push(value);
        }
    }
    messages
}

/// The tail of a response body, for an error message: enough to recognise
/// the failure, never the whole document.
pub(crate) fn body_tail(body: &str) -> String {
    let trimmed = body.trim();
    const MAX: usize = 300;
    if trimmed.chars().count() <= MAX {
        if trimmed.is_empty() { "(empty body)".to_string() } else { trimmed.to_string() }
    } else {
        let cut: String = trimmed.chars().take(MAX).collect();
        format!("{cut}\u{2026}")
    }
}


