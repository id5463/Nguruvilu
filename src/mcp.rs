//! Minimal MCP client over stdio.
//!
//! MCP is JSON-RPC 2.0 with newline-delimited frames. This implements the
//! subset an agent needs to *use* a server: `initialize`, `tools/list`, and
//! `tools/call`. Resources and prompts are deliberately out of scope.
//!
//! A server runs as a child process. One reader task owns stdout and fans
//! responses out by request id, so concurrent tool calls do not serialize on
//! reading.

use std::collections::HashMap;
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, Command};
use tokio::sync::{oneshot, Mutex};

/// Protocol revision this client speaks.
pub const PROTOCOL_VERSION: &str = "2024-11-05";

/// How to launch one MCP server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpSpec {
    /// Identifier used for namespacing tools and for the ledger.
    pub id: String,
    /// Executable to launch.
    pub command: String,
    /// Arguments.
    #[serde(default)]
    pub args: Vec<String>,
    /// Extra environment variables.
    #[serde(default)]
    pub env: HashMap<String, String>,
    /// Working directory.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cwd: Option<String>,
    /// Per-call timeout.
    #[serde(default = "default_timeout")]
    pub timeout_ms: u64,
}

fn default_timeout() -> u64 {
    60_000
}

impl McpSpec {
    /// Build a stdio spec.
    pub fn stdio(id: impl Into<String>, command: impl Into<String>, args: Vec<String>) -> Self {
        Self {
            id: id.into(),
            command: command.into(),
            args,
            env: HashMap::new(),
            cwd: None,
            timeout_ms: default_timeout(),
        }
    }
}

/// One tool advertised by a server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpTool {
    /// Tool name as the server knows it.
    pub name: String,
    /// Description for the model.
    #[serde(default)]
    pub description: String,
    /// JSON Schema for arguments.
    #[serde(default, rename = "inputSchema")]
    pub input_schema: Value,
}

/// A connected MCP server.
pub struct McpClient {
    server: String,
    stdin: Arc<Mutex<ChildStdin>>,
    pending: Arc<Mutex<HashMap<u64, oneshot::Sender<Value>>>>,
    next_id: AtomicU64,
    child: Mutex<Child>,
    tools: Vec<McpTool>,
    timeout: Duration,
}

impl McpClient {
    
/// Resolve a command name to something the OS can actually launch.
///
/// On Windows `npx` is `npx.cmd`, and `Command::new` does not consult `PATHEXT`
/// — so a pack that says `"command": "npx"` fails with "program not found" even
/// though the command works in a shell. Every MCP server published as an npm
/// package is launched that way, so this is the common case rather than an
/// edge.
///
/// The lookup order is deliberate: an explicit `.exe` beats a `.cmd` shim,
/// because a shim needs a shell to interpret it and a real executable does not.
fn resolve_program(command: &str) -> std::path::PathBuf {
    let path = std::path::Path::new(command);

    // A path with a separator is the caller's business: they named a file.
    if path.components().count() > 1 || path.extension().is_some() {
        return path.to_path_buf();
    }

    #[cfg(windows)]
    {
        // `where` is the same lookup a shell does, so whatever works in a
        // terminal works here.
        if let Ok(output) = std::process::Command::new("where").arg(command).output() {
            if output.status.success() {
                let text = String::from_utf8_lossy(&output.stdout);
                let candidates: Vec<&str> = text.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
                // An `.exe` first: a `.cmd` needs a shell, and spawning one
                // through `Command` without `cmd /c` would not run it.
                if let Some(exe) = candidates.iter().find(|c| c.to_ascii_lowercase().ends_with(".exe")) {
                    return std::path::PathBuf::from(exe);
                }
                if let Some(cmd) = candidates.iter().find(|c| {
                    let lower = c.to_ascii_lowercase();
                    lower.ends_with(".cmd") || lower.ends_with(".bat")
                }) {
                    return std::path::PathBuf::from(cmd);
                }
                if let Some(any) = candidates.first() {
                    return std::path::PathBuf::from(any);
                }
            }
        }
    }

    path.to_path_buf()
}

/// Launch a server, complete the handshake, and fetch its tool list.
    pub async fn connect(spec: &McpSpec) -> Result<Self> {
        let program = Self::resolve_program(&spec.command);
        let mut command = Command::new(&program);
        command
            .args(&spec.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true);
        for (key, value) in &spec.env {
            command.env(key, value);
        }
        if let Some(cwd) = &spec.cwd {
            command.current_dir(cwd);
        }

        let mut child = command
            .spawn()
            .with_context(|| format!("launching MCP server '{}' ({})", spec.id, spec.command))?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| anyhow!("MCP server '{}' has no stdin", spec.id))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow!("MCP server '{}' has no stdout", spec.id))?;

        let pending: Arc<Mutex<HashMap<u64, oneshot::Sender<Value>>>> =
            Arc::new(Mutex::new(HashMap::new()));

        // One reader owns stdout for the connection's lifetime.
        {
            let pending = Arc::clone(&pending);
            let server = spec.id.clone();
            tokio::spawn(async move {
                let mut lines = BufReader::new(stdout).lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    let line = line.trim();
                    if line.is_empty() {
                        continue;
                    }
                    let Ok(message) = serde_json::from_str::<Value>(line) else {
                        continue;
                    };
                    let Some(id) = message.get("id").and_then(|i| i.as_u64()) else {
                        // A notification or a server-initiated request: not used here.
                        continue;
                    };
                    let sender = pending.lock().await.remove(&id);
                    if let Some(sender) = sender {
                        let _ = sender.send(message);
                    }
                }
                // The pipe closed: fail every waiter rather than hang them.
                let mut waiting = pending.lock().await;
                for (_, sender) in waiting.drain() {
                    let _ = sender.send(json!({ "error": { "message": format!("MCP server '{server}' closed the connection") } }));
                }
            });
        }

        let mut client = Self {
            server: spec.id.clone(),
            stdin: Arc::new(Mutex::new(stdin)),
            pending,
            next_id: AtomicU64::new(1),
            child: Mutex::new(child),
            tools: Vec::new(),
            timeout: Duration::from_millis(spec.timeout_ms),
        };

        client.initialize().await?;
        client.tools = client.list_tools().await?;
        Ok(client)
    }

    /// Server id.
    pub fn server(&self) -> &str {
        &self.server
    }

    /// Tools advertised by the server.
    pub fn tools(&self) -> &[McpTool] {
        &self.tools
    }

    /// The tool name this client exposes for one server tool.
    ///
    /// Namespacing keeps two servers from claiming the same tool name, which
    /// the tool table would otherwise reject.
    pub fn exposed_name(&self, tool: &str) -> String {
        format!("mcp__{}__{}", sanitize(&self.server), sanitize(tool))
    }

    async fn initialize(&mut self) -> Result<()> {
        let params = json!({
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": { "name": "nguruvilu", "version": env!("CARGO_PKG_VERSION") }
        });
        let response = self.request("initialize", params).await?;
        if response.get("error").is_some() {
            return Err(anyhow!(
                "MCP server '{}' refused initialize: {}",
                self.server,
                response["error"]
            ));
        }
        // The handshake is complete only after the client says so.
        self.notify("notifications/initialized", json!({})).await?;
        Ok(())
    }

    async fn list_tools(&mut self) -> Result<Vec<McpTool>> {
        let response = self.request("tools/list", json!({})).await?;
        if let Some(error) = response.get("error") {
            return Err(anyhow!("MCP server '{}' tools/list failed: {error}", self.server));
        }
        let tools = response
            .get("result")
            .and_then(|r| r.get("tools"))
            .and_then(|t| t.as_array())
            .cloned()
            .unwrap_or_default();
        let mut parsed = Vec::with_capacity(tools.len());
        for tool in tools {
            match serde_json::from_value::<McpTool>(tool) {
                Ok(t) => parsed.push(t),
                // A malformed entry must not take down the whole server.
                Err(_) => continue,
            }
        }
        Ok(parsed)
    }

    /// Call one tool and flatten its content into text.
    pub async fn call(&self, tool: &str, arguments: Value) -> Result<String> {
        let params = json!({ "name": tool, "arguments": arguments });
        let response = self.request("tools/call", params).await?;

        if let Some(error) = response.get("error") {
            let message = error.get("message").and_then(|m| m.as_str()).unwrap_or("unknown error");
            return Err(anyhow!("MCP tool '{tool}' failed: {message}"));
        }

        let result = response.get("result").cloned().unwrap_or(Value::Null);
        let is_error = result.get("isError").and_then(|e| e.as_bool()).unwrap_or(false);
        let text = flatten_content(&result);

        if is_error {
            return Err(anyhow!("MCP tool '{tool}' reported an error: {text}"));
        }
        Ok(if text.is_empty() { "(no content)".to_string() } else { text })
    }

    /// Terminate the server process.
    pub async fn shutdown(&self) {
        let mut child = self.child.lock().await;
        let _ = child.kill().await;
        let _ = child.wait().await;
    }

    async fn request(&self, method: &str, params: Value) -> Result<Value> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (sender, receiver) = oneshot::channel();
        self.pending.lock().await.insert(id, sender);

        let message = json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params });
        self.write_line(&message).await?;

        match tokio::time::timeout(self.timeout, receiver).await {
            Ok(Ok(response)) => Ok(response),
            Ok(Err(_)) => Err(anyhow!("MCP server '{}' dropped request {id}", self.server)),
            Err(_) => {
                self.pending.lock().await.remove(&id);
                Err(anyhow!(
                    "MCP server '{}' timed out after {:?} on {method}",
                    self.server,
                    self.timeout
                ))
            }
        }
    }

    async fn notify(&self, method: &str, params: Value) -> Result<()> {
        let message = json!({ "jsonrpc": "2.0", "method": method, "params": params });
        self.write_line(&message).await
    }

    async fn write_line(&self, message: &Value) -> Result<()> {
        let mut line = serde_json::to_string(message)?;
        line.push('\n');
        let mut stdin = self.stdin.lock().await;
        stdin
            .write_all(line.as_bytes())
            .await
            .with_context(|| format!("writing to MCP server '{}'", self.server))?;
        stdin.flush().await?;
        Ok(())
    }
}

/// Flatten an MCP content array into plain text.
fn flatten_content(result: &Value) -> String {
    let Some(items) = result.get("content").and_then(|c| c.as_array()) else {
        // Some servers return a bare object.
        return result.as_str().unwrap_or("").to_string();
    };
    let mut parts = Vec::with_capacity(items.len());
    for item in items {
        match item.get("type").and_then(|t| t.as_str()) {
            Some("text") => {
                if let Some(text) = item.get("text").and_then(|t| t.as_str()) {
                    parts.push(text.to_string());
                }
            }
            Some(other) => parts.push(format!("[{other} content omitted]")),
            None => {}
        }
    }
    parts.join("\n")
}

/// Make a string safe for use inside a tool name.
fn sanitize(value: &str) -> String {
    value
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '_' })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_names_are_namespaced_and_sanitized() {
        let spec = McpSpec::stdio("my server", "x", vec![]);
        assert_eq!(spec.id, "my server");
        assert_eq!(sanitize("my server"), "my_server");
        assert_eq!(sanitize("tools/read.file"), "tools_read_file");
    }

    #[test]
    fn content_is_flattened_to_text() {
        let result = json!({
            "content": [
                { "type": "text", "text": "first" },
                { "type": "image", "data": "..." },
                { "type": "text", "text": "second" }
            ]
        });
        let text = flatten_content(&result);
        assert!(text.contains("first"));
        assert!(text.contains("second"));
        assert!(text.contains("image content omitted"));

        // A bare string result is passed through.
        assert_eq!(flatten_content(&json!("plain")), "plain");
        // An empty result yields nothing.
        assert_eq!(flatten_content(&json!({})), "");
    }

    #[test]
    fn exposed_names_include_the_server() {
        let spec = McpSpec::stdio("github", "npx", vec![]);
        assert_eq!(spec.id, "github");
        // The naming rule is what keeps two servers from colliding.
        assert_eq!(format!("mcp__{}__{}", sanitize("github"), sanitize("create_issue")), "mcp__github__create_issue");
    }
}
