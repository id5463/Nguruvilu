//! Minimal MCP client over stdio and Streamable HTTP.
//!
//! MCP is JSON-RPC 2.0. This implements the subset an agent needs to *use* a
//! server: `initialize`, `tools/list`, and `tools/call`. Resources and
//! prompts are deliberately out of scope.
//!
//! Two transports carry the same messages. A stdio server runs as a child
//! process with newline-delimited frames on its pipes: one reader task owns
//! stdout and fans responses out by request id, so concurrent tool calls do
//! not serialize on reading. A Streamable HTTP server is one URL: each
//! message is a `POST`, answered either as a single JSON document or as a
//! `text/event-stream` whose events end in the response, with a
//! `Mcp-Session-Id` assigned at `initialize` travelling on every later
//! request.

use std::collections::{BTreeMap, HashMap};
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
    /// Endpoint for the `streamable-http` transport; `None` means stdio.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    /// Headers for `streamable-http` — `Authorization` and friends, sent on
    /// every request to the endpoint.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub headers: BTreeMap<String, String>,
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
            url: None,
            headers: BTreeMap::new(),
        }
    }

    /// Build a Streamable HTTP spec: one endpoint, no child process.
    pub fn http(id: impl Into<String>, url: impl Into<String>, headers: BTreeMap<String, String>) -> Self {
        Self {
            id: id.into(),
            command: String::new(),
            args: Vec::new(),
            env: HashMap::new(),
            cwd: None,
            timeout_ms: default_timeout(),
            url: Some(url.into()),
            headers,
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
    /// The child's stdin. `None` for an HTTP server, which has no child.
    stdin: Option<Arc<Mutex<ChildStdin>>>,
    pending: Arc<Mutex<HashMap<u64, oneshot::Sender<Value>>>>,
    next_id: AtomicU64,
    /// The child itself. `None` for an HTTP server.
    child: Option<Mutex<Child>>,
    /// The HTTP wiring, present exactly when [`McpSpec::url`] was.
    http: Option<HttpSession>,
    tools: Vec<McpTool>,
    timeout: Duration,
}

/// One Streamable HTTP connection.
///
/// Every message is a `POST` to `url`; the server may answer with a single
/// JSON body or an SSE stream, and hands out the `Mcp-Session-Id` with the
/// `initialize` response, which travels on every later request.
struct HttpSession {
    client: reqwest::Client,
    url: String,
    headers: BTreeMap<String, String>,
    /// The assigned session id, once `initialize` returned one.
    session: Mutex<Option<String>>,
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
///
/// `pub(crate)` beyond its MCP role: pack install resolves `npm` the same way,
/// so a pack that declares an `npm:` source runs the same lookup rather than
/// re-inventing the Windows `where` dance.
pub(crate) fn resolve_program(command: &str) -> std::path::PathBuf {
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
        if let Some(url) = spec.url.clone() {
            return Self::connect_http(spec, &url).await;
        }
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
            stdin: Some(Arc::new(Mutex::new(stdin))),
            pending,
            next_id: AtomicU64::new(1),
            child: Some(Mutex::new(child)),
            http: None,
            tools: Vec::new(),
            timeout: Duration::from_millis(spec.timeout_ms),
        };

        client.initialize().await?;
        client.tools = client.list_tools().await?;
        Ok(client)
    }

    /// Open a Streamable HTTP server: handshake, then the tool list — the
    /// same sequence a stdio server gets, over `POST`s instead of pipes.
    async fn connect_http(spec: &McpSpec, url: &str) -> Result<Self> {
        let http = HttpSession {
            client: reqwest::Client::builder()
                .user_agent(format!("nguruvilu/{}", env!("CARGO_PKG_VERSION")))
                // An endpoint a pack names travels there directly: picking up
                // the shell's ambient `HTTP_PROXY` is how a proxy set for
                // another tool silently reroutes MCP traffic (the same rule
                // the model route follows).
                .no_proxy()
                .build()
                .with_context(|| format!("building an HTTP client for MCP server '{}'", spec.id))?,
            url: url.to_string(),
            headers: spec.headers.clone(),
            session: Mutex::new(None),
        };
        let mut client = Self {
            server: spec.id.clone(),
            stdin: None,
            pending: Arc::new(Mutex::new(HashMap::new())),
            next_id: AtomicU64::new(1),
            child: None,
            http: Some(http),
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

    /// Call one tool: text flattened into the readable result, image parts
    /// kept as attachments.
    ///
    /// A tool message may only carry text (the OpenAI format says so), so the
    /// images travel in the [`ToolOutput`](crate::tools::ToolOutput) the loop
    /// attaches to a user message after the tool results — the same path the
    /// `read` tool uses. Servers that answer with a screenshot (a browser
    /// tool, say) are visible to the model this way instead of being reduced
    /// to "[image content omitted]".
    pub async fn call(&self, tool: &str, arguments: Value) -> Result<crate::tools::ToolOutput> {
        let params = json!({ "name": tool, "arguments": arguments });
        let response = self.request("tools/call", params).await?;

        if let Some(error) = response.get("error") {
            let message = error.get("message").and_then(|m| m.as_str()).unwrap_or("unknown error");
            return Err(anyhow!("MCP tool '{tool}' failed: {message}"));
        }

        let result = response.get("result").cloned().unwrap_or(Value::Null);
        let is_error = result.get("isError").and_then(|e| e.as_bool()).unwrap_or(false);
        let mut output = flatten_result(&result);

        if is_error {
            return Err(anyhow!("MCP tool '{tool}' reported an error: {}", output.text));
        }
        if output.text.is_empty() && output.images.is_empty() {
            output.text = "(no content)".to_string();
        }
        Ok(output)
    }

    /// Terminate the server: the child process, or the HTTP session.
    pub async fn shutdown(&self) {
        if let Some(http) = &self.http {
            // `DELETE` on the endpoint ends the session server-side. Without
            // one (a handshake that never finished) there is nothing to end,
            // and a failed DELETE just leaves the server to time the session
            // out — neither leaks anything here.
            let session = http.session.lock().await.clone();
            if let Some(session) = session {
                let mut request = http
                    .client
                    .delete(&http.url)
                    .header("Mcp-Session-Id", session);
                for (key, value) in &http.headers {
                    request = request.header(key.as_str(), value.as_str());
                }
                let _ = tokio::time::timeout(
                    std::time::Duration::from_secs(3),
                    request.send(),
                )
                .await;
            }
            return;
        }
        let Some(child) = &self.child else {
            return;
        };
        let mut child = child.lock().await;
        let _ = child.kill().await;
        let _ = child.wait().await;
    }

    /// A bounded close: returns whether the shutdown finished inside `deadline`.
    ///
    /// On timeout the wait is cancelled — which releases the child lock — and
    /// the server is killed synchronously instead. A close must finish whatever
    /// the server is doing: an unbounded wait here is how a window turns into a
    /// ghost on quit (it did).
    pub async fn shutdown_within(&self, deadline: std::time::Duration) -> bool {
        if tokio::time::timeout(deadline, self.shutdown()).await.is_ok() {
            return true;
        }
        self.force_kill();
        false
    }

    /// Best-effort synchronous kill: no await, never panics.
    ///
    /// The close path calls this *first* so the servers are already on their
    /// way out before any waiting starts; the `Drop` path is the same call for
    /// handles nobody shut down explicitly.
    pub fn force_kill(&self) {
        // An HTTP server has no process to kill; its session ends server-side
        // (see `shutdown`) or times out there. Nothing is leaked here.
        if let Some(child) = &self.child {
            if let Ok(mut child) = child.try_lock() {
                let _ = child.start_kill();
            }
        }
    }

    /// A client over an already-running child, without the handshake.
    ///
    /// Tests only: the real path is [`McpClient::connect`], which must
    /// initialize and discover tools before anything may call the server.
    #[cfg(test)]
    pub(crate) fn from_child_for_tests(mut child: Child, server: &str) -> Self {
        let stdin = child.stdin.take().expect("the test child pipes stdin");
        Self {
            server: server.to_string(),
            stdin: Some(Arc::new(Mutex::new(stdin))),
            pending: Arc::new(Mutex::new(HashMap::new())),
            next_id: AtomicU64::new(1),
            child: Some(Mutex::new(child)),
            http: None,
            tools: Vec::new(),
            timeout: Duration::from_secs(60),
        }
    }

    async fn request(&self, method: &str, params: Value) -> Result<Value> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let message = json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params });
        if let Some(http) = &self.http {
            return self.http_post(http, Some(id), method, &message).await;
        }

        let (sender, receiver) = oneshot::channel();
        self.pending.lock().await.insert(id, sender);

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
        if let Some(http) = &self.http {
            self.http_post(http, None, method, &message).await?;
            return Ok(());
        }
        self.write_line(&message).await
    }

    /// One POST over Streamable HTTP: the message out, the answer back.
    ///
    /// `expected` is `Some(id)` for a request — whose answer is either one
    /// JSON document or an SSE stream ending in the response — and `None` for
    /// a notification, which the server acknowledges with `202 Accepted` and
    /// no message at all.
    async fn http_post(
        &self,
        http: &HttpSession,
        expected: Option<u64>,
        method: &str,
        message: &Value,
    ) -> Result<Value> {
        let mut request = http
            .client
            .post(&http.url)
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .header(
                reqwest::header::ACCEPT,
                "application/json, text/event-stream",
            )
            .body(serde_json::to_string(message)?);
        for (key, value) in &http.headers {
            request = request.header(key.as_str(), value.as_str());
        }
        if let Some(session) = http.session.lock().await.as_deref() {
            request = request.header("Mcp-Session-Id", session);
        }

        let response = tokio::time::timeout(self.timeout, request.send())
            .await
            .map_err(|_| {
                anyhow!(
                    "MCP server '{}' timed out after {:?} on {method}",
                    self.server,
                    self.timeout
                )
            })?
            .with_context(|| format!("posting to MCP server '{}' at {}", self.server, http.url))?;

        // The session the server assigned rides every later message.
        if let Some(value) = response
            .headers()
            .get("mcp-session-id")
            .and_then(|value| value.to_str().ok())
        {
            *http.session.lock().await = Some(value.to_string());
        }

        let status = response.status();
        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .unwrap_or("")
            .to_ascii_lowercase();
        let body = response
            .text()
            .await
            .with_context(|| format!("reading the body from MCP server '{}'", self.server))?;

        let Some(id) = expected else {
            // A notification has no answer: `202 Accepted` is the reply.
            if status.is_success() {
                return Ok(Value::Null);
            }
            return Err(anyhow!(
                "MCP server '{}' rejected {method}: HTTP {}: {}",
                self.server,
                status.as_u16(),
                body_tail(&body)
            ));
        };

        if content_type.contains("event-stream") {
            return sse_message(&body, id).ok_or_else(|| {
                anyhow!(
                    "MCP server '{}' streamed no answer to {method} (request {id})",
                    self.server
                )
            });
        }
        serde_json::from_str::<Value>(&body).map_err(|error| {
            anyhow!(
                "MCP server '{}' answered {method} (request {id}) with HTTP {}: {}: {}",
                self.server,
                status.as_u16(),
                error,
                body_tail(&body)
            )
        })
    }

    async fn write_line(&self, message: &Value) -> Result<()> {
        let Some(stdin) = &self.stdin else {
            return Err(anyhow!(
                "MCP server '{}' has no stdio to write to",
                self.server
            ));
        };
        let mut line = serde_json::to_string(message)?;
        line.push('\n');
        let mut stdin = stdin.lock().await;
        stdin
            .write_all(line.as_bytes())
            .await
            .with_context(|| format!("writing to MCP server '{}'", self.server))?;
        stdin.flush().await?;
        Ok(())
    }
}

impl Drop for McpClient {
    /// Stop the server when its last handle goes.
    ///
    /// Synchronous and best-effort: this runs while a host is on its way out,
    /// where awaiting a kill is not on offer — and depending on the child's own
    /// drop instead kept the process alive long after everything it had to say
    /// was printed. Unloading a pack reaches here too: its tools are the
    /// handles, so the last one going takes the server with it.
    fn drop(&mut self) {
        self.force_kill();
    }
}

/// Flatten an MCP content array: text parts joined, image parts kept as
/// attachments the loop can hand the model.
///
/// A bare (non-array) `result` passes through as text, the way a few servers
/// answer instead of the specified shape.
fn flatten_result(result: &Value) -> crate::tools::ToolOutput {
    let Some(items) = result.get("content").and_then(|c| c.as_array()) else {
        return crate::tools::ToolOutput::text(result.as_str().unwrap_or(""));
    };
    let mut texts: Vec<String> = Vec::with_capacity(items.len());
    let mut out = crate::tools::ToolOutput::default();
    for item in items {
        match item.get("type").and_then(|t| t.as_str()) {
            Some("text") => {
                if let Some(text) = item.get("text").and_then(|t| t.as_str()) {
                    texts.push(text.to_string());
                }
            }
            Some("image") => {
                // The MCP image part: already-base64 bytes plus their type.
                // Wrapped as a data URI directly — decoding only to re-encode
                // would be the same bytes with more work.
                let Some(data) = item.get("data").and_then(|d| d.as_str()) else {
                    continue;
                };
                let mime = item
                    .get("mimeType")
                    .and_then(|m| m.as_str())
                    .unwrap_or("image/png");
                out.images.push(
                    crate::message::ImageAttachment::url(format!("data:{mime};base64,{data}"))
                        .labelled(format!("image from MCP ({mime})")),
                );
            }
            Some(other) => texts.push(format!("[{other} content omitted]")),
            None => {}
        }
    }
    out.text = texts.join("\n");
    out
}

/// Make a string safe for use inside a tool name.
fn sanitize(value: &str) -> String {
    value
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '_' })
        .collect()
}

/// The answer to `id` inside an SSE body.
///
/// The stream may also carry server-initiated messages; the id is what makes
/// one of them *the* response rather than a side channel event.
fn sse_message(body: &str, id: u64) -> Option<Value> {
    sse_messages(body)
        .into_iter()
        .find(|message| message.get("id").and_then(|i| i.as_u64()) == Some(id))
}

/// Every JSON-RPC message an SSE body carries.
///
/// One event's `data:` lines hold one message (several join with newlines);
/// an event ends at the blank line, and a stream that ends without one still
/// flushes its last payload.
fn sse_messages(body: &str) -> Vec<Value> {
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
fn body_tail(body: &str) -> String {
    let trimmed = body.trim();
    const MAX: usize = 300;
    if trimmed.chars().count() <= MAX {
        if trimmed.is_empty() { "(empty body)".to_string() } else { trimmed.to_string() }
    } else {
        let cut: String = trimmed.chars().take(MAX).collect();
        format!("{cut}\u{2026}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The text-only view of a result: the pre-attachment behavior the test
    /// below pins. Production code reads images too (flatten_result), so this
    /// helper lives with the test that asks for the old contract.
    fn flatten_content(result: &Value) -> String {
        flatten_result(result).text
    }

    #[test]
    fn tool_names_are_namespaced_and_sanitized() {
        let spec = McpSpec::stdio("my server", "x", vec![]);
        assert_eq!(spec.id, "my server");
        assert_eq!(sanitize("my server"), "my_server");
        assert_eq!(sanitize("tools/read.file"), "tools_read_file");
    }

    #[test]
    fn content_is_flattened_to_text() {
        // Text parts join; a part type the model cannot render is named in the
        // text. Images are not — they leave as attachments (see the loopback
        // test below, which proves the attachment path end to end).
        let result = json!({
            "content": [
                { "type": "text", "text": "first" },
                { "type": "audio", "data": "..." },
                { "type": "text", "text": "second" }
            ]
        });
        let text = flatten_content(&result);
        assert!(text.contains("first"));
        assert!(text.contains("second"));
        assert!(text.contains("audio content omitted"), "{text}");

        // A bare string result is passed through.
        assert_eq!(flatten_content(&json!("plain")), "plain");
        // An empty result yields nothing.
        assert_eq!(flatten_content(&json!({})), "");
    }

    #[test]
    fn image_parts_leave_as_attachments() {
        let result = json!({
            "content": [
                { "type": "text", "text": "shot" },
                { "type": "image", "data": "aGVsbG8=", "mimeType": "image/png" }
            ]
        });
        let output = flatten_result(&result);
        assert_eq!(output.text, "shot");
        assert_eq!(output.images.len(), 1, "{:?}", output.images);
        assert_eq!(
            output.images[0].url,
            "data:image/png;base64,aGVsbG8=",
            "wrapped as a data URI without a re-encode"
        );
        // A malformed image part (no data) is skipped, not fatal.
        let broken = json!({ "content": [{ "type": "image", "mimeType": "image/png" }] });
        assert!(flatten_result(&broken).images.is_empty());
    }

    #[test]
    fn exposed_names_include_the_server() {
        let spec = McpSpec::stdio("github", "npx", vec![]);
        assert_eq!(spec.id, "github");
        // The naming rule is what keeps two servers from colliding.
        assert_eq!(format!("mcp__{}__{}", sanitize("github"), sanitize("create_issue")), "mcp__github__create_issue");
    }

    /// An SSE body answers the request whose id it carries; a server-initiated
    /// message in the same stream is not that answer.
    #[test]
    fn sse_answers_fly_by_request_id() {
        let body = "event: message\ndata: {\"jsonrpc\":\"2.0\",\"method\":\"notifications/foo\"}\n\n\
                    event: message\ndata: {\"jsonrpc\":\"2.0\",\"id\":7,\"result\":{\"ok\":1}}\n\n";
        let answer = sse_message(body, 7).expect("the id 7 answer");
        assert_eq!(answer["result"]["ok"], 1);
        assert!(sse_message(body, 8).is_none(), "no other id is answered");
        // Several data lines join into one message.
        let multiline = "data: {\"jsonrpc\":\"2.0\",\ndata: \"id\":9,\ndata: \"result\":{}}\n\n";
        assert!(sse_message(multiline, 9).is_some(), "multi-line data joins");
    }

    /// A minimal Streamable HTTP MCP server, in-process: `initialize` hands
    /// out a session, `tools/list` *insists* the session came back, and a
    /// tool call flattens like the stdio path does.
    #[tokio::test]
    async fn an_http_server_is_reached_over_post_with_its_session() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            loop {
                let Ok((mut stream, _)) = listener.accept().await else {
                    return;
                };
                tokio::spawn(async move {
                    let mut buf: Vec<u8> = Vec::new();
                    let mut chunk = [0u8; 4096];
                    // Headers first, then exactly Content-Length bytes of body.
                    let header_end = loop {
                        let Ok(n) = stream.read(&mut chunk).await else {
                            return;
                        };
                        if n == 0 {
                            return;
                        }
                        buf.extend_from_slice(&chunk[..n]);
                        if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                            break pos + 4;
                        }
                    };
                    let head = String::from_utf8_lossy(&buf[..header_end]).to_string();
                    let len: usize = head
                        .lines()
                        .find(|line| line.to_ascii_lowercase().starts_with("content-length:"))
                        .and_then(|line| line.split(':').nth(1))
                        .and_then(|value| value.trim().parse().ok())
                        .unwrap_or(0);
                    while buf.len() < header_end + len {
                        let Ok(n) = stream.read(&mut chunk).await else {
                            return;
                        };
                        if n == 0 {
                            return;
                        }
                        buf.extend_from_slice(&chunk[..n]);
                    }
                    let body = String::from_utf8_lossy(&buf[header_end..header_end + len]).to_string();

                    // A DELETE (session close) carries no body and no JSON.
                    if body.trim().is_empty() {
                        let _ = stream
                            .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                            .await;
                        return;
                    }
                    let message: Value = serde_json::from_str(&body).unwrap();
                    let id = message.get("id").cloned().unwrap_or(Value::Null);
                    let method = message
                        .get("method")
                        .and_then(|m| m.as_str())
                        .unwrap_or("")
                        .to_string();

                    let (payload, status, session) = match method.as_str() {
                        "initialize" => (
                            json!({
                                "jsonrpc": "2.0", "id": id,
                                "result": {
                                    "protocolVersion": PROTOCOL_VERSION,
                                    "capabilities": {},
                                    "serverInfo": { "name": "loopback", "version": "0" }
                                }
                            }),
                            "200 OK",
                            Some("mcp-session-1"),
                        ),
                        "notifications/initialized" => (Value::Null, "202 Accepted", None),
                        "tools/list" => {
                            // The contract under test: the session assigned at
                            // initialize travels on every later request.
                            if !head
                                .to_ascii_lowercase()
                                .contains("mcp-session-id: mcp-session-1")
                            {
                                (
                                    json!({
                                        "jsonrpc": "2.0", "id": id,
                                        "error": { "message": "missing Mcp-Session-Id" }
                                    }),
                                    "400 Bad Request",
                                    None,
                                )
                            } else {
                                (
                                    json!({
                                        "jsonrpc": "2.0", "id": id,
                                        "result": {
                                            "tools": [{
                                                "name": "ping",
                                                "description": "answers pong",
                                                "inputSchema": { "type": "object", "properties": {} }
                                            }]
                                        }
                                    }),
                                    "200 OK",
                                    None,
                                )
                            }
                        }
                        "tools/call" => (
                            json!({
                                "jsonrpc": "2.0", "id": id,
                                "result": {
                                    "content": [
                                        { "type": "text", "text": "pong" },
                                        { "type": "image", "data": "aGVsbG8=", "mimeType": "image/png" }
                                    ]
                                }
                            }),
                            "200 OK",
                            None,
                        ),
                        other => (
                            json!({
                                "jsonrpc": "2.0", "id": id,
                                "error": { "message": format!("unexpected {other}") }
                            }),
                            "400 Bad Request",
                            None,
                        ),
                    };

                    let payload = if payload.is_null() {
                        String::new()
                    } else {
                        payload.to_string()
                    };
                    let session = session
                        .map(|value| format!("Mcp-Session-Id: {value}\r\n"))
                        .unwrap_or_default();
                    let response = format!(
                        "HTTP/1.1 {status}\r\nContent-Type: application/json\r\n{session}Content-Length: {}\r\nConnection: close\r\n\r\n{payload}",
                        payload.len()
                    );
                    let _ = stream.write_all(response.as_bytes()).await;
                    let _ = stream.flush().await;
                });
            }
        });

        let spec = McpSpec::http("remote", format!("http://{addr}/mcp"), BTreeMap::new());
        let client = McpClient::connect(&spec).await.expect("handshake");
        assert_eq!(client.tools().len(), 1, "tools/list answered");
        assert_eq!(client.tools()[0].name, "ping");
        let output = client.call("ping", json!({})).await.expect("call");
        assert_eq!(output.text, "pong", "content flattens like the stdio path");
        // The image part survives as an attachment — the loop hands these to
        // the model; flattening them away is what made screenshots invisible.
        assert_eq!(output.images.len(), 1, "{:?}", output.images);
        assert!(
            output.images[0].url.starts_with("data:image/png;base64,aGVsbG8="),
            "{}",
            output.images[0].url
        );
        client.shutdown().await;
    }

    /// A child that would happily sit there for thirty seconds — the shape of
    /// the server that hung a window's quit.
    #[cfg(windows)]
    async fn stubborn_child() -> Child {
        tokio::process::Command::new("cmd")
            .args(["/c", "ping -n 30 127.0.0.1"])
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .spawn()
            .expect("spawning the stub server")
    }

    #[cfg(windows)]
    #[tokio::test]
    async fn a_bounded_close_kills_instead_of_waiting_out_the_server() {
        let client = McpClient::from_child_for_tests(stubborn_child().await, "stub");
        let started = std::time::Instant::now();
        let finished = client
            .shutdown_within(std::time::Duration::from_secs(30))
            .await;
        assert!(finished, "the kill path finishes");
        assert!(
            started.elapsed() < std::time::Duration::from_secs(5),
            "and quickly, not in thirty: {:?}",
            started.elapsed()
        );
    }

    #[cfg(windows)]
    #[tokio::test]
    async fn force_kill_ends_the_child_so_the_next_close_is_instant() {
        let client = McpClient::from_child_for_tests(stubborn_child().await, "stub");
        client.force_kill();
        tokio::time::sleep(std::time::Duration::from_millis(300)).await;
        let started = std::time::Instant::now();
        assert!(
            client
                .shutdown_within(std::time::Duration::from_secs(10))
                .await
        );
        assert!(
            started.elapsed() < std::time::Duration::from_secs(2),
            "already dead: {:?}",
            started.elapsed()
        );
    }
}
