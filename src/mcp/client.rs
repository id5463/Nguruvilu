//! MCP 客户端的行为：连接、握手、列工具、调工具、收尾。
//!
//! 两种传输走同一套消息：stdio 是一个子进程，由唯一的 reader task 独占
//! stdout，按请求 id 分发给等待者；Streamable HTTP 是一个 URL，每条消息一个
//! POST，`initialize` 拿到的 `Mcp-Session-Id` 带在之后每个请求上。

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::Command;
use tokio::sync::{oneshot, Mutex};

use super::model::{CloseReport, HttpSession, IdSpace, KillReport, McpClient, McpSpec, Pending};
use super::util::{body_tail, sanitize, sse_message, sse_messages};

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
        // UTF-8 in and out, always.
        //
        // A Windows Python child reads stdin with the console's code page — `gbk`
        // on a Chinese machine — and a request containing any non-ASCII byte
        // makes its `json.loads` throw, which exits the process and surfaces to
        // the caller as "MCP server closed the connection". The kernel writes
        // UTF-8 (JSON is UTF-8), so the two disagreed and every server written in
        // Python handled ASCII requests and died on the first Chinese one.
        //
        // Declared entries still win: a server that needs another encoding says
        // so, and this is a default rather than a rule.
        if !spec.env.contains_key("PYTHONIOENCODING") {
            command.env("PYTHONIOENCODING", "utf-8");
        }
        if !spec.env.contains_key("PYTHONUTF8") {
            command.env("PYTHONUTF8", "1");
        }
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

        let pending: Arc<Mutex<Pending>> = Arc::new(Mutex::new(Pending::default()));
        let ids = IdSpace::new();
        let tag = ids.tag;

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
                    let mut waiting = pending.lock().await;
                    if !spoken_by(tag, id) {
                        // An id this connection never spoke — an answer to the
                        // request of a connection this one replaced, or one the
                        // server made up. It goes to no waiter, and the count of
                        // such answers is what a report can point at.
                        waiting.stray += 1;
                        continue;
                    }
                    waiting.deliver(id, message);
                }
                // The pipe closed: fail every waiter rather than hang them.
                pending.lock().await.fail_all(json!({
                    "error": { "message": format!("MCP server '{server}' closed the connection") }
                }));
            });
        }

        let mut client = Self {
            server: spec.id.clone(),
            stdin: Some(Arc::new(Mutex::new(stdin))),
            pending,
            ids,
            // Captured before the child goes behind a lock: killing by pid is
            // the one route that does not need a turn with that lock.
            pid: AtomicU32::new(child.id().unwrap_or(0)),
            child: Some(Mutex::new(child)),
            http: None,
            tools: Vec::new(),
            declared_timeout_ms: spec.timeout_ms,
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
            pending: Arc::new(Mutex::new(Pending::default())),
            ids: IdSpace::new(),
            // No child, so no pid: there is nothing here to signal.
            pid: AtomicU32::new(0),
            child: None,
            http: Some(http),
            tools: Vec::new(),
            declared_timeout_ms: spec.timeout_ms,
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
        if child.wait().await.is_ok() {
            // Waited on: the pid belongs to the operating system again, so the
            // pid fallback in `force_kill` must never signal it.
            self.pid.store(0, Ordering::Relaxed);
        }
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

    /// How many answers this connection received that matched no request.
    ///
    /// Zero for a well-behaved server. Anything else means it replayed an id, or
    /// invented one — useful to know before blaming a slow answer.
    pub fn stray_answers(&self) -> u64 {
        match self.pending.try_lock() {
            Ok(pending) => pending.stray,
            // A request is in flight this instant; the count is a diagnostic,
            // not something to wait behind.
            Err(_) => 0,
        }
    }

    /// Best-effort synchronous kill: no await, never panics, and says what it
    /// managed.
    ///
    /// The close path calls this *first* so the servers are already on their way
    /// out before any waiting starts; the `Drop` path is the same call for
    /// handles nobody shut down explicitly, and an unload calls it for the
    /// servers its ledger entry named.
    ///
    /// The lock on the child is retried a bounded number of times, then stepped
    /// around: a call in flight or a close already inside its own wait holds it,
    /// and a teardown that quietly did nothing because the lock was busy is
    /// exactly the leak this path exists to prevent. The pid captured at spawn is
    /// the way in that needs no lock at all.
    pub fn force_kill(&self) -> KillReport {
        // An HTTP server has no process to kill; its session ends server-side
        // (see `shutdown`) or times out there. Nothing is leaked here.
        let Some(child) = &self.child else {
            return KillReport::NoProcess;
        };

        for attempt in 0..=KILL_RETRIES {
            if attempt > 0 {
                std::thread::sleep(KILL_RETRY_PAUSE);
            }
            if let Ok(mut child) = child.try_lock() {
                let _ = child.start_kill();
                return KillReport::Signalled;
            }
        }

        // Still locked. The handle cannot be reached, but the process can: the
        // pid captured at spawn is still this child's — a process nobody has
        // waited on keeps its number, and a held lock means exactly that.
        match self.pid.load(Ordering::Relaxed) {
            0 => KillReport::Blocked(
                "the child handle stayed locked and the process id was already released"
                    .to_string(),
            ),
            pid => match kill_by_pid(pid) {
                Ok(()) => KillReport::SignalledByPid(pid),
                Err(error) => KillReport::Blocked(format!(
                    "the child handle stayed locked and stopping pid {pid} failed: {error}"
                )),
            },
        }
    }

    /// A client over an already-running child, without the handshake.
    ///
    /// Tests only: the real path is [`McpClient::connect`], which must
    /// initialize and discover tools before anything may call the server.

    async fn request(&self, method: &str, params: Value) -> Result<Value> {
        let id = self.ids.next();
        let message = json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params });
        if let Some(http) = &self.http {
            return self.http_post(http, Some(id), method, &message).await;
        }

        let (sender, receiver) = oneshot::channel();
        if !self.pending.lock().await.register(id, sender) {
            return Err(anyhow!(
                "MCP server '{}' has {} requests already waiting and takes no more",
                self.server,
                Pending::LIMIT
            ));
        }

        if let Err(error) = self.write_line(&message).await {
            // Nothing was sent, so nothing is coming: the waiter goes with the
            // request rather than sitting in the table until the timeout.
            self.pending.lock().await.forget(id);
            return Err(error);
        }

        match tokio::time::timeout(deadline_for(self.declared_timeout_ms), receiver).await {
            Ok(Ok(response)) => Ok(response),
            Ok(Err(_)) => {
                self.pending.lock().await.forget(id);
                Err(anyhow!("MCP server '{}' dropped request {id}", self.server))
            }
            Err(_) => {
                // If the entry is already gone, the answer arrived just as the
                // deadline did and went with the dropped receiver — worth
                // saying, because it is not the same as a silent server.
                let waiting = self.pending.lock().await.forget(id);
                let late = if waiting {
                    ""
                } else {
                    "; its answer arrived too late to be taken"
                };
                Err(anyhow!(
                    "MCP server '{}' timed out after {:?} on {method}{late}",
                    self.server,
                    deadline_for(self.declared_timeout_ms)
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

        let response = tokio::time::timeout(deadline_for(self.declared_timeout_ms), request.send())
            .await
            .map_err(|_| {
                anyhow!(
                    "MCP server '{}' timed out after {:?} on {method}",
                    self.server,
                    deadline_for(self.declared_timeout_ms)
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
