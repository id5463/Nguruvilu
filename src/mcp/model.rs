//! MCP 的类型模型与连接注册表。
//!
//! 这一层只有数据与登记：一个 server 怎么声明（[`McpSpec`]）、它提供什么工具
//! （[`McpTool`]）、一个连接长什么样（[`McpClient`] 的字段）、请求 id 怎么分配
//! （[`IdSpace`]）、进程怎么收尾（[`KillReport`] / [`CloseReport`]）。
//! 真正发消息的逻辑在 [`super::client`]，解析响应的逻辑在 [`super::util`]。

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex as StdMutex, OnceLock, Weak};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::process::{Child, ChildStdin};
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
    /// Per-call timeout, when the declaration names one.
    ///
    /// `None` means "follow the process budget" (`mcp_timeout_secs`), and the
    /// number is then read **at each call** rather than captured at connect.
    /// That distinction is the whole point: the settings panel republishes the
    /// budget while a server is already running, and a deadline frozen at
    /// handshake time would ignore it — measured as "changing the MCP timeout in
    /// the panel does nothing".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_ms: Option<u64>,
    /// Endpoint for the `streamable-http` transport; `None` means stdio.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    /// Headers for `streamable-http` — `Authorization` and friends, sent on
    /// every request to the endpoint.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub headers: BTreeMap<String, String>,
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
            timeout_ms: None,
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
            timeout_ms: None,
            url: Some(url.into()),
            headers,
        }
    }
}

/// The deadline for one MCP call.
///
/// Read **at each call**, not captured at connect: `mcp_timeout_secs` is a
/// process budget that the settings panel republishes and that `ngu config set`
/// writes, and a deadline frozen during the handshake ignored both — which is
/// what "I change the MCP timeout and nothing happens" was. A declaration that
/// names its own number keeps it; everything else follows the process.
pub(crate) fn deadline_for(declared: Option<u64>) -> std::time::Duration {
    let ms = declared.unwrap_or_else(|| crate::tools::budgets().mcp_timeout_ms as u64);
    std::time::Duration::from_millis(ms.max(1_000))
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
    pub(crate) server: String,
    /// The child's stdin. `None` for an HTTP server, which has no child.
    pub(crate) stdin: Option<Arc<Mutex<ChildStdin>>>,
    pub(crate) pending: Arc<Mutex<Pending>>,
    /// The request ids of this connection: monotonic within it, and drawn from a
    /// slice of the id space no other connection ever uses.
    pub(crate) ids: IdSpace,
    /// The child's process id, captured at spawn, or 0 once it is no longer
    /// safely ours to signal. Kept because the last-resort kill must not depend
    /// on getting a turn with the child handle.
    pub(crate) pid: AtomicU32,
    /// The child itself. `None` for an HTTP server.
    pub(crate) child: Option<Mutex<Child>>,
    /// The HTTP wiring, present exactly when [`McpSpec::url`] was.
    pub(crate) http: Option<HttpSession>,
    pub(crate) tools: Vec<McpTool>,
    /// A declaration's own number, if it named one. `None` follows the process
    /// budget, which is read at each call — see [`Client::deadline`].
    pub(crate) declared_timeout_ms: Option<u64>,
}

/// The requests one connection is waiting on, and the answers that belonged to
/// no request at all.
#[derive(Default)]
pub(crate) struct Pending {
    waiters: HashMap<u64, oneshot::Sender<Value>>,
    /// Answers whose id no request of this connection was waiting for: a repeat
    /// of one already delivered, or a number this connection never spoke. Kept
    /// as a count rather than discarded silently, because a server that does it
    /// is the thing a "wrong answer" report has to be able to name.
    stray: u64,
}

impl Pending {
    /// The most requests one connection may have in flight at once.
    ///
    /// A server that stops answering would otherwise grow this table for as long
    /// as the conversation lasted. Refusing a request is visible; an unbounded
    /// table is not.
    const LIMIT: usize = 1_024;

    /// Register a waiter. `false` when the table is full.
    ///
    /// The ones whose caller has gone are dropped first: a cancelled turn or a
    /// closed window leaves a sender nobody will take, and those would otherwise
    /// count against the limit forever.
    pub(crate) fn register(&mut self, id: u64, sender: oneshot::Sender<Value>) -> bool {
        self.waiters.retain(|_, waiting| !waiting.is_closed());
        if self.waiters.len() >= Self::LIMIT {
            return false;
        }
        self.waiters.insert(id, sender);
        true
    }

    /// Stop waiting for one request. `false` means the answer got out first.
    pub(crate) fn forget(&mut self, id: u64) -> bool {
        self.waiters.remove(&id).is_some()
    }

    /// Hand one answer to the request that asked for it.
    ///
    /// `false` means nobody was waiting for that id — an answer delivered twice,
    /// or one to a request this table never held. It is counted, not accepted.
    pub(crate) fn deliver(&mut self, id: u64, message: Value) -> bool {
        match self.waiters.remove(&id) {
            Some(sender) => sender.send(message).is_ok(),
            None => {
                self.stray += 1;
                false
            }
        }
    }

    /// Fail every waiter with the same message: the pipe closed.
    pub(crate) fn fail_all(&mut self, reason: Value) {
        for (_, sender) in self.waiters.drain() {
            let _ = sender.send(reason.clone());
        }
    }
}

/// The request ids one connection speaks.
///
/// A plain counter starting at 1 would hand out the same numbers again every
/// time a pack is reloaded, so an answer still travelling from the connection
/// being replaced is indistinguishable from an answer to the connection that
/// replaced it — the request that "answered" with someone else's result. Each
/// connection therefore numbers its requests inside its own slice of the id
/// space: the connection's `tag` in the high half, a counter climbing from 1 in
/// the low half. Ids stay monotonic within a connection and are never reused by
/// another one for as long as this process lives, and [`IdSpace::owns`] is the
/// check the reader makes before delivering anything.
pub(crate) struct IdSpace {
    /// This connection's number, in the high half of every id it speaks.
    tag: u64,
    /// The low half: climbs by one per request.
    next: AtomicU64,
}

/// Connection numbers, handed out one per [`IdSpace`].
static CONNECTIONS: AtomicU64 = AtomicU64::new(1);

impl IdSpace {
    pub(crate) fn new() -> Self {
        // Masked into the high half: a process cannot make 2^32 connections, and
        // wrapping is preferable to failing the 2^32nd connect outright.
        let tag = CONNECTIONS.fetch_add(1, Ordering::Relaxed) & 0xffff_ffff;
        Self {
            tag,
            next: AtomicU64::new(1),
        }
    }

    /// The next request id of this connection.
    pub(crate) fn next(&self) -> u64 {
        let count = self.next.fetch_add(1, Ordering::Relaxed) & 0xffff_ffff;
        (self.tag << 32) | count
    }

    /// Whether `id` is one this connection could have spoken.
    ///
    /// The reader task does the same check on the hot path (it only holds the
    /// tag, not the space); this is the form the tests assert against.
    pub(crate) fn owns(&self, id: u64) -> bool {
        spoken_by(self.tag, id)
    }
}

/// Whether `id` carries `tag` in its high half, the way every id of that
/// connection does.
pub(crate) fn spoken_by(tag: u64, id: u64) -> bool {
    (id >> 32) == tag
}

/// What a best-effort kill managed to do.
///
/// [`McpClient::force_kill`] runs where nothing can await anything — an unload,
/// a drop at exit — so it has to say what it achieved. Returning `()` here (as
/// this did) is how a child process went on running with nobody noticing: the
/// handle was busy, nothing was signalled, and the caller had no way to tell
/// that apart from success.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KillReport {
    /// An HTTP server owns no child process; its session ends server-side.
    NoProcess,
    /// The child was signalled through its handle.
    Signalled,
    /// The handle stayed locked for the whole retry budget — a request being
    /// written, or a close already inside its own wait — so the process was
    /// stopped by the pid captured when it was spawned.
    SignalledByPid(u32),
    /// Neither route reached the process. The reason is reported, never
    /// swallowed.
    Blocked(String),
}

impl KillReport {
    /// Whether the server is on its way out (or had no process to stop).
    pub fn stopped(&self) -> bool {
        !matches!(self, KillReport::Blocked(_))
    }
}

impl std::fmt::Display for KillReport {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KillReport::NoProcess => write!(formatter, "no child process"),
            KillReport::Signalled => write!(formatter, "signalled"),
            KillReport::SignalledByPid(pid) => write!(formatter, "signalled by pid {pid}"),
            KillReport::Blocked(reason) => write!(formatter, "not stopped: {reason}"),
        }
    }
}

/// How many times a kill waits for a busy child handle before going around it.
pub(crate) const KILL_RETRIES: usize = 4;

/// The pause between those attempts. Bounded and short on purpose: this runs on
/// the way out of an unload, where waiting forever is worse than a pid kill.
pub(crate) const KILL_RETRY_PAUSE: Duration = Duration::from_millis(50);

/// Live clients by server id.
///
/// An unload reads the ledger and the kernel; it does not hold the handles that
/// started the servers — `load_pack` handed those to the caller, and every
/// registered tool holds one of its own. Without this table the only thing that
/// could stop a server on unload would be the last `Arc` going away, which is an
/// assumption about code elsewhere, and one that leaked a child process per
/// load/unload cycle.
pub(crate) fn live() -> &'static StdMutex<HashMap<String, Vec<Weak<McpClient>>>> {
    static LIVE: OnceLock<StdMutex<HashMap<String, Vec<Weak<McpClient>>>>> = OnceLock::new();
    LIVE.get_or_init(|| StdMutex::new(HashMap::new()))
}

pub(crate) fn live_lock() -> std::sync::MutexGuard<'static, HashMap<String, Vec<Weak<McpClient>>>> {
    live().lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Remember a live client under its server id, so [`close_server`] can reach its
/// process without being handed the handle.
pub fn track(client: &Arc<McpClient>) {
    let mut live = live_lock();
    let entry = live.entry(client.server().to_string()).or_default();
    entry.retain(|known| known.strong_count() > 0);
    entry.push(Arc::downgrade(client));
}

/// What closing one server id by name produced.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct CloseReport {
    /// Clients reached — stopped, already gone, or with no process at all.
    pub stopped: usize,
    /// Clients whose process could not be reached, each with the reason.
    pub failed: Vec<String>,
}

impl CloseReport {
    /// Whether the close had nothing to complain about.
    pub fn clean(&self) -> bool {
        self.failed.is_empty()
    }
}

/// Stop every live client of one server id, without awaiting.
///
/// This is what an unload calls. The id's entries are dropped either way, so a
/// second unload of the same pack finds nothing here and returns at once; a
/// server this could not reach is put back and reported, and keeps the `Drop`
/// path as a second chance like any other handle.
pub fn close_server(id: &str) -> CloseReport {
    let mut report = CloseReport::default();
    let targets: Vec<Arc<McpClient>> = live_lock()
        .remove(id)
        .unwrap_or_default()
        .into_iter()
        .filter_map(|known| known.upgrade())
        .collect();

    for client in targets {
        match client.force_kill() {
            KillReport::Blocked(reason) => {
                report.failed.push(format!("{}: {reason}", client.server()));
                // Put the handle back: a process this did not reach is worth
                // another try later, not something to forget it existed.
                live_lock()
                    .entry(client.server().to_string())
                    .or_default()
                    .push(Arc::downgrade(&client));
            }
            _ => report.stopped += 1,
        }
    }
    report
}

/// Stop a process by its id. The last resort when the child handle is locked.
///
/// Signal, not reap: whoever holds the [`Child`] still owns the wait.
#[cfg(windows)]
pub(crate) fn kill_by_pid(pid: u32) -> std::io::Result<()> {
    use std::os::windows::process::CommandExt;
    // No console window flashes up for a kill nobody asked to see.
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    let status = std::process::Command::new("taskkill")
        .args(["/PID", &pid.to_string(), "/T", "/F"])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .creation_flags(CREATE_NO_WINDOW)
        .status()?;
    if status.success() {
        Ok(())
    } else {
        Err(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("taskkill exited with {status}"),
        ))
    }
}

/// Stop a process by its id. The last resort when the child handle is locked.
#[cfg(unix)]
pub(crate) fn kill_by_pid(pid: u32) -> std::io::Result<()> {
    // SAFETY: signalling a process needs no memory access. `ESRCH` means the
    // process went on its own, which is the outcome being asked for.
    if unsafe { libc::kill(pid as libc::pid_t, libc::SIGKILL) } == 0 {
        return Ok(());
    }
    let error = std::io::Error::last_os_error();
    if error.raw_os_error() == Some(libc::ESRCH) {
        Ok(())
    } else {
        Err(error)
    }
}

/// Stop a process by its id, where this platform offers no way to do that.
#[cfg(not(any(windows, unix)))]
pub(crate) fn kill_by_pid(pid: u32) -> std::io::Result<()> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Other,
        format!("no way to signal process {pid} on this platform"),
    ))
}

/// One Streamable HTTP connection.
///
/// Every message is a `POST` to `url`; the server may answer with a single
/// JSON body or an SSE stream, and hands out the `Mcp-Session-Id` with the
/// `initialize` response, which travels on every later request.
pub(crate) struct HttpSession {
    pub(crate) client: reqwest::Client,
    pub(crate) url: String,
    pub(crate) headers: BTreeMap<String, String>,
    /// The assigned session id, once `initialize` returned one.
    pub(crate) session: Mutex<Option<String>>,
}
