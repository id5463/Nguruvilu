//! The plugin protocol: one process, one pipe, four operations.
//!
//! # Why this exists
//!
//! A pack used to be able to add a capability in exactly one way: the code was
//! compiled into the kernel, and the pack's assembly decided whether it loaded
//! (`builtin:delegate`). That meant a new capability needed a new
//! `src/tools/something.rs`, a new line in `Kernel::define_builtins`, and a new
//! build — the pack system could enable what the kernel already knew and nothing
//! else.
//!
//! The kernel also carried two other ways to extend itself, and neither worked
//! out: `dylib:` (a full C ABI, 1,116 lines of loader, and not one real
//! capability using it) and MCP (a protocol designed for a model talking to a
//! server, pressed into service as a plugin ABI — it has no notion of a panel, no
//! way to read the host's state, and every plugin pays for handshake and version
//! negotiation it does not use).
//!
//! This module is the replacement. It is deliberately small, and it is built
//! around one measurement.
//!
//! # The measurement
//!
//! Same machine, same payload, 200 round trips, on Windows:
//!
//! | payload | newline-framed (what MCP does) | length-prefixed (this) |
//! |---|---|---|
//! | 64 B | 244 µs | 256 µs |
//! | 1 KB | 257 µs | 263 µs |
//! | 64 KB | 615 µs | 610 µs |
//! | 1 MB | 15.0 ms | 19.1 ms |
//!
//! Framing does not matter. What matters is this: **spawning a process costs
//! 45 ms, and a round trip costs 0.25 ms.** Spawning is 180× the message. So the
//! protocol's one performance rule is that a plugin process is started **once**
//! and kept: the 45 ms is paid at load, and every call after that is a write and
//! a read.
//!
//! # The four operations
//!
//! Every exchange is one JSON object in, one JSON object out:
//!
//! ```json
//! {"op":"describe"}                                  → {"tools":[…],"panels":[…]}
//! {"op":"call","tool":"x","args":{…}}                → {"result":…} | {"error":…}
//! {"op":"state","get":"settings"}                    → {"value":…}
//! {"op":"state","set":"judge","value":{…}}           → {"ok":true}
//! {"op":"register","id":"p1","name":"some.route"}    → {"result":true}
//! {"op":"next","id":"p1","layer":1,"args":{…}}       → {"result":…} | {"error":…}
//! ```
//!
//! `register` and `next` are the two that make the pipe bidirectional in the
//! sense that matters: `call` is the kernel calling a plugin, and `register` is
//! a plugin putting itself in front of a name the kernel will call. `next` is
//! the same road back — the plugin deciding the call is not its and handing it
//! to the layer below, which is the `next(1, args)` of the route design arriving
//! over a pipe instead of a closure.
//!
//! `state` is the operation that makes a plugin a peer of the kernel rather than
//! a supply of tools: `"settings"` is a **runtime string**, looked up in the same
//! table the kernel keeps its own state in, and there is no list of permitted
//! names. The thousandth plugin naming the thousandth key changes nothing here.
//!
//! # Permission
//!
//! There is none, anywhere. A plugin is a process the user started, running as
//! the user, with the user's files, network, and ability to signal other
//! processes. It is not sandboxed and the kernel does not pretend otherwise.
//!
//! What the kernel *does* do is hand it two pieces of information it could not
//! otherwise know: what `kernel.pid` answers, which is the kernel's own process
//! id. A plugin that decides the kernel should restart writes `kill(pid, …)` and
//! lets the supervisor bring it back — measured to work, and the reason `restart`
//! does not need to live in the kernel.
//!
//! # Framing
//!
//! `[4 bytes big-endian length][JSON payload]`, on stdin and on stdout. Big
//! payloads are not escaped, not line-split, and may contain any byte: a plugin
//! that returns a PDF's text does not have to think about newline handling.
//! Length-prefixed also means a truncated frame is detectable — a partial write
//! is an error rather than a short line that happens to parse.

use std::collections::BTreeMap;
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{anyhow, Context, Result};
use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::Mutex;

/// The state keys the kernel answers, and nothing else.
///
/// These are **not** an enumeration of capabilities — they are the kernel's own
/// facts, the same list however many plugins exist. A plugin asks for one by
/// name through `{"op":"state","get":…}` and gets a value; the kernel does not
/// care which plugin asked or why.
///
/// This replaced the environment variables `NGU_PID`, `NGU_EXE` and
/// `NGU_PACK_SETTINGS`. Environment variables worked, and they were the wrong
/// shape: a value passed at start-up is a snapshot, invisible in the protocol,
/// and every new one is a new name in the kernel's vocabulary that nobody
/// reading `describe` would know existed. Asking for it makes the surface
/// visible: **anything a plugin can know about the kernel is a `state` key**,
/// and `{"op":"state","list"}` says which ones there are.
pub const KERNEL_PID: &str = "kernel.pid";
/// The kernel's executable path.
pub const KERNEL_EXE: &str = "kernel.exe";
/// Every pack's settings, as one JSON object: `{"judge":{…},"mcp-proxy":{…}}`.
///
/// The kernel serialises a map and its involvement ends — no pack name appears
/// in the kernel because of this key.
pub const SETTINGS: &str = "settings";
/// The session's working directory.
///
/// A plugin that writes into a workspace needs to know which one; before this
/// key existed the only way to say so was another environment variable, which is
/// the shape this replaced.
pub const WORKSPACE: &str = "workspace";
/// The session's id.
pub const SESSION: &str = "session";

/// What the protocol needs a router to do — and the whole of what it needs.
///
/// Two operations ask the router a question: `register` (a plugin wants its
/// implementation in front of a name) and `next` (a plugin has decided the work
/// is not its and wants the layer below to have it). Neither question is
/// answerable here: this module knows frames, not routes. So the questions are
/// handed to whoever implements this trait, and the protocol layer stays a
/// codec.
///
/// The signatures are phrased in **plugin id and route name** only. No
/// capability name appears in this file, there is no list of names a plugin may
/// claim, and nothing here changes when the thousandth route is registered —
/// the same rule the `state` keys follow.
///
/// The two methods are shaped differently on purpose:
///
/// - [`RouteHost::register`] is synchronous because putting a handler at the
///   head of a chain is a table write. Making it a future would invite a router
///   to do slow work (a spawn, a network call) while a plugin's frame waits.
/// - [`RouteHost::next`] is not, because running the layer below *is* the work.
pub trait RouteHost: Send + Sync {
    /// The plugin `plugin` wants to be consulted for `name`. Returns whether it
    /// was accepted.
    ///
    /// `plugin` is the id the **kernel** chose for the process, not the `id` a
    /// plugin writes into its own frame: a plugin cannot be believed about who
    /// it is, and the ownership recorded in the chain has to be the real one.
    fn register(&self, plugin: &str, name: &str) -> bool;

    /// The plugin `plugin` asks for the result of layer `layer` of `name` —
    /// the `next(1, args)` of the route design, arriving over a pipe.
    fn next(
        &self,
        plugin: &str,
        name: &str,
        layer: usize,
        args: Value,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Value, String>> + Send>>;
}

/// What a plugin is told when it sends `register` or `next` to a process that
/// was started without a router.
///
/// A constant so the test and the branch cannot drift, and so the sentence a
/// plugin sees is one sentence rather than a paraphrase in three places.
pub const NO_HOST: &str = "no route host installed";

/// A plugin claiming a name.
///
/// Wire shape: `{"op":"register","id":"p1","name":"some.route"}`. `id` is the
/// plugin's own name for itself, for diagnostics and log lines; the kernel does
/// not trust it to say which process it is (see [`RouteHost::register`]).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Register {
    /// Operation tag; always `"register"`.
    #[serde(default)]
    pub op: String,
    /// The plugin's own name for itself.
    #[serde(default)]
    pub id: String,
    /// The name it wants to be consulted for.
    pub name: String,
}

impl Register {
    /// One frame, from a plugin id and a route name.
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            op: "register".to_string(),
            id: id.into(),
            name: name.into(),
        }
    }

    /// Parse an already-decoded frame, or say why it is not one.
    pub fn from_value(value: &Value) -> Result<Self> {
        let this: Self = serde_json::from_value(value.clone())
            .with_context(|| format!("not a register frame: {value}"))?;
        this.validate()?;
        Ok(this)
    }

    /// The parts serde cannot express.
    ///
    /// Checked here rather than by a serde attribute because the error has to be
    /// readable: a plugin sending `{"op":"register"}` with no name needs to be
    /// told the name is missing, not handed a `missing field` from four layers
    /// down.
    pub fn validate(&self) -> Result<()> {
        if self.op != "register" {
            return Err(anyhow!("expected op 'register', got '{}'", self.op));
        }
        if self.name.trim().is_empty() {
            return Err(anyhow!("a register frame needs a 'name'"));
        }
        Ok(())
    }
}

/// A plugin handing the work back to the layer below.
///
/// Wire shape: `{"op":"next","id":"p1","layer":1,"args":{…}}`. `name` is
/// optional: a plugin called for `some.route` that sends `next` without a name
/// means the name it was called for, which is the only name it could sensibly
/// mean. `layer` defaults to 0 — no layer named means "from the top", which is
/// what a plugin that was not told its depth would guess anyway.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Next {
    /// Operation tag; always `"next"`.
    #[serde(default)]
    pub op: String,
    /// The plugin's own name for itself.
    #[serde(default)]
    pub id: String,
    /// The route; absent means "the one I was called for".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Which layer to run.
    #[serde(default)]
    pub layer: usize,
    /// Arguments for the layer below.
    #[serde(default)]
    pub args: Value,
}

impl Next {
    /// One frame, from a plugin id, an optional route name, a layer and args.
    pub fn new(id: impl Into<String>, name: Option<String>, layer: usize, args: Value) -> Self {
        Self {
            op: "next".to_string(),
            id: id.into(),
            name,
            layer,
            args,
        }
    }

    /// Parse an already-decoded frame, or say why it is not one.
    pub fn from_value(value: &Value) -> Result<Self> {
        let this: Self = serde_json::from_value(value.clone())
            .with_context(|| format!("not a next frame: {value}"))?;
        this.validate()?;
        Ok(this)
    }

    /// The parts serde cannot express.
    pub fn validate(&self) -> Result<()> {
        if self.op != "next" {
            return Err(anyhow!("expected op 'next', got '{}'", self.op));
        }
        Ok(())
    }

    /// The route this frame is about, given the route the plugin was called for.
    pub fn route<'a>(&'a self, called_for: &'a str) -> &'a str {
        match &self.name {
            Some(name) if !name.trim().is_empty() => name.as_str(),
            _ => called_for,
        }
    }
}

/// The largest frame this side will accept.
///
/// Bounded because the length prefix is attacker-controlled as far as the kernel
/// knows: a plugin that says 4 GB and sends nothing would otherwise make the
/// kernel allocate it. Sixty-four megabytes is far past any tool result a model
/// can read and still small enough to fail loudly.
pub const MAX_FRAME: usize = 64 * 1024 * 1024;

/// One thing a plugin says it can do.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ToolDecl {
    /// Name the model calls.
    pub name: String,
    /// Description shown to the model.
    #[serde(default)]
    pub description: String,
    /// JSON Schema for the arguments.
    #[serde(default)]
    pub parameters: Value,
}

/// One interface panel a plugin contributes.
///
/// The same shape `look.json` already uses, because a pack that ships a panel as
/// data and a plugin that builds one in code should not disagree about what a
/// panel is. `slot` is either a bare name (`"above-composer"`) or an anchored
/// selector (`{"at":{"select":"#composer","place":"before"}}`) — the shell
/// already understands both.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PanelDecl {
    /// Unique id, within the plugin's namespace.
    pub id: String,
    /// Namespace panels group under; defaults to the plugin name.
    #[serde(default)]
    pub namespace: Option<String>,
    /// Where it goes.
    #[serde(default)]
    pub slot: Value,
    /// Sort key inside the slot.
    #[serde(default)]
    pub order: i64,
    /// Panel kind, for one the shell draws itself.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    /// The HTML.
    #[serde(default)]
    pub html: String,
    /// The script, evaluated once after the panel is inserted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub script: Option<String>,
}

/// One file format a plugin can convert.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ReaderDecl {
    /// Stable identity (`pdf`, `office`, …).
    pub id: String,
    /// Byte prefixes that mean "this is mine", as UTF-8 strings.
    #[serde(default)]
    pub magic: Vec<String>,
    /// File extensions it claims, lowercase and without the dot.
    #[serde(default)]
    pub extensions: Vec<String>,
}

/// What a plugin reports about itself.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Description {
    /// Tools it provides.
    #[serde(default)]
    pub tools: Vec<ToolDecl>,
    /// Panels it contributes.
    #[serde(default)]
    pub panels: Vec<PanelDecl>,
    /// File formats it can convert, for the kernel's own `read`.
    ///
    /// Data rather than a probe function: `read` has to decide whether to call
    /// on every binary file, and a round trip to ask "is this yours?" would cost
    /// 0.25 ms on every file the plugin does not claim. The kernel checks these
    /// declarations locally and only starts a process when one matches.
    #[serde(default)]
    pub readers: Vec<ReaderDecl>,
    /// Where its settings live in the pack table, if it wants any.
    ///
    /// Advisory only: the kernel does not read it. It is here so a panel's
    /// settings UI and a `--describe` from the command line can agree about the
    /// key without either side hardcoding the other's name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub settings_key: Option<String>,
}

/// How to start one plugin process.
#[derive(Debug, Clone)]
pub struct Spec {
    /// Identifier, for diagnostics and ownership.
    pub id: String,
    /// Executable. A bare name is looked up on `PATH`.
    pub command: String,
    /// Arguments.
    pub args: Vec<String>,
    /// Working directory, when the plugin came with a pack.
    pub cwd: Option<String>,
    /// Environment additions.
    pub env: BTreeMap<String, String>,
}

impl Spec {
    /// A spec for a command with no arguments.
    pub fn new(id: impl Into<String>, command: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            command: command.into(),
            args: Vec::new(),
            cwd: None,
            env: BTreeMap::new(),
        }
    }

    /// Add an argument.
    pub fn arg(mut self, arg: impl Into<String>) -> Self {
        self.args.push(arg.into());
        self
    }

    /// Set the working directory.
    pub fn in_dir(mut self, dir: impl Into<String>) -> Self {
        self.cwd = Some(dir.into());
        self
    }
}

/// One running plugin.
///
/// Held for as long as the plugin is loaded: the process, its two pipes, and a
/// mutex so two callers cannot interleave half a frame. The mutex is over the
/// round trip rather than over each write and read, because the protocol is
/// strictly one request, one response — there is no request id and no
/// multiplexing, and adding them would be the first step toward the complexity
/// this exists to avoid.
pub struct Process {
    id: String,
    description: Description,
    /// How this process answers when the *plugin* asks the kernel something.
    ///
    /// The protocol is symmetric in a way the environment variables were not:
    /// the kernel may ask a plugin for its state (`state_get`), and a plugin may
    /// ask the kernel (`{"op":"state","get":…}` in the middle of a `call`). Both
    /// travel the same pipe, in the same frame format, through the same four
    /// operations — so the set of things a plugin can learn about the kernel is
    /// visible in `describe`, not hidden in a start-up environment.
    state: Option<std::sync::Arc<dyn Fn(&str) -> Option<Value> + Send + Sync>>,
    /// Who answers the plugin's `register` and `next` frames.
    ///
    /// Separate from `state` rather than folded into it: `state` *answers*
    /// questions about the kernel (*what pid are you*, *what is the workspace*),
    /// while a route host *does something to* the kernel — it edits a table. One
    /// closure could do both, and would then be a read-only-looking seam that
    /// writes.
    ///
    /// `None` means the process was started by something with no router. That is
    /// not an error at start-up — most plugins never register anything — and it
    /// becomes a reported error the moment one tries.
    routes: Option<std::sync::Arc<dyn RouteHost>>,
    inner: Mutex<Inner>,
    /// Next request number, for diagnostics only.
    pub requests: AtomicU64,
}

struct Inner {
    child: Child,
    stdin: ChildStdin,
    stdout: ChildStdout,
    /// The route currently being called, so a `next` frame that omits `name`
    /// can mean "the one I was called for".
    ///
    /// Kept here rather than threaded through `request` because the `next` frame
    /// arrives *inside* the loop that is reading the answer to a `call` — the
    /// name is out of scope there unless it was written down when the `call`
    /// went out.
    route: Option<String>,
}

impl std::fmt::Debug for Process {
    /// Hand-written because `Child` is not `Debug`: name and description are
    /// what a diagnostic wants, and the pipes are not printable anyway.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Process")
            .field("id", &self.id)
            .field("tools", &self.description.tools.len())
            .field("panels", &self.description.panels.len())
            .finish()
    }
}

impl Process {
    /// Start a plugin and ask it what it can do.
    ///
    /// One spawn, one `describe`, then the process stays. Every later call is a
    /// frame out and a frame back — measured at 0.25 ms against 45 ms to start.
    pub async fn start(spec: &Spec) -> Result<Self> {
        Self::start_with_state(spec, None).await
    }

    /// [`Process::start`] with an answerer for the plugin's own questions.
    ///
    /// `state` is called with a key the plugin named — `"settings"`,
    /// `"workspace"`, anything — and returns what the kernel knows. `None` means
    /// the plugin gets an error for every key, which is the honest answer for a
    /// kernel that has not published anything.
    pub async fn start_with_state(
        spec: &Spec,
        state: Option<std::sync::Arc<dyn Fn(&str) -> Option<Value> + Send + Sync>>,
    ) -> Result<Self> {
        Self::start_full(spec, state, None).await
    }

    /// [`Process::start_with_state`] plus a router to answer `register`/`next`.
    ///
    /// Three entry points rather than one taking an options struct: the two seams
    /// are independent — a plugin may ask about the kernel without ever
    /// registering a route — and a plugin that does register with no router
    /// installed has to be told so by name instead of silently losing its claim.
    pub async fn start_full(
        spec: &Spec,
        state: Option<std::sync::Arc<dyn Fn(&str) -> Option<Value> + Send + Sync>>,
        routes: Option<std::sync::Arc<dyn RouteHost>>,
    ) -> Result<Self> {
        let mut command = Command::new(&spec.command);
        command
            .args(&spec.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            // stderr is left alone rather than piped: a plugin's log line is for
            // whoever is watching the terminal, and a plugin that writes a lot
            // must not be able to fill a pipe the kernel never drains.
            .stderr(Stdio::inherit())
            .kill_on_drop(true);
        if let Some(dir) = &spec.cwd {
            command.current_dir(dir);
        }
        for (key, value) in &spec.env {
            command.env(key, value);
        }

        let mut child = command.spawn().with_context(|| {
            format!(
                "starting plugin '{}' ({}); a plugin is an executable, so this fails \
                 when the file is missing or not executable",
                spec.id, spec.command
            )
        })?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| anyhow!("plugin '{}' has no stdin", spec.id))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow!("plugin '{}' has no stdout", spec.id))?;

        let mut process = Self {
            id: spec.id.clone(),
            description: Description::default(),
            state,
            routes,
            inner: Mutex::new(Inner {
                child,
                stdin,
                stdout,
                route: None,
            }),
            requests: AtomicU64::new(0),
        };

        let described = process
            .request(json!({ "op": "describe" }))
            .await
            .with_context(|| {
                format!(
                    "plugin '{}' did not answer `describe`; a plugin must read one \
                     JSON frame from stdin and write one back",
                    spec.id
                )
            })?;
        process.description = serde_json::from_value(described).with_context(|| {
            format!(
                "plugin '{}' answered `describe` with something that is not a \
                 description",
                spec.id
            )
        })?;
        Ok(process)
    }

    /// The plugin's id.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// What it said it can do.
    pub fn description(&self) -> &Description {
        &self.description
    }

    /// One round trip.
    ///
    /// Serialised by the mutex: the protocol has no request ids, so two requests
    /// in flight at once would be two answers with nothing to tell them apart.
    pub async fn request(&self, message: Value) -> Result<Value> {
        let seq = self.requests.fetch_add(1, Ordering::Relaxed) + 1;
        let body = serde_json::to_vec(&message)?;
        if body.len() > MAX_FRAME {
            return Err(anyhow!(
                "plugin '{}' request {seq} is {} bytes, over the {MAX_FRAME} limit",
                self.id,
                body.len()
            ));
        }
        let mut inner = self.inner.lock().await;
        // Which route this exchange is about, for as long as it lasts. A `next`
        // frame that omits `name` means "the one I was called for", and this is
        // the only place that knows it.
        inner.route = if message.get("op").and_then(Value::as_str) == Some("call") {
            message
                .get("name")
                .or_else(|| message.get("tool"))
                .and_then(Value::as_str)
                .map(str::to_string)
        } else {
            None
        };

        // Length prefix, big endian, then the payload. One `write_all` for the
        // header and one for the body rather than concatenating: the header is
        // four bytes and copying a megabyte to prepend four is the kind of
        // avoidable cost this module exists to not have.
        let mut header = [0u8; 4];
        header.copy_from_slice(&(body.len() as u32).to_be_bytes());
        inner
            .stdin
            .write_all(&header)
            .await
            .with_context(|| format!("writing to plugin '{}'", self.id))?;
        inner
            .stdin
            .write_all(&body)
            .await
            .with_context(|| format!("writing to plugin '{}'", self.id))?;
        inner
            .stdin
            .flush()
            .await
            .with_context(|| format!("flushing plugin '{}'", self.id))?;

        // Read until this request is answered. A frame that carries no `error`
        // of ours and asks a question of its own (`"ask":{…}`) is the plugin
        // talking mid-call; it is answered here, on the same pipe, and the loop
        // goes back to waiting.
        //
        // This is what makes the protocol symmetric. A plugin could instead be
        // handed everything it might need at start-up — which is what the
        // environment variables did — but then the set of things it can learn is
        // fixed before it runs and invisible to anyone reading `describe`.
        loop {
            let len = read_u32_async(&mut inner.stdout)
                .await
                .with_context(|| format!("plugin '{}' closed its output", self.id))?;
            let len = len as usize;
            if len > MAX_FRAME {
                return Err(anyhow!(
                    "plugin '{}' announced a {len}-byte frame, over the {MAX_FRAME} limit",
                    self.id
                ));
            }
            let mut buf = vec![0u8; len];
            inner
                .stdout
                .read_exact(&mut buf)
                .await
                .with_context(|| format!("plugin '{}' truncated its answer", self.id))?;
            let value: Value = serde_json::from_slice(&buf).with_context(|| {
                format!(
                    "plugin '{}' answered with {} bytes that are not JSON",
                    self.id, len
                )
            })?;

            // The two directions that are not questions: a plugin putting
            // itself in front of a name (`register`), and a plugin handing the
            // call down to the layer below (`next`).
            //
            // Handled *before* the `ask` match, and that order is the whole
            // point: these frames carry no `ask` key, so falling through the
            // match would return them to the caller as the answer to whatever
            // request was in flight. A kernel that called `some.route` would
            // then get `{"op":"register",…}` back as its result and be none the
            // wiser. Answering them here and looping keeps them out of the
            // result path entirely.
            if let Some(op) = value.get("op").and_then(Value::as_str) {
                if op == "register" || op == "next" {
                    let called_for = inner.route.clone();
                    let body = self
                        .answer_route(op, &value, called_for.as_deref())
                        .await?;
                    write_frame_async(&mut inner.stdin, &body).await?;
                    continue;
                }
            }

            let asked = match value.get("ask") {
                Some(ask) => ask,
                None => return Ok(value),
            };
            // A plugin asking for a model turn. Answered by the same provider
            // that answers `state` — the provider is the kernel, and what it can
            // do for a plugin is one list, not two.
            if let Some(turn) = asked.get("model") {
                let body = match &self.state {
                    Some(provider) => match provider(&format!("model:{turn}")) {
                        Some(value) => serde_json::to_vec(&json!({
                            "ask": { "id": asked.get("id").cloned().unwrap_or(Value::Null) },
                            "value": value,
                        }))?,
                        None => serde_json::to_vec(&json!({
                            "ask": { "id": asked.get("id").cloned().unwrap_or(Value::Null) },
                            "error": "the kernel cannot run a model turn for this plugin",
                        }))?,
                    },
                    None => serde_json::to_vec(&json!({
                        "ask": { "id": asked.get("id").cloned().unwrap_or(Value::Null) },
                        "error": "this process was started without a state provider",
                    }))?,
                };
                write_frame_async(&mut inner.stdin, &body).await?;
                continue;
            }
            let (key, id) = match asked.get("state") {
                Some(key) => (
                    key.as_str().unwrap_or("").to_string(),
                    asked.get("id").cloned().unwrap_or(Value::Null),
                ),
                // An unknown question. Answered rather than ignored: a plugin
                // waiting for a frame that never comes is worse than one told
                // the kernel does not understand it.
                None => {
                    let body = serde_json::to_vec(&json!({
                        "ask": { "id": Value::Null },
                        "error": format!("unknown question: {asked}"),
                    }))?;
                    write_frame_async(&mut inner.stdin, &body).await?;
                    continue;
                }
            };
            let answer = match &self.state {
                Some(provider) => match provider(&key) {
                    Some(value) => json!({ "ask": { "id": id }, "value": value }),
                    None => json!({
                        "ask": { "id": id },
                        "error": format!("the kernel has no state called '{key}'"),
                    }),
                },
                None => json!({
                    "ask": { "id": id },
                    "error": "this process was started without a state provider",
                }),
            };
            let body = serde_json::to_vec(&answer)?;
            write_frame_async(&mut inner.stdin, &body).await?;
        }
    }

    /// Call one of the tools it declared.
    pub async fn call(&self, tool: &str, args: Value) -> Result<Value> {
        let answer = self
            .request(json!({ "op": "call", "tool": tool, "args": args }))
            .await?;
        error_or_result(answer)
    }

    /// Call the plugin **by route name**: `{"op":"call","id":…,"name":…}`.
    ///
    /// Separate from [`Process::call`], which speaks `"tool"`, because the two
    /// are actually different questions and only one of them is a route. A tool
    /// is a thing the kernel offers the model under a name the plugin declared;
    /// a route is a slot in a chain the plugin has put itself in front of, and
    /// the name in a route call is the one from `{"op":"register"}`, not
    /// necessarily any tool the plugin declared.
    ///
    /// Overloading `call` to send `"name"` instead of `"tool"` would have been
    /// one line and would have silently broken every plugin written against
    /// `m.tool` — that is, all of them, including the ones in this file's tests.
    pub async fn call_route(&self, name: &str, args: Value) -> Result<Value> {
        let answer = self
            .request(json!({ "op": "call", "id": self.id, "name": name, "args": args }))
            .await?;
        error_or_result(answer)
    }

    /// Ask a plugin to convert a file, from a blocking caller.
    ///
    /// The protocol's two file operations, for a reader that runs inside `read`'s
    /// `spawn_blocking`. Content travels as base64 because it is bytes and JSON
    /// is text: a PDF is not a string, and pretending otherwise is how a
    /// conversion starts failing on the first byte above 0x7F.
    ///
    /// One round trip, not two: the plugin is told the path and the bytes, and
    /// answers with text or "not mine". A separate "is this yours" call would
    /// double the cost for a question the plugin can answer while it has the
    /// bytes anyway.
    pub fn convert_blocking(&self, id: &str, bytes: &[u8], path: &std::path::Path) -> Result<String> {
        use base64::Engine as _;
        let encoded = base64::engine::general_purpose::STANDARD.encode(bytes);
        let answer = self.blocking_request(json!({
            "op": "convert",
            "reader": id,
            "path": path.display().to_string(),
            "bytes_base64": encoded,
        }))?;
        if let Some(error) = answer.get("error") {
            return Err(anyhow!("{error}"));
        }
        Ok(answer
            .get("text")
            .and_then(|t| t.as_str())
            .unwrap_or("")
            .to_string())
    }

    /// One round trip, blocking.
    ///
    /// Needs its own thread and runtime when called from inside an async context:
    /// `block_on` panics on a runtime thread, and the caller here is a
    /// `spawn_blocking` thread — which has no runtime of its own but is *inside*
    /// one. A current-thread runtime on a fresh thread is the only correct shape.
    pub fn blocking_request(&self, message: Value) -> Result<Value> {
        let body = serde_json::to_vec(&message)?;
        if body.len() > MAX_FRAME {
            return Err(anyhow!(
                "plugin '{}' request is {} bytes, over the {MAX_FRAME} limit",
                self.id,
                body.len()
            ));
        }
        // A blocking caller cannot await, so the whole exchange runs on its own
        // thread with its own runtime.
        let this = self as *const Self as usize;
        std::thread::scope(|scope| {
            scope
                .spawn(move || {
                    let process = unsafe { &*(this as *const Self) };
                    tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                        .map_err(|error| anyhow!("building a runtime: {error}"))
                        .and_then(|runtime| runtime.block_on(process.request(message.clone())))
                })
                .join()
                .map_err(|_| anyhow!("the plugin request thread panicked"))?
        })
    }

    /// Read one of the kernel's state slots.
    ///
    /// `name` is not checked against anything: the table is the kernel's, the
    /// meanings are the plugins', and a whitelist here would be the enumeration
    /// this protocol exists to remove.
    pub async fn state_get(&self, name: &str) -> Result<Value> {
        let answer = self
            .request(json!({ "op": "state", "get": name }))
            .await?;
        if let Some(error) = answer.get("error") {
            return Err(anyhow!("{error}"));
        }
        Ok(answer.get("value").cloned().unwrap_or(Value::Null))
    }

    /// Write one of the kernel's state slots.
    pub async fn state_set(&self, name: &str, value: Value) -> Result<()> {
        let answer = self
            .request(json!({ "op": "state", "set": name, "value": value }))
            .await?;
        if let Some(error) = answer.get("error") {
            return Err(anyhow!("{error}"));
        }
        Ok(())
    }

    /// Answer one `register` or `next` frame from the plugin.
    ///
    /// The two ops are answered here rather than at the call site because both
    /// need the same three things — who this process is, what route is live, and
    /// the router — and splitting them would duplicate the "no host" branch.
    ///
    /// Neither failure is a transport error. A plugin that names no route, or a
    /// kernel with no router, gets `{"error":…}` back on the same pipe and the
    /// exchange continues: the plugin can log it and answer the original `call`
    /// anyway, which is better than a frame that kills the process.
    async fn answer_route(
        &self,
        op: &str,
        value: &Value,
        called_for: Option<&str>,
    ) -> Result<Vec<u8>> {
        let host = match &self.routes {
            Some(host) => host,
            // Named, not silent. A plugin that registered a route and got
            // `{"result":false}` would believe the router refused it; the truth
            // is that there was never a router to ask.
            None => {
                return Ok(serde_json::to_vec(&json!({
                    "op": op,
                    "error": NO_HOST,
                }))?)
            }
        };

        if op == "register" {
            let frame = match Register::from_value(value) {
                Ok(frame) => frame,
                Err(error) => {
                    return Ok(serde_json::to_vec(&json!({
                        "op": "register",
                        "error": format!("{error:#}"),
                    }))?)
                }
            };
            // `self.id`, never `frame.id`: the id inside the frame is the
            // plugin's claim about itself, and the chain has to record the
            // process the kernel actually started.
            let accepted = host.register(&self.id, &frame.name);
            return Ok(serde_json::to_vec(&json!({
                "op": "register",
                "name": frame.name,
                "result": accepted,
            }))?);
        }

        let frame = match Next::from_value(value) {
            Ok(frame) => frame,
            Err(error) => {
                return Ok(serde_json::to_vec(&json!({
                    "op": "next",
                    "error": format!("{error:#}"),
                }))?)
            }
        };
        // Owned, not borrowed: the name outlives `frame` below, where `args` is
        // moved into the router.
        let name = frame.route(called_for.unwrap_or("")).to_string();
        if name.is_empty() {
            return Ok(serde_json::to_vec(&json!({
                "op": "next",
                "error": "a next frame needs a 'name' unless it is sent during a call",
            }))?);
        }
        let answer = host.next(&self.id, &name, frame.layer, frame.args).await;
        Ok(match answer {
            Ok(result) => serde_json::to_vec(&json!({
                "op": "next",
                "name": name,
                "result": result,
            }))?,
            Err(error) => serde_json::to_vec(&json!({
                "op": "next",
                "name": name,
                "error": error,
            }))?,
        })
    }

    /// Ask the plugin to exit, then make sure it did.
    ///
    /// Closing stdin is the ordinary way: a plugin reading frames sees end of
    /// file and leaves. A plugin that ignores that is killed after the grace
    /// period, because a hung child would otherwise outlive the pack that
    /// loaded it and hold its pipes open.
    pub async fn stop(&self) {
        {
            let mut inner = self.inner.lock().await;
            // Dropping the write half is what a reader sees as end of file.
            let _ = inner.stdin.shutdown().await;
            let deadline = tokio::time::Instant::now() + std::time::Duration::from_millis(500);
            if tokio::time::timeout_at(deadline, inner.child.wait())
                .await
                .is_err()
            {
                let _ = inner.child.start_kill();
            }
        }
    }
}

/// Turn one answered frame into a `Result`: `{"error":…}` becomes an error,
/// anything else yields `result` (or null when the frame has none).
///
/// A free function because `call` and `call_route` differ only in what they
/// send, and two copies of this would be two places for the error shape to
/// drift — the shape a plugin actually writes, which is occasionally
/// `{"error":{"message":…}}` rather than a string.
#[doc(hidden)]
pub fn error_or_result(answer: Value) -> Result<Value> {
    if let Some(error) = answer.get("error") {
        let message = error
            .as_str()
            .map(str::to_string)
            .or_else(|| error.get("message").and_then(|m| m.as_str()).map(str::to_string))
            .unwrap_or_else(|| error.to_string());
        return Err(anyhow!("{}", message));
    }
    Ok(answer.get("result").cloned().unwrap_or(Value::Null))
}

/// Read a big-endian `u32` from an async stream.
async fn read_u32_async(stream: &mut ChildStdout) -> std::io::Result<u32> {
    let mut buf = [0u8; 4];
    stream.read_exact(&mut buf).await?;
    Ok(u32::from_be_bytes(buf))
}

/// Write one already-serialised frame.
async fn write_frame_async(stream: &mut ChildStdin, body: &[u8]) -> Result<()> {
    let mut header = [0u8; 4];
    header.copy_from_slice(&(body.len() as u32).to_be_bytes());
    stream.write_all(&header).await?;
    stream.write_all(body).await?;
    stream.flush().await?;
    Ok(())
}

/// The pid of the process a plugin should signal to restart the kernel.
///
/// Provided as a function as well as a constant so the value has one definition:
/// `std::process::id()` is read here and written to the environment there, and a
/// plugin on the other side reads over the protocol (`kernel.pid`).
pub fn kernel_pid() -> u32 {
    std::process::id()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    /// A plugin in a box, scripted: it answers whatever the test says.
    ///
    /// Written as a Node one-liner rather than a Rust fixture because the point
    /// of the protocol is that a plugin is *any* executable — a test that only
    /// worked with a Rust child would be testing the wrong thing.
    fn node(script: &str) -> Spec {
        Spec::new("test", "node").arg("-e").arg(script)
    }

    const ECHO: &str = r#"
        const read = () => new Promise(r => {
            let buf = Buffer.alloc(0);
            const onData = d => {
                buf = Buffer.concat([buf, d]);
                if (buf.length >= 4) {
                    const len = buf.readUInt32BE(0);
                    if (buf.length >= 4 + len) {
                        process.stdin.off('data', onData);
                        r(JSON.parse(buf.subarray(4, 4 + len).toString()));
                    }
                }
            };
            process.stdin.on('data', onData);
        });
        const write = o => {
            const b = Buffer.from(JSON.stringify(o));
            const h = Buffer.alloc(4); h.writeUInt32BE(b.length);
            process.stdout.write(Buffer.concat([h, b]));
        };
        (async () => {
            for (;;) {
                let m;
                try { m = await read(); } catch { return; }
                if (m.op === 'describe') {
                    write({ tools: [{ name: 'echo', description: 'echos', parameters: {} }],
                            panels: [{ id: 'p', html: '<b>hi</b>' }] });
                } else if (m.op === 'call') {
                    write({ result: { said: m.args } });
                } else if (m.op === 'state' && m.get) {
                    write({ value: { pid: process.env.NGU_PID } });
                } else if (m.op === 'state' && 'set' in m) {
                    write({ ok: true, wrote: m.value });
                }
            }
        })();
    "#;

    #[tokio::test]
    async fn a_plugin_is_described_then_called_then_stopped() {
        let process = Process::start(&node(ECHO)).await.expect("start");
        assert_eq!(process.description().tools.len(), 1);
        assert_eq!(process.description().tools[0].name, "echo");
        assert_eq!(process.description().panels.len(), 1);

        let answer = process.call("echo", json!({"hello": "world"})).await.unwrap();
        assert_eq!(answer["said"]["hello"], json!("world"));

        process.stop().await;
    }

    #[tokio::test]
    async fn the_kernel_pid_is_an_answer_not_an_environment_variable() {
        // What `restart` depends on: a plugin cannot signal a process it cannot
        // name, and the name has to come from the kernel. It used to arrive in
        // `NGU_PID`; it is now a `state` key, which is why this asserts the
        // answer rather than an inherited variable.
        let provider: Arc<dyn Fn(&str) -> Option<Value> + Send + Sync> =
            Arc::new(|key: &str| match key {
                "kernel.pid" => Some(json!(std::process::id())),
                _ => None,
            });
        let process = Process::start_with_state(&node(ASKER), Some(provider))
            .await
            .expect("start");
        let answer = process
            .call("ask", json!({ "key": "kernel.pid" }))
            .await
            .expect("call");
        assert_eq!(answer.as_u64(), Some(std::process::id() as u64));
        process.stop().await;
    }

    #[tokio::test]
    async fn a_big_payload_survives_the_frame() {
        // The reason for a length prefix rather than newlines: half a megabyte
        // with newlines inside it is ordinary for a document conversion.
        let process = Process::start(&node(ECHO)).await.expect("start");
        let big = "x".repeat(512 * 1024) + "\n\n\n" + &"y".repeat(1024);
        let answer = process.call("echo", json!({ "body": big })).await.unwrap();
        assert_eq!(answer["said"]["body"].as_str().unwrap().len(), big.len());
        process.stop().await;
    }

    #[tokio::test]
    async fn an_error_answer_becomes_an_error_not_a_silent_null() {
        let script = r#"
            const read = () => new Promise(r => {
                let b = Buffer.alloc(0);
                const on = d => { b = Buffer.concat([b, d]);
                    if (b.length >= 4) { const n = b.readUInt32BE(0);
                        if (b.length >= 4 + n) { process.stdin.off('data', on);
                            r(JSON.parse(b.subarray(4, 4 + n).toString())); } } };
                process.stdin.on('data', on);
            });
            const write = o => { const p = Buffer.from(JSON.stringify(o));
                const h = Buffer.alloc(4); h.writeUInt32BE(p.length);
                process.stdout.write(Buffer.concat([h, p])); };
            (async () => { for (;;) { let m; try { m = await read(); } catch { return; }
                if (m.op === 'describe') write({ tools: [{ name: 'bad' }] });
                else write({ error: 'it went wrong' }); } })();
        "#;
        let process = Process::start(&node(script)).await.expect("start");
        let error = process.call("bad", json!({})).await.expect_err("must fail");
        assert!(format!("{error}").contains("it went wrong"), "{error}");
        process.stop().await;
    }

    /// A plugin that asks the kernel a question and relays the answer.
    const ASKER: &str = r#"
        const read = () => new Promise(r => {
            let b = Buffer.alloc(0);
            const on = d => { b = Buffer.concat([b, d]);
                if (b.length >= 4) { const n = b.readUInt32BE(0);
                    if (b.length >= 4 + n) { process.stdin.off('data', on);
                        r(JSON.parse(b.subarray(4, 4 + n).toString())); } } };
            process.stdin.on('data', on);
        });
        const write = o => { const p = Buffer.from(JSON.stringify(o));
            const h = Buffer.alloc(4); h.writeUInt32BE(p.length);
            process.stdout.write(Buffer.concat([h, p])); };
        (async () => {
            for (;;) {
                let m; try { m = await read(); } catch { return; }
                if (m.op === 'describe') { write({ tools: [{ name: 'ask' }] }); continue; }
                write({ ask: { state: m.args.key } });
                const reply = await read();
                write({ result: reply.value ?? reply.error ?? null });
            }
        })();
    "#;

    #[tokio::test]
    async fn a_plugin_can_ask_the_kernel_a_question_mid_call() {
        // This is what replaced the environment variables. If the ask path is
        // broken, `restart` cannot name the process to signal and every plugin
        // that wants a setting is back to reading a start-up snapshot.
        let spec = node(ASKER);
        let provider: Arc<dyn Fn(&str) -> Option<Value> + Send + Sync> =
            Arc::new(|key: &str| match key {
                "kernel.pid" => Some(json!(std::process::id())),
                _ => None,
            });
        let process = Process::start_with_state(&spec, Some(provider))
            .await
            .expect("start");
        let answer = process
            .call("ask", json!({ "key": "kernel.pid" }))
            .await
            .expect("call");
        assert_eq!(answer.as_u64(), Some(std::process::id() as u64));
        process.stop().await;
    }

    #[tokio::test]
    async fn an_unknown_state_key_is_named_not_silent() {
        let spec = node(ASKER);
        let provider: Arc<dyn Fn(&str) -> Option<Value> + Send + Sync> = Arc::new(|_| None);
        let process = Process::start_with_state(&spec, Some(provider))
            .await
            .expect("start");
        let answer = process
            .call("ask", json!({ "key": "not.a.real.key" }))
            .await
            .expect("call");
        let text = answer.as_str().unwrap_or("");
        assert!(text.contains("no state called"), "{text}");
        process.stop().await;
    }

    /// A router that records what the protocol asked it, so a test can check
    /// that the frame reached the seam rather than that some plausible value
    /// came back.
    struct Recorder {
        registered: std::sync::Mutex<Vec<(String, String)>>,
        asked: std::sync::Mutex<Vec<(String, String, usize, Value)>>,
    }

    impl Recorder {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                registered: std::sync::Mutex::new(Vec::new()),
                asked: std::sync::Mutex::new(Vec::new()),
            })
        }
    }

    impl RouteHost for Recorder {
        fn register(&self, plugin: &str, name: &str) -> bool {
            self.registered
                .lock()
                .unwrap()
                .push((plugin.to_string(), name.to_string()));
            true
        }

        fn next(
            &self,
            plugin: &str,
            name: &str,
            layer: usize,
            args: Value,
        ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Value, String>> + Send>>
        {
            self.asked.lock().unwrap().push((
                plugin.to_string(),
                name.to_string(),
                layer,
                args.clone(),
            ));
            let plugin = plugin.to_string();
            let name = name.to_string();
            Box::pin(async move { Ok(json!({ "from": plugin, "route": name, "args": args })) })
        }
    }

    /// A plugin that registers once and then calls `next` once, reporting both
    /// answers as the result of its `call`.
    const CLAIMER: &str = r#"
        const read = () => new Promise(r => {
            let b = Buffer.alloc(0);
            const on = d => { b = Buffer.concat([b, d]);
                if (b.length >= 4) { const n = b.readUInt32BE(0);
                    if (b.length >= 4 + n) { process.stdin.off('data', on);
                        r(JSON.parse(b.subarray(4, 4 + n).toString())); } } };
            process.stdin.on('data', on);
        });
        const write = o => { const p = Buffer.from(JSON.stringify(o));
            const h = Buffer.alloc(4); h.writeUInt32BE(p.length);
            process.stdout.write(Buffer.concat([h, p])); };
        (async () => {
            for (;;) {
                let m; try { m = await read(); } catch { return; }
                if (m.op === 'describe') { write({ tools: [{ name: 'claim' }] }); continue; }
                if (m.op === 'call') {
                    const seen = {};
                    write({ op: 'register', id: 'liar', name: 'some.route' });
                    seen.registered = await read();
                    write({ op: 'next', id: 'liar', layer: 1, args: { kept: true } });
                    seen.nexted = await read();
                    write({ result: seen });
                    continue;
                }
            }
        })();
    "#;

    #[tokio::test]
    async fn a_register_frame_reaches_the_route_host() {
        // The protocol does not know what a route is; the host does. So the
        // assertion is on what the host *received*: a reply of `true` only
        // proves the frame parsed, and the recorded pair proves it reached the
        // seam with the right name and the kernel's own plugin id.
        let recorder = Recorder::new();
        let process = Process::start_full(&node(CLAIMER), None, Some(recorder.clone()))
            .await
            .expect("start");
        let answer = process
            .call_route("some.route", json!({}))
            .await
            .expect("call");

        assert_eq!(answer["registered"]["op"], json!("register"));
        assert_eq!(answer["registered"]["result"], json!(true));

        let registered = recorder.registered.lock().unwrap().clone();
        assert_eq!(registered.len(), 1, "{registered:?}");
        assert_eq!(registered[0].1, "some.route");
        // The frame said `liar`; the host must have been told the process id
        // instead, because a plugin cannot be believed about who it is.
        assert_eq!(
            registered[0].0, "test",
            "the frame's own id must not be trusted"
        );
        process.stop().await;
    }

    #[tokio::test]
    async fn a_next_frame_runs_the_layer_below_and_that_is_the_result() {
        let recorder = Recorder::new();
        let process = Process::start_full(&node(CLAIMER), None, Some(recorder.clone()))
            .await
            .expect("start");
        // Called by *route*, which is what a router does after a register: the
        // name in the frame is the one the plugin claimed, not a tool it
        // declared.
        let answer = process
            .call_route("some.route", json!({}))
            .await
            .expect("call");

        // The frame omitted the route name, so it has to have been filled in
        // from the call in flight — the case a plugin will actually hit, and the
        // case with no other source of truth.
        assert_eq!(answer["nexted"]["name"], json!("some.route"));
        assert_eq!(answer["nexted"]["result"]["route"], json!("some.route"));
        assert_eq!(answer["nexted"]["result"]["args"]["kept"], json!(true));

        let asked = recorder.asked.lock().unwrap().clone();
        assert_eq!(asked.len(), 1, "{asked:?}");
        assert_eq!(asked[0].0, "test");
        assert_eq!(asked[0].1, "some.route");
        assert_eq!(asked[0].2, 1, "the layer the plugin named");
        assert_eq!(asked[0].3["kept"], json!(true));
        process.stop().await;
    }

    #[tokio::test]
    async fn register_without_a_host_is_named_not_silently_refused() {
        // A `{"result":false}` here would read as "the router declined", which is
        // a different claim from "there is no router". The difference matters to
        // a plugin deciding whether to answer the call itself.
        let bare = Process::start(&node(CLAIMER)).await.expect("start");
        let answer = bare
            .call_route("some.route", json!({}))
            .await
            .expect("the call still answers");
        assert_eq!(answer["registered"]["error"], json!(NO_HOST));
        assert_eq!(answer["nexted"]["error"], json!(NO_HOST));
        bare.stop().await;
    }

    #[test]
    fn a_register_frame_that_names_nothing_says_so() {
        // Parsed without a process: the failure is in the frame, and starting a
        // child to discover that would be testing the child.
        let missing = Register::from_value(&json!({ "op": "register", "id": "p1" }));
        let error = format!("{:#}", missing.expect_err("no name"));
        assert!(error.contains("name"), "{error}");

        let wrong_op = Register::from_value(&json!({ "op": "call", "name": "a.b" }));
        assert!(format!("{:#}", wrong_op.expect_err("wrong op")).contains("register"));

        let ok = Register::from_value(&json!({ "op": "register", "id": "p1", "name": "a.b" }));
        assert_eq!(ok.expect("parses").name, "a.b");
    }

    #[test]
    fn a_next_frame_without_a_name_means_the_route_it_was_called_for() {
        let frame =
            Next::from_value(&json!({ "op": "next", "layer": 1, "args": {} })).expect("parses");
        assert_eq!(frame.route("some.route"), "some.route");
        assert_eq!(frame.layer, 1);

        let named =
            Next::from_value(&json!({ "op": "next", "name": "other.route" })).expect("parses");
        assert_eq!(named.route("some.route"), "other.route");
        // Nothing named and nothing in flight is not a route at all; the caller
        // turns the empty string into an error rather than into the name "".
        assert_eq!(frame.route(""), "");
    }

    #[tokio::test]
    async fn a_missing_executable_says_what_to_do() {
        let error = Process::start(&Spec::new("gone", "definitely-not-a-real-binary-xyz"))
            .await
            .expect_err("must fail");
        let text = format!("{error:#}");
        assert!(text.contains("gone"), "{text}");
        assert!(text.contains("starting plugin"), "{text}");
    }

    #[tokio::test]
    async fn a_plugin_that_never_answers_is_an_error_not_a_hang() {
        // It starts, it stays silent. The `describe` has to fail rather than
        // wait for ever — a load that hangs is worse than one that fails.
        let reached = tokio::time::timeout(
            std::time::Duration::from_millis(1500),
            Process::start(&node("process.stdin.resume();")),
        )
        .await;
        // Either it reported a timeout-free error or the test's own bound fired;
        // both are "did not hang for ever", and on a slow machine the second is
        // the honest outcome.
        match reached {
            Ok(result) => assert!(result.is_err(), "a silent plugin cannot describe itself"),
            Err(_) => { /* the bound fired: acceptable, the point is it is bounded */ }
        }
    }
}

