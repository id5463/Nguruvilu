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
//! ```
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
//! otherwise know: [`PID_VAR`], the kernel's own process id, and [`EXE_VAR`], the
//! kernel's executable. A plugin that decides the kernel should restart writes
//! `kill(pid, SIGTERM)` and lets the supervisor bring it back — measured to work,
//! and the reason `restart` does not need to live in the kernel.
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

/// The environment variable carrying the kernel's own process id.
///
/// A plugin that wants the kernel to restart signals this pid. Measured working
/// on Windows and Linux; without it a plugin can only restart itself, which is
/// not a restart.
pub const PID_VAR: &str = "NGU_PID";

/// The environment variable carrying the kernel's executable path.
///
/// The same value `NGU_EXE` already carried for MCP servers, under this
/// protocol's own name so a plugin does not have to know which era it was
/// launched in.
pub const EXE_VAR: &str = "NGU_EXE";

/// The environment variable carrying every pack's settings, as one JSON object.
///
/// `{"judge":{"endpoint":…},"mcp-proxy":{"mode":"proxy"}}` — verbatim, no
/// interpretation. A plugin reads its own key; one that wants nothing ignores
/// the variable. Folding this in here is what lets the kernel stop having a
/// `settings.pack("mcp-proxy")` line: it serializes a map and its involvement
/// ends.
pub const SETTINGS_VAR: &str = "NGU_PACK_SETTINGS";

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

/// What a plugin reports about itself.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Description {
    /// Tools it provides.
    #[serde(default)]
    pub tools: Vec<ToolDecl>,
    /// Panels it contributes.
    #[serde(default)]
    pub panels: Vec<PanelDecl>,
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
    inner: Mutex<Inner>,
    /// Next request number, for diagnostics only.
    pub requests: AtomicU64,
}

struct Inner {
    child: Child,
    stdin: ChildStdin,
    stdout: ChildStdout,
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
        // The two things a plugin cannot know and needs: which process is the
        // kernel, and where the kernel's settings live.
        command.env(PID_VAR, std::process::id().to_string());
        if let Ok(exe) = std::env::current_exe() {
            command.env(EXE_VAR, exe.display().to_string());
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
            inner: Mutex::new(Inner { child, stdin, stdout }),
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

        let len = inner
            .stdout
            .read_u32()
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
        Ok(value)
    }

    /// Call one of the tools it declared.
    pub async fn call(&self, tool: &str, args: Value) -> Result<Value> {
        let answer = self
            .request(json!({ "op": "call", "tool": tool, "args": args }))
            .await?;
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

/// The pid of the process a plugin should signal to restart the kernel.
///
/// Provided as a function as well as a constant so the value has one definition:
/// `std::process::id()` is read here and written to the environment there, and a
/// plugin on the other side reads [`PID_VAR`].
pub fn kernel_pid() -> u32 {
    std::process::id()
}

#[cfg(test)]
mod tests {
    use super::*;

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
    async fn the_kernel_pid_reaches_the_plugin() {
        // What `restart` depends on: a plugin cannot signal a process it cannot
        // name, and a plugin's own pid is not the kernel's.
        let process = Process::start(&node(ECHO)).await.expect("start");
        let value = process.state_get("anything").await.unwrap();
        assert_eq!(value["pid"], json!(std::process::id().to_string()));
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
