//! The tool registry.
//!
//! The registry is the kernel's mutable tool table. Plugins contribute to it
//! through `register`; the four base tools are registered at boot.
//!
//! Name collisions fail loud by default. Silent override is the most
//! dangerous failure mode a plugin system can have: one plugin quietly
//! replacing another's safety check leaves no trace. A contribution that
//! genuinely intends to replace an existing tool says so explicitly.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use anyhow::{anyhow, Result};
use serde_json::{json, Value};

pub mod base;pub mod documents;
pub mod judge;
pub mod jobs;
pub mod longhorizon;
pub mod pack;pub mod readers;
pub mod search;
pub mod spill;
pub mod subagent;

/// What a tool produced.
///
/// Text is what the model reads. Images cannot travel in a tool message — the
/// OpenAI format restricts that content to text — so the loop collects them and
/// attaches them to a user message placed after the tool results.
#[derive(Debug, Clone, Default)]
pub struct ToolOutput {
    /// Text the model reads.
    pub text: String,
    /// Images the model should see.
    pub images: Vec<crate::message::ImageAttachment>,
}

impl ToolOutput {
    /// Text only.
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            images: Vec::new(),
        }
    }

    /// Attach an image.
    pub fn with_image(mut self, image: crate::message::ImageAttachment) -> Self {
        self.images.push(image);
        self
    }

    /// Whether this output carries anything for the model to look at.
    pub fn has_images(&self) -> bool {
        !self.images.is_empty()
    }
}

impl From<String> for ToolOutput {
    fn from(text: String) -> Self {
        Self::text(text)
    }
}

impl From<&str> for ToolOutput {
    fn from(text: &str) -> Self {
        Self::text(text)
    }
}

/// Boxed future returned by a tool handler.
pub type ToolFuture = Pin<Box<dyn Future<Output = Result<ToolOutput>> + Send>>;

/// A tool implementation.
pub type ToolHandler = Arc<dyn Fn(Value) -> ToolFuture + Send + Sync>;

/// What to do when a tool name is already registered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConflictPolicy {
    /// Refuse the registration and report both owners (default).
    #[default]
    Error,
    /// Replace the existing tool. The replaced owner is recorded.
    Override,
}

/// One registered tool.
#[derive(Clone)]
pub struct ToolDef {
    /// Name the model calls.
    pub name: String,
    /// Description shown to the model.
    pub description: String,
    /// JSON Schema for the arguments object.
    pub parameters: Value,
    /// Implementation.
    pub handler: ToolHandler,
    /// Name of the plugin that contributed this tool (`"kernel"` for base tools).
    pub owner: String,
}

impl ToolDef {
    /// Build a tool definition from a handler closure.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: Value,
        owner: impl Into<String>,
        handler: impl Fn(Value) -> ToolFuture + Send + Sync + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
            handler: Arc::new(handler),
            owner: owner.into(),
        }
    }

    /// The OpenAI `tools` entry for this tool.
    pub fn schema(&self) -> Value {
        json!({
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        })
    }
}

/// Record of one override, kept so conflicts stay inspectable.
#[derive(Debug, Clone)]
pub struct OverrideRecord {
    /// Tool name.
    pub name: String,
    /// Owner that was replaced.
    pub previous_owner: String,
    /// Owner that replaced it.
    pub new_owner: String,
}

/// Whose tool call is running: the session it belongs to and that session's
/// working directory.
///
/// This is the *per-call* identity. It travels with the call — the agent that
/// dispatches a turn builds one from the session the turn belongs to and hands
/// it to [`ToolRegistry::execute_as`], which pins it over every poll of the
/// handler.
///
/// It exists because one registry serves several conversations: the desktop's
/// single `AppState` shares one tool table across sessions, and two sessions
/// can run turns at the same time. A process-level "current session" that the
/// shell rewrites on every switch is read at whatever moment a handler polls,
/// so switching to B in the gap between two polls of A's handler silently
/// re-homes A's `bash` cwd, relative paths, and job attribution. Carrying the
/// identity in the call makes that impossible by construction.
///
/// `None` in either field keeps the old meaning: no session (jobs and packs
/// see no owner) and the process directory for relative paths.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ToolIdentity {
    /// The session the call belongs to.
    pub session: Option<String>,
    /// Working directory of that session; `None` is the process directory.
    pub workspace: Option<std::path::PathBuf>,
}

impl ToolIdentity {
    /// An identity from its parts.
    pub fn new(session: Option<String>, workspace: Option<std::path::PathBuf>) -> Self {
        Self { session, workspace }
    }

    /// The identity of one named session in its workspace.
    pub fn for_session(
        session: impl Into<String>,
        workspace: Option<std::path::PathBuf>,
    ) -> Self {
        Self {
            session: Some(session.into()),
            workspace,
        }
    }

    /// Whether this identity names nothing at all.
    pub fn is_empty(&self) -> bool {
        self.session.is_none() && self.workspace.is_none()
    }

    /// The identity pinned around the call running on this thread, if any.
    ///
    /// A tool that runs another agent (a subagent, a long-horizon mission)
    /// inherits the host turn's identity through this: the nested agent runs
    /// inside the parent's handler, which runs inside the parent's pin.
    pub fn ambient() -> Self {
        Self {
            session: current_session(),
            workspace: current_workspace(),
        }
    }

    /// [`ToolIdentity::ambient`], or `None` when nothing is pinned.
    pub fn ambient_if_set() -> Option<Self> {
        let ambient = Self::ambient();
        if ambient.is_empty() {
            None
        } else {
            Some(ambient)
        }
    }
}

// The session the current tool call belongs to, for the duration of one
// `execute_as`.
//
// A thread-local rather than a handler parameter because the tool handler
// signature carries only the arguments — threading a session id through it
// would change every tool for the sake of jobs and packs, which need it to tell
// one conversation's work from another's. Set by `execute_as` around each
// handler call, *from that call's own identity*, and restored after every poll;
// handlers read it via `current_session`.
//
// The working directory of the same session travels alongside it in
// `CURRENT_WORKSPACE`, so path resolution in handlers can default to it.
thread_local! {
    static CURRENT_SESSION: std::cell::RefCell<Option<String>> = const {
        std::cell::RefCell::new(None)
    };
    static CURRENT_WORKSPACE: std::cell::RefCell<Option<std::path::PathBuf>> = const {
        std::cell::RefCell::new(None)
    };
}

/// The session id of the tool call now running on this thread, if any.
///
/// Handlers that need to know *whose* conversation called them (job
/// attribution, session-scoped pack state) read this. Absent outside an
/// `execute`, or when the caller set no session.
pub fn current_session() -> Option<String> {
    CURRENT_SESSION.with(|cell| cell.borrow().clone())
}

/// The working directory of the session whose tool call is running here.
///
/// `None` outside an `execute`, or when the session has no workspace bound;
/// callers then use the process directory, which is the CLI behaviour.
pub fn current_workspace() -> Option<std::path::PathBuf> {
    CURRENT_WORKSPACE.with(|cell| cell.borrow().clone())
}

/// The tool deadline in milliseconds: both the default and the ceiling.
///
/// A process global rather than a field on the registry: the deadline belongs
/// to the deployment (how long this machine is willing to let one command hold
/// a turn), not to a conversation, and both hosts — CLI and desktop shell —
/// set it once from the same `settings.json` key.
///
/// It used to be only a default, and that turned out to be a hole a measured
/// session walked straight through: a model asking for `timeout_ms: 1500000`
/// was granted twenty-five minutes, because the argument was honoured whenever
/// it was merely positive. The same key now caps it. `background: true` is the
/// way out and is not capped — a job's lifetime belongs to `job_kill`, and a
/// subagent's or a pack tool's long work is a deliberate exception to any
/// per-call deadline.
static DEFAULT_TOOL_TIMEOUT_MS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(base::DEFAULT_BASH_TIMEOUT_MS);

/// Set the tool deadline, in milliseconds. Zero is ignored: a misconfigured
/// "0 seconds" would make every tool call instant.
pub fn set_default_tool_timeout_ms(ms: u64) {
    if ms > 0 {
        DEFAULT_TOOL_TIMEOUT_MS.store(ms, std::sync::atomic::Ordering::Relaxed);
    }
}

/// The tool deadline in milliseconds: default and ceiling at once.
pub fn default_tool_timeout_ms() -> u64 {
    DEFAULT_TOOL_TIMEOUT_MS.load(std::sync::atomic::Ordering::Relaxed)
}

/// The deadline one foreground call actually gets.
///
/// Returns the duration and, when the request was cut, what was asked for — so
/// the caller can say so out loud rather than silently running short. A model
/// that asks for less than the ceiling gets exactly what it asked for: the
/// ceiling is a limit on the deployment's patience, not a minimum.
///
/// `requested` of `None` or `0` means "no opinion": take the ceiling uncapped.
pub(crate) fn effective_timeout(requested_ms: Option<u64>, ceiling_ms: u64) -> (u64, Option<u64>) {
    let ceiling = if ceiling_ms == 0 {
        base::DEFAULT_BASH_TIMEOUT_MS
    } else {
        ceiling_ms
    };
    match requested_ms {
        Some(asked) if asked > 0 && asked > ceiling => (ceiling, Some(asked)),
        Some(asked) if asked > 0 => (asked, None),
        _ => (ceiling, None),
    }
}

/// Scopes a session id and its workspace over every poll of an inner future.
///
/// A thread-local set around `handler(args)` alone is empty by the time an
/// `async` body runs, because the body executes when it is first polled.
/// Wrapping the future keeps the scope in force for each poll and restores
/// the previous values after each one, so nested tool calls never leak — and,
/// because the values are copied into the wrapper before the first poll, a
/// registry default that changes mid-flight cannot reach the handler.
pub(crate) struct ScopedSession<F> {
    pub(crate) session: Option<String>,
    pub(crate) workspace: Option<std::path::PathBuf>,
    pub(crate) inner: F,
}

impl<F> ScopedSession<F> {
    /// Scope one call's identity over its handler.
    pub(crate) fn for_identity(identity: &ToolIdentity, inner: F) -> Self {
        Self {
            session: identity.session.clone(),
            workspace: identity.workspace.clone(),
            inner,
        }
    }
}

impl<F: Future + Unpin> Future for ScopedSession<F> {
    type Output = F::Output;

    fn poll(mut self: Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> std::task::Poll<Self::Output> {
        let this = &mut *self;
        let prev_session = CURRENT_SESSION.with(|cell| cell.replace(this.session.clone()));
        let prev_workspace = CURRENT_WORKSPACE.with(|cell| cell.replace(this.workspace.clone()));
        let polled = Pin::new(&mut this.inner).poll(cx);
        CURRENT_SESSION.with(|cell| *cell.borrow_mut() = prev_session);
        CURRENT_WORKSPACE.with(|cell| *cell.borrow_mut() = prev_workspace);
        polled
    }
}

/// The kernel's mutable tool table.
#[derive(Clone)]
pub struct ToolRegistry {
    tools: HashMap<String, ToolDef>,
    overrides: Vec<OverrideRecord>,
    /// The event bus every clone shares — the registry is where tool
    /// execution happens, so it is where the bus lives, and the kernel hands
    /// out clones of it (`kernel.events()`).
    events: crate::events::EventBus,
    /// The **fallback** identity, used only by [`ToolRegistry::execute`] — the
    /// legacy entry point with no identity argument.
    ///
    /// This was the whole mechanism before per-call identities: the desktop
    /// rewrote it on every session switch, and `execute` read it as each poll
    /// came in, so a switch landing between two polls of one handler moved that
    /// handler's cwd and job ownership to the other conversation. Nothing on
    /// the per-call path ([`ToolRegistry::execute_as`]) reads these any more;
    /// they survive so a caller that cannot supply an identity yet keeps
    /// working, and so [`ToolIdentity::ambient`] has a last resort.
    ///
    /// Do not build new behaviour on them: two conversations running at the
    /// same time cannot both be "the" caller.
    caller: std::sync::Arc<std::sync::RwLock<Option<String>>>,
    /// Working directory of the fallback identity; `None` means the process
    /// directory. Set together with `caller`.
    workspace: std::sync::Arc<std::sync::RwLock<Option<std::path::PathBuf>>>,
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self {
            tools: HashMap::new(),
            overrides: Vec::new(),
            events: crate::events::EventBus::default(),
            caller: std::sync::Arc::new(std::sync::RwLock::new(None)),
            workspace: std::sync::Arc::new(std::sync::RwLock::new(None)),
        }
    }
}

impl ToolRegistry {
    /// Set the **default** identity: the session and working directory that
    /// [`ToolRegistry::execute`] uses when a caller supplies none.
    ///
    /// Deprecated as a mechanism for naming the live conversation. It is a
    /// process-level value on a registry that every conversation shares, so
    /// rewriting it mid-turn is exactly the race that
    /// [`ToolRegistry::execute_as`] exists to close: it must not decide which
    /// session a caller's tool call belongs to. Prefer building a
    /// [`ToolIdentity`] from the turn's own session and passing it per call.
    ///
    /// Kept for shells that have not been converted yet, and for the paths that
    /// legitimately have one caller for the whole process (a one-shot CLI
    /// turn).
    pub fn set_session(&self, session: Option<String>, workspace: Option<std::path::PathBuf>) {
        *self.caller.write().expect("registry caller") = session;
        *self.workspace.write().expect("registry workspace") = workspace;
    }

    /// The fallback session, if any.
    ///
    /// Deprecated in the same sense as [`ToolRegistry::set_session`]: this is
    /// the default, not "the current session" — with two turns in flight there
    /// is no single answer.
    pub fn session(&self) -> Option<String> {
        self.caller.read().expect("registry caller").clone()
    }

    /// The fallback working directory, if any.
    ///
    /// Deprecated in the same sense as [`ToolRegistry::set_session`].
    pub fn workspace(&self) -> Option<std::path::PathBuf> {
        self.workspace.read().expect("registry workspace").clone()
    }

    /// The fallback identity, read in one go.
    pub fn default_identity(&self) -> ToolIdentity {
        ToolIdentity {
            session: self.caller.read().expect("registry caller").clone(),
            workspace: self.workspace.read().expect("registry workspace").clone(),
        }
    }
}


impl ToolRegistry {
    /// An empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// The event bus this registry's executions report to (shared by clones).
    pub fn events(&self) -> crate::events::EventBus {
        self.events.clone()
    }

    /// A registry with the four base tools registered.
    pub fn with_base_tools() -> Result<Self> {
        let mut registry = Self::new();
        base::register_all(&mut registry)?;
        Ok(registry)
    }

    /// Register a tool.
    ///
    /// Fails when the name is taken and the policy is [`ConflictPolicy::Error`].
    pub fn register(&mut self, def: ToolDef, policy: ConflictPolicy) -> Result<()> {
        if let Some(existing) = self.tools.get(&def.name) {
            match policy {
                ConflictPolicy::Error => {
                    return Err(anyhow!(
                        "tool name conflict: '{}' is already registered by '{}' (requested by '{}')",
                        def.name,
                        existing.owner,
                        def.owner
                    ));
                }
                ConflictPolicy::Override => {
                    self.overrides.push(OverrideRecord {
                        name: def.name.clone(),
                        previous_owner: existing.owner.clone(),
                        new_owner: def.owner.clone(),
                    });
                }
            }
        }
        self.tools.insert(def.name.clone(), def);
        Ok(())
    }

    /// Remove a tool, returning whether it existed.
    pub fn unregister(&mut self, name: &str) -> bool {
        self.tools.remove(name).is_some()
    }

    /// Remove every tool contributed by `owner`, returning how many went.
    ///
    /// Unloading a plugin must not leave its tools callable.
    pub fn retain_owner(&mut self, owner: &str) -> usize {
        let before = self.tools.len();
        self.tools.retain(|_, def| def.owner != owner);
        before - self.tools.len()
    }

    /// Look up a tool.
    pub fn get(&self, name: &str) -> Option<&ToolDef> {
        self.tools.get(name)
    }

    /// All tool names, sorted.
    pub fn names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.tools.keys().cloned().collect();
        names.sort();
        names
    }

    /// Number of registered tools.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Owner of a registered tool, if any.
    pub fn owner(&self, name: &str) -> Option<&str> {
        self.tools.get(name).map(|t| t.owner.as_str())
    }

    /// Every recorded override.
    pub fn overrides(&self) -> &[OverrideRecord] {
        &self.overrides
    }

    /// Tool schemas in a stable order, for the model request.
    pub fn schemas(&self) -> Vec<Value> {
        let mut entries: Vec<(&String, &ToolDef)> = self.tools.iter().collect();
        entries.sort_by(|a, b| a.0.cmp(b.0));
        entries.into_iter().map(|(_, def)| def.schema()).collect()
    }

    /// Execute a tool by name with raw JSON arguments, under a **caller-named**
    /// identity.
    ///
    /// The one funnel the per-call path goes through: the identity is copied
    /// into the scope before the handler is first polled, so every later poll
    /// of this handler sees this call's session and workspace no matter what
    /// the process-level defaults do in the meantime. Two sessions' turns can
    /// run concurrently through one registry without touching each other's cwd,
    /// relative paths, job ownership, or pack-scoped state.
    ///
    /// Also where `tool.start` / `tool.end` are emitted: the payload carries
    /// truncated arguments and results, because an event must not become a copy
    /// of a 2 MB file read.
    pub async fn execute_as(
        &self,
        identity: ToolIdentity,
        name: &str,
        arguments: &str,
    ) -> Result<ToolOutput> {
        let def = self
            .tools
            .get(name)
            .ok_or_else(|| anyhow!("unknown tool '{name}'"))?;

        let args: Value = if arguments.trim().is_empty() {
            json!({})
        } else {
            serde_json::from_str(arguments)
                .map_err(|e| anyhow!("invalid JSON arguments for '{name}': {e}"))?
        };

        self.events.emit(
            crate::events::TOOL_START,
            &json!({
                "tool": name,
                "arguments": crate::events::truncate(arguments, 4000),
                "session": identity.session.as_deref(),
            }),
        );
        let started = std::time::Instant::now();
        let result = ScopedSession::for_identity(&identity, (def.handler)(args)).await;
        let duration_ms = started.elapsed().as_millis() as u64;
        match &result {
            Ok(output) => self.events.emit(
                crate::events::TOOL_END,
                &json!({
                    "tool": name,
                    "ok": true,
                    "duration_ms": duration_ms,
                    "text": crate::events::truncate(&output.text, 4000),
                    "session": identity.session.as_deref(),
                }),
            ),
            Err(error) => self.events.emit(
                crate::events::TOOL_END,
                &json!({
                    "tool": name,
                    "ok": false,
                    "duration_ms": duration_ms,
                    "error": crate::events::truncate(&format!("{error:#}"), 1000),
                    "session": identity.session.as_deref(),
                }),
            ),
        }
        // Every result worth keeping is spilled before it enters the history, so
        // the pointer is in the transcript from the start: pruning may thin this
        // text out later, and the way back has to already be there.
        match result {
            Ok(mut output) => {
                if let Some(path) = spill::store(identity.session.as_deref(), name, &output.text) {
                    output.text.push_str(&spill::marker_line(&path));
                }
                // Inline images are the heavier leak: a screenshot costs
                // hundreds of kilobytes on every later step. Spilling the bytes
                // now is what makes dropping one from a request reversible.
                for image in output.images.iter_mut() {
                    if image.stored.is_none() {
                        if let Some(path) =
                            spill::store_image(identity.session.as_deref(), name, &image.url)
                        {
                            image.stored = Some(path.display().to_string());
                        }
                    }
                }
                Ok(output)
            }
            Err(error) => Err(error),
        }
    }

    /// Execute a tool by name with raw JSON arguments under the registry's
    /// **default** identity.
    ///
    /// Legacy entry point: it reads the process-level default (see
    /// [`ToolRegistry::set_session`]) instead of taking one from the caller, so
    /// it inherits that mechanism's race and is only correct when a process
    /// really has one conversation. Hosts that run turns for several sessions
    /// call [`ToolRegistry::execute_as`] with the turn's own identity;
    /// everything else — tests, embedding, and tools that execute tools — can
    /// stay here.
    pub async fn execute(&self, name: &str, arguments: &str) -> Result<ToolOutput> {
        let identity = self.default_identity();
        self.execute_as(identity, name, arguments).await
    }
}

#[cfg(test)]
mod freeze_tests {
    use super::*;
    use serde_json::json;
    use std::path::PathBuf;

    /// The whole point of the change: the settings key bounds what a model may
    /// ask for, instead of merely filling in what it forgot to ask for.
    #[test]
    fn the_tool_deadline_is_a_ceiling_and_not_only_a_default() {
        // No opinion, or a zero: the ceiling, and nothing to report.
        assert_eq!(effective_timeout(None, 120_000), (120_000, None));
        assert_eq!(effective_timeout(Some(0), 120_000), (120_000, None));
        // Under the ceiling, the request is honoured exactly — including at the
        // boundary, where it is not a cut and must not be reported as one.
        assert_eq!(effective_timeout(Some(800), 120_000), (800, None));
        assert_eq!(effective_timeout(Some(120_000), 120_000), (120_000, None));
        // Over the ceiling: cut, and the asked-for value comes back so the
        // caller can say it out loud. These are the real numbers measured on a
        // long server session, where the old code granted every one of them.
        assert_eq!(
            effective_timeout(Some(300_000), 120_000),
            (120_000, Some(300_000))
        );
        assert_eq!(
            effective_timeout(Some(600_000), 120_000),
            (120_000, Some(600_000))
        );
        assert_eq!(
            effective_timeout(Some(1_500_000), 120_000),
            (120_000, Some(1_500_000)),
            "twenty-five minutes must not be grantable by typing a bigger number"
        );
        // A nonsense ceiling of 0 falls back to the built-in, and a huge ask is
        // still cut against it.
        assert_eq!(
            effective_timeout(Some(9_999_999), 0),
            (base::DEFAULT_BASH_TIMEOUT_MS, Some(9_999_999))
        );
    }

    /// The table a turn copies at its start keeps a tool the live table drops.
    ///
    /// This is what stands between a background turn and a plugin released
    /// while it runs: the desktop lets go of a conversation's plugins when that
    /// conversation is left, and the turn still going has to finish with the
    /// tools it started with rather than fail on "unknown tool" halfway
    /// through its work.
    #[tokio::test]
    async fn a_copy_of_the_table_keeps_a_tool_the_live_table_drops() {
        let mut live = ToolRegistry::with_base_tools().unwrap();
        live.register(
            ToolDef::new(
                "frozen_probe",
                "a tool that only exists to be taken away",
                json!({ "type": "object", "properties": {} }),
                "test",
                |_| Box::pin(async { Ok(ToolOutput::text("still here")) }) as ToolFuture,
            ),
            ConflictPolicy::Error,
        )
        .unwrap();

        // The copy `Agent::run` takes at the start of a turn.
        let frozen = live.clone();
        assert!(
            live.unregister("frozen_probe"),
            "the live table has something to drop"
        );
        assert!(live.get("frozen_probe").is_none(), "the live table dropped it");
        assert!(
            frozen.get("frozen_probe").is_some(),
            "the turn's copy has to keep the tool it started with"
        );

        let output = frozen
            .execute_as(
                ToolIdentity::for_session("s1", Some(PathBuf::from("."))),
                "frozen_probe",
                "{}",
            )
            .await
            .expect("the frozen copy still runs it");
        assert_eq!(output.text, "still here");

        // And it is a copy, not a second view of one table: a tool registered
        // after the turn began must not become callable inside that turn.
        live.register(
            ToolDef::new(
                "late_probe",
                "registered after the turn began",
                json!({ "type": "object", "properties": {} }),
                "test",
                |_| Box::pin(async { Ok(ToolOutput::text("too late")) }) as ToolFuture,
            ),
            ConflictPolicy::Error,
        )
        .unwrap();
        assert!(
            frozen.get("late_probe").is_none(),
            "the turn's table is fixed"
        );
    }

    /// The funnel, not the helper: everything that later makes thinning reversible
    /// has to be attached where every tool result passes, before it enters a
    /// conversation.
    #[tokio::test]
    async fn a_large_result_leaves_the_funnel_with_a_pointer_to_its_full_text() {
        let mut registry = ToolRegistry::with_base_tools().unwrap();
        let body = "y".repeat(5_000);
        let expected = body.clone();
        registry.register(
            ToolDef::new(
                "big_text",
                "a tool that answers at length",
                json!({ "type": "object", "properties": {} }),
                "test",
                move |_| {
                    let body = expected.clone();
                    Box::pin(async move { Ok(ToolOutput::text(body)) }) as ToolFuture
                },
            ),
            ConflictPolicy::Error,
        )
        .unwrap();

        let session = format!("funnel-{}", std::process::id());
        let output = registry
            .execute_as(
                ToolIdentity::for_session(&session, Some(PathBuf::from("."))),
                "big_text",
                "{}",
            )
            .await
            .expect("the tool runs");

        let pointer = spill::marker_of(&output.text)
            .unwrap_or_else(|| panic!("the funnel left no pointer: {}", output.text));
        let path = pointer
            .trim()
            .strip_prefix("[full output: ")
            .and_then(|rest| rest.strip_suffix(']'))
            .expect("the pointer names a path");
        assert_eq!(
            std::fs::read_to_string(path).expect("the spilled file reads back"),
            body,
            "the file holds what the model no longer has to re-read"
        );
        // The text the model sees is the original plus one line — never a loss.
        assert!(output.text.starts_with(&body), "the head stays intact");

        let _ = std::fs::remove_dir_all(spill::root_dir().join(&session));
    }

    /// A screenshot is the heavier leak, so its bytes leave the funnel on disk too.
    #[tokio::test]
    async fn an_inline_image_leaves_the_funnel_with_a_stored_copy() {
        use base64::{engine::general_purpose, Engine as _};
        let mut registry = ToolRegistry::with_base_tools().unwrap();
        // A 1x1 PNG.
        let png: Vec<u8> = vec![
            0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44,
            0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x06, 0x00, 0x00, 0x00, 0x1f,
            0x15, 0xc4, 0x89, 0x00, 0x00, 0x00, 0x0a, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9c, 0x63, 0x00,
            0x01, 0x00, 0x00, 0x05, 0x00, 0x01, 0x0d, 0x0a, 0x2d, 0xb4, 0x00, 0x00, 0x00, 0x00, 0x49,
            0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82,
        ];
        let uri = format!("data:image/png;base64,{}", general_purpose::STANDARD.encode(&png));
        let for_tool = uri.clone();
        registry.register(
            ToolDef::new(
                "big_image",
                "a tool that answers with a screenshot",
                json!({ "type": "object", "properties": {} }),
                "test",
                move |_| {
                    let uri = for_tool.clone();
                    Box::pin(async move {
                        Ok(ToolOutput::text("captured").with_image(crate::message::ImageAttachment::url(uri)))
                    })
                        as ToolFuture
                },
            ),
            ConflictPolicy::Error,
        )
        .unwrap();

        let session = format!("funnel-img-{}", std::process::id());
        let output = registry
            .execute_as(
                ToolIdentity::for_session(&session, Some(PathBuf::from("."))),
                "big_image",
                "{}",
            )
            .await
            .expect("the tool runs");

        assert_eq!(output.images.len(), 1);
        let image = &output.images[0];
        assert_eq!(image.url, uri, "the model still sees the bytes it needs now");
        let stored = image
            .stored
            .as_ref()
            .unwrap_or_else(|| panic!("the funnel kept no copy: {:?}", image.stored));
        assert_eq!(std::fs::read(stored).expect("the copy reads back"), png);

        let _ = std::fs::remove_dir_all(spill::root_dir().join(&session));
    }
}
