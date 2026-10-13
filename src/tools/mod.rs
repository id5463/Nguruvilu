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
pub mod pack;pub mod readers;
pub mod search;
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

/// The session the current tool call belongs to, for the duration of one
/// `execute`.
///
/// A thread-local rather than a parameter because the tool handler signature
/// carries only the arguments — threading a session id through it would change
/// every tool for the sake of jobs and packs, which need it to tell one
/// conversation's work from another's. Set by `execute` around each handler
/// call; handlers read it via `current_session`.
///
/// The working directory of the same session travels alongside it in
/// `CURRENT_WORKSPACE`, so path resolution in handlers can default to it.
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

/// Scopes a session id and its workspace over every poll of an inner future.
///
/// A thread-local set around `handler(args)` alone is empty by the time an
/// `async` body runs, because the body executes when it is first polled.
/// Wrapping the future keeps the scope in force for each poll and restores
/// the previous values after each one, so nested tool calls never leak.
pub(crate) struct ScopedSession<F> {
    pub(crate) session: Option<String>,
    pub(crate) workspace: Option<std::path::PathBuf>,
    pub(crate) inner: F,
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
    /// The session whose turn calls into this table.
    ///
    /// The desktop swaps sessions in place (one `AppState`, one registry), so
    /// the id moves here rather than living in the session struct; `execute`
    /// reads it and pins the thread-local for the duration of each handler.
    caller: std::sync::Arc<std::sync::RwLock<Option<String>>>,
    /// Working directory of the pinned session; `None` means the process
    /// directory. Set together with `caller` by [`ToolRegistry::set_session`].
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
    /// Pin the session whose turn will call into this table, with its
    /// working directory.
    ///
    /// Called when the desktop switches conversations: from then on every
    /// `execute` through this registry is attributed to `session`, and
    /// path-taking tools without an explicit directory resolve under
    /// `workspace`.
    pub fn set_session(&self, session: Option<String>, workspace: Option<std::path::PathBuf>) {
        *self.caller.write().expect("registry caller") = session;
        *self.workspace.write().expect("registry workspace") = workspace;
    }

    /// The session currently pinned to this table, if any.
    pub fn session(&self) -> Option<String> {
        self.caller.read().expect("registry caller").clone()
    }

    /// The working directory currently pinned to this table, if any.
    pub fn workspace(&self) -> Option<std::path::PathBuf> {
        self.workspace.read().expect("registry workspace").clone()
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

    /// Execute a tool by name with raw JSON arguments.
    ///
    /// The single funnel every caller goes through (the agent loop, the pack
    /// tool, delegates), which makes it where `tool.start` / `tool.end` are
    /// emitted: the payload carries truncated arguments and results, because
    /// an event must not become a copy of a 2 MB file read.
    pub async fn execute(&self, name: &str, arguments: &str) -> Result<ToolOutput> {
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
            }),
        );
        let started = std::time::Instant::now();
        let caller = self.caller.read().expect("registry caller").clone();
        let workspace = self.workspace.read().expect("registry workspace").clone();
        let result = ScopedSession {
            session: caller,
            workspace,
            inner: (def.handler)(args),
        }
        .await;
        let duration_ms = started.elapsed().as_millis() as u64;
        match &result {
            Ok(output) => self.events.emit(
                crate::events::TOOL_END,
                &json!({
                    "tool": name,
                    "ok": true,
                    "duration_ms": duration_ms,
                    "text": crate::events::truncate(&output.text, 4000),
                }),
            ),
            Err(error) => self.events.emit(
                crate::events::TOOL_END,
                &json!({
                    "tool": name,
                    "ok": false,
                    "duration_ms": duration_ms,
                    "error": crate::events::truncate(&format!("{error:#}"), 1000),
                }),
            ),
        }
        result
    }
}
