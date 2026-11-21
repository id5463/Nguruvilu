//! The runtime budgets, and the session scope a tool call runs inside.
//!
//! Two process globals and one thread-local, together because they answer one
//! question: how long may a call take, and whose call is it. Both hosts set the
//! budgets from `settings.json`; the thread-local carries the identity of the
//! call being polled, which is the only way two conversations sharing one
//! registry stay apart.

use std::future::Future;
use std::pin::Pin;

use super::base;
use super::registry::ToolIdentity;


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

/// The runtime budgets, none of which is a constant any more.
///
/// Every one of these used to be a `const` in the file that used it — twelve
/// steps, fifteen minutes, thirty seconds, sixty seconds, two levels deep. A
/// number that only a rebuild can change is not a setting: the deployment, the
/// panel, and in turn the model could not move any of them, and the ones that
/// bit hardest (the subagent's wall clock, the MCP deadline) were the ones
/// nobody could see. They now come from `settings.json`, are editable in the
/// settings panel, and are *named in the answer* whenever one of them cuts
/// something short — so hitting a budget teaches which knob moves it.
///
/// `Budgets` is a process global, like the tool deadline already was: a budget
/// belongs to the machine, not to a conversation, and both hosts set it from the
/// same file — at boot, and again on every panel save.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Budgets {
    /// How long one foreground `bash` call may hold the turn.
    pub tool_deadline_ms: u64,
    /// How long one MCP request may take.
    pub mcp_timeout_ms: u64,
    /// Model rounds a subagent may take.
    pub subagent_max_steps: usize,
    /// Wall clock a subagent may run.
    pub subagent_wall_ms: u64,
    /// How deep subagents may nest.
    pub subagent_max_depth: usize,
    /// The largest `wait_ms` a job read may block for.
    pub job_wait_max_ms: u64,
}

impl Default for Budgets {
    /// The shipped numbers are the old constants, unchanged: turning a hardcoded
    /// value into a setting must not silently change behaviour.
    fn default() -> Self {
        Self {
            tool_deadline_ms: base::DEFAULT_BASH_TIMEOUT_MS,
            mcp_timeout_ms: 60_000,
            subagent_max_steps: 12,
            subagent_wall_ms: 15 * 60_000,
            subagent_max_depth: 2,
            job_wait_max_ms: 30_000,
        }
    }
}

/// The budgets in force. A zero anywhere reads as "the shipped value": a
/// half-written settings file must not hand out instant deadlines.
pub fn budgets() -> Budgets {
    let stored = BUDGETS.load(std::sync::atomic::Ordering::Relaxed);
    let fallback = Budgets::default();
    let pick = |value: u64, default: u64| if value == 0 { default } else { value };
    Budgets {
        tool_deadline_ms: pick(stored.tool_deadline_ms, fallback.tool_deadline_ms),
        mcp_timeout_ms: pick(stored.mcp_timeout_ms, fallback.mcp_timeout_ms),
        subagent_max_steps: if stored.subagent_max_steps == 0 {
            fallback.subagent_max_steps
        } else {
            stored.subagent_max_steps
        },
        subagent_wall_ms: pick(stored.subagent_wall_ms, fallback.subagent_wall_ms),
        subagent_max_depth: if stored.subagent_max_depth == 0 {
            fallback.subagent_max_depth
        } else {
            stored.subagent_max_depth
        },
        job_wait_max_ms: pick(stored.job_wait_max_ms, fallback.job_wait_max_ms),
    }
}

/// A packed form of [`Budgets`], stored in six atomics so reading one needs no
/// lock in the middle of a tool call.
#[derive(Debug)]
struct PackedBudgets {
    tool_deadline_ms: std::sync::atomic::AtomicU64,
    mcp_timeout_ms: std::sync::atomic::AtomicU64,
    subagent_max_steps: std::sync::atomic::AtomicU64,
    subagent_wall_ms: std::sync::atomic::AtomicU64,
    subagent_max_depth: std::sync::atomic::AtomicU64,
    job_wait_max_ms: std::sync::atomic::AtomicU64,
}

impl PackedBudgets {
    fn store(&self, budgets: &Budgets) {
        use std::sync::atomic::Ordering::Relaxed;
        self.tool_deadline_ms.store(budgets.tool_deadline_ms, Relaxed);
        self.mcp_timeout_ms.store(budgets.mcp_timeout_ms, Relaxed);
        self.subagent_max_steps.store(budgets.subagent_max_steps as u64, Relaxed);
        self.subagent_wall_ms.store(budgets.subagent_wall_ms, Relaxed);
        self.subagent_max_depth.store(budgets.subagent_max_depth as u64, Relaxed);
        self.job_wait_max_ms.store(budgets.job_wait_max_ms, Relaxed);
    }

    fn load(&self, _ordering: std::sync::atomic::Ordering) -> Budgets {
        use std::sync::atomic::Ordering::Relaxed;
        Budgets {
            tool_deadline_ms: self.tool_deadline_ms.load(Relaxed),
            mcp_timeout_ms: self.mcp_timeout_ms.load(Relaxed),
            subagent_max_steps: self.subagent_max_steps.load(Relaxed) as usize,
            subagent_wall_ms: self.subagent_wall_ms.load(Relaxed),
            subagent_max_depth: self.subagent_max_depth.load(Relaxed) as usize,
            job_wait_max_ms: self.job_wait_max_ms.load(Relaxed),
        }
    }
}

static BUDGETS: PackedBudgets = PackedBudgets {
    tool_deadline_ms: std::sync::atomic::AtomicU64::new(base::DEFAULT_BASH_TIMEOUT_MS),
    mcp_timeout_ms: std::sync::atomic::AtomicU64::new(60_000),
    subagent_max_steps: std::sync::atomic::AtomicU64::new(12),
    subagent_wall_ms: std::sync::atomic::AtomicU64::new(15 * 60_000),
    subagent_max_depth: std::sync::atomic::AtomicU64::new(2),
    job_wait_max_ms: std::sync::atomic::AtomicU64::new(30_000),
};

/// Publish the budgets for the whole process. Both hosts call this at boot and
/// again on every settings save, so a panel change lands on the next tool call
/// rather than the next launch.
pub fn set_budgets(budgets: Budgets) {
    BUDGETS.store(&budgets);
    // The tool deadline keeps its own global too: it is read by `bash` and was
    // published from settings long before this module existed.
    set_default_tool_timeout_ms(budgets.tool_deadline_ms);
}

/// The tool deadline in milliseconds: default and ceiling at once.
pub fn default_tool_timeout_ms() -> u64 {
    DEFAULT_TOOL_TIMEOUT_MS.load(std::sync::atomic::Ordering::Relaxed)
}

/// Where a budget came from, for the note that names it when it bites.
///
/// A message that only says "timed out" sends the reader to a log; one that
/// says which setting owns the number, and that it is editable, ends the hunt.
pub fn budget_hint(what: &str, key: &str) -> String {
    format!(
        "{what} is set by `{key}` — change it in the settings panel, or with \
         `ngu config set --{key} <value>`; it applies to the next call, no restart."
    )
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
