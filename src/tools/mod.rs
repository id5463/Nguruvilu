//! The tool table and the tools that fill it.
//!
//! The registry is the kernel's mutable tool table. Plugins contribute to it
//! through `register`; the four base tools are registered at boot.
//!
//! Name collisions fail loud by default. Silent override is the most
//! dangerous failure mode a plugin system can have: one plugin quietly
//! replacing another's safety check leaves no trace. A contribution that
//! genuinely intends to replace an existing tool says so explicitly.
//!
//! # Layout
//!
//! One file per function:
//!
//! * [`output`] — what a tool produced ([`ToolOutput`]).
//! * [`registry`] — the table itself: the tool definition, the identity a call
//!   runs under, and the conflict policy.
//! * [`budget`] — the runtime budgets and the session scope pinned over a call.
//! * [`base`], [`judge`], [`readers`], [`spill`] — the tools themselves.
//! * [`jobs`] — background jobs: the job model, the process plumbing, the
//!   registry that owns both.
//! * [`pack`] — the `pack` tool: its actions, and the queue a load waits in.
//!
//! Every item this module exported before is re-exported here, so
//! `crate::tools::X` still names the same item.

pub mod base;
pub mod judge;
pub mod jobs;
pub mod pack;
pub mod readers;

pub mod spill;

mod budget;
mod output;
mod registry;

pub use budget::{
    budget_hint, budgets, current_session, current_workspace, default_tool_timeout_ms, set_budgets,
    set_default_tool_timeout_ms, Budgets,
};
pub use output::ToolOutput;
pub use registry::{
    ConflictPolicy, OverrideRecord, ToolDef, ToolFuture, ToolHandler, ToolIdentity, ToolRegistry,
};

pub(crate) use budget::effective_timeout;
// Re-exported for `base`'s Windows-only path tests; nothing else outside the
// budget module names it, so on a Unix build the import reads as unused.
#[allow(unused_imports)]
pub(crate) use budget::ScopedSession;
