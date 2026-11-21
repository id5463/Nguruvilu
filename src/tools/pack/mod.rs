//! The `pack` tool: build, inspect, install, load, and unload packs.
//!
//! `pack` is one tool with an `action` argument rather than five tools, because
//! the actions share their description and a model choosing between five names
//! spends attention on the choice. What it can do is bounded on purpose:
//! building writes an archive in the workspace, installing places code where
//! the kernel will load it, and loading a pack into a running conversation is
//! queued for the host to apply at the end of the turn.
//!
//! # Layout
//!
//! One file per function:
//!
//! * [`action`] — the actions themselves: `build`, `verify`, `list`,
//!   `install`, `load`/`unload`, `apply`, and the dispatcher over them.
//! * [`pending`] — [`PendingQueue`], where a load or unload waits until the
//!   turn ends, and the tool's registration (which names that queue).
//!
//! Every item this module exported before is re-exported here, so
//! `crate::tools::pack::X` still names the same item.

mod action;
mod pending;

pub use pending::{register, register_with, Pending, PendingQueue};
