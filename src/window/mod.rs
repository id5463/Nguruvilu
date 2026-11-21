//! How large a model's context window is, and where that number came from.
//!
//! The kernel needs this number to know when history is about to overflow. It
//! is resolved from the most trustworthy source available, in order:
//!
//! 1. **Configured** — an explicit setting always wins.
//! 2. **Provider** — some gateways report `context_length` / `max_model_len`
//!    per model; when they do, that is the truth for that route. This covers
//!    two ways of learning it: a number a caller passed in when it built the
//!    policy, and a number a provider stated at runtime, in an error that
//!    named the limit it was enforcing.
//! 3. **Known table** — a small table of well-known model families.
//! 4. **Assumed** — a deliberately low floor, so an unknown model compacts
//!    early rather than failing late.
//!
//! The source is part of the answer, not a hidden detail: a window that was
//! *assumed*, or that only matched the built-in table, is a guess and the UI
//! should be able to say so.
//!
//! # Interface
//!
//! [`ContextPolicy`] is the seam. The kernel ships a default implementation and
//! consults it every turn; a plugin that provides the `context.policy` service
//! replaces it wholesale — different window, different threshold, different
//! number of messages kept verbatim.
//!
//! The seam is also why [`ReportedWindows`] exists. A policy is built once and
//! then held behind `Arc<dyn ContextPolicy>`, which is a read-only handle, while
//! the thing that learns a real limit is the agent loop — from a provider
//! rejecting an oversized prompt with a message that states the limit. The loop
//! has no way to hand that number to the policy it is using, so both sides
//! instead share one store: the loop records into it, and [`DefaultContextPolicy`]
//! reads it when it resolves a window. The next compaction decision then
//! measures against what this deployment actually enforces rather than against
//! the table's guess about the model id.

mod policy;
mod resolve;

/// The same ruler as [`crate::context::estimate_tokens`], re-exported here.
///
/// One estimate, called from both sides: compaction asks "should this history
/// be trimmed?" and the context budget asks "how much of it fits?", and two
/// rulers made those answers disagree — a plain four-characters-per-token count
/// undercounts CJK by three to four times in exactly the module that decides
/// when to compact.
pub use crate::context::estimate_tokens;
pub use policy::{estimate_history, ContextPolicy, DefaultContextPolicy};
pub use resolve::{
    known_window, reported_windows, ContextWindow, ReportedWindows, WindowSource, ASSUMED_WINDOW,
    MAX_PLAUSIBLE_WINDOW, MIN_PLAUSIBLE_WINDOW,
};

