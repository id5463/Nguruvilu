//! The dynamic loading layer: turning a pack's manifest into running state.
//!
//! Three kinds of thing — plugins, MCP servers, skills — share one manifest,
//! one ordering pass, one ledger and one failure policy, so none of them is a
//! special case. Every load is idempotent through the ledger's recorded hashes.
//!
//! # Layout
//!
//! * [`model`] — what a loader holds and what one entry's outcome looks like.
//! * [`apply`] — the loader's behaviour: build it, run a plan, load each kind.
//! * [`helpers`] — the standalone functions: path resolution, process start,
//!   naming and labels.
//!
//! Every item this module exported before is re-exported here, so
//! `crate::loader::X` still names the same item.

pub mod apply;
pub mod helpers;
pub mod model;

pub use apply::{load_pack, merge_skills, unload_pack};
pub use model::{FailedEntry, LoadReport, Loader, Outcome, UnloadReport};
