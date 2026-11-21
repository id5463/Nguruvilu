//! The dynamic loading layer: turning a pack's manifest into running state.
//!
//! Three kinds of thing — plugins, MCP servers, skills — share one manifest,
//! one ordering pass, one ledger and one failure policy, so none of them is a
//! special case. Every load is idempotent through the ledger's recorded hashes.
//!
//! # Layout
//!
//! * [`model`] — what a loader holds, and what one entry's outcome looks like.
//! * [`accessors`] — build a loader, and read out what it holds. No loading.
//! * [`entry`] — run a plan: walk its steps, and load each kind of entry.
//! * [`process`] — the process half of a plugin: find its binary, build it if
//!   the pack ships source, and keep it running for the loader's life.
//! * [`tail`] — the queries and the two ways to finish: hand back, or stop.
//! * [`helpers`] — standalone functions: path resolution, naming and labels.
//!
//! Every item this module exported before is re-exported here, so
//! `crate::loader::X` still names the same item.

pub mod accessors;
pub mod entry;
pub mod helpers;
pub mod model;
pub mod process;
pub mod tail;

pub use helpers::{load_pack, merge_skills, unload_pack};
pub use model::{FailedEntry, LoadReport, Loader, Outcome, UnloadReport};
