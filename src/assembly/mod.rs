//! The assembly manifest: how a pack declares what it loads and in what order.
//!
//! A pack ships an `assembly.yaml` alongside its identity manifest. It states
//! which plugins, MCP servers, and skills to load — the three are peers, not
//! nested — grouped into stages, with explicit dependencies and per-entry
//! policy.
//!
//! Ordering has four levels, highest first:
//!
//! 1. `after` / `before` — a real dependency, never violated.
//! 2. stage order — coarse grouping (`foundation` before `extensions`).
//! 3. `order` — a numeric sort within a stage.
//! 4. declaration order — the stable tiebreak.
//!
//! A cycle in the dependency graph is an error, not something to paper over.

pub mod manifest;
pub mod plan;

pub use manifest::{
    Assembly, BuildSpec, ConflictKind, Defaults, EntryBase, LoadMode, McpEntry, OnFailure,
    PluginEntry, Scope, SkillEntry, Stage, Teardown, Transport, ASSEMBLY_VERSION,
};
pub use plan::{EntryKind, Plan, PlannedStep, SkippedEntry};
