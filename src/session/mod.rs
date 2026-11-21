//! Sessions and durable storage.
//!
//! Storage is a trait so the backend can be replaced (SQLite later) without
//! touching callers. The shipped backend is JSONL: a header line followed by
//! one line per message.
//!
//! JSONL is append-only, which is what makes it crash-safe — a truncated
//! final line costs at most the last message, and nothing before it is lost.
//!
//! # Layout
//!
//! One file per functional unit, each reachable on its own path: `model` holds
//! the records, `store` the replaceable backend trait and the service a plugin
//! sees, `jsonl` the shipped backend and `replay` the folding. Everything this
//! module exported before is re-exported here, so `crate::session::X` still
//! names the same item.

pub mod jsonl;
pub mod model;
pub mod replay;
pub mod store;

pub use jsonl::{JsonlStore, StoreHandle};
pub use model::{Session, SessionSummary, SessionTotals, TurnStats};
pub use replay::{ensure_workspace, new_id, now_iso, totals_of, workspace_dir};
pub use store::{install_store, InjectionRecord, SessionService, SessionStore, SERVICE};
