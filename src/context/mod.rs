//! The context injection engine.
//!
//! Mechanism adapted from SillyTavern's World Info (`world-info.js`), which is
//! the most carefully built context-budgeting design in this space, then
//! changed for an agent rather than a roleplay client.
//!
//! What it does: given a conversation and a budget, decide which extra
//! fragments to inject, where to put them, and when to stop.
//!
//! Two changes from the original matter:
//!
//! * **No randomness.** SillyTavern rolls dice (`probability`) and picks
//!   group winners at random. An agent must be reproducible — the same input
//!   has to produce the same context — so selection is deterministic: sort by
//!   `order`, then by id.
//! * **Cache awareness.** Injection position is not cosmetic. Writing into the
//!   prompt prefix invalidates the provider's cached prefix; writing into the
//!   history does not. The cache policy therefore decides *where* a fragment
//!   may go, and under `CacheFirst` a prefix injection is refused and moved
//!   into the history instead.

mod engine;
mod entry;
mod position;

pub use crate::hotreload::CachePolicy;
pub use engine::{estimate_tokens, InjectionEngine, InjectionSource};
pub use entry::{Injection, InjectionEntry};
pub use position::{Position, PositionKind, ScanScope};

