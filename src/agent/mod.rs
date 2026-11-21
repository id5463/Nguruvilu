//! The agent loop, one file per function.
//!
//! One turn is: send the history, stream the reply, dispatch any requested tools
//! in parallel, append their results, and repeat until the model stops asking
//! for tools.
//!
//! * `loop` — [`Agent`], [`AgentLoop`], and the process-wide installed slot.
//! * `observer` — [`AgentObserver`], what a host renders a turn with.
//! * `config` — [`TurnConfig`] and the frozen [`TurnSettings`] of one turn.
//! * `hooks` — [`PluginHooks`], the seam a pack answers a turn through.
//! * `state` — [`LoopState`], [`AgentOutcome`], and one step's result.
//!
//! Three rules from the spec live across these files:
//!
//! * **Independent tool calls run concurrently.** Serial dispatch is the single
//!   largest avoidable cost in an agent turn.
//! * **Each turn takes one snapshot.** Settings are read once at the start of a
//!   turn, so a hot reload landing mid-turn cannot change a request already in
//!   flight. A turn sees changes; a turn never sees half of one.
//! * **Timing is measured, not guessed.** Every turn reports how long the model
//!   took versus how long tools took.

mod config;
mod hooks;
mod r#loop;
mod observer;
mod state;

pub use config::{StaticConfig, TurnConfig, TurnSettings};
pub use hooks::PluginHooks;
pub use observer::{AgentObserver, SilentObserver};
pub use r#loop::{install_loop, installed_loop, uninstall_loop, Agent, AgentLoop, BuiltinLoop};
pub use state::{AgentOutcome, LoopState, TurnTiming, CONTINUATION_BATCHES};
