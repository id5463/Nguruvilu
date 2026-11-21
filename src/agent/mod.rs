//! The agent loop, one file per concern.
//!
//! A turn sends the history, streams the reply, dispatches the tools the model
//! asked for — in parallel — and repeats until the model asks for nothing.
//!
//! # Layout
//!
//! * [`contract`] — `AgentLoop`: what a turn runs on, and how to install one.
//!   This is the largest seam in the kernel: replacing it changes how a turn
//!   thinks.
//! * [`builder`] — `Agent` and its `with_*` methods: assembling a turn.
//! * [`run`] — running it: the request, the stream, the tools, the accounting.
//! * [`observer`] — progress out of the loop, and where it goes.
//! * [`config`] — where a turn's settings come from (`TurnConfig`).
//! * [`hooks`] — what the loop asks plugins (`PluginHooks`).
//! * [`state`] — the data a turn carries.
//!
//! Every item this module exported before is re-exported here, so
//! `crate::agent::X` still names the same item.

pub mod builder;
pub mod config;
pub mod contract;
pub mod hooks;
pub mod observer;
pub mod run;
pub mod state;

pub use builder::Agent;
pub use config::{StaticConfig, TurnConfig, TurnSettings};
pub use contract::{install_loop, installed_loop, uninstall_loop, AgentLoop, BuiltinLoop};
pub use hooks::PluginHooks;
pub use observer::{AgentObserver, SilentObserver};
pub use state::{AgentOutcome, LoopState, StepOutput, TurnTiming};
