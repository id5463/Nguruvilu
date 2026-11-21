//! The command-line surface, one functional unit per file.
//!
//! Each subcommand is its own file because each is a thing that can be
//! replaced on its own: how statistics are printed, what an export looks like,
//! how a plugin is driven. None of them share state — a subcommand reads what
//! it needs from disk or takes it as an argument — so one can change without
//! touching another.
//!
//! * [`args`] — the flag and subcommand types. Nothing here reaches the
//!   kernel, so a change to a flag cannot change what a turn does.
//! * [`events`] — `ngu events`: the audit log, newest first.
//! * [`injections`] — `ngu injections`: what would be injected, and why.
//! * [`observer`] — progress to stderr, keeping stdout clean for the answer.
//! * [`plugin`] — `ngu plugin`: load a plugin binary and call one of its tools.
//!
//! Everything this module exports is re-exported below, so `cli::X` names the
//! same item as the file that defines it.

pub mod args;
pub mod config;
pub mod events;
pub mod export;
pub mod injections;
pub mod mcp;
pub mod mission;
pub mod observer;
pub mod pack;
pub mod ui;
pub mod plugin;
pub mod runtime;
pub mod stats;
pub mod tui;

pub use args::*;
pub use events::events_command;
pub use config::config_command;
pub use export::export_command;
pub use injections::show_injections;
pub use mcp::mcp_import_command;
pub use mission::{active_mission_block, clip_mission_text};
pub use pack::{install_command, list_packs, pack_command, uninstall_command, verify_command};
pub use plugin::plugin_command;
pub use runtime::{list_skills, show_assembly, show_runtime, snapshot_command};
pub use stats::stats_command;
pub use tui::interactive;
pub use ui::ui_command;
