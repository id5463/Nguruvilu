//! The command-line surface, one functional unit per file.
//!
//! Each subcommand is its own file because each is a thing that can be replaced
//! on its own: how statistics are printed, what an export looks like, how a
//! plugin is driven. None of them share state — a subcommand reads what it
//! needs from disk or takes it as an argument — so one can change without
//! touching another.
//!
//! # The pieces
//!
//! * [`args`] — flags and subcommand types. Nothing here reaches the kernel, so
//!   a change to a flag cannot change what a turn does.
//! * [`observer`] — progress to stderr, keeping stdout clean for the answer.
//! * [`tui`] — interactive mode: read a line, run a turn, print the answer.
//! * [`mission`] — the mission strip: the five lines the model sees.
//!
//! Subcommands, one file each:
//! [`stats`], [`export`], [`config`], [`pack`], [`mcp`], [`plugin`],
//! [`events`], [`injections`], [`ui`], [`skills`], [`assembly_cli`],
//! [`snapshot`], [`runtime_cmd`], [`routes_cli`].
//!
//! # The routes
//!
//! Ten of these register a `dispatch` route, so a pack can replace one without
//! touching the kernel: [`stats::register`], [`export::register`],
//! [`config::register`], [`pack::register`], [`cli_routes::register`].
//!
//! Everything this module exports is re-exported below, so `cli::X` names the
//! same item as the file that defines it.

pub mod args;
pub mod assembly_cli;
pub mod cli_routes;
pub mod config;
pub mod content_cli;
pub mod events;
pub mod export;
pub mod injections;
pub mod mcp;
pub mod mission;
pub mod model_route;
pub mod observer;
pub mod pack;
pub mod plugin;
pub mod routes_cli;
pub mod runtime_cmd;
pub mod skills;
pub mod snapshot;
pub mod stats;
pub mod tui;
pub mod ui;

pub use args::*;
pub use assembly_cli::{apply_assembly, show_assembly};
pub use cli_routes::register as register_cli_routes;
pub use config::config_command;
pub use content_cli::adopt_content;
pub use events::events_command;
pub use export::export_command;
pub use injections::show_injections;
pub use mcp::mcp_import_command;
pub use mission::{active_mission_block, clip_mission_text};
pub use model_route::model_route;
pub use observer::{PrettyObserver, SessionEnd};
pub use pack::{install_command, list_packs, pack_command, uninstall_command, verify_command};
pub use plugin::plugin_command;
pub use routes_cli::routes_command;
pub use runtime_cmd::show_runtime;
pub use skills::list_skills;
pub use snapshot::snapshot_command;
pub use stats::stats_command;
pub use tui::interactive;
pub use ui::ui_command;
