//! Minimal MCP client over stdio and Streamable HTTP.
//!
//! MCP is JSON-RPC 2.0. This implements the subset an agent needs to *use* a
//! server: `initialize`, `tools/list`, and `tools/call`. Resources and
//! prompts are deliberately out of scope.
//!
//! Two transports carry the same messages. A stdio server runs as a child
//! process with newline-delimited frames on its pipes: one reader task owns
//! stdout and fans responses out by request id, so concurrent tool calls do
//! not serialize on reading. A Streamable HTTP server is one URL: each
//! message is a `POST`, answered either as a single JSON document or as a
//! `text/event-stream` whose events end in the response, with a
//! `Mcp-Session-Id` assigned at `initialize` travelling on every later
//! request.
//!
//! # Layout
//!
//! * [`model`] — the data: how a server is declared, what a connection holds,
//!   how request ids are drawn, what a kill or close reports.
//! * [`client`] — the behaviour: connect, handshake, list tools, call a tool.
//! * [`util`] — turning what a server answered into what the kernel uses.
//!
//! Every item this module exported before is re-exported here, so
//! `crate::mcp::X` still names the same item.

pub mod client;
pub mod model;
pub mod util;

pub use model::{
    close_server, track, CloseReport, KillReport, McpClient, McpSpec, McpTool, PROTOCOL_VERSION,
};
