//! Nguruvilu — an independent agent kernel.
//!
//! The kernel does four things: read and write files, run commands, drive the
//! model loop, and manage sessions. Everything else is loaded dynamically.
//!
//! This crate is the minimum viable kernel: neutral message format, an
//! OpenAI-compatible streaming client, a mutable tool registry with the four
//! base tools, the agent loop with parallel tool dispatch, and JSONL session
//! persistence.

pub mod agent;
pub mod llm;
pub mod message;
pub mod session;
pub mod tools;

pub use agent::{Agent, AgentOutcome};
pub use llm::{LlmClient, LlmConfig, LlmEvent};
pub use message::{Message, Role, ToolCall};
pub use session::{Session, SessionStore};
pub use tools::{ToolDef, ToolRegistry};
