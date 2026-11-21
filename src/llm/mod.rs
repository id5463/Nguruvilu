//! OpenAI-compatible model client with streaming and tool calls.
//!
//! Only the OpenAI wire format is supported by design: multi-provider
//! adaptation is the largest source of accidental complexity in agent
//! frameworks, and it buys nothing here.
//!
//! Streaming is the default request mode. Tool-call arguments arrive in
//! fragments keyed by `index`; this module accumulates them so the loop sees
//! one complete call per index.
//!
//! # Why a non-streaming mode exists at all
//!
//! A gateway that answers `stream: true` by opening the response, writing one
//! empty frame, and then never writing again is not a gateway the loop can use:
//! the turn hangs until the whole-request timeout, and the retry reopens the
//! same broken stream. Measured against one such endpoint, ten consecutive
//! streaming requests produced no answer and no `[DONE]` — while the same body
//! with `stream: false` returned the complete answer, including tool calls, in
//! under six seconds.
//!
//! So the mode is a choice rather than an assumption, and both modes feed the
//! loop the same [`LlmEvent`] sequence: the difference is on the wire, never in
//! `agent.rs`.

use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::sync::mpsc;

use crate::message::{Message, ToolCall};

pub mod assemble;
pub mod caps;
pub mod client;
pub mod parse;
pub mod retry;
pub mod sse;
pub mod wire;

pub use assemble::*;
pub use caps::*;
pub use client::*;
pub use parse::*;
pub use retry::*;
pub use sse::*;
pub use wire::*;
