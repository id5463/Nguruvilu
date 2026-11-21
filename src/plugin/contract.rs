//! `Plugin` 契约：一个插件能被内核问到的每一件事。
//!
//! 除了 `name` 和 `apply`，所有方法都有默认实现 —— 一个只贡献一个工具的
//! 插件只写三个方法就够了。默认值是「这一问我不管」，所以不关心的插件
//! 只付一次匹配的代价。
//!
//! 这是本仓库最大的替换点之一：实现这个 trait 就是写一个插件。

use anyhow::Result;

use crate::plugin::ctx::{Contributions, PluginCtx};
use crate::plugin::seam::{Seam, TurnContext, ToolCallView};


use crate::tools::{ConflictPolicy, ToolDef, ToolRegistry};
use anyhow::{anyhow, Result};
use serde_json::Value;
use std::any::Any;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use super::ctx::{Contributions, Disposer, PluginCtx, StateHandle};
use super::fiber::{Epoch, Fiber, FiberState, ReleaseOutcome, Staged, StagedClaims};
use super::realm::{RealmId, RealmMap};
use super::seam::{Seam, ToolCallView, TurnContext};
use super::service::{ServiceEntry, ServiceTable, ServiceView};

/// A plugin.
pub trait Plugin: Send + Sync {
    /// Plugin name, unique per kernel.
    fn name(&self) -> &str;

    /// Services that must exist before this plugin loads.
    fn inject(&self) -> Vec<String> {
        Vec::new()
    }

    /// Services this plugin will provide. Advisory: it lets the kernel reject a
    /// clash before `apply` runs, but the authoritative check happens at
    /// install time, where a plugin may provide something it did not declare.
    fn provide(&self) -> Vec<String> {
        Vec::new()
    }

    /// Produce contributions. Called only when every dependency is present.
    fn apply(&self, ctx: &PluginCtx) -> Result<Contributions>;

    // ---- what a plugin may do while the agent runs ----
    //
    // These are methods, not a registry of named points. The difference
    // matters: a registry means the kernel publishes a fixed list of moments
    // and a plugin hopes one matches; a method means the kernel asks every
    // plugin the question it already knows how to ask. Adding a moment is
    // adding a method here — the plugin says what it can do by implementing
    // it, and there is no name to misspell and no table to look up.

    /// Text to append to this turn's system prompt.
    ///
    /// Called once per turn, after the base prompt and the skill catalog are
    /// assembled, with the user's message in hand. Returning nothing is the
    /// common case; returning text is how a retriever says "you will probably
    /// need these skills" without the kernel knowing what a retriever is.
    fn build_prompt(&self, _ctx: &TurnContext) -> Option<String> {
        None
    }

    /// Which of these tool names survive for one request.
    ///
    /// Runs after the registry is snapshotted and before the body is built, so
    /// a server advertising fifty-five primitives can be cut to the six a task
    /// needs. Names rather than definitions: the registry owns the handlers, and
    /// handing them out to be rebuilt invites a plugin to drop one by accident.
    /// `None` leaves the table whole, and nothing is unregistered either way —
    /// the next turn starts from the full set again.
    fn filter_tools(&self, _ctx: &TurnContext, _names: Vec<String>) -> Option<Vec<String>> {
        None
    }

    /// A last word before a tool call runs.
    ///
    /// Returning `Some(reason)` refuses the call and hands the reason back to
    /// the model as the tool result. This is advice with teeth the other
    /// directions do not have: it is the only place a plugin can say no.
    fn before_tool_call(&self, _ctx: &TurnContext, _call: &ToolCallView) -> Option<String> {
        None
    }

    /// News after a tool call returned.
    ///
    /// Observational, like the event bus, but typed: the output is here rather
    /// than in a JSON blob, and a plugin that wants to count calls does not
    /// have to parse anything.
    fn after_tool_call(&self, _ctx: &TurnContext, _call: &ToolCallView, _ok: bool) {}

    /// Change one piece of data as it passes through the loop.
    ///
    /// One method for every seam instead of one method per seam. A seam that
    /// needs its own method means the kernel grows a method each time someone
    /// wants to touch something new; a seam that is a `Slot` variant means the
    /// plugin says which data it cares about and does not care how many kinds
    /// exist. Adding a kind is a variant here plus one call site, not a new
    /// trait method that every plugin has to keep implementing.
    ///
    /// The default returns the value untouched, so a plugin that does not care
    /// pays one match.
    fn intercept(&self, _ctx: &TurnContext, seam: Seam) -> Seam {
        seam
    }
}
