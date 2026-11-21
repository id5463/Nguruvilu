//! 一轮对话跑在什么上：`AgentLoop` 契约，以及把它装进/卸出进程。
//!
//! 换掉循环是最大的一个替换点 —— 它决定一轮对话怎么想。默认实现是
//! [`BuiltinLoop`]，一个包可以装自己的。
//!
//! 安装是进程级的而不是会话级：一台机器装了某个循环，就期望每一轮都那样想。
//! 后来的覆盖先前的（最后写的赢），与 `ServiceTable` 同一条规则 —— 拒绝的话
//! 等于内核替用户决定两个包里哪个合法。

use std::sync::Arc;

use async_trait::async_trait;
use anyhow::Result;

use crate::agent::config::TurnConfig;
use crate::agent::state::{AgentOutcome, LoopState};
use crate::llm::LlmClient;
use crate::message::Message;


use crate::compaction::{Compaction, Summarizer};
use crate::llm::{LlmClient, LlmEvent, PartialCallView, TokenUsage};
use crate::message::{Message, ToolCall};
use crate::mission::MissionBrief;
use crate::tools::ToolRegistry;
use crate::window::{ContextPolicy, ContextWindow};
use anyhow::Result;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Instant;
use tokio::task::JoinSet;

use super::config::{StaticConfig, StaticStateConfig, TurnConfig, TurnSettings};
use super::observer::{AgentObserver, SilentObserver};
use super::state::{
    record_reported_window, stop_flag, summarize_tool_error, AgentOutcome, LoopState, StepOutput,
    TurnTiming, CONTINUATION_BATCHES, MISSION_NUDGE, MISSION_RETRY_CAP, STOPPED,
};
use crate::compaction;
use crate::llm;

/// The loop a turn runs on.
///
/// One method, and the whole turn's state goes in and comes out. The default is
/// [`Agent`]'s own loop, which is what every existing caller gets; a pack that
/// wants a different shape — a tree search, a vote across routes, a loop that
/// stops only when a person says so — supplies its own and is not second-class:
/// nothing in the kernel asks whether the loop is "the real one".
///
/// Being replaceable does not mean being replaced. A build that never installs
/// another loop runs exactly the code it ran before.
pub trait AgentLoop: Send + Sync {
    /// Run one turn to its end.
    ///
    /// `state` is owned, not borrowed: a loop that spawns work with the
    /// conversation in hand cannot do that through a borrow that must outlive
    /// the spawn. Whatever the loop changes is left in `state` when it returns,
    /// and the caller reads it back out.
    fn run<'a>(
        &'a self,
        state: LoopState,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = (LoopState, Result<AgentOutcome>)> + Send + 'a>,
    >;
}

/// How many times one answer may be continued after hitting the provider's
/// output ceiling, before the loop stops extending it.
const MAX_CONTINUATIONS: usize = 3;

/// The loop every build has, wrapped so it can be chosen like any other.
///
/// It does not reimplement anything: it builds an [`Agent`] from the state,
/// runs the same `run` the kernel always had, and hands the conversation back.
/// That is deliberate — a "built-in loop" that had drifted from the one every
/// caller gets would mean two behaviours to keep correct, and the point of the
/// trait is that replacing the loop is ordinary, not that the default is
/// special.
pub struct BuiltinLoop;

impl AgentLoop for BuiltinLoop {
    fn run<'a>(
        &'a self,
        mut state: LoopState,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = (LoopState, Result<AgentOutcome>)> + Send + 'a>,
    > {
        Box::pin(async move {
            let mut agent = Agent::new(
                state.client.clone(),
                Arc::new(StaticStateConfig {
                    settings: state.settings.clone(),
                }),
                std::mem::take(&mut state.messages),
            )
            .as_builtin();
            agent.max_steps = state.max_steps;
            agent.observer = Arc::clone(&state.observer);
            agent.summarizer = Arc::clone(&state.summarizer);
            agent.identity = state.identity.clone();
            agent.cancel = state.cancel.clone();
            agent.inbox = state.inbox.clone();
            agent.subtask = state.subtask;
            agent.stop_reason = state.stop_reason.clone();
            agent.injection_override = state.injection_override.clone();
            let prompt = state.start_prompt.clone();
            let result = agent.run(&prompt).await;
            state.messages = std::mem::take(&mut agent.messages);
            (state, result)
        })
    }
}

/// thinks, and a machine that has installed one expects every turn to think that
/// way. A caller that wants the built-in loop simply does not ask.
static INSTALLED_LOOP: std::sync::RwLock<Option<Arc<dyn AgentLoop>>> = std::sync::RwLock::new(None);

/// Install the loop every subsequent turn runs on.
///
/// Replacing an installed loop is allowed: a second pack's loop wins, and the
/// loser's turns run the winner's code until the process restarts. That is the
/// same "last writer wins" rule the service table uses, and for the same reason
/// — refusing would make the kernel decide which of two packs is legitimate.
pub fn install_loop(loop_impl: Arc<dyn AgentLoop>) {
    let mut slot = INSTALLED_LOOP.write().unwrap_or_else(|p| p.into_inner());
    *slot = Some(loop_impl);
}

/// Remove the installed loop, returning it. Turns go back to the built-in one.
pub fn uninstall_loop() -> Option<Arc<dyn AgentLoop>> {
    let mut slot = INSTALLED_LOOP.write().unwrap_or_else(|p| p.into_inner());
    slot.take()
}

/// The installed loop, if any.
pub fn installed_loop() -> Option<Arc<dyn AgentLoop>> {
    INSTALLED_LOOP
        .read()
        .unwrap_or_else(|p| p.into_inner())
        .clone()
}
