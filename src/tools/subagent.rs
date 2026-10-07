//! A tool that runs a subtask in a fresh agent.
//!
//! A subagent is not separate machinery — it is another [`Agent`] with its own
//! history. What makes it useful is the **empty** history: a large side
//! investigation happens out of sight, and only its answer enters the
//! conversation that the user is watching.
//!
//! # What it inherits
//!
//! Everything, from the session, through [`ModelAccess`]: the same route, the
//! same tool table, the same standing instructions, the same context policy. A
//! subagent that reached a tool the session cannot would be a way around the
//! session's own limits, and one running on a different model would be a
//! surprise the user did not choose.
//!
//! # What it does not inherit
//!
//! The conversation. That is the point — the parent's transcript stays small.
//! The cost is that the subagent cannot ask a follow-up question, so the task
//! it is given has to stand on its own, and the tool description says so.

use std::sync::Arc;

use anyhow::{anyhow, Result};
use serde_json::{json, Value};

use crate::agent::{Agent, StaticConfig};
use crate::model::ModelAccess;
use crate::tools::{ToolDef, ToolFuture, ToolOutput};

/// The tool name.
///
/// Not `subagent` or `task`: a short verb that reads as an instruction, and no
/// collision with a name a provider reserves.
pub const TOOL: &str = "delegate";

/// The most steps a subagent may take.
///
/// Bounded because a subagent's loop is invisible: one that runs away spends
/// the user's budget on something they cannot see. The parent's own limit is
/// higher; this is deliberately lower.
pub const MAX_STEPS: usize = 12;

/// Register the plugin a pack loads to bring the tool in.
///
/// Defining is not loading: nothing appears in a tool table until a pack's
/// assembly asks for `builtin:delegate`. The code ships inside this binary —
/// a pack never carries a binary — but *whether a conversation has it* is the
/// pack's decision, which is what keeps the kernel from growing a capability
/// of its own.
pub fn define(kernel: &mut crate::plugin::Kernel) {
    kernel.define(std::sync::Arc::new(Delegate));
}

/// The plugin behind [`TOOL`].
pub struct Delegate;

impl crate::plugin::Plugin for Delegate {
    fn name(&self) -> &str {
        TOOL
    }

    /// The route publishes as a service, so the tool waits for it rather than
    /// capturing a client at startup.
    fn inject(&self) -> Vec<String> {
        vec![crate::model::SERVICE.to_string()]
    }

    fn apply(&self, ctx: &crate::plugin::PluginCtx) -> Result<crate::plugin::Contributions> {
        let access = crate::model::ModelHandle::from_view(&ctx.services)
            .ok_or_else(|| anyhow!("the session has not published a route"))?;
        Ok(crate::plugin::Contributions::new().tool(tool(access)))
    }
}

fn tool(access: Arc<dyn ModelAccess>) -> ToolDef {
    ToolDef::new(
        TOOL,
        format!(
            "Run a self-contained subtask in a fresh agent with no memory of this \
             conversation, and get back only its answer. Use it to keep a large \
             side investigation out of this transcript. \
             The subtask sees the same tools and the same model you do, but not \
             what was said here — so state everything it needs, including paths \
             and what 'done' looks like. It cannot ask you anything, and it stops \
             after {MAX_STEPS} steps."
        ),
        json!({
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The whole subtask, standing on its own: what to do, where, and what counts as finished."
                },
                "model": {
                    "type": "string",
                    "description": "Optional model for the subtask. Defaults to the session's. A cheaper model is usually right for mechanical work."
                }
            },
            "required": ["task"]
        }),
        TOOL,
        move |args| {
            let access = Arc::clone(&access);
            Box::pin(async move { run(access, args).await }) as ToolFuture
        },
    )
}

async fn run(access: Arc<dyn ModelAccess>, args: Value) -> Result<ToolOutput> {
    let task = args
        .get("task")
        .and_then(|t| t.as_str())
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .ok_or_else(|| {
            anyhow!(
                "missing required argument: task. The subtask has no memory of this \
                 conversation, so it must be written out in full."
            )
        })?;

    // A native async recursion guard: a subtask that could itself delegate
    // would let one instruction spawn an unbounded tree, and nothing in the
    // transcript would show where the budget went.
    if !DEPTH.with(|depth| {
        let current = depth.get();
        if current >= MAX_DEPTH {
            false
        } else {
            depth.set(current + 1);
            true
        }
    }) {
        return Err(anyhow!(
            "subagents are already {MAX_DEPTH} deep; this one may not delegate further. \
             Do the work here instead."
        ));
    }
    let _guard = DepthGuard;

    let client = match args.get("model").and_then(|m| m.as_str()) {
        Some(model) => access.client_for(model)?,
        None => access.client()?,
    };

    // The same instructions the session runs under, so a subtask is bounded by
    // exactly the permissions and conventions the session has.
    let config: Arc<dyn crate::agent::TurnConfig> = Arc::new(
        StaticConfig::new(
            access.system_prompt(),
            access.tools(),
            client.model(),
        )
        .with_policy(access.context_policy()),
    );

    // An empty history, then the task as the first user message.
    let mut agent = Agent::new(client, config, Vec::new()).with_max_steps(MAX_STEPS);
    let outcome = agent.run(task).await?;

    // A failed subtask reports the failure as the tool result, carrying any
    // partial answer with it: the parent model reads what went wrong instead
    // of a bare transport error surfacing as a crashed tool.
    if let Some(error) = &outcome.error {
        let mut message = format!("The subtask failed: {error}");
        if !outcome.text.trim().is_empty() {
            message.push_str("\n\nPartial answer before the failure:\n");
            message.push_str(&outcome.text);
        }
        return Err(anyhow!(message));
    }

    let mut text = format!(
        "The subtask finished after {} step(s) and {} tool call(s).\n\n{}",
        outcome.steps, outcome.tool_calls, outcome.text
    );
    // A subtask that produced nothing is worth saying out loud: an empty answer
    // reads as "it found nothing", which is a different claim from "it ran out
    // of steps".
    if outcome.text.trim().is_empty() {
        text.push_str(
            "\n\n(It returned no text. It may have run out of steps before \
             concluding; a narrower task, or a larger budget, would help.)",
        );
    }
    Ok(ToolOutput::text(text))
}

/// How deep delegation may nest.
const MAX_DEPTH: usize = 2;

thread_local! {
    /// Depth of delegation on this thread.
    ///
    /// A thread-local rather than a parameter because the tool handler signature
    /// carries only the arguments, and threading depth through it would change
    /// every tool for the sake of one.
    static DEPTH: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Puts the depth back however the subtask ends.
struct DepthGuard;

impl Drop for DepthGuard {
    fn drop(&mut self) {
        DEPTH.with(|depth| depth.set(depth.get().saturating_sub(1)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hotreload::ModelRoute;
    use crate::llm::{LlmClient, LlmConfig};
    use crate::tools::ToolRegistry;
    use crate::window::{ContextPolicy, DefaultContextPolicy};

    struct FakeAccess {
        tools: Arc<ToolRegistry>,
    }

    impl ModelAccess for FakeAccess {
        fn route(&self) -> ModelRoute {
            ModelRoute {
                provider: "openai".into(),
                base_url: "http://127.0.0.1:1/v1".into(),
                api_key: "k".into(),
                model: "m".into(),
                ..Default::default()
            }
        }

        fn client(&self) -> Result<LlmClient> {
            self.client_for("m")
        }

        fn client_for(&self, model: &str) -> Result<LlmClient> {
            Ok(LlmClient::new(LlmConfig::new(
                "http://127.0.0.1:1/v1",
                "k",
                model,
            ))?)
        }

        fn tools(&self) -> Arc<ToolRegistry> {
            Arc::clone(&self.tools)
        }

        fn system_prompt(&self) -> String {
            "you are a subtask".into()
        }

        fn context_policy(&self) -> Arc<dyn ContextPolicy> {
            Arc::new(DefaultContextPolicy::default())
        }
    }

    fn access() -> Arc<dyn ModelAccess> {
        Arc::new(FakeAccess {
            tools: Arc::new(ToolRegistry::with_base_tools().unwrap()),
        })
    }

    /// The state a session is in once the subagent pack has loaded: the plugin
    /// defined by the kernel, the route published, and a pack that asked for
    /// it.
    fn loaded_kernel() -> crate::plugin::Kernel {
        let mut kernel = crate::plugin::Kernel::new();
        define(&mut kernel);
        crate::model::install(&mut kernel, access()).unwrap();
        kernel
            .load(TOOL, crate::plugin::RealmMap::new(), serde_json::Value::Null)
            .unwrap();
        kernel
    }

    /// The tool's schema, wherever it sits in the table.
    fn schema_of(kernel: &crate::plugin::Kernel) -> serde_json::Value {
        kernel
            .tools()
            .schemas()
            .into_iter()
            .find(|schema| schema["function"]["name"] == TOOL)
            .expect("the delegate tool has a schema")
    }

    #[test]
    fn defining_the_plugin_alone_loads_nothing() {
        // The kernel holds the code; the pack decides it is there. Between
        // `define` and a pack's assembly asking for it, the tool must not be
        // in the table — otherwise the kernel grew a capability it claims not
        // to have.
        let mut kernel = crate::plugin::Kernel::new();
        define(&mut kernel);
        crate::model::install(&mut kernel, access()).unwrap();

        assert!(kernel.tools().get(TOOL).is_none());
        assert!(kernel.plugin(TOOL).is_some(), "but the code is available");
    }

    #[test]
    fn without_a_published_route_nothing_activates() {
        // The tool waits for the route rather than capturing a client at
        // startup: a pack that loads before the host publishes one must leave
        // the kernel usable, not half-configured.
        let mut kernel = crate::plugin::Kernel::new();
        define(&mut kernel);
        kernel
            .load(TOOL, crate::plugin::RealmMap::new(), serde_json::Value::Null)
            .unwrap();

        assert!(kernel.tools().get(TOOL).is_none());
    }

    #[test]
    fn the_tool_is_registered_under_the_pack_that_asked_for_it() {
        let kernel = loaded_kernel();
        assert!(kernel.tools().get(TOOL).is_some());
        assert_eq!(
            kernel.tools().owner(TOOL),
            Some(TOOL),
            "the kernel stamps ownership, so the pack cannot claim someone else's tool"
        );
    }

    #[test]
    fn the_tool_name_does_not_collide_with_a_reserved_one() {
        // `web_search` is dropped by a provider without any error; this is the
        // cheap version of that guard.
        assert!(!TOOL.is_empty());
        assert!(TOOL.chars().all(|c| c.is_ascii_lowercase() || c == '_'));
    }

    #[test]
    fn the_description_says_the_subtask_starts_blind() {
        // The single most important thing about this tool for a caller to know:
        // the subtask has no memory of the conversation.
        let description = schema_of(&loaded_kernel())["function"]["description"]
            .as_str()
            .unwrap()
            .to_string();
        assert!(description.contains("no memory"), "{description}");
        assert!(description.contains("cannot ask"), "{description}");
    }

    #[test]
    fn the_description_carries_the_step_budget() {
        // A caller that does not know the budget cannot size its task.
        let description = schema_of(&loaded_kernel())["function"]["description"]
            .as_str()
            .unwrap()
            .to_string();
        assert!(description.contains(&MAX_STEPS.to_string()), "{description}");
    }

    #[tokio::test]
    async fn an_empty_task_is_refused_with_the_reason() {
        for args in [json!({}), json!({ "task": "" }), json!({ "task": "   " })] {
            let error = run(access(), args).await.expect_err("must refuse");
            let text = format!("{error:#}");
            assert!(text.contains("task"), "{text}");
            assert!(
                text.contains("no memory"),
                "it should say why the task must be complete: {text}"
            );
        }
    }

    #[tokio::test]
    async fn delegating_past_the_depth_limit_is_refused_by_name() {
        // Depth is entered before the call, so pretend we are already there.
        DEPTH.with(|depth| depth.set(MAX_DEPTH));
        let error = run(access(), json!({ "task": "x" }))
            .await
            .expect_err("must refuse");
        let text = format!("{error:#}");
        assert!(text.contains("deep"), "{text}");
        DEPTH.with(|depth| depth.set(0));
    }

    #[test]
    fn the_depth_guard_restores_the_count() {
        DEPTH.with(|depth| depth.set(0));
        DEPTH.with(|depth| depth.set(1));
        {
            let _guard = DepthGuard;
            DEPTH.with(|depth| depth.set(depth.get() + 1));
        }
        assert_eq!(DEPTH.with(|depth| depth.get()), 1, "restored to what it was");
        DEPTH.with(|depth| depth.set(0));
    }
}
