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
    // A native async recursion guard: a subtask that could itself delegate
    // would let one instruction spawn an unbounded tree, and nothing in the
    // transcript would show where the budget went. The scope is entered around
    // the body (below), so an error return ends it as surely as success.
    let current = delegation_depth();
    if current >= MAX_DEPTH {
        return Err(anyhow!(
            "subagents are already {MAX_DEPTH} deep; this one may not delegate further. \
             Do the work here instead."
        ));
    }
    // The scope *is* the guard: the value lives inside the future and ends
    // with it, however the delegation ends. The old thread-local + Drop guard
    // could be incremented on one worker and dropped on another — tokio polls
    // one task on whichever worker is free — and the count then leaked, after
    // which a legitimate delegation was refused as "already deepest" forever.
    enter_delegation(delegate(access, args)).await.1
}

/// Run `body` one delegation level deeper, and report the depth it ran at.
///
/// The depth travels with the future, so it ends where the body ends — on
/// whichever worker that happens to be. Handing the depth back as well is what
/// lets a test assert what a real delegation saw, instead of reading another
/// task's locals from the outside (which a task-local deliberately refuses).
async fn enter_delegation<F>(body: F) -> (usize, F::Output)
where
    F: std::future::Future,
{
    let depth = delegation_depth() + 1;
    DEPTH.scope(depth, async move { (depth, body.await) }).await
}

/// The body of one delegation, run inside the depth scope.
async fn delegate(access: Arc<dyn ModelAccess>, args: Value) -> Result<ToolOutput> {
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

tokio::task_local! {
    /// Depth of delegation on this task.
    ///
    /// A task-local rather than a parameter because the tool handler signature
    /// carries only the arguments, and threading depth through it would change
    /// every tool for the sake of one. Explicitly *not* a thread-local: tokio
    /// polls one task on whichever worker thread is free, so an increment on
    /// one thread and its restore on another leak the count — silently, until
    /// a legitimate delegation is refused as "already deepest" forever. A
    /// task-local travels with the future, so the scope's value ends when the
    /// delegation does, on any thread.
    static DEPTH: usize;
}

/// How deep the current delegation is; zero outside one.
///
/// Read-only, for tests and diagnostics: outside a delegation there is no
/// scope at all, which is the honest answer to "how deep".
pub(crate) fn delegation_depth() -> usize {
    DEPTH.try_with(|depth| *depth).unwrap_or(0)
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
        let error = DEPTH
            .scope(MAX_DEPTH, run(access(), json!({ "task": "x" })))
            .await
            .expect_err("must refuse");
        let text = format!("{error:#}");
        assert!(text.contains("deep"), "{text}");
        // The refusal is the scope check, not a counter somebody has to reset.
        assert_eq!(delegation_depth(), 0);
    }

    #[tokio::test]
    async fn the_depth_scope_ends_when_the_delegation_does() {
        DEPTH.scope(1, async {
            assert_eq!(delegation_depth(), 1);
            DEPTH.scope(2, async {
                assert_eq!(delegation_depth(), 2);
            })
            .await;
            // The inner scope ended: back to the outer value, with nothing
            // leaked upward — the property a thread-local could not promise.
            assert_eq!(delegation_depth(), 1);
        })
        .await;
        assert_eq!(delegation_depth(), 0);
    }

    /// Delegations running at the same time on two workers: each must see
    /// *itself* at depth 1, and none may leave a count behind.
    ///
    /// With the old thread-local guard the increment and its restore could land
    /// on different threads — tokio polls one task on whichever worker is free
    /// — so the count leaked and a later legitimate delegation was refused as
    /// "already deepest" forever. Each body parks across an await so the
    /// workers really do swap tasks under it.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn concurrent_delegations_see_depth_one_and_leave_nothing_behind() {
        let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
        let mut handles = Vec::new();
        for _ in 0..4 {
            let seen = Arc::clone(&seen);
            handles.push(tokio::spawn(async move {
                // The same entry point the tool itself uses, so this is the
                // real scope rather than a copy of its arithmetic.
                let (entered, observed) = enter_delegation(async {
                    tokio::task::yield_now().await;
                    delegation_depth()
                })
                .await;
                seen.lock().expect("seen").push((entered, observed));
            }));
        }
        for handle in handles {
            handle.await.expect("the task ran");
        }
        let seen = seen.lock().expect("seen").clone();
        assert_eq!(seen.len(), 4);
        assert!(
            seen.iter().all(|(entered, observed)| *entered == 1 && *observed == 1),
            "each concurrent delegation runs at depth 1, not at a shared count: {seen:?}"
        );
        assert_eq!(
            delegation_depth(),
            0,
            "a finished delegation must leave no count behind"
        );

        // And the entry point ends the scope on its failure path too: this is
        // the same wrapper `run` uses, driven with a body that returns an error
        // instead of calling a provider. Testing the real error path needs a
        // live transport and a connection timeout — an integration test's job,
        // not something that should stall the unit suite behind a hang.
        let (depth, (result, seen)) = enter_delegation(async {
            tokio::task::yield_now().await;
            (Err::<(), &'static str>("the transport refused"), delegation_depth())
        })
        .await;
        assert_eq!(depth, 1);
        assert!(result.is_err(), "the body's error is handed back untouched");
        assert_eq!(seen, 1, "the body still saw its own depth");
        assert_eq!(
            delegation_depth(),
            0,
            "a failed delegation left a count behind"
        );
    }
}
