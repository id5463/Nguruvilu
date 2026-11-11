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
//!
//! # What bounds it
//!
//! Two budgets, both stated in the tool description, because a caller sizes a
//! task against them: [`MAX_STEPS`] model rounds — the whole budget, not one
//! batch of it; [`Agent::as_subtask`] switches off the batch renewal a
//! conversation enjoys — and [`WALL_BUDGET_SECS`] of wall clock, because a step
//! counts rounds and says nothing about how long one lasts. While it runs, the
//! subtask is silent: it appends every step, tool call and message to a file
//! under [`TRACE_DIR`] and the result names the path, so a slow subtask is
//! distinguishable from a wedged one and the user can watch it work.

use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use anyhow::{anyhow, Result};
use serde_json::{json, Value};

use crate::agent::{Agent, AgentObserver, SilentObserver, StaticConfig};
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
///
/// This is the whole budget, not one batch of it: the turn is built with
/// [`Agent::as_subtask`], which switches off the batch renewal a conversation
/// enjoys. A subtask promised twelve steps must not quietly spend thirty-six —
/// that is a caller waiting half an hour for a `delegate` the description said
/// would come back sooner, which is the shape of the bug this budget exists to
/// prevent.
///
/// This is the shipped *default*; the number in force is
/// `subagent_max_steps` in the settings, read at registration.
pub const MAX_STEPS: usize = 12;

/// How long one subtask may run, wall clock, before it is cut off.
///
/// Steps bound how many model rounds a subtask takes; they say nothing about how
/// long one round lasts. A slow route, a gateway that hangs, a tool of its own —
/// each is inside the step count and outside any clock the parent can see. The
/// parent is simply waiting: its tool call has not returned, and nothing in its
/// transcript says whether the subtask is slow or wedged.
///
/// So the subtask carries a wall clock of its own, and the tool description
/// states it, because a caller sizes a task against the budget it is given.
/// Generous on purpose — fifteen minutes is a dozen slow steps — and finite on
/// purpose: an answer that arrives is worth more than work that never does.
///
/// The shipped default; the number in force is `subagent_wall_secs` in the
/// settings, read when the task is launched.
pub const WALL_BUDGET_SECS: u64 = 15 * 60;

/// Where a subtask's own transcript is written, relative to the session
/// workspace.
///
/// A directory of the session's own, not the system temp: the point is that the
/// user can open it, and a path under the workspace is one they already know.
const TRACE_DIR: &str = ".ngu-subtasks";

/// How much of a tool's arguments, result, or a message body one trace line
/// keeps.
///
/// The trace is read by a person and possibly by a model, and it lives in the
/// session's workspace: a 2 MB `read` result must not become a 2 MB line in a
/// file somebody is tailing.
const TRACE_ARGUMENTS: usize = 400;
const TRACE_RESULT: usize = 600;
const TRACE_REASON: usize = 300;
const TRACE_MESSAGE: usize = 4000;

/// Distinguishes two subtasks that started in the same millisecond.
static TRACE_SEQ: AtomicUsize = AtomicUsize::new(0);

/// How many transcripts to keep in [`TRACE_DIR`].
///
/// The directory sits in the session workspace, and a conversation that
/// delegates a lot writes one file per call — unbounded, it becomes the largest
/// thing in a workspace nobody meant to fill with traces. The oldest go first
/// because the newest is the one being read; fifty is enough to look back over
/// a working session.
const TRACE_KEEP: usize = 50;

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

/// What `delegate` tells the model about itself.
///
/// A function of [`crate::tools::Budgets`] rather than a literal, so the
/// sentence states the numbers in force — and so a test can ask what the
/// description would say at any budget without rewriting a process global under
/// the other tests' feet.
pub(crate) fn description(budgets: crate::tools::Budgets) -> String {
    let steps = budgets.subagent_max_steps;
    let minutes = (budgets.subagent_wall_ms / 60_000).max(1);
    let depth = budgets.subagent_max_depth;
    format!(
        "Run a self-contained subtask in a fresh agent with no memory of this \
         conversation, and get back only its answer. Use it to keep a large \
         side investigation out of this transcript. \
         The subtask sees the same tools and the same model you do, but not \
         what was said here — so state everything it needs, including paths \
         and what 'done' looks like. It cannot ask you anything. \
         It stops after {steps} steps or {minutes} minutes, whichever comes \
         first — size the task to fit, and split it rather than promising more \
         than it can finish. Both budgets are yours to move: `subagent_max_steps` \
         and `subagent_wall_secs` in the settings panel; subagents may nest \
         {depth} deep (`subagent_max_depth`). \
         It is silent while it runs: every step and tool \
         call is appended to its own transcript under {TRACE_DIR}/ in the \
         working directory, named in the result, so read that file to see what \
         it is doing while it works."
    )
}

fn tool(access: Arc<dyn ModelAccess>) -> ToolDef {
    // The budgets the caller is told about are the ones in force: registration
    // runs after the host has published them, so this states real numbers.
    let budgets = super::budgets();
    ToolDef::new(
        TOOL,
        description(budgets),
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
    let budgets = super::budgets();
    if current >= budgets.subagent_max_depth {
        return Err(anyhow!(
            "subagents are already {} deep; this one may not delegate further. Do the \
             work here instead, or raise `subagent_max_depth` in the settings panel.",
            budgets.subagent_max_depth
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
            access.subagent_prompt(),
            access.tools(),
            client.model(),
        )
        .with_policy(access.context_policy()),
    );

    // The subtask's own framing, ahead of the task itself. It runs under the
    // session's standing instructions, and a persona that tells its model to
    // split work up and hand the critique to another agent would spend this
    // subtask's budget spawning a tree of its own — each layer invisible to the
    // user, each taking steps. One paragraph says which job this is; the task
    // stays the caller's words, verbatim, below it.
    let prompt = format!(
        "You are a subagent running one self-contained subtask. Do this work \
         yourself with the tools you have and end your turn with the answer — \
         no further subagents, no questions back.\n\nSubtask:\n{task}"
    );

    // A transcript nobody has to wait for. While the subtask runs, the parent's
    // transcript shows only a tool call that has not returned; this file is the
    // witness, named in the result below, so a slow subtask is distinguishable
    // from a wedged one and a bad answer can be diagnosed rather than guessed.
    let trace = new_trace_path();
    let observer: Arc<dyn AgentObserver> = match &trace {
        Some(path) => Arc::new(Trace::new(path.clone())),
        None => Arc::new(SilentObserver),
    };

    // The wall clock. Steps count rounds; they do not bound one round, and a
    // parent waiting on this call cannot tell "slow" from "never returning".
    // Both numbers come from settings, and the stop reason says so: the parent
    // model reads that text, so it should learn which knob moves, not that some
    // anonymous user set a limit.
    let budgets = super::budgets();
    let wall_secs = budgets.subagent_wall_ms / 1000;
    let (stop, stopped) = tokio::sync::watch::channel(false);
    let deadline = tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_millis(budgets.subagent_wall_ms)).await;
        let _ = stop.send(true);
    });

    // An empty history, then the task as the first user message. The turn is a
    // subtask: an exact step budget — no batch renewal — and no hold from the
    // parent's mission, because the parent's turn is the one that owes the
    // mission its time.
    let mut agent = Agent::new(client, config, Vec::new())
        .with_max_steps(budgets.subagent_max_steps)
        .as_subtask()
        .with_observer(observer)
        .with_cancel(stopped)
        .with_stop_reason(format!(
            "the subtask ran out of its {wall_secs}-second wall clock (`subagent_wall_secs`) \
             or its {}-step budget (`subagent_max_steps`) before it finished — both are \
             editable in the settings panel, and a narrower task would also fit",
            budgets.subagent_max_steps
        ));
    let outcome = agent.run(&prompt).await;
    // The budget no longer matters once the subtask is over; a sleeping timer
    // that outlived it would be work nobody cancels.
    deadline.abort();
    let outcome = outcome?;

    let trace_note = match &trace {
        Some(path) => format!(
            "\n\nIts own transcript — every step, tool call and message, in order — \
             is at {}.",
            path.display()
        ),
        None => String::new(),
    };

    // A failed subtask reports the failure as the tool result, carrying any
    // partial answer with it: the parent model reads what went wrong instead
    // of a bare transport error surfacing as a crashed tool.
    if let Some(error) = &outcome.error {
        let mut message = format!("The subtask failed: {error}");
        if !outcome.text.trim().is_empty() {
            message.push_str("\n\nPartial answer before the failure:\n");
            message.push_str(&outcome.text);
        }
        message.push_str(&trace_note);
        return Err(anyhow!(message));
    }

    let mut text = format!(
        "The subtask finished after {} step(s), {} tool call(s) and {} retry(ies) \
         (model {} ms, tools {} ms).{}\n\n{}",
        outcome.steps,
        outcome.tool_calls,
        outcome.retries,
        outcome.timing.model_ms,
        outcome.timing.tools_ms,
        trace_note,
        outcome.text
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

/// A fresh file for one subtask's transcript, or `None` when there is no
/// workspace to put it in.
///
/// The path resolves from the identity the *call* runs under, so it lands in
/// the session the parent conversation belongs to — the place its user already
/// knows how to open.
fn new_trace_path() -> Option<std::path::PathBuf> {
    let workspace = crate::tools::current_workspace()?;
    let dir = workspace.join(TRACE_DIR);
    if let Err(error) = std::fs::create_dir_all(&dir) {
        // A subtask must not fail because its transcript cannot be written:
        // the answer is the product, the transcript is the evidence.
        eprintln!("[subagent] no trace directory {}: {error}", dir.display());
        return None;
    }
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|since| since.as_millis())
        .unwrap_or(0);
    let seq = TRACE_SEQ.fetch_add(1, Ordering::Relaxed);
    prune_traces(&dir, TRACE_KEEP);
    Some(dir.join(format!("{stamp}-{seq}.jsonl")))
}

/// Keep the newest `keep` transcripts in `dir`, removing the rest.
///
/// Names are `<millis>-<seq>.jsonl`, so ordering by name is ordering by time and
/// no metadata has to be read per file. A removal that fails is ignored: a stale
/// trace is a smaller problem than a subtask that cannot start.
fn prune_traces(dir: &std::path::Path, keep: usize) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut names: Vec<String> = entries
        .flatten()
        .filter_map(|entry| entry.file_name().into_string().ok())
        .filter(|name| name.ends_with(".jsonl"))
        .collect();
    if names.len() <= keep {
        return;
    }
    // Oldest first, so `take(excess)` removes exactly the files that pushed the
    // directory over the bound.
    names.sort();
    let excess = names.len() - keep;
    for name in names.into_iter().take(excess) {
        let _ = std::fs::remove_file(dir.join(name));
    }
}

/// Writes the subtask's own transcript, one JSONL line per event.
///
/// A subtask is silent while it runs — nothing it says enters the parent's
/// transcript until it answers — so a parent waiting on `delegate` sees a tool
/// call that has not come back and has no way to tell a slow subtask from a
/// wedged one. This file is that witness: every step, tool call, retry and
/// message appends a line, and the tool result names the path, so the user or
/// the parent model can read what the subtask is doing *while* it works.
///
/// Failures are reported and swallowed. A trace is evidence, not a product: an
/// unwritable file must not fail the subtask it was watching.
struct Trace {
    path: std::path::PathBuf,
}

impl Trace {
    fn new(path: std::path::PathBuf) -> Self {
        Self { path }
    }

    /// Append one record.
    fn line(&self, kind: &str, payload: Value) {
        let record = json!({
            "ts": std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|since| since.as_millis())
                .unwrap_or(0),
            "kind": kind,
            "payload": payload,
        });
        let Ok(mut line) = serde_json::to_string(&record) else {
            return;
        };
        line.push('\n');
        let written = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .and_then(|mut file| file.write_all(line.as_bytes()));
        if let Err(error) = written {
            eprintln!("[subagent] trace {}: {error}", self.path.display());
        }
    }
}

impl AgentObserver for Trace {
    fn on_step(&self, step: usize) {
        self.line("step", json!({ "step": step }));
    }

    fn on_tool_start(&self, name: &str, arguments: &str) {
        self.line(
            "tool.start",
            json!({
                "tool": name,
                "arguments": crate::events::truncate(arguments, TRACE_ARGUMENTS),
            }),
        );
    }

    fn on_tool_end(&self, name: &str, ok: bool, result: &str) {
        self.line(
            "tool.end",
            json!({
                "tool": name,
                "ok": ok,
                "result": crate::events::truncate(result, TRACE_RESULT),
            }),
        );
    }

    fn on_step_retry(&self, attempt: usize, reason: &str) {
        self.line(
            "retry",
            json!({
                "attempt": attempt,
                "reason": crate::events::truncate(reason, TRACE_REASON),
            }),
        );
    }

    fn on_message(&self, message: &crate::message::Message) {
        // The heart of it: the subtask's own transcript, in order, including
        // the final answer the parent is about to receive.
        self.line(
            "message",
            json!({
                "role": message.role.as_wire(),
                "tool_calls": message
                    .tool_calls
                    .iter()
                    .map(|call| call.name.as_str())
                    .collect::<Vec<_>>(),
                "text": crate::events::truncate(message.text(), TRACE_MESSAGE),
            }),
        );
    }
}

/// How deep delegation may nest — the shipped default; the number in force is
/// `subagent_max_depth` in the settings, read on every call.
pub const MAX_DEPTH: usize = 2;

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

    /// The sentence the model reads has to follow the settings. A description
    /// stating a number the machine no longer uses is how a caller sizes a task
    /// against a budget that does not exist — which is why this text is built
    /// from [`crate::tools::Budgets`] and tested at two different budgets.
    #[test]
    fn the_description_states_the_budgets_it_was_built_with() {
        let shipped = description(crate::tools::Budgets::default());
        assert!(shipped.contains("12 steps"), "{shipped}");
        assert!(shipped.contains("15 minutes"), "{shipped}");
        assert!(shipped.contains("2 deep"), "{shipped}");

        let moved = description(crate::tools::Budgets {
            subagent_max_steps: 40,
            subagent_wall_ms: 45 * 60_000,
            subagent_max_depth: 5,
            ..crate::tools::Budgets::default()
        });
        assert!(moved.contains("40 steps"), "{moved}");
        assert!(moved.contains("45 minutes"), "{moved}");
        assert!(moved.contains("5 deep"), "{moved}");
        assert!(!moved.contains("12 steps"), "the old number must be gone");
        // And each one names where to move it.
        assert!(moved.contains("subagent_max_steps"), "{moved}");
        assert!(moved.contains("subagent_wall_secs"), "{moved}");
        assert!(moved.contains("subagent_max_depth"), "{moved}");
    }

    #[test]
    fn a_wall_clock_of_under_a_minute_is_still_promised_as_a_minute() {
        let text = description(crate::tools::Budgets {
            subagent_wall_ms: 1_000,
            ..crate::tools::Budgets::default()
        });
        assert!(text.contains("1 minutes"), "{text}");
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

    #[test]
    fn the_description_states_both_budgets_and_where_the_trace_goes() {
        // A caller sizes its task against what it is told it is buying: how many
        // steps, and how long. A budget discovered by waiting — the bundle of
        // twenty minutes nobody warned about — is the bug this description
        // exists to prevent, and a silent wait must be announced as silent so
        // that it is not read as a hang.
        let description = schema_of(&loaded_kernel())["function"]["description"]
            .as_str()
            .unwrap()
            .to_string();
        assert!(
            description.contains(&MAX_STEPS.to_string()),
            "{description}"
        );
        assert!(
            description.contains(&format!("{} minutes", WALL_BUDGET_SECS / 60)),
            "{description}"
        );
        assert!(description.contains(TRACE_DIR), "{description}");
        assert!(description.contains("silent"), "{description}");
    }

    #[tokio::test]
    async fn the_trace_records_what_the_subtask_did() {
        // A subtask's answer is one message; what it did on the way is a file.
        // That file is the only thing a user can read while `delegate` is still
        // waiting, and the only way to diagnose an answer that came back thin.
        let dir = std::env::temp_dir().join(format!(
            "ngu-subagent-trace-{}",
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&dir).expect("a scratch directory");
        let path = dir.join("trace.jsonl");

        let trace = Trace::new(path.clone());
        trace.on_step(1);
        trace.on_tool_start("read", r#"{"path":"notes.md"}"#);
        trace.on_tool_end("read", true, &"x".repeat(TRACE_RESULT * 2));
        trace.on_step_retry(2, "the provider returned an empty response");
        trace.on_message(&crate::message::Message::user("write the critique"));

        let text = std::fs::read_to_string(&path).expect("the trace was written");
        let records: Vec<Value> = text
            .lines()
            .map(|line| serde_json::from_str(line).expect("every line is JSON"))
            .collect();
        assert_eq!(records.len(), 5, "{text}");

        assert_eq!(records[0]["kind"], "step");
        assert_eq!(records[0]["payload"]["step"], 1);
        assert_eq!(records[1]["kind"], "tool.start");
        assert_eq!(records[1]["payload"]["tool"], "read");
        assert_eq!(records[2]["kind"], "tool.end");
        assert_eq!(records[2]["payload"]["ok"], true);
        assert!(
            records[2]["payload"]["result"].as_str().unwrap().len() < TRACE_RESULT * 2,
            "a huge tool result is cut, not copied whole into the trace"
        );
        assert_eq!(records[3]["kind"], "retry");
        assert_eq!(records[4]["kind"], "message");
        assert_eq!(records[4]["payload"]["role"], "user");
        assert_eq!(records[4]["payload"]["text"], "write the critique");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn old_transcripts_are_pruned_and_newer_ones_kept() {
        // The trace directory lives in the user's workspace: a session that
        // delegates a lot writes one file per call, and nothing else would ever
        // clean it up. Pruning by name is pruning by time, because the stamp
        // leads the name and is fixed width.
        let dir = std::env::temp_dir().join(format!(
            "ngu-subagent-prune-{}",
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&dir).expect("a scratch directory");
        for n in 0..(TRACE_KEEP + 5) {
            std::fs::write(dir.join(format!("{n:013}-0.jsonl")), "{}").expect("a trace");
        }
        // Not a trace: never the pruner's business.
        std::fs::write(dir.join("notes.txt"), "keep me").expect("a bystander");

        prune_traces(&dir, TRACE_KEEP);

        let mut kept: Vec<String> = std::fs::read_dir(&dir)
            .expect("the directory reads")
            .flatten()
            .filter_map(|entry| entry.file_name().into_string().ok())
            .collect();
        kept.sort();
        assert_eq!(kept.len(), TRACE_KEEP + 1, "{kept:?}");
        assert!(kept.contains(&"notes.txt".to_string()), "{kept:?}");
        assert!(
            !kept.contains(&format!("{:013}-0.jsonl", 0)),
            "the oldest went first: {kept:?}"
        );
        assert!(
            kept.contains(&format!("{:013}-0.jsonl", TRACE_KEEP + 4)),
            "the newest survived: {kept:?}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
