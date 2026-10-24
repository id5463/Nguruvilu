//! Application state and command handling for the desktop shell.
//!
//! The window is a view over the same kernel the CLI drives. Nothing here
//! reimplements kernel behaviour: it holds a [`Kernel`], a [`Runtime`], and a
//! session, and translates UI commands into calls on them.
//!
//! Threading: the window and the webview live on the tao event loop thread,
//! while agent turns run on a tokio runtime. They meet at one seam — a
//! [`EventSink`] — the page in window mode, stdout in headless mode.

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Result};
use serde_json::{json, Value};

use nguruvilu::agent::{Agent, AgentObserver, TurnConfig, TurnSettings};
use nguruvilu::hotreload::{Change, ChangePayload, ModelRoute, Runtime};
use nguruvilu::llm::LlmClient;
use nguruvilu::plugin::Kernel;
use nguruvilu::session::{ensure_workspace, JsonlStore, Session, SessionStore};
use nguruvilu::settings::Settings;
use nguruvilu::window::ContextPolicy;
use nguruvilu::skills::{register_skill_tool, SkillRegistry};

use crate::sink::EventSink;

/// System prompt for the desktop shell.
pub const BASE_PROMPT: &str = "You are a coding agent. Use the tools to inspect and change \
files and to run commands. Prefer `read` over shelling out to view files, and `edit` for \
surgical changes. Keep answers short and concrete. When you tell the user about a file, \
give its absolute path — a bare or relative name is not something they can open.";

/// Newest transcript entries sent to the page on connect and on each refresh.
/// Older history is fetched on demand as the page scrolls up.
const TRANSCRIPT_PAGE: usize = 40;

/// How many per-turn usage rows the `stats` command returns.
///
/// The totals cover the whole session; the tail exists so a spike can be
/// traced to the turn that caused it. Twenty is a screenful, not a ledger.
const STATS_TAIL: usize = 20;

/// The wire shape of a session's cumulative usage.
///
/// Built field by field rather than serialising the Rust struct: the page's
/// names are part of the IPC contract, and `cache_hit_ratio` is computed here
/// (as a fraction, 0..=1) so no client has to repeat the division.
fn totals_json(totals: &nguruvilu::session::SessionTotals) -> Value {
    json!({
        "turns": totals.turns,
        "steps": totals.steps,
        "tool_calls": totals.tool_calls,
        "input": totals.input,
        "output": totals.output,
        "cached": totals.cached,
        "cache_created": totals.cache_created,
        "reasoning": totals.reasoning,
        "model_ms": totals.model_ms,
        "tools_ms": totals.tools_ms,
        "cache_hit_ratio": totals.cache_hit_ratio(),
    })
}

/// One stored turn's usage, in the same field names as the totals.
fn turn_stats_json(stats: &nguruvilu::session::TurnStats) -> Value {
    json!({
        "ts": stats.ts,
        "steps": stats.steps,
        "tool_calls": stats.tool_calls,
        "retries": stats.retries,
        "input": stats.input,
        "output": stats.output,
        "cached": stats.cached,
        "cache_created": stats.cache_created,
        "reasoning": stats.reasoning,
        "model_ms": stats.model_ms,
        "tools_ms": stats.tools_ms,
    })
}

/// One session's ledger as the page reads it: what the conversation cost, and
/// the turns that made up the cost.
///
/// One definition for both entry points — [`AppState::stats`], which answers
/// while the state lock is held, and [`AppState::stats_outside_lock`], which
/// reads the rows first and formats after the lock is gone — so the two cannot
/// answer different JSON for the same session. `totals` arrives already built
/// because the totals are summed from a whole session record, which the caller
/// has and this function does not need.
fn stats_json(id: &str, totals: Value, stats: &[nguruvilu::session::TurnStats]) -> Value {
    // Newest first for the tail, then back to chronological order: a log reads
    // forward, and the page lists it as it happened.
    let turns: Vec<Value> = stats
        .iter()
        .rev()
        .take(STATS_TAIL)
        .rev()
        .map(turn_stats_json)
        .collect();
    json!({
        "session": id,
        "totals": totals,
        "turns": turns,
    })
}

/// The `[first, end)` range of the page that ends just before `before`, for a
/// transcript of `total` entries. `before` is clamped to `total`, so a stale
/// request cannot index past the log.
fn page_before(total: usize, before: usize) -> (usize, usize) {
    let end = before.min(total);
    let first = end.saturating_sub(TRANSCRIPT_PAGE);
    (first, end)
}

#[cfg(test)]
mod page_tests {
    use super::{page_before, TRANSCRIPT_PAGE};

    #[test]
    fn the_first_page_before_the_tail_is_the_page_just_above_it() {
        // A 100-entry log: the page shown starts at 60, so the next page up
        // covers 20..60.
        assert_eq!(page_before(100, 60), (20, 60));
    }

    #[test]
    fn the_page_at_the_start_of_the_log_is_shorter_than_a_full_page() {
        assert_eq!(page_before(100, 10), (0, 10));
    }

    #[test]
    fn an_out_of_range_offset_clamps_to_the_log_instead_of_panicking() {
        assert_eq!(page_before(5, 999), (0, 5));
    }

    #[test]
    fn a_full_page_is_always_the_page_size() {
        let (first, end) = page_before(500, 400);
        assert_eq!(end - first, TRANSCRIPT_PAGE);
    }
}

#[cfg(test)]
mod stats_tests {
    use super::{turn_stats_json, totals_json};
    use nguruvilu::session::{SessionTotals, TurnStats};

    /// The page reads these names; renaming one here without renaming it in
    /// the script blanks the usage line, so the keys themselves are the test.
    #[test]
    fn the_totals_wire_shape_carries_every_field_the_page_draws() {
        let totals = SessionTotals {
            turns: 3,
            steps: 7,
            tool_calls: 4,
            input: 1000,
            output: 200,
            cached: 250,
            cache_created: 120,
            reasoning: 60,
            model_ms: 1500,
            tools_ms: 400,
        };
        let json = totals_json(&totals);
        assert_eq!(json["turns"], 3);
        assert_eq!(json["input"], 1000);
        assert_eq!(json["output"], 200);
        assert_eq!(json["cached"], 250);
        assert_eq!(json["cache_created"], 120);
        assert_eq!(json["reasoning"], 60);
        assert_eq!(json["model_ms"], 1500);
        assert_eq!(json["tools_ms"], 400);
        // A fraction, not a percentage: the page multiplies.
        assert!((json["cache_hit_ratio"].as_f64().unwrap() - 0.25).abs() < 1e-9);
    }

    #[test]
    fn a_session_with_nothing_recorded_reports_zero_rather_than_dividing_by_zero() {
        let json = totals_json(&SessionTotals::default());
        assert_eq!(json["turns"], 0);
        assert_eq!(json["cache_hit_ratio"], 0.0);
    }

    #[test]
    fn one_turn_keeps_its_own_counts_when_it_is_sent_as_a_row() {
        let stats = TurnStats {
            ts: "2026-10-24T00:00:00.000Z".to_string(),
            steps: 2,
            tool_calls: 1,
            retries: 1,
            input: 500,
            output: 90,
            cached: 100,
            cache_created: 40,
            reasoning: 30,
            model_ms: 700,
            tools_ms: 50,
        };
        let json = turn_stats_json(&stats);
        assert_eq!(json["ts"], "2026-10-24T00:00:00.000Z");
        assert_eq!(json["steps"], 2);
        assert_eq!(json["retries"], 1);
        assert_eq!(json["cached"], 100);
        assert_eq!(json["cache_created"], 40);
        assert_eq!(json["model_ms"], 700);
    }
}

/// The session's route, published so a plugin can build a subagent on it.
///
/// Reads through to the same `Runtime` the turn uses, so a plugin sees the
/// model and proxy in force now rather than the ones at startup.
struct DesktopModelAccess {
    runtime: Arc<Mutex<Runtime>>,
    skills: Arc<Mutex<SkillRegistry>>,
    base_prompt: String,
    policy: Arc<Mutex<Arc<dyn ContextPolicy>>>,
}

impl nguruvilu::model::ModelAccess for DesktopModelAccess {
    fn route(&self) -> nguruvilu::hotreload::ModelRoute {
        self.runtime.lock().expect("runtime lock").snapshot().model_route
    }

    fn client(&self) -> Result<LlmClient> {
        self.client_for(&self.route().model)
    }

    fn client_for(&self, model: &str) -> Result<LlmClient> {
        // The route's own builder, so a subtask is sent with the same proxy,
        // effort, output ceiling, extra fields, and connection behaviour the
        // session itself is using right now.
        self.route().client_for(model)
    }

    fn tools(&self) -> Arc<nguruvilu::tools::ToolRegistry> {
        self.runtime.lock().expect("runtime lock").snapshot().tools
    }

    fn system_prompt(&self) -> String {
        let mut prompt = self
            .runtime
            .lock()
            .expect("runtime lock")
            .snapshot()
            .system_prompt(&self.base_prompt);
        if let Some(catalog) = self.skills.lock().expect("skills lock").catalog() {
            prompt.push_str("\n\n");
            prompt.push_str(&catalog);
        }
        prompt
    }

    fn context_policy(&self) -> Arc<dyn ContextPolicy> {
        Arc::clone(&self.policy.lock().expect("policy lock"))
    }
}

/// State the agent loop reads at the start of each turn.
///
/// This is the desktop shell's [`TurnConfig`]: it composes the runtime's persona
/// and route with the skill catalog, exactly as the CLI does.
pub struct SharedConfig {
    /// The hot-reload runtime owning persona, route, and the tool table.
    ///
    /// Shared with the published model service, so a plugin sees the route in
    /// force now rather than the one at startup.
    pub runtime: Arc<Mutex<Runtime>>,
    /// Base system prompt.
    pub base_prompt: String,
    /// Skill registry, scanned at boot.
    pub skills: Arc<Mutex<SkillRegistry>>,
    /// Context window, threshold, and how much survives a compaction.
    /// Rebuilt whenever the settings panel saves, so a window change takes
    /// effect on the next turn.
    pub policy: Arc<Mutex<Arc<dyn ContextPolicy>>>,
    /// Extra fragments to place in each request.
    pub injection: Mutex<Arc<nguruvilu::context::InjectionEngine>>,
    /// Working directory of the session the next turn runs in. Changed on
    /// session switch so the model is told where it is, and the tools agree.
    pub cwd: Mutex<Option<String>>,
}

impl SharedConfig {
    /// Point the model's stated working directory at the given session directory.
    pub fn set_cwd(&self, cwd: Option<String>) {
        *self.cwd.lock().expect("cwd lock") = cwd;
    }
}

impl TurnConfig for SharedConfig {
    fn settings(&self) -> TurnSettings {
        let runtime = self.runtime.lock().expect("runtime lock");
        let snapshot = runtime.snapshot();
        let mut system_prompt = snapshot.system_prompt(&self.base_prompt);
        if let Some(cwd) = self.cwd.lock().expect("cwd lock").as_deref() {
            system_prompt.push_str(&format!(
                "\n\nYour working directory for this conversation is {cwd}. \
                 Relative paths resolve against it, and files the user uploads land in its uploads/ folder."
            ));
        }
        if let Some(catalog) = self.skills.lock().expect("skills lock").catalog() {
            system_prompt.push_str("\n\n");
            system_prompt.push_str(&catalog);
        }
        TurnSettings {
            system_prompt,
            tools: snapshot.tools,
            model: snapshot.model_route.model,
            version: snapshot.version,
            policy: Arc::clone(&self.policy.lock().expect("policy lock")),
            injection: Arc::clone(&self.injection.lock().expect("injection lock")),
            cache_policy: snapshot.cache_policy,
        }
    }
}

/// Everything the shell knows.
/// One notify job's owner and the event stream of that owner's session.
///
/// Stored per job so the settlement hook can wake the right session through the
/// right sink (the install-time sink is only correct for the window/headless
/// paths; for SSE it serves just the viewing client), and so cross-session
/// reads/kills against the process-global kernel registry can be filtered here.
pub struct AppJob {
    /// The session whose turn started the job (its wake target).
    pub session: String,
    /// That session's live event stream.
    pub sink: Arc<dyn crate::sink::EventSink>,
}

pub struct AppState {
    /// The plugin kernel, owning the tool table.
    pub kernel: Kernel,
    /// Hot-reload runtime.
    pub config: Arc<SharedConfig>,
    /// The session currently shown.
    pub session: Session,
    /// Durable session storage.
    pub store: JsonlStore,
    /// Model route settings, for rebuilding clients.
    pub route: ModelRoute,
    /// Sessions with a turn running right now.
    ///
    /// Keyed by session id rather than one flag: a session that is working
    /// must not lock out a different one, and the id is what routes a turn's
    /// events back to the right history when the user switches away mid-run.
    pub running: HashSet<String>,
    /// Stop requests for the sessions that are running right now.
    ///
    /// The sender lives here so any command can flip the flag while the turn
    /// runs unguarded; the turn itself holds only the receiving end and checks
    /// it at its next safe point.
    pub cancels: HashMap<String, tokio::sync::watch::Sender<bool>>,
    /// Messages typed while a session's turn runs.
    ///
    /// Delivered at the turn's next step boundary (steering); whatever arrives
    /// after the last claim is chained into a follow-up turn when the current
    /// one completes, so nothing the user sent is stranded.
    pub inboxes: HashMap<String, Arc<std::sync::Mutex<VecDeque<String>>>>,
    /// Background jobs that asked to wake their session, by job id.
    ///
    /// Attribution happens where the session is known: the turn's observer
    /// pairs each `bash` call's start (`notify` flag) with its end (job id).
    /// The settlement hook looks the session up here and removes the entry; a
    /// job without one never asked to be woken.
    pub job_sessions: HashMap<String, String>,
    /// Every notify job's owner and that owner's live event stream.
    ///
    /// Written where attribution already happens (the turn's `UiObserver`, the
    /// only place that knows both the session and its sink) and read by the
    /// settlement hook. The kernel's job registry is process-global — jobs from
    /// every session land in one table — so cross-session reads and kills are
    /// filtered by this map, and the wake uses the *owner's* sink rather than
    /// whatever sink happened to install the hook (which is wrong for the
    /// multi-session SSE path, where one sink serves only the viewing client).
    pub app_jobs: HashMap<String, AppJob>,

    /// Effective settings, as the settings panel should display them.
    pub settings: Settings,
    /// Prompt tokens the provider last reported, for the context readout.
    pub last_prompt_tokens: Option<usize>,
    /// The last failure, kept here rather than only in the page: a message
    /// drawn into the DOM vanishes the moment the transcript is re-rendered,
    /// which is exactly when a user switches sessions to look for it.
    pub last_error: Option<String>,
    /// Pack loads and unloads the agent asked for during a turn.
    ///
    /// The tool records what it wants while the turn owns the kernel; the
    /// shell drains this between turns, which is the next moment the tool
    /// table is read anyway.
    pub pending: Arc<nguruvilu::tools::pack::PendingQueue>,
    /// The MCP servers the app started.
    ///
    /// Kept here rather than only in the tool table so a server survives its
    /// pack's reload, is stopped when the pack unloads, and is stopped for
    /// good when the window closes.
    pub mcp: Vec<Arc<nguruvilu::mcp::McpClient>>,
    /// The search plugin's settings cell.
    ///
    /// The plugin holds this and reads it when its fibers apply; the panel
    /// writes it through `search::configure`, which also re-applies them.
    pub search_cell: nguruvilu::tools::search::SettingsCell,
    /// The judge plugin's settings cell — same contract as search.
    pub judge_cell: nguruvilu::tools::judge::SettingsCell,
    /// The user's own settings, before any pack folded over them.
    ///
    /// [`recompute_pack_content`](Self::recompute_pack_content) starts from
    /// this baseline on every load and unload, so an unloaded pack's route
    /// and window leave with it while the user's stored values come back.
    /// A save from the settings panel replaces it: what the user typed is
    /// their own choice, and the next pack toggle must not undo it.
    pub pack_baseline: Settings,
}

impl AppState {
    /// Build the shell's state from the environment.
    pub fn bootstrap() -> Result<Self> {
        // Settings, not raw environment variables: the shell has to be usable by
        // someone who has never set one, and changeable without a restart.
        let settings = Settings::resolve();
        let model = settings.model_or_default();

        // Skills are scanned once at boot; the catalog is part of the prompt.
        let mut skills = SkillRegistry::with_roots(SkillRegistry::default_roots());
        let scan = skills.scan();

        let mut kernel = Kernel::new();
        // The agent can build, install, load, and unload packs itself; the
        // shell applies the load and unload between turns.
        let pending = Arc::new(nguruvilu::tools::pack::PendingQueue::new());
        nguruvilu::tools::pack::register_with(kernel.tools_mut(), Some(Arc::clone(&pending)))?;
        // Available to a pack, not loaded by the kernel: `packs/subagent` asks
        // for `builtin:delegate` and it appears.
        nguruvilu::tools::subagent::define(&mut kernel);
        // Same rule for long tasks (长程任务): `packs/long-horizon` asks for
        // `builtin:long-horizon` and the mission tools appear — missions,
        // recall, mailboxes, and the tick that runs one step of one in a
        // fresh worker.
        nguruvilu::tools::longhorizon::define(&mut kernel);
        // Same for document reading (读不了就装包): `packs/documents` asks for
        // `builtin:documents` and `read` learns PDF/Office.
        nguruvilu::tools::documents::define(&mut kernel);
        // Available to a pack, not loaded by the kernel: `packs/search` asks
        // for `builtin:search` and it appears — with a tool only while a key
        // is configured (see `search::configure`).
        let search_cell = nguruvilu::tools::search::cell(settings.search.clone());
        nguruvilu::tools::search::install(&mut kernel, search_cell.clone());
        // The judge follows the same rule: code available, pack decides,
        // settings from the host's cell.
        let judge_cell = nguruvilu::tools::judge::cell(settings.judge.clone());
        nguruvilu::tools::judge::install(&mut kernel, judge_cell.clone());
        // The audit log: every event appended to <data>/events.jsonl for this
        // process. Dropping the returned disposer is deliberate — a disposer
        // unsubscribes only when called, never when dropped.
        let _audit = nguruvilu::events::install_audit(
            &kernel.events(),
            nguruvilu::events::audit_path(),
        );
        if !skills.is_empty() {
            register_skill_tool(kernel.tools_mut(), Arc::new(skills.clone()))?;
        }

        let tools = Arc::new(kernel.tools().clone());
        // One route from the settings, used both by the runtime a turn reads
        // and by the state the panel edits, so the two cannot drift.
        let route = ModelRoute::from_settings(&settings);
        let runtime = Runtime::new(tools).with_route(route.clone());

        let policy: Arc<dyn ContextPolicy> = Arc::new(settings.context_policy(None));

        let runtime = Arc::new(Mutex::new(runtime));
        let skills = Arc::new(Mutex::new(skills));
        let policy = Arc::new(Mutex::new(policy));

        // Published before any pack loads, so a plugin that wants to spawn a
        // subagent finds the service already there.
        let access: Arc<dyn nguruvilu::model::ModelAccess> =
            Arc::new(DesktopModelAccess {
                runtime: Arc::clone(&runtime),
                skills: Arc::clone(&skills),
                base_prompt: BASE_PROMPT.to_string(),
                policy: Arc::clone(&policy),
            });
        nguruvilu::model::install(&mut kernel, Arc::clone(&access))?;

        let config = Arc::new(SharedConfig {
            runtime,
            base_prompt: BASE_PROMPT.to_string(),
            skills,
            policy,
            injection: Mutex::new(Arc::new(nguruvilu::context::InjectionEngine::load_default())),
            cwd: Mutex::new(None),
        });

        let store = JsonlStore::open(sessions_root())?;

        // Resume the most recent session when there is one: a desktop app that
        // forgets the conversation on every launch is not much of a desktop app.
        let mut session = match store.latest()? {
            Some(summary) => store.load(&summary.id)?.unwrap_or_else(|| Session::new(Some(model.clone()))),
            None => Session::new(Some(model.clone())),
        };
        let stored = store.root().join(format!("{}.jsonl", session.id)).exists();
        if !stored {
            session.cwd = Some(ensure_workspace(&workspaces_root()?, &session.id)?.display().to_string());
            store.create(&session)?;
        } else {
            session.cwd = Some(adopt_workspace_for(&session)?);
        }

        // The state of the configuration belongs on stderr at startup: it is
        // the first thing to check when a request fails with a 403 or a 401.
        if settings.is_configured() {
            eprintln!(
                "[settings] endpoint {} | model {} | key {} | file {}",
                settings.base_url,
                settings.model_or_default(),
                settings.masked_key(),
                Settings::path().display()
            );
        } else {
            eprintln!(
                "[settings] NOT CONFIGURED (missing: {}) — fill it in from the Settings panel, or run: ngu config set --base-url <url> --api-key <key>",
                settings.missing().join(", ")
            );
        }

        // A scan no longer aborts on one bad root or entry: the skills it did
        // find stay in the catalog and the failures arrive as `Ok(report)`.
        // That makes the partial result silent unless it is printed here, and
        // silence is the wrong answer to "my skills disappeared". The reason
        // is rendered by `summary` so this shell and every other front end
        // report a skipped path with the same words.
        match &scan {
            Ok(report) if !report.skipped.is_empty() => {
                eprintln!("[skills] {}", report.summary());
            }
            Ok(_) => {}
            Err(error) => eprintln!("[skills] scan failed: {error:#}"),
        }

        config.set_cwd(session.cwd.clone());

        // Before any pack folds: boot loads the packs after this point, so the
        // snapshot is exactly the user's own state a recompute returns to.
        let pack_baseline = settings.clone();
        let state = Self {
            kernel,
            config,
            session,
            store,
            route,
            running: HashSet::new(),
            cancels: HashMap::new(),
            inboxes: HashMap::new(),
            job_sessions: HashMap::new(),
            app_jobs: HashMap::new(),
            settings,
            last_error: None,
            last_prompt_tokens: None,
            pending,
            mcp: Vec::new(),
            search_cell,
            judge_cell,
            pack_baseline,
        };
        // The session this process boots into is already current: its start
        // belongs to bootstrap, not to the first prompt. The switch pins its
        // workspace on the tool table; the same id means no SESSION_END.
        let boot_workspace = session_workspace(&state.session)?;
        state.session_switch(&state.session.id.clone(), &boot_workspace);
        Ok(state)
    }

    /// A description of the current state, for the details panel and status bar.
    pub fn describe(&self) -> Value {
        // This session's mission board, read before any lock is taken: it is a
        // handful of small files in a directory that usually does not exist, and
        // the page paints it the moment a panel frame loads — a reader who starts a
        // mission has to see it without waiting for the next tool call.
        let missions = missions_workspace(
            self.store.root(),
            &self.session.id,
            self.session.cwd.as_deref(),
        )
        .map(|workspace| missions_under(&workspace))
        .unwrap_or_else(|| json!([]));
        let runtime = self.config.runtime.lock().expect("runtime lock");
        let snapshot = runtime.snapshot();
        let skills = self.config.skills.lock().expect("skills lock");
        // Sorted so the page's badge order does not shuffle between pushes.
        let mut running: Vec<&String> = self.running.iter().collect();
        running.sort();
        let running: Vec<String> = running.into_iter().cloned().collect();
        json!({
            "session": self.session.id,
            "title": self.session.title(),
            "messages": self.session.messages.len(),
            "model": snapshot.model_route.model,
            "reasoning_effort": self.settings.reasoning_effort,
            // The word the settings use for how an answer is carried; empty
            // means the default, `stream`.
            "delivery": self.settings.delivery,
            "proxy": self.settings.proxy,
            "max_output_tokens": self.settings.max_output_tokens.map(nguruvilu::size::format_size),
            "network": json!({
                "request_timeout_secs": self.settings.network.request_timeout_secs,
                "pool_idle_timeout_secs": self.settings.network.pool_idle_timeout_secs,
                "retry_attempts": self.settings.network.retry_attempts,
                "retry_backoff_ms": self.settings.network.retry_backoff_ms,
            }),
            "base_url": snapshot.model_route.base_url,
            "cache_policy": format!("{:?}", snapshot.cache_policy),
            "config_version": snapshot.version,
            "tools": snapshot.tools.names(),
            "skills": skills.list().iter().map(|s| json!({
                "id": s.id,
                "description": s.description,
            })).collect::<Vec<_>>(),
            "skill_roots": skills.roots().iter().map(|r| r.display().to_string()).collect::<Vec<_>>(),
            "plugins": self.kernel.plugin_names(),
            // Which pack brought each one: `delegate ← subagent`, so a
            // capability can be read back to the pack it came from.
            "plugin_origins": self.kernel.plugin_origins(),
            "services": self.kernel.service_list().iter().map(|(name, realm, owner)| json!({
                "name": name,
                "realm": realm.to_string(),
                "owner": owner,
            })).collect::<Vec<_>>(),
            // Busy means *this* session is working. Another session running a
            // turn is reported separately so the page can let one conversation
            // continue while a different one is busy.
            "busy": self.running.contains(&self.session.id),
            "running_sessions": running,
            // Every mission in this session's workspace, newest first. It is not
            // derived from anything the page can see: the mission tools write to
            // disk, and this is how the board reaches the page at open.
            "missions": missions,
            "configured": self.settings.is_configured(),
            "missing": self.settings.missing(),
            "api_key_masked": self.settings.masked_key(),
            // The panel sets these; the key itself never comes back.
            "search": match &self.settings.search {
                Some(search) => json!({
                    "provider": serde_json::to_value(&search.provider).unwrap_or(Value::Null),
                    "endpoint": search.endpoint,
                    "key": if search.api_key.trim().is_empty() { "(not set)" } else { "(set)" },
                }),
                None => json!({ "provider": "", "endpoint": "", "key": "(not set)" }),
            },
            // Same shape for the judge: endpoint and model are not secrets;
            // the key is reported as presence only.
            "judge": match &self.settings.judge {
                Some(judge) => json!({
                    "endpoint": judge.endpoint,
                    "model": judge.model,
                    "key": if judge.api_key.trim().is_empty() { "(not set)" } else { "(set)" },
                }),
                None => json!({ "endpoint": "", "model": "", "key": "(not set)" }),
            },
            "settings_path": Settings::path().display().to_string(),
            "last_error": self.last_error,
            // The session's cumulative usage, pushed with every status so the
            // page can draw the usage line without a second round trip. It is
            // a read of rows the session already loaded — no I/O, no lock of
            // its own.
            "totals": totals_json(&nguruvilu::session::totals_of(&self.session)),
            "context": {
                "window": self.config.policy.lock().expect("policy lock").window(&self.route.model).tokens,
                "window_source": self.config.policy.lock().expect("policy lock").window(&self.route.model).source.as_str(),
                "threshold_percent": self.settings.compact_percent,
                "keep_recent": self.settings.compact_keep_recent,
                "configured_window": self.settings.context_window,
                "last_prompt_tokens": self.last_prompt_tokens,
            },
        })
    }

    /// Stored sessions, newest first.
    pub fn sessions(&self) -> Result<Value> {
        let list = self.store.list()?;
        Ok(json!(list
            .iter()
            .map(|s| json!({
                "id": s.id,
                "title": s.title,
                "messages": s.message_count,
                "updated_at": s.updated_at,
                "current": s.id == self.session.id,
                "running": self.running.contains(&s.id),
            }))
            .collect::<Vec<_>>()))
    }

    /// The usage ledger of one session — totals plus its most recent turns.
    ///
    /// Read-only and answered from the rows in memory, so this never waits on a
    /// running turn and never writes. The rows of the open conversation were
    /// loaded with it.
    ///
    /// `session` defaults to the open conversation. Asking for another one
    /// re-reads it from the store, which is why one conversation's numbers
    /// cannot leak into another's — and which means the caller must be someone
    /// for whom a file read is acceptable: this runs wherever it is called,
    /// including with the state lock held. Use
    /// [`stats_outside_lock`](Self::stats_outside_lock) when the request may
    /// name a session that is not the open one.
    pub fn stats(&self, session: Option<&str>) -> Result<Value> {
        let requested = session.map(str::trim).filter(|id| !id.is_empty());
        let loaded;
        let record: &Session = match requested {
            Some(id) if id != self.session.id => {
                loaded = self
                    .store
                    .load(id)?
                    .ok_or_else(|| anyhow!("session not found: {id}"))?;
                &loaded
            }
            _ => &self.session,
        };
        Ok(stats_json(
            &record.id,
            totals_json(&nguruvilu::session::totals_of(record)),
            &record.stats,
        ))
    }

    /// [`stats`](Self::stats) with the disk read outside the state lock.
    ///
    /// `stats` runs under the process-wide lock, so asking it about a session
    /// that is not on screen reads that session's file with the whole shell
    /// waiting behind the lock: every other session's events and every command
    /// the page sends stop until the page of JSONL is parsed. This entry point
    /// takes the lock only long enough to tell the two cases apart — the session
    /// on screen is answered from memory with no I/O at all, and anything else
    /// carries nothing out of the lock but a second handle on the store, which
    /// is then read after the lock has been released.
    ///
    /// A caller already holding the lock must use [`stats`](Self::stats):
    /// this function takes that same non-reentrant lock.
    pub fn stats_outside_lock(state: &Mutex<AppState>, session: Option<&str>) -> Result<Value> {
        let requested = session.map(str::trim).filter(|id| !id.is_empty());
        let store = {
            let guard = state.lock().expect("state lock");
            // The usage rows of the open conversation are already in memory, so
            // its own report costs no I/O and the in-memory path answers it right
            // here. Only a request for another session needs anything carried out
            // of the lock, and all it carries is a second handle on the store.
            match requested {
                Some(id) if id != guard.session.id => store_handle(&guard.store)?,
                _ => return guard.stats(None),
            }
        };
        let id = requested.expect("only a request for another session reaches here");
        let record = store
            .load(id)?
            .ok_or_else(|| anyhow!("session not found: {id}"))?;
        Ok(stats_json(
            &record.id,
            totals_json(&nguruvilu::session::totals_of(&record)),
            &record.stats,
        ))
    }

    /// The read-only configuration check, as an event.
    ///
    /// The same examination `ngu doctor` prints, asked from inside the running
    /// shell: on a machine where the shell is the only front end there is, this
    /// is the only way to see it. The failures it catches are the ones that look
    /// like "nothing happened" rather than like an error — a settings file that
    /// does not parse, a pack manifest naming an assembly that is gone, a skill
    /// directory the scan walked past.
    pub fn doctor(&self) -> Value {
        let options = nguruvilu::doctor::Options {
            sessions: self.store.root().to_path_buf(),
            // The cap this shell's turns actually use, so the report answers the
            // question a user has ("why did my turn stop?") instead of repeating
            // a default they never chose.
            max_steps: Some(self.settings.max_steps()),
            ..nguruvilu::doctor::Options::default()
        };
        let report = nguruvilu::doctor::run(&options);
        let mut event = report.json();
        event["summary"] = json!(report.summary());
        event
    }

    /// Every stored message of the current session as renderable entries,
    /// oldest first. Pagination slices this list; it is not sent whole.
    fn all_entries(&self) -> Vec<Value> {
        self.session
            .messages
            .iter()
            .map(|message| {
                let role = match message.role {
                    nguruvilu::message::Role::System => "system",
                    nguruvilu::message::Role::User => "user",
                    nguruvilu::message::Role::Assistant => "assistant",
                    nguruvilu::message::Role::Tool => "tool",
                };
                json!({
                    "role": role,
                    "text": message.text(),
                    "tool_calls": message.tool_calls.iter().map(|c| json!({
                        "name": c.name,
                        "arguments": c.arguments,
                    })).collect::<Vec<_>>(),
                    "images": message.images.iter().map(|i| json!({
                        "url": i.url,
                        "label": i.label,
                    })).collect::<Vec<_>>(),
                })
            })
            .collect()
    }

    /// The newest page of the transcript, plus the failure line if one is set.
    ///
    /// Only the newest `TRANSCRIPT_PAGE` entries cross to the page: re-rendering
    /// a whole long session on every reconnect is the stall. `total` is the
    /// number of stored entries and `first` the absolute index of the first
    /// entry returned, so the page can request earlier history by `first`.
    pub fn transcript(&self) -> Value {
        let all = self.all_entries();
        let total = all.len();
        let first = total.saturating_sub(TRANSCRIPT_PAGE);
        let mut entries: Vec<Value> = all[first..].to_vec();

        // A failure belongs in the transcript, not only in a status line: it is
        // what a user scrolls back to read, and re-rendering the session must
        // not erase it.
        if let Some(error) = &self.last_error {
            entries.push(json!({
                "role": "error",
                "text": error,
                "tool_calls": [],
            }));
        }
        json!({
            "entries": entries,
            "total": total,
            "first": first,
        })
    }

    /// One page of history that ends just before absolute index `before`.
    ///
    /// Returns the same shape as [`transcript`](Self::transcript) without the
    /// failure line: an older page is history, not the current error. An
    /// out-of-range `before` clamps to the stored length, so a stale page
    /// request cannot panic or return entries that do not exist.
    pub fn transcript_before(&self, before: usize) -> Value {
        let all = self.all_entries();
        let total = all.len();
        let (first, end) = page_before(total, before);
        json!({
            "entries": all[first..end].to_vec(),
            "total": total,
            "first": first,
        })
    }

    /// Start a fresh session.
    pub fn new_session(&mut self) -> Result<()> {
        let session = self.new_session_record()?;
        self.store.create(&session)?;
        let workspace = session_workspace(&session)?;
        self.session_switch(&session.id, &workspace);
        self.session = session;
        Ok(())
    }

    /// Build a session with its own workspace directory.
    ///
    /// The directory is created before the session is stored, so a failure
    /// here aborts the new session instead of leaving one with no directory.
    fn new_session_record(&self) -> Result<Session> {
        let mut session = Session::new(Some(self.route.model.clone()));
        let dir = ensure_workspace(&workspaces_root()?, &session.id)?;
        session.cwd = Some(dir.display().to_string());
        Ok(session)
    }

    /// Switch to a stored session.
    pub fn open_session(&mut self, id: &str) -> Result<()> {
        let mut session = self
            .store
            .load(id)?
            .ok_or_else(|| anyhow!("session not found: {id}"))?;
        session.cwd = Some(self.adopt_workspace(&session)?);
        let workspace = session_workspace(&session)?;
        self.session_switch(&session.id, &workspace);
        self.session = session;
        Ok(())
    }

    /// The workspace directory of an existing session.
    ///
    /// A session stored before per-session workspaces records the process
    /// directory (or nothing). Keep the recorded `cwd` when it exists, and
    /// otherwise create the session's own workspace, so that no old session
    /// keeps running in a shared directory after the upgrade.
    fn adopt_workspace(&self, session: &Session) -> Result<String> {
        adopt_workspace_for(session)
    }

    /// Emit the session boundary for a switch: the old id ends (when there
    /// was a different one), the new one starts.
    ///
    /// These belong at the switch itself, not at process exit — a process can
    /// change sessions several times, and the audit log joins each turn to
    /// its session through this pair.
    fn session_switch(&self, next: &str, workspace: &str) {
        let events = self.kernel.events();
        // Pin the new conversation onto the shared tool table: from here on,
        // every tool call through this registry is attributed to `next` (job
        // spawn/list/kill filtering, and pack-scoped state read via the
        // execute-pinned thread-local), and relative paths and the default
        // bash directory resolve under its own workspace. The registry is
        // process-wide for the desktop's single AppState, so this swap is what
        // makes jobs, packs, and file tools conversation-scoped.
        self.kernel.set_session(Some(next.to_string()), Some(PathBuf::from(workspace)));
        // The model is told the same directory the tools resolve against.
        self.config.set_cwd(Some(workspace.to_string()));
        if self.session.id != next {
            events.emit(
                nguruvilu::events::SESSION_END,
                &serde_json::json!({ "session": &self.session.id }),
            );
        }
        events.emit(
            nguruvilu::events::SESSION_START,
            &serde_json::json!({ "session": next }),
        );
    }

    /// Delete a stored session, moving to a new one when it was current.
    ///
    /// A turn still running for that session is stopped first. Deleting the file
    /// underneath it would not stop the turn: it would keep appending, recreate
    /// the file the user just removed, and leave the transcript split across a
    /// deleted session and a fresh one.
    pub fn delete_session(&mut self, id: &str) -> Result<()> {
        if let Some(stop) = self.cancels.remove(id) {
            // The receiver may already be gone (the turn finished in between);
            // sending into a closed watch channel is an error, not a failure of
            // the delete.
            let _ = stop.send(true);
        }
        self.running.remove(id);
        self.store.delete(id)?;
        if self.session.id == id {
            self.new_session()?;
        }
        Ok(())
    }

    /// Change the model route (a session-scoped hot reload).
    pub fn set_model(&mut self, model: &str, base_url: Option<&str>) -> Result<()> {
        let mut route = self.route.clone();
        route.model = model.to_string();
        if let Some(url) = base_url {
            route.base_url = url.to_string();
        }
        self.route = route.clone();
        // The choice is the user's, so the baseline follows it too — a later
        // pack recompute must not send the route back to the boot value.
        self.pack_baseline.model = model.to_string();
        if let Some(url) = base_url {
            self.pack_baseline.base_url = url.to_string();
        }

        let mut runtime = self.config.runtime.lock().expect("runtime lock");
        runtime.apply(Change::session(
            self.session.id.clone(),
            ChangePayload::ModelRoute(route),
        ))?;
        Ok(())
    }

    /// Persist settings from the settings panel and apply them immediately.
    ///
    /// Applying matters as much as saving: a user who just typed a key expects
    /// the next message to use it, not the next launch.
    pub fn save_settings(
        &mut self,
        base_url: &str,
        api_key: &str,
        model: &str,
        reasoning_effort: &str,
        proxy: &str,
        context_window: Option<usize>,
        compact_percent: u32,
        compact_keep_recent: usize,
        max_output_tokens: Option<usize>,
        search_provider: &str,
        search_api_key: &str,
        search_endpoint: &str,
        search_present: bool,
        judge_endpoint: &str,
        judge_api_key: &str,
        judge_model: &str,
        judge_present: bool,
    ) -> Result<()> {
        let settings = Settings {
            base_url: base_url.trim().to_string(),
            api_key: api_key.trim().to_string(),
            model: model.trim().to_string(),
            reasoning_effort: reasoning_effort.trim().to_string(),
            // The delivery mode is not a panel field: it is a property of the
            // gateway rather than of the conversation, and `ngu config set
            // --delivery buffered` or the settings file is where it is written.
            // Preserving it here is what stops an unrelated save from quietly
            // putting a route that cannot stream back into streaming mode.
            delivery: self.settings.delivery.clone(),
            degrade_on_stall: self.settings.degrade_on_stall,
            proxy: proxy.trim().to_string(),
            context_window: context_window.filter(|t| *t > 0),
            compact_percent,
            compact_keep_recent: compact_keep_recent.max(1),
            // Not a panel field: the shell's settings panel has no control for
            // the step cap, and a save from it must not reset what the file or
            // the command line chose.
            max_steps: self.settings.max_steps,
            // Not panel fields: a save from the panel must keep the stored relay
            // settings instead of clearing them.
            relay_url: self.settings.relay_url.clone(),
            relay_token: self.settings.relay_token.clone(),
            max_output_tokens: max_output_tokens.filter(|t| *t > 0),
            network: self.settings.network.clone(),
            // A section the page does not show (its pack is unloaded) is not a
            // section the user cleared: absent means "keep what is stored",
            // present means "these fields are the truth".
            search: if search_present {
                resolve_search(
                    &self.settings.search,
                    search_provider,
                    search_api_key,
                    search_endpoint,
                )?
            } else {
                self.settings.search.clone()
            },
            judge: if judge_present {
                resolve_judge(&self.settings.judge, judge_endpoint, judge_api_key, judge_model)?
            } else {
                self.settings.judge.clone()
            },
            extra_body: self.settings.extra_body.clone(),
            // Keys this build does not know about are carried through: the panel
            // saving a field must not delete what a plugin or a newer build wrote.
            unknown: self.settings.unknown.clone(),
        };
        settings.save()?;
        self.settings = settings;
        // The panel just wrote the user's own file, so what it shows is now
        // their choice: the baseline follows, or the next pack toggle would
        // recompute away the values they entered.
        self.pack_baseline = self.settings.clone();

        // The context policy is derived from settings, so it has to be rebuilt
        // here too — otherwise the panel would save a window that only takes
        // effect after a restart.
        {
            let mut policy = self.config.policy.lock().expect("policy lock");
            *policy = Arc::new(self.settings.context_policy(None));
        }

        // The route is what a turn actually reads, so it is rebuilt from the
        // settings that just changed — every field of it, not only the three
        // the panel is most likely to have touched.
        self.route = ModelRoute::from_settings(&self.settings);

        // And the search tool follows its key at once: a key typed into the
        // panel goes into the plugin's cell and its fibers re-apply, so
        // `search_web` appears in the very next status — and vanishes when
        // the key does.
        nguruvilu::tools::search::configure(
            &mut self.kernel,
            &self.search_cell,
            self.settings.search.clone(),
        )?;
        // The judge takes the panel's three fields the same way.
        nguruvilu::tools::judge::configure(
            &mut self.kernel,
            &self.judge_cell,
            self.settings.judge.clone(),
        )?;
        let tools = Arc::new(self.kernel.tools().clone());

        let mut runtime = self.config.runtime.lock().expect("runtime lock");
        runtime.set_tools(tools);
        runtime.apply(Change::session(
            self.session.id.clone(),
            ChangePayload::ModelRoute(self.route.clone()),
        ))?;
        Ok(())
    }

    /// A snapshot of the saved API profiles for the settings page.
    ///
    /// Keys are not part of it: the page sees whether a profile has one, and a
    /// switch needs a name. The key lives on this machine only.
    pub fn routes_snapshot(&self) -> Value {
        let routes = nguruvilu::routes::Routes::load();
        json!({
            "active": routes.active,
            "profiles": routes.profiles.iter().map(|profile| json!({
                "name": profile.name,
                "base_url": profile.base_url,
                "model": profile.model,
                "proxy": profile.proxy,
                "has_key": !profile.api_key.trim().is_empty(),
            })).collect::<Vec<_>>(),
        })
    }

    /// Bookmark the live settings as a named profile. Saving under a name that
    /// already exists rewrites that profile: two chips the panel cannot tell
    /// apart are one chip updated.
    pub fn save_route(&mut self, name: &str) -> Result<()> {
        let name = name.trim();
        if name.is_empty() {
            return Err(anyhow!("save_route needs a profile name"));
        }
        let mut routes = nguruvilu::routes::Routes::load();
        routes.upsert(nguruvilu::routes::Routes::capture(&self.settings, name));
        // The live settings are exactly this profile now, so the bookmark
        // marks the active line.
        routes.active = name.to_string();
        routes.save()
    }

    /// Switch to a saved profile: the profile becomes the live settings, and
    /// the current session takes the new route on its next turn — the same
    /// apply path a panel save uses, so settings become real exactly one way.
    pub fn switch_route(&mut self, name: &str) -> Result<()> {
        let name = name.trim();
        let mut routes = nguruvilu::routes::Routes::load();
        routes.apply(name, &mut self.settings)?;
        routes.active = name.to_string();
        routes.save()?;
        self.settings.save()?;
        // The user's own choice moved the route: a later pack recompute must
        // not send it back to the boot value.
        self.pack_baseline = self.settings.clone();

        self.route = ModelRoute::from_settings(&self.settings);
        let mut runtime = self.config.runtime.lock().expect("runtime lock");
        runtime.apply(Change::session(
            self.session.id.clone(),
            ChangePayload::ModelRoute(self.route.clone()),
        ))?;
        Ok(())
    }

    /// Remove a saved profile. If it was the active line, the live settings
    /// stay as they are: deleting a bookmark does not disconnect the route it
    /// described.
    pub fn delete_route(&mut self, name: &str) -> Result<()> {
        let mut routes = nguruvilu::routes::Routes::load();
        routes.remove(name)?;
        routes.save()
    }

    /// Adopt what the enabled packs carry, recomputed as one set.
    ///
    /// [`nguruvilu::content::AdoptedContent::recompute`] folds every enabled
    /// pack over [`pack_baseline`](Self::pack_baseline) — the state a fresh
    /// boot arrives at — so a load and an unload converge on the same result
    /// instead of an unload having no inverse: the pack that brought a route
    /// is gone and its route goes with it. Persona, injection rules, the
    /// context policy, the route, and the tool table all move together, so
    /// the next turn reads one consistent view.
    pub fn recompute_pack_content(&mut self) -> Result<()> {
        let adopted = nguruvilu::content::AdoptedContent::recompute(
            &self.pack_baseline,
            &nguruvilu::pack::default_packs_dir(),
        )?;
        self.settings = adopted.settings;

        // The context policy is rebuilt from the settings that just changed,
        // so a pack's window takes effect on the next turn rather than the
        // next launch.
        if let Ok(mut policy) = self.config.policy.lock() {
            *policy = std::sync::Arc::new(self.settings.context_policy(None));
        }
        // Injection rules: the standing file with each pack's engine applied
        // over it, the same composition a boot arrives at.
        if let Ok(mut injection) = self.config.injection.lock() {
            *injection = std::sync::Arc::new(adopted.injections);
        }

        // The route is rebuilt from the adopted settings and folds in whatever
        // network policy the kernel currently publishes — the truth after a
        // pack's plugin loaded or unloaded.
        self.route = ModelRoute::from_settings(&self.settings);
        let view = self
            .kernel
            .service_view(nguruvilu::plugin::RealmMap::new());
        if let Some(network) = nguruvilu::network::NetworkHandle::from_view(&view) {
            self.route.network = network;
        }
        let tools = Arc::new(self.kernel.tools().clone());

        if let Ok(mut runtime) = self.config.runtime.lock() {
            // The joined persona — cleared when no enabled pack carries one,
            // so an unloaded pack's voice leaves with it.
            let _ = runtime.apply(nguruvilu::hotreload::Change::session(
                "pack",
                nguruvilu::hotreload::ChangePayload::Persona(adopted.persona.unwrap_or_default()),
            ));
            runtime.set_tools(tools);
            let _ = runtime.apply(nguruvilu::hotreload::Change::session(
                "pack",
                nguruvilu::hotreload::ChangePayload::ModelRoute(self.route.clone()),
            ));
        }
        Ok(())
    }

    /// Installed packs.
    pub fn packs(&self) -> Result<Value> {
        let dir = nguruvilu::pack::default_packs_dir();
        let packs = nguruvilu::pack::installed(&dir)?;
        Ok(json!({
            "dir": dir.display().to_string(),
            "packs": packs.iter().map(|pack| json!({
                "name": pack.manifest.name,
                "description": pack.manifest.description,
                "version": pack.manifest.version_id,
                "license": pack.manifest.license,
                // What the pack fetches, and what it carries itself.
                "skills": pack.manifest.skills.len(),
                "plugins": pack.manifest.plugins.len(),
                "content": nguruvilu::pack::CONTENT_FILES
                    .iter()
                    .filter(|(field, _)| pack.manifest.content_file(field).is_some())
                    .map(|(_, file)| (*file).to_string())
                    .collect::<Vec<_>>(),
                "accepts": pack.manifest.dependencies.nguruvilu,
                "compatible": pack.manifest.accepts_kernel(nguruvilu::pack::kernel_version()),
                "path": pack.path.display().to_string(),
                "assembly": pack.assembly.as_ref().map(|p| p.display().to_string()),
                "loaded": pack.enabled,
            })).collect::<Vec<_>>(),
        }))
    }

    /// Install an archive and report what landed.

    /// Build an archive from a pack directory.
    pub fn pack_dir(&self, dir: &Path, out: Option<&Path>) -> Result<Value> {
        let manifest = nguruvilu::pack::read_manifest(dir)?;
        let archive = match out {
            Some(path) => path.to_path_buf(),
            None => dir.join(format!("{}-{}.dshpack", manifest.name, manifest.version_id)),
        };
        let packed = nguruvilu::pack::pack(dir, &archive)?;
        let contents = nguruvilu::pack::inspect(dir)?;
        Ok(json!({
            "name": packed.name,
            "version": packed.version_id,
            "archive": archive.display().to_string(),
            "files": contents.files.len(),
            "has_assembly": contents.has_assembly,
        }))
    }

    /// Verify an archive without installing it.
    pub fn verify_pack(&self, archive: &Path) -> Result<Value> {
        let report = nguruvilu::pack::verify(archive)?;
        Ok(json!({
            "name": report.manifest.name,
            "version": report.manifest.version_id,
            "license": report.manifest.license,
            "files": report.contents.files.len(),
            "has_assembly": report.contents.has_assembly,
            "warnings": report.warnings,
        }))
    }

    /// Take the kernel out, so an async load can run without holding the lock.
    pub fn take_kernel(&mut self) -> Kernel {
        std::mem::replace(&mut self.kernel, Kernel::new())
    }

    /// Put a kernel back and make its tools visible to the next turn.
    ///
    /// Applying a pack changes the tool table, and the runtime is what a turn
    /// reads it from — so both have to move together. The route moves with
    /// them: by now the pack's plugins are loaded, so a network policy it
    /// published is folded in, and the built-in service publishes the
    /// settings only when no plugin provided one.
    pub fn restore_kernel(&mut self, kernel: Kernel) -> Result<()> {
        self.kernel = kernel;
        let network = self.settings.network.clone();
        nguruvilu::network::install(&mut self.kernel, network)?;

        // A pack may be what configured search: the settings go into the
        // plugin's cell and its fibers re-apply against this kernel — the
        // tool present while a key is configured, absent when it is not (a
        // search tool that cannot authenticate costs a turn every time the
        // model tries it).
        nguruvilu::tools::search::configure(
            &mut self.kernel,
            &self.search_cell,
            self.settings.search.clone(),
        )?;
        // A pack may have configured the judge too — same re-apply.
        nguruvilu::tools::judge::configure(
            &mut self.kernel,
            &self.judge_cell,
            self.settings.judge.clone(),
        )?;

        let tools = Arc::new(self.kernel.tools().clone());
        let mut route = ModelRoute::from_settings(&self.settings);
        let view = self.kernel.service_view(nguruvilu::plugin::RealmMap::new());
        if let Some(network) = nguruvilu::network::NetworkHandle::from_view(&view) {
            route.network = network;
        }
        self.route = route;

        let mut runtime = self.config.runtime.lock().expect("runtime lock");
        runtime.set_tools(tools);
        let _ = runtime.apply(Change::session(
            self.session.id.clone(),
            ChangePayload::ModelRoute(self.route.clone()),
        ));
        Ok(())
    }

    /// A client for the current route.
    ///
    /// The route's own builder, so this and a subagent the model delegates to
    /// send the same body on the same connection settings.
    pub fn client(&self) -> Result<LlmClient> {
        self.route.client()
    }
}

/// Fold the panel's three search fields into the stored settings.
///
/// The panel never receives the stored key, so a blank box means "keep it" —
/// the same rule as the model key. The provider decides whether search is on:
/// an empty value means off, and off drops what was stored, because a key that
/// no reachable tool can use is worse than no key. The panel's hint says so,
/// and `search_web` appears only while a key is actually configured.
fn resolve_search(
    stored: &Option<nguruvilu::tools::search::SearchSettings>,
    provider: &str,
    api_key: &str,
    endpoint: &str,
) -> Result<Option<nguruvilu::tools::search::SearchSettings>> {
    let provider = provider.trim();
    if provider.is_empty() {
        return Ok(None);
    }
    let dialect = nguruvilu::tools::search::Dialect::parse(provider).ok_or_else(|| {
        anyhow!(
            "unknown search provider '{provider}'; expected {}",
            nguruvilu::tools::search::Dialect::all()
                .iter()
                .map(|candidate| format!("{candidate:?}").to_lowercase())
                .collect::<Vec<_>>()
                .join(", ")
        )
    })?;
    let mut merged = stored.clone().unwrap_or_default();
    merged.provider = dialect;
    if !api_key.trim().is_empty() {
        merged.api_key = api_key.trim().to_string();
    }
    merged.endpoint = if endpoint.trim().is_empty() {
        None
    } else {
        Some(endpoint.trim().to_string())
    };
    Ok(Some(merged))
}

/// Fold the panel's judge fields into the stored settings.
///
/// The panel never receives the stored key, so a blank box means "keep it" —
/// the model key's rule again. Three empties mean the judge is off, and off
/// drops what was stored: a key no tool can reach is worse than no key. A
/// provided endpoint must be an http(s) URL — it is pasted straight into a
/// request, and a typo there becomes a failed call with no hint about why.
fn resolve_judge(
    stored: &Option<nguruvilu::tools::judge::JudgeSettings>,
    endpoint: &str,
    api_key: &str,
    model: &str,
) -> Result<Option<nguruvilu::tools::judge::JudgeSettings>> {
    let endpoint = endpoint.trim().trim_end_matches('/');
    let api_key = api_key.trim();
    let model = model.trim();

    if endpoint.is_empty() && api_key.is_empty() && model.is_empty() {
        return Ok(None);
    }

    let mut merged = stored.clone().unwrap_or_default();
    if !endpoint.is_empty() {
        if !(endpoint.starts_with("http://") || endpoint.starts_with("https://")) {
            anyhow::bail!("judge endpoint must be an http(s) URL, got '{endpoint}'");
        }
        merged.endpoint = endpoint.to_string();
    }
    if !api_key.is_empty() {
        merged.api_key = api_key.to_string();
    }
    if !model.is_empty() {
        merged.model = model.to_string();
    }
    Ok(Some(merged))
}

/// The recorded working directory of a session, which must be present.
///
/// A session without one has no workspace to run its tools in, so the switch
/// is refused instead of binding some other directory.
fn session_workspace(session: &Session) -> Result<String> {
    session
        .cwd
        .clone()
        .ok_or_else(|| anyhow!("session {} has no working directory", session.id))
}

/// Where per-session working directories live: `<data dir>/workspaces`.
///
/// Derived from the same base as [`sessions_root`], so the session store and
/// the workspaces share one data directory.
fn workspaces_root() -> Result<PathBuf> {
    // `ensure_workspace` adds the `workspaces/` segment itself, so this is the
    // data root, not the workspaces directory.
    sessions_root()
        .parent()
        .map(|dir| dir.to_path_buf())
        .ok_or_else(|| anyhow!("sessions root has no parent directory to hold workspaces"))
}

/// The workspace directory an existing session should run in.
///
/// A recorded `cwd` that still exists is kept. A session stored before
/// per-session workspaces (or whose directory was removed) gets its own
/// `workspaces/<id>` directory. Creation failure is returned, never replaced
/// with another directory.
fn adopt_workspace_for(session: &Session) -> Result<String> {
    if let Some(recorded) = &session.cwd {
        if Path::new(recorded).is_dir() {
            return Ok(recorded.clone());
        }
    }
    let dir = ensure_workspace(&workspaces_root()?, &session.id)?;
    Ok(dir.display().to_string())
}

/// Every mission of one session workspace, newest first.
///
/// The mission tools already put this board in every tool answer they return, but
/// the page also draws it when the window opens — before any tool has answered.
/// Otherwise the mission bar sits on "idle, no mission in progress" while the
/// tasks exist on disk, which reads exactly like the feature is missing.
///
/// Missions are directories under `<workspace>/missions`, each holding a
/// `mission.json`; their ids are time-ordered (`m-YYYYmmdd-HHMMSS`), so sorting the
/// directory names newest-first gives the order a reader expects. Nothing here may
/// fail: a session with no missions is the ordinary case, and this runs from
/// `AppState::describe` under the state lock, where a half-readable board must
/// never turn a status push into an error. A record that does not parse is skipped
/// rather than reported as a nameless mission.
fn missions_under(workspace: &Path) -> Value {
    let root = workspace.join("missions");
    let Ok(entries) = std::fs::read_dir(&root) else {
        return json!([]);
    };
    let mut rows: Vec<(String, Value)> = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let name = match entry.file_name().to_str() {
            Some(name) => name.to_string(),
            None => continue,
        };
        let Ok(text) = std::fs::read_to_string(path.join("mission.json")) else {
            continue;
        };
        let Ok(record) = serde_json::from_str::<Value>(&text) else {
            continue;
        };
        let row = json!({
            "id": record
                .get("id")
                .cloned()
                .unwrap_or_else(|| Value::String(name.clone())),
            "objective": record.get("objective").cloned().unwrap_or(Value::Null),
            "status": record.get("status").cloned().unwrap_or(Value::Null),
        });
        rows.push((name, row));
    }
    rows.sort_by(|a, b| b.0.cmp(&a.0));
    Value::Array(rows.into_iter().map(|(_, row)| row).collect())
}

/// The directory whose `missions/` this session's tools wrote into.
///
/// A mission lives in the session workspace, which is the recorded `cwd` while it
/// still exists — a session can be pinned to a real project — and the per-session
/// `<home>/workspaces/<id>` directory otherwise. That is the rule
/// [`adopt_workspace_for`] applies, so the board is read where the tools wrote
/// instead of where a guess says they should have. `store_root` is the sessions
/// directory, so its parent is the state home.
fn missions_workspace(store_root: &Path, session_id: &str, cwd: Option<&str>) -> Option<PathBuf> {
    if let Some(cwd) = cwd.map(Path::new) {
        if cwd.is_dir() {
            return Some(cwd.to_path_buf());
        }
    }
    nguruvilu::session::workspace_dir(store_root.parent()?, session_id).ok()
}

/// Where sessions live for the desktop shell.
///
/// Deliberately not the current directory: a windowed app is launched from
/// wherever its executable happens to sit — a desktop shortcut, a read-only
/// share, `dist/` — and writing a session store there either litters that
/// directory or fails outright.
fn sessions_root() -> PathBuf {
    if let Ok(home) = std::env::var("NGU_HOME") {
        if !home.trim().is_empty() {
            return PathBuf::from(home).join("sessions");
        }
    }
    match Settings::path().parent() {
        Some(dir) => dir.join("sessions"),
        None => JsonlStore::default_root(),
    }
}

/// Where WebView2 keeps its profile for this app.
///
/// The webview exists only on the window path, and `windowed` is this
/// function's only caller: without a window there is no profile to point
/// anywhere.
#[cfg(windows)]
pub fn data_directory() -> PathBuf {
    sessions_root().parent().map(|dir| dir.join("webview")).unwrap_or_else(|| PathBuf::from("."))
}

/// A second handle on the same store directory.
///
/// `AppState` owns the only `JsonlStore` and that type is a bare root path with
/// no shared state behind it — every append and load is a stateless file
/// operation under the root — so a handle opened at the same root is the same
/// store. Opening one is how a caller carries the disk work out of the state
/// lock without the store having to be shared or cloned.
fn store_handle(store: &JsonlStore) -> Result<JsonlStore> {
    JsonlStore::open(store.root())
}

/// Write the metadata rows a finished turn leaves behind, with no lock held.
///
/// Messages were already written by `on_message`, the moment each one was
/// committed — a kill mid-turn loses only what never completed, never a whole
/// turn's worth. What is left is the metadata that has no message to ride along
/// with: the injections, the compactions, and the turn's usage. None of it is
/// in-memory, which is why it can be done against a handle carried out of the
/// state lock: the append costs no reader its turn.
///
/// Returns the usage row when it reached the file. The caller records that same
/// row in the live session, and only a row that was actually written belongs
/// there.
fn write_turn_rows(
    store: &JsonlStore,
    session_id: &str,
    outcome: &nguruvilu::agent::AgentOutcome,
) -> Option<nguruvilu::session::TurnStats> {
    // The user may have deleted this conversation mid-turn. The file is gone,
    // and appending to it would resurrect an empty shell of a conversation the
    // user asked to be rid of: the metadata rides the same loss its messages
    // already took. A read that *fails* (rather than reporting "not there") is
    // not evidence of deletion, so the appends still run on an error.
    let still_stored = match store.load(session_id) {
        Ok(found) => found.is_some(),
        Err(error) => {
            eprintln!("[store] could not check {session_id} before appending: {error:#}");
            true
        }
    };
    if !still_stored {
        eprintln!(
            "[store] session {session_id} was deleted mid-turn; skipping injection, \
             compaction and usage rows"
        );
    }
    if still_stored {
        for record in &outcome.injections {
            if let Err(error) = store.append_injection(session_id, record) {
                eprintln!("[store] injection append failed: {error:#}");
            }
        }
    }

    // Record compactions after the messages they replace.
    for compaction in &outcome.compactions {
        if still_stored {
            if let Err(error) = store.append_compaction(session_id, compaction) {
                eprintln!("[store] compaction append failed: {error:#}");
            }
        }
        eprintln!(
            "[compacted {} messages into {} chars]",
            compaction.replaced,
            compaction.summary.chars().count()
        );
    }

    if !still_stored {
        return None;
    }

    // The turn's usage, appended to the session log: the one place that
    // outlives this process. Token counts are spent money and a duration is the
    // only honest measure of a slow turn, so both are recorded even when the
    // turn ended on an error — what was spent was spent either way. A failed
    // append is reported, never fatal: the answer already exists, and losing
    // its ledger row must not also lose the turn.
    let stats = nguruvilu::session::TurnStats::from_outcome(outcome);
    match store.append_stats(session_id, &stats) {
        Ok(()) => Some(stats),
        Err(error) => {
            eprintln!("[store] stats append failed: {error:#}");
            None
        }
    }
}

/// Streams loop progress into the page.
///
/// Text is forwarded the moment it arrives. An earlier version batched deltas
/// into a 25 ms window to save IPC round trips, but measurement showed that was
/// backwards: the endpoint already delivers ~5 characters per delta, often
/// several at once, so an extra window only merged those bursts into larger,
/// more visible lumps. The page coalesces within a frame anyway, which is where
/// redundant DOM work actually gets avoided.
pub struct UiObserver {
    sink: Arc<dyn EventSink>,
    /// The shell's state, so a committed message can be written the moment it
    /// exists instead of batched until the turn ends.
    state: Arc<Mutex<AppState>>,
    /// The session this turn belongs to.
    ///
    /// Stamped onto every event so the page can tell one conversation's
    /// stream from another's once two sessions may be working at once — and
    /// the key every incremental write goes under, whichever session the page
    /// happens to be showing when the turn finishes.
    session: String,
    /// Per `bash` call in this turn: did it ask for a completion wake?
    ///
    /// Starts and ends arrive in tool-call order, so a FIFO pairs them even
    /// when a step dispatches several calls at once.
    bash_notify: Mutex<VecDeque<bool>>,
}

impl UiObserver {
    /// Build an observer that pushes events for `session` to the window and
    /// writes each committed message to `state`'s store as it happens.
    pub fn new(sink: Arc<dyn EventSink>, state: Arc<Mutex<AppState>>, session: String) -> Self {
        Self {
            sink,
            state,
            session,
            bash_notify: Mutex::new(VecDeque::new()),
        }
    }

    fn emit(&self, event: Value) {
        let tagged = match event {
            Value::Object(mut map) => {
                map.insert("session".to_string(), Value::String(self.session.clone()));
                Value::Object(map)
            }
            other => other,
        };
        // The page is the only consumer; a closed window just means the send
        // fails, which is not an error worth surfacing.
        self.sink.emit(tagged);
    }
}

impl AgentObserver for UiObserver {
    fn on_step(&self, step: usize) {
        self.emit(json!({ "ev": "step", "step": step }));
    }

    fn on_text(&self, delta: &str) {
        if !delta.is_empty() {
            self.emit(json!({ "ev": "text", "delta": delta }));
        }
    }

    fn on_reasoning(&self, delta: &str) {
        self.emit(json!({ "ev": "reasoning", "delta": delta }));
    }

    fn on_tool_start(&self, name: &str, arguments: &str) {
        self.emit(json!({ "ev": "tool_start", "name": name, "arguments": arguments }));
        if name == "bash" {
            let notify = serde_json::from_str::<Value>(arguments)
                .ok()
                .and_then(|args| args.get("notify").and_then(|value| value.as_bool()))
                .unwrap_or(false);
            self.bash_notify.lock().expect("bash notify").push_back(notify);
        }
    }

    fn on_tool_end(&self, name: &str, ok: bool, result: &str) {
        self.emit(json!({ "ev": "tool_end", "name": name, "ok": ok, "result": result }));
        if name == "bash" {
            let notify = self
                .bash_notify
                .lock()
                .expect("bash notify")
                .pop_front()
                .unwrap_or(false);
            // A job id is only in a successful background reply — pair the
            // flag with it here, where the session is known.
            if notify && ok {
                if let Some(job_id) = result.split_whitespace().find(|t| t.starts_with("job-")) {
                    let mut guard = self.state.lock().expect("state lock");
                    guard
                        .job_sessions
                        .insert(job_id.to_string(), self.session.clone());
                    // The owner's sink, captured where both session and sink are
                    // known - this is the only place the pairing happens, so it
                    // is also the right place to remember which stream to wake on.
                    guard.app_jobs.insert(
                        job_id.to_string(),
                        AppJob {
                            session: self.session.clone(),
                            sink: Arc::clone(&self.sink),
                        },
                    );
                }
            }
        }
    }

    fn on_step_retry(&self, attempt: usize, reason: &str) {
        self.emit(json!({ "ev": "step_retry", "attempt": attempt, "reason": reason }));
    }

    fn on_continuation(&self, reason: &str) {
        self.emit(json!({ "ev": "continuation", "reason": reason }));
    }

    fn on_message(&self, message: &nguruvilu::message::Message) {
        // Written the moment it exists. This is what makes a force-kill
        // survivable: the file already holds every message the conversation
        // produced, so a restart reads exactly what happened — instead of
        // waiting for a turn-end batch the killed process never reaches.
        //
        // The write happens outside the state lock. The guard is held only long
        // enough to take a handle on the store, so a slow disk delays this
        // session's own append and nothing else: not the page's commands, and
        // not another session's events.
        //
        // Writing outside the lock cannot reorder the appends, which is the
        // property the append exists for: a session runs at most one turn at a
        // time (`running` is keyed by session), so every append to one file
        // comes from that session's single turn, in the order the agent
        // committed the messages. Two appends for the same session cannot
        // overlap, so dropping the lock lets no later message overtake an
        // earlier one.
        let handle = {
            let guard = self.state.lock().expect("state lock");
            store_handle(&guard.store)
        };
        let appended = match &handle {
            Ok(store) => store.append(&self.session, std::slice::from_ref(message)),
            // A handle could not be opened, which means the store directory
            // itself is unusable. The message still goes through the handle the
            // state holds, under the lock: a message that exists must reach the
            // file even when the cheap copy of the handle cannot be made.
            Err(_) => {
                let guard = self.state.lock().expect("state lock");
                guard.store.append(&self.session, std::slice::from_ref(message))
            }
        };
        if let Err(error) = appended {
            eprintln!("[store] incremental append failed: {error:#}");
        }
        // The lock is taken again only for the in-memory transcript. Whether
        // this turn belongs to the conversation on screen is decided here
        // rather than earlier: the user may have switched while the append was
        // on its way to disk.
        let mut guard = self.state.lock().expect("state lock");
        if guard.session.id == self.session {
            guard.session.messages.push(message.clone());
        }
    }

    fn on_compaction_start(&self, messages: usize, window: nguruvilu::window::ContextWindow) {
        self.emit(json!({
            "ev": "compaction_start",
            "messages": messages,
            "window": window.tokens,
            "window_source": window.source.as_str(),
        }));
    }

    fn on_injection(&self, injection: &nguruvilu::context::Injection) {
        if injection.is_empty() {
            return;
        }
        self.emit(json!({
            "ev": "injection",
            "activated": injection.activated,
            "relocated": injection.relocated,
            "budget_used": injection.budget_used,
        }));
    }

    fn on_compaction(&self, compaction: &nguruvilu::compaction::Compaction) {
        self.emit(json!({
            "ev": "compaction",
            "replaced": compaction.replaced,
            "summary": compaction.summary,
        }));
    }
}

/// Queue `text` as steering for the session the shell is showing, but only
/// while that session's turn is running.
///
/// Returns the session id when the message was queued; `None` means the
/// session is idle (or the text is blank) and the caller should treat the
/// text as an ordinary prompt. The serve shell calls this directly: its
/// command queue would otherwise hold the message until the very turn it
/// steers has already ended.
pub fn try_steer(guard: &mut AppState, text: &str) -> Option<String> {
    if text.trim().is_empty() {
        return None;
    }
    let session_id = guard.session.id.clone();
    if !guard.running.contains(&session_id) {
        return None;
    }
    let queue = guard
        .inboxes
        .entry(session_id.clone())
        .or_insert_with(|| Arc::new(std::sync::Mutex::new(VecDeque::new())));
    queue.lock().expect("inbox").push_back(text.to_string());
    Some(session_id)
}

/// Run one turn, streaming progress to the page.
///
/// A turn belongs to the session it started in: it finishes against that
/// session's file even when the user switches to a different conversation
/// mid-run, and one session runs one turn at a time. A prompt sent while its
/// turn runs is steering — queued for the next step boundary, or, when the
/// running one ends before claiming it, returned as the next turn for
/// [`run_turn_cascading`] to run. Settings are taken once, at the start, so a
/// change landing while the turn runs applies to the next turn.
///
/// `target` is the conversation this turn belongs to: an explicit id wins
/// (background cascades and job wakes carry their own), `None` means the
/// conversation the shell is showing. A target that is not on screen never
/// takes the view — the text is parked in that session's inbox instead.
///
/// A failure keeps its progress: every message the agent committed is written
/// as it happens, and the error is reported alongside it.
///
/// Returns the oldest unclaimed steering message, if one arrived too late to
/// be claimed by this turn.
pub async fn run_turn(
    state: Arc<Mutex<AppState>>,
    target: Option<String>,
    text: String,
    sink: Arc<dyn EventSink>,
) -> Result<Option<String>> {
    // One turn per session, decided under the lock so two prompts racing for
    // the same history cannot both pass the check.
    let (stop_tx, stop_rx) = tokio::sync::watch::channel(false);
    let (session_id, client, config, messages, events, inbox, workspace, max_steps) = {
        let mut guard = state.lock().expect("state lock");
        // The conversation this turn belongs to, resolved once under the lock:
        // an explicit target wins, without one the conversation on screen. A
        // follow-up never re-reads "what is on screen now".
        let target_session = match target {
            Some(id) => id,
            None => guard.session.id.clone(),
        };
        // The turn's conversation is not the one the user is looking at, and
        // grabbing the view is the wrong behaviour: an implicit switch would
        // yank the page mid-read and re-home the shared tool table to another
        // conversation. The text waits in that session's inbox — its next turn
        // (a later prompt there, or its next wake) claims the queue at the
        // first step boundary and runs it where it belongs. Nothing is emitted
        // to the page: it is showing the other conversation, and a prompt for
        // one the user cannot see is not news.
        if guard.session.id != target_session {
            let parked = guard
                .inboxes
                .entry(target_session.clone())
                .or_insert_with(|| Arc::new(std::sync::Mutex::new(VecDeque::new())));
            parked.lock().expect("inbox").push_back(text);
            eprintln!(
                "[turn] parked prompt for session {target_session} ({} is on screen); it runs \
                 when that session next gets a turn",
                guard.session.id
            );
            drop(guard);
            return Ok(None);
        }
        // Steering, not a refusal: the message joins the running turn at its
        // next step boundary — or, when the turn ends before claiming it, the
        // cascade runs it as a follow-up turn.
        if let Some(session) = try_steer(&mut guard, &text) {
            drop(guard);
            sink.emit(json!({
                "ev": "steer",
                "text": text,
                "session": session,
            }));
            return Ok(None);
        }
        let session_id = target_session;
        // A new attempt supersedes the previous failure; leaving a stale error
        // on screen while a retry runs would be a lie.
        guard.last_error = None;
        // The fallible part runs *before* the session is marked running: a `?`
        // here must not leave the page blocking every later send for a turn
        // that never started.
        let client = guard.client()?;
        guard.running.insert(session_id.clone());
        // Published before the first await: a stop arriving any time after
        // this line finds its target.
        guard.cancels.insert(session_id.clone(), stop_tx);
        let inbox = guard
            .inboxes
            .entry(session_id.clone())
            .or_insert_with(|| Arc::new(std::sync::Mutex::new(VecDeque::new())))
            .clone();
        (
            session_id,
            client,
            Arc::clone(&guard.config),
            guard.session.messages.clone(),
            guard.kernel.events(),
            inbox,
            guard.session.cwd.clone(),
            // The step cap is a setting, not a constant of the shell: a machine
            // whose turns are long-running work needs a different number than one
            // answering short questions, and hard-coding 50 meant the desktop was
            // the one front end that could not be told.
            guard.settings.max_steps(),
        )
    };
    let observer = Arc::new(UiObserver::new(
        Arc::clone(&sink),
        Arc::clone(&state),
        session_id.clone(),
    ));
    let mut agent = Agent::new(client, config, messages)
        .with_max_steps(max_steps)
        .with_observer(Arc::clone(&observer) as Arc<dyn AgentObserver>)
        .with_cancel(stop_rx)
        .with_inbox(inbox)
        // This turn's session, fixed here rather than read from the shared
        // registry on every tool call: switching conversations mid-turn must
        // not re-home this turn's `bash` cwd, relative paths or job owner.
        .with_identity(nguruvilu::tools::ToolIdentity::for_session(
            session_id.clone(),
            workspace.map(std::path::PathBuf::from),
        ));

    events.emit(
        nguruvilu::events::PROMPT_SUBMIT,
        &serde_json::json!({
            "session": &session_id,
            "text": nguruvilu::events::truncate(&text, 4000),
        }),
    );
    events.emit(
        nguruvilu::events::TURN_START,
        &serde_json::json!({ "session": &session_id }),
    );
    // The page sees the turn start only once the running slot is held, and
    // with the session id attached: everything streamed from here on belongs
    // to that conversation.
    sink.emit(json!({ "ev": "turn_start", "text": &text, "session": &session_id }));

    let turn_started = std::time::Instant::now();
    let result = agent.run(&text).await;
    let turn_ms = turn_started.elapsed().as_millis() as u64;
    match &result {
        Ok(outcome) if outcome.error.is_none() => events.emit(
            nguruvilu::events::TURN_COMPLETE,
            &serde_json::json!({
                "session": &session_id,
                "ok": true,
                "duration_ms": turn_ms,
                "steps": outcome.steps,
                "tool_calls": outcome.tool_calls,
                "retries": outcome.retries,
            }),
        ),
        Ok(outcome) => events.emit(
            nguruvilu::events::TURN_COMPLETE,
            &serde_json::json!({
                "session": &session_id,
                "ok": false,
                "duration_ms": turn_ms,
                "steps": outcome.steps,
                "retries": outcome.retries,
                "error": nguruvilu::events::truncate(
                    outcome.error.as_deref().unwrap_or("turn failed"),
                    1000,
                ),
            }),
        ),
        Err(error) => events.emit(
            nguruvilu::events::TURN_COMPLETE,
            &serde_json::json!({
                "session": &session_id,
                "ok": false,
                "duration_ms": turn_ms,
                "error": nguruvilu::events::truncate(&format!("{error:#}"), 1000),
            }),
        ),
    }

    // The turn gives its slots back under a short lock, and carries out of it
    // one thing for the file work below: a second handle on the store. Nothing
    // is written while the lock is held, so a slow append costs this turn's own
    // bookkeeping and nothing else — not the page's commands, and not another
    // session's events.
    let (followup, store) = {
        let mut guard = state.lock().expect("state lock");
        guard.running.remove(&session_id);
        guard.cancels.remove(&session_id);
        // Whatever arrived after this turn's last claim still deserves its turn:
        // take the oldest pending message; the chained run's own first step
        // claims the rest. A cancel cleared the queue already, so stopping stops
        // the chain too.
        let followup = guard
            .inboxes
            .get(&session_id)
            .and_then(|queue| queue.lock().expect("inbox").pop_front());
        // A handle opened at the same root is the same store, because the store
        // is a bare root path with no shared state behind it. Taking that copy
        // is what lets the rows be written with the lock released.
        (followup, store_handle(&guard.store))
    };
    let mut end: Option<Value> = None;
    let mut failure: Option<String> = None;
    // What the turn leaves in memory, applied under the one lock taken after
    // the disk work. The reduced history is a flag rather than a copy: it is
    // only worth taking when this session is still the one on screen, which is
    // decided under the lock that applies it.
    let mut adopt_history = false;
    let mut prompt_tokens: Option<usize> = None;
    let mut usage_row: Option<nguruvilu::session::TurnStats> = None;

    match result {
        Ok(outcome) => {
            // A one-line trace on stderr: the window shows the answer, but a
            // terminal launch should also be able to see that a turn happened.
            let preview: String = outcome.text.chars().take(100).collect();
            eprintln!(
                "[turn] {} steps, {} tools, {} retries, {} in / {} out ({} cached), model {} ms, tools {} ms: {}",
                outcome.steps,
                outcome.tool_calls,
                outcome.retries,
                outcome.usage.input,
                outcome.usage.output,
                outcome.usage.cached,
                outcome.timing.model_ms,
                outcome.timing.tools_ms,
                preview.replace('\n', " ")
            );

            // Messages were already written by `on_message`, the moment each
            // one was committed — a kill mid-turn loses only what never
            // completed, never a whole turn's worth. What remains here is the
            // metadata that has no message to ride along with.
            //
            // That metadata is written with the state lock released: the rows go
            // out through the second handle taken above, so the page's commands
            // and every other session's events stay live while the disk works.
            // A handle that could not be opened means the store directory itself
            // is unusable, which is the one case where falling back to the
            // state's own handle makes sense: those appends are about to fail
            // anyway, and the slower path still reports each failure instead of
            // losing the row silently.
            let usage = match &store {
                Ok(store) => write_turn_rows(store, &session_id, &outcome),
                Err(_) => {
                    let guard = state.lock().expect("state lock");
                    write_turn_rows(&guard.store, &session_id, &outcome)
                }
            };

            // The in-memory half of the tail: adopting the agent's reduced
            // history so the live session matches the file, and remembering the
            // prompt size and the usage row. All three need the lock, which is
            // taken again below, after the writes.
            adopt_history = !outcome.compactions.is_empty();
            if outcome.usage.input > 0 {
                prompt_tokens = Some(outcome.usage.input as usize);
            }
            usage_row = usage;

            if let Some(error) = outcome.error {
                failure = Some(error);
            } else {
                end = Some(json!({
                    "ev": "turn_end",
                    "steps": outcome.steps,
                    "tool_calls": outcome.tool_calls,
                    "retries": outcome.retries,
                    "usage": {
                        "input": outcome.usage.input,
                        "output": outcome.usage.output,
                        "cached": outcome.usage.cached,
                    },
                    "timing": {
                        "model_ms": outcome.timing.model_ms,
                        "tools_ms": outcome.timing.tools_ms,
                    },
                    "config_version": outcome.config_version,
                    "session": &session_id,
                    "messages": agent.messages().len(),
                }));
            }
        }
        Err(error) => {
            // Nothing to salvage here: every message the agent committed was
            // already written by `on_message` as it happened.
            failure = Some(format!("{error:#}"));
        }
    }

    // Only memory is left, and only memory needs the lock. It is taken here,
    // after the disk work, and held for the rest of the function exactly as
    // before. Whether this session is the one on screen is decided now rather
    // than before the writes: the user may have switched conversations while
    // the rows were on their way out, and every update below belongs to the
    // conversation being shown.
    let mut guard = state.lock().expect("state lock");
    let is_current = guard.session.id == session_id;
    if adopt_history && is_current {
        // The file holds the summary in place of the messages it replaced, so
        // the live transcript follows the agent's reduced history or the page
        // keeps drawing what the model no longer sees.
        guard.session.messages = agent.messages().to_vec();
    }
    if let Some(tokens) = prompt_tokens {
        guard.last_prompt_tokens = Some(tokens);
    }
    if let Some(row) = usage_row {
        if is_current {
            // The live session is the source of `describe`'s totals; the row
            // just written must count there too, or the usage line would trail
            // the turn that produced it by one.
            guard.session.stats.push(row);
        }
    }

    // The error belongs to the session that hit it: switching away mid-turn
    // must not paint the session now on screen as failed.
    if let (Some(error), true) = (&failure, is_current) {
        guard.last_error = Some(error.clone());
    }
    let status = guard.describe();
    let sessions = match guard.sessions() {
        Ok(list) => Some(list),
        Err(error) => {
            eprintln!("[sessions] refresh failed: {error:#}");
            None
        }
    };
    drop(guard);

    if let Some(list) = sessions {
        sink.emit(json!({ "ev": "sessions", "list": list }));
    }
    sink.emit(json!({ "ev": "status", "status": status }));

    match (end, failure) {
        (_, Some(error)) => sink.emit(
            json!({ "ev": "error", "message": error, "session": session_id }),
        ),
        (Some(payload), None) => sink.emit(payload),
        (None, None) => {}
    }

    // One chained turn per completion: emitted events above first, so the page
    // sees the old turn end before the new one starts. If a fresher prompt won
    // the running slot meanwhile, the message is queued again — that turn
    // claims it — rather than spun in a loop here.
    // Whatever arrived after this turn's last claim still deserves its turn:
    // the oldest pending message is returned for the cascade to run, and its
    // own first step claims the rest. A cancel cleared the queue already, so
    // stopping stops the chain too.
    Ok(followup)
}

/// Run one prompt, then every steering message that arrived too late for the
/// turn to claim — each as its own turn, in order.
///
/// A loop instead of a spawn from inside `run_turn`: awaiting the same async
/// fn it lives in makes the future reference itself, which the compiler
/// settles as "not Send". Here the runs are plain sequential awaits, so the
/// queue drains until empty — and if another turn won the running slot in
/// between, the message is queued for *it* to claim and this cascade ends.
///
/// The whole cascade belongs to the conversation its first turn started in:
/// the target is resolved once, here, and every follow-up carries that id.
/// Re-reading "what is on screen" for each link of the chain is how a
/// background conversation ends up writing into whichever one the user
/// switched to — the follow-up would run in the wrong session's history, with
/// the wrong file behind it.
pub async fn run_turn_cascading(
    state: Arc<Mutex<AppState>>,
    target: Option<String>,
    text: String,
    sink: Arc<dyn EventSink>,
) -> Result<()> {
    let target = match target {
        Some(id) => id,
        None => state.lock().expect("state lock").session.id.clone(),
    };
    let mut next = run_turn(
        Arc::clone(&state),
        Some(target.clone()),
        text,
        Arc::clone(&sink),
    )
    .await?;
    while let Some(text) = next {
        next = run_turn(
            Arc::clone(&state),
            Some(target.clone()),
            text,
            Arc::clone(&sink),
        )
        .await?;
    }
    Ok(())
}

/// Route settled `notify` jobs to the session that started them.
///
/// Installed once per shell process (the job registry is process-wide).
/// While that session's turn still runs, the message is steering, claimed at
/// the next step; when the session is idle, it opens as its own turn — a
/// program that finishes after the model moved on still hands back its result
/// and the work continues without anyone pressing anything.
pub fn install_job_wake(state: Arc<Mutex<AppState>>, sink: Arc<dyn EventSink>) {
    nguruvilu::tools::jobs::set_settlement_hook(Arc::new(move |job_id, label| {
        // A job that never asked to notify (or ran in a host without an
        // observer) has nobody to wake -- leave it to job_list.
        let registered = {
            let guard = state.lock().expect("state lock");
            guard.app_jobs.get(job_id).map(|job| (job.session.clone(), Arc::clone(&job.sink)))
        };
        let Some((session, owner_sink)) = registered else {
            let _ = sink; // install-time sink unused when nothing is registered
            return;
        };
        // Drop the attribution: one settlement, one wake.
        {
            let mut guard = state.lock().expect("state lock");
            guard.app_jobs.remove(job_id);
            guard.job_sessions.remove(job_id);
        }

        let text = format!(
            "Background job {job_id} finished: {label}. Its output is ready — \
             collect it with job_output, then continue the task."
        );
        // Queue as steering first: while the owner's turn runs, the next step
        // boundary claims it -- the same path a user prompt takes.
        let steered = {
            let mut guard = state.lock().expect("state lock");
            let is_running = guard.running.contains(&session);
            if is_running {
                let queue = guard
                    .inboxes
                    .entry(session.clone())
                    .or_insert_with(|| Arc::new(std::sync::Mutex::new(VecDeque::new())));
                queue.lock().expect("inbox").push_back(text.clone());
            }
            is_running
        };
        if steered {
            // A live turn claims it at the next step; announce the steering so
            // the page shows it (only a real steer reaches the model this way).
            owner_sink.emit(json!({ "ev": "steer", "text": text, "session": session }));
        } else {
            // Idle: open its own turn through the same entry a prompt uses, on
            // the *owner's* sink (the install-time sink is only correct for
            // window/headless, not the multi-session SSE path). A distinct
            // event -- not "steer" -- so the page does not render the wake as
            // if the user had typed it. The wake carries its owner's id: if
            // the user has moved to another conversation, the turn parks in
            // the owner's inbox rather than opening in the one on screen.
            let wake_state = Arc::clone(&state);
            tokio::spawn(run_turn_cascading(
                wake_state,
                Some(session.clone()),
                text.clone(),
                owner_sink.clone(),
            ));
            owner_sink.emit(json!({
                "ev": "job_wake",
                "job": job_id,
                "text": text,
                "session": session
            }));
        }
    }));
}

/// Install an archive and report what landed.
///
/// A free function rather than a method: installing fetches, and holding the
/// shell's state lock across an await would stall every other reader for the
/// length of a download.
pub async fn install_pack(archive: &Path) -> Result<Value> {
    let dir = nguruvilu::pack::default_packs_dir();
    let placed = nguruvilu::pack::install(archive, &dir).await?;
    Ok(json!({
        "name": placed.manifest.name,
        "version": placed.manifest.version_id,
        "path": placed.path.display().to_string(),
        "assembly": placed.assembly.as_ref().map(|p| p.display().to_string()),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex as StdMutex;
    use std::time::Instant;

    /// Records everything the shell pushes, so a test can assert which session
    /// each event belonged to.
    struct RecordingSink {
        events: StdMutex<Vec<Value>>,
        /// When each event was emitted, with its name and session.
        ///
        /// Concurrency is about ordering in time. A test that compares two
        /// durations assumes both were produced under the same load; "B reached
        /// the provider before A's turn ended" instead needs only the second
        /// half to be a fact, so the sink records it.
        stamps: StdMutex<Vec<(Instant, String, Option<String>)>>,
        /// Run for every event as it is emitted.
        ///
        /// Whether a file already holds a row at the moment a turn reports
        /// itself done is a fact about that moment, not about a later read: a
        /// test that checks the disk after `run_turn` returned cannot tell a row
        /// written before the event from one written after it.
        hook: StdMutex<Option<Arc<dyn Fn(&Value) + Send + Sync>>>,
    }

    impl RecordingSink {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                events: StdMutex::new(Vec::new()),
                stamps: StdMutex::new(Vec::new()),
                hook: StdMutex::new(None),
            })
        }

        /// Run `hook` for every event emitted from here on.
        fn watch(&self, hook: Arc<dyn Fn(&Value) + Send + Sync>) {
            *self.hook.lock().expect("recording sink hook") = Some(hook);
        }

        fn events(&self) -> Vec<Value> {
            self.events.lock().expect("recording sink").clone()
        }

        /// When the event named `name` was emitted for `session`.
        fn stamp(&self, name: &str, session: &str) -> Option<Instant> {
            self.stamps
                .lock()
                .expect("recording sink stamps")
                .iter()
                .find(|(_, event, event_session)| {
                    event == name && event_session.as_deref() == Some(session)
                })
                .map(|(at, _, _)| *at)
        }
    }

    impl EventSink for RecordingSink {
        fn emit(&self, event: Value) {
            let name = event["ev"].as_str().unwrap_or_default().to_string();
            let session = event["session"].as_str().map(str::to_string);
            self.stamps
                .lock()
                .expect("recording sink stamps")
                .push((Instant::now(), name, session));
            // Taken out from under the hook's lock before it runs: the hook
            // looks at the disk, and holding this lock while it did would make
            // the check's own timing the thing it measures.
            let hook = self.hook.lock().expect("recording sink hook").clone();
            if let Some(hook) = hook {
                hook(&event);
            }
            self.events.lock().expect("recording sink").push(event);
        }
    }

    /// A hook that records, for `session`, whether its usage row was already in
    /// the file when the turn reported itself done.
    ///
    /// The file is read through a handle opened here, at the moment of the
    /// event, so what it reports is the disk and not the shell's in-memory
    /// transcript.
    fn usage_row_watcher(
        root: &Path,
        session: &str,
        seen: &Arc<StdMutex<Vec<(String, bool)>>>,
    ) -> Arc<dyn Fn(&Value) + Send + Sync> {
        let root = root.to_path_buf();
        let session = session.to_string();
        let seen = Arc::clone(seen);
        Arc::new(move |event: &Value| {
            if event["ev"] != "turn_end" {
                return;
            }
            let written = JsonlStore::open(&root)
                .and_then(|store| store.load(&session))
                .ok()
                .flatten()
                .map(|record| !record.stats.is_empty())
                .unwrap_or(false);
            seen.lock().expect("rows at done").push((session.clone(), written));
        })
    }

    /// A provider that holds every answer for `delay` before sending it, and
    /// records when each request arrived.
    ///
    /// The hold is what makes concurrency observable: if two turns run at the
    /// same time, the second request reaches the provider while the first
    /// answer is still being held. If they are serialized, the second arrives
    /// only after the first hold releases — a gap no scheduling jitter can
    /// fake.
    async fn held_provider(
        delay: std::time::Duration,
    ) -> (String, Arc<StdMutex<Vec<(Instant, String)>>>) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let arrivals: Arc<StdMutex<Vec<(Instant, String)>>> = Arc::new(StdMutex::new(Vec::new()));
        let seen = Arc::clone(&arrivals);

        tokio::spawn(async move {
            while let Ok((mut socket, _)) = listener.accept().await {
                // Each connection is handled on its own task: holding the
                // accept loop across a delay would queue the second client
                // behind the first and make two concurrent turns look
                // serialized.
                let seen = Arc::clone(&seen);
                tokio::spawn(async move {
                    // Drain the request head, then the declared body length,
                    // so the reply can close the connection without racing a
                    // client that is still writing.
                    let mut head = Vec::new();
                    loop {
                        let mut byte = [0u8; 1];
                        match socket.read(&mut byte).await {
                            Ok(0) | Err(_) => break,
                            Ok(_) => {
                                head.push(byte[0]);
                                if head.ends_with(b"\r\n\r\n") {
                                    break;
                                }
                            }
                        }
                    }
                    let head_text = String::from_utf8_lossy(&head);
                    let declared = head_text
                        .lines()
                        .filter_map(|line| {
                            let (key, value) = line.split_once(':')?;
                            key.eq_ignore_ascii_case("content-length")
                                .then(|| value.trim().parse::<usize>().ok())
                                .flatten()
                        })
                        .next()
                        .unwrap_or(0);
                    let mut body = Vec::with_capacity(declared);
                    while body.len() < declared {
                        let mut chunk = [0u8; 4096];
                        match socket.read(&mut chunk).await {
                            Ok(0) | Err(_) => break,
                            Ok(read) => body.extend_from_slice(&chunk[..read]),
                        }
                    }

                    let text = String::from_utf8_lossy(&body);
                    let label = if text.contains("TASK-A") {
                        "TASK-A"
                    } else if text.contains("TASK-B") {
                        "TASK-B"
                    } else {
                        "other"
                    };
                    seen.lock().expect("arrivals").push((Instant::now(), label.to_string()));

                    tokio::time::sleep(delay).await;
                    let reply = format!("Mock reply for {label}: done.");
                    let frame = serde_json::json!({
                        "choices": [{ "delta": { "content": reply }, "finish_reason": "stop" }]
                    });
                    let response = format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\nconnection: close\r\n\r\ndata: {frame}\n\ndata: [DONE]\n\n"
                    );
                    let _ = socket.write_all(response.as_bytes()).await;
                    let _ = socket.flush().await;
                });
            }
        });
        (format!("http://{addr}"), arrivals)
    }

    /// Two unrelated sessions run their turns at the same time.
    ///
    /// This is the window path: every page command is its own task, and a turn
    /// belongs to the session it started in. Two things must hold — both
    /// requests are in flight together, and each session's file ends up with
    /// its own answer and nothing from the other.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn two_unrelated_sessions_run_their_turns_at_the_same_time() {
        // Everything the shell persists lives under NGU_HOME, so this test
        // gets a directory of its own and the real sessions stay untouched.
        let home = std::env::temp_dir().join(format!("ngu-state-test-{}", uuid::Uuid::new_v4()));
        std::env::set_var("NGU_HOME", &home);

        let delay = std::time::Duration::from_millis(1500);
        let (base, arrivals) = held_provider(delay).await;

        let state = Arc::new(Mutex::new(AppState::bootstrap().expect("bootstrap")));
        state
            .lock()
            .expect("state lock")
            .set_model("mock-model", Some(&base))
            .expect("route points at the mock");

        // A handle on the store directory, taken before any turn runs: what the
        // watchers below record is the disk, not the shell's in-memory
        // transcript.
        let store_root = state.lock().expect("state lock").store.root().to_path_buf();
        // Per session: was its usage row already in its file at the moment its
        // turn reported itself done.
        let rows_at_done: Arc<StdMutex<Vec<(String, bool)>>> =
            Arc::new(StdMutex::new(Vec::new()));

        // Session A starts its turn.
        state.lock().expect("state lock").new_session().expect("session A");
        let session_a = state.lock().expect("state lock").session.id.clone();
        let sink_a = RecordingSink::new();
        sink_a.watch(usage_row_watcher(&store_root, &session_a, &rows_at_done));
        let sink_a_trait: Arc<dyn EventSink> = sink_a.clone();
        let task_a = tokio::spawn(run_turn(
            Arc::clone(&state),
            None,
            "TASK-A: first task".into(),
            sink_a_trait,
        ));

        // B must start while A is genuinely in flight, not merely scheduled
        // before it: wait for A's running slot.
        let deadline = Instant::now() + std::time::Duration::from_secs(10);
        loop {
            if state.lock().expect("state lock").running.contains(&session_a) {
                break;
            }
            assert!(Instant::now() < deadline, "session A never started running");
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }

        // Session B switches in and sends its own task while A runs.
        state.lock().expect("state lock").new_session().expect("session B");
        let session_b = state.lock().expect("state lock").session.id.clone();
        assert_ne!(session_a, session_b);
        let sink_b = RecordingSink::new();
        sink_b.watch(usage_row_watcher(&store_root, &session_b, &rows_at_done));
        let sink_b_trait: Arc<dyn EventSink> = sink_b.clone();
        let task_b = tokio::spawn(run_turn(
            Arc::clone(&state),
            None,
            "TASK-B: second task".into(),
            sink_b_trait,
        ));

        task_a.await.expect("task A panicked").expect("task A failed");
        task_b.await.expect("task B panicked").expect("task B failed");

        // Neither turn was refused: each session saw its own start and end.
        for (sink, session) in [(&sink_a, &session_a), (&sink_b, &session_b)] {
            let events = sink.events();
            let mine = |name: &str| {
                events
                    .iter()
                    .filter(|event| event["ev"] == name)
                    .filter(|event| match event.get("session") {
                        Some(value) => value.as_str() == Some(session.as_str()),
                        None => true,
                    })
                    .count()
            };
            assert_eq!(mine("turn_start"), 1, "{session} saw one turn start");
            assert_eq!(mine("turn_end"), 1, "{session} saw its own turn end");
            let rejections = events
                .iter()
                .filter(|event| {
                    event["ev"] == "error"
                        && event["message"]
                            .as_str()
                            .is_some_and(|m| m.contains("already working"))
                })
                .count();
            assert_eq!(rejections, 0, "{session} was never refused as busy");
        }

        // Both requests were in flight together: B arrived at the provider
        // while A's answer was still inside its hold window.
        let seen = arrivals.lock().expect("arrivals").clone();
        assert_eq!(seen.len(), 2, "both turns reached the provider: {seen:?}");
        let labels: Vec<&str> = seen.iter().map(|(_, label)| label.as_str()).collect();
        assert!(labels.contains(&"TASK-A"), "{labels:?}");
        assert!(labels.contains(&"TASK-B"), "{labels:?}");
        // The ordering that proves overlap is "B reached the provider while A's
        // turn was still running", not "the two arrivals are close together":
        // how long a turn spends before its first request leaves the shell is
        // machine load, and a gap comparison reads that as serialization.
        let b_at = seen
            .iter()
            .find(|(_, label)| label == "TASK-B")
            .map(|(at, _)| *at)
            .expect("B reached the provider");
        let a_ended = sink_a
            .stamp("turn_end", &session_a)
            .expect("A's turn end is stamped");
        assert!(
            b_at < a_ended,
            "B's request reached the provider after A's turn had already ended \
             (A ended {:?} before B's request): the turns ran one after the other",
            b_at.duration_since(a_ended)
        );

        // The usage row was in the file when the turn reported itself done,
        // read through a handle opened at that moment. The tail writes its rows
        // with the state lock released, and that concession must not move them
        // past the event the page redraws its usage line on.
        let watched = rows_at_done.lock().expect("rows at done").clone();
        for session in [session_a.as_str(), session_b.as_str()] {
            let at_done = watched
                .iter()
                .find(|(seen, _)| seen.as_str() == session)
                .map(|(_, written)| *written);
            assert_eq!(
                at_done,
                Some(true),
                "{session} had its usage row on disk when turn_end was emitted: {watched:?}"
            );
        }

        // Each session's file holds its own answer, and only its own.
        //
        // The guard is scoped: `let store = &state.lock()…store;` extends the
        // guard's temporary to the end of the test, and the `state.lock()` at
        // the bottom of this function then waits on the lock it is itself
        // holding — a self-deadlock that reads as a hung suite, not a failure.
        let (a, b) = {
            let guard = state.lock().expect("state lock");
            (
                guard.store.load(&session_a).expect("load A").expect("session A exists"),
                guard.store.load(&session_b).expect("load B").expect("session B exists"),
            )
        };
        let text_of = |messages: &[nguruvilu::message::Message]| -> String {
            messages.iter().map(|m| m.text()).collect::<Vec<_>>().join("\n")
        };
        let a_text = text_of(&a.messages);
        let b_text = text_of(&b.messages);
        assert!(a_text.contains("Mock reply for TASK-A"), "A has its answer: {a_text}");
        assert!(!a_text.contains("TASK-B"), "B never leaked into A");
        assert!(b_text.contains("Mock reply for TASK-B"), "B has its answer: {b_text}");
        assert!(!b_text.contains("TASK-A"), "A never leaked into B");

        // Nothing left marked busy once both turns settled.
        assert!(state.lock().expect("state lock").running.is_empty());

        let _ = std::fs::remove_dir_all(&home);
    }

    /// A turn whose session was deleted mid-turn writes no rows at all.
    ///
    /// The metadata of a finished turn belongs to the file the conversation
    /// lives in. When the user deletes that conversation while the turn runs,
    /// the tail must not write into what is left of it: rows appended to the
    /// remnant would resurrect an empty shell of the conversation the user asked
    /// to be rid of. What the mid-turn message writes leave behind is a
    /// header-less file that no longer loads as a session, which is exactly the
    /// state the deleted conversation already fails to pass.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_turn_whose_session_was_deleted_mid_turn_writes_no_rows() {
        // Everything the shell persists lives under NGU_HOME, so this test
        // gets a directory of its own and the real sessions stay untouched.
        let home = std::env::temp_dir().join(format!("ngu-state-test-{}", uuid::Uuid::new_v4()));
        std::env::set_var("NGU_HOME", &home);

        let delay = std::time::Duration::from_millis(1500);
        let (base, _arrivals) = held_provider(delay).await;

        let state = Arc::new(Mutex::new(AppState::bootstrap().expect("bootstrap")));
        state
            .lock()
            .expect("state lock")
            .set_model("mock-model", Some(&base))
            .expect("route points at the mock");

        state.lock().expect("state lock").new_session().expect("session A");
        let session_a = state.lock().expect("state lock").session.id.clone();
        let store_root = state.lock().expect("state lock").store.root().to_path_buf();
        let sink = RecordingSink::new();
        let sink_trait: Arc<dyn EventSink> = sink.clone();
        let turn = tokio::spawn(run_turn(
            Arc::clone(&state),
            None,
            "DELETED: mid-turn".into(),
            sink_trait,
        ));

        // The deletion has to land while the turn is genuinely in flight, so
        // wait for its running slot — taken before the request is sent — and
        // delete afterwards.
        let deadline = Instant::now() + std::time::Duration::from_secs(10);
        loop {
            if state.lock().expect("state lock").running.contains(&session_a) {
                break;
            }
            assert!(Instant::now() < deadline, "session A never started running");
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
        // The conversation is on disk before the delete, so what this test
        // asserts is a row the tail chose not to write, not one with nowhere to
        // go.
        assert!(
            JsonlStore::open(&store_root)
                .expect("reopen the store")
                .load(&session_a)
                .expect("load A")
                .is_some(),
            "session A existed before the delete"
        );
        state.lock().expect("state lock").store.delete(&session_a).expect("delete A");

        turn.await.expect("turn panicked").expect("turn returns");

        // The deletion costs the rows, not the turn: the page still hears that
        // the turn ended.
        assert!(
            sink.events().iter().any(|event| event["ev"] == "turn_end"
                && event["session"].as_str() == Some(session_a.as_str())),
            "the turn still reported its end: {:?}",
            sink.events()
        );

        // No usage row for a session that is no longer there. The remnant the
        // turn's own messages left is read raw: a header-based check takes it
        // for "not a session", and a line the tail should never have written
        // would hide behind that answer.
        let path = store_root.join(format!("{session_a}.jsonl"));
        if let Ok(remnant) = std::fs::read_to_string(&path) {
            assert!(
                !remnant.contains("\"type\":\"stats\""),
                "a deleted session takes its usage row with it: {remnant}"
            );
            assert!(
                !remnant.contains("\"type\":\"compaction\""),
                "a deleted session takes its compaction rows with it: {remnant}"
            );
        }
        assert!(
            JsonlStore::open(&store_root)
                .expect("reopen the store")
                .load(&session_a)
                .expect("load A")
                .is_none(),
            "a session deleted mid-turn is not reloadable afterwards"
        );

        // The compaction row is checked where a turn would actually write one.
        // The mock provider cannot drive a real compaction, so the row is handed
        // to the same function the tail uses, for the session just deleted.
        let compacted = nguruvilu::agent::AgentOutcome {
            text: "gone".to_string(),
            steps: 1,
            tool_calls: 0,
            usage: nguruvilu::llm::TokenUsage { input: 40, output: 8, ..Default::default() },
            new_messages: Vec::new(),
            timing: nguruvilu::agent::TurnTiming::default(),
            config_version: 1,
            injections: Vec::new(),
            compactions: vec![nguruvilu::compaction::Compaction {
                replaced: 3,
                summary: "earlier turns".to_string(),
            }],
            error: None,
            retries: 0,
        };
        let store = JsonlStore::open(&store_root).expect("reopen the store");
        assert!(
            write_turn_rows(&store, &session_a, &compacted).is_none(),
            "no usage row is reported written for a deleted session"
        );
        let after = std::fs::read_to_string(&path).unwrap_or_default();
        assert!(
            !after.contains("\"type\":\"stats\"") && !after.contains("\"type\":\"compaction\""),
            "a compaction-bearing turn adds no rows to a deleted session: {after}"
        );

        let _ = std::fs::remove_dir_all(&home);
    }

    /// A cascade's follow-up belongs to the session its first turn started in.
    ///
    /// The user switches away while A's turn runs; the steering that arrives
    /// too late for that turn to claim is the follow-up. The old code fed it
    /// back through "what is on screen", so it ran in B and landed in B's
    /// file — a whole answer for conversation A written into conversation B.
    /// With the target carried through the chain, it runs in A.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_cascades_followup_lands_in_a_even_after_switching_to_b() {
        // Everything the shell persists lives under NGU_HOME, so this test
        // gets a directory of its own and the real sessions stay untouched.
        let home = std::env::temp_dir().join(format!("ngu-state-test-{}", uuid::Uuid::new_v4()));
        std::env::set_var("NGU_HOME", &home);

        // The hold is what makes the switch observable: A's turn is genuinely
        // in flight when the user moves to B, so anything the cascade does
        // afterwards happens with B on screen.
        let delay = std::time::Duration::from_millis(1500);
        let (base, _arrivals) = held_provider(delay).await;

        let state = Arc::new(Mutex::new(AppState::bootstrap().expect("bootstrap")));
        state
            .lock()
            .expect("state lock")
            .set_model("mock-model", Some(&base))
            .expect("route points at the mock");

        // Session A gets a conversation of its own.
        state.lock().expect("state lock").new_session().expect("session A");
        let session_a = state.lock().expect("state lock").session.id.clone();
        let sink = RecordingSink::new();
        let sink_dyn: Arc<dyn EventSink> = sink.clone();

        // A's first turn starts, steering queued during it becomes the
        // follow-up the cascade runs. Sent through the cascade, exactly the
        // way a prompt enters the shell.
        let cascade = tokio::spawn(run_turn_cascading(
            Arc::clone(&state),
            Some(session_a.clone()),
            "TASK-A: follow me".into(),
            Arc::clone(&sink_dyn),
        ));

        // Wait for A's turn to hold the running slot, then park the follow-up
        // in A's inbox the way a steering message arrives.
        let deadline = Instant::now() + std::time::Duration::from_secs(10);
        loop {
            if state.lock().expect("state lock").running.contains(&session_a) {
                break;
            }
            assert!(Instant::now() < deadline, "session A never started running");
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
        // The user switches to B while A runs.
        state.lock().expect("state lock").new_session().expect("session B");
        let session_b = state.lock().expect("state lock").session.id.clone();
        assert_ne!(session_a, session_b);
        // The late steering lands in A's inbox — A is no longer the
        // conversation on screen, which is exactly the case this test is
        // about. When the turn already finished between the switch and this
        // push, the same text goes through the cascade entry a job wake uses:
        // the point is where it ends up, not which door it came through.
        let turn_still_running = {
            let mut guard = state.lock().expect("state lock");
            let running = guard.running.contains(&session_a);
            if running {
                guard
                    .inboxes
                    .entry(session_a.clone())
                    .or_insert_with(|| Arc::new(StdMutex::new(VecDeque::new())))
                    .lock()
                    .expect("inbox")
                    .push_back("TASK-A: second task".to_string());
            }
            running
        };
        if !turn_still_running {
            run_turn_cascading(
                Arc::clone(&state),
                Some(session_a.clone()),
                "TASK-A: second task".into(),
                Arc::clone(&sink_dyn),
            )
            .await
            .expect("parked cascade");
        }

        cascade.await.expect("cascade panicked").expect("cascade failed");

        let text_of = |messages: &[nguruvilu::message::Message]| -> String {
            messages.iter().map(|m| m.text()).collect::<Vec<_>>().join("\n")
        };

        // Nothing ran in B: the follow-up was aimed at A, and A is not on
        // screen. This is the assertion the old code failed — it ran the
        // follow-up in B and wrote A's answer into B's file.
        {
            let guard = state.lock().expect("state lock");
            let b = guard.store.load(&session_b).expect("load B").expect("session B exists");
            let b_text = text_of(&b.messages);
            assert!(!b_text.contains("TASK-A"), "A's follow-up must not run in B: {b_text}");
        }
        // Not lost either: it waits in A's inbox until A gets a turn again —
        // unless the running turn claimed the steering before the switch, in
        // which case it is already in A's file.
        let parked: Vec<String> = state
            .lock()
            .expect("state lock")
            .inboxes
            .get(&session_a)
            .map(|queue| queue.lock().expect("inbox").iter().cloned().collect())
            .unwrap_or_default();
        assert!(
            parked.is_empty() || parked == vec!["TASK-A: second task".to_string()],
            "A's inbox holds at most the text parked for A: {parked:?}"
        );

        // The user comes back to A; its next turn claims the parked text.
        state
            .lock()
            .expect("state lock")
            .open_session(&session_a)
            .expect("back to A");
        let sink_dyn: Arc<dyn EventSink> = sink.clone();
        let mut next = run_turn(
            Arc::clone(&state),
            None,
            "TASK-A: collecting".into(),
            Arc::clone(&sink_dyn),
        )
        .await
        .expect("turn after returning to A");
        while let Some(text) = next {
            next = run_turn(Arc::clone(&state), None, text, Arc::clone(&sink_dyn))
                .await
                .expect("follow-up turn for A");
        }

        // A's file holds both of its tasks and nothing of B's was touched.
        let (a_text, b_text) = {
            let guard = state.lock().expect("state lock");
            let a = guard.store.load(&session_a).expect("load A").expect("session A exists");
            let b = guard.store.load(&session_b).expect("load B").expect("session B exists");
            (text_of(&a.messages), text_of(&b.messages))
        };
        assert!(a_text.contains("TASK-A: follow me"), "A ran its own first task: {a_text}");
        assert!(
            a_text.contains("TASK-A: second task"),
            "A's follow-up ran in A: {a_text}"
        );
        assert!(a_text.contains("Mock reply for TASK-A"), "A has its answers: {a_text}");
        assert!(!b_text.contains("TASK-A"), "B holds none of A's conversation: {b_text}");

        let _ = std::fs::remove_dir_all(&home);
    }

    /// A turn aimed at a conversation the user is not looking at must not
    /// take the view: the text parks in the target's inbox, nothing runs, and
    /// the conversation on screen is untouched. The parked text is claimed by
    /// the target's next turn.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_prompt_for_a_background_session_parks_instead_of_grabbing_the_view() {
        let home = std::env::temp_dir().join(format!("ngu-state-test-{}", uuid::Uuid::new_v4()));
        std::env::set_var("NGU_HOME", &home);

        let delay = std::time::Duration::from_millis(1500);
        let (base, _arrivals) = held_provider(delay).await;

        let state = Arc::new(Mutex::new(AppState::bootstrap().expect("bootstrap")));
        state
            .lock()
            .expect("state lock")
            .set_model("mock-model", Some(&base))
            .expect("route points at the mock");

        // A exists in the store; B is the conversation on screen.
        state.lock().expect("state lock").new_session().expect("session A");
        let session_a = state.lock().expect("state lock").session.id.clone();
        state.lock().expect("state lock").new_session().expect("session B");
        let session_b = state.lock().expect("state lock").session.id.clone();
        let sink = RecordingSink::new();
        let sink_trait: Arc<dyn EventSink> = sink.clone();

        // A turn aimed at A while B is on screen: it must not switch.
        let outcome = run_turn(
            Arc::clone(&state),
            Some(session_a.clone()),
            "PARKED: not on screen".into(),
            sink_trait,
        )
        .await
        .expect("parked turn returns");
        assert!(outcome.is_none(), "nothing runs for a parked prompt");
        assert_eq!(
            state.lock().expect("state lock").session.id,
            session_b,
            "the view stayed on B"
        );
        let events = sink.events();
        assert!(
            !events.iter().any(|event| event["ev"] == "turn_start"),
            "no turn started while parked: {events:?}"
        );
        // The text waits where A's next turn will claim it.
        let parked: Vec<String> = state
            .lock()
            .expect("state lock")
            .inboxes
            .get(&session_a)
            .map(|queue| queue.lock().expect("inbox").iter().cloned().collect())
            .unwrap_or_default();
        assert_eq!(
            parked,
            vec!["PARKED: not on screen".to_string()],
            "the prompt waits in A's inbox"
        );
        assert!(
            state.lock().expect("state lock").running.is_empty(),
            "a parked prompt never marked anything running"
        );

        let _ = std::fs::remove_dir_all(&home);
    }

    /// A committed message reaches the file before `on_message` returns.
    ///
    /// The append now happens outside the state lock, and this is the property
    /// that change could have broken without a single failing command: the
    /// message must be on disk by the time the observer hands control back,
    /// because the force-kill that a turn-end batch would lose can happen
    /// immediately afterwards. The file is read through a handle that never
    /// passed through the state lock, so what is asserted here is the disk, not
    /// the in-memory transcript.
    #[test]
    fn an_observed_message_is_on_disk_when_on_message_returns() {
        // Everything the shell persists lives under NGU_HOME, so this test
        // gets a directory of its own and the real sessions stay untouched.
        let home = std::env::temp_dir().join(format!("ngu-state-test-{}", uuid::Uuid::new_v4()));
        std::env::set_var("NGU_HOME", &home);

        let state = Arc::new(Mutex::new(AppState::bootstrap().expect("bootstrap")));
        let (session_id, root) = {
            let guard = state.lock().expect("state lock");
            (guard.session.id.clone(), guard.store.root().to_path_buf())
        };

        let sink: Arc<dyn EventSink> = RecordingSink::new();
        let observer = UiObserver::new(sink, Arc::clone(&state), session_id.clone());
        observer.on_message(&nguruvilu::message::Message::user("COMMITTED: written already"));

        let stored = JsonlStore::open(&root)
            .expect("reopen the store")
            .load(&session_id)
            .expect("load the session")
            .expect("the session exists");
        let on_disk: String =
            stored.messages.iter().map(|message| message.text()).collect::<Vec<_>>().join("\n");
        assert!(
            on_disk.contains("COMMITTED: written already"),
            "the message is on disk when on_message returns: {on_disk}"
        );

        // The transcript the page draws holds it too: this turn belongs to the
        // conversation on screen.
        let shown: String = state
            .lock()
            .expect("state lock")
            .session
            .messages
            .iter()
            .map(|message| message.text())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            shown.contains("COMMITTED: written already"),
            "the on-screen transcript holds it: {shown}"
        );

        let _ = std::fs::remove_dir_all(&home);
    }

    /// `stats_outside_lock` answers exactly what `stats` answers.
    ///
    /// The two entry points exist so the page can ask about a session that is
    /// not on screen without parking the whole shell behind the state lock, and
    /// they are interchangeable only if the JSON is identical. Both cases are
    /// compared: the open conversation, answered from the rows already in
    /// memory, and a stored one, answered by reading its file.
    #[test]
    fn stats_outside_lock_answers_the_same_json_as_stats() {
        let home = std::env::temp_dir().join(format!("ngu-state-test-{}", uuid::Uuid::new_v4()));
        std::env::set_var("NGU_HOME", &home);

        let state = Arc::new(Mutex::new(AppState::bootstrap().expect("bootstrap")));
        let (session_a, session_b) = {
            let mut guard = state.lock().expect("state lock");
            let row = nguruvilu::session::TurnStats {
                ts: "2026-01-01T00:00:00.000Z".to_string(),
                steps: 2,
                tool_calls: 1,
                input: 100,
                output: 20,
                cached: 10,
                cache_created: 5,
                reasoning: 3,
                model_ms: 40,
                tools_ms: 7,
                ..Default::default()
            };
            let session_a = guard.session.id.clone();
            guard.new_session().expect("session B");
            let session_b = guard.session.id.clone();
            // B's ledger exists only in its file, so asking about it takes the
            // load path; A is asked about as the open conversation, from
            // memory. An empty ledger on both sides would compare equal for the
            // wrong reason.
            guard.store.append_stats(&session_b, &row).expect("B's stats row");
            guard.open_session(&session_a).expect("back to A");
            guard.session.stats.push(row);
            (session_a, session_b)
        };
        assert_ne!(session_a, session_b);

        for requested in [None, Some(session_b.as_str())] {
            let under_lock = {
                state
                    .lock()
                    .expect("state lock")
                    .stats(requested)
                    .expect("stats under the lock")
            };
            let outside =
                AppState::stats_outside_lock(&state, requested).expect("stats outside the lock");
            assert_eq!(outside, under_lock, "{requested:?} answers the same JSON");
            // The page draws these three names; renaming one here would blank
            // the usage line even though both entry points agreed.
            assert!(outside.get("session").is_some(), "{requested:?} names its session");
            assert!(outside.get("totals").is_some(), "{requested:?} carries totals");
            assert!(outside.get("turns").is_some(), "{requested:?} carries the turn tail");
            assert_eq!(outside["totals"]["turns"], 1, "{requested:?} totals its one turn");
            assert_eq!(
                outside["turns"].as_array().map(|turns| turns.len()),
                Some(1),
                "{requested:?} sends its one turn row"
            );
        }

        // A session that does not exist fails through both doors; neither
        // invents an empty ledger for it.
        let missing = "20260101-000000-missing";
        assert!(state.lock().expect("state lock").stats(Some(missing)).is_err());
        assert!(AppState::stats_outside_lock(&state, Some(missing)).is_err());

        let _ = std::fs::remove_dir_all(&home);
    }

    /// The status push carries the session's mission board.
    ///
    /// The page paints the mission list from `status.missions` the moment a panel
    /// frame loads, so this field is what makes a mission visible before — and
    /// without — any tool answering. The layout asserted here is the one the
    /// mission tools write: `<workspace>/missions/<id>/mission.json`.
    #[test]
    fn the_status_board_lists_the_session_missions() {
        let home = std::env::temp_dir().join(format!("ngu-state-test-{}", uuid::Uuid::new_v4()));
        // The sessions directory is the handle `describe` has; the state home and the
        // workspaces under it are derived from it.
        let sessions = home.join("sessions");
        let workspace = missions_workspace(&sessions, "s-1", None).expect("a workspace path");
        assert_eq!(workspace, home.join("workspaces").join("s-1"));

        for id in ["m-20260101-000000", "m-20260102-000000"] {
            let dir = workspace.join("missions").join(id);
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(
                dir.join("mission.json"),
                json!({
                    "id": id,
                    "objective": format!("objective for {id}"),
                    "status": "running",
                })
                .to_string(),
            )
            .unwrap();
        }
        // A directory without a readable record is not a mission: the board skips it
        // instead of reporting a nameless row.
        std::fs::create_dir_all(workspace.join("missions").join("broken")).unwrap();

        let board = missions_under(&workspace);
        let rows = board.as_array().expect("an array");
        assert_eq!(rows.len(), 2, "{board}");
        // Newest first: ids are time-ordered, and the newest mission is the one a
        // reader is looking for.
        assert_eq!(rows[0]["id"], "m-20260102-000000");
        assert_eq!(rows[1]["id"], "m-20260101-000000");
        assert_eq!(rows[0]["objective"], "objective for m-20260102-000000");
        assert_eq!(rows[0]["status"], "running");

        // A recorded working directory that still exists is where the tools wrote, so
        // it is where the board is read. A stale one falls back to the per-session
        // workspace instead of reading a path that is gone.
        let project = home.join("projects").join("pinned");
        std::fs::create_dir_all(&project).unwrap();
        assert_eq!(
            missions_workspace(&sessions, "s-1", project.to_str()),
            Some(project.clone())
        );
        assert_eq!(
            missions_workspace(&sessions, "s-1", Some("this-project-does-not-exist")),
            Some(home.join("workspaces").join("s-1"))
        );

        let _ = std::fs::remove_dir_all(&home);
    }

    /// A session with no missions reports an empty board, never an error.
    #[test]
    fn a_workspace_without_missions_reports_an_empty_board() {
        let home = std::env::temp_dir().join(format!("ngu-state-test-{}", uuid::Uuid::new_v4()));
        assert_eq!(missions_under(&home.join("nowhere")), json!([]));
        // An empty `missions/` directory is the same fact as a missing one, and both
        // are the ordinary case on a machine that never started a mission.
        std::fs::create_dir_all(home.join("missions")).unwrap();
        assert_eq!(missions_under(&home), json!([]));
        let _ = std::fs::remove_dir_all(&home);
    }
}
