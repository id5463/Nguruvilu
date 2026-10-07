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
use nguruvilu::session::{JsonlStore, Session, SessionStore};
use nguruvilu::settings::Settings;
use nguruvilu::window::ContextPolicy;
use nguruvilu::skills::{register_skill_tool, SkillRegistry};

use crate::sink::EventSink;

/// System prompt for the desktop shell.
pub const BASE_PROMPT: &str = "You are a coding agent. Use the tools to inspect and change \
files and to run commands. Prefer `read` over shelling out to view files, and `edit` for \
surgical changes. Keep answers short and concrete.";

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
}

impl TurnConfig for SharedConfig {
    fn settings(&self) -> TurnSettings {
        let runtime = self.runtime.lock().expect("runtime lock");
        let snapshot = runtime.snapshot();
        let mut system_prompt = snapshot.system_prompt(&self.base_prompt);
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
        });

        let store = JsonlStore::open(sessions_root())?;

        // Resume the most recent session when there is one: a desktop app that
        // forgets the conversation on every launch is not much of a desktop app.
        let session = match store.latest()? {
            Some(summary) => store.load(&summary.id)?.unwrap_or_else(|| Session::new(Some(model.clone()))),
            None => Session::new(Some(model.clone())),
        };
        if !store.root().join(format!("{}.jsonl", session.id)).exists() {
            store.create(&session)?;
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

        if let Err(error) = &scan {
            eprintln!("[skills] scan failed: {error:#}");
        }

        let state = Self {
            kernel,
            config,
            session,
            store,
            route,
            running: HashSet::new(),
            cancels: HashMap::new(),
            inboxes: HashMap::new(),
            settings,
            last_error: None,
            last_prompt_tokens: None,
            pending,
            mcp: Vec::new(),
            search_cell,
            judge_cell,
        };
        // The session this process boots into is already current: its start
        // belongs to bootstrap, not to the first prompt.
        state.kernel.events().emit(
            nguruvilu::events::SESSION_START,
            &serde_json::json!({ "session": &state.session.id }),
        );
        Ok(state)
    }

    /// A description of the current state, for the details panel and status bar.
    pub fn describe(&self) -> Value {
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

    /// The transcript of the current session, as renderable entries.
    pub fn transcript(&self) -> Value {
        let mut entries: Vec<Value> = self
            .session
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
            .collect::<Vec<_>>();

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
        json!(entries)
    }

    /// Start a fresh session.
    pub fn new_session(&mut self) -> Result<()> {
        let session = Session::new(Some(self.route.model.clone()));
        self.store.create(&session)?;
        self.session_switch(&session.id);
        self.session = session;
        Ok(())
    }

    /// Switch to a stored session.
    pub fn open_session(&mut self, id: &str) -> Result<()> {
        let session = self
            .store
            .load(id)?
            .ok_or_else(|| anyhow!("session not found: {id}"))?;
        self.session_switch(&session.id);
        self.session = session;
        Ok(())
    }

    /// Emit the session boundary for a switch: the old id ends (when there
    /// was a different one), the new one starts.
    ///
    /// These belong at the switch itself, not at process exit — a process can
    /// change sessions several times, and the audit log joins each turn to
    /// its session through this pair.
    fn session_switch(&self, next: &str) {
        let events = self.kernel.events();
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
    pub fn delete_session(&mut self, id: &str) -> Result<()> {
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
            proxy: proxy.trim().to_string(),
            context_window: context_window.filter(|t| *t > 0),
            compact_percent,
            compact_keep_recent: compact_keep_recent.max(1),
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
        };
        settings.save()?;
        self.settings = settings;

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

    /// Adopt what a pack carried: persona, route, context policy, rules.
    ///
    /// A pack's settings replace the standing ones — the pack is the unit the
    /// user chose, so it decides. The next turn reads them.
    pub fn adopt_pack_content(&mut self, content: nguruvilu::content::PackContent) {
        if let Some(models) = &content.models {
            if let Some(url) = &models.base_url {
                self.settings.base_url = url.clone();
            }
            if let Some(name) = &models.model {
                self.settings.model = name.clone();
            }
            if let Some(effort) = &models.reasoning_effort {
                self.settings.reasoning_effort = effort.clone();
            }
            if let Some(proxy) = &models.proxy {
                self.settings.proxy = proxy.clone();
            }
            if let Some(ceiling) = &models.max_output_tokens {
                self.settings.max_output_tokens = nguruvilu::size::parse_size(ceiling).ok();
            }
            for (key, value) in &models.extra_body {
                self.settings.extra_body.insert(key.clone(), value.clone());
            }
            // The key itself is never in a pack; only the name of the variable
            // holding it.
            if let Some(var) = &models.api_key_env {
                if let Ok(value) = std::env::var(var) {
                    if !value.trim().is_empty() {
                        self.settings.api_key = value;
                    }
                }
            }
        }

        if let Some(models) = &content.models {
            if let Some(network) = &models.network {
                self.settings.network = network.clone();
            }
        }

        if let Some(context) = &content.context {
            if let Some(window) = &context.window {
                self.settings.context_window = nguruvilu::size::parse_size(window).ok();
            }
            if let Some(percent) = context.compact_percent {
                self.settings.compact_percent = percent.min(100);
            }
            if let Some(keep) = context.compact_keep_recent {
                self.settings.compact_keep_recent = keep.max(1);
            }
        }

        // The context policy is rebuilt from the settings that just changed, so
        // a pack's window takes effect on the next turn rather than the next
        // launch.
        if let Ok(mut policy) = self.config.policy.lock() {
            *policy = std::sync::Arc::new(self.settings.context_policy(None));
        }

        // The persona is the joined soul of every installed pack — never this
        // pack's alone: applied one by one, load order would decide who the
        // model is. Best-effort here because adoption must not fail over the
        // persona; `installed_persona` itself reports read errors upward.
        match nguruvilu::content::installed_persona(
            &nguruvilu::pack::default_packs_dir(),
        ) {
            Ok(Some(persona)) => {
                if let Ok(mut runtime) = self.config.runtime.lock() {
                    let _ = runtime.apply(nguruvilu::hotreload::Change::session(
                        "pack",
                        nguruvilu::hotreload::ChangePayload::Persona(persona),
                    ));
                }
            }
            Ok(None) => {}
            Err(error) => eprintln!("[pack] persona: {error:#}"),
        }

        // The pack's settings replace the standing ones, so the route is
        // rebuilt and published with them: a pack that names another endpoint
        // or another output ceiling decides the next turn, not the next launch.
        self.route = ModelRoute::from_settings(&self.settings);
        if let Ok(mut runtime) = self.config.runtime.lock() {
            let _ = runtime.apply(nguruvilu::hotreload::Change::session(
                "pack",
                nguruvilu::hotreload::ChangePayload::ModelRoute(self.route.clone()),
            ));
        }

        if let Some(rules) = content.injections {
            if let Ok(mut injection) = self.config.injection.lock() {
                *injection = std::sync::Arc::new(rules);
            }
        }

        // A pack may be what configured search; the tool is put in step with
        // the settings when the kernel comes back (see `restore_kernel`).
        if let Some(search) = &content.search {
            search.apply(&mut self.settings.search);
        }
    }

    /// Installed packs.
    pub fn packs(&self) -> Result<Value> {
        let dir = nguruvilu::pack::default_packs_dir();
        let packs = nguruvilu::pack::installed(&dir)?;
        Ok(json!({
            "dir": dir.display().to_string(),
            "packs": packs.iter().map(|pack| json!({
                "name": pack.manifest.name,
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
pub fn data_directory() -> PathBuf {
    sessions_root().parent().map(|dir| dir.join("webview")).unwrap_or_else(|| PathBuf::from("."))
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
}

impl UiObserver {
    /// Build an observer that pushes events for `session` to the window and
    /// writes each committed message to `state`'s store as it happens.
    pub fn new(sink: Arc<dyn EventSink>, state: Arc<Mutex<AppState>>, session: String) -> Self {
        Self { sink, state, session }
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
    }

    fn on_tool_end(&self, name: &str, ok: bool, result: &str) {
        self.emit(json!({ "ev": "tool_end", "name": name, "ok": ok, "result": result }));
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
        let mut guard = self.state.lock().expect("state lock");
        if let Err(error) = guard
            .store
            .append(&self.session, std::slice::from_ref(message))
        {
            eprintln!("[store] incremental append failed: {error:#}");
        }
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
/// A failure keeps its progress: every message the agent committed is written
/// as it happens, and the error is reported alongside it.
///
/// Returns the oldest unclaimed steering message, if one arrived too late to
/// be claimed by this turn.
pub async fn run_turn(
    state: Arc<Mutex<AppState>>,
    text: String,
    sink: Arc<dyn EventSink>,
) -> Result<Option<String>> {
    // One turn per session, decided under the lock so two prompts racing for
    // the same history cannot both pass the check.
    let (stop_tx, stop_rx) = tokio::sync::watch::channel(false);
    let (session_id, client, config, messages, events, inbox) = {
        let mut guard = state.lock().expect("state lock");
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
        let session_id = guard.session.id.clone();
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
        )
    };
    let observer = Arc::new(UiObserver::new(
        Arc::clone(&sink),
        Arc::clone(&state),
        session_id.clone(),
    ));
    let mut agent = Agent::new(client, config, messages)
        .with_max_steps(50)
        .with_observer(Arc::clone(&observer) as Arc<dyn AgentObserver>)
        .with_cancel(stop_rx)
        .with_inbox(inbox);

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

    let mut guard = state.lock().expect("state lock");
    guard.running.remove(&session_id);
    guard.cancels.remove(&session_id);
    // Whatever arrived after this turn's last claim still deserves its turn:
    // take the oldest pending message; the chained run's own first step claims
    // the rest. A cancel cleared the queue already, so stopping stops the
    // chain too.
    let followup = guard
        .inboxes
        .get(&session_id)
        .and_then(|queue| queue.lock().expect("inbox").pop_front());
    let is_current = guard.session.id == session_id;
    let mut end: Option<Value> = None;
    let mut failure: Option<String> = None;

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
            for record in &outcome.injections {
                if let Err(error) = guard.store.append_injection(&session_id, record) {
                    eprintln!("[store] injection append failed: {error:#}");
                }
            }

            // Record compactions after the messages they replace, then adopt
            // the agent's reduced history so the live session matches the file.
            for compaction in &outcome.compactions {
                if let Err(error) = guard.store.append_compaction(&session_id, compaction) {
                    eprintln!("[store] compaction append failed: {error:#}");
                }
                eprintln!(
                    "[compacted {} messages into {} chars]",
                    compaction.replaced,
                    compaction.summary.chars().count()
                );
            }
            if !outcome.compactions.is_empty() && is_current {
                guard.session.messages = agent.messages().to_vec();
            }

            if outcome.usage.input > 0 {
                guard.last_prompt_tokens = Some(outcome.usage.input as usize);
            }

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
pub async fn run_turn_cascading(
    state: Arc<Mutex<AppState>>,
    text: String,
    sink: Arc<dyn EventSink>,
) -> Result<()> {
    let mut next = run_turn(Arc::clone(&state), text, Arc::clone(&sink)).await?;
    while let Some(text) = next {
        next = run_turn(Arc::clone(&state), text, Arc::clone(&sink)).await?;
    }
    Ok(())
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
    }

    impl RecordingSink {
        fn new() -> Arc<Self> {
            Arc::new(Self { events: StdMutex::new(Vec::new()) })
        }

        fn events(&self) -> Vec<Value> {
            self.events.lock().expect("recording sink").clone()
        }
    }

    impl EventSink for RecordingSink {
        fn emit(&self, event: Value) {
            self.events.lock().expect("recording sink").push(event);
        }
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

        // Session A starts its turn.
        state.lock().expect("state lock").new_session().expect("session A");
        let session_a = state.lock().expect("state lock").session.id.clone();
        let sink_a = RecordingSink::new();
        let sink_a_trait: Arc<dyn EventSink> = sink_a.clone();
        let task_a = tokio::spawn(run_turn(
            Arc::clone(&state),
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
        let sink_b_trait: Arc<dyn EventSink> = sink_b.clone();
        let task_b = tokio::spawn(run_turn(
            Arc::clone(&state),
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
        let gap = if seen[0].0 >= seen[1].0 {
            seen[0].0.duration_since(seen[1].0)
        } else {
            seen[1].0.duration_since(seen[0].0)
        };
        assert!(
            gap < delay,
            "the second request arrived {gap:?} after the first — outside the \
             {delay:?} hold, so the turns ran one after the other"
        );

        // Each session's file holds its own answer, and only its own.
        let store = &state.lock().expect("state lock").store;
        let a = store.load(&session_a).expect("load A").expect("session A exists");
        let b = store.load(&session_b).expect("load B").expect("session B exists");
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
}
