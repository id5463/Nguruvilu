//! Application state and command handling for the desktop shell.
//!
//! The window is a view over the same kernel the CLI drives. Nothing here
//! reimplements kernel behaviour: it holds a [`Kernel`], a [`Runtime`], and a
//! session, and translates UI commands into calls on them.
//!
//! Threading: the window and the webview live on the tao event loop thread,
//! while agent turns run on a tokio runtime. They meet at one seam — a
//! [`EventSink`] — the page in window mode, stdout in headless mode.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Context, Result};
use serde_json::{json, Value};

use nguruvilu::agent::{Agent, AgentObserver, TurnConfig, TurnSettings};
use nguruvilu::hotreload::{Change, ChangePayload, ModelRoute, Runtime};
use nguruvilu::llm::{LlmClient, LlmConfig};
use nguruvilu::plugin::Kernel;
use nguruvilu::session::{JsonlStore, Session, SessionStore};
use nguruvilu::settings::Settings;
use nguruvilu::window::ContextPolicy;
use nguruvilu::skills::{register_skill_tool, SkillRegistry};

use crate::sink::EventSink;

/// Default model when neither `NGU_MODEL` nor the environment supplies one.
pub const DEFAULT_MODEL: &str = "deepseek-v4.1-flash";

/// Default API base.
pub const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

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
        let route = self.route();
        let mut config = LlmConfig::new(route.base_url, route.api_key, model);
        config.proxy = route.proxy.clone().unwrap_or_default();
        config.reasoning_effort = route.reasoning_effort.clone();
        config.max_tokens = route.max_tokens;
        Ok(LlmClient::new(config)?)
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
    /// Whether a turn is running, so the UI can disable input.
    pub busy: bool,
    /// Effective settings, as the settings panel should display them.
    pub settings: Settings,
    /// Prompt tokens the provider last reported, for the context readout.
    pub last_prompt_tokens: Option<usize>,
    /// The last failure, kept here rather than only in the page: a message
    /// drawn into the DOM vanishes the moment the transcript is re-rendered,
    /// which is exactly when a user switches sessions to look for it.
    pub last_error: Option<String>,
}

impl AppState {
    /// Build the shell's state from the environment.
    pub fn bootstrap() -> Result<Self> {
        // Settings, not raw environment variables: the shell has to be usable by
        // someone who has never set one, and changeable without a restart.
        let settings = Settings::resolve();
        let base_url = settings.base_url.clone();
        let api_key = settings.api_key.clone();
        let model = settings.model_or_default();

        // Skills are scanned once at boot; the catalog is part of the prompt.
        let mut skills = SkillRegistry::with_roots(SkillRegistry::default_roots());
        let scan = skills.scan();

        let mut kernel = Kernel::new();
        // The agent can build and install packs itself.
        nguruvilu::tools::pack::register(kernel.tools_mut())?;
        if !skills.is_empty() {
            register_skill_tool(kernel.tools_mut(), Arc::new(skills.clone()))?;
        }

        let tools = Arc::new(kernel.tools().clone());
        let runtime = Runtime::new(tools).with_route(ModelRoute {
            provider: "openai".into(),
            base_url: base_url.clone(),
            api_key: String::new(),
            model: model.clone(),
            temperature: None,
            max_tokens: None,
            reasoning_effort: if settings.reasoning_effort.trim().is_empty() {
                None
            } else {
                Some(settings.reasoning_effort.clone())
            },
            proxy: Some(settings.proxy.clone()),
        });

        let policy: Arc<dyn ContextPolicy> = Arc::new(settings.context_policy(None));

        let runtime = Arc::new(Mutex::new(runtime));
        let skills = Arc::new(Mutex::new(skills));
        let policy = Arc::new(Mutex::new(policy));

        // Published before any pack loads, so a plugin that wants to spawn a
        // subagent finds the service already there.
        nguruvilu::model::install(
            &mut kernel,
            Arc::new(DesktopModelAccess {
                runtime: Arc::clone(&runtime),
                skills: Arc::clone(&skills),
                base_prompt: BASE_PROMPT.to_string(),
                policy: Arc::clone(&policy),
            }),
        )?;

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

        Ok(Self {
            kernel,
            config,
            session,
            store,
            route: ModelRoute {
                provider: "openai".into(),
                base_url,
                api_key,
                model,
                temperature: None,
                max_tokens: None,
                reasoning_effort: if settings.reasoning_effort.trim().is_empty() {
                    None
                } else {
                    Some(settings.reasoning_effort.clone())
                },
                proxy: Some(settings.proxy.clone()),
            },
            busy: false,
            settings,
            last_error: None,
            last_prompt_tokens: None,
        })
    }

    /// A description of the current state, for the details panel and status bar.
    pub fn describe(&self) -> Value {
        let runtime = self.config.runtime.lock().expect("runtime lock");
        let snapshot = runtime.snapshot();
        let skills = self.config.skills.lock().expect("skills lock");
        json!({
            "session": self.session.id,
            "title": self.session.title(),
            "messages": self.session.messages.len(),
            "model": snapshot.model_route.model,
            "reasoning_effort": self.settings.reasoning_effort,
            "proxy": self.settings.proxy,
            "max_output_tokens": self.settings.max_output_tokens.map(nguruvilu::size::format_size),
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
            "services": self.kernel.service_list().iter().map(|(name, realm, owner)| json!({
                "name": name,
                "realm": realm.to_string(),
                "owner": owner,
            })).collect::<Vec<_>>(),
            "busy": self.busy,
            "configured": self.settings.is_configured(),
            "missing": self.settings.missing(),
            "api_key_masked": self.settings.masked_key(),
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
        self.session = session;
        Ok(())
    }

    /// Switch to a stored session.
    pub fn open_session(&mut self, id: &str) -> Result<()> {
        let session = self
            .store
            .load(id)?
            .ok_or_else(|| anyhow!("session not found: {id}"))?;
        self.session = session;
        Ok(())
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

        // The route is what a turn actually reads, so it has to move too.
        self.route.base_url = self.settings.base_url.clone();
        self.route.api_key = self.settings.api_key.clone();
        self.route.model = self.settings.model_or_default();

        let mut runtime = self.config.runtime.lock().expect("runtime lock");
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

        if let Some(soul) = content.soul {
            if let Ok(mut runtime) = self.config.runtime.lock() {
                let _ = runtime.apply(nguruvilu::hotreload::Change::session(
                    "pack",
                    nguruvilu::hotreload::ChangePayload::Persona(soul),
                ));
            }
        }

        if let Some(rules) = content.injections {
            if let Ok(mut injection) = self.config.injection.lock() {
                *injection = std::sync::Arc::new(rules);
            }
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
    /// reads it from — so both have to move together.
    pub fn restore_kernel(&mut self, kernel: Kernel) -> Result<()> {
        self.kernel = kernel;
        let tools = Arc::new(self.kernel.tools().clone());
        let mut runtime = self.config.runtime.lock().expect("runtime lock");
        runtime.set_tools(tools);
        Ok(())
    }

    /// A client for the current route.
    pub fn client(&self) -> Result<LlmClient> {
        let mut config = LlmConfig::new(
            self.route.base_url.clone(),
            self.route.api_key.clone(),
            self.route.model.clone(),
        );
        // Reasoning effort is the main cost lever on a reasoning model, so it
        // travels with every request rather than being chosen per call.
        let effort = self.settings.reasoning_effort.trim();
        if !effort.is_empty() && effort != "default" {
            config.reasoning_effort = Some(effort.to_string());
        }
        // Empty means direct, and means the process's own proxy variables stay
        // out of the way.
        config.proxy = self.settings.proxy.clone();
        LlmClient::new(config)
            .map(|client| {
                client.with_shaper(nguruvilu::request::from_extra_fields(
                    self.settings.extra_body.clone(),
                ))
            })
            .context("building the model client")
    }
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
}

impl UiObserver {
    /// Build an observer that pushes events to the window.
    pub fn new(sink: Arc<dyn EventSink>) -> Self {
        Self { sink }
    }

    fn emit(&self, event: Value) {
        // The page is the only consumer; a closed window just means the send
        // fails, which is not an error worth surfacing.
        self.sink.emit(event);
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

/// Run one turn, streaming progress to the page.
///
/// The turn takes its settings once, at the start, so a change landing while it
/// runs applies to the next turn rather than to a request already in flight.
pub async fn run_turn(
    state: Arc<Mutex<AppState>>,
    text: String,
    sink: Arc<dyn EventSink>,
) -> Result<()> {
    let (client, config, messages) = {
        let mut guard = state.lock().expect("state lock");
        guard.busy = true;
        // A new attempt supersedes the previous failure; leaving a stale error
        // on screen while a retry runs would be a lie.
        guard.last_error = None;
        (guard.client()?, Arc::clone(&guard.config), guard.session.messages.clone())
    };

    let observer = Arc::new(UiObserver::new(Arc::clone(&sink)));
    let mut agent = Agent::new(client, config, messages)
        .with_max_steps(50)
        .with_observer(Arc::clone(&observer) as Arc<dyn AgentObserver>);

    let result = agent.run(&text).await;

    let mut guard = state.lock().expect("state lock");
    guard.busy = false;

    match result {
        Ok(outcome) => {
            // A one-line trace on stderr: the window shows the answer, but a
            // terminal launch should also be able to see that a turn happened.
            let preview: String = outcome.text.chars().take(100).collect();
            eprintln!(
                "[turn] {} steps, {} tools, {} in / {} out ({} cached), model {} ms, tools {} ms: {}",
                outcome.steps,
                outcome.tool_calls,
                outcome.usage.input,
                outcome.usage.output,
                outcome.usage.cached,
                outcome.timing.model_ms,
                outcome.timing.tools_ms,
                preview.replace('\n', " ")
            );
            guard.session.messages.extend(outcome.new_messages.clone());
            if let Err(error) = guard.store.append(&guard.session.id, &outcome.new_messages) {
                eprintln!("[store] append failed: {error:#}");
            }

            for record in &outcome.injections {
                if let Err(error) = guard.store.append_injection(&guard.session.id, record) {
                    eprintln!("[store] injection append failed: {error:#}");
                }
            }

            // Record compactions after the messages they replace, then adopt the
            // agent's reduced history so the live session matches the file.
            for compaction in &outcome.compactions {
                if let Err(error) = guard.store.append_compaction(&guard.session.id, compaction) {
                    eprintln!("[store] compaction append failed: {error:#}");
                }
                eprintln!(
                    "[compacted {} messages into {} chars]",
                    compaction.replaced,
                    compaction.summary.chars().count()
                );
            }
            if !outcome.compactions.is_empty() {
                guard.session.messages = agent.messages().to_vec();
            }

            let payload = json!({
                "ev": "turn_end",
                "steps": outcome.steps,
                "tool_calls": outcome.tool_calls,
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
                "session": guard.session.id,
                "messages": guard.session.messages.len(),
            });
            sink.emit(payload);
        }
        Err(error) => {
            let message = format!("{error:#}");
            guard.last_error = Some(message.clone());
            sink.emit(json!({
                "ev": "error",
                "message": message,
            }));
        }
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
