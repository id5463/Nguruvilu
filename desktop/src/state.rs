//! Application state and command handling for the desktop shell.
//!
//! The window is a view over the same kernel the CLI drives. Nothing here
//! reimplements kernel behaviour: it holds a [`Kernel`], a [`Runtime`], and a
//! session, and translates UI commands into calls on them.
//!
//! Threading: the window and the webview live on the tao event loop thread,
//! while agent turns run on a tokio runtime. They meet at one seam — a
//! [`EventLoopProxy`] carrying JSON to evaluate in the page.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Context, Result};
use serde_json::{json, Value};
use tao::event_loop::EventLoopProxy;

use nguruvilu::agent::{Agent, AgentObserver, TurnConfig, TurnSettings};
use nguruvilu::hotreload::{Change, ChangePayload, ModelRoute, Runtime};
use nguruvilu::llm::{LlmClient, LlmConfig};
use nguruvilu::plugin::Kernel;
use nguruvilu::session::{JsonlStore, Session, SessionStore};
use nguruvilu::settings::Settings;
use nguruvilu::skills::{register_skill_tool, SkillRegistry};

use crate::UserEvent;

/// Default model when neither `NGU_MODEL` nor the environment supplies one.
pub const DEFAULT_MODEL: &str = "deepseek-v4.1-flash";

/// Default API base.
pub const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

/// System prompt for the desktop shell.
pub const BASE_PROMPT: &str = "You are a coding agent. Use the tools to inspect and change \
files and to run commands. Prefer `read` over shelling out to view files, and `edit` for \
surgical changes. Keep answers short and concrete.";

/// State the agent loop reads at the start of each turn.
///
/// This is the desktop shell's [`TurnConfig`]: it composes the runtime's persona
/// and route with the skill catalog, exactly as the CLI does.
pub struct SharedConfig {
    /// The hot-reload runtime owning persona, route, and the tool table.
    pub runtime: Mutex<Runtime>,
    /// Base system prompt.
    pub base_prompt: String,
    /// Skill registry, scanned at boot.
    pub skills: Mutex<SkillRegistry>,
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
        });

        let config = Arc::new(SharedConfig {
            runtime: Mutex::new(runtime),
            base_prompt: BASE_PROMPT.to_string(),
            skills: Mutex::new(skills),
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
            },
            busy: false,
            settings,
            last_error: None,
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
    ) -> Result<()> {
        let settings = Settings {
            base_url: base_url.trim().to_string(),
            api_key: api_key.trim().to_string(),
            model: model.trim().to_string(),
            reasoning_effort: reasoning_effort.trim().to_string(),
        };
        settings.save()?;
        self.settings = settings;

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
                "summary": pack.manifest.summary,
                "path": pack.path.display().to_string(),
                "assembly": pack.assembly.as_ref().map(|p| p.display().to_string()),
            })).collect::<Vec<_>>(),
        }))
    }

    /// Install an archive and report what landed.
    pub fn install_pack(&mut self, archive: &Path) -> Result<Value> {
        let dir = nguruvilu::pack::default_packs_dir();
        let placed = nguruvilu::pack::install(archive, &dir)?;
        Ok(json!({
            "name": placed.manifest.name,
            "version": placed.manifest.version_id,
            "path": placed.path.display().to_string(),
            "assembly": placed.assembly.as_ref().map(|p| p.display().to_string()),
        }))
    }

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
        LlmClient::new(config).context("building the model client")
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
pub struct UiObserver {
    proxy: EventLoopProxy<UserEvent>,
}

impl UiObserver {
    /// Build an observer that pushes events to the window.
    pub fn new(proxy: EventLoopProxy<UserEvent>) -> Self {
        Self { proxy }
    }

    fn emit(&self, event: Value) {
        // The page is the only consumer; a closed window just means the send
        // fails, which is not an error worth surfacing.
        let _ = self.proxy.send_event(UserEvent::ToUi(event));
    }
}

impl AgentObserver for UiObserver {
    fn on_step(&self, step: usize) {
        self.emit(json!({ "ev": "step", "step": step }));
    }

    fn on_text(&self, delta: &str) {
        self.emit(json!({ "ev": "text", "delta": delta }));
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
}

/// Run one turn, streaming progress to the page.
///
/// The turn takes its settings once, at the start, so a change landing while it
/// runs applies to the next turn rather than to a request already in flight.
pub async fn run_turn(
    state: Arc<Mutex<AppState>>,
    text: String,
    proxy: EventLoopProxy<UserEvent>,
) -> Result<()> {
    let (client, config, messages) = {
        let mut guard = state.lock().expect("state lock");
        guard.busy = true;
        // A new attempt supersedes the previous failure; leaving a stale error
        // on screen while a retry runs would be a lie.
        guard.last_error = None;
        (guard.client()?, Arc::clone(&guard.config), guard.session.messages.clone())
    };

    let observer = Arc::new(UiObserver::new(proxy.clone()));
    let mut agent = Agent::new(client, config, messages)
        .with_max_steps(50)
        .with_observer(observer);

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
            let _ = proxy.send_event(UserEvent::ToUi(payload));
        }
        Err(error) => {
            let message = format!("{error:#}");
            guard.last_error = Some(message.clone());
            let _ = proxy.send_event(UserEvent::ToUi(json!({
                "ev": "error",
                "message": message,
            })));
        }
    }
    Ok(())
}
