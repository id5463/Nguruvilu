//! Nguruvilu desktop shell.
//!
//! A native window (tao) rendering the interface through the system webview
//! (wry). The interface is compiled into the binary with `include_str!`, so the
//! shipped artifact is **one executable** — nothing is unpacked to disk to show
//! a window, and there are no sidecar assets to lose.
//!
//! Layout:
//!
//! * the tao event loop thread owns the window and the webview;
//! * a tokio runtime owns agent turns;
//! * they meet at one seam: [`UserEvent::ToUi`] carries JSON to evaluate in the
//!   page, and the webview's IPC handler carries commands the other way.

mod sink;
mod state;

use std::io::{BufRead, Read, Write};
use std::path::PathBuf;
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};

use serde_json::{json, Value};
use tao::dpi::LogicalSize;
use tao::rwh_06::{HasDisplayHandle, HasWindowHandle};
use tao::event::{Event, WindowEvent};
use tao::event_loop::{ControlFlow, EventLoopBuilder};
use tao::window::WindowBuilder;
use wry::WebViewBuilder;

use nguruvilu::assembly::Assembly;
use nguruvilu::ledger::default_ledger_path;
use nguruvilu::loader::Loader;

use sink::{EventSink, StdoutSink, WindowSink};
use state::AppState;

/// The interface, embedded at compile time.
const UI_HTML: &str = include_str!("../ui/index.html");

/// Events the loop delivers to the page.
pub enum UserEvent {
    /// Evaluate this value as a call into the page's `window.ngu.receive`.
    ToUi(Value),
}

/// Command-line options the shell understands.
struct Options {
    /// Send this prompt as soon as the shell is ready.
    prompt: Option<String>,
    /// Id of the interface to use. Defaults to the first installed pack that
    /// declares one; without either, the built-in interface.
    ui: Option<String>,
    /// Run without a window, reading commands from stdin.
    headless: bool,
}

fn options() -> Options {
    let args: Vec<String> = std::env::args().collect();
    let value_of = |flag: &str| -> Option<String> {
        args.iter()
            .position(|arg| arg == flag)
            .and_then(|index| args.get(index + 1).cloned())
    };
    Options {
        prompt: value_of("--prompt").or_else(|| value_of("-p")),
        ui: value_of("--ui"),
        headless: args.iter().any(|arg| arg == "--headless"),
    }
}

fn main() -> anyhow::Result<()> {
    let options = options();
    if options.headless {
        return headless(options.prompt);
    }
    windowed(options.prompt.clone(), options.ui.clone())
}

/// Run without a window: JSON commands in on stdin, JSON events out on stdout.
///
/// This exists so the shell can be driven — by a script, by another agent, or by
/// whoever is trying to work out why something behaves the way it does. It runs
/// the same [`dispatch`] as the window, so it exercises the real path rather
/// than a parallel one.
///
/// Commands are one JSON object per line, the same objects the page sends:
///
/// ```text
/// {"cmd":"prompt","text":"say hi"}
/// {"cmd":"list_packs"}
/// {"cmd":"status"}
/// ```
///
/// Events are one JSON object per line, the same objects the page receives.
fn headless(startup_prompt: Option<String>) -> anyhow::Result<()> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;

    let state = Arc::new(Mutex::new(AppState::bootstrap()?));
    let sink: Arc<dyn EventSink> = Arc::new(StdoutSink);

    let result = runtime.block_on(async {
        // The windowed path starts from the page's "ready"; headless has no page,
        // so the shell announces itself the same way.
        dispatch(
            Arc::clone(&state),
            "{\"cmd\":\"ready\"}",
            Arc::clone(&sink),
            startup_prompt,
        )
        .await?;

        // Reading stdin is blocking, so it runs off the async scheduler.
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();
        std::thread::spawn(move || {
            let stdin = std::io::stdin();
            for line in stdin.lock().lines() {
                match line {
                    Ok(line) => {
                        if tx.send(line).is_err() {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }
        });

        while let Some(line) = rx.recv().await {
            let line = line.trim().to_string();
            if line.is_empty() {
                continue;
            }
            if line == "exit" || line == "quit" {
                break;
            }
            if let Err(error) =
                dispatch(Arc::clone(&state), &line, Arc::clone(&sink), None).await
            {
                sink.emit(json!({ "ev": "error", "message": format!("{error:#}") }));
            }
        }
        // Stop the MCP servers before the runtime tears down: a server still
        // registered with a runtime that is shutting down is what keeps the
        // process alive after everything it has printed. Bounded — a server
        // that refuses to die must not postpone the exit it is blocking.
        let clients = state.lock().expect("state lock").mcp.clone();
        for client in &clients {
            client.shutdown_within(std::time::Duration::from_secs(3)).await;
        }
        Ok::<(), anyhow::Error>(())
    });

    // Deliberately not dropping the runtime — the same reason the CLI's main
    // exits instead: teardown waits on state a spawned server left behind, and
    // the process would then never leave. Everything it owned was stopped
    // above.
    match result {
        Ok(()) => std::process::exit(0),
        Err(error) => {
            eprintln!("ngu: {error:#}");
            std::process::exit(1);
        }
    }
}

/// Run with a window.
fn windowed(startup_prompt: Option<String>, ui_id: Option<String>) -> anyhow::Result<()> {
    // WebView2 takes its profile location from the environment. Setting it here
    // keeps the browser profile out of whatever directory the app was launched
    // from, which may be read-only.
    std::env::set_var("WEBVIEW2_USER_DATA_FOLDER", state::data_directory());

    // EventLoopBuilder is how tao attaches a custom user event type.
    let event_loop = EventLoopBuilder::<UserEvent>::with_user_event().build();
    let proxy = event_loop.create_proxy();

    let window = WindowBuilder::new()
        .with_title("Nguruvilu")
        .with_inner_size(LogicalSize::new(1280.0, 820.0))
        .with_min_inner_size(LogicalSize::new(760.0, 500.0))
        .build(&event_loop)?;

    // Captured once for the file dialogs: the IPC callback must be `'static`
    // so it cannot borrow the window, and a dialog without an owner opens
    // *behind* the main window — where the click looks dead.
    let (parent_window, parent_display) = {
        let window_handle = window
            .window_handle()
            .expect("the window has a handle before any dialog can open");
        let display_handle = window.display_handle().expect("display handle");
        (window_handle.as_raw(), display_handle.as_raw())
    };

    let state = Arc::new(Mutex::new(AppState::bootstrap()?));
    let sink: Arc<dyn EventSink> = Arc::new(WindowSink::new(proxy));

    // Agent turns are async; the window is not. The runtime lives for the whole
    // process, moved into the event loop closure so it is not dropped early.
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    let handle = runtime.handle().clone();

    // Before the interface is chosen: which interface runs is a pack's
    // decision, so the packs this build ships have to be there first. Without
    // this the very first launch would fall back to the built-in one and the
    // second would not — the same window, two answers, depending on nothing
    // the user did.
    match runtime.block_on(nguruvilu::preinstall::seed(
        &nguruvilu::pack::default_packs_dir(),
    )) {
        Ok(placed) if !placed.is_empty() => eprintln!("[preinstall] {}", placed.join(", ")),
        Ok(_) => {}
        Err(error) => eprintln!("[preinstall] {error:#}"),
    }

    let webview = {
        let state = Arc::clone(&state);
        let sink = Arc::clone(&sink);
        let handle = handle.clone();
        let startup_prompt = startup_prompt.clone();
        // Decided before the window exists: the page is loaded once, so an
        // interface cannot be swapped in afterwards without a reload.
        if let Some(directory) = resolve_active_ui(ui_id.as_deref()) {
            set_active_ui(directory);
        }

        let ui_port = serve_interface_on_loopback()?;
        WebViewBuilder::new()
            // The page is served over loopback HTTP rather than a custom
            // scheme: WebView2 refuses an unknown scheme as the document's own
            // origin, so `ngu://localhost/` navigates to nothing and the window
            // comes up blank. A real origin also means the script runs (the
            // opaque origin of `with_html` gets inline scripts refused) and the
            // same `serve_ui` answers every request — a file server, because a
            // pack brings a directory of assets whose stylesheets, scripts, and
            // images resolve as relative paths. Which directory is served is
            // decided when a pack declares one; until then it is the built-in
            // interface, so the window is never blank.
            //
            // The custom scheme stays registered for anything that still asks
            // for `ngu://`; the window itself does not.
            .with_custom_protocol("ngu".into(), move |_id, request| {
                let path = request.uri().path().trim_start_matches('/').to_string();
                let (body, mime) = serve_ui(&path);
                wry::http::Response::builder()
                    .header("Content-Type", mime)
                    // An interface is edited in place while being written, so a
                    // cached copy is a stale one.
                    .header("Cache-Control", "no-store")
                    .body(std::borrow::Cow::Owned(body))
                    .expect("building the response")
            })
            .with_url(&format!("http://127.0.0.1:{ui_port}/"))
            .with_ipc_handler(move |request| {
                let body = request.body().to_string();
                let command: Value = serde_json::from_str(&body).unwrap_or(Value::Null);
                // Owned, not borrowed: the value crosses into the runtime task below.
                let name: String = command
                    .get("cmd")
                    .and_then(|c| c.as_str())
                    .unwrap_or("")
                    .to_string();

                // A native file dialog has to run on this thread, so the picker
                // is opened here and only the follow-up work is handed to the
                // runtime. `None` means the user cancelled. The dialog is owned
                // by the main window (see `DialogParent`) so it cannot hide
                // behind it.
                let picked: Option<PathBuf> = {
                    let parent = DialogParent::new(parent_window, parent_display);
                    match name.as_str() {
                        "pick_install" | "pick_verify" => rfd::FileDialog::new()
                            .set_title("Choose a .dshpack archive")
                            .add_filter("pack archive", &["dshpack"])
                            .set_parent(&parent)
                            .pick_file(),
                        "pick_pack" => rfd::FileDialog::new()
                            .set_title("Choose a pack directory (it must contain dsh.index.json)")
                            .set_parent(&parent)
                            .pick_folder(),
                        _ => None,
                    }
                };

                let state = Arc::clone(&state);
                let sink = Arc::clone(&sink);
                let startup_prompt = startup_prompt.clone();
                // The handler is synchronous and must return quickly, so the
                // work goes onto the runtime and answers arrive as events.
                handle.spawn(async move {
                    let outcome = if name.starts_with("pick_") {
                        match picked {
                            Some(path) => {
                                dispatch_path(state, &name, path, Arc::clone(&sink)).await
                            }
                            // Cancelling a dialog is not an error.
                            None => Ok(()),
                        }
                    } else {
                        dispatch(state, &body, Arc::clone(&sink), startup_prompt.clone()).await
                    };
                    if let Err(error) = outcome {
                        sink.emit(json!({
                            "ev": "error",
                            "message": format!("{error:#}"),
                        }));
                    }
                });
            })
            .build(&window)?
    };

    event_loop.run(move |event, _, control_flow| {
        // Holding the runtime here keeps it alive for the process lifetime.
        let _ = &runtime;
        *control_flow = ControlFlow::Wait;

        match event {
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
                // Close must finish — this runs on the UI thread, so anything
                // unbounded here turns the window into a ghost. It did: a
                // server that ignored the kill hung the quit forever (CPU
                // idle, children alive, window never responding again).
                //
                // Three rules: take the client list without trusting the state
                // lock blindly (a panic elsewhere poisons it; a task may hold
                // it for a moment); kill the servers first so they are already
                // exiting; then give each a short window to be reaped.
                let clients = {
                    let mut found = None;
                    for _ in 0..20 {
                        match state.try_lock() {
                            Ok(guard) => {
                                found = Some(guard.mcp.clone());
                                break;
                            }
                            Err(std::sync::TryLockError::Poisoned(poisoned)) => {
                                found = Some(poisoned.into_inner().mcp.clone());
                                break;
                            }
                            Err(std::sync::TryLockError::WouldBlock) => {
                                std::thread::sleep(std::time::Duration::from_millis(50));
                            }
                        }
                    }
                    // No list in a second: leave anyway. An orphaned server is
                    // a bad exit; a window that never closes is a broken app.
                    found.unwrap_or_default()
                };
                for client in &clients {
                    client.force_kill();
                }
                runtime.block_on(async {
                    for client in &clients {
                        client
                            .shutdown_within(std::time::Duration::from_millis(500))
                            .await;
                    }
                });
                *control_flow = ControlFlow::Exit;
            }
            Event::UserEvent(UserEvent::ToUi(value)) => {
                let script = format!("window.ngu && window.ngu.receive({value})");
                if let Err(error) = webview.evaluate_script(&script) {
                    eprintln!("[ui] evaluate failed: {error}");
                }
            }
            _ => {}
        }
    })
}

/// Handle one command from the page.
async fn dispatch(
    state: Arc<Mutex<AppState>>,
    body: &str,
    sink: Arc<dyn EventSink>,
    startup_prompt: Option<String>,
) -> anyhow::Result<()> {
    let command: Value = serde_json::from_str(body).unwrap_or(Value::Null);
    // Every command is logged while the shell is young: a silent page is
    // indistinguishable from a broken IPC path otherwise.
    eprintln!("[ipc] {body}");
    let name = command.get("cmd").and_then(|c| c.as_str()).unwrap_or("");

    match name {
        // The page announces itself once loaded, which is when initial state
        // can safely be pushed.
        "ready" => {
            // Proof that the page loaded, ran its script, and the IPC path works.
            eprintln!("[ui] page ready; pushing initial state");

            // Load every installed pack before the first paint. A pack the user
            // installed but cannot see is a pack they will install again.
            load_installed_packs(Arc::clone(&state), Arc::clone(&sink)).await;

            let (sessions, transcript, status) = {
                let guard = state.lock().expect("state lock");
                (guard.sessions()?, guard.transcript(), guard.describe())
            };
            emit(&sink, json!({ "ev": "sessions", "list": sessions }));
            emit(&sink, json!({ "ev": "transcript", "entries": transcript }));
            emit(&sink, json!({ "ev": "status", "status": status }));
            // Plugin panels and the theme they imply, pushed with the rest of
            // the initial state so the page never paints the wrong palette.
            emit_panels(&state, &sink);

            // A prompt given on the command line runs once the page can show it.

            // Fetch the catalog up front when the endpoint is configured: a picker
            // that only fills after someone finds a button is a picker nobody finds.
            // Failure is reported by the fetch itself and is not fatal.
            let configured = state.lock().expect("state lock").settings.is_configured();
            if configured {
                // Deliberately not awaited: fetching the catalog is a nicety, and
                // making a slow endpoint delay the first turn would be a poor trade.
                let models_state = Arc::clone(&state);
                let models_sink = Arc::clone(&sink);
                tokio::spawn(async move {
                    let _ = fetch_models(models_state, models_sink).await;
                });
            }
            if let Some(text) = startup_prompt {
                emit(&sink, json!({ "ev": "turn_start", "text": text }));
                state::run_turn(Arc::clone(&state), text, Arc::clone(&sink)).await?;
                drain_packs(state, Arc::clone(&sink)).await;
            }
        }

        "prompt" => {
            let text = command
                .get("text")
                .and_then(|t| t.as_str())
                .unwrap_or("")
                .trim()
                .to_string();
            if text.is_empty() {
                return Ok(());
            }
            emit(&sink, json!({ "ev": "turn_start", "text": text }));
            state::run_turn(Arc::clone(&state), text, Arc::clone(&sink)).await?;
            drain_packs(state, Arc::clone(&sink)).await;
        }

        "new_session" | "open_session" | "delete_session" => {
            let (sessions, transcript, status) = {
                let mut guard = state.lock().expect("state lock");
                match name {
                    "new_session" => guard.new_session()?,
                    "open_session" => {
                        let id = command.get("id").and_then(|i| i.as_str()).unwrap_or("");
                        guard.open_session(id)?
                    }
                    _ => {
                        let id = command.get("id").and_then(|i| i.as_str()).unwrap_or("");
                        guard.delete_session(id)?
                    }
                }
                (guard.sessions()?, guard.transcript(), guard.describe())
            };
            emit(&sink, json!({ "ev": "sessions", "list": sessions }));
            emit(&sink, json!({ "ev": "transcript", "entries": transcript }));
            emit(&sink, json!({ "ev": "status", "status": status }));
        }

        "set_model" => {
            let model = command.get("model").and_then(|m| m.as_str()).unwrap_or("");
            let base_url = command.get("base_url").and_then(|u| u.as_str());
            let status = {
                let mut guard = state.lock().expect("state lock");
                guard.set_model(model, base_url)?;
                guard.describe()
            };
            emit(&sink, json!({ "ev": "status", "status": status }));
        }

        "save_settings" => {
            let base_url = command.get("base_url").and_then(|v| v.as_str()).unwrap_or("");
            let api_key_field = command.get("api_key").and_then(|v| v.as_str()).unwrap_or("");
            let keep_key = command.get("keep_api_key").and_then(|v| v.as_bool()).unwrap_or(false);
            let model = command.get("model").and_then(|v| v.as_str()).unwrap_or("");
            let effort = command.get("reasoning_effort").and_then(|v| v.as_str()).unwrap_or("");
            let proxy = command.get("proxy").and_then(|v| v.as_str()).unwrap_or("");
            let context_window = command
                .get("context_window")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize);
            let compact_percent = command
                .get("compact_percent")
                .and_then(|v| v.as_u64())
                .map(|v| v as u32)
                .unwrap_or(75);
            let keep_recent = command
                .get("compact_keep_recent")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(8);
            // Accepts a number or a sized string like 8K.
            let max_output = command
                .get("max_output_tokens")
                .and_then(|v| v.as_str())
                .and_then(|v| nguruvilu::size::parse_size(v).ok());
            // Web search: provider decides on/off, blank key box keeps the
            // stored key — the same two rules the panel applies to the model key.
            let search_provider = command
                .get("search_provider")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let search_api_key = command
                .get("search_api_key")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let search_endpoint = command
                .get("search_endpoint")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            // The judge's three: same rules as search (blank key keeps the
            // stored one; blank endpoint with blank key turns it off).
            let judge_endpoint = command
                .get("judge_endpoint")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let judge_api_key = command
                .get("judge_api_key")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let judge_model = command
                .get("judge_model")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            // Present flags: the page says whether it *shows* each section, so
            // an unloaded pack's absence never reads as "the user erased it".
            // A payload without flags (scripted call) falls back to "present
            // when any of its fields was sent".
            let search_present = command
                .get("search_present")
                .and_then(|v| v.as_bool())
                .unwrap_or_else(|| {
                    command.get("search_provider").is_some()
                        || command.get("search_api_key").is_some()
                        || command.get("search_endpoint").is_some()
                });
            let judge_present = command
                .get("judge_present")
                .and_then(|v| v.as_bool())
                .unwrap_or_else(|| {
                    command.get("judge_endpoint").is_some()
                        || command.get("judge_api_key").is_some()
                        || command.get("judge_model").is_some()
                });

            let outcome = {
                let mut guard = state.lock().expect("state lock");
                // A blank key box means "keep the stored key": the panel never
                // receives the key back, so it cannot send it again, and treating
                // blank as "erase" would silently break the next request.
                let api_key = if keep_key || api_key_field.trim().is_empty() {
                    guard.settings.api_key.clone()
                } else {
                    api_key_field.to_string()
                };
                guard.save_settings(
                    base_url,
                    &api_key,
                    model,
                    effort,
                    proxy,
                    context_window,
                    compact_percent,
                    keep_recent,
                    max_output,
                    search_provider,
                    search_api_key,
                    search_endpoint,
                    search_present,
                    judge_endpoint,
                    judge_api_key,
                    judge_model,
                    judge_present,
                )
            };

            match outcome {
                Ok(()) => {
                    eprintln!("[settings] saved to {}", nguruvilu::settings::Settings::path().display());
                    let status = state.lock().expect("state lock").describe();
                    emit(&sink, json!({ "ev": "status", "status": status }));
                    emit(&sink, json!({ "ev": "settings_saved" }));
                }
                Err(error) => {
                    emit(
                        &sink,
                        json!({ "ev": "error", "message": format!("saving settings: {error:#}") }),
                    );
                }
            }
        }

        "fetch_models" => {
            fetch_models(state, Arc::clone(&sink)).await?;
        }

        "list_packs" => {
            refresh_packs(&state, &sink);
        }

        "apply_pack" => {
            let path = command.get("path").and_then(|v| v.as_str()).unwrap_or("");
            if path.is_empty() {
                emit(&sink, json!({ "ev": "error", "message": "apply_pack needs a path" }));
            } else {
                apply_pack(state, PathBuf::from(path), Arc::clone(&sink)).await?;
            }
        }

        "unload_plugin" => {
            let name = command
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();
            let removed = {
                let mut guard = state.lock().expect("state lock");
                let ids: Vec<u64> = guard
                    .kernel
                    .fibers()
                    .iter()
                    .filter(|fiber| fiber.plugin == name)
                    .map(|fiber| fiber.id)
                    .collect();
                let count = ids.len();
                for id in ids {
                    if let Err(error) = guard.kernel.unload(id) {
                        eprintln!("[unload] {name}: {error:#}");
                    }
                }
                count
            };
            sink.emit(json!({ "ev": "notice", "text": format!("Unloaded {removed} instance(s) of {name}") }));
            // The panels and theme just changed, so the page has to be told.
            emit_panels(&state, &sink);
        }

        "uninstall_pack" | "load_pack" => {
            let enable = name == "load_pack";
            let name = command
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();
            let version = command
                .get("version")
                .and_then(|v| v.as_str())
                .map(str::to_string);

            let mut matched: Vec<String> = Vec::new();
            match nguruvilu::pack::installed(&nguruvilu::pack::default_packs_dir()) {
                Ok(packs) => {
                    for pack in packs
                        .iter()
                        .filter(|pack| pack.manifest.name == name)
                        .filter(|pack| {
                            version
                                .as_deref()
                                .map_or(true, |v| pack.manifest.version_id == v)
                        })
                    {
                        match nguruvilu::pack::set_enabled(&pack.path, enable) {
                            Ok(_) => matched.push(format!(
                                "{}@{}",
                                pack.manifest.name, pack.manifest.version_id
                            )),
                            Err(error) => emit(
                                &sink,
                                json!({ "ev": "error", "message": format!("{}: {error:#}", pack.manifest.name) }),
                            ),
                        }
                    }
                }
                Err(error) => emit(
                    &sink,
                    json!({ "ev": "error", "message": format!("listing packs: {error:#}") }),
                ),
            }

            if matched.is_empty() {
                emit(
                    &sink,
                    json!({ "ev": "error", "message": format!("no installed pack matches '{name}'") }),
                );
            } else {
                // The state is written; now the conversation itself has to
                // follow, or a loaded pack would keep working until a restart.
                state.lock().expect("state lock").pending.push(if enable {
                    nguruvilu::tools::pack::Pending::Load(name.clone())
                } else {
                    nguruvilu::tools::pack::Pending::Unload(name.clone())
                });
                drain_packs(Arc::clone(&state), Arc::clone(&sink)).await;
                emit(
                    &sink,
                    json!({
                        "ev": "notice",
                        "text": if enable {
                            format!("Loaded {}", matched.join(", "))
                        } else {
                            format!("Unloaded {} (files kept)", matched.join(", "))
                        },
                    }),
                );
            }
            refresh_packs(&state, &sink);
        }

        "delete_pack" => {
            let name = command
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();
            let version = command
                .get("version")
                .and_then(|v| v.as_str())
                .map(str::to_string);
            let target = match &version {
                Some(version) => format!("{name}@{version}"),
                None => name.clone(),
            };
            // Unload first, then delete: removing files under a live kernel
            // would leave tools whose backing directory no longer exists.
            state.lock().expect("state lock").pending.push(
                nguruvilu::tools::pack::Pending::Unload(name.clone()),
            );
            drain_packs(Arc::clone(&state), Arc::clone(&sink)).await;
            match remove_pack(&target) {
                Ok(()) => {
                    emit(&sink, json!({ "ev": "notice", "text": format!("Deleted {target}") }));
                }
                Err(error) => emit(
                    &sink,
                    json!({ "ev": "error", "message": format!("deleting {target}: {error:#}") }),
                ),
            }
            refresh_packs(&state, &sink);
        }

        "ui_panels" => {
            emit_panels(&state, &sink);
        }

        "status" => {
            let status = state.lock().expect("state lock").describe();
            emit(&sink, json!({ "ev": "status", "status": status }));
        }






        "" => {
            emit(&sink, json!({ "ev": "error", "message": "empty command" }));
        }

        other => {
            emit(
                &sink,
                json!({ "ev": "error", "message": format!("unknown command '{other}'") }),
            );
        }
    }

    Ok(())
}

/// Handle a command whose path came from a native dialog.
async fn dispatch_path(
    state: Arc<Mutex<AppState>>,
    name: &str,
    path: PathBuf,
    sink: Arc<dyn EventSink>,
) -> anyhow::Result<()> {
    match name {
        "pick_install" => {
            // Verify before unpacking: a corrupt or unidentifiable archive
            // should be reported, not scattered into the packs directory.
            let verified = {
                let guard = state.lock().expect("state lock");
                guard.verify_pack(&path)
            };
            match verified {
                Ok(report) => emit(&sink, json!({ "ev": "pack_verified", "report": report })),
                Err(error) => {
                    emit(
                        &sink,
                        json!({ "ev": "error", "message": format!("{}: {error:#}", path.display()) }),
                    );
                    return Ok(());
                }
            }

            // No lock held: installing fetches, and the shell must stay
            // readable while it does.
            let installed = state::install_pack(&path).await;
            match installed {
                Ok(info) => {
                    eprintln!("[pack] installed {} {}", info["name"], info["version"]);
                    emit(&sink, json!({ "ev": "pack_installed", "pack": info }));
                }
                Err(error) => emit(
                    &sink,
                    json!({ "ev": "error", "message": format!("installing {}: {error:#}", path.display()) }),
                ),
            }
            refresh_packs(&state, &sink);
        }

        "pick_verify" => {
            let result = { state.lock().expect("state lock").verify_pack(&path) };
            match result {
                Ok(report) => emit(
                    &sink,
                    json!({ "ev": "pack_verified", "report": report, "path": path.display().to_string() }),
                ),
                Err(error) => emit(&sink, json!({ "ev": "error", "message": format!("{error:#}") })),
            }
        }

        "pick_pack" => {
            let result = { state.lock().expect("state lock").pack_dir(&path, None) };
            match result {
                Ok(info) => {
                    eprintln!("[pack] built {}", info["archive"]);
                    emit(&sink, json!({ "ev": "pack_built", "pack": info }));
                }
                Err(error) => emit(
                    &sink,
                    json!({ "ev": "error", "message": format!("packing {}: {error:#}", path.display()) }),
                ),
            }
        }

        other => {
            emit(
                &sink,
                json!({ "ev": "error", "message": format!("unknown picker command '{other}'") }),
            );
        }
    }
    Ok(())
}

/// Push the installed-pack list to the page.
fn refresh_packs(state: &Arc<Mutex<AppState>>, sink: &Arc<dyn EventSink>) {
    match state.lock().expect("state lock").packs() {
        Ok(list) => emit(sink, json!({ "ev": "packs", "list": list })),
        Err(error) => emit(
            sink,
            json!({ "ev": "error", "message": format!("listing packs: {error:#}") }),
        ),
    }
}

/// Apply the pack load and unload the last turn asked for.
///
/// The tool records what it wants while the turn owns the kernel; this runs
/// between turns, which is the next moment the tool table is read anyway. No
/// restart, no pausing — the conversation continues with a new table, and a
/// tool that was unloaded answers "no tool named …" from then on.
async fn drain_packs(state: Arc<Mutex<AppState>>, sink: Arc<dyn EventSink>) {
    let actions = {
        let guard = state.lock().expect("state lock");
        guard.pending.drain()
    };
    if actions.is_empty() {
        return;
    }

    let mut changed = false;
    for action in actions {
        match action {
            nguruvilu::tools::pack::Pending::Unload(name) => {
                // The guard lives inside this block and the kill happens
                // outside it: a guard held across an await would make the whole
                // future non-Send, and the UI thread would wait on it.
                let (found, to_stop) = {
                    let mut guard = state.lock().expect("state lock");
                    let packs =
                        match nguruvilu::pack::installed(&nguruvilu::pack::default_packs_dir()) {
                            Ok(packs) => packs,
                            Err(error) => {
                                emit(
                                    &sink,
                                    json!({ "ev": "error", "message": format!("listing packs: {error:#}") }),
                                );
                                continue;
                            }
                        };
                    let mut found = false;
                    let mut to_stop: Vec<Arc<nguruvilu::mcp::McpClient>> = Vec::new();
                    for pack in packs.iter().filter(|pack| pack.manifest.name == name) {
                        let label =
                            format!("{}-{}", pack.manifest.name, pack.manifest.version_id);
                        let skills = Arc::clone(&guard.config.skills);
                        let result = {
                            let mut host = skills.lock().expect("skills lock");
                            nguruvilu::loader::unload_pack(
                                &mut guard.kernel,
                                &mut *host,
                                &nguruvilu::ledger::default_ledger_path(),
                                &label,
                                &pack.manifest.name,
                            )
                        };
                        match result {
                            Ok(report) => {
                                found = true;
                                changed = true;
                                eprintln!(
                                    "[pack] unloaded {label}: {} plugin instance(s), {} mcp server(s), {} skill root(s)",
                                    report.plugins.len(),
                                    report.mcp.len(),
                                    report.skills.len()
                                );
                                to_stop.extend(
                                    guard
                                        .mcp
                                        .iter()
                                        .filter(|client| {
                                            report.mcp.iter().any(|id| id == client.server())
                                        })
                                        .cloned(),
                                );
                            }
                            Err(error) => emit(
                                &sink,
                                json!({ "ev": "error", "message": format!("unloading {label}: {error:#}") }),
                            ),
                        }
                    }
                    (found, to_stop)
                };

                // The tools are gone, so the servers they belonged to have no
                // reason to keep running. Bounded: an unload must not wedge
                // because one server ignores the kill.
                for client in &to_stop {
                    client.shutdown_within(std::time::Duration::from_secs(5)).await;
                }
                if !to_stop.is_empty() {
                    let mut guard = state.lock().expect("state lock");
                    guard
                        .mcp
                        .retain(|kept| !to_stop.iter().any(|stop| Arc::ptr_eq(kept, stop)));
                }
                if !found {
                    emit(
                        &sink,
                        json!({ "ev": "notice", "text": format!("{name} is not installed; nothing was loaded to unload") }),
                    );
                }
            }
            nguruvilu::tools::pack::Pending::Load(name) => {
                let targets: Vec<(String, PathBuf, PathBuf)> = {
                    match nguruvilu::pack::installed(&nguruvilu::pack::default_packs_dir()) {
                        Ok(packs) => packs
                            .iter()
                            .filter(|pack| pack.manifest.name == name && pack.enabled)
                            .filter_map(|pack| {
                                pack.assembly.clone().map(|assembly| {
                                    (
                                        format!("{}-{}", pack.manifest.name, pack.manifest.version_id),
                                        assembly,
                                        pack.path.clone(),
                                    )
                                })
                            })
                            .collect(),
                        Err(_) => Vec::new(),
                    }
                };
                if targets.is_empty() {
                    emit(
                        &sink,
                        json!({ "ev": "notice", "text": format!("nothing to load for '{name}'") }),
                    );
                }

                for (label, assembly, dir) in targets {
                    // The kernel moves out for the await: holding the state
                    // lock across it would deadlock the UI thread.
                    let mut kernel = {
                        let mut guard = state.lock().expect("state lock");
                        guard.take_kernel()
                    };
                    let loaded = nguruvilu::loader::load_pack(
                        &mut kernel,
                        &nguruvilu::ledger::default_ledger_path(),
                        &assembly,
                        &label,
                    )
                    .await;

                    let mut guard = state.lock().expect("state lock");
                    match loaded {
                        Ok((pack_skills, report, servers)) => {
                            guard.mcp.extend(servers);
                            {
                                let mut host = guard.config.skills.lock().expect("skills lock");
                                if let Err(error) =
                                    nguruvilu::loader::merge_skills(&mut kernel, &mut *host, pack_skills)
                                {
                                    emit(
                                        &sink,
                                        json!({ "ev": "error", "message": format!("skills for {label}: {error:#}") }),
                                    );
                                }
                            }
                            if let Ok(manifest) = nguruvilu::pack::read_manifest(&dir) {
                                match nguruvilu::content::apply(&mut kernel, &dir, &manifest) {
                                    Ok(content) => guard.adopt_pack_content(content),
                                    Err(error) => emit(
                                        &sink,
                                        json!({ "ev": "error", "message": format!("{label} content: {error:#}") }),
                                    ),
                                }
                            }
                            eprintln!(
                                "[pack] loaded {label}: {} entr(ies), {} failed",
                                report.loaded.len(),
                                report.failed.len()
                            );
                            changed = true;
                        }
                        Err(error) => emit(
                            &sink,
                            json!({ "ev": "error", "message": format!("loading {label}: {error:#}") }),
                        ),
                    }
                    if let Err(error) = guard.restore_kernel(kernel) {
                        emit(
                            &sink,
                            json!({ "ev": "error", "message": format!("restoring the kernel: {error:#}") }),
                        );
                    }
                }
            }
        }
    }

    if changed {
        // Panels, theme, and labels move with the kernel, and the tool table
        // is what the next turn reads.
        emit_panels(&state, &sink);
        let status = state.lock().expect("state lock").describe();
        emit(&sink, json!({ "ev": "status", "status": status }));
        emit(
            &sink,
            json!({ "ev": "notice", "text": "The next turn sees the new tool table." }),
        );
    }
}

/// Load an installed pack's plugins, MCP servers, and skills into the kernel.
/// Load every installed pack, in name order.
///
/// A pack is a thing the user chose to install, so it loads without being asked
/// again. Failures are reported and do not stop the others: one pack with a
/// bad manifest must not leave the shell with no appearance at all.
async fn load_installed_packs(state: Arc<Mutex<AppState>>, sink: Arc<dyn EventSink>) {
    let dir = nguruvilu::pack::default_packs_dir();

    // Place what this build ships before listing: the packs decide what a
    // conversation has, and a report rather than a failure keeps a full or
    // read-only disk from stopping the shell.
    match nguruvilu::preinstall::seed(&dir).await {
        Ok(placed) if !placed.is_empty() => eprintln!("[preinstall] {}", placed.join(", ")),
        Ok(_) => {}
        Err(error) => eprintln!("[preinstall] {error:#}"),
    }

    let packs = match nguruvilu::pack::installed(&dir) {
        Ok(packs) => packs,
        Err(error) => {
            eprintln!("[pack] cannot list {}: {error:#}", dir.display());
            return;
        }
    };

    for pack in packs {
        // An unloaded pack keeps its files and stays out of every
        // conversation until it is loaded again.
        if !pack.enabled {
            eprintln!("[pack] {} is unloaded; files kept", pack.manifest.name);
            continue;
        }
        let Some(assembly) = pack.assembly.clone() else {
            continue;
        };
        let name = format!("{}-{}", pack.manifest.name, pack.manifest.version_id);
        match apply_pack(Arc::clone(&state), assembly, Arc::clone(&sink)).await {
            Ok(()) => eprintln!("[pack] loaded {name}"),
            Err(error) => {
                eprintln!("[pack] {name} failed: {error:#}");
                emit(
                    &sink,
                    json!({ "ev": "error", "message": format!("pack {name}: {error:#}") }),
                );
            }
        }
    }
}

/// Apply a pack's assembly and content.
async fn apply_pack(
    state: Arc<Mutex<AppState>>,
    assembly: PathBuf,
    sink: Arc<dyn EventSink>,
) -> anyhow::Result<()> {
    // Parse before taking the kernel: a malformed manifest must not be able to
    // strand it, leaving the app with no tools at all.
    let document = Assembly::from_file(&assembly)?;
    let plan = document.plan(platform_tag())?;

    // The kernel moves out for the duration: loading is async, and holding the
    // state lock across an await would deadlock the UI thread.
    let kernel = {
        let mut guard = state.lock().expect("state lock");
        guard.take_kernel()
    };

    let pack_name = assembly
        .parent()
        .and_then(|dir| nguruvilu::pack::read_manifest(dir).ok())
        .map(|manifest| format!("{}-{}", manifest.name, manifest.version_id))
        .unwrap_or_else(|| "pack".to_string());

    let mut loader = Loader::new(kernel, default_ledger_path(), pack_name)
        .with_pack_dir(assembly.parent().map(PathBuf::from).unwrap_or_else(|| PathBuf::from(".")));

    let report = loader.apply(&plan).await?;
    // The MCP servers stay up: their tools are in this kernel and are meant to
    // stay callable for as long as the app runs. The pack's skill roots join
    // the shell's catalog rather than replacing it.
    let (mut kernel, pack_skills, servers) = loader.finish().await;
    {
        let mut guard = state.lock().expect("state lock");
        guard.mcp.extend(servers);
        let mut host = guard.config.skills.lock().expect("skills lock");
        nguruvilu::loader::merge_skills(&mut kernel, &mut host, pack_skills)?;
    }
    // Apply what the pack carries beyond its entries. Appearance lands on the
    // kernel; the rest describes the session and is applied to the shell's
    // settings, which the state owns.
    let pack_dir = assembly
        .parent()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    let content = match nguruvilu::pack::read_manifest(&pack_dir) {
        Ok(manifest) => {
            let content = nguruvilu::content::apply(&mut kernel, &pack_dir, &manifest)?;
            for line in content.summary() {
                eprintln!("[pack] content: {line}");
            }
            Some(content)
        }
        Err(error) => {
            // An assembly with no manifest beside it is still loadable — it is
            // the offline shape from before packs had manifests.
            eprintln!("[pack] no manifest beside the assembly ({error:#})");
            None
        }
    };

    let status = {
        let mut guard = state.lock().expect("state lock");
        if let Some(content) = content {
            guard.adopt_pack_content(content);
        }
        guard.restore_kernel(kernel)?;
        guard.describe()
    };

    eprintln!(
        "[pack] applied {} entries from {}",
        report.loaded.len(),
        assembly.display()
    );
    emit(
        &sink,
        json!({
            "ev": "pack_applied",
            "loaded": report.loaded,
            "failed": report.failed.iter().map(|f| json!({"id": f.id, "error": f.error})).collect::<Vec<_>>(),
            "skipped": report.skipped.iter().map(|s| json!({"id": s.id, "reason": s.reason})).collect::<Vec<_>>(),
        }),
    );
    emit(&sink, json!({ "ev": "status", "status": status }));
    Ok(())
}

/// Fetch the provider's model catalog and hand it to the page.
///
/// Asking the endpoint beats making a user type a model id from memory: a
/// gateway can front dozens of models and its list changes without notice.
async fn fetch_models(
    state: Arc<Mutex<AppState>>,
    sink: Arc<dyn EventSink>,
) -> anyhow::Result<()> {
    let client = {
        let guard = state.lock().expect("state lock");
        guard.client()?
    };

    match client.list_models_detailed().await {
        Ok(models) => {
            eprintln!("[models] provider reported {} entries", models.len());
            emit(
                &sink,
                json!({
                    "ev": "models",
                    "list": models.iter().map(|model| json!({
                        "id": model.id,
                        "owned_by": model.owned_by,
                        "label": model.label(),
                        "endpoint_types": model.endpoint_types,
                    })).collect::<Vec<_>>(),
                }),
            );
        }
        Err(error) => emit(
            &sink,
            json!({ "ev": "error", "message": format!("listing models: {error:#}") }),
        ),
    }
    Ok(())
}

/// Platform tag used by assembly manifests.
fn platform_tag() -> &'static str {
    if cfg!(windows) {
        "win32"
    } else if cfg!(target_os = "macos") {
        "darwin"
    } else {
        "linux"
    }
}

/// Remove an installed pack by `name` or `name@version`.
fn remove_pack(target: &str) -> anyhow::Result<()> {
    use nguruvilu::source::PackSource;

    let root = nguruvilu::pack::default_packs_dir();
    let source = nguruvilu::source::FilesystemSource::new(&root);

    let (name, version) = match target.split_once('@') {
        Some((name, version)) => (name, Some(version)),
        None => (target, None),
    };

    let matches: Vec<nguruvilu::pack::InstalledPack> = source
        .list()?
        .into_iter()
        .filter(|pack| pack.manifest.name == name)
        .filter(|pack| match version {
            Some(version) => pack.manifest.version_id == version,
            None => true,
        })
        .collect();

    if matches.is_empty() {
        anyhow::bail!("no installed pack matches '{target}'");
    }
    for pack in &matches {
        source.remove(&pack.path)?;
    }
    Ok(())
}

/// Find the interface to serve, before the window exists.
///
/// The page is loaded once when the window is created, so which interface it
/// gets has to be decided first — switching afterwards would mean reloading.
///
/// `wanted` names one by id; without it, the first installed pack that declares
/// an interface wins. The built-in is what a machine with no such pack gets,
/// which is why nothing has to be configured for the program to work.
fn resolve_active_ui(wanted: Option<&str>) -> Option<PathBuf> {
    let packs = nguruvilu::pack::installed(&nguruvilu::pack::default_packs_dir()).ok()?;

    for pack in &packs {
        // An unloaded pack is out of every conversation, and that includes the
        // interface it brought: unloading it puts the built-in one back.
        if !pack.enabled {
            continue;
        }
        let Some(ui) = &pack.manifest.ui else {
            continue;
        };
        if let Some(wanted) = wanted {
            if ui.id != wanted {
                continue;
            }
        }
        // `entry` is relative to the installed pack, so its parent is the
        // directory the assets live in.
        let entry = pack.path.join(&ui.entry);
        if !entry.is_file() {
            eprintln!(
                "[ui] {} declares interface '{}' but {} is not there",
                pack.manifest.name,
                ui.id,
                entry.display()
            );
            continue;
        }
        let directory = entry.parent().map(PathBuf::from)?;
        eprintln!("[ui] serving interface '{}' from {}", ui.id, directory.display());
        return Some(directory);
    }

    if wanted.is_some() {
        eprintln!(
            "[ui] no installed pack declares interface '{}'; using the built-in one",
            wanted.unwrap_or_default()
        );
    }
    None
}

/// Where the served interface lives.
///
/// Set once when a pack declares one. A global rather than a parameter because
/// the custom-protocol handler is a plain closure that the webview builder owns
/// for the life of the window, and it must answer without knowing about state
/// locks — a handler that blocks on the state mutex while the UI thread holds
/// it would deadlock the window.
static ACTIVE_UI: OnceLock<PathBuf> = OnceLock::new();

/// Serve a path from the active interface directory, or the built-in page.
///
/// A path that is not there falls back to the entry file rather than 404: a
/// single-page interface routes its own URLs, and a reload on one of those must
/// still render the application.
/// Serve the interface over a loopback HTTP origin, and return the port.
///
/// The window navigates to `http://127.0.0.1:<port>/` because WebView2 refuses
/// an unknown scheme as the document's own origin — `ngu://localhost/` lands on
/// nothing and the window comes up blank. A real origin is what lets the script
/// run, and [`serve_ui`] answers every request exactly as before.
///
/// One thread, one request per connection: this is a file server for a window
/// that loads once, not a service. Bound to loopback and sending no CORS
/// headers, so a page on another origin cannot read these responses.
/// The file dialog's owner, rebuilt from raw handles captured at window
/// creation.
///
/// The IPC callback must be `'static`, so it cannot borrow the tao window;
/// instead the two raw handles are copied once, and the window outlives every
/// dialog because dialogs only open while the event loop is running. Without
/// an owner the dialog opens *behind* the main window and the click looks like
/// nothing happened.
struct DialogParent {
    window: tao::rwh_06::RawWindowHandle,
    display: tao::rwh_06::RawDisplayHandle,
}

impl DialogParent {
    fn new(
        window: tao::rwh_06::RawWindowHandle,
        display: tao::rwh_06::RawDisplayHandle,
    ) -> Self {
        Self { window, display }
    }
}

impl tao::rwh_06::HasWindowHandle for DialogParent {
    fn window_handle(
        &self,
    ) -> Result<tao::rwh_06::WindowHandle<'_>, tao::rwh_06::HandleError> {
        // SAFETY: these handles came from a window that outlives every dialog —
        // dialogs only open while the event loop, and so the window, runs.
        Ok(unsafe { tao::rwh_06::WindowHandle::borrow_raw(self.window) })
    }
}

impl tao::rwh_06::HasDisplayHandle for DialogParent {
    fn display_handle(
        &self,
    ) -> Result<tao::rwh_06::DisplayHandle<'_>, tao::rwh_06::HandleError> {
        // SAFETY: same window as `window_handle` above.
        Ok(unsafe { tao::rwh_06::DisplayHandle::borrow_raw(self.display) })
    }
}

fn serve_interface_on_loopback() -> std::io::Result<u16> {
    let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
    let port = listener.local_addr()?.port();
    // Logged so a stalled page can be checked with one curl instead of a
    // debugger: `curl -I http://127.0.0.1:<port>/` shows the Content-Type.
    eprintln!("[ui] serving http://127.0.0.1:{port}/");
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            let mut request = [0u8; 8192];
            let read = stream.read(&mut request).unwrap_or(0);
            let text = String::from_utf8_lossy(&request[..read]).to_string();
            // GET <path> HTTP/1.1 — anything unparseable gets the entry page.
            let path = text
                .split_whitespace()
                .nth(1)
                .and_then(|target| target.split('?').next())
                .map(|target| target.trim_start_matches('/').to_string())
                .unwrap_or_default();
            let (body, mime) = serve_ui(&path);
            let head = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: {mime}\r\nContent-Length: {}\r\n\
                 Cache-Control: no-store\r\nConnection: close\r\n\r\n",
                body.len()
            );
            if stream.write_all(head.as_bytes()).is_err() {
                continue;
            }
            let _ = stream.write_all(&body);
            let _ = stream.flush();
        }
    });
    Ok(port)
}

fn serve_ui(path: &str) -> (Vec<u8>, &'static str) {
    let builtin = || (UI_HTML.as_bytes().to_vec(), "text/html; charset=utf-8");

    let Some(root) = ACTIVE_UI.get() else {
        return builtin();
    };

    // A path from a URL is untrusted input that becomes a filesystem path.
    // Refusing anything but a plain relative descent is what keeps `..` from
    // reaching outside the interface directory.
    let requested = if path.is_empty() { "index.html" } else { path };
    let candidate = if requested
        .split('/')
        .all(|part| !part.is_empty() && part != "." && part != "..")
    {
        root.join(requested)
    } else {
        root.join("index.html")
    };

    let file = if candidate.is_file() {
        candidate
    } else {
        root.join("index.html")
    };

    match std::fs::read(&file) {
        Ok(bytes) => (bytes, mime_for(&file)),
        Err(error) => {
            eprintln!("[ui] cannot read {}: {error}", file.display());
            builtin()
        }
    }
}

/// A content type for the extensions an interface actually uses.
fn mime_for(path: &Path) -> &'static str {
    match path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase()
        .as_str()
    {
        // The entry document first: without `text/html` the browser treats the
        // page as a download and the window comes up blank — which is exactly
        // what every window did while `html` was missing from this match.
        "html" | "htm" => "text/html; charset=utf-8",
        "css" => "text/css; charset=utf-8",
        "js" | "mjs" => "text/javascript; charset=utf-8",
        "json" => "application/json",
        "svg" => "image/svg+xml",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "woff" => "font/woff",
        "woff2" => "font/woff2",
        "ttf" => "font/ttf",
        "wasm" => "application/wasm",
        "mp3" => "audio/mpeg",
        "mp4" => "video/mp4",
        _ => "application/octet-stream",
    }
}

/// Point the window at an interface directory.
fn set_active_ui(dir: PathBuf) {
    // First writer wins: the interface is chosen once, at startup, and a second
    // call would silently serve a different directory than the page was loaded
    // from.
    if ACTIVE_UI.set(dir).is_ok() {
        // nothing to do; the handler reads it on the next request
    }
}

/// Send the panels, theme, and labels plugins contributed.
fn emit_panels(state: &Arc<Mutex<AppState>>, sink: &Arc<dyn EventSink>) {
    let (panels, theme, strings) = {
        let guard = state.lock().expect("state lock");
        (
            guard.kernel.ui_panels(),
            guard.kernel.resolved_theme(),
            guard.kernel.resolved_strings(),
        )
    };
    sink.emit(json!({ "ev": "ui_panels", "panels": panels }));
    // The theme and the labels travel with the panels: all three are plugin
    // contributions, and sending them together means the page never applies
    // one without the others — a translated interface with the wrong palette,
    // or the right palette with the previous pack's words.
    sink.emit(json!({ "ev": "theme", "tokens": theme.tokens, "name": theme.name }));
    sink.emit(json!({ "ev": "strings", "strings": strings }));
}

/// Send one event to the page.
fn emit(sink: &Arc<dyn EventSink>, value: Value) {
    sink.emit(value);
}
