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

use std::io::BufRead;
use std::path::PathBuf;
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};

use serde_json::{json, Value};
use tao::dpi::LogicalSize;
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

    runtime.block_on(async {
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
        Ok::<(), anyhow::Error>(())
    })
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

    let state = Arc::new(Mutex::new(AppState::bootstrap()?));
    let sink: Arc<dyn EventSink> = Arc::new(WindowSink::new(proxy));

    // Agent turns are async; the window is not. The runtime lives for the whole
    // process, moved into the event loop closure so it is not dropped early.
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    let handle = runtime.handle().clone();

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

        WebViewBuilder::new()
            // `with_html` serves the page from an opaque origin, where WebView2
            // refuses to run inline scripts — the page renders but stays inert.
            // A custom protocol gives it a real origin, so its script runs.
            //
            // The handler is a file server rather than a single page: an
            // interface a pack brings is a directory of assets, and its own
            // stylesheets, scripts, and images resolve as relative paths
            // against it. Which directory is served is decided when a pack
            // declares one; until then it is the built-in interface, so the
            // window is never blank.
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
            .with_url("ngu://localhost/")
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
                // runtime. `None` means the user cancelled.
                let picked: Option<PathBuf> = match name.as_str() {
                    "pick_install" | "pick_verify" => rfd::FileDialog::new()
                        .set_title("Choose a .dshpack archive")
                        .add_filter("pack archive", &["dshpack"])
                        .pick_file(),
                    "pick_pack" => rfd::FileDialog::new()
                        .set_title("Choose a pack directory (it must contain dsh.index.json)")
                        .pick_folder(),
                    _ => None,
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
                state::run_turn(state, text, Arc::clone(&sink)).await?;
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
            state::run_turn(state, text, Arc::clone(&sink)).await?;
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

        "uninstall_pack" => {
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
            match remove_pack(&target) {
                Ok(()) => {
                    sink.emit(json!({ "ev": "notice", "text": format!("Removed {target}") }));
                    // The shell loaded the pack at startup, so the panels and
                    // theme it contributed are still live. Say what to do about
                    // it rather than leaving a removed pack on screen.
                    sink.emit(json!({
                        "ev": "notice",
                        "text": "Restart the app to drop what it loaded.",
                    }));
                }
                Err(error) => emit(
                    &sink,
                    json!({ "ev": "error", "message": format!("removing {target}: {error:#}") }),
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

/// Load an installed pack's plugins, MCP servers, and skills into the kernel.
/// Load every installed pack, in name order.
///
/// A pack is a thing the user chose to install, so it loads without being asked
/// again. Failures are reported and do not stop the others: one pack with a
/// bad manifest must not leave the shell with no appearance at all.
async fn load_installed_packs(state: Arc<Mutex<AppState>>, sink: Arc<dyn EventSink>) {
    let dir = nguruvilu::pack::default_packs_dir();
    let packs = match nguruvilu::pack::installed(&dir) {
        Ok(packs) => packs,
        Err(error) => {
            eprintln!("[pack] cannot list {}: {error:#}", dir.display());
            return;
        }
    };

    for pack in packs {
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
    let mut kernel = loader.finish().await;

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

/// Send the panels plugins contributed.
fn emit_panels(state: &Arc<Mutex<AppState>>, sink: &Arc<dyn EventSink>) {
    let (panels, theme) = {
        let guard = state.lock().expect("state lock");
        (guard.kernel.ui_panels(), guard.kernel.resolved_theme())
    };
    sink.emit(json!({ "ev": "ui_panels", "panels": panels }));
    // The theme travels with the panels: both are plugin contributions, and
    // sending them together means the page never applies one without the other.
    sink.emit(json!({ "ev": "theme", "tokens": theme.tokens, "name": theme.name }));
}

/// Send one event to the page.
fn emit(sink: &Arc<dyn EventSink>, value: Value) {
    sink.emit(value);
}
