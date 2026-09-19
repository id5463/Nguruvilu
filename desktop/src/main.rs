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

mod state;

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use serde_json::{json, Value};
use tao::dpi::LogicalSize;
use tao::event::{Event, WindowEvent};
use tao::event_loop::{ControlFlow, EventLoopBuilder, EventLoopProxy};
use tao::window::WindowBuilder;
use wry::WebViewBuilder;

use nguruvilu::assembly::Assembly;
use nguruvilu::ledger::default_ledger_path;
use nguruvilu::loader::Loader;

use state::AppState;

/// The interface, embedded at compile time.
const UI_HTML: &str = include_str!("../ui/index.html");

/// Events the loop delivers to the page.
pub enum UserEvent {
    /// Evaluate this value as a call into the page's `window.ngu.receive`.
    ToUi(Value),
}

fn main() -> anyhow::Result<()> {
    // WebView2 takes its profile location from the environment. Setting it here
    // keeps the browser profile out of whatever directory the app was launched
    // from, which may be read-only.
    std::env::set_var("WEBVIEW2_USER_DATA_FOLDER", state::data_directory());

    // A prompt on the command line is sent as soon as the page is ready. It
    // makes the window scriptable and gives the shell a smoke test that does
    // not depend on someone watching the screen.
    let startup_prompt: Option<String> = {
        let args: Vec<String> = std::env::args().collect();
        args.iter()
            .position(|arg| arg == "--prompt" || arg == "-p")
            .and_then(|index| args.get(index + 1).cloned())
    };
    // EventLoopBuilder is how tao attaches a custom user event type.
    let event_loop = EventLoopBuilder::<UserEvent>::with_user_event().build();
    let proxy = event_loop.create_proxy();

    let window = WindowBuilder::new()
        .with_title("Nguruvilu")
        .with_inner_size(LogicalSize::new(1280.0, 820.0))
        .with_min_inner_size(LogicalSize::new(760.0, 500.0))
        .build(&event_loop)?;

    let state = Arc::new(Mutex::new(AppState::bootstrap()?));

    // Agent turns are async; the window is not. The runtime lives for the whole
    // process, moved into the event loop closure so it is not dropped early.
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    let handle = runtime.handle().clone();

    let webview = {
        let state = Arc::clone(&state);
        let proxy = proxy.clone();
                let startup_prompt = startup_prompt.clone();
        WebViewBuilder::new()
            // `with_html` serves the page from an opaque origin, where WebView2
            // refuses to run inline scripts — the page renders but stays inert.
            // A custom protocol gives it a real origin, so its script runs.
            .with_custom_protocol("ngu".into(), move |_id, _request| {
                wry::http::Response::builder()
                    .header("Content-Type", "text/html; charset=utf-8")
                    .body(std::borrow::Cow::Borrowed(UI_HTML.as_bytes()))
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
                let proxy = proxy.clone();
                let startup_prompt = startup_prompt.clone();
                // The handler is synchronous and must return quickly, so the
                // work goes onto the runtime and answers arrive as events.
                handle.spawn(async move {
                    let outcome = if name.starts_with("pick_") {
                        match picked {
                            Some(path) => {
                                dispatch_path(state, &name, path, proxy.clone()).await
                            }
                            // Cancelling a dialog is not an error.
                            None => Ok(()),
                        }
                    } else {
                        dispatch(state, &body, proxy.clone(), startup_prompt.clone()).await
                    };
                    if let Err(error) = outcome {
                        let _ = proxy.send_event(UserEvent::ToUi(json!({
                            "ev": "error",
                            "message": format!("{error:#}"),
                        })));
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
    proxy: EventLoopProxy<UserEvent>,
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
            let (sessions, transcript, status) = {
                let guard = state.lock().expect("state lock");
                (guard.sessions()?, guard.transcript(), guard.describe())
            };
            emit(&proxy, json!({ "ev": "sessions", "list": sessions }));
            emit(&proxy, json!({ "ev": "transcript", "entries": transcript }));
            emit(&proxy, json!({ "ev": "status", "status": status }));

            // A prompt given on the command line runs once the page can show it.

            // Fetch the catalog up front when the endpoint is configured: a picker
            // that only fills after someone finds a button is a picker nobody finds.
            // Failure is reported by the fetch itself and is not fatal.
            let configured = state.lock().expect("state lock").settings.is_configured();
            if configured {
                let _ = fetch_models(Arc::clone(&state), proxy.clone()).await;
            }
            if let Some(text) = startup_prompt {
                emit(&proxy, json!({ "ev": "turn_start", "text": text }));
                state::run_turn(state, text, proxy.clone()).await?;
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
            emit(&proxy, json!({ "ev": "turn_start", "text": text }));
            state::run_turn(state, text, proxy.clone()).await?;
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
            emit(&proxy, json!({ "ev": "sessions", "list": sessions }));
            emit(&proxy, json!({ "ev": "transcript", "entries": transcript }));
            emit(&proxy, json!({ "ev": "status", "status": status }));
        }

        "set_model" => {
            let model = command.get("model").and_then(|m| m.as_str()).unwrap_or("");
            let base_url = command.get("base_url").and_then(|u| u.as_str());
            let status = {
                let mut guard = state.lock().expect("state lock");
                guard.set_model(model, base_url)?;
                guard.describe()
            };
            emit(&proxy, json!({ "ev": "status", "status": status }));
        }

        "save_settings" => {
            let base_url = command.get("base_url").and_then(|v| v.as_str()).unwrap_or("");
            let api_key_field = command.get("api_key").and_then(|v| v.as_str()).unwrap_or("");
            let keep_key = command.get("keep_api_key").and_then(|v| v.as_bool()).unwrap_or(false);
            let model = command.get("model").and_then(|v| v.as_str()).unwrap_or("");
            let effort = command.get("reasoning_effort").and_then(|v| v.as_str()).unwrap_or("");

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
                guard.save_settings(base_url, &api_key, model, effort)
            };

            match outcome {
                Ok(()) => {
                    eprintln!("[settings] saved to {}", nguruvilu::settings::Settings::path().display());
                    let status = state.lock().expect("state lock").describe();
                    emit(&proxy, json!({ "ev": "status", "status": status }));
                    emit(&proxy, json!({ "ev": "settings_saved" }));
                }
                Err(error) => {
                    emit(
                        &proxy,
                        json!({ "ev": "error", "message": format!("saving settings: {error:#}") }),
                    );
                }
            }
        }

        "fetch_models" => {
            fetch_models(state, proxy.clone()).await?;
        }

        "list_packs" => {
            refresh_packs(&state, &proxy);
        }

        "apply_pack" => {
            let path = command.get("path").and_then(|v| v.as_str()).unwrap_or("");
            if path.is_empty() {
                emit(&proxy, json!({ "ev": "error", "message": "apply_pack needs a path" }));
            } else {
                apply_pack(state, PathBuf::from(path), proxy.clone()).await?;
            }
        }

        "status" => {
            let status = state.lock().expect("state lock").describe();
            emit(&proxy, json!({ "ev": "status", "status": status }));
        }






        "" => {
            emit(&proxy, json!({ "ev": "error", "message": "empty command" }));
        }

        other => {
            emit(
                &proxy,
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
    proxy: EventLoopProxy<UserEvent>,
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
                Ok(report) => emit(&proxy, json!({ "ev": "pack_verified", "report": report })),
                Err(error) => {
                    emit(
                        &proxy,
                        json!({ "ev": "error", "message": format!("{}: {error:#}", path.display()) }),
                    );
                    return Ok(());
                }
            }

            let installed = {
                let mut guard = state.lock().expect("state lock");
                guard.install_pack(&path)
            };
            match installed {
                Ok(info) => {
                    eprintln!("[pack] installed {} {}", info["name"], info["version"]);
                    emit(&proxy, json!({ "ev": "pack_installed", "pack": info }));
                }
                Err(error) => emit(
                    &proxy,
                    json!({ "ev": "error", "message": format!("installing {}: {error:#}", path.display()) }),
                ),
            }
            refresh_packs(&state, &proxy);
        }

        "pick_verify" => {
            let result = { state.lock().expect("state lock").verify_pack(&path) };
            match result {
                Ok(report) => emit(
                    &proxy,
                    json!({ "ev": "pack_verified", "report": report, "path": path.display().to_string() }),
                ),
                Err(error) => emit(&proxy, json!({ "ev": "error", "message": format!("{error:#}") })),
            }
        }

        "pick_pack" => {
            let result = { state.lock().expect("state lock").pack_dir(&path, None) };
            match result {
                Ok(info) => {
                    eprintln!("[pack] built {}", info["archive"]);
                    emit(&proxy, json!({ "ev": "pack_built", "pack": info }));
                }
                Err(error) => emit(
                    &proxy,
                    json!({ "ev": "error", "message": format!("packing {}: {error:#}", path.display()) }),
                ),
            }
        }

        other => {
            emit(
                &proxy,
                json!({ "ev": "error", "message": format!("unknown picker command '{other}'") }),
            );
        }
    }
    Ok(())
}

/// Push the installed-pack list to the page.
fn refresh_packs(state: &Arc<Mutex<AppState>>, proxy: &EventLoopProxy<UserEvent>) {
    match state.lock().expect("state lock").packs() {
        Ok(list) => emit(proxy, json!({ "ev": "packs", "list": list })),
        Err(error) => emit(
            proxy,
            json!({ "ev": "error", "message": format!("listing packs: {error:#}") }),
        ),
    }
}

/// Load an installed pack's plugins, MCP servers, and skills into the kernel.
async fn apply_pack(
    state: Arc<Mutex<AppState>>,
    assembly: PathBuf,
    proxy: EventLoopProxy<UserEvent>,
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
    let kernel = loader.finish().await;

    let status = {
        let mut guard = state.lock().expect("state lock");
        guard.restore_kernel(kernel)?;
        guard.describe()
    };

    eprintln!(
        "[pack] applied {} entries from {}",
        report.loaded.len(),
        assembly.display()
    );
    emit(
        &proxy,
        json!({
            "ev": "pack_applied",
            "loaded": report.loaded,
            "failed": report.failed.iter().map(|f| json!({"id": f.id, "error": f.error})).collect::<Vec<_>>(),
            "skipped": report.skipped.iter().map(|s| json!({"id": s.id, "reason": s.reason})).collect::<Vec<_>>(),
        }),
    );
    emit(&proxy, json!({ "ev": "status", "status": status }));
    Ok(())
}

/// Fetch the provider's model catalog and hand it to the page.
///
/// Asking the endpoint beats making a user type a model id from memory: a
/// gateway can front dozens of models and its list changes without notice.
async fn fetch_models(
    state: Arc<Mutex<AppState>>,
    proxy: EventLoopProxy<UserEvent>,
) -> anyhow::Result<()> {
    let client = {
        let guard = state.lock().expect("state lock");
        guard.client()?
    };

    match client.list_models_detailed().await {
        Ok(models) => {
            eprintln!("[models] provider reported {} entries", models.len());
            emit(
                &proxy,
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
            &proxy,
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

/// Send one event to the page.
fn emit(proxy: &EventLoopProxy<UserEvent>, value: Value) {
    let _ = proxy.send_event(UserEvent::ToUi(value));
}
