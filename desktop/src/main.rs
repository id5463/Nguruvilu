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

use std::sync::{Arc, Mutex};

use serde_json::{json, Value};
use tao::dpi::LogicalSize;
use tao::event::{Event, WindowEvent};
use tao::event_loop::{ControlFlow, EventLoopBuilder, EventLoopProxy};
use tao::window::WindowBuilder;
use wry::WebViewBuilder;

use state::AppState;

/// The interface, embedded at compile time.
const UI_HTML: &str = include_str!("../ui/index.html");

/// Events the loop delivers to the page.
pub enum UserEvent {
    /// Evaluate this value as a call into the page's `window.ngu.receive`.
    ToUi(Value),
}

fn main() -> anyhow::Result<()> {
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
            // Diagnostic: report whether the page can reach the shell at all, and
            // surface any script error, instead of a silent blank window.
            .with_ipc_handler(move |request| {
                let body = request.body().to_string();
                let state = Arc::clone(&state);
                let proxy = proxy.clone();
                let startup_prompt = startup_prompt.clone();
                // The handler is synchronous and must return quickly, so the
                // work goes onto the runtime and answers arrive as events.
                handle.spawn(async move {
                    let outcome =
                        dispatch(state, &body, proxy.clone(), startup_prompt.clone()).await;
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

/// Send one event to the page.
fn emit(proxy: &EventLoopProxy<UserEvent>, value: Value) {
    let _ = proxy.send_event(UserEvent::ToUi(value));
}
