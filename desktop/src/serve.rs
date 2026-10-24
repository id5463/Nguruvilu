//! `--serve`: the same shell over a socket.
//!
//! The window exists only where the webview stack exists; everything under it —
//! state, dispatch, headless — does not care. This module adds the last layer:
//! the page the window renders, served over HTTP/SSE so a browser on any machine
//! can be the window. Commands come in on `POST /cmd`, events go out on
//! `GET /events`, and both are the page's own `dispatch` — not a parallel
//! implementation that can drift from it.
//!
//! The door is open. This server used to be locked with a bearer token,
//! generated fresh each run and printed with the URL to open; the owner's
//! verdict on that was final — 安全是不需要的。越危险越好，安全没有任何用处 —
//! because the token cost more than it bought. The printed URL was also the only
//! way in, so every request that arrived without `?token=…` got `401`: a reload
//! from history, a bookmark, the address typed on a phone, the browser asking
//! for a favicon. A page that answers `401` to its own user reads as a broken
//! app, and that was the first complaint a real user made about `--serve`. The
//! exposure is accepted now instead of gated: whoever reaches the port drives
//! the shell, reads files under the session's directory, and installs packs.
//!
//! What is left in this file is capacity and correctness, never permission —
//! connection and stream caps, body and download limits, read timeouts, and the
//! path router that keeps an interface request inside the interface directory.
//! Those stay because dropping them would not free the page, it would break it:
//! without the caps one slow-loris takes the server down, and without the router
//! the entry document cannot be found at all.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{broadcast, mpsc};

use crate::sink::EventSink;
use crate::state::AppState;
use crate::{resolve_active_ui, serve_ui, set_active_ui};

/// Longest download `/file` will serve, in bytes.
///
/// The body is copied in chunks and never buffered whole, but a cap still
/// exists: an unbounded stream is one request away from saturating the link, and
/// the download path is the one place a size can be checked before a byte of it
/// is sent. 64 MiB is past any transcript or upload round-trip and short of a
/// machine image.
const MAX_FILE_BYTES: u64 = 64 * 1024 * 1024;

/// How many SSE clients may be attached at once.
///
/// Every subscriber holds a socket and a place in the broadcast queue, and with
/// the token gone this cap is the only thing between the port and a tab
/// reconnecting in a loop: one page opens one stream and one page on two
/// machines opens two, so 16 leaves room for real use and stops the unbounded
/// case.
const MAX_SSE_SUBSCRIBERS: usize = 16;

/// Accepted connections currently inside [`handle`], and the ceiling on them.
///
/// The counter is what stops a slow-loris that never finishes its headers from
/// occupying every accept slot: past the limit a new connection is closed
/// immediately, which costs one line in the log instead of a task and a file
/// handle per idle client.
static ACTIVE_CONNECTIONS: AtomicUsize = AtomicUsize::new(0);
const MAX_ACTIVE_CONNECTIONS: usize = 64;

/// How long the server waits for any single read while assembling a request.
///
/// A client that drips one byte a second would otherwise hold a connection slot
/// and a task for as long as it pleases; past this the connection is closed and
/// the slot returned.
const READ_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

/// One connection's claim on the [`ACTIVE_CONNECTIONS`] count, released by
/// `Drop` so every early return in [`handle`] returns its slot.
struct ConnectionSlot;

impl ConnectionSlot {
    /// Claim a slot, or `None` when the ceiling is already reached.
    fn acquire() -> Option<Self> {
        // A saturated counter is not a reason to keep accepting: closing here
        // is what makes the ceiling real, because the socket never reaches a
        // task that could hold it.
        if ACTIVE_CONNECTIONS.fetch_add(1, Ordering::Relaxed) >= MAX_ACTIVE_CONNECTIONS {
            ACTIVE_CONNECTIONS.fetch_sub(1, Ordering::Relaxed);
            return None;
        }
        Some(Self)
    }
}

impl Drop for ConnectionSlot {
    fn drop(&mut self) {
        ACTIVE_CONNECTIONS.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Subscriber count for the event stream, limited to [`MAX_SSE_SUBSCRIBERS`].
static SSE_SUBSCRIBERS: AtomicUsize = AtomicUsize::new(0);

/// One stream's claim on the subscriber count, released when the stream ends.
struct SubscriberSlot;

impl SubscriberSlot {
    fn acquire() -> Option<Self> {
        if SSE_SUBSCRIBERS.fetch_add(1, Ordering::Relaxed) >= MAX_SSE_SUBSCRIBERS {
            SSE_SUBSCRIBERS.fetch_sub(1, Ordering::Relaxed);
            return None;
        }
        Some(Self)
    }
}

impl Drop for SubscriberSlot {
    fn drop(&mut self) {
        SSE_SUBSCRIBERS.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Events fan out to every page currently listening on `/events`.
///
/// One sink, many connections: the window has exactly one page, but a served
/// shell may have a tab open on two machines, and both must see the same turn.
pub struct SseSink {
    bus: broadcast::Sender<String>,
}

impl EventSink for SseSink {
    fn emit(&self, event: Value) {
        // No listener yet (the page is between loads) drops the event, exactly
        // as a closed window drops it; the next `ready` re-sends the full
        // transcript, so nothing durable lives in the gap.
        let _ = self.bus.send(event.to_string());
    }
}

/// Serve the shell on `addr` until the process is killed.
///
/// `startup_prompt` runs once — on the first page's `ready`, not on every
/// reload, or a refresh would re-send the prompt as a new turn.
pub fn serve(
    addr: String,
    // Accepted and ignored. The caller still generates and passes a token, and
    // this keeps the parameter so its call site does not change; nothing on the
    // served side reads it any more. It is kept rather than removed because the
    // function is `pub` and its one caller lives in `main.rs`, which this change
    // does not own.
    token: String,
    startup_prompt: Option<String>,
    ui_id: Option<String>,
) -> anyhow::Result<()> {
    // The one line that says so at the point of use: the token exists, and no
    // door below consults it.
    let _ = token;
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    let state = Arc::new(Mutex::new(AppState::bootstrap()?));

    runtime.block_on(async move {
        // Same first-launch seeding as the window: the interface may come from
        // a pack, and a pack the build ships but never places is a pack the
        // first launch falls back without.
        match nguruvilu::preinstall::seed(&nguruvilu::pack::default_packs_dir()).await {
            Ok(placed) if !placed.is_empty() => eprintln!("[preinstall] {}", placed.join(", ")),
            Ok(_) => {}
            Err(error) => eprintln!("[preinstall] {error:#}"),
        }
        // Decided before the first request, for the same reason the window
        // decides before it loads: the page is one document, and which
        // interface it is cannot change under a running session.
        if let Some(directory) = resolve_active_ui(ui_id.as_deref()) {
            set_active_ui(directory);
        }

        let (bus, _) = broadcast::channel::<String>(256);
        let sink: Arc<dyn EventSink> = Arc::new(SseSink { bus: bus.clone() });
        // notify-jobs wake their session over the same event stream.
        crate::state::install_job_wake(Arc::clone(&state), Arc::clone(&sink));
        // Packs before the first request, not on a page's `ready`: a served
        // shell may be driven by a phone that never sends one, and a
        // conversation with only the base tools is indistinguishable from a
        // missing feature. `ready` still asks; the flag makes it a no-op.
        crate::load_packs_once(Arc::clone(&state), Arc::clone(&sink)).await;

        // Outbound remote control, same as the window: the process dials the
        // relay and every command still passes the allow-list in `remote`.
        {
            let (url, relay_token) = {
                let guard = state.lock().expect("state lock");
                (guard.settings.relay_url.clone(), guard.settings.relay_token.clone())
            };
            match crate::remote::disabled_reason(Some(&url), Some(&relay_token)) {
                Some(reason) => eprintln!("[remote] {reason}"),
                None => {
                    let config = crate::remote::RelayConfig { url, token: relay_token };
                    let relay_state = Arc::clone(&state);
                    let relay_sink = Arc::clone(&sink);
                    tokio::spawn(crate::remote::run_relay(config, move |body: String| {
                        let state = Arc::clone(&relay_state);
                        let sink = Arc::clone(&relay_sink);
                        Box::pin(async move {
                            // The body, rate and `prompt` gates all ran in
                            // `remote::admit` before this closure was called.
                            // What is left is the one check that needs the
                            // state: `open_session` moves the working
                            // directory, and with it the root `/file` and
                            // `/upload` resolve against, so the relay may only
                            // switch to a session this machine already has.
                            if let Some(id) = crate::remote::open_session_id(&body) {
                                if !session_exists(&state, &id) {
                                    eprintln!(
                                        "[remote] audit: open_session refused for an unknown session"
                                    );
                                    return r#"{"ok":false,"error":"unknown session: open_session may only open a session that already exists on this machine"}"#
                                        .to_string();
                                }
                            }
                            match crate::dispatch(state, &body, sink, None).await {
                                Ok(()) => r#"{"ok":true}"#.to_string(),
                                Err(error) => {
                                    serde_json::json!({ "ok": false, "error": format!("{error:#}") })
                                        .to_string()
                                }
                            }
                        })
                    }));
                }
            }
        }

        // Two lanes, split by what a command may *write*. A command that reads
        // state, or writes only the payload this page is showing, goes straight
        // to its own task — a ten-minute turn must not freeze `status`, the
        // numbers in the header, or the stop button behind it. Everything that
        // rewrites what a running turn reads, and everything unknown, stays
        // here in the ordered lane: `session_switch` re-pins the process-wide
        // tool table, and a session command issued beside a live turn would
        // move that turn's bash cwd, its relative paths and its job ownership.
        // `is_concurrent` is that judgement, and its test holds the two halves
        // against each other rather than restating the names.
        let (commands_tx, mut commands_rx) = mpsc::unbounded_channel::<String>();
        let startup = Arc::new(Mutex::new(startup_prompt));
        {
            let state = Arc::clone(&state);
            let sink = Arc::clone(&sink);
            tokio::spawn(async move {
                while let Some(body) = commands_rx.recv().await {
                    if is_concurrent(&body) {
                        let state = Arc::clone(&state);
                        let sink = Arc::clone(&sink);
                        tokio::spawn(async move {
                            // `stats` is answered here rather than by the
                            // shell's command table, and the same holds on this
                            // second route in: whichever door a command came
                            // through, it gets the same read-only answer.
                            if is_stats(&body) {
                                sink.emit(stats_event(&state, &body));
                                return;
                            }
                            if let Err(error) =
                                crate::dispatch(state, &body, sink.clone(), None).await
                            {
                                sink.emit(json!({ "ev": "error", "message": format!("{error:#}") }));
                            }
                        });
                        continue;
                    }
                    // Only the page's `ready` may collect the prompt; any other
                    // command arriving first must not steal it from the page
                    // that has not loaded yet.
                    let startup_prompt = if is_ready(&body) {
                        startup.lock().expect("startup lock").take()
                    } else {
                        None
                    };
                    if let Err(error) =
                        crate::dispatch(Arc::clone(&state), &body, Arc::clone(&sink), startup_prompt)
                            .await
                    {
                        sink.emit(json!({ "ev": "error", "message": format!("{error:#}") }));
                    }
                }
            });
        }

        let listener = TcpListener::bind(&addr)
            .await
            .map_err(|error| anyhow::anyhow!("cannot bind {addr}: {error}"))?;
        if !is_loopback(&addr) {
            eprintln!(
                "[serve] {addr} is not loopback: anyone who reaches this port can drive the \
                 shell"
            );
        }
        eprintln!("[serve] listening on http://{addr}");
        // The URL the owner opens, with nothing appended: there is no token to
        // carry, and a `?token=…` in the printed line was the thing that made a
        // hand-typed address 401.
        eprintln!("[serve] open http://{addr}/");

        loop {
            let (stream, peer) = match listener.accept().await {
                Ok(connection) => connection,
                // A connection that dies between accept and now is not a reason
                // to take the server with it.
                Err(error) => {
                    eprintln!("[serve] accept: {error}");
                    continue;
                }
            };
            // The slot is claimed before a task exists: a server that has run
            // out of capacity must not spawn the work it is about to refuse.
            let Some(slot) = ConnectionSlot::acquire() else {
                eprintln!(
                    "[serve] {peer} refused: {MAX_ACTIVE_CONNECTIONS} connections already open"
                );
                // Dropping the stream closes the connection with no reply:
                // there is nothing a client can do with "busy" here that
                // retrying will not do better.
                drop(stream);
                continue;
            };
            let bus = bus.clone();
            let commands_tx = commands_tx.clone();
            let state = Arc::clone(&state);
            tokio::spawn(async move {
                let _slot = slot;
                handle(stream, peer, bus, commands_tx, state).await;
            });
        }

        #[allow(unreachable_code)]
        Ok::<(), anyhow::Error>(())
    })?;

    // Deliberately not dropping the runtime — the same reason headless does
    // not: teardown would wait on servers that outlive their commands.
    Ok(())
}

/// One request in, one response out. Connections are not pooled: the page's
/// fetches are small and rare, and a closed connection is one less thing to
/// police against a client that went away mid-headers.
///
/// Nothing here decides *who* may ask. The method and the path pick a door
/// (see [`route`]) and the door answers; the peer address and the headers are
/// only ever read for log lines, never for permission.
async fn handle(
    mut stream: TcpStream,
    peer: std::net::SocketAddr,
    bus: broadcast::Sender<String>,
    commands_tx: mpsc::UnboundedSender<String>,
    state: Arc<Mutex<AppState>>,
) {
    let request = match read_request(&mut stream).await {
        ReadOutcome::Request(request) => request,
        ReadOutcome::TooLarge(declared) => {
            let limit_mb = max_body_bytes() / (1024 * 1024);
            // Rejections are logged too: a client that never appears in the
            // log with a big file is otherwise indistinguishable from one
            // that never tried.
            eprintln!(
                "[serve] {peer} rejected: {declared} bytes declared exceeds the \
                 {limit_mb} MB upload limit"
            );
            // The one case where silence is worse than an error: the client
            // picked a size this server refuses, and it deserves to hear that
            // before retrying — a dropped connection reads as a network fault.
            let body = format!(
                "payload too large: {declared} bytes declared, this server accepts up to \
                 {limit_mb} MB. Set NGU_MAX_UPLOAD_MB to raise the limit."
            );
            respond(
                &mut stream,
                413,
                "text/plain; charset=utf-8",
                body.as_bytes(),
            )
            .await;
            return;
        }
        ReadOutcome::Refused { status, reason } => {
            // A refusal this server chose to make, not a malformed request:
            // the client asked for something the server deliberately does not
            // do (a chunked body, a body with no declared length, a request
            // that arrived too slowly), and it gets told which.
            eprintln!("[serve] {peer} rejected: {status} {reason}");
            respond(&mut stream, status, "text/plain; charset=utf-8", reason.as_bytes()).await;
            return;
        }
        ReadOutcome::Malformed => return,
    };
    let (path, query) = split_target(&request.target);

    let (status, content_type, body) = match route(&request.method, path) {
        // The event stream. Attaching is one of the doors the owner opened: a
        // phone that opens the page has to get its transcript, and there is no
        // credential left to weigh it against. What stays is capacity — a tab
        // reconnecting in a loop still cannot grow the subscriber count away.
        Route::Events => match SubscriberSlot::acquire() {
            // A bare close here reads as a network fault; 429 plus a
            // reason is something the page can put in front of a user,
            // and it is the answer that says "this machine is already
            // streaming to as many listeners as it will".
            None => {
                eprintln!(
                    "[serve] {peer} refused an event stream: {MAX_SSE_SUBSCRIBERS} \
                     subscribers already attached"
                );
                respond(
                    &mut stream,
                    429,
                    "text/plain; charset=utf-8",
                    format!(
                        "too many event streams: this server keeps at most \
                         {MAX_SSE_SUBSCRIBERS} open. Close another tab and reload."
                    )
                    .as_bytes(),
                )
                .await;
                return;
            }
            Some(_slot) => {
                // One line per subscriber: it answers the only question
                // that matters when a page comes up blank — was anyone
                // listening.
                eprintln!("[serve] events stream to {peer}");
                // The slot lives until `sse` returns, so a client that
                // walks away returns its place with the socket.
                sse(&mut stream, bus).await;
                return;
            }
        },
        Route::Command => {
            let body_text = std::str::from_utf8(&request.body).unwrap_or("");
            // No credential is read here, and none is missing: this door
            // answers whoever reached the port. What it still refuses is a body
            // it cannot read as the one JSON object every command is.
            if serde_json::from_str::<Value>(body_text).is_err() {
                plain(400, "body must be one JSON object")
            } else {
                // A stop must not queue behind the very turn it is stopping:
                // this server runs commands one at a time, so a queued
                // `cancel` would only be reached after the turn already
                // ended. The flag is set under the lock and the turn does
                // the rest at its next safe point.
                let parsed: Value = serde_json::from_str(body_text).unwrap_or(Value::Null);
                // The two commands that accept a target — `prompt` and `cancel`
                // — share one reading of it here, because both have the same
                // failure mode: a `session` that is present but is not a string
                // would otherwise fall back to "no session", which aims the
                // command at the conversation on screen. That is the one answer
                // an explicit id must never get, so an unreadable one is the
                // caller's error rather than a value quietly dropped.
                let explicit = match parsed.get("cmd").and_then(Value::as_str) {
                    Some("prompt") | Some("cancel") => match optional_session(&parsed) {
                        Ok(session) => session,
                        Err(reason) => {
                            respond(
                                &mut stream,
                                400,
                                "text/plain; charset=utf-8",
                                reason.as_bytes(),
                            )
                            .await;
                            return;
                        }
                    },
                    // No other command on this door reads a `session`, so a
                    // value it cannot parse is not this door's business.
                    _ => None,
                };
                if parsed.get("cmd").and_then(|c| c.as_str()) == Some("cancel") {
                    let stopped = {
                        let guard = state.lock().expect("state lock");
                        // The optional `session` the ordered lane also reads:
                        // without it the only stoppable turn is the one on
                        // screen, and a stop aimed at a turn the page has
                        // steered away from would stop the wrong one — and
                        // clear that session's pending steering with it.
                        let session = explicit
                            .clone()
                            .unwrap_or_else(|| guard.session.id.clone());
                        let sent = guard
                            .cancels
                            .get(&session)
                            .is_some_and(|sender| sender.send(true).is_ok());
                        // Stop means stop: pending steering goes too.
                        if let Some(queue) = guard.inboxes.get(&session) {
                            queue.lock().expect("inbox").clear();
                        }
                        (sent, session)
                    };
                    if stopped.0 {
                        eprintln!("[serve] cancel for {}", stopped.1);
                    }
                    let body = format!("{{\"ok\":true,\"stopped\":{}}}", stopped.0);
                    (200, "application/json", body.into_bytes())
                } else if parsed.get("cmd").and_then(|c| c.as_str()) == Some("prompt") {
                    // Steering must reach the running turn's next step — a
                    // queued command would arrive only after that turn ended,
                    // which is the difference between redirecting the model
                    // and starting over. Idle sessions still queue normally.
                    //
                    // A `text` that is not a string is not a prompt that can be
                    // run: reading it as empty answers `ok` for a turn that
                    // never happens, which is the one reply worse than an error.
                    let text = match parsed.get("text") {
                        Some(Value::String(text)) => text.clone(),
                        None => String::new(),
                        Some(_) => {
                            respond(
                                &mut stream,
                                400,
                                "text/plain; charset=utf-8",
                                b"`text` must be a string",
                            )
                            .await;
                            return;
                        }
                    };
                    // A prompt aimed at another conversation is parked in that
                    // session's inbox instead of run here, so the id has to name
                    // a conversation this shell actually has: `run_turn` parks
                    // the message and returns, and an id that names nothing
                    // leaves the text in an inbox no turn can ever claim while
                    // this answer says it was accepted. The session on screen is
                    // never looked up — it is the one conversation this shell is
                    // certain to hold.
                    let target_check = {
                        let guard = state.lock().expect("state lock");
                        match explicit.as_deref() {
                            Some(id) if id != guard.session.id.as_str() => {
                                Some(listed_session(&guard, id))
                            }
                            _ => None,
                        }
                    };
                    match target_check {
                        Some(Ok(false)) => {
                            // Two consequences in one line without quoting the
                            // id back: a client that echoes its own untrusted
                            // text into this body would otherwise be handed a
                            // response built from it.
                            let reason = "no session on this machine has that id: a prompt for \
                                          another conversation is parked in that session's \
                                          inbox, so this text was not stored anywhere";
                            eprintln!("[serve] prompt from {peer} refused: unknown session");
                            respond(
                                &mut stream,
                                404,
                                "text/plain; charset=utf-8",
                                reason.as_bytes(),
                            )
                            .await;
                            return;
                        }
                        // The store could not be read, which is this server's
                        // failure rather than the caller's mistake, and the
                        // detail stays in the log: a store error quotes paths
                        // this response has no business handing out.
                        Some(Err(error)) => {
                            eprintln!(
                                "[serve] prompt from {peer} refused: cannot read the session \
                                 list: {error}"
                            );
                            let reason = "cannot read the session list right now: the prompt \
                                          was not queued";
                            respond(
                                &mut stream,
                                503,
                                "text/plain; charset=utf-8",
                                reason.as_bytes(),
                            )
                            .await;
                            return;
                        }
                        _ => {}
                    }
                    let steered = {
                        // A prompt aimed at another conversation must not steer
                        // the one on screen: `try_steer` queues into whatever
                        // session the shell is showing, so an explicit id that
                        // is not that session goes straight to `dispatch`,
                        // which parks it in the target's inbox instead.
                        let mut guard = state.lock().expect("state lock");
                        match explicit.as_deref() {
                            Some(id) if id != guard.session.id.as_str() => None,
                            _ => crate::state::try_steer(&mut guard, &text),
                        }
                    };
                    match steered {
                        Some(session) => {
                            eprintln!("[serve] steer queued for {session}");
                            let _ = bus.send(
                                json!({ "ev": "steer", "text": text, "session": session })
                                    .to_string(),
                            );
                            (200, "application/json", b"{\"ok\":true,\"steer\":true}".to_vec())
                        }
                        None if commands_tx.send(body_text.to_string()).is_err() => {
                            plain(503, "shell is shutting down")
                        }
                        None => (200, "application/json", b"{\"ok\":true}".to_vec()),
                    }
                } else if parsed.get("cmd").and_then(|c| c.as_str()) == Some("stats") {
                    // Read-only, and answered from this connection's own task:
                    // the ledger is already in memory, so there is nothing to
                    // queue for. Routing it through the ordered lane would put
                    // a usage read behind whatever pack work that lane holds —
                    // the same "one long command freezes the page" problem the
                    // two-lane split exists to prevent, seen from the other
                    // side. The answer leaves on the event stream like every
                    // other reply, so the page has one place to listen.
                    let event = stats_event(&state, body_text);
                    let _ = bus.send(event.to_string());
                    (200, "application/json", b"{\"ok\":true}".to_vec())
                } else if commands_tx.send(body_text.to_string()).is_err() {
                    // Queued, not awaited: the turn runs for seconds to minutes and
                    // its progress belongs on the event stream, not in this
                    // response. The page's fetch acknowledges delivery, not completion.
                    plain(503, "shell is shutting down")
                } else {
                    (200, "application/json", b"{\"ok\":true}".to_vec())
                }
            }
        }
        // Files both ways: the page hands a file to the shell, the shell's
        // paths come back as download links. Neither direction asks who is
        // asking. The token used to decide that, and all it ever changed was
        // that an upload from the page's own user could be refused; the file
        // still lands in `uploads/` and nowhere else, because the name rule
        // below is what keeps it there.
        Route::Upload => {
            // Logged before anything can reject it: every attempt leaves one
            // line, success or not.
            eprintln!(
                "[serve] POST /upload from {peer}: {} bytes",
                request.body.len()
            );
            if request.body.is_empty() {
                eprintln!("[serve] upload from {peer}: empty body");
                plain(400, "empty upload")
            } else if session_workdir_in(&state).is_none() {
                // No cwd means no landing place. Falling back to the process
                // directory would put an upload outside every session's root
                // and into whatever directory the shell was started from.
                let reason = "this session has no working directory yet: open or create a \
                              session first, then upload again";
                eprintln!("[serve] upload from {peer}: no session workdir");
                plain(409, reason)
            } else {
                let name = sanitize_name(query.get("name").map(String::as_str).unwrap_or(""));
                let directory = session_workdir_in(&state)
                    .expect("checked above")
                    .join("uploads");
                match tokio::fs::create_dir_all(&directory).await {
                    Err(error) => {
                        eprintln!("[serve] upload from {peer}: cannot create uploads: {error}");
                        plain(500, &format!("cannot create uploads: {error}"))
                    }
                    Ok(()) => {
                        // An upload never overwrites: the same name twice is
                        // two files with `(1)`, `(2)`… rather than one silent
                        // replacement the sender cannot see.
                        let name = free_upload_name(&directory, &name).await;
                        let file_path = directory.join(&name);
                        match tokio::fs::write(&file_path, &request.body).await {
                            Err(error) => {
                                eprintln!("[serve] upload from {peer}: cannot write: {error}");
                                plain(500, &format!("cannot write upload: {error}"))
                            }
                            Ok(()) => {
                                eprintln!(
                                    "[serve] uploaded {} bytes → {}",
                                    request.body.len(),
                                    file_path.display()
                                );
                                let payload = json!({
                                    "ok": true,
                                    "path": file_path.display().to_string(),
                                    "name": name,
                                    "size": request.body.len(),
                                });
                                (200, "application/json", payload.to_string().into_bytes())
                            }
                        }
                    }
                }
            }
        }
        // Files out: the page asks for a path under the session's own working
        // directory, and what it may read is decided by the path rules below
        // rather than by who is asking. The download cap stays — that one keeps
        // a single request from occupying the link, which is capacity, not
        // permission.
        Route::File => {
            let Some(raw_path) = query.get("path") else {
                respond(&mut stream, 400, "text/plain; charset=utf-8", b"missing path").await;
                return;
            };
            // Already percent-decoded by `split_target`, once, for every
            // door into the query string.
            let decoded: &str = raw_path;
            let Some(workdir) = session_workdir_in(&state) else {
                // Same rule as `/upload`: no cwd, no root. The old fallback
                // to the process directory quietly widened what any caller
                // could read to wherever the shell happened to be started —
                // and now that every caller is served, an invented root would
                // be worse than the same refusal it always was.
                let reason = "this session has no working directory yet: open or create a \
                              session first, then download";
                eprintln!("[serve] {peer} GET /file: no session workdir");
                respond(&mut stream, 409, "text/plain; charset=utf-8", reason.as_bytes()).await;
                return;
            };
            let candidate = if std::path::Path::new(decoded).is_absolute() {
                // An absolute path is what the page holds for an upload it
                // just made, so it stays allowed; the canonicalized check
                // below is what decides whether it may be read.
                std::path::PathBuf::from(decoded)
            } else {
                // A relative path is joined to the session root, and that
                // join is only safe for a descent through plain names: this
                // is the same rule the interface files get, so a `..`, a
                // drive letter, a device name or a `file:stream` alternate
                // data stream is refused before it becomes a path to open.
                let Some(relative) = safe_relative(decoded) else {
                    respond(&mut stream, 404, "text/plain; charset=utf-8", b"not found").await;
                    return;
                };
                workdir.join(relative)
            };
            // Canonicalize both ends: a symlink that points outside the
            // working directory fails the same check as `../`, and the
            // settings file with the API key inside stays unreachable.
            let resolved = tokio::fs::canonicalize(&candidate).await;
            let root = tokio::fs::canonicalize(&workdir).await;
            match (resolved, root) {
                (Err(_), _) => {
                    respond(&mut stream, 404, "text/plain; charset=utf-8", b"not found").await;
                    return;
                }
                (Ok(path), Ok(root)) if path.starts_with(&root) && path.is_file() => {
                    let mut file = match tokio::fs::File::open(&path).await {
                        Ok(file) => file,
                        Err(error) => {
                            respond(
                                &mut stream,
                                500,
                                "text/plain; charset=utf-8",
                                format!("cannot read: {error}").as_bytes(),
                            )
                            .await;
                            return;
                        }
                    };
                    let length = match file.metadata().await {
                        Ok(metadata) => metadata.len(),
                        Err(error) => {
                            respond(
                                &mut stream,
                                500,
                                "text/plain; charset=utf-8",
                                format!("cannot read: {error}").as_bytes(),
                            )
                            .await;
                            return;
                        }
                    };
                    // Size is asked before a byte is sent: a body that is
                    // about to be refused should not first occupy the link.
                    if over_file_limit(length) {
                        let body = format!(
                            "file too large: {length} bytes, this server serves up to \
                             {} MB per download",
                            MAX_FILE_BYTES / (1024 * 1024)
                        );
                        eprintln!("[serve] download {} refused: {body}", path.display());
                        respond(&mut stream, 413, "text/plain; charset=utf-8", body.as_bytes())
                            .await;
                        return;
                    }
                    let filename = path
                        .file_name()
                        .map(|name| name.to_string_lossy().into_owned())
                        .unwrap_or_else(|| "download".into());
                    eprintln!("[serve] download {} ({length} bytes)", path.display());
                    respond_attachment(&mut stream, &filename, &mut file, length).await;
                    return;
                }
                (Ok(_), Ok(_)) => {
                    respond(
                        &mut stream,
                        403,
                        "text/plain; charset=utf-8",
                        b"outside the working directory",
                    )
                    .await;
                    return;
                }
                (Ok(_), Err(_)) => {
                    respond(&mut stream, 500, "text/plain; charset=utf-8", b"workdir vanished").await;
                    return;
                }
            }
        }
        // Health for the upstream the shell cannot afford to guess at: the
        // page asks, this answers by opening the same door the LLM client
        // would — a CONNECT handshake through the configured proxy. 200 from
        // the handshake means the node completed the upstream leg; anything
        // else (5xx from the proxy, silence) means the tunnel is down.
        Route::Health => {
            let started = std::time::Instant::now();
            let ok = probe_upstream().await;
            if !ok {
                // A failed check is the moment to refresh the evidence
                // behind the proxy's own pick: re-measuring every node
                // now gives the URLTest group fresh numbers to evict the
                // one that just dropped us, instead of waiting out its
                // interval while turns keep failing.
                kick_node_probes().await;
            }
            let payload = json!({ "ok": ok, "ms": started.elapsed().as_millis() });
            (200, "application/json", payload.to_string().into_bytes())
        }
        // The interface's own files: the page, and the door the printed URL
        // opens. Answered from the path alone — `ui_response` takes no peer, no
        // header and no credential — because a page its own user cannot open is
        // not a guarded page, it is a broken one, and that was the whole
        // complaint. `safe_relative` is still the router: a path that would
        // climb out of the interface directory answers 404, exactly as a
        // missing file does.
        Route::Ui => ui_response(path),
        // Route existence is not a secret any more either: the door that is not
        // there says 405 or 404 in the open, so a page that fetches the wrong
        // thing can be debugged from the browser it is running in.
        Route::MethodNotAllowed => plain(405, "method not allowed"),
        Route::NotFound => plain(404, "not found"),
    };

    respond(&mut stream, status, content_type, &body).await;
}

/// The served shell's doors, one per arm of [`handle`].
///
/// Nothing here is a permission: the enum names *what* to answer, and the method
/// and the path are the only inputs that pick it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Route {
    /// `GET /events`: the page's event stream.
    Events,
    /// `POST /cmd`: the page's own `dispatch`.
    Command,
    /// `POST /upload`: a file the page hands to the shell.
    Upload,
    /// `GET /file?path=…`: a file the shell hands back.
    File,
    /// `GET /health`: whether the upstream the shell needs is reachable.
    Health,
    /// `GET /` and every other path: the interface's own files.
    Ui,
    /// A path this server has, asked for with the wrong verb.
    MethodNotAllowed,
    /// No such path — or no such door for this verb.
    NotFound,
}

/// Which door a request goes to, from its method and path alone.
///
/// Pure and total on purpose: this table *is* the surface of the served shell,
/// and it takes nothing a credential could arrive in. Order is the routing rule
/// — the named pairs come first, because `GET /cmd` is a wrong verb (405) rather
/// than an interface file, and `POST` to anything unnamed is the same mistake.
fn route(method: &str, path: &str) -> Route {
    match (method, path) {
        ("GET", "events") => Route::Events,
        ("POST", "cmd") => Route::Command,
        ("POST", "upload") => Route::Upload,
        ("GET", "file") => Route::File,
        ("GET", "health") => Route::Health,
        // The wrong verb on a door that exists, and every other `POST`: one
        // answer, because the mistake is the same one.
        ("GET", "cmd") | ("POST", _) => Route::MethodNotAllowed,
        // Any other read-shaped request is the interface, paths and all.
        ("GET", _) => Route::Ui,
        _ => Route::NotFound,
    }
}

/// What a `GET` for an interface file answers, from the path alone.
///
/// The signature is the statement the owner asked for: no peer address, no
/// header, no credential — a caller with the URL gets the document. The one rule
/// kept from the old gate is the router. `GET /` is the URL this server prints
/// at startup and carries no path at all, so it is the entry document rather
/// than a path to refuse; and a path that would climb out of the interface
/// directory answers 404, exactly as a file that is not there does.
fn ui_response(path: &str) -> (u16, &'static str, Vec<u8>) {
    let entry = if path.is_empty() { "index.html" } else { path };
    if safe_relative(entry).is_none() {
        // A plain 404 with no reason in it: a path this server will not read
        // should tell a prober no more than a missing file does.
        plain(404, "not found")
    } else {
        let (bytes, content_type) = serve_ui(entry);
        (200, content_type, bytes)
    }
}

fn plain(status: u16, message: &str) -> (u16, &'static str, Vec<u8>) {
    (
        status,
        "text/plain; charset=utf-8",
        message.as_bytes().to_vec(),
    )
}

/// Longest upload name kept, in characters.
///
/// Characters rather than bytes so a name does not lose a third of its length
/// on the way in: Windows counts UTF-16 units, where 200 characters is inside
/// the 255 it accepts. A 200-character CJK name is 600 bytes, so on a
/// byte-limited file system the write itself still fails — with an error the
/// upload answers `500` for, rather than a name silently cut mid-character.
const MAX_NAME_CHARS: usize = 200;

/// How many duplicates of one name are tried before the suffix stops growing.
const MAX_NAME_SUFFIX: u32 = 999;

/// An upload's name, stripped of anything that could make it a path: the
/// file lands in `uploads/` and nowhere else, whatever the client sent.
///
/// `file_name()` is what defeats `../` and absolute paths; the rest is about
/// names that stay inside `uploads/` and still are not what they look like.
/// Control and format characters (U+202E reverses the rendered extension) are
/// dropped even though the character whitelist below would reject them anyway:
/// the whitelist is the rule most likely to be widened next, and this keeps
/// that change from also being a disguise change.
pub(crate) fn sanitize_name(name: &str) -> String {
    let base = std::path::Path::new(name)
        .file_name()
        .map(|part| part.to_string_lossy().into_owned())
        .unwrap_or_default();
    let cleaned: String = base
        .chars()
        .filter(|c| !c.is_control() && !is_format_char(*c))
        .filter(|c| c.is_alphanumeric() || matches!(c, '-' | '_' | '.' | ' '))
        .collect();
    let shortened: String = cleaned.chars().take(MAX_NAME_CHARS).collect();
    let trimmed = shortened.trim().trim_matches(['.', ' ']);
    if trimmed.is_empty() {
        return "upload.bin".to_string();
    }
    // A Windows device name is not a file there, whatever its extension: an
    // upload called `CON` would open the console instead. One leading
    // underscore makes it an ordinary file and keeps the name recognisable.
    if is_windows_device_name(trimmed) {
        return format!("_{trimmed}");
    }
    trimmed.to_string()
}

/// Whether a segment names a Windows device rather than a file.
///
/// Case does not matter (`con` opens the console too), neither does the
/// extension (`CON.txt` is `CON`), and Windows drops a trailing space before it
/// compares (`CON .txt` is still the console), so the stem before the first dot
/// is trimmed and uppercased before it is checked. Extension-proof on purpose:
/// a `NUL.md` that opens like a file and swallows writes is worse than a
/// refused path.
fn is_windows_device_name(name: &str) -> bool {
    let stem = name
        .split('.')
        .next()
        .unwrap_or(name)
        .trim_end_matches(' ')
        .to_ascii_uppercase();
    if matches!(
        stem.as_str(),
        "CON" | "PRN" | "AUX" | "NUL" | "CONIN$" | "CONOUT$"
    ) {
        return true;
    }
    // COM1-9 and LPT1-9; COM0 and LPT0 are ordinary names, and the
    // superscript spellings are not something a browser upload sends.
    let bytes = stem.as_bytes();
    bytes.len() == 4
        && (stem.starts_with("COM") || stem.starts_with("LPT"))
        && bytes[3].is_ascii_digit()
        && bytes[3] != b'0'
}

/// Unicode `Cf` (format) characters: they render as nothing and can reorder
/// the text around them.
///
/// The ranges are spelled out rather than taken from a Unicode table crate:
/// this crate depends on none, and the ones listed here are the ones that
/// matter for a file name a human reads — the bidi controls, the joiners, the
/// byte-order mark, and the Arabic letter-mark family.
fn is_format_char(c: char) -> bool {
    matches!(
        c as u32,
        0x00AD
            | 0x0600..=0x0605
            | 0x061C
            | 0x06DD
            | 0x070F
            | 0x0890..=0x0891
            | 0x08E2
            | 0x180E
            | 0x200B..=0x200F
            | 0x202A..=0x202E
            | 0x2060..=0x2064
            | 0x2066..=0x206F
            | 0xFEFF
            | 0xFFF9..=0xFFFB
            | 0x110BD
            | 0x110CD
            | 0x13430..=0x1343F
            | 0x1BCA0..=0x1BCA3
            | 0x1D173..=0x1D17A
            | 0xE0001
            | 0xE0020..=0xE007F
    )
}

/// `photo.png` and 3 as `photo(3).png`.
///
/// The extension stays last so both the browser's download and the file
/// system's own type detection still see it; a name with no extension just
/// gains the suffix.
fn suffixed(name: &str, n: u32) -> String {
    match name.rsplit_once('.') {
        Some((stem, extension)) if !stem.is_empty() && !extension.is_empty() => {
            format!("{stem}({n}).{extension}")
        }
        _ => format!("{name}({n})"),
    }
}

/// The first name not already taken, `taken` saying what is.
///
/// Split out from the directory read so the rule is testable on its own: the
/// caller snapshots `uploads/` and this decides what to do about it.
fn free_name(mut taken: impl FnMut(&str) -> bool, name: &str) -> String {
    if !taken(name) {
        return name.to_string();
    }
    for n in 1..=MAX_NAME_SUFFIX {
        let candidate = suffixed(name, n);
        if !taken(&candidate) {
            return candidate;
        }
    }
    // A thousand uploads under one name is not a naming problem any more;
    // keeping the name and letting the write replace the last candidate is
    // better than failing an upload the user asked for.
    suffixed(name, MAX_NAME_SUFFIX + 1)
}

/// [`free_name`] against the live contents of `directory`.
async fn free_upload_name(directory: &std::path::Path, name: &str) -> String {
    let mut taken: HashSet<String> = HashSet::new();
    if let Ok(mut entries) = tokio::fs::read_dir(directory).await {
        while let Ok(Some(entry)) = entries.next_entry().await {
            taken.insert(entry.file_name().to_string_lossy().into_owned());
        }
    }
    // Two uploads racing for one name can still collide between this read and
    // the write; the cost is one replaced upload, not an escaped path.
    free_name(|candidate| taken.contains(candidate), name)
}

/// `(host, port)` out of a URL, with `default_port` when the URL does not
/// spell one: `https://api.b.ai/v1` -> ("api.b.ai", 443).
fn host_port(url: &str, default_port: u16) -> Option<(String, u16)> {
    let rest = url.split_once("://").map(|(_, tail)| tail).unwrap_or(url);
    let authority = rest.split(['/', '?', '#']).next()?;
    if authority.is_empty() {
        return None;
    }
    match authority.rsplit_once(':') {
        Some((host, port)) if !host.is_empty() && port.chars().all(|c| c.is_ascii_digit()) => {
            Some((host.to_string(), port.parse().ok()?))
        }
        _ => Some((authority.to_string(), default_port)),
    }
}

/// Last time the node probes were kicked, unix seconds — a dead link must
/// not re-kick on every 5-second retry tick.
static LAST_KICK: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(0);

/// One plain HTTP GET against the local Clash-meta control port.
async fn api_get(path: &str) -> Option<String> {
    let mut stream = tokio::net::TcpStream::connect("127.0.0.1:9090").await.ok()?;
    let request = format!(
        "GET {path} HTTP/1.1\r\nHost: 127.0.0.1:9090\r\nConnection: close\r\n\r\n"
    );
    stream.write_all(request.as_bytes()).await.ok()?;
    let mut response = Vec::new();
    tokio::time::timeout(std::time::Duration::from_secs(8), stream.read_to_end(&mut response))
        .await
        .ok()?
        .ok()?;
    let text = String::from_utf8_lossy(&response);
    text.split("\r\n\r\n").nth(1).map(str::to_string)
}

/// Re-measure every node of the local proxy's `PROXY` group after a failed
/// health check. The group is a URLTest: fresh delay numbers are exactly
/// what it ranks members by, so this is the nudge that moves traffic off a
/// dropping node now rather than at the end of its test interval. Entirely
/// best-effort — without a local controller this is a no-op.
async fn kick_node_probes() {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let last = LAST_KICK.load(Ordering::Relaxed);
    if now - last < 30 {
        return;
    }
    if LAST_KICK
        .compare_exchange(last, now, Ordering::Relaxed, Ordering::Relaxed)
        .is_err()
    {
        return;
    }
    let Some(body) = api_get("/proxies").await else {
        // Same outage that failed the probe usually took the controller with
        // it (they are one process); the line says the kick ran and found
        // nothing to kick rather than leaving silence to guess at.
        eprintln!("[health] kick: no local controller on 127.0.0.1:9090");
        return;
    };
    let Ok(document) = serde_json::from_str::<Value>(&body) else {
        return;
    };
    let Some(members) = document
        .get("proxies")
        .and_then(|proxies| proxies.get("PROXY"))
        .and_then(|group| group.get("all"))
        .and_then(Value::as_array)
    else {
        return;
    };
    let targets: Vec<String> = members
        .iter()
        .filter_map(Value::as_str)
        .map(str::to_string)
        .collect();
    if targets.is_empty() {
        return;
    }
    eprintln!(
        "[health] link down; refreshing delay probes for {} nodes",
        targets.len()
    );
    let test_url = percent_encode_utf8("http://www.gstatic.com/generate_204");
    for name in targets {
        let path = format!(
            "/proxies/{}/delay?timeout=5000&url={test_url}",
            percent_encode_utf8(&name)
        );
        tokio::spawn(async move {
            let _ = api_get(&path).await;
        });
    }
}

/// Ask the upstream a question only a working tunnel can answer: open the
/// configured proxy, `CONNECT` the LLM host, read the verdict. No TLS is
/// spoken — the handshake reply *is* the answer, and it crosses the same
/// leg as every chat request would. Any 2xx from the proxy means the node
/// completed the upstream connection; 5xx, silence, or a refused socket
/// means the tunnel is down.
async fn probe_upstream() -> bool {
    let Ok(settings) = nguruvilu::settings::Settings::load() else {
        return false;
    };
    let Some((host, port)) = host_port(&settings.base_url, 443) else {
        return false;
    };
    let target = format!("{host}:{port}");
    let probe = async {
        if settings.proxy.trim().is_empty() {
            // Direct mode: reachability of the host is the whole truth.
            return tokio::net::TcpStream::connect(target.as_str()).await.is_ok();
        }
        let Some((proxy_host, proxy_port)) = host_port(settings.proxy.trim(), 80) else {
            return false;
        };
        let mut stream = match tokio::net::TcpStream::connect((proxy_host.as_str(), proxy_port))
            .await
        {
            Ok(stream) => stream,
            Err(_) => return false,
        };
        let handshake = format!("CONNECT {target} HTTP/1.1\r\nHost: {target}\r\n\r\n");
        if stream.write_all(handshake.as_bytes()).await.is_err() {
            return false;
        }
        let mut reply = [0u8; 256];
        let read = match tokio::time::timeout(std::time::Duration::from_secs(6), stream.read(&mut reply))
            .await
        {
            Ok(Ok(read)) => read,
            _ => return false,
        };
        let line = String::from_utf8_lossy(&reply[..read]);
        line.starts_with("HTTP/") && line.contains(" 200")
    };
    tokio::time::timeout(std::time::Duration::from_secs(7), probe)
        .await
        .unwrap_or(false)
}

/// `%XX` in a query value back to text, for file paths with spaces or CJK.
pub(crate) fn percent_decode(value: &str) -> String {
    let bytes = value.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'%' && index + 2 < bytes.len() {
            let hex = std::str::from_utf8(&bytes[index + 1..index + 3]).ok();
            if let Some(byte) = hex.and_then(|h| u8::from_str_radix(h, 16).ok()) {
                out.push(byte);
                index += 3;
                continue;
            }
        }
        out.push(bytes[index]);
        index += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Percent-encode for the `filename*` half of Content-Disposition.
fn percent_encode_utf8(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for byte in value.as_bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(*byte as char)
            }
            _ => out.push_str(&format!("%{byte:02X}")),
        }
    }
    out
}

/// A download response: octet-stream so the browser saves rather than
/// renders, with the name in both spellings — ASCII for old clients, UTF-8
/// for the CJK names humans actually use.
///
/// The body is copied out chunk by chunk rather than handed over as one slice:
/// a 64 MiB read used to be 64 MiB of resident memory per parallel download,
/// and the length is known up front so `Content-Length` stays exact.
async fn respond_attachment(
    stream: &mut TcpStream,
    filename: &str,
    file: &mut tokio::fs::File,
    length: u64,
) {
    let ascii: String = filename
        .chars()
        .map(|c| if c.is_ascii_graphic() || c == ' ' { c } else { '_' })
        .collect();
    let head = format!(
        "HTTP/1.1 200 OK\r\n\
         Content-Type: application/octet-stream\r\n\
         Content-Disposition: attachment; filename=\"{}\"; filename*=UTF-8''{}\r\n\
         Content-Length: {length}\r\n\
         Connection: close\r\n\r\n",
        ascii.replace('"', "'"),
        percent_encode_utf8(filename)
    );
    if stream.write_all(head.as_bytes()).await.is_ok() {
        // A read error mid-copy cannot be reported any more — the head is
        // already on the wire — so the connection closing short of the
        // declared length is the signal, which is what HTTP gives us.
        let _ = tokio::io::copy(file, stream).await;
    }
}

/// Whether a download is over [`MAX_FILE_BYTES`].
///
/// A named function rather than an inline comparison so the boundary — one
/// byte over the limit, not one byte under — is testable without a socket.
fn over_file_limit(length: u64) -> bool {
    length > MAX_FILE_BYTES
}

/// A parsed HTTP/1.1 request: the two things a response is decided by, plus the
/// body.
///
/// The headers are deliberately not carried any further than [`read_head`]. They
/// used to travel here so `authorized` could read `Authorization` and the
/// interface branch could read `Host`; both checks are gone, so a copy that
/// nothing reads would be the only trace of a gate that no longer exists.
struct Request {
    method: String,
    target: String,
    /// Raw bytes, kept raw because `/upload` carries files that are not UTF-8;
    /// `/cmd` decodes on its own where JSON needs it.
    body: Vec<u8>,
}

/// The largest body this server will read, in bytes.
///
/// A request body is buffered in memory here, so this one number is both the
/// upload limit a user runs into and the answer to "how much RAM may one
/// request ask for". It used to be a silent 64 MB drop — the client saw a
/// connection with no reply and no reason. `NGU_MAX_UPLOAD_MB` overrides it
/// per deployment; failures now answer `413` with the limit spelled out.
pub(crate) fn max_body_bytes() -> u64 {
    let megabytes = std::env::var("NGU_MAX_UPLOAD_MB")
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(512);
    megabytes.saturating_mul(1024 * 1024)
}

/// What reading one request produced.
enum ReadOutcome {
    /// A complete request.
    Request(Request),
    /// The declared `Content-Length` exceeds [`max_body_bytes`].
    ///
    /// The body is deliberately *not* read: the reply explains the limit and
    /// the connection closes, rather than the server swallowing a payload it
    /// is about to refuse anyway.
    TooLarge(u64),
    /// Refused with a reason the client can act on: a body this server cannot
    /// frame, or one that arrived too slowly.
    Refused { status: u16, reason: String },
    /// Malformed, truncated, or gone — dropped without a reply; an
    /// unsolicited connection owes nobody an explanation.
    Malformed,
}

/// The head of one request: the request line and the headers, lowercased by
/// name. The body is read separately, from the length this says.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Head {
    method: String,
    target: String,
    headers: HashMap<String, String>,
    /// `Content-Length`, when the header is present and a byte count.
    content_length: Option<u64>,
}

/// Why a head cannot be served, and what to tell the client.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HeadError {
    /// Not a request line and headers this server reads.
    Malformed,
    /// A body framed by `Transfer-Encoding`, which this server does not read.
    TransferEncoding,
    /// A `POST` with no `Content-Length`: there is no other way to know where
    /// the body ends.
    LengthRequired,
    /// `Content-Length` that is not a number.
    BadLength,
}

impl HeadError {
    /// The status this refusal answers with.
    fn status(self) -> u16 {
        match self {
            // 501, not 400: the request is well formed and this server simply
            // does not implement that framing. A client can tell the two apart
            // and retry with the one that works.
            Self::TransferEncoding => 501,
            Self::Malformed | Self::LengthRequired | Self::BadLength => 400,
        }
    }

    /// Why, in words the client's author can act on.
    fn reason(self) -> &'static str {
        match self {
            Self::Malformed => "malformed request head",
            Self::TransferEncoding => {
                "Transfer-Encoding is not supported: send the body with Content-Length, \
                 which is the only framing this server reads"
            }
            Self::LengthRequired => {
                "missing Content-Length: this server takes the body length from that header \
                 alone, and reads no body without it"
            }
            Self::BadLength => "Content-Length is not a byte count",
        }
    }
}

impl Head {
    /// The body framing this head declares.
    ///
    /// `Transfer-Encoding` is checked first and refused outright, even when a
    /// `Content-Length` is also present: reading the declared length and
    /// ignoring the transfer coding is how a request is smuggled past a proxy.
    fn framing(&self) -> Result<u64, HeadError> {
        if self.headers.contains_key("transfer-encoding") {
            return Err(HeadError::TransferEncoding);
        }
        match self.content_length {
            Some(length) => Ok(length),
            // A GET with no body is the normal case, so no header means no
            // body. A POST with no header is ambiguous, and guessing zero is
            // how a body silently goes missing.
            None if self.method.eq_ignore_ascii_case("POST") => Err(HeadError::LengthRequired),
            None => Ok(0),
        }
    }
}

/// Parse a request head out of the bytes read before the blank line.
///
/// Pure on purpose: every rejection path that a client can trigger is decided
/// here, which is what makes them testable without a socket.
fn read_head(bytes: &[u8]) -> Result<Head, HeadError> {
    let text = std::str::from_utf8(bytes).map_err(|_| HeadError::Malformed)?;
    let mut lines = text.split("\r\n");
    let mut request_line = lines.next().ok_or(HeadError::Malformed)?.split_whitespace();
    let method = request_line.next().unwrap_or("").to_string();
    let target = request_line.next().unwrap_or("").to_string();
    if method.is_empty() || target.is_empty() {
        return Err(HeadError::Malformed);
    }

    let mut headers = HashMap::new();
    for line in lines {
        if let Some((name, value)) = line.split_once(':') {
            headers.insert(name.trim().to_ascii_lowercase(), value.trim().to_string());
        }
    }

    // A `Content-Length` that is present but unreadable is an error rather
    // than "no body": the client meant to send something, and reading zero
    // would answer a different request than the one it made.
    let content_length = match headers.get("content-length") {
        None => None,
        Some(value) => Some(value.parse::<u64>().map_err(|_| HeadError::BadLength)?),
    };

    Ok(Head {
        method,
        target,
        headers,
        content_length,
    })
}

/// Read one request off the socket: head until `\r\n\r\n`, then exactly
/// `Content-Length` bytes of body.
///
/// Every read is on [`READ_TIMEOUT`]: the socket is the client's, and a client
/// that sends one byte a minute would otherwise hold a connection slot for as
/// long as it liked.
async fn read_request(stream: &mut TcpStream) -> ReadOutcome {
    let mut buffer = Vec::with_capacity(4096);
    let mut chunk = [0u8; 8192];

    let head_end = loop {
        if let Some(position) = find_subslice(&buffer, b"\r\n\r\n") {
            break position;
        }
        if buffer.len() > 64 * 1024 {
            return ReadOutcome::Malformed;
        }
        let read = match tokio::time::timeout(READ_TIMEOUT, stream.read(&mut chunk)).await {
            Ok(Ok(read)) if read > 0 => read,
            // Closed, or closed short of the blank line: nothing to answer.
            Ok(_) => return ReadOutcome::Malformed,
            Err(_) => {
                return ReadOutcome::Refused {
                    status: 408,
                    reason: format!(
                        "timed out after {} s while reading the request head",
                        READ_TIMEOUT.as_secs()
                    ),
                }
            }
        };
        buffer.extend_from_slice(&chunk[..read]);
    };

    let head = match read_head(&buffer[..head_end]) {
        Ok(head) => head,
        Err(error) => {
            return ReadOutcome::Refused {
                status: error.status(),
                reason: error.reason().to_string(),
            }
        }
    };
    let content_length = match head.framing() {
        Ok(length) => length,
        Err(error) => {
            return ReadOutcome::Refused {
                status: error.status(),
                reason: error.reason().to_string(),
            }
        }
    };
    if content_length > max_body_bytes() {
        return ReadOutcome::TooLarge(content_length);
    }
    // A length that does not fit in a `usize` cannot be a buffer here either;
    // the limit above already covers every default, this covers a deployment
    // that raised it past the address space.
    let Ok(content_length) = usize::try_from(content_length) else {
        return ReadOutcome::TooLarge(content_length);
    };

    let mut body_bytes = buffer[head_end + 4..].to_vec();
    while body_bytes.len() < content_length {
        let read = match tokio::time::timeout(READ_TIMEOUT, stream.read(&mut chunk)).await {
            Ok(Ok(read)) if read > 0 => read,
            Ok(_) => return ReadOutcome::Malformed,
            Err(_) => {
                return ReadOutcome::Refused {
                    status: 408,
                    reason: format!(
                        "timed out after {} s while reading the {} bytes of body this \
                         request declared",
                        READ_TIMEOUT.as_secs(),
                        content_length
                    ),
                }
            }
        };
        body_bytes.extend_from_slice(&chunk[..read]);
    }
    body_bytes.truncate(content_length);

    ReadOutcome::Request(Request {
        method: head.method,
        target: head.target,
        body: body_bytes,
    })
}

/// Write one response and close. `Connection: close` is the whole policy:
/// keep-alive would mean tracking half-closed clients for no gain here.
///
/// `401` is deliberately absent from the reasons below: no door in this file
/// asks a caller for a credential any more, so a status this server can never
/// answer with would only invite the gate back.
async fn respond(stream: &mut TcpStream, status: u16, content_type: &str, body: &[u8]) {
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        408 => "Request Timeout",
        409 => "Conflict",
        413 => "Payload Too Large",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        501 => "Not Implemented",
        503 => "Service Unavailable",
        _ => "",
    };
    let head = format!(
        "HTTP/1.1 {status} {reason}\r\n\
         Content-Type: {content_type}\r\n\
         Content-Length: {}\r\n\
         Cache-Control: no-store\r\n\
         Connection: close\r\n\r\n",
        body.len()
    );
    if stream.write_all(head.as_bytes()).await.is_ok() {
        let _ = stream.write_all(body).await;
    }
}

/// Hand the socket to the event stream: headers once, then every event as it
/// comes, with a heartbeat so an idle proxy cannot decide the stream is dead.
///
/// Every event carries a monotonic `id:`, and a reader that fell behind the bus
/// is told so with an `event: resync` frame. It used to be `Lagged(…) => {}`:
/// the events were dropped and the page was left rendering a stale transcript
/// with no hint that it had missed anything, so the only recovery was a manual
/// refresh.
async fn sse(stream: &mut TcpStream, bus: broadcast::Sender<String>) {
    let head = "HTTP/1.1 200 OK\r\n\
                Content-Type: text/event-stream\r\n\
                Cache-Control: no-cache\r\n\
                Connection: keep-alive\r\n\r\n";
    if stream.write_all(head.as_bytes()).await.is_err() {
        return;
    }

    let mut events = bus.subscribe();
    let mut heartbeat = tokio::time::interval(std::time::Duration::from_secs(15));
    // Numbered per stream, from 1: the counter exists so a listener can tell
    // events apart and spot a gap, which is all a `Last-Event-ID` would be used
    // for here. It is not a position in the bus — the bus keeps no history.
    let mut id: u64 = 0;
    loop {
        tokio::select! {
            event = events.recv() => match event {
                Ok(data) => {
                    id += 1;
                    if stream.write_all(sse_event(id, &data).as_bytes()).await.is_err() {
                        break;
                    }
                }
                // The connection survives, the backlog does not. Saying so is
                // the difference between a page that reloads its state and one
                // that shows the user the past.
                Err(broadcast::error::RecvError::Lagged(missed)) => {
                    id += 1;
                    eprintln!(
                        "[serve] event stream fell {missed} events behind: asking the page to resync"
                    );
                    if stream.write_all(sse_resync(id).as_bytes()).await.is_err() {
                        break;
                    }
                }
                Err(broadcast::error::RecvError::Closed) => break,
            },
            _ = heartbeat.tick() => {
                if stream.write_all(b": ping\n\n").await.is_err() {
                    break;
                }
            }
        }
    }
}

/// One event frame: its monotonic `id:` and the payload on one `data:` line.
///
/// The payload is a serialized JSON object, which never contains a newline, so
/// one `data:` line is enough and there is no multi-line assembly for the page
/// to get wrong.
fn sse_event(id: u64, data: &str) -> String {
    format!("id: {id}\ndata: {data}\n\n")
}

/// The frame that tells a lagging listener its backlog was dropped.
///
/// `resync` is a request, not a state: the page answers it by re-sending
/// `ready`, which is the one command that re-sends everything durable. That is
/// why this carries no payload beyond its reason — the recovery is the same
/// whether one event or a thousand were missed.
fn sse_resync(id: u64) -> String {
    format!("id: {id}\nevent: resync\ndata: {{\"reason\":\"lagged\"}}\n\n")
}

/// The current session's working directory, or `None` when it has none.
///
/// `None` is a real answer, not a reason to guess: uploads and downloads are
/// rooted here, so inventing a root means serving a directory the session never
/// had. The callers answer `409` and say why. A second function used to sit
/// beside this one and fall back to the process directory; it lost its last
/// caller once both doors learned to refuse, and leaving it in place would have
/// kept the invented root one call away.
pub(crate) fn session_workdir_in(state: &Arc<Mutex<AppState>>) -> Option<std::path::PathBuf> {
    state
        .lock()
        .expect("state lock")
        .session
        .cwd
        .clone()
        .map(std::path::PathBuf::from)
}

/// Whether `id` is a session this machine already has, for the relay's
/// `open_session` gate.
///
/// The listing comes from the store, so a session deleted a moment ago is not
/// a session the cloud may switch to. A store that cannot be read answers
/// `false`: no answer must not become permission.
fn session_exists(state: &Arc<Mutex<AppState>>, id: &str) -> bool {
    if id.trim().is_empty() {
        return false;
    }
    let guard = state.lock().expect("state lock");
    listed_session(&guard, id).unwrap_or(false)
}

/// Whether `id` names a session in the store, with the store's own failure kept
/// apart from "no such session".
///
/// [`session_exists`] folds both into `false`, which is the right answer when
/// the question is permission to move this shell's working directory: no answer
/// must not become permission. Where the question is whether to accept text a
/// user typed, the two are not interchangeable — "no such session" is the
/// caller's mistake and answers `404`, while a store this process cannot read is
/// its own fault and must not be reported as a typo the caller never made.
fn listed_session(state: &AppState, id: &str) -> Result<bool, String> {
    match state.sessions() {
        Ok(Value::Array(sessions)) => Ok(sessions
            .iter()
            .any(|session| session.get("id").and_then(Value::as_str) == Some(id))),
        Ok(_) => Err("the session list is not a list".to_string()),
        Err(error) => Err(format!("{error:#}")),
    }
}

/// Split `path?query` into both halves, query as a map.
///
/// Query values arrive percent-encoded (`?name=my%20file.txt`, and a path with
/// CJK characters is encoded twice over) and are decoded here, once, for every
/// caller. The `/file` handler used to decode on its own, which meant the two
/// doors into the query string disagreed about what a value was.
pub(crate) fn split_target(target: &str) -> (&str, HashMap<String, String>) {
    let (path, query) = target.split_once('?').unwrap_or((target, ""));
    let query = query
        .split('&')
        .filter_map(|pair| pair.split_once('='))
        .map(|(name, value)| (name.to_string(), percent_decode(value)))
        .collect();
    (path.trim_start_matches('/'), query)
}

fn is_ready(body: &str) -> bool {
    serde_json::from_str::<Value>(body)
        .ok()
        .and_then(|command| command.get("cmd").and_then(Value::as_str).map(str::to_string))
        .is_some_and(|cmd| cmd == "ready")
}

/// Whether a command is the read-only usage ledger read.
fn is_stats(body: &str) -> bool {
    serde_json::from_str::<Value>(body)
        .ok()
        .and_then(|command| command.get("cmd").and_then(Value::as_str).map(str::to_string))
        .is_some_and(|cmd| cmd == "stats")
}

/// The `session` a command body may carry, or why the value cannot be read.
///
/// The two doors that accept a target — this one's fast path and the ordered
/// lane's `dispatch` — must agree on what an explicit id means, and a value
/// that is not a string is the one shape neither can honour. Reading it as
/// "absent" would aim the command at the conversation on screen, which is the
/// opposite of what the caller wrote; `null` is the same statement spelled out
/// as absence, so it stays the ordinary case rather than becoming an error.
fn optional_session(command: &Value) -> Result<Option<String>, String> {
    match command.get("session") {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(id)) => {
            let id = id.trim();
            Ok(if id.is_empty() {
                None
            } else {
                Some(id.to_string())
            })
        }
        Some(_) => Err(
            "`session` must be a string session id; omit it to address the session on screen"
                .to_string(),
        ),
    }
}

/// Answer `stats`: the requested session's cumulative usage and its recent
/// turns, wrapped as the event the page listens for.
///
/// Synchronous by design. The session on screen already holds its usage rows, so
/// that answer needs no I/O at all; a request for a session that is not the open
/// one has to read the file, and the read happens after the state lock has been
/// released — a stranger's report must not hold up every other session's events.
/// Failures come back as an `error` field on the same event rather than as a
/// separate event: a page asking for one number should not have to watch two.
fn stats_event(state: &Arc<Mutex<AppState>>, body: &str) -> Value {
    let requested = serde_json::from_str::<Value>(body)
        .ok()
        .and_then(|command| {
            command
                .get("session")
                .and_then(Value::as_str)
                .map(str::to_string)
        });
    let mut event = match AppState::stats_outside_lock(state, requested.as_deref()) {
        Ok(payload) => payload,
        Err(error) => json!({ "error": format!("{error:#}") }),
    };
    if let Value::Object(fields) = &mut event {
        fields.insert("ev".to_string(), json!("stats"));
    }
    event
}

/// Commands that may run on their own task beside the ordered lane: they read
/// state, or write only the payload this page is showing.
///
/// Deliberately short, and deliberately not "the cheap ones": `status` and
/// `load_older` read, `list_packs`/`list_routes`/`fetch_models` list, `ui_panels`
/// and `stats` say what the session already holds. None of them reaches the
/// kernel, and a page must be able to ask "how much has this cost so far" while
/// a turn is still running, which is exactly when the question is interesting.
const READ_ONLY_COMMANDS: [&str; 8] = [
    "status",
    "list_packs",
    "list_routes",
    "fetch_models",
    "ui_panels",
    "load_older",
    "stats",
    "doctor",
];

/// `prompt` and `cancel` start and stop work rather than read it, and they are
/// the exception that has to be on the concurrent lane: while a turn is running
/// they are the only controls the page has, so queueing them behind that turn
/// locks the user out of their own agent. Both are safe there because neither
/// touches the kernel — `prompt` claims the per-session `running` slot atomically
/// (a second prompt steers instead of racing) and `cancel` flips a per-session
/// flag the turn checks at its next safe point.
const STEERING_COMMANDS: [&str; 2] = ["prompt", "cancel"];

/// Commands that rewrite what a running turn reads, and therefore stay in the
/// ordered lane.
///
/// `session_switch` is the sharp one: it re-pins the process-wide tool table, so
/// a session command running next to a live turn would move that turn's bash
/// cwd, its relative paths and its job ownership. Settings, model, routes and
/// packs swap the runtime, the policy or the tool set under the turn's feet.
/// Named here so a test can hold them against the concurrent list instead of
/// trusting that nobody added one back.
///
/// Together with [`READ_ONLY_COMMANDS`] and [`STEERING_COMMANDS`] this names
/// every command `dispatch` has an arm for, apart from the empty and unknown
/// fallbacks, plus `stats`, which this module answers itself. No command should
/// reach its lane by being forgotten: the default is the ordered lane, and a
/// default is not a classification a reviewer can check.
const KERNEL_WRITE_COMMANDS: [&str; 17] = [
    "new_session",
    "open_session",
    "delete_session",
    "save_settings",
    "set_model",
    "save_route",
    "switch_route",
    "delete_route",
    "apply_pack",
    "load_pack",
    "uninstall_pack",
    "delete_pack",
    "unload_plugin",
    // `ready` loads every installed pack into the kernel and may run the
    // command line's startup prompt as a real turn; the pickers build, verify
    // and install a pack from a path the page uploaded. All four reached the
    // ordered lane only because an unlisted command defaults to it, and one
    // edit to that default would have put a pack install beside a live turn.
    "ready",
    "pick_install",
    "pick_verify",
    "pick_pack",
];

/// Whether a command may run on its own task without waiting for the ordered
/// lane — the judgement behind the whole scheduling split.
///
/// The rule is what a command may write, not what it costs: read-only (or
/// page-payload-only) work may run beside a turn; anything that rewrites the
/// kernel, the settings or the session stays ordered. White-listed on purpose —
/// a command nobody classified, including one added later, defaults to the
/// ordered lane, where the worst it can do is run after its neighbours.
fn is_concurrent(body: &str) -> bool {
    serde_json::from_str::<Value>(body)
        .ok()
        .and_then(|command| command.get("cmd").and_then(Value::as_str).map(str::to_string))
        .is_some_and(|cmd| {
            let cmd = cmd.as_str();
            // Names that rewrite the kernel are refused here rather than by
            // being left off a whitelist: this list is the classification, so a
            // later edit that adds one of them to a read-only list is still a
            // mistake the scheduler will not act on.
            if KERNEL_WRITE_COMMANDS.contains(&cmd) {
                return false;
            }
            READ_ONLY_COMMANDS.contains(&cmd) || STEERING_COMMANDS.contains(&cmd)
        })
}

/// An address that only the machine itself can reach.
fn is_loopback(addr: &str) -> bool {
    addr.starts_with("127.")
        || addr.starts_with("localhost")
        || addr.starts_with("[::1]")
        || addr.starts_with("::1")
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

#[cfg(test)]
mod scheduling_tests {
    use super::{
        is_concurrent, is_stats, KERNEL_WRITE_COMMANDS, READ_ONLY_COMMANDS, STEERING_COMMANDS,
    };

    /// The ledger read must never enter the ordered lane: queued behind pack
    /// work it answers late, and a usage line that arrives after the turn is
    /// useless. This is the whitelist entry that keeps it on its own task.
    #[test]
    fn stats_runs_on_the_concurrent_lane() {
        assert!(is_concurrent(r#"{"cmd":"stats"}"#));
        assert!(is_concurrent(r#"{"cmd":"stats","session":"20260101-000000-abc123"}"#));
    }

    #[test]
    fn stats_is_recognised_with_or_without_a_session_argument() {
        assert!(is_stats(r#"{"cmd":"stats"}"#));
        assert!(is_stats(r#"{"cmd":"stats","session":"20260101-000000-abc123"}"#));
        assert!(!is_stats(r#"{"cmd":"status"}"#));
        assert!(!is_stats("not json"));
    }

    /// The lane split still puts the mutating commands where they were: pack
    /// work and `ready` stay ordered.
    #[test]
    fn pack_work_and_ready_stay_on_the_ordered_lane() {
        assert!(!is_concurrent(r#"{"cmd":"ready"}"#));
        assert!(!is_concurrent(r#"{"cmd":"install_pack","path":"x.dshpack"}"#));
    }

    /// The lane split is one rule, and this is the rule: read-only (or
    /// page-payload-only) commands may run beside a turn, and nothing that
    /// rewrites the kernel, the settings or the session may. The two constant
    /// lists are compared against the whitelist rather than restated name by
    /// name, so an addition in one place cannot be forgotten in the other —
    /// the mistake that put `session_switch`'s callers on the fast lane.
    #[test]
    fn only_read_only_commands_run_concurrently() {
        for cmd in READ_ONLY_COMMANDS.iter().chain(STEERING_COMMANDS.iter()) {
            let body = serde_json::json!({ "cmd": cmd }).to_string();
            assert!(
                is_concurrent(&body),
                "{cmd} is on the concurrent list but the code runs it ordered"
            );
        }
        for cmd in KERNEL_WRITE_COMMANDS {
            let body = serde_json::json!({ "cmd": cmd }).to_string();
            assert!(
                !is_concurrent(&body),
                "{cmd} rewrites what a running turn reads and must stay ordered"
            );
            // …and no name may be claimed by both lists: that is the shape the
            // two-constant comparison exists to catch.
            assert!(
                !READ_ONLY_COMMANDS.contains(&cmd) && !STEERING_COMMANDS.contains(&cmd),
                "{cmd} is listed as both concurrent and kernel-writing"
            );
        }
        // The lists themselves do not drift into each other by accident either.
        for cmd in READ_ONLY_COMMANDS {
            assert!(
                !KERNEL_WRITE_COMMANDS.contains(&cmd),
                "{cmd} cannot be both read-only and a kernel write"
            );
        }
    }
}

/// What replaced the gate's tests.
///
/// The module that stood here asserted the opposite of this one: a request with
/// no token was refused, a foreign `Host:` was refused, a header that was not a
/// `Bearer` credential was refused. The owner deleted that contract on purpose,
/// so what is asserted now is the contract that replaced it — every door answers
/// its method and path whoever knocks, and what is left (`405`, `404`) is
/// routing, not permission.
#[cfg(test)]
mod routing_tests {
    use super::{route, ui_response, Route};

    /// The complaint this contract exists to end, in one test.
    ///
    /// `route` takes a method and a path and nothing else — no header, no query
    /// string, no peer address — so there is no argument a credential could even
    /// arrive in. The entry document the server prints then answers `200` to a
    /// caller that presented nothing at all.
    #[test]
    fn a_request_without_a_token_is_served() {
        // The printed URL: `GET /`, no path, no credential, no `Host:` named.
        assert_eq!(route("GET", ""), Route::Ui);
        let (status, content_type, _) = ui_response("");
        assert_eq!(status, 200, "the page is not a 401 to its own user");
        assert!(content_type.starts_with("text/html"), "{content_type}");
    }

    /// Every door the page actually uses, reached from its method and its path
    /// alone — the same answer for a loopback browser, a phone on the LAN, and
    /// anything else that can open the port.
    #[test]
    fn every_door_is_reached_by_method_and_path_alone() {
        assert_eq!(route("GET", "events"), Route::Events);
        assert_eq!(route("POST", "cmd"), Route::Command);
        assert_eq!(route("POST", "upload"), Route::Upload);
        assert_eq!(route("GET", "file"), Route::File);
        assert_eq!(route("GET", "health"), Route::Health);
        assert_eq!(route("GET", "index.html"), Route::Ui);
    }

    /// The two answers that used to be `401` first are now just routing: the
    /// wrong verb on a door that exists is `405`, an unknown path is `404`, and
    /// an unknown verb is `404` too. `401` is a status this file can no longer
    /// produce at all.
    #[test]
    fn a_wrong_verb_and_an_unknown_path_answer_without_a_credential() {
        assert_eq!(route("GET", "cmd"), Route::MethodNotAllowed);
        assert_eq!(route("POST", "events"), Route::MethodNotAllowed);
        assert_eq!(route("POST", "file"), Route::MethodNotAllowed);
        assert_eq!(route("POST", "no/such/thing"), Route::MethodNotAllowed);
        assert_eq!(route("DELETE", "cmd"), Route::NotFound);
        assert_eq!(route("HEAD", ""), Route::NotFound);
        assert_eq!(route("PATCH", "events"), Route::NotFound);
        // A `GET` on an unknown path is still the interface branch; that is
        // where the 404 for a path this server will not read comes from.
        assert_eq!(route("GET", "no/such/thing"), Route::Ui);
    }

    /// The documents themselves, answered to a caller that has nothing to
    /// present: the entry document the printed URL opens, and an asset beside
    /// it. What a path may *not* name is `path_tests`' job — this one is only
    /// that the reading side of the old gate is gone.
    #[test]
    fn the_interface_files_are_served() {
        assert_eq!(ui_response("").0, 200);
        assert_eq!(ui_response("index.html").0, 200);
        assert_eq!(ui_response("nested/asset.css").0, 200);
    }
}

#[cfg(test)]
mod transfer_tests {
    use super::{read_head, HeadError};

    /// The head is parsed without a socket, so every rejection a client can
    /// provoke is decided — and tested — here.
    #[test]
    fn a_plain_head_parses_with_its_method_and_target() {
        let head = read_head(b"GET /file?path=x HTTP/1.1\r\nHost: localhost\r\n").expect("head");
        assert_eq!(head.method, "GET");
        assert_eq!(head.target, "/file?path=x");
        // Header names are lowercased on the way in, so lookups do not have to
        // guess the client's spelling.
        assert_eq!(head.headers.get("host").map(String::as_str), Some("localhost"));
        // A GET has no body to declare, and no body is the right reading.
        assert_eq!(head.framing(), Ok(0));
    }

    #[test]
    fn content_length_is_the_only_body_framing() {
        let post = read_head(b"POST /cmd HTTP/1.1\r\nContent-Length: 12\r\n").expect("head");
        assert_eq!(post.framing(), Ok(12));
        // A POST with no length is refused rather than read as empty: the
        // client sent something and guessing zero answers a different request.
        let bare = read_head(b"POST /cmd HTTP/1.1\r\nHost: h\r\n").expect("head");
        assert_eq!(bare.framing(), Err(HeadError::LengthRequired));
        assert_eq!(HeadError::LengthRequired.status(), 400);
        // A length that is not a number is an error, not a zero. The head
        // reader decides it, so there is no parsed head left to frame: a
        // `Head` holding an unreadable length would let the body reader pick
        // its own interpretation of how many bytes follow.
        assert_eq!(
            read_head(b"POST /cmd HTTP/1.1\r\nContent-Length: twelve\r\n").err(),
            Some(HeadError::BadLength)
        );
        assert_eq!(HeadError::BadLength.status(), 400);
        assert_eq!(
            read_head(b"POST / HTTP/1.1\r\nContent-Length: twelve\r\n").err(),
            Some(HeadError::BadLength)
        );
        // A negative length is not a length either.
        assert_eq!(
            read_head(b"POST /cmd HTTP/1.1\r\nContent-Length: -1\r\n").err(),
            Some(HeadError::BadLength)
        );
    }

    /// A chunked body is not something this server reads, and it says so with
    /// `501` — the request is well formed, the implementation is missing.
    /// Reading a `Content-Length` while ignoring a `Transfer-Encoding` is the
    /// classic request-smuggling shape, so both together are refused too.
    #[test]
    fn transfer_encoding_is_identified_and_refused() {
        let chunked = read_head(b"POST /cmd HTTP/1.1\r\nTransfer-Encoding: chunked\r\n")
            .expect("head");
        assert_eq!(chunked.framing(), Err(HeadError::TransferEncoding));
        assert_eq!(HeadError::TransferEncoding.status(), 501);
        assert!(HeadError::TransferEncoding.reason().contains("Content-Length"));
        let both = read_head(
            b"POST /cmd HTTP/1.1\r\nTransfer-Encoding: chunked\r\nContent-Length: 4\r\n",
        )
        .expect("head");
        assert_eq!(both.framing(), Err(HeadError::TransferEncoding));
    }

    #[test]
    fn a_head_that_is_not_a_head_is_malformed() {
        assert_eq!(read_head(b"").err(), Some(HeadError::Malformed));
        // No target on the request line.
        assert_eq!(read_head(b"GET\r\n").err(), Some(HeadError::Malformed));
        // Not UTF-8 at all.
        assert_eq!(read_head(&[0xff, 0xfe, 0xfd]).err(), Some(HeadError::Malformed));
    }
}

#[cfg(test)]
mod name_tests {
    use super::{
        free_name, is_windows_device_name, sanitize_name, suffixed, MAX_NAME_CHARS, MAX_NAME_SUFFIX,
    };

    /// The path rules are the ones that were already right; these assert they
    /// still are after the added filters.
    #[test]
    fn names_cannot_become_paths() {
        assert_eq!(sanitize_name("photo.png"), "photo.png");
        assert_eq!(sanitize_name("../../etc/passwd"), "passwd");
        assert_eq!(sanitize_name("/etc/passwd"), "passwd");
        assert_eq!(sanitize_name("with space.txt"), "with space.txt");
        assert_eq!(sanitize_name("中文名.txt"), "中文名.txt");
        // No separator survives, whichever separator the sender meant.
        let flat = sanitize_name("a/b\\c.txt");
        assert!(!flat.contains('/') && !flat.contains('\\'), "{flat:?}");
    }

    /// Control and format characters are dropped: U+202E reverses the rendered
    /// name, which is how `photo\u{202e}gnp.exe` reads as a `.png`.
    #[test]
    fn invisible_and_control_characters_are_dropped() {
        assert_eq!(sanitize_name("a\u{0}b.txt"), "ab.txt");
        assert_eq!(sanitize_name("a\nb.txt"), "ab.txt");
        let reversed = sanitize_name("photo\u{202e}gnp.exe");
        assert!(!reversed.contains('\u{202e}'), "{reversed:?}");
        assert_eq!(reversed, "photognp.exe");
        // A byte-order mark and a zero-width space are not part of the name.
        assert_eq!(sanitize_name("\u{feff}note.txt"), "note.txt");
        assert_eq!(sanitize_name("no\u{200b}te.txt"), "note.txt");
    }

    /// Windows device names are not files there, whatever the case or the
    /// extension, so the name gets one underscore and stays recognisable.
    #[test]
    fn windows_device_names_are_refused_as_written() {
        assert_eq!(sanitize_name("CON"), "_CON");
        assert_eq!(sanitize_name("con.txt"), "_con.txt");
        assert_eq!(sanitize_name("COM1"), "_COM1");
        assert_eq!(sanitize_name("Lpt9.log"), "_Lpt9.log");
        assert!(is_windows_device_name("nul.tar.gz"));
        // Names that merely start like one are ordinary files.
        assert_eq!(sanitize_name("console.txt"), "console.txt");
        assert_eq!(sanitize_name("COM0"), "COM0");
        assert_eq!(sanitize_name("COM10.txt"), "COM10.txt");
        assert!(!is_windows_device_name("contract.md"));
    }

    /// The truncation is counted in characters, so a CJK name keeps as many
    /// characters as an ASCII one keeps bytes.
    #[test]
    fn long_names_are_truncated_to_two_hundred_characters() {
        let long = sanitize_name(&"a".repeat(300));
        assert_eq!(long.chars().count(), MAX_NAME_CHARS);
        let cjk = sanitize_name(&"中".repeat(300));
        assert_eq!(cjk.chars().count(), MAX_NAME_CHARS);
        // A truncated name must not end in the dot the trim exists for.
        assert!(!sanitize_name(&format!("{}.txt", "a".repeat(197))).ends_with('.'));
    }

    #[test]
    fn a_name_with_nothing_left_falls_back() {
        assert_eq!(sanitize_name(""), "upload.bin");
        assert_eq!(sanitize_name("..."), "upload.bin");
        assert_eq!(sanitize_name("///"), "upload.bin");
        assert_eq!(sanitize_name("***"), "upload.bin");
    }

    /// The duplicate suffix keeps the extension last, so the type survives.
    #[test]
    fn duplicates_are_numbered_not_overwritten() {
        assert_eq!(suffixed("photo.png", 1), "photo(1).png");
        assert_eq!(suffixed("archive.tar.gz", 2), "archive.tar(2).gz");
        assert_eq!(suffixed("README", 1), "README(1)");
        assert_eq!(suffixed(".hidden", 1), ".hidden(1)");

        assert_eq!(free_name(|_| false, "photo.png"), "photo.png");
        assert_eq!(
            free_name(|candidate| candidate == "photo.png", "photo.png"),
            "photo(1).png"
        );
        assert_eq!(
            free_name(|candidate| matches!(candidate, "photo.png" | "photo(1).png"), "photo.png"),
            "photo(2).png"
        );
        // Exhausting the suffixes still returns a usable name rather than
        // failing the upload.
        assert_eq!(
            free_name(|_| true, "photo.png"),
            suffixed("photo.png", MAX_NAME_SUFFIX + 1)
        );
    }
}

#[cfg(test)]
mod sse_tests {
    use super::{sse_event, sse_resync};

    /// The page can tell events apart and spot a gap only if the ids are there
    /// and monotonic, so the frame format is what this pins down.
    #[test]
    fn every_event_carries_a_monotonic_id() {
        let first = sse_event(1, r#"{"ev":"status"}"#);
        let second = sse_event(2, r#"{"ev":"status"}"#);
        assert_eq!(first, "id: 1\ndata: {\"ev\":\"status\"}\n\n");
        assert_eq!(second, "id: 2\ndata: {\"ev\":\"status\"}\n\n");
        assert!(first.ends_with("\n\n"));
        assert!(!first.contains("event:"));
        // The payload stays on one `data:` line: the bus carries serialized
        // JSON, which never contains a newline.
        assert_eq!(first.lines().filter(|line| line.starts_with("data: ")).count(), 1);
    }

    /// A lagging listener is told it missed events, with a payload it can parse
    /// like any other event — the page answers it by re-sending `ready`.
    #[test]
    fn a_lagging_listener_is_told_to_resync() {
        let frame = sse_resync(7);
        assert_eq!(frame, "id: 7\nevent: resync\ndata: {\"reason\":\"lagged\"}\n\n");
        let data = frame
            .lines()
            .find_map(|line| line.strip_prefix("data: "))
            .expect("data line");
        let parsed: serde_json::Value = serde_json::from_str(data).expect("json payload");
        assert_eq!(parsed["reason"], "lagged");
    }
}

#[cfg(test)]
mod limit_tests {
    use super::{
        over_file_limit, ConnectionSlot, SubscriberSlot, MAX_ACTIVE_CONNECTIONS, MAX_FILE_BYTES,
        MAX_SSE_SUBSCRIBERS, READ_TIMEOUT,
    };

    /// The numbers are policy, so they are asserted rather than left to drift
    /// with a refactor.
    #[test]
    fn the_limits_are_the_ones_that_were_agreed() {
        assert_eq!(MAX_FILE_BYTES, 64 * 1024 * 1024);
        assert_eq!(MAX_ACTIVE_CONNECTIONS, 64);
        assert_eq!(MAX_SSE_SUBSCRIBERS, 16);
        assert_eq!(READ_TIMEOUT, std::time::Duration::from_secs(10));
    }

    /// The boundary is one byte over, not one byte under: a file of exactly the
    /// limit is served, the next byte is `413`.
    #[test]
    fn the_file_limit_refuses_only_what_is_over_it() {
        assert!(!over_file_limit(0));
        assert!(!over_file_limit(MAX_FILE_BYTES - 1));
        assert!(!over_file_limit(MAX_FILE_BYTES));
        assert!(over_file_limit(MAX_FILE_BYTES + 1));
        assert!(over_file_limit(u64::MAX));
    }

    /// Both caps are process-global counters, so this drives them to the limit
    /// and back: a slot that never comes back is a server that refuses every
    /// request an hour later, which is the failure the guards exist to avoid.
    #[test]
    fn connection_slots_are_capped_and_returned() {
        let mut slots = Vec::new();
        for _ in 0..MAX_ACTIVE_CONNECTIONS {
            slots.push(ConnectionSlot::acquire().expect("under the limit"));
        }
        // At the limit: and a refused claim must not consume a slot either.
        for _ in 0..8 {
            assert!(ConnectionSlot::acquire().is_none());
        }
        // Giving one back makes room for exactly one more.
        drop(slots.pop());
        assert!(ConnectionSlot::acquire().is_some());
        drop(slots);
    }

    #[test]
    fn subscriber_slots_are_capped_and_returned() {
        let mut slots = Vec::new();
        for _ in 0..MAX_SSE_SUBSCRIBERS {
            slots.push(SubscriberSlot::acquire().expect("under the limit"));
        }
        for _ in 0..8 {
            assert!(SubscriberSlot::acquire().is_none());
        }
        drop(slots.pop());
        assert!(SubscriberSlot::acquire().is_some());
        drop(slots);
    }
}

#[cfg(test)]
mod query_tests {
    use super::split_target;

    /// One decode, in one place: `/file` used to decode its own value, so the
    /// two doors into the query string disagreed about what a value was.
    #[test]
    fn query_values_are_percent_decoded_once() {
        let (path, query) = split_target("/file?path=my%20file.txt&name=r%C3%A9sum%C3%A9.pdf");
        assert_eq!(path, "file");
        assert_eq!(query.get("path").map(String::as_str), Some("my file.txt"));
        assert_eq!(query.get("name").map(String::as_str), Some("résumé.pdf"));
        // A lone `%` is not an encoding and survives as itself.
        let (_, query) = split_target("/file?path=100%done.txt");
        assert_eq!(query.get("path").map(String::as_str), Some("100%done.txt"));
        // The leading slash is trimmed, and a target with no query is empty.
        let (path, query) = split_target("/status");
        assert_eq!(path, "status");
        assert!(query.is_empty());
    }

    /// The query string is data now, never a credential, and the doubling that
    /// used to be answered `400` is the case worth pinning: the URL an old
    /// bookmark holds — printed by an older build, `?token=` and all, sometimes
    /// twice — has to open the page like any other address. The reading is the
    /// one every other parameter gets: the last occurrence wins, and only the
    /// names a door actually asks for are ever looked up.
    #[test]
    fn an_old_token_in_the_query_is_ordinary_data() {
        let (path, query) = split_target("/?token=6f1c&token=stale");
        assert_eq!(path, "", "the entry document, whatever the query carries");
        assert_eq!(query.get("token").map(String::as_str), Some("stale"));
        // A token beside a real parameter changes nothing about the parameter.
        let (path, query) = split_target("/file?path=uploads%2Fx.png&token=stale");
        assert_eq!(path, "file");
        assert_eq!(query.get("path").map(String::as_str), Some("uploads/x.png"));
        // What a door reads is untouched: no token, no difference.
        let without_token = split_target("/file?path=uploads%2Fx.png").1;
        assert_eq!(
            without_token.get("path").map(String::as_str),
            Some("uploads/x.png")
        );
    }

    /// Decoding happens after the split, so a value cannot smuggle a separator
    /// in: `%26` arrives as `&` only once the pairs have been cut, which is why
    /// a `path` that contains `&token=` is still one value and one lookup.
    #[test]
    fn a_value_that_looks_like_another_parameter_is_still_one_value() {
        let (_, query) = split_target("/file?path=a%26token%3Dx");
        assert_eq!(query.get("path").map(String::as_str), Some("a&token=x"));
        assert_eq!(query.get("token"), None);
    }
}

/// A URL path turned into a path that is *only* ever relative, or nothing.
///
/// The one rule this enforces: whatever the client typed, the answer is a
/// descent through plain file names - never out of the directory it is joined
/// to, and never a name the file system reads as something else. Four things
/// make that harder than splitting on `/`:
///
/// - `%2e%2e` is `..` by the time a browser, a proxy, or a second decode has
///   finished with it, so the check runs on the decoded form.
/// - `\` is a separator on Windows and an ordinary character in a URL, so
///   `..\..\x` arrives as one harmless-looking segment. Backslashes become
///   separators here before anything is split.
/// - An absolute path makes `Path::join` *replace* the root instead of
///   descending it, so the leading separator is dropped and any segment
///   carrying a `:` - a drive letter or an alternate data stream - is refused.
/// - Windows reads `CON`, `NUL`, `COM1` and friends as devices wherever they
///   appear, extension or not, so they are refused as file names.
pub(crate) fn safe_relative(target: &str) -> Option<std::path::PathBuf> {
    let decoded = percent_decode(target);
    let unified = decoded.replace('\\', "/");
    let trimmed = unified.trim_start_matches('/');
    if trimmed.is_empty() {
        return None;
    }
    let mut relative = std::path::PathBuf::new();
    for part in trimmed.split('/') {
        if part.is_empty() || part == "." || part == ".." {
            return None;
        }
        if part.contains(':') || part.contains('\0') {
            return None;
        }
        if is_windows_device_name(part) {
            return None;
        }
        relative.push(part);
    }
    // Pushing plain names cannot produce anything else, but the caller joins
    // this to a root and then trusts the result, so the shape is asserted
    // rather than assumed.
    if !relative
        .components()
        .all(|part| matches!(part, std::path::Component::Normal(_)))
    {
        return None;
    }
    Some(relative)
}

#[cfg(test)]
mod path_tests {
    use super::{safe_relative, ui_response};

    /// Every spelling of "escape the root" a URL can carry: an absolute path
    /// (which makes `join` replace the root), a raw backslash climb, a
    /// percent-encoded climb, and a climb hidden behind a harmless first
    /// segment.
    #[test]
    fn a_path_that_climbs_out_of_the_ui_directory_is_refused() {
        for target in ["/C:\\Windows\\win.ini", "/..\\..\\x", "/%2e%2e/x", "/a\\..\\..\\b"] {
            assert!(
                safe_relative(target).is_none(),
                "{target} was accepted as a relative path"
            );
        }
    }

    /// The paths the page actually asks for keep working, spaces included.
    #[test]
    fn a_plain_asset_path_stays_relative() {
        assert_eq!(
            safe_relative("/a/b.md"),
            Some(std::path::PathBuf::from("a/b.md"))
        );
        assert_eq!(
            safe_relative("/a/%20b.md"),
            Some(std::path::PathBuf::from("a/ b.md"))
        );
    }

    /// The gate that used to answer this question is gone, and the rule that
    /// replaced it is the path itself. Any caller may ask for an interface
    /// file — no token, any `Host:` — and the answer is decided here: the entry
    /// document and the assets the page really uses are served, and a path that
    /// would climb out of the interface directory answers `404`, which is what
    /// a file that is not there answers too.
    #[test]
    fn anyone_may_ask_but_the_path_still_decides() {
        // The printed URL: no path, so the entry document.
        let (status, content_type, _) = ui_response("");
        assert_eq!(status, 200);
        assert!(content_type.starts_with("text/html"), "{content_type}");
        // An asset the interface really asks for.
        let (status, _, _) = ui_response("app.js");
        assert_eq!(status, 200);
        // A climb, in every spelling `safe_relative` refuses: a refusal here is
        // a 404, the same answer a missing file gets, and never a 401 and never
        // a file.
        for path in ["../settings.json", "..%2f..%2fx", "C:/Windows/win.ini", "a\\..\\..\\b"] {
            let (status, content_type, body) = ui_response(path);
            assert_eq!(status, 404, "{path} answered {status}");
            assert_eq!(content_type, "text/plain; charset=utf-8");
            assert_eq!(body, b"not found".to_vec());
        }
    }
}

#[cfg(test)]
mod optional_session_tests {
    use super::optional_session;

    /// The three shapes a `session` can arrive in, and the one that must not be
    /// guessed at. Absent and null both mean "the conversation on screen"; a
    /// string is that conversation, trimmed, with the empty (or all-whitespace)
    /// string counting as absent because it names nothing; anything else is an
    /// error the caller answers `400` to, never a value quietly read as "on
    /// screen" — the one answer an explicit id must never get.
    #[test]
    fn an_optional_session_is_read_or_refused() {
        assert_eq!(optional_session(&serde_json::json!({})), Ok(None));
        assert_eq!(optional_session(&serde_json::json!({ "session": null })), Ok(None));
        assert_eq!(optional_session(&serde_json::json!({ "session": "" })), Ok(None));
        assert_eq!(optional_session(&serde_json::json!({ "session": "   " })), Ok(None));
        assert_eq!(
            optional_session(&serde_json::json!({ "session": " 20260101-000000-abc123 " })),
            Ok(Some("20260101-000000-abc123".to_string()))
        );
        // A value that is not a string is a body this server cannot read.
        let error = optional_session(&serde_json::json!({ "session": 17 })).expect_err("a number");
        assert!(error.contains("session"), "{error}");
        assert!(optional_session(&serde_json::json!({ "session": ["a"] })).is_err());
        assert!(optional_session(&serde_json::json!({ "session": true })).is_err());
    }
}
