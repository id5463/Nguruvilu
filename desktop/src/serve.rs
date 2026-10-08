//! `--serve`: the same shell over a socket.
//!
//! The window exists only where the webview stack exists; everything under it —
//! state, dispatch, headless — does not care. This module adds the last layer:
//! the page the window renders, served over HTTP/SSE so a browser on any machine
//! can be the window. Commands come in on `POST /cmd`, events go out on
//! `GET /events`, and both are the page's own `dispatch` — not a parallel
//! implementation that can drift from it.
//!
//! The door is locked with a bearer token. A shell behind `--serve` can run
//! turns, install packs, and read settings; on `0.0.0.0` anyone who reaches the
//! port owns it. The token is generated fresh each run and printed with the URL
//! to open.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{broadcast, mpsc};

use crate::sink::EventSink;
use crate::state::AppState;
use crate::{resolve_active_ui, serve_ui, set_active_ui};

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
    token: String,
    startup_prompt: Option<String>,
    ui_id: Option<String>,
) -> anyhow::Result<()> {
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

        // Two lanes, split by what a command touches. Short commands and turn
        // claims go straight to their own task — the window's IPC handler has
        // always worked this way, and a ten-minute turn must not freeze
        // `status`, `open_session`, or a second session's prompt behind it.
        // The ordered lane keeps only what genuinely needs to run one at a
        // time: `ready`'s initial push (so two refreshes cannot race pack
        // loading) and pack work, which mutates the kernel.
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
                "[serve] {addr} is not loopback: anyone who reaches this port and holds the \
                 token can drive the shell"
            );
        }
        eprintln!("[serve] listening on http://{addr}");
        eprintln!("[serve] open http://{addr}/?token={token}");

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
            let bus = bus.clone();
            let token = token.clone();
            let commands_tx = commands_tx.clone();
            let state = Arc::clone(&state);
            tokio::spawn(async move {
                handle(stream, peer, bus, token, commands_tx, state).await;
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
async fn handle(
    mut stream: TcpStream,
    peer: std::net::SocketAddr,
    bus: broadcast::Sender<String>,
    token: String,
    commands_tx: mpsc::UnboundedSender<String>,
    state: Arc<Mutex<AppState>>,
) {
    let Some(request) = read_request(&mut stream).await else {
        return;
    };
    let (path, query) = split_target(&request.target);

    let (status, content_type, body) = match (request.method.as_str(), path) {
        // The event stream and the command door share one key, checked the
        // same way: a token that only guards half the shell guards nothing.
        ("GET", "events") => {
            if !authorized(&request, &query, &token) {
                plain(401, "unauthorized")
            } else {
                // One line per subscriber: it answers the only question that
                // matters when a page comes up blank — was anyone listening.
                eprintln!("[serve] events stream to {peer}");
                sse(&mut stream, bus).await;
                return;
            }
        }
        ("POST", "cmd") => {
            let body_text = std::str::from_utf8(&request.body).unwrap_or("");
            if !authorized(&request, &query, &token) {
                plain(401, "unauthorized")
            } else if serde_json::from_str::<Value>(body_text).is_err() {
                plain(400, "body must be one JSON object")
            } else {
                // A stop must not queue behind the very turn it is stopping:
                // this server runs commands one at a time, so a queued
                // `cancel` would only be reached after the turn already
                // ended. The flag is set under the lock and the turn does
                // the rest at its next safe point.
                let parsed: Value = serde_json::from_str(body_text).unwrap_or(Value::Null);
                if parsed.get("cmd").and_then(|c| c.as_str()) == Some("cancel") {
                    let stopped = {
                        let guard = state.lock().expect("state lock");
                        let session = guard.session.id.clone();
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
                    let text = parsed
                        .get("text")
                        .and_then(|t| t.as_str())
                        .unwrap_or_default()
                        .to_string();
                    let steered = {
                        let mut guard = state.lock().expect("state lock");
                        crate::state::try_steer(&mut guard, &text)
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
        // paths come back as download links. Same door as everything else —
        // a token that did not earn the right to drive the shell has not
        // earned the right to read its disk either.
        ("POST", "upload") => {
            if !authorized(&request, &query, &token) {
                plain(401, "unauthorized")
            } else if request.body.is_empty() {
                plain(400, "empty upload")
            } else {
                let name = sanitize_name(query.get("name").map(String::as_str).unwrap_or(""));
                let workdir = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
                let directory = workdir.join("uploads");
                match tokio::fs::create_dir_all(&directory).await {
                    Err(error) => plain(500, &format!("cannot create uploads: {error}")),
                    Ok(()) => {
                        let file_path = directory.join(&name);
                        match tokio::fs::write(&file_path, &request.body).await {
                            Err(error) => plain(500, &format!("cannot write upload: {error}")),
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
        ("GET", "file") => {
            if !authorized(&request, &query, &token) {
                plain(401, "unauthorized")
            } else {
                let Some(raw_path) = query.get("path") else {
                    respond(&mut stream, 400, "text/plain; charset=utf-8", b"missing path").await;
                    return;
                };
                let decoded = percent_decode(raw_path);
                let workdir = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
                let candidate = if std::path::Path::new(&decoded).is_absolute() {
                    std::path::PathBuf::from(&decoded)
                } else {
                    workdir.join(&decoded)
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
                        match tokio::fs::read(&path).await {
                            Ok(bytes) => {
                                let filename = path
                                    .file_name()
                                    .map(|name| name.to_string_lossy().into_owned())
                                    .unwrap_or_else(|| "download".into());
                                eprintln!("[serve] download {} ({} bytes)", path.display(), bytes.len());
                                respond_attachment(&mut stream, &filename, &bytes).await;
                                return;
                            }
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
                        }
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
        }
        // Health for the upstream the shell cannot afford to guess at: the
        // page asks, this answers by opening the same door the LLM client
        // would — a CONNECT handshake through the configured proxy. 200 from
        // the handshake means the node completed the upstream leg; anything
        // else (5xx from the proxy, silence) means the tunnel is down.
        ("GET", "health") => {
            if !authorized(&request, &query, &token) {
                plain(401, "unauthorized")
            } else {
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
        }
        ("GET", _) if path != "events" && path != "cmd" => {
            let (bytes, content_type) = serve_ui(path);
            (200, content_type, bytes)
        }
        ("POST", _) | ("GET", "cmd") => plain(405, "method not allowed"),
        _ => plain(404, "not found"),
    };

    respond(&mut stream, status, content_type, &body).await;
}

fn plain(status: u16, message: &str) -> (u16, &'static str, Vec<u8>) {
    (
        status,
        "text/plain; charset=utf-8",
        message.as_bytes().to_vec(),
    )
}

/// An upload's name, stripped of anything that could make it a path: the
/// file lands in `uploads/` and nowhere else, whatever the client sent.
fn sanitize_name(name: &str) -> String {
    let base = std::path::Path::new(name)
        .file_name()
        .map(|part| part.to_string_lossy().into_owned())
        .unwrap_or_default();
    let cleaned: String = base
        .chars()
        .filter(|c| c.is_alphanumeric() || matches!(c, '-' | '_' | '.' | ' '))
        .collect();
    let trimmed = cleaned.trim().trim_matches(['.', ' ']);
    if trimmed.is_empty() {
        "upload.bin".to_string()
    } else {
        trimmed.to_string()
    }
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
    use std::sync::atomic::Ordering;
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
fn percent_decode(value: &str) -> String {
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
async fn respond_attachment(stream: &mut TcpStream, filename: &str, body: &[u8]) {
    let ascii: String = filename
        .chars()
        .map(|c| if c.is_ascii_graphic() || c == ' ' { c } else { '_' })
        .collect();
    let head = format!(
        "HTTP/1.1 200 OK\r\n\
         Content-Type: application/octet-stream\r\n\
         Content-Disposition: attachment; filename=\"{}\"; filename*=UTF-8''{}\r\n\
         Content-Length: {}\r\n\
         Connection: close\r\n\r\n",
        ascii.replace('"', "'"),
        percent_encode_utf8(filename),
        body.len()
    );
    if stream.write_all(head.as_bytes()).await.is_ok() {
        let _ = stream.write_all(body).await;
    }
}

/// A parsed HTTP/1.1 request head plus body, as far as this server needs.
struct Request {
    method: String,
    target: String,
    headers: HashMap<String, String>,
    /// Raw bytes, kept raw because `/upload` carries files that are not UTF-8;
    /// `/cmd` decodes on its own where JSON needs it.
    body: Vec<u8>,
}

/// Read one request off the socket: head until `\r\n\r\n`, then exactly
/// `Content-Length` bytes of body. Anything malformed is dropped without a
/// reply — an unsolicited connection owes nobody an explanation.
async fn read_request(stream: &mut TcpStream) -> Option<Request> {
    let mut buffer = Vec::with_capacity(4096);
    let mut chunk = [0u8; 8192];

    let head_end = loop {
        if let Some(position) = find_subslice(&buffer, b"\r\n\r\n") {
            break position;
        }
        if buffer.len() > 64 * 1024 {
            return None;
        }
        let read = stream.read(&mut chunk).await.ok()?;
        if read == 0 {
            return None;
        }
        buffer.extend_from_slice(&chunk[..read]);
    };

    let head = std::str::from_utf8(&buffer[..head_end]).ok()?;
    let mut lines = head.split("\r\n");
    let mut request_line = lines.next()?.split_whitespace();
    let method = request_line.next()?.to_string();
    let target = request_line.next()?.to_string();

    let mut headers = HashMap::new();
    for line in lines {
        if let Some((name, value)) = line.split_once(':') {
            headers.insert(name.trim().to_ascii_lowercase(), value.trim().to_string());
        }
    }

    let content_length: usize = headers
        .get("content-length")
        .and_then(|value| value.parse().ok())
        .unwrap_or(0);
    // Wide enough for an uploaded file, narrow enough that a hostile length
    // cannot ask a 2 GB box for memory it does not have.
    if content_length > 64 * 1024 * 1024 {
        return None;
    }

    let mut body_bytes = buffer[head_end + 4..].to_vec();
    while body_bytes.len() < content_length {
        let read = stream.read(&mut chunk).await.ok()?;
        if read == 0 {
            return None;
        }
        body_bytes.extend_from_slice(&chunk[..read]);
    }
    body_bytes.truncate(content_length);

    Some(Request {
        method,
        target,
        headers,
        body: body_bytes,
    })
}

/// Write one response and close. `Connection: close` is the whole policy:
/// keep-alive would mean tracking half-closed clients for no gain here.
async fn respond(stream: &mut TcpStream, status: u16, content_type: &str, body: &[u8]) {
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        _ => "",
    };
    let head = format!(
        "HTTP/1.1 {status} {reason}\r\n\
         Content-Type: {content_type}\r\n\
         Content-Length: {}\r\n\
         Connection: close\r\n\r\n",
        body.len()
    );
    if stream.write_all(head.as_bytes()).await.is_ok() {
        let _ = stream.write_all(body).await;
    }
}

/// Hand the socket to the event stream: headers once, then every event as it
/// comes, with a heartbeat so an idle proxy cannot decide the stream is dead.
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
    loop {
        tokio::select! {
            event = events.recv() => match event {
                Ok(data) => {
                    if stream.write_all(format!("data: {data}\n\n").as_bytes()).await.is_err() {
                        break;
                    }
                }
                // A reader too slow to keep up loses events, not the connection:
                // the next `ready` re-sends everything durable.
                Err(broadcast::error::RecvError::Lagged(_)) => continue,
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

/// Check the bearer token: header where a client can send one, query string
/// where it cannot — `EventSource` has no header of its own.
fn authorized(request: &Request, query: &HashMap<String, String>, token: &str) -> bool {
    let header_ok = request
        .headers
        .get("authorization")
        .and_then(|value| value.strip_prefix("Bearer "))
        .is_some_and(|value| value == token);
    header_ok || query.get("token").is_some_and(|value| value == token)
}

/// Split `path?query` into both halves, query as a map.
fn split_target(target: &str) -> (&str, HashMap<String, String>) {
    let (path, query) = target.split_once('?').unwrap_or((target, ""));
    let query = query
        .split('&')
        .filter_map(|pair| pair.split_once('='))
        .map(|(name, value)| (name.to_string(), value.to_string()))
        .collect();
    (path.trim_start_matches('/'), query)
}

fn is_ready(body: &str) -> bool {
    serde_json::from_str::<Value>(body)
        .ok()
        .and_then(|command| command.get("cmd").and_then(Value::as_str).map(str::to_string))
        .is_some_and(|cmd| cmd == "ready")
}

/// Whether a command may run on its own task without waiting for the ordered
/// lane — the judgement behind the whole scheduling split.
///
/// White-listed on purpose: a short critical section (a read or one locked
/// write, released before any await) or a turn claim that the per-session
/// `running` set arbitrates. `prompt` qualifies because `run_turn` claims its
/// session slot atomically — a second prompt for a busy session goes through
/// `try_steer`/pending rather than racing. Everything else (pack operations,
/// `ready`, commands added later) defaults to the ordered lane, where the
/// worst an unknown command can do is run after its neighbours.
fn is_concurrent(body: &str) -> bool {
    serde_json::from_str::<Value>(body)
        .ok()
        .and_then(|command| command.get("cmd").and_then(Value::as_str).map(str::to_string))
        .is_some_and(|cmd| {
            matches!(
                cmd.as_str(),
                "status"
                    | "new_session"
                    | "open_session"
                    | "delete_session"
                    | "list_packs"
                    | "fetch_models"
                    | "ui_panels"
                    | "save_settings"
                    | "set_model"
                    | "prompt"
            )
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
