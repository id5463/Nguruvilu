//! Cloud-to-local remote control, with the local machine as the only party
//! that dials out.
//!
//! The local app connects outward to the cloud relay and waits for commands.
//! Nothing listens on a local port for this. Every command passes the
//! allow-list below before it reaches the shell dispatcher, and every executed
//! command is written to the audit log.
//!
//! Holding the relay token is not, by itself, permission to make this machine
//! run a turn: `prompt` additionally needs the local switch in [`allow_prompt`],
//! and every frame is measured against [`MAX_BODY_BYTES`] and a rolling
//! [`MAX_COMMANDS_PER_MINUTE`] window in [`admit`] before anything reaches the
//! dispatcher. A leaked token should cost a bounded amount of work, not the
//! machine.
//!
//! The token is still one shared secret, with no machine identity and no
//! rotation — the residual risk is spelled out on [`RelayConfig`]. What this
//! module can do about it without a relay-side protocol change is remember which
//! relay and which credential each link used, as a target plus a fingerprint,
//! so two of either can be told apart afterwards.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Commands the cloud may run. Read-only status and turn-starting commands only.
/// Destructive commands (delete, settings writes, pack uninstall, shell exit)
/// stay local-only on purpose.
pub const ALLOWED_COMMANDS: &[&str] = &["ready", "status", "new_session", "open_session", "prompt", "cancel"];

/// Largest command body the relay may hand this machine, in bytes.
///
/// The body is read whole into memory and parsed before the allow-list sees
/// it, so this bounds what one relayed frame can cost. 64 KiB is far past any
/// real command and far short of what pushing a file through would need.
pub const MAX_BODY_BYTES: usize = 64 * 1024;

/// Commands accepted in any rolling [`RATE_WINDOW`].
///
/// The relay lives on somebody else's machine: a leak should cost a bounded
/// number of agent turns per minute, not an unbounded one.
pub const MAX_COMMANDS_PER_MINUTE: usize = 30;

/// The rolling window the rate limit is measured over.
pub const RATE_WINDOW: Duration = Duration::from_secs(60);

/// How long one relayed command may run before its caller is told it timed out.
///
/// A turn can legitimately take minutes; two is past any turn that was going to
/// finish, and the frame loop must not sit behind one forever — a relay reads
/// silence as a dead machine and drops the link anyway.
pub const COMMAND_TIMEOUT: Duration = Duration::from_secs(120);

/// Consecutive attempts that ended within [`FAST_FAILURE`] before reconnecting
/// stops for good.
pub const MAX_FAST_FAILURES: u32 = 5;

/// An attempt that ends sooner than this counts as a fast failure.
pub const FAST_FAILURE: Duration = Duration::from_secs(5);

/// How far a retry wait is stretched or shrunk: ±30%.
pub const JITTER: f64 = 0.3;

/// Whether a remote command body may run. Rejects anything that is not one
/// JSON object with a `cmd` field from [`ALLOWED_COMMANDS`].
///
/// This is the name-only half of the gate: [`admit`] adds the local `prompt`
/// switch, the body limit and the rate window on top of it.
pub fn is_allowed(body: &str) -> bool {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(body) else {
        return false;
    };
    value
        .get("cmd")
        .and_then(|c| c.as_str())
        .is_some_and(|cmd| ALLOWED_COMMANDS.contains(&cmd))
}

/// Configuration for the outbound relay link. Both values come from the
/// settings file; nothing is defaulted, so a missing value keeps remote control
/// off rather than dialing a guessed host.
///
/// Accepted residual risk (REMOTE-15), deliberately not fixed by a redesign
/// here: the token is one shared secret. It names no machine and no operator,
/// it is not rotated when the link reconnects, and whoever holds it holds this
/// machine's remote access — up to every command the allow-list above permits.
/// What limits the damage is around it, not in it: the allow-list, the local
/// `prompt` switch in [`allow_prompt`], the body and rate limits in [`admit`],
/// and the target-plus-fingerprint line this module logs so two credentials or
/// two relays can at least be told apart afterwards. Replacing the secret with a
/// device id and short-lived paired credentials is a relay-side protocol change.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelayConfig {
    /// Relay WebSocket URL the local app dials out to.
    pub url: String,
    /// Bearer token the relay checks on connect.
    pub token: String,
}

/// One command forwarded by the relay: the relay's request `id` and the raw
/// command JSON in `body`.
struct RelayFrame {
    id: serde_json::Value,
    body: String,
}

impl RelayFrame {
    fn parse(frame: &str) -> Option<Self> {
        let value: serde_json::Value = serde_json::from_str(frame).ok()?;
        let id = value.get("id")?.clone();
        let body = value.get("body")?.as_str()?.to_string();
        Some(Self { id, body })
    }
}

/// Why remote control is off, or `None` when both values are present.
pub fn disabled_reason(url: Option<&str>, token: Option<&str>) -> Option<&'static str> {
    match (url.filter(|u| !u.trim().is_empty()), token.filter(|t| !t.trim().is_empty())) {
        (None, _) => Some("remote control is off: no relay URL is configured"),
        (_, None) => Some("remote control is off: no relay token is configured"),
        (Some(_), Some(_)) => None,
    }
}

/// The local switch that lets a relayed `prompt` drive this machine.
///
/// Off unless somebody turned it on here. The relay token alone must not be
/// enough: with `prompt` the cloud can make the agent read files, run tools and
/// send data out, so "the relay is configured" and "the cloud may start turns"
/// are two decisions, not one.
///
/// The flag lives in `relay.json` beside the settings file
/// (`{"relay_allow_prompt": true}`) rather than in `settings.json`, because the
/// settings struct rejects unknown fields and silently falls back to defaults
/// when it meets one — a stray key there would wipe every other setting.
/// `NGU_RELAY_ALLOW_PROMPT` overrides the file for a scripted deployment.
pub fn allow_prompt() -> bool {
    if let Ok(value) = std::env::var("NGU_RELAY_ALLOW_PROMPT") {
        let value = value.trim().to_ascii_lowercase();
        return matches!(value.as_str(), "1" | "true" | "yes" | "on");
    }
    prompt_flag_from_file(&nguruvilu::settings::Settings::path())
}

/// The switch as recorded beside the settings file.
fn prompt_flag_from_file(settings_path: &std::path::Path) -> bool {
    let path = settings_path.with_file_name("relay.json");
    match std::fs::read_to_string(&path) {
        Ok(text) => prompt_flag_from_json(&text),
        // No file is the default: off. So is an unreadable one — a permission
        // problem must not be the thing that grants the cloud a turn.
        Err(_) => false,
    }
}

/// The switch as written in `relay.json`. Strictly a boolean: the string
/// `"true"` or the number `1` is a hand-edit that meant something else, and
/// guessing in the permissive direction is the wrong way to guess.
fn prompt_flag_from_json(text: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(text)
        .ok()
        .and_then(|value| value.get("relay_allow_prompt").and_then(|flag| flag.as_bool()))
        .unwrap_or(false)
}

/// The rolling window that caps how many commands the relay may send.
///
/// A deque of accept times rather than a per-minute counter: a fixed bucket
/// lets the cloud send 2× the limit across a bucket boundary, which is exactly
/// the burst a rate limit exists to stop.
#[derive(Debug, Default)]
pub struct CommandWindow {
    accepted: VecDeque<Instant>,
    rejected: u64,
}

impl CommandWindow {
    /// An empty window: nothing accepted, nothing refused yet.
    pub fn new() -> Self {
        Self::default()
    }

    /// Count one attempt at `now`; false when the window is already full.
    pub fn admit(&mut self, now: Instant) -> bool {
        while let Some(at) = self.accepted.front() {
            if now.saturating_duration_since(*at) < RATE_WINDOW {
                break;
            }
            self.accepted.pop_front();
        }
        if self.accepted.len() >= MAX_COMMANDS_PER_MINUTE {
            // Counted so the log can say how hard this machine is being
            // pushed: a refusal that is not counted is a refusal nobody sees.
            self.rejected += 1;
            return false;
        }
        self.accepted.push_back(now);
        true
    }

    /// How many attempts were refused for being over the limit.
    pub fn rejected(&self) -> u64 {
        self.rejected
    }
}

/// The verdict for one relayed frame.
#[derive(Debug, PartialEq, Eq)]
pub enum Admission {
    /// It may run; the parsed `cmd`, for the audit line.
    Allowed(String),
    /// The body is larger than [`MAX_BODY_BYTES`].
    TooLong { bytes: usize },
    /// More than [`MAX_COMMANDS_PER_MINUTE`] frames in the last [`RATE_WINDOW`].
    RateLimited,
    /// `prompt` with the local switch off.
    PromptDisabled,
    /// Not one JSON object with a `cmd` from [`ALLOWED_COMMANDS`].
    NotAllowed,
}

impl Admission {
    /// The reason this refusal puts in `{"ok":false,"error":…}`, or `None`
    /// when the frame may run.
    ///
    /// Every refusal is worded for whoever is reading the cloud side: "not
    /// allowed" for a command name it should not have sent, and a plain
    /// statement of the local policy for the two limits.
    pub fn error(&self) -> Option<String> {
        match self {
            Self::Allowed(_) => None,
            Self::TooLong { bytes } => Some(format!(
                "command body is {bytes} bytes: the relay limit is {} KiB",
                MAX_BODY_BYTES / 1024
            )),
            Self::RateLimited => Some(format!(
                "too many commands over the relay: at most {MAX_COMMANDS_PER_MINUTE} per {} seconds",
                RATE_WINDOW.as_secs()
            )),
            Self::PromptDisabled => {
                Some("prompt over the relay is disabled on this machine".to_string())
            }
            Self::NotAllowed => Some("command not allowed".to_string()),
        }
    }
}

/// The whole gate for one relayed frame: window, length, allow-list, `prompt`.
///
/// Order matters: the rate window is charged first, so a flood is throttled
/// even when every frame in it is also refused for its size or its name.
pub fn admit(
    body: &str,
    allow_prompt: bool,
    window: &mut CommandWindow,
    now: Instant,
) -> Admission {
    if !window.admit(now) {
        return Admission::RateLimited;
    }
    if body.len() > MAX_BODY_BYTES {
        return Admission::TooLong { bytes: body.len() };
    }
    // The name check goes through [`is_allowed`] rather than repeating the
    // list here: one place to widen or narrow the allow-list, and the two
    // cannot drift into disagreeing about what the cloud may send.
    if !is_allowed(body) {
        return Admission::NotAllowed;
    }
    let cmd = serde_json::from_str::<serde_json::Value>(body)
        .ok()
        .and_then(|value| value.get("cmd").and_then(|c| c.as_str()).map(str::to_string))
        // `is_allowed` already proved the shape, so a missing command here is
        // unreachable; refusing is still the safe answer if that ever changes.
        .unwrap_or_default();
    if cmd == "prompt" && !allow_prompt {
        return Admission::PromptDisabled;
    }
    Admission::Allowed(cmd)
}

/// The session id an `open_session` body names, when the command is that one.
///
/// A body that names no session comes back as an empty string on purpose: it
/// matches no stored session, so the caller's existence check refuses it too,
/// and there is no second path that opens "whatever was last".
///
/// The id is copied out rather than borrowed from the parsed body: the body is
/// local to this call, and a `&str` into it would outlive what owns it.
pub fn open_session_id(body: &str) -> Option<String> {
    let value: serde_json::Value = serde_json::from_str(body).ok()?;
    if value.get("cmd").and_then(|c| c.as_str()) != Some("open_session") {
        return None;
    }
    Some(
        value
            .get("id")
            .and_then(|id| id.as_str())
            .unwrap_or("")
            .to_string(),
    )
}

/// A retry wait stretched or shrunk by up to [`JITTER`], with `unit` in
/// `[0, 1]` picking where in the band the result lands.
///
/// The jitter is what stops every machine that lost the relay at the same
/// moment from reconnecting in lockstep and stampeding it on the way back.
pub fn jittered(delay: Duration, unit: f64) -> Duration {
    let factor = 1.0 - JITTER + 2.0 * JITTER * unit.clamp(0.0, 1.0);
    Duration::from_secs_f64(delay.as_secs_f64() * factor)
}

/// Where in `[0, 1)` a wait should land: the clock's sub-second part, which is
/// as much randomness as this needs and all this crate can reach without a new
/// dependency.
fn jitter_unit() -> f64 {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|since| since.subsec_nanos())
        .unwrap_or(0);
    f64::from(nanos) / 1_000_000_000.0
}

/// One audit line: the command name, the body size, and a fingerprint — never
/// the body itself.
///
/// The audit log is written to stderr, which on a served machine lands in a
/// terminal, a service log or someone's scrollback. A prompt is the private
/// part of a turn; a hash answers "was this the same command" without writing
/// it down.
pub fn audit_line(body: &str) -> String {
    let cmd = serde_json::from_str::<serde_json::Value>(body)
        .ok()
        .and_then(|value| {
            value
                .get("cmd")
                .and_then(|c| c.as_str())
                .map(str::to_string)
        })
        .unwrap_or_else(|| "?".to_string());
    let digest = sha1_hex(body.as_bytes());
    format!("cmd={cmd} bytes={} sha1={}", body.len(), &digest[..8])
}

/// sha1 of a body, lowercase hex.
///
/// Computed here rather than pulled from the kernel: the `sha1` crate is a
/// dependency of the root crate and is not reachable from this one, and adding
/// it would mean touching the manifest. It is a fingerprint for a log line,
/// never a check, so the algorithm's age costs nothing. Tested against the
/// RFC 3174 vectors below.
fn sha1_hex(bytes: &[u8]) -> String {
    let mut state: [u32; 5] = [0x6745_2301, 0xEFCD_AB89, 0x98BA_DCFE, 0x1032_5476, 0xC3D2_E1F0];
    let mut message = bytes.to_vec();
    let bit_length = (bytes.len() as u64).wrapping_mul(8);
    message.push(0x80);
    while message.len() % 64 != 56 {
        message.push(0);
    }
    message.extend_from_slice(&bit_length.to_be_bytes());

    for block in message.chunks_exact(64) {
        let mut words = [0u32; 80];
        for (index, word) in block.chunks_exact(4).enumerate() {
            words[index] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for index in 16..80 {
            words[index] =
                (words[index - 3] ^ words[index - 8] ^ words[index - 14] ^ words[index - 16])
                    .rotate_left(1);
        }
        let (mut a, mut b, mut c, mut d, mut e) =
            (state[0], state[1], state[2], state[3], state[4]);
        for (index, word) in words.iter().enumerate() {
            let (f, k) = match index {
                0..=19 => ((b & c) | (!b & d), 0x5A82_7999),
                20..=39 => (b ^ c ^ d, 0x6ED9_EBA1),
                40..=59 => ((b & c) | (b & d) | (c & d), 0x8F1B_BCDC),
                _ => (b ^ c ^ d, 0xCA62_C1D6),
            };
            let temp = a
                .rotate_left(5)
                .wrapping_add(f)
                .wrapping_add(e)
                .wrapping_add(k)
                .wrapping_add(*word);
            e = d;
            d = c;
            c = b.rotate_left(30);
            b = a;
            a = temp;
        }
        state[0] = state[0].wrapping_add(a);
        state[1] = state[1].wrapping_add(b);
        state[2] = state[2].wrapping_add(c);
        state[3] = state[3].wrapping_add(d);
        state[4] = state[4].wrapping_add(e);
    }

    state.iter().fold(String::with_capacity(40), |mut hex, word| {
        hex.push_str(&format!("{word:08x}"));
        hex
    })
}

/// The relay a dial is aimed at, with any query or fragment removed.
///
/// A relay URL may carry a token in its query string (`wss://host/ws?t=…`), and
/// this line goes to a log file. The scheme, host and path are what answer
/// "which relay was this", and they are all the log needs.
fn relay_target(url: &str) -> String {
    let base = url.split(['?', '#']).next().unwrap_or(url);
    base.trim().to_string()
}

/// The first eight hex digits of the token's sha1, for telling credentials
/// apart in a log without writing the credential down.
fn token_fingerprint(token: &str) -> String {
    let digest = sha1_hex(token.as_bytes());
    digest[..8].to_string()
}

/// Dial the relay and serve commands until the connection ends, then retry.
///
/// Each received command goes through [`admit`] before it is handed on. A
/// refused command gets an error reply and is never executed. The loop is
/// bounded in three ways: retries back off with jitter to a fixed ceiling, a
/// single command cannot hold the frame loop past [`COMMAND_TIMEOUT`], and a
/// relay that keeps failing within [`FAST_FAILURE`] stops being retried at
/// [`MAX_FAST_FAILURES`] — at that point something needs a human, and a silent
/// reconnect loop would only hide it.
pub async fn run_relay(
    config: RelayConfig,
    mut execute: impl FnMut(String) -> std::pin::Pin<Box<dyn std::future::Future<Output = String> + Send>> + Send + 'static,
) {
    use futures_util::{SinkExt, StreamExt};
    use tokio_tungstenite::tungstenite::{client::IntoClientRequest, Message};

    let prompt_allowed = allow_prompt();
    if prompt_allowed {
        eprintln!("[remote] prompt over the relay is enabled by local settings");
    } else {
        eprintln!("[remote] prompt over the relay is disabled on this machine");
    }
    // One window for the whole process, not one per connection: reconnecting
    // must not hand the relay a fresh budget.
    let mut window = CommandWindow::new();

    let mut delay = std::time::Duration::from_secs(1);
    let ceiling = std::time::Duration::from_secs(60);
    let mut fast_failures = 0u32;
    loop {
        let started = Instant::now();
        let attempt = async {
            // Which relay, and which credential: after the fact this is the
            // only way to tell "the token was rotated" from "the relay moved",
            // and neither the token nor a token in the URL's query is written
            // down.
            eprintln!(
                "[remote] dialing {} (token sha1 {})",
                relay_target(&config.url),
                token_fingerprint(&config.token)
            );
            let mut request = config.url.as_str().into_client_request()?;
            request.headers_mut().insert(
                "authorization",
                format!("Bearer {}", config.token).parse()?,
            );
            let (mut socket, _) = tokio_tungstenite::connect_async(request).await?;
            eprintln!("[remote] connected to relay");
            while let Some(message) = socket.next().await {
                let Message::Text(frame) = message? else { continue };
                // The relay wraps each command as `{"id","body"}` and expects the
                // same `id` back in `{"id","reply"}`, so replies pair with requests.
                let Some(envelope) = RelayFrame::parse(&frame) else {
                    eprintln!("[remote] audit: malformed frame refused");
                    continue;
                };
                let reply = match admit(&envelope.body, prompt_allowed, &mut window, Instant::now()) {
                    Admission::Allowed(cmd) => {
                        eprintln!("[remote] audit: executing {}", audit_line(&envelope.body));
                        // One command must not park the frame loop: the relay
                        // reads a silent socket as a dead machine and would
                        // drop a link that is merely busy.
                        match tokio::time::timeout(COMMAND_TIMEOUT, execute(envelope.body.clone())).await {
                            Ok(reply) => reply,
                            Err(_) => {
                                eprintln!(
                                    "[remote] audit: {cmd} timed out after {} s",
                                    COMMAND_TIMEOUT.as_secs()
                                );
                                r#"{"ok":false,"error":"timeout"}"#.to_string()
                            }
                        }
                    }
                    refusal => {
                        let error = refusal.error().unwrap_or_else(|| "command not allowed".to_string());
                        eprintln!(
                            "[remote] audit: refused {error} ({}, {} refused by the rate window so far)",
                            audit_line(&envelope.body),
                            window.rejected()
                        );
                        serde_json::json!({ "ok": false, "error": error }).to_string()
                    }
                };
                let answer = serde_json::json!({ "id": envelope.id, "reply": reply }).to_string();
                socket.send(Message::Text(answer.into())).await?;
            }
            anyhow::Ok(())
        };
        let outcome = attempt.await;
        // A link that dies immediately is not the same as one that ran for an
        // hour and then dropped: the first is a relay that refuses this
        // machine, the second is a restart worth reconnecting to.
        let fast = started.elapsed() < FAST_FAILURE;
        match outcome {
            Ok(()) => {
                eprintln!("[remote] relay closed the link");
                delay = std::time::Duration::from_secs(1);
            }
            Err(error) => {
                eprintln!("[remote] relay link ended: {error}");
                delay = (delay * 2).min(ceiling);
            }
        }
        fast_failures = if fast { fast_failures + 1 } else { 0 };
        if fast_failures >= MAX_FAST_FAILURES {
            eprintln!(
                "[remote] {fast_failures} connection attempts ended within {} s of starting: \
                 remote control is stopped and needs manual recovery (check the relay URL and \
                 token, then restart the app)",
                FAST_FAILURE.as_secs()
            );
            return;
        }
        // Jittered even after a clean close: a relay that accepts and closes
        // at once would otherwise be reconnected in a tight one-second loop.
        let wait = jittered(delay, jitter_unit());
        tokio::time::sleep(wait).await;
    }
}

#[cfg(test)]
mod config_tests {
    use super::*;

    #[test]
    fn remote_control_stays_off_without_both_values() {
        assert!(disabled_reason(None, Some("t")).is_some());
        assert!(disabled_reason(Some("wss://relay"), None).is_some());
        assert!(disabled_reason(Some("  "), Some("t")).is_some());
    }

    #[test]
    fn remote_control_is_enabled_only_with_url_and_token() {
        assert!(disabled_reason(Some("wss://relay"), Some("t")).is_none());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allows_turn_and_status_commands() {
        assert!(is_allowed(r#"{"cmd":"prompt","text":"hi"}"#));
        assert!(is_allowed(r#"{"cmd":"status"}"#));
    }

    #[test]
    fn rejects_destructive_and_unknown_commands() {
        assert!(!is_allowed(r#"{"cmd":"delete_session","id":"x"}"#));
        assert!(!is_allowed(r#"{"cmd":"save_settings"}"#));
        assert!(!is_allowed(r#"{"cmd":"uninstall_pack"}"#));
    }

    #[test]
    fn rejects_malformed_bodies() {
        assert!(!is_allowed("not json"));
        assert!(!is_allowed(r#"{"text":"no cmd"}"#));
        assert!(!is_allowed(r#"["cmd","prompt"]"#));
    }
}

#[cfg(test)]
mod gate_tests {
    use super::*;

    /// A fresh window and a body that names an allowed command.
    fn body(cmd: &str) -> String {
        format!(r#"{{"cmd":"{cmd}"}}"#)
    }

    /// The prompt gate is the whole of REMOTE-13's first half: off unless the
    /// local switch says otherwise, and the refusal says which policy refused.
    #[test]
    fn prompt_is_refused_unless_the_local_switch_is_on() {
        let mut window = CommandWindow::new();
        let now = Instant::now();
        let refused = admit(&body("prompt"), false, &mut window, now);
        assert_eq!(refused, Admission::PromptDisabled);
        assert_eq!(
            refused.error().as_deref(),
            Some("prompt over the relay is disabled on this machine")
        );
        // Everything else that was allowed before still is: the switch is
        // about starting turns, not about remote control as a whole.
        assert_eq!(
            admit(&body("status"), false, &mut window, now),
            Admission::Allowed("status".to_string())
        );
        assert_eq!(
            admit(&body("prompt"), true, &mut window, now),
            Admission::Allowed("prompt".to_string())
        );
    }

    /// The window is a sliding one: the 31st frame inside the minute is
    /// refused, and the same frame is fine once the oldest has aged out.
    #[test]
    fn the_rate_window_slides_at_thirty_per_minute() {
        let mut window = CommandWindow::new();
        let start = Instant::now();
        for index in 0..MAX_COMMANDS_PER_MINUTE {
            assert_eq!(
                admit(&body("status"), false, &mut window, start + Duration::from_millis(index as u64)),
                Admission::Allowed("status".to_string())
            );
        }
        assert_eq!(
            admit(&body("status"), false, &mut window, start + Duration::from_millis(30)),
            Admission::RateLimited
        );
        assert_eq!(window.rejected(), 1);
        // Still inside the window a moment later.
        assert_eq!(
            admit(&body("status"), false, &mut window, start + RATE_WINDOW - Duration::from_millis(1)),
            Admission::RateLimited
        );
        assert_eq!(window.rejected(), 2);
        // The first frame is now older than the window, so one slot is free.
        assert_eq!(
            admit(&body("status"), false, &mut window, start + RATE_WINDOW),
            Admission::Allowed("status".to_string())
        );
        // …and only one: the other twenty-nine are still inside it.
        assert_eq!(
            admit(&body("status"), false, &mut window, start + RATE_WINDOW),
            Admission::RateLimited
        );
    }

    /// A flood counts against the window even when each frame is also refused
    /// for its size — otherwise oversized frames would be free to send.
    #[test]
    fn oversized_bodies_are_refused_and_still_charged_to_the_window() {
        let mut window = CommandWindow::new();
        let start = Instant::now();
        let oversized = format!(r#"{{"cmd":"status","pad":"{}"}}"#, "x".repeat(MAX_BODY_BYTES));
        assert!(oversized.len() > MAX_BODY_BYTES);
        match admit(&oversized, true, &mut window, start) {
            Admission::TooLong { bytes } => assert_eq!(bytes, oversized.len()),
            other => panic!("expected TooLong, got {other:?}"),
        }
        // The refused frame still occupies a slot.
        for index in 1..MAX_COMMANDS_PER_MINUTE {
            assert!(matches!(
                admit(
                    &body("status"),
                    false,
                    &mut window,
                    start + Duration::from_millis(index as u64)
                ),
                Admission::Allowed(_)
            ));
        }
        assert_eq!(
            admit(&body("status"), false, &mut window, start + Duration::from_millis(31)),
            Admission::RateLimited
        );
        // A body of exactly the limit is not over it.
        let prefix = r#"{"cmd":"status","p":""#;
        let suffix = r#""}"#;
        let exact = format!(
            "{prefix}{}{suffix}",
            "x".repeat(MAX_BODY_BYTES - prefix.len() - suffix.len())
        );
        assert_eq!(exact.len(), MAX_BODY_BYTES);
        assert_eq!(
            admit(&exact, false, &mut CommandWindow::new(), start),
            Admission::Allowed("status".to_string())
        );
    }

    /// `open_session` is the one command that can move the working directory
    /// (and with it what `/file` may read), so it may only name a session that
    /// already exists here.
    #[test]
    fn open_session_names_a_session_or_nothing() {
        assert_eq!(
            open_session_id(r#"{"cmd":"open_session","id":"20260101-000000-abc123"}"#),
            Some("20260101-000000-abc123".to_string())
        );
        // No id at all matches no stored session, so the caller refuses it.
        assert_eq!(
            open_session_id(r#"{"cmd":"open_session"}"#),
            Some(String::new())
        );
        assert_eq!(
            open_session_id(r#"{"cmd":"open_session","id":7}"#),
            Some(String::new())
        );
        assert_eq!(open_session_id(r#"{"cmd":"new_session"}"#), None);
        assert_eq!(open_session_id("not json"), None);
    }

    /// The switch is read strictly: only a JSON boolean `true` turns it on.
    #[test]
    fn the_prompt_flag_is_a_strict_boolean() {
        assert!(prompt_flag_from_json(r#"{"relay_allow_prompt":true}"#));
        assert!(!prompt_flag_from_json(r#"{"relay_allow_prompt":false}"#));
        assert!(!prompt_flag_from_json(r#"{"relay_allow_prompt":"true"}"#));
        assert!(!prompt_flag_from_json(r#"{"relay_allow_prompt":1}"#));
        assert!(!prompt_flag_from_json("{}"));
        assert!(!prompt_flag_from_json("not json"));
        assert!(!prompt_flag_from_json(""));
    }

    /// Jitter stays inside ±30%, and the ends of the band are the specified
    /// ±30% — a band that silently widened would be a retry storm waiting to
    /// happen.
    #[test]
    fn retry_jitter_stays_within_thirty_percent() {
        let delay = Duration::from_secs(10);
        // Compared with a tolerance: the band is built from floats, and the
        // question here is the band, not the last bit of a duration.
        for (unit, expected) in [(0.0, 7.0), (0.5, 10.0), (1.0, 13.0)] {
            let waited = jittered(delay, unit).as_secs_f64();
            assert!(
                (waited - expected).abs() < 1e-9,
                "jitter {unit} produced {waited} s, expected {expected} s"
            );
        }
        for step in 0..=100 {
            let unit = f64::from(step) / 100.0;
            let waited = jittered(delay, unit).as_secs_f64();
            assert!(
                (7.0..=13.0).contains(&waited),
                "jitter {unit} produced {waited} s, outside ±30% of 10 s"
            );
        }
        // Out-of-band input is clamped, not extrapolated.
        assert!((jittered(delay, -5.0).as_secs_f64() - 7.0).abs() < 1e-9);
        assert!((jittered(delay, 5.0).as_secs_f64() - 13.0).abs() < 1e-9);
    }

    /// The audit line identifies a command without writing it down: two bodies
    /// that differ only in their private text share a command name and length,
    /// and differ in the fingerprint.
    #[test]
    fn audit_lines_fingerprint_instead_of_quoting() {
        let secret = r#"{"cmd":"prompt","text":"my private question"}"#;
        let other = r#"{"cmd":"prompt","text":"my private questiom"}"#;
        let line = audit_line(secret);
        assert!(line.starts_with("cmd=prompt bytes="), "{line}");
        assert!(!line.contains("private"), "the body leaked into the audit line: {line}");
        let digest = line.rsplit("sha1=").next().expect("fingerprint");
        assert_eq!(digest.len(), 8);
        assert!(digest.chars().all(|c| c.is_ascii_hexdigit()));
        assert_eq!(audit_line(secret), line);
        assert_ne!(audit_line(other).rsplit("sha1=").next(), Some(digest));
        // A body that is not JSON at all still gets a line.
        assert!(audit_line("not json").starts_with("cmd=? bytes=8 sha1="));
    }

    /// RFC 3174 (and the widely published) vectors, so a broken digest cannot
    /// pass as a fingerprint.
    #[test]
    fn sha1_matches_the_published_vectors() {
        for (input, expected) in [
            ("", "da39a3ee5e6b4b0d3255bfef95601890afd80709"),
            ("abc", "a9993e364706816aba3e25717850c26c9cd0d89d"),
            ("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
             "84983e441c3bd26ebaae4aa1f95129e5e54670f1"),
        ] {
            assert_eq!(sha1_hex(input.as_bytes()), expected, "sha1({input:?})");
        }
        // The 1 000 000 × 'a' vector, which exercises multi-block padding.
        assert_eq!(
            sha1_hex("a".repeat(1_000_000).as_bytes()),
            "34aa973cd4c4daa4f61eeb2bdbad27316534016f"
        );
    }
}

#[cfg(test)]
mod log_tests {
    use super::{relay_target, sha1_hex, token_fingerprint};

    /// The log line says which relay and which credential, and neither the
    /// token nor a token hidden in the URL's query is ever written down.
    #[test]
    fn the_dial_line_names_the_target_and_fingerprints_the_token() {
        assert_eq!(relay_target("wss://relay.example/ws"), "wss://relay.example/ws");
        // A URL that carries its own secret keeps only the part that identifies
        // the relay.
        assert_eq!(
            relay_target("wss://relay.example/ws?token=s3cret"),
            "wss://relay.example/ws"
        );
        assert_eq!(relay_target("wss://relay.example/ws#frag"), "wss://relay.example/ws");
        assert_eq!(relay_target("  wss://relay.example  "), "wss://relay.example");

        let fingerprint = token_fingerprint("a-relay-token");
        assert_eq!(fingerprint.len(), 8);
        assert!(fingerprint.chars().all(|c| c.is_ascii_hexdigit()));
        assert_eq!(fingerprint, &sha1_hex(b"a-relay-token")[..8]);
        // Stable for one token, different for another, and not the token.
        assert_eq!(token_fingerprint("a-relay-token"), fingerprint);
        assert_ne!(token_fingerprint("another-relay-token"), fingerprint);
        assert!(!fingerprint.contains("relay"));
    }
}
