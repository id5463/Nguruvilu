//! Cloud-to-local remote control, with the local machine as the only party
//! that dials out.
//!
//! The local app connects outward to the cloud relay and waits for commands.
//! Nothing listens on a local port for this. Every command passes the
//! allow-list below before it reaches the shell dispatcher, and every executed
//! command is written to the audit log.

/// Commands the cloud may run. Read-only status and turn-starting commands only.
/// Destructive commands (delete, settings writes, pack uninstall, shell exit)
/// stay local-only on purpose.
pub const ALLOWED_COMMANDS: &[&str] = &["ready", "status", "new_session", "open_session", "prompt", "cancel"];

/// Whether a remote command body may run. Rejects anything that is not one
/// JSON object with a `cmd` field from [`ALLOWED_COMMANDS`].
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

/// Dial the relay and serve commands until the connection ends, then retry.
///
/// Each received command is checked against [`is_allowed`] before it is
/// handed on. A refused command gets an error reply and is never executed.
/// The loop is bounded: retries back off to a fixed ceiling and never stop
/// the local app.
pub async fn run_relay(
    config: RelayConfig,
    mut execute: impl FnMut(String) -> std::pin::Pin<Box<dyn std::future::Future<Output = String> + Send>> + Send + 'static,
) {
    use futures_util::{SinkExt, StreamExt};
    use tokio_tungstenite::tungstenite::{client::IntoClientRequest, Message};

    let mut delay = std::time::Duration::from_secs(1);
    let ceiling = std::time::Duration::from_secs(60);
    loop {
        let attempt = async {
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
                let reply = if is_allowed(&envelope.body) {
                    eprintln!("[remote] audit: executing {}", envelope.body);
                    execute(envelope.body.clone()).await
                } else {
                    eprintln!("[remote] audit: refused {}", envelope.body);
                    r#"{"ok":false,"error":"command not allowed"}"#.to_string()
                };
                let answer = serde_json::json!({ "id": envelope.id, "reply": reply }).to_string();
                socket.send(Message::Text(answer.into())).await?;
            }
            anyhow::Ok(())
        };
        match attempt.await {
            Ok(()) => delay = std::time::Duration::from_secs(1),
            Err(error) => eprintln!("[remote] relay link ended: {error}"),
        }
        tokio::time::sleep(delay).await;
        delay = (delay * 2).min(ceiling);
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
