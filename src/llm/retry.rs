use super::*;

/// Whether a chat failure is worth another attempt.
///
/// Walks the whole cause chain, because a classification placed on the root
/// error must still be visible after a `POST <url>` context is layered on top.
pub fn retryable(error: &anyhow::Error) -> bool {
    error.chain().any(|cause| {
        if let Some(failure) = cause.downcast_ref::<RequestError>() {
            return failure.retryable;
        }
        // Every transport-level failure is worth another ask: connection
        // refused, reset, TLS torn down, DNS blip, a body that stopped early.
        // None of them say anything about the request's contents, and none of
        // them mean an answer was delivered — the request never produced one.
        // This used to retry only timeouts, so a dropped connection ended a
        // turn (and with it a running task) on the first hiccup.
        //
        // A builder error is different: a malformed URL cannot start working,
        // so it stays final.
        if let Some(transport) = cause.downcast_ref::<reqwest::Error>() {
            return !transport.is_builder();
        }
        false
    })
}

/// Whether an HTTP status is worth sending the same request again.
///
/// Rate limiting and a broken upstream are the two families every gateway
/// retries on its own, and the third is a request the gateway itself could not
/// take: 408 Request Timeout and 425 Too Early both mean the server declined
/// before the answer began, so repeating the request cannot duplicate an answer
/// the caller has already seen. One function because three paths classify the
/// same response — the streaming open, the buffered open, and an error object
/// carried inside a successful one — and they used to hold three copies of
/// `429 || 5xx`: a 408 was fatal on all of them, and fixing that in three
/// places is how the copies drift apart again.
pub(crate) fn retryable_status(code: u16) -> bool {
    matches!(code, 408 | 425 | 429) || (500..=599).contains(&code)
}

/// Whether a provider error message describes a failure no retry can fix.
///
/// Used when the provider reports an error without a numeric status: retrying
/// a request the API rejected on its merits only repeats the rejection.
pub fn looks_fatal(message: &str) -> bool {
    let lower = message.to_lowercase();
    [
        "context_length",
        "context length",
        "invalid_request",
        "invalid request",
        "unauthorized",
        "authentication",
        "not_found",
        "insufficient_quota",
        "quota exceeded",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
}

/// Whether an error is the peer closing a connection mid-request.
///
/// reqwest reports this as a body or request error rather than a connect error,
/// because the connection was established and then went away. The distinction
/// matters: it is the one failure that is certainly safe to retry.
///
/// Shared with [`crate::fetch`], which faces the same pooled-connection race
/// while downloading and must classify it the same way.
pub(crate) fn is_connection_closed(error: &reqwest::Error) -> bool {
    error_chain_matches(error, matches_closed_text)
}

/// Whether TLS records arrived that this session's keys cannot open.
///
/// rustls raises these after a completed handshake, so reqwest reports a request
/// failure rather than a connect error. Nothing of the response has been read
/// when they surface and the connection is unusable behind them, which puts the
/// next attempt on a fresh session: retryable on the same terms as
/// [`is_connection_closed`].
pub(crate) fn is_tls_stream_damage(error: &reqwest::Error) -> bool {
    error_chain_matches(error, matches_tls_damage_text)
}

/// Test `pred` against every link of a reqwest error's cause chain.
///
/// The text that identifies these failures is below the outermost error:
/// reqwest reports "error sending request for url (...)" and the actionable
/// words — "connection closed before message completed", "cannot decrypt peer's
/// message" — arrive one or more levels down, so testing only
/// `error.to_string()` classifies every transport loss as unretryable.
pub fn error_chain_matches(error: &reqwest::Error, pred: impl Fn(&str) -> bool) -> bool {
    let mut current: Option<&(dyn std::error::Error + 'static)> = Some(error);
    while let Some(link) = current {
        if pred(&link.to_string()) {
            return true;
        }
        current = link.source();
    }
    false
}

/// The message test behind [`is_connection_closed`].
///
/// Separated from the error type so it can be exercised against the exact
/// strings a peer produces, which is the whole of the decision.
pub fn matches_closed_text(text: &str) -> bool {
    text.contains("close_notify")
        || text.contains("connection closed")
        || text.contains("Connection reset")
        || text.contains("connection reset")
        || text.contains("broken pipe")
        || text.contains("IncompleteMessage")
        || os_error_is_connection_loss(text)
}

/// The message test behind [`is_tls_stream_damage`].
///
/// These are rustls's own words for a record whose integrity check fails or
/// whose declared length does not describe a record. They come from an endpoint
/// that tears a session down mid-stream, or from an intermediary that writes
/// another session's bytes onto this one — the two ways a proxy chain in front
/// of a model API breaks TLS without breaking TCP.
///
/// Matching is case-insensitive because the same failure reaches the chain as
/// both a rustls `Display` string and a lower-cased `source` sentence.
pub fn matches_tls_damage_text(text: &str) -> bool {
    const DAMAGED: [&str; 7] = [
        "cannot decrypt peer's message",
        "couldn't decrypt",
        "decrypt failed",
        "bad record mac",
        "corrupt message",
        "record overflow",
        "illegal packet length",
    ];
    let lower = text.to_lowercase();
    DAMAGED.iter().any(|needle| lower.contains(needle))
}

/// Whether the message carries an OS error code for a connection that went
/// away.
///
/// Windows writes that sentence in the system language — "你的主机中的软件中止了
/// 一个已建立的连接。(os error 10053)" on a Chinese machine — so no English
/// phrase matches it there. The number after "os error" is the same on every
/// machine, which is what makes the classification survive a translation.
pub fn os_error_is_connection_loss(text: &str) -> bool {
    const LOST: [u32; 5] = [
        10053, // WSAECONNABORTED: the connection was aborted
        10054, // WSAECONNRESET: the peer reset it
        103,   // ECONNABORTED
        104,   // ECONNRESET
        32,    // EPIPE: writing to a socket the peer closed
    ];

    let mut rest = text;
    while let Some(at) = rest.find("os error ") {
        rest = &rest[at + "os error ".len()..];
        let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
        // `take_while` stops at a non-digit, so "os error 10060" is read as
        // 10060 rather than as 1006 with a trailing digit.
        if let Ok(code) = digits.parse::<u32>() {
            if LOST.contains(&code) {
                return true;
            }
        }
    }
    false
}

