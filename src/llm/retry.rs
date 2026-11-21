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

#[cfg(test)]
mod retry_tests {
    use super::*;

    /// Build a reqwest error of the kind the peer-closed path produces.
    fn error_from(message: &str) -> reqwest::Error {
        // A request to a port nothing listens on gives a real connect error;
        // the classification below is what is under test, so the exact variant
        // matters less than the message it carries.
        //
        // `no_proxy`, or the machine's `HTTP_PROXY` would route this loopback
        // address somewhere that answers — turning a refused connection into a
        // successful response and making the test depend on the environment.
        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        let url = "http://127.0.0.1:1/";
        let error = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(async { client.get(url).send().await.unwrap_err() });
        let _ = message;
        error
    }

    #[test]
    fn a_connect_error_is_recognised() {
        let error = error_from("connection refused");
        assert!(error.is_connect(), "{error}");
    }

    #[test]
    fn the_close_notify_message_is_recognised_as_closed() {
        // The exact string a peer that closes without a TLS shutdown produces.
        // It cannot be built from a real reqwest error here, so the predicate
        // is exercised through the messages it must match.
        for text in [
            "peer closed connection without sending TLS close_notify",
            "connection closed before message completed",
            "Connection reset by peer",
            "broken pipe",
        ] {
            assert!(
                matches_closed_text(text),
                "{text:?} should be treated as a closed connection"
            );
        }
    }

    #[test]
    fn an_unrelated_failure_is_not_treated_as_closed() {
        for text in [
            "provider returned 400 Bad Request",
            "invalid api key",
            "operation timed out",
        ] {
            assert!(
                !matches_closed_text(text),
                "{text:?} must not be retried"
            );
        }
    }

    #[test]
    fn a_tls_record_the_session_cannot_open_is_recognised_as_damage() {
        // The sentences rustls puts on the chain when bytes arrive that its
        // keys cannot open, including the one a proxy chain in front of the
        // provider produces when it tears a session down mid-stream.
        for text in [
            "connection error: cannot decrypt peer's message",
            "received corrupt message",
            "Failed to create packet: record overflow",
            "illegal packet length",
            "couldn't decrypt message",
        ] {
            assert!(
                matches_tls_damage_text(text),
                "{text:?} should be treated as a lost connection"
            );
        }
    }

    #[test]
    fn a_request_the_provider_rejected_is_not_tls_damage() {
        // Retrying these repeats the rejection. A bare "decrypt" appearing in
        // unrelated prose must not pull them into the retryable set, which is
        // why the needles above are whole rustls sentences.
        for text in [
            "provider returned 400 Bad Request",
            "invalid api key",
            "operation timed out",
            "unable to decrypt the local session store",
            "connection closed before message completed",
        ] {
            assert!(
                !matches_tls_damage_text(text),
                "{text:?} must not be classified as TLS damage"
            );
        }
    }

    #[test]
    fn a_translated_connection_error_is_still_recognised() {
        // On a Chinese Windows the sentence is translated and only the code is
        // stable, so a machine that reads English and one that does not must
        // classify the same failure the same way.
        for text in [
            "你的主机中的软件中止了一个已建立的连接。(os error 10053)",
            "另一端强行关闭了一个现有的连接。(os error 10054)",
            "Connection reset by peer (os error 10054)",
        ] {
            assert!(matches_closed_text(text), "{text:?} must be retried");
        }
        // A timeout is translated too, and it must stay unretryable: the
        // request may already have been answered.
        assert!(
            !matches_closed_text("由于连接方在一段时间后没有反应...(os error 10060)"),
            "a timed-out connection must not be retried"
        );
    }

    #[tokio::test]
    async fn a_connection_dropped_before_the_response_is_retried() {
        // A server that closes the first connection without answering and
        // answers every one after. This is the shape of the failure a peer
        // produces when it discards a pooled connection, and the retry is the
        // whole of the defence against it.
        //
        // It keeps accepting rather than answering exactly the second: under a
        // parallel test run the client may make its first attempt before the
        // listener is ready, so counting connections makes the test depend on
        // scheduling. Answering everything after the first does not.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        tokio::spawn(async move {
            let mut first = true;
            while let Ok((mut socket, _)) = listener.accept().await {
                if first {
                    first = false;
                    // Closed without a response, which is what a reaped
                    // connection looks like from the client's side.
                    drop(socket);
                    continue;
                }
                use tokio::io::AsyncWriteExt;
                let body = "{\"ok\":true}";
                let response = format!(
                    "HTTP/1.1 200 OK\r\ncontent-length: {}\r\ncontent-type: application/json\r\n\r\n{body}",
                    body.len()
                );
                let _ = socket.write_all(response.as_bytes()).await;
                let _ = socket.flush().await;
                // Held open until the client has read it; closing immediately
                // can truncate the response instead of ending it cleanly.
                tokio::time::sleep(std::time::Duration::from_secs(2)).await;
            }
        });

        let config = LlmConfig::new(format!("http://{addr}"), "k", "m");
        let client = LlmClient::new(config).unwrap();
        let url = format!("http://{addr}/x");

        let response = client
            .send_retrying(|| client.http.get(&url))
            .await
            .expect("the retry should recover");

        assert_eq!(response.status(), 200);
        assert_eq!(response.text().await.unwrap(), "{\"ok\":true}");
    }

    #[tokio::test]
    async fn a_server_that_always_drops_fails_after_the_attempts_run_out() {
        // Retrying must not become an infinite loop against a server that is
        // simply down.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        tokio::spawn(async move {
            while let Ok((socket, _)) = listener.accept().await {
                drop(socket);
            }
        });

        let config = LlmConfig::new(format!("http://{addr}"), "k", "m");
        let client = LlmClient::new(config).unwrap();
        let url = format!("http://{addr}/x");

        let started = std::time::Instant::now();
        let _ = client
            .send_retrying(|| client.http.get(&url))
            .await
            .expect_err("it must give up");
        assert!(
            started.elapsed() < std::time::Duration::from_secs(10),
            "giving up took {:?}",
            started.elapsed()
        );
    }

    #[test]
    fn a_retry_classification_survives_added_context() {
        // chat_stream labels its failures with the URL it posted to; the loop
        // must still read the classification the failure carries underneath.
        let busy = RequestError {
            status: Some(503),
            retryable: true,
            message: "overloaded".into(),
        };
        let error =
            Result::<(), RequestError>::Err(busy)
                .context("POST http://x/chat/completions")
                .unwrap_err();
        assert!(retryable(&error));

        let rejected = RequestError {
            status: Some(401),
            retryable: false,
            message: "bad key".into(),
        };
        let error =
            Result::<(), RequestError>::Err(rejected)
                .context("POST http://x/chat/completions")
                .unwrap_err();
        assert!(!retryable(&error));
    }

    #[test]
    fn an_unclassified_error_is_not_retried() {
        // Unknown failures are not given the benefit of the doubt: a retry
        // loop that invents retryable failures turns a bug into a hang.
        let error = anyhow::anyhow!("unexpected failure");
        assert!(!retryable(&error));
    }

    #[tokio::test]
    async fn a_dropped_connection_is_worth_another_ask() {
        // A name that cannot resolve stands in for a gateway that went away: a
        // transport failure that is neither a timeout nor a builder error. The
        // old rule retried only timeouts, so this ended the turn — and with it
        // whatever task the turn was holding.
        let error = reqwest::Client::builder()
            .timeout(std::time::Duration::from_millis(5000))
            .build()
            .expect("a client")
            .get("http://drops-this-connection.invalid/never")
            .send()
            .await
            .expect_err("this name cannot resolve");
        assert!(!error.is_builder(), "not a builder error: {error}");
        assert!(
            !error.is_timeout(),
            "not a timeout either — the old rule already retried those: {error}"
        );
        let error = anyhow::Error::new(error);
        assert!(
            retryable(&error),
            "a dropped connection must be retryable: {error:#}"
        );
    }


    #[test]
    fn a_fatal_provider_message_ends_the_step_instead_of_retrying() {
        // Mid-stream provider errors often arrive without a numeric status;
        // the message is the only thing left to classify on.
        assert!(looks_fatal("context_length_exceeded for this model"));
        assert!(looks_fatal("invalid request: tools schema"));
        assert!(looks_fatal("unauthorized"));
        assert!(looks_fatal("insufficient_quota"));
        assert!(!looks_fatal("The model is overloaded, please try again"));
        assert!(!looks_fatal("rate_limit_exceeded"));
    }

    #[test]
    fn a_window_is_read_from_every_shape_a_provider_reports() {
        // One sentence per wire dialect: the prose OpenAI and Anthropic-shaped
        // gateways send, the terse field names vLLM and llama.cpp send, and the
        // same number written with the underscore separators a provider uses
        // when it prints a value from its own configuration.
        for (message, tokens) in [
            ("This model's maximum context length is 8192 tokens", 8192),
            ("context length is 128000", 128_000),
            ("maximum context window: 65536", 65_536),
            ("max_model_len=32768", 32_768),
            ("this model's max_model_len = 32_768", 32_768),
            ("Context Window: 1_000_000 exceeded", 1_000_000),
            (
                "This model's maximum context length is 8192 tokens. However, you requested 20000 tokens",
                8192,
            ),
        ] {
            assert_eq!(reported_window(message), Some(tokens), "{message:?}");
        }
    }

    #[test]
    fn a_number_that_is_not_about_the_context_window_is_not_a_window() {
        // The rejected request's own token count sits in the same message, is
        // often larger than the window, and is the number a naive "first big
        // number" parser would take. Sizing later prompts to it would keep the
        // loop building requests the provider still refuses.
        assert_eq!(reported_window("tokens: 123456"), None);
        assert_eq!(reported_window("you sent 200000 tokens in one request"), None);
        assert_eq!(
            reported_window("maximum context length is 8192 tokens; you sent 200000 tokens"),
            Some(8192),
            "the number after the phrase wins, not the larger one later on"
        );
    }

    #[test]
    fn an_implausible_or_absent_window_reads_as_no_window() {
        // Every refusal here matters: a wrong report silently resizes every
        // later prompt, so anything short of a plausible number is `None`.
        assert_eq!(reported_window(""), None);
        assert_eq!(reported_window("   "), None);
        assert_eq!(reported_window("context length exceeded"), None);
        assert_eq!(
            reported_window("context window: 42"),
            None,
            "a two-digit number is a token count or an id, not a window"
        );
        assert_eq!(
            reported_window("context window: 99999999999"),
            None,
            "a window larger than any provider sells is a byte count"
        );
        assert_eq!(reported_window("invalid request: tools schema"), None);
        assert_eq!(reported_window("上下文长度超出限制"), None);
    }
}
