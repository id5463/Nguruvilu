use crate::llm::{LlmClient, PartialCallView, TokenUsage};
use crate::message::{Message, ToolCall};
use crate::tools::ToolRegistry;
use crate::window::ContextPolicy;
use anyhow::Result;
use std::collections::VecDeque;
use std::sync::Arc;

use super::config::{StaticConfig, TurnConfig, TurnSettings};
use super::observer::AgentObserver;
use super::r#loop::{Agent, AgentLoop};
use super::state::{
    summarize_tool_error, AgentOutcome, LoopState, StepOutput, TurnTiming, CONTINUATION_BATCHES,
    MAX_TOOL_ERROR_BYTES, MISSION_NUDGE, STOPPED,
};
use crate::window;

/// The gate this exists for: a strict OpenAI-compatible route answers
/// `400 System message must be at the beginning` for a request whose second
/// message is also `system`. One leading block, whatever the pack config.
#[test]
fn the_leading_block_is_one_message_even_with_injections() {
    let tools = Arc::new(ToolRegistry::with_base_tools().unwrap());
    let mut engine = crate::context::InjectionEngine::new();
    engine.add(
        crate::context::InjectionEntry::constant("r1", "用中文回答。")
            .at(crate::context::PositionKind::Prefix),
    );
    engine.add(
        crate::context::InjectionEntry::constant("r2", "不要开场白。")
            .at(crate::context::PositionKind::Prefix),
    );

    let config: Arc<dyn TurnConfig> = Arc::new(
        StaticConfig::new("你是编程助手", Arc::clone(&tools), "m")
            .with_policy(Arc::new(window::DefaultContextPolicy::default())),
    );
    let client = LlmClient::new(crate::llm::LlmConfig::new("http://localhost:1", "k", "m"))
        .expect("client builds");
    let mut agent = Agent::new(client, Arc::clone(&config), vec![Message::user("问题")]);
    agent.set_injection(Arc::new(engine));

    let (request, _) = agent.build_request(&config.settings(), None);
    let leading = request
        .iter()
        .take_while(|m| m.role == crate::message::Role::System)
        .count();
    assert_eq!(
        leading, 1,
        "exactly one leading system message: {request:?}"
    );
    assert_eq!(request[0].role, crate::message::Role::System);
    assert_eq!(
        request[1].role,
        crate::message::Role::User,
        "the history follows the single block"
    );
    let text = request[0].text();
    for fragment in ["你是编程助手", "用中文回答。", "不要开场白。"] {
        assert!(
            text.contains(fragment),
            "the block keeps {fragment:?} verbatim"
        );
    }
}

#[test]
fn a_static_config_reports_its_settings() {
    let tools = Arc::new(ToolRegistry::with_base_tools().unwrap());
    let config = StaticConfig::new("prompt", Arc::clone(&tools), "test-model");
    let settings = config.settings();
    assert_eq!(settings.system_prompt, "prompt");
    assert_eq!(settings.model, "test-model");
    assert_eq!(settings.version, 0);
    assert!(settings.tools.get("read").is_some());
}

#[test]
fn the_request_prepends_the_system_prompt_without_storing_it() {
    let tools = Arc::new(ToolRegistry::with_base_tools().unwrap());
    let config: Arc<dyn TurnConfig> = Arc::new(StaticConfig::new("SYS", Arc::clone(&tools), "m"));
    let client = LlmClient::new(crate::llm::LlmConfig::new("http://localhost:1", "k", "m"))
        .expect("client builds");
    let agent = Agent::new(client, Arc::clone(&config), vec![Message::user("hi")]);

    let request = agent.build_request(&config.settings(), None).0;
    assert_eq!(request.len(), 2);
    assert_eq!(request[0].role, crate::message::Role::System);
    assert_eq!(request[0].text(), "SYS");
    assert_eq!(request[1].text(), "hi");

    // The history itself is untouched, so recording stays clean.
    assert_eq!(agent.messages().len(), 1);

    // An empty prompt contributes no message at all.
    let bare: Arc<dyn TurnConfig> = Arc::new(StaticConfig::new("", Arc::clone(&tools), "m"));
    assert_eq!(agent.build_request(&bare.settings(), None).0.len(), 1);
}

/// The whole point of thinning, end to end: what leaves this process is
/// smaller, what is stored is not.
#[test]
/// The rule that makes the ladder worth having: while the window has room,
/// a tool result reaches the model exactly as the tool produced it.
///
/// Before this, thinning ran on every request regardless of how full the
/// window was. A model that ran a build and got 200 KB back saw
/// `[old tool output thinned: 204800 chars · ...]` on the very next step —
/// the answer to its own question, replaced by a note about the answer.
#[test]
fn a_tool_result_reaches_the_model_intact_while_there_is_room() {
    let tools = Arc::new(ToolRegistry::with_base_tools().unwrap());
    let mut config_inner = StaticConfig::new("SYS", Arc::clone(&tools), "m");
    config_inner.policy = Arc::new(crate::window::DefaultContextPolicy {
        // A million tokens of room for a history of a few thousand: nowhere
        // near the quiet line.
        configured: Some(1_000_000),
        threshold_percent: 75,
        keep_recent: 8,
        provider_window: None,
        reserve: 0,
        prune: crate::compaction::PrunePolicy::default(),
    });
    let config: Arc<dyn TurnConfig> = Arc::new(config_inner);

    let mut history = Vec::new();
    for index in 0..12 {
        history.push(Message::tool_result(
            format!("call-{index}"),
            "bash",
            format!("r{index} {}", "x".repeat(20_000)),
        ));
    }
    let agent = Agent::new(
        LlmClient::new(crate::llm::LlmConfig::new("http://localhost:1", "k", "m")).unwrap(),
        Arc::clone(&config),
        history,
    );

    let request = agent.build_request(&config.settings(), None).0;
    let thinned = request
        .iter()
        .filter(|message| message.text().contains("chars thinned"))
        .count();
    assert_eq!(
        thinned, 0,
        "nothing is thinned while the window is mostly empty"
    );
    let full = request
        .iter()
        .filter(|message| message.text().len() >= 20_000)
        .count();
    assert_eq!(full, 12, "every result arrives at its full size");
}

fn an_old_tool_result_and_image_leave_the_request_but_not_the_history() {
    let tools = Arc::new(ToolRegistry::with_base_tools().unwrap());
    // A deliberately small window: the ladder only thins once history has
    // grown into a real share of what the model can hold, and this test is
    // about what thinning does — not about when it starts. The "when" has its
    // own test below.
    let mut config_inner = StaticConfig::new("SYS", Arc::clone(&tools), "m");
    config_inner.policy = Arc::new(crate::window::DefaultContextPolicy {
        // Sized so this history lands on the middle rung. Measured
        // rather than guessed: the estimate counts CJK and base64
        // differently from ASCII, and a window picked by arithmetic on
        // character counts landed on the wrong rung twice. 80 k against
        // this history sits above the quiet line and below the
        // compaction threshold, so the default preset runs.
        configured: Some(120_000),
        threshold_percent: 75,
        keep_recent: 8,
        provider_window: None,
        reserve: 0,
        prune: crate::compaction::PrunePolicy::default(),
    });
    let config: Arc<dyn TurnConfig> = Arc::new(config_inner);

    let mut history = Vec::new();
    for index in 0..12 {
        history.push(Message::tool_result(
            format!("call-{index}"),
            "bash",
            format!("r{index} {}", "x".repeat(20_000)),
        ));
    }
    for index in 0..6 {
        history.push(Message::user_with_images(
            "看看这个页面",
            vec![crate::message::ImageAttachment::url(format!(
                "data:image/png;base64,{}",
                "A".repeat(40_000)
            ))
            .stored_at(format!("/tmp/img-{index}.png"))],
        ));
    }
    let agent = Agent::new(
        LlmClient::new(crate::llm::LlmConfig::new("http://localhost:1", "k", "m")).unwrap(),
        Arc::clone(&config),
        history,
    );

    let request = agent.build_request(&config.settings(), None).0;

    // Twelve results, the newest eight exact: the older four say they were
    // thinned instead of shipping twenty thousand characters each.
    let thinned = request
        .iter()
        .filter(|message| message.text().contains("chars thinned"))
        .count();
    assert_eq!(thinned, 4, "the four oldest tool results are thinned");

    // Six inline images, the newest two stay: the request carries a fraction
    // of the bytes the history holds.
    let sent_images: usize = request.iter().map(|message| message.images.len()).sum();
    assert_eq!(sent_images, 2, "only the newest images travel");
    let sent_bytes: usize = request
        .iter()
        .flat_map(|message| message.images.iter())
        .map(|image| image.url.len())
        .sum();
    assert!(sent_bytes < 100_000, "{sent_bytes} bytes of image inline");

    // And the dropped ones left a way back, not a hole.
    let note = request
        .iter()
        .find(|message| message.text().contains("已省略"))
        .expect("a note replaces the dropped images");
    assert!(note.text().contains("/tmp/img-0.png"), "{}", note.text());

    // The stored conversation keeps every byte: thinning is a view.
    let stored = agent.messages();
    assert_eq!(stored.len(), 18, "no message was dropped from history");
    assert_eq!(
        stored[0].text().len(),
        20_003,
        "the full tool result is kept"
    );
    assert_eq!(stored[12].images.len(), 1, "the old image is kept");
}

#[test]
fn an_injected_prefix_fragment_sits_above_the_history() {
    let tools = Arc::new(ToolRegistry::with_base_tools().unwrap());
    let mut engine = crate::context::InjectionEngine::new();
    engine.add(
        crate::context::InjectionEntry::constant("rule", "never edit vendor/")
            .at(crate::context::PositionKind::Prefix),
    );

    let config: Arc<dyn TurnConfig> = Arc::new(
        StaticConfig::new("SYS", tools, "m")
            .with_policy(Arc::new(window::DefaultContextPolicy::default())),
    );
    let client = LlmClient::new(crate::llm::LlmConfig::new("http://localhost:1", "k", "m"))
        .expect("client builds");
    let mut agent = Agent::new(client, Arc::clone(&config), vec![Message::user("hi")]);
    agent.set_injection(Arc::new(engine));

    let (request, injection) = agent.build_request(&config.settings(), None);
    assert_eq!(injection.activated, vec!["rule"]);
    // The rule is above the history — and inside the same leading block as
    // the persona, because a strict gateway refuses a second `system`
    // message. Verbatim text, one envelope.
    assert_eq!(request.len(), 2, "one leading block, then the history");
    assert_eq!(request[0].role, crate::message::Role::System);
    assert!(request[0].text().contains("SYS"), "persona kept");
    assert!(
        request[0].text().contains("never edit vendor/"),
        "rule kept verbatim: {}",
        request[0].text()
    );
    assert_eq!(request[1].text(), "hi");

    // Injected text never enters the recorded history.
    assert_eq!(agent.messages().len(), 1);
}

#[test]
fn an_injected_tail_fragment_comes_last() {
    let tools = Arc::new(ToolRegistry::with_base_tools().unwrap());
    let mut engine = crate::context::InjectionEngine::new();
    engine.add(
        crate::context::InjectionEntry::constant("note", "remember the deadline")
            .at(crate::context::PositionKind::HistoryTail),
    );

    let config: Arc<dyn TurnConfig> = Arc::new(StaticConfig::new("SYS", tools, "m"));
    let client = LlmClient::new(crate::llm::LlmConfig::new("http://localhost:1", "k", "m"))
        .expect("client builds");
    let mut agent = Agent::new(client, Arc::clone(&config), vec![Message::user("hi")]);
    agent.set_injection(Arc::new(engine));

    let (request, _) = agent.build_request(&config.settings(), None);
    assert_eq!(request.len(), 3);
    assert_eq!(request.last().unwrap().text(), "remember the deadline");
}

#[test]
fn an_injected_depth_fragment_never_splits_a_tool_pair() {
    // A provider rejects a request where an assistant's tool_calls message
    // is not immediately followed by its tool results, so an insertion that
    // would land there has to move earlier.
    let tools = Arc::new(ToolRegistry::with_base_tools().unwrap());
    let mut engine = crate::context::InjectionEngine::new();
    engine.add(crate::context::InjectionEntry::constant("deep", "inserted").at_depth(1));

    let config: Arc<dyn TurnConfig> = Arc::new(StaticConfig::new("SYS", tools, "m"));
    let client = LlmClient::new(crate::llm::LlmConfig::new("http://localhost:1", "k", "m"))
        .expect("client builds");

    let calling = Message::assistant_tools(
        vec![ToolCall {
            id: "c1".into(),
            name: "read".into(),
            arguments: "{}".into(),
        }],
        None,
    );
    let result = Message::tool_result("c1", "read", "contents");
    let mut agent = Agent::new(
        client,
        Arc::clone(&config),
        vec![Message::user("please read"), calling, result],
    );
    agent.set_injection(Arc::new(engine));

    let (request, _) = agent.build_request(&config.settings(), None);

    // Find the assistant call and assert the very next message is its result.
    let call_index = request
        .iter()
        .position(|m| !m.tool_calls.is_empty())
        .expect("the call survives");
    assert_eq!(
        request[call_index + 1].role,
        crate::message::Role::Tool,
        "the call must be immediately followed by its result: {request:#?}"
    );
}

#[test]
fn an_injected_depth_fragment_lands_that_many_messages_back() {
    let tools = Arc::new(ToolRegistry::with_base_tools().unwrap());
    let mut engine = crate::context::InjectionEngine::new();
    engine.add(crate::context::InjectionEntry::constant("deep", "inserted here").at_depth(1));

    let config: Arc<dyn TurnConfig> = Arc::new(StaticConfig::new("SYS", tools, "m"));
    let client = LlmClient::new(crate::llm::LlmConfig::new("http://localhost:1", "k", "m"))
        .expect("client builds");
    let mut agent = Agent::new(
        client,
        Arc::clone(&config),
        vec![Message::user("first"), Message::user("second")],
    );
    agent.set_injection(Arc::new(engine));

    let (request, _) = agent.build_request(&config.settings(), None);
    // system, first, injected (depth 1 = before the newest), second.
    assert_eq!(request.len(), 4);
    assert_eq!(request[1].text(), "first");
    assert_eq!(request[2].text(), "inserted here");
    assert_eq!(request[3].text(), "second");
}

#[test]
fn settings_are_frozen_per_turn() {
    use std::sync::atomic::{AtomicU64, Ordering};

    struct Counting {
        calls: AtomicU64,
        tools: Arc<ToolRegistry>,
    }

    impl TurnConfig for Counting {
        fn settings(&self) -> TurnSettings {
            let n = self.calls.fetch_add(1, Ordering::Relaxed);
            TurnSettings {
                system_prompt: format!("v{n}"),
                tools: Arc::clone(&self.tools),
                model: "m".into(),
                version: n,
                policy: Arc::new(window::DefaultContextPolicy::default()),
                injection: Arc::new(crate::context::InjectionEngine::new()),
                cache_policy: crate::hotreload::CachePolicy::default(),
                plugin_hooks: None,
            }
        }
    }

    let tools = Arc::new(ToolRegistry::with_base_tools().unwrap());
    let config: Arc<dyn TurnConfig> = Arc::new(Counting {
        calls: AtomicU64::new(0),
        tools,
    });
    let client = LlmClient::new(crate::llm::LlmConfig::new("http://localhost:1", "k", "m"))
        .expect("client builds");
    let agent = Agent::new(client, Arc::clone(&config), Vec::new());

    // Each read advances the counter, which is what makes "one snapshot per
    // turn" observable rather than assumed.
    assert_eq!(agent.config.settings().version, 0);
    assert_eq!(agent.config.settings().version, 1);
    assert_eq!(agent.config_version(), 2);
}

#[test]
fn an_empty_step_is_nothing_at_all() {
    let empty = StepOutput::default();
    assert!(empty.is_empty());

    let mut spoken = StepOutput::default();
    spoken.text = "hi".into();
    assert!(!spoken.is_empty());

    let mut thought = StepOutput::default();
    thought.saw_reasoning = true;
    assert!(!thought.is_empty());

    let mut called = StepOutput::default();
    called.calls.push(PartialCallView::default());
    assert!(!called.is_empty());
}

/// A provider answering canned HTTP responses in order, repeating the
/// last one, one response per connection.
///
/// Enough of the wire for the loop under test: the loop's decisions are
/// made from statuses, SSE frames, and connection endings, not from
/// anything else the request carries — so the request head and body are
/// drained and ignored.
async fn provider(
    responses: Vec<String>,
) -> (String, std::sync::Arc<std::sync::atomic::AtomicUsize>) {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let hits = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let served = std::sync::Arc::clone(&hits);

    tokio::spawn(async move {
        while let Ok((mut socket, _)) = listener.accept().await {
            // Drain the request head byte-wise, then the declared body, so
            // a close after the response cannot race a client still
            // writing and turn every test into a flake.
            let mut head = Vec::new();
            loop {
                let mut byte = [0u8; 1];
                match socket.read(&mut byte).await {
                    Ok(0) | Err(_) => break,
                    Ok(_) => {
                        head.push(byte[0]);
                        if head.ends_with(b"\r\n\r\n") {
                            break;
                        }
                    }
                }
            }
            let head_text = String::from_utf8_lossy(&head);
            let declared = head_text
                .lines()
                .filter_map(|line| {
                    let (key, value) = line.split_once(':')?;
                    key.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().ok())
                        .flatten()
                })
                .next()
                .unwrap_or(0);
            let mut remaining = declared;
            let mut buf = [0u8; 4096];
            while remaining > 0 {
                match socket.read(&mut buf).await {
                    Ok(0) | Err(_) => break,
                    Ok(read) => remaining = remaining.saturating_sub(read),
                }
            }

            let index = served.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let body = responses
                .get(index)
                .or_else(|| responses.last())
                .cloned()
                .unwrap_or_default();
            let _ = socket.write_all(body.as_bytes()).await;
            let _ = socket.flush().await;
        }
    });
    (format!("http://{addr}"), hits)
}

/// One SSE response carrying a single content frame with `finish`.
fn sse_answer(text: &str, finish: &str) -> String {
    let frame = serde_json::json!({
        "choices": [{ "delta": { "content": text }, "finish_reason": finish }]
    });
    format!(
        "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\nconnection: close\r\n\r\ndata: {frame}\n\ndata: [DONE]\n\n"
    )
}

/// The same provider, except the **first** connection is accepted, read and
/// then closed without a byte: the shape of a gateway that took the request
/// and died. Every later connection answers like `provider`, so a client
/// that retries lands on a working route.
///
/// The request is drained first for the same reason `provider` drains it: a
/// close while the client is still writing is a race that turns the test
/// into a flake.
/// Connections the drop provider kills before it starts answering.
///
/// More than the client's own connection retry budget, so the failure
/// reaches the agent loop's retry rather than being absorbed below it.
const DROPS: usize = 6;

async fn provider_dropping_first() -> (String, std::sync::Arc<std::sync::atomic::AtomicUsize>) {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let hits = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let served = std::sync::Arc::clone(&hits);

    tokio::spawn(async move {
        while let Ok((mut socket, _)) = listener.accept().await {
            let index = served.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

            let mut head = Vec::new();
            loop {
                let mut byte = [0u8; 1];
                match socket.read(&mut byte).await {
                    Ok(0) | Err(_) => break,
                    Ok(_) => {
                        head.push(byte[0]);
                        if head.ends_with(b"\r\n\r\n") {
                            break;
                        }
                    }
                }
            }
            let head_text = String::from_utf8_lossy(&head);
            let declared = head_text
                .lines()
                .filter_map(|line| {
                    let (key, value) = line.split_once(':')?;
                    key.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().ok())
                        .flatten()
                })
                .next()
                .unwrap_or(0);
            let mut remaining = declared;
            let mut buf = [0u8; 4096];
            while remaining > 0 {
                match socket.read(&mut buf).await {
                    Ok(0) | Err(_) => break,
                    Ok(read) => remaining = remaining.saturating_sub(read),
                }
            }

            if index < DROPS {
                // Read the request, then vanish: whatever the transport
                // layer calls this, the retry rule must treat it as worth
                // another ask. Several of them, because the client retries
                // a connection failure on its own before the agent loop ever
                // sees one.
                let _ = socket.shutdown().await;
                continue;
            }
            let body = sse_answer("all done", "stop");
            let _ = socket.write_all(body.as_bytes()).await;
            let _ = socket.flush().await;
        }
    });
    (format!("http://{addr}"), hits)
}

fn agent_against(base: String) -> Agent {
    let tools = Arc::new(ToolRegistry::with_base_tools().unwrap());
    let config: Arc<dyn TurnConfig> = Arc::new(StaticConfig::new("SYS", tools, "test-model"));
    let client = LlmClient::new(crate::llm::LlmConfig::new(base, "k", "test-model")).unwrap();
    Agent::new(client, config, Vec::new())
}

#[tokio::test]
async fn a_truncated_answer_is_continued_rather_than_taken_as_finished() {
    // A small model hits the output ceiling mid-answer. The turn must go
    // on asking for the rest instead of ending half-done and looking
    // complete.
    let (base, hits) = provider(vec![
        sse_answer("part one ", "length"),
        sse_answer("part two", "stop"),
    ])
    .await;
    let mut agent = agent_against(base);

    let outcome = agent.run("go").await.expect("the loop finishes");

    assert!(outcome.error.is_none(), "{:?}", outcome.error);
    assert_eq!(outcome.steps, 2, "the cut-off step runs again");
    assert_eq!(outcome.text, "part one part two");
    assert_eq!(
        hits.load(std::sync::atomic::Ordering::Relaxed),
        2,
        "one request per step"
    );

    // The history keeps both segments and the nudge between them: the
    // model must see where it was cut off to continue from there.
    let messages = agent.messages();
    assert_eq!(messages.len(), 4);
    assert_eq!(messages[0].text(), "go");
    assert_eq!(messages[1].text(), "part one ");
    assert!(
        messages[2].text().contains("cut off"),
        "{:?}",
        messages[2].text()
    );
    assert_eq!(messages[3].text(), "part two");
}

#[tokio::test]
async fn a_transient_provider_failure_is_retried_within_the_step() {
    let (base, hits) = provider(vec![
        "HTTP/1.1 503 Service Unavailable\r\nconnection: close\r\n\r\noverloaded".to_string(),
        sse_answer("recovered", "stop"),
    ])
    .await;
    let mut agent = agent_against(base);

    let outcome = agent.run("go").await.expect("the loop finishes");

    assert!(outcome.error.is_none(), "{:?}", outcome.error);
    assert_eq!(outcome.retries, 1);
    assert_eq!(hits.load(std::sync::atomic::Ordering::Relaxed), 2);
    assert_eq!(outcome.text, "recovered");
    // Only the final attempt's answer enters the history — the failed one
    // produced nothing worth keeping.
    assert_eq!(agent.messages().len(), 2);
}

#[tokio::test]
async fn a_permanent_failure_reports_an_error_without_losing_the_turn() {
    let (base, hits) = provider(vec![
        "HTTP/1.1 400 Bad Request\r\nconnection: close\r\n\r\nmalformed".to_string(),
    ])
    .await;
    let mut agent = agent_against(base);

    let outcome = agent.run("go").await.expect("the loop still answers");

    let error = outcome
        .error
        .expect("a 400 must surface as the turn's error");
    assert!(error.contains("400"), "{error}");
    assert_eq!(outcome.retries, 0, "a rejected request is not retried");
    assert_eq!(hits.load(std::sync::atomic::Ordering::Relaxed), 1);
    // The prompt itself survives: the next turn continues from what was
    // said, not from a conversation that pretends the ask never happened.
    assert_eq!(outcome.new_messages.len(), 1);
    assert_eq!(outcome.new_messages[0].text(), "go");
}

#[tokio::test]
async fn a_rejected_prompt_teaches_the_route_its_real_window() {
    // A gateway whose model accepts far less than the family table assumes
    // answers an oversized prompt with a 400 that names the limit. The turn
    // must still fail exactly as it did before, and the next compaction
    // decision must measure against the number the provider stated instead
    // of against the table's estimate for the model id.
    //
    // The id is unique because the report is stored process-wide: recording
    // one for a name another test resolves would make these tests depend on
    // the order they ran in.
    let model = format!("window-probe-{}", uuid::Uuid::new_v4().simple());
    let body = serde_json::json!({
        "error": { "message": "This model's maximum context length is 8192 tokens" }
    })
    .to_string();
    let (base, hits) = provider(vec![format!(
        "HTTP/1.1 400 Bad Request\r\ncontent-type: application/json\r\nconnection: close\r\n\r\n{body}"
    )])
    .await;

    let tools = Arc::new(ToolRegistry::with_base_tools().unwrap());
    let config: Arc<dyn TurnConfig> = Arc::new(StaticConfig::new("SYS", tools, model.clone()));
    let client = LlmClient::new(crate::llm::LlmConfig::new(base, "k", model.clone())).unwrap();
    let mut agent = Agent::new(client, config, Vec::new());

    let outcome = agent.run("go").await.expect("the loop still answers");

    assert!(
        outcome.error.is_some(),
        "a rejected request still fails the turn"
    );
    assert_eq!(
        hits.load(std::sync::atomic::Ordering::Relaxed),
        1,
        "and it is still not retried"
    );
    assert_eq!(
        window::reported_windows().get(&model),
        Some(8192),
        "the provider's own number is learned from its refusal"
    );
    let resolved = window::DefaultContextPolicy::default().window(&model);
    assert_eq!(resolved.tokens, 8192);
    assert_eq!(resolved.source, window::WindowSource::Provider);
}

#[tokio::test]
async fn an_empty_response_is_retried_and_then_reported() {
    let (base, hits) = provider(vec!["HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\nconnection: close\r\n\r\ndata: [DONE]\n\n".to_string()])
        .await;
    let mut agent = agent_against(base);

    let outcome = agent.run("go").await.expect("the loop still answers");

    let error = outcome.error.expect("silence must not read as success");
    assert!(error.contains("empty response"), "{error}");
    // Default budget: one attempt plus three retries.
    assert_eq!(outcome.retries, 3);
    assert_eq!(hits.load(std::sync::atomic::Ordering::Relaxed), 4);
}

#[tokio::test]
async fn a_poisoned_inbox_does_not_end_the_turn() {
    // A panic in another thread while the inbox was locked leaves the mutex
    // poisoned. `expect("inbox")` then panics on *every* remaining step of
    // the turn, so one unrelated panic silently breaks the whole loop. The
    // queue is a plain `VecDeque` and is still intact, so the step goes on
    // with what is there.
    //
    // The panic printed while this test sets that up is the point: poisoning
    // can only happen by panicking, so it cannot be arranged quietly.
    let inbox: Arc<std::sync::Mutex<VecDeque<String>>> =
        Arc::new(std::sync::Mutex::new(VecDeque::new()));
    let doomed = Arc::clone(&inbox);
    let _ = std::thread::spawn(move || {
        let mut queue = doomed.lock().expect("a fresh lock");
        queue.push_back("said while the thread was dying".to_string());
        panic!("this panic is what poisons the inbox");
    })
    .join();
    assert!(inbox.lock().is_err(), "the mutex is poisoned");

    let (base, hits) = provider(vec![sse_answer("answered", "stop")]).await;
    let mut agent = agent_against(base).with_inbox(Arc::clone(&inbox));

    let outcome = agent.run("go").await.expect("the turn must not panic");

    assert!(outcome.error.is_none(), "{:?}", outcome.error);
    assert_eq!(outcome.text, "answered");
    assert_eq!(hits.load(std::sync::atomic::Ordering::Relaxed), 1);

    // What was queued before the poison still reached the model, and the
    // queue was drained rather than left to be re-read on every step.
    let texts: Vec<String> = agent
        .messages()
        .iter()
        .map(|message| message.text().to_string())
        .collect();
    assert!(
        texts
            .iter()
            .any(|text| text == "said while the thread was dying"),
        "the steering text survived: {texts:?}"
    );
    let drained = match inbox.lock() {
        Ok(queue) => queue.is_empty(),
        Err(poisoned) => poisoned.into_inner().is_empty(),
    };
    assert!(drained, "claimed once, not on every step");
}

/// A registry holding exactly one tool, built from a handler.
fn registry_with(
    name: &str,
    handler: impl Fn(serde_json::Value) -> crate::tools::ToolFuture + Send + Sync + 'static,
) -> Arc<ToolRegistry> {
    let mut registry = ToolRegistry::new();
    registry
        .register(
            crate::tools::ToolDef::new(
                name,
                "a tool that does exactly what this test needs",
                serde_json::json!({ "type": "object", "properties": {} }),
                "test",
                handler,
            ),
            crate::tools::ConflictPolicy::Error,
        )
        .expect("the tool registers");
    Arc::new(registry)
}

/// One SSE response asking for a single tool call.
fn sse_tool_call(id: &str, name: &str, arguments: &str) -> String {
    let frame = serde_json::json!({
        "choices": [{
            "delta": { "tool_calls": [{
                "index": 0,
                "id": id,
                "function": { "name": name, "arguments": arguments }
            }]},
            "finish_reason": "tool_calls"
        }]
    });
    format!(
        "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\nconnection: close\r\n\r\ndata: {frame}\n\ndata: [DONE]\n\n"
    )
}

/// An agent against `base`, with `tools` and a cap of one step per batch.
///
/// The mock provider repeats its last response, so a script that ends on a
/// tool call keeps the model asking for tools until the whole step budget
/// is spent. A test that wants one tool round and then an end scripts a
/// final answer as the last response.
fn agent_with_tools(base: String, tools: Arc<ToolRegistry>) -> Agent {
    let config: Arc<dyn TurnConfig> = Arc::new(StaticConfig::new("SYS", tools, "test-model"));
    let client =
        LlmClient::new(crate::llm::LlmConfig::new(base, "k", "test-model")).expect("client builds");
    Agent::new(client, config, Vec::new()).with_max_steps(1)
}

/// The tool message answering call `id`, if the history holds one.
fn answer_to(agent: &Agent, id: &str) -> Option<String> {
    agent.messages().iter().find_map(|message| {
        (message.role == crate::message::Role::Tool && message.tool_call_id.as_deref() == Some(id))
            .then(|| message.text().to_string())
    })
}

#[tokio::test]
async fn a_panicking_tool_is_answered_instead_of_ending_the_turn() {
    // A panic inside a tool makes its task return a `JoinError`. The
    // assistant message that asked for the call is already in the history,
    // so letting that error escape leaves a call with no result — a history
    // no provider accepts, refused with a 400 on every later request, which
    // ends the conversation rather than the call. The call is answered
    // instead, and the turn goes on.
    let tools = registry_with("boom", |_arguments| -> crate::tools::ToolFuture {
        Box::pin(async move {
            panic!("the tool went down");
        })
    });
    // The script ends on a final answer. The provider repeats its last
    // response, and the cap is one step per batch, so a script that ended on
    // the tool call would keep the model calling it until the whole step
    // budget was spent — this test is about the panic, not the budget.
    let (base, hits) = provider(vec![
        sse_tool_call("call_1", "boom", "{}"),
        sse_answer("done", "stop"),
    ])
    .await;
    let mut agent = agent_with_tools(base, tools);

    let outcome = agent
        .run("go")
        .await
        .expect("a panicking tool must not end the turn");

    assert!(outcome.error.is_none(), "{:?}", outcome.error);
    assert_eq!(outcome.tool_calls, 1);
    assert_eq!(
        hits.load(std::sync::atomic::Ordering::Relaxed),
        2,
        "the tool round, then the answer that follows the ask to continue"
    );

    // The pairing, asserted where a request builder looks for it: the call
    // is immediately followed by its result, and the result names the call.
    let messages = agent.messages();
    let call_at = messages
        .iter()
        .position(|message| !message.tool_calls.is_empty())
        .expect("the call was committed");
    assert_eq!(messages[call_at].tool_calls[0].id, "call_1");
    assert_eq!(messages[call_at + 1].role, crate::message::Role::Tool);
    assert_eq!(
        messages[call_at + 1].tool_call_id.as_deref(),
        Some("call_1")
    );

    // The model reads a failure it can act on, not a missing answer.
    let answer = answer_to(&agent, "call_1").expect("the call has an answer");
    assert!(answer.starts_with("error:"), "{answer}");
    assert!(answer.contains("tool panicked: boom"), "{answer}");
}

#[tokio::test]
async fn a_tool_failure_reaches_the_model_as_a_bounded_summary() {
    // The whole cause chain is a developer artifact — nested contexts,
    // absolute paths, internal hostnames, a token pasted into a failed URL —
    // and it enters the history and travels to the provider on every request
    // after it. The model gets the top-level reason, bounded and on one line;
    // the chain underneath stays in the log.
    let tools = registry_with("explode", |_arguments| -> crate::tools::ToolFuture {
        Box::pin(async move {
            let long = "C:\\Users\\someone\\project\\".to_string() + &"nested\\".repeat(200);
            let failed: Result<crate::tools::ToolOutput> =
                Err(anyhow::anyhow!("no such device (os error 6)")
                    .context(format!("copy failed for {long}artifact.bin")));
            failed
        })
    });
    // One tool round, then a final answer: the failure summary is what this
    // test measures, so the turn is allowed to end after the call instead of
    // running the whole step budget.
    let (base, _) = provider(vec![
        sse_tool_call("call_2", "explode", "{}"),
        sse_answer("done", "stop"),
    ])
    .await;
    let mut agent = agent_with_tools(base, tools);

    let outcome = agent.run("go").await.expect("the turn finishes");
    assert!(outcome.error.is_none(), "{:?}", outcome.error);

    let answer = answer_to(&agent, "call_2").expect("the call has an answer");
    assert!(answer.starts_with("error: copy failed"), "{answer}");
    assert!(
        answer.len() <= "error: ".len() + MAX_TOOL_ERROR_BYTES + "… [truncated]".len(),
        "the failure is bounded; it was {} bytes",
        answer.len()
    );
    assert!(answer.ends_with("[truncated]"), "{answer}");
    assert!(
        !answer.contains("no such device"),
        "the cause chain stays out of the transcript: {answer}"
    );
    assert!(!answer.contains('\n'), "one line, not several: {answer}");
}

#[test]
fn a_long_tool_error_is_cut_at_a_character_boundary() {
    // Cutting by byte would split a multi-byte character and put U+FFFD in
    // the transcript. Chinese text is the easy way to reach the limit.
    let error = anyhow::anyhow!("读取失败：{}", "路径".repeat(400));
    let summary = summarize_tool_error(&error);
    assert!(summary.ends_with("[truncated]"), "{summary}");
    assert!(summary.len() <= MAX_TOOL_ERROR_BYTES + "… [truncated]".len());
    assert!(
        !summary.contains('\u{FFFD}'),
        "no torn characters: {summary}"
    );
    assert!(summary.starts_with("读取失败："), "the reason survives");

    // A short failure passes through untouched, newlines collapsed.
    let short = summarize_tool_error(&anyhow::anyhow!("line one\nline two"));
    assert_eq!(short, "line one line two");
}

/// A tool that only returns, so a script that keeps calling it spends steps
fn quiet_tools() -> Arc<ToolRegistry> {
    registry_with("step", |_arguments| -> crate::tools::ToolFuture {
        Box::pin(async move {
            let answered: Result<crate::tools::ToolOutput> =
                Ok(crate::tools::ToolOutput::text("step"));
            answered
        })
    })
}

/// The part of the continuation ask a test matches on.
///
/// A fragment rather than the whole sentence, so rewording the question the
/// model is asked does not break the tests that only care that it was asked.
const CONTINUATION_FRAGMENT: &str = "对话轮数过长是否正常";

/// How many of `messages` carry the continuation ask.
fn continuation_asks<'a>(messages: impl IntoIterator<Item = &'a Message>) -> usize {
    messages
        .into_iter()
        .filter(|message| message.text().contains(CONTINUATION_FRAGMENT))
        .count()
}

#[tokio::test]
async fn the_step_cap_asks_the_model_to_continue_instead_of_ending_the_turn() {
    // A model that only ever asks for tools never produces a final answer.
    // The cap used to end the turn there, abandoning a long task mid-work
    // with nothing but a note saying it had been abandoned. Reaching it now
    // asks whether a round count this long is normal, tells the model to
    // keep working, and grants a bounded number of batches of the same
    // length.
    let (base, hits) = provider(vec![sse_tool_call("call_1", "step", "{}")]).await;
    let mut agent = agent_with_tools(base, quiet_tools()).with_max_steps(2);

    let outcome = agent.run("go").await.expect("the loop finishes");

    assert!(outcome.error.is_none(), "{:?}", outcome.error);
    assert_eq!(
        continuation_asks(&outcome.new_messages),
        CONTINUATION_BATCHES,
        "one ask per spent batch, and no more: {:?}",
        outcome
            .new_messages
            .iter()
            .map(Message::text)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        outcome.steps,
        2 * (1 + CONTINUATION_BATCHES),
        "each renewal grants another full cap of steps"
    );
    assert_eq!(
        hits.load(std::sync::atomic::Ordering::Relaxed),
        outcome.steps,
        "every one of those steps was a real request"
    );
    assert_eq!(
        outcome.text,
        format!(
            "stopped after {} steps without a final answer ({} continuation(s) asked)",
            2 * (1 + CONTINUATION_BATCHES),
            CONTINUATION_BATCHES
        ),
        "the report is honest about the real total and what was asked"
    );
    // The ask is a request, not a remark: it has to say what to do next.
    assert!(
        outcome.new_messages.iter().any(|message| message
            .text()
            .contains("如果已经可以收尾，请现在直接给出结论")),
        "the ask carries the instruction to keep working"
    );

    // Committed to the conversation, not merely reported: a message the
    // model never reads is not a request to continue.
    assert_eq!(continuation_asks(agent.messages()), CONTINUATION_BATCHES);
}

#[tokio::test]
async fn the_continuation_message_reaches_the_observer() {
    // Hosts persist the conversation from `on_message`, and that is what the
    // user reads as the transcript. An ask the model receives but the
    // transcript never shows makes a turn that overran its cap look like one
    // that hung.
    #[derive(Default)]
    struct Recorder {
        seen: std::sync::Mutex<Vec<String>>,
    }

    impl AgentObserver for Recorder {
        fn on_message(&self, message: &Message) {
            self.seen
                .lock()
                .expect("the recorder lock is not poisoned")
                .push(message.text().to_string());
        }
    }

    let (base, _) = provider(vec![sse_tool_call("call_1", "step", "{}")]).await;
    let recorder = Arc::new(Recorder::default());
    // Cloned through the binding, not `Arc::clone(&recorder)`: the turbofish
    // is inferred from the argument, so the unsizing to `dyn AgentObserver`
    // has to happen where the expected type is known.
    let observing: Arc<dyn AgentObserver> = recorder.clone();
    let mut agent = agent_with_tools(base, quiet_tools())
        .with_max_steps(1)
        .with_observer(observing);

    let outcome = agent.run("go").await.expect("the loop finishes");

    let seen = recorder
        .seen
        .lock()
        .expect("the recorder lock is not poisoned")
        .clone();
    assert_eq!(
        seen.iter()
            .filter(|text| text.contains(CONTINUATION_FRAGMENT))
            .count(),
        CONTINUATION_BATCHES,
        "the transcript shows every ask: {seen:?}"
    );
    // The prompt is observed too, so the count above is not simply "every
    // committed message".
    assert_eq!(seen.first().map(String::as_str), Some("go"));
    assert_eq!(outcome.steps, 1 + CONTINUATION_BATCHES);
}

#[tokio::test]
async fn a_turn_that_finishes_early_never_asks_for_a_continuation() {
    // The renewal must be invisible to a turn that ends on its own: asking a
    // finished model whether to keep working would invite it to invent more
    // work.
    let (base, hits) = provider(vec![sse_answer("all done", "stop")]).await;
    let mut agent = agent_against(base).with_max_steps(2);

    let outcome = agent.run("go").await.expect("the loop finishes");

    assert!(outcome.error.is_none(), "{:?}", outcome.error);
    assert_eq!(outcome.text, "all done");
    assert!(
        outcome.steps < 2,
        "a finished turn leaves the rest of its batch unused"
    );
    assert_eq!(
        continuation_asks(&outcome.new_messages),
        0,
        "nothing to renew"
    );
    assert_eq!(continuation_asks(agent.messages()), 0);
    assert_eq!(hits.load(std::sync::atomic::Ordering::Relaxed), 1);
}

#[tokio::test]
async fn a_stop_during_the_second_batch_ends_the_turn_without_another_ask() {
    // A renewal must not outlive the user's decision to stop. The stop lands
    // while the second batch is running, so the ask that authorized that
    // batch is already committed — and nothing may be appended after it.
    struct StopAtStep {
        at: usize,
        cancel: tokio::sync::watch::Sender<bool>,
    }

    impl AgentObserver for StopAtStep {
        fn on_step(&self, step: usize) {
            if step == self.at {
                let _ = self.cancel.send(true);
            }
        }
    }

    let (base, hits) = provider(vec![sse_tool_call("call_1", "step", "{}")]).await;
    let (cancel, cancelled) = tokio::sync::watch::channel(false);
    // Step 3 is the first step of the second batch, so batch one is spent
    // and its ask is in the history when the stop is requested.
    let stopper: Arc<dyn AgentObserver> = Arc::new(StopAtStep { at: 3, cancel });
    let mut agent = agent_with_tools(base, quiet_tools())
        .with_max_steps(2)
        .with_observer(stopper)
        .with_cancel(cancelled);

    let outcome = agent.run("go").await.expect("the loop still answers");

    assert_eq!(
        outcome.error.as_deref(),
        Some(STOPPED),
        "{:?}",
        outcome.error
    );
    assert_eq!(
        outcome.steps, 3,
        "the stop lands before that step's request"
    );
    assert_eq!(
        hits.load(std::sync::atomic::Ordering::Relaxed),
        2,
        "no request is made for the step the user stopped"
    );
    assert_eq!(
        continuation_asks(&outcome.new_messages),
        1,
        "the ask that started batch two, and none after the stop"
    );
    assert_eq!(continuation_asks(agent.messages()), 1);
}

// ------------------------------------------- a mission outlives the answer

/// The id every mission test in this file uses.
const MISSION_TEST_ID: &str = "m-20260101-000000";

/// A scratch workspace holding one mission record and the pointer to it.
///
/// Pinned through [`crate::tools::ToolIdentity`], never through a
/// process-global: the loop reads the mission from the identity the turn was
/// built with, so a test that rewrote a global would pass or fail on
/// whatever else happened to be running.
fn workspace_with_mission(tag: &str, status: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "ngu-agent-mission-{tag}-{}",
        uuid::Uuid::new_v4().simple()
    ));
    let mdir = dir.join("missions").join(MISSION_TEST_ID);
    std::fs::create_dir_all(&mdir).expect("the mission directory");
    std::fs::write(
        mdir.join("mission.json"),
        serde_json::json!({
            "id": MISSION_TEST_ID,
            "objective": "ship the crawler",
            "status": status,
            "step": "write the failing test",
            "done_when": "the tests pass",
        })
        .to_string(),
    )
    .expect("the mission record");
    std::fs::write(
        dir.join("missions").join("active.json"),
        serde_json::json!({ "mission": MISSION_TEST_ID }).to_string(),
    )
    .expect("the active pointer");
    dir
}

/// How many of `messages` are the bare continuation handed to a running
/// mission. Compared whole rather than by fragment: 「继续」 is short enough
/// to appear inside other injected text, and a `contains` would count those.
fn mission_nudges<'a>(messages: impl IntoIterator<Item = &'a Message>) -> usize {
    messages
        .into_iter()
        .filter(|message| message.text() == MISSION_NUDGE)
        .count()
}

#[tokio::test]
async fn a_running_mission_never_ends_the_turn() {
    // The model answered; the mission on disk says the work is not over. The
    // shell may not end that turn — and it may not cap it either: a step
    // count here is what used to turn a task into something that stops and
    // waits for the user. The turn goes round again until the mission itself
    // stops running, which is a decision only the mission's record makes.
    //
    // The step at which the record is flipped to `blocked` is far past every
    // bound the old version had (two idle asks, sixty-four batches), so this
    // test fails if any of them comes back.
    const BLOCK_AT: usize = 70;

    struct BlockAt {
        at: usize,
        record: std::path::PathBuf,
        seen: std::sync::atomic::AtomicUsize,
    }

    impl AgentObserver for BlockAt {
        fn on_step(&self, step: usize) {
            if step == self.at {
                std::fs::write(
                    &self.record,
                    serde_json::json!({
                        "id": MISSION_TEST_ID,
                        "objective": "ship the crawler",
                        "status": "blocked",
                        "step": "waiting on a decision only the user can make",
                        "done_when": "the tests pass",
                    })
                    .to_string(),
                )
                .expect("the mission record is writable");
            }
        }

        fn on_message(&self, message: &Message) {
            if message.text() == MISSION_NUDGE {
                self.seen.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
    }

    let dir = workspace_with_mission("running", "running");
    // The provider repeats its last response and the cap is one step per
    // batch, so every request is one batch and the batch count and the
    // request count are the same number.
    let (base, hits) = provider(vec![sse_answer("all done", "stop")]).await;
    let seen = Arc::new(BlockAt {
        at: BLOCK_AT,
        record: dir
            .join("missions")
            .join(MISSION_TEST_ID)
            .join("mission.json"),
        seen: std::sync::atomic::AtomicUsize::new(0),
    });
    let observing: Arc<dyn AgentObserver> = seen.clone();
    let mut agent = agent_with_tools(base, quiet_tools())
        .with_max_steps(1)
        .with_observer(observing)
        .with_identity(crate::tools::ToolIdentity::for_session(
            "test-session",
            Some(dir.clone()),
        ));

    let outcome = agent.run("go").await.expect("the loop finishes");

    assert!(outcome.error.is_none(), "{:?}", outcome.error);
    assert_eq!(
        outcome.steps, BLOCK_AT,
        "the turn ran until the mission stopped running, not until a count ran out"
    );
    assert_eq!(
        hits.load(std::sync::atomic::Ordering::Relaxed),
        BLOCK_AT,
        "every one of those steps was a real request"
    );
    assert_eq!(
        outcome.text, "all done",
        "the answer that ended it is the model's"
    );
    assert_eq!(
        mission_nudges(&outcome.new_messages),
        BLOCK_AT - 1,
        "one continuation per step the mission held open"
    );
    assert_eq!(mission_nudges(agent.messages()), BLOCK_AT - 1);
    assert_eq!(
        seen.seen.load(std::sync::atomic::Ordering::Relaxed),
        BLOCK_AT - 1,
        "the observer saw every one"
    );

    // The continuation is the two-character ask and nothing else: the
    // mission's objective, next step and done-when already travel in the
    // system prompt, and repeating them here is context the task cannot spare.
    for message in agent
        .messages()
        .iter()
        .filter(|message| message.text() == MISSION_NUDGE)
    {
        assert_eq!(message.text(), "继续");
    }

    // Nothing in this file wrote the mission's own record: the flip above was
    // the test's, standing in for the model recording `blocked`.
    let record = std::fs::read_to_string(
        dir.join("missions")
            .join(MISSION_TEST_ID)
            .join("mission.json"),
    )
    .expect("the record is still there");
    assert!(
        record.contains("\"blocked\""),
        "the shell never ends a mission: {record}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn a_dropped_connection_does_not_end_a_turn_with_a_running_mission() {
    // The route accepts the request and closes without a byte — a gateway
    // restart. The mission is running, so the turn may not end there: it
    // asks again and finishes the work. Without this the first hiccup ends
    // a task someone is waiting on.
    const BLOCK_AT: usize = 2;

    struct BlockAt {
        at: usize,
        record: std::path::PathBuf,
        retries: std::sync::atomic::AtomicUsize,
    }

    impl AgentObserver for BlockAt {
        fn on_step(&self, step: usize) {
            if step == self.at {
                std::fs::write(
                    &self.record,
                    serde_json::json!({
                        "id": MISSION_TEST_ID,
                        "objective": "ship the crawler",
                        "status": "blocked",
                        "step": "the route came back and the work is done",
                        "done_when": "the tests pass",
                    })
                    .to_string(),
                )
                .expect("the mission record is writable");
            }
        }

        fn on_step_retry(&self, _attempt: usize, _reason: &str) {
            self.retries
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }

    let dir = workspace_with_mission("drop", "running");
    let (base, hits) = provider_dropping_first().await;
    let seen = Arc::new(BlockAt {
        at: BLOCK_AT,
        record: dir
            .join("missions")
            .join(MISSION_TEST_ID)
            .join("mission.json"),
        retries: std::sync::atomic::AtomicUsize::new(0),
    });
    let observing: Arc<dyn AgentObserver> = seen.clone();
    let mut agent = agent_with_tools(base, quiet_tools())
        .with_max_steps(1)
        .with_observer(observing)
        .with_identity(crate::tools::ToolIdentity::for_session(
            "test-session",
            Some(dir.clone()),
        ));

    let outcome = agent.run("go").await.expect("the loop finishes");

    assert!(outcome.error.is_none(), "{:?}", outcome.error);
    assert_eq!(outcome.text, "all done", "the turn finished its work");
    assert_eq!(outcome.steps, BLOCK_AT);
    assert!(
        outcome.retries >= 1,
        "the dropped connection was retried, not fatal (retries={} hits={} steps={})",
        outcome.retries,
        hits.load(std::sync::atomic::Ordering::Relaxed),
        outcome.steps
    );
    assert!(
        seen.retries.load(std::sync::atomic::Ordering::Relaxed) >= 1,
        "the observer saw the retry, so the page can say why it is waiting"
    );
    assert!(
        hits.load(std::sync::atomic::Ordering::Relaxed) >= DROPS + 2,
        "the dropped attempts plus two answers"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn a_mission_that_is_not_running_lets_the_turn_end_immediately() {
    // A finished mission is the model having decided the work is over — the
    // one decision the shell must not second-guess — so the answer ends the
    // turn exactly as it did before missions existed.
    let dir = workspace_with_mission("done", "done");
    let (base, hits) = provider(vec![sse_answer("all done", "stop")]).await;
    let mut agent = agent_with_tools(base, quiet_tools())
        .with_max_steps(1)
        .with_identity(crate::tools::ToolIdentity::for_session(
            "test-session",
            Some(dir.clone()),
        ));

    let outcome = agent.run("go").await.expect("the loop finishes");

    assert!(outcome.error.is_none(), "{:?}", outcome.error);
    assert_eq!(
        outcome.text, "all done",
        "the model's answer is the turn's answer"
    );
    assert_eq!(outcome.steps, 1);
    assert_eq!(
        hits.load(std::sync::atomic::Ordering::Relaxed),
        1,
        "one request, not another batch"
    );
    assert_eq!(mission_nudges(&outcome.new_messages), 0);
    assert_eq!(mission_nudges(agent.messages()), 0);
    let _ = std::fs::remove_dir_all(&dir);

    // A workspace with no mission at all is the ordinary case, and it must
    // stay exactly as cheap: one answer, one turn.
    let empty = std::env::temp_dir().join(format!(
        "ngu-agent-mission-none-{}",
        uuid::Uuid::new_v4().simple()
    ));
    std::fs::create_dir_all(&empty).expect("a scratch workspace");
    let (base, hits) = provider(vec![sse_answer("all done", "stop")]).await;
    let mut agent = agent_with_tools(base, quiet_tools())
        .with_max_steps(1)
        .with_identity(crate::tools::ToolIdentity::for_session(
            "test-session",
            Some(empty.clone()),
        ));

    let outcome = agent.run("go").await.expect("the loop finishes");

    assert_eq!(outcome.text, "all done");
    assert_eq!(hits.load(std::sync::atomic::Ordering::Relaxed), 1);
    assert_eq!(mission_nudges(&outcome.new_messages), 0);
    let _ = std::fs::remove_dir_all(&empty);
}

#[tokio::test]
async fn a_stop_request_still_ends_a_turn_with_a_running_mission() {
    // A running mission must not outlive the user's decision to stop. The
    // stop lands at the top of the second step, after the message that kept
    // the turn open is already committed — and nothing may be appended after
    // it.
    struct StopAtStep {
        at: usize,
        cancel: tokio::sync::watch::Sender<bool>,
    }

    impl AgentObserver for StopAtStep {
        fn on_step(&self, step: usize) {
            if step == self.at {
                let _ = self.cancel.send(true);
            }
        }
    }

    let dir = workspace_with_mission("stop", "running");
    let (base, hits) = provider(vec![sse_answer("all done", "stop")]).await;
    let (cancel, cancelled) = tokio::sync::watch::channel(false);
    let stopper: Arc<dyn AgentObserver> = Arc::new(StopAtStep { at: 2, cancel });
    let mut agent = agent_with_tools(base, quiet_tools())
        .with_max_steps(1)
        .with_observer(stopper)
        .with_cancel(cancelled)
        .with_identity(crate::tools::ToolIdentity::for_session(
            "test-session",
            Some(dir.clone()),
        ));

    let outcome = agent.run("go").await.expect("the loop still answers");

    assert_eq!(
        outcome.error.as_deref(),
        Some(STOPPED),
        "a stop outranks the mission: {:?}",
        outcome.error
    );
    assert_eq!(
        outcome.steps, 2,
        "the stop lands before that step's request"
    );
    assert_eq!(
        hits.load(std::sync::atomic::Ordering::Relaxed),
        1,
        "no request is made for the step the user stopped"
    );
    assert_eq!(
        mission_nudges(&outcome.new_messages),
        1,
        "the message that kept the turn open, and none after the stop"
    );
    assert!(
        outcome
            .new_messages
            .last()
            .expect("the turn committed messages")
            .text()
            == MISSION_NUDGE,
        "nothing is injected after the stop"
    );
    assert_eq!(mission_nudges(agent.messages()), 1);
    let _ = std::fs::remove_dir_all(&dir);
}

// ------------------------------------------------- a subtask is not a turn

#[tokio::test]
async fn a_subtask_stops_at_its_budget_instead_of_asking_for_more() {
    // A subtask's budget is the one its caller was told about. A top-level
    // turn that spends its steps asks whether a round count this long is
    // normal and earns another batch — right for a long task someone is
    // watching, and exactly wrong for a `delegate` whose description
    // promised twelve steps: the caller waits, sees nothing, and the tool
    // comes back after three times what was promised. The budget is spent
    // and the subtask reports failure instead of renewing.
    let (base, hits) = provider(vec![sse_tool_call("call_1", "step", "{}")]).await;
    let mut agent = agent_with_tools(base, quiet_tools()).as_subtask();

    let outcome = agent.run("go").await.expect("the loop finishes");

    assert_eq!(
        outcome.steps, 1,
        "one batch, and nothing after it: the budget is the whole budget"
    );
    assert_eq!(hits.load(std::sync::atomic::Ordering::Relaxed), 1);
    assert_eq!(
        continuation_asks(agent.messages()),
        0,
        "no subtask is asked whether a round count this long is normal"
    );
    let error = outcome
        .error
        .expect("a subtask without an answer has failed");
    assert!(error.contains("1-step budget"), "{error}");
    assert!(
        outcome.text.is_empty(),
        "the stop marker is not dressed up as an answer: {:?}",
        outcome.text
    );
}

#[tokio::test]
async fn a_subtask_is_not_held_open_by_the_parents_mission() {
    // The mission record lives in the parent's workspace, and a subtask may
    // even inherit the parent's identity with it. Holding the subtask open
    // until that mission stops running is the runaway `delegate` must not
    // produce: the parent would wait inside one tool call, invisibly, for
    // as long as its own mission lasts. The parent's turn is the one that
    // owes the mission its time; the subtask ends when it has an answer.
    let dir = workspace_with_mission("subtask", "running");
    let (base, hits) = provider(vec![sse_answer("all done", "stop")]).await;
    let mut agent = agent_with_tools(base, quiet_tools())
        .as_subtask()
        .with_identity(crate::tools::ToolIdentity::for_session(
            "test-session",
            Some(dir.clone()),
        ));

    let outcome = agent.run("go").await.expect("the loop finishes");

    assert!(outcome.error.is_none(), "{:?}", outcome.error);
    assert_eq!(outcome.text, "all done");
    assert_eq!(outcome.steps, 1, "the answer ends a subtask mid-mission");
    assert_eq!(hits.load(std::sync::atomic::Ordering::Relaxed), 1);
    assert_eq!(mission_nudges(agent.messages()), 0);
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn a_stop_reports_the_reason_the_turn_was_given() {
    // The string a stopped turn reports is read by whoever asked for it. For
    // a subtask cut off at its own wall clock that reader is the parent
    // model, and "stopped by the user" would send it looking for a user who
    // did no such thing — so the host that arms the deadline names what
    // happened, and the loop repeats it verbatim.
    let (base, hits) = provider(vec![sse_answer("all done", "stop")]).await;
    let (stop, stopped) = tokio::sync::watch::channel(true);
    let mut agent = agent_with_tools(base, quiet_tools())
        .as_subtask()
        .with_cancel(stopped)
        .with_stop_reason("the subtask's time budget ran out");

    let outcome = agent.run("go").await.expect("the loop answers the stop");

    assert_eq!(
        outcome.error.as_deref(),
        Some("the subtask's time budget ran out"),
        "{:?}",
        outcome.error
    );
    assert_eq!(
        hits.load(std::sync::atomic::Ordering::Relaxed),
        0,
        "no request is made for a turn already asked to stop"
    );
    drop(stop);
}

/// A loop that answers without asking the model, so the test can tell
/// whether the replaced loop ran at all.
struct ScriptedLoop {
    answer: String,
}

impl AgentLoop for ScriptedLoop {
    fn run<'a>(
        &'a self,
        mut state: LoopState,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = (LoopState, Result<AgentOutcome>)> + Send + 'a>,
    > {
        Box::pin(async move {
            let message = Message::assistant(&self.answer);
            state.messages.push(message.clone());
            let outcome = AgentOutcome {
                text: self.answer.clone(),
                new_messages: vec![message],
                steps: 0,
                tool_calls: 0,
                usage: TokenUsage::default(),
                timing: TurnTiming::default(),
                config_version: 0,
                injections: Vec::new(),
                compactions: Vec::new(),
                error: None,
                retries: 0,
            };
            (state, Ok(outcome))
        })
    }
}

#[tokio::test]
async fn a_replaced_loop_runs_instead_of_the_built_in_one() {
    // The point of the trait: a host that installs a loop gets that loop,
    // and the kernel does not quietly run its own anyway. No provider is
    // started here — a built-in loop would fail reaching one, so a passing
    // answer is proof the replaced loop handled the turn.
    let tools = Arc::new(ToolRegistry::with_base_tools().unwrap());
    let config: Arc<dyn TurnConfig> = Arc::new(StaticConfig::new("SYS", tools, "m"));
    let client =
        LlmClient::new(crate::llm::LlmConfig::new("http://127.0.0.1:1", "k", "m")).unwrap();
    let mut agent = Agent::new(client, config, Vec::new()).with_loop(Arc::new(ScriptedLoop {
        answer: "from the replaced loop".into(),
    }));

    let outcome = agent.run("hello").await.expect("the scripted loop answers");

    assert_eq!(outcome.text, "from the replaced loop");
    // The conversation the loop built is the one the agent keeps: a turn
    // that ran on a replaced loop must not lose its history.
    assert_eq!(agent.messages.len(), 1);
    assert_eq!(agent.messages[0].text(), "from the replaced loop");
}

#[tokio::test]
async fn the_built_in_loop_still_runs_when_none_is_installed() {
    // The default is not special-cased away: without `with_loop`, the agent
    // asks the provider exactly as it always did.
    let (base, hits) = provider(vec![sse_answer("plain answer", "stop")]).await;
    let mut agent = agent_against(base);
    let outcome = agent.run("hello").await.expect("the built-in loop answers");
    assert_eq!(outcome.text, "plain answer");
    assert_eq!(hits.load(std::sync::atomic::Ordering::SeqCst), 1);
}
