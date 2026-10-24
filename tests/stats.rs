//! Usage-ledger tests across the seams.
//!
//! The unit tests beside `session.rs` cover the rows themselves; these check
//! that a turn recorded through the store reads back through the aggregation,
//! that a session written before the ledger existed still loads, and that
//! `ngu stats` reports exactly what is on disk.

use std::path::{Path, PathBuf};
use std::process::Command;

use nguruvilu::message::Message;
use nguruvilu::session::{
    now_iso, totals_of, JsonlStore, Session, SessionStore, SessionTotals, TurnStats,
};

// ------------------------------------------------------------------ helpers

fn temp_root(tag: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!(
        "ngu-stats-it-{tag}-{}",
        uuid::Uuid::new_v4().simple()
    ));
    std::fs::create_dir_all(&root).expect("scratch root");
    root
}

fn turn(steps: u32, input: u64, cached: u64) -> TurnStats {
    TurnStats {
        ts: now_iso(),
        steps,
        tool_calls: 3,
        retries: 1,
        input,
        output: 11,
        cached,
        cache_created: 6,
        reasoning: 4,
        model_ms: 250,
        tools_ms: 40,
    }
}

/// A session file with a header and one message, written by hand: the shape
/// every session had before usage rows existed.
fn write_legacy_session(root: &Path, id: &str) -> PathBuf {
    let path = root.join(format!("{id}.jsonl"));
    let now = now_iso();
    let header = serde_json::json!({
        "type": "header",
        "id": id,
        "created_at": now,
        "updated_at": now,
        "model": "test-model",
    });
    // Built from the real message type so the fixture cannot drift from the
    // format: whatever `Message` serialises to is what this file holds.
    let mut message = serde_json::to_value(Message::user("an old prompt")).expect("message json");
    message["type"] = serde_json::json!("message");
    std::fs::write(
        &path,
        format!(
            "{}\n{}\n",
            serde_json::to_string(&header).expect("header json"),
            serde_json::to_string(&message).expect("message line")
        ),
    )
    .expect("write legacy session");
    path
}

// -------------------------------------------------------------------- tests

#[test]
fn a_recorded_turn_reads_back_through_totals_of() {
    let root = temp_root("totals");
    let store = JsonlStore::open(&root).expect("open store");

    let session = Session::new(Some("test-model".into()));
    store.create(&session).expect("create");
    store
        .append(&session.id, &[Message::user("hello"), Message::assistant("hi")])
        .expect("append");
    store.append_stats(&session.id, &turn(2, 100, 25)).expect("stats 1");
    store.append_stats(&session.id, &turn(5, 300, 275)).expect("stats 2");

    let loaded = store.load(&session.id).expect("load").expect("exists");
    let totals = totals_of(&loaded);
    assert_eq!(totals.turns, 2);
    assert_eq!(totals.steps, 7);
    assert_eq!(totals.tool_calls, 6);
    assert_eq!(totals.input, 400);
    assert_eq!(totals.output, 22);
    assert_eq!(totals.cached, 300);
    assert!(
        (totals.cache_hit_ratio() - 0.75).abs() < 1e-9,
        "cached/input: {}",
        totals.cache_hit_ratio()
    );

    // The conversation itself is unaffected by the ledger beside it.
    assert_eq!(loaded.messages.len(), 2);
    assert_eq!(loaded.title().as_deref(), Some("hello"));

    std::fs::remove_dir_all(&root).ok();
}

#[test]
fn a_session_written_before_the_ledger_totals_to_zero() {
    let root = temp_root("legacy");
    let store = JsonlStore::open(&root).expect("open store");
    write_legacy_session(&root, "legacy-session");

    let loaded = store
        .load("legacy-session")
        .expect("a file with no usage rows still loads")
        .expect("exists");
    assert!(loaded.stats.is_empty());

    let totals = totals_of(&loaded);
    assert_eq!(totals, SessionTotals::default());
    assert_eq!(totals.cache_hit_ratio(), 0.0);

    std::fs::remove_dir_all(&root).ok();
}

#[test]
fn the_stats_command_reports_what_was_written() {
    let root = temp_root("cli");
    let store = JsonlStore::open(&root).expect("open store");
    let session = Session::new(Some("test-model".into()));
    store.create(&session).expect("create");
    store
        .append(&session.id, &[Message::user("hi")])
        .expect("append");
    store.append_stats(&session.id, &turn(1, 120, 60)).expect("stats 1");
    store.append_stats(&session.id, &turn(3, 240, 200)).expect("stats 2");

    let output = Command::new(env!("CARGO_BIN_EXE_ngu"))
        .arg("--home")
        .arg(&root)
        .arg("--json")
        .arg("stats")
        .arg(&session.id)
        .output()
        .expect("run ngu stats");
    assert!(
        output.status.success(),
        "ngu stats failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let doc: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("stats --json is valid JSON");
    assert_eq!(doc["sessions"][0]["session"], session.id.as_str());
    assert_eq!(doc["sessions"][0]["found"], true);
    assert_eq!(doc["sessions"][0]["totals"]["turns"], 2);
    assert_eq!(doc["sessions"][0]["totals"]["input"], 360);
    assert_eq!(doc["sessions"][0]["totals"]["cached"], 260);
    // The per-turn tail is included for one session, so a spike can be traced.
    assert_eq!(
        doc["sessions"][0]["turns"].as_array().map(|t| t.len()),
        Some(2)
    );
    assert_eq!(doc["total"]["turns"], 2);

    std::fs::remove_dir_all(&root).ok();
}

#[test]
fn the_stats_command_tolerates_a_missing_or_untracked_session() {
    let root = temp_root("cli-missing");
    let store = JsonlStore::open(&root).expect("open store");
    write_legacy_session(&root, "legacy-session");

    // A session that predates the ledger: zeros, and a successful exit.
    let legacy = Command::new(env!("CARGO_BIN_EXE_ngu"))
        .arg("--home")
        .arg(&root)
        .arg("--json")
        .arg("stats")
        .arg("legacy-session")
        .output()
        .expect("run ngu stats");
    assert!(
        legacy.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&legacy.stderr)
    );
    let doc: serde_json::Value = serde_json::from_slice(&legacy.stdout).expect("json");
    assert_eq!(doc["sessions"][0]["totals"]["turns"], 0);
    assert_eq!(doc["sessions"][0]["totals"]["cache_hit_ratio"], 0.0);

    // An id with no file at all: zeros rather than an error.
    let missing = Command::new(env!("CARGO_BIN_EXE_ngu"))
        .arg("--home")
        .arg(&root)
        .arg("stats")
        .arg("no-such-session")
        .output()
        .expect("run ngu stats");
    assert!(
        missing.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&missing.stderr)
    );
    let text = String::from_utf8_lossy(&missing.stdout);
    assert!(text.contains("not found"), "{text}");

    // `--all` covers every session — the untracked one and a tracked one — and
    // ends with a grand total.
    let tracked = Session::new(Some("test-model".into()));
    store.create(&tracked).expect("create");
    store.append_stats(&tracked.id, &turn(1, 90, 30)).expect("stats");

    let all = Command::new(env!("CARGO_BIN_EXE_ngu"))
        .arg("--home")
        .arg(&root)
        .arg("stats")
        .arg("--all")
        .output()
        .expect("run ngu stats --all");
    assert!(
        all.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&all.stderr)
    );
    let text = String::from_utf8_lossy(&all.stdout);
    assert!(text.contains("legacy-session"), "{text}");
    assert!(text.contains(&tracked.id), "{text}");
    // One session recorded a turn, the other none: the grand total counts the
    // one and the untracked session still reports nothing rather than an error.
    let (_, total_block) = text
        .split_once("session total")
        .unwrap_or_else(|| panic!("no grand total line: {text}"));
    assert!(total_block.contains("turns 1"), "{total_block}");
    assert!(text.contains("no turns recorded"), "{text}");

    std::fs::remove_dir_all(&root).ok();
}

/// A turn with exactly the token counts a price can be checked against.
fn priced_turn(input: u64, output: u64, cached: u64, cache_created: u64) -> TurnStats {
    TurnStats {
        ts: now_iso(),
        steps: 1,
        tool_calls: 0,
        retries: 0,
        input,
        output,
        cached,
        cache_created,
        reasoning: 0,
        model_ms: 10,
        tools_ms: 0,
    }
}

#[test]
fn the_stats_command_turns_tokens_into_money_when_it_is_given_rates() {
    let root = temp_root("cli-cost");
    // `NGU_HOME` is the state home for this run, so the default price file is
    // `$NGU_HOME/prices.json` and the store is `$NGU_HOME/sessions` — the same
    // layout the desktop uses.
    let store = JsonlStore::open(root.join("sessions")).expect("open store");
    let session = Session::new(Some("priced-model".into()));
    store.create(&session).expect("create");
    // One million prompt tokens, of which 200k came from cache and 100k were
    // written to it, plus 500k completion tokens.
    store
        .append_stats(&session.id, &priced_turn(1_000_000, 500_000, 200_000, 100_000))
        .expect("stats");

    std::fs::write(
        root.join("prices.json"),
        r#"{"priced-model": {"input": 1.0, "output": 4.0, "cache_read": 0.1, "cache_write": 2.0}}"#,
    )
    .expect("write the price file");

    let priced = Command::new(env!("CARGO_BIN_EXE_ngu"))
        .arg("stats")
        .arg("--cost")
        .arg("--json")
        .env("NGU_HOME", &root)
        .output()
        .expect("run ngu stats --cost --json");
    assert!(
        priced.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&priced.stderr)
    );
    let doc: serde_json::Value = serde_json::from_slice(&priced.stdout).expect("json");
    let cost = &doc["sessions"][0]["cost"];
    assert_eq!(cost["priced"], true, "{cost}");
    assert_eq!(cost["model"], "priced-model");
    // 0.7M at 1.0 + 0.2M at 0.1 + 0.1M at 2.0 + 0.5M at 4.0 = 2.92
    let amount = cost["amount"].as_f64().expect("an amount");
    assert!((amount - 2.92).abs() < 1e-6, "{amount}");
    assert_eq!(cost["fresh_input"], 700_000);
    // The grand total is priced too, and equals the one session.
    assert_eq!(doc["total_cost"]["priced"], true, "{}", doc["total_cost"]);
    assert!((doc["total_cost"]["amount"].as_f64().unwrap_or(-1.0) - 2.92).abs() < 1e-6);

    let human = Command::new(env!("CARGO_BIN_EXE_ngu"))
        .arg("stats")
        .arg("--cost")
        .env("NGU_HOME", &root)
        .output()
        .expect("run ngu stats --cost");
    assert!(
        human.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&human.stderr)
    );
    let text = String::from_utf8_lossy(&human.stdout);
    assert!(text.contains("cost    2.9200 (priced-model)"), "{text}");

    // With no price file, the report says so instead of printing zero.
    std::fs::remove_file(root.join("prices.json")).expect("remove the price file");
    let unpriced = Command::new(env!("CARGO_BIN_EXE_ngu"))
        .arg("stats")
        .arg("--cost")
        .env("NGU_HOME", &root)
        .output()
        .expect("run ngu stats --cost without a price file");
    assert!(unpriced.status.success());
    let text = String::from_utf8_lossy(&unpriced.stdout);
    assert!(text.contains("cost    unpriced"), "{text}");
    assert!(text.contains("no price table"), "{text}");

    // A price file that cannot be parsed is an error that names the file, not a
    // report of free turns.
    std::fs::write(root.join("prices.json"), "{ not json").expect("write a broken file");
    let broken = Command::new(env!("CARGO_BIN_EXE_ngu"))
        .arg("stats")
        .arg("--cost")
        .env("NGU_HOME", &root)
        .output()
        .expect("run ngu stats --cost with a broken price file");
    assert!(!broken.status.success(), "a broken price file must fail");
    assert!(
        String::from_utf8_lossy(&broken.stderr).contains("prices.json"),
        "stderr: {}",
        String::from_utf8_lossy(&broken.stderr)
    );

    std::fs::remove_dir_all(&root).ok();
}
