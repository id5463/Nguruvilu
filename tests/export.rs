//! Export tests across the seam.
//!
//! The unit tests beside `src/export.rs` cover the rendering itself. These check
//! the command: that a session written through the store reaches the export,
//! that both formats carry the conversation and the ledger, that `--out` writes
//! the same bytes stdout would have carried, and that an id with no session
//! fails instead of producing an empty file that looks like a real export.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use nguruvilu::message::Message;
use nguruvilu::session::{now_iso, JsonlStore, Session, SessionStore, TurnStats};

// ------------------------------------------------------------------ helpers

fn temp_root(tag: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!(
        "ngu-export-it-{tag}-{}",
        uuid::Uuid::new_v4().simple()
    ));
    std::fs::create_dir_all(&root).expect("scratch root");
    root
}

/// One recorded turn with round token counts, so a ratio can be checked exactly.
fn turn() -> TurnStats {
    TurnStats {
        ts: now_iso(),
        steps: 1,
        tool_calls: 2,
        retries: 0,
        input: 400,
        output: 30,
        cached: 200,
        cache_created: 50,
        reasoning: 10,
        model_ms: 300,
        tools_ms: 25,
    }
}

/// A session with a short conversation and one recorded turn, plus the root it
/// lives in so the caller can point `--home` at it and clean it up.
fn seeded(tag: &str) -> (PathBuf, String) {
    let root = temp_root(tag);
    let store = JsonlStore::open(&root).expect("open store");

    let mut session = Session::new(Some("export-model".into()));
    // Set before the header is written: `Session::new` deliberately inherits no
    // working directory, and the export's fact block has to have one to report.
    session.cwd = Some("C:/export/work".into());
    store.create(&session).expect("create");
    store
        .append(
            &session.id,
            &[
                Message::user("which file holds the greeting"),
                Message::assistant("src/greeting.rs"),
                Message::user("thanks"),
            ],
        )
        .expect("append");
    store.append_stats(&session.id, &turn()).expect("stats");

    (root, session.id)
}

/// Run `ngu` against a throwaway home, so the test never reads the real one.
fn ngu(home: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_ngu"))
        .arg("--home")
        .arg(home)
        .args(args)
        .output()
        .expect("run ngu")
}

// -------------------------------------------------------------------- tests

#[test]
fn the_json_export_carries_the_conversation_and_the_ledger() {
    let (root, id) = seeded("json");

    let output = ngu(&root, &["export", id.as_str(), "--format", "json"]);
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let doc: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("the export is one JSON object");

    assert_eq!(doc["session"]["id"], id.as_str());
    assert_eq!(doc["session"]["model"], "export-model");
    assert_eq!(doc["session"]["cwd"], "C:/export/work");
    assert!(doc["session"]["created_at"].is_string(), "{}", doc["session"]);

    let messages = doc["messages"].as_array().expect("a messages array");
    assert_eq!(messages.len(), 3);
    assert_eq!(messages[0]["role"], "user");
    assert_eq!(messages[0]["text"], "which file holds the greeting");
    let texts: Vec<&str> = messages.iter().filter_map(|m| m["text"].as_str()).collect();
    assert!(texts.contains(&"src/greeting.rs"), "{texts:?}");
    assert!(texts.contains(&"thanks"), "{texts:?}");

    // The recorded turn travels verbatim, and the totals use the stats
    // command's names and its fraction-not-percentage ratio.
    assert_eq!(doc["turns"][0]["steps"], 1);
    assert_eq!(doc["turns"][0]["cache_created"], 50);
    assert_eq!(doc["totals"]["turns"], 1);
    assert_eq!(doc["totals"]["input"], 400);
    assert_eq!(doc["totals"]["cached"], 200);
    let ratio = doc["totals"]["cache_hit_ratio"].as_f64().expect("a ratio");
    assert!((ratio - 0.5).abs() < 1e-9, "{ratio}");

    std::fs::remove_dir_all(&root).ok();
}

#[test]
fn the_default_export_is_markdown_naming_each_role() {
    let (root, id) = seeded("markdown");

    let output = ngu(&root, &["export", id.as_str()]);
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8(output.stdout).expect("the export is UTF-8");

    for expected in [
        "## user",
        "## assistant",
        "which file holds the greeting",
        "src/greeting.rs",
        "thanks",
    ] {
        assert!(text.contains(expected), "missing {expected:?} in:\n{text}");
    }
    assert!(text.contains("- messages: 3"), "{text}");
    assert!(text.contains("- recorded turns: 1"), "{text}");
    assert!(text.contains("- cache hit ratio: 50.0%"), "{text}");

    std::fs::remove_dir_all(&root).ok();
}

#[test]
fn writing_to_a_file_produces_the_bytes_stdout_produced() {
    let (root, id) = seeded("out");
    // A directory that does not exist yet: `--out` creates what it needs.
    let path = root.join("nested").join("dir").join("session.md");

    let to_file = ngu(
        &root,
        &["export", id.as_str(), "--out", path.to_str().expect("a UTF-8 path")],
    );
    assert!(
        to_file.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&to_file.stderr)
    );
    let written = std::fs::read(&path).expect("the exported file");

    let to_stdout = ngu(&root, &["export", id.as_str()]);
    assert!(
        to_stdout.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&to_stdout.stderr)
    );
    assert_eq!(
        written, to_stdout.stdout,
        "a file export and a stdout export must be the same bytes"
    );

    // The one line printed says where it went and how big it is.
    let note = String::from_utf8_lossy(&to_file.stdout);
    assert!(note.contains(&path.display().to_string()), "{note}");
    assert!(note.contains(&format!("{} bytes", written.len())), "{note}");

    std::fs::remove_dir_all(&root).ok();
}

#[test]
fn an_unknown_session_id_fails_and_names_the_id() {
    let (root, _id) = seeded("missing");

    let output = ngu(&root, &["export", "no-such-session"]);
    assert!(
        !output.status.success(),
        "exporting a session that does not exist must fail"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("no-such-session"), "stderr: {stderr}");
    // Nothing goes to stdout, so a caller piping the export cannot end up with
    // an empty file that looks like a real one.
    assert!(
        output.stdout.is_empty(),
        "stdout: {}",
        String::from_utf8_lossy(&output.stdout)
    );

    std::fs::remove_dir_all(&root).ok();
}

#[test]
fn an_unknown_format_names_the_ones_it_accepts() {
    let (root, id) = seeded("format");

    let output = ngu(&root, &["export", id.as_str(), "--format", "yaml"]);
    assert!(!output.status.success(), "an unknown format must fail");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("yaml"), "stderr: {stderr}");
    assert!(stderr.contains("json"), "stderr: {stderr}");
    assert!(stderr.contains("markdown"), "stderr: {stderr}");

    std::fs::remove_dir_all(&root).ok();
}
