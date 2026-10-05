//! The plugin event bus: facts about what just happened, delivered to
//! whoever asked to hear them.
//!
//! # What this is — and what it deliberately is not
//!
//! Version one is **observational**: subscribers receive `(event, payload)`
//! and can do nothing but watch. There is no veto, no argument rewrite, no
//! result substitution. Those need a chain with `next()` semantics and a
//! decision about where they sit relative to the permission stack — that is
//! the hooks milestone, and until it exists this bus must not pretend:
//! a hook system that *looks* like it can deny a tool and cannot is worse
//! than one that says nothing.
//!
//! What it does today:
//!
//! * **Names are plain strings**, payloads are plain JSON — no live Cordis
//!   objects cross the boundary (they cannot be serialized honestly).
//! * **Subscribing returns a disposer.** Dropping the disposer does *not*
//!   unsubscribe — it must be called, which is the same effect discipline the
//!   plugin system already runs on: effects are undone, never GC'd.
//! * **A panicking observer is contained.** It runs inside the tool's own
//!   execution; one broken listener must not fail the call it was watching.
//! * **The bus keeps no history.** The host installs an audit observer
//!   ([`audit_observer`]) that appends every event to a JSONL file — the log
//!   lives on disk, not in memory, so a long session costs nothing here.
//!
//! # Events emitted today
//!
//! | Name | Who emits | Payload |
//! |---|---|---|
//! | `session.start` | hosts, at session creation/switch | `session` |
//! | `session.end` | hosts, at switch/delete/exit | `session` |
//! | `prompt.submit` | hosts, before a turn runs | `session`, `text` (truncated) |
//! | `turn.start` | hosts, before `agent.run` | `session` |
//! | `turn.complete` | hosts, after `agent.run` | `session`, `ok`, `duration_ms`, plus `steps`/`tool_calls` or `error` |
//! | `tool.start` | the tool table, before a handler | `tool`, `arguments` (truncated) |
//! | `tool.end` | the tool table, after a handler | `tool`, `ok`, `duration_ms`, plus `text` or `error` (truncated) |
//!
//! Truncation is part of the contract: payloads land in a file and in
//! subscribers' memory, and a 2 MB tool result must not become a 2 MB event.

use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use serde_json::{json, Value};

/// A session became current.
pub const SESSION_START: &str = "session.start";
/// A session stopped being current.
pub const SESSION_END: &str = "session.end";
/// The user submitted a prompt.
pub const PROMPT_SUBMIT: &str = "prompt.submit";
/// A turn is about to run.
pub const TURN_START: &str = "turn.start";
/// A turn finished, successfully or not.
pub const TURN_COMPLETE: &str = "turn.complete";
/// A tool handler is about to run (after lookup and argument parsing).
pub const TOOL_START: &str = "tool.start";
/// A tool handler returned, successfully or not.
pub const TOOL_END: &str = "tool.end";

/// One subscriber's callback.
type Observer = Arc<dyn Fn(&str, &Value) + Send + Sync>;

/// Removal of one subscription. Call it to unsubscribe; dropping it does
/// nothing (effects are undone, not garbage-collected).
pub type Disposer = Box<dyn FnOnce() + Send>;

#[derive(Default)]
struct EventState {
    by_name: HashMap<String, Vec<(u64, Observer)>>,
    /// Subscribers that asked for every event (the audit log).
    all: Vec<(u64, Observer)>,
}

/// The bus itself: cheap to clone, every clone shares the same subscribers.
#[derive(Clone, Default)]
pub struct EventBus {
    state: Arc<Mutex<EventState>>,
    next_id: Arc<AtomicU64>,
}

impl EventBus {
    /// Subscribe to one event name. Registration order is delivery order.
    pub fn subscribe(
        &self,
        event: &str,
        observer: impl Fn(&str, &Value) + Send + Sync + 'static,
    ) -> Disposer {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        {
            let mut state = self.state.lock().expect("event bus");
            state
                .by_name
                .entry(event.to_string())
                .or_default()
                .push((id, Arc::new(observer)));
        }
        let state = Arc::clone(&self.state);
        let event = event.to_string();
        Box::new(move || {
            let mut state = state.lock().expect("event bus");
            if let Some(list) = state.by_name.get_mut(&event) {
                list.retain(|(seen, _)| *seen != id);
            }
        })
    }

    /// Subscribe to every event, in emission order per event.
    pub fn subscribe_all(
        &self,
        observer: impl Fn(&str, &Value) + Send + Sync + 'static,
    ) -> Disposer {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        self.state
            .lock()
            .expect("event bus")
            .all
            .push((id, Arc::new(observer)));
        let state = Arc::clone(&self.state);
        Box::new(move || {
            state
                .lock()
                .expect("event bus")
                .all
                .retain(|(seen, _)| *seen != id);
        })
    }

    /// Deliver one event to name-specific subscribers, then to the wildcard
    /// list. Called with no lock held: an observer may subscribe while
    /// listening, and a deadlock in the event bus would wedge the tool call.
    pub fn emit(&self, event: &str, payload: &Value) {
        let (named, all) = {
            let state = self.state.lock().expect("event bus");
            let named: Vec<Observer> = state
                .by_name
                .get(event)
                .map(|list| list.iter().map(|(_, obs)| Arc::clone(obs)).collect())
                .unwrap_or_default();
            let all: Vec<Observer> = state.all.iter().map(|(_, obs)| Arc::clone(obs)).collect();
            (named, all)
        };
        for observer in named.into_iter().chain(all) {
            // A subscriber runs inside someone else's execution. If it
            // panics, the execution continues without it — reported, because
            // silence about a broken listener is how it stays broken.
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                observer(event, payload)
            }));
            if result.is_err() {
                eprintln!("[events] an observer of '{event}' panicked; continuing");
            }
        }
    }
}

/// Cut a string to `max` characters with a marker, so payloads stay bounded.
pub fn truncate(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_string();
    }
    let kept: String = text.chars().take(max).collect();
    format!("{kept}…[+{} chars]", text.chars().count() - max)
}

/// An observer that appends every event to a JSONL file.
///
/// Rotation happens once the file passes `max_bytes`: the current file is
/// renamed to `<name>.1`, replacing any previous rotation, so the log is
/// bounded at roughly twice `max_bytes` with one generation kept. All
/// failures are swallowed on purpose — an unwritable audit log must not fail
/// the turn it was watching (the events themselves are still delivered to
/// other subscribers).
pub fn audit_observer(path: PathBuf, max_bytes: u64) -> impl Fn(&str, &Value) + Send + Sync {
    move |event, payload| {
        let line = json!({
            "ts": std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis())
                .unwrap_or(0),
            "event": event,
            "payload": payload,
        });
        let Ok(mut line) = serde_json::to_string(&line) else {
            return;
        };
        line.push('\n');

        if let Ok(meta) = std::fs::metadata(&path) {
            if meta.len() + line.len() as u64 > max_bytes {
                let rotated = path.with_extension("jsonl.1");
                let _ = std::fs::remove_file(&rotated);
                let _ = std::fs::rename(&path, &rotated);
            }
        }
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
        {
            let _ = file.write_all(line.as_bytes());
        }
    }
}

/// The default audit path: `<data>/events.jsonl`.
pub fn audit_path() -> PathBuf {
    crate::settings::data_dir().join("events.jsonl")
}

/// Install the audit observer on a bus; the subscription stays for the life
/// of the bus (dropping the returned disposer does not remove it — call it to).
pub fn install_audit(bus: &EventBus, path: PathBuf) -> Disposer {
    bus.subscribe_all(audit_observer(path, 5 * 1024 * 1024))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn subscribers_see_events_in_registration_order_with_the_payload() {
        let bus = EventBus::default();
        let order = Arc::new(Mutex::new(Vec::new()));
        let seen = Arc::new(Mutex::new(Vec::new()));

        let a = Arc::clone(&order);
        bus.subscribe("tool.end", move |name, payload| {
            a.lock().unwrap().push(format!("a:{name}:{}", payload["ok"]));
        });
        let b = Arc::clone(&order);
        bus.subscribe("tool.end", move |name, _| {
            b.lock().unwrap().push(format!("b:{name}"));
        });
        let s = Arc::clone(&seen);
        bus.subscribe_all(move |name, _| s.lock().unwrap().push(name.to_string()));

        bus.emit("tool.end", &json!({ "ok": true }));
        assert_eq!(
            order.lock().unwrap().as_slice(),
            ["a:tool.end:true", "b:tool.end"]
        );
        assert_eq!(seen.lock().unwrap().as_slice(), ["tool.end"]);
    }

    #[test]
    fn a_disposer_unsubscribes_only_when_called() {
        let bus = EventBus::default();
        let count = Arc::new(AtomicUsize::new(0));
        let inner = Arc::clone(&count);
        let dispose = bus.subscribe("x", move |_, _| {
            inner.fetch_add(1, Ordering::Relaxed);
        });

        bus.emit("x", &json!({}));
        // Dropping the disposer must not unsubscribe: effects are undone by
        // calling, not by going out of scope.
        let dispose = dispose;
        dispose();
        bus.emit("x", &json!({}));
        assert_eq!(count.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn a_panicking_observer_cannot_stop_the_run() {
        let bus = EventBus::default();
        let after = Arc::new(AtomicUsize::new(0));
        let inner = Arc::clone(&after);
        bus.subscribe("x", |_, _| panic!("broken listener"));
        bus.subscribe("x", move |_, _| {
            inner.fetch_add(1, Ordering::Relaxed);
        });

        bus.emit("x", &json!({}));
        assert_eq!(after.load(Ordering::Relaxed), 1, "the second still ran");
    }

    #[test]
    fn payloads_and_strings_stay_bounded() {
        let long = "x".repeat(50);
        assert!(truncate(&long, 10).ends_with("[+40 chars]"));
        assert_eq!(truncate("short", 10), "short");
    }

    #[test]
    fn the_audit_log_appends_parseable_lines_and_rotates() {
        let dir = std::env::temp_dir().join(format!(
            "ngu-events-{}",
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("events.jsonl");

        // A small threshold makes rotation testable without writing
        // megabytes. 180 keeps two events on the first pass and rotates on
        // the third, so the first generation survives in `.1`.
        let audit = audit_observer(path.clone(), 180);
        audit("session.start", &json!({ "session": "s1" }));
        audit("turn.start", &json!({ "session": "s1" }));
        audit("tool.end", &json!({ "tool": "read", "ok": true }));

        let rotated = dir.join("events.jsonl.1");
        assert!(rotated.exists(), "the file rotated past the threshold");
        let current = std::fs::read_to_string(&path).unwrap();
        let last = current.lines().last().expect("a line in the current file");
        let parsed: Value = serde_json::from_str(last).unwrap();
        assert_eq!(parsed["event"], "tool.end");
        assert!(parsed["ts"].as_u64().unwrap() > 0);
        assert_eq!(parsed["payload"]["tool"], "read");

        let first: Value = serde_json::from_str(
            std::fs::read_to_string(&rotated)
                .unwrap()
                .lines()
                .next()
                .unwrap(),
        )
        .unwrap();
        assert_eq!(first["event"], "session.start");

        std::fs::remove_dir_all(&dir).ok();
    }
}
