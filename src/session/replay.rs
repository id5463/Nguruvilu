//! Replaying a stored session: folding compactions back in, and the helpers a
//! session is created with.

use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use sha2::{Digest, Sha256};

use crate::message::Message;

use super::jsonl::{CompactionLine, JsonlLine, JsonlStore};
use super::model::{Session, SessionTotals, TurnStats};

/// Sum a session's recorded turns.
///
/// A session with no stats lines — every session written before this existed —
/// totals to all zeros rather than failing: the absence of a ledger is not an
/// error, it is a session with nothing recorded.
pub fn totals_of(session: &Session) -> SessionTotals {
    session.stats.iter().map(totals_of_turn).sum()
}

/// One turn's contribution to [`SessionTotals`].
fn totals_of_turn(stats: &TurnStats) -> SessionTotals {
    SessionTotals {
        turns: 1,
        steps: stats.steps as u64,
        tool_calls: stats.tool_calls as u64,
        input: stats.input,
        output: stats.output,
        cached: stats.cached,
        cache_created: stats.cache_created,
        reasoning: stats.reasoning,
        model_ms: stats.model_ms,
        tools_ms: stats.tools_ms,
    }
}

/// How many hex characters of the digest identify a replaced run.
///
/// Sixteen hex characters is 64 bits — enough that a dropped or substituted
/// message is caught — while the line stays short enough to read by eye.
const FINGERPRINT_HEX: usize = 16;

/// A short, order-sensitive fingerprint of a run of messages.
///
/// Length-prefixed fields rather than a delimiter: two different runs must not
/// hash the same because a message happened to contain the separator. Every
/// part of a message the model saw or produced is covered — role, text, tool
/// calls with their arguments, the tool result id, and where an image points.
pub(crate) fn fingerprint_of(messages: &[Message]) -> String {
    let mut hasher = Sha256::new();
    for message in messages {
        feed(&mut hasher, message.role.as_wire().as_bytes());
        feed(&mut hasher, message.text().as_bytes());
        hasher.update((message.tool_calls.len() as u64).to_le_bytes());
        for call in &message.tool_calls {
            feed(&mut hasher, call.id.as_bytes());
            feed(&mut hasher, call.name.as_bytes());
            feed(&mut hasher, call.arguments.as_bytes());
        }
        feed(
            &mut hasher,
            message.tool_call_id.as_deref().unwrap_or("").as_bytes(),
        );
        hasher.update((message.images.len() as u64).to_le_bytes());
        for image in &message.images {
            feed(&mut hasher, image.url.as_bytes());
        }
    }
    let hex = format!("{:x}", hasher.finalize());
    hex[..FINGERPRINT_HEX.min(hex.len())].to_string()
}

/// Feed one length-prefixed field into the digest.
fn feed(hasher: &mut Sha256, bytes: &[u8]) {
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

/// Apply the recorded compactions to the stored messages.
///
/// Returns the reduced history and one note per record that could not be
/// applied as written.
///
/// A record is only folded when it still describes the messages in front of it.
/// When it carries anchors, the span is found by them: that is the whole reason
/// they are stored, because a count says "the first N" and a missing line makes
/// N point somewhere the summary never stood in for. When it carries only a
/// count — the shape every older file has — the count is trusted as it always
/// was, with the fingerprint as its guard.
pub(crate) fn replay_compactions(
    messages: Vec<Message>,
    records: &[CompactionLine],
) -> (Vec<Message>, Vec<String>) {
    let mut history = messages;
    let mut drift = Vec::new();
    for (index, record) in records.iter().enumerate() {
        let number = index + 1;
        if record.replaced == 0 {
            drift.push(format!(
                "compaction #{number}: replaces nothing (replaced 0); not applied"
            ));
            continue;
        }
        // The anchors decide when they are present. Whether they resolve or not,
        // the count is not used in their place: a record that cannot find the
        // messages it replaced cannot say which messages its summary stood in
        // for, and folding at the recorded offset anyway is the silent deletion
        // this anchor exists to prevent.
        match (&record.first_anchor, &record.last_anchor) {
            (Some(first), Some(last)) => {
                match crate::compaction::locate_span(&history, record.replaced, first, last) {
                    Ok((start, end)) => {
                        // The located span is also held to the fingerprint: two
                        // independent checks on one decision cost nothing, and
                        // the anchors alone cannot see a substitution in the
                        // middle of an otherwise well-placed run.
                        if let Some(expected) = &record.replaced_fingerprint {
                            let actual = fingerprint_of(&history[start..=end]);
                            if &actual != expected {
                                drift.push(format!(
                                    "compaction #{number}: the run it located does not match the \
                                     record (recorded {expected}, found {actual}); kept all {} \
                                     messages",
                                    history.len()
                                ));
                                continue;
                            }
                        }
                        history = fold_span(&history, start, end, record);
                    }
                    Err(reason) => drift.push(format!("compaction #{number}: {reason}")),
                }
                continue;
            }
            // Half a pair brackets nothing. The missing end would have to be
            // derived from the count, which is the assumption anchors are here
            // to stop relying on, so the record is reported as unreadable.
            (None, None) => {}
            _ => {
                drift.push(format!(
                    "compaction #{number}: the record states only one end of the run it \
                     replaced; not applied, kept all {} messages",
                    history.len()
                ));
                continue;
            }
        }
        if record.replaced > history.len() {
            drift.push(format!(
                "compaction #{number}: replaces {} messages but only {} are stored; not applied",
                record.replaced,
                history.len()
            ));
            continue;
        }
        if let Some(expected) = &record.replaced_fingerprint {
            let actual = fingerprint_of(&history[..record.replaced]);
            if &actual != expected {
                drift.push(format!(
                    "compaction #{number}: the stored messages do not match the record \
                     (recorded {expected}, found {actual}); kept all {} messages",
                    history.len()
                ));
                continue;
            }
        }
        // No anchors: a record written before they existed. The count is
        // trusted exactly as it was before this guard — this is the
        // compatibility path, not an oversight.
        history = fold(&history, record);
    }
    (history, drift)
}

/// `[summary] + the messages after the replaced run`, for a record that states
/// only a count.
fn fold(history: &[Message], record: &CompactionLine) -> Vec<Message> {
    fold_span(history, 0, record.replaced - 1, record)
}

/// `[anything before the span] + [summary] + everything after the span`.
///
/// The span is a leading one in the ordinary case, and this is then exactly the
/// fold the count alone used to perform: one summary standing where the run was.
/// Honouring a non-zero `start` matters only when an anchor found the run
/// further down the transcript — a message that was written later but now sits
/// in front of the run — and moving the summary ahead of that message there
/// would reorder a conversation that was never reordered.
fn fold_span(history: &[Message], start: usize, end: usize, record: &CompactionLine) -> Vec<Message> {
    let mut next = Vec::with_capacity(history.len() - (end - start));
    next.extend(history[..start].iter().cloned());
    next.push(record.summary_message());
    next.extend(history[end + 1..].iter().cloned());
    next
}

pub(crate) fn read_first_user_message(path: &Path) -> Option<String> {
    let file = File::open(path).ok()?;
    let reader = BufReader::new(file);
    for line in reader.lines().map_while(Result::ok) {
        if let Ok(JsonlLine::Message(message)) = serde_json::from_str::<JsonlLine>(&line) {
            if message.role == crate::message::Role::User {
                let text = message.text().trim().replace('\n', " ");
                if text.is_empty() {
                    continue;
                }
                return Some(if text.chars().count() > 60 {
                    format!("{}…", text.chars().take(60).collect::<String>())
                } else {
                    text
                });
            }
        }
    }
    None
}

/// Current time as an ISO-8601 string.
pub fn now_iso() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
}

/// A fresh session id: timestamp plus random suffix, filename-safe.
pub fn new_id() -> String {
    let stamp = chrono::Utc::now().format("%Y%m%d-%H%M%S");
    let suffix = uuid::Uuid::new_v4().simple().to_string();
    format!("{stamp}-{}", &suffix[..6])
}

/// The private working directory of one session: `<root>/workspaces/<id>`.
///
/// `root` is chosen by the caller (the desktop derives it from its data
/// directory). This function only joins path segments; it does not touch
/// the filesystem. Ids that would escape `root` are rejected, the same rule
/// the session store applies to file stems.
pub fn workspace_dir(root: &Path, id: &str) -> Result<PathBuf> {
    JsonlStore::validate_id(id)?;
    Ok(root.join("workspaces").join(id))
}

/// Create the working directory of session `id` and return its path.
///
/// Fails when the directory cannot be created. Callers must not fall back to
/// another directory on failure: a session that silently ran in a shared or
/// process directory is the bug this function exists to prevent.
pub fn ensure_workspace(root: &Path, id: &str) -> Result<PathBuf> {
    let dir = workspace_dir(root, id)?;
    fs::create_dir_all(&dir).with_context(|| format!("creating workspace {}", dir.display()))?;
    Ok(dir)
}
