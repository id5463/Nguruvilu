//! Sessions and durable storage.
//!
//! Storage is a trait so the backend can be replaced (SQLite later) without
//! touching callers. The shipped backend is JSONL: a header line followed by
//! one line per message.
//!
//! JSONL is append-only, which is what makes it crash-safe — a truncated
//! final line costs at most the last message, and nothing before it is lost.

use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::message::Message;

/// A conversation with its metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    /// Stable id, also the file stem.
    pub id: String,
    /// ISO-8601 creation time.
    pub created_at: String,
    /// ISO-8601 last-update time.
    pub updated_at: String,
    /// Working directory this session was started in.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cwd: Option<String>,
    /// Model route used.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Packs this session has loaded, by pack label. Persisted so that a
    /// restart still knows which sessions depend on a pack: an unload in one
    /// conversation must not remove a pack another conversation still uses.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub packs: Vec<String>,
    /// Conversation history.
    #[serde(default)]
    pub messages: Vec<Message>,
    /// Usage of every turn this session ran, oldest first.
    ///
    /// Kept beside the conversation rather than inside it: a compaction folds
    /// messages away, but what a conversation cost is a ledger, not dialogue.
    #[serde(default)]
    pub stats: Vec<TurnStats>,
    /// Compaction records that no longer line up with the stored messages.
    ///
    /// Empty for a healthy file. One entry per record whose replaced run does
    /// not match what the file actually holds — a dropped line, a truncated
    /// append, an outer tool that rewrote the file. Such a record is *not*
    /// applied: folding the wrong messages would delete history silently, and
    /// this field is how a caller (or the log) can say so out loud.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub drift: Vec<String>,
}

impl Session {
    /// Start a new session with a generated id and no working directory.
    ///
    /// The caller assigns [`Session::cwd`] once it has created the session's
    /// workspace with [`ensure_workspace`]. Leaving it unset here means a
    /// session never inherits the process directory by accident.
    pub fn new(model: Option<String>) -> Self {
        let now = now_iso();
        Self {
            id: new_id(),
            created_at: now.clone(),
            updated_at: now,
            cwd: None,
            model,
            packs: Vec::new(),
            messages: Vec::new(),
            stats: Vec::new(),
            drift: Vec::new(),
        }
    }

    /// A short title derived from the first user message.
    pub fn title(&self) -> Option<String> {
        self.messages
            .iter()
            .find(|m| m.role == crate::message::Role::User)
            .map(|m| {
                let text = m.text().trim().replace('\n', " ");
                if text.chars().count() > 60 {
                    format!("{}…", text.chars().take(60).collect::<String>())
                } else {
                    text
                }
            })
            .filter(|t| !t.is_empty())
    }
}

/// Summary of a stored session, for listing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSummary {
    /// Session id.
    pub id: String,
    /// Creation time.
    pub created_at: String,
    /// Last update time.
    pub updated_at: String,
    /// Number of stored messages.
    pub message_count: usize,
    /// Derived title, when the session has a user message.
    pub title: Option<String>,
}

/// One turn's usage, as recorded in the session log.
///
/// Written as its own JSONL line at the end of every turn. The cached-prefix
/// plumbing (parsing, summing) already existed; without this row it was printed
/// once and lost, so "what did this session cost" was unanswerable after the
/// fact.
///
/// A **struct** variant, not a bare sequence: the line type is internally
/// tagged, and serde can only buffer a struct into a map — a newtype around a
/// `Vec` fails to serialize at all (the lesson from the `Packs` line).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct TurnStats {
    /// When the turn ended, RFC3339.
    pub ts: String,
    /// Model steps taken.
    pub steps: u32,
    /// Tool calls executed.
    pub tool_calls: u32,
    /// Step retries.
    pub retries: u32,
    /// Prompt tokens, provider semantics — cached tokens included.
    pub input: u64,
    /// Completion tokens.
    pub output: u64,
    /// Prompt tokens the provider served from its cache.
    pub cached: u64,
    /// Prompt tokens the provider wrote *into* its cache this turn.
    ///
    /// Separate from `cached` because the two are billed differently: a write
    /// usually costs a premium and a read a discount, so a ledger that folds
    /// them together cannot be turned into a bill.
    pub cache_created: u64,
    /// Reasoning tokens, when the provider reports them.
    pub reasoning: u64,
    /// Time spent waiting on the model.
    pub model_ms: u64,
    /// Time spent executing tools.
    pub tools_ms: u64,
}

impl TurnStats {
    /// The stats of a finished turn, read from what the loop reported.
    pub fn from_outcome(outcome: &crate::agent::AgentOutcome) -> Self {
        Self {
            ts: now_iso(),
            steps: u32::try_from(outcome.steps).unwrap_or(u32::MAX),
            tool_calls: u32::try_from(outcome.tool_calls).unwrap_or(u32::MAX),
            retries: outcome.retries,
            input: outcome.usage.input,
            output: outcome.usage.output,
            cached: outcome.usage.cached,
            cache_created: outcome.usage.cache_created,
            reasoning: outcome.usage.reasoning,
            model_ms: u64::try_from(outcome.timing.model_ms).unwrap_or(u64::MAX),
            tools_ms: u64::try_from(outcome.timing.tools_ms).unwrap_or(u64::MAX),
        }
    }

    /// The row for a turn that died before reporting anything.
    ///
    /// Written rather than skipped, so the turn count stays honest; every
    /// measured field is zero because nothing was reported to measure.
    pub fn unreported() -> Self {
        Self {
            ts: now_iso(),
            ..Self::default()
        }
    }
}

/// A session's usage summed over all of its turns.
///
/// `#[serde(default)]` so a totals object written by an older build — one
/// without `cache_created` — still reads back instead of being rejected as a
/// missing field.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct SessionTotals {
    /// Turns recorded.
    pub turns: u64,
    /// Model steps, summed.
    pub steps: u64,
    /// Tool calls, summed.
    pub tool_calls: u64,
    /// Prompt tokens, summed (cached included, provider semantics).
    pub input: u64,
    /// Completion tokens, summed.
    pub output: u64,
    /// Prompt tokens served from the provider's cache, summed.
    pub cached: u64,
    /// Prompt tokens written into the provider's cache, summed.
    pub cache_created: u64,
    /// Reasoning tokens, summed.
    pub reasoning: u64,
    /// Model wait time, summed.
    pub model_ms: u64,
    /// Tool execution time, summed.
    pub tools_ms: u64,
}

impl SessionTotals {
    /// The share of prompt tokens the provider's cache served, `0..=1`.
    ///
    /// Zero prompt tokens means the question is meaningless, not a perfect hit:
    /// a session with no stats, or one that never got a reply, reads as 0.0.
    pub fn cache_hit_ratio(&self) -> f64 {
        if self.input == 0 {
            0.0
        } else {
            self.cached as f64 / self.input as f64
        }
    }
}

impl std::iter::Sum for SessionTotals {
    fn sum<I: Iterator<Item = SessionTotals>>(iter: I) -> SessionTotals {
        iter.fold(SessionTotals::default(), |mut total, next| {
            total.turns += next.turns;
            total.steps += next.steps;
            total.tool_calls += next.tool_calls;
            total.input += next.input;
            total.output += next.output;
            total.cached += next.cached;
            total.cache_created += next.cache_created;
            total.reasoning += next.reasoning;
            total.model_ms += next.model_ms;
            total.tools_ms += next.tools_ms;
            total
        })
    }
}

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

/// Durable session storage.
pub trait SessionStore: Send + Sync {
    /// Write a new session (header plus any messages).
    fn create(&self, session: &Session) -> Result<()>;
    /// Append messages to an existing session.
    fn append(&self, id: &str, messages: &[Message]) -> Result<()>;
    /// Record that a leading run of messages was replaced by a summary.
    ///
    /// Defaulted to a no-op so a store that does not model compaction still
    /// satisfies the trait; the JSONL store implements it.
    fn append_compaction(&self, _id: &str, _compaction: &crate::compaction::Compaction) -> Result<()> {
        Ok(())
    }
    /// Record what was injected into a request, so the log explains what the
    /// model was actually sent.
    fn append_injection(&self, _id: &str, _record: &InjectionRecord) -> Result<()> {
        Ok(())
    }
    /// Record the full set of packs this session has loaded. The latest set
    /// recorded is the current one; a store without pack tracking keeps none.
    fn append_packs(&self, _id: &str, _packs: &[String]) -> Result<()> {
        Ok(())
    }
    /// Record one turn's usage. Defaulted to a no-op, like the other optional
    /// records: a store that keeps no ledger is still a valid store.
    fn append_stats(&self, _id: &str, _stats: &TurnStats) -> Result<()> {
        Ok(())
    }
    /// Load a session, or `None` when it does not exist.
    fn load(&self, id: &str) -> Result<Option<Session>>;
    /// List sessions, most recently updated first.
    fn list(&self) -> Result<Vec<SessionSummary>>;
    /// Delete a session. Succeeds when it is already absent.
    fn delete(&self, id: &str) -> Result<()>;
    /// The most recently updated session.
    fn latest(&self) -> Result<Option<SessionSummary>> {
        Ok(self.list()?.into_iter().next())
    }
    /// Root directory holding the sessions.
    fn root(&self) -> &Path;
}

/// One injection that happened, for the session log.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InjectionRecord {
    /// Ids of the entries that were placed.
    pub activated: Vec<String>,
    /// Entries that were moved because the cache policy refused their position.
    #[serde(default)]
    pub relocated: Vec<String>,
    /// Estimated tokens the fragments cost.
    #[serde(default)]
    pub budget_used: usize,
    /// The text that was placed, so a reader can see it verbatim.
    #[serde(default)]
    pub texts: Vec<String>,
}

/// A compaction, as it appears in the session file.
///
/// `replaced` is the offset the replay starts from, so it is also the length the
/// fingerprint covers — one number, not two that could disagree.
///
/// `replaced_fingerprint` is the guard against a count that no longer describes
/// the file. JSONL is append-only, but a line can still vanish (something
/// outside this program rewrote the file, an append was truncated at the wrong
/// boundary): the count then points at a different run than the summary was
/// written for, and folding at it deletes history without a word. A fingerprint
/// turns that silent misalignment into a reported one.
///
/// `first_anchor` and `last_anchor` say *which* messages the run held, not just
/// how many, by carrying [`crate::compaction::anchor_of`] for its two ends. They
/// are the primary key the replay uses: a fingerprint can only report that the
/// count has drifted, while an anchor can still find the run wherever it now
/// sits. Both are `Option` with `#[serde(default)]`, so a record written before
/// anchors existed still parses, and its replay falls back to the count exactly
/// as it did before.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct CompactionLine {
    /// How many leading messages the summary stands in for.
    replaced: usize,
    /// The summary text.
    summary: String,
    /// Short fingerprint of the run it replaces, when recorded.
    ///
    /// Absent in files written before it existed, and the replay falls back to
    /// counting for those — the compatibility path, with a comment saying so.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    replaced_fingerprint: Option<String>,
    /// Anchor of the first message the summary replaced, when recorded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    first_anchor: Option<String>,
    /// Anchor of the last message the summary replaced, when recorded.
    ///
    /// Both ends are kept: a single anchor would find a run but could not tell
    /// how far it extends, and guessing that extent from `replaced` is the
    /// mistake the anchors exist to catch.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    last_anchor: Option<String>,
}

/// What the store knows about the run a compaction record is describing.
///
/// Read from the session file at write time, because the loader reduces the
/// file and the record has to describe what the loader will see, not what the
/// live agent happens to hold in memory. Every field is best effort: a file that
/// cannot be read yields an empty description, and the record is then written
/// the way an older build wrote it.
#[derive(Default)]
struct ReplacedRun {
    /// Fingerprint of the run, for the count-based check.
    fingerprint: Option<String>,
    /// Anchor of the run's first message.
    first: Option<String>,
    /// Anchor of the run's last message.
    last: Option<String>,
}

impl CompactionLine {
    fn of(compaction: &crate::compaction::Compaction, run: ReplacedRun) -> Self {
        Self {
            replaced: compaction.replaced,
            summary: compaction.summary.clone(),
            replaced_fingerprint: run.fingerprint,
            first_anchor: run.first,
            last_anchor: run.last,
        }
    }

    /// The record as the rest of the kernel knows it.
    fn compaction(&self) -> crate::compaction::Compaction {
        crate::compaction::Compaction {
            replaced: self.replaced,
            summary: self.summary.clone(),
        }
    }

    /// The message that stands in for the replaced run.
    ///
    /// Built by [`crate::compaction::Compaction::message`] rather than here, so
    /// a summary is marked the same way whichever layer produced it.
    fn summary_message(&self) -> Message {
        self.compaction().message()
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
fn fingerprint_of(messages: &[Message]) -> String {
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
fn replay_compactions(
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

/// Everything a session file holds, in the order it was read.
#[derive(Default)]
struct StoredLines {
    header: Option<SessionHeader>,
    messages: Vec<Message>,
    compactions: Vec<CompactionLine>,
    packs: Vec<String>,
    stats: Vec<TurnStats>,
}

/// JSONL-backed store: one file per session.
pub struct JsonlStore {
    root: PathBuf,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum JsonlLine {
    Header(SessionHeader),
    Message(Message),
    /// Fragments injected into a request.
    ///
    /// Recorded because the session log has to explain what the model was
    /// actually sent. Without this, a question like "why did it believe that"
    /// is unanswerable from the transcript: the injected text appears nowhere
    /// in it.
    Injection(InjectionRecord),
    /// A compaction: the leading run of messages is stood in for by a summary.
    ///
    /// Recorded as its own line rather than by rewriting the file, because the
    /// store is append-only so that an interrupted write cannot corrupt it.
    /// Rewriting history to compact it would give that up.
    Compaction(CompactionLine),
    /// The packs this session has loaded, as a full set.
    ///
    /// The latest such line is the current set. Recorded as a line rather than
    /// a header field because the store is append-only: changing a header would
    /// mean rewriting the file. A full set (not a delta) keeps a replay correct
    /// even when an earlier line was lost to an interrupted write.
    Packs { packs: Vec<String> },
    /// One turn's token and time usage.
    ///
    /// Its own line, so the accounting survives anything that folds messages
    /// away: a compaction rewrites the conversation, not the ledger.
    Stats(TurnStats),
}

#[derive(Debug, Serialize, Deserialize)]
struct SessionHeader {
    id: String,
    created_at: String,
    updated_at: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    cwd: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    model: Option<String>,
}

impl JsonlStore {
    /// Open a store rooted at `root`, creating the directory when needed.
    pub fn open(root: impl Into<PathBuf>) -> Result<Self> {
        let root = root.into();
        fs::create_dir_all(&root).with_context(|| format!("creating {}", root.display()))?;
        Ok(Self { root })
    }

    /// The default root: `$NGU_HOME/sessions`, else `~/.nguruvilu/sessions`.
    pub fn default_root() -> PathBuf {
        if let Ok(home) = std::env::var("NGU_HOME") {
            if !home.trim().is_empty() {
                return PathBuf::from(home).join("sessions");
            }
        }
        // The user's home, not the current directory. Sessions belong to the
        // user, and the directory a command happened to run in is not a place to
        // leave state — it may be a read-only checkout, or someone else's repo.
        // The desktop shell resolves the same path, so both frontends see one
        // history.
        crate::settings::home_dir().join(".nguruvilu").join("sessions")
    }

    fn path_for(&self, id: &str) -> PathBuf {
        self.root.join(format!("{id}.jsonl"))
    }

    /// Reject ids that could escape the store directory.
    fn validate_id(id: &str) -> Result<()> {
        if id.is_empty()
            || id.contains('/')
            || id.contains('\\')
            || id.contains("..")
            || id.contains(':')
        {
            return Err(anyhow!("invalid session id: {id:?}"));
        }
        Ok(())
    }

    fn read_header(path: &Path) -> Result<Option<(SessionHeader, usize)>> {
        let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let reader = BufReader::new(file);
        let mut header: Option<SessionHeader> = None;
        let mut count = 0usize;
        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            match serde_json::from_str::<JsonlLine>(&line) {
                Ok(JsonlLine::Header(h)) => header = Some(h),
                Ok(JsonlLine::Message(_)) => count += 1,
                // None of these carries a message.
                Ok(JsonlLine::Compaction(_))
                | Ok(JsonlLine::Injection(_))
                | Ok(JsonlLine::Packs { .. })
                // Neither does a usage row: the list carries the conversation,
                // not its ledger.
                | Ok(JsonlLine::Stats(_)) => {}
                Err(_) => continue,
            }
        }
        Ok(header.map(|h| (h, count)))
    }
}

impl SessionStore for JsonlStore {
    fn create(&self, session: &Session) -> Result<()> {
        Self::validate_id(&session.id)?;
        let path = self.path_for(&session.id);
        let mut file = File::create(&path).with_context(|| format!("creating {}", path.display()))?;

        let header = SessionHeader {
            id: session.id.clone(),
            created_at: session.created_at.clone(),
            updated_at: session.updated_at.clone(),
            cwd: session.cwd.clone(),
            model: session.model.clone(),
        };
        writeln!(file, "{}", serde_json::to_string(&JsonlLine::Header(header))?)?;

        for message in &session.messages {
            writeln!(file, "{}", serde_json::to_string(&JsonlLine::Message(message.clone()))?)?;
        }
        file.flush()?;
        file.sync_data()?;
        Ok(())
    }

    fn append(&self, id: &str, messages: &[Message]) -> Result<()> {
        Self::validate_id(id)?;
        if messages.is_empty() {
            return Ok(());
        }
        let path = self.path_for(id);
        let mut file = OpenOptions::new()
            .append(true)
            .create(true)
            .open(&path)
            .with_context(|| format!("opening {} for append", path.display()))?;

        for message in messages {
            writeln!(file, "{}", serde_json::to_string(&JsonlLine::Message(message.clone()))?)?;
        }
        file.flush()?;
        // Written, then on the platter. `flush` only leaves the user-space
        // buffer; without this a machine that loses power right after a
        // successful turn loses the turn — the exact window this store exists
        // to survive.
        file.sync_data()?;
        // The handle has to go before the header is rewritten: on Windows an
        // open handle blocks the rename the rewrite does, and a handle to the
        // replaced file would keep appending to the old inode.
        drop(file);

        self.touch(id)?;
        Ok(())
    }

    fn append_injection(&self, id: &str, record: &InjectionRecord) -> Result<()> {
        Self::validate_id(id)?;
        let path = self.path_for(id);
        let mut file = OpenOptions::new()
            .append(true)
            .create(true)
            .open(&path)
            .with_context(|| format!("opening {} for append", path.display()))?;
        writeln!(
            file,
            "{}",
            serde_json::to_string(&JsonlLine::Injection(record.clone()))?
        )?;
        file.flush()?;
        file.sync_data()?;
        drop(file);
        self.touch(id)?;
        Ok(())
    }

    fn append_compaction(&self, id: &str, compaction: &crate::compaction::Compaction) -> Result<()> {
        Self::validate_id(id)?;
        let path = self.path_for(id);
        // Fingerprinted and anchored at write time, from the file itself: the
        // loader will reduce the file, so the file is what the record has to
        // describe. Both come from one read and one replay, so what is written
        // is always what the loader will find when it reads the same file back.
        let line = CompactionLine::of(compaction, self.describe_replaced_run(id, compaction.replaced));
        let mut file = OpenOptions::new()
            .append(true)
            .create(true)
            .open(&path)
            .with_context(|| format!("opening {} for append", path.display()))?;
        writeln!(
            file,
            "{}",
            serde_json::to_string(&JsonlLine::Compaction(line))?
        )?;
        file.flush()?;
        file.sync_data()?;
        drop(file);
        self.touch(id)?;
        Ok(())
    }

    fn append_packs(&self, id: &str, packs: &[String]) -> Result<()> {
        Self::validate_id(id)?;
        let path = self.path_for(id);
        let mut file = fs::OpenOptions::new()
            .append(true)
            .create(true)
            .open(&path)
            .with_context(|| format!("opening {} for append", path.display()))?;
        writeln!(file, "{}", serde_json::to_string(&JsonlLine::Packs { packs: packs.to_vec() })?)?;
        file.flush()?;
        file.sync_data()?;
        drop(file);
        self.touch(id)?;
        Ok(())
    }

    fn append_stats(&self, id: &str, stats: &TurnStats) -> Result<()> {
        Self::validate_id(id)?;
        let path = self.path_for(id);
        let mut file = OpenOptions::new()
            .append(true)
            .create(true)
            .open(&path)
            .with_context(|| format!("opening {} for append", path.display()))?;
        writeln!(file, "{}", serde_json::to_string(&JsonlLine::Stats(stats.clone()))?)?;
        file.flush()?;
        // The ledger is only worth writing if it survives the power cut that
        // makes it interesting: flush alone stops at the OS buffer.
        file.sync_data()?;
        drop(file);
        self.touch(id)?;
        Ok(())
    }


    fn load(&self, id: &str) -> Result<Option<Session>> {
        Self::validate_id(id)?;
        let path = self.path_for(id);
        if !path.exists() {
            return Ok(None);
        }

        let stored = Self::read_lines(&path)?;
        let Some(header) = stored.header else {
            return Ok(None);
        };

        // The file holds the original messages *and* the records that replaced
        // them; reducing the two gives the history the model actually saw.
        // Without this a reloaded session would replay messages that were
        // compacted away, and diverge from the live conversation.
        let (messages, drift) = replay_compactions(stored.messages, &stored.compactions);
        for note in &drift {
            // Loud on purpose. A record that no longer lines up means the file
            // lost or gained a line, and applying it would delete messages the
            // summary never stood in for — the one failure nothing else in the
            // system would ever notice.
            eprintln!("[session {}] {note}", header.id);
        }

        Ok(Some(Session {
            id: header.id,
            created_at: header.created_at,
            updated_at: header.updated_at,
            cwd: header.cwd,
            model: header.model,
            packs: stored.packs,
            messages,
            stats: stored.stats,
            drift,
        }))
    }

    fn list(&self) -> Result<Vec<SessionSummary>> {
        let mut summaries = Vec::new();
        for entry in fs::read_dir(&self.root).with_context(|| format!("reading {}", self.root.display()))? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
                continue;
            }
            let Ok(Some((header, count))) = Self::read_header(&path) else {
                continue;
            };
            let session = Session {
                id: header.id.clone(),
                created_at: header.created_at.clone(),
                updated_at: header.updated_at.clone(),
                cwd: header.cwd.clone(),
                model: header.model.clone(),
                packs: Vec::new(),
                messages: Vec::new(),
                stats: Vec::new(),
                drift: Vec::new(),
            };
            // Title needs the first user message; read it lazily from the file.
            let title = read_first_user_message(&path);
            summaries.push(SessionSummary {
                id: header.id,
                created_at: header.created_at,
                updated_at: header.updated_at,
                message_count: count,
                title: title.or_else(|| session.title()),
            });
        }
        summaries.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
        Ok(summaries)
    }

    fn delete(&self, id: &str) -> Result<()> {
        Self::validate_id(id)?;
        let path = self.path_for(id);
        if path.exists() {
            fs::remove_file(&path).with_context(|| format!("removing {}", path.display()))?;
        }
        Ok(())
    }

    fn root(&self) -> &Path {
        &self.root
    }
}

impl JsonlStore {
    /// Read every line of a session file, in file order.
    ///
    /// Shared by `load` and by the fingerprint taken when a compaction is
    /// written, so both see the same history from the same code path.
    fn read_lines(path: &Path) -> Result<StoredLines> {
        let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let reader = BufReader::new(file);
        let mut stored = StoredLines::default();
        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            match serde_json::from_str::<JsonlLine>(&line) {
                Ok(JsonlLine::Header(h)) => stored.header = Some(h),
                Ok(JsonlLine::Message(m)) => stored.messages.push(m),
                Ok(JsonlLine::Compaction(c)) => stored.compactions.push(c),
                // An injection record is evidence, not history: it is not
                // replayed into the conversation.
                Ok(JsonlLine::Injection(_)) => {}
                // The last full set wins: each `Packs` line replaces the one
                // before it.
                Ok(JsonlLine::Packs { packs: set }) => stored.packs = set,
                // Usage is a ledger, not history: it is collected here and
                // never replayed into the conversation.
                Ok(JsonlLine::Stats(s)) => stored.stats.push(s),
                // A corrupt line is skipped rather than failing the load: an
                // interrupted append must not cost the whole conversation.
                Err(_) => continue,
            }
        }
        Ok(stored)
    }

    /// Describe the run a compaction is about to replace, by reading the file.
    ///
    /// Best effort: a file that cannot be read yields an empty description, and
    /// the record is then written exactly as an older build wrote it — no
    /// fingerprint, no anchors. Failing the append instead would be worse: the
    /// turn is already on disk, and a count-based record still carries the
    /// summary.
    ///
    /// The anchors and the fingerprint come from one read of the file, and the
    /// run is located the way the loader will locate it: earlier compactions are
    /// replayed first, so the record describes the history the loader reduces
    /// rather than the raw lines on disk.
    fn describe_replaced_run(&self, id: &str, replaced: usize) -> ReplacedRun {
        if replaced == 0 {
            return ReplacedRun::default();
        }
        let path = self.path_for(id);
        if !path.exists() {
            return ReplacedRun::default();
        }
        let Ok(stored) = Self::read_lines(&path) else {
            return ReplacedRun::default();
        };
        let (history, _drift) = replay_compactions(stored.messages, &stored.compactions);
        if history.len() < replaced {
            return ReplacedRun::default();
        }
        let run = &history[..replaced];
        let Some((first, last)) = crate::compaction::anchors_of(run) else {
            return ReplacedRun::default();
        };
        ReplacedRun {
            fingerprint: Some(fingerprint_of(run)),
            first: Some(first),
            last: Some(last),
        }
    }

    /// Refresh the header's `updated_at`.
    ///
    /// The header is the first line, so this rewrites the file — but never in
    /// place. The new contents go to a sibling temp file that is synced and
    /// renamed over the original: the file on disk is then either the old one
    /// or the complete new one. `File::create` on the real path truncates it
    /// first, so a crash — or a full disk — between the truncate and the last
    /// line used to cost the whole conversation.
    ///
    /// Every other line is copied through verbatim. Rewriting the header must
    /// not drop what the log holds: messages, compaction records, and usage
    /// rows all have to survive it.
    fn touch(&self, id: &str) -> Result<()> {
        let path = self.path_for(id);
        let Some((mut header, _)) = Self::read_header(&path)? else {
            return Ok(());
        };
        header.updated_at = now_iso();

        let content = fs::read_to_string(&path)?;
        let mut lines = content.lines();
        let _ = lines.next();
        let rest: String = lines.map(|l| format!("{l}\n")).collect();

        // Unique per call, and inside the store root: a rename is only atomic
        // within one volume, so the temp file has to be a sibling.
        let temp = self.root.join(format!(
            "{id}.{}.{}.tmp",
            std::process::id(),
            uuid::Uuid::new_v4().simple()
        ));

        let written = (|| -> Result<()> {
            let mut file = File::create(&temp).with_context(|| format!("creating {}", temp.display()))?;
            writeln!(file, "{}", serde_json::to_string(&JsonlLine::Header(header))?)?;
            file.write_all(rest.as_bytes())?;
            file.flush()?;
            file.sync_data()?;
            // Closed before the swap: an open handle is what makes a rename
            // fail on Windows.
            drop(file);
            Self::replace(&temp, &path)
        })();

        if written.is_err() {
            // A leftover temp file would be read back as a session by
            // anything that globs the directory, so clean it up. The original
            // is untouched at this point, which is the whole point of the swap.
            let _ = fs::remove_file(&temp);
        }
        written
    }

    /// Rename `from` over `to`, retrying briefly.
    ///
    /// A reader — the desktop listing sessions, a scanner — can hold the target
    /// open for a moment, and on Windows that fails the swap outright. A few
    /// short retries turn that race into a delay. The header timestamp is
    /// cosmetic; failing an otherwise complete append over it would not be.
    fn replace(from: &Path, to: &Path) -> Result<()> {
        let mut last: Option<std::io::Error> = None;
        for attempt in 0..8u64 {
            match fs::rename(from, to) {
                Ok(()) => return Ok(()),
                Err(error) => {
                    last = Some(error);
                    std::thread::sleep(std::time::Duration::from_millis(5 * (attempt + 1)));
                }
            }
        }
        Err(anyhow!(
            "replacing {} with {}: {}",
            to.display(),
            from.display(),
            last.map(|e| e.to_string()).unwrap_or_default()
        ))
    }
}

fn read_first_user_message(path: &Path) -> Option<String> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workspace_dir_nests_under_workspaces_and_session_id() {
        let root = Path::new("data");
        let dir = workspace_dir(root, "20260101-000000-abc123").expect("valid id");
        assert_eq!(dir, root.join("workspaces").join("20260101-000000-abc123"));
    }

    #[test]
    fn workspace_dir_rejects_ids_that_escape_the_root() {
        let root = Path::new("data");
        for bad in ["", "../x", "a/b", "a\\b", "c:x"] {
            assert!(workspace_dir(root, bad).is_err(), "{bad:?} must be rejected");
        }
    }

    #[test]
    fn ensure_workspace_creates_the_directory_and_is_idempotent() {
        let root = std::env::temp_dir().join(format!("ngu-ws-test-{}", uuid::Uuid::new_v4()));
        let id = new_id();
        let first = ensure_workspace(&root, &id).expect("create");
        assert!(first.is_dir());
        let second = ensure_workspace(&root, &id).expect("idempotent");
        assert_eq!(first, second);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn ensure_workspace_fails_loudly_when_the_path_is_a_file() {
        let root = std::env::temp_dir().join(format!("ngu-ws-test-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(root.join("workspaces")).expect("root");
        let id = "blocked";
        fs::write(root.join("workspaces").join(id), b"not a dir").expect("blocker");
        assert!(ensure_workspace(&root, id).is_err());
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn new_session_has_no_inherited_working_directory() {
        assert!(Session::new(None).cwd.is_none());
    }

    // ------------------------------------------------------------ usage stats

    /// A store rooted in a fresh temporary directory, plus that directory so
    /// the caller can clean it up.
    fn temp_store(tag: &str) -> (JsonlStore, PathBuf) {
        let root = std::env::temp_dir().join(format!(
            "ngu-stats-{}-{}",
            tag,
            uuid::Uuid::new_v4().simple()
        ));
        let store = JsonlStore::open(&root).expect("open store");
        (store, root)
    }

    fn turn(steps: u32, input: u64, cached: u64) -> TurnStats {
        TurnStats {
            ts: now_iso(),
            steps,
            tool_calls: 2,
            retries: 1,
            input,
            output: 7,
            cached,
            cache_created: 11,
            reasoning: 3,
            model_ms: 400,
            tools_ms: 50,
        }
    }

    #[test]
    fn a_usage_line_is_tagged_and_round_trips() {
        // The shape the file has to hold: an internally tagged enum can only
        // wrap a struct, never a bare sequence — `Packs(Vec<..>)` failed to
        // serialize for exactly this reason. This test is the guard.
        let original = turn(3, 100, 40);
        let line = serde_json::to_string(&JsonlLine::Stats(original.clone())).expect("serializes");
        assert!(line.contains("\"type\":\"stats\""), "{line}");
        let JsonlLine::Stats(back) = serde_json::from_str::<JsonlLine>(&line).expect("parses back")
        else {
            panic!("expected a stats line, got {line}");
        };
        assert_eq!(back, original);
    }

    #[test]
    fn a_usage_line_missing_fields_still_loads() {
        // An interrupted or hand-edited row must not fail the whole load:
        // every field has a default.
        let stats: TurnStats =
            serde_json::from_str(r#"{"steps":2,"input":10}"#).expect("defaults fill in");
        assert_eq!(stats.steps, 2);
        assert_eq!(stats.output, 0);
        assert_eq!(stats.ts, "");
    }

    #[test]
    fn append_stats_replays_through_load() {
        let (store, root) = temp_store("replay");
        let session = Session::new(Some("m".into()));
        store.create(&session).expect("create");
        store
            .append(&session.id, &[Message::user("hi"), Message::assistant("yo")])
            .expect("append");

        store.append_stats(&session.id, &turn(1, 100, 25)).expect("stats 1");
        store.append_stats(&session.id, &turn(4, 200, 100)).expect("stats 2");

        let loaded = store.load(&session.id).expect("load").expect("exists");
        assert_eq!(loaded.stats.len(), 2, "one row per turn, in order");
        assert_eq!(loaded.stats[0].steps, 1);
        assert_eq!(loaded.stats[1].input, 200);
        // Usage is a ledger beside the conversation, never inside it.
        assert_eq!(loaded.messages.len(), 2);

        let totals = totals_of(&loaded);
        assert_eq!(totals.turns, 2);
        assert_eq!(totals.steps, 5);
        assert_eq!(totals.tool_calls, 4);
        assert_eq!(totals.input, 300);
        assert_eq!(totals.output, 14);
        assert_eq!(totals.cached, 125);
        assert_eq!(totals.reasoning, 6);
        assert_eq!(totals.model_ms, 800);
        assert_eq!(totals.tools_ms, 100);
        assert!((totals.cache_hit_ratio() - 125.0 / 300.0).abs() < 1e-9);

        store.delete(&session.id).expect("delete");
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn usage_rows_survive_a_compaction_and_the_header_rewrite() {
        let (store, root) = temp_store("compaction");
        let session = Session::new(Some("m".into()));
        store.create(&session).expect("create");
        store
            .append(&session.id, &[Message::user("one"), Message::assistant("two")])
            .expect("append");
        store.append_stats(&session.id, &turn(1, 100, 10)).expect("stats");

        // A compaction is appended, then the *next* turn's writes rewrite the
        // header through `touch` — the path that used to rebuild the file from
        // its own lines and could drop anything it did not understand.
        store
            .append_compaction(
                &session.id,
                &crate::compaction::Compaction { replaced: 1, summary: "asked about one thing".into() },
            )
            .expect("compaction");
        store.append_stats(&session.id, &turn(2, 200, 150)).expect("stats after");
        store
            .append(&session.id, &[Message::assistant("three")])
            .expect("append after");

        let path = root.join(format!("{}.jsonl", session.id));
        let raw = fs::read_to_string(&path).expect("read raw");
        assert_eq!(
            raw.matches("\"type\":\"stats\"").count(),
            2,
            "both usage rows still in the file: {raw}"
        );

        let loaded = store.load(&session.id).expect("load").expect("exists");
        assert_eq!(loaded.stats.len(), 2, "the ledger is not folded away");
        assert_eq!(totals_of(&loaded).input, 300);
        // The conversation itself was still reduced by the compaction: three
        // messages were written (one, two, three) and the record stands in for
        // the first of them, so the summary plus the two survivors remain.
        assert_eq!(
            loaded.messages.len(),
            3,
            "summary + the messages the compaction kept: {:?}",
            loaded.messages.iter().map(|m| m.text()).collect::<Vec<_>>()
        );
        assert!(
            crate::compaction::Compaction::is_summary(&loaded.messages[0]),
            "the replaced run is stood in for by a summary"
        );
        assert_eq!(loaded.messages[1].text(), "two");
        assert_eq!(loaded.messages[2].text(), "three");
        // And the record still lined up: a clean file reports no drift.
        assert!(loaded.drift.is_empty(), "{:?}", loaded.drift);

        store.delete(&session.id).expect("delete");
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn the_header_rewrite_leaves_no_temp_file() {
        let (store, root) = temp_store("atomic");
        let session = Session::new(Some("m".into()));
        store.create(&session).expect("create");
        store.append(&session.id, &[Message::user("hi")]).expect("append");
        store.append_stats(&session.id, &turn(1, 10, 0)).expect("stats");

        let leftovers: Vec<String> = fs::read_dir(&root)
            .expect("read dir")
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().to_string())
            .filter(|name| name.ends_with(".tmp"))
            .collect();
        assert!(leftovers.is_empty(), "temp files left behind: {leftovers:?}");

        store.delete(&session.id).expect("delete");
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_session_with_no_usage_rows_loads_and_totals_to_zero() {
        // A file written before stats existed — and a session that recorded
        // nothing yet — must load clean and total to zeros, not error.
        let (store, root) = temp_store("legacy");
        let path = root.join("legacy-session.jsonl");
        let header = serde_json::to_string(&JsonlLine::Header(SessionHeader {
            id: "legacy-session".into(),
            created_at: now_iso(),
            updated_at: now_iso(),
            cwd: None,
            model: Some("m".into()),
        }))
        .expect("header");
        let message = serde_json::to_string(&JsonlLine::Message(Message::user("old prompt")))
            .expect("message");
        fs::write(&path, format!("{header}\n{message}\n")).expect("write legacy file");

        let loaded = store.load("legacy-session").expect("load").expect("exists");
        assert!(loaded.stats.is_empty());
        let totals = totals_of(&loaded);
        assert_eq!(totals, SessionTotals::default());
        assert_eq!(totals.cache_hit_ratio(), 0.0, "no input is not a perfect hit");

        // And the listing still sees it: a stats line is not a message.
        let summaries = store.list().expect("list");
        assert_eq!(summaries.len(), 1);
        assert_eq!(summaries[0].message_count, 1);

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn the_listing_does_not_count_usage_rows() {
        let (store, root) = temp_store("listing");
        let session = Session::new(Some("m".into()));
        store.create(&session).expect("create");
        store.append(&session.id, &[Message::user("hi")]).expect("append");
        for _ in 0..3 {
            store.append_stats(&session.id, &turn(1, 10, 5)).expect("stats");
        }

        let summaries = store.list().expect("list");
        assert_eq!(summaries.len(), 1);
        assert_eq!(summaries[0].message_count, 1, "the ledger is not dialogue");
        assert_eq!(summaries[0].title.as_deref(), Some("hi"));

        store.delete(&session.id).expect("delete");
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn totals_grand_sum_over_sessions() {
        let totals = vec![
            SessionTotals { turns: 1, steps: 2, input: 10, cached: 5, ..Default::default() },
            SessionTotals { turns: 3, steps: 4, input: 20, cached: 20, ..Default::default() },
        ]
        .into_iter()
        .sum::<SessionTotals>();
        assert_eq!(totals.turns, 4);
        assert_eq!(totals.steps, 6);
        assert_eq!(totals.input, 30);
        assert!((totals.cache_hit_ratio() - 25.0 / 30.0).abs() < 1e-9);
    }

    // ------------------------------------------- compaction alignment (kernel #11)

    /// A session file with a hand-written header, plus the given lines verbatim.
    ///
    /// Used where a test needs to control the file's exact shape — a lost line,
    /// a record in an older format — rather than go through the appends.
    fn write_session_file(root: &Path, id: &str, lines: &[String]) {
        let header = serde_json::to_string(&JsonlLine::Header(SessionHeader {
            id: id.into(),
            created_at: now_iso(),
            updated_at: now_iso(),
            cwd: None,
            model: Some("m".into()),
        }))
        .expect("header");
        let mut text = header;
        text.push('\n');
        for line in lines {
            text.push_str(line);
            text.push('\n');
        }
        fs::write(root.join(format!("{id}.jsonl")), text).expect("write session file");
    }

    fn message_line(text: &str) -> String {
        serde_json::to_string(&JsonlLine::Message(Message::user(text))).expect("message line")
    }

    #[test]
    fn a_fingerprinted_compaction_folds_the_run_it_recorded() {
        let (store, root) = temp_store("fp-match");
        let session = Session::new(Some("m".into()));
        store.create(&session).expect("create");
        store
            .append(
                &session.id,
                &[Message::user("one"), Message::assistant("two"), Message::user("three")],
            )
            .expect("append");
        store
            .append_compaction(
                &session.id,
                &crate::compaction::Compaction { replaced: 2, summary: "one and two".into() },
            )
            .expect("compaction");

        // The record on disk carries a fingerprint of the run it replaces.
        let raw = fs::read_to_string(root.join(format!("{}.jsonl", session.id))).expect("raw");
        assert!(raw.contains("\"replaced_fingerprint\":\""), "{raw}");

        let loaded = store.load(&session.id).expect("load").expect("exists");
        assert!(loaded.drift.is_empty(), "{:?}", loaded.drift);
        assert_eq!(loaded.messages.len(), 2, "summary + the message after the run");
        assert!(crate::compaction::Compaction::is_summary(&loaded.messages[0]));
        assert_eq!(loaded.messages[1].text(), "three");

        store.delete(&session.id).expect("delete");
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_substituted_message_inside_the_run_is_reported_instead_of_folded() {
        let (store, root) = temp_store("fp-substituted");
        let session = Session::new(Some("m".into()));
        store.create(&session).expect("create");
        store
            .append(
                &session.id,
                &[
                    Message::user("one"),
                    Message::assistant("two"),
                    Message::user("three"),
                    Message::user("four"),
                ],
            )
            .expect("append");
        store
            .append_compaction(
                &session.id,
                &crate::compaction::Compaction { replaced: 3, summary: "one two three".into() },
            )
            .expect("compaction");

        // Something outside this program edited a message in the middle of the
        // run. Both anchors still bracket it, so the span is found; what notices
        // the damage is the fingerprint of the run, which no longer matches what
        // the summary was written against.
        let path = root.join(format!("{}.jsonl", session.id));
        let raw = fs::read_to_string(&path).expect("raw");
        let edited = raw.replace("\"two\"", "\"two, edited\"");
        assert_ne!(edited, raw, "the message was in the file to edit");
        fs::write(&path, edited).expect("write the edited file");

        let loaded = store.load(&session.id).expect("load").expect("exists");
        // Folding here would stand the summary in for a run that no longer says
        // what it said. Nothing is folded, every message is kept, and the
        // mismatch is reported.
        assert_eq!(
            texts(&loaded),
            vec!["one", "two, edited", "three", "four"],
            "every stored message is kept"
        );
        assert!(!holds_a_summary(&loaded), "nothing was stood in for");
        assert_eq!(loaded.drift.len(), 1, "{:?}", loaded.drift);
        // Two paths can notice this — the span the anchors located, and the span
        // the count located — and they word the accusation differently ("does
        // not" against "do not"). What both say is that the record and the run
        // disagree, and that is the claim worth holding.
        assert!(
            loaded.drift[0].contains("match the record"),
            "{:?}",
            loaded.drift
        );

        store.delete(&session.id).expect("delete");
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_record_without_a_fingerprint_keeps_the_old_behaviour() {
        // A file written before fingerprints existed: the count is all there is,
        // and it is trusted. This is the compatibility path, not an oversight.
        let (store, root) = temp_store("fp-legacy");
        write_session_file(
            &root,
            "legacy-compaction",
            &[
                message_line("one"),
                message_line("two"),
                message_line("three"),
                serde_json::to_string(&serde_json::json!({
                    "type": "compaction",
                    "replaced": 2,
                    "summary": "one and two",
                }))
                .expect("record"),
            ],
        );

        let loaded = store.load("legacy-compaction").expect("load").expect("exists");
        assert!(loaded.drift.is_empty(), "{:?}", loaded.drift);
        assert_eq!(loaded.messages.len(), 2);
        assert!(crate::compaction::Compaction::is_summary(&loaded.messages[0]));
        assert_eq!(loaded.messages[1].text(), "three");

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_record_pointing_past_the_end_is_reported_not_applied() {
        let (store, root) = temp_store("fp-past-end");
        write_session_file(
            &root,
            "short-session",
            &[
                message_line("one"),
                serde_json::to_string(&serde_json::json!({
                    "type": "compaction",
                    "replaced": 5,
                    "summary": "five messages",
                }))
                .expect("record"),
            ],
        );

        let loaded = store.load("short-session").expect("load").expect("exists");
        assert_eq!(loaded.messages.len(), 1, "the file's messages are untouched");
        assert_eq!(loaded.drift.len(), 1, "{:?}", loaded.drift);
        assert!(loaded.drift[0].contains("only 1 are stored"), "{:?}", loaded.drift);

        let _ = fs::remove_dir_all(&root);
    }

    // -------------------------------------------------- compaction anchors

    /// The anchor of a plain user message, as the writer stores it.
    fn anchor(text: &str) -> String {
        crate::compaction::anchor_of(&Message::user(text))
    }

    /// A compaction record naming its run by anchor, as a newer file holds it.
    fn anchored_record(replaced: usize, summary: &str, first: &str, last: &str) -> String {
        serde_json::to_string(&serde_json::json!({
            "type": "compaction",
            "replaced": replaced,
            "summary": summary,
            "first_anchor": anchor(first),
            "last_anchor": anchor(last),
        }))
        .expect("record")
    }

    fn texts(session: &Session) -> Vec<String> {
        session
            .messages
            .iter()
            .map(|m| m.text().to_string())
            .collect()
    }

    fn holds_a_summary(session: &Session) -> bool {
        session.messages.iter().any(crate::compaction::Compaction::is_summary)
    }

    #[test]
    fn an_anchored_compaction_replays_the_same_messages_when_the_file_is_intact() {
        let (store, root) = temp_store("anchor-intact");
        let session = Session::new(Some("m".into()));
        store.create(&session).expect("create");
        store
            .append(
                &session.id,
                &[
                    Message::user("one"),
                    Message::assistant("two"),
                    Message::user("three"),
                    Message::user("four"),
                ],
            )
            .expect("append");
        store
            .append_compaction(
                &session.id,
                &crate::compaction::Compaction { replaced: 2, summary: "one and two".into() },
            )
            .expect("compaction");

        // The writer stores both ends of the run it replaced.
        let raw = fs::read_to_string(root.join(format!("{}.jsonl", session.id))).expect("raw");
        assert!(raw.contains("\"first_anchor\":\""), "{raw}");
        assert!(raw.contains("\"last_anchor\":\""), "{raw}");

        // And the reader accepts them: the same span, from the same record.
        let loaded = store.load(&session.id).expect("load").expect("exists");
        assert!(loaded.drift.is_empty(), "{:?}", loaded.drift);
        assert_eq!(texts(&loaded), vec!["[summary of earlier conversation]\none and two", "three", "four"]);
        assert!(crate::compaction::Compaction::is_summary(&loaded.messages[0]));

        store.delete(&session.id).expect("delete");
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_message_lost_from_an_anchored_run_is_reported_and_nothing_is_folded() {
        let (store, root) = temp_store("anchor-lost");
        let session = Session::new(Some("m".into()));
        store.create(&session).expect("create");
        store
            .append(
                &session.id,
                &[
                    Message::user("one"),
                    Message::user("two"),
                    Message::user("three"),
                    Message::user("four"),
                ],
            )
            .expect("append");
        store
            .append_compaction(
                &session.id,
                &crate::compaction::Compaction { replaced: 3, summary: "one two three".into() },
            )
            .expect("compaction");

        // One line inside the replaced run is gone. The count 3 now points at
        // ["one", "three", "four"], so folding at it would hand the model a
        // conversation the summary never described and swallow "four" without a
        // trace. The anchors are what refuse that.
        let path = root.join(format!("{}.jsonl", session.id));
        let raw = fs::read_to_string(&path).expect("raw");
        let kept: Vec<&str> = raw
            .lines()
            .filter(|line| {
                !matches!(
                    serde_json::from_str::<JsonlLine>(line),
                    Ok(JsonlLine::Message(ref m)) if m.text() == "two"
                )
            })
            .collect();
        assert_eq!(kept.len(), 5, "the header, three messages, and the record: {kept:?}");
        fs::write(&path, format!("{}\n", kept.join("\n"))).expect("rewrite without the line");

        let loaded = store.load(&session.id).expect("load").expect("exists");
        assert_eq!(
            texts(&loaded),
            vec!["one", "three", "four"],
            "every surviving message is still there"
        );
        assert!(!holds_a_summary(&loaded), "the summary was not applied");
        assert_eq!(loaded.drift.len(), 1, "{:?}", loaded.drift);
        assert!(
            loaded.drift[0].contains("not the 3 it recorded"),
            "the mismatch is reported: {:?}",
            loaded.drift
        );

        store.delete(&session.id).expect("delete");
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn an_anchor_finds_its_run_where_the_count_would_have_folded_the_wrong_one() {
        // The record was written when the file began with "one": the count 2
        // then described ["one", "two"]. A line now sits in front of that run —
        // a merge between machines, a hand edit — so the count points at
        // ["extra", "one"], which the summary never stood in for. The anchors
        // still name the run, and it is found where it actually is.
        let (store, root) = temp_store("anchor-shifted");
        write_session_file(
            &root,
            "shifted",
            &[
                message_line("extra"),
                message_line("one"),
                message_line("two"),
                message_line("three"),
                anchored_record(2, "one and two", "one", "two"),
            ],
        );

        let loaded = store.load("shifted").expect("load").expect("exists");
        assert!(loaded.drift.is_empty(), "{:?}", loaded.drift);
        let found = texts(&loaded);
        assert_eq!(found.len(), 3, "{found:?}");
        assert_eq!(found[0], "extra", "the message in front of the run survives");
        assert!(crate::compaction::Compaction::is_summary(&loaded.messages[1]));
        assert_eq!(found[2], "three");

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_record_without_anchors_still_replays_by_count() {
        // The shape of a file written before anchors existed: a count, a
        // summary, and nothing else. It is trusted exactly as it always was.
        let (store, root) = temp_store("anchor-legacy");
        write_session_file(
            &root,
            "pre-anchor",
            &[
                message_line("one"),
                message_line("two"),
                message_line("three"),
                serde_json::to_string(&serde_json::json!({
                    "type": "compaction",
                    "replaced": 2,
                    "summary": "one and two",
                }))
                .expect("record"),
            ],
        );

        let loaded = store.load("pre-anchor").expect("load").expect("exists");
        assert!(loaded.drift.is_empty(), "{:?}", loaded.drift);
        assert_eq!(loaded.messages.len(), 2);
        assert!(crate::compaction::Compaction::is_summary(&loaded.messages[0]));
        assert_eq!(loaded.messages[1].text(), "three");

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_record_whose_run_grew_is_reported_not_silently_applied() {
        // A message was inserted inside the replaced run. Both anchors are still
        // in the file, so a span can be found — but it is longer than the run
        // the summary was written for, and folding it would delete a message the
        // summary never described.
        let (store, root) = temp_store("anchor-grown");
        write_session_file(
            &root,
            "grown",
            &[
                message_line("one"),
                message_line("inserted later"),
                message_line("two"),
                anchored_record(2, "one and two", "one", "two"),
            ],
        );

        let loaded = store.load("grown").expect("load").expect("exists");
        assert_eq!(texts(&loaded), vec!["one", "inserted later", "two"]);
        assert!(!holds_a_summary(&loaded));
        assert_eq!(loaded.drift.len(), 1, "{:?}", loaded.drift);
        assert!(
            loaded.drift[0].contains("not the 2 it recorded"),
            "{:?}",
            loaded.drift
        );

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_record_holding_half_an_anchor_is_reported_not_guessed() {
        // One end without the other brackets nothing: the missing end would have
        // to come from the count, which is the assumption anchors remove.
        let (store, root) = temp_store("anchor-half");
        write_session_file(
            &root,
            "half-anchor",
            &[
                message_line("one"),
                message_line("two"),
                message_line("three"),
                serde_json::to_string(&serde_json::json!({
                    "type": "compaction",
                    "replaced": 2,
                    "summary": "one and two",
                    "first_anchor": anchor("one"),
                }))
                .expect("record"),
            ],
        );

        let loaded = store.load("half-anchor").expect("load").expect("exists");
        assert_eq!(loaded.messages.len(), 3, "nothing is folded");
        assert!(!holds_a_summary(&loaded));
        assert_eq!(loaded.drift.len(), 1, "{:?}", loaded.drift);
        assert!(
            loaded.drift[0].contains("only one end"),
            "{:?}",
            loaded.drift
        );

        let _ = fs::remove_dir_all(&root);
    }
}
