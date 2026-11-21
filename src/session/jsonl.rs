//! The shipped backend: one append-only JSONL file per session.

use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};

use crate::message::Message;

use super::model::{Session, SessionSummary, TurnStats};
use super::replay::{fingerprint_of, now_iso, read_first_user_message, replay_compactions};
use super::store::{InjectionRecord, SessionService, SessionStore};

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
pub(crate) struct CompactionLine {
    /// How many leading messages the summary stands in for.
    pub(crate) replaced: usize,
    /// The summary text.
    pub(crate) summary: String,
    /// Short fingerprint of the run it replaces, when recorded.
    ///
    /// Absent in files written before it existed, and the replay falls back to
    /// counting for those — the compatibility path, with a comment saying so.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) replaced_fingerprint: Option<String>,
    /// Anchor of the first message the summary replaced, when recorded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) first_anchor: Option<String>,
    /// Anchor of the last message the summary replaced, when recorded.
    ///
    /// Both ends are kept: a single anchor would find a run but could not tell
    /// how far it extends, and guessing that extent from `replaced` is the
    /// mistake the anchors exist to catch.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) last_anchor: Option<String>,
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
    pub(crate) fn summary_message(&self) -> Message {
        self.compaction().message()
    }
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
pub(crate) enum JsonlLine {
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
    pub(crate) fn validate_id(id: &str) -> Result<()> {
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

    /// The header and message count of a session file, for the list.
    ///
    /// This does **not** deserialize each line. The list shows two things — the
    /// header and how many messages there are — and decoding every message body
    /// as JSON just to learn that it was one cost 0.36 s of CPU per call on a
    /// machine with 354 sessions and 249 MB of history, and the list is rebuilt
    /// on every reconnect, every session switch and every new session.
    ///
    /// The tag is found by searching rather than by assuming a position:
    /// `serde(tag = "type")` writes it **last** for an internally tagged enum
    /// (`{"id":…,"type":"header"}`), so a prefix test would never match — which
    /// is exactly the bug this version fixes. `"type":"message"` is checked as a
    /// substring first because it is the common case, and a message body cannot
    /// contain that exact byte run as a bare key without being an object of its
    /// own (whose `type` would have to be `message` anyway).
    fn read_header(path: &Path) -> Result<Option<(SessionHeader, usize)>> {
        const MESSAGE_TAG: &[u8] = b"\"type\":\"message\"";
        const HEADER_TAG: &[u8] = b"\"type\":\"header\"";

        /// A byte-substring search, without pulling in a string dependency.
        fn contains(haystack: &[u8], needle: &[u8]) -> bool {
            haystack
                .windows(needle.len())
                .any(|window| window == needle)
        }

        let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let reader = BufReader::new(file);
        let mut header: Option<SessionHeader> = None;
        let mut count = 0usize;
        for line in reader.lines() {
            let line = line?;
            let text = line.trim();
            if !text.starts_with('{') {
                continue;
            }
            let bytes = text.as_bytes();
            if header.is_none() && contains(bytes, HEADER_TAG) {
                if let Ok(JsonlLine::Header(h)) = serde_json::from_str::<JsonlLine>(text) {
                    header = Some(h);
                }
            } else if contains(bytes, MESSAGE_TAG) {
                count += 1;
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

/// Adapts a [`SessionStore`] to the plugin-facing [`SessionService`].
///
/// A wrapper rather than an impl on `JsonlStore` directly: the two traits share
/// method names (`create`, `append`, `load`), and implementing both on one type
/// makes every existing `store.create(...)` call ambiguous. A newtype keeps the
/// old calls meaning what they meant and gives the service a name of its own.
pub struct StoreHandle<S: SessionStore> {
    inner: std::sync::Arc<S>,
}

impl<S: SessionStore + 'static> StoreHandle<S> {
    /// Wrap a store for the service table.
    pub fn new(inner: std::sync::Arc<S>) -> Self {
        Self { inner }
    }
}

impl<S: SessionStore + 'static> SessionService for StoreHandle<S> {
    fn create(&self, session: &Session) -> Result<()> {
        self.inner.create(session)
    }
    fn append(&self, id: &str, messages: &[Message]) -> Result<()> {
        self.inner.append(id, messages)
    }
    fn load(&self, id: &str) -> Result<Option<Session>> {
        self.inner.load(id)
    }
    fn ids(&self) -> Result<Vec<String>> {
        Ok(self.inner.list()?.into_iter().map(|s| s.id).collect())
    }
}
