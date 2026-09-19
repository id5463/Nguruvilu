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
    /// Conversation history.
    #[serde(default)]
    pub messages: Vec<Message>,
}

impl Session {
    /// Start a new session with a generated id.
    pub fn new(model: Option<String>) -> Self {
        let now = now_iso();
        Self {
            id: new_id(),
            created_at: now.clone(),
            updated_at: now,
            cwd: std::env::current_dir().ok().map(|p| p.display().to_string()),
            model,
            messages: Vec::new(),
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

/// Durable session storage.
pub trait SessionStore: Send + Sync {
    /// Write a new session (header plus any messages).
    fn create(&self, session: &Session) -> Result<()>;
    /// Append messages to an existing session.
    fn append(&self, id: &str, messages: &[Message]) -> Result<()>;
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

/// JSONL-backed store: one file per session.
pub struct JsonlStore {
    root: PathBuf,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum JsonlLine {
    Header(SessionHeader),
    Message(Message),
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
            .open(&path)
            .with_context(|| format!("opening {} for append", path.display()))?;

        for message in messages {
            writeln!(file, "{}", serde_json::to_string(&JsonlLine::Message(message.clone()))?)?;
        }
        file.flush()?;

        self.touch(id)?;
        Ok(())
    }

    fn load(&self, id: &str) -> Result<Option<Session>> {
        Self::validate_id(id)?;
        let path = self.path_for(id);
        if !path.exists() {
            return Ok(None);
        }

        let file = File::open(&path).with_context(|| format!("opening {}", path.display()))?;
        let reader = BufReader::new(file);

        let mut header: Option<SessionHeader> = None;
        let mut messages = Vec::new();
        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            match serde_json::from_str::<JsonlLine>(&line) {
                Ok(JsonlLine::Header(h)) => header = Some(h),
                Ok(JsonlLine::Message(m)) => messages.push(m),
                // A corrupt line is skipped rather than failing the load: an
                // interrupted append must not cost the whole conversation.
                Err(_) => continue,
            }
        }

        let Some(header) = header else {
            return Ok(None);
        };

        Ok(Some(Session {
            id: header.id,
            created_at: header.created_at,
            updated_at: header.updated_at,
            cwd: header.cwd,
            model: header.model,
            messages,
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
                messages: Vec::new(),
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
    /// Refresh the header's `updated_at` in place.
    ///
    /// The header is the first line, so this rewrites the file. Sessions are
    /// small text files and this happens once per turn, not per message.
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

        let mut file = File::create(&path)?;
        writeln!(file, "{}", serde_json::to_string(&JsonlLine::Header(header))?)?;
        file.write_all(rest.as_bytes())?;
        file.flush()?;
        Ok(())
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
