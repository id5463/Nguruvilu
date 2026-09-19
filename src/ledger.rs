//! The install ledger: what has already been loaded, and from where.
//!
//! The ledger is what makes loading idempotent. An entry that is already
//! present with the same content hash is reused rather than fetched and
//! installed again, so applying the same pack twice is cheap and cannot
//! double-register anything.
//!
//! It records references and hashes, never content.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// Current ledger format.
pub const LEDGER_VERSION: u32 = 1;

/// One recorded installation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LedgerEntry {
    /// `plugin`, `mcp`, `skill`, or `pack`.
    pub kind: String,
    /// Entry id from the assembly.
    pub id: String,
    /// Where it came from.
    pub source: String,
    /// Content hash, when the source pins one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha1: Option<String>,
    /// Where it was installed on disk, for skills.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    /// Scope it was loaded under.
    #[serde(default)]
    pub scope: String,
    /// Pack that brought it in.
    #[serde(default)]
    pub pack: String,
    /// When it was recorded.
    pub installed_at: String,
}

/// The ledger file's contents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ledger {
    /// Format version.
    #[serde(default = "default_version")]
    pub version: u32,
    /// Recorded installations.
    #[serde(default)]
    pub entries: Vec<LedgerEntry>,
}

fn default_version() -> u32 {
    LEDGER_VERSION
}

impl Default for Ledger {
    fn default() -> Self {
        Self { version: LEDGER_VERSION, entries: Vec::new() }
    }
}

impl Ledger {
    /// An empty ledger.
    pub fn new() -> Self {
        Self::default()
    }

    /// Read the ledger, or start empty when the file is absent.
    pub fn load(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading ledger {}", path.display()))?;
        if text.trim().is_empty() {
            return Ok(Self::default());
        }
        serde_json::from_str(&text)
            .with_context(|| format!("parsing ledger {}", path.display()))
    }

    /// Write the ledger atomically: a temp file then a rename, so a crash
    /// mid-write cannot truncate the record.
    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating {}", parent.display()))?;
        }
        let text = serde_json::to_string_pretty(self)?;
        let temp = path.with_extension("json.tmp");
        std::fs::write(&temp, text)
            .with_context(|| format!("writing {}", temp.display()))?;
        std::fs::rename(&temp, path)
            .with_context(|| format!("renaming {} into place", temp.display()))?;
        Ok(())
    }

    /// Whether an identical installation is already recorded.
    ///
    /// A recorded hash must match when the caller supplies one; an entry
    /// recorded without a hash is treated as a match on kind and id alone.
    pub fn has(&self, kind: &str, id: &str, sha1: Option<&str>) -> bool {
        self.entries.iter().any(|entry| {
            entry.kind == kind
                && entry.id == id
                && match (sha1, entry.sha1.as_deref()) {
                    (Some(wanted), Some(recorded)) => wanted == recorded,
                    (Some(_), None) => true,
                    (None, _) => true,
                }
        })
    }

    /// Look up a recorded installation.
    pub fn find(&self, kind: &str, id: &str) -> Option<&LedgerEntry> {
        self.entries
            .iter()
            .find(|entry| entry.kind == kind && entry.id == id)
    }

    /// Record an installation, replacing any earlier record for the same
    /// kind and id.
    pub fn record(&mut self, entry: LedgerEntry) {
        self.entries
            .retain(|existing| !(existing.kind == entry.kind && existing.id == entry.id));
        self.entries.push(entry);
    }

    /// Remove a record, returning whether anything was removed.
    pub fn remove(&mut self, kind: &str, id: &str) -> bool {
        let before = self.entries.len();
        self.entries
            .retain(|entry| !(entry.kind == kind && entry.id == id));
        self.entries.len() != before
    }

    /// Everything recorded for one kind.
    pub fn of_kind(&self, kind: &str) -> Vec<&LedgerEntry> {
        self.entries.iter().filter(|e| e.kind == kind).collect()
    }

    /// Number of records.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether nothing is recorded.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// The default ledger path: `$NGU_HOME/installed.json`, else
/// `<cwd>/.nguruvilu/installed.json`.
pub fn default_ledger_path() -> PathBuf {
    if let Ok(home) = std::env::var("NGU_HOME") {
        return PathBuf::from(home).join("installed.json");
    }
    std::env::current_dir()
        .unwrap_or_else(|_| PathBuf::from("."))
        .join(".nguruvilu")
        .join("installed.json")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(kind: &str, id: &str, sha1: Option<&str>) -> LedgerEntry {
        LedgerEntry {
            kind: kind.into(),
            id: id.into(),
            source: "test".into(),
            sha1: sha1.map(str::to_string),
            path: None,
            scope: "session".into(),
            pack: "test-pack".into(),
            installed_at: "2026-01-01T00:00:00.000Z".into(),
        }
    }

    #[test]
    fn recording_then_finding_round_trips() {
        let mut ledger = Ledger::new();
        ledger.record(entry("skill", "pdf-tools", Some("abc")));

        assert!(ledger.has("skill", "pdf-tools", Some("abc")));
        assert_eq!(ledger.find("skill", "pdf-tools").unwrap().source, "test");
        assert_eq!(ledger.of_kind("skill").len(), 1);
        assert!(ledger.of_kind("plugin").is_empty());
    }

    #[test]
    fn a_different_hash_is_not_a_hit() {
        let mut ledger = Ledger::new();
        ledger.record(entry("skill", "pdf-tools", Some("abc")));

        // Same id, different content: this must be installed again.
        assert!(!ledger.has("skill", "pdf-tools", Some("def")));
        // Asking without a hash matches on identity alone.
        assert!(ledger.has("skill", "pdf-tools", None));
    }

    #[test]
    fn recording_twice_replaces_rather_than_duplicates() {
        let mut ledger = Ledger::new();
        ledger.record(entry("skill", "pdf-tools", Some("abc")));
        ledger.record(entry("skill", "pdf-tools", Some("def")));

        assert_eq!(ledger.len(), 1);
        assert!(ledger.has("skill", "pdf-tools", Some("def")));
        assert!(!ledger.has("skill", "pdf-tools", Some("abc")));
    }

    #[test]
    fn different_kinds_with_the_same_id_coexist() {
        let mut ledger = Ledger::new();
        ledger.record(entry("skill", "shared", None));
        ledger.record(entry("plugin", "shared", None));
        assert_eq!(ledger.len(), 2);

        assert!(ledger.remove("skill", "shared"));
        assert_eq!(ledger.len(), 1);
        assert!(!ledger.remove("skill", "shared"));
        assert!(ledger.has("plugin", "shared", None));
    }

    #[test]
    fn save_and_load_round_trip() {
        let dir = std::env::temp_dir().join(format!("ngu-ledger-{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("installed.json");

        // A missing file is an empty ledger, not an error.
        assert!(Ledger::load(&path).unwrap().is_empty());

        let mut ledger = Ledger::new();
        ledger.record(entry("mcp", "github", None));
        ledger.save(&path).unwrap();

        let reloaded = Ledger::load(&path).unwrap();
        assert_eq!(reloaded.len(), 1);
        assert!(reloaded.has("mcp", "github", None));

        // No temp file is left behind.
        assert!(!path.with_extension("json.tmp").exists());
    }

    #[test]
    fn an_empty_file_loads_as_an_empty_ledger() {
        let dir = std::env::temp_dir().join(format!("ngu-ledger-empty-{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("installed.json");
        std::fs::write(&path, "   \n").unwrap();
        assert!(Ledger::load(&path).unwrap().is_empty());
    }
}
