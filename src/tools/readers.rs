//! The reader registry: how `read` learns to open formats the kernel itself
//! does not parse.
//!
//! # Two rules this exists for
//!
//! **读不了就装包** — a format the kernel cannot read is not the kernel's
//! problem to grow into; it is a pack's to provide. The kernel keeps the
//! contract (`read` either shows content or says precisely why it cannot),
//! and packs contribute *readers*: byte-sniffers that turn one container into
//! text. The `documents` pack reads PDF/Office this way; a future pack can
//! read anything else without this file changing.
//!
//! **缺环境就自己装** — a reader that fails because a tool is missing must
//! fail with the command that installs it (the model has `bash` and no gate
//! to ask). The reason string travels with `Unreadable` and is shown: that is
//! where `winget install Python.Python.3.13` and
//! `python -m pip install --user pymupdf markitdown` live. Self-provisioning
//! beats a hand-rolled parser guessing at font tables.
//!
//! # Discipline
//!
//! * **A reader is an effect**: registration returns a disposer the pack's
//!   fiber runs on unload. Registrations carry a nonce, so a reload's new
//!   registration survives the old fiber's disposal (replace-then-dispose).
//! * **Probes are magic-first**: bytes, not names.
//! * **`Unclaimed` keeps plain text working** with no readers installed at
//!   all, and images are deliberately not readers — an image is attached as
//!   message bytes, not interpreted into text.
//! * Closures are stored behind `Arc` so a snapshot can leave the lock before
//!   any conversion runs: a reader that spawns Python must never hold up the
//!   registry, and another registration arriving mid-read must not invalidate
//!   the snapshot.

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

/// What one reader produces for one file.
pub enum ReadOutcome {
    /// Readable text.
    Text(String),
    /// These bytes are not this reader's format.
    NotMine,
    /// The format is mine, but extraction failed — with a reason worth
    /// showing (an install-command card belongs here).
    Unreadable(String),
}

/// Claims probe: do these bytes belong to this reader's format?
type Claims = Arc<dyn Fn(&[u8], &Path) -> bool + Send + Sync>;
/// The conversion itself.
type Convert = Arc<dyn Fn(&[u8], &Path) -> ReadOutcome + Send + Sync>;

/// What a plugin puts into [`crate::plugin::Contributions`]; the kernel
/// stamps the owner (the plugin's name) on the way into the registry.
pub struct ReaderSpec {
    /// Stable identity: re-registering replaces instead of stacking.
    pub id: String,
    /// Magic/extension probe over the raw bytes and the path.
    pub claims: Claims,
    /// The conversion.
    pub convert: Convert,
}

/// One registered reader.
struct Reader {
    id: String,
    owner: String,
    nonce: u64,
    claims: Claims,
    convert: Convert,
}

/// How `read` sees the registry's answer.
pub enum ReadAnswer {
    /// A reader produced text.
    Converted(String),
    /// Nothing claimed the bytes: try the plain-text path.
    Unclaimed,
    /// Claimed and failed; each reason is shown — the install-command card
    /// lives here.
    Unreadable(Vec<String>),
}

fn registry() -> &'static Mutex<Vec<Reader>> {
    static REGISTRY: OnceLock<Mutex<Vec<Reader>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(Vec::new()))
}

fn next_nonce() -> u64 {
    static NONCE: AtomicU64 = AtomicU64::new(1);
    NONCE.fetch_add(1, Ordering::Relaxed)
}

/// Register one reader; the disposer removes **exactly this registration**
/// (id + nonce), so an old fiber's disposal cannot unseat the replacement its
/// own reload registered a moment earlier.
pub fn register(spec: ReaderSpec, owner: impl Into<String>) -> impl FnOnce() + Send {
    let id = spec.id.clone();
    let nonce = next_nonce();
    {
        let mut guard = registry().lock().expect("reader registry");
        // Replace, don't stack: two readers under one id would leave the
        // first one's verdict winning forever (conversion stops at the first
        // text produced).
        guard.retain(|r| r.id != id);
        guard.push(Reader {
            id: id.clone(),
            owner: owner.into(),
            nonce,
            claims: spec.claims,
            convert: spec.convert,
        });
    }
    move || {
        registry()
            .lock()
            .expect("reader registry")
            .retain(|r| !(r.id == id && r.nonce == nonce));
    }
}

/// Every installed reader (`id`, `owner`): named in `read`'s failure message
/// so "can't read it" is diagnosable at a glance.
pub fn installed() -> Vec<(String, String)> {
    registry()
        .lock()
        .expect("reader registry")
        .iter()
        .map(|r| (r.id.clone(), r.owner.clone()))
        .collect()
}

/// Ask the registered readers to convert this file.
///
/// The claiming snapshot is taken under the lock (Arc clones) and every
/// conversion runs outside it: readers spawn processes, and registrations may
/// arrive while a read is in flight.
pub fn convert(bytes: &[u8], path: &Path) -> ReadAnswer {
    let snapshot: Vec<Convert> = {
        let guard = registry().lock().expect("reader registry");
        guard
            .iter()
            .filter(|r| (r.claims)(bytes, path))
            .map(|r| Arc::clone(&r.convert))
            .collect()
    };

    let mut claimed = false;
    let mut reasons: Vec<String> = Vec::new();
    for convert in snapshot {
        match convert(bytes, path) {
            ReadOutcome::Text(text) if !text.trim().is_empty() => {
                return ReadAnswer::Converted(text);
            }
            ReadOutcome::Text(_) | ReadOutcome::NotMine => {}
            ReadOutcome::Unreadable(reason) => {
                claimed = true;
                reasons.push(reason);
            }
        }
    }
    if claimed {
        ReadAnswer::Unreadable(reasons)
    } else {
        ReadAnswer::Unclaimed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn spec(
        id: &str,
        claims: fn(&[u8], &Path) -> bool,
        convert: fn(&[u8], &Path) -> ReadOutcome,
    ) -> ReaderSpec {
        ReaderSpec {
            id: id.into(),
            claims: Arc::new(claims),
            convert: Arc::new(convert),
        }
    }

    fn path(name: &str) -> PathBuf {
        PathBuf::from(name)
    }

    #[test]
    fn a_reader_claims_by_bytes_and_its_disposer_removes_it() {
        let dispose = register(
            spec(
                "demo",
                |bytes, _| bytes.starts_with(b"MAGIC"),
                |bytes, _| ReadOutcome::Text(String::from_utf8_lossy(&bytes[5..]).into_owned()),
            ),
            "test-pack",
        );

        assert!(
            matches!(
                convert(b"MAGIC hello", &path("x.anything")),
                ReadAnswer::Converted(ref text) if text == " hello"
            ),
            "claimed by magic, not by name"
        );
        assert!(
            matches!(convert(b"other bytes", &path("x.anything")), ReadAnswer::Unclaimed),
            "an unclaimed file passes through to plain text"
        );
        // Subset assertions, not equality: the registry is process-wide and
        // other tests register concurrently in the same run.
        assert!(
            installed().contains(&("demo".into(), "test-pack".into())),
            "registered with its owner"
        );

        dispose();
        assert!(matches!(
            convert(b"MAGIC hello", &path("x.anything")),
            ReadAnswer::Unclaimed
        ));
        assert!(
            !installed().iter().any(|(id, _)| id == "demo"),
            "the disposer removed exactly this registration"
        );
    }

    #[test]
    fn a_reader_that_cannot_read_falls_through_with_its_reason() {
        let first = register(
            spec(
                "empty",
                |b, _| b.starts_with(b"GO!"),
                |_, _| ReadOutcome::Unreadable("python missing: winget install Python".into()),
            ),
            "test-pack",
        );
        let second = register(
            spec(
                "backup",
                |b, _| b.starts_with(b"GO!"),
                |_, _| ReadOutcome::Text("from backup".into()),
            ),
            "test-pack",
        );

        assert!(
            matches!(
                convert(b"GO!", &path("x")),
                ReadAnswer::Converted(ref text) if text == "from backup"
            ),
            "one engine failing must not lock the format out"
        );
        first();
        second();

        // All claiming readers failing keeps the reasons, not the text.
        let only = register(
            spec(
                "only",
                |b, _| b.starts_with(b"GO!"),
                |_, _| ReadOutcome::Unreadable("install X".into()),
            ),
            "test-pack",
        );
        match convert(b"GO!", &path("x")) {
            ReadAnswer::Unreadable(reasons) => {
                assert_eq!(reasons, vec!["install X".to_string()]);
            }
            _ => panic!("reasons must survive to the caller"),
        }
        only();
    }

    #[test]
    fn re_registering_an_id_replaces_and_nonce_disposal_is_surgical() {
        let old = register(
            spec(
                "same",
                |b, _| b.starts_with(b"XYZ"),
                |_, _| ReadOutcome::Text("old".into()),
            ),
            "p",
        );
        let new = register(
            spec(
                "same",
                |b, _| b.starts_with(b"XYZ"),
                |_, _| ReadOutcome::Text("new".into()),
            ),
            "p",
        );
        assert!(matches!(
            convert(b"XYZ", &path("x")),
            ReadAnswer::Converted(ref t) if t == "new"
        ));
        // Counted by id, not by list length: the registry is process-wide and
        // the other tests are registering while this one runs.
        assert_eq!(
            installed().iter().filter(|(id, _)| id == "same").count(),
            1,
            "one id, one reader"
        );

        // The OLD fiber unloads after the new one registered: it must remove
        // itself only, not the replacement.
        old();
        assert!(matches!(
            convert(b"XYZ", &path("x")),
            ReadAnswer::Converted(ref t) if t == "new"
        ));
        new();
        assert!(matches!(convert(b"XYZ", &path("x")), ReadAnswer::Unclaimed));
    }
}
