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

/// A reader whose converter is a plugin process rather than a closure.
///
/// The closure form cannot cross a process boundary, which is why `documents`
/// used to be kernel code: it registered two `Arc<dyn Fn>` and `read` called them
/// inline. A plugin instead answers two questions over the protocol — "is this
/// yours?" and "convert it" — and the kernel keeps the same registry, with a
/// process where the closure was.
pub struct ProcessReader {
    /// Stable identity (`pdf`, `office`, …).
    pub id: String,
    /// The plugin to ask.
    pub process: std::sync::Arc<crate::protocol::Process>,
    /// What it claims, as magic prefixes and file extensions.
    ///
    /// Declared as data rather than as a probe function: the probe has to run on
    /// every `read` of a non-text file, and a process round trip to answer "is
    /// this yours" would cost 0.25 ms per byte of every binary file read. The
    /// kernel checks the cheap declaration; the plugin only runs when it matches.
    pub magic: Vec<String>,
    /// Extensions it claims, lowercase and without the dot.
    pub extensions: Vec<String>,
}

/// Register a plugin-backed reader, returning a disposer.
pub fn register_process(spec: ProcessReader) -> impl FnOnce() + Send {
    let id = spec.id.clone();
    let nonce = next_nonce();
    let process = std::sync::Arc::clone(&spec.process);
    let magic = spec.magic.clone();
    let extensions = spec.extensions.clone();
    let claimed_id = id.clone();
    {
        let mut guard = registry_lock();
        guard.retain(|r| r.id != id);
        guard.push(Reader {
            id: id.clone(),
            owner: format!("plugin:{}", spec.process.id()),
            nonce,
            // The probe is the declaration, checked locally.
            claims: Arc::new(move |bytes, path| {
                if magic.iter().any(|prefix| bytes.starts_with(prefix.as_bytes())) {
                    return true;
                }
                match path.extension().and_then(|e| e.to_str()) {
                    Some(ext) => {
                        let lower = ext.to_ascii_lowercase();
                        extensions.iter().any(|known| *known == lower)
                    }
                    None => false,
                }
            }),
            convert: Arc::new(move |bytes, path| {
                process_convert(&process, &claimed_id, bytes, path)
            }),
        });
    }
    move || {
        registry_lock().retain(|r| !(r.id == id && r.nonce == nonce));
    }
}

/// Ask a plugin to convert a file, on the thread `read` already runs readers on.
///
/// Blocking rather than async because `readers::convert` is called from
/// `spawn_blocking`: the kernel already treats a conversion as slow work and has
/// parked a thread for it, so awaiting inside would need a second runtime for no
/// gain.
fn process_convert(
    process: &std::sync::Arc<crate::protocol::Process>,
    id: &str,
    bytes: &[u8],
    path: &Path,
) -> ReadOutcome {
    match process.convert_blocking(id, bytes, path) {
        Ok(text) if !text.trim().is_empty() => ReadOutcome::Text(text),
        Ok(_) => ReadOutcome::NotMine,
        Err(error) => ReadOutcome::Unreadable(format!("{error:#}")),
    }
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

/// Take the registry, recovering the data if a previous holder panicked.
///
/// A reader runs inside a pack's factory, so any panic on that thread poisons
/// the lock. `.expect()` there would turn one bad reader into a process where
/// *every* later read panics; the registry's `Vec<Reader>` is still perfectly
/// usable, so the poison is stepped over.
fn registry_lock() -> std::sync::MutexGuard<'static, Vec<Reader>> {
    registry().lock().unwrap_or_else(|error| error.into_inner())
}

/// Register one reader; the disposer removes **exactly this registration**
/// (id + nonce), so an old fiber's disposal cannot unseat the replacement its
/// own reload registered a moment earlier.
pub fn register(spec: ReaderSpec, owner: impl Into<String>) -> impl FnOnce() + Send {
    let id = spec.id.clone();
    let nonce = next_nonce();
    {
        let mut guard = registry_lock();
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
        registry_lock().retain(|r| !(r.id == id && r.nonce == nonce));
    }
}

/// Every installed reader (`id`, `owner`): named in `read`'s failure message
/// so "can't read it" is diagnosable at a glance.
pub fn installed() -> Vec<(String, String)> {
    registry_lock()
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
        let guard = registry_lock();
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

