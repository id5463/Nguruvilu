//! Content addressing, and the numbers that describe a download.
//!
//! A pack names content by sha256, so this is where bytes become that name: one
//! slice, one file, or a whole tree hashed in a fixed order so a pack built on
//! Windows verifies on Linux.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use sha2::{Digest, Sha256};

/// sha256 of a byte slice, lowercase hex.
pub fn sha256(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

/// sha256 of a file.
pub fn sha256_of_file(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    Ok(sha256(&bytes))
}

/// sha256 of a directory tree.
///
/// Deterministic across platforms: entries are sorted by their forward-slash
/// relative path, and each contributes `path`, a colon, its bytes, and a
/// newline. A pack built on Windows therefore verifies on Linux.
///
/// The algorithm is fixed by the format, so changing it invalidates every
/// published hash. That is the reason it is written out rather than delegated
/// to a hashing library's directory walk.
pub fn sha256_of_dir(dir: &Path) -> Result<String> {
    let mut files: Vec<(String, PathBuf)> = Vec::new();
    collect(dir, dir, &mut files)?;
    files.sort_by(|a, b| a.0.cmp(&b.0));

    let mut hasher = Sha256::new();
    for (relative, path) in &files {
        hasher.update(relative.as_bytes());
        hasher.update(b":");
        let bytes =
            std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        hasher.update(&bytes);
        hasher.update(b"\n");
    }
    Ok(format!("{:x}", hasher.finalize()))
}

pub(crate) fn collect(root: &Path, dir: &Path, out: &mut Vec<(String, PathBuf)>) -> Result<()> {
    let entries = std::fs::read_dir(dir)
        .with_context(|| format!("listing {}", dir.display()))?;
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            collect(root, &path, out)?;
        } else if file_type.is_file() {
            let relative = path
                .strip_prefix(root)
                .unwrap_or(&path)
                .to_string_lossy()
                .replace('\\', "/");
            out.push((relative, path));
        }
    }
    Ok(())
}

/// A byte count a user can read: bytes, kibibytes, or mebibytes, whichever
/// makes the sentence say something.
pub(crate) fn human_size(bytes: u64) -> String {
    const KIB: u64 = 1024;
    const MIB: u64 = 1024 * 1024;
    if bytes >= MIB {
        format!("{} MiB", bytes / MIB)
    } else if bytes >= KIB {
        format!("{} KiB", bytes / KIB)
    } else {
        format!("{bytes} B")
    }
}

/// The default cache directory: `$NGU_HOME/cache`, else `~/.nguruvilu/cache`.
pub fn default_cache_dir() -> PathBuf {
    crate::settings::data_dir().join("cache")
}
