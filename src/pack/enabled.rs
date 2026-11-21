//! 已安装包的状态与清单读写：开没开、清单在哪、怎么读怎么写。
//!
//! 启用与卸载是两件事：卸载只放一个标记文件、文件全留着；删除才移除目录。
//! 所以磁盘上「卸载了但还在」是正常状态，不是错误。

use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};

use super::format::{DISABLED_MARKER, LEGACY_MANIFEST_NAMES, MANIFEST_NAME};
use super::manifest::PackManifest;

pub fn is_enabled(dir: &Path) -> bool {
    !dir.join(DISABLED_MARKER).exists()
}

/// Enable or disable an installed pack, keeping its files either way.
///
/// Returns the state before the change, so a caller can report a no-op as a
/// no-op instead of as a success it did not have to earn.
pub fn set_enabled(dir: &Path, enabled: bool) -> Result<bool> {
    let marker = dir.join(DISABLED_MARKER);
    let was = is_enabled(dir);
    if enabled {
        if marker.exists() {
            std::fs::remove_file(&marker)
                .with_context(|| format!("removing {}", marker.display()))?;
        }
    } else if !marker.exists() {
        std::fs::write(&marker, "unloaded by the user; files kept.\n")
            .with_context(|| format!("writing {}", marker.display()))?;
    }
    Ok(was)
}

/// Read `ngu.index.json` from a pack directory.
///
/// Reads, and does not judge. A manifest is a declaration by the person who
/// wrote it, and this kernel is a tool rather than a gatekeeper: a pack whose
/// licence is blank or whose version range does not match still installs, and
/// whatever is actually wrong shows up when the thing it describes is used.
/// Refusing here substitutes this program's judgement for the user's, and the
/// user has the authority.
/// The manifest file this directory actually has, if either name is present.
///
/// `ds.index.json` wins when a directory carries both: it is the name a rebuild
/// writes, and a directory holding both spellings is one in the middle of a
/// migration.
pub fn manifest_path(dir: &Path) -> Option<PathBuf> {
    let modern = dir.join(MANIFEST_NAME);
    if modern.is_file() {
        return Some(modern);
    }
    for name in LEGACY_MANIFEST_NAMES {
        let legacy = dir.join(name);
        if legacy.is_file() {
            return Some(legacy);
        }
    }
    None
}

/// Rewrite a legacy manifest name to the current one, in place.
///
/// Returns whether anything changed. Idempotent, and a no-op for a directory
/// that already carries the current name — so a caller may run it over every
/// installed pack on each start without paying for it twice.
///
/// Callers should actually call it. It existed for a while and nothing did, so
/// every pack on every machine kept the old spelling and the rename was a
/// constant in a test rather than a thing that happened.
pub fn migrate_manifest_name(dir: &Path) -> Result<bool> {
    let modern = dir.join(MANIFEST_NAME);
    if modern.exists() {
        return Ok(false);
    }
    for name in LEGACY_MANIFEST_NAMES {
        let legacy = dir.join(name);
        if !legacy.is_file() {
            continue;
        }
        std::fs::rename(&legacy, &modern).with_context(|| {
            format!("renaming {} to {}", legacy.display(), modern.display())
        })?;
        return Ok(true);
    }
    Ok(false)
}
pub fn read_manifest(dir: &Path) -> Result<PackManifest> {
    let path = manifest_path(dir).ok_or_else(|| {
        anyhow!(
            "{} has no manifest: a pack needs {} (or the older {})",
            dir.display(),
            MANIFEST_NAME,
            LEGACY_MANIFEST_NAMES.join(" / ")
        )
    })?;
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("reading {} (a pack needs one)", path.display()))?;
    let manifest: PackManifest = serde_json::from_str(&text)
        .with_context(|| format!("{} is not readable as a manifest", path.display()))?;
    // A manifest that parses but names nothing is not a pack. With an empty name
    // and version the installed set sorts under the empty string and every label
    // reads "-", which is how a leftover directory becomes a thing the startup
    // log claims to have loaded. Refused here, the directory is reported and
    // left out instead.
    if manifest.name.trim().is_empty() {
        return Err(anyhow!(
            "{} names no pack: `name` is empty",
            path.display()
        ));
    }
    if manifest.version_id.trim().is_empty() {
        return Err(anyhow!(
            "{} names no version: `version` is empty",
            path.display()
        ));
    }
    Ok(manifest)
}

/// Write a manifest into a pack directory.
pub fn write_manifest(dir: &Path, manifest: &PackManifest) -> Result<()> {
    std::fs::create_dir_all(dir)
        .with_context(|| format!("creating {}", dir.display()))?;
    let path = dir.join(MANIFEST_NAME);
    let text = serde_json::to_string_pretty(manifest)?;
    std::fs::write(&path, format!("{text}\n"))
        .with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}
