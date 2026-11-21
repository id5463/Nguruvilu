//! Unpacking an archive a pack named, under a ceiling that keeps it bounded.
//!
//! An archive is a compressor's output, and its whole purpose is turning a
//! little input into a lot of output, so every unpack here runs against
//! [`UnpackLimits`] and refuses the shapes that would exhaust the machine: too
//! many entries, one entry past its size, the whole archive past its total, or
//! a name that escapes the directory it was given.

use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};

use super::hash::human_size;

/// Whether a file name points at an archive this kernel can unpack.
///
/// The check is on the name alone: a query string is not part of it, and an
/// unknown extension is refused by [`extract_archive`] rather than guessed at.
pub fn is_archive_name(name: &str) -> bool {
    let path = name.split('?').next().unwrap_or(name).to_ascii_lowercase();
    path.ends_with(".zip") || path.ends_with(".tar.gz") || path.ends_with(".tgz")
}

/// Unpack a `.zip`, `.tar.gz`, or `.tgz` into `into`, created as needed.
///
/// Every entry name is resolved against `into` first: one carrying `..` or an
/// absolute path aborts the extraction with the same refusal a pack archive
/// gets, so an archive cannot write a byte outside the directory it was given.
/// Executable permission bits are kept on unix; link entries are never
/// materialized, because writing through a link is the other way out.
///
/// The unpack also runs under [`UnpackLimits::DEFAULT`]. An archive is a
/// compressor's output, and a compressor's whole purpose is to turn a little
/// input into a lot of output — a few kilobytes of deflate expands to gigabytes,
/// which is why this is a shape worth refusing rather than a theoretical one.
/// The input here can be a remote artifact named by a pack, so it is untrusted
/// by construction.
pub fn extract_archive(archive: &Path, into: &Path) -> Result<()> {
    extract_archive_with(archive, into, UnpackLimits::DEFAULT)
}

/// Unpack an archive under explicit ceilings.
///
/// The ceilings are a parameter so the rule can be exercised with numbers small
/// enough to test quickly; production always passes
/// [`UnpackLimits::DEFAULT`].
pub fn extract_archive_with(archive: &Path, into: &Path, limits: UnpackLimits) -> Result<()> {
    std::fs::create_dir_all(into)
        .with_context(|| format!("creating {}", into.display()))?;
    let name = archive
        .file_name()
        .map(|n| n.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();
    if name.ends_with(".zip") {
        return extract_zip(archive, into, limits);
    }
    if name.ends_with(".tar.gz") || name.ends_with(".tgz") {
        return extract_tar(archive, into, limits);
    }
    Err(anyhow!(
        "{} is not an archive this kernel can unpack; expected .zip, .tar.gz or .tgz",
        archive.display()
    ))
}

/// The most entries one archive may carry.
const MAX_ARCHIVE_ENTRIES: usize = 4096;

/// The most bytes one entry may expand to.
const MAX_ENTRY_BYTES: u64 = 64 * 1024 * 1024;

/// The most bytes one archive may expand to in total.
const MAX_ARCHIVE_BYTES: u64 = 256 * 1024 * 1024;

/// The ceilings one unpack runs under: entries, one entry, the whole archive.
#[derive(Debug, Clone, Copy)]
pub struct UnpackLimits {
    /// Most entries the archive may list.
    pub entries: usize,
    /// Most bytes a single entry may expand to.
    pub entry_bytes: u64,
    /// Most bytes the archive may expand to in total.
    pub total_bytes: u64,
}

impl UnpackLimits {
    /// The ceilings every unpack uses.
    pub const DEFAULT: Self = Self {
        entries: MAX_ARCHIVE_ENTRIES,
        entry_bytes: MAX_ENTRY_BYTES,
        total_bytes: MAX_ARCHIVE_BYTES,
    };
}

/// What one unpack has spent so far, and the refusals that keep it bounded.
struct UnpackBudget {
    archive: PathBuf,
    limits: UnpackLimits,
    entries: usize,
    bytes: u64,
}

impl UnpackBudget {
    fn new(archive: &Path, limits: UnpackLimits) -> Self {
        Self {
            archive: archive.to_path_buf(),
            limits,
            entries: 0,
            bytes: 0,
        }
    }

    /// Count one entry before reading it, refusing an archive with too many.
    fn entry(&mut self, name: &str) -> Result<()> {
        self.entries += 1;
        if self.entries > self.limits.entries {
            return Err(anyhow!(
                "{} carries more than {} entries (stopped at {name}); refusing to unpack it",
                self.archive.display(),
                self.limits.entries
            ));
        }
        Ok(())
    }

    /// Read one entry's bytes, refusing an entry or an archive that is too big.
    ///
    /// The read stops one byte past the ceiling in force, which is enough to
    /// know the limit was broken without materialising the rest of a bomb.
    fn read_entry(&mut self, source: impl std::io::Read, name: &str) -> Result<Vec<u8>> {
        use std::io::Read;

        let headroom = self.limits.total_bytes.saturating_sub(self.bytes);
        let ceiling = self.limits.entry_bytes.min(headroom);
        let mut buffer = Vec::new();
        source.take(ceiling.saturating_add(1)).read_to_end(&mut buffer)?;

        let size = buffer.len() as u64;
        if size > self.limits.entry_bytes {
            return Err(anyhow!(
                "{} expands the entry {name} past the {} ceiling for a single entry; \
                 refusing to unpack it",
                self.archive.display(),
                human_size(self.limits.entry_bytes)
            ));
        }
        if self.bytes.saturating_add(size) > self.limits.total_bytes {
            return Err(anyhow!(
                "{} expands past the {} ceiling for one archive, at entry {name}; \
                 refusing to unpack the rest of it",
                self.archive.display(),
                human_size(self.limits.total_bytes)
            ));
        }
        self.bytes += size;
        Ok(buffer)
    }
}

/// Resolve one entry name under `into`, refusing anything that escapes it.
fn contained(raw: &str, archive: &Path) -> Result<PathBuf> {
    use std::path::Component;
    let escapes = Path::new(raw).components().any(|component| {
        matches!(
            component,
            Component::ParentDir | Component::RootDir | Component::Prefix(_)
        )
    });
    if escapes {
        return Err(anyhow!(
            "{} contains an entry that escapes the archive: {raw}",
            archive.display()
        ));
    }
    Ok(PathBuf::from(raw))
}

fn extract_zip(archive: &Path, into: &Path, limits: UnpackLimits) -> Result<()> {
    let file = std::fs::File::open(archive)
        .with_context(|| format!("opening {}", archive.display()))?;
    let mut zip = zip::ZipArchive::new(file)
        .with_context(|| format!("{} is not a zip archive", archive.display()))?;
    let mut budget = UnpackBudget::new(archive, limits);
    for index in 0..zip.len() {
        let mut entry = zip.by_index(index)?;
        // `enclosed_name` answers containment for zip the way `contained`
        // does for tar: no `..`, no absolute path, no drive prefix.
        let Some(relative) = entry.enclosed_name() else {
            return Err(anyhow!(
                "{} contains an entry that escapes the archive: {}",
                archive.display(),
                entry.name()
            ));
        };
        // Counted before it is read, so an archive listing millions of tiny
        // entries is refused on the list rather than after the work.
        budget.entry(&relative.to_string_lossy())?;
        let target = into.join(relative);
        if entry.is_dir() {
            std::fs::create_dir_all(&target)?;
            continue;
        }
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let buffer = budget.read_entry(&mut entry, &target.to_string_lossy())?;
        std::fs::write(&target, buffer)
            .with_context(|| format!("writing {}", target.display()))?;

        #[cfg(unix)]
        if let Some(mode) = entry.unix_mode() {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(&target, std::fs::Permissions::from_mode(safe_mode(mode)));
        }
    }
    Ok(())
}

fn extract_tar(archive: &Path, into: &Path, limits: UnpackLimits) -> Result<()> {
    let file = std::fs::File::open(archive)
        .with_context(|| format!("opening {}", archive.display()))?;
    let decoder = flate2::read::GzDecoder::new(file);
    let mut archive_reader = tar::Archive::new(decoder);
    let mut budget = UnpackBudget::new(archive, limits);
    for entry in archive_reader
        .entries()
        .with_context(|| format!("reading the entries of {}", archive.display()))?
    {
        let mut entry = entry.with_context(|| format!("reading an entry of {}", archive.display()))?;
        let raw = entry
            .path()
            .map_err(|error| anyhow!("{} has an unreadable entry name: {error}", archive.display()))?
            .to_string_lossy()
            .to_string();
        let relative = contained(&raw, archive)?;
        budget.entry(&raw)?;
        let target = into.join(&relative);
        if entry.header().entry_type().is_dir() {
            std::fs::create_dir_all(&target)?;
            continue;
        }
        if !entry.header().entry_type().is_file() {
            // A link, device, or fifo: never materialized, so nothing can be
            // written through one on a later entry.
            continue;
        }
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let buffer = budget.read_entry(&mut entry, &target.to_string_lossy())?;
        std::fs::write(&target, buffer)
            .with_context(|| format!("writing {}", target.display()))?;

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            if let Ok(mode) = entry.header().mode() {
                let _ = std::fs::set_permissions(&target, std::fs::Permissions::from_mode(safe_mode(mode)));
            }
        }
    }
    Ok(())
}

/// The permission bits an extracted file is allowed to end up with.
///
/// An archive carries whatever its author put in it, and that includes the
/// setuid and setgid bits. Installing a pack is not a reason to hand out a
/// setuid binary that some later process will run with the installer's
/// privileges, so the mode is normalised to the two shapes this program
/// actually ships — `0o644` for a file, `0o755` for something meant to be run —
/// and every other bit, the special bits included, is dropped rather than
/// trimmed.
#[cfg_attr(not(unix), allow(dead_code))]
pub(crate) fn safe_mode(mode: u32) -> u32 {
    if mode & 0o111 != 0 {
        0o755
    } else {
        0o644
    }
}

pub(crate) fn copy_tree(from: &Path, to: &Path) -> Result<()> {
    std::fs::create_dir_all(to)?;
    for entry in std::fs::read_dir(from)? {
        let entry = entry?;
        let target = to.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_tree(&entry.path(), &target)?;
        } else {
            std::fs::copy(entry.path(), &target)?;
        }
    }
    Ok(())
}
