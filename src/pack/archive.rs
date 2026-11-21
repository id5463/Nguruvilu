//! 归档的读写：一个目录怎么变成一个 `.ngupack`，一个归档怎么解开。
//!
//! 解包是唯一处理陌生人输入的地方，所以它带三道上限（条目数、单条目字节、
//! 总字节），且每个成员都在落盘前检查展开后的名字仍在目标目录里。

use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use serde_json::Value;

use super::format::{
    ARCHIVE_EXTS, FILES_DIR, MANIFEST_NAME, MAX_ARCHIVE_ENTRIES, MAX_ENTRY_BYTES,
    MAX_UNPACKED_BYTES,
};
use super::manifest::PackManifest;

pub fn inspect(dir: &Path) -> Result<PackContents> {
    let mut contents = PackContents::default();
    collect_files(dir, dir, &mut contents.files)?;
    contents.files.sort();
    // Either spelling counts: a directory mid-migration is still a pack.
    contents.has_manifest = manifest_path(dir).is_some();
    contents.has_assembly = dir.join(ASSEMBLY_NAME).is_file();
    Ok(contents)
}

fn collect_files(root: &Path, dir: &Path, out: &mut Vec<String>) -> Result<()> {
    let entries = std::fs::read_dir(dir)
        .with_context(|| format!("listing {}", dir.display()))?;
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        if entry.file_type()?.is_dir() {
            collect_files(root, &path, out)?;
        } else {
            let relative = path
                .strip_prefix(root)
                .unwrap_or(&path)
                .to_string_lossy()
                .replace('\\', "/");
            out.push(relative);
        }
    }
    Ok(())
}

/// The files a pack would contain, excluding any archive.
///
/// Whether a file name is a pack archive, in either the current or the older
/// spelling.
///
/// Used to keep a leftover build out of the pack it is being built from, so it
/// has to recognise both names: an archive from before the rename is just as
/// wrong to embed as a fresh one.
fn is_pack_archive_name(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    ARCHIVE_EXTS
        .iter()
        .any(|ext| lower.ends_with(&format!(".{ext}")))
}

/// The default output path sits beside the tree being packed, so a scan that
/// does not exclude it counts the archive as a member of itself. Every
/// `.ngupack` (and the older `.dshpack`) is excluded, not only the one being
/// written: an archive left over
/// from an earlier build is still an archive, and embedding one is always wrong
/// — it ships a stale pack inside a fresh one, and the build after that embeds
/// both.
pub fn packable_files(dir: &Path, out: &Path) -> Result<PackContents> {
    let _ = out;
    let mut contents = inspect(dir)?;
    contents.files.retain(|relative| {
        !is_pack_archive_name(relative) && !is_build_output(relative)
    });
    Ok(contents)
}

/// Whether a path is a build tool's working directory rather than pack content.
///
/// A pack that ships a plugin's source has a `src/target/` beside it after the
/// first build — hundreds of megabytes of object files that are not part of the
/// pack and would be re-fetched by nobody. Measured on `packs/documents`, whose
/// archive was 779 files for three of source.
///
/// The names are the conventions of the tools a pack realistically uses, and a
/// pack that keeps real content under one of them can say so by shipping it
/// outside — but the default has to be "do not ship a compiler's scratch space",
/// because nobody notices a 779-file archive going by.
fn is_build_output(relative: &str) -> bool {
    relative.split('/').any(|segment| {
        matches!(
            segment,
            "target" | "node_modules" | "__pycache__" | ".git" | "dist" | ".venv"
        )
    })
}

/// Write a zip of the given files.
fn write_archive(dir: &Path, files: &[String], out: &Path) -> Result<()> {
    if let Some(parent) = out.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating {}", parent.display()))?;
        }
    }
    let file = std::fs::File::create(out)
        .with_context(|| format!("creating {}", out.display()))?;
    let mut writer = zip::ZipWriter::new(file);

    for relative in files {
        let path = dir.join(relative);
        let metadata = std::fs::metadata(&path)
            .with_context(|| format!("reading metadata for {relative}"))?;

        #[allow(unused_mut)]
        let mut options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated)
            .large_file(metadata.len() > u32::MAX as u64);
        // A pack may ship a script; dropping the executable bit would break it.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            options = options.unix_permissions(metadata.permissions().mode());
        }

        writer
            .start_file(relative, options)
            .with_context(|| format!("adding {relative}"))?;
        let data = std::fs::read(&path).with_context(|| format!("reading {relative}"))?;
        writer
            .write_all(&data)
            .with_context(|| format!("writing {relative} into the archive"))?;
    }

    writer.finish().context("finalizing the archive")?;
    Ok(())
}

/// Build a manifest pack: the manifest and its content files, no payload.
pub fn pack(dir: &Path, out: &Path) -> Result<PackManifest> {
    let manifest = read_manifest(dir)?;
    let contents = packable_files(dir, out)?;

    // A pack must not smuggle payload in. Anything under `files/` is what an
    // offline build embeds, and a manifest pack that carries it is neither
    // shape — it would install stale bytes instead of fetching.
    let smuggled: Vec<&String> = contents
        .files
        .iter()
        .filter(|f| f.starts_with(&format!("{FILES_DIR}/")))
        .collect();
    if !smuggled.is_empty() {
        return Err(anyhow!(
            "{} holds payload under {FILES_DIR}/, which a manifest pack must not: \
             build with --offline to embed it deliberately",
            dir.display()
        ));
    }

    write_archive(dir, &contents.files, out)?;
    Ok(manifest)
}

/// Build an offline pack: fetch everything the manifest names and embed it.
pub async fn pack_offline(dir: &Path, out: &Path) -> Result<PackManifest> {
    let manifest = read_manifest(dir)?;
    let fetcher = Fetcher::new()?;
    let files_dir = dir.join(FILES_DIR);
    std::fs::create_dir_all(&files_dir)?;

    for reference in manifest.references()? {
        let Reference { kind, id, source, sha256 } = reference;
        let (source, expected) = (&source, sha256);
        if !source.is_remote() {
            continue;
        }
        let fetched = fetcher.fetch(source, expected).await?;
        let destination = files_dir.join(id);
        if destination.exists() {
            std::fs::remove_dir_all(&destination)?;
        }
        copy_tree(&fetched.path, &destination)?;
        eprintln!("[offline] embedded {kind} {id} ({})", fetched.sha256);
    }

    let contents = packable_files(dir, out)?;
    write_archive(dir, &contents.files, out)?;
    Ok(manifest)
}

fn copy_tree(from: &Path, to: &Path) -> Result<()> {
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

/// The most entries one archive may hold.
///
/// A pack is a manifest and a handful of content files; thousands of members is
/// a different animal, and the cheap thing to refuse before reading a byte.
pub const MAX_ARCHIVE_ENTRIES: usize = 4_096;

/// The most bytes one entry may expand to.
pub const MAX_ENTRY_BYTES: u64 = 64 * 1024 * 1024;

/// The most bytes one archive may expand to in total.
///
/// These three numbers bound `unpack`, which both `install` and `verify` run —
/// `verify` on an archive nobody has agreed to install yet. Without them a few
/// hundred kilobytes of deflated zeros was enough to fill a directory and the
/// memory of whoever looked at it: an archive is input from a stranger, and
/// reading a member whole into a buffer is how the kernel dies instead of
/// refusing.
pub const MAX_UNPACKED_BYTES: u64 = 256 * 1024 * 1024;

/// The permission bits a pack may keep: owner/group/other rwx, nothing else.
///
/// `setuid` and `setgid` are not in it. A pack's file lands owned by the person
/// unpacking it, and a bit that makes it run as somebody *else* is not something
/// an archive gets to decide — the same reasoning that drops world-writeable.
/// (This is `0o755` rather than `mode & 0o7777`: the high three bits go away.)
#[cfg(unix)]
const PACK_PERMISSION_BITS: u32 = 0o755;

/// The three limits an extraction runs under.
///
/// A struct rather than three arguments so a test can hold them all to something
/// it can afford to build: the accounting is what needs proving, not the size of
/// a real archive.
#[derive(Debug, Clone, Copy)]
struct Limits {
    entries: usize,
    entry_bytes: u64,
    total_bytes: u64,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            entries: MAX_ARCHIVE_ENTRIES,
            entry_bytes: MAX_ENTRY_BYTES,
            total_bytes: MAX_UNPACKED_BYTES,
        }
    }
}

/// Extract an archive into a directory, under the limits above.
///
/// Fails — rather than half-extracting silently — on an archive that expands
/// past any of them, on one that names the same path twice, and on a symlink
/// entry. Callers own the destination and are expected to discard it on failure
/// (`install` and `verify` both remove their scratch directory).
pub fn unpack(pack_path: &Path, dest: &Path) -> Result<PackManifest> {
    unpack_under(pack_path, dest, Limits::default())
}

fn unpack_under(pack_path: &Path, dest: &Path, limits: Limits) -> Result<PackManifest> {
    std::fs::create_dir_all(dest)
        .with_context(|| format!("creating {}", dest.display()))?;

    let file = std::fs::File::open(pack_path)
        .with_context(|| format!("opening {}", pack_path.display()))?;
    let mut archive = zip::ZipArchive::new(file)
        .with_context(|| format!("{} is not a zip archive", pack_path.display()))?;

    let entries = archive.len() as usize;
    if entries > limits.entries {
        return Err(anyhow!(
            "{} holds {entries} entries; a pack may hold at most {}",
            pack_path.display(),
            limits.entries
        ));
    }

    // Bytes that actually landed on disk, not the bytes the headers promise:
    // a member's declared size is the archive's claim, and a claim is free.
    let mut expanded = 0u64;
    // Paths already written, so a repeat name is caught instead of the second
    // entry quietly winning over the first.
    let mut seen: std::collections::HashSet<PathBuf> = std::collections::HashSet::new();

    for index in 0..archive.len() {
        let mut entry = archive.by_index(index)?;
        let name = entry.name().to_string();
        let Some(relative) = entry.enclosed_name() else {
            return Err(anyhow!(
                "{} contains an entry that escapes the archive: {name}",
                pack_path.display()
            ));
        };

        // A symlink is a second path the archive does not name, and this format
        // has no use for one: whatever it points at is written outside the
        // control of `enclosed_name`.
        if entry.is_symlink() {
            return Err(anyhow!(
                "{} carries {name} as a symbolic link; a pack holds plain files only",
                pack_path.display()
            ));
        }

        // The comparison is on what the name *means*, not on how it is spelled:
        // `a.md` and `./a.md` land on one file, and on Windows so do `A.md` and
        // `a.md`. An archive carrying both would install one and lose the other
        // with nothing said. A repeat of the *same* spelling never reaches here
        // — the reader collapses a repeated central-directory name into a single
        // entry — so the collision above is the one an archive can hand over.
        let key: PathBuf = relative
            .components()
            .filter(|component| !matches!(component, std::path::Component::CurDir))
            .collect();
        #[cfg(windows)]
        let key = PathBuf::from(key.to_string_lossy().to_lowercase());
        if !seen.insert(key) {
            return Err(anyhow!(
                "{} names {name} twice; which copy would install is not a choice \
                 this program should be making",
                pack_path.display()
            ));
        }

        let target = dest.join(&relative);
        if entry.is_dir() {
            std::fs::create_dir_all(&target)?;
            continue;
        }
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // The declared size is checked before anything is written, so an
        // obviously enormous member costs nothing; the bytes that arrive are
        // checked again as they stream out, so an understated header cannot
        // get around the limit either.
        let declared = entry.size();
        let room = limits.total_bytes.saturating_sub(expanded);
        if declared > limits.entry_bytes {
            return Err(expansion_error(
                pack_path,
                &name,
                "one entry's",
                limits.entry_bytes,
                declared,
                expanded,
            ));
        }
        if declared > room {
            return Err(expansion_error(
                pack_path,
                &name,
                "the whole archive's",
                limits.total_bytes,
                expanded + declared,
                expanded,
            ));
        }
        // Which limit is the one this member could cross: both are in play, and
        // the smaller of the two is what stops it.
        let per_entry = limits.entry_bytes <= room;
        let limit = if per_entry { limits.entry_bytes } else { room };
        let which = if per_entry { "one entry's" } else { "the whole archive's" };

        let mut written = std::fs::File::create(&target)
            .with_context(|| format!("creating {}", target.display()))?;
        // One byte past the limit: enough to tell "exactly at it" from "past
        // it", without ever holding the member in memory. The copy streams, so
        // a header that understated the size still stops at the limit instead of
        // filling the disk first.
        let copied = std::io::copy(&mut (&mut entry).take(limit + 1), &mut written)
            .with_context(|| format!("reading {name} out of {}", pack_path.display()))?;
        if copied > limit {
            // Windows will not delete a file that is still open.
            drop(written);
            let _ = std::fs::remove_file(&target);
            let reached = if per_entry { copied } else { expanded + copied };
            let limit = if per_entry { limits.entry_bytes } else { limits.total_bytes };
            return Err(expansion_error(pack_path, &name, which, limit, reached, expanded));
        }
        expanded += copied;

        #[cfg(unix)]
        if let Some(mode) = entry.unix_mode() {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(
                &target,
                std::fs::Permissions::from_mode(mode & PACK_PERMISSION_BITS),
            );
        }
    }

    read_manifest(dest)
}

/// The complaint about a member that expands too much: which limit it crosses,
/// how far the archive reaches, and how much had already landed — the three
/// things needed to tell a bomb from an honest pack that grew too big.
fn expansion_error(
    pack: &Path,
    name: &str,
    which: &str,
    limit: u64,
    reached: u64,
    already: u64,
) -> anyhow::Error {
    if already == 0 {
        anyhow!(
            "{} is refused: {name} crosses {which} limit of {} — this archive \
             would expand to {}",
            pack.display(),
            human_bytes(limit),
            human_bytes(reached),
        )
    } else {
        anyhow!(
            "{} is refused: {name} crosses {which} limit of {} — this archive \
             would expand to {}, with {} already unpacked",
            pack.display(),
            human_bytes(limit),
            human_bytes(reached),
            human_bytes(already),
        )
    }
}

/// A byte count in the largest unit that still reads as a whole number.
fn human_bytes(bytes: u64) -> String {
    const MIB: u64 = 1024 * 1024;
    if bytes >= MIB && bytes % MIB == 0 {
        format!("{} MiB", bytes / MIB)
    } else if bytes >= 1024 && bytes % 1024 == 0 {
        format!("{} KiB", bytes / 1024)
    } else {
        format!("{bytes} bytes")
    }
}
