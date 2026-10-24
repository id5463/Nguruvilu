//! The official packs, shipped inside the binary and placed on first run.
//!
//! A pack is data — a manifest plus a few small documents — so shipping five of
//! them costs a few dozen kilobytes of executable and buys a program that works
//! the moment it runs: 中文、界面、搜索、子代理、computer use, without a
//! download step, an app store, or an account.
//!
//! # Three rules this keeps to
//!
//! * **The user's decisions outlast ours.** A pack that was unloaded or deleted
//!   is never resurrected: the record remembers it, and that memory is the only
//!   thing that stops us from placing it again.
//! * **Edited packs are never overwritten.** At placement time every file's
//!   sha256 is recorded; a later build replaces the pack only when what is on
//!   disk still hashes to exactly what *we* put there. Files the machine writes
//!   by itself — logs, caches, editor droppings — are not edits and do not count
//!   (see [`is_machine_noise`]). A hand-edited pack is left alone, permanently,
//!   and the report says so.
//! * **Placement is idempotent.** Called on every start, costs a directory
//!   listing and some hashing, and does nothing at all on the steady state.

use std::collections::BTreeMap;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// The packs this build ships: name, version, and the archive itself.
///
/// Embedded rather than sitting beside the executable: the single-file
/// requirement applies to what the user downloads, and a program that needs a
/// payload folder to be complete is not a single-file program.
pub const SHIPPED: &[(&str, &str, &[u8])] = &[
    (
        "starter",
        "1.0.0",
        include_bytes!("../packs/starter/starter-1.0.0.dshpack"),
    ),
    (
        "judge",
        "1.0.0",
        include_bytes!("../packs/judge/judge-1.0.0.dshpack"),
    ),
];

/// The five packs this project once shipped separately, now merged into
/// `starter`: chinese, ui, search, subagent, computer-use.
///
/// A machine that still has the old five would load their souls and plugins
/// beside starter's — two personas, two `builtin:search`, two interfaces.
/// Retiring them follows the same rule as everything here: only bytes *we*
/// placed and nobody edited go. Recorded, still hashing to the record, and
/// `shipped` (ours by construction — an adopted pack was the user's first).
/// Anything else is kept and named, because "kept" is the safe answer when
/// touching it could destroy work.
const MERGED_INTO_STARTER: &[&str] = &["chinese", "ui", "search", "subagent", "computer-use"];

/// What this program has placed, so a later run can tell its own work apart
/// from the user's.
#[derive(Debug, Default, Serialize, Deserialize)]
struct Record {
    #[serde(default)]
    packs: BTreeMap<String, Seeded>,
}

#[derive(Debug, Serialize, Deserialize)]
struct Seeded {
    /// The version placed, for messages.
    version: String,
    /// sha256 of every file as *we* installed it.
    ///
    /// The whole of the "did anyone edit this" test: a file that no longer
    /// hashes to this was changed by somebody, and somebody else's work is not
    /// ours to replace. Machine droppings are excluded when the comparison is
    /// made, so this is the set we placed rather than the set on disk.
    files: BTreeMap<String, String>,
    /// sha256 of the archive, when this build placed it.
    ///
    /// `None` means the pack was adopted — already installed before shipping
    /// began — and an adopted pack is never upgraded either.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    shipped: Option<String>,
}

/// Place the shipped packs into `dir`, reporting what changed.
///
/// Returns one line per pack actually placed or updated; a steady-state run
/// returns nothing, which is the point.
pub async fn seed(dir: &Path) -> Result<Vec<String>> {
    seed_with_record(dir, &crate::settings::data_dir().join("preinstalled.json")).await
}

/// [`seed`] with the record somewhere a caller chooses — tests pick a scratch
/// path so they cannot touch the real one.
///
/// Returns one line per pack placed, updated, retired, or kept with a reason;
/// a steady-state run on a fully consolidated machine returns nothing, which
/// is the point.
pub async fn seed_with_record(dir: &Path, record_path: &Path) -> Result<Vec<String>> {
    let lock_path = lock_path_for(record_path);
    if let Some(parent) = lock_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }

    // One writer at a time, across processes. Two instances starting together
    // both find nothing installed and both place the same packs into the same
    // directories, and the record is written whole: the second writer drops the
    // first one's entries, and a dropped entry reads as "the user deleted that
    // pack" on the next start — the one decision this file never undoes. The
    // lock covers the whole pass, and only its holder writes the record.
    let acquired = SeedLock::acquire(&lock_path)?;
    let Some(lock) = acquired else {
        return Ok(vec![
            "seed skipped: another instance is placing the shipped packs".to_string(),
        ]);
    };

    let mut record = read_record(record_path);
    let mut placed = Vec::new();
    let mut dirty = false;

    // Read once: placement adds directories, but every shipped pack is looked
    // up by its own name, and an unreadable listing is a root cause rather than
    // one pack's problem.
    let installed = crate::pack::installed(dir)?;

    for (name, version, bytes) in SHIPPED {
        // Restate that this instance is still working. A first start fetches
        // packs and can outlast the staleness window, and a seed that is
        // half-way through placing is exactly the one nobody may take over.
        lock.touch();
        // One pack at a time. A locked directory, an unreadable file, a full
        // disk is one line in the report; the packs around it are placed and
        // stay placed, which is what the retire branch below already promises.
        let outcome: Result<()> = async {
            let Some(pack) = pick_installed(&installed, name, version) else {
                // Not installed. Once it has been and is gone again, that was a
                // decision: the record is what keeps us from placing it back.
                if record.packs.contains_key(*name) {
                    return Ok(());
                }
                let path = place(dir, name, bytes).await?;
                record.packs.insert(
                    name.to_string(),
                    Seeded {
                        version: version.to_string(),
                        files: hashes_of(&path)?,
                        shipped: Some(sha256(bytes)),
                    },
                );
                dirty = true;
                placed.push(format!("placed {name}@{version}"));
                return Ok(());
            };

            // Installed already: unloaded means the user said no.
            if !pack.enabled {
                return Ok(());
            }
            // First sighting of a pack we did not place: adopt it — record what
            // is there, so an edit from here on is recognisably not ours.
            // Adoption then falls straight through to the same upgrade check
            // everything else takes, because "we have never seen this" and
            // "this is ours, older" deserve the same question: is it
            // untouched, and do we ship something newer?
            if !record.packs.contains_key(*name) {
                record.packs.insert(
                    name.to_string(),
                    Seeded {
                        version: pack.manifest.version_id.clone(),
                        files: hashes_of(&pack.path)?,
                        shipped: None,
                    },
                );
                dirty = true;
            }
            let (was_shipped, recorded_version, recorded_files) = {
                let entry = &record.packs[*name];
                (
                    entry.shipped.clone(),
                    entry.version.clone(),
                    entry.files.clone(),
                )
            };
            // Two reasons to replace what is on disk: this build placed it and
            // is now shipping a different archive, or the pack was adopted and
            // this build ships a different *version* of it. Both are worthless
            // without the next test — an edited pack stays, whatever we ship.
            let upgrade = match &was_shipped {
                Some(archive) => archive != &sha256(bytes),
                None => recorded_version != *version,
            };
            if !upgrade {
                return Ok(());
            }
            // The archive's sha256 is compared against the record, and that is
            // all it is: a consistency check on our own bookkeeping ("are these
            // the bytes we placed?"), never a statement about where the archive
            // came from. A local pack is not a signed artifact and this record
            // is not a root of trust; proving provenance is a design decision
            // this file does not make. A mismatch means the contents are no
            // longer ours to overwrite, so the update is reported and the pack
            // on disk is kept — silently skipping it is how a user ends up with
            // an old pack and no idea why.
            if !untouched(&pack.path, &recorded_files)? {
                placed.push(format!(
                    "{name}@{recorded_version} kept: its files changed since it was placed, \
                     so the shipped {version} was not installed"
                ));
                return Ok(());
            }
            match replace(dir, name, version, bytes, &pack.path).await {
                Ok((path, warning)) => {
                    record.packs.insert(
                        name.to_string(),
                        Seeded {
                            version: version.to_string(),
                            files: hashes_of(&path)?,
                            shipped: Some(sha256(bytes)),
                        },
                    );
                    dirty = true;
                    placed.push(format!("placed {name}@{version} (updated)"));
                    if let Some(warning) = warning {
                        placed.push(warning);
                    }
                }
                Err(error) => {
                    // The record keeps its old values: the installed copy is
                    // still on disk (see `replace`), and the next start must
                    // retry this update rather than read the entry as a pack
                    // the user deleted.
                    placed.push(format!(
                        "{name}@{recorded_version} kept: the update to {version} failed \
                         ({error:#}) — will retry next run"
                    ));
                }
            }
            Ok(())
        }
        .await;
        if let Err(error) = outcome {
            placed.push(format!("{name} could not be installed: {error:#}"));
        }
    }

    // A placement that succeeded must survive even if the listing below turns
    // out to be unreadable: an unrecorded pack reads as one the user deleted on
    // the next start.
    if dirty {
        write_record(record_path, &record)?;
        dirty = false;
    }

    // Retire the packs `starter` merged into: a twin soul or a second
    // `builtin:search` loading beside starter's is a conflict nobody asked
    // for. Only our own untouched bytes go (recorded, still matching, and
    // `shipped`); everything else stays and says why.
    lock.touch();
    for name in MERGED_INTO_STARTER {
        for pack in crate::pack::installed(dir)?
            .into_iter()
            .filter(|pack| pack.manifest.name == *name)
            .collect::<Vec<_>>()
        {
            let version = pack.manifest.version_id.clone();
            let Some(entry) = record.packs.get(*name) else {
                placed.push(format!(
                    "{name}@{version} kept: never placed by this program, so untouched cannot be proven"
                ));
                continue;
            };
            if entry.shipped.is_none() {
                placed.push(format!(
                    "{name}@{version} kept: adopted from your own install — starter carries the same content"
                ));
                continue;
            }
            // An unreadable file is one pack's problem, reported here instead
            // of ending the pass: the rest of this report is worth more than a
            // single `?`.
            match untouched(&pack.path, &entry.files) {
                Ok(true) => {}
                Ok(false) => {
                    placed.push(format!(
                        "{name}@{version} kept: you have edits; it may now duplicate starter's copy"
                    ));
                    continue;
                }
                Err(error) => {
                    placed.push(format!(
                        "{name}@{version} kept: its files could not all be read ({error:#})"
                    ));
                    continue;
                }
            }
            // Removal can fail for reasons that are not the pack's content —
            // the desktop is running the driver right now. Report it, keep the
            // record so the next run retries, and finish the seed: one locked
            // pack must not swallow the report of everything else.
            if let Err(error) = std::fs::remove_dir_all(&pack.path) {
                placed.push(format!(
                    "{name}@{version} kept: cannot remove yet ({error:#}) — in use; will retry next run"
                ));
                continue;
            }
            record.packs.remove(*name);
            dirty = true;
            placed.push(format!("{name}@{version} retired: merged into starter"));
        }
    }

    // A retire that died halfway leaves a husk: the manifest gone, the locked
    // files left — and installed() cannot see a directory without a manifest,
    // so the loop above would miss it forever. Sweep by name, touching only a
    // directory this program has a record of placing.
    for name in MERGED_INTO_STARTER {
        let Ok(entries) = std::fs::read_dir(dir) else {
            continue;
        };
        let prefix = format!("{name}-");
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_dir()
                || !entry.file_name().to_string_lossy().starts_with(&prefix)
                || path.join("dsh.index.json").exists()
            {
                continue;
            }
            let ours = record
                .packs
                .get(*name)
                .is_some_and(|entry| entry.shipped.is_some());
            if !ours {
                continue;
            }
            match std::fs::remove_dir_all(&path) {
                Ok(()) => {
                    record.packs.remove(*name);
                    dirty = true;
                    placed.push(format!(
                        "{name} leftover retired: the remains of an earlier attempt"
                    ));
                }
                Err(error) => placed.push(format!(
                    "{name} leftover kept: cannot remove yet ({error:#}) — in use; will retry next run"
                )),
            }
        }
    }

    if dirty {
        write_record(record_path, &record)?;
    }
    Ok(placed)
}

/// The record, or an empty one.
///
/// A record that cannot be parsed is treated as empty rather than as an error:
/// refusing to start over our own bookkeeping helps nobody, and an empty record
/// is exactly the "never seen this" state the adopt path already handles.
fn read_record(path: &Path) -> Record {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok())
        .unwrap_or_default()
}

/// Write the record whole, through a temporary file and a rename.
///
/// A half-written record reads as "nothing was ever placed", which resurrects
/// every pack the user deleted.
fn write_record(path: &Path, record: &Record) -> Result<()> {
    let name = path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| "preinstalled.json".to_string());
    let temp = path.with_file_name(format!(
        ".{name}.{}.tmp",
        uuid::Uuid::new_v4().simple()
    ));
    std::fs::write(&temp, serde_json::to_string_pretty(record)?)
        .with_context(|| format!("writing {}", temp.display()))?;
    if let Err(error) = std::fs::rename(&temp, path) {
        let _ = std::fs::remove_file(&temp);
        return Err(error).with_context(|| format!("writing {}", path.display()));
    }
    Ok(())
}

/// Where the seed lock lives: beside the record, because the record is the one
/// thing two simultaneous starts can corrupt beyond recovery.
fn lock_path_for(record_path: &Path) -> PathBuf {
    let parent = match record_path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent,
        _ => Path::new("."),
    };
    parent.join(".seed.lock")
}

/// How long an untouched seed lock counts as held.
///
/// Longer than any seed this program performs (the steady state is a directory
/// listing and some hashing) and short enough that a seed killed mid-place
/// cannot wedge every later start.
const LOCK_STALE_AFTER: Duration = Duration::from_secs(30);

/// The seed's cross-process lock.
///
/// A file created with `create_new`: exactly one process wins, and the losers
/// report that somebody else is seeding instead of interleaving a half-written
/// pack with somebody else's half-written pack.
struct SeedLock {
    path: PathBuf,
    /// Identifies this holder: written into the file and compared on every
    /// refresh and on release, so a lock taken over as stale is never refreshed
    /// (or released) by the instance that lost it.
    id: String,
}

impl SeedLock {
    /// Take the lock, or `None` when another instance holds a live one.
    ///
    /// A lock nobody has refreshed for [`LOCK_STALE_AFTER`] is treated as
    /// abandoned and taken over: the alternative is a crash during the first
    /// start costing every start after it.
    fn acquire(path: &Path) -> Result<Option<Self>> {
        let lock = Self {
            path: path.to_path_buf(),
            id: uuid::Uuid::new_v4().simple().to_string(),
        };
        match Self::create(path, &lock.contents()) {
            Ok(()) => Ok(Some(lock)),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                if !Self::is_stale(path) {
                    return Ok(None);
                }
                let _ = std::fs::remove_file(path);
                match Self::create(path, &lock.contents()) {
                    // Somebody else may have taken the abandoned lock first,
                    // and that is not an error: they are seeding.
                    Ok(()) => Ok(Some(lock)),
                    Err(_) => Ok(None),
                }
            }
            Err(error) => {
                Err(error).with_context(|| format!("creating {}", path.display()))
            }
        }
    }

    /// What a lock file holds: who holds it, since when, and which instance.
    fn contents(&self) -> String {
        format!("{}\n{}\n{}\n", std::process::id(), unix_now(), self.id)
    }

    fn create(path: &Path, contents: &str) -> std::io::Result<()> {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?;
        file.write_all(contents.as_bytes())
    }

    /// Whether the lock file is still this holder's.
    fn owns(&self) -> bool {
        std::fs::read_to_string(&self.path)
            .map(|text| text.lines().nth(2).map(str::trim) == Some(self.id.as_str()))
            .unwrap_or(false)
    }

    /// Restate that this holder is still working.
    ///
    /// A first start fetches packs and can take minutes; without this, any seed
    /// running longer than [`LOCK_STALE_AFTER`] would be "abandoned" while it
    /// was busy placing, which is the collision the lock exists to prevent.
    fn touch(&self) {
        if self.owns() {
            let _ = std::fs::write(&self.path, self.contents());
        }
    }

    /// Whether a lock file was abandoned.
    ///
    /// The moment it was taken is written into the file rather than read from
    /// its modification time: the content is what a hand-built lock in a test
    /// can state, and it survives a filesystem that does not record times.
    fn is_stale(path: &Path) -> bool {
        let Ok(text) = std::fs::read_to_string(path) else {
            // Unreadable is indistinguishable from abandoned, and treating it
            // as live would let one bad file stop every later start.
            return true;
        };
        match text
            .lines()
            .nth(1)
            .and_then(|line| line.trim().parse::<u64>().ok())
        {
            Some(taken) => unix_now().saturating_sub(taken) >= LOCK_STALE_AFTER.as_secs(),
            None => true,
        }
    }

    /// Give the lock back, but only if it is still ours.
    fn release(&self) {
        if self.owns() {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

impl Drop for SeedLock {
    fn drop(&mut self) {
        self.release();
    }
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs())
        .unwrap_or(0)
}

/// The installed pack this build is talking about.
///
/// More than one directory can carry the same name — an update that could not
/// remove the copy it superseded leaves one behind — so the copy at the shipped
/// version wins: it is the one this build placed, and the one the record is
/// expected to describe.
fn pick_installed<'a>(
    installed: &'a [crate::pack::InstalledPack],
    name: &str,
    version: &str,
) -> Option<&'a crate::pack::InstalledPack> {
    let mut first = None;
    for pack in installed.iter().filter(|pack| pack.manifest.name == name) {
        if pack.manifest.version_id == version {
            return Some(pack);
        }
        if first.is_none() {
            first = Some(pack);
        }
    }
    first
}

/// Place `bytes` over the pack already installed at `old`.
///
/// The old order — remove the installed directory, then install — left nothing
/// behind when the install failed: the record still claimed the pack, so the
/// next start read it as one the user deleted and never placed it again. This
/// order keeps a copy on disk at every moment. The copy is moved aside rather
/// than deleted (and `installed()` still reads a moved-aside directory, so even
/// a process killed here leaves a loadable pack), the new copy is unpacked where
/// it belongs, and only then does the old copy go. Any failure before that puts
/// the old copy back, so a failed update costs a retry and nothing else.
///
/// The new copy is unpacked *at the destination* rather than in a sibling
/// staging directory, because `pack::install` resolves the absolute command
/// paths it bakes into `mcp.json` against the packs directory it is given: a
/// staged directory would leave those commands pointing at a path that is about
/// to be renamed away.
async fn replace(
    dir: &Path,
    name: &str,
    version: &str,
    bytes: &[u8],
    old: &Path,
) -> Result<(PathBuf, Option<String>)> {
    let target = dir.join(format!("{name}-{version}"));
    // Same path means a rebuild of the same version, and `pack::install`
    // removes the destination it is about to write: move it aside first so
    // there is never a moment with no copy on disk.
    let aside = (old == target.as_path()).then(|| {
        target.with_file_name(format!(
            "{name}-{version}.old-{}",
            uuid::Uuid::new_v4().simple()
        ))
    });
    if let Some(aside) = aside.as_ref() {
        if let Err(error) = std::fs::rename(&target, aside) {
            // Nothing else has been touched: the pack is exactly where it was.
            return Err(error).with_context(|| format!("moving {} aside", target.display()));
        }
    }

    let placed = match place(dir, name, bytes).await {
        Ok(path) => path,
        Err(error) => {
            if let Some(aside) = aside.as_ref() {
                // Put the installed copy back. If even this fails the copy is
                // still on disk under its `.old-` name, which `installed()`
                // reads like any other pack: no loss either way.
                let _ = std::fs::rename(aside, &target);
            }
            return Err(error);
        }
    };

    // The new copy is complete, so the superseded ones can go. A failure here
    // is a line in the report, not a lost update.
    let mut stale: Vec<PathBuf> = Vec::new();
    if let Some(aside) = aside {
        stale.push(aside);
    }
    if old != placed.as_path() {
        stale.push(old.to_path_buf());
    }
    let mut warning = None;
    for path in stale {
        if let Err(error) = std::fs::remove_dir_all(&path) {
            warning = Some(format!(
                "{name}@{version}: the superseded copy in {} could not be removed yet ({error:#}) — will retry next run",
                path.display()
            ));
        }
    }
    Ok((placed, warning))
}

/// Whether a pack directory still holds exactly the files this program placed.
///
/// This is a consistency check on our own bookkeeping ("are these the bytes we
/// put there?"), not a statement about where the archive came from; see the
/// comment where an archive's sha256 is compared. Only files count, and only
/// the ones recorded: a pack directory is also a working directory for whatever
/// the pack loads, and treating a driver's log or an editor's droppings as the
/// user's edit pins the pack to its old version for good.
///
/// The record's own set is the list to compare against: a manifest declares
/// content references and fetches, not a file inventory, so the files this
/// program wrote are the only trustworthy statement of what it put there.
fn untouched(dir: &Path, recorded: &BTreeMap<String, String>) -> Result<bool> {
    Ok(without_noise(hashes_of(dir)?) == without_noise(recorded.clone()))
}

/// The same file set with machine droppings removed, on both sides.
fn without_noise(files: BTreeMap<String, String>) -> BTreeMap<String, String> {
    files
        .into_iter()
        .filter(|(relative, _)| !is_machine_noise(relative))
        .collect()
}

/// Files that belong to the machine rather than to the pack.
///
/// Deliberately not on this list: `.ngu-disabled`, because unloading a pack is
/// the user's decision and must keep counting as one.
fn is_machine_noise(relative: &str) -> bool {
    let name = relative.rsplit('/').next().unwrap_or(relative);
    let lower = name.to_ascii_lowercase();
    lower.ends_with(".log")
        || lower.ends_with(".tmp")
        || lower.ends_with(".swp")
        || lower.ends_with(".bak")
        || name == ".DS_Store"
        || name == "Thumbs.db"
        || name == "desktop.ini"
        || name.starts_with("~$")
        || relative.starts_with("target/")
        || relative.contains("/target/")
        || relative.starts_with(".git/")
}

/// Unpack one embedded archive into the packs directory.
async fn place(dir: &Path, name: &str, bytes: &[u8]) -> Result<PathBuf> {
    // Unique per call: two hosts (or two tests) may place at the same moment,
    // and a shared path would have one writing over the other's archive.
    let temp = std::env::temp_dir().join(format!(
        "ngu-shipped-{name}-{}.dshpack",
        uuid::Uuid::new_v4().simple()
    ));
    std::fs::write(&temp, bytes).with_context(|| format!("writing {}", temp.display()))?;
    let result = crate::pack::install(&temp, dir)
        .await
        .with_context(|| format!("placing the shipped pack '{name}'"));
    let _ = std::fs::remove_file(&temp);
    Ok(result?.path)
}

/// sha256 of every file in a directory, by relative path.
fn hashes_of(dir: &Path) -> Result<BTreeMap<String, String>> {
    let mut out = BTreeMap::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(current) = stack.pop() {
        for entry in std::fs::read_dir(&current)
            .with_context(|| format!("listing {}", current.display()))?
        {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            let relative = path
                .strip_prefix(dir)
                .unwrap_or(&path)
                .to_string_lossy()
                .replace('\\', "/");
            let bytes =
                std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
            out.insert(relative, sha256(&bytes));
        }
    }
    Ok(out)
}

fn sha256(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn scratch(tag: &str) -> (PathBuf, PathBuf) {
        let root = std::env::temp_dir().join(format!(
            "ngu-seed-{tag}-{}",
            uuid::Uuid::new_v4().simple()
        ));
        let dir = root.join("packs");
        std::fs::create_dir_all(&dir).unwrap();
        // The record lives outside the packs directory: it is this program's
        // bookkeeping, not a pack, and a stray file inside the packs directory
        // would be a stray file inside the packs directory.
        (dir, root.join("preinstalled.json"))
    }

    #[test]
    fn every_shipped_pack_is_present_and_named() {
        // A renamed archive would fail at build time (the include), but a
        // mismatched name inside the manifest would only show up on a machine
        // nobody has run it on yet.
        assert!(!SHIPPED.is_empty());
        for (name, version, bytes) in SHIPPED {
            assert!(!bytes.is_empty(), "{name} ships nothing");
            let temp = std::env::temp_dir().join(format!(
                "ngu-shipped-check-{}-{}.dshpack",
                name,
                uuid::Uuid::new_v4().simple()
            ));
            std::fs::write(&temp, bytes).unwrap();
            let manifest = crate::pack::verify(&temp).unwrap().manifest;
            let _ = std::fs::remove_file(&temp);
            assert_eq!(manifest.name, *name, "{name} archive holds another pack");
            assert_eq!(manifest.version_id, *version);
        }
    }

    #[test]
    fn every_shipped_pack_source_parses_and_matches_the_shipped_version() {
        // The archives are checked above; the *source directory* is what the
        // next edit touches, and a broken manifest there only surfaces when
        // somebody packs it — one missing comma in chinese's manifest went out
        // exactly that way. Version drift matters for the same reason: the
        // record compares versions to decide whether to replace what is on
        // disk.
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("packs");
        for (name, version, _) in SHIPPED {
            let manifest = crate::pack::read_manifest(&root.join(name))
                .unwrap_or_else(|error| panic!("{name} source manifest: {error:#}"));
            assert_eq!(manifest.name, *name, "{name}");
            assert_eq!(
                manifest.version_id, *version,
                "{name} source version drifted from SHIPPED"
            );
        }
    }

    #[tokio::test]
    async fn seeding_places_every_pack_once_and_never_again() {
        let (dir, record) = scratch("place");

        let placed = seed_with_record(&dir, &record).await.unwrap();
        assert_eq!(placed.len(), SHIPPED.len(), "{placed:?}");
        assert_eq!(crate::pack::installed(&dir).unwrap().len(), SHIPPED.len());

        let again = seed_with_record(&dir, &record).await.unwrap();
        assert!(again.is_empty(), "the steady state does nothing: {again:?}");

        // Deleting one is the user's decision, and the record is what keeps it
        // deleted — a program that reinstalls what you just removed is a
        // program you cannot turn off.
        let installed = crate::pack::installed(&dir).unwrap();
        let target = installed
            .iter()
            .find(|pack| pack.manifest.name == "judge")
            .unwrap();
        std::fs::remove_dir_all(&target.path).unwrap();
        let after = seed_with_record(&dir, &record).await.unwrap();
        assert!(
            after.is_empty(),
            "a deleted pack must not come back: {after:?}"
        );
    }

    #[tokio::test]
    async fn an_edited_pack_is_never_replaced() {
        let (dir, record) = scratch("edited");
        seed_with_record(&dir, &record).await.unwrap();

        let installed = crate::pack::installed(&dir).unwrap();
        let soul = installed
            .iter()
            .find(|pack| pack.manifest.name == "starter")
            .unwrap()
            .path
            .join("soul.md");
        std::fs::write(&soul, "You are someone else entirely.\n").unwrap();

        // Pretend this build ships something different: the record's idea of
        // what was placed no longer matches any archive we have.
        let mut value: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&record).unwrap()).unwrap();
        value["packs"]["starter"]["shipped"] = json!("a-different-archive");
        std::fs::write(&record, serde_json::to_string(&value).unwrap()).unwrap();

        let again = seed_with_record(&dir, &record).await.unwrap();
        assert!(
            !again.iter().any(|line| line.starts_with("placed starter@")),
            "an edited pack is left alone: {again:?}"
        );
        // ...and says so. A pack that keeps its old version with no word about
        // it is indistinguishable from a bug; the line is the whole of the
        // "reported, old pack kept" behaviour.
        assert!(
            again
                .iter()
                .any(|line| line.contains("starter@1.0.0") && line.contains("kept")),
            "the mismatch is reported: {again:?}"
        );
        assert_eq!(
            std::fs::read_to_string(&soul).unwrap(),
            "You are someone else entirely.\n"
        );
        // The record keeps what it had: this build placed nothing, so it
        // records nothing, and the next start asks the same question again.
        let value: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&record).unwrap()).unwrap();
        assert_eq!(value["packs"]["starter"]["shipped"], json!("a-different-archive"));
    }

    #[tokio::test]
    async fn an_untouched_pack_is_replaced_by_the_newer_shipped_one() {
        let (dir, record) = scratch("upgrade");
        seed_with_record(&dir, &record).await.unwrap();

        // Same trick as above — a build we have never seen — but this time
        // what is on disk is untouched, which is the whole difference.
        let mut value: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&record).unwrap()).unwrap();
        value["packs"]["starter"]["shipped"] = json!("a-different-archive");
        std::fs::write(&record, serde_json::to_string(&value).unwrap()).unwrap();

        let again = seed_with_record(&dir, &record).await.unwrap();
        assert!(
            again.iter().any(|line| line.starts_with("placed starter@")),
            "an untouched pack takes the update: {again:?}"
        );
        let installed = crate::pack::installed(&dir).unwrap();
        assert!(
            installed
                .iter()
                .any(|pack| pack.manifest.name == "starter" && pack.enabled),
            "and is still installed and enabled"
        );

        // And after that the record agrees with the archive again.
        let steady = seed_with_record(&dir, &record).await.unwrap();
        assert!(steady.is_empty(), "{steady:?}");
    }

    #[tokio::test]
    async fn an_unloaded_pack_stays_unloaded() {
        let (dir, record) = scratch("unloaded");
        seed_with_record(&dir, &record).await.unwrap();

        let installed = crate::pack::installed(&dir).unwrap();
        let target = installed
            .iter()
            .find(|pack| pack.manifest.name == "judge")
            .unwrap();
        crate::pack::set_enabled(&target.path, false).unwrap();

        let again = seed_with_record(&dir, &record).await.unwrap();
        assert!(
            again.is_empty(),
            "an unloaded pack is not touched: {again:?}"
        );
        let installed = crate::pack::installed(&dir).unwrap();
        assert!(
            !installed
                .iter()
                .find(|pack| pack.manifest.name == "judge")
                .unwrap()
                .enabled,
            "and is still unloaded"
        );
    }

    #[tokio::test]
    async fn merged_twins_retire_only_when_ours_and_untouched() {
        let (dir, record) = scratch("retire");
        seed_with_record(&dir, &record).await.unwrap();

        // Four twins a pre-merge machine could still have. The first is the
        // normal case — ours, recorded, untouched — and retires. The rest are
        // the three reasons *not* to touch a pack, one each.
        let manifest = |name: &str, version: &str| {
            format!(
                r#"{{"formatVersion":1,"game":"nguruvilu","name":"{name}","versionId":"{version}","license":"MIT","kernelVersion":"0.1.0","dependencies":{{"nguruvilu":">=0.1.0"}}}}"#
            )
        };
        let make = |dir: &Path, name: &str, version: &str, file: &str, bytes: &str| {
            let pack = dir.join(format!("{name}-{version}"));
            std::fs::create_dir_all(&pack).unwrap();
            std::fs::write(pack.join("dsh.index.json"), manifest(name, version)).unwrap();
            std::fs::write(pack.join(file), bytes).unwrap();
            pack
        };

        let untouched = make(&dir, "chinese", "1.1.0", "soul.md", "old soul\n");
        let edited = make(&dir, "ui", "1.0.0", "index.html", "pristine\n");
        let adopted = make(&dir, "search", "1.1.0", "search.json", "{}\n");
        let stranger = make(&dir, "subagent", "1.0.0", "notes.txt", "x\n");

        let mut value: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&record).unwrap()).unwrap();
        value["packs"]["chinese"] = json!({
            "version": "1.1.0",
            "files": hashes_of(&untouched).unwrap(),
            "shipped": "our-archive",
        });
        let pristine = hashes_of(&edited).unwrap();
        std::fs::write(edited.join("index.html"), "user edited\n").unwrap();
        value["packs"]["ui"] = json!({
            "version": "1.0.0",
            "files": pristine,
            "shipped": "our-archive",
        });
        value["packs"]["search"] = json!({
            "version": "1.1.0",
            "files": hashes_of(&adopted).unwrap(),
        });
        std::fs::write(&record, serde_json::to_string(&value).unwrap()).unwrap();

        let report = seed_with_record(&dir, &record).await.unwrap();
        let line = |needle: &str| {
            report
                .iter()
                .find(|l| l.contains(needle))
                .cloned()
                .unwrap_or_else(|| panic!("no line for {needle}: {report:?}"))
        };

        assert!(!untouched.exists(), "ours and untouched retires: {report:?}");
        assert!(line("chinese@1.1.0").contains("retired"));
        assert!(edited.exists(), "an edit is never destroyed: {report:?}");
        assert!(line("ui@1.0.0").contains("edits"));
        assert!(adopted.exists(), "an adopted pack was theirs first");
        assert!(line("search@1.1.0").contains("adopted"));
        assert!(stranger.exists(), "unproven stays");
        assert!(line("subagent@1.0.0").contains("never placed"));

        // The retired one stays gone; the kept ones are named again next run —
        // reporting every time is the honest version of a warning.
        let again = seed_with_record(&dir, &record).await.unwrap();
        assert!(!untouched.exists());
        assert!(!again.iter().any(|l| l.contains("chinese@")), "{again:?}");
        assert!(again.iter().any(|l| l.contains("ui@1.0.0")), "{again:?}");

        let names: Vec<String> = crate::pack::installed(&dir)
            .unwrap()
            .into_iter()
            .map(|pack| pack.manifest.name)
            .collect();
        assert!(names.contains(&"starter".to_string()), "{names:?}");
        assert!(names.contains(&"judge".to_string()), "{names:?}");
        assert!(!names.contains(&"chinese".to_string()), "{names:?}");
    }

    #[tokio::test]
    async fn a_manifest_less_husk_is_swept_only_when_ours() {
        let (dir, record) = scratch("husk");
        seed_with_record(&dir, &record).await.unwrap();

        // What a retire that died halfway leaves: the manifest gone, the
        // record still saying we placed it.
        let husk = dir.join("computer-use-1.2.0");
        std::fs::create_dir_all(husk.join("files")).unwrap();
        std::fs::write(husk.join("files").join("locked.bin"), "x\n").unwrap();
        let mut value: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&record).unwrap()).unwrap();
        value["packs"]["computer-use"] = json!({
            "version": "1.2.0",
            "files": json!({}),
            "shipped": "our-archive",
        });
        std::fs::write(&record, serde_json::to_string(&value).unwrap()).unwrap();

        // And one nobody placed: a husk under a merged name with no record is
        // not provably ours, so it stays.
        let stranger_husk = dir.join("search-1.1.0");
        std::fs::create_dir_all(&stranger_husk).unwrap();
        std::fs::write(stranger_husk.join("notes.txt"), "someone's\n").unwrap();

        let report = seed_with_record(&dir, &record).await.unwrap();
        assert!(!husk.exists(), "our husk is swept: {report:?}");
        assert!(
            report.iter().any(|l| l.contains("computer-use") && l.contains("leftover")),
            "{report:?}"
        );
        assert!(stranger_husk.exists(), "an unproven husk stays: {report:?}");
    }

    /// A pack directory of the shape an older install leaves behind.
    fn write_pack(dir: &Path, name: &str, version: &str, file: &str, body: &str) -> PathBuf {
        let pack = dir.join(format!("{name}-{version}"));
        std::fs::create_dir_all(&pack).unwrap();
        std::fs::write(
            pack.join("dsh.index.json"),
            format!(
                r#"{{"formatVersion":1,"game":"nguruvilu","name":"{name}","versionId":"{version}","license":"MIT","kernelVersion":"0.1.0","dependencies":{{"nguruvilu":">=0.1.0"}}}}"#
            ),
        )
        .unwrap();
        std::fs::write(pack.join(file), body).unwrap();
        pack
    }

    fn arm_an_update(record: &Path, name: &str) {
        let mut value: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(record).unwrap()).unwrap();
        value["packs"][name]["shipped"] = json!("a-different-archive");
        std::fs::write(record, serde_json::to_string(&value).unwrap()).unwrap();
    }

    #[tokio::test]
    async fn a_machine_written_file_does_not_pin_a_pack_to_its_old_version() {
        // A driver's log and a build cache are not edits. Comparing every file
        // in the directory made them read as one, and an "edited" pack is never
        // replaced again — permanently, silently.
        let (dir, record) = scratch("noise");
        seed_with_record(&dir, &record).await.unwrap();
        let starter = crate::pack::installed(&dir)
            .unwrap()
            .into_iter()
            .find(|pack| pack.manifest.name == "starter")
            .unwrap()
            .path;
        std::fs::write(starter.join("driver.log"), "connected\n").unwrap();
        std::fs::create_dir_all(starter.join("target")).unwrap();
        std::fs::write(starter.join("target").join("cache.bin"), "x\n").unwrap();

        arm_an_update(&record, "starter");
        let again = seed_with_record(&dir, &record).await.unwrap();
        assert!(
            again
                .iter()
                .any(|line| line.starts_with("placed starter@") && line.contains("updated")),
            "machine droppings are not edits: {again:?}"
        );
    }

    #[tokio::test]
    async fn one_pack_that_cannot_be_placed_does_not_stop_the_others() {
        // judge's destination is occupied by a file, so its install cannot run.
        // starter's is not, and starter must still be placed and still be
        // recorded: the pass reports per pack rather than ending at the first.
        let (dir, record) = scratch("aggregate");
        std::fs::write(dir.join("judge-1.0.0"), "in the way\n").unwrap();

        let report = seed_with_record(&dir, &record)
            .await
            .expect("one pack's failure is not the pass's failure");
        assert!(
            report.iter().any(|line| line.starts_with("placed starter@")),
            "{report:?}"
        );
        assert!(
            report
                .iter()
                .any(|line| line.contains("judge") && line.contains("could not")),
            "the failure is reported: {report:?}"
        );
        assert!(dir.join("starter-1.0.0").is_dir(), "{report:?}");

        let value: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&record).unwrap()).unwrap();
        assert_eq!(value["packs"]["starter"]["version"], json!("1.0.0"));
        assert!(
            value["packs"]["judge"].is_null(),
            "a pack that never landed is not recorded as placed"
        );

        // And the next start still tries it, because it was never recorded as
        // "placed and then deleted".
        std::fs::remove_file(dir.join("judge-1.0.0")).unwrap();
        let again = seed_with_record(&dir, &record).await.unwrap();
        assert!(
            again.iter().any(|line| line.contains("placed judge@1.0.0")),
            "{again:?}"
        );
    }

    #[tokio::test]
    async fn an_adopted_pack_whose_update_fails_keeps_its_copy_and_retries() {
        // The old order deleted the installed directory before installing, so a
        // failed install left nothing — and the record still claimed the pack,
        // which the next start reads as "the user deleted it" and skips for
        // ever. Both halves are checked here: the copy survives, and the retry
        // happens.
        let (dir, record) = scratch("failed-update");
        let old = write_pack(&dir, "judge", "0.9.0", "soul.md", "old judge\n");
        let blocked = dir.join("judge-1.0.0");
        std::fs::write(&blocked, "in the way\n").unwrap();

        let report = seed_with_record(&dir, &record).await.unwrap();
        assert!(
            report.iter().any(|line| line.starts_with("placed starter@")),
            "{report:?}"
        );
        assert!(
            report
                .iter()
                .any(|line| line.contains("judge@0.9.0") && line.contains("kept")),
            "the failed update is reported: {report:?}"
        );
        assert!(old.join("soul.md").is_file(), "the installed copy survives");
        let value: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&record).unwrap()).unwrap();
        assert_eq!(value["packs"]["judge"]["version"], json!("0.9.0"));

        // Take the obstacle away: the same upgrade goes through.
        std::fs::remove_file(&blocked).unwrap();
        let again = seed_with_record(&dir, &record).await.unwrap();
        assert!(
            again.iter().any(|line| line.contains("placed judge@1.0.0")),
            "{again:?}"
        );
        assert!(!old.exists(), "the superseded copy goes once the new one is in place: {again:?}");
    }

    #[tokio::test]
    async fn a_failed_replacement_puts_the_installed_copy_back() {
        // The seam: a rebuild of the same version moves the installed copy
        // aside before installing, and this failure lands exactly there. The
        // copy must come back, and nothing half-done may be left behind.
        let (dir, _record) = scratch("rollback");
        let installed = write_pack(&dir, "judge", "1.0.0", "soul.md", "old judge\n");

        replace(&dir, "judge", "1.0.0", b"not an archive at all", &installed)
            .await
            .expect_err("a corrupt archive cannot be placed");

        assert!(
            installed.join("soul.md").is_file(),
            "the installed copy is put back"
        );
        assert_eq!(
            std::fs::read_to_string(installed.join("soul.md")).unwrap(),
            "old judge\n"
        );
        let names: Vec<String> = std::fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .map(|entry| entry.file_name().to_string_lossy().into_owned())
            .collect();
        assert!(
            !names.iter().any(|name| name.contains(".old-")),
            "no half-done copy is left: {names:?}"
        );
    }

    #[test]
    fn the_seed_lock_admits_one_writer_and_is_released_with_its_holder() {
        let root = std::env::temp_dir().join(format!(
            "ngu-seed-lock-{}",
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let path = lock_path_for(&root.join("preinstalled.json"));

        let first = SeedLock::acquire(&path)
            .unwrap()
            .expect("the first instance takes the lock");
        assert!(path.is_file());
        assert!(
            SeedLock::acquire(&path).unwrap().is_none(),
            "a second instance must not seed"
        );
        drop(first);
        assert!(!path.exists(), "the lock goes with its holder");
        assert!(
            SeedLock::acquire(&path).unwrap().is_some(),
            "and the next start seeds"
        );
    }

    #[test]
    fn an_abandoned_lock_is_taken_over() {
        // A seed killed mid-place leaves its lock behind; treating it as live
        // would stop every start after it.
        let root = std::env::temp_dir().join(format!(
            "ngu-seed-stale-{}",
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let path = lock_path_for(&root.join("preinstalled.json"));
        std::fs::write(&path, format!("999\n{}\nx\n", unix_now() - 3600)).unwrap();

        assert!(SeedLock::acquire(&path).unwrap().is_some());
    }

    #[test]
    fn a_working_seed_does_not_look_abandoned() {
        // A first start fetches packs and can take minutes. Without a refresh,
        // any pass longer than the staleness window would read as abandoned
        // while it was busy placing — the exact collision the lock prevents.
        let root = std::env::temp_dir().join(format!(
            "ngu-seed-touch-{}",
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let path = lock_path_for(&root.join("preinstalled.json"));
        let held = SeedLock::acquire(&path).unwrap().expect("held");

        // Age our own lock, as if this pass had been running for an hour.
        std::fs::write(&path, format!("1\n{}\n{}\n", unix_now() - 3600, held.id)).unwrap();
        assert!(SeedLock::is_stale(&path), "the aged lock reads as abandoned");

        held.touch();
        assert!(!SeedLock::is_stale(&path), "a seed still working does not");
        assert!(
            SeedLock::acquire(&path).unwrap().is_none(),
            "and nobody takes it away"
        );
    }

    #[tokio::test]
    async fn a_seed_while_another_instance_holds_the_lock_places_nothing() {
        let (dir, record) = scratch("locked");
        let held = SeedLock::acquire(&lock_path_for(&record))
            .unwrap()
            .expect("held");

        let report = seed_with_record(&dir, &record).await.unwrap();
        assert!(
            report.iter().any(|line| line.contains("another instance")),
            "{report:?}"
        );
        assert!(
            crate::pack::installed(&dir).unwrap().is_empty(),
            "nothing is placed while somebody else is seeding: {report:?}"
        );
        assert!(!record.exists(), "and nothing is recorded");
        drop(held);
    }
}
