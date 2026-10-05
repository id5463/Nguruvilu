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
//!   disk still hashes to exactly what *we* put there. A hand-edited pack is
//!   left alone, silently and permanently.
//! * **Placement is idempotent.** Called on every start, costs a directory
//!   listing and some hashing, and does nothing at all on the steady state.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

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
    /// ours to replace.
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
    let mut record = match std::fs::read_to_string(record_path) {
        Ok(text) => serde_json::from_str(&text)
            .with_context(|| format!("parsing {}", record_path.display()))
            .unwrap_or_default(),
        Err(_) => Record::default(),
    };
    let mut placed = Vec::new();

    for (name, version, bytes) in SHIPPED {
        let Some(pack) = crate::pack::installed(dir)?
            .into_iter()
            .find(|pack| pack.manifest.name == *name)
        else {
            // Not installed. Once it has been and is gone again, that was a
            // decision: the record is what keeps us from placing it back.
            if record.packs.contains_key(*name) {
                continue;
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
            placed.push(format!("placed {name}@{version}"));
            continue;
        };

        // Installed already: unloaded means the user said no.
        if !pack.enabled {
            continue;
        }
        // First sighting of a pack we did not place: adopt it — record what is
        // there, so an edit from here on is recognisably not ours. Adoption
        // then falls straight through to the same upgrade check everything else
        // takes, because "we have never seen this" and "this is ours, older"
        // deserve the same question: is it untouched, and do we ship something
        // newer?
        if !record.packs.contains_key(*name) {
            record.packs.insert(
                name.to_string(),
                Seeded {
                    version: pack.manifest.version_id.clone(),
                    files: hashes_of(&pack.path)?,
                    shipped: None,
                },
            );
        }
        let (was_shipped, recorded_version, recorded_files) = {
            let entry = &record.packs[*name];
            (
                entry.shipped.clone(),
                entry.version.clone(),
                entry.files.clone(),
            )
        };
        // Two reasons to replace what is on disk: this build placed it and is
        // now shipping a different archive, or the pack was adopted and this
        // build ships a different *version* of it. Both are worthless without
        // the next test — an edited pack stays, whatever we ship.
        let upgrade = match &was_shipped {
            Some(archive) => archive != &sha256(bytes),
            None => recorded_version != *version,
        };
        if !upgrade {
            continue;
        }
        if hashes_of(&pack.path)? != recorded_files {
            continue;
        }
        std::fs::remove_dir_all(&pack.path)
            .with_context(|| format!("replacing {}", pack.path.display()))?;
        let path = place(dir, name, bytes).await?;
        record.packs.insert(
            name.to_string(),
            Seeded {
                version: version.to_string(),
                files: hashes_of(&path)?,
                shipped: Some(sha256(bytes)),
            },
        );
        placed.push(format!("placed {name}@{version} (updated)"));
    }

    // Retire the packs `starter` merged into: a twin soul or a second
    // `builtin:search` loading beside starter's is a conflict nobody asked
    // for. Only our own untouched bytes go (recorded, still matching, and
    // `shipped`); everything else stays and says why.
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
            if entry.files != hashes_of(&pack.path)? {
                placed.push(format!(
                    "{name}@{version} kept: you have edits; it may now duplicate starter's copy"
                ));
                continue;
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

    if !placed.is_empty() {
        if let Some(parent) = record_path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating {}", parent.display()))?;
        }
        std::fs::write(record_path, serde_json::to_string_pretty(&record)?)
            .with_context(|| format!("writing {}", record_path.display()))?;
    }
    Ok(placed)
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
            again.is_empty(),
            "an edited pack is left alone: {again:?}"
        );
        assert_eq!(
            std::fs::read_to_string(&soul).unwrap(),
            "You are someone else entirely.\n"
        );
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
}
