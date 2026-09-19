//! Pack archives: `.dshpack`.
//!
//! A pack is a zip archive holding an identity manifest, an assembly manifest,
//! and whatever the assembly references (skills, plugin payloads, patch
//! files). It is the distribution unit of the loading layer.
//!
//! The archive is deliberately uncompressed. Packs are small text trees, the
//! components inside them are already compressed when that matters, and a
//! plain tar keeps the format inspectable with ordinary tools.
//!
//! Identity is separate from assembly on purpose: `dsh.index.json` says *what
//! this pack is* (name, version, license, what kernel it needs), while
//! `assembly.yaml` says *what it loads and in what order*. A pack can be
//! inspected for provenance without executing any of its load logic.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};

/// Current pack format.
pub const PACK_FORMAT_VERSION: u32 = 1;

/// Marker identifying the archive as a pack.
pub const PACK_MAGIC: &str = "dsh";

/// The identity manifest's filename.
pub const MANIFEST_NAME: &str = "dsh.index.json";

/// The assembly manifest's filename.
pub const ASSEMBLY_NAME: &str = "assembly.yaml";

/// Identity of one pack.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PackManifest {
    /// Format version.
    pub format_version: u32,
    /// Marker field, always `dsh`.
    pub game: String,
    /// Pack name.
    pub name: String,
    /// Version of this pack's contents.
    pub version_id: String,
    /// License the pack is distributed under.
    pub license: String,
    /// Kernel version the pack was built against.
    pub kernel_version: String,
    /// What the pack requires of the kernel.
    #[serde(default)]
    pub dependencies: Dependencies,
    /// One-line description.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
}

/// Version requirements a pack declares.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Dependencies {
    /// Kernel version range, e.g. `>=0.1.0 <1.0.0`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kernel: Option<String>,
}

impl PackManifest {
    /// Build a manifest for a pack being created.
    pub fn new(name: impl Into<String>, version_id: impl Into<String>) -> Self {
        Self {
            format_version: PACK_FORMAT_VERSION,
            game: PACK_MAGIC.to_string(),
            name: name.into(),
            version_id: version_id.into(),
            license: "MIT".to_string(),
            kernel_version: env!("CARGO_PKG_VERSION").to_string(),
            dependencies: Dependencies {
                kernel: Some(format!(">={}", env!("CARGO_PKG_VERSION"))),
            },
            summary: None,
        }
    }

    /// Reject a manifest that could never be loaded.
    pub fn validate(&self) -> Result<()> {
        if self.format_version != PACK_FORMAT_VERSION {
            return Err(anyhow!(
                "pack format version {} is not supported (expected {PACK_FORMAT_VERSION})",
                self.format_version
            ));
        }
        if self.game != PACK_MAGIC {
            return Err(anyhow!(
                "pack marker is {:?}, expected {PACK_MAGIC:?}",
                self.game
            ));
        }
        for (field, value) in [
            ("name", &self.name),
            ("version_id", &self.version_id),
            ("license", &self.license),
            ("kernel_version", &self.kernel_version),
        ] {
            if value.trim().is_empty() {
                return Err(anyhow!("pack manifest is missing a value for '{field}'"));
            }
        }
        Ok(())
    }

    /// Whether this pack declares a license for its contents.
    ///
    /// A pack without a license cannot be redistributed, so it is worth
    /// surfacing rather than assuming.
    pub fn has_license(&self) -> bool {
        !self.license.trim().is_empty()
    }
}

/// What a pack directory contains.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PackContents {
    /// Files relative to the pack root, sorted.
    pub files: Vec<String>,
    /// Whether an assembly manifest is present.
    pub has_assembly: bool,
}

/// Result of verifying an archive.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifyReport {
    /// The manifest.
    pub manifest: PackManifest,
    /// Contents found.
    pub contents: PackContents,
    /// Problems that do not prevent loading but should be reported.
    pub warnings: Vec<String>,
}

/// Read a manifest from a pack directory.
pub fn read_manifest(dir: &Path) -> Result<PackManifest> {
    let path = dir.join(MANIFEST_NAME);
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("reading {}", path.display()))?;
    let manifest: PackManifest = serde_json::from_str(&text)
        .with_context(|| format!("parsing {}", path.display()))?;
    manifest.validate()?;
    Ok(manifest)
}

/// Write a manifest into a pack directory.
pub fn write_manifest(dir: &Path, manifest: &PackManifest) -> Result<()> {
    manifest.validate()?;
    let path = dir.join(MANIFEST_NAME);
    let text = serde_json::to_string_pretty(manifest)?;
    std::fs::write(&path, format!("{text}\n"))
        .with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

/// List what a pack directory holds.
pub fn inspect(dir: &Path) -> Result<PackContents> {
    let mut files = Vec::new();
    collect_files(dir, dir, &mut files)?;
    files.sort();
    Ok(PackContents {
        has_assembly: dir.join(ASSEMBLY_NAME).is_file(),
        files,
    })
}

fn collect_files(root: &Path, dir: &Path, out: &mut Vec<String>) -> Result<()> {
    for entry in std::fs::read_dir(dir).with_context(|| format!("reading {}", dir.display()))? {
        let entry = entry?;
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().to_string();
        // Never ship a repository's internals inside a pack.
        if name == ".git" || name == "target" || name == "node_modules" {
            continue;
        }
        if path.is_dir() {
            collect_files(root, &path, out)?;
        } else if let Ok(relative) = path.strip_prefix(root) {
            out.push(relative.to_string_lossy().replace('\\', "/"));
        }
    }
    Ok(())
}

/// Create a `.dshpack` from a directory.
///
/// The directory must carry a manifest; an archive without one cannot be
/// identified, and a nameless artifact is worse than no artifact.
pub fn pack(dir: &Path, out: &Path) -> Result<PackManifest> {
    let manifest = read_manifest(dir)?;

    if let Some(parent) = out.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating {}", parent.display()))?;
        }
    }

    let file = std::fs::File::create(out)
        .with_context(|| format!("creating {}", out.display()))?;
    let mut writer = zip::ZipWriter::new(file);

    // `inspect` returns a sorted list, so the archive is byte-comparable
    // between builds of the same tree.
    let contents = inspect(dir)?;
    for relative in &contents.files {
        let path = dir.join(relative);
        let metadata = std::fs::metadata(&path)
            .with_context(|| format!("reading metadata for {relative}"))?;

        // mut is only needed on Unix, where the mode is attached below.
        #[allow(unused_mut)]
        let mut options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated)
            .large_file(metadata.len() > u32::MAX as u64);

        // Preserve the mode: a pack may ship a script, and silently dropping
        // the executable bit would break it on unpack.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            options = options.unix_permissions(metadata.permissions().mode());
        }

        writer
            .start_file(relative, options)
            .with_context(|| format!("adding {relative} to the archive"))?;
        let data = std::fs::read(&path).with_context(|| format!("reading {relative}"))?;
        writer
            .write_all(&data)
            .with_context(|| format!("writing {relative} into the archive"))?;
    }

    writer.finish().context("finalizing the archive")?;
    Ok(manifest)
}

/// Unpack a `.dshpack` into a directory, returning its manifest.
pub fn unpack(pack_path: &Path, dest: &Path) -> Result<PackManifest> {
    std::fs::create_dir_all(dest)
        .with_context(|| format!("creating {}", dest.display()))?;

    let file = std::fs::File::open(pack_path)
        .with_context(|| format!("opening {}", pack_path.display()))?;
    let mut archive = zip::ZipArchive::new(file)
        .with_context(|| format!("reading the archive {}", pack_path.display()))?;

    // Extract entry by entry rather than calling `extract()` wholesale: an
    // entry whose path escapes the destination must be refused, not written.
    for index in 0..archive.len() {
        let mut entry = archive.by_index(index)?;
        let Some(relative) = entry.enclosed_name() else {
            return Err(anyhow!(
                "archive entry {:?} escapes the destination directory",
                entry.name()
            ));
        };
        let target = dest.join(relative);

        if entry.is_dir() {
            std::fs::create_dir_all(&target)
                .with_context(|| format!("creating {}", target.display()))?;
            continue;
        }
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating {}", parent.display()))?;
        }

        let mut out = std::fs::File::create(&target)
            .with_context(|| format!("creating {}", target.display()))?;
        std::io::copy(&mut entry, &mut out)
            .with_context(|| format!("writing {}", target.display()))?;
    }

    read_manifest(dest)
}

/// Verify an archive without keeping its contents.
pub fn verify(pack_path: &Path) -> Result<VerifyReport> {
    let temp = std::env::temp_dir().join(format!(
        "ngu-verify-{}",
        uuid::Uuid::new_v4().simple()
    ));
    std::fs::create_dir_all(&temp)?;

    let result = (|| -> Result<VerifyReport> {
        let manifest = unpack(pack_path, &temp)?;
        let contents = inspect(&temp)?;

        let mut warnings = Vec::new();
        if !contents.has_assembly {
            warnings.push(format!(
                "no {ASSEMBLY_NAME}: the pack identifies itself but loads nothing"
            ));
        }
        if !manifest.has_license() {
            warnings.push("no license: the pack cannot be redistributed".to_string());
        }

        Ok(VerifyReport { manifest, contents, warnings })
    })();

    let _ = std::fs::remove_dir_all(&temp);
    result
}

/// A pack installed on disk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstalledPack {
    /// Manifest.
    pub manifest: PackManifest,
    /// Where it was unpacked.
    pub path: PathBuf,
    /// Path to its assembly manifest, when it has one.
    pub assembly: Option<PathBuf>,
}

/// Install an archive into a packs directory.
///
/// The destination is `<packs_dir>/<name>-<version>`, so two versions of the
/// same pack coexist instead of overwriting each other.
pub fn install(pack_path: &Path, packs_dir: &Path) -> Result<InstalledPack> {
    let staging = packs_dir.join(format!(".staging-{}", uuid::Uuid::new_v4().simple()));
    std::fs::create_dir_all(&staging)
        .with_context(|| format!("creating {}", staging.display()))?;

    let manifest = match unpack(pack_path, &staging) {
        Ok(manifest) => manifest,
        Err(error) => {
            let _ = std::fs::remove_dir_all(&staging);
            return Err(error);
        }
    };

    let slug = format!("{}-{}", sanitize(&manifest.name), sanitize(&manifest.version_id));
    let target = packs_dir.join(&slug);
    if target.exists() {
        std::fs::remove_dir_all(&target)
            .with_context(|| format!("replacing {}", target.display()))?;
    }
    std::fs::rename(&staging, &target)
        .with_context(|| format!("moving the pack into {}", target.display()))?;

    let assembly = target.join(ASSEMBLY_NAME);
    Ok(InstalledPack {
        manifest,
        assembly: assembly.is_file().then_some(assembly),
        path: target,
    })
}

/// List installed packs under a directory.
pub fn installed(packs_dir: &Path) -> Result<Vec<InstalledPack>> {
    if !packs_dir.exists() {
        return Ok(Vec::new());
    }

    let mut packs = Vec::new();
    for entry in std::fs::read_dir(packs_dir)
        .with_context(|| format!("reading {}", packs_dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with('.') {
            continue; // staging leftovers
        }
        let Ok(manifest) = read_manifest(&path) else {
            continue;
        };
        let assembly = path.join(ASSEMBLY_NAME);
        packs.push(InstalledPack {
            manifest,
            assembly: assembly.is_file().then_some(assembly),
            path,
        });
    }
    packs.sort_by(|a, b| a.manifest.name.cmp(&b.manifest.name));
    Ok(packs)
}

/// Make a name safe for a directory.
fn sanitize(value: &str) -> String {
    value
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' { c } else { '-' })
        .collect()
}

/// The default packs directory: `$NGU_HOME/packs`, else `<cwd>/.nguruvilu/packs`.
pub fn default_packs_dir() -> PathBuf {
    if let Ok(home) = std::env::var("NGU_HOME") {
        if !home.trim().is_empty() {
            return PathBuf::from(home).join("packs");
        }
    }
    // The user's home, not the current directory: packs are installed for the
    // user, and the directory the command happened to run in is not a place to
    // leave state.
    crate::settings::home_dir().join(".nguruvilu").join("packs")
}

/// Read the pack's assembly manifest, when present.
pub fn read_assembly(dir: &Path) -> Result<Option<crate::assembly::Assembly>> {
    let path = dir.join(ASSEMBLY_NAME);
    if !path.is_file() {
        return Ok(None);
    }
    Ok(Some(crate::assembly::Assembly::from_file(&path)?))
}

/// A summary of what a pack would load, for display.
pub fn describe(manifest: &PackManifest) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    out.insert("name".into(), manifest.name.clone());
    out.insert("version".into(), manifest.version_id.clone());
    out.insert("license".into(), manifest.license.clone());
    out.insert("kernel".into(), manifest.kernel_version.clone());
    if let Some(range) = &manifest.dependencies.kernel {
        out.insert("requires".into(), range.clone());
    }
    if let Some(summary) = &manifest.summary {
        out.insert("summary".into(), summary.clone());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("ngu-pack-{tag}-{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_pack_dir(dir: &Path) {
        write_manifest(dir, &PackManifest::new("demo-pack", "1.0.0")).unwrap();
        std::fs::write(
            dir.join(ASSEMBLY_NAME),
            "version: 1\nstages:\n  - name: foundation\n    skills:\n      - id: s\n        source: ./skills\n",
        )
        .unwrap();
        std::fs::create_dir_all(dir.join("skills/pdf-tools")).unwrap();
        std::fs::write(
            dir.join("skills/pdf-tools/SKILL.md"),
            "---\nname: pdf-tools\ndescription: Read PDFs\n---\n\nBody.\n",
        )
        .unwrap();
    }

    #[test]
    fn a_manifest_round_trips() {
        let dir = temp_dir("manifest");
        let manifest = PackManifest::new("my-pack", "2.1.0");
        write_manifest(&dir, &manifest).unwrap();

        let read = read_manifest(&dir).unwrap();
        assert_eq!(read, manifest);
        assert_eq!(read.game, "dsh");
        assert!(read.has_license());
        assert_eq!(read.dependencies.kernel.is_some(), true);
    }

    #[test]
    fn a_missing_manifest_is_an_error() {
        let dir = temp_dir("no-manifest");
        let error = read_manifest(&dir).expect_err("no manifest");
        assert!(format!("{error:#}").contains(MANIFEST_NAME));
    }

    #[test]
    fn an_invalid_manifest_is_rejected() {
        let dir = temp_dir("bad-manifest");
        std::fs::write(
            dir.join(MANIFEST_NAME),
            r#"{"format_version": 99, "game": "dsh", "name": "x", "version_id": "1", "license": "MIT", "kernel_version": "0.1.0"}"#,
        )
        .unwrap();
        let error = read_manifest(&dir).expect_err("bad version");
        assert!(format!("{error:#}").contains("not supported"));
    }

    #[test]
    fn a_manifest_missing_a_required_field_is_rejected() {
        let dir = temp_dir("missing-field");
        std::fs::write(
            dir.join(MANIFEST_NAME),
            r#"{"format_version": 1, "game": "dsh", "name": "", "version_id": "1", "license": "MIT", "kernel_version": "0.1.0"}"#,
        )
        .unwrap();
        let error = read_manifest(&dir).expect_err("empty name");
        assert!(format!("{error:#}").contains("name"));
    }

    #[test]
    fn packing_and_unpacking_preserves_the_tree() {
        let src = temp_dir("pack-src");
        write_pack_dir(&src);
        let archive = temp_dir("pack-out").join("demo.dshpack");

        let manifest = pack(&src, &archive).unwrap();
        assert_eq!(manifest.name, "demo-pack");
        assert!(archive.is_file());

        let dest = temp_dir("pack-dest");
        let unpacked = unpack(&archive, &dest).unwrap();
        assert_eq!(unpacked, manifest);

        // The manifest, the assembly, and the skill all came through.
        assert!(dest.join(MANIFEST_NAME).is_file());
        assert!(dest.join(ASSEMBLY_NAME).is_file());
        assert!(dest.join("skills/pdf-tools/SKILL.md").is_file());

        // And the assembly still parses on the other side.
        let assembly = read_assembly(&dest).unwrap().expect("assembly present");
        assert_eq!(assembly.stages.len(), 1);
    }

    #[test]
    fn packing_without_a_manifest_fails() {
        let src = temp_dir("pack-nomanifest");
        std::fs::write(src.join("stray.txt"), "x").unwrap();
        let error = pack(&src, &temp_dir("pack-out2").join("x.dshpack")).expect_err("no manifest");
        assert!(format!("{error:#}").contains(MANIFEST_NAME));
    }

    #[test]
    fn build_output_is_not_shipped_inside_a_pack() {
        let src = temp_dir("pack-ignore");
        write_pack_dir(&src);
        // These directories must never end up in an archive.
        for junk in ["target", "node_modules", ".git"] {
            std::fs::create_dir_all(src.join(junk)).unwrap();
            std::fs::write(src.join(junk).join("big.bin"), "x").unwrap();
        }

        let archive = temp_dir("pack-out3").join("demo.dshpack");
        pack(&src, &archive).unwrap();

        let dest = temp_dir("pack-dest3");
        unpack(&archive, &dest).unwrap();
        assert!(!dest.join("target").exists());
        assert!(!dest.join("node_modules").exists());
        assert!(!dest.join(".git").exists());
        assert!(dest.join(ASSEMBLY_NAME).exists());
    }

    #[test]
    fn verify_reports_missing_pieces_as_warnings() {
        let src = temp_dir("verify-warn");
        // A manifest with no assembly: identifies itself, loads nothing.
        write_manifest(&src, &PackManifest::new("bare", "1.0.0")).unwrap();
        let archive = temp_dir("verify-out").join("bare.dshpack");
        pack(&src, &archive).unwrap();

        let report = verify(&archive).unwrap();
        assert_eq!(report.manifest.name, "bare");
        assert!(!report.contents.has_assembly);
        assert!(report
            .warnings
            .iter()
            .any(|w| w.contains(ASSEMBLY_NAME)));
    }

    #[test]
    fn verify_is_clean_for_a_complete_pack() {
        let src = temp_dir("verify-ok");
        write_pack_dir(&src);
        let archive = temp_dir("verify-out2").join("ok.dshpack");
        pack(&src, &archive).unwrap();

        let report = verify(&archive).unwrap();
        assert!(report.contents.has_assembly);
        assert!(report.warnings.is_empty(), "{:?}", report.warnings);
    }

    #[test]
    fn installing_places_the_pack_under_a_versioned_directory() {
        let src = temp_dir("install-src");
        write_pack_dir(&src);
        let archive = temp_dir("install-out").join("demo.dshpack");
        pack(&src, &archive).unwrap();

        let packs_dir = temp_dir("install-packs");
        let placed = install(&archive, &packs_dir).unwrap();

        assert_eq!(placed.manifest.name, "demo-pack");
        assert!(placed.path.ends_with("demo-pack-1.0.0"), "{:?}", placed.path);
        assert!(placed.assembly.is_some());

        let listed = installed(&packs_dir).unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].manifest.version_id, "1.0.0");
    }

    #[test]
    fn installing_twice_replaces_rather_than_failing() {
        let src = temp_dir("install-twice");
        write_pack_dir(&src);
        let archive = temp_dir("install-twice-out").join("demo.dshpack");
        pack(&src, &archive).unwrap();

        let packs_dir = temp_dir("install-twice-packs");
        install(&archive, &packs_dir).unwrap();
        install(&archive, &packs_dir).unwrap();

        assert_eq!(installed(&packs_dir).unwrap().len(), 1);
        // No staging leftovers.
        let leftovers: Vec<String> = std::fs::read_dir(&packs_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().to_string())
            .filter(|n| n.starts_with('.'))
            .collect();
        assert!(leftovers.is_empty(), "{leftovers:?}");
    }

    #[test]
    fn two_versions_of_one_pack_coexist() {
        let packs_dir = temp_dir("install-versions");

        for version in ["1.0.0", "2.0.0"] {
            let src = temp_dir("install-version-src");
            write_manifest(&src, &PackManifest::new("multi", version)).unwrap();
            let archive = temp_dir("install-version-out").join("multi.dshpack");
            pack(&src, &archive).unwrap();
            install(&archive, &packs_dir).unwrap();
        }

        let listed = installed(&packs_dir).unwrap();
        assert_eq!(listed.len(), 2, "{listed:?}");
    }

    #[test]
    fn a_corrupt_archive_fails_loudly() {
        let archive = temp_dir("corrupt").join("bad.dshpack");
        std::fs::write(&archive, b"this is not a tar archive").unwrap();
        assert!(verify(&archive).is_err());
    }

    #[test]
    fn describing_a_manifest_yields_its_identity() {
        let mut manifest = PackManifest::new("demo", "1.2.3");
        manifest.summary = Some("a demo".into());
        let described = describe(&manifest);
        assert_eq!(described.get("name").unwrap(), "demo");
        assert_eq!(described.get("version").unwrap(), "1.2.3");
        assert_eq!(described.get("summary").unwrap(), "a demo");
    }

    #[test]
    fn sanitizing_keeps_names_path_safe() {
        assert_eq!(sanitize("my pack/1.0"), "my-pack-1.0");
        assert_eq!(sanitize("ok-name_1.0"), "ok-name_1.0");
    }
}
