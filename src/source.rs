//! Where packs come from, and where they go.
//!
//! Installing a pack is a filesystem operation today: unpack a zip under
//! `~/.nguruvilu/packs/<name>-<version>/`. That is one possible source, not the
//! only sensible one — a pack could come from a URL, a git checkout, a shared
//! directory, or a database, and different deployments will want different
//! answers.
//!
//! [`PackSource`] is the seam. The kernel ships [`FilesystemSource`] and uses it
//! by default; a plugin providing the `pack.source` service replaces it, and
//! every path that installs, lists, or resolves a pack goes through whichever
//! one is installed.
//!
//! The trait is deliberately small. Anything that only *reads* a pack — verify,
//! inspect, plan — works from a directory and needs no source at all, so those
//! are left out.

use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;

use anyhow::{Context, Result};

use crate::pack::{InstalledPack, PackManifest};

/// Boxed future returned by [PackSource::install].
pub type InstallFuture<'a> =
    Pin<Box<dyn Future<Output = Result<InstalledPack>> + Send + 'a>>;

/// Where packs are found and installed.
pub trait PackSource: Send + Sync {
    /// A name for diagnostics.
    fn name(&self) -> &str;

    /// The directory packs live in, when the source has one.
    fn root(&self) -> Option<PathBuf> {
        None
    }

    /// Every installed pack.
    fn list(&self) -> Result<Vec<InstalledPack>>;

    /// Install an archive, replacing any existing copy of the same version.
    fn install<'a>(&'a self, archive: &'a Path) -> InstallFuture<'a>;

    /// Locate an installed pack by name, newest version first when several exist.
    fn find(&self, name: &str) -> Result<Option<InstalledPack>>;

    /// Remove an installed pack by its directory.
    fn remove(&self, path: &Path) -> Result<()>;
}

/// The default source: a directory of unpacked archives.
#[derive(Debug, Clone)]
pub struct FilesystemSource {
    root: PathBuf,
}

impl FilesystemSource {
    /// Use a specific directory.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// Use `$NGU_HOME/packs`, else `~/.nguruvilu/packs`.
    pub fn default_root() -> Self {
        Self::new(crate::pack::default_packs_dir())
    }
}

impl PackSource for FilesystemSource {
    fn name(&self) -> &str {
        "filesystem"
    }

    fn root(&self) -> Option<PathBuf> {
        Some(self.root.clone())
    }

    fn list(&self) -> Result<Vec<InstalledPack>> {
        crate::pack::installed(&self.root)
    }

    fn install<'a>(&'a self, archive: &'a Path) -> InstallFuture<'a> {
        Box::pin(async move { crate::pack::install(archive, &self.root).await })
    }

    fn find(&self, name: &str) -> Result<Option<InstalledPack>> {
        let mut matches: Vec<InstalledPack> = self
            .list()?
            .into_iter()
            .filter(|pack| pack.manifest.name == name)
            .collect();

        // Newest first. Versions are compared as strings, which is wrong for
        // arbitrary schemes and right for the one this format uses: a
        // `<major>.<minor>.<patch>` string sorts correctly when the parts are
        // zero-padded, and a pack that does not sort correctly is still found
        // by its directory name.
        matches.sort_by(|a, b| b.manifest.version_id.cmp(&a.manifest.version_id));
        Ok(matches.into_iter().next())
    }

    fn remove(&self, path: &Path) -> Result<()> {
        // Refuse anything outside the root: a caller passing an arbitrary path
        // must not turn this into a delete-anything primitive.
        let root = self
            .root
            .canonicalize()
            .with_context(|| format!("resolving {}", self.root.display()))?;
        let target = path
            .canonicalize()
            .with_context(|| format!("resolving {}", path.display()))?;
        if !target.starts_with(&root) {
            anyhow::bail!(
                "{} is outside the pack directory {}",
                target.display(),
                root.display()
            );
        }
        std::fs::remove_dir_all(&target)
            .with_context(|| format!("removing {}", target.display()))?;
        Ok(())
    }
}

/// Read a pack's manifest from wherever it lives.
///
/// Separate from [`PackSource`] because a pack directory is readable without
/// knowing how it was installed.
pub fn manifest_of(dir: &Path) -> Result<PackManifest> {
    crate::pack::read_manifest(dir)
}

