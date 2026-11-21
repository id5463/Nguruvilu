//! Packs: `.ngupack`, the distribution format.
//!
//! A pack is a zip holding a **manifest of references** plus the small data
//! files that cannot be fetched from anywhere — persona, model route, context
//! policy, MCP servers, appearance, injection rules.
//!
//! ```text
//! my-pack-1.0.0.ngupack
//! ├── ngu.index.json     the manifest: what to fetch, and what is inside
//! ├── soul.md            persona
//! ├── models.json        how to talk to the model
//! ├── context.json       how much to remember
//! ├── mcp.json           MCP servers
//! ├── look.json          themes and panels
//! └── injections.json    injection rules
//! ```
//!
//! The payload — skills and plugins — is **not** in the archive. It is named by
//! `github:` or `https:` reference with a sha256, and fetched on install into a
//! content-addressed cache. Two packs naming the same skill share one download.
//!
//! # Two shapes
//!
//! | | manifest pack | offline pack |
//! |---|---|---|
//! | size | kilobytes | megabytes per platform |
//! | install | fetches what it names | needs no network |
//! | platforms | one manifest covers all | one archive per platform |
//!
//! The offline shape is the same zip with the payload embedded under `files/`,
//! which is what `--offline` produces. It exists for air-gapped and archived
//! installs; the manifest shape is the normal one.
//!
//! # Install materialises a loadable directory
//!
//! The loader reads `assembly.yaml`. Install therefore writes one, generated
//! from the manifest and the content files, so the loader needs to know nothing
//! about the distribution format. The two formats change for different reasons
//! and are allowed to change separately.
//!
//! # This module is a directory
//!
//! `pack` is split by job. [`manifest`] holds the manifest and the data types it
//! names; this file keeps the archive format, install, and the functions over a
//! pack directory. The manifest's public items are re-exported below, so
//! `crate::pack::PackManifest` resolves exactly as it did when the two were one
//! file, and `pub mod manifest` lets a caller name the data layer on its own.

pub mod manifest;

pub use manifest::*;

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use serde::Serialize;

use crate::fetch::{Fetcher, Source};

/// Manifest file name inside the archive.
///
/// The name carries the project, deliberately. A format file is the one place a
/// name travels to every pack on every machine, so a name borrowed from another
/// project is a name that makes people ask which project they are running —
/// which is exactly what happened: this file was `dsh.index.json`, then a
/// half-finished rename left `ds.index.json`, and neither says Nguruvilu.
///
/// [`LEGACY_MANIFEST_NAME`] is still read so an installed directory keeps
/// loading, and [`migrate_manifest_name`] rewrites it on sight.
pub const MANIFEST_NAME: &str = "ngu.index.json";

/// Manifest names packs written before this one carry, newest first.
///
/// Read wherever [`MANIFEST_NAME`] is read, and reported by `pack` so a user can
/// see which spelling a directory uses. Never written: a pack built now is
/// `ngu.index.json`, and [`migrate_manifest_name`] rewrites an old one in place.
pub const LEGACY_MANIFEST_NAMES: &[&str] = &["ds.index.json", "dsh.index.json"];

/// The oldest spelling, kept as its own constant because it is the one a large
/// number of packs already on disk wear.
pub const LEGACY_MANIFEST_NAME: &str = "dsh.index.json";

/// Archive extension this build writes.
pub const ARCHIVE_EXT: &str = "ngupack";

/// Archive extension packs written before the rename carry.
pub const LEGACY_ARCHIVE_EXT: &str = "dshpack";

/// Every archive extension [`crate::pack::open`] accepts, preferred first.
pub const ARCHIVE_EXTS: &[&str] = &[ARCHIVE_EXT, LEGACY_ARCHIVE_EXT];

/// Assembly file name the loader reads.
pub const ASSEMBLY_NAME: &str = "assembly.yaml";

/// Format version this build writes and accepts.
pub const FORMAT_VERSION: u32 = 1;

/// The `game` field's expected value.
pub const GAME: &str = "nguruvilu";

/// The value earlier packs wrote. Accepted so an existing offline pack keeps
/// loading; new packs use [`GAME`].
pub const LEGACY_GAME: &str = "dsh";

/// Directory holding embedded payload in an offline pack.
pub const FILES_DIR: &str = "files";

/// Field name to file name, for the content a pack carries.
///
/// The order is the order they are written and reported, so diagnostics read
/// the same way twice.
pub const CONTENT_FILES: &[(&str, &str)] = &[
    ("soul", "soul.md"),
    ("models", "models.json"),
    ("context", "context.json"),
    ("mcp", "mcp.json"),
    ("look", "look.json"),
    ("injections", "injections.json"),
    ("search", "search.json"),
];

// ---------------------------------------------------------------- archive

/// What a pack directory holds.
#[derive(Debug, Clone, Default)]
pub struct PackContents {
    /// Files, relative to the pack root, sorted.
    pub files: Vec<String>,
    /// Whether a manifest is present.
    pub has_manifest: bool,
    /// Whether an assembly is present (offline packs carry one).
    pub has_assembly: bool,
}

/// Result of checking an archive without installing it.
#[derive(Debug, Clone)]
pub struct VerifyReport {
    /// The manifest that was read.
    pub manifest: PackManifest,
    /// What the archive contains.
    pub contents: PackContents,
    /// Whether the declared kernel range accepts this kernel.
    pub kernel_compatible: bool,
    /// Problems found, in the order they were discovered.
    pub warnings: Vec<String>,
}

/// An installed pack, as it sits in the packs directory.
#[derive(Debug, Clone)]
pub struct InstalledPack {
    /// Its manifest.
    pub manifest: PackManifest,
    /// Where it is installed.
    pub path: PathBuf,
    /// Its assembly, when it has one.
    pub assembly: Option<PathBuf>,
    /// Whether it is loaded into conversations.
    ///
    /// Unloading flips this off and keeps the files; deleting removes the
    /// directory. The two are different acts, so a pack that is unloaded but
    /// still on disk is a normal state rather than an error.
    pub enabled: bool,
}

/// Marker file inside an installed pack meaning "do not load this".
///
/// A file rather than a field in the manifest: the manifest is the pack
/// author's declaration, and this is the *user's* decision about it. Keeping
/// them apart is what lets a user unload a pack without editing — or
/// invalidating — what the pack says about itself.
pub const DISABLED_MARKER: &str = ".ngu-disabled";

/// Whether an installed pack directory is enabled.

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
