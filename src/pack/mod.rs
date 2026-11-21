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
//! so one code path reads both.
//!
//! # Layout
//!
//! * [`manifest`] — the data: what a manifest declares, file by file.
//! * [`format`] — the archive's names, versions and limits, plus the result
//!   types (`PackContents`, `VerifyReport`, `InstalledPack`).
//! * [`enabled`] — installed state: enabled or not, and manifest read/write.
//! * [`archive`] — a directory into a `.ngupack`, and an archive back out.
//! * [`verify`] — check an archive without installing it.
//! * [`install`] — unpack, fetch what it names, write the assembly.
//!
//! Every item this module exported before is re-exported here, so
//! `crate::pack::X` still names the same item.

pub mod archive;
pub mod enabled;
pub mod format;
pub mod install;
pub mod manifest;
pub mod verify;

pub use archive::{inspect, pack, pack_offline, packable_files, unpack};
pub use enabled::{is_enabled, manifest_path, migrate_manifest_name, read_manifest, set_enabled, write_manifest};
pub use format::{
    PackContents, VerifyReport, InstalledPack, ARCHIVE_EXT, ARCHIVE_EXTS, ASSEMBLY_NAME,
    CONTENT_FILES, DISABLED_MARKER, FILES_DIR, FORMAT_VERSION, GAME, LEGACY_ARCHIVE_EXT,
    LEGACY_GAME, LEGACY_MANIFEST_NAME, LEGACY_MANIFEST_NAMES, MANIFEST_NAME,
    MAX_ARCHIVE_ENTRIES, MAX_ENTRY_BYTES, MAX_UNPACKED_BYTES,
};
pub use install::{default_packs_dir, describe, ensure_assembly, install, installed, read_assembly};
pub use manifest::*;
pub use verify::verify;
