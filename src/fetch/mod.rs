//! Fetching what a pack references.
//!
//! A pack carries references, not payloads: `github:owner/repo@path@ref`, a
//! direct URL, or a name the kernel already provides. This module turns a
//! reference into a directory on disk, verifying it against the hash the pack
//! declared.
//!
//! # Why the hash is sha256 and not sha1
//!
//! The hash answers one question: are these the bytes the pack named? sha1
//! collisions are constructible, so a sha1 check can be satisfied by content
//! the pack did not name. That makes it decoration rather than verification.
//!
//! # Why not shell out to `gh`
//!
//! The reference implementation of this format calls the `gh` CLI. That would
//! make every pack install depend on a tool the user may not have, in a program
//! whose point is being one self-contained executable. GitHub's REST API and
//! `raw.githubusercontent.com` are reachable with the HTTP client the kernel
//! already links.
//!
//! # Caching
//!
//! Downloads land in `~/.nguruvilu/cache/<sha256>/`, addressed by content. Two
//! packs naming the same skill share one copy, and a second install of the same
//! pack does no network work at all — which is what makes an install work with
//! no connection once the cache is warm.

//! # Where the pieces live
//!
//! `source` is the vocabulary — [`Source`], [`SchemeHandler`], [`Fetcher`] and
//! the cache entry points; `http` is the download path with its redirect and
//! retry rules; `unpack` is the archive reader with its ceilings; `hash` is
//! content addressing and the sizes reported with it.

mod hash;
mod http;
mod source;
mod unpack;

pub use self::source::{
    platform_tag, DirectoryScheme, Fetched, Fetcher, SchemeFuture, SchemeHandler, Source,
    KNOWN_SCHEMES, PLATFORMS,
};
pub use self::unpack::{extract_archive, extract_archive_with, is_archive_name, UnpackLimits};
