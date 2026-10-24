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

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Ceiling on one download, in bytes.
///
/// A reference in a pack is a skill, a tool binary, or an archive of either.
/// 256 MiB is far above anything the format is meant to carry and far below
/// what a runaway or hostile response would need to exhaust this process:
/// without a bound, one bad URL is an out-of-memory kill.
const MAX_DOWNLOAD_BYTES: u64 = 256 * 1024 * 1024;

/// How many times a connection-level failure is retried before giving up.
///
/// The same order as the model client's bound: enough to step over a pooled
/// connection the peer closed and a TLS session that arrived damaged, not
/// enough to hide an endpoint that cannot be reached at all.
const DOWNLOAD_RETRIES: u32 = 2;

/// The pause before the first retry, in milliseconds; each retry doubles it.
const RETRY_BACKOFF_MS: u64 = 250;

/// The default request timeout for a fetch, in seconds.
const DEFAULT_FETCH_TIMEOUT_SECS: u64 = 120;

/// The timeout a fetch uses: `NGU_FETCH_TIMEOUT_SECS`, else two minutes.
///
/// A download can reasonably take longer than a model request on a slow link,
/// and a mirror on the far side of the world can need longer still, so the
/// number is a setting rather than a constant baked into the client.
fn fetch_timeout() -> std::time::Duration {
    std::time::Duration::from_secs(fetch_timeout_secs(
        std::env::var("NGU_FETCH_TIMEOUT_SECS").ok().as_deref(),
    ))
}

/// The parsed override, separated from the environment so the rule is
/// testable without touching process state.
///
/// An unparsable value, or zero, falls back to the default rather than
/// producing a timeout that fires immediately: a typo in an environment
/// variable must not read as "every download fails".
fn fetch_timeout_secs(raw: Option<&str>) -> u64 {
    raw.and_then(|value| value.trim().parse::<u64>().ok())
        .filter(|secs| *secs > 0)
        .unwrap_or(DEFAULT_FETCH_TIMEOUT_SECS)
}

/// A platform tag, as it appears in a pack's `platforms` map.
pub fn platform_tag() -> &'static str {
    if cfg!(target_os = "windows") {
        if cfg!(target_arch = "aarch64") {
            "win-arm64"
        } else {
            "win-x64"
        }
    } else if cfg!(target_os = "macos") {
        if cfg!(target_arch = "aarch64") {
            "mac-arm64"
        } else {
            "mac-x64"
        }
    } else if cfg!(target_arch = "aarch64") {
        "linux-arm64"
    } else {
        "linux-x64"
    }
}

/// Schemes the kernel handles without help.
///
/// A source naming anything else is kept and resolved by a registered handler;
/// this list exists for diagnostics, not as a closed set.
pub const KNOWN_SCHEMES: &[&str] = &["github", "https", "http", "npm", "builtin", "dylib"];

/// Every platform tag the format knows.
pub const PLATFORMS: &[&str] = &[
    "win-x64",
    "win-arm64",
    "linux-x64",
    "linux-arm64",
    "mac-x64",
    "mac-arm64",
];

/// Where a referenced thing comes from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case", tag = "kind")]
pub enum Source {
    /// `github:owner/repo@path@ref` — a directory or file inside a repository.
    Github {
        /// Repository owner.
        owner: String,
        /// Repository name, without a trailing `.git`.
        repo: String,
        /// Path inside the repository; `.` means the whole tree.
        path: String,
        /// Branch, tag, or commit.
        #[serde(rename = "ref")]
        git_ref: String,
    },
    /// A direct URL to one file.
    Url {
        /// The URL.
        url: String,
    },
    /// `npm:package@version` — a package installed from the npm registry into
    /// the pack's own directory when the pack is installed.
    ///
    /// This is not fetched as a file: a tarball alone has no dependency tree,
    /// so `pack::install` runs `npm install --prefix` and points the server at
    /// the resulting `.bin` shim. The network moment is install, exactly like
    /// every other remote source; after that the pack runs offline.
    Npm {
        /// Package name, possibly scoped (`@scope/name`).
        package: String,
        /// Exact version, or `None` for whatever the registry calls latest.
        ///
        /// An unpinned version installs — but says so, the way an unpinned git
        /// ref does.
        version: Option<String>,
    },
    /// A name the kernel already registered; nothing to fetch.
    Builtin {
        /// Plugin name.
        name: String,
    },
    /// A path inside the pack itself.
    Local {
        /// Path relative to the pack directory.
        path: String,
    },
    /// A scheme the kernel does not handle itself.
    ///
    /// Resolved by a handler registered on the [`Fetcher`], so a deployment can
    /// serve packs from wherever it keeps them without this format knowing that
    /// place exists.
    Custom {
        /// The scheme, lowercased: `gitlab`, `s3`, `internal-registry`.
        scheme: String,
        /// Everything after the colon.
        rest: String,
    },
}

impl Source {
    /// Parse one of the reference forms.
    pub fn parse(raw: &str) -> Result<Self> {
        let raw = raw.trim();
        if let Some(rest) = raw.strip_prefix("github:") {
            return Self::parse_github(rest, raw);
        }
        if let Some(rest) = raw.strip_prefix("builtin:") {
            if rest.trim().is_empty() {
                return Err(anyhow!("'{raw}' names no plugin"));
            }
            return Ok(Source::Builtin {
                name: rest.trim().to_string(),
            });
        }
        if let Some(rest) = raw.strip_prefix("npm:") {
            return Self::parse_npm(rest, raw);
        }
        if let Some(rest) = raw.strip_prefix("dylib:") {
            return Ok(Source::Local {
                path: rest.to_string(),
            });
        }
        if raw.starts_with("https://") || raw.starts_with("http://") {
            return Ok(Source::Url {
                url: raw.to_string(),
            });
        }

        // Anything else shaped like `scheme:rest` is kept rather than refused.
        // Which schemes exist is not the format's business: a deployment may
        // serve packs from an internal registry, a bucket, or a git host that
        // is not GitHub, and the set of names cannot be enumerated in advance.
        // An unregistered scheme fails at fetch time, where the message can
        // list what *is* registered.
        if let Some((scheme, rest)) = raw.split_once(':') {
            if !scheme.is_empty()
                && scheme
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '-')
                && !rest.is_empty()
            {
                return Ok(Source::Custom {
                    scheme: scheme.to_ascii_lowercase(),
                    rest: rest.to_string(),
                });
            }
        }

        // A path inside the pack: `skills/pdf`, `./skills/pdf` — the offline
        // shape the format documents and what a pack uses to ship its own
        // skills. The scheme branch above already claimed every colon-bearing
        // string, so this cannot swallow a URL or a Windows absolute path, and
        // a bare word (`no-colon-here`) still fails: a word is not a path.
        if raw.starts_with('.') || raw.contains('/') || raw.contains('\\') {
            return Ok(Source::Local {
                path: raw.to_string(),
            });
        }

        Err(anyhow!(
            "'{raw}' is not a usable source; expected a scheme such as {}, \
             or a plain https:// URL",
            KNOWN_SCHEMES.join(", ")
        ))
    }

    /// `package@version`, where the version is optional and a scope may carry
    /// its own `@` (`@scope/name@1.2.3`).
    ///
    /// The version is split from the right: a package name can contain one `@`
    /// (the scope marker) but the *last* `@` before the end is always the
    /// version separator when there is one.
    fn parse_npm(rest: &str, raw: &str) -> Result<Self> {
        let rest = rest.trim();
        if rest.is_empty() {
            return Err(anyhow!("'{raw}' names no package"));
        }
        // Find the version separator: the last '@' that is not the leading
        // scope marker of the name itself.
        let (package, version) = match rest.rfind('@') {
            Some(index) if index > 0 => {
                let (name, version) = rest.split_at(index);
                (name, Some(version[1..].to_string()))
            }
            _ => (rest, None),
        };
        if package.is_empty() {
            return Err(anyhow!("'{raw}' names no package"));
        }
        if matches!(&version, Some(v) if v.trim().is_empty()) {
            return Err(anyhow!("'{raw}' has an empty version after '@'"));
        }
        Ok(Source::Npm {
            package: package.to_string(),
            version,
        })
    }

    /// `owner/repo@path@ref`, where `@path` and `@ref` are both optional.
    ///
    /// The trailing `@ref` is taken from the right, because a path may itself
    /// contain `@`.
    fn parse_github(rest: &str, raw: &str) -> Result<Self> {
        let (slug, tail) = match rest.split_once('@') {
            Some((slug, tail)) => (slug, tail),
            None => (rest, ""),
        };

        let (owner, repo) = slug
            .split_once('/')
            .ok_or_else(|| anyhow!("'{raw}' is missing the repository; expected github:owner/repo"))?;
        if owner.is_empty() || repo.is_empty() {
            return Err(anyhow!("'{raw}' has an empty owner or repository"));
        }

        let (path, git_ref) = match tail.rsplit_once('@') {
            Some((path, git_ref)) => (path, git_ref),
            None => (tail, ""),
        };

        Ok(Source::Github {
            owner: owner.to_string(),
            repo: repo.trim_end_matches(".git").to_string(),
            path: if path.is_empty() { ".".into() } else { path.into() },
            git_ref: if git_ref.is_empty() {
                "HEAD".into()
            } else {
                git_ref.into()
            },
        })
    }

    /// Whether fetching this needs the network.
    pub fn is_remote(&self) -> bool {
        // A custom scheme is assumed to need the network: most do, and a
        // handler that answers from the filesystem can only make a fetch
        // cheaper, never wrong.
        !matches!(self, Source::Builtin { .. } | Source::Local { .. })
    }

    /// A short description, for diagnostics.
    pub fn describe(&self) -> String {
        match self {
            Source::Github {
                owner,
                repo,
                path,
                git_ref,
            } => format!("{owner}/{repo}@{path}@{git_ref}"),
            Source::Url { url } => url.clone(),
            Source::Npm { package, version } => match version {
                Some(version) => format!("npm:{package}@{version}"),
                None => format!("npm:{package}"),
            },
            Source::Builtin { name } => format!("builtin:{name}"),
            Source::Local { path } => format!("dylib:{path}"),
            Source::Custom { scheme, rest } => format!("{scheme}:{rest}"),
        }
    }
}

/// sha256 of a byte slice, lowercase hex.
pub fn sha256(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

/// sha256 of a file.
pub fn sha256_of_file(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    Ok(sha256(&bytes))
}

/// sha256 of a directory tree.
///
/// Deterministic across platforms: entries are sorted by their forward-slash
/// relative path, and each contributes `path`, a colon, its bytes, and a
/// newline. A pack built on Windows therefore verifies on Linux.
///
/// The algorithm is fixed by the format, so changing it invalidates every
/// published hash. That is the reason it is written out rather than delegated
/// to a hashing library's directory walk.
pub fn sha256_of_dir(dir: &Path) -> Result<String> {
    let mut files: Vec<(String, PathBuf)> = Vec::new();
    collect(dir, dir, &mut files)?;
    files.sort_by(|a, b| a.0.cmp(&b.0));

    let mut hasher = Sha256::new();
    for (relative, path) in &files {
        hasher.update(relative.as_bytes());
        hasher.update(b":");
        let bytes =
            std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        hasher.update(&bytes);
        hasher.update(b"\n");
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn collect(root: &Path, dir: &Path, out: &mut Vec<(String, PathBuf)>) -> Result<()> {
    let entries = std::fs::read_dir(dir)
        .with_context(|| format!("listing {}", dir.display()))?;
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            collect(root, &path, out)?;
        } else if file_type.is_file() {
            let relative = path
                .strip_prefix(root)
                .unwrap_or(&path)
                .to_string_lossy()
                .replace('\\', "/");
            out.push((relative, path));
        }
    }
    Ok(())
}

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

/// What a fetch produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fetched {
    /// Directory in the cache holding the content.
    pub path: PathBuf,
    /// sha256 of what was fetched.
    pub sha256: String,
    /// Whether the cache already had it.
    pub cached: bool,
}

/// A handler for one source scheme.
///
/// The kernel handles `github`, `https`, `builtin`, and `dylib` itself. A
/// deployment that serves packs from an internal registry, a bucket, or a git
/// host that is not GitHub registers a handler for its scheme and the format
/// needs no change.
///
/// Returns a boxed future rather than being `async`, because a trait method
/// cannot be `async` in this edition and blocking inside an async runtime would
/// stall the executor for the length of a download.
pub trait SchemeHandler: Send + Sync {
    /// The scheme this handles, lowercased and without the colon.
    fn scheme(&self) -> &str;

    /// Fetch `rest` (everything after the colon) into `into`.
    ///
    /// `into` exists and is empty when this is called; whatever the handler
    /// writes there becomes the fetched content.
    fn fetch<'a>(&'a self, rest: &'a str, into: &'a Path) -> SchemeFuture<'a>;
}

/// Boxed future returned by [`SchemeHandler::fetch`].
pub type SchemeFuture<'a> = std::pin::Pin<
    Box<dyn std::future::Future<Output = Result<()>> + Send + 'a>,
>;

/// A scheme handler that answers from the filesystem, for tests.
pub struct DirectoryScheme {
    scheme: String,
    root: PathBuf,
}

impl DirectoryScheme {
    /// Serve `scheme:` out of `root`, where `rest` is a path below it.
    pub fn new(scheme: impl Into<String>, root: impl Into<PathBuf>) -> Self {
        Self {
            scheme: scheme.into(),
            root: root.into(),
        }
    }
}

impl SchemeHandler for DirectoryScheme {
    fn scheme(&self) -> &str {
        &self.scheme
    }

    fn fetch<'a>(&'a self, rest: &'a str, into: &'a Path) -> SchemeFuture<'a> {
        Box::pin(async move {
            let source = self.root.join(rest.trim_start_matches('/'));
            if !source.exists() {
                return Err(anyhow!(
                    "{}:{} has nothing at {}",
                    self.scheme,
                    rest,
                    source.display()
                ));
            }
            if source.is_dir() {
                copy_tree(&source, into)?;
            } else {
                let name = source
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_else(|| "download".into());
                std::fs::copy(&source, into.join(name))?;
            }
            Ok(())
        })
    }
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

/// Fetches references into a content-addressed cache.
pub struct Fetcher {
    http: reqwest::Client,
    cache: PathBuf,
    token: Option<String>,
    /// Handlers for schemes the kernel does not handle itself.
    handlers: std::collections::HashMap<String, Arc<dyn SchemeHandler>>,
}

impl Fetcher {
    /// A fetcher writing to `$NGU_HOME/cache`, else `~/.nguruvilu/cache`.
    pub fn new() -> Result<Self> {
        Self::at(default_cache_dir())
    }

    /// A fetcher writing to a specific directory.
    pub fn at(cache: impl Into<PathBuf>) -> Result<Self> {
        let mut builder = reqwest::Client::builder()
            .user_agent(concat!("nguruvilu/", env!("CARGO_PKG_VERSION")))
            .timeout(fetch_timeout())
            // Redirects are followed by [`download_once`], not by the client.
            // The client's own policy follows up to ten hops wherever they
            // point, including down from https to http, and a bearer token
            // riding along on that is a token handed to the network.
            .redirect(reqwest::redirect::Policy::none());

        // Same rule as the model client, and now the same function: a
        // configured proxy, `NO_PROXY` honoured by suffix, and a direct
        // connection otherwise. The environment's proxy variables are ignored
        // unless this program was told to use a proxy.
        let configured = std::env::var("NGU_PROXY").unwrap_or_default();
        builder = match crate::llm::proxy_rules(&configured)? {
            Some(proxy) => builder.proxy(proxy),
            None => builder.no_proxy(),
        };

        Ok(Self {
            http: builder.build().context("building the fetch client")?,
            cache: cache.into(),
            handlers: std::collections::HashMap::new(),
            // Unauthenticated GitHub allows 60 requests an hour, which a
            // directory fetch exhausts quickly. A token is optional and only
            // ever read from the environment.
            token: std::env::var("GITHUB_TOKEN").ok().filter(|t| !t.trim().is_empty()),
        })
    }

    /// Register a handler for a scheme the kernel does not handle itself.
    ///
    /// A second handler for the same scheme replaces the first, so a plugin
    /// that reloads does not stack.
    pub fn register_scheme(&mut self, handler: Arc<dyn SchemeHandler>) {
        self.handlers
            .insert(handler.scheme().to_ascii_lowercase(), handler);
    }

    /// Every scheme this fetcher can resolve, for diagnostics.
    pub fn schemes(&self) -> Vec<String> {
        let mut schemes: Vec<String> = KNOWN_SCHEMES.iter().map(|s| (*s).to_string()).collect();
        schemes.extend(self.handlers.keys().cloned());
        schemes.sort();
        schemes.dedup();
        schemes
    }

    /// The cache root.
    pub fn cache_root(&self) -> &Path {
        &self.cache
    }

    /// Fetch one URL as a single file whose own sha256 the pack declares.
    ///
    /// `fetch` hashes the *directory* it writes, which is right for content
    /// arriving as a tree. A release asset is one file, and the project that
    /// ships it publishes the file's hash (a `checksums.txt`); declaring
    /// exactly that keeps the pack's hash checkable against the release with
    /// no derivative to recompute, and the cache is keyed by the same hash so
    /// a second install of the pack does no network work.
    pub async fn fetch_file(&self, url: &str, expect: &str) -> Result<Fetched> {
        if let Some(path) = self.cache_hit(expect) {
            return Ok(Fetched {
                path,
                sha256: expect.to_string(),
                cached: true,
            });
        }

        // Download into a scratch directory and only then move under the hash
        // the bytes actually have, so a failed download leaves no partial
        // entry — the same order `fetch` keeps.
        let scratch = self.scratch()?;
        let result = self.get_bytes(url).await;
        let bytes = match result {
            Ok(bytes) => bytes,
            Err(error) => {
                let _ = std::fs::remove_dir_all(&scratch);
                return Err(error);
            }
        };
        let actual = sha256(&bytes);
        if actual != expect {
            let _ = std::fs::remove_dir_all(&scratch);
            return Err(anyhow!(
                "hash mismatch for {url}: the pack declares {expect}, the download is {actual}. \
                 The content is not what the pack named; refusing to install it."
            ));
        }

        // The name comes from the URL, so the cached entry holds the archive
        // under the name the pack asked for.
        let name = url
            .rsplit('/')
            .next()
            .filter(|n| !n.is_empty() && !n.contains('?'))
            .unwrap_or("download");
        if let Err(error) = std::fs::write(scratch.join(name), bytes) {
            let _ = std::fs::remove_dir_all(&scratch);
            return Err(error).with_context(|| format!("writing {name}"));
        }

        let destination = self.cache.join(&actual);
        if destination.exists() {
            let _ = std::fs::remove_dir_all(&scratch);
        } else {
            std::fs::create_dir_all(&self.cache)
                .with_context(|| format!("creating {}", self.cache.display()))?;
            std::fs::rename(&scratch, &destination).with_context(|| {
                format!("moving {} into the cache", scratch.display())
            })?;
        }

        Ok(Fetched {
            path: destination,
            sha256: actual,
            cached: false,
        })
    }

    /// Fetch a source, reusing the cache when the expected hash is present.
    ///
    /// `expect` is the hash the pack declared. When it is `Some` and the cache
    /// already holds that hash, nothing touches the network — which is what
    /// makes a warm cache usable offline.
    pub async fn fetch(&self, source: &Source, expect: Option<&str>) -> Result<Fetched> {
        if let Some(expected) = expect {
            if let Some(path) = self.cache_hit(expected) {
                return Ok(Fetched {
                    path,
                    sha256: expected.to_string(),
                    cached: true,
                });
            }
        }

        // Download into a scratch directory, then move into place under the
        // hash it actually has. A failed download therefore never leaves a
        // partial entry in the cache.
        let scratch = self.scratch()?;
        let result = self.download_into(source, &scratch).await;

        let actual = match result {
            Ok(()) => match sha256_of_dir(&scratch) {
                Ok(hash) => hash,
                Err(error) => {
                    let _ = std::fs::remove_dir_all(&scratch);
                    return Err(error);
                }
            },
            Err(error) => {
                let _ = std::fs::remove_dir_all(&scratch);
                return Err(error);
            }
        };

        if let Some(expected) = expect {
            if actual != expected {
                let _ = std::fs::remove_dir_all(&scratch);
                return Err(anyhow!(
                    "hash mismatch for {}: the pack declares {expected}, the download is {actual}. \
                     The content is not what the pack named; refusing to install it.",
                    source.describe()
                ));
            }
        }

        let destination = self.cache.join(&actual);
        if destination.exists() {
            let _ = std::fs::remove_dir_all(&scratch);
        } else {
            std::fs::create_dir_all(&self.cache)
                .with_context(|| format!("creating {}", self.cache.display()))?;
            std::fs::rename(&scratch, &destination).with_context(|| {
                format!("moving {} into the cache", scratch.display())
            })?;
        }

        Ok(Fetched {
            path: destination,
            sha256: actual,
            cached: false,
        })
    }

    /// The cache directory for a hash, when it holds something.
    pub fn cache_hit(&self, hash: &str) -> Option<PathBuf> {
        // A hash from a pack is untrusted input that becomes a path. Anything
        // but hex would let `../..` escape the cache.
        if hash.is_empty() || !hash.chars().all(|c| c.is_ascii_hexdigit()) {
            return None;
        }
        let dir = self.cache.join(hash);
        if dir.is_dir() {
            Some(dir)
        } else {
            None
        }
    }

    fn scratch(&self) -> Result<PathBuf> {
        let dir = self
            .cache
            .join(format!(".tmp-{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("creating {}", dir.display()))?;
        Ok(dir)
    }

    async fn download_into(&self, source: &Source, into: &Path) -> Result<()> {
        match source {
            Source::Url { url } => {
                let bytes = self.get_bytes(url).await?;
                // The name comes from the URL, so a single-file reference lands
                // as a file rather than a directory holding one file.
                let name = url
                    .rsplit('/')
                    .next()
                    .filter(|n| !n.is_empty() && !n.contains('?'))
                    .unwrap_or("download");
                std::fs::write(into.join(name), bytes)
                    .with_context(|| format!("writing {}", name))?;
                Ok(())
            }
            Source::Github {
                owner,
                repo,
                path,
                git_ref,
            } => self.download_github(owner, repo, path, git_ref, into).await,
            Source::Builtin { .. } | Source::Local { .. } => Err(anyhow!(
                "{} is not a remote source and cannot be fetched",
                source.describe()
            )),
            // npm is installed, not downloaded: a tarball has no dependency
            // tree, so `pack::install` runs `npm install` for this variant
            // before anything reaches the fetcher.
            Source::Npm { .. } => Err(anyhow!(
                "{} is installed by npm during pack install, never fetched directly",
                source.describe()
            )),
            // A scheme the kernel does not handle. Resolved by a registered
            // handler; the failure names what is registered, because "no
            // handler" without that list is a dead end.
            Source::Custom { scheme, rest } => {
                let handler = self.handlers.get(scheme).ok_or_else(|| {
                    anyhow!(
                        "no handler for the '{scheme}' scheme. Registered: {}. \
                         A plugin can add one.",
                        self.schemes().join(", ")
                    )
                })?;
                handler.fetch(rest, into).await
            }
        }
    }

    /// Fetch a path inside a repository.
    ///
    /// The tree listing is one API call; the files come from
    /// `raw.githubusercontent.com`, which is not rate-limited the way the API
    /// is. That keeps a directory fetch to a single API request.
    async fn download_github(
        &self,
        owner: &str,
        repo: &str,
        path: &str,
        git_ref: &str,
        into: &Path,
    ) -> Result<()> {
        let prefix = path.trim_matches('/');
        let listing = format!(
            "https://api.github.com/repos/{owner}/{repo}/git/trees/{git_ref}?recursive=1"
        );
        let body = self.get_bytes(&listing).await.with_context(|| {
            format!("listing {owner}/{repo} at {git_ref}; a private repository needs GITHUB_TOKEN")
        })?;
        let tree: serde_json::Value =
            serde_json::from_slice(&body).context("GitHub returned something other than JSON")?;

        let blobs: Vec<String> = tree
            .get("tree")
            .and_then(|t| t.as_array())
            .map(|entries| {
                entries
                    .iter()
                    .filter(|e| e.get("type").and_then(|t| t.as_str()) == Some("blob"))
                    .filter_map(|e| e.get("path").and_then(|p| p.as_str()))
                    .filter(|p| {
                        prefix.is_empty()
                            || *p == prefix
                            || p.starts_with(&format!("{prefix}/"))
                    })
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default();

        if blobs.is_empty() {
            return Err(anyhow!(
                "no files at '{prefix}' in {owner}/{repo}@{git_ref}"
            ));
        }

        for blob in &blobs {
            let relative = if prefix.is_empty() {
                blob.clone()
            } else {
                blob.strip_prefix(prefix)
                    .unwrap_or(blob)
                    .trim_start_matches('/')
                    .to_string()
            };
            // A single-file reference keeps its own name; a directory keeps the
            // tree below it.
            let relative = if relative.is_empty() {
                blob.rsplit('/').next().unwrap_or(blob).to_string()
            } else {
                relative
            };

            let url =
                format!("https://raw.githubusercontent.com/{owner}/{repo}/{git_ref}/{blob}");
            let bytes = self.get_bytes(&url).await.with_context(|| {
                format!("fetching {blob} from {owner}/{repo}@{git_ref}")
            })?;

            let destination = into.join(&relative);
            if let Some(parent) = destination.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&destination, bytes)
                .with_context(|| format!("writing {relative}"))?;
        }
        Ok(())
    }

    /// Download one URL into memory, retrying a connection-level failure.
    ///
    /// The retry is bounded and it is safe because a GET has no side effects:
    /// repeating it cannot duplicate an install. A refusal (4xx/5xx) is not
    /// retried — that is the server's final word on the request, and asking
    /// again spends the round trip twice for the same answer.
    async fn get_bytes(&self, url: &str) -> Result<Vec<u8>> {
        self.get_bytes_capped(url, MAX_DOWNLOAD_BYTES).await
    }

    /// Download `url`, refusing a body over `cap` bytes.
    ///
    /// The cap is a parameter so the rule can be exercised without moving
    /// gigabytes through a test.
    async fn get_bytes_capped(&self, url: &str, cap: u64) -> Result<Vec<u8>> {
        let mut attempt = 0u32;
        loop {
            attempt += 1;
            match self.download_once(url, cap).await {
                Ok(bytes) => return Ok(bytes),
                Err(error) => {
                    if attempt > DOWNLOAD_RETRIES || !transient_download_failure(&error) {
                        return Err(error);
                    }
                    // Exponential, starting small: a pooled connection the peer
                    // closed is gone by the second attempt, and an endpoint that
                    // is merely busy is not hit again at the worst moment.
                    let pause = std::time::Duration::from_millis(
                        RETRY_BACKOFF_MS << (attempt - 1).min(8),
                    );
                    tokio::time::sleep(pause).await;
                }
            }
        }
    }

    /// One attempt at downloading `url`, bounded by `cap`.
    ///
    /// Redirects are followed here rather than by the HTTP client, because
    /// following one is a decision: the chain may step down from https to http,
    /// or leave the origin the `GITHUB_TOKEN` was issued for, and either one
    /// hands a credential to whoever is listening. This loop keeps the token on
    /// its own origin, refuses the downgrade, and stops after five hops.
    ///
    /// The body is accumulated chunk by chunk rather than materialised whole
    /// and copied: `bytes()` allocates the entire response and `to_vec()` then
    /// copies it again, which for a large asset means two buffers of the size
    /// the ceiling exists to bound. A declared `content-length` is checked
    /// before a byte is read; a chunked or lying response is caught by the
    /// running total.
    async fn download_once(&self, url: &str, cap: u64) -> Result<Vec<u8>> {
        let first = reqwest::Url::parse(url).with_context(|| format!("{url} is not a URL"))?;
        let mut chain: Vec<reqwest::Url> = vec![first];
        // The hop cap bounds how many requests a chain may make; this bounds how
        // long it may keep making them. Without it a redirector that answers
        // just inside the timeout on every hop holds a step for the cap times
        // the timeout — five hops of two minutes each.
        let budget = fetch_timeout();
        let started = std::time::Instant::now();

        let response = loop {
            let mut request = self
                .http
                .get(chain.last().expect("the chain starts with the request").clone());
            if let Some(token) = self.credential_for(&chain) {
                request = request.bearer_auth(token);
            }
            let response = request.send().await?;

            match redirect_target(&response, &chain)? {
                Some(next) => {
                    if started.elapsed() > budget {
                        return Err(anyhow!(
                            "{} is still redirecting after {:.0}s across {} hops; giving up",
                            chain[0],
                            started.elapsed().as_secs_f64(),
                            chain.len()
                        ));
                    }
                    chain.push(next);
                }
                // Not a redirect (or one with nowhere to go): this is the answer.
                None => break response,
            }
        };

        let status = response.status();
        if !status.is_success() {
            return Err(HttpStatus { status, url: url.to_string() }.into());
        }

        let declared = response.content_length();
        if let Some(declared) = declared {
            if declared > cap {
                return Err(download_too_large(url, declared, cap));
            }
        }

        // Reserve what the response says it needs, but never on the strength of
        // a number that would itself be an allocation worth avoiding.
        let reserve = declared.unwrap_or(0).min(8 * 1024 * 1024) as usize;
        let mut bytes: Vec<u8> = Vec::with_capacity(reserve);
        let mut stream = response.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            let total = bytes.len() as u64 + chunk.len() as u64;
            if total > cap {
                return Err(download_too_large(url, total, cap));
            }
            bytes.extend_from_slice(&chunk);
        }
        Ok(bytes)
    }

    /// The token to send with this hop of a redirect chain, if any.
    ///
    /// The token is issued for the GitHub API and is sent to that origin and
    /// nothing else: once a redirect leaves the origin the request started on,
    /// the header is dropped rather than carried along. A mirror on another
    /// host has not been trusted with it, and a hostile one phrased to look
    /// like the API host in its *path* gets nothing either way — the host is
    /// read from the URL, not matched inside the whole string.
    fn credential_for(&self, chain: &[reqwest::Url]) -> Option<&str> {
        let token = self.token.as_deref()?;
        let first = chain.first()?;
        if !is_api_host(first) {
            return None;
        }
        chain
            .iter()
            .all(|hop| same_origin(first, hop))
            .then_some(token)
    }
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
fn safe_mode(mode: u32) -> u32 {
    if mode & 0o111 != 0 {
        0o755
    } else {
        0o644
    }
}

/// The most hops one download may follow.
const MAX_REDIRECTS: usize = 5;

/// Where a 3xx points, or `None` when the response is the answer.
fn redirect_target(
    response: &reqwest::Response,
    chain: &[reqwest::Url],
) -> Result<Option<reqwest::Url>> {
    if !response.status().is_redirection() {
        return Ok(None);
    }
    let Some(location) = response.headers().get(reqwest::header::LOCATION) else {
        // A 3xx with nowhere to go *is* an answer; the status check below
        // turns it into an error worth reporting.
        return Ok(None);
    };
    let location = location.to_str().with_context(|| {
        format!(
            "a redirect from {} carries a Location that is not text",
            chain
                .last()
                .map(|url| url.as_str())
                .unwrap_or("the request")
        )
    })?;
    redirect_follow(chain, location).map(Some)
}

/// Follow `chain` one more hop to the `Location` it names.
///
/// The rules a reflexive follow would skip: the chain is bounded, and a hop
/// that steps down from https to http is refused outright, because whatever
/// credentials the first host was given would travel in clear text from there.
fn redirect_follow(chain: &[reqwest::Url], location: &str) -> Result<reqwest::Url> {
    let from = chain.last().expect("the chain starts with the request");
    if chain.len() > MAX_REDIRECTS {
        return Err(anyhow!(
            "{} redirected more than {MAX_REDIRECTS} times; refusing to keep following",
            chain[0]
        ));
    }
    let next = from
        .join(location)
        .with_context(|| format!("{from} redirected to an unusable Location: {location}"))?;
    if from.scheme() == "https" && next.scheme() == "http" {
        return Err(anyhow!(
            "refusing a redirect that downgrades https to http ({from} -> {next}): \
             the first host's credential would travel in clear text"
        ));
    }
    Ok(next)
}

/// Whether a URL is the GitHub API, the one host this fetcher holds a token for.
///
/// Named by *host* rather than matched inside the whole URL: the old
/// `url.contains("api.github.com")` also matched a path such as
/// `https://elsewhere.test/api.github.com/x`, which is a way to be handed this
/// token by a link that has nothing to do with GitHub.
fn is_api_host(url: &reqwest::Url) -> bool {
    url.host_str() == Some("api.github.com")
}

/// Whether two URLs share a scheme, host, and port.
fn same_origin(a: &reqwest::Url, b: &reqwest::Url) -> bool {
    a.scheme() == b.scheme()
        && a.host_str() == b.host_str()
        && a.port_or_known_default() == b.port_or_known_default()
}

/// The provider answered, and the answer was a refusal.
///
/// A distinct type rather than a formatted string, because whether to retry
/// turns on exactly this: a status is final, a transport failure is not.
#[derive(Debug)]
struct HttpStatus {
    status: reqwest::StatusCode,
    url: String,
}

impl std::fmt::Display for HttpStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "HTTP {} from {}", self.status, self.url)
    }
}

impl std::error::Error for HttpStatus {}

/// The error a download over the ceiling reports.
fn download_too_large(url: &str, size: u64, cap: u64) -> anyhow::Error {
    anyhow!(
        "{url} is {}, over the {} ceiling for a single download; \
         refusing to buffer it",
        human_size(size),
        human_size(cap)
    )
}

/// A byte count a user can read: bytes, kibibytes, or mebibytes, whichever
/// makes the sentence say something.
fn human_size(bytes: u64) -> String {
    const KIB: u64 = 1024;
    const MIB: u64 = 1024 * 1024;
    if bytes >= MIB {
        format!("{} MiB", bytes / MIB)
    } else if bytes >= KIB {
        format!("{} KiB", bytes / KIB)
    } else {
        format!("{bytes} B")
    }
}

/// Whether a failed download is worth another attempt.
///
/// Only connection-level failures are: a refused connection, a pooled one the
/// peer closed, or a TLS session that arrived damaged. A status is the server's
/// verdict and a body that broke the ceiling will break it again; a timeout is
/// excluded so an endpoint that never answers is not waited on three times over.
fn transient_download_failure(error: &anyhow::Error) -> bool {
    if error.chain().any(|cause| cause.downcast_ref::<HttpStatus>().is_some()) {
        return false;
    }
    error.chain().any(|cause| match cause.downcast_ref::<reqwest::Error>() {
        Some(transport) => {
            !transport.is_timeout()
                && (transport.is_connect()
                    || crate::llm::is_connection_closed(transport)
                    || crate::llm::is_tls_stream_damage(transport))
        }
        None => false,
    })
}

/// The default cache directory: `$NGU_HOME/cache`, else `~/.nguruvilu/cache`.
pub fn default_cache_dir() -> PathBuf {
    crate::settings::data_dir().join("cache")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn a_github_reference_parses_into_its_parts() {
        let source = Source::parse("github:owner/repo@skills/pdf@v1.2.0").unwrap();
        assert_eq!(
            source,
            Source::Github {
                owner: "owner".into(),
                repo: "repo".into(),
                path: "skills/pdf".into(),
                git_ref: "v1.2.0".into(),
            }
        );
    }

    #[test]
    fn a_github_reference_may_omit_the_path_and_the_ref() {
        assert_eq!(
            Source::parse("github:owner/repo").unwrap(),
            Source::Github {
                owner: "owner".into(),
                repo: "repo".into(),
                path: ".".into(),
                git_ref: "HEAD".into(),
            }
        );
        // Only a ref.
        assert_eq!(
            Source::parse("github:owner/repo@v2").unwrap(),
            Source::Github {
                owner: "owner".into(),
                repo: "repo".into(),
                path: "v2".into(),
                git_ref: "HEAD".into(),
            }
        );
    }

    #[test]
    fn the_ref_is_taken_from_the_right() {
        // A path containing `@` must not be mistaken for the ref.
        let source = Source::parse("github:owner/repo@dir/we@ird/file@v1").unwrap();
        match source {
            Source::Github { path, git_ref, .. } => {
                assert_eq!(path, "dir/we@ird/file");
                assert_eq!(git_ref, "v1");
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn a_trailing_git_suffix_is_dropped() {
        match Source::parse("github:owner/repo.git@v1").unwrap() {
            Source::Github { repo, .. } => assert_eq!(repo, "repo"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn the_other_forms_parse() {
        assert_eq!(
            Source::parse("https://example.com/a.md").unwrap(),
            Source::Url {
                url: "https://example.com/a.md".into()
            }
        );
        assert_eq!(
            Source::parse("builtin:search").unwrap(),
            Source::Builtin {
                name: "search".into()
            }
        );
        assert_eq!(
            Source::parse("dylib:files/x.dll").unwrap(),
            Source::Local {
                path: "files/x.dll".into()
            }
        );
    }

    #[test]
    fn nonsense_is_refused_with_the_accepted_forms() {
        for bad in [
            "",
            "   ",
            "github:",
            "github:owner",
            "builtin:",
            "no-colon-here",
            ":empty-scheme",
        ] {
            let error = Source::parse(bad).expect_err(bad);
            let text = format!("{error:#}");
            assert!(
                text.contains("github") || text.contains("names no plugin"),
                "{bad:?} produced: {text}"
            );
        }
    }

    #[test]
    fn a_path_inside_the_pack_parses_as_local() {
        // The offline shape pack-format documents ("相对路径"): a pack shipping
        // its own skills writes them without a scheme. A colon-bearing string
        // was already claimed by the scheme branch, so this cannot swallow a
        // URL or `C:\...`; a bare word still fails.
        for raw in ["./skills/pdf", "skills/pdf", "..\\shared\\skills"] {
            assert_eq!(
                Source::parse(raw).unwrap(),
                Source::Local {
                    path: raw.to_string()
                },
                "{raw}"
            );
        }
        assert!(Source::parse("no-colon-here").is_err(), "a word is not a path");
    }

    #[test]
    fn an_unknown_scheme_is_kept_rather_than_refused() {
        // Which schemes exist is not the format's business: a deployment may
        // serve packs from an internal registry or a git host that is not
        // GitHub, and the set of names cannot be enumerated in advance.
        let source = Source::parse("gitlab:owner/repo@path@v1").unwrap();
        assert_eq!(
            source,
            Source::Custom {
                scheme: "gitlab".into(),
                rest: "owner/repo@path@v1".into(),
            }
        );
        assert!(source.is_remote(), "most custom schemes fetch over the network");
    }

    #[test]
    fn a_custom_scheme_reports_a_clean_error_when_no_handler_is_registered() {
        let fetcher = Fetcher::at(scratch("noscheme")).unwrap();
        let source = Source::parse("gitlab:owner/repo@x@v1").unwrap();
        let error = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(async { fetcher.fetch(&source, None).await })
            .expect_err("must refuse");
        let text = format!("{error:#}");
        assert!(text.contains("gitlab"), "{text}");
        assert!(
            text.contains("github"),
            "it should list what is registered: {text}"
        );
    }

    #[tokio::test]
    async fn a_registered_scheme_is_fetched_through_its_handler() {
        // The extension point: a deployment serves packs from wherever it keeps
        // them, and the format learns nothing about that place.
        let served = scratch("served");
        std::fs::create_dir_all(served.join("myskill")).unwrap();
        std::fs::write(served.join("myskill/SKILL.md"), "a skill").unwrap();

        let mut fetcher = Fetcher::at(scratch("custom-cache")).unwrap();
        fetcher.register_scheme(Arc::new(DirectoryScheme::new("internal", &served)));
        assert!(fetcher.schemes().contains(&"internal".to_string()));

        let source = Source::parse("internal:myskill").unwrap();
        let fetched = fetcher.fetch(&source, None).await.unwrap();
        assert!(fetched.path.join("SKILL.md").is_file());
        assert_eq!(
            std::fs::read_to_string(fetched.path.join("SKILL.md")).unwrap(),
            "a skill"
        );
    }

    #[test]
    fn only_remote_sources_need_the_network() {
        assert!(Source::parse("github:o/r@x@v").unwrap().is_remote());
        assert!(Source::parse("https://x/y").unwrap().is_remote());
        assert!(!Source::parse("builtin:search").unwrap().is_remote());
        assert!(!Source::parse("dylib:x.dll").unwrap().is_remote());
    }

    #[test]
    fn a_directory_hash_is_stable_regardless_of_creation_order() {
        let a = scratch("order-a");
        let b = scratch("order-b");
        for dir in [&a, &b] {
            std::fs::create_dir_all(dir.join("sub")).unwrap();
        }
        // Same content, written in opposite orders.
        std::fs::write(a.join("one.txt"), "first").unwrap();
        std::fs::write(a.join("sub/two.txt"), "second").unwrap();
        std::fs::write(b.join("sub/two.txt"), "second").unwrap();
        std::fs::write(b.join("one.txt"), "first").unwrap();

        assert_eq!(sha256_of_dir(&a).unwrap(), sha256_of_dir(&b).unwrap());
    }

    #[test]
    fn a_directory_hash_changes_with_content() {
        let dir = scratch("content");
        std::fs::write(dir.join("f.txt"), "before").unwrap();
        let before = sha256_of_dir(&dir).unwrap();
        std::fs::write(dir.join("f.txt"), "after").unwrap();
        assert_ne!(before, sha256_of_dir(&dir).unwrap());
    }

    #[test]
    fn a_directory_hash_notices_a_renamed_file() {
        // The path is part of the hash, not only the bytes.
        let a = scratch("rename-a");
        let b = scratch("rename-b");
        std::fs::write(a.join("x.txt"), "same").unwrap();
        std::fs::write(b.join("y.txt"), "same").unwrap();
        assert_ne!(sha256_of_dir(&a).unwrap(), sha256_of_dir(&b).unwrap());
    }

    #[test]
    fn a_file_hash_matches_the_byte_hash() {
        let dir = scratch("file");
        let path = dir.join("f.bin");
        std::fs::write(&path, [1u8, 2, 3]).unwrap();
        assert_eq!(sha256_of_file(&path).unwrap(), sha256(&[1, 2, 3]));
    }

    #[test]
    fn the_known_sha256_of_a_known_input_is_stable() {
        // Pinning one vector catches an accidental algorithm change, which
        // would silently invalidate every published pack hash.
        assert_eq!(
            sha256(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn a_cache_lookup_refuses_a_hash_that_is_a_path() {
        let fetcher = Fetcher::at(scratch("cache")).unwrap();
        assert!(fetcher.cache_hit("../../etc").is_none());
        assert!(fetcher.cache_hit("").is_none());
        assert!(fetcher.cache_hit("not-hex").is_none());
    }

    #[test]
    fn a_cache_lookup_finds_what_is_there() {
        let root = scratch("hit");
        let hash = "a".repeat(64);
        std::fs::create_dir_all(root.join(&hash)).unwrap();
        let fetcher = Fetcher::at(&root).unwrap();
        assert_eq!(fetcher.cache_hit(&hash), Some(root.join(&hash)));
        assert!(fetcher.cache_hit(&"b".repeat(64)).is_none());
    }

    #[tokio::test]
    async fn a_cached_hash_is_reused_without_touching_the_network() {
        let root = scratch("reuse");
        let hash = "c".repeat(64);
        std::fs::create_dir_all(root.join(&hash)).unwrap();
        let fetcher = Fetcher::at(&root).unwrap();

        // A source that cannot resolve, proving the cache short-circuits it.
        let source = Source::parse("github:nobody/nothing@x@y").unwrap();
        let fetched = fetcher.fetch(&source, Some(&hash)).await.unwrap();

        assert!(fetched.cached);
        assert_eq!(fetched.path, root.join(&hash));
    }

    #[tokio::test]
    async fn a_non_remote_source_is_refused_by_fetch() {
        let fetcher = Fetcher::at(scratch("local")).unwrap();
        let source = Source::parse("builtin:search").unwrap();
        let error = fetcher.fetch(&source, None).await.expect_err("must refuse");
        assert!(format!("{error:#}").contains("not a remote source"));
    }

    #[test]
    fn the_platform_tag_is_one_the_format_knows() {
        assert!(PLATFORMS.contains(&platform_tag()), "{}", platform_tag());
    }

    #[test]
    fn a_source_describes_itself() {
        assert_eq!(
            Source::parse("github:o/r@p@v").unwrap().describe(),
            "o/r@p@v"
        );
        assert_eq!(Source::parse("builtin:x").unwrap().describe(), "builtin:x");
    }

    #[test]
    fn archive_recognition_looks_at_the_name() {
        assert!(is_archive_name("cua-driver-rs-0.28.0-windows-x86_64-binary.zip"));
        assert!(is_archive_name("https://host/x/driver.tar.gz"));
        assert!(is_archive_name("driver.tgz?download=1"));
        assert!(!is_archive_name("driver.tar"));
        assert!(!is_archive_name("driver"));
    }

    #[test]
    fn a_zip_unpacks_into_the_directory_it_is_given() {
        let dir = scratch("extract-zip");
        let archive = dir.join("bundle.zip");
        {
            let file = std::fs::File::create(&archive).unwrap();
            let mut writer = zip::ZipWriter::new(file);
            let options = zip::write::SimpleFileOptions::default();
            writer.start_file("tool.exe", options).unwrap();
            writer.write_all(b"binary").unwrap();
            writer.start_file("docs/readme.txt", options).unwrap();
            writer.write_all(b"notes").unwrap();
            writer.finish().unwrap();
        }

        let into = dir.join("out");
        extract_archive(&archive, &into).unwrap();
        assert_eq!(std::fs::read(into.join("tool.exe")).unwrap(), b"binary");
        assert_eq!(
            std::fs::read_to_string(into.join("docs/readme.txt")).unwrap(),
            "notes"
        );
    }

    #[test]
    fn a_zip_entry_that_escapes_is_refused() {
        let dir = scratch("extract-escape");
        let archive = dir.join("evil.zip");
        {
            let file = std::fs::File::create(&archive).unwrap();
            let mut writer = zip::ZipWriter::new(file);
            writer
                .start_file("../outside.txt", zip::write::SimpleFileOptions::default())
                .unwrap();
            writer.write_all(b"escaped").unwrap();
            writer.finish().unwrap();
        }

        let error = extract_archive(&archive, &dir.join("out")).expect_err("must refuse");
        assert!(
            format!("{error:#}").contains("escapes the archive"),
            "{error:#}"
        );
        assert!(!dir.join("outside.txt").exists(), "nothing was written outside");
    }

    #[test]
    fn a_tgz_unpacks_and_keeps_the_executable_bit() {
        let dir = scratch("extract-tgz");
        let archive = dir.join("bundle.tar.gz");
        write_tgz(&archive, "bin/tool", 0o755, b"script");

        let into = dir.join("out");
        extract_archive(&archive, &into).unwrap();
        assert_eq!(std::fs::read(into.join("bin/tool")).unwrap(), b"script");

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(into.join("bin/tool"))
                .unwrap()
                .permissions()
                .mode();
            assert_ne!(mode & 0o111, 0, "the executable bit survives: {mode:o}");
        }
    }

    #[test]
    fn a_tgz_entry_that_escapes_is_refused() {
        // The same refusal for tar as for zip: a `..` name aborts before a
        // single byte is written.
        let dir = scratch("extract-tgz-escape");
        let archive = dir.join("evil.tar.gz");
        write_tgz_raw_name(&archive, "../outside.txt", 0o644, b"bad");

        let error = extract_archive(&archive, &dir.join("out")).expect_err("must refuse");
        assert!(
            format!("{error:#}").contains("escapes the archive"),
            "{error:#}"
        );
        assert!(!dir.join("outside.txt").exists());
    }

    #[test]
    fn a_name_that_is_not_an_archive_says_so() {
        let dir = scratch("extract-unknown");
        let plain = dir.join("bundle.rar");
        std::fs::write(&plain, b"not an archive").unwrap();
        let error = extract_archive(&plain, &dir.join("out")).expect_err("must refuse");
        assert!(format!("{error:#}").contains("expected .zip"), "{error:#}");
    }

    #[tokio::test]
    async fn a_single_file_fetch_reuses_the_cache_keyed_by_the_file_hash() {
        // The cache hit answers before any network call, which is what makes
        // a second install of the same pack work with no connection.
        let root = scratch("file-fetch");
        let bytes = b"release asset";
        let hash = sha256(bytes);
        std::fs::create_dir_all(root.join(&hash)).unwrap();
        std::fs::write(root.join(&hash).join("asset.zip"), bytes).unwrap();

        let fetcher = Fetcher::at(&root).unwrap();
        let fetched = fetcher
            .fetch_file("https://example.invalid/asset.zip", &hash)
            .await
            .unwrap();
        assert!(fetched.cached, "the seeded entry answers the fetch");
        assert_eq!(fetched.path, root.join(&hash));
        assert_eq!(
            std::fs::read(fetched.path.join("asset.zip")).unwrap(),
            bytes
        );
    }

    /// A loopback HTTP server answering every request with `body`.
    ///
    /// With `declared` the response carries a `content-length`; without it the
    /// body runs to the end of the connection, which is how a chunked or
    /// length-lying server reaches the client. Returns the base URL and a count
    /// of the requests actually served.
    async fn serving(
        body: Vec<u8>,
        declared: bool,
    ) -> (String, Arc<std::sync::atomic::AtomicUsize>) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let hits = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counter = Arc::clone(&hits);

        tokio::spawn(async move {
            while let Ok((mut socket, _)) = listener.accept().await {
                let body = body.clone();
                let counter = Arc::clone(&counter);
                tokio::spawn(async move {
                    // Drain the request before answering: a close that races a
                    // client still writing turns every test into a flake.
                    let mut head = Vec::new();
                    let mut byte = [0u8; 1];
                    loop {
                        match socket.read(&mut byte).await {
                            Ok(0) | Err(_) => return,
                            Ok(_) => {
                                head.push(byte[0]);
                                if head.ends_with(b"\r\n\r\n") {
                                    break;
                                }
                            }
                        }
                    }
                    counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

                    let mut response = b"HTTP/1.1 200 OK\r\nconnection: close\r\n".to_vec();
                    if declared {
                        response.extend_from_slice(
                            format!("content-length: {}\r\n", body.len()).as_bytes(),
                        );
                    }
                    response.extend_from_slice(b"\r\n");
                    response.extend_from_slice(&body);
                    let _ = socket.write_all(&response).await;
                    let _ = socket.flush().await;
                    // Hold the socket a moment so the client's read sees an
                    // orderly end of body.
                    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                });
            }
        });

        (format!("http://{addr}"), hits)
    }

    #[tokio::test]
    async fn a_download_whose_declared_length_is_over_the_ceiling_is_refused() {
        // Before the fix, nothing bounded a download at all: one URL that
        // answers 4 GiB was an out-of-memory kill rather than an error.
        let (base, hits) = serving(vec![b'z'; 4096], true).await;
        let fetcher = Fetcher::at(scratch("declared-ceiling")).unwrap();

        let error = fetcher
            .get_bytes_capped(&base, 1024)
            .await
            .expect_err("a declared body over the ceiling is refused");
        assert!(format!("{error:#}").contains("ceiling"), "{error:#}");
        assert_eq!(
            hits.load(std::sync::atomic::Ordering::Relaxed),
            1,
            "a size refusal is not a transport failure, so it is not retried"
        );
    }

    #[tokio::test]
    async fn a_download_without_a_declared_length_is_capped_as_it_streams() {
        // The same ceiling for a server that declares nothing: the running
        // total has to trip, not the header.
        let (base, hits) = serving(vec![b'z'; 8192], false).await;
        let fetcher = Fetcher::at(scratch("streaming-ceiling")).unwrap();

        let error = fetcher
            .get_bytes_capped(&base, 1024)
            .await
            .expect_err("an undeclared body is still bounded");
        assert!(format!("{error:#}").contains("ceiling"), "{error:#}");
        assert_eq!(hits.load(std::sync::atomic::Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn a_download_within_the_ceiling_arrives_whole() {
        // The happy path, so the cap is proved not to eat a normal body.
        let payload: Vec<u8> = (0..600u32).map(|index| (index % 251) as u8).collect();
        let (base, _hits) = serving(payload.clone(), true).await;
        let fetcher = Fetcher::at(scratch("whole")).unwrap();

        let bytes = fetcher
            .get_bytes_capped(&base, 1024 * 1024)
            .await
            .expect("the download completes");
        assert_eq!(bytes, payload, "byte for byte, one copy");
    }

    #[tokio::test]
    async fn a_connection_dropped_before_the_answer_is_retried() {
        // The fetch path used to have no retry at all, while the model client
        // had one: the same peer closing a connection ended a download and so
        // an install. A GET has no side effects, so asking again is safe.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let hits = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counter = Arc::clone(&hits);
        let payload = b"the pack contents".to_vec();
        let served = payload.clone();

        tokio::spawn(async move {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            let mut answered = false;
            while let Ok((mut socket, _)) = listener.accept().await {
                let seen = counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if !answered && seen == 0 {
                    // Hang up without writing a single response byte. The
                    // request itself is already in the socket buffer, so the
                    // client sees a connection that went away mid-answer.
                    drop(socket);
                    continue;
                }
                answered = true;

                let mut head = Vec::new();
                let mut byte = [0u8; 1];
                loop {
                    match socket.read(&mut byte).await {
                        Ok(0) | Err(_) => break,
                        Ok(_) => {
                            head.push(byte[0]);
                            if head.ends_with(b"\r\n\r\n") {
                                break;
                            }
                        }
                    }
                }
                let response = format!(
                    "HTTP/1.1 200 OK\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                    served.len()
                );
                let _ = socket.write_all(response.as_bytes()).await;
                let _ = socket.write_all(&served).await;
                let _ = socket.flush().await;
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            }
        });

        let fetcher = Fetcher::at(scratch("dropped")).unwrap();
        let bytes = fetcher
            .get_bytes_capped(&format!("http://{addr}/asset.bin"), 1024 * 1024)
            .await
            .expect("the retry recovers from a dropped connection");
        assert_eq!(bytes, payload);
        assert_eq!(
            hits.load(std::sync::atomic::Ordering::Relaxed),
            2,
            "one dropped connection, one retry"
        );
    }

    #[tokio::test]
    async fn a_refusal_is_not_retried() {
        // A 4xx is the server's final word on the request: retrying it spends
        // the round trip twice for the same answer, and turns a missing file
        // into three.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let hits = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counter = Arc::clone(&hits);
        tokio::spawn(async move {
            use tokio::io::AsyncWriteExt;
            while let Ok((mut socket, _)) = listener.accept().await {
                counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let _ = socket
                    .write_all(b"HTTP/1.1 404 Not Found\r\ncontent-length: 0\r\nconnection: close\r\n\r\n")
                    .await;
                let _ = socket.flush().await;
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            }
        });

        let fetcher = Fetcher::at(scratch("refused")).unwrap();
        let error = fetcher
            .get_bytes(&format!("http://{addr}/missing.bin"))
            .await
            .expect_err("a 404 is an error");
        assert!(format!("{error:#}").contains("404"), "{error:#}");
        assert_eq!(hits.load(std::sync::atomic::Ordering::Relaxed), 1);
    }

    #[test]
    fn the_fetch_timeout_follows_its_environment_variable() {
        // The old code had 120 seconds hardcoded and no way to say otherwise;
        // a pack mirror on a slow link could not be fetched at all.
        assert_eq!(fetch_timeout_secs(Some("45")), 45);
        assert_eq!(fetch_timeout_secs(Some(" 45 ")), 45, "padded is still a number");
        assert_eq!(fetch_timeout_secs(None), DEFAULT_FETCH_TIMEOUT_SECS);
        // A typo falls back rather than producing a timeout that always fires.
        assert_eq!(fetch_timeout_secs(Some("soon")), DEFAULT_FETCH_TIMEOUT_SECS);
        assert_eq!(fetch_timeout_secs(Some("0")), DEFAULT_FETCH_TIMEOUT_SECS);
        assert!(fetch_timeout().as_secs() >= 1, "never an instant timeout");
    }

    /// A `.zip` holding one entry of `size` zero bytes: a few kilobytes on
    /// disk, arbitrarily large once expanded. This is the shape of a
    /// decompression bomb in the smallest useful form.
    fn write_zip_of_zeros(into: &Path, name: &str, size: usize) {
        let file = std::fs::File::create(into).unwrap();
        let mut writer = zip::ZipWriter::new(file);
        writer
            .start_file(name, zip::write::SimpleFileOptions::default())
            .unwrap();
        writer.write_all(&vec![0u8; size]).unwrap();
        writer.finish().unwrap();
    }

    #[test]
    fn an_archive_that_expands_past_the_entry_ceiling_is_refused() {
        // A small archive that unpacks to something enormous is exactly what a
        // compressor is for, so an untrusted one has to be refused on the
        // expanded size rather than believed about it.
        let dir = scratch("zip-bomb");
        let archive = dir.join("bomb.zip");
        write_zip_of_zeros(&archive, "payload.bin", 4 * 1024 * 1024);
        assert!(
            std::fs::metadata(&archive).unwrap().len() < 512 * 1024,
            "the archive on disk is small; only its expansion is not"
        );

        let out = dir.join("out");
        let limits = UnpackLimits {
            entries: 16,
            entry_bytes: 64 * 1024,
            total_bytes: 1024 * 1024,
        };
        let error =
            extract_archive_with(&archive, &out, limits).expect_err("an expanding archive is refused");

        let text = format!("{error:#}");
        assert!(text.contains("payload.bin"), "the entry is named: {text}");
        assert!(text.contains("ceiling"), "the ceiling is named: {text}");
        assert!(
            !out.join("payload.bin").exists(),
            "nothing of the entry was written"
        );
    }

    #[test]
    fn an_archive_with_too_many_entries_is_refused() {
        // The cheap half of the same attack: many tiny entries, refused before
        // any of them is read.
        let dir = scratch("zip-many");
        let archive = dir.join("many.zip");
        {
            let file = std::fs::File::create(&archive).unwrap();
            let mut writer = zip::ZipWriter::new(file);
            for index in 0..8 {
                writer
                    .start_file(format!("f{index}.txt"), zip::write::SimpleFileOptions::default())
                    .unwrap();
                writer.write_all(b"x").unwrap();
            }
            writer.finish().unwrap();
        }

        let limits = UnpackLimits {
            entries: 4,
            entry_bytes: 1024,
            total_bytes: 4096,
        };
        let error = extract_archive_with(&archive, &dir.join("out"), limits)
            .expect_err("an archive with too many entries is refused");
        let text = format!("{error:#}");
        assert!(
            text.contains("more than 4 entries"),
            "the entry ceiling is named: {text}"
        );
    }

    #[test]
    fn an_archive_that_expands_past_the_total_ceiling_is_refused() {
        // Each entry fits on its own; together they do not.
        let dir = scratch("zip-total");
        let archive = dir.join("total.zip");
        {
            let file = std::fs::File::create(&archive).unwrap();
            let mut writer = zip::ZipWriter::new(file);
            for index in 0..4 {
                writer
                    .start_file(format!("p{index}.bin"), zip::write::SimpleFileOptions::default())
                    .unwrap();
                writer.write_all(&vec![b'z'; 4096]).unwrap();
            }
            writer.finish().unwrap();
        }

        let limits = UnpackLimits {
            entries: 16,
            entry_bytes: 8192,
            total_bytes: 8192,
        };
        let error = extract_archive_with(&archive, &dir.join("out"), limits)
            .expect_err("the whole archive is capped");
        let text = format!("{error:#}");
        assert!(
            text.contains("one archive"),
            "the total ceiling is named: {text}"
        );
    }

    #[test]
    fn a_normal_archive_is_unpacked_under_the_default_ceilings() {
        // The ceilings must not get in the way of the thing they exist for: a
        // real archive whose entries are ordinary files.
        let dir = scratch("zip-normal");
        let archive = dir.join("normal.zip");
        {
            let file = std::fs::File::create(&archive).unwrap();
            let mut writer = zip::ZipWriter::new(file);
            writer
                .start_file("readme.txt", zip::write::SimpleFileOptions::default())
                .unwrap();
            writer.write_all(b"hello").unwrap();
            writer.finish().unwrap();
        }
        let out = dir.join("out");
        extract_archive(&archive, &out).expect("a small archive is unpacked");
        assert_eq!(std::fs::read_to_string(out.join("readme.txt")).unwrap(), "hello");
    }

    #[test]
    fn an_archive_cannot_hand_out_a_setuid_binary() {
        // Mode bits come from the archive's author, and a setuid binary dropped
        // into a plugin directory is a privilege escalation the install never
        // asked for.
        assert_eq!(safe_mode(0o4755), 0o755, "the setuid bit is dropped");
        assert_eq!(safe_mode(0o2755), 0o755, "so is setgid");
        assert_eq!(safe_mode(0o6755), 0o755, "both of them");
        assert_eq!(safe_mode(0o7777), 0o755, "and the sticky bit");
        assert_eq!(
            safe_mode(0o4644),
            0o644,
            "setuid without an execute bit is just a file"
        );
        assert_eq!(safe_mode(0o600), 0o644, "a private file becomes readable");
        assert_eq!(safe_mode(0o100), 0o755, "any execute bit means 0o755");
    }

    #[cfg(unix)]
    #[test]
    fn an_extracted_binary_is_not_setuid() {
        use std::os::unix::fs::PermissionsExt;
        let dir = scratch("setuid-unpack");
        let archive = dir.join("tool.tgz");
        write_tgz(&archive, "tool", 0o4755, b"binary");

        let into = dir.join("out");
        extract_archive(&archive, &into).unwrap();

        let mode = std::fs::metadata(into.join("tool"))
            .unwrap()
            .permissions()
            .mode()
            & 0o7777;
        assert_eq!(mode, 0o755, "the setuid bit must not survive the unpack");
    }

    #[test]
    fn a_redirect_that_downgrades_to_http_is_refused() {
        // Following this one hands the first host's credential to whoever is
        // listening on the plain-text side of the hop.
        let chain = vec![reqwest::Url::parse("https://api.github.com/repos/o/r/zipball").unwrap()];
        let error = redirect_follow(&chain, "http://mirror.test/x.zip").expect_err("downgrade refused");
        let text = format!("{error:#}");
        assert!(text.contains("downgrades"), "{text}");
        assert!(text.contains("mirror.test"), "the hop is named: {text}");
    }

    #[test]
    fn a_redirect_chain_is_bounded() {
        let mut chain = vec![reqwest::Url::parse("https://host.test/0").unwrap()];
        for hop in 1..=5 {
            chain.push(reqwest::Url::parse(&format!("https://host.test/{hop}")).unwrap());
        }
        let error = redirect_follow(&chain, "/next").expect_err("the sixth hop is refused");
        assert!(
            format!("{error:#}").contains("5 times"),
            "the bound is named: {error:#}"
        );
    }

    #[test]
    fn a_relative_location_resolves_against_the_hop_it_came_from() {
        let chain = vec![reqwest::Url::parse("https://host.test/a/b.zip").unwrap()];
        assert_eq!(
            redirect_follow(&chain, "c.zip").unwrap().as_str(),
            "https://host.test/a/c.zip"
        );
        assert_eq!(
            redirect_follow(&chain, "/d.zip").unwrap().as_str(),
            "https://host.test/d.zip"
        );
    }

    #[test]
    fn a_credential_stays_on_the_origin_it_was_issued_for() {
        let api = reqwest::Url::parse("https://api.github.com/repos/o/r/git/trees/main").unwrap();
        let also_api = reqwest::Url::parse("https://api.github.com/raw/x").unwrap();
        let codeload = reqwest::Url::parse("https://codeload.github.com/o/r/zip").unwrap();
        let plain = reqwest::Url::parse("http://api.github.com/x").unwrap();

        assert!(is_api_host(&api));
        assert!(same_origin(&api, &also_api));
        assert!(!same_origin(&api, &codeload), "another host is another origin");
        assert!(!same_origin(&api, &plain), "and another scheme is too");

        // A path that merely spells the API host is not the API host: the old
        // substring test handed the token to this URL.
        let impostor =
            reqwest::Url::parse("https://elsewhere.test/api.github.com/x").unwrap();
        assert!(!is_api_host(&impostor), "{impostor}");
    }

    #[tokio::test]
    async fn a_cross_origin_redirect_is_followed_without_a_credential() {
        // GitHub answers a release-asset request on one host with a redirect to
        // another, so refusing every cross-origin hop would break downloads.
        // What must not happen is a credential crossing with it — and no
        // credential is attached here, because the first host has none.
        let payload = b"release asset".to_vec();
        let (destination, destination_hits) = serving(payload.clone(), true).await;

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let target = format!("{destination}/asset.zip");
        tokio::spawn(async move {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            while let Ok((mut socket, _)) = listener.accept().await {
                let target = target.clone();
                tokio::spawn(async move {
                    let mut head = Vec::new();
                    let mut byte = [0u8; 1];
                    loop {
                        match socket.read(&mut byte).await {
                            Ok(0) | Err(_) => return,
                            Ok(_) => {
                                head.push(byte[0]);
                                if head.ends_with(b"\r\n\r\n") {
                                    break;
                                }
                            }
                        }
                    }
                    let response = format!(
                        "HTTP/1.1 302 Found\r\nlocation: {target}\r\ncontent-length: 0\r\nconnection: close\r\n\r\n"
                    );
                    let _ = socket.write_all(response.as_bytes()).await;
                    let _ = socket.flush().await;
                    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                });
            }
        });

        let fetcher = Fetcher::at(scratch("redirect-cross")).unwrap();
        let bytes = fetcher
            .get_bytes_capped(&format!("http://{addr}/asset.zip"), 1024 * 1024)
            .await
            .expect("the redirect is followed");
        assert_eq!(bytes, payload);
        assert_eq!(
            destination_hits.load(std::sync::atomic::Ordering::Relaxed),
            1,
            "the payload came from the redirect target"
        );
    }

    #[tokio::test]
    async fn a_redirect_loop_is_bounded() {
        // The client's own policy would follow ten hops before giving up, and
        // every one of them would carry whatever header the first one had.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let hits = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counter = Arc::clone(&hits);
        let location = format!("http://{addr}/again");

        tokio::spawn(async move {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            while let Ok((mut socket, _)) = listener.accept().await {
                let location = location.clone();
                let counter = Arc::clone(&counter);
                tokio::spawn(async move {
                    let mut head = Vec::new();
                    let mut byte = [0u8; 1];
                    loop {
                        match socket.read(&mut byte).await {
                            Ok(0) | Err(_) => return,
                            Ok(_) => {
                                head.push(byte[0]);
                                if head.ends_with(b"\r\n\r\n") {
                                    break;
                                }
                            }
                        }
                    }
                    counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let response = format!(
                        "HTTP/1.1 302 Found\r\nlocation: {location}\r\ncontent-length: 0\r\nconnection: close\r\n\r\n"
                    );
                    let _ = socket.write_all(response.as_bytes()).await;
                    let _ = socket.flush().await;
                    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                });
            }
        });

        let fetcher = Fetcher::at(scratch("redirect-loop")).unwrap();
        let error = fetcher
            .get_bytes_capped(&format!("http://{addr}/start"), 1024 * 1024)
            .await
            .expect_err("an endless redirect is an error");
        assert!(
            format!("{error:#}").contains("5 times"),
            "the bound is named: {error:#}"
        );
        assert_eq!(
            hits.load(std::sync::atomic::Ordering::Relaxed),
            MAX_REDIRECTS + 1,
            "the first request plus the five hops it allowed"
        );
    }

    /// One small `.tar.gz` holding a single file, built in the test rather
    /// than kept as a fixture.
    fn write_tgz(into: &Path, name: &str, mode: u32, contents: &[u8]) {
        let file = std::fs::File::create(into).unwrap();
        let encoder = flate2::write::GzEncoder::new(file, flate2::Compression::default());
        let mut builder = tar::Builder::new(encoder);
        let mut header = tar::Header::new_gnu();
        header.set_size(contents.len() as u64);
        header.set_mode(mode);
        header.set_cksum();
        builder
            .append_data(&mut header, name, contents)
            .unwrap();
        builder.finish().unwrap();
        builder.into_inner().unwrap().finish().unwrap();
    }

    /// A `.tar.gz` whose entry name is written straight into the header
    /// bytes: `tar::Builder` refuses to *set* a `..` path, so an archive
    /// that actually carries one has to be laid down by hand.
    fn write_tgz_raw_name(into: &Path, name: &str, mode: u32, contents: &[u8]) {
        let file = std::fs::File::create(into).unwrap();
        let encoder = flate2::write::GzEncoder::new(file, flate2::Compression::default());
        let mut builder = tar::Builder::new(encoder);
        let mut header = tar::Header::new_gnu();
        header.set_size(contents.len() as u64);
        header.set_mode(mode);
        {
            let bytes = header.as_mut_bytes();
            for (index, byte) in name.as_bytes().iter().enumerate() {
                bytes[index] = *byte;
            }
        }
        header.set_cksum();
        builder.append(&header, contents).unwrap();
        builder.finish().unwrap();
        builder.into_inner().unwrap().finish().unwrap();
    }

    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir()
            .join(format!("ngu-fetch-{tag}-{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }
}
