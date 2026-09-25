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
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

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
pub const KNOWN_SCHEMES: &[&str] = &["github", "https", "http", "builtin", "dylib"];

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

        Err(anyhow!(
            "'{raw}' is not a usable source; expected a scheme such as {}, \
             or a plain https:// URL",
            KNOWN_SCHEMES.join(", ")
        ))
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
            .timeout(std::time::Duration::from_secs(120));
        // Same rule as the model client: the environment's proxy variables are
        // ignored unless this program was told to use a proxy.
        if let Ok(proxy) = std::env::var("NGU_PROXY") {
            if !proxy.trim().is_empty() {
                builder = builder.proxy(reqwest::Proxy::all(proxy.trim())?);
            }
        } else {
            builder = builder.no_proxy();
        }

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

    async fn get_bytes(&self, url: &str) -> Result<Vec<u8>> {
        let mut request = self.http.get(url);
        if url.contains("api.github.com") {
            if let Some(token) = &self.token {
                request = request.header("Authorization", format!("Bearer {token}"));
            }
        }
        let response = request.send().await?;
        let status = response.status();
        if !status.is_success() {
            return Err(anyhow!("HTTP {status} from {url}"));
        }
        Ok(response.bytes().await?.to_vec())
    }
}

/// The default cache directory: `$NGU_HOME/cache`, else `~/.nguruvilu/cache`.
pub fn default_cache_dir() -> PathBuf {
    crate::settings::data_dir().join("cache")
}

#[cfg(test)]
mod tests {
    use super::*;

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

    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir()
            .join(format!("ngu-fetch-{tag}-{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }
}
