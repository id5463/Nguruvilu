//! What a pack may name, and the object that turns a name into content.
//!
//! This is the vocabulary half of the module: the [`Source`] forms a pack is
//! allowed to declare, the [`SchemeHandler`] a deployment registers for a
//! scheme the kernel does not know, and [`Fetcher`] itself — the cache, the
//! handlers, and the two public entry points that read from it.
//!
//! The download path is `super::http`, the archive reader `super::unpack`, and
//! content addressing `super::hash`.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};

use super::hash::{default_cache_dir, sha256, sha256_of_dir};
use super::http::fetch_timeout;
use super::unpack::copy_tree;

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
        // `plugin:<path>` is a process the pack carries. Local, not remote: the
        // kernel starts it from the pack directory and there is nothing to
        // fetch. Parsed as a local source so every "does this need the network"
        // question answers itself.
        if let Some(rest) = raw.strip_prefix("plugin:") {
            if rest.trim().is_empty() {
                return Err(anyhow!("'{raw}' names no plugin executable"));
            }
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
        //
        // `plugin:` is the exception, and it is not custom in the sense the
        // others are: it names a process the pack carries and the kernel starts,
        // so there is nothing to fetch. Left as remote, installing a pack that
        // ships a plugin failed with "no handler for the 'plugin' scheme" —
        // measured on `packs/restart`, whose archive installs fine and whose
        // install step tried to download a file that was already inside it.
        if let Source::Custom { scheme, .. } = self {
            if scheme == "plugin" {
                return false;
            }
        }
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

/// Fetches references into a content-addressed cache.
pub struct Fetcher {
    pub(super) http: reqwest::Client,
    pub(super) cache: PathBuf,
    pub(super) token: Option<String>,
    /// Handlers for schemes the kernel does not handle itself.
    pub(super) handlers: std::collections::HashMap<String, Arc<dyn SchemeHandler>>,
}

impl Fetcher {
    /// A fetcher writing to `$NGU_HOME/cache`, else `~/.nguruvilu/cache`.
    pub fn new() -> Result<Self> {
        Self::at(default_cache_dir())
    }

    /// A fetcher writing to a specific directory.

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

    /// Register a handler for a scheme the kernel does not handle itself.
    ///
    /// A second handler for the same scheme replaces the first, so a plugin
    /// that reloads does not stack.
    pub fn register_scheme(&mut self, handler: Arc<dyn SchemeHandler>) {
        self.handlers
            .insert(handler.scheme().to_ascii_lowercase(), handler);
    }

    /// Every scheme this fetcher can resolve, for diagnostics.

    /// Every scheme this fetcher can resolve, for diagnostics.
    pub fn schemes(&self) -> Vec<String> {
        let mut schemes: Vec<String> = KNOWN_SCHEMES.iter().map(|s| (*s).to_string()).collect();
        schemes.extend(self.handlers.keys().cloned());
        schemes.sort();
        schemes.dedup();
        schemes
    }

    /// The cache root.

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

    pub(crate) fn scratch(&self) -> Result<PathBuf> {
        let dir = self
            .cache
            .join(format!(".tmp-{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("creating {}", dir.display()))?;
        Ok(dir)
    }
}
