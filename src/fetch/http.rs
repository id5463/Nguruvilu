//! Downloads over HTTP(S): the client, the redirect rule, and the retry rule.
//!
//! Redirects are followed here rather than by the HTTP client, because
//! following one is a decision: the chain may step down from https to http, or
//! leave the origin the token was issued for, and either one hands a credential
//! to whoever is listening. The retry is bounded and only repeats a transport
//! failure, because a status is the server's final word.

use std::path::Path;

use anyhow::{anyhow, Context, Result};
use futures_util::StreamExt;

use super::hash::human_size;
use super::source::{Fetcher, Source};

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
pub(crate) const DEFAULT_FETCH_TIMEOUT_SECS: u64 = 120;

/// The timeout a fetch uses: `NGU_FETCH_TIMEOUT_SECS`, else two minutes.
///
/// A download can reasonably take longer than a model request on a slow link,
/// and a mirror on the far side of the world can need longer still, so the
/// number is a setting rather than a constant baked into the client.
pub(crate) fn fetch_timeout() -> std::time::Duration {
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
pub(crate) fn fetch_timeout_secs(raw: Option<&str>) -> u64 {
    raw.and_then(|value| value.trim().parse::<u64>().ok())
        .filter(|secs| *secs > 0)
        .unwrap_or(DEFAULT_FETCH_TIMEOUT_SECS)
}

/// The most hops one download may follow.
pub(crate) const MAX_REDIRECTS: usize = 5;

/// Where a 3xx points, or `None` when the response is the answer.
pub(crate) fn redirect_target(
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
pub(crate) fn redirect_follow(chain: &[reqwest::Url], location: &str) -> Result<reqwest::Url> {
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
pub(crate) fn is_api_host(url: &reqwest::Url) -> bool {
    url.host_str() == Some("api.github.com")
}

/// Whether two URLs share a scheme, host, and port.
pub(crate) fn same_origin(a: &reqwest::Url, b: &reqwest::Url) -> bool {
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

/// The error a download over the ceiling reports.
pub(crate) fn download_too_large(url: &str, size: u64, cap: u64) -> anyhow::Error {
    anyhow!(
        "{url} is {}, over the {} ceiling for a single download; \
         refusing to buffer it",
        human_size(size),
        human_size(cap)
    )
}

/// Whether a failed download is worth another attempt.
///
/// Only connection-level failures are: a refused connection, a pooled one the
/// peer closed, or a TLS session that arrived damaged. A status is the server's
/// verdict and a body that broke the ceiling will break it again; a timeout is
/// excluded so an endpoint that never answers is not waited on three times over.
pub(crate) fn transient_download_failure(error: &anyhow::Error) -> bool {
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

impl std::error::Error for HttpStatus {}

impl Fetcher {
    pub(crate) async fn download_into(&self, source: &Source, into: &Path) -> Result<()> {
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

    /// Fetch a path inside a repository.
    ///
    /// The tree listing is one API call; the files come from
    /// `raw.githubusercontent.com`, which is not rate-limited the way the API
    /// is. That keeps a directory fetch to a single API request.
    pub(crate) async fn download_github(
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

    /// Download one URL into memory, retrying a connection-level failure.
    ///
    /// The retry is bounded and it is safe because a GET has no side effects:
    /// repeating it cannot duplicate an install. A refusal (4xx/5xx) is not
    /// retried — that is the server's final word on the request, and asking
    /// again spends the round trip twice for the same answer.
    pub(crate) async fn get_bytes(&self, url: &str) -> Result<Vec<u8>> {
        self.get_bytes_capped(url, MAX_DOWNLOAD_BYTES).await
    }

    /// Download `url`, refusing a body over `cap` bytes.
    ///
    /// The cap is a parameter so the rule can be exercised without moving
    /// gigabytes through a test.

    /// Download `url`, refusing a body over `cap` bytes.
    ///
    /// The cap is a parameter so the rule can be exercised without moving
    /// gigabytes through a test.
    pub(crate) async fn get_bytes_capped(&self, url: &str, cap: u64) -> Result<Vec<u8>> {
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
    pub(crate) async fn download_once(&self, url: &str, cap: u64) -> Result<Vec<u8>> {
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

    /// The token to send with this hop of a redirect chain, if any.
    ///
    /// The token is issued for the GitHub API and is sent to that origin and
    /// nothing else: once a redirect leaves the origin the request started on,
    /// the header is dropped rather than carried along. A mirror on another
    /// host has not been trusted with it, and a hostile one phrased to look
    /// like the API host in its *path* gets nothing either way — the host is
    /// read from the URL, not matched inside the whole string.
    pub(crate) fn credential_for(&self, chain: &[reqwest::Url]) -> Option<&str> {
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
