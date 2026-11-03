//! Turning served interface files into something a phone can afford to fetch.
//!
//! The measurement that produced this module: `GET /` answered with 147,897 bytes,
//! `Cache-Control: no-store`, no `Content-Encoding`, and `Connection: close`. So
//! every page open and every switch over a relay tunnel re-downloaded the whole
//! interface, uncompressed, and the browser was explicitly forbidden to keep a
//! copy. Lazy rendering in the page does not help when the page itself is what
//! gets re-sent.
//!
//! Two cheap things fix the bulk of it, and neither changes what the interface is:
//! an `ETag` so a re-open can be answered with `304 Not Modified`, and gzip so the
//! first load costs a quarter of what it did.

use sha2::{Digest, Sha256};

/// A strong validator for one body: stable across restarts, so a phone that
/// cached the page yesterday can revalidate it today with one cheap request.
pub fn etag(body: &[u8]) -> String {
    let digest = Sha256::digest(body);
    // 16 hex characters (64 bits) is far past the point where an accidental
    // collision could matter for an interface asset, and keeps the header short.
    let hex: String = digest.iter().take(8).map(|byte| format!("{byte:02x}")).collect();
    format!("\"{hex}\"")
}

/// Does this `Accept-Encoding` value allow gzip?
///
/// Deliberately naive about `q=0`: a client that says `identity;q=0, gzip` is
/// handled, and a client that says `gzip;q=0` is not — that combination does not
/// occur in practice, and guessing wrong costs a retry, not correctness.
pub fn accepts_gzip(header: &str) -> bool {
    header
        .split(',')
        .map(|part| part.trim().to_ascii_lowercase())
        .any(|part| part == "gzip" || part.starts_with("gzip;"))
}

/// gzip a body. Returns `None` when compression would not help (a small body has
/// more header than payload) so callers can send it as-is.
pub fn gzip(body: &[u8]) -> Option<Vec<u8>> {
    use std::io::Write as _;
    let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
    encoder.write_all(body).ok()?;
    let packed = encoder.finish().ok()?;
    // Under 1 KB the framing overhead can exceed the saving.
    (packed.len() + 64 < body.len()).then_some(packed)
}

/// The caching policy for an interface asset.
///
/// `must-revalidate` with a zero max-age, not `no-store`: the browser may keep a
/// copy and must ask whether it is still current, which is one 304 instead of a
/// fresh 147 KB. It is not `immutable` either — packs can replace the interface,
/// and a stale page after a pack switch would be a wrong page.
pub const UI_CACHE_CONTROL: &str = "public, max-age=0, must-revalidate";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_etag_is_quoted_stable_and_content_dependent() {
        let body = b"<html>hello</html>";
        let first = etag(body);
        assert_eq!(first, etag(body), "the same body always gets the same tag");
        assert!(first.starts_with('"') && first.ends_with('"'), "quoted: {first}");
        assert_ne!(first, etag(b"<html>hello</html> "), "one byte of change is a new tag");
    }

    #[test]
    fn gzip_helps_a_real_page_and_is_skipped_for_tiny_bodies() {
        let page = "<div class='row'>".repeat(2000).into_bytes();
        let packed = gzip(&page).expect("a 37 KB page compresses");
        assert!(packed.len() < page.len() / 2, "{} vs {}", packed.len(), page.len());
        assert_eq!(&packed[..2], &[0x1f, 0x8b], "gzip magic bytes");

        assert!(gzip(b"ok").is_none(), "a two-byte body is not worth framing");
    }

    #[test]
    fn accept_encoding_is_read_the_way_clients_write_it() {
        assert!(accepts_gzip("gzip"));
        assert!(accepts_gzip("deflate, gzip, br"));
        assert!(accepts_gzip("GZIP"));
        assert!(accepts_gzip("gzip;q=1.0, identity;q=0.5"));
        assert!(!accepts_gzip("identity"));
        assert!(!accepts_gzip("br, deflate"));
        assert!(!accepts_gzip(""));
    }

    #[test]
    fn the_cache_policy_allows_a_copy_but_forbids_a_stale_one() {
        assert!(UI_CACHE_CONTROL.contains("must-revalidate"));
        assert!(!UI_CACHE_CONTROL.contains("no-store"), "no-store is the bug this replaces");
    }
}
