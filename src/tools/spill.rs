//! Where a tool's full output goes, so the request can keep only a pointer.
//!
//! Two problems meet here.
//!
//! The first is the one this module was written for: a long session accumulates
//! every tool result it ever produced, and each step re-sends all of them. A
//! measured session of 2,749 steps carried ~36万 tokens per step, almost exactly
//! the sum of its 3,245 tool results — the window was not the problem, the junk
//! in it was. [`crate::compaction::prune_tool_results`] thins that history out;
//! this module is what makes thinning *safe*, because a thinned result still
//! names a file the model can `read`.
//!
//! The second problem is older and quieter: [`crate::tools::base::truncate_output`]
//! kept the head and tail of a huge output and dropped the middle with a note.
//! That middle was gone — not slow, not expensive, gone. Spilling turns it into
//! a path instead.
//!
//! Everything here is best-effort. A spilled file that cannot be written costs a
//! pointer, never a tool call: the caller keeps the text it already has.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// Outputs shorter than this are not spilled: a pointer costs more than the text.
pub const SPILL_FLOOR: usize = 1024;

/// Bytes one spilled file keeps.
///
/// A command that prints gigabytes must not fill the disk. Above this the file
/// is cut and says so, and the in-context head/tail view is unaffected.
pub const MAX_FILE_BYTES: usize = 8 * 1024 * 1024;

/// How many sessions keep their spilled outputs; older ones are removed.
const KEEP_SESSIONS: usize = 20;

/// The prefix that marks a pointer inside a tool result.
pub const MARKER: &str = "[full output:";

/// The extension a `data:` URI's mime type maps to.
fn extension(mime: &str) -> &'static str {
    match mime {
        "image/png" => "png",
        "image/jpeg" => "jpg",
        "image/gif" => "gif",
        "image/webp" => "webp",
        _ => "bin",
    }
}

/// Write an image's bytes to disk and return the path.
///
/// Only `data:` URIs are spilled: a remote URL costs a few characters in the
/// request, while an inline screenshot costs hundreds of kilobytes *every step*,
/// forever. This is the larger of the two context leaks a measured session
/// showed — 62.9 MB of its 67.7 MB of message bytes were inline images, against
/// 2.0 MB of tool text.
pub fn store_image(session: Option<&str>, tool: &str, url: &str) -> Option<PathBuf> {
    let rest = url.strip_prefix("data:")?;
    let (mime, encoded) = rest.split_once(";base64,")?;
    use base64::{engine::general_purpose, Engine as _};
    let bytes = general_purpose::STANDARD.decode(encoded).ok()?;
    let directory = root().join(safe(session.unwrap_or("no-session")));
    if std::fs::create_dir_all(&directory).is_err() {
        return None;
    }
    let millis = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let path = directory.join(format!(
        "{}-{millis}-{}.{}",
        safe(tool),
        sequence(),
        extension(mime)
    ));
    let temporary = path.with_extension("tmp");
    if std::fs::write(&temporary, &bytes).is_err() {
        return None;
    }
    if std::fs::rename(&temporary, &path).is_err() {
        let _ = std::fs::remove_file(&temporary);
        return None;
    }
    Some(path)
}

/// Readable size of a spilled file, for the note that replaces a dropped image.
pub fn describe_size(path: &str) -> String {
    match std::fs::metadata(path) {
        Ok(meta) => {
            let bytes = meta.len();
            if bytes >= 1024 * 1024 {
                format!("{:.1} MB", bytes as f64 / 1048576.0)
            } else if bytes >= 1024 {
                format!("{} KB", (bytes as f64 / 1024.0).round() as u64)
            } else {
                format!("{bytes} B")
            }
        }
        Err(_) => "size unknown".to_string(),
    }
}

fn root() -> PathBuf {
    // Tests write to a scratch directory of their own: the real one lives under
    // the user's data dir, which a sandboxed test run may not be allowed to
    // touch, and a test that depends on that would fail for the wrong reason.
    #[cfg(test)]
    {
        std::env::temp_dir().join(format!("ngu-spill-test-{}", std::process::id()))
    }
    #[cfg(not(test))]
    {
        crate::settings::data_dir().join("tool-output")
    }
}

/// Where spilled outputs live, for a panel that wants to show the path.
pub fn root_dir() -> PathBuf {
    root()
}

fn sequence() -> u64 {
    static SEQ: AtomicU64 = AtomicU64::new(1);
    SEQ.fetch_add(1, Ordering::Relaxed)
}

/// Keep only the letters, digits, dash and underscore of a tool name.
fn safe(text: &str) -> String {
    let cleaned: String = text
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
        .collect();
    if cleaned.is_empty() {
        "tool".to_string()
    } else {
        cleaned
    }
}

/// Write a tool's full output and return where it went.
///
/// `None` means the output was short enough to keep whole, or the write failed.
pub fn store(session: Option<&str>, tool: &str, text: &str) -> Option<PathBuf> {
    if text.len() < SPILL_FLOOR {
        return None;
    }
    let directory = root().join(safe(session.unwrap_or("no-session")));
    if std::fs::create_dir_all(&directory).is_err() {
        return None;
    }
    let millis = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let path = directory.join(format!("{}-{millis}-{}.txt", safe(tool), sequence()));
    let body = if text.len() > MAX_FILE_BYTES {
        format!(
            "{}\n\n--- output cut at {} MB; the rest was longer than this file may be ---\n",
            &text[..floor_boundary(text, MAX_FILE_BYTES)],
            MAX_FILE_BYTES / (1024 * 1024)
        )
    } else {
        text.to_string()
    };
    // Temporary file, then rename: a crash leaves either the whole file or none.
    let temporary = path.with_extension("tmp");
    if std::fs::write(&temporary, body.as_bytes()).is_err() {
        return None;
    }
    if std::fs::rename(&temporary, &path).is_err() {
        let _ = std::fs::remove_file(&temporary);
        return None;
    }
    prune_sessions();
    Some(path)
}

/// The largest byte index at or below `limit` that does not split a character.
fn floor_boundary(text: &str, limit: usize) -> usize {
    let mut end = limit.min(text.len());
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    end
}

/// The line a tool result carries to name its spilled output.
pub fn marker_line(path: &Path) -> String {
    format!("\n{MARKER} {}]", path.display())
}

/// The pointer already inside a tool result, if it has one.
///
/// The pruner keeps this line when it thins an old result: the text may go, the
/// way back to it may not.
pub fn marker_of(text: &str) -> Option<&str> {
    text.lines()
        .find(|line| line.trim_start().starts_with(MARKER))
        .map(str::trim)
}

/// Remove the session directories of spilled outputs beyond the newest few.
///
/// Best-effort and rare: it costs a directory listing after a write, and a
/// failure here never fails the tool call that triggered it.
fn prune_sessions() {
    let Ok(entries) = std::fs::read_dir(root()) else {
        return;
    };
    let mut directories: Vec<(std::time::SystemTime, PathBuf)> = entries
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.file_type().map(|kind| kind.is_dir()).unwrap_or(false))
        .map(|entry| {
            let modified = entry
                .metadata()
                .and_then(|meta| meta.modified())
                .unwrap_or(std::time::UNIX_EPOCH);
            (modified, entry.path())
        })
        .collect();
    if directories.len() <= KEEP_SESSIONS {
        return;
    }
    directories.sort_by(|a, b| b.0.cmp(&a.0));
    for (_, path) in directories.into_iter().skip(KEEP_SESSIONS) {
        let _ = std::fs::remove_dir_all(path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A session name of this test's own: the tests run in parallel, and one that
    /// removes "its" directory must never be removing another's.
    fn scratch_session() -> String {
        format!("test-{}-{}", std::process::id(), uuid::Uuid::new_v4())
    }

    #[test]
    fn a_short_output_is_not_spilled() {
        assert!(store(Some("s"), "bash", "ok").is_none());
    }

    #[test]
    fn a_long_output_is_written_whole_and_named_by_a_marker() {
        let session = scratch_session();
        let text = "x".repeat(SPILL_FLOOR + 10);
        let path = store(Some(&session), "bash", &text).expect("spills");
        assert!(path.is_file(), "{}", path.display());
        let stored = std::fs::read_to_string(&path).expect("reads back");
        assert_eq!(stored, text, "the file keeps what the model did not");
        let marked = format!("kept\n{}", marker_line(&path));
        assert_eq!(marker_of(&marked), Some(marker_line(&path).trim()));
        let _ = std::fs::remove_dir_all(root().join(safe(&session)));
    }

    #[test]
    fn two_calls_never_share_a_file() {
        let session = scratch_session();
        let text = "y".repeat(SPILL_FLOOR + 10);
        let first = store(Some(&session), "bash", &text).expect("first");
        let second = store(Some(&session), "bash", &text).expect("second");
        assert_ne!(first, second);
        let _ = std::fs::remove_dir_all(root().join(safe(&session)));
    }

    #[test]
    fn an_oversized_output_is_cut_rather_than_unbounded() {
        let session = scratch_session();
        let text = "z".repeat(MAX_FILE_BYTES + 1000);
        let path = store(Some(&session), "bash", &text).expect("spills");
        let stored = std::fs::read_to_string(&path).expect("reads back");
        assert!(stored.len() <= MAX_FILE_BYTES + 200, "{} bytes", stored.len());
        assert!(stored.contains("output cut at"), "the cut is named");
        let _ = std::fs::remove_dir_all(root().join(safe(&session)));
    }

    #[test]
    fn a_tool_name_cannot_escape_the_directory() {
        assert_eq!(safe("../../etc/passwd"), "______etc_passwd");
        assert_eq!(safe(""), "tool");
    }

    #[test]
    fn a_result_without_a_marker_has_none() {
        assert_eq!(marker_of("exit 0"), None);
    }

    #[test]
    fn an_inline_image_is_written_as_bytes_not_base64() {
        let session = scratch_session();
        let payload = general_payload("hello world");
        let url = format!("data:image/png;base64,{payload}");
        let path = store_image(Some(&session), "read", &url).expect("spills");
        assert_eq!(path.extension().and_then(|e| e.to_str()), Some("png"));
        assert_eq!(std::fs::read(&path).expect("reads back"), b"hello world");
        let _ = std::fs::remove_dir_all(root().join(safe(&session)));
    }

    #[test]
    fn a_remote_url_is_left_alone() {
        assert!(store_image(Some("s"), "read", "https://example.com/a.png").is_none());
    }

    #[test]
    fn a_broken_data_uri_is_not_spilled() {
        assert!(store_image(Some("s"), "read", "data:image/png;base64,@@not base64@@").is_none());
    }

    /// Base64 of `text`, without pulling the engine into the test scope.
    fn general_payload(text: &str) -> String {
        use base64::{engine::general_purpose, Engine as _};
        general_purpose::STANDARD.encode(text)
    }
}
