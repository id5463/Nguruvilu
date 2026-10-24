//! Where the shell's events go.
//!
//! The window and the headless mode produce exactly the same events; only the
//! destination differs. Keeping that difference behind one trait means both
//! paths run the same dispatch code, so a headless run is a real test of the
//! windowed behaviour rather than a parallel implementation that can drift.

use serde_json::Value;
#[cfg(windows)]
use tao::event_loop::EventLoopProxy;

/// A destination for shell events.
pub trait EventSink: Send + Sync + 'static {
    /// Deliver one event.
    fn emit(&self, event: Value);
}

/// Sends events into the page through the window's event loop.
#[cfg(windows)]
pub struct WindowSink {
    proxy: EventLoopProxy<crate::UserEvent>,
}

#[cfg(windows)]
impl WindowSink {
    /// Build a sink that drives a webview.
    pub fn new(proxy: EventLoopProxy<crate::UserEvent>) -> Self {
        Self { proxy }
    }
}

#[cfg(windows)]
impl EventSink for WindowSink {
    fn emit(&self, event: Value) {
        // A closed window makes the send fail, which is not worth surfacing.
        let _ = self.proxy.send_event(crate::UserEvent::ToUi(event));
    }
}

/// The most bytes of one event that reach a line-based reader.
///
/// A single event can be an entire tool result or file body. Writing one of
/// those whole would either block the sink or push megabytes through a pipe
/// whose reader is expecting lines, so the line is cut and marked instead —
/// a truncated line is unparseable, a marked one at least says why.
const MAX_LINE_BYTES: usize = 1024 * 1024;

/// The marker that says a line was cut.
const TRUNCATION_MARK: &str = "…[truncated]";

/// `text` cut to at most `limit` bytes, on a character boundary, with
/// [`TRUNCATION_MARK`] appended when it had to cut.
fn truncate_line(text: &str, limit: usize) -> std::borrow::Cow<'_, str> {
    if text.len() <= limit {
        return std::borrow::Cow::Borrowed(text);
    }
    // Never split a multi-byte character: the bytes are a string's, not a
    // buffer's.
    let mut end = limit;
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    std::borrow::Cow::Owned(format!("{}{TRUNCATION_MARK}", &text[..end]))
}

/// Writes events to stdout as newline-delimited JSON.
///
/// This is what makes the shell callable: any process that can write a JSON
/// line to stdin and read one from stdout can drive it, including another
/// agent.
pub struct StdoutSink;

impl EventSink for StdoutSink {
    fn emit(&self, event: Value) {
        use std::io::Write;
        let line = event.to_string();
        let line = truncate_line(&line, MAX_LINE_BYTES);
        let stdout = std::io::stdout();
        let mut handle = stdout.lock();
        // One JSON object per line, flushed immediately: a reader that waits for
        // a full buffer would see nothing until the turn ended.
        match writeln!(handle, "{line}") {
            // The reader is gone. That is the normal end of a `| head` pipeline,
            // not a failure: there is nobody left to tell, so say nothing and
            // never panic on it.
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::BrokenPipe => return,
            Err(error) => {
                eprintln!("[stdout] cannot write event: {error}");
                return;
            }
        }
        let _ = handle.flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_lines_are_written_whole() {
        let line = truncate_line(r#"{"ev":"ready"}"#, MAX_LINE_BYTES);
        assert_eq!(line.as_ref(), r#"{"ev":"ready"}"#);
        assert!(!line.contains(TRUNCATION_MARK));
    }

    #[test]
    fn long_lines_are_cut_to_one_mib_and_marked() {
        let text = "x".repeat(MAX_LINE_BYTES + 10);
        let line = truncate_line(&text, MAX_LINE_BYTES);
        assert!(line.ends_with(TRUNCATION_MARK));
        assert_eq!(line.len(), MAX_LINE_BYTES + TRUNCATION_MARK.len());
    }

    #[test]
    fn truncation_never_splits_a_character() {
        // Two bytes per character: an odd limit lands inside the second one.
        let text = "é".repeat(16);
        let line = truncate_line(&text, 5);
        assert!(line.is_char_boundary(4), "cut fell inside a character");
        assert_eq!(line.as_ref(), format!("éé{TRUNCATION_MARK}"));
    }
}
