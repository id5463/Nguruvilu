//! Where the shell's events go.
//!
//! The window and the headless mode produce exactly the same events; only the
//! destination differs. Keeping that difference behind one trait means both
//! paths run the same dispatch code, so a headless run is a real test of the
//! windowed behaviour rather than a parallel implementation that can drift.

use serde_json::Value;
use tao::event_loop::EventLoopProxy;

/// A destination for shell events.
pub trait EventSink: Send + Sync + 'static {
    /// Deliver one event.
    fn emit(&self, event: Value);
}

/// Sends events into the page through the window's event loop.
pub struct WindowSink {
    proxy: EventLoopProxy<crate::UserEvent>,
}

impl WindowSink {
    /// Build a sink that drives a webview.
    pub fn new(proxy: EventLoopProxy<crate::UserEvent>) -> Self {
        Self { proxy }
    }
}

impl EventSink for WindowSink {
    fn emit(&self, event: Value) {
        // A closed window makes the send fail, which is not worth surfacing.
        let _ = self.proxy.send_event(crate::UserEvent::ToUi(event));
    }
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
        let stdout = std::io::stdout();
        let mut handle = stdout.lock();
        // One JSON object per line, flushed immediately: a reader that waits for
        // a full buffer would see nothing until the turn ended.
        let _ = writeln!(handle, "{event}");
        let _ = handle.flush();
    }
}
