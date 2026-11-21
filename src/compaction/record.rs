//! The record of one compaction, and what its summary message looks like.

use serde::{Deserialize, Serialize};

use crate::message::Message;

/// The kernel's record of one compaction.
///
/// The count and the summary are all this in-memory record carries. The ids of
/// the run it replaced — the anchors a replay needs — are added when the record
/// is written down, because only a session store can see which messages a
/// loader will reduce; see [`crate::session`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Compaction {
    /// How many leading messages the summary stands in for.
    ///
    /// Kept as the fallback for records written before anchors existed, and as
    /// the length the anchors are measured against.
    pub replaced: usize,
    /// The summary text.
    pub summary: String,
}

impl Compaction {
    /// The marker that prefixes a summary message, so a reader — and
    /// [`is_summary`](Self::is_summary) — can tell a summary from something the
    /// user actually typed. Kept as the first line so sessions written before
    /// the framing below still read correctly.
    pub const MARKER: &'static str = "[summary of earlier conversation]";

    /// What the successor model is told about the summary it inherits.
    ///
    /// Wording follows codex's `SUMMARY_PREFIX`: the shortest framing that tells
    /// a resuming model "this is a handoff, not speech from the user, and the
    /// tool state comes with it". Without it a summary reads like an
    /// instruction from the user and gets argued with or restated.
    pub const HANDOFF: &'static str = "Another language model started to solve this \
problem and produced a summary of its thinking process. You also have access to the \
state of the tools that were used by that language model. Use this to build on the \
work that has already been done and avoid duplicating work. Here is the summary \
produced by the other language model, use the information in this summary to assist \
with your own analysis:";

    /// The message that stands in for the summarized run.
    pub fn message(&self) -> Message {
        Message::user(format!("{}\n{}\n{}", Self::MARKER, Self::HANDOFF, self.summary))
    }

    /// Whether a message is a compaction summary.
    pub fn is_summary(message: &Message) -> bool {
        message.text().starts_with(Self::MARKER)
    }
}
