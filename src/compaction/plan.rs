//! Deciding which leading messages to summarize.

use crate::message::Message;

/// Which leading messages to summarize.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Plan {
    /// Number of leading messages to replace with a summary.
    pub summarize: usize,
    /// Number of trailing messages kept verbatim.
    pub keep: usize,
}

/// Fewest messages worth summarizing.
///
/// Summarizing a single message costs a model call and produces text that says
/// less than the message did. Observed in practice: a run of one user message
/// became a 548-character summary whose main content was that nothing had
/// happened yet. A user message plus its reply is the smallest run that can
/// carry information the summary might preserve.
///
/// The floor is deliberately low: raising it means a short conversation is
/// never compacted at all, and the window it was meant to protect still fills.
pub const MIN_SUMMARIZE: usize = 2;

/// Decide what to compact, or `None` when nothing can safely be removed.
///
/// `keep_recent` is a target, not a promise. The split is moved earlier until
/// the kept region can stand on its own — which it cannot do if it begins with a
/// `tool` message, since a tool result is only valid beside the assistant
/// message that requested it. Any other role is a valid start for a suffix, so
/// the search stops at the first non-tool message rather than demanding a user
/// message and giving up when there is none.
pub fn plan(messages: &[Message], keep_recent: usize) -> Option<Plan> {
    let keep_target = keep_recent.max(1);
    if messages.len() <= keep_target {
        return None;
    }

    // Start from the newest split that would keep `keep_recent` messages.
    let mut split = messages.len() - keep_target;

    while split > 0 && messages[split].role == crate::message::Role::Tool {
        split -= 1;
    }

    if split < MIN_SUMMARIZE {
        // Nothing worth summarizing, or no position where the remainder stands
        // on its own. Leaving the history alone is correct.
        return None;
    }

    Some(Plan {
        summarize: split,
        keep: messages.len() - split,
    })
}
