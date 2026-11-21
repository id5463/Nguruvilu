//! Applying a compaction to a history.

use crate::message::Message;

use super::plan::Plan;
use super::record::Compaction;

/// Replace a leading run of messages with a summary.
///
/// Returns the new history. Callers pass the result straight back to the agent.
///
/// The record built on the way carries the count and the summary, and nothing
/// else: the anchors that identify the run are taken when the record is stored,
/// because only the store can see the messages the loader will reduce.
pub fn apply(messages: &[Message], plan: Plan, summary: &str) -> Vec<Message> {
    let compaction = Compaction {
        replaced: plan.summarize,
        summary: summary.to_string(),
    };

    let mut next = Vec::with_capacity(plan.keep + 1);
    next.push(compaction.message());
    next.extend(messages[plan.summarize..].iter().cloned());
    next
}

/// Apply every compaction recorded for a session, in order.
///
/// Used when loading: the file holds the original messages *and* the records
/// that replaced them, and this reduces the two into the history the model
/// actually saw.
///
/// This is the count-only form. A [`Compaction`] carries the offset, not the
/// anchors, so it cannot check itself against the transcript — which is why a
/// session file's records also store [`crate::compaction::anchors_of`] the run they replaced, and
/// [`crate::session`] replays those through [`crate::compaction::locate_span`] rather than here.
/// A caller that has anchors should use them; this function exists for the
/// records that do not.
pub fn replay(messages: Vec<Message>, compactions: &[Compaction]) -> Vec<Message> {
    let mut history = messages;
    for compaction in compactions {
        if compaction.replaced == 0 || compaction.replaced > history.len() {
            continue;
        }
        let mut next = Vec::with_capacity(history.len() - compaction.replaced + 1);
        next.push(compaction.message());
        next.extend(history[compaction.replaced..].iter().cloned());
        history = next;
    }
    history
}
