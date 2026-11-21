//! Finding the run a compaction record replaced, by content-derived anchors.

use sha2::{Digest, Sha256};

use crate::message::Message;

/// How many hex characters of an anchor's digest are kept.
///
/// Sixteen hex characters is 64 bits, which no session will ever collide on by
/// accident, and the identifier still fits in a log line that a person reads.
pub const ANCHOR_HEX: usize = 16;

/// A stable identifier for one message.
///
/// A message carries no generated id — the wire format has none, and inventing
/// one would have to be threaded through every producer and every file already
/// on disk — so the identifier is derived from what the message *is*: its role,
/// its text, the calls it makes or answers, and where its images point. A
/// message's anchor therefore survives being reloaded, copied between machines,
/// or written by a different build.
///
/// This digest is deliberately independent of the run fingerprint in
/// [`crate::session`]. That one guards a whole stored run against a count that
/// no longer describes it; this one names a single message. They cover the same
/// fields, but deriving one from the other would mean a change to either
/// invalidates records the other has already written.
pub fn anchor_of(message: &Message) -> String {
    let mut hasher = Sha256::new();
    // A version tag, so a later change to this digest cannot be mistaken for a
    // match against an anchor written by an older build.
    hasher.update(b"nguruvilu/anchor/1");
    anchor_field(&mut hasher, message.role.as_wire().as_bytes());
    anchor_field(&mut hasher, message.text().as_bytes());
    hasher.update((message.tool_calls.len() as u64).to_le_bytes());
    for call in &message.tool_calls {
        anchor_field(&mut hasher, call.id.as_bytes());
        anchor_field(&mut hasher, call.name.as_bytes());
        anchor_field(&mut hasher, call.arguments.as_bytes());
    }
    anchor_field(
        &mut hasher,
        message.tool_call_id.as_deref().unwrap_or("").as_bytes(),
    );
    hasher.update((message.images.len() as u64).to_le_bytes());
    for image in &message.images {
        anchor_field(&mut hasher, image.url.as_bytes());
    }
    let hex = format!("{:x}", hasher.finalize());
    hex[..ANCHOR_HEX.min(hex.len())].to_string()
}

/// Feed one length-prefixed field into an anchor digest.
///
/// Length-prefixed rather than delimited: two different messages must not hash
/// the same because one of them happened to contain the separator.
fn anchor_field(hasher: &mut Sha256, bytes: &[u8]) {
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

/// The anchors naming both ends of a run, when there is a run to name.
///
/// Both ends are stored because one id cannot tell a run that lost its last
/// message from a run that is intact: the record has to bracket the span it
/// replaced, not merely point into it.
pub fn anchors_of(run: &[Message]) -> Option<(String, String)> {
    let first = run.first()?;
    let last = run.last()?;
    Some((anchor_of(first), anchor_of(last)))
}

/// The refusal for a record whose anchor is not in the transcript.
///
/// One wording for either end, because which end went missing is a detail of
/// the damage, not of the conclusion: the run cannot be identified, so the
/// summary must not be applied to whatever happens to sit at the recorded
/// offset.
fn anchor_missing(which: &str, count: usize) -> String {
    format!(
        "the {which} message it replaced is no longer in the file; \
         not applied, kept all {count} messages"
    )
}

/// Locate the span a compaction record replaced, by the anchors it stored.
///
/// Returns the inclusive index range into `messages`, or the reason the span
/// cannot be trusted. The reason is a lowercase clause written to be read after
/// the record's number ("compaction #2: ..."): the session loader prefixes it
/// and reports it as drift.
///
/// A count says "the first N", and a count is exactly what a missing line
/// changes. So when a record carries anchors, the count is only used to
/// sanity-check the span the anchors found: the anchors say *which* messages,
/// the count says *how many* must be between them. A record whose anchors
/// cannot both be found, or whose found span is not the length it recorded, is
/// refused rather than guessed at — folding the wrong run deletes messages no
/// summary ever stood in for, and nothing else in the system would notice.
pub fn locate_span(
    messages: &[Message],
    replaced: usize,
    first: &str,
    last: &str,
) -> Result<(usize, usize), String> {
    let start = messages
        .iter()
        .position(|message| anchor_of(message) == first)
        .ok_or_else(|| anchor_missing("first", messages.len()))?;

    // The count is checked first, because on an intact transcript it is exact:
    // the last replaced message sits precisely `replaced` slots after the first
    // one, and an identical message repeating elsewhere cannot fool that check.
    // Only when it is not there is the anchor itself searched for, and then the
    // span is measured against the count instead of being trusted.
    let end = match start.checked_add(replaced.saturating_sub(1)) {
        Some(candidate) if messages.get(candidate).map(anchor_of).as_deref() == Some(last) => {
            candidate
        }
        _ => messages
            .iter()
            .enumerate()
            .skip(start)
            .find(|(_, message)| anchor_of(*message) == last)
            .map(|(index, _)| index)
            .ok_or_else(|| anchor_missing("last", messages.len()))?,
    };

    let span = end - start + 1;
    if span != replaced {
        return Err(format!(
            "the run it replaced is now {span} messages long, not the {replaced} it recorded; \
             not applied, kept all {} messages",
            messages.len()
        ));
    }
    Ok((start, end))
}
