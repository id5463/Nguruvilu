//! Compaction: reducing history when it approaches the window.
//!
//! Built into the kernel, not a plugin, because a conversation that cannot be
//! continued is a broken kernel rather than a missing feature. What *is* a
//! seam is how the summary is written — see [`Summarizer`].
//!
//! # What the kernel guarantees
//!
//! Two invariants hold no matter which summarizer is installed, because the
//! kernel owns the mechanics:
//!
//! * **Tool pairs stay together.** A `tool` result is only valid next to the
//!   assistant message that requested it. Splitting them produces a request the
//!   provider rejects, so the split point is always moved to a user message.
//! * **The newest messages are never touched.** Compaction only ever removes a
//!   leading run of messages; what the user just said survives verbatim.
//!
//! # How it is recorded
//!
//! A compaction is written to the session as its own record rather than by
//! rewriting the file. The store is append-only precisely so a crash cannot
//! corrupt it, and rewriting would give that up. On load, a compaction record
//! replaces the run it covers, so a reloaded session matches what the model saw.
//!
//! # How the record finds its run again
//!
//! A count alone — "the first N messages" — says where the run *started*, not
//! which messages it held. N is exactly what a missing line changes, and a
//! session file can lose a line for reasons this program never sees: an
//! interrupted write, a hand edit, a merge between machines, an unparsable line
//! skipped by the reader. So a record also stores [`anchors_of`] the run it
//! replaced, and the store finds the span by those ids. When they are gone, the
//! store says so instead of folding at the old offset. The count stays as well:
//! it is the fallback for records written before anchors existed, and it is the
//! check the anchors are measured against.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::llm::LlmClient;
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
    /// The marker that prefixes a summary message, so a reader can tell a
    /// summary from something the user actually typed.
    pub const MARKER: &'static str = "[summary of earlier conversation]";

    /// The message that stands in for the summarized run.
    pub fn message(&self) -> Message {
        Message::user(format!("{}\n{}", Self::MARKER, self.summary))
    }

    /// Whether a message is a compaction summary.
    pub fn is_summary(message: &Message) -> bool {
        message.text().starts_with(Self::MARKER)
    }
}

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

/// Build the request that asks the model for a summary.
///
/// The transcript is rendered as plain text rather than replayed as messages:
/// a summary request is a fresh, single-purpose question, and replayed tool
/// calls would drag their schemas along for no benefit.
pub fn summary_request(history: &[Message]) -> Vec<Message> {
    let mut transcript = String::new();
    for message in history {
        let who = match message.role {
            crate::message::Role::System => "system",
            crate::message::Role::User => "user",
            crate::message::Role::Assistant => "assistant",
            crate::message::Role::Tool => "tool",
        };
        for call in &message.tool_calls {
            transcript.push_str(&format!("[{who}] called {}({})\n", call.name, call.arguments));
        }
        let text = message.text();
        if !text.is_empty() {
            transcript.push_str(&format!("[{who}] {text}\n"));
        }
    }

    vec![Message::user(format!(
        "Summarize the conversation below so work can continue without the original messages.\n\
         \n\
         Write it as six fixed sections, in this order, with these headings:\n\
         1. 「目标」— what the user asked for, with every constraint and preference they \
         stated, quoted where the wording matters.\n\
         2. 「已完成」— what was actually done: files created or changed (exact paths), \
         commands run and their outcomes, accounts/URLs/credentials that were created or used.\n\
         3. 「关键事实」— anything expensive to rediscover: paths, versions, error causes, \
         endpoints, prices, IDs. Copy them verbatim; never paraphrase a path, a command or a \
         number.\n\
         4. 「未完成」— what remains, each item with its concrete next action.\n\
         5. 「坑」— mistakes and dead ends already tried, so they are not repeated.\n\
         6. 「下一步」— the single next action to take when work resumes.\n\
         \n\
         Rules: specific beats general; file paths, commands, identifiers, URLs and numbers \
         travel verbatim; when a fact was stated more than once, keep the latest version; do \
         not describe the summarization itself, do not add pleasantries, do not invent \
         anything the transcript does not contain.\n\
         \n\
         Length: this summary is all that survives, so it has to be as long as the work it \
         stands in for. A long conversation deserves a long summary — tens of thousands of \
         characters is normal, and one that fits on a screen has thrown away the reason it \
         exists. Never shorten it by dropping a fact; shorten the prose around the facts \
         instead.\n\
         \n\
         Conversation:\n\
         {transcript}"
    ))]
}

/// How a summary is produced.
///
/// The kernel ships [`ModelSummarizer`]. A plugin providing the `compaction`
/// service replaces it — a different prompt, a different model, or a summarizer
/// that calls an external service.
pub trait Summarizer: Send + Sync {
    /// A name for diagnostics.
    fn name(&self) -> &str;

    /// The request to send when summarizing `history`.
    fn request(&self, history: &[Message]) -> Vec<Message>;

    /// Model to summarize with; empty or `None` means the session's own model.
    ///
    /// Summarizing is a good place to spend less than the main conversation, so
    /// this is worth being able to point elsewhere.
    fn model(&self) -> Option<&str> {
        None
    }

    /// Clean up the model's reply before it is stored.
    fn accept(&self, reply: &str) -> String {
        reply.trim().to_string()
    }
}

/// The built-in summarizer: ask the same model to summarize the transcript.
#[derive(Debug, Clone, Default)]
pub struct ModelSummarizer {
    /// Optional model override, so summaries can run on something cheaper than
    /// the main route. Empty means the session's own model.
    pub model: String,
}

impl Summarizer for ModelSummarizer {
    fn name(&self) -> &str {
        "model"
    }

    fn request(&self, history: &[Message]) -> Vec<Message> {
        summary_request(history)
    }

    fn model(&self) -> Option<&str> {
        if self.model.trim().is_empty() {
            None
        } else {
            Some(&self.model)
        }
    }

    fn accept(&self, reply: &str) -> String {
        // Some models wrap a summary in a preamble or a code fence; stripping
        // the fence keeps the stored text clean.
        let trimmed = reply.trim();
        let without_fence = trimmed
            .strip_prefix("```")
            .and_then(|rest| rest.split_once('\n').map(|(_, body)| body))
            .and_then(|body| body.strip_suffix("```"))
            .map(str::trim)
            .unwrap_or(trimmed);
        without_fence.to_string()
    }
}

/// Run a summarization request to completion and return the text.
///
/// Uses the streaming call because that is the only call the client offers, and
/// collects the deltas. Summaries are short, so nothing is gained by streaming
/// them anywhere.
pub async fn summarize(
    client: &LlmClient,
    summarizer: &dyn Summarizer,
    history: &[Message],
) -> anyhow::Result<String> {
    let request = summarizer.request(history);
    let mut rx = client.chat_stream(&request, &[]).await?;

    let mut text = String::new();
    while let Some(event) = rx.recv().await {
        match event? {
            crate::llm::LlmEvent::TextDelta(delta) => text.push_str(&delta),
            crate::llm::LlmEvent::Finished { .. } => break,
            _ => {}
        }
    }

    let accepted = summarizer.accept(&text);
    if accepted.trim().is_empty() {
        return Err(anyhow::anyhow!(
            "the summarizer returned nothing; refusing to replace history with an empty summary"
        ));
    }
    Ok(accepted)
}

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
/// session file's records also store [`anchors_of`] the run they replaced, and
/// [`crate::session`] replays those through [`locate_span`] rather than here.
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

/// How much of an old tool result stays in the request.
///
/// This is the second lever on context size, and the one that matters in long
/// sessions. Compaction (above) rewrites the *front* of a conversation once it
/// passes a fraction of the window; pruning rewrites the *middle* continuously
/// and depends on no window at all.
///
/// The measured case it exists for: 2,749 steps, ~36万 tokens per step, and
/// 3,245 tool results whose combined size was almost exactly that per-step
/// figure. Nothing was wrong with the window — the history was simply never
/// thinned, so every step re-read every old `bash` echo.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrunePolicy {
    /// Whether pruning runs at all.
    pub enabled: bool,
    /// Characters of the head of an old result that stay.
    pub head_chars: usize,
    /// Characters of the tail of an old result that stay.
    pub tail_chars: usize,
    /// How many of the newest tool results stay untouched.
    pub keep_recent: usize,
    /// How many of the newest inline images stay in the request.
    ///
    /// The heavier dial by far: one screenshot is hundreds of kilobytes of
    /// base64, and it travels on *every* later step. The session this exists for
    /// carried 62.9 MB of inline images against 2.0 MB of tool text.
    pub keep_images: usize,
}

impl Default for PrunePolicy {
    /// The "standard" preset: the newest eight results are exact, older ones
    /// keep a readable head and tail, and every one of them keeps its pointer to
    /// the spilled full output. Two of the newest images stay.
    fn default() -> Self {
        Self {
            enabled: true,
            head_chars: 4_000,
            tail_chars: 1_000,
            keep_recent: 8,
            keep_images: 2,
        }
    }
}

impl PrunePolicy {
    /// The "loose" preset: exactly what this kernel did before pruning existed.
    pub fn loose() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }

    /// The "minimal" preset: small windows on old results.
    pub fn minimal() -> Self {
        Self {
            enabled: true,
            head_chars: 1_000,
            tail_chars: 256,
            keep_recent: 4,
            keep_images: 1,
        }
    }

    /// The "pointer only" preset: an old result becomes one line naming its file.
    pub fn pointer_only() -> Self {
        Self {
            enabled: true,
            head_chars: 0,
            tail_chars: 0,
            keep_recent: 1,
            keep_images: 0,
        }
    }

    /// Whether this policy would change anything at all.
    pub fn is_noop(&self) -> bool {
        !self.enabled
    }
}

/// Thin out old tool results, leaving everything else alone.
///
/// The newest [`PrunePolicy::keep_recent`] tool results are returned untouched —
/// they are the ones the model is working with. Older ones keep a head, a tail,
/// and the `[full output: …]` pointer that `crate::tools::spill` put there, so
/// thinning never removes the way back to the text.
///
/// Nothing is mutated and nothing is stored: the caller gets a view for one
/// request. The session file keeps every original, which is what makes the knob
/// safe to turn back up, and what keeps an audit honest.
pub fn prune_tool_results(messages: &[Message], policy: &PrunePolicy) -> Vec<Message> {
    if policy.is_noop() {
        return messages.to_vec();
    }
    let tool_indices: Vec<usize> = messages
        .iter()
        .enumerate()
        .filter(|(_, message)| message.role == crate::message::Role::Tool)
        .map(|(index, _)| index)
        .collect();
    if tool_indices.len() <= policy.keep_recent {
        return messages.to_vec();
    }
    let mut pruned = messages.to_vec();
    for index in tool_indices
        .iter()
        .take(tool_indices.len() - policy.keep_recent)
    {
        let content = pruned[*index].content.clone().unwrap_or_default();
        pruned[*index].content = Some(shrink(&content, policy));
    }
    pruned
}

/// Drop old inline images from the request, keeping the newest few.
///
/// A `data:` URI is the bytes themselves, base64-encoded, in every request that
/// follows. Screenshots are also the least reproducible thing in a session — the
/// page has usually moved on — so the rule here is: keep the newest
/// [`PrunePolicy::keep_images`], and replace each older one with a note naming
/// the file where its bytes were spilled.
///
/// Two things are never touched: a remote `https://` URL (it costs a few
/// characters, not a payload), and an image whose bytes were never spilled —
/// dropping those would be irreversible, so they stay.
pub fn prune_images(messages: &[Message], policy: &PrunePolicy) -> Vec<Message> {
    if policy.is_noop() {
        return messages.to_vec();
    }
    if messages.iter().all(|message| message.images.is_empty()) {
        return messages.to_vec();
    }
    let mut budget = policy.keep_images;
    let mut replacements: Vec<Option<(Vec<crate::message::ImageAttachment>, Vec<String>)>> =
        vec![None; messages.len()];
    // Newest first, so the budget is spent on what the model is looking at now.
    for index in (0..messages.len()).rev() {
        let message = &messages[index];
        if message.images.is_empty() {
            continue;
        }
        let mut kept = Vec::with_capacity(message.images.len());
        let mut dropped = Vec::new();
        for image in message.images.iter().rev() {
            let droppable = image.url.starts_with("data:") && image.stored.is_some();
            if !droppable || budget > 0 {
                if droppable {
                    budget -= 1;
                }
                kept.push(image.clone());
            } else {
                dropped.push(image.stored.clone().unwrap_or_default());
            }
        }
        kept.reverse();
        if !dropped.is_empty() {
            replacements[index] = Some((kept, dropped));
        }
    }
    let mut pruned = messages.to_vec();
    for (index, replacement) in replacements.into_iter().enumerate() {
        let Some((kept, dropped)) = replacement else {
            continue;
        };
        let note = format!(
            "[已省略 {} 张旧图片（{}）— 需要时用 read 打开这些文件]",
            dropped.len(),
            dropped.join(", ")
        );
        pruned[index].images = kept;
        pruned[index].content = Some(match pruned[index].content.as_deref() {
            Some(text) if !text.is_empty() => format!("{text}\n{note}"),
            _ => note,
        });
    }
    pruned
}

/// One old tool result, reduced to what still earns its place.
///
/// Returns the text unchanged when thinning would not actually save anything:
/// a note and a pointer can be longer than the result they replace, and a
/// "thinned" message that grows is worse than no thinning at all.
fn shrink(text: &str, policy: &PrunePolicy) -> String {
    let pointer = crate::tools::spill::marker_of(text).unwrap_or("");
    let bytes = text.len();
    let with_pointer = |body: String| {
        if pointer.is_empty() {
            body
        } else {
            format!("{body}\n{pointer}")
        }
    };
    if policy.head_chars == 0 && policy.tail_chars == 0 {
        let first = text
            .lines()
            .find(|line| {
                !line.trim().is_empty() && !line.trim_start().starts_with(crate::tools::spill::MARKER)
            })
            .map(str::trim)
            .unwrap_or("");
        let first: String = first.chars().take(160).collect();
        let thinned = with_pointer(format!("[old tool output thinned: {bytes} chars · {first}]"));
        return if thinned.len() < text.len() { thinned } else { text.to_string() };
    }
    // The pointer is stripped before measuring so it cannot be counted twice,
    // and put back verbatim because it may be longer than the tail it would
    // otherwise have to survive inside.
    let body = match crate::tools::spill::marker_of(text) {
        Some(pointer) => text.replace(pointer, ""),
        None => text.to_string(),
    };
    let trimmed = body.trim_end();
    let chars: Vec<char> = trimmed.chars().collect();
    // Enough margin that the note and the pointer never make the result longer.
    if chars.len() <= policy.head_chars + policy.tail_chars + 200 {
        return with_pointer(trimmed.to_string());
    }
    let head: String = chars.iter().take(policy.head_chars).collect();
    let tail: String = chars
        .iter()
        .skip(chars.len().saturating_sub(policy.tail_chars))
        .collect();
    let omitted = chars.len() - policy.head_chars - policy.tail_chars;
    let thinned = with_pointer(format!("{head}\n… {omitted} chars thinned …\n{tail}"));
    if thinned.len() < text.len() {
        thinned
    } else {
        text.to_string()
    }
}

#[cfg(test)]
mod prune_tests {
    use super::*;
    use crate::message::{Message, Role};

    fn tool(text: &str) -> Message {
        Message::tool_result("call-1", "bash", text)
    }

    fn user(text: &str) -> Message {
        Message::user(text)
    }

    #[test]
    fn the_newest_results_are_never_touched() {
        let messages: Vec<Message> = (0..12)
            .map(|index| tool(&format!("result {index} {}", "x".repeat(50_000))))
            .collect();
        let policy = PrunePolicy::default();
        let pruned = prune_tool_results(&messages, &policy);
        assert_eq!(pruned.len(), messages.len(), "nothing is added or removed");
        for index in 4..12 {
            assert_eq!(
                pruned[index].content, messages[index].content,
                "result {index} is among the newest {} and must be exact",
                policy.keep_recent
            );
        }
        assert!(
            pruned[0].content.as_deref().unwrap().len() < messages[0].content.as_deref().unwrap().len(),
            "the oldest result is thinned"
        );
        assert!(
            pruned[0].content.as_deref().unwrap().contains("chars thinned"),
            "the cut says so"
        );
    }

    #[test]
    fn non_tool_messages_are_never_changed() {
        let mut messages = vec![user(&"y".repeat(9000))];
        for index in 0..10 {
            messages.push(tool(&format!("r{index} {}", "z".repeat(20_000))));
        }
        let pruned = prune_tool_results(&messages, &PrunePolicy::default());
        assert_eq!(pruned[0].content, messages[0].content, "the question stays whole");
        assert!(pruned[1].content.as_deref().unwrap().contains("chars thinned"));
    }

    #[test]
    fn the_pointer_to_the_full_output_survives() {
        let pointer = "[full output: /tmp/ngu/tool-output/s/bash-1.txt]";
        let body = format!("line one\n{}\n{}", "m".repeat(6000), pointer);
        let mut messages: Vec<Message> = (0..10).map(|index| tool(&format!("filler {index}"))).collect();
        messages[0] = tool(&body);
        let pruned = prune_tool_results(&messages, &PrunePolicy::default());
        let thinned = pruned[0].content.as_deref().unwrap();
        assert!(thinned.contains(pointer), "the way back is kept: {thinned}");
    }

    #[test]
    fn a_pointer_only_result_becomes_one_line() {
        let pointer = "[full output: /tmp/x.txt]";
        let body = format!("{}\n{pointer}", "q".repeat(5000));
        let mut messages: Vec<Message> = (0..10).map(|index| tool(&format!("filler {index}"))).collect();
        messages[0] = tool(&body);
        let pruned = prune_tool_results(&messages, &PrunePolicy::pointer_only());
        let thinned = pruned[0].content.as_deref().unwrap();
        assert!(thinned.starts_with("[old tool output thinned:"), "{thinned}");
        assert!(thinned.contains(pointer), "{thinned}");
        assert!(thinned.len() < 400, "one line, not a document: {}", thinned.len());
    }

    #[test]
    fn a_loose_policy_changes_nothing() {
        let messages: Vec<Message> = (0..20).map(|index| tool(&format!("r{index}"))).collect();
        let pruned = prune_tool_results(&messages, &PrunePolicy::loose());
        assert_eq!(pruned.len(), messages.len());
        for (before, after) in messages.iter().zip(pruned.iter()) {
            assert_eq!(before.content, after.content);
        }
    }

    #[test]
    fn a_short_history_is_left_alone() {
        let messages: Vec<Message> = (0..3).map(|index| tool(&format!("r{index}"))).collect();
        let pruned = prune_tool_results(&messages, &PrunePolicy::default());
        assert_eq!(pruned[0].content, messages[0].content);
    }

    #[test]
    fn a_result_shorter_than_the_windows_is_kept_whole() {
        let mut messages: Vec<Message> = (0..10).map(|index| tool(&format!("r{index}"))).collect();
        messages[0] = tool("short");
        let pruned = prune_tool_results(&messages, &PrunePolicy::default());
        assert_eq!(pruned[0].content.as_deref(), Some("short"));
    }

    #[test]
    fn the_presets_are_ordered_from_most_to_least_detail() {
        let loose = PrunePolicy::loose();
        let standard = PrunePolicy::default();
        let minimal = PrunePolicy::minimal();
        let pointer = PrunePolicy::pointer_only();
        assert!(loose.is_noop());
        assert!(!standard.is_noop());
        assert!(minimal.head_chars < standard.head_chars);
        assert_eq!(pointer.head_chars + pointer.tail_chars, 0);
    }

    #[test]
    fn the_role_is_preserved_so_providers_still_accept_the_request() {
        let messages: Vec<Message> = (0..10).map(|index| tool(&format!("r{index}"))).collect();
        let pruned = prune_tool_results(&messages, &PrunePolicy::default());
        for message in pruned {
            assert_eq!(message.role, Role::Tool);
            assert_eq!(message.tool_call_id.as_deref(), Some("call-1"));
        }
    }

    /// An inline image whose bytes were spilled (the droppable kind).
    fn image(message: usize) -> Message {
        Message::user_with_images(
            "看看这个页面",
            vec![crate::message::ImageAttachment::url(format!(
                "data:image/png;base64,{}",
                "a".repeat(4_000)
            ))
            .stored_at(format!("/tmp/img-{message}.png"))],
        )
    }

    /// An inline image with no spilled copy: dropping it would lose it.
    fn unspillable(message: usize) -> Message {
        Message::user_with_images(
            "看看这个页面",
            vec![crate::message::ImageAttachment::url(format!(
                "data:image/png;base64,{}",
                "a".repeat(4_000)
            ))
            .labelled(format!("img-{message}"))],
        )
    }

    fn remote(message: usize) -> Message {
        Message::user_with_images(
            "外链图",
            vec![crate::message::ImageAttachment::url(format!(
                "https://example.com/{message}.png"
            ))],
        )
    }

    #[test]
    fn only_the_newest_images_stay_in_the_request() {
        let messages: Vec<Message> = (0..6).map(image).collect();
        let pruned = prune_images(&messages, &PrunePolicy::default());
        assert_eq!(pruned.len(), messages.len(), "no message is dropped");
        assert_eq!(pruned[5].images.len(), 1, "the newest stays");
        assert_eq!(pruned[4].images.len(), 1, "and the one before it");
        for index in 0..4 {
            assert!(pruned[index].images.is_empty(), "older image {index} is dropped");
            let content = pruned[index].content.as_deref().unwrap();
            assert!(content.contains("已省略 1 张旧图片"), "{content}");
            assert!(
                content.contains(&format!("/tmp/img-{index}.png")),
                "the note names the file: {content}"
            );
        }
    }

    #[test]
    fn the_text_of_an_image_message_survives_the_drop() {
        let messages = vec![image(0), image(1), image(2), image(3), image(4)];
        let pruned = prune_images(&messages, &PrunePolicy {
            keep_images: 1,
            ..PrunePolicy::default()
        });
        assert!(
            pruned[0].content.as_deref().unwrap().starts_with("看看这个页面"),
            "what the model was told stays; what it was shown goes: {:?}",
            pruned[0].content
        );
    }

    #[test]
    fn an_unspillable_image_is_never_dropped() {
        // An image whose bytes were never written to disk cannot be re-read, so
        // dropping it would be irreversible — it keeps its place instead.
        let messages: Vec<Message> = (0..6)
            .map(|index| if index == 0 { unspillable(index) } else { image(index) })
            .collect();
        let pruned = prune_images(&messages, &PrunePolicy {
            keep_images: 1,
            ..PrunePolicy::default()
        });
        assert_eq!(pruned[0].images.len(), 1, "the unspillable one stays");
        assert_eq!(pruned[5].images.len(), 1, "and so does the newest");
    }

    #[test]
    fn a_remote_url_is_left_in_place() {
        let messages: Vec<Message> = (0..6).map(remote).collect();
        let pruned = prune_images(&messages, &PrunePolicy {
            keep_images: 0,
            ..PrunePolicy::default()
        });
        for message in pruned {
            assert_eq!(message.images.len(), 1, "a URL costs characters, not bytes");
        }
    }

    #[test]
    fn a_session_without_images_is_returned_untouched() {
        let messages: Vec<Message> = (0..5).map(|index| tool(&format!("r{index}"))).collect();
        let pruned = prune_images(&messages, &PrunePolicy::pointer_only());
        assert_eq!(pruned.len(), messages.len());
        assert_eq!(pruned[0].content, messages[0].content);
    }

    #[test]
    fn a_loose_policy_drops_no_image() {
        let messages: Vec<Message> = (0..6).map(image).collect();
        let pruned = prune_images(&messages, &PrunePolicy::loose());
        for message in pruned {
            assert_eq!(message.images.len(), 1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::{Role, ToolCall};

    fn user(text: &str) -> Message {
        Message::user(text)
    }

    fn assistant_calling(name: &str, id: &str) -> Message {
        Message::assistant_tools(
            vec![ToolCall {
                id: id.into(),
                name: name.into(),
                arguments: "{}".into(),
            }],
            None,
        )
    }

    fn tool_result(id: &str) -> Message {
        Message::tool_result(id, "read", "contents")
    }

    #[test]
    fn nothing_is_planned_when_history_is_short() {
        let messages = vec![user("a"), user("b")];
        assert_eq!(plan(&messages, 8), None);
    }

    #[test]
    fn a_short_history_relative_to_the_keep_target_is_untouched() {
        let messages: Vec<Message> = (0..4).map(|i| user(&format!("m{i}"))).collect();
        assert_eq!(plan(&messages, 4), None);
    }

    #[test]
    fn the_split_lands_on_a_user_message() {
        let messages = vec![
            user("one"),
            user("two"),
            user("three"),
            user("four"),
        ];
        let plan = plan(&messages, 2).expect("a plan");
        assert_eq!(plan.summarize, 2);
        assert_eq!(plan.keep, 2);
        assert_eq!(messages[plan.summarize].role, Role::User);
    }

    #[test]
    fn a_tool_pair_is_never_split() {
        // keep_recent = 2 would land on the tool result, which cannot begin a
        // kept region.
        let messages = vec![
            user("first"),
            user("please read it"),
            assistant_calling("read", "c1"),
            tool_result("c1"),
            user("thanks"),
        ];
        let plan = plan(&messages, 2).expect("a plan");

        // The split moved one earlier, so the call and its result stay together
        // in the kept region.
        assert_eq!(plan.summarize, 2);
        assert_eq!(plan.keep, 3);
        assert_ne!(
            messages[plan.summarize].role,
            Role::Tool,
            "the kept region must not begin with a tool result"
        );
    }

    #[test]
    fn a_kept_region_may_begin_with_an_assistant_message() {
        // An assistant message followed by its tool result is a valid suffix:
        // the call is present, so the result has something to belong to.
        let messages = vec![
            user("one"),
            user("two"),
            assistant_calling("read", "c1"),
            tool_result("c1"),
        ];
        let plan = plan(&messages, 2).expect("a plan");
        assert_eq!(plan.summarize, 2);
        assert_eq!(messages[plan.summarize].role, Role::Assistant);
    }

    #[test]
    fn a_history_that_is_all_tool_results_is_left_alone() {
        // Every message is a tool result with no assistant call in front: there
        // is no position where the remainder stands on its own.
        let messages = vec![tool_result("c1"), tool_result("c2")];
        assert_eq!(plan(&messages, 1), None);
    }

    #[test]
    fn applying_a_plan_replaces_the_leading_run_with_one_message() {
        let messages = vec![user("one"), user("two"), user("three"), user("four")];
        let plan = plan(&messages, 2).expect("a plan");
        let next = apply(&messages, plan, "they said one and two");

        assert_eq!(next.len(), 3, "one summary plus two kept");
        assert!(Compaction::is_summary(&next[0]));
        assert!(next[0].text().contains("they said one and two"));
        assert_eq!(next[1].text(), "three");
        assert_eq!(next[2].text(), "four");
    }

    #[test]
    fn the_newest_messages_are_kept_verbatim() {
        let messages: Vec<Message> = (0..12).map(|i| user(&format!("m{i}"))).collect();
        let plan = plan(&messages, 5).expect("a plan");
        let next = apply(&messages, plan, "older stuff");

        for i in 0..5 {
            let expected = format!("m{}", 12 - 5 + i);
            assert_eq!(next[next.len() - 5 + i].text(), expected);
        }
    }

    #[test]
    fn a_summary_is_recognisable() {
        let compaction = Compaction {
            replaced: 3,
            summary: "did some things".into(),
        };
        let message = compaction.message();
        assert!(Compaction::is_summary(&message));
        assert!(!Compaction::is_summary(&user("hello")));
    }

    #[test]
    fn replaying_compactions_reproduces_what_the_model_saw() {
        // The file holds the originals plus the record that replaced them.
        let originals = vec![
            user("one"),
            user("two"),
            user("three"),
            user("four"),
            user("five"),
        ];
        let compactions = vec![Compaction {
            replaced: 3,
            summary: "one two three".into(),
        }];

        let history = replay(originals, &compactions);
        assert_eq!(history.len(), 3);
        assert!(Compaction::is_summary(&history[0]));
        assert_eq!(history[1].text(), "four");
        assert_eq!(history[2].text(), "five");
    }

    #[test]
    fn replaying_stacked_compactions_works() {
        // A long session compacts twice; both records are in the file.
        let originals: Vec<Message> = (0..10).map(|i| user(&format!("m{i}"))).collect();
        let compactions = vec![
            Compaction {
                replaced: 4,
                summary: "first summary".into(),
            },
            Compaction {
                replaced: 3,
                summary: "second summary".into(),
            },
        ];

        let history = replay(originals, &compactions);
        // 10 → (1 summary + 6 kept) = 7 → (1 summary + 4 kept) = 5.
        assert_eq!(history.len(), 5);
        assert!(history[0].text().contains("second summary"));
    }

    #[test]
    fn a_compaction_record_that_does_not_fit_is_ignored() {
        let originals = vec![user("one")];
        let compactions = vec![Compaction {
            replaced: 99,
            summary: "impossible".into(),
        }];
        assert_eq!(replay(originals, &compactions).len(), 1);
    }

    #[test]
    fn the_summary_request_includes_the_transcript_and_the_instructions() {
        let history = vec![
            user("change the port to 8080"),
            assistant_calling("edit", "c1"),
            tool_result("c1"),
        ];
        let request = summary_request(&history);
        assert_eq!(request.len(), 1);

        let text = request[0].text();
        assert!(text.contains("change the port to 8080"), "user request kept");
        assert!(text.contains("edit"), "tool call kept");
        assert!(text.contains("Summarize the conversation below"));
        // The instructions are a fixed structure now: the remaining work and the
        // single next action have sections of their own, so a resumed conversation
        // knows what to do instead of reading prose for it.
        assert!(text.contains("「未完成」"), "{text}");
        assert!(text.contains("「下一步」"), "{text}");
    }

    #[test]
    fn the_default_summarizer_strips_a_code_fence() {
        let summarizer = ModelSummarizer::default();
        assert_eq!(summarizer.accept("```\nplain text\n```"), "plain text");
        assert_eq!(summarizer.accept("  plain text  "), "plain text");
    }

    #[test]
    fn the_default_summarizer_is_named_for_diagnostics() {
        assert_eq!(ModelSummarizer::default().name(), "model");
    }

    #[test]
    fn a_run_too_small_to_be_worth_summarizing_is_left_alone() {
        // Summarizing one or two messages costs a model call and says little
        // more than they did.
        let two = vec![user("one"), user("two")];
        assert_eq!(plan(&two, 1), None, "one message is not worth a summary");

        let four = vec![user("one"), user("two"), user("three"), user("four")];
        // keep_recent = 3 would summarize exactly one; below the floor.
        assert_eq!(plan(&four, 3), None);
    }

    // ------------------------------------------------------- anchors

    #[test]
    fn an_anchor_names_a_message_by_what_it_is_not_by_where_it_is() {
        let one = user("one");
        // The same message read back from a file is the same message, so the
        // anchor has to be equal: that is the whole point of it.
        let same = user("one");
        assert_eq!(anchor_of(&one), anchor_of(&same));
        assert_ne!(anchor_of(&one), anchor_of(&user("two")));
        assert_ne!(anchor_of(&one), anchor_of(&assistant_calling("read", "c1")));
        assert_ne!(
            anchor_of(&assistant_calling("read", "c1")),
            anchor_of(&assistant_calling("write", "c1")),
            "the tool a call names is part of what the message is"
        );
        assert_eq!(anchor_of(&one).len(), ANCHOR_HEX);
    }

    #[test]
    fn the_anchors_of_a_run_name_its_two_ends() {
        let run = vec![user("one"), user("two"), user("three")];
        let (first, last) = anchors_of(&run).expect("a run has two ends");
        assert_eq!(first, anchor_of(&run[0]));
        assert_eq!(last, anchor_of(&run[2]));

        assert!(anchors_of(&[]).is_none(), "an empty run names nothing");
    }

    #[test]
    fn an_intact_run_is_located_by_its_anchors() {
        let messages = vec![user("one"), user("two"), user("three"), user("four")];
        let (first, last) = anchors_of(&messages[..2]).expect("anchors");
        assert_eq!(
            locate_span(&messages, 2, &first, &last),
            Ok::<(usize, usize), String>((0, 1))
        );

        // A run that does not begin at the head is found where it actually
        // sits; a count cannot express that at all, which is the case anchors
        // exist for.
        let (first, last) = anchors_of(&messages[2..]).expect("anchors");
        assert_eq!(
            locate_span(&messages, 2, &first, &last),
            Ok::<(usize, usize), String>((2, 3))
        );
    }

    #[test]
    fn a_run_that_lost_its_last_message_is_refused_rather_than_guessed() {
        let (first, last) = anchors_of(&[user("one"), user("two")]).expect("anchors");
        let now = vec![user("one")];
        let refused = locate_span(&now, 2, &first, &last).expect_err("must refuse");
        assert!(refused.contains("no longer in the file"), "{refused}");
        assert!(refused.contains("not applied"), "{refused}");
    }

    #[test]
    fn a_run_that_grew_is_refused_rather_than_mis_applied() {
        // Both anchors are still present, so the span can be found — but a
        // message was inserted inside it, and folding this span would delete a
        // message the summary was never written for.
        let (first, last) = anchors_of(&[user("one"), user("two")]).expect("anchors");
        let now = vec![user("one"), user("inserted"), user("two"), user("three")];
        let refused = locate_span(&now, 2, &first, &last).expect_err("must refuse");
        assert!(refused.contains("not the 2 it recorded"), "{refused}");
    }
}
