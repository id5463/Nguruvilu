//! The second lever on context size: thinning old tool results and images.

use crate::message::Message;

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

    /// The "aggressive" preset: what runs once a turn is close to compacting.
    ///
    /// Near the threshold the question stops being "how much of this result does
    /// the model still need" and becomes "which of these can we keep at all".
    /// Two thousand characters of head and no tail keeps the command and the
    /// start of its output — enough to know what happened — while a tool call's
    /// output is exactly the thing that grows without bound.
    pub fn aggressive() -> Self {
        Self {
            enabled: true,
            head_chars: 300,
            tail_chars: 0,
            keep_recent: 2,
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
