//! How a summary is produced: the seam a plugin replaces.

use crate::llm::LlmClient;
use crate::message::Message;

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
        "{transcript}\n\n---\n\n{}",
        SUMMARY_INSTRUCTION
    ))]
}

/// What the summarizer is asked for: codex's four-bullet handoff, verbatim in
/// substance.
///
/// The six fixed sections this replaced were mine, and they were wrong for the
/// model: a rigid template spends output on headings and invites a `(none)` stub
/// per empty section, which is the opposite of what a summary that *is* the
/// memory needs. A competent model already knows what a handoff contains; the
/// instruction only has to name the four things that matter and get out of the
/// way. The one addition to codex's wording is the closing sentence: this turn
/// exists to produce text, and a tool call here means the turn produced nothing.
const SUMMARY_INSTRUCTION: &str = "You are performing a CONTEXT CHECKPOINT \
COMPACTION. Create a handoff summary for another LLM that will resume the task.\n\
\n\
Include:\n\
- Current progress and key decisions made\n\
- Important context, constraints, or user preferences\n\
- What remains to be done (clear next steps)\n\
- Any critical data, examples, or references needed to continue\n\
\n\
Be concise, structured, and focused on helping the next LLM seamlessly continue the \
work. Preserve exact paths, commands, identifiers and numbers. Do not call any tools; \
reply with the summary only.";

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
    // Two attempts, because the failure this guards against is a *decision* by
    // the model, not a bug: measured on a real route, two of four summaries came
    // back as a tool call instead of a summary, and the text before the call was
    // stored as if it were the whole thing (1,005 characters standing in for
    // 2,160 messages). The same shape covers an empty reply, a reply cut at the
    // token cap, and a refusal — none of them is a summary, and none of them may
    // be written to disk as one.
    let mut complaint = String::from("no attempt ran");
    for attempt in 1..=2 {
        let mut rx = client.chat_stream(&request, &[]).await?;
        let mut text = String::new();
        let mut reason = String::new();
        while let Some(event) = rx.recv().await {
            match event? {
                crate::llm::LlmEvent::TextDelta(delta) => text.push_str(&delta),
                crate::llm::LlmEvent::Finished { reason: why } => {
                    reason = why.unwrap_or_default().to_ascii_lowercase();
                    break;
                }
                _ => {}
            }
        }

        let accepted = summarizer.accept(&text);
        let cut = reason.contains("length");
        let sidetracked = reason.contains("tool");
        if !accepted.trim().is_empty() && !cut && !sidetracked {
            return Ok(accepted);
        }
        complaint = if accepted.trim().is_empty() {
            format!("attempt {attempt}: no text at all (finish reason '{reason}')")
        } else {
            format!(
                "attempt {attempt}: {} characters, then finish reason '{reason}'",
                accepted.chars().count()
            )
        };
    }
    Err(anyhow::anyhow!(
        "refusing to compact: the summarizer produced no usable summary ({complaint}). \
         The history is kept as it was."
    ))
}
