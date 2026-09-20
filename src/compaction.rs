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

use serde::{Deserialize, Serialize};

use crate::llm::LlmClient;
use crate::message::Message;

/// The kernel's record of one compaction.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Compaction {
    /// How many leading messages the summary stands in for.
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

/// Which leading messages to summarize.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Plan {
    /// Number of leading messages to replace with a summary.
    pub summarize: usize,
    /// Number of trailing messages kept verbatim.
    pub keep: usize,
}

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

    if split == 0 {
        // Nothing left to summarize, or no position where the remainder stands
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
        "Summarize the conversation below so it can continue without the original messages.\n\
         \n\
         Preserve, in this order:\n\
         1. What the user asked for, including any constraints or preferences they stated.\n\
         2. What was already done — files created or changed, commands run, and their outcomes.\n\
         3. Facts discovered that would be expensive to rediscover (paths, versions, error causes).\n\
         4. What remains to be done.\n\
         \n\
         Be specific and concrete. Keep file paths, identifiers, and numbers exactly as they \
         appear. Do not add commentary about the summarization itself. Write prose, not a \
         list of pleasantries.\n\
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
            user("please read it"),
            assistant_calling("read", "c1"),
            tool_result("c1"),
            user("thanks"),
        ];
        let plan = plan(&messages, 2).expect("a plan");

        // The split moved one earlier, so the call and its result stay together
        // in the kept region.
        assert_eq!(plan.summarize, 1);
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
        assert!(text.contains("What remains to be done"));
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
}
