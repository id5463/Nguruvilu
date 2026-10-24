//! Exporting a stored session into a file somebody else can read.
//!
//! A session on disk is a JSONL log written for this program to replay: a
//! header line, one line per message, and a ledger beside it. That is the right
//! shape for a store and the wrong one for a person or another agent who was
//! handed the file, so an export restates the same conversation in a shape that
//! stands on its own, together with the facts that make it interpretable —
//! which model ran, in which directory, and what the session cost in tokens.
//!
//! Three rules decide the details below.
//!
//! * **Nothing is cut without saying so.** A long tool result is capped, because
//!   an export that is mostly one command's output is not shareable, but a
//!   shortened record that does not admit to being shortened is a lie. Every cut
//!   carries both numbers: how much was kept and how much there was.
//! * **A summary is not a reply.** A compaction folds an earlier run of messages
//!   into one message marked with [`Compaction::MARKER`]. Rendering that as an
//!   ordinary turn would tell a reader the model said something it never said, so
//!   it gets its own section and an explanation of where it came from.
//! * **Rendering does not print.** [`render`] returns a `String`; where those
//!   bytes go — a terminal, a file, a pipe — is the caller's decision, which is
//!   also what makes the module testable without a child process.

use std::fmt::Write as _;

use serde::Serialize;

use crate::compaction::Compaction;
use crate::message::{Message, Role, ToolCall};
use crate::session::{Session, SessionTotals, TurnStats};

/// How many characters of one tool result an export keeps by default.
///
/// Long enough for an ordinary listing or error trace, short enough that a
/// session which read one large file still exports to something a person can
/// read. Cutting at all is a compromise, which is why every cut is reported.
pub const DEFAULT_TOOL_OUTPUT_CAP: usize = 8_000;

/// The shape an export is written in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    /// Readable text: a title, a fact block, then the conversation with role
    /// headings and fenced tool traffic.
    Markdown,
    /// One JSON object, for a script or another agent.
    Json,
}

/// What an export is rendered with.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExportOptions {
    /// Characters kept from one tool result. Zero keeps all of it.
    ///
    /// Zero means "no cap" rather than "keep nothing": a caller who asks for the
    /// whole record should get the whole record, not an export of empty blocks.
    pub tool_output_cap: usize,
    /// Which format to render.
    pub format: Format,
}

impl Default for ExportOptions {
    fn default() -> Self {
        Self {
            tool_output_cap: DEFAULT_TOOL_OUTPUT_CAP,
            format: Format::Markdown,
        }
    }
}

/// Render one session, with the totals its ledger summed to, as text.
///
/// Infallible on purpose: the session is already loaded, and rendering it is
/// formatting rather than I/O. The caller decides where the bytes go.
///
/// The result ends with exactly one newline whichever format was asked for, so
/// printing it to a terminal and writing it to a file produce the same bytes —
/// a difference of one byte between the two paths would be an export that does
/// not round-trip through a pipe.
pub fn render(session: &Session, totals: &SessionTotals, options: &ExportOptions) -> String {
    match options.format {
        Format::Markdown => markdown(session, totals, options.tool_output_cap),
        Format::Json => json(session, totals, options.tool_output_cap),
    }
}

// ------------------------------------------------------------------ markdown

fn markdown(session: &Session, totals: &SessionTotals, cap: usize) -> String {
    let mut out = String::new();
    // The id and the creation time go in the title, because an exported file
    // gets separated from its name and a reader then has no way to say which
    // session this was or when it happened.
    let _ = writeln!(out, "# Session {} (created {})", session.id, session.created_at);
    out.push('\n');

    let _ = writeln!(out, "- model: {}", session.model.as_deref().unwrap_or("(not recorded)"));
    let _ = writeln!(
        out,
        "- working directory: {}",
        session.cwd.as_deref().unwrap_or("(not recorded)")
    );
    let _ = writeln!(out, "- messages: {}", session.messages.len());
    let _ = writeln!(out, "- recorded turns: {}", totals.turns);
    let _ = writeln!(
        out,
        "- tokens: input {}, output {}, cached {}, cache-created {}, reasoning {}",
        totals.input, totals.output, totals.cached, totals.cache_created, totals.reasoning
    );
    // A percentage, because this is a line for a reader; the JSON export keeps
    // the fraction, which is the convention the stats command already prints.
    let _ = writeln!(
        out,
        "- cache hit ratio: {:.1}%",
        totals.cache_hit_ratio() * 100.0
    );
    out.push('\n');

    if session.messages.is_empty() {
        out.push_str("This session holds no messages.\n");
        return out;
    }

    for message in &session.messages {
        message_markdown(&mut out, message, cap);
    }
    out
}

fn message_markdown(out: &mut String, message: &Message, cap: usize) {
    // Checked before the role match: a compaction summary is stored as a `user`
    // message, so the ordinary path below would present it as something a person
    // typed. Only the roles a summary can actually have are considered, so a
    // tool result that happens to start with the same marker still renders as the
    // tool result it is.
    if matches!(message.role, Role::User | Role::Assistant) && Compaction::is_summary(message) {
        out.push_str("## summary of earlier conversation\n\n");
        out.push_str(
            "_A compaction replaced an earlier run of messages with the summary below. It was \
             written by the session's summarizer, not by the user and not as an ordinary reply to \
             the reader, and it is what the conversation continued from. Its opening marker line \
             is kept verbatim because it is part of the stored record._\n\n",
        );
        push_text(out, message.text());
        out.push('\n');
        return;
    }

    // `as_wire` is the same spelling the provider sees, so the heading and the
    // stored log agree on what a role is called.
    let _ = writeln!(out, "## {}\n", message.role.as_wire());

    let text = message.text();
    if text.is_empty() {
        if message.tool_calls.is_empty() {
            out.push_str("(empty message)\n\n");
        }
    } else if message.role == Role::Tool {
        tool_result_markdown(out, message, cap);
    } else {
        // Verbatim, uncapped: this is the conversation itself, and shortening
        // what a participant said would change the record rather than compress it.
        push_text(out, text);
        out.push('\n');
    }

    for call in &message.tool_calls {
        tool_call_markdown(out, call);
    }
}

fn tool_call_markdown(out: &mut String, call: &ToolCall) {
    let _ = writeln!(out, "### tool call `{}` (call `{}`)\n", call.name, call.id);
    // Fenced as JSON because that is what the arguments are: a raw JSON string
    // as the wire format carries it, parsed only when the call is executed.
    fenced(out, "json", &call.arguments);
    out.push('\n');
}

fn tool_result_markdown(out: &mut String, message: &Message, cap: usize) {
    let mut heading = String::from("### tool result");
    if let Some(name) = &message.name {
        let _ = write!(heading, " from `{name}`");
    }
    if let Some(id) = &message.tool_call_id {
        let _ = write!(heading, " (call `{id}`)");
    }
    let _ = writeln!(out, "{heading}\n");

    let (body, cut) = capped(message.text(), cap);
    fenced(out, "", &body);
    if let Some(note) = cut {
        out.push('\n');
        out.push_str(&note);
    }
    out.push('\n');
}

/// Append message text verbatim, ending the line so the next block starts fresh.
fn push_text(out: &mut String, text: &str) {
    out.push_str(text);
    if !text.ends_with('\n') {
        out.push('\n');
    }
}

// ---------------------------------------------------------------------- json

fn json(session: &Session, totals: &SessionTotals, cap: usize) -> String {
    let export = JsonExport {
        session: JsonSession {
            id: &session.id,
            created_at: &session.created_at,
            // Absent rather than null when the session records nothing: the
            // stored session omits these fields the same way, and a consumer
            // that sees no key cannot mistake a missing fact for an empty one.
            model: session.model.as_deref(),
            cwd: session.cwd.as_deref(),
        },
        totals: JsonTotals::of(totals),
        // `as_slice` rather than a generic `&Vec<_>`→`&[_]` coercion, so the
        // field is plainly a borrowed run of recorded rows, not a copy of them.
        turns: session.stats.as_slice(),
        messages: session
            .messages
            .iter()
            .map(|message| JsonMessage::of(message, cap))
            .collect(),
    };

    let mut text = match serde_json::to_string_pretty(&export) {
        Ok(text) => text,
        // Every type here is made of strings and integers, so this cannot happen
        // in practice. Returning an object that says what went wrong beats an
        // `unwrap` that would crash the command, or an empty string that would
        // look like an empty session.
        Err(error) => serde_json::json!({
            "error": format!("the export could not be serialized: {error}"),
        })
        .to_string(),
    };
    text.push('\n');
    text
}

/// The export as a whole. Field order is the order a reader wants them in:
/// which session this is, what it cost, then the transcript.
#[derive(Serialize)]
struct JsonExport<'a> {
    session: JsonSession<'a>,
    totals: JsonTotals,
    /// The ledger rows, serialized as the struct itself. Verbatim rather than
    /// re-spelled, so the export cannot drift from what the session recorded.
    turns: &'a [TurnStats],
    messages: Vec<JsonMessage<'a>>,
}

#[derive(Serialize)]
struct JsonSession<'a> {
    id: &'a str,
    created_at: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cwd: Option<&'a str>,
}

/// A session's usage, spelled exactly as `ngu stats` prints it.
///
/// The names are written out here rather than derived from [`SessionTotals`]
/// because they are a contract shared with that command and with the desktop
/// shell: a consumer reads `cache_created` from either frontend. `cache_hit_ratio`
/// is a fraction in `0..=1`, matching the stats convention, so no consumer has to
/// guess at the scale or repeat the division.
#[derive(Serialize)]
struct JsonTotals {
    turns: u64,
    steps: u64,
    tool_calls: u64,
    input: u64,
    output: u64,
    cached: u64,
    cache_created: u64,
    reasoning: u64,
    model_ms: u64,
    tools_ms: u64,
    cache_hit_ratio: f64,
}

impl JsonTotals {
    fn of(totals: &SessionTotals) -> Self {
        Self {
            turns: totals.turns,
            steps: totals.steps,
            tool_calls: totals.tool_calls,
            input: totals.input,
            output: totals.output,
            cached: totals.cached,
            cache_created: totals.cache_created,
            reasoning: totals.reasoning,
            model_ms: totals.model_ms,
            tools_ms: totals.tools_ms,
            cache_hit_ratio: totals.cache_hit_ratio(),
        }
    }
}

#[derive(Serialize)]
struct JsonMessage<'a> {
    /// The wire spelling — `user`, `assistant`, `tool`, `system`.
    role: &'static str,
    /// The message's text. Cut short only for a tool result, and only when the
    /// two fields below say so.
    text: String,
    /// The calls this message made, serialized as the struct: id, name, and the
    /// raw argument string the model produced.
    tool_calls: &'a [ToolCall],
    /// Present on a tool result, which is the only message that answers a call.
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<&'a str>,
    /// Present only when `text` was cut, so an absent marker always means the
    /// text is whole.
    #[serde(skip_serializing_if = "Option::is_none")]
    truncated: Option<bool>,
    /// How long the uncut text was, in characters, so a reader of the JSON can
    /// tell how much of the record is missing without opening the session file.
    #[serde(skip_serializing_if = "Option::is_none")]
    original_chars: Option<usize>,
}

impl<'a> JsonMessage<'a> {
    fn of(message: &'a Message, cap: usize) -> Self {
        // Only a tool result is capped. Assistant and user text is what was said,
        // and a shortened quotation of a conversation is a different conversation.
        let (text, original) = if message.role == Role::Tool {
            let original = message.text().chars().count();
            let (body, cut) = capped(message.text(), cap);
            (body, cut.map(|_| original))
        } else {
            (message.text().to_string(), None)
        };

        Self {
            role: message.role.as_wire(),
            text,
            tool_calls: message.tool_calls.as_slice(),
            tool_call_id: message.tool_call_id.as_deref(),
            truncated: original.map(|_| true),
            original_chars: original,
        }
    }
}

// ------------------------------------------------------------------- shared

/// Cut a tool result to the cap, and say what was cut.
///
/// Returns the body and, when something was removed, the note that admits it.
/// The note names both numbers, because "truncated" on its own does not tell a
/// reader whether a line was trimmed or everything after the first page was
/// dropped.
fn capped(text: &str, cap: usize) -> (String, Option<String>) {
    // Counted in characters rather than bytes: cutting mid-character would
    // produce text that is not valid UTF-8 for a reader, and a tool result with
    // non-ASCII in it is ordinary.
    let total = text.chars().count();
    if cap == 0 || total <= cap {
        return (text.to_string(), None);
    }
    let body: String = text.chars().take(cap).collect();
    let note = format!(
        "[cut {} of {} characters: this tool result is longer than the {} character export cap]\n",
        total - cap,
        total,
        cap
    );
    (body, Some(note))
}

/// Write a fenced code block whose fence the body cannot close.
///
/// A tool result often contains backticks — an export of source code, an error
/// message quoting markdown — and the usual three-backtick fence would end in the
/// middle of it, turning the rest of the record into prose and the reader's
/// understanding with it. The fence is therefore at least as long as the longest
/// run of backticks in the body, plus one.
fn fenced(out: &mut String, info: &str, body: &str) {
    let fence = "`".repeat(fence_length(body));
    out.push_str(&fence);
    out.push_str(info);
    out.push('\n');
    out.push_str(body);
    if !body.ends_with('\n') {
        out.push('\n');
    }
    out.push_str(&fence);
    out.push('\n');
}

fn fence_length(body: &str) -> usize {
    let mut longest = 0;
    let mut run = 0;
    for character in body.chars() {
        if character == '`' {
            run += 1;
            longest = longest.max(run);
        } else {
            run = 0;
        }
    }
    // Three is the shortest fence a markdown reader recognises as one.
    (longest + 1).max(3)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn session_with(messages: Vec<Message>) -> Session {
        let mut session = Session::new(Some("test-model".into()));
        session.cwd = Some("C:/work".into());
        session.messages = messages;
        session
    }

    fn turn(input: u64, cached: u64) -> TurnStats {
        TurnStats {
            ts: "2026-01-01T00:00:00.000Z".into(),
            steps: 2,
            tool_calls: 3,
            retries: 0,
            input,
            output: 10,
            cached,
            cache_created: 5,
            reasoning: 1,
            model_ms: 100,
            tools_ms: 20,
        }
    }

    #[test]
    fn markdown_names_the_session_and_counts_what_it_holds() {
        let session = session_with(vec![Message::user("hello"), Message::assistant("hi")]);
        let text = render(&session, &SessionTotals::default(), &ExportOptions::default());

        assert!(text.contains(&format!("# Session {} (created {}", session.id, session.created_at)));
        assert!(text.contains("- model: test-model"), "{text}");
        assert!(text.contains("- working directory: C:/work"), "{text}");
        assert!(text.contains("- messages: 2"), "{text}");
        assert!(text.contains("- recorded turns: 0"), "{text}");
        assert!(text.contains("## user"), "{text}");
        assert!(text.contains("## assistant"), "{text}");
        assert!(text.contains("hello"), "{text}");
        assert!(text.ends_with('\n'), "a rendered export ends with one newline");
    }

    #[test]
    fn a_tool_call_and_its_result_are_both_rendered() {
        let session = session_with(vec![
            Message::user("read it"),
            Message::assistant_tools(
                vec![ToolCall {
                    id: "c1".into(),
                    name: "read".into(),
                    arguments: "{\"path\":\"a.txt\"}".into(),
                }],
                None,
            ),
            Message::tool_result("c1", "read", "the contents"),
        ]);
        let text = render(&session, &SessionTotals::default(), &ExportOptions::default());

        assert!(text.contains("### tool call `read` (call `c1`)"), "{text}");
        assert!(text.contains("{\"path\":\"a.txt\"}"), "{text}");
        assert!(text.contains("### tool result from `read` (call `c1`)"), "{text}");
        assert!(text.contains("the contents"), "{text}");
    }

    #[test]
    fn a_capped_tool_result_says_how_much_was_cut() {
        let long = "x".repeat(50);
        let session = session_with(vec![Message::tool_result("c1", "read", long)]);
        let options = ExportOptions {
            tool_output_cap: 20,
            format: Format::Markdown,
        };
        let text = render(&session, &SessionTotals::default(), &options);

        assert!(text.contains("[cut 30 of 50 characters"), "{text}");
        assert!(text.contains("20 character export cap"), "{text}");
        assert!(!text.contains(&"x".repeat(50)), "the body itself is cut");
    }

    #[test]
    fn a_cap_of_zero_keeps_everything() {
        let long = "y".repeat(50);
        let session = session_with(vec![Message::tool_result("c1", "read", long.clone())]);
        let options = ExportOptions {
            tool_output_cap: 0,
            format: Format::Markdown,
        };
        let text = render(&session, &SessionTotals::default(), &options);

        assert!(text.contains(&long), "{text}");
        assert!(!text.contains("[cut"), "{text}");
    }

    #[test]
    fn a_summary_is_marked_rather_than_shown_as_an_ordinary_turn() {
        let summary = Compaction {
            replaced: 4,
            summary: "they discussed the port".into(),
        }
        .message();
        let session = session_with(vec![summary, Message::user("and then")]);
        let text = render(&session, &SessionTotals::default(), &ExportOptions::default());

        assert!(text.contains("## summary of earlier conversation"), "{text}");
        assert!(text.contains("they discussed the port"), "{text}");
        assert!(text.contains("compaction replaced an earlier run"), "{text}");
        // The summary must not be presented as an ordinary turn. The history
        // holds exactly one user message, after the summary, so that section has
        // to come later and there must be exactly one of it: a summary rendered
        // under `## user` would make this two.
        let summary_at = text
            .find("## summary of earlier conversation")
            .expect("a summary section");
        let user_at = text.find("## user").expect("the later user turn");
        assert!(summary_at < user_at, "{text}");
        assert_eq!(text.matches("## user").count(), 1, "{text}");
    }

    #[test]
    fn a_fence_is_longer_than_any_backtick_run_in_the_body() {
        let session = session_with(vec![Message::tool_result(
            "c1",
            "read",
            "a ``` fenced block inside the result",
        )]);
        let text = render(&session, &SessionTotals::default(), &ExportOptions::default());
        assert!(text.contains("````"), "the fence grew to hold the body: {text}");
    }

    #[test]
    fn json_carries_the_session_the_totals_the_turns_and_the_messages() {
        let mut session = session_with(vec![
            Message::user("hello"),
            Message::assistant_tools(
                vec![ToolCall {
                    id: "c1".into(),
                    name: "read".into(),
                    arguments: "{\"path\":\"a.txt\"}".into(),
                }],
                None,
            ),
            Message::tool_result("c1", "read", "the contents"),
        ]);
        session.stats = vec![turn(100, 25)];

        // Summed by the kernel rather than by the test, so what the export
        // reports is what the rest of the program would report for this ledger.
        let totals = crate::session::totals_of(&session);
        let options = ExportOptions {
            tool_output_cap: DEFAULT_TOOL_OUTPUT_CAP,
            format: Format::Json,
        };
        let text = render(&session, &totals, &options);
        let doc: serde_json::Value = serde_json::from_str(&text).expect("valid JSON");

        assert_eq!(doc["session"]["id"], session.id.as_str());
        assert_eq!(doc["session"]["model"], "test-model");
        assert_eq!(doc["session"]["cwd"], "C:/work");
        assert_eq!(doc["totals"]["turns"], 1);
        assert_eq!(doc["totals"]["cache_created"], 5);
        assert!((doc["totals"]["cache_hit_ratio"].as_f64().unwrap() - 0.25).abs() < 1e-9);
        assert_eq!(doc["turns"][0]["steps"], 2);
        let messages = doc["messages"].as_array().expect("messages");
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0]["role"], "user");
        assert_eq!(messages[0]["text"], "hello");
        assert_eq!(messages[1]["tool_calls"][0]["name"], "read");
        assert_eq!(messages[2]["tool_call_id"], "c1");
        // Nothing was cut, so no message carries a truncation marker.
        assert!(messages[2].get("truncated").is_none(), "{}", messages[2]);
    }

    #[test]
    fn json_marks_a_cut_tool_result() {
        let session = session_with(vec![Message::tool_result("c1", "read", "z".repeat(40))]);
        let options = ExportOptions {
            tool_output_cap: 10,
            format: Format::Json,
        };
        let text = render(&session, &SessionTotals::default(), &options);
        let doc: serde_json::Value = serde_json::from_str(&text).expect("valid JSON");

        assert_eq!(doc["messages"][0]["truncated"], true);
        assert_eq!(doc["messages"][0]["original_chars"], 40);
        assert_eq!(doc["messages"][0]["text"].as_str().map(|t| t.chars().count()), Some(10));
    }

    #[test]
    fn a_session_with_no_messages_still_renders() {
        let session = session_with(Vec::new());
        let markdown = render(&session, &SessionTotals::default(), &ExportOptions::default());
        assert!(markdown.contains("This session holds no messages."), "{markdown}");

        let options = ExportOptions {
            tool_output_cap: DEFAULT_TOOL_OUTPUT_CAP,
            format: Format::Json,
        };
        let text = render(&session, &SessionTotals::default(), &options);
        let doc: serde_json::Value = serde_json::from_str(&text).expect("valid JSON");
        assert_eq!(doc["messages"].as_array().map(|m| m.len()), Some(0));
        assert_eq!(doc["turns"].as_array().map(|t| t.len()), Some(0));
    }
}
