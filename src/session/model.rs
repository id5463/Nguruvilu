//! Session records and their usage ledger.

use serde::{Deserialize, Serialize};

use crate::message::Message;

use super::replay::{new_id, now_iso};

/// A conversation with its metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    /// Stable id, also the file stem.
    pub id: String,
    /// ISO-8601 creation time.
    pub created_at: String,
    /// ISO-8601 last-update time.
    pub updated_at: String,
    /// Working directory this session was started in.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cwd: Option<String>,
    /// Model route used.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Packs this session has loaded, by pack label. Persisted so that a
    /// restart still knows which sessions depend on a pack: an unload in one
    /// conversation must not remove a pack another conversation still uses.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub packs: Vec<String>,
    /// Conversation history.
    #[serde(default)]
    pub messages: Vec<Message>,
    /// Usage of every turn this session ran, oldest first.
    ///
    /// Kept beside the conversation rather than inside it: a compaction folds
    /// messages away, but what a conversation cost is a ledger, not dialogue.
    #[serde(default)]
    pub stats: Vec<TurnStats>,
    /// Compaction records that no longer line up with the stored messages.
    ///
    /// Empty for a healthy file. One entry per record whose replaced run does
    /// not match what the file actually holds — a dropped line, a truncated
    /// append, an outer tool that rewrote the file. Such a record is *not*
    /// applied: folding the wrong messages would delete history silently, and
    /// this field is how a caller (or the log) can say so out loud.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub drift: Vec<String>,
}

impl Session {
    /// Start a new session with a generated id and no working directory.
    ///
    /// The caller assigns [`Session::cwd`] once it has created the session's
    /// workspace with [`crate::session::ensure_workspace`]. Leaving it unset here means a
    /// session never inherits the process directory by accident.
    pub fn new(model: Option<String>) -> Self {
        let now = now_iso();
        Self {
            id: new_id(),
            created_at: now.clone(),
            updated_at: now,
            cwd: None,
            model,
            packs: Vec::new(),
            messages: Vec::new(),
            stats: Vec::new(),
            drift: Vec::new(),
        }
    }

    /// A short title derived from the first user message.
    pub fn title(&self) -> Option<String> {
        self.messages
            .iter()
            .find(|m| m.role == crate::message::Role::User)
            .map(|m| {
                let text = m.text().trim().replace('\n', " ");
                if text.chars().count() > 60 {
                    format!("{}…", text.chars().take(60).collect::<String>())
                } else {
                    text
                }
            })
            .filter(|t| !t.is_empty())
    }
}

/// Summary of a stored session, for listing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSummary {
    /// Session id.
    pub id: String,
    /// Creation time.
    pub created_at: String,
    /// Last update time.
    pub updated_at: String,
    /// Number of stored messages.
    pub message_count: usize,
    /// Derived title, when the session has a user message.
    pub title: Option<String>,
}

/// One turn's usage, as recorded in the session log.
///
/// Written as its own JSONL line at the end of every turn. The cached-prefix
/// plumbing (parsing, summing) already existed; without this row it was printed
/// once and lost, so "what did this session cost" was unanswerable after the
/// fact.
///
/// A **struct** variant, not a bare sequence: the line type is internally
/// tagged, and serde can only buffer a struct into a map — a newtype around a
/// `Vec` fails to serialize at all (the lesson from the `Packs` line).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct TurnStats {
    /// When the turn ended, RFC3339.
    pub ts: String,
    /// Model steps taken.
    pub steps: u32,
    /// Tool calls executed.
    pub tool_calls: u32,
    /// Step retries.
    pub retries: u32,
    /// Prompt tokens, provider semantics — cached tokens included.
    pub input: u64,
    /// Completion tokens.
    pub output: u64,
    /// Prompt tokens the provider served from its cache.
    pub cached: u64,
    /// Prompt tokens the provider wrote *into* its cache this turn.
    ///
    /// Separate from `cached` because the two are billed differently: a write
    /// usually costs a premium and a read a discount, so a ledger that folds
    /// them together cannot be turned into a bill.
    pub cache_created: u64,
    /// Reasoning tokens, when the provider reports them.
    pub reasoning: u64,
    /// Time spent waiting on the model.
    pub model_ms: u64,
    /// Time spent executing tools.
    pub tools_ms: u64,
}

impl TurnStats {
    /// The stats of a finished turn, read from what the loop reported.
    pub fn from_outcome(outcome: &crate::agent::AgentOutcome) -> Self {
        Self {
            ts: now_iso(),
            steps: u32::try_from(outcome.steps).unwrap_or(u32::MAX),
            tool_calls: u32::try_from(outcome.tool_calls).unwrap_or(u32::MAX),
            retries: outcome.retries,
            input: outcome.usage.input,
            output: outcome.usage.output,
            cached: outcome.usage.cached,
            cache_created: outcome.usage.cache_created,
            reasoning: outcome.usage.reasoning,
            model_ms: u64::try_from(outcome.timing.model_ms).unwrap_or(u64::MAX),
            tools_ms: u64::try_from(outcome.timing.tools_ms).unwrap_or(u64::MAX),
        }
    }

    /// The row for a turn that died before reporting anything.
    ///
    /// Written rather than skipped, so the turn count stays honest; every
    /// measured field is zero because nothing was reported to measure.
    pub fn unreported() -> Self {
        Self {
            ts: now_iso(),
            ..Self::default()
        }
    }
}

/// A session's usage summed over all of its turns.
///
/// `#[serde(default)]` so a totals object written by an older build — one
/// without `cache_created` — still reads back instead of being rejected as a
/// missing field.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct SessionTotals {
    /// Turns recorded.
    pub turns: u64,
    /// Model steps, summed.
    pub steps: u64,
    /// Tool calls, summed.
    pub tool_calls: u64,
    /// Prompt tokens, summed (cached included, provider semantics).
    pub input: u64,
    /// Completion tokens, summed.
    pub output: u64,
    /// Prompt tokens served from the provider's cache, summed.
    pub cached: u64,
    /// Prompt tokens written into the provider's cache, summed.
    pub cache_created: u64,
    /// Reasoning tokens, summed.
    pub reasoning: u64,
    /// Model wait time, summed.
    pub model_ms: u64,
    /// Tool execution time, summed.
    pub tools_ms: u64,
}

impl SessionTotals {
    /// The share of prompt tokens the provider's cache served, `0..=1`.
    ///
    /// Zero prompt tokens means the question is meaningless, not a perfect hit:
    /// a session with no stats, or one that never got a reply, reads as 0.0.
    pub fn cache_hit_ratio(&self) -> f64 {
        if self.input == 0 {
            0.0
        } else {
            self.cached as f64 / self.input as f64
        }
    }
}

impl std::iter::Sum for SessionTotals {
    fn sum<I: Iterator<Item = SessionTotals>>(iter: I) -> SessionTotals {
        iter.fold(SessionTotals::default(), |mut total, next| {
            total.turns += next.turns;
            total.steps += next.steps;
            total.tool_calls += next.tool_calls;
            total.input += next.input;
            total.output += next.output;
            total.cached += next.cached;
            total.cache_created += next.cache_created;
            total.reasoning += next.reasoning;
            total.model_ms += next.model_ms;
            total.tools_ms += next.tools_ms;
            total
        })
    }
}
