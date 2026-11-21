//! Where a fragment may sit, and which messages a trigger may read.
//!
//! [`Position`] is where a fragment goes in the request; [`ScanScope`] is which
//! messages a trigger may read. Both are decisions the engine consults rather
//! than details it hides, which is why they live together here.
//!
//! The `default_*` helpers belong to the types they default: a settings file is
//! written by hand, so every field needs a value nobody typed. [`is_cjk`] is
//! here rather than next to the token estimator because it is knowledge about
//! scripts — which characters cost a token each — not about budgets.

use serde::{Deserialize, Serialize};

use crate::message::Role;

/// Where a fragment is placed in the request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Position {
    /// The prompt prefix. Changing this invalidates the cached prefix.
    Prefix,
    /// Appended after the conversation, before the newest user message.
    HistoryTail,
    /// Inserted `depth` messages back from the end of the history.
    AtDepth(usize),
}

/// Where an entry goes, as written in a configuration file.
///
/// A plain enum with a separate `depth` field, rather than the engine's own
/// [`Position`], because a settings file has to be writable by hand:
/// `"position": "at-depth", "depth": 3` is a line someone can type, while the
/// derived form of a newtype variant (`{"at-depth": 3}`) is not.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PositionKind {
    /// The prompt prefix. Changing this invalidates the cached prefix.
    Prefix,
    /// Appended after the conversation.
    #[default]
    HistoryTail,
    /// Inserted `depth` messages back from the end of the history.
    AtDepth,
}

impl PositionKind {
    /// A short label for display.
    pub fn as_str(&self) -> &'static str {
        match self {
            PositionKind::Prefix => "prefix",
            PositionKind::HistoryTail => "history-tail",
            PositionKind::AtDepth => "at-depth",
        }
    }

    /// Resolve to the engine's position, using `depth` for `AtDepth`.
    pub fn resolve(self, depth: Option<usize>) -> Position {
        match self {
            PositionKind::Prefix => Position::Prefix,
            PositionKind::HistoryTail => Position::HistoryTail,
            // A depth of zero would mean "after the last message", which the
            // tail position already covers; one is the nearest useful point.
            PositionKind::AtDepth => Position::AtDepth(depth.unwrap_or(1).max(1)),
        }
    }
}

pub(super) fn default_role() -> Role {
    Role::System
}
pub(super) fn default_weight() -> u32 {
    100
}
pub(super) fn default_true() -> bool {
    true
}

/// Whether a character belongs to a script that costs about one token each.
///
/// The ranges are the blocks a tokenizer splits per character: Han ideographs
/// and their extensions, Japanese kana (which are also one token each even
/// though they are three bytes), and Hangul syllables.
pub(super) fn is_cjk(character: char) -> bool {
    matches!(character as u32,
        // CJK Unified Ideographs Extension A.
        0x3400..=0x4DBF
        // CJK Unified Ideographs.
        | 0x4E00..=0x9FFF
        // Compatibility Ideographs.
        | 0xF900..=0xFAFF
        // Hiragana and Katakana.
        | 0x3040..=0x30FF
        // CJK Symbols and Punctuation: the full-width marks CJK text is written
        // with, one token each.
        | 0x3000..=0x303F
        // Hangul Syllables.
        | 0xAC00..=0xD7AF
        // Half-width and full-width forms: the full-width punctuation of CJK
        // text costs a token each as well.
        | 0xFF00..=0xFF60
        // CJK Unified Ideographs Extension B and beyond.
        | 0x20000..=0x3FFFF
    )
}

/// Which messages a trigger may match against.
///
/// This is not a cosmetic setting. Matching the assistant's own output lets an
/// injected fragment feed itself: an entry saying "this is a Rust project" makes
/// the model look for Cargo, its search results contain the word "cargo", the
/// trigger fires again, and the model keeps believing something that was never
/// true. Observed in practice, and it cost a dozen wasted tool calls.
///
/// So the default reads only what the *user* said. Tool output can be included
/// deliberately, because external facts are legitimate evidence; the model's own
/// words never are.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ScanScope {
    /// Only user messages. The safe default.
    #[default]
    User,
    /// User messages and tool results, but never the assistant's own text.
    UserAndTool,
    /// Every message, including the assistant's output. Can self-reinforce.
    All,
}

impl ScanScope {
    /// Whether a role is visible to triggers.
    pub fn includes(&self, role: Role) -> bool {
        match self {
            ScanScope::User => role == Role::User,
            ScanScope::UserAndTool => matches!(role, Role::User | Role::Tool),
            ScanScope::All => true,
        }
    }

    /// A short label for display.
    pub fn as_str(&self) -> &'static str {
        match self {
            ScanScope::User => "user",
            ScanScope::UserAndTool => "user+tool",
            ScanScope::All => "all",
        }
    }
}
