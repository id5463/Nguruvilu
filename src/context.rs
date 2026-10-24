//! The context injection engine.
//!
//! Mechanism adapted from SillyTavern's World Info (`world-info.js`), which is
//! the most carefully built context-budgeting design in this space, then
//! changed for an agent rather than a roleplay client.
//!
//! What it does: given a conversation and a budget, decide which extra
//! fragments to inject, where to put them, and when to stop.
//!
//! Two changes from the original matter:
//!
//! * **No randomness.** SillyTavern rolls dice (`probability`) and picks
//!   group winners at random. An agent must be reproducible — the same input
//!   has to produce the same context — so selection is deterministic: sort by
//!   `order`, then by id.
//! * **Cache awareness.** Injection position is not cosmetic. Writing into the
//!   prompt prefix invalidates the provider's cached prefix; writing into the
//!   history does not. The cache policy therefore decides *where* a fragment
//!   may go, and under `CacheFirst` a prefix injection is refused and moved
//!   into the history instead.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

pub use crate::hotreload::CachePolicy;
use crate::message::{Message, Role};

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

/// One injectable fragment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InjectionEntry {
    /// Identifier, used in reports and group tie-breaking.
    pub id: String,
    /// The text to inject.
    pub content: String,
    /// Always inject, regardless of triggers.
    #[serde(default)]
    pub constant: bool,
    /// Substrings that activate this entry. Matching is case-insensitive.
    #[serde(default)]
    pub triggers: Vec<String>,
    /// Where to place it.
    #[serde(default)]
    pub position: PositionKind,
    /// Messages back from the end, for `position: at-depth`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub depth: Option<usize>,
    /// Role to inject under, for history positions.
    #[serde(default = "default_role")]
    pub role: Role,
    /// Sort key; higher is considered first.
    #[serde(default)]
    pub order: i32,
    /// Group name. At most one member of a group is injected.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub group: Option<String>,
    /// Weight within the group; higher wins.
    #[serde(default = "default_weight")]
    pub group_weight: u32,
    /// Only scan this many of the most recent messages for triggers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scan_depth: Option<usize>,
    /// Exempt from the budget.
    #[serde(default)]
    pub ignore_budget: bool,
    /// Whether the entry may trigger further entries.
    #[serde(default = "default_true")]
    pub recursive: bool,
    /// Whether the entry is active at all.
    #[serde(default = "default_true")]
    pub enabled: bool,
}

fn default_role() -> Role {
    Role::System
}
fn default_weight() -> u32 {
    100
}
fn default_true() -> bool {
    true
}

impl InjectionEntry {
    /// A constant entry placed in the history tail.
    pub fn constant(id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            content: content.into(),
            constant: true,
            triggers: Vec::new(),
            position: PositionKind::HistoryTail,
            depth: None,
            role: Role::System,
            order: 0,
            group: None,
            group_weight: default_weight(),
            scan_depth: None,
            ignore_budget: false,
            recursive: true,
            enabled: true,
        }
    }

    /// A trigger-activated entry.
    pub fn triggered(
        id: impl Into<String>,
        content: impl Into<String>,
        triggers: Vec<String>,
    ) -> Self {
        Self {
            constant: false,
            triggers,
            ..Self::constant(id, content)
        }
    }

    /// Place this entry at a position.
    pub fn at(mut self, position: PositionKind) -> Self {
        self.position = position;
        self
    }

    /// Place this entry a number of messages back from the end.
    pub fn at_depth(mut self, depth: usize) -> Self {
        self.position = PositionKind::AtDepth;
        self.depth = Some(depth);
        self
    }

    /// Give this entry an order.
    pub fn ordered(mut self, order: i32) -> Self {
        self.order = order;
        self
    }

    /// Put this entry in a group.
    pub fn in_group(mut self, group: impl Into<String>, weight: u32) -> Self {
        self.group = Some(group.into());
        self.group_weight = weight;
        self
    }

    /// Exempt this entry from the budget.
    pub fn ignoring_budget(mut self) -> Self {
        self.ignore_budget = true;
        self
    }

    /// Restrict trigger scanning to the most recent `depth` messages.
    pub fn scanning(mut self, depth: usize) -> Self {
        self.scan_depth = Some(depth);
        self
    }

    /// Whether this entry matches `haystack`.
    fn matches(&self, haystack: &str) -> bool {
        if self.constant {
            return true;
        }
        if self.triggers.is_empty() {
            return false;
        }
        let lowered = haystack.to_lowercase();
        self.triggers
            .iter()
            .any(|trigger| lowered.contains(&trigger.to_lowercase()))
    }
}

/// The resolved injection for one request.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Injection {
    /// Text for the prompt prefix, in order.
    pub prefix: Vec<String>,
    /// Fragments placed inside the history: `(depth from end, role, text)`.
    pub at_depth: Vec<(usize, Role, String)>,
    /// Fragments appended at the history tail, in order.
    pub tail: Vec<String>,
    /// Ids of the entries that were injected.
    pub activated: Vec<String>,
    /// Estimated tokens used.
    pub budget_used: usize,
    /// Whether the budget stopped further activation.
    pub overflowed: bool,
    /// Entries that were activated but moved because the cache policy refused
    /// their preferred position.
    pub relocated: Vec<String>,
}

impl Injection {
    /// Whether anything was injected.
    pub fn is_empty(&self) -> bool {
        self.prefix.is_empty() && self.at_depth.is_empty() && self.tail.is_empty()
    }
}

/// Rough token estimate for budgeting, weighted by character class.
///
/// Deliberately approximate — the exact count comes from the provider — but not
/// approximate in one direction only, and not the same for every script.
///
/// A flat four-characters-per-token rule is calibrated on Latin text, where it
/// is roughly right. Chinese, Japanese, and Korean are three to four bytes and
/// one token per character: the same rule then undercounts by three to four
/// times, so a budget that reads as 25% of the window is really filling the
/// window, and the injection that was supposed to fit gets rejected by the
/// provider or silently truncates the history.
///
/// The weights below are per character, in quarter-tokens so the arithmetic is
/// whole numbers:
///
/// * ASCII: 0.25 token per character — the four-characters rule.
/// * CJK (Han, kana, Hangul): 1 token per character — one character is one
///   token, sometimes a little more.
/// * Any other non-ASCII: 0.5 token per character — accented Latin, Cyrillic,
///   Greek, emoji, and everything else outside the two ranges above.
///
/// Rounded up, and never zero, so an empty fragment still costs something.
pub fn estimate_tokens(text: &str) -> usize {
    let quarters: usize = text
        .chars()
        .map(|character| {
            if character.is_ascii() {
                // Four ASCII characters per token.
                1
            } else if is_cjk(character) {
                // One character per token, sometimes a bit more.
                4
            } else {
                // Two characters per token for the rest of the non-ASCII world.
                2
            }
        })
        .sum();
    quarters.div_ceil(4).max(1)
}

/// Whether a character belongs to a script that costs about one token each.
///
/// The ranges are the blocks a tokenizer splits per character: Han ideographs
/// and their extensions, Japanese kana (which are also one token each even
/// though they are three bytes), and Hangul syllables.
fn is_cjk(character: char) -> bool {
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

/// Decides what to inject.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InjectionEngine {
    /// Entries, in declaration order.
    #[serde(default)]
    pub entries: Vec<InjectionEntry>,
    /// Which messages triggers may match against.
    #[serde(default)]
    pub scan: ScanScope,
    /// Budget as a percentage of the context window.
    #[serde(default = "default_budget_percent")]
    pub budget_percent: u32,
    /// Absolute ceiling on the budget, in tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub budget_cap: Option<usize>,
    /// How many recursion passes to allow.
    #[serde(default = "default_max_recursion")]
    pub max_recursion: usize,
}

fn default_budget_percent() -> u32 {
    25
}
fn default_max_recursion() -> usize {
    3
}

impl Default for InjectionEngine {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
            scan: ScanScope::default(),
            budget_percent: default_budget_percent(),
            budget_cap: None,
            max_recursion: default_max_recursion(),
        }
    }
}

/// How much activated text later recursion passes may scan, in bytes.
///
/// Recursion feeds on what earlier passes injected. Without a ceiling, a long
/// chain of entries grows the haystack on every pass — and the trigger surface
/// widens the more the engine has already said — so the accumulation stops
/// here. Real entries are small; only a pathological configuration reaches it.
const MAX_RECURSION_SCAN_BYTES: usize = 64 * 1024;

impl InjectionEngine {
    /// An engine with no entries.
    pub fn new() -> Self {
        Self::default()
    }

    /// Where injection entries are read from.
    ///
    /// `$NGU_HOME/injections.json` when set, else `<home>/.nguruvilu/injections.json`.
    pub fn default_path() -> std::path::PathBuf {
        if let Ok(home) = std::env::var("NGU_HOME") {
            if !home.trim().is_empty() {
                return std::path::PathBuf::from(home).join("injections.json");
            }
        }
        crate::settings::home_dir()
            .join(".nguruvilu")
            .join("injections.json")
    }

    /// Read entries from a file.
    ///
    /// The file is this struct's own serialized form, so it carries the budget
    /// settings alongside the entries.
    pub fn from_file(path: &std::path::Path) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path)
            .map_err(|error| anyhow::anyhow!("reading {}: {error}", path.display()))?;
        let engine: Self = serde_json::from_str(&text)
            .map_err(|error| anyhow::anyhow!("parsing {}: {error}", path.display()))?;
        Ok(engine)
    }

    /// Load the default file, treating "no file" as "no entries".
    ///
    /// A missing file is the normal case for someone who has never written one,
    /// so it is not an error. A *malformed* file is reported rather than
    /// swallowed: silently ignoring it would leave standing rules quietly
    /// unapplied, which is the worst of both outcomes.
    pub fn load_default() -> Self {
        let path = Self::default_path();
        if !path.is_file() {
            return Self::new();
        }
        match Self::from_file(&path) {
            Ok(engine) => engine,
            Err(error) => {
                eprintln!("[injections] {error:#}");
                Self::new()
            }
        }
    }

    /// Write the engine to a file, creating its directory.
    pub fn save(&self, path: &std::path::Path) -> anyhow::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|error| anyhow::anyhow!("creating {}: {error}", parent.display()))?;
        }
        let text = serde_json::to_string_pretty(self)?;
        std::fs::write(path, format!("{text}\n"))
            .map_err(|error| anyhow::anyhow!("writing {}: {error}", path.display()))?;
        Ok(())
    }

    /// Add an entry.
    pub fn add(&mut self, entry: InjectionEntry) {
        self.entries.push(entry);
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether there are no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The budget in tokens for a given context window.
    ///
    /// Both settings are clamped, because both are written by hand in a JSON
    /// file. `budget_percent: 500` would otherwise mean a budget five times the
    /// window — the ceiling gone, which is the opposite of what the setting is
    /// for — and a cap of zero would refuse every entry with no diagnostic
    /// anyone could act on.
    pub fn budget(&self, context_window: usize) -> usize {
        let percent =
            (context_window as u64 * self.budget_percent.min(100) as u64 / 100) as usize;
        match self.budget_cap {
            Some(cap) => percent.min(cap.max(1)),
            None => percent,
        }
    }

    /// Resolve the injection for one request.
    ///
    /// `messages` is the conversation so far; `context_window` sizes the
    /// budget; `policy` decides whether the prefix is available.
    pub fn inject(
        &self,
        messages: &[Message],
        context_window: usize,
        policy: CachePolicy,
    ) -> Injection {
        let budget = self.budget(context_window);
        let mut result = Injection::default();
        let mut used = 0usize;
        let mut activated: BTreeSet<String> = BTreeSet::new();
        // One member per group for the whole request — recorded only once a
        // fragment actually landed, so a group that injected nothing keeps its
        // members eligible for the next pass.
        let mut group_winners: BTreeMap<String, String> = BTreeMap::new();
        // What earlier passes activated: the only new evidence recursion adds.
        // Bounded, because an unbounded haystack widens the trigger surface the
        // more the engine has already said.
        let mut scan_text = String::new();
        // Entries the budget refused, kept so the report at the end says
        // whether anything was really kept out.
        let mut refused: BTreeSet<String> = BTreeSet::new();

        for pass in 0..=self.max_recursion {
            // What this pass may consider: enabled, not yet injected, and
            // triggered by text the entry is itself allowed to read. This set is
            // also the only place a group winner may come from.
            let matched: Vec<&InjectionEntry> = self
                .entries
                .iter()
                .filter(|entry| entry.enabled && !activated.contains(&entry.id))
                .filter(|entry| {
                    // The declared scan depth holds on every pass, not only the
                    // first: recursion contributes activated fragments as
                    // evidence, it does not hand an entry the whole history to
                    // search for its trigger.
                    let mut haystack = self.scan_window(messages, entry.scan_depth);
                    if pass > 0 && !scan_text.is_empty() {
                        haystack.push('\n');
                        haystack.push_str(&scan_text);
                    }
                    entry.matches(&haystack)
                })
                .collect();

            if matched.is_empty() {
                break;
            }

            // Candidates in a deterministic order: higher order first, then id.
            let matching: BTreeSet<&str> = matched.iter().map(|entry| entry.id.as_str()).collect();
            let mut candidates: Vec<&InjectionEntry> = matched;
            candidates.sort_by(|a, b| b.order.cmp(&a.order).then_with(|| a.id.cmp(&b.id)));

            // Group decisions for this pass. A group with no qualifying member
            // injects nothing this pass, and the decision is not remembered: a
            // member that only activates later through recursion still gets its
            // turn at winning.
            let mut decided: BTreeMap<String, (Option<String>, Vec<String>)> = BTreeMap::new();
            let mut pass_activated = false;
            let mut newly_activated_text = String::new();

            for entry in candidates {
                // Group competition: at most one member of a group is injected,
                // and the winner is chosen by weight then order then id — never
                // at random, so the same input yields the same context.
                if let Some(group) = &entry.group {
                    // A group is closed by the member that already landed: that
                    // member is in `activated` and cannot be a candidate again,
                    // and every other member is out.
                    if group_winners.contains_key(group) {
                        continue;
                    }
                    let (winner, unaffordable) = match decided.get(group) {
                        Some(decision) => decision.clone(),
                        None => {
                            let decision = self.pick_group_winner(group, &matching, budget, used);
                            decided.insert(group.clone(), decision.clone());
                            decision
                        }
                    };
                    // A member the budget could not pay for is a budget
                    // outcome, not a group decision, and is reported as one.
                    for id in unaffordable {
                        refused.insert(id);
                    }
                    match winner {
                        Some(winner) if winner == entry.id => {}
                        _ => continue,
                    }
                }

                let cost = estimate_tokens(&entry.content);

                if !entry.ignore_budget && used + cost > budget {
                    // The budget stops *new* activations; what is already
                    // injected stays.
                    refused.insert(entry.id.clone());
                    continue;
                }

                let preferred = entry.position.resolve(entry.depth);
                let (position, relocated) =
                    self.resolve_position(&preferred, policy, cost, budget);

                match position {
                    Position::Prefix => result.prefix.push(entry.content.clone()),
                    Position::HistoryTail => result.tail.push(entry.content.clone()),
                    Position::AtDepth(depth) => {
                        result
                            .at_depth
                            .push((depth, entry.role, entry.content.clone()));
                    }
                }

                if relocated {
                    result.relocated.push(entry.id.clone());
                }

                if let Some(group) = &entry.group {
                    group_winners.insert(group.clone(), entry.id.clone());
                }

                // An entry exempt from the budget must not spend it either:
                // counting its cost would push every ordinary entry behind it
                // out of an injection it was never meant to lose.
                if !entry.ignore_budget {
                    used += cost;
                }
                activated.insert(entry.id.clone());
                pass_activated = true;
                if entry.recursive {
                    newly_activated_text.push_str(&entry.content);
                    newly_activated_text.push('\n');
                }
            }

            if !pass_activated {
                break;
            }
            if scan_text.len() < MAX_RECURSION_SCAN_BYTES {
                scan_text.push_str(&newly_activated_text);
            }
        }

        // Overflow means "the budget kept something out". An entry refused on
        // one pass that lands on another was not kept out, so the flag is
        // recomputed from what is still missing instead of being latched by the
        // first refusal — a latched flag blames the budget for what was really
        // a trigger or group decision.
        let overflowed = refused.iter().any(|id| !activated.contains(id));

        // Prefix fragments keep the deterministic order too.
        result.activated = activated.into_iter().collect();
        result.budget_used = used;
        result.overflowed = overflowed;
        result
    }

    /// Choose the winner of a group, from the members that actually qualify.
    ///
    /// Qualifying means all three: enabled, its own trigger fired in this pass
    /// (`matching`), and the budget can still pay for it. Choosing from the
    /// whole group instead — what this did before — made a group whose heaviest
    /// member was *not* triggered inject nothing at all: that member was never a
    /// candidate, and every member that did match was skipped as "someone else
    /// won", while the group was then recorded as decided. A group with no
    /// qualifying member injects nothing, which is the honest outcome.
    ///
    /// Returns the winner and the matched members the budget ruled out, so the
    /// report can say that the budget — not the group — kept them out.
    fn pick_group_winner(
        &self,
        group: &str,
        matching: &BTreeSet<&str>,
        budget: usize,
        used: usize,
    ) -> (Option<String>, Vec<String>) {
        let mut members: Vec<&InjectionEntry> = self
            .entries
            .iter()
            .filter(|entry| entry.enabled && entry.group.as_deref() == Some(group))
            .filter(|entry| matching.contains(entry.id.as_str()))
            .collect();
        members.sort_by(|a, b| {
            b.group_weight
                .cmp(&a.group_weight)
                .then_with(|| b.order.cmp(&a.order))
                .then_with(|| a.id.cmp(&b.id))
        });
        let mut winner = None;
        let mut unaffordable = Vec::new();
        for entry in members {
            let affordable =
                entry.ignore_budget || used + estimate_tokens(&entry.content) <= budget;
            if !affordable {
                unaffordable.push(entry.id.clone());
                continue;
            }
            // The list is already in preference order, so the first affordable
            // member is the winner.
            if winner.is_none() {
                winner = Some(entry.id.clone());
            }
        }
        (winner, unaffordable)
    }

    /// Apply the cache policy to a preferred position.
    ///
    /// Returns the effective position and whether it had to move. `cost` and
    /// `budget` are what the middle policy needs to actually decide something.
    fn resolve_position(
        &self,
        preferred: &Position,
        policy: CachePolicy,
        cost: usize,
        budget: usize,
    ) -> (Position, bool) {
        match policy {
            // Freshness accepts any position, including the prefix.
            CachePolicy::Freshness => (preferred.clone(), false),
            // Balanced prefers history but allows the prefix — for a fragment
            // small enough that invalidating the cached prefix is a rounding
            // error: a quarter of the injection budget. Anything larger moves to
            // the history tail, which costs the cache nothing. Without this the
            // policy was a copy of Freshness and the three settings were really
            // two.
            CachePolicy::Balanced => match preferred {
                Position::Prefix if cost.saturating_mul(4) > budget => {
                    (Position::HistoryTail, true)
                }
                other => (other.clone(), false),
            },
            // CacheFirst refuses to touch the prefix: the fragment moves into
            // the history tail instead of being dropped. Capability is never
            // gated on cache cost.
            CachePolicy::CacheFirst => match preferred {
                Position::Prefix => (Position::HistoryTail, true),
                other => (other.clone(), false),
            },
        }
    }

    /// The text triggers are matched against.
    fn scan_window(&self, messages: &[Message], depth: Option<usize>) -> String {
        let slice = match depth {
            Some(depth) if depth < messages.len() => &messages[messages.len() - depth..],
            _ => messages,
        };
        let mut text = String::new();
        for message in slice {
            if !self.scan.includes(message.role) {
                continue;
            }
            text.push_str(message.text());
            text.push('\n');
            for call in &message.tool_calls {
                text.push_str(&call.name);
                text.push('\n');
            }
        }
        text
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn conversation(turns: &[&str]) -> Vec<Message> {
        turns.iter().map(|t| Message::user(*t)).collect()
    }

    #[test]
    fn a_constant_entry_is_always_injected() {
        let mut engine = InjectionEngine::new();
        engine.add(InjectionEntry::constant("always", "standing rule"));

        let injection = engine.inject(&conversation(&["hello"]), 8000, CachePolicy::Balanced);
        assert_eq!(injection.tail, vec!["standing rule"]);
        assert_eq!(injection.activated, vec!["always"]);
        assert!(!injection.overflowed);
    }

    #[test]
    fn a_triggered_entry_needs_its_trigger() {
        let mut engine = InjectionEngine::new();
        engine.add(InjectionEntry::triggered(
            "pdf",
            "use pdftotext",
            vec!["pdf".into()],
        ));

        let quiet = engine.inject(&conversation(&["hello"]), 8000, CachePolicy::Balanced);
        assert!(quiet.is_empty());

        let hit = engine.inject(
            &conversation(&["please read this pdf"]),
            8000,
            CachePolicy::Balanced,
        );
        assert_eq!(hit.tail, vec!["use pdftotext"]);
    }

    #[test]
    fn triggers_are_case_insensitive() {
        let mut engine = InjectionEngine::new();
        engine.add(InjectionEntry::triggered(
            "grep-tip",
            "prefer ripgrep",
            vec!["grep".into()],
        ));

        let upper = engine.inject(&conversation(&["GREP the repo"]), 8000, CachePolicy::Balanced);
        assert_eq!(upper.activated, vec!["grep-tip"]);
    }

    #[test]
    fn the_assistants_own_output_does_not_trigger_anything() {
        // The loop this prevents: an entry claims something, the model acts on
        // it, the model's own words match the trigger, and the claim keeps
        // renewing itself. It cost a dozen wasted tool calls in practice.
        let mut engine = InjectionEngine::new();
        engine.add(InjectionEntry::triggered(
            "rust-style",
            "This is a Rust project",
            vec!["cargo".into()],
        ));

        let mut history = conversation(&["draw me a picture"]);
        history.push(Message::assistant("I will look for Cargo.toml"));
        history.push(Message::assistant_tools(
            vec![crate::message::ToolCall {
                id: "c1".into(),
                name: "bash".into(),
                arguments: r#"{"command":"ls Cargo.toml"}"#.into(),
            }],
            None,
        ));

        let injection = engine.inject(&history, 8000, CachePolicy::Balanced);
        assert!(
            injection.is_empty(),
            "the model's own mention of cargo must not activate the entry: {:?}",
            injection.activated
        );
    }

    #[test]
    fn tool_results_can_be_scanned_when_asked_for() {
        let mut engine = InjectionEngine::new();
        engine.scan = ScanScope::UserAndTool;
        engine.add(InjectionEntry::triggered(
            "grep-tip",
            "prefer ripgrep",
            vec!["grep".into()],
        ));

        let mut history = conversation(&["run it"]);
        history.push(Message::tool_result("c1", "bash", "grep: command not found"));

        let injection = engine.inject(&history, 8000, CachePolicy::Balanced);
        assert_eq!(injection.activated, vec!["grep-tip"]);
    }

    #[test]
    fn the_full_scope_scans_everything_and_says_so() {
        assert!(ScanScope::All.includes(Role::Assistant));
        assert!(ScanScope::All.includes(Role::Tool));
        assert!(ScanScope::All.includes(Role::User));

        assert!(!ScanScope::User.includes(Role::Assistant));
        assert!(!ScanScope::User.includes(Role::Tool));
        assert!(ScanScope::User.includes(Role::User));

        assert!(ScanScope::UserAndTool.includes(Role::Tool));
        assert!(!ScanScope::UserAndTool.includes(Role::Assistant));

        assert_eq!(ScanScope::default(), ScanScope::User, "the safe default");
    }

    #[test]
    fn scan_depth_limits_what_can_trigger() {
        let mut engine = InjectionEngine::new();
        engine.add(
            InjectionEntry::triggered("pdf", "use pdftotext", vec!["pdf".into()]).scanning(1),
        );

        // The trigger is far back, outside the scan window.
        let old = conversation(&["read this pdf", "unrelated", "unrelated"]);
        assert!(engine
            .inject(&old, 8000, CachePolicy::Balanced)
            .is_empty());

        // Inside the window it fires.
        let recent = conversation(&["unrelated", "unrelated", "read this pdf"]);
        assert_eq!(
            engine.inject(&recent, 8000, CachePolicy::Balanced).activated,
            vec!["pdf"]
        );
    }

    #[test]
    fn the_budget_stops_new_activations() {
        let mut engine = InjectionEngine::new();
        engine.budget_percent = 25;
        // 1000-token window → 250-token budget. Each entry is ~250 tokens.
        let big = "x".repeat(1000);
        engine.add(InjectionEntry::constant("first", big.clone()).ordered(10));
        engine.add(InjectionEntry::constant("second", big.clone()).ordered(5));

        let injection = engine.inject(&conversation(&["hi"]), 1000, CachePolicy::Balanced);
        assert!(injection.overflowed, "the second entry could not fit");
        assert_eq!(injection.activated, vec!["first"], "higher order wins the budget");
        assert!(injection.budget_used <= 250);
    }

    #[test]
    fn ignore_budget_entries_get_in_anyway() {
        let mut engine = InjectionEngine::new();
        let big = "x".repeat(4000);
        engine.add(InjectionEntry::constant("bulk", big.clone()).ordered(10));
        engine.add(
            InjectionEntry::constant("critical", "safety rule")
                .ordered(-10)
                .ignoring_budget(),
        );

        let injection = engine.inject(&conversation(&["hi"]), 1000, CachePolicy::Balanced);
        assert!(injection.overflowed, "the bulk entry overflowed the budget");
        assert!(
            injection.activated.contains(&"critical".to_string()),
            "a budget-exempt entry still lands: {:?}",
            injection.activated
        );
    }

    #[test]
    fn the_budget_respects_its_cap() {
        let mut engine = InjectionEngine::new();
        engine.budget_percent = 50;
        engine.budget_cap = Some(1000);

        // 50% of 100_000 would be 50_000, but the cap wins.
        assert_eq!(engine.budget(100_000), 1000);
        // Below the cap the percentage applies.
        assert_eq!(engine.budget(1000), 500);
    }

    #[test]
    fn recursion_lets_activated_content_trigger_more() {
        let mut engine = InjectionEngine::new();
        engine.add(InjectionEntry::triggered(
            "outer",
            "the word needle appears here",
            vec!["start".into()],
        ));
        engine.add(InjectionEntry::triggered(
            "inner",
            "inner detail",
            vec!["needle".into()],
        ));

        let injection = engine.inject(&conversation(&["start"]), 100_000, CachePolicy::Balanced);
        let mut activated = injection.activated.clone();
        activated.sort();
        assert_eq!(activated, vec!["inner", "outer"], "the inner entry followed the outer");
    }

    #[test]
    fn a_non_recursive_entry_does_not_trigger_further_ones() {
        let mut engine = InjectionEngine::new();
        let mut outer = InjectionEntry::triggered("outer", "mentions needle", vec!["start".into()]);
        outer.recursive = false;
        engine.add(outer);
        engine.add(InjectionEntry::triggered("inner", "detail", vec!["needle".into()]));

        let injection = engine.inject(&conversation(&["start"]), 100_000, CachePolicy::Balanced);
        assert_eq!(injection.activated, vec!["outer"]);
    }

    #[test]
    fn a_group_injects_exactly_one_deterministic_winner() {
        let mut engine = InjectionEngine::new();
        engine.add(
            InjectionEntry::constant("low", "low weight")
                .in_group("style", 10)
                .ordered(100),
        );
        engine.add(
            InjectionEntry::constant("high", "high weight")
                .in_group("style", 90)
                .ordered(1),
        );

        let first = engine.inject(&conversation(&["hi"]), 100_000, CachePolicy::Balanced);
        assert_eq!(first.tail.len(), 1, "only one member of the group");
        assert_eq!(first.activated, vec!["high"], "the heavier weight wins");

        // Repeating must give the same answer: an agent cannot be random.
        let second = engine.inject(&conversation(&["hi"]), 100_000, CachePolicy::Balanced);
        assert_eq!(first, second);
    }

    #[test]
    fn group_ties_break_on_order_then_id() {
        let mut engine = InjectionEngine::new();
        engine.add(InjectionEntry::constant("b", "b text").in_group("g", 50).ordered(5));
        engine.add(InjectionEntry::constant("a", "a text").in_group("g", 50).ordered(5));

        let injection = engine.inject(&conversation(&["hi"]), 100_000, CachePolicy::Balanced);
        assert_eq!(injection.activated, vec!["a"], "the lower id wins the tie");
    }

    #[test]
    fn position_decides_where_a_fragment_lands() {
        let mut engine = InjectionEngine::new();
        engine.add(InjectionEntry::constant("p", "prefix text").at(PositionKind::Prefix));
        engine.add(InjectionEntry::constant("t", "tail text").at(PositionKind::HistoryTail));
        engine.add(InjectionEntry::constant("d", "deep text").at_depth(3));

        let injection = engine.inject(&conversation(&["hi"]), 100_000, CachePolicy::Balanced);
        assert_eq!(injection.prefix, vec!["prefix text"]);
        assert_eq!(injection.tail, vec!["tail text"]);
        assert_eq!(injection.at_depth.len(), 1);
        assert_eq!(injection.at_depth[0].0, 3);
        assert_eq!(injection.at_depth[0].2, "deep text");
    }

    #[test]
    fn cache_first_relocates_prefix_injections_instead_of_dropping_them() {
        let mut engine = InjectionEngine::new();
        engine.add(InjectionEntry::constant("p", "prefix text").at(PositionKind::Prefix));

        let injection = engine.inject(&conversation(&["hi"]), 100_000, CachePolicy::CacheFirst);

        // The fragment still lands — capability is not gated on cache cost —
        // but it moves out of the prefix.
        assert!(injection.prefix.is_empty());
        assert_eq!(injection.tail, vec!["prefix text"]);
        assert_eq!(injection.relocated, vec!["p"]);
        assert_eq!(injection.activated, vec!["p"]);
    }

    #[test]
    fn freshness_leaves_the_prefix_alone() {
        let mut engine = InjectionEngine::new();
        engine.add(InjectionEntry::constant("p", "prefix text").at(PositionKind::Prefix));

        let injection = engine.inject(&conversation(&["hi"]), 100_000, CachePolicy::Freshness);
        assert_eq!(injection.prefix, vec!["prefix text"]);
        assert!(injection.relocated.is_empty());
    }

    #[test]
    fn disabled_entries_are_skipped() {
        let mut engine = InjectionEngine::new();
        let mut entry = InjectionEntry::constant("off", "should not appear");
        entry.enabled = false;
        engine.add(entry);

        assert!(engine
            .inject(&conversation(&["hi"]), 100_000, CachePolicy::Balanced)
            .is_empty());
    }

    #[test]
    fn token_estimation_is_four_ascii_characters_per_token() {
        assert_eq!(estimate_tokens("abcd"), 1);
        assert_eq!(estimate_tokens("abcdefgh"), 2);
        // Never zero, so an empty fragment still costs something.
        assert_eq!(estimate_tokens(""), 1);
        // Rounded up rather than down: a partial character still costs a token.
        assert_eq!(estimate_tokens("abcde"), 2);
    }

    #[test]
    fn chinese_costs_a_token_per_character_not_per_four() {
        // The defect this pins down: a flat characters/4 estimate reads a
        // hundred-character Chinese fragment as 25 tokens when the provider
        // counts about a hundred, so a Chinese session's injection budget is
        // really three to four times the window it was set to.
        let chinese = "这是一段用于测试中文令牌估算的说明文字";
        let characters = chinese.chars().count();
        assert!(characters > 10, "a fragment worth measuring: {characters}");
        assert_eq!(
            estimate_tokens(chinese),
            characters,
            "one token per CJK character"
        );
        // Under the old rule this was 5; anything near it is the bug.
        assert!(estimate_tokens(chinese) >= characters);

        // Full-width punctuation carries the same weight as the ideographs it
        // sits between.
        assert_eq!(estimate_tokens("说明：很好。"), 6);

        // A mixed fragment is the sum of its parts, not the count of them.
        // 20 ASCII characters (5 tokens) + 4 ideographs (4 tokens) = 9.
        let mixed = "the model answered 你好世界";
        assert_eq!(estimate_tokens(mixed), 5 + 4);
    }

    #[test]
    fn other_scripts_fall_between_ascii_and_cjk() {
        // Accented Latin, Cyrillic, and emoji are two characters per token:
        // more than ASCII, less than Han.
        assert_eq!(estimate_tokens("éééé"), 2);
        assert_eq!(estimate_tokens("привет"), 3);
        // Emoji are four bytes and cost more than a quarter-token character.
        assert!(estimate_tokens("🚀🚀🚀🚀") >= 2);
    }

    #[test]
    fn a_chinese_fragment_is_measured_against_the_same_budget() {
        // The budget decision, not just the number: a Chinese entry that fits a
        // window sized by the old Latin-only rule must be refused when the
        // estimate says what Chinese actually costs.
        let chinese = "这是一段用于测试中文令牌估算的说明文字".repeat(10);
        let mut engine = InjectionEngine::new();
        // The whole window, so the comparison is about the estimate alone.
        engine.budget_percent = 100;
        engine.add(InjectionEntry::constant("cn", chinese.clone()));

        let used = estimate_tokens(&chinese);
        let old_estimate = chinese.chars().count().div_ceil(4);
        assert!(
            used > old_estimate,
            "the weighted estimate is the larger one: {used} vs {old_estimate}"
        );

        // A window sized by the old rule cannot actually hold the text.
        let tight = engine.inject(&conversation(&["hi"]), old_estimate, CachePolicy::Balanced);
        assert!(tight.overflowed, "the budget refuses it");
        assert!(tight.activated.is_empty(), "and nothing was injected");

        // One sized by what Chinese costs holds it.
        let fitting = engine.inject(&conversation(&["hi"]), used, CachePolicy::Balanced);
        assert_eq!(fitting.activated, vec!["cn".to_string()]);
        assert_eq!(fitting.budget_used, used);
    }

    #[test]
    fn an_empty_engine_injects_nothing() {
        let engine = InjectionEngine::new();
        let injection = engine.inject(&conversation(&["hi"]), 100_000, CachePolicy::Balanced);
        assert!(injection.is_empty());
        assert_eq!(injection.budget_used, 0);
        assert!(!injection.overflowed);
    }

    #[test]
    fn a_group_injects_the_member_whose_trigger_fired() {
        // The heavy member of the group is not triggered. Choosing the winner
        // from the whole group without checking that made the entire group
        // inject nothing: the winner was never a candidate, and the member that
        // did match was skipped as "someone else won".
        let mut engine = InjectionEngine::new();
        engine.add(
            InjectionEntry::triggered("heavy", "heavy text", vec!["quantum".into()])
                .in_group("style", 90),
        );
        engine.add(
            InjectionEntry::triggered("light", "light text", vec!["pdf".into()])
                .in_group("style", 10),
        );

        let injection = engine.inject(
            &conversation(&["please read this pdf"]),
            100_000,
            CachePolicy::Balanced,
        );
        assert_eq!(
            injection.activated,
            vec!["light"],
            "the member that matched is the one that lands"
        );
        assert_eq!(injection.tail, vec!["light text"]);
        assert!(!injection.overflowed, "nothing was refused by the budget");
    }

    #[test]
    fn a_group_injects_nothing_when_no_member_is_triggered() {
        let mut engine = InjectionEngine::new();
        engine.add(
            InjectionEntry::triggered("heavy", "heavy text", vec!["quantum".into()])
                .in_group("style", 90),
        );
        engine.add(
            InjectionEntry::triggered("light", "light text", vec!["tensor".into()])
                .in_group("style", 10),
        );

        let injection = engine.inject(
            &conversation(&["please read this pdf"]),
            100_000,
            CachePolicy::Balanced,
        );
        assert!(injection.is_empty(), "{:?}", injection.activated);
        assert!(!injection.overflowed);
    }

    #[test]
    fn a_group_winner_must_also_fit_the_budget() {
        // Both members match, the heavier one is far too big for what is left.
        // The group still injects — the affordable member wins — and the report
        // says the budget is what ruled the other one out.
        let mut engine = InjectionEngine::new();
        engine.budget_percent = 25; // 1000-token window → 250 tokens.
        engine.add(
            InjectionEntry::triggered("heavy", "x".repeat(4000), vec!["pdf".into()])
                .in_group("style", 90),
        );
        engine.add(
            InjectionEntry::triggered("light", "short note", vec!["pdf".into()])
                .in_group("style", 10),
        );

        let injection = engine.inject(
            &conversation(&["read this pdf"]),
            1000,
            CachePolicy::Freshness,
        );
        assert_eq!(
            injection.activated,
            vec!["light"],
            "the member the budget can pay for wins"
        );
        assert!(injection.overflowed, "and the heavy one is reported as unfitted");
    }

    #[test]
    fn a_group_loser_is_not_reported_as_a_budget_overflow() {
        // Nothing ran out of budget here: a group decided. The flag has to name
        // the right cause or the diagnosis goes to the wrong place.
        let mut engine = InjectionEngine::new();
        engine.add(InjectionEntry::constant("high", "high text").in_group("style", 90));
        engine.add(InjectionEntry::constant("low", "low text").in_group("style", 10));

        let injection = engine.inject(&conversation(&["hi"]), 100_000, CachePolicy::Balanced);
        assert_eq!(injection.activated, vec!["high"]);
        assert!(!injection.overflowed);
    }

    #[test]
    fn an_exempt_entry_does_not_spend_the_budget_it_is_exempt_from() {
        // "Ignore the budget" has to mean both halves: never refused, and never
        // consuming. Counting it pushed every ordinary entry behind it out.
        let mut engine = InjectionEngine::new();
        engine.budget_percent = 25; // 1000-token window → 250 tokens.
        engine.add(
            InjectionEntry::constant("exempt", "x".repeat(4000))
                .ignoring_budget()
                .ordered(10),
        );
        engine.add(InjectionEntry::constant("normal", "a standing rule").ordered(5));

        let injection = engine.inject(&conversation(&["hi"]), 1000, CachePolicy::Balanced);
        assert!(
            injection.activated.contains(&"normal".to_string()),
            "the ordinary entry still fits: {:?}",
            injection.activated
        );
        assert!(
            injection.activated.contains(&"exempt".to_string()),
            "and the exempt one lands too: {:?}",
            injection.activated
        );
        assert!(!injection.overflowed);
        assert_eq!(
            injection.budget_used,
            estimate_tokens("a standing rule"),
            "only the ordinary entry is charged"
        );
    }

    #[test]
    fn an_out_of_range_budget_setting_is_clamped() {
        let mut engine = InjectionEngine::new();

        engine.budget_percent = 500;
        assert_eq!(
            engine.budget(1000),
            1000,
            "a percent above 100 must not lift the ceiling past the window"
        );

        engine.budget_percent = 25;
        assert_eq!(engine.budget(1000), 250);

        // A cap of zero would refuse every entry with nothing to diagnose.
        engine.budget_cap = Some(0);
        assert_eq!(engine.budget(1000), 1);
    }

    #[test]
    fn scan_depth_is_honoured_on_recursion_passes_too() {
        // "Only look at the last message" is a property of the entry, not of
        // the first pass. Recursion used to rescan the *whole* history on later
        // passes, so an entry declared with a depth of one was triggered by a
        // word from the beginning of the conversation — and the more passes ran,
        // the wider its trigger surface got.
        let mut engine = InjectionEngine::new();
        engine.add(InjectionEntry::triggered(
            "outer",
            "outer says go",
            vec!["start".into()],
        ));
        engine.add(
            InjectionEntry::triggered("deep", "deep detail", vec!["needle".into()]).scanning(1),
        );

        // "needle" is far outside the one-message window `deep` declared.
        let history = conversation(&["needle", "unrelated", "start"]);
        let injection = engine.inject(&history, 100_000, CachePolicy::Freshness);
        assert_eq!(
            injection.activated,
            vec!["outer"],
            "the depth limit holds on recursion passes"
        );

        // Recursion itself still works: `deep` fires when the activated text
        // carries its trigger, which is the evidence recursion is for.
        let mut chained = InjectionEngine::new();
        chained.add(InjectionEntry::triggered(
            "outer",
            "the word needle appears here",
            vec!["start".into()],
        ));
        chained.add(
            InjectionEntry::triggered("deep", "deep detail", vec!["needle".into()]).scanning(1),
        );
        let chained_injection =
            chained.inject(&conversation(&["start"]), 100_000, CachePolicy::Freshness);
        let mut activated = chained_injection.activated.clone();
        activated.sort();
        assert_eq!(activated, vec!["deep", "outer"]);
    }

    #[test]
    fn balanced_allows_a_cheap_prefix_and_moves_an_expensive_one() {
        // The middle policy has to have its own behaviour, or the three
        // settings are really two. A prefix is cheap when re-writing the cached
        // prefix costs a rounding error of the budget; past that it moves to the
        // history, where it costs the cache nothing.
        let mut engine = InjectionEngine::new();
        engine.budget_percent = 25; // 1000-token window → 250 tokens.
        engine.add(InjectionEntry::constant("cheap", "short rule").at(PositionKind::Prefix));
        engine.add(
            InjectionEntry::constant("large", "x".repeat(900))
                .at(PositionKind::Prefix)
                .ordered(-1),
        );

        let balanced = engine.inject(&conversation(&["hi"]), 1000, CachePolicy::Balanced);
        assert_eq!(
            balanced.prefix,
            vec!["short rule"],
            "a cheap prefix keeps its place"
        );
        assert_eq!(balanced.tail, vec!["x".repeat(900)], "an expensive one moves");
        assert_eq!(balanced.relocated, vec!["large"]);
        assert_eq!(
            balanced.activated,
            vec!["cheap".to_string(), "large".to_string()],
            "moving a fragment is not dropping it"
        );

        // Freshness takes either price and leaves the prefix alone.
        let fresh = engine.inject(&conversation(&["hi"]), 1000, CachePolicy::Freshness);
        assert_eq!(fresh.prefix.len(), 2);
        assert!(fresh.relocated.is_empty());
    }
}
