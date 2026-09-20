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

fn default_position() -> Position {
    Position::HistoryTail
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

/// Rough token estimate: four characters per token.
///
/// Deliberately approximate. The exact count comes from the provider; this is
/// for budgeting, where being slightly conservative is enough.
pub fn estimate_tokens(text: &str) -> usize {
    text.chars().count().div_ceil(4).max(1)
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
    pub fn budget(&self, context_window: usize) -> usize {
        let percent = (context_window as u64 * self.budget_percent as u64 / 100) as usize;
        match self.budget_cap {
            Some(cap) => percent.min(cap),
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
        let mut group_winners: BTreeMap<String, String> = BTreeMap::new();

        // Triggers are matched against the scanned window, which starts as the
        // recent history and grows with each recursion pass as activated
        // content becomes new evidence.
        let mut scan_text = self.scan_window(messages, None);
        let mut overflowed = false;

        for pass in 0..=self.max_recursion {
            // Candidates in a deterministic order: higher order first, then id.
            let mut candidates: Vec<&InjectionEntry> = self
                .entries
                .iter()
                .filter(|entry| entry.enabled && !activated.contains(&entry.id))
                .filter(|entry| {
                    if pass == 0 {
                        entry.matches(&self.scan_window(messages, entry.scan_depth))
                    } else {
                        entry.matches(&scan_text)
                    }
                })
                .collect();
            candidates.sort_by(|a, b| b.order.cmp(&a.order).then_with(|| a.id.cmp(&b.id)));

            if candidates.is_empty() {
                break;
            }

            let mut pass_activated = false;
            let mut newly_activated_text = String::new();

            for entry in candidates {
                // Group competition: at most one member of a group is injected,
                // and the winner is chosen by weight then order then id — never
                // at random, so the same input yields the same context.
                if let Some(group) = &entry.group {
                    match group_winners.get(group) {
                        Some(winner) if winner != &entry.id => continue,
                        Some(_) => {}
                        None => {
                            let winner = self.pick_group_winner(group, &activated);
                            match winner {
                                Some(id) if id != entry.id => {
                                    group_winners.insert(group.clone(), id);
                                    continue;
                                }
                                _ => {
                                    group_winners.insert(group.clone(), entry.id.clone());
                                }
                            }
                        }
                    }
                }

                let cost = estimate_tokens(&entry.content);

                if !entry.ignore_budget && used + cost > budget {
                    // The budget stops *new* activations; what is already
                    // injected stays.
                    overflowed = true;
                    continue;
                }

                let preferred = entry.position.resolve(entry.depth);
                let (position, relocated) = self.resolve_position(&preferred, policy);

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

                used += cost;
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
            scan_text.push_str(&newly_activated_text);
        }

        // Prefix fragments keep the deterministic order too.
        result.activated = activated.into_iter().collect();
        result.budget_used = used;
        result.overflowed = overflowed;
        result
    }

    /// Choose the winner of a group deterministically.
    fn pick_group_winner(&self, group: &str, activated: &BTreeSet<String>) -> Option<String> {
        let mut members: Vec<&InjectionEntry> = self
            .entries
            .iter()
            .filter(|entry| entry.enabled && entry.group.as_deref() == Some(group))
            .filter(|entry| !activated.contains(&entry.id))
            .collect();
        members.sort_by(|a, b| {
            b.group_weight
                .cmp(&a.group_weight)
                .then_with(|| b.order.cmp(&a.order))
                .then_with(|| a.id.cmp(&b.id))
        });
        members.first().map(|entry| entry.id.clone())
    }

    /// Apply the cache policy to a preferred position.
    ///
    /// Returns the effective position and whether it had to move.
    fn resolve_position(&self, preferred: &Position, policy: CachePolicy) -> (Position, bool) {
        match policy {
            // Freshness accepts any position, including the prefix.
            CachePolicy::Freshness => (preferred.clone(), false),
            // Balanced prefers history but allows the prefix.
            CachePolicy::Balanced => (preferred.clone(), false),
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
    fn token_estimation_is_roughly_four_chars_per_token() {
        assert_eq!(estimate_tokens("abcd"), 1);
        assert_eq!(estimate_tokens("abcdefgh"), 2);
        // Never zero, so an empty fragment still costs something.
        assert_eq!(estimate_tokens(""), 1);
    }

    #[test]
    fn an_empty_engine_injects_nothing() {
        let engine = InjectionEngine::new();
        let injection = engine.inject(&conversation(&["hi"]), 100_000, CachePolicy::Balanced);
        assert!(injection.is_empty());
        assert_eq!(injection.budget_used, 0);
        assert!(!injection.overflowed);
    }
}
