//! The kernel's own injection source: entries, budget, recursion, cache policy.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::hotreload::CachePolicy;
use crate::message::Message;

use super::entry::{Injection, InjectionEntry};
use super::position::{is_cjk, Position, ScanScope};

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

/// Decides what to inject.
///
/// A trait rather than a concrete type so a pack can answer the question itself.
/// The kernel's own engine is [`InjectionEngine`], which reads entries from
/// `injections.json` and matches substrings; a pack that wants regular
/// expressions, a decision model, or anything else supplies its own and the loop
/// asks it the same question it asks the built-in one.
///
/// The shape is one method returning [`Injection`] rather than a pipeline of
/// hooks: what goes into a prompt is decided in one place, and a pack that had to
/// cooperate with the kernel's own engine through several calls would be an
/// engine *next to* it, not *instead of* it.
pub trait InjectionSource: Send + Sync + std::fmt::Debug {
    /// What to add to this request, given the conversation so far.
    fn inject(
        &self,
        messages: &[Message],
        context_window: usize,
        policy: CachePolicy,
    ) -> Injection;
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

    /// Fold another pack's engine into this one, instead of replacing it.
    ///
    /// A pack's `injections.json` used to *replace* the standing engine, so
    /// installing a second pack that carries rules silently deleted the first
    /// one's — the rules were still on disk, still enabled, and simply not
    /// there. Measured on a real machine with two packs carrying rules: only the
    /// alphabetically last one's text ever reached the model.
    ///
    /// Entries are appended in pack-name order, which keeps the result
    /// deterministic, and an id that is already present is skipped: two packs
    /// that both ship `chinese-only` mean one rule, not two contradictory
    /// copies of the same sentence.
    ///
    /// The tuning knobs (budget, scan scope, recursion) keep the standing file's
    /// values: they describe the user's own machine, not any one pack.
    pub fn absorb(&mut self, other: InjectionEngine) {
        for entry in other.entries {
            if self.entries.iter().any(|kept| kept.id == entry.id) {
                eprintln!(
                    "[injections] a second entry with id {:?} was ignored: ids must be unique \
                     across everything a session loads",
                    entry.id
                );
                continue;
            }
            self.entries.push(entry);
        }
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
    fn run(
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
}

impl InjectionSource for InjectionEngine {

    /// The kernel engine as one source among any others.
    fn inject(
        &self,
        messages: &[Message],
        context_window: usize,
        policy: CachePolicy,
    ) -> Injection {
        // The work lives in an inherent method so this impl cannot call itself.
        self.run(messages, context_window, policy)
    }
}

impl InjectionEngine {
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
