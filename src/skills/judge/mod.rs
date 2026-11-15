//! Which skills this turn's catalog shows.
//!
//! The catalog is the model's only view of what it can load, so deciding what
//! belongs in it is the whole problem. Four mechanisms, each a different trade:
//!
//! - [`full`] — everything, always. No selection at all. The catalog has to be
//!   the shape the model reads rather than a search result, and that is the
//!   point: the model, not an algorithm, decides what applies.
//! - [`regex`] — rules from the skill's own frontmatter. Free and predictable,
//!   and wrong exactly when a request does not use the words the rules expect.
//! - [`jev`] — a decision model asked once per turn about every skill at the
//!   same time. It answers a probability per skill, which is a number a
//!   threshold can act on.
//! - [`llm`] — a language model shown the whole catalog. Most flexible, most
//!   expensive, and the only one that can reason about a request no rule
//!   anticipated.
//!
//! Every mechanism narrows what is *shown*. None of them load a skill: loading
//! stays the model's decision, made by reading the catalog and calling `skill`.
//! A selector that force-loaded would be guessing with the user's context.
//!
//! An absent mode is [`full`], because a catalog that silently drops skills is
//! worse than a long one — the failure is invisible from inside the turn.

pub mod full;
pub mod jev;
pub mod llm;
pub mod regex;

use std::collections::HashSet;

use std::future::Future;
use std::pin::Pin;

use crate::skills::Skill;

/// The boxed future a judge returns.
pub type JudgeFuture<'a> = Pin<Box<dyn Future<Output = anyhow::Result<HashSet<String>>> + Send + 'a>>;

/// The catalog a turn starts from, and the request that will read it.
pub struct Ask<'a> {
    /// The user's message, verbatim. Summaries lose the words rules match on.
    pub query: &'a str,
    /// Every skill registered right now.
    pub skills: &'a [&'a Skill],
}

/// One mechanism for deciding which skills a turn's catalog shows.
pub trait SkillJudge: Send + Sync {
    /// The mode's name, as it appears in settings and the panel.
    fn mode(&self) -> &'static str;

    /// Which skill ids this turn's catalog should show.
    ///
    /// Returning an empty set would hide every skill, which reads as "there are
    /// no skills" rather than "none matched". A caller seeing an empty set gets
    /// the full catalog instead, so a selector that finds nothing is a no-op
    /// rather than a blindfold.
    fn keep<'a>(&'a self, ask: &'a Ask<'a>) -> JudgeFuture<'a>;
}

/// The judge a mode name selects, or `full` when the name is not one of ours.
///
/// An unknown mode is not an error: a pack is allowed to store a name this
/// build does not implement (it may be newer, or a typo), and the safe reading
/// of "I do not know that mode" is to show everything rather than nothing.
pub fn for_mode(mode: Option<&str>, settings: &crate::settings::SkillJudgeSettings) -> Box<dyn SkillJudge> {
    match mode.map(str::trim).filter(|m| !m.is_empty()) {
        Some("regex") => Box::new(regex::RegexJudge::new(settings)),
        Some("jev") => Box::new(jev::JevJudge::new(settings)),
        Some("llm") => Box::new(llm::LlmJudge::new(settings)),
        _ => Box::new(full::FullJudge),
    }
}

/// Every mode name this build implements, for a panel that offers a choice.
pub const MODES: [&str; 4] = ["full", "regex", "jev", "llm"];
