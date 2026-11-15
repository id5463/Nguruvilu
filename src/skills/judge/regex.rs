//! Rules and no model: frontmatter `triggers:` matched against the request.
//!
//! Free and predictable, which is why it is the mode to reach for when a wrong
//! decision has to be explainable. It is also the one that fails silently: a
//! request phrased in words the author did not anticipate matches nothing, and
//! the skill it should have loaded never appears.
//!
//! Matching is case-insensitive substring containment, not word boundaries.
//! Chinese has no spaces, so a word boundary would never match, and a request
//! that says "看看moodle" should match a trigger "moodle" — demanding whitespace
//! around a term is what turned a correct rule into zero results.

use std::collections::HashSet;


use super::{Ask, JudgeFuture, SkillJudge};
use crate::settings::SkillJudgeSettings;

pub struct RegexJudge {
    /// Extra phrases that apply to every skill, from settings.
    extra: Vec<String>,
}

impl RegexJudge {
    pub fn new(settings: &SkillJudgeSettings) -> Self {
        Self {
            extra: settings
                .regex
                .as_ref()
                .map(|r| r.extra_triggers.clone())
                .unwrap_or_default()
                .into_iter()
                .map(|t| t.trim().to_lowercase())
                .filter(|t| !t.is_empty())
                .collect(),
        }
    }
}

impl SkillJudge for RegexJudge {
    fn mode(&self) -> &'static str {
        "regex"
    }

    fn keep<'a>(&'a self, ask: &'a Ask<'a>) -> JudgeFuture<'a> {
        Box::pin(async move {
        let haystack = ask.query.to_lowercase();
        // A trigger written for another language is useless here, so both sides
        // are lowercased with the same function — `to_lowercase` on the query
        // and on the trigger — and matched as bytes of the same encoding.
        let mut keep = HashSet::new();
        for skill in ask.skills {
            let mut hit = skill
                .triggers
                .iter()
                .any(|t| !t.is_empty() && haystack.contains(&t.to_lowercase()));
            // A skill with no triggers at all matches on its own id and name:
            // the author wrote neither, so the two things they did write are
            // the only honest fallback.
            if !hit && skill.triggers.is_empty() {
                hit = haystack.contains(&skill.id.to_lowercase())
                    || (!skill.name.is_empty() && haystack.contains(&skill.name.to_lowercase()));
            }
            // Settings-wide extras: phrases the user keeps needing that no
            // skill author anticipated.
            if !hit {
                hit = self.extra.iter().any(|t| haystack.contains(t));
            }
            if hit {
                keep.insert(skill.id.clone());
            }
        }
        Ok(keep)
        })
    }
}
