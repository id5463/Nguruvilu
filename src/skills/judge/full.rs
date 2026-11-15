//! Show every skill. The default, and the only mode with nothing to go wrong.

use std::collections::HashSet;


use super::{Ask, JudgeFuture, SkillJudge};

pub struct FullJudge;

impl SkillJudge for FullJudge {
    fn mode(&self) -> &'static str {
        "full"
    }

    fn keep<'a>(&'a self, ask: &'a Ask<'a>) -> JudgeFuture<'a> {
        Box::pin(async move {
        Ok(ask.skills.iter().map(|s| s.id.clone()).collect())
        })
    }
}
