//! Ask a decision model, once per turn, about every skill at the same time.
//!
//! The provider answers `noul` questions with a probability, so one call takes
//! the whole catalog as `state` and asks one yes/no per skill as `questions`.
//! Measured at 70–500 ms and about $0.00002 for six skills: cheap enough to run
//! every turn, slow enough that it is not free.
//!
//! One call rather than one call per skill is not an optimisation detail. The
//! questions share a single `state`, so the model judges every skill against
//! the same reading of the request; asked separately it re-reads the request
//! six times and the answers stop being comparable.

use std::collections::HashSet;

use serde_json::{json, Value};

use super::{Ask, JudgeFuture, SkillJudge};
use crate::settings::SkillJudgeSettings;
use crate::tools::judge;

pub struct JevJudge {
    settings: judge::JudgeSettings,
    threshold: f64,
}

impl JevJudge {
    pub fn new(settings: &SkillJudgeSettings) -> Self {
        let jev = settings.jev.clone().unwrap_or_default();
        Self {
            settings: jev.settings(),
            threshold: jev.threshold,
        }
    }
}

impl SkillJudge for JevJudge {
    fn mode(&self) -> &'static str {
        "jev"
    }

    fn keep<'a>(&'a self, ask: &'a Ask<'a>) -> JudgeFuture<'a> {
        Box::pin(async move {
        if !self.settings.is_configured() {
            anyhow::bail!(
                "jev 模式需要密钥：在设置面板填写 JeV 的 api_key，或导出 NGU_JEV_KEY"
            );
        }
        if ask.skills.is_empty() {
            return Ok(HashSet::new());
        }

        // The catalog as the model reads it: name, description, and the words
        // the author said should trigger it. The body is deliberately absent —
        // it is long, and the question is whether the skill applies, not what
        // it says.
        let mut state = String::from("用户请求：\n");
        state.push_str(ask.query);
        state.push_str("\n\n可用技能：\n");
        for skill in ask.skills {
            state.push_str(&format!("- {}: {}", skill.id, skill.description));
            if !skill.triggers.is_empty() {
                state.push_str(&format!(" [触发词: {}]", skill.triggers.join(", ")));
            }
            state.push('\n');
        }

        // One question per skill, keyed by id so the answer maps back without
        // relying on order.
        let mut questions = serde_json::Map::new();
        for skill in ask.skills {
            questions.insert(
                skill.id.clone(),
                json!({
                    "type": "noul",
                    "instructions": format!(
                        "处理这个用户请求时，需要用到「{}」这个技能吗？",
                        skill.id
                    ),
                }),
            );
        }

        let answer = judge::evaluate(&self.settings, Value::String(state), Value::Object(questions))
            .await?;

        // `answers.<id>.noul` is the probability. A provider that answers with
        // a boolean under `value` instead is read the same way, because the
        // shape is the provider's to choose and refusing to read it would turn
        // a correct answer into an error.
        let mut keep = HashSet::new();
        let answers = answer.get("answers").and_then(Value::as_object);
        for skill in ask.skills {
            let Some(entry) = answers.and_then(|a| a.get(&skill.id)) else {
                continue;
            };
            let probability = entry
                .get("noul")
                .and_then(Value::as_f64)
                .or_else(|| entry.get("value").and_then(Value::as_bool).map(|b| if b { 1.0 } else { 0.0 }))
                .unwrap_or(0.0);
            if probability >= self.threshold {
                keep.insert(skill.id.clone());
            }
        }
        Ok(keep)
        })
    }
}
