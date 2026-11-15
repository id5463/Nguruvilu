//! Ask a language model to pick from the whole catalog at once.
//!
//! The only mechanism that can reason about a request no rule anticipated, and
//! the only one that costs a real request. It reuses the judge transport rather
//! than the chat endpoint because the answer wanted here is a short JSON list,
//! not prose: a chat model asked to "pick the skills" writes a paragraph, and
//! parsing a paragraph is how a correct pick becomes a missed one.

use std::collections::HashSet;

use serde_json::{json, Value};

use super::{Ask, JudgeFuture, SkillJudge};
use crate::settings::SkillJudgeSettings;
use crate::tools::judge;

pub struct LlmJudge {
    settings: judge::JudgeSettings,
    max: usize,
}

impl LlmJudge {
    pub fn new(settings: &SkillJudgeSettings) -> Self {
        let llm = settings.llm.clone().unwrap_or_default();
        Self {
            settings: llm.settings(),
            max: llm.max_skills,
        }
    }
}

impl SkillJudge for LlmJudge {
    fn mode(&self) -> &'static str {
        "llm"
    }

    fn keep<'a>(&'a self, ask: &'a Ask<'a>) -> JudgeFuture<'a> {
        Box::pin(async move {
        if !self.settings.is_configured() {
            anyhow::bail!(
                "llm 模式需要密钥：在设置面板填写大模型的 api_key，或导出 NGU_LLM_KEY"
            );
        }
        if ask.skills.is_empty() {
            return Ok(HashSet::new());
        }

        let mut state = String::from("用户请求：\n");
        state.push_str(ask.query);
        state.push_str("\n\n可用技能（id: 描述）：\n");
        let mut ids = Vec::new();
        for skill in ask.skills {
            ids.push(skill.id.clone());
            state.push_str(&format!("- {}: {}", skill.id, skill.description));
            if !skill.triggers.is_empty() {
                state.push_str(&format!(" [触发词: {}]", skill.triggers.join(", ")));
            }
            state.push('\n');
        }

        // A `choice` question with every skill as an option would break on a
        // catalog wider than the provider's option cap, so the answer is asked
        // for as a score per skill: same information, no cap.
        let mut questions = serde_json::Map::new();
        for skill in ask.skills {
            questions.insert(
                skill.id.clone(),
                json!({
                    "type": "score",
                    "instructions": format!(
                        "处理这个用户请求需要用到「{}」的程度，0 表示完全不需要，100 表示必须用。",
                        skill.id
                    ),
                    "range": [0, 100],
                }),
            );
        }

        let answer = judge::evaluate(&self.settings, Value::String(state), Value::Object(questions))
            .await?;

        let mut scored: Vec<(f64, String)> = Vec::new();
        let answers = answer.get("answers").and_then(Value::as_object);
        for skill in ask.skills {
            let Some(entry) = answers.and_then(|a| a.get(&skill.id)) else {
                continue;
            };
            let score = entry
                .get("score")
                .and_then(Value::as_f64)
                .or_else(|| entry.get("value").and_then(Value::as_f64))
                .unwrap_or(0.0);
            if score > 0.0 {
                scored.push((score, skill.id.clone()));
            }
        }
        scored.sort_by(|a, b| b.0.total_cmp(&a.0));
        scored.truncate(self.max.max(1));
        Ok(scored.into_iter().map(|(_, id)| id).collect())
        })
    }
}
