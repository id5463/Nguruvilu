//! The `skill` tool: the model's way to list and load a skill.
//!
//! One tool with two actions rather than two tools: listing and loading are the
//! same concern, and one schema costs fewer prompt tokens. The catalog is
//! already in the prompt; this tool is how a body is fetched once the model has
//! decided one applies.

use std::sync::Arc;

use anyhow::Result;
use serde_json::json;

use crate::tools::{ConflictPolicy, ToolDef, ToolFuture, ToolRegistry};

use super::model::SkillRegistry;

/// Owner recorded on the `skill` tool.
///
/// Not `"kernel"`. The tool is a *view* of the catalog: `loader::merge_skills`
/// unregisters and re-registers it whenever the roots change, and any future
/// unload path needs to tell it apart from the tools the kernel itself owns. The
/// old hard-coded owner claimed kernel ownership for a tool that is rebuilt from
/// pack content.
pub const SKILL_TOOL_OWNER: &str = "skills";

/// Register the `skill` tool against a registry.
///
/// The tool exposes two actions instead of two tools: listing and loading are
/// the same concern, and one schema costs fewer prompt tokens.
pub fn register_skill_tool(tools: &mut ToolRegistry, skills: Arc<SkillRegistry>) -> Result<()> {
    register_skill_tool_owned(tools, skills, SKILL_TOOL_OWNER)
}

/// [`register_skill_tool`] with an explicit owner, for a caller that reloads the
/// catalog and needs to recognise its own registration (see [`SKILL_TOOL_OWNER`]).
pub fn register_skill_tool_owned(
    tools: &mut ToolRegistry,
    skills: Arc<SkillRegistry>,
    owner: &'static str,
) -> Result<()> {
    tools.register(
        ToolDef::new(
            "skill",
            "List the available skills, or load one to read its full instructions. \
             Load a skill before doing work it covers.",
            json!({
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["list", "load"],
                        "description": "list the catalog, or load one skill's instructions"
                    },
                    "id": {
                        "type": "string",
                        "description": "Skill id, required when action is 'load'"
                    }
                },
                "required": ["action"]
            }),
            owner,
            move |args| {
                let skills = Arc::clone(&skills);
                Box::pin(async move { skill_tool(skills, args).await.map(Into::into) }) as ToolFuture
            },
        ),
        ConflictPolicy::Error,
    )?;
    Ok(())
}

async fn skill_tool(skills: Arc<SkillRegistry>, args: serde_json::Value) -> Result<String> {
    let action = args
        .get("action")
        .and_then(|a| a.as_str())
        .unwrap_or("list");

    match action {
        "list" => {
            if skills.is_empty() {
                return Ok(format!(
                    "no skills found (searched: {})",
                    skills
                        .roots()
                        .iter()
                        .map(|r| r.display().to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
            }
            let mut out = String::from("Skills:\n");
            for skill in skills.list() {
                out.push_str(&skill.catalog_line());
                out.push('\n');
            }
            Ok(out)
        }
        "load" => {
            let id = args
                .get("id")
                .and_then(|i| i.as_str())
                .ok_or_else(|| anyhow::anyhow!("action 'load' requires an 'id'"))?;
            let skill = skills
                .get(id)
                .ok_or_else(|| anyhow::anyhow!("unknown skill '{id}'"))?;
            let mut out = format!("# Skill: {}\n", skill.id);
            if !skill.description.is_empty() {
                out.push_str(&format!("{}\n", skill.description));
            }
            out.push_str("\n");
            out.push_str(&skill.body);
            Ok(out)
        }
        other => Err(anyhow::anyhow!("unknown action '{other}'; expected 'list' or 'load'")),
    }
}

