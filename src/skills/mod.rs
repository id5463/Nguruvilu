//! Skills: on-demand instruction packs.
//!
//! A skill is a directory containing `SKILL.md`. The file starts with a small
//! frontmatter block naming the skill and describing when it applies:
//!
//! ```markdown
//! ---
//! name: pdf-tools
//! description: Extract text and tables from PDF files
//! ---
//!
//! # PDF tools
//! ...full instructions...
//! ```
//!
//! Only the catalog (id + description) goes into the prompt. The body is
//! loaded on demand through the `skill` tool, which keeps the standing prompt
//! small and stable — a requirement of the cache policy, not just tidiness.
//!
//! Scanning is cheap and idempotent, so a rescan is how hot reload picks up
//! added or changed skills.
//!
//! # Layout
//!
//! One file per function:
//!
//! * [`model`] — [`Skill`], [`ScanReport`], [`SkillRegistry`], and the scan
//!   that fills the registry from its roots.
//! * [`parse`] — reading one `SKILL.md`: `parse_skill`, `split_frontmatter`,
//!   `unique_id`, `unquote`.
//! * [`tool`] — the `skill` tool: `register_skill_tool`,
//!   `register_skill_tool_owned`, `skill_tool`.
//!
//! Every item this module exported before is re-exported here, so
//! `crate::skills::X` still names the same item.

mod model;
mod parse;
mod tool;

pub use model::{ScanReport, Skill, SkillRegistry, CATALOG_RULES};
pub use tool::{register_skill_tool, register_skill_tool_owned, SKILL_TOOL_OWNER};
