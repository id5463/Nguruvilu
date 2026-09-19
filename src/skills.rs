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

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use serde_json::json;

use crate::tools::{ConflictPolicy, ToolDef, ToolFuture, ToolRegistry};

/// One skill discovered on disk.
#[derive(Debug, Clone)]
pub struct Skill {
    /// Identifier used by the `skill` tool and for de-duplication.
    pub id: String,
    /// Display name from frontmatter, falling back to the id.
    pub name: String,
    /// One-line description shown in the catalog.
    pub description: String,
    /// The `SKILL.md` path.
    pub path: PathBuf,
    /// The directory holding the skill.
    pub dir: PathBuf,
    /// Instructions, with the frontmatter removed.
    pub body: String,
    /// Root this skill was found under.
    pub source: PathBuf,
}

impl Skill {
    /// The catalog line for this skill.
    fn catalog_line(&self) -> String {
        if self.description.is_empty() {
            format!("- {}", self.id)
        } else {
            format!("- {}: {}", self.id, self.description)
        }
    }
}

/// Outcome of a scan.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ScanReport {
    /// Skills now registered.
    pub loaded: usize,
    /// Roots that did not exist.
    pub missing_roots: Vec<PathBuf>,
    /// Directories that looked like skills but had no usable `SKILL.md`.
    pub skipped: Vec<PathBuf>,
}

/// Every skill visible to the kernel, across all configured roots.
#[derive(Debug, Default, Clone)]
pub struct SkillRegistry {
    roots: Vec<PathBuf>,
    skills: BTreeMap<String, Skill>,
}

impl SkillRegistry {
    /// An empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// A registry scanning the given roots.
    pub fn with_roots(roots: impl IntoIterator<Item = PathBuf>) -> Self {
        let mut registry = Self::new();
        for root in roots {
            registry.add_root(root);
        }
        registry
    }

    /// The default roots: `$NGU_HOME/skills`, then `<cwd>/.nguruvilu/skills`.
    pub fn default_roots() -> Vec<PathBuf> {
        let mut roots = Vec::new();
        if let Ok(home) = std::env::var("NGU_HOME") {
            roots.push(PathBuf::from(home).join("skills"));
        }
        roots.push(
            std::env::current_dir()
                .unwrap_or_else(|_| PathBuf::from("."))
                .join(".nguruvilu")
                .join("skills"),
        );
        roots
    }

    /// Add a scan root. Adding the same root twice is a no-op.
    pub fn add_root(&mut self, root: impl Into<PathBuf>) {
        let root = root.into();
        if !self.roots.contains(&root) {
            self.roots.push(root);
        }
    }

    /// Remove a scan root and forget the skills that came from it.
    pub fn remove_root(&mut self, root: &Path) -> usize {
        let before = self.skills.len();
        self.roots.retain(|r| r != root);
        self.skills.retain(|_, s| s.source != root);
        before - self.skills.len()
    }

    /// The configured roots.
    pub fn roots(&self) -> &[PathBuf] {
        &self.roots
    }

    /// Rescan every root, replacing the current contents.
    ///
    /// A rescan is the reload path for skills: it is idempotent, so calling it
    /// after files change is enough to pick the change up.
    pub fn scan(&mut self) -> Result<ScanReport> {
        let mut report = ScanReport::default();
        let mut found: BTreeMap<String, Skill> = BTreeMap::new();

        for root in self.roots.clone() {
            if !root.exists() {
                report.missing_roots.push(root);
                continue;
            }
            let entries = std::fs::read_dir(&root)
                .with_context(|| format!("reading skills root {}", root.display()))?;

            for entry in entries {
                let entry = entry?;
                let dir = entry.path();
                if !dir.is_dir() {
                    continue;
                }
                let manifest = dir.join("SKILL.md");
                if !manifest.is_file() {
                    report.skipped.push(dir);
                    continue;
                }
                match parse_skill(&manifest, &dir, &root) {
                    Ok(skill) => {
                        // A later root shadows an earlier one with the same id,
                        // so a user root can override a shipped skill.
                        found.insert(skill.id.clone(), skill);
                    }
                    Err(_) => report.skipped.push(dir),
                }
            }
        }

        report.loaded = found.len();
        self.skills = found;
        Ok(report)
    }

    /// Look a skill up by id (case-insensitive).
    pub fn get(&self, id: &str) -> Option<&Skill> {
        let needle = id.trim().to_ascii_lowercase();
        self.skills
            .get(id.trim())
            .or_else(|| self.skills.values().find(|s| s.id.to_ascii_lowercase() == needle))
    }

    /// Every skill, ordered by id.
    pub fn list(&self) -> Vec<&Skill> {
        self.skills.values().collect()
    }

    /// Number of registered skills.
    pub fn len(&self) -> usize {
        self.skills.len()
    }

    /// Whether no skills are registered.
    pub fn is_empty(&self) -> bool {
        self.skills.is_empty()
    }

    /// The catalog block injected into the prompt, or `None` when empty.
    pub fn catalog(&self) -> Option<String> {
        if self.skills.is_empty() {
            return None;
        }
        let mut out = String::from(
            "Available skills. Load one with the `skill` tool when it applies:\n",
        );
        for skill in self.skills.values() {
            out.push_str(&skill.catalog_line());
            out.push('\n');
        }
        Some(out)
    }
}

/// Parse one `SKILL.md`.
fn parse_skill(manifest: &Path, dir: &Path, root: &Path) -> Result<Skill> {
    let raw = std::fs::read_to_string(manifest)
        .with_context(|| format!("reading {}", manifest.display()))?;
    let (fields, body) = split_frontmatter(&raw);

    let fallback_id = dir
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "skill".into());

    let id = fields
        .get("name")
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or(fallback_id);

    let description = fields
        .get("description")
        .map(|s| s.trim().to_string())
        .unwrap_or_default();

    Ok(Skill {
        name: id.clone(),
        id,
        description,
        path: manifest.to_path_buf(),
        dir: dir.to_path_buf(),
        body: body.trim().to_string(),
        source: root.to_path_buf(),
    })
}

/// Split a `---` frontmatter block from the body.
///
/// Only flat `key: value` pairs are read; skills do not need a full YAML
/// parser and pulling one in would be more machinery than the format deserves.
fn split_frontmatter(raw: &str) -> (BTreeMap<String, String>, String) {
    let mut fields = BTreeMap::new();
    let trimmed = raw.strip_prefix('\u{feff}').unwrap_or(raw);

    let mut lines = trimmed.lines();
    let Some(first) = lines.next() else {
        return (fields, String::new());
    };
    if first.trim() != "---" {
        return (fields, trimmed.to_string());
    }

    let mut body_start = None;
    let mut offset = first.len() + 1;
    for line in lines {
        offset += line.len() + 1;
        if line.trim() == "---" {
            body_start = Some(offset);
            break;
        }
        if let Some((key, value)) = line.split_once(':') {
            fields.insert(key.trim().to_ascii_lowercase(), value.trim().to_string());
        }
    }

    match body_start {
        Some(start) => (fields, trimmed.get(start..).unwrap_or("").to_string()),
        // Unterminated frontmatter: treat the whole file as body.
        None => (BTreeMap::new(), trimmed.to_string()),
    }
}

/// Register the `skill` tool against a registry.
///
/// The tool exposes two actions instead of two tools: listing and loading are
/// the same concern, and one schema costs fewer prompt tokens.
pub fn register_skill_tool(tools: &mut ToolRegistry, skills: Arc<SkillRegistry>) -> Result<()> {
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
            "kernel",
            move |args| {
                let skills = Arc::clone(&skills);
                Box::pin(async move { skill_tool(skills, args).await }) as ToolFuture
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

#[cfg(test)]
mod tests {
    use super::*;

    fn write_skill(root: &Path, id: &str, name: &str, description: &str, body: &str) {
        let dir = root.join(id);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("SKILL.md"),
            format!("---\nname: {name}\ndescription: {description}\n---\n\n{body}\n"),
        )
        .unwrap();
    }

    fn temp_root(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("ngu-skill-{tag}-{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn frontmatter_is_split_from_body() {
        let (fields, body) = split_frontmatter("---\nname: a\ndescription: d\n---\n\nBody text\n");
        assert_eq!(fields.get("name").unwrap(), "a");
        assert_eq!(fields.get("description").unwrap(), "d");
        assert_eq!(body.trim(), "Body text");
    }

    #[test]
    fn a_file_without_frontmatter_is_all_body() {
        let (fields, body) = split_frontmatter("# Just a heading\n");
        assert!(fields.is_empty());
        assert_eq!(body, "# Just a heading\n");
    }

    #[test]
    fn scan_finds_skills_and_reports_missing_roots() {
        let root = temp_root("scan");
        write_skill(&root, "pdf-tools", "pdf-tools", "Read PDFs", "Use pdftotext.");
        write_skill(&root, "csv-tools", "csv-tools", "Work with CSV", "Use xsv.");
        // A directory without SKILL.md must be reported, not silently ignored.
        std::fs::create_dir_all(root.join("not-a-skill")).unwrap();

        let mut registry = SkillRegistry::with_roots([root.clone()]);
        registry.add_root(root.join("does-not-exist"));
        let report = registry.scan().unwrap();

        assert_eq!(report.loaded, 2);
        assert_eq!(report.missing_roots.len(), 1);
        assert_eq!(report.skipped.len(), 1);
        assert_eq!(registry.len(), 2);

        let pdf = registry.get("pdf-tools").expect("skill found");
        assert_eq!(pdf.description, "Read PDFs");
        assert!(pdf.body.contains("pdftotext"));

        let catalog = registry.catalog().expect("catalog exists");
        assert!(catalog.contains("pdf-tools: Read PDFs"));
        assert!(catalog.contains("csv-tools: Work with CSV"));
        // The body must not leak into the standing prompt.
        assert!(!catalog.contains("pdftotext"));
    }

    #[test]
    fn later_roots_shadow_earlier_ones() {
        let base = temp_root("shadow-base");
        let user = temp_root("shadow-user");
        write_skill(&base, "search", "search", "Base search", "base body");
        write_skill(&user, "search", "search", "User search", "user body");

        let mut registry = SkillRegistry::with_roots([base, user]);
        registry.scan().unwrap();
        assert_eq!(registry.len(), 1);
        assert_eq!(registry.get("search").unwrap().description, "User search");
    }

    #[test]
    fn rescan_picks_up_new_skills() {
        let root = temp_root("rescan");
        let mut registry = SkillRegistry::with_roots([root.clone()]);
        registry.scan().unwrap();
        assert_eq!(registry.len(), 0);

        write_skill(&root, "late", "late", "Added later", "body");
        registry.scan().unwrap();
        assert_eq!(registry.len(), 1);
        assert!(registry.get("late").is_some());
    }

    #[test]
    fn removing_a_root_forgets_its_skills() {
        let root = temp_root("remove");
        write_skill(&root, "temp", "temp", "d", "b");
        let mut registry = SkillRegistry::with_roots([root.clone()]);
        registry.scan().unwrap();
        assert_eq!(registry.len(), 1);

        assert_eq!(registry.remove_root(&root), 1);
        assert_eq!(registry.len(), 0);
        assert!(registry.roots().is_empty());
    }

    #[tokio::test]
    async fn skill_tool_lists_and_loads() {
        let root = temp_root("tool");
        write_skill(&root, "pdf-tools", "pdf-tools", "Read PDFs", "Run pdftotext first.");
        let mut scanned = SkillRegistry::with_roots([root]);
        scanned.scan().unwrap();
        let registry = Arc::new(scanned);
        let mut tools = ToolRegistry::with_base_tools().unwrap();
        register_skill_tool(&mut tools, Arc::clone(&registry)).unwrap();

        let listed = tools.execute("skill", &json!({ "action": "list" }).to_string()).await.unwrap();
        assert!(listed.contains("pdf-tools"));

        let loaded = tools
            .execute("skill", &json!({ "action": "load", "id": "pdf-tools" }).to_string())
            .await
            .unwrap();
        assert!(loaded.contains("Run pdftotext first"));

        let missing = tools
            .execute("skill", &json!({ "action": "load", "id": "nope" }).to_string())
            .await
            .expect_err("unknown skill");
        assert!(format!("{missing:#}").contains("unknown skill"));
    }
}
