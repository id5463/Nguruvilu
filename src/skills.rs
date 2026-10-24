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
    ///
    /// Separate from [`Self::skipped`] because an absent root is ordinary — a
    /// checkout simply has no `.nguruvilu/skills` — and a caller that only
    /// wants to know which roots are absent should not have to filter a list of
    /// failures. Nothing lands in both.
    pub missing_roots: Vec<PathBuf>,
    /// Paths that *were* there and contributed no skill, each with the reason: a
    /// directory without `SKILL.md`, an entry or root that could not be read, a
    /// file that would not parse.
    ///
    /// The reason is carried rather than just the path because the failure a
    /// user actually reports is "my skills disappeared", and a bare path in a
    /// log gives them nothing to act on: permission denied, not a directory,
    /// unreadable file are three different fixes.
    pub skipped: Vec<(PathBuf, String)>,
    /// Ids that were already taken: `(id offered, id used)`.
    ///
    /// The later root does **not** replace the earlier one. Both stay in the
    /// catalog under different ids, because a skill body is read into the model
    /// as instructions: letting a workspace directory silently displace a skill
    /// the user installed themselves would hand the model an instruction set
    /// nobody asked for.
    pub collisions: Vec<(String, String)>,
}

impl ScanReport {
    /// The scan as user-facing text, one fact per line.
    ///
    /// This is the only place the reasons are rendered, so every caller that
    /// wants to tell a user what happened to their skills prints this instead of
    /// reformatting the fields itself — otherwise a reason added here would go
    /// unread by half the front ends.
    pub fn summary(&self) -> String {
        let mut out = format!("{} skill(s) loaded", self.loaded);
        for (path, reason) in &self.skipped {
            out.push_str(&format!("\nskipped {}: {reason}", path.display()));
        }
        for (offered, used) in &self.collisions {
            out.push_str(&format!(
                "\nskill '{offered}' was already taken; the later copy is catalogued as '{used}'"
            ));
        }
        out
    }
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
    ///
    /// Priority follows this order — the first root to offer an id keeps it. The
    /// user's own skills are therefore listed first, so a `.nguruvilu/skills`
    /// directory inside a checkout cannot take over a skill the user installed
    /// (a skill body becomes model instructions, and a workspace is content
    /// nobody vetted). A workspace copy of an existing id is still catalogued,
    /// under a suffixed id.
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
    ///
    /// It is deliberately hard to fail. One unreadable root or one broken entry
    /// loses that path and nothing else: returning `Err` there used to throw
    /// away every skill the scan had already found, so a single permission
    /// problem anywhere turned the whole catalog into "no skills". Failures are
    /// recorded in [`ScanReport::skipped`] with the reason instead, and the
    /// `Result` remains only so a future whole-scan failure has a shape to use.
    pub fn scan(&mut self) -> Result<ScanReport> {
        let mut report = ScanReport::default();
        let mut found: BTreeMap<String, Skill> = BTreeMap::new();

        for root in self.roots.clone() {
            if !root.exists() {
                // A root that is simply not there is not a failure: `default_roots`
                // offers paths the user may never have created, and reporting
                // that at every boot trains people to ignore the report that
                // also carries real failures. `missing_roots` is the quiet list
                // for exactly this case; `skipped` means "present and broken".
                report.missing_roots.push(root);
                continue;
            }
            let entries = match std::fs::read_dir(&root) {
                Ok(entries) => entries,
                Err(error) => {
                    // A root that cannot even be listed is recorded and passed
                    // over: the other roots still hold skills worth cataloguing.
                    report.skipped.push((
                        root,
                        format!("skills root could not be read: {error}"),
                    ));
                    continue;
                }
            };

            for entry in entries {
                let entry = match entry {
                    Ok(entry) => entry,
                    Err(error) => {
                        // The iterator itself can fail per entry; attributing it
                        // to the root is the most precise thing available,
                        // because the failing entry has no path to name.
                        report.skipped.push((
                            root.clone(),
                            format!("an entry in this root could not be read: {error}"),
                        ));
                        continue;
                    }
                };
                let dir = entry.path();
                if !dir.is_dir() {
                    continue;
                }
                let manifest = dir.join("SKILL.md");
                if !manifest.is_file() {
                    report.skipped.push((dir, "no SKILL.md in this directory".to_string()));
                    continue;
                }
                match parse_skill(&manifest, &dir, &root) {
                    Ok(mut skill) => {
                        // Roots are scanned in priority order and the first one
                        // to offer an id keeps it: `default_roots` lists
                        // `$NGU_HOME/skills` first so the user's own skill
                        // outranks a copy checked into a workspace. The later
                        // copy is neither dropped nor allowed to overwrite — it
                        // lands under a suffixed id and the collision is
                        // reported, so a repository cannot displace a skill the
                        // user installed.
                        let id = unique_id(&skill.id, &found);
                        if id != skill.id {
                            report.collisions.push((skill.id.clone(), id.clone()));
                            skill.id = id;
                        }
                        found.insert(skill.id.clone(), skill);
                    }
                    // Unreadable or unusable `SKILL.md`: one lost skill, not a
                    // lost scan. The context chain names the file and the OS
                    // error, which is exactly what the reason has to say.
                    Err(error) => report.skipped.push((dir, format!("{error:#}"))),
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
/// The two shapes real skills use beyond flat pairs — block scalars and
/// quoted values — are handled explicitly (see inside).
///
/// The walk runs over the **original** text and advances by the bytes it really
/// consumed. `str::lines()` is deliberately not used for offsets: it strips the
/// `\r` of a CRLF file while a caller has to guess the newline width, so every
/// frontmatter line in a Windows `SKILL.md` (CRLF is the default there) shifted
/// the body one byte early. The body then began inside the closing `---`, and
/// when the drift landed mid-character `str::get` returned `None` — which the
/// old `.unwrap_or("")` turned into a **silently empty body**. A skill that
/// loads with no instructions is the worst of the three outcomes.
fn split_frontmatter(raw: &str) -> (BTreeMap<String, String>, String) {
    let mut fields = BTreeMap::new();
    let trimmed = raw.strip_prefix('\u{feff}').unwrap_or(raw);

    // The opening delimiter has to be the first line, exactly `---`.
    let Some(first_newline) = trimmed.find('\n') else {
        // No newline at all: nothing can be frontmatter, so all of it is body.
        return (fields, trimmed.to_string());
    };
    if trimmed[..first_newline].trim() != "---" {
        return (fields, trimmed.to_string());
    }

    // A YAML block scalar (`>` folded or `|` literal, optionally chomped with
    // -/+) keeps its value on the following indented lines. Treating those as
    // pairs would silently drop a long description and smuggle its sentences
    // in as fake keys — Claude Code skills use this shape routinely, so
    // reading their frontmatter needs it.
    let mut block: Option<(String, bool)> = None; // (key, folded?)
    let mut cursor = first_newline + 1;
    while cursor < trimmed.len() {
        let rest = &trimmed[cursor..];
        // Byte-exact: the line, plus whatever newline actually ended it.
        let (line, next) = match rest.find('\n') {
            Some(newline) => (&rest[..newline], cursor + newline + 1),
            None => (rest, trimmed.len()),
        };
        let line = line.strip_suffix('\r').unwrap_or(line);
        cursor = next;

        if line.trim() == "---" {
            // A slice of the original text: it cannot drift, and it cannot be
            // cut mid-character.
            return (fields, trimmed[cursor..].to_string());
        }

        if let Some((key, folded)) = block.clone() {
            let indented = line.starts_with([' ', '\t']) || line.trim().is_empty();
            if indented {
                let text = line.trim();
                if !text.is_empty() {
                    let slot = fields.entry(key).or_default();
                    if folded {
                        if !slot.is_empty() {
                            slot.push(' ');
                        }
                    } else if !slot.is_empty() {
                        slot.push('\n');
                    }
                    slot.push_str(text);
                }
                continue;
            }
            // Not indented any more: the block ended.
            block = None;
        }

        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        let key = key.trim().to_ascii_lowercase();
        let value = value.trim();
        // `>`, `|-`, `|+` … — an indicator, never a value.
        let scalar_shape = !value.is_empty()
            && matches!(value.as_bytes()[0], b'>' | b'|')
            && value.bytes().all(|b| matches!(b, b'>' | b'|' | b'-' | b'+'));
        if scalar_shape {
            fields.entry(key.clone()).or_default();
            block = Some((key, value.as_bytes()[0] == b'>'));
            continue;
        }
        fields.insert(key, unquote(value).to_string());
    }

    // Unterminated frontmatter: treat the whole file as body.
    (BTreeMap::new(), trimmed.to_string())
}

/// A key that is not taken yet: `id`, else `id-2`, `id-3`, …
///
/// Used when two roots offer the same skill id. Neither copy is discarded: a
/// workspace that happens to check in `.nguruvilu/skills/pdf` must not be able
/// to make the user's own `pdf` skill disappear from the catalog.
fn unique_id(wanted: &str, taken: &BTreeMap<String, Skill>) -> String {
    if !taken.contains_key(wanted) {
        return wanted.to_string();
    }
    for suffix in 2.. {
        let candidate = format!("{wanted}-{suffix}");
        if !taken.contains_key(&candidate) {
            return candidate;
        }
    }
    unreachable!("a free suffix always exists")
}

/// Strip one matching pair of surrounding quotes.
///
/// Quoted values are how a description carries a colon without the flat
/// parser splitting on it, so the quotes have to go after the split.
fn unquote(value: &str) -> &str {
    let bytes = value.as_bytes();
    if value.len() >= 2 {
        let (first, last) = (bytes[0], bytes[value.len() - 1]);
        if (first == b'"' && last == b'"') || (first == b'\'' && last == b'\'') {
            return &value[1..value.len() - 1];
        }
    }
    value
}

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
    fn block_scalar_and_quoted_values_survive() {
        // The Claude Code skill shape: long descriptions run as a folded
        // block, unknown fields ride along, and quoted values carry colons
        // the flat parser must not split on.
        let raw = concat!(
            "---\n",
            "name: pdf-tools\n",
            "description: >\n",
            "  Extract text and tables from PDF files.\n",
            "  Works when: pdftotext is missing.\n",
            "allowed-tools: Read, Grep\n",
            "license: MIT\n",
            "---\n",
            "\n",
            "Do the thing.\n"
        );
        let (fields, body) = split_frontmatter(raw);
        assert_eq!(
            fields.get("description").unwrap(),
            "Extract text and tables from PDF files. Works when: pdftotext is missing."
        );
        assert_eq!(fields.get("allowed-tools").unwrap(), "Read, Grep");
        assert_eq!(fields.get("license").unwrap(), "MIT");
        assert_eq!(body.trim(), "Do the thing.");

        let quoted = "---\ndescription: \"Extract: text from PDFs\"\n---\n\nBody\n";
        let (fields, body) = split_frontmatter(quoted);
        assert_eq!(fields.get("description").unwrap(), "Extract: text from PDFs");
        assert_eq!(body.trim(), "Body");
    }

    #[test]
    fn a_claude_code_shaped_skill_end_to_end() {
        let root = temp_root("claude-shape");
        let dir = root.join("incident-review");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("SKILL.md"),
            "---\nname: incident-review\ndescription: >\n  Review an incident timeline and\n  draft the postmortem.\nallowed-tools: Read, Bash(git log:*)\nlicense: MIT\n---\n\nSteps:\n1. Read the timeline.\n",
        )
        .unwrap();

        let mut registry = SkillRegistry::with_roots([root]);
        registry.scan().unwrap();

        let skill = registry.get("incident-review").expect("loaded");
        assert_eq!(skill.description, "Review an incident timeline and draft the postmortem.");
        assert!(skill.body.contains("1. Read the timeline."));
        // The catalog is what the model sees: description in, body out.
        let catalog = registry.catalog().unwrap();
        assert!(catalog.contains("incident-review: Review an incident timeline"));
        assert!(!catalog.contains("Read the timeline"));
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
        // Only the present-but-empty directory is a skip: an absent root is
        // ordinary and lives in `missing_roots`, so a boot on a checkout
        // without `.nguruvilu/skills` stays quiet.
        assert_eq!(report.skipped.len(), 1, "{report:?}");
        assert_eq!(report.skipped[0].0, root.join("not-a-skill"));
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
    fn a_missing_root_is_reported_with_a_reason_and_keeps_the_other_roots() {
        // One absent root must cost that root and nothing else: the skills the
        // other roots offered are still catalogued.
        let good = temp_root("missing-root-good");
        write_skill(&good, "pdf-tools", "pdf-tools", "Read PDFs", "Use pdftotext.");
        let absent = good.join("nowhere");

        let mut registry = SkillRegistry::with_roots([good.clone()]);
        registry.add_root(absent.clone());
        // A root that exists but is a file cannot be listed. That is the case
        // that has to be loud, and it must not cost the readable root either.
        let not_a_directory = good.join("root-is-a-file");
        std::fs::write(&not_a_directory, b"not a directory").unwrap();
        registry.add_root(not_a_directory.clone());
        let report = registry.scan().unwrap();

        assert_eq!(report.loaded, 1, "the readable root still loads: {report:?}");
        assert!(registry.get("pdf-tools").is_some());
        assert_eq!(report.missing_roots, vec![absent.clone()]);

        // The absent root is quiet: it is in `missing_roots` and nowhere else,
        // because every checkout is missing roots the user never created.
        assert!(
            !report.skipped.iter().any(|(path, _)| path == &absent),
            "an absent root is not a failure: {report:?}"
        );
        assert!(
            !report.summary().contains(&absent.display().to_string()),
            "the summary stays quiet about a root that is simply not there: {}",
            report.summary()
        );

        // The unlistable root is the opposite: named, with the OS reason.
        let reason = report
            .skipped
            .iter()
            .find_map(|(path, reason)| (path == &not_a_directory).then(|| reason.clone()))
            .expect("the broken root is reported, not dropped");
        assert!(!reason.is_empty(), "a skip without a reason explains nothing");
        assert!(
            report.summary().contains(&not_a_directory.display().to_string()),
            "the summary names the path: {}",
            report.summary()
        );
        assert!(
            report.summary().contains(&reason),
            "the summary prints the reason: {}",
            report.summary()
        );
    }

    #[test]
    fn an_unreadable_entry_is_skipped_with_a_reason_while_its_siblings_load() {
        // A `SKILL.md` that is not valid UTF-8 fails `read_to_string`. The old
        // scan propagated that as an error and discarded every skill it had
        // already found, so one bad file meant an empty catalog.
        let root = temp_root("unreadable-entry");
        write_skill(&root, "good", "good", "Still loads", "body");
        let broken = root.join("broken");
        std::fs::create_dir_all(&broken).unwrap();
        std::fs::write(broken.join("SKILL.md"), [b'-', b'-', b'-', b'\n', 0xff, 0xfe, b'\n'])
            .unwrap();

        let mut registry = SkillRegistry::with_roots([root]);
        let report = registry.scan().unwrap();

        assert_eq!(report.loaded, 1, "the sibling survives: {report:?}");
        assert!(registry.get("good").is_some());
        assert_eq!(registry.len(), 1);

        let (path, reason) = report
            .skipped
            .iter()
            .find_map(|(path, reason)| {
                let named = path
                    .file_name()
                    .map(|name| name.to_string_lossy() == "broken")
                    .unwrap_or(false);
                named.then(|| (path.clone(), reason.clone()))
            })
            .expect("the broken entry is reported");
        assert_eq!(path, broken);
        assert!(!reason.is_empty(), "a skip without a reason explains nothing");
        assert!(
            report.summary().contains(&broken.display().to_string()),
            "the summary names the directory: {}",
            report.summary()
        );
    }

    #[test]
    fn a_duplicate_id_from_a_later_root_keeps_the_first_and_adds_the_second() {
        // The first root is the user's own, the second is a checkout. A skill
        // body becomes model instructions, so the workspace copy must not be
        // able to displace the user's — but it is not dropped either.
        let user = temp_root("collide-user");
        let workspace = temp_root("collide-workspace");
        write_skill(&user, "search", "search", "User search", "user body");
        write_skill(&workspace, "search", "search", "Workspace search", "workspace body");

        let mut registry = SkillRegistry::with_roots([user, workspace]);
        let report = registry.scan().unwrap();

        assert_eq!(registry.len(), 2, "both copies stay in the catalog");
        assert_eq!(
            registry.get("search").unwrap().description,
            "User search",
            "the first root keeps the plain id"
        );
        let second = registry.get("search-2").expect("the later copy under a free id");
        assert_eq!(second.description, "Workspace search");
        assert_eq!(
            report.collisions,
            vec![("search".to_string(), "search-2".to_string())],
            "the collision is reported, not silent"
        );
    }

    #[test]
    fn a_crlf_skill_keeps_its_body() {
        // Windows writes `SKILL.md` with CRLF. The old offset arithmetic counted
        // one byte per line after `lines()` had already eaten the `\r`, so the
        // body slice started inside the closing delimiter — and once the drift
        // grew past it the cut landed mid-character, `str::get` returned `None`,
        // and `.unwrap_or("")` turned the whole body into an empty string.
        let raw = concat!(
            "---\r\n",
            "name: 中文技能\r\n",
            "description: 中文描述\r\n",
            "allowed-tools: Read\r\n",
            "license: MIT\r\n",
            "version: 1.0.0\r\n",
            "author: someone\r\n",
            "tags: a, b\r\n",
            "notes: 更多中文内容\r\n",
            "---\r\n",
            "\r\n",
            "步骤一：先读文件。\r\n",
            "步骤二：再改代码。\r\n"
        );
        let (fields, body) = split_frontmatter(raw);
        assert_eq!(fields.get("name").unwrap(), "中文技能");
        assert_eq!(fields.get("description").unwrap(), "中文描述");
        assert!(!body.trim().is_empty(), "the body must not be empty: {body:?}");
        assert!(
            body.trim_start().starts_with("步骤一"),
            "the body must start at the body, not inside the delimiter: {body:?}"
        );
        assert!(body.contains("步骤二"));
        assert!(!body.contains("---"), "no frontmatter residue: {body:?}");
    }

    #[test]
    fn a_crlf_skill_loads_through_a_scan() {
        let root = temp_root("crlf-scan");
        let dir = root.join("cn");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("SKILL.md"),
            "---\r\nname: cn\r\ndescription: 中文\r\nallowed-tools: Read\r\nlicense: MIT\r\n\
             owner: me\r\nversion: 2\r\n---\r\n\r\n第一行指令。\r\n第二行指令。\r\n",
        )
        .unwrap();

        let mut registry = SkillRegistry::with_roots([root]);
        let report = registry.scan().unwrap();
        assert_eq!(report.loaded, 1, "the skill is not skipped: {report:?}");

        let skill = registry.get("cn").expect("loaded");
        assert!(
            skill.body.starts_with("第一行指令"),
            "the body survived the CRLF frontmatter: {:?}",
            skill.body
        );
        assert!(skill.body.contains("第二行指令"));
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
        assert!(listed.text.contains("pdf-tools"));

        let loaded = tools
            .execute("skill", &json!({ "action": "load", "id": "pdf-tools" }).to_string())
            .await
            .unwrap();
        assert!(loaded.text.contains("Run pdftotext first"));

        let missing = tools
            .execute("skill", &json!({ "action": "load", "id": "nope" }).to_string())
            .await
            .expect_err("unknown skill");
        assert!(format!("{missing:#}").contains("unknown skill"));
    }
}
