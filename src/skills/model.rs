//! The skill model: a discovered skill, what a scan found, and the registry.
//!
//! A skill is a directory holding a `SKILL.md` whose frontmatter names it and
//! says when it applies. This file holds the three types that describe one —
//! [`Skill`], [`ScanReport`], [`SkillRegistry`] — and the scan that fills the
//! registry.
//!
//! Only the catalog (id + description + path) goes into the prompt. The body is
//! loaded on demand through the `skill` tool, which keeps the standing prompt
//! small and stable — a requirement of the cache policy, not just tidiness.
//! Scanning is cheap and idempotent, so a rescan is how hot reload picks up
//! added or changed skills.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::Result;

use super::parse::{parse_skill, unique_id};

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
    /// Words or phrases that should make this skill apply, from frontmatter
    /// `triggers:`.
    ///
    /// Separate from the description because they are read by different
    /// readers: the model reads the description and judges, and the `regex`
    /// mode reads these and matches. A description written for a human ("检查
    /// Moodle 作业") makes a poor matcher, and a matcher list makes a poor
    /// description. Splitting them lets an author write each well.
    ///
    /// Accepts either a comma-separated string or a YAML list, since both are
    /// what people write.
    pub triggers: Vec<String>,
}

impl Skill {
    /// The catalog line for this skill.
    ///
    /// The `SKILL.md` path is part of the line, not an extra: a model that is
    /// told only a name and a description can decide a skill applies but has no
    /// way to read it, so the catalog and the body live in two separate worlds.
    /// Naming the file closes that gap with the one `read` it already knows how
    /// to make. It also survives a front end that has no `skill` tool.
    pub(super) fn catalog_line(&self) -> String {
        let where_ = self.path.display();
        if self.description.is_empty() {
            format!("- {} ({})", self.id, where_)
        } else {
            format!("- {}: {} ({})", self.id, self.description, where_)
        }
    }
}

/// What the model is told to do with the catalog.
///
/// These are not a style preference. Each one is a failure this catalog had in
/// a real session: a skill the user named by name being ignored, a model
/// answering from what a skill's *name* suggested instead of reading it, three
/// overlapping skills all loaded and their instructions fighting, and a
/// subagent handed the reading task so the main agent never saw the rules it
/// was meant to follow.
pub const CATALOG_RULES: &str = r###"- Discovery: The list above is the skills available in this session (name + description + short path). Skill bodies live on disk at the absolute paths listed above.
- Trigger rules: If the user names a skill (with `$SkillName` or plain text) OR the task clearly matches a skill's description shown above, you must use that skill for that turn. Multiple mentions mean use them all. Do not carry skills across turns unless re-mentioned.
- Missing/blocked: If a named skill isn't in the list or the path can't be read, say so briefly and continue with the best fallback.
- How to use a skill (progressive disclosure):
  1) After deciding to use a skill, the main agent must expand the listed short `path` with the matching alias from `### Skill roots`, then open and read its `SKILL.md` completely before taking task actions. If a read is truncated or paginated, continue until EOF.
  2) When `SKILL.md` references relative paths (e.g., `scripts/foo.py`), resolve them relative to the directory containing that expanded `SKILL.md` first, and only consider other paths if needed.
  3) If `SKILL.md` points to extra folders such as `references/`, use its routing instructions to identify the files required for the task. The main agent must read each required instruction or reference file itself before acting on it. Do not delegate reading, summarizing, or interpreting skill instructions to a subagent. Subagents may still perform task work when the selected skill allows it.
  4) If `scripts/` exist, prefer running or patching them instead of retyping large code blocks.
  5) If `assets/` or templates exist, reuse them instead of recreating from scratch.
- Coordination and sequencing:
  - If multiple skills apply, choose the minimal set that covers the request and state the order you'll use them.
  - Announce which skill(s) you're using and why (one short line). If you skip an obvious skill, say why.
- Context hygiene:
  - Progressive disclosure applies to selecting relevant files, not partially reading a selected instruction file. Do not load unrelated references, scripts, or assets.
  - Avoid deep reference-chasing: prefer opening only files directly linked from `SKILL.md` unless you're blocked.
  - When variants exist (frameworks, providers, domains), pick only the relevant reference file(s) and note that choice.
- Safety and fallback: If a skill can't be applied cleanly (missing files, unclear instructions), state the issue, pick the next-best approach, and continue."###;

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
        self.catalog_with_tools(&[], None)
    }

    /// The catalog, plus which tools each skill's MCP server actually provides.
    ///
    /// The tool table is where the model looks for an action, and a skill's
    /// instructions are a *second* lookup. Measured in a real session, that gap
    /// was decisive: with six `mcp__moodle-herzen__*` tools among 114 and a
    /// perfectly good skill description, the model ran
    /// `bash` + `find / -name "*moodle*"` 441 times against 198 calls to the
    /// tools themselves. Nothing in the prompt said the two were the same
    /// thing, so the familiar shell won.
    ///
    /// A skill named `moodle` owns the tools whose name starts with
    /// `mcp__moodle`, which is how the MCP host namespaces them. Listing them
    /// beside the skill is the whole fix: the model now reads "here is the
    /// skill, and here are the exact tools it governs".
    pub fn catalog_with_tools(
        &self,
        tools: &[String],
        keep: Option<&std::collections::HashSet<String>>,
    ) -> Option<String> {
        if self.skills.is_empty() {
            return None;
        }
        let mut out = String::from(
            "## 技能\n\
             技能是一份写在 `SKILL.md` 里的指令。下面是本会话可用的技能。\n\
             一个技能名下的 MCP 工具直接用它，不要用 `bash` 重新实现服务。\n\n\
             ### 可用技能\n",
        );
        for skill in self.skills.values() {
            // A mode may narrow this turn's catalog. `None` means no narrowing,
            // which is the old behaviour and stays the default so nothing that
            // does not opt in changes shape.
            if let Some(keep) = keep {
                if !keep.contains(&skill.id) {
                    continue;
                }
            }
            out.push_str(&skill.catalog_line());
            let owned: Vec<&String> = tools
                .iter()
                .filter(|tool| {
                    let name = tool.as_str();
                    name.starts_with(&format!("mcp__{}__", skill.id))
                        || name.starts_with(&format!("mcp__{}-", skill.id))
                })
                .collect();
            if !owned.is_empty() {
                let mut names: Vec<&str> = owned.iter().map(|t| t.as_str()).collect();
                names.sort_unstable();
                out.push_str("  tools: ");
                out.push_str(&names.join(", "));
            }
            out.push('\n');
        }
        Some(out)
    }
}

