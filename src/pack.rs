//! Packs: `.dshpack`, the distribution format.
//!
//! A pack is a zip holding a **manifest of references** plus the small data
//! files that cannot be fetched from anywhere — persona, model route, context
//! policy, MCP servers, appearance, injection rules.
//!
//! ```text
//! my-pack-1.0.0.dshpack
//! ├── dsh.index.json     the manifest: what to fetch, and what is inside
//! ├── soul.md            persona
//! ├── models.json        how to talk to the model
//! ├── context.json       how much to remember
//! ├── mcp.json           MCP servers
//! ├── look.json          themes and panels
//! └── injections.json    injection rules
//! ```
//!
//! The payload — skills and plugins — is **not** in the archive. It is named by
//! `github:` or `https:` reference with a sha256, and fetched on install into a
//! content-addressed cache. Two packs naming the same skill share one download.
//!
//! # Two shapes
//!
//! | | manifest pack | offline pack |
//! |---|---|---|
//! | size | kilobytes | megabytes per platform |
//! | install | fetches what it names | needs no network |
//! | platforms | one manifest covers all | one archive per platform |
//!
//! The offline shape is the same zip with the payload embedded under `files/`,
//! which is what `--offline` produces. It exists for air-gapped and archived
//! installs; the manifest shape is the normal one.
//!
//! # Install materialises a loadable directory
//!
//! The loader reads `assembly.yaml`. Install therefore writes one, generated
//! from the manifest and the content files, so the loader needs to know nothing
//! about the distribution format. The two formats change for different reasons
//! and are allowed to change separately.

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};

use crate::fetch::{Fetcher, Source};

/// Manifest file name inside the archive.
pub const MANIFEST_NAME: &str = "dsh.index.json";

/// Assembly file name the loader reads.
pub const ASSEMBLY_NAME: &str = "assembly.yaml";

/// Format version this build writes and accepts.
pub const FORMAT_VERSION: u32 = 1;

/// The `game` field's expected value.
pub const GAME: &str = "nguruvilu";

/// The value earlier packs wrote. Accepted so an existing offline pack keeps
/// loading; new packs use [`GAME`].
pub const LEGACY_GAME: &str = "dsh";

/// Directory holding embedded payload in an offline pack.
pub const FILES_DIR: &str = "files";

/// Field name to file name, for the content a pack carries.
///
/// The order is the order they are written and reported, so diagnostics read
/// the same way twice.
pub const CONTENT_FILES: &[(&str, &str)] = &[
    ("soul", "soul.md"),
    ("models", "models.json"),
    ("context", "context.json"),
    ("mcp", "mcp.json"),
    ("look", "look.json"),
    ("injections", "injections.json"),
];

/// The kernel version this build is.
pub fn kernel_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

// ---------------------------------------------------------------- manifest

/// A pack's manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PackManifest {
    /// Format version; must equal [`FORMAT_VERSION`].
    ///
    /// The snake_case spelling is the older one; both read, so a pack written
    /// before this format settled keeps loading.
    #[serde(alias = "format_version")]
    pub format_version: u32,
    /// Must equal [`GAME`].
    ///
    /// Defaulted rather than required, and both the current and the older value
    /// are accepted, for the same reason.
    #[serde(default = "default_game")]
    pub game: String,
    /// Pack name.
    #[serde(default)]
    pub name: String,
    /// Pack version.
    #[serde(alias = "version_id", default)]
    pub version_id: String,
    /// Distribution licence.
    ///
    /// Absent is not an error: this kernel reports what an archive says
    /// rather than requiring it to say anything in particular.
    #[serde(default)]
    pub license: String,
    /// Kernel version this pack was built against.
    #[serde(alias = "kernel_version", default)]
    pub kernel_version: String,
    /// Accepted kernel range.
    #[serde(default)]
    pub dependencies: Dependencies,
    /// Skills to fetch.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub skills: Vec<SkillRef>,
    /// Plugins to fetch.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub plugins: Vec<PluginRef>,
    /// Licence attribution for content the pack carries itself.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub components: Vec<Component>,

    /// Persona file, relative to the pack root.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub soul: Option<ContentRef>,
    /// Model route file.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub models: Option<ContentRef>,
    /// Context policy file.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context: Option<ContentRef>,
    /// MCP server file.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mcp: Option<ContentRef>,
    /// Appearance file.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub look: Option<ContentRef>,
    /// Injection rule file.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub injections: Option<ContentRef>,
    /// A user interface this pack brings.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ui: Option<UiDecl>,
}

/// Accepted kernel version range.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Dependencies {
    /// A range such as `>=0.1.0 <0.2.0`.
    ///
    /// Accepts the older `kernel` spelling, which named the same thing before
    /// this program settled on one name for itself. An absent range accepts
    /// anything: refusing would make every pack written before the field
    /// existed unloadable, and the field exists to help, not to gate.
    #[serde(alias = "kernel", default = "default_range")]
    pub nguruvilu: String,
}

fn default_game() -> String {
    GAME.to_string()
}

fn default_range() -> String {
    ">=0.0.0".to_string()
}

    /// A user interface a pack brings.
///
/// The interface is a whole client, not a panel inside the built-in one. It is
/// plain web assets — the desktop shell is a web view — so a single set of
/// files covers every platform, and it can be as large as it needs to be
/// because it is fetched rather than carried.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct UiDecl {
    /// Identifier, used to select it.
    #[serde(default)]
    pub id: String,
    /// Title shown when choosing.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub title: String,
    /// The entry file within the interface directory.
    #[serde(default = "default_entry")]
    pub entry: String,
    /// Where the interface comes from, when it is fetched.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    /// sha256 of the fetched directory.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
}

fn default_entry() -> String {
    "index.html".into()
}

impl UiDecl {
    /// A carried interface.
    pub fn carried(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            title: String::new(),
            entry: default_entry(),
            source: None,
            sha256: None,
        }
    }

    /// Whether this interface is fetched rather than carried.
    pub fn is_fetched(&self) -> bool {
        self.source.is_some()
    }
}

/// Where one content file comes from.
///
/// Either carried in the pack or fetched, and the author chooses. A persona or
/// a context policy is small and belongs in the pack; a user interface is not —
/// one can carry a rendering engine — and fetching it keeps the pack small
/// enough to send.
///
/// A bare string is the carried form, so the common case stays one line:
///
/// ```jsonc
/// "soul": "soul.md",
/// "ui": { "path": "index.html", "source": "github:owner/ui@dist@v1.0.0" }
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ContentRef {
    /// A path inside the pack.
    Carried(String),
    /// A path inside a fetched directory.
    Fetched {
        /// Path inside the fetched directory.
        path: String,
        /// Where the directory comes from.
        source: String,
        /// sha256 of the fetched directory.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        sha256: Option<String>,
    },
}

impl From<&str> for ContentRef {
    fn from(path: &str) -> Self {
        ContentRef::Carried(path.to_string())
    }
}

impl From<String> for ContentRef {
    fn from(path: String) -> Self {
        ContentRef::Carried(path)
    }
}

impl ContentRef {
    /// A path inside the pack.
    pub fn carried(path: impl Into<String>) -> Self {
        ContentRef::Carried(path.into())
    }

    /// The file within whichever directory holds it.
    pub fn path(&self) -> &str {
        match self {
            ContentRef::Carried(path) => path,
            ContentRef::Fetched { path, .. } => path,
        }
    }

    /// The source to fetch from, when it is not carried.
    pub fn source(&self) -> Option<&str> {
        match self {
            ContentRef::Carried(_) => None,
            ContentRef::Fetched { source, .. } => Some(source),
        }
    }

    /// The declared hash, when it is fetched.
    pub fn sha256(&self) -> Option<&str> {
        match self {
            ContentRef::Carried(_) => None,
            ContentRef::Fetched { sha256, .. } => sha256.as_deref(),
        }
    }

    /// A one-line description, for diagnostics.
    pub fn describe(&self) -> String {
        match self {
            ContentRef::Carried(path) => path.clone(),
            ContentRef::Fetched { path, source, .. } => format!("{source} → {path}"),
        }
    }
}


impl PackManifest {
    /// A manifest for a new pack, built against this kernel.
    pub fn new(name: impl Into<String>, version_id: impl Into<String>) -> Self {
        Self {
            format_version: FORMAT_VERSION,
            game: GAME.into(),
            name: name.into(),
            version_id: version_id.into(),
            license: "MIT".into(),
            kernel_version: kernel_version().into(),
            dependencies: Dependencies {
                nguruvilu: format!(">={}", kernel_version()),
            },
            skills: Vec::new(),
            plugins: Vec::new(),
            components: Vec::new(),
            soul: None,
            models: None,
            context: None,
            mcp: None,
            look: None,
            injections: None,
            ui: None,
        }
    }

    /// The content file named by a field, if any.
    pub fn content_file(&self, field: &str) -> Option<&ContentRef> {
        match field {
            "soul" => self.soul.as_ref(),
            "models" => self.models.as_ref(),
            "context" => self.context.as_ref(),
            "mcp" => self.mcp.as_ref(),
            "look" => self.look.as_ref(),
            "injections" => self.injections.as_ref(),
            _ => None,
        }
    }

    /// Every reference this pack names.
    /// Every reference this pack names, with its source parsed.
    ///
    /// Returns an error rather than skipping an unparseable source: silently
    /// dropping one would install a pack that is missing something it asked
    /// for, which is the failure mode that is hardest to notice.
    pub fn references(&self) -> Result<Vec<Reference<'_>>> {
        let mut out = Vec::new();
        for skill in &self.skills {
            out.push(Reference {
                kind: "skill",
                id: &skill.id,
                source: Source::parse(&skill.source)
                    .with_context(|| format!("skill '{}'", skill.id))?,
                sha256: skill.sha256.as_deref(),
            });
        }
        for plugin in &self.plugins {
            out.push(Reference {
                kind: "plugin",
                id: &plugin.id,
                source: Source::parse(&plugin.source)
                    .with_context(|| format!("plugin '{}'", plugin.id))?,
                sha256: plugin.sha256.as_deref(),
            });
        }
        Ok(out)
    }

    /// Point a content field at a new location.
    ///
    /// Used by install: content that arrived over the wire is rewritten to the
    /// path it was placed at, so the installed pack is self-contained and
    /// nothing downstream needs to know where it came from.
    pub fn set_content(&mut self, field: &str, reference: ContentRef) {
        match field {
            "soul" => self.soul = Some(reference),
            "models" => self.models = Some(reference),
            "context" => self.context = Some(reference),
            "mcp" => self.mcp = Some(reference),
            "look" => self.look = Some(reference),
            "injections" => self.injections = Some(reference),
            _ => {}
        }
    }

    /// Record a fetched hash against the reference it belongs to.
    ///
    /// Used by pack --pin, which fetches each reference so an install can
    /// find it in the cache instead of downloading it again.
    pub fn set_hash(&mut self, kind: &str, id: &str, sha256: &str) -> Result<()> {
        match kind {
            "skill" => {
                let entry = self
                    .skills
                    .iter_mut()
                    .find(|skill| skill.id == id)
                    .ok_or_else(|| anyhow!("no skill '{id}'"))?;
                entry.sha256 = Some(sha256.to_string());
            }
            "plugin" => {
                let entry = self
                    .plugins
                    .iter_mut()
                    .find(|plugin| plugin.id == id)
                    .ok_or_else(|| anyhow!("no plugin '{id}'"))?;
                // A per-platform pack pins the artifact for this platform;
                // the others are pinned when built on them.
                let tag = crate::fetch::platform_tag();
                if let Some(artifact) = entry.platforms.get_mut(tag) {
                    artifact.sha256 = sha256.to_string();
                } else {
                    entry.sha256 = Some(sha256.to_string());
                }
            }
            other => return Err(anyhow!("cannot pin a '{other}'")),
        }
        Ok(())
    }

    /// Things that do not block an install but that an author should know.
    ///
    /// Separate from [`PackManifest::problems`] because these are judgement
    /// calls rather than mistakes: a pack under development legitimately
    /// references a branch, and refusing to build it would make the format
    /// annoying to work with. A pack being published does not.
    pub fn warnings(&self) -> Vec<String> {
        let mut warnings = Vec::new();

        // A remote reference with no hash is floating: it fetches whatever the
        // ref points at today. A tag is not a pin — tags move — so the hash is
        // the only thing that fixes the bytes. `ngu pack --pin` records it.
        for reference in self.references().unwrap_or_default() {
            if !reference.source.is_remote() || reference.sha256.is_some() {
                continue;
            }
            let detail = match &reference.source {
                Source::Github { git_ref, .. } if git_ref == "HEAD" => {
                    "it names no ref and no sha256, so it follows the default branch".to_string()
                }
                Source::Github { git_ref, .. } => format!(
                    "ref '{git_ref}' can move and there is no sha256, so the content is not fixed"
                ),
                _ => "there is no sha256, so the content is not fixed".to_string(),
            };
            warnings.push(format!(
                "{} '{}' is not pinned: {detail}. Run `ngu pack --pin` to record the hash.",
                reference.kind, reference.id
            ));
        }

        warnings
    }

/// Problems that make this manifest unusable.
    ///
    /// Returned as a list rather than an error so a validator can report every
    /// problem at once. Fixing them one per run is how a format becomes
    /// annoying to author.
    pub fn problems(&self) -> Vec<String> {
        let mut problems = Vec::new();

        if self.format_version != FORMAT_VERSION {
            problems.push(format!(
                "formatVersion is {}, this build reads {FORMAT_VERSION}",
                self.format_version
            ));
        }
        if !matches!(self.game.as_str(), GAME | LEGACY_GAME) {
            problems.push(format!(
                "game must be \"{GAME}\", found \"{}\"",
                self.game
            ));
        }
        for (field, value) in [
            ("name", &self.name),
            ("versionId", &self.version_id),
            ("license", &self.license),
            ("kernelVersion", &self.kernel_version),
        ] {
            if value.trim().is_empty() {
                problems.push(format!("{field} is empty"));
            }
        }
        if self.dependencies.nguruvilu.trim().is_empty() {
            problems.push("dependencies.nguruvilu is empty; it is the accepted kernel range".into());
        }

        let mut seen: BTreeMap<(&str, &str), ()> = BTreeMap::new();
        for skill in &self.skills {
            if skill.id.trim().is_empty() {
                problems.push("a skill has no id".into());
            }
            if skill.license.trim().is_empty() {
                problems.push(format!(
                    "skill '{}' has no license; every fetched component states its licence",
                    skill.id
                ));
            }
            if seen.insert(("skill", skill.id.as_str()), ()).is_some() {
                problems.push(format!("skill '{}' appears twice", skill.id));
            }
            if let Err(error) = Source::parse(skill.source.as_str()) {
                problems.push(format!("skill '{}': {error:#}", skill.id));
            }
        }

        for plugin in &self.plugins {
            if plugin.id.trim().is_empty() {
                problems.push("a plugin has no id".into());
            }
            if plugin.license.trim().is_empty() {
                problems.push(format!("plugin '{}' has no license", plugin.id));
            }
            if seen.insert(("plugin", plugin.id.as_str()), ()).is_some() {
                problems.push(format!("plugin '{}' appears twice", plugin.id));
            }
            if let Err(error) = Source::parse(plugin.source.as_str()) {
                problems.push(format!("plugin '{}': {error:#}", plugin.id));
            }
            for (tag, artifact) in &plugin.platforms {
                if !crate::fetch::PLATFORMS.contains(&tag.as_str()) {
                    problems.push(format!(
                        "plugin '{}' names platform '{tag}', which is not one of {}",
                        plugin.id,
                        crate::fetch::PLATFORMS.join(", ")
                    ));
                }
                if artifact.file.trim().is_empty() {
                    problems.push(format!(
                        "plugin '{}' platform '{tag}' has no file",
                        plugin.id
                    ));
                }
            }
        }

        if let Some(reference) = self.content_file("mcp") {
            if let Some(source) = reference.source() {
                if let Err(error) = Source::parse(source) {
                    problems.push(format!("mcp: {error:#}"));
                }
            }
        }

        for component in &self.components {
            if component.license.trim().is_empty() {
                problems.push(format!(
                    "component '{}' has no license",
                    component.id
                ));
            }
        }

        for (field, _) in CONTENT_FILES {
            if let Some(reference) = self.content_file(field) {
                if reference.path().trim().is_empty() {
                    problems.push(format!("{field} is set but names no file"));
                }
                // A fetched content file needs a source that parses, for the
                // same reason a skill does.
                if let Some(source) = reference.source() {
                    if let Err(error) = Source::parse(source) {
                        problems.push(format!("{field}: {error:#}"));
                    }
                }
            }
        }

        problems
    }

    /// Whether the declared range accepts `version`.
    pub fn accepts_kernel(&self, version: &str) -> bool {
        accepts_range(&self.dependencies.nguruvilu, version)
    }
}

/// One thing a pack names and where it comes from.
#[derive(Debug, Clone)]
pub struct Reference<'a> {
    /// skill or plugin.
    pub kind: &'static str,
    /// Identifier from the manifest.
    pub id: &'a str,
    /// Parsed source.
    pub source: Source,
    /// sha256 the pack declares.
    pub sha256: Option<&'a str>,
}

/// A skill to fetch.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SkillRef {
    /// Identifier.
    pub id: String,
    /// Where it comes from.
    pub source: String,
    /// sha256 of the directory.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
    /// Distribution licence.
    ///
    /// Absent is not an error: this kernel reports what an archive says
    /// rather than requiring it to say anything in particular.
    #[serde(default)]
    pub license: String,
    /// Runtime the skill expects, for the reader's information.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub deps: BTreeMap<String, String>,
}

/// A plugin to fetch.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PluginRef {
    /// Identifier.
    pub id: String,
    /// Where it comes from.
    pub source: String,
    /// Distribution licence.
    ///
    /// Absent is not an error: this kernel reports what an archive says
    /// rather than requiring it to say anything in particular.
    #[serde(default)]
    pub license: String,
    /// sha256 of the whole fetched directory, when not per-platform.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
    /// Per-platform file and hash.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub platforms: BTreeMap<String, PlatformArtifact>,
}

/// One platform's build of a plugin.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PlatformArtifact {
    /// File name inside the fetched directory.
    pub file: String,
    /// sha256 of the fetched directory.
    pub sha256: String,
}

/// Licence attribution for content a pack carries.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Component {
    /// Identifier.
    #[serde(default)]
    pub id: String,
    /// `skill`, `plugin`, `soul`, `look`, or `rule`.
    pub kind: String,
    /// Distribution licence.
    ///
    /// Absent is not an error: this kernel reports what an archive says
    /// rather than requiring it to say anything in particular.
    #[serde(default)]
    pub license: String,
}

// ------------------------------------------------------------ content files

/// `models.json`: how to talk to the model.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ModelsFile {
    /// Endpoint.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    /// **Name** of the environment variable holding the key. Never the key.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key_env: Option<String>,
    /// Model id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Reasoning effort.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    /// Proxy; empty means direct.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proxy: Option<String>,
    /// Output ceiling, as `8K` or a number.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<String>,
    /// How requests are sent: timeouts, pooling, retries.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub network: Option<crate::network::NetworkSettings>,
    /// Fields merged into every request body.
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub extra_body: serde_json::Map<String, serde_json::Value>,
}

/// `context.json`: how much to remember.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ContextFile {
    /// Window, as `128K` or a number.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub window: Option<String>,
    /// Percentage at which compaction runs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compact_percent: Option<u32>,
    /// Messages kept verbatim.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compact_keep_recent: Option<usize>,
    /// `freshness`, `balanced`, or `cache-first`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_policy: Option<String>,
}

/// `mcp.json`: MCP servers.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct McpFile {
    /// The servers.
    #[serde(default)]
    pub servers: Vec<McpServerDecl>,
}

/// One MCP server declaration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct McpServerDecl {
    /// Identifier.
    pub id: String,
    /// `stdio` today; `streamable-http` is declared but not connected.
    #[serde(default = "default_transport")]
    pub transport: String,
    /// Where the server comes from, when it is fetched rather than installed.
    ///
    /// Without this an MCP server can only be something already on the machine,
    /// which in practice means an npm package run through `npx` — and far more
    /// servers exist in git repositories than in npm. A source makes the
    /// channel a declaration, so a server published anywhere can be used.
    ///
    /// A fetched server lands under `files/mcp/<id>/`, and `command` is
    /// resolved inside it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    /// sha256 of the fetched server.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
    /// Executable. Relative to the fetched directory when there is a source.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub command: Option<String>,
    /// Arguments.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub args: Vec<String>,
    /// Environment additions. `${VAR}` is replaced from the environment.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub env: BTreeMap<String, String>,
    /// `session` or `global`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope: Option<String>,
}

impl LookFile {
    /// The theme in force: the named one, else the first.
    ///
    /// An ctiveTheme naming a theme that is not in the file is reported by
    /// [LookFile::problems] rather than silently falling back.
    pub fn active(&self) -> Option<&crate::theme::Theme> {
        match &self.active_theme {
            Some(name) => self.themes.iter().find(|theme| &theme.name == name),
            None => self.themes.first(),
        }
    }

    /// Problems that would make the appearance file misleading.
    pub fn problems(&self) -> Vec<String> {
        let mut problems = Vec::new();
        if let Some(name) = &self.active_theme {
            if !self.themes.iter().any(|theme| &theme.name == name) {
                problems.push(format!(
                    "activeTheme names '{name}', which is not one of: {}",
                    self.themes
                        .iter()
                        .map(|t| t.name.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
            }
        }
        problems
    }
}

fn default_transport() -> String {
    "stdio".into()
}

/// `look.json`: themes and panels.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct LookFile {
    /// Themes this pack offers.
    ///
    /// Variants, not layers: a pack that ships a light and a dark theme means
    /// the reader picks one. Applying them all would let the last win and hide
    /// the choice.
    #[serde(default)]
    pub themes: Vec<crate::theme::Theme>,
    /// Which of them is in force. Absent means the first.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_theme: Option<String>,
    /// Interface panels.
    #[serde(default)]
    pub panels: Vec<crate::ui::UiPanel>,
}

/// `injections.json`: the injection engine's own serialised form.
///
/// Reusing the engine's format means a pack author can write the same file the
/// kernel already reads, and there is one format rather than two.
pub fn read_injections(path: &Path) -> Result<crate::context::InjectionEngine> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_str(&text)
        .with_context(|| format!("{} is not a valid injection file", path.display()))
}

// ---------------------------------------------------------------- versions

/// Whether a `>=x <y` style range accepts `version`.
///
/// Deliberately small: space-separated comparators, each `>=`, `>`, `<=`, `<`,
/// or `=`. A pack's compatibility claim is a blunt instrument and a full semver
/// grammar would suggest a precision it does not have.
pub fn accepts_range(range: &str, version: &str) -> bool {
    let Some(current) = parse_version(version) else {
        return false;
    };
    for clause in range.split_whitespace() {
        let (operator, rest) = ["<=", ">=", "<", ">", "="]
            .iter()
            .find_map(|op| clause.strip_prefix(op).map(|rest| (*op, rest)))
            .unwrap_or(("=", clause));
        let Some(bound) = parse_version(rest) else {
            // An unparseable bound cannot be satisfied; refusing is safer than
            // accepting a claim nobody can evaluate.
            return false;
        };
        let satisfied = match operator {
            ">=" => current >= bound,
            ">" => current > bound,
            "<=" => current <= bound,
            "<" => current < bound,
            _ => current == bound,
        };
        if !satisfied {
            return false;
        }
    }
    true
}

/// Parse `1.2.3`, `1.2`, or `1` into comparable numbers, ignoring any suffix.
fn parse_version(text: &str) -> Option<(u64, u64, u64)> {
    let core = text.trim().trim_start_matches('v');
    // Pre-release and build metadata do not participate: a preview kernel is
    // compared on its numbers, which is what the pack author can reason about.
    let core = core.split(['-', '+']).next()?;
    let mut parts = core.split('.').map(|p| p.parse::<u64>().ok());
    let major = parts.next()??;
    let minor = parts.next().flatten().unwrap_or(0);
    let patch = parts.next().flatten().unwrap_or(0);
    Some((major, minor, patch))
}

#[cfg(test)]
mod version_tests {
    use super::*;

    #[test]
    fn a_bounded_range_accepts_what_it_names() {
        assert!(accepts_range(">=0.1.0 <0.2.0", "0.1.0"));
        assert!(accepts_range(">=0.1.0 <0.2.0", "0.1.9"));
        assert!(!accepts_range(">=0.1.0 <0.2.0", "0.2.0"));
        assert!(!accepts_range(">=0.1.0 <0.2.0", "0.0.9"));
    }

    #[test]
    fn a_single_comparator_works() {
        assert!(accepts_range(">=0.1.0", "9.9.9"));
        assert!(!accepts_range(">=0.1.0", "0.0.1"));
        assert!(accepts_range("=1.0.0", "1.0.0"));
    }

    #[test]
    fn short_versions_are_filled_in() {
        assert!(accepts_range(">=1", "1.0.0"));
        assert!(accepts_range(">=1.2", "1.2.0"));
    }

    #[test]
    fn a_pre_release_suffix_is_ignored_for_ordering() {
        // A preview kernel compares on its numbers, which is what an author can
        // reason about.
        assert!(accepts_range(">=0.1.0 <0.2.0", "0.1.0-rc.5"));
    }

    #[test]
    fn an_unparseable_range_refuses_rather_than_guessing() {
        assert!(!accepts_range("^0.1.0", "0.1.5"));
        assert!(!accepts_range("latest", "0.1.5"));
        assert!(!accepts_range(">=x", "0.1.5"));
    }

    #[test]
    fn a_leading_v_is_accepted() {
        assert!(accepts_range(">=0.1.0", "v0.1.5"));
    }
}


// ---------------------------------------------------------------- archive

/// What a pack directory holds.
#[derive(Debug, Clone, Default)]
pub struct PackContents {
    /// Files, relative to the pack root, sorted.
    pub files: Vec<String>,
    /// Whether a manifest is present.
    pub has_manifest: bool,
    /// Whether an assembly is present (offline packs carry one).
    pub has_assembly: bool,
}

/// Result of checking an archive without installing it.
#[derive(Debug, Clone)]
pub struct VerifyReport {
    /// The manifest that was read.
    pub manifest: PackManifest,
    /// What the archive contains.
    pub contents: PackContents,
    /// Whether the declared kernel range accepts this kernel.
    pub kernel_compatible: bool,
    /// Problems found, in the order they were discovered.
    pub warnings: Vec<String>,
}

/// An installed pack, as it sits in the packs directory.
#[derive(Debug, Clone)]
pub struct InstalledPack {
    /// Its manifest.
    pub manifest: PackManifest,
    /// Where it is installed.
    pub path: PathBuf,
    /// Its assembly, when it has one.
    pub assembly: Option<PathBuf>,
}

/// Read `dsh.index.json` from a pack directory.
///
/// Reads, and does not judge. A manifest is a declaration by the person who
/// wrote it, and this kernel is a tool rather than a gatekeeper: a pack whose
/// licence is blank or whose version range does not match still installs, and
/// whatever is actually wrong shows up when the thing it describes is used.
/// Refusing here substitutes this program's judgement for the user's, and the
/// user has the authority.
pub fn read_manifest(dir: &Path) -> Result<PackManifest> {
    let path = dir.join(MANIFEST_NAME);
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("reading {} (a pack needs one)", path.display()))?;
    let manifest: PackManifest = serde_json::from_str(&text)
        .with_context(|| format!("{} is not readable as a manifest", path.display()))?;
    Ok(manifest)
}

/// Write a manifest into a pack directory.
pub fn write_manifest(dir: &Path, manifest: &PackManifest) -> Result<()> {
    std::fs::create_dir_all(dir)
        .with_context(|| format!("creating {}", dir.display()))?;
    let path = dir.join(MANIFEST_NAME);
    let text = serde_json::to_string_pretty(manifest)?;
    std::fs::write(&path, format!("{text}\n"))
        .with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

/// List what a pack directory holds.
pub fn inspect(dir: &Path) -> Result<PackContents> {
    let mut contents = PackContents::default();
    collect_files(dir, dir, &mut contents.files)?;
    contents.files.sort();
    contents.has_manifest = dir.join(MANIFEST_NAME).is_file();
    contents.has_assembly = dir.join(ASSEMBLY_NAME).is_file();
    Ok(contents)
}

fn collect_files(root: &Path, dir: &Path, out: &mut Vec<String>) -> Result<()> {
    let entries = std::fs::read_dir(dir)
        .with_context(|| format!("listing {}", dir.display()))?;
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        if entry.file_type()?.is_dir() {
            collect_files(root, &path, out)?;
        } else {
            let relative = path
                .strip_prefix(root)
                .unwrap_or(&path)
                .to_string_lossy()
                .replace('\\', "/");
            out.push(relative);
        }
    }
    Ok(())
}

/// The files a pack would contain, excluding any archive.
///
/// The default output path sits beside the tree being packed, so a scan that
/// does not exclude it counts the archive as a member of itself. Every
/// `.dshpack` is excluded, not only the one being written: an archive left over
/// from an earlier build is still an archive, and embedding one is always wrong
/// — it ships a stale pack inside a fresh one, and the build after that embeds
/// both.
pub fn packable_files(dir: &Path, out: &Path) -> Result<PackContents> {
    let _ = out;
    let mut contents = inspect(dir)?;
    contents
        .files
        .retain(|relative| !relative.to_ascii_lowercase().ends_with(".dshpack"));
    Ok(contents)
}

/// Write a zip of the given files.
fn write_archive(dir: &Path, files: &[String], out: &Path) -> Result<()> {
    if let Some(parent) = out.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating {}", parent.display()))?;
        }
    }
    let file = std::fs::File::create(out)
        .with_context(|| format!("creating {}", out.display()))?;
    let mut writer = zip::ZipWriter::new(file);

    for relative in files {
        let path = dir.join(relative);
        let metadata = std::fs::metadata(&path)
            .with_context(|| format!("reading metadata for {relative}"))?;

        #[allow(unused_mut)]
        let mut options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated)
            .large_file(metadata.len() > u32::MAX as u64);
        // A pack may ship a script; dropping the executable bit would break it.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            options = options.unix_permissions(metadata.permissions().mode());
        }

        writer
            .start_file(relative, options)
            .with_context(|| format!("adding {relative}"))?;
        let data = std::fs::read(&path).with_context(|| format!("reading {relative}"))?;
        writer
            .write_all(&data)
            .with_context(|| format!("writing {relative} into the archive"))?;
    }

    writer.finish().context("finalizing the archive")?;
    Ok(())
}

/// Build a manifest pack: the manifest and its content files, no payload.
pub fn pack(dir: &Path, out: &Path) -> Result<PackManifest> {
    let manifest = read_manifest(dir)?;
    let contents = packable_files(dir, out)?;

    // A pack must not smuggle payload in. Anything under `files/` is what an
    // offline build embeds, and a manifest pack that carries it is neither
    // shape — it would install stale bytes instead of fetching.
    let smuggled: Vec<&String> = contents
        .files
        .iter()
        .filter(|f| f.starts_with(&format!("{FILES_DIR}/")))
        .collect();
    if !smuggled.is_empty() {
        return Err(anyhow!(
            "{} holds payload under {FILES_DIR}/, which a manifest pack must not: \
             build with --offline to embed it deliberately",
            dir.display()
        ));
    }

    write_archive(dir, &contents.files, out)?;
    Ok(manifest)
}

/// Build an offline pack: fetch everything the manifest names and embed it.
pub async fn pack_offline(dir: &Path, out: &Path) -> Result<PackManifest> {
    let manifest = read_manifest(dir)?;
    let fetcher = Fetcher::new()?;
    let files_dir = dir.join(FILES_DIR);
    std::fs::create_dir_all(&files_dir)?;

    for reference in manifest.references()? {
        let Reference { kind, id, source, sha256 } = reference;
        let (source, expected) = (&source, sha256);
        if !source.is_remote() {
            continue;
        }
        let fetched = fetcher.fetch(source, expected).await?;
        let destination = files_dir.join(id);
        if destination.exists() {
            std::fs::remove_dir_all(&destination)?;
        }
        copy_tree(&fetched.path, &destination)?;
        eprintln!("[offline] embedded {kind} {id} ({})", fetched.sha256);
    }

    let contents = packable_files(dir, out)?;
    write_archive(dir, &contents.files, out)?;
    Ok(manifest)
}

fn copy_tree(from: &Path, to: &Path) -> Result<()> {
    std::fs::create_dir_all(to)?;
    for entry in std::fs::read_dir(from)? {
        let entry = entry?;
        let target = to.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_tree(&entry.path(), &target)?;
        } else {
            std::fs::copy(entry.path(), &target)?;
        }
    }
    Ok(())
}

/// Extract an archive into a directory.
pub fn unpack(pack_path: &Path, dest: &Path) -> Result<PackManifest> {
    std::fs::create_dir_all(dest)
        .with_context(|| format!("creating {}", dest.display()))?;

    let file = std::fs::File::open(pack_path)
        .with_context(|| format!("opening {}", pack_path.display()))?;
    let mut archive = zip::ZipArchive::new(file)
        .with_context(|| format!("{} is not a zip archive", pack_path.display()))?;

    for index in 0..archive.len() {
        let mut entry = archive.by_index(index)?;
        let Some(relative) = entry.enclosed_name() else {
            return Err(anyhow!(
                "{} contains an entry that escapes the archive: {}",
                pack_path.display(),
                entry.name()
            ));
        };
        let target = dest.join(relative);
        if entry.is_dir() {
            std::fs::create_dir_all(&target)?;
            continue;
        }
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut buffer = Vec::new();
        entry.read_to_end(&mut buffer)?;
        std::fs::write(&target, buffer)
            .with_context(|| format!("writing {}", target.display()))?;

        #[cfg(unix)]
        if let Some(mode) = entry.unix_mode() {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(&target, std::fs::Permissions::from_mode(mode));
        }
    }

    read_manifest(dest)
}

/// Check an archive without installing it.
pub fn verify(pack_path: &Path) -> Result<VerifyReport> {
    let scratch = std::env::temp_dir().join(format!(
        "ngu-verify-{}",
        uuid::Uuid::new_v4().simple()
    ));
    let result = (|| -> Result<VerifyReport> {
        let manifest = unpack(pack_path, &scratch)?;
        let contents = inspect(&scratch)?;

        // What the archive holds, described rather than judged. erify exists
        // so a person can see inside a pack; it does not decide whether the pack
        // is acceptable, because that is the person's call.
        let mut warnings = Vec::new();
        for (field, _) in CONTENT_FILES {
            let Some(reference) = manifest.content_file(field) else {
                continue;
            };
            // A fetched file is not in the archive; it arrives at install.
            if reference.source().is_some() {
                continue;
            }
            if !scratch.join(reference.path()).is_file() {
                warnings.push(format!(
                    "{field} names {}, which is not in the archive",
                    reference.path()
                ));
            }
        }
        let kernel_compatible = manifest.accepts_kernel(kernel_version());
        if !kernel_compatible {
            warnings.push(format!(
                "this pack accepts nguruvilu {}, this kernel is {}",
                manifest.dependencies.nguruvilu,
                kernel_version()
            ));
        }
        if contents.has_assembly && manifest.references().map(|r| !r.is_empty()).unwrap_or(false) {
            // An offline pack embeds payload; a manifest pack must not.
            warnings.push(
                "the archive carries an assembly and also names references; \
                 it will install as an offline pack"
                    .into(),
            );
        }

        Ok(VerifyReport {
            manifest,
            contents,
            kernel_compatible,
            warnings,
        })
    })();

    let _ = std::fs::remove_dir_all(&scratch);
    result
}

// ------------------------------------------------------------------ install

/// Install an archive into the packs directory.
///
/// Manifest packs fetch what they name; offline packs use what they carry. The
/// result is the same either way: a directory the loader can read.
pub async fn install(pack_path: &Path, packs_dir: &Path) -> Result<InstalledPack> {
    let staging = std::env::temp_dir().join(format!(
        "ngu-install-{}",
        uuid::Uuid::new_v4().simple()
    ));
    let result = install_inner(pack_path, packs_dir, &staging).await;
    let _ = std::fs::remove_dir_all(&staging);
    result
}

async fn install_inner(
    pack_path: &Path,
    packs_dir: &Path,
    staging: &Path,
) -> Result<InstalledPack> {
    let manifest = unpack(pack_path, staging)?;

    // The declared kernel range is not enforced. A pack that names a range this
    // kernel is outside of installs anyway, and whatever the mismatch actually
    // breaks — a tool that never appears, a field that reads as absent — is
    // reported where it happens. Refusing here would be this program deciding
    // for the user that they may not try it.

    // Fetch what the pack names. Anything already in the cache is not
    // re-downloaded, and an offline pack carries its own copy.
    let embedded = staging.join(FILES_DIR);
    let fetcher = Fetcher::new()?;
    for reference in manifest.references()? {
        let Reference { kind, id, source, sha256 } = reference;
        let (source, expected) = (&source, sha256);
        let local = embedded.join(id);
        if local.is_dir() {
            continue;
        }
        if !source.is_remote() {
            continue;
        }
        let fetched = fetcher.fetch(source, expected).await.with_context(|| {
            format!("fetching {kind} '{id}' from {}", source.describe())
        })?;
        copy_tree(&fetched.path, &local)?;
    }

    // Fetch content that lives elsewhere, then rewrite the manifest so the
    // installed pack is self-contained: everything downstream reads a path
    // inside it, and nothing needs to know the content arrived over the wire.
    let mut manifest = manifest;
    let mut rewritten = false;
    for (field, _) in CONTENT_FILES {
        let Some(reference) = manifest.content_file(field).cloned() else {
            continue;
        };
        let Some(source) = reference.source() else {
            continue;
        };
        let parsed = Source::parse(source)
            .with_context(|| format!("{field} source"))?;
        let fetched = fetcher
            .fetch(&parsed, reference.sha256())
            .await
            .with_context(|| format!("fetching {field} from {source}"))?;
        let local = staging.join(FILES_DIR).join(field);
        copy_tree(&fetched.path, &local)?;
        manifest.set_content(
            field,
            ContentRef::carried(format!("{FILES_DIR}/{field}/{}", reference.path())),
        );
        rewritten = true;
    }
    if rewritten {
        write_manifest(staging, &manifest)?;
    }

    // MCP servers named by a source are fetched alongside skills and plugins.
    // A server only reachable through npx is a server the user has to install a
    // runtime for.
    let mcp_decls = match manifest.mcp.as_ref() {
        Some(reference) => {
            let path = staging.join(reference.path());
            if path.is_file() {
                let text = std::fs::read_to_string(&path)?;
                serde_json::from_str::<McpFile>(&text)
                    .with_context(|| format!("{} is not a valid mcp file", path.display()))?
                    .servers
            } else {
                Vec::new()
            }
        }
        None => Vec::new(),
    };
    for server in &mcp_decls {
        let Some(source) = &server.source else {
            continue;
        };
        let parsed = Source::parse(source)
            .with_context(|| format!("mcp '{}' source", server.id))?;
        let fetched = fetcher
            .fetch(&parsed, server.sha256.as_deref())
            .await
            .with_context(|| format!("fetching mcp '{}' from {source}", server.id))?;
        copy_tree(&fetched.path, &staging.join(FILES_DIR).join("mcp").join(&server.id))?;
    }

    // A fetched interface lands under files/ui/, and the declaration is
    // rewritten to point inside the pack so the shell reads one layout whether
    // the interface arrived over the wire or was carried.
    if let Some(ui) = manifest.ui.clone() {
        if let Some(source) = &ui.source {
            let parsed = Source::parse(source)
                .with_context(|| format!("ui '{}' source", ui.id))?;
            let fetched = fetcher
                .fetch(&parsed, ui.sha256.as_deref())
                .await
                .with_context(|| format!("fetching ui '{}' from {source}", ui.id))?;
            let local = staging.join(FILES_DIR).join("ui");
            copy_tree(&fetched.path, &local)?;
            let mut carried = ui.clone();
            carried.source = None;
            carried.sha256 = None;
            carried.entry = format!("{FILES_DIR}/ui/{}", ui.entry.trim_start_matches('/'));
            manifest.ui = Some(carried);
            write_manifest(staging, &manifest)?;
        }
    }

    let destination = packs_dir.join(format!("{}-{}", manifest.name, manifest.version_id));
    if destination.exists() {
        std::fs::remove_dir_all(&destination)
            .with_context(|| format!("replacing {}", destination.display()))?;
    }
    std::fs::create_dir_all(packs_dir)
        .with_context(|| format!("creating {}", packs_dir.display()))?;

    let assembly = render_assembly(&manifest, staging)?;
    std::fs::write(staging.join(ASSEMBLY_NAME), &assembly)?;
    copy_tree(staging, &destination)?;

    let assembly_path = destination.join(ASSEMBLY_NAME);
    Ok(InstalledPack {
        manifest,
        path: destination,
        assembly: assembly_path.is_file().then_some(assembly_path),
    })
}

/// Render the assembly the loader reads, from the manifest and content files.
///
/// Install writes it rather than shipping it, so a pack author never has to
/// maintain two descriptions of the same thing — and so the loader keeps
/// reading one format while the distribution format evolves.
fn render_assembly(manifest: &PackManifest, staging: &Path) -> Result<String> {
    #[derive(Serialize)]
    struct OutBase {
        id: String,
        source: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        scope: Option<String>,
    }
    #[derive(Serialize)]
    struct OutPlugin {
        #[serde(flatten)]
        base: OutBase,
    }
    #[derive(Serialize)]
    struct OutMcp {
        #[serde(flatten)]
        base: OutBase,
        transport: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        command: Option<String>,
        #[serde(skip_serializing_if = "Vec::is_empty")]
        args: Vec<String>,
        #[serde(skip_serializing_if = "BTreeMap::is_empty")]
        env: BTreeMap<String, String>,
    }
    #[derive(Serialize)]
    struct OutSkill {
        #[serde(flatten)]
        base: OutBase,
    }
    #[derive(Serialize)]
    struct OutStage {
        name: String,
        #[serde(skip_serializing_if = "Vec::is_empty")]
        plugins: Vec<OutPlugin>,
        #[serde(skip_serializing_if = "Vec::is_empty")]
        mcp: Vec<OutMcp>,
        #[serde(skip_serializing_if = "Vec::is_empty")]
        skills: Vec<OutSkill>,
    }
    #[derive(Serialize)]
    struct OutDefaults {
        scope: String,
        on_failure: String,
    }
    #[derive(Serialize)]
    struct Out {
        version: u32,
        name: String,
        defaults: OutDefaults,
        stages: Vec<OutStage>,
    }
    let platform = crate::fetch::platform_tag();

    let plugins: Vec<OutPlugin> = manifest
        .plugins
        .iter()
        .map(|plugin| {
            // The loader takes a path inside the pack; the fetched directory is
            // copied to `files/<id>` at install time.
            let file = plugin
                .platforms
                .get(platform)
                .map(|artifact| artifact.file.clone())
                .unwrap_or_else(|| plugin.id.clone());
            OutPlugin {
                base: OutBase {
                    id: plugin.id.clone(),
                    source: format!("dylib:{FILES_DIR}/{}/{file}", plugin.id),
                    scope: Some("global".into()),
                },
            }
        })
        .collect();

    let skills: Vec<OutSkill> = manifest
        .skills
        .iter()
        .map(|skill| OutSkill {
            base: OutBase {
                id: skill.id.clone(),
                source: format!("{FILES_DIR}/{}", skill.id),
                scope: Some("session".into()),
            },
        })
        .collect();

    let mcp: Vec<OutMcp> = match manifest.mcp.as_ref() {
        Some(file) => {
            let path = staging.join(file.path());
            let text = std::fs::read_to_string(&path)
                .with_context(|| format!("reading {}", path.display()))?;
            let parsed: McpFile = serde_json::from_str(&text)
                .with_context(|| format!("{} is not a valid mcp file", path.display()))?;
            parsed
                .servers
                .into_iter()
                .map(|server| {
                    // A fetched server runs from where it landed, so the
                    // command is resolved inside that directory. Without this
                    // the loader would look for it on PATH.
                    let command = match (&server.source, &server.command) {
                        (Some(_), Some(command)) => Some(
                            Path::new(FILES_DIR)
                                .join("mcp")
                                .join(&server.id)
                                .join(command)
                                .to_string_lossy()
                                .replace('\\', "/"),
                        ),
                        (_, command) => command.clone(),
                    };
                    OutMcp {
                        base: OutBase {
                            id: server.id.clone(),
                            source: command
                                .clone()
                                .map(|c| format!("stdio:{c} {}", server.args.join(" ")))
                                .unwrap_or_default()
                                .trim()
                                .to_string(),
                            scope: server.scope,
                        },
                        transport: server.transport,
                        command,
                        args: server.args,
                        env: server.env,
                    }
                })
                .collect()
        }
        None => Vec::new(),
    };

    let stages = if plugins.is_empty() && skills.is_empty() && mcp.is_empty() {
        Vec::new()
    } else {
        vec![OutStage {
            name: "pack".into(),
            plugins,
            mcp,
            skills,
        }]
    };

    let assembly = Out {
        version: 1,
        name: manifest.name.clone(),
        defaults: OutDefaults {
            scope: "session".into(),
            on_failure: "skip".into(),
        },
        stages,
    };
    Ok(serde_yaml::to_string(&assembly)?)
}

/// Every installed pack, newest name first.
pub fn installed(packs_dir: &Path) -> Result<Vec<InstalledPack>> {
    let mut out = Vec::new();
    let entries = match std::fs::read_dir(packs_dir) {
        Ok(entries) => entries,
        // A missing directory means nothing is installed, which is not an error.
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(out),
        Err(error) => {
            return Err(error).with_context(|| format!("listing {}", packs_dir.display()))
        }
    };

    for entry in entries {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let path = entry.path();
        if !path.join(MANIFEST_NAME).is_file() {
            continue;
        }
        // A pack with a broken manifest is reported rather than skipped: a
        // silent omission looks like the pack was never installed.
        let manifest = read_manifest(&path)
            .with_context(|| format!("reading the pack at {}", path.display()))?;
        let assembly = path.join(ASSEMBLY_NAME);
        out.push(InstalledPack {
            manifest,
            path,
            assembly: assembly.is_file().then_some(assembly),
        });
    }
    out.sort_by(|a, b| a.manifest.name.cmp(&b.manifest.name));
    Ok(out)
}

/// Where packs are installed: `$NGU_HOME/packs`, else `~/.nguruvilu/packs`.
pub fn default_packs_dir() -> PathBuf {
    crate::settings::data_dir().join("packs")
}

/// Read a pack's assembly, when it has one.
pub fn read_assembly(dir: &Path) -> Result<Option<crate::assembly::Assembly>> {
    let path = dir.join(ASSEMBLY_NAME);
    if !path.is_file() {
        return Ok(None);
    }
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("reading {}", path.display()))?;
    let assembly: crate::assembly::Assembly = serde_yaml::from_str(&text)
        .with_context(|| format!("{} is not a valid assembly", path.display()))?;
    Ok(Some(assembly))
}

/// A manifest as displayable pairs.
pub fn describe(manifest: &PackManifest) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    out.insert("name".into(), manifest.name.clone());
    out.insert("version".into(), manifest.version_id.clone());
    out.insert("license".into(), manifest.license.clone());
    out.insert("built against".into(), manifest.kernel_version.clone());
    out.insert("accepts".into(), manifest.dependencies.nguruvilu.clone());
    out.insert("skills".into(), manifest.skills.len().to_string());
    out.insert("plugins".into(), manifest.plugins.len().to_string());
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir()
            .join(format!("ngu-pack-{tag}-{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// A pack directory with a manifest and one content file.
    fn sample(dir: &Path, name: &str, version: &str) -> PackManifest {
        let manifest = PackManifest::new(name, version);
        write_manifest(dir, &manifest).unwrap();
        std::fs::write(dir.join("soul.md"), "You are terse.\n").unwrap();
        let mut manifest = manifest;
        manifest.soul = Some("soul.md".into());
        write_manifest(dir, &manifest).unwrap();
        manifest
    }

    #[test]
    fn a_fresh_manifest_is_valid() {
        let manifest = PackManifest::new("demo", "1.0.0");
        assert!(manifest.problems().is_empty(), "{:?}", manifest.problems());
        assert_eq!(manifest.format_version, FORMAT_VERSION);
        assert_eq!(manifest.game, GAME);
    }

    #[test]
    fn a_manifest_round_trips_through_json() {
        let mut manifest = PackManifest::new("demo", "1.0.0");
        manifest.skills.push(SkillRef {
            id: "pdf".into(),
            source: "github:o/r@skills/pdf@v1".into(),
            sha256: Some("a".repeat(64)),
            license: "MIT".into(),
            deps: BTreeMap::new(),
        });
        let text = serde_json::to_string(&manifest).unwrap();
        let back: PackManifest = serde_json::from_str(&text).unwrap();
        assert_eq!(back.name, "demo");
        assert_eq!(back.skills.len(), 1);
        assert_eq!(back.skills[0].id, "pdf");
    }

    #[test]
    fn the_wrong_game_is_refused() {
        let mut manifest = PackManifest::new("demo", "1.0.0");
        manifest.game = "minecraft".into();
        let problems = manifest.problems();
        assert!(problems.iter().any(|p| p.contains("game must be")), "{problems:?}");
    }

    #[test]
    fn the_legacy_game_value_still_loads() {
        // Packs written before this format settled on a name for the program
        // say "dsh". Refusing them would break every existing offline pack.
        let mut manifest = PackManifest::new("demo", "1.0.0");
        manifest.game = LEGACY_GAME.into();
        assert!(manifest.problems().is_empty(), "{:?}", manifest.problems());
    }

    #[test]
    fn a_manifest_written_in_the_old_spelling_still_parses() {
        // The old shape: snake_case keys, `dependencies.kernel`, no `game`.
        let text = "{
            \"format_version\": 1,
            \"game\": \"dsh\",
            \"name\": \"demo-pack\",
            \"version_id\": \"1.0.0\",
            \"license\": \"MIT\",
            \"kernel_version\": \"0.1.0\",
            \"dependencies\": { \"kernel\": \">=0.1.0\" },
            \"summary\": \"an old pack\"
        }";
        let manifest: PackManifest = serde_json::from_str(text).unwrap();
        assert_eq!(manifest.name, "demo-pack");
        assert_eq!(manifest.version_id, "1.0.0");
        assert_eq!(manifest.dependencies.nguruvilu, ">=0.1.0");
        assert!(manifest.problems().is_empty(), "{:?}", manifest.problems());
    }

    #[test]
    fn an_absent_dependency_range_accepts_anything() {
        // The field exists to help, not to gate.
        let text = "{\"formatVersion\":1,\"name\":\"x\",\"versionId\":\"1.0.0\",\"license\":\"MIT\",\"kernelVersion\":\"0.1.0\",\"dependencies\":{}}";
        let manifest: PackManifest = serde_json::from_str(text).unwrap();
        assert!(manifest.accepts_kernel("9.9.9"));
    }

    #[test]
    fn the_wrong_format_version_is_refused() {
        let mut manifest = PackManifest::new("demo", "1.0.0");
        manifest.format_version = 99;
        let problems = manifest.problems();
        assert!(problems.iter().any(|p| p.contains("formatVersion")), "{problems:?}");
    }

    #[test]
    fn a_component_without_a_licence_is_refused() {
        // Every fetched component states its licence: a pack that pulls in GPL
        // code owes the person installing it that fact.
        let mut manifest = PackManifest::new("demo", "1.0.0");
        manifest.skills.push(SkillRef {
            id: "pdf".into(),
            source: "github:o/r@p@v".into(),
            sha256: None,
            license: String::new(),
            deps: BTreeMap::new(),
        });
        let problems = manifest.problems();
        assert!(problems.iter().any(|p| p.contains("no license")), "{problems:?}");
    }

    #[test]
    fn an_unparseable_source_is_reported_at_validation_time() {
        let mut manifest = PackManifest::new("demo", "1.0.0");
        manifest.plugins.push(PluginRef {
            id: "x".into(),
            source: "no-colon-here".into(),
            license: "MIT".into(),
            sha256: None,
            platforms: BTreeMap::new(),
        });
        let problems = manifest.problems();
        assert!(problems.iter().any(|p| p.contains("plugin 'x'")), "{problems:?}");
    }

    #[test]
    fn an_unknown_platform_tag_is_reported() {
        let mut manifest = PackManifest::new("demo", "1.0.0");
        let mut platforms = BTreeMap::new();
        platforms.insert(
            "windows".into(),
            PlatformArtifact {
                file: "x.dll".into(),
                sha256: "a".repeat(64),
            },
        );
        manifest.plugins.push(PluginRef {
            id: "x".into(),
            source: "github:o/r@p@v".into(),
            license: "MIT".into(),
            sha256: None,
            platforms,
        });
        let problems = manifest.problems();
        assert!(
            problems.iter().any(|p| p.contains("names platform 'windows'")),
            "{problems:?}"
        );
    }

    #[test]
    fn a_duplicate_id_is_reported() {
        let mut manifest = PackManifest::new("demo", "1.0.0");
        for _ in 0..2 {
            manifest.skills.push(SkillRef {
                id: "same".into(),
                source: "github:o/r@p@v".into(),
                sha256: None,
                license: "MIT".into(),
                deps: BTreeMap::new(),
            });
        }
        let problems = manifest.problems();
        assert!(problems.iter().any(|p| p.contains("appears twice")), "{problems:?}");
    }

    #[test]
    fn references_parse_into_sources() {
        let mut manifest = PackManifest::new("demo", "1.0.0");
        manifest.skills.push(SkillRef {
            id: "pdf".into(),
            source: "github:o/r@skills/pdf@v1".into(),
            sha256: None,
            license: "MIT".into(),
            deps: BTreeMap::new(),
        });
        let refs = manifest.references().unwrap();
        assert_eq!(refs.len(), 1);
        assert_eq!(refs[0].kind, "skill");
        assert_eq!(refs[0].id, "pdf");
        assert!(refs[0].source.is_remote());
    }

    #[test]
    fn a_manifest_pack_round_trips_through_the_archive() {
        let dir = scratch("round");
        sample(&dir, "demo", "1.0.0");
        let archive = dir.join("demo-1.0.0.dshpack");

        let built = pack(&dir, &archive).unwrap();
        assert_eq!(built.name, "demo");
        assert!(archive.is_file());

        let report = verify(&archive).unwrap();
        assert_eq!(report.manifest.name, "demo");
        assert!(report.kernel_compatible, "{:?}", report.warnings);
        assert!(report.contents.has_manifest);
        assert!(report.warnings.is_empty(), "{:?}", report.warnings);
    }

    #[test]
    fn a_manifest_pack_refuses_to_carry_payload() {
        // Smuggling payload would install stale bytes instead of fetching.
        let dir = scratch("smuggle");
        sample(&dir, "demo", "1.0.0");
        std::fs::create_dir_all(dir.join(FILES_DIR)).unwrap();
        std::fs::write(dir.join(FILES_DIR).join("x.dll"), b"payload").unwrap();

        let error = pack(&dir, &dir.join("out.dshpack")).expect_err("must refuse");
        let text = format!("{error:#}");
        assert!(text.contains("--offline"), "{text}");
    }

    #[test]
    fn the_archive_is_not_a_member_of_itself() {
        let dir = scratch("self");
        sample(&dir, "demo", "1.0.0");
        let archive = dir.join("demo-1.0.0.dshpack");

        pack(&dir, &archive).unwrap();
        let again = pack(&dir, &archive).unwrap();
        assert_eq!(again.name, "demo");

        let report = verify(&archive).unwrap();
        assert!(
            !report.contents.files.iter().any(|f| f.ends_with(".dshpack")),
            "{:?}",
            report.contents.files
        );
    }

    #[test]
    fn a_content_field_naming_a_missing_file_is_a_warning() {
        let dir = scratch("missing-content");
        let mut manifest = PackManifest::new("demo", "1.0.0");
        manifest.soul = Some("soul.md".into());
        write_manifest(&dir, &manifest).unwrap();
        // soul.md deliberately not written.

        let archive = dir.join("out.dshpack");
        pack(&dir, &archive).unwrap();
        let report = verify(&archive).unwrap();
        assert!(
            report.warnings.iter().any(|w| w.contains("soul.md")),
            "{:?}",
            report.warnings
        );
    }

    #[test]
    fn an_incompatible_kernel_range_is_flagged() {
        let dir = scratch("incompatible");
        let mut manifest = PackManifest::new("demo", "1.0.0");
        manifest.dependencies.nguruvilu = ">=99.0.0".into();
        write_manifest(&dir, &manifest).unwrap();

        let archive = dir.join("out.dshpack");
        pack(&dir, &archive).unwrap();
        let report = verify(&archive).unwrap();
        assert!(!report.kernel_compatible);
        assert!(
            report.warnings.iter().any(|w| w.contains("accepts nguruvilu")),
            "{:?}",
            report.warnings
        );
    }

    #[tokio::test]
    async fn a_pack_outside_its_declared_kernel_range_still_installs() {
        // The kernel is a tool, not a gatekeeper. A pack that names a range
        // this kernel is outside of installs anyway, and whatever the mismatch
        // breaks is reported where it happens rather than here.
        let dir = scratch("range");
        let mut manifest = PackManifest::new("demo", "1.0.0");
        manifest.dependencies.nguruvilu = ">=99.0.0".into();
        write_manifest(&dir, &manifest).unwrap();
        let archive = dir.join("out.dshpack");
        pack(&dir, &archive).unwrap();

        let placed = install(&archive, &dir.join("packs"))
            .await
            .expect("it installs");
        assert_eq!(placed.manifest.name, "demo");
        assert!(
            !placed.manifest.accepts_kernel(kernel_version()),
            "and the mismatch is still answerable for anyone who asks"
        );
    }

    #[tokio::test]
    async fn an_offline_pack_installs_without_the_network() {
        // The payload is embedded, so install must not reach for it. A
        // reference that could not resolve proves nothing was fetched.
        let dir = scratch("offline");
        let mut manifest = PackManifest::new("offline-demo", "1.0.0");
        manifest.skills.push(SkillRef {
            id: "pdf".into(),
            source: "github:nobody/nothing@x@y".into(),
            sha256: None,
            license: "MIT".into(),
            deps: BTreeMap::new(),
        });
        manifest.soul = Some("soul.md".into());
        write_manifest(&dir, &manifest).unwrap();
        std::fs::write(dir.join("soul.md"), "Be terse.\n").unwrap();

        // What `--offline` would have embedded.
        let embedded = dir.join(FILES_DIR).join("pdf");
        std::fs::create_dir_all(&embedded).unwrap();
        std::fs::write(embedded.join("SKILL.md"), "pdf skill\n").unwrap();

        let archive = dir.join("offline.dshpack");
        {
            let contents = packable_files(&dir, &archive).unwrap();
            write_archive(&dir, &contents.files, &archive).unwrap();
        }

        let packs = dir.join("packs");
        let placed = install(&archive, &packs).await.unwrap();
        assert_eq!(placed.manifest.name, "offline-demo");
        assert!(placed.path.join("soul.md").is_file());
        assert!(placed.path.join(FILES_DIR).join("pdf").join("SKILL.md").is_file());
    }

    #[tokio::test]
    async fn install_writes_an_assembly_the_loader_can_read() {
        let dir = scratch("assembly");
        let mut manifest = PackManifest::new("demo", "1.0.0");
        manifest.skills.push(SkillRef {
            id: "pdf".into(),
            source: "github:nobody/nothing@x@y".into(),
            sha256: None,
            license: "MIT".into(),
            deps: BTreeMap::new(),
        });
        write_manifest(&dir, &manifest).unwrap();
        let embedded = dir.join(FILES_DIR).join("pdf");
        std::fs::create_dir_all(&embedded).unwrap();
        std::fs::write(embedded.join("SKILL.md"), "x").unwrap();

        let archive = dir.join("demo.dshpack");
        {
            let contents = packable_files(&dir, &archive).unwrap();
            write_archive(&dir, &contents.files, &archive).unwrap();
        }

        let placed = install(&archive, &dir.join("packs")).await.unwrap();
        let assembly = placed.assembly.expect("an assembly");
        assert!(assembly.is_file());

        // It must parse as an assembly, which is the contract with the loader.
        let parsed = read_assembly(&placed.path).unwrap().expect("readable");
        assert_eq!(parsed.stages.len(), 1);
        assert_eq!(parsed.stages[0].skills.len(), 1);
        assert_eq!(parsed.stages[0].skills[0].base.id, "pdf");
    }

    #[tokio::test]
    async fn a_pack_with_no_references_still_installs_and_gets_an_assembly() {
        // A pure appearance or persona pack: nothing to fetch.
        let dir = scratch("no-refs");
        let mut manifest = PackManifest::new("look", "1.0.0");
        manifest.look = Some("look.json".into());
        write_manifest(&dir, &manifest).unwrap();
        std::fs::write(
            dir.join("look.json"),
            "{\"themes\":[{\"name\":\"light\",\"tokens\":{\"--bg\":\"#ffffff\"}}],\"panels\":[]}",
        )
        .unwrap();

        let archive = dir.join("look.dshpack");
        pack(&dir, &archive).unwrap();

        let placed = install(&archive, &dir.join("packs")).await.unwrap();
        assert!(placed.path.join("look.json").is_file());
        // No stages, because there is nothing to load.
        let parsed = read_assembly(&placed.path).unwrap().expect("readable");
        assert!(parsed.stages.is_empty());
    }

    #[test]
    fn the_look_file_parses_into_themes_and_panels() {
        let mut look = LookFile::default();
        look.themes.push(
            crate::theme::Theme::new("light").set("--bg", "#ffffff"),
        );
        look.panels.push(crate::ui::UiPanel::new(
            "p",
            crate::ui::UiSlot::StatusBar,
            "<b>x</b>",
        ));

        // Round-trip: this is the file a pack author writes.
        let text = serde_json::to_string(&look).unwrap();
        let back: LookFile = serde_json::from_str(&text).unwrap();
        assert_eq!(back.themes.len(), 1);
        assert_eq!(back.themes[0].tokens["--bg"], "#ffffff");
        assert_eq!(back.panels.len(), 1);
        assert_eq!(back.panels[0].slot, crate::ui::UiSlot::StatusBar);
    }

    #[test]
    fn an_unknown_field_in_a_content_file_is_refused() {
        // A typo in a content file would otherwise be a setting that silently
        // does nothing.
        let text = "{\"window\": \"128K\", \"compactPercentage\": 75}";
        let error = serde_json::from_str::<ContextFile>(text).expect_err("must refuse");
        assert!(format!("{error}").contains("compactPercentage"), "{error}");
    }

    #[test]
    fn the_models_file_holds_an_env_var_name_never_a_key() {
        let text = "{\"baseUrl\": \"https://x/v1\", \"apiKeyEnv\": \"MY_KEY\", \"model\": \"m\"}";
        let models: ModelsFile = serde_json::from_str(text).unwrap();
        assert_eq!(models.api_key_env.as_deref(), Some("MY_KEY"));
        // The field is named for what it holds; there is no field for a value.
        assert!(!serde_json::to_string(&models).unwrap().contains("apiKey\""));
    }

    #[test]
    fn an_installed_pack_is_listed_and_a_broken_one_is_reported() {
        let dir = scratch("list");
        let packs = dir.join("packs");
        let good = packs.join("demo-1.0.0");
        std::fs::create_dir_all(&good).unwrap();
        write_manifest(&good, &PackManifest::new("demo", "1.0.0")).unwrap();

        // A directory with no manifest is not a pack and is skipped.
        std::fs::create_dir_all(packs.join("not-a-pack")).unwrap();

        let listed = installed(&packs).unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].manifest.name, "demo");
    }

    #[test]
    fn listing_a_missing_directory_is_empty_not_an_error() {
        let dir = scratch("nothing");
        assert!(installed(&dir.join("never-created")).unwrap().is_empty());
    }

    #[test]
    fn the_platform_artifact_map_picks_this_platform() {
        let mut platforms = BTreeMap::new();
        platforms.insert(
            crate::fetch::platform_tag().to_string(),
            PlatformArtifact {
                file: "here.dll".into(),
                sha256: "a".repeat(64),
            },
        );
        let plugin = PluginRef {
            id: "p".into(),
            source: "github:o/r@p@v".into(),
            license: "MIT".into(),
            sha256: None,
            platforms,
        };
        let tag = crate::fetch::platform_tag();
        assert_eq!(plugin.platforms[tag].file, "here.dll");
    }

    #[test]
    fn the_content_file_table_matches_the_manifest_fields() {
        // Guards against a field being added without its file name.
        let manifest = PackManifest::new("demo", "1.0.0");
        for (field, file) in CONTENT_FILES {
            assert!(manifest.content_file(field).is_none(), "{field}");
            assert!(!file.is_empty());
        }
        assert_eq!(manifest.content_file("nonsense"), None);
    }

    #[test]
    fn an_archive_never_contains_an_archive() {
        // A leftover archive from an earlier build is still an archive, and
        // embedding one ships a stale pack inside a fresh one.
        let dir = scratch("no-nesting");
        sample(&dir, "demo", "1.0.0");
        std::fs::write(dir.join("stale-0.9.0.dshpack"), b"an old archive").unwrap();

        let archive = dir.join("demo-1.0.0.dshpack");
        pack(&dir, &archive).unwrap();

        let report = verify(&archive).unwrap();
        assert!(
            !report.contents.files.iter().any(|f| f.ends_with(".dshpack")),
            "{:?}",
            report.contents.files
        );
        assert!(report.contents.files.iter().any(|f| f == "soul.md"));

    #[test]
    fn an_archive_never_contains_an_archive() {
        // A leftover archive from an earlier build is still an archive, and
        // embedding one ships a stale pack inside a fresh one.
        let dir = scratch("no-nesting");
        sample(&dir, "demo", "1.0.0");
        std::fs::write(dir.join("stale-0.9.0.dshpack"), b"an old archive").unwrap();

        let archive = dir.join("demo-1.0.0.dshpack");
        pack(&dir, &archive).unwrap();

        let report = verify(&archive).unwrap();
        assert!(
            !report.contents.files.iter().any(|f| f.ends_with(".dshpack")),
            "{:?}",
            report.contents.files
        );
        assert!(report.contents.files.iter().any(|f| f == "soul.md"));
    }
}

    #[test]
    fn an_unpinned_reference_is_a_warning_not_a_problem() {
        // A pack under development legitimately points at a branch; a pack
        // being published does not, and the difference is worth saying out loud.
        let mut manifest = PackManifest::new("demo", "1.0.0");
        manifest.skills.push(SkillRef {
            id: "floating".into(),
            source: "github:owner/repo@skills/x@main".into(),
            sha256: None,
            license: "MIT".into(),
            deps: BTreeMap::new(),
        });

        assert!(manifest.problems().is_empty(), "it still builds");
        let warnings = manifest.warnings();
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("not pinned"), "{}", warnings[0]);
        assert!(warnings[0].contains("--pin"), "it should say the fix: {}", warnings[0]);
    }

    #[test]
    fn a_reference_with_no_ref_at_all_says_it_follows_the_default_branch() {
        let mut manifest = PackManifest::new("demo", "1.0.0");
        manifest.plugins.push(PluginRef {
            id: "x".into(),
            source: "github:owner/repo@bin".into(),
            license: "MIT".into(),
            sha256: None,
            platforms: BTreeMap::new(),
        });
        let warnings = manifest.warnings();
        assert!(
            warnings[0].contains("default branch"),
            "{}",
            warnings[0]
        );
    }

    #[test]
    fn a_pinned_reference_warns_about_nothing() {
        let mut manifest = PackManifest::new("demo", "1.0.0");
        manifest.skills.push(SkillRef {
            id: "fixed".into(),
            source: "github:owner/repo@skills/x@main".into(),
            sha256: Some("a".repeat(64)),
            license: "MIT".into(),
            deps: BTreeMap::new(),
        });
        assert!(manifest.warnings().is_empty());
    }

    #[test]
    fn a_local_reference_is_not_reported_as_floating() {
        // Nothing to pin: the content is already inside the pack.
        let mut manifest = PackManifest::new("demo", "1.0.0");
        manifest.plugins.push(PluginRef {
            id: "local".into(),
            source: "dylib:files/x.dll".into(),
            license: "MIT".into(),
            sha256: None,
            platforms: BTreeMap::new(),
        });
        assert!(manifest.warnings().is_empty());
    }

    #[test]
    fn verify_describes_an_archive_rather_than_judging_it() {
        // verify exists so a person can see inside a pack. Whether the pack is
        // acceptable is the person's call: an unpinned reference installs, and
        // if the floating content turns out to be wrong that shows up when it
        // is used.
        let dir = scratch("unpinned");
        let mut manifest = PackManifest::new("demo", "1.0.0");
        manifest.skills.push(SkillRef {
            id: "floating".into(),
            source: "github:owner/repo@skills/x@main".into(),
            sha256: None,
            license: "MIT".into(),
            deps: BTreeMap::new(),
        });
        write_manifest(&dir, &manifest).unwrap();

        let archive = dir.join("out.dshpack");
        pack(&dir, &archive).unwrap();
        let report = verify(&archive).unwrap();
        assert_eq!(report.manifest.name, "demo");
        assert_eq!(report.manifest.skills.len(), 1, "the reference is described");
    }

    #[test]
    fn an_mcp_server_naming_a_source_is_validated_like_any_other() {
        // An MCP server that can only be an npm package is a server the user
        // has to install a runtime for. A source makes the channel a
        // declaration, so it has to parse.
        let text = "{\"servers\":[{\"id\":\"fs\",\"source\":\"no-colon-here\"}]}";
        let mcp: McpFile = serde_json::from_str(text).unwrap();
        assert_eq!(mcp.servers[0].source.as_deref(), Some("no-colon-here"));

        let mut manifest = PackManifest::new("demo", "1.0.0");
        manifest.mcp = Some(ContentRef::carried("mcp.json"));
        // The source lives in the file rather than the manifest, so the
        // manifest alone cannot reject it; it is checked where the file is
        // read.
        assert!(manifest.problems().is_empty());
    }

    #[test]
    fn a_fetched_mcp_server_records_its_source_and_hash() {
        let text = "{\"servers\":[{\"id\":\"fs\",\"command\":\"mcp-fs\",\
                     \"source\":\"github:owner/mcp@bin@v1\",\"sha256\":\"aa\"}]}";
        let mcp: McpFile = serde_json::from_str(text).unwrap();
        assert_eq!(mcp.servers[0].command.as_deref(), Some("mcp-fs"));
        assert!(mcp.servers[0].source.is_some());
        assert_eq!(mcp.servers[0].sha256.as_deref(), Some("aa"));
    }

    #[test]
    fn an_mcp_server_without_a_source_round_trips_unchanged() {
        // The npm-run form must keep working: most servers are still there.
        let text = "{\"servers\":[{\"id\":\"fs\",\"command\":\"npx\",\"args\":[\"-y\",\"x\"]}]}";
        let mcp: McpFile = serde_json::from_str(text).unwrap();
        assert!(mcp.servers[0].source.is_none());
        assert_eq!(mcp.servers[0].args.len(), 2);
    }
}
