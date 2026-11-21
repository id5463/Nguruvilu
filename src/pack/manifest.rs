//! The manifest, and the data types it names.
//!
//! A pack's identity and its content wiring: [`PackManifest`] and the shapes it
//! is made of, the files it carries, and the version range that decides whether
//! its kernel claim is met.
//!
//! Everything here is data. Nothing fetches, installs, or unpacks; the one
//! function that touches disk is [`read_injections`], which reads the file it is
//! handed and nothing else. Archiving, install, and the pack directories live in
//! [`super`].
//!
//! The format's constants and the content file table stay in [`super`], beside
//! the code that writes them, and are read from here: this file states the
//! shapes, the parent states what they mean.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};

use super::{CONTENT_FILES, FORMAT_VERSION, GAME, LEGACY_GAME};
use crate::fetch::Source;

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
    /// One line saying what the pack is for, shown where packs are listed.
    ///
    /// Optional: an older pack without one still installs — it just lists as
    /// name and version, the way every pack did before this field existed.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub description: String,
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
    /// Search provider file.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub search: Option<ContentRef>,
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
#[serde(rename_all = "camelCase")]
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
    /// This pack owns the document, rather than overlaying its files.
    ///
    /// The page is one document, so exactly one pack can supply `index.html` —
    /// that is the closest analogue to Patchouli's rule that a resource pack
    /// may override a book's contents but "not include book.json". Everything
    /// else a pack brings (stylesheets, fragments, images) is an overlay: a
    /// lookup walks the stack and the first directory with the file wins, so
    /// several packs extend one interface at once instead of one excluding the
    /// rest.
    ///
    /// Defaults to false: a pack that says nothing is an overlay, because that
    /// is the safe answer — an overlay cannot break the page it does not own.
    /// The pack that ships the interface declares `base: true`.
    #[serde(default)]
    pub base: bool,
    /// Where this pack sits among the overlays. Higher wins.
    ///
    /// Overlays are ordered, not merged by name: a pack that wants its
    /// stylesheet to land after another's has to be able to say so, the same
    /// way Patchouli's `sortnum` orders entries inside a category. Ties fall
    /// back to the pack name so two runs never resolve differently.
    #[serde(default)]
    pub overlay_order: i32,
    /// Whether this interface accepts overlays from other packs at all.
    ///
    /// The analogue of `use_resource_pack: true` in a Patchouli `book.json`.
    /// Defaults to true for an overlay (it is only supplying files, so it has
    /// nothing to protect) and is what the base pack can switch off when it
    /// wants its page served exactly as shipped.
    #[serde(default = "default_true")]
    pub overlays: bool,
    /// Where the interface comes from, when it is fetched.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    /// sha256 of the fetched directory.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
}

/// Serde default for fields that are on unless a manifest turns them off.
fn default_true() -> bool {
    true
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
            base: false,
            overlay_order: 0,
            overlays: true,
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
            description: String::new(),
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
            search: None,
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
            "search" => self.search.as_ref(),
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
            "search" => self.search = Some(reference),
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
    ///
    /// `builtin:<name>` is a plugin compiled into this kernel, `plugin:<path>` a
    /// process the kernel starts, `dylib:<path>` a shared library. The three are
    /// rendered into the assembly verbatim, so a pack that ships a process
    /// plugin says so here and nothing has to translate it.
    pub source: String,
    /// How to build the executable from source shipped in the pack.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub build: Option<crate::assembly::BuildSpec>,
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
#[serde(rename_all = "camelCase")]
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
    /// How the answer is delivered: `stream` or `buffered`.
    ///
    /// A pack property rather than a user one, because the thing it describes is
    /// the gateway: a pack that ships against an endpoint whose stream never
    /// completes has to say so, or every install of it hangs on the first turn.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub delivery: Option<String>,
    /// Whether a stalled stream switches this route to buffered delivery.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub degrade_on_stall: Option<bool>,
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

/// `search.json`: which search provider this pack expects.
///

/// `context.json`: how much to remember.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
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
#[serde(rename_all = "camelCase")]
pub struct McpFile {
    /// The servers.
    #[serde(default)]
    pub servers: Vec<McpServerDecl>,
}

/// One tool an MCP server exposes, as its pack declares it.
///
/// The same fields the protocol's own `tools/list` returns, spelled the way
/// `mcp.json` already spells things (`inputSchema`, matching the wire format
/// rather than this project's Rust style) so a pack author can copy a
/// `tools/list` response into a manifest without translating it.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct McpToolDecl {
    /// Name as the server knows it, without the `mcp__server__` prefix.
    pub name: String,
    /// Description shown to the model.
    #[serde(default)]
    pub description: String,
    /// JSON Schema for the arguments.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_schema: Option<serde_json::Value>,
}

/// One MCP server declaration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct McpServerDecl {
    /// Identifier.
    pub id: String,
    /// `stdio` (the default) or `streamable-http`.
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
    /// Executable. Resolved inside the fetched directory when there is a
    /// source; install rewrites it to the absolute path it unpacked into.
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
    /// Hand this server to the named MCP proxy instead of registering its tools.
    ///
    /// Same field, same meaning as the one on an assembly entry: a pack keeps
    /// declaring its servers in `mcp.json`, and this says who starts them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub via_proxy: Option<String>,
    /// One source per platform, when a server ships a different artifact to
    /// each. A non-empty map replaces the single `source` at install time:
    /// the entry for `fetch::platform_tag()` is fetched — an archive URL is
    /// unpacked into `files/mcp/<id>/` — and the choice is written back into
    /// the mcp file so the assembly resolves the command from inside the
    /// pack. A tag the map does not carry is an error naming the tags it
    /// does, never another platform's binary.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub platforms: BTreeMap<String, PlatformSource>,
    /// The endpoint, for `transport: "streamable-http"`: the one URL every
    /// message is `POST`ed to. Required there, ignored for stdio.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    /// Headers for `streamable-http` — `Authorization` and friends, sent on
    /// every request. Ignored for stdio.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub headers: BTreeMap<String, String>,
    /// The tools this server provides, declared rather than discovered.
    ///
    /// Listing a server's tools is one round trip — but starting the server to
    /// ask is not: measured on this machine, thirteen servers cost 6.2 s on a
    /// cold start (2.7 s of it `chrome-devtools-mcp` alone) against 47 ms for a
    /// process that loads nothing. Almost none of them is used in any given
    /// turn, and a subagent answering one question pays the whole bill.
    ///
    /// So a pack says what its server exposes, the kernel registers those tools
    /// at load, and the server starts the first time one of them is called. A
    /// pack that omits this is still correct — the kernel starts the server to
    /// learn the list, exactly as before — it is just slower.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<McpToolDecl>,
}

/// One platform's source for an MCP server.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PlatformSource {
    /// Where this platform's artifact comes from. An archive URL is unpacked
    /// into the server's directory; every other source fetches and copies,
    /// exactly as the single-source form does.
    pub source: String,
    /// sha256 the fetch is checked against: the file's own hash for a URL —
    /// the value a release's `checksums.txt` publishes — and the directory
    /// hash for a directory source.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
    /// Command name override, for an artifact whose executable is not named
    /// like the others: Windows ships `cua-driver.exe`, macOS nests its
    /// binary under the release directory.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub command: Option<String>,
}

impl LookFile {
    /// The theme in force: the named one, else the first.
    ///
    /// An `activeTheme` naming a theme that is not in the file is reported by
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
#[serde(rename_all = "camelCase")]
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
    /// Interface labels, by element id.
    ///
    /// Text the pack supplies for the shell's own elements, so a pack can put
    /// the interface into a language without shipping a whole interface. Keys
    /// the interface does not have are passed through untouched rather than
    /// rejected: a label written for another interface is not an error, and
    /// the no-gates rule applies to text as much as to anything else here.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub strings: BTreeMap<String, String>,

    /// Stylesheet rules appended after the interface's own.
    ///
    /// The gap this fills: `themes` can only change custom properties the
    /// interface already declared, and `panels` can only add fragments at a few
    /// fixed slots. Anything else — spacing, a column width, the look of a
    /// control — used to require shipping a complete copy of the 150 KB
    /// interface, because the only other way in was to replace `index.html`
    /// wholesale. That made every small change expensive twice over: the pack
    /// had to carry the whole page, and `tests/ui_sync.rs` then required it to
    /// stay byte-identical to the built-in.
    ///
    /// So this holds CSS text, appended after the built-in sheet, which is what
    /// lets a pack restyle a corner of the interface in a dozen lines instead of
    /// copying a megabyte of markup it does not care about.
    ///
    /// Rules here apply to the page the pack is loaded into, which is why the
    /// same trust rule as `panels` applies: loading a pack already means
    /// trusting it with the interface.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub css: Option<String>,

    /// Named points this pack hangs code on, and the script that does it.
    ///
    /// The shape Patchouli add-ons use: the host names a point
    /// (`session.row.end`), the add-on supplies the code that runs there. Here
    /// the point names are the page's own — see `runHook` in the interface —
    /// and the scripts are evaluated in the page after it has loaded, in the
    /// order the packs resolved.
    ///
    /// This is what lets a pack extend a rendering step instead of replacing
    /// the document, which was the only option before: one badge in the session
    /// list used to cost a copy of the whole 3 000-line page.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub hooks: BTreeMap<String, String>,

    /// Panel kinds this pack defines.
    ///
    /// The registry Patchouli keeps as `pageTypes`: a pack that wants a *new
    /// kind of panel* — a chart, a diff view, a colour picker — registers a
    /// name here, and every panel that declares `"type": "<name>"` is handed to
    /// this renderer instead of the built-in HTML one.
    ///
    /// A type is data, not code from the kernel's point of view: the pack
    /// supplies a renderer script and an optional fixture (default `html`), and
    /// the page runs it inside the same sandbox a normal panel gets. That keeps
    /// the boundary where it already was — a panel has never been able to reach
    /// the shell, and defining a type does not change that.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub types: BTreeMap<String, PanelType>,
}

/// One pack-defined panel kind.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PanelType {
    /// Which built-in shape this type is drawn with. Only `html` exists today:
    /// a type decides how the fragment is produced, not how it is hosted.
    #[serde(default = "default_renderer")]
    pub renderer: String,
    /// A script that runs once per panel of this type.
    ///
    /// It receives `(host, data)` — `host` is the panel's own element inside
    /// the sandboxed frame, `data` is whatever the panel's `html` field holds,
    /// parsed as JSON when it looks like JSON. The script fills `host`; the
    /// page does not interpret what it draws.
    pub script: String,
    /// Shown in the pack panel, for a human choosing what to install.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub description: String,
}

fn default_renderer() -> String {
    "html".to_string()
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

