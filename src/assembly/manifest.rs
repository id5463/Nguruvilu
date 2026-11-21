//! What a pack declares it loads: the manifest types.
//!
//! Parsing lives here; turning a parsed manifest into an ordered list of steps
//! lives in [`super::plan`].

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Current manifest format.
pub const ASSEMBLY_VERSION: u32 = 1;

/// Which plane a change lands on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Scope {
    /// Only the session that loaded the pack (the default).
    #[default]
    Session,
    /// Every session, including ones created later. Requires user consent.
    Global,
}

/// What to do when a contribution claims a name that is taken.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ConflictKind {
    /// Refuse the load and report both owners.
    #[default]
    Error,
    /// Replace the existing contribution.
    Override,
    /// Prefix the contribution with the pack id.
    Namespaced,
}

/// What to do when an entry fails to load.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum OnFailure {
    /// Fail the whole assembly.
    #[default]
    Abort,
    /// Skip this entry and continue.
    Skip,
    /// Retry once before giving up.
    Retry,
}

/// When to load an entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum LoadMode {
    /// At assembly time.
    #[default]
    Eager,
    /// On first use.
    Lazy,
}

/// Transport for an MCP entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum Transport {
    /// A child process speaking JSON-RPC on stdio.
    #[default]
    Stdio,
    /// A remote Streamable HTTP endpoint (declared, not connected here).
    StreamableHttp,
}

/// Defaults applied to every entry in a manifest.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Defaults {
    /// Default scope.
    #[serde(default)]
    pub scope: Scope,
    /// Default conflict policy.
    #[serde(default)]
    pub conflict: ConflictKind,
    /// Default failure policy.
    #[serde(default)]
    pub on_failure: OnFailure,
    /// Default load mode.
    #[serde(default)]
    pub load: LoadMode,
}

/// One loadable entry, shared shape across the three kinds.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EntryBase {
    /// Identifier, unique within the manifest.
    pub id: String,
    /// Where it comes from.
    #[serde(default)]
    pub source: String,
    /// Numeric sort key inside the stage.
    #[serde(default)]
    pub order: i64,
    /// Entries that must load first.
    #[serde(default)]
    pub after: Vec<String>,
    /// Entries that must load later.
    #[serde(default)]
    pub before: Vec<String>,
    /// Per-entry overrides of the defaults.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope: Option<Scope>,
    /// Conflict policy override.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub conflict: Option<ConflictKind>,
    /// Failure policy override.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub on_failure: Option<OnFailure>,
    /// Load mode override.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub load: Option<LoadMode>,
    /// Restrict to these platforms (`win32`, `darwin`, `linux`).
    #[serde(default)]
    pub platform: Vec<String>,
    /// Version or capability requirement, recorded for diagnostics.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requires: Option<String>,
}

/// A plugin entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginEntry {
    /// Shared entry fields.
    #[serde(flatten)]
    pub base: EntryBase,
    /// Plugin configuration passed to `apply`.
    ///
    /// A plugin that runs as a process receives this verbatim in its
    /// `describe` reply's surroundings; one compiled into the kernel gets it
    /// through `PluginCtx`.
    #[serde(default)]
    pub config: Value,
    /// Build the executable from source inside the pack, when it is not there.
    ///
    /// A plugin is an executable, and a pack that ships one has two choices: put
    /// the binary in the archive (platform-specific, large, and unverifiable on
    /// a machine that did not build it) or ship the source and let the machine
    /// build it. This says the second, and names what to run.
    ///
    /// Conceptually `cargo`, `go`, or `make`: the field names a **directory to
    /// build in** and a **binary to end up with**, not a toolchain. Keeping
    /// `cargo` out of the field name is what stops this from being an enum of
    /// build systems in a year.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub build: Option<BuildSpec>,
}

/// How to produce a plugin's executable from source.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BuildSpec {
    /// Directory to build in, relative to the pack.
    ///
    /// `dir` rather than `crate` because `crate` is a Rust keyword and this field
    /// must not assume Rust is the build tool.
    #[serde(default = "default_build_dir")]
    pub dir: String,
    /// The binary to expect afterwards; also the name passed to the build tool.
    pub binary: String,
    /// Arguments for the build command. Defaults to what a Cargo release build
    /// wants, which is the only one implemented.
    #[serde(default)]
    pub args: Vec<String>,
    /// The build command. Defaults to `cargo`; a pack that builds with
    /// something else says so and does not need a kernel change.
    #[serde(default)]
    pub command: Option<String>,
}

fn default_build_dir() -> String {
    "src".to_string()
}

/// An MCP server entry.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct McpEntry {
    /// Shared entry fields.
    #[serde(flatten)]
    pub base: EntryBase,
    /// How to reach the server.
    #[serde(default)]
    pub transport: Transport,
    /// Executable, for `stdio`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub command: Option<String>,
    /// Arguments, for `stdio`.
    #[serde(default)]
    pub args: Vec<String>,
    /// Environment additions, for `stdio`.
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    /// Endpoint, for `streamable-http`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    /// Headers, for `streamable-http`.
    #[serde(default)]
    pub headers: BTreeMap<String, String>,
    /// Per-call timeout.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_ms: Option<u64>,

    /// Hand this server to a proxy instead of registering its tools directly.
    ///
    /// A pack declares its server the same way it always did; this says *who
    /// starts it*. Set it and the kernel does not spawn the server at all — it
    /// appends the command line to the proxy's target list, and the proxy
    /// exposes one entry that reaches all of them.
    ///
    /// Measured reason: the packs installed here declare 102 tools between them
    /// (57 + 29 + 8 + 6 + 2). Every one of those is a line the model reads before
    /// it answers, and a conversation uses three or four. Behind a proxy the same
    /// capability costs one tool.
    ///
    /// The alternative — editing every pack's `mcp.json` — was tried first and
    /// rejected: it makes a pack's own manifest a lie about what it provides, and
    /// uninstalling the proxy would leave every pack with no tools at all rather
    /// than the tools it declared.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub via_proxy: Option<String>,
    /// The tools this server provides, declared rather than discovered.
    ///
    /// When present, the kernel registers these tools at load and starts the
    /// server on the first call to one of them. When absent it starts the
    /// server to ask, which is correct and slow — see
    /// [`crate::pack::McpToolDecl`] for the measurement.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<crate::pack::McpToolDecl>,
}

/// A skill entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillEntry {
    /// Shared entry fields.
    #[serde(flatten)]
    pub base: EntryBase,
    /// Expected content hash, when the pack pins one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha1: Option<String>,
}

/// One stage: a coarse ordering group.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stage {
    /// Stage name, for diagnostics.
    pub name: String,
    /// Plugins in this stage.
    #[serde(default)]
    pub plugins: Vec<PluginEntry>,
    /// MCP servers in this stage.
    #[serde(default)]
    pub mcp: Vec<McpEntry>,
    /// Skills in this stage.
    #[serde(default)]
    pub skills: Vec<SkillEntry>,
}

/// Teardown policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Teardown {
    /// Unload stages in reverse order.
    #[serde(default = "default_true")]
    pub reverse_stages: bool,
}

fn default_true() -> bool {
    true
}

impl Default for Teardown {
    fn default() -> Self {
        Self { reverse_stages: true }
    }
}

/// A parsed `assembly.yaml`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Assembly {
    /// Format version.
    #[serde(default = "default_version")]
    pub version: u32,
    /// Pack name, for diagnostics.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Defaults for every entry.
    #[serde(default)]
    pub defaults: Defaults,
    /// Ordered stages.
    #[serde(default)]
    pub stages: Vec<Stage>,
    /// Teardown policy.
    #[serde(default)]
    pub teardown: Teardown,
}

fn default_version() -> u32 {
    ASSEMBLY_VERSION
}

impl Assembly {
    /// Parse a manifest from YAML text.
    pub fn parse(text: &str) -> Result<Self> {
        let assembly: Assembly =
            serde_yaml::from_str(text).context("parsing assembly manifest")?;
        if assembly.version != ASSEMBLY_VERSION {
            return Err(anyhow!(
                "assembly version {} is not supported (expected {ASSEMBLY_VERSION})",
                assembly.version
            ));
        }
        Ok(assembly)
    }

    /// Read a manifest from disk.
    pub fn from_file(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading {}", path.display()))?;
        Self::parse(&text)
    }

    /// An empty manifest.
    pub fn empty() -> Self {
        Self {
            version: ASSEMBLY_VERSION,
            name: None,
            defaults: Defaults::default(),
            stages: Vec::new(),
            teardown: Teardown::default(),
        }
    }
}
