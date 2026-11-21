//! The command line: the flags `ngu` takes and the subcommands it runs.
//!
//! One module per shape rather than one per subcommand. `Cli` is the whole
//! surface a run is started with, and the five enums below are the parts of it
//! that name a subcommand — each arm here is a verb the binary answers to.
//!
//! Kept out of `main.rs` because parsing and doing are different jobs: nothing
//! in this file reaches the kernel, so a change to a flag cannot change what a
//! turn does.

use std::path::PathBuf;

use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(
    name = "ngu",
    version,
    about = "Nguruvilu — an independent agent kernel (four tools, a loop, everything else is loaded)"
)]
pub struct Cli {
    /// One-shot prompt. When omitted, stdin is read if it is not a terminal.
    #[arg(short, long)]
    pub prompt: Option<String>,

    /// Which packs to load: `all`, `none`, or a comma-separated list.
    ///
    /// A subagent answering one question rarely needs the thirteen MCP servers
    /// an interactive session has, and each one costs a start. Measured on this
    /// machine: `none` is 121 ms against 339 ms for `all`.
    ///
    /// Deliberately a plain list rather than named modes. "The same packs the
    /// parent loaded" is a decision the *parent* makes — it knows what it loaded
    /// and can write the list itself — and putting that word here would mean the
    /// kernel had to know what a parent is.
    #[arg(long, env = "NGU_PACKS", default_value = "all")]
    pub packs: String,

    /// Start minimal: no persona, no injection rules, no pack content, no
    /// skills.
    ///
    /// For a subagent. A worker that is started to answer one question should
    /// not inherit the persona, the retrieval rules, or the theme of the
    /// conversation that started it — those describe a *conversation*, and this
    /// process is not one. It keeps the base prompt (the four tools and how to
    /// use them), the tool table, and the model route: without those it could
    /// not work at all.
    ///
    /// Measured against a full start on this machine: what this skips is the
    /// pack content and skills, which is where most of a session's non-MCP
    /// start cost lives.
    #[arg(long, env = "NGU_BARE")]
    pub bare: bool,

    /// Model id.
    #[arg(short, long, env = "NGU_MODEL")]
    pub model: Option<String>,

    /// API base URL, including the version segment.
    #[arg(long, env = "NGU_BASE_URL")]
    pub base_url: Option<String>,

    /// API key.
    #[arg(long, env = "NGU_API_KEY")]
    pub api_key: Option<String>,

    /// Continue an existing session by id.
    #[arg(short, long)]
    pub session: Option<String>,

    /// Start a new session even when `--session` would resolve one.
    #[arg(long)]
    pub new_session: bool,

    /// Emit a structured JSON result instead of plain text.
    #[arg(long, global = true)]
    pub json: bool,

    /// Suppress progress output.
    #[arg(short, long)]
    pub quiet: bool,

    /// Maximum model steps in one turn.
    ///
    /// Unset means the settings file's `max_steps`, which is 50 when the file
    /// says nothing. The flag wins when it is given, so one long session does
    /// not have to rewrite the file — and `ngu config set max_steps` is what
    /// makes the number stick.
    #[arg(long)]
    pub max_steps: Option<usize>,

    /// System prompt override.
    #[arg(long)]
    pub system: Option<String>,

    /// Working directory for the session.
    #[arg(long, global = true)]
    pub cwd: Option<PathBuf>,

    /// Session storage root, overriding `$NGU_HOME/sessions`.
    ///
    /// Deliberately not bound to `NGU_HOME`: that variable names the state
    /// *home*, and both frontends have to read it the same way. Binding this
    /// flag to it meant `NGU_HOME=/x` put the command line's sessions in `/x`
    /// while the desktop put them in `/x/sessions` — one variable, two
    /// directories, and a history that looked split for no visible reason.
    #[arg(long, global = true)]
    pub home: Option<PathBuf>,

    /// Extra skill search root (repeatable).
    #[arg(long = "skill-dir", global = true)]
    pub skill_dirs: Vec<PathBuf>,

    /// Reasoning effort: minimal, low, medium, or high.
    #[arg(long, env = "NGU_REASONING_EFFORT", global = true)]
    pub reasoning_effort: Option<String>,

    /// How the answer is delivered: stream, or buffered for a gateway whose
    /// stream never completes. Overrides the stored setting for this run.
    #[arg(long, env = "NGU_DELIVERY", global = true)]
    pub delivery: Option<String>,

    /// Persona text, applied as a session-scoped hot change.
    #[arg(long)]
    pub persona: Option<String>,

    /// Cache policy: freshness, balanced, or cache-first.
    #[arg(long, default_value = "balanced")]
    pub cache_policy: String,

    /// Snapshot the workspace with git before and after each turn.
    #[arg(long)]
    pub git_snapshot: bool,

    /// Load an assembly manifest before running.
    #[arg(long)]
    pub assembly: Option<PathBuf>,

    #[command(subcommand)]
    pub command: Option<Command>,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// List stored sessions.
    Sessions,
    /// Print a stored session's transcript.
    Show {
        /// Session id.
        id: String,
    },
    /// Delete a stored session.
    Delete {
        /// Session id.
        id: String,
    },
    /// Show the token and time usage recorded by stored sessions.
    Stats {
        /// Session id. Defaults to the most recently updated session.
        id: Option<String>,
        /// Report every session, plus a grand total.
        #[arg(long)]
        all: bool,
        /// Also report money, using the rates in the price file.
        #[arg(long)]
        cost: bool,
        /// Price file to read instead of the default one.
        #[arg(long)]
        prices: Option<PathBuf>,
    },
    /// Export a stored session as a file meant to be shared.
    ///
    /// The format is Markdown unless `--format json` asks for JSON. The global
    /// `--json` flag also selects JSON, and it wins over `--format`: a script
    /// that already passes `--json` to every subcommand must not have to know
    /// which of two flags this one happens to read first.
    Export {
        /// Session id.
        id: String,
        /// Write to this file instead of stdout. Parent directories are created.
        #[arg(long)]
        out: Option<PathBuf>,
        /// Output format: `md` (or `markdown`) or `json`.
        #[arg(long)]
        format: Option<String>,
    },
    /// List models available on the configured route.
    Models {
        /// Also print the context window each model reports.
        ///
        /// A provider that reports one is the reason the compaction policy can
        /// stop guessing, so the number is recorded here as well: a turn in this
        /// process then measures its history against what the provider actually
        /// enforces instead of against a built-in table.
        #[arg(long)]
        windows: bool,
    },
    /// Check this machine's configuration without changing anything.
    Doctor,
    /// List skills found under the configured roots.
    Skills {
        /// Also print each skill's full instructions.
        #[arg(long)]
        verbose: bool,
    },
    /// Validate an assembly manifest and print its load plan.
    Assembly {
        /// Path to `assembly.yaml`.
        file: PathBuf,
        /// Platform to plan for (win32, darwin, linux).
        #[arg(long)]
        platform: Option<String>,
    },
    /// Apply an assembly manifest: load its plugins, MCP servers, and skills.
    Apply {
        /// Path to `assembly.yaml`.
        file: PathBuf,
        /// Platform to plan for.
        #[arg(long)]
        platform: Option<String>,
    },
    /// Inspect or restore workspace snapshots.
    Snapshot {
        /// `list` or `restore`.
        #[arg(default_value = "list")]
        action: String,
        /// Commit to restore, for `restore`.
        commit: Option<String>,
        /// Maximum snapshots to list.
        #[arg(long, default_value_t = 20)]
        limit: usize,
    },
    /// Build a `.ngupack` archive from a pack directory.
    Pack {
        /// Pack directory (must contain ngu.index.json; older spellings are read).
        dir: PathBuf,
        /// Output archive path. Defaults to <name>-<version>.ngupack.
        #[arg(short, long)]
        out: Option<PathBuf>,
        /// Fetch every reference and write its hash into the manifest.
        ///
        /// Without a hash an install cannot use the cache — there is nothing to
        /// look the content up by — so every install re-downloads. Pinning also
        /// makes the pack reproducible: the bytes are fixed rather than
        /// whatever the branch happens to hold today.
        #[arg(long)]
        pin: bool,
        /// Embed the referenced content, for an air-gapped install.
        #[arg(long)]
        offline: bool,
    },
    /// Install a `.ngupack` archive (the older `.dshpack` is accepted).
    Install {
        /// What to install: a local `.ngupack` path, `github:owner/repo[@path][@ref]`,
        /// or an `https://` URL to an archive.
        file: PathBuf,
        /// Directory to install into.
        #[arg(long)]
        into: Option<PathBuf>,
    },
    /// MCP helpers (`ngu mcp import`, matching `config set` / `plugin load`).
    Mcp {
        #[command(subcommand)]
        action: McpAction,
    },
    /// Show the plugin event log (the audit trail on disk).
    Events {
        /// Lines from the end to show.
        #[arg(default_value_t = 20)]
        last: usize,
    },
    /// List installed packs.
    Packs,

    /// Take an installed pack out of every conversation, keeping its files.
    Uninstall {
        /// Pack name, or `name@version` to pick one of several.
        name: String,
        /// Remove every installed version of that name.
        #[arg(long)]
        all: bool,
        /// Delete the files instead of only unloading them.
        #[arg(long)]
        delete: bool,
        /// Directory to unload from (or delete from, with `--delete`).
        #[arg(long)]
        into: Option<PathBuf>,
    },
    /// Write the built-in interface into a directory, to start a pack from.
    ///
    /// Without this, replacing the interface means rewriting every event
    /// handler from nothing, which nobody does. The built-in one is the
    /// worked example.
    Ui {
        /// Where to write it. Defaults to ./ui.
        dir: Option<PathBuf>,
    },
    /// Verify a `.ngupack` archive without installing it.
    Verify {
        /// Archive to verify.
        file: PathBuf,
    },
    /// Load a dynamically linked plugin and report what it contributes.
    Plugin {
        #[command(subcommand)]
        action: PluginAction,
    },
    /// Show or change the stored API settings (endpoint, key, model).
    Config {
        #[command(subcommand)]
        action: Option<ConfigAction>,
    },
    /// Print the runtime's policy table and effective configuration.
    /// Show the injection entries and what they would match.
    Injections {
        /// Show which entries would activate for this text.
        #[arg(long)]
        r#for: Option<String>,
    },
    Runtime,
    /// List the routes the kernel can answer, and call one.
    ///
    /// 这是"一切皆调用"的验证入口：`ngu routes` 看到表里有什么，
    /// `ngu routes --call <名字> [--args <JSON>]` 真的走一遍。
    Routes {
        /// Call this route instead of listing them.
        #[arg(long)]
        call: Option<String>,
        /// JSON arguments for `--call`.
        #[arg(long, default_value = "{}")]
        args: String,
    },
}

#[derive(Subcommand, Debug)]
pub enum ConfigAction {
    /// Print the effective settings and where they are stored.
    Show,
    /// Write settings to the settings file.
    Set {
        /// API endpoint, including the version segment, e.g. https://host/v1.
        #[arg(long)]
        base_url: Option<String>,
        /// API key.
        #[arg(long)]
        api_key: Option<String>,
        /// Model id.
        #[arg(long)]
        model: Option<String>,
        /// Reasoning effort: none, minimal, low, medium, high, xhigh, or max.
        #[arg(long)]
        reasoning_effort: Option<String>,
        /// How the answer is delivered: `stream` (the default), or `buffered`
        /// for a gateway whose stream never completes.
        #[arg(long)]
        delivery: Option<String>,
        /// Proxy URL, e.g. http://127.0.0.1:7890. Pass an empty string to clear
        /// it and go direct.
        #[arg(long)]
        proxy: Option<String>,
        /// Context window: 128K, 1M, or a plain token count. Pass 0 for automatic.
        #[arg(long)]
        context_window: Option<String>,
        /// Compact once this percentage of the window is in use.
        #[arg(long)]
        compact_percent: Option<u32>,
        /// Recent messages kept verbatim when compacting.
        #[arg(long)]
        compact_keep_recent: Option<usize>,
        /// Most model steps one turn may take, 1..=200.
        #[arg(long)]
        max_steps: Option<u32>,
        /// Output token ceiling: 8K, 32K, or a plain count. Pass 0 to clear.
        #[arg(long)]
        max_output_tokens: Option<String>,
        /// Foreground `bash` budget in seconds: how long one command may hold the
        /// turn before the answer says it is taking too long.
        #[arg(long)]
        tool_timeout: Option<u64>,
        /// How long one MCP request may take, in seconds.
        #[arg(long)]
        mcp_timeout: Option<u64>,
        /// Model rounds a `delegate` subagent may take.
        #[arg(long)]
        subagent_steps: Option<usize>,
        /// Wall clock a `delegate` subagent may run, in seconds.
        #[arg(long)]
        subagent_wall: Option<u64>,
        /// How deep subagents may nest.
        #[arg(long)]
        subagent_depth: Option<usize>,
        /// The longest `job_output` may block waiting, in seconds.
        #[arg(long)]
        job_wait: Option<u64>,
        /// Network tuning preset: default, reasoning, local, or flaky.
        #[arg(long)]
        network: Option<String>,
        /// Whole-request timeout, in seconds.
        #[arg(long)]
        request_timeout: Option<u64>,
        /// How long an idle connection is kept for reuse, in seconds.
        #[arg(long)]
        pool_idle_timeout: Option<u64>,
        /// How many times a dropped connection is retried.
        #[arg(long)]
        retry_attempts: Option<u32>,
        /// Milliseconds before the first retry; doubles each attempt.
        #[arg(long)]
        retry_backoff_ms: Option<u64>,
        /// Judge endpoint, e.g. https://api.typesafe.ai (no path).
        #[arg(long)]
        judge_endpoint: Option<String>,
        /// Judge API key. The `judge` tool appears only when this is set.
        #[arg(long)]
        judge_key: Option<String>,
        /// Judge model alias (default: jev-latest).
        #[arg(long)]
        judge_model: Option<String>,
    },
    /// Print the settings file path.
    Path,
    /// List the saved API profiles, marking the active one.
    Routes,
    /// Point the stored settings at a saved profile — the next start uses it.
    Use {
        /// Profile name, as shown by `config routes`.
        name: String,
    },
    /// Save the current stored settings as a named profile.
    SaveRoute {
        /// Profile name. An existing profile of this name is replaced.
        name: String,
    },
}

#[derive(Subcommand, Debug)]
pub enum PluginAction {
    /// Load a plugin library and list the tools it registers.
    Load {
        /// Path to the shared library (.dll / .so / .dylib).
        path: PathBuf,
    },
    /// Load a plugin library and call one of its tools.
    Call {
        /// Path to the shared library.
        path: PathBuf,
        /// Tool name.
        tool: String,
        /// Arguments as a JSON object.
        #[arg(default_value = "{}")]
        args: String,
    },
}

#[derive(Subcommand, Debug)]
pub enum McpAction {
    /// Import MCP servers from a Claude Code / Claude Desktop config into a pack.
    Import {
        /// The `.mcp.json` (or claude_desktop_config.json) to read.
        file: PathBuf,
        /// Pack directory to create (default: ./<name>).
        #[arg(long)]
        out: Option<PathBuf>,
        /// Pack name (default: derived from the file name).
        #[arg(long)]
        name: Option<String>,
        /// License for the generated pack — the servers' own licenses stay yours to check.
        #[arg(long, default_value = "MIT")]
        license: String,
    },
}
