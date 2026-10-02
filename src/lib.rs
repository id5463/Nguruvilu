//! Nguruvilu — an independent agent kernel.
//!
//! The kernel does four things: read and write files, run commands, drive the
//! model loop, and manage sessions. Everything else — plugins, MCP servers,
//! skills — is loaded dynamically by the loader layer, which treats the three
//! as peers.
//!
//! Module map:
//!
//! * [`message`] — neutral, provider-independent message format.
//! * [`llm`] — OpenAI-format streaming client with tool-call assembly.
//! * [`tools`] — the mutable tool table and the four base tools.
//! * [`agent`] — the loop: multi-step, parallel tool dispatch, per-turn snapshot.
//! * [`session`] — sessions and JSONL persistence behind a store trait.
//! * [`plugin`] — isolation realms, epochs, reversible effects, permissions.
//! * [`skills`] — on-demand instruction packs.
//! * [`mcp`] — a minimal MCP client over stdio.
//! * [`assembly`] — the pack manifest: what loads, and in what order.
//! * [`ledger`] — the install ledger that makes loading idempotent.
//! * [`loader`] — the dynamic loading layer tying all of the above together.

pub mod agent;
pub mod assembly;
pub mod compaction;
pub mod content;
pub mod context;
pub mod dylib;
pub mod fetch;
pub mod git;
pub mod hotreload;
pub mod ledger;
pub mod llm;
pub mod loader;
pub mod mcp;
pub mod message;
pub mod network;
pub mod pack;
pub mod model;
pub mod plugin;
pub mod preinstall;
pub mod request;
pub mod session;
pub mod window;
pub mod settings;
pub mod source;
pub mod theme;
pub mod ui;
pub mod size;
pub mod skills;
pub mod tools;

pub use agent::{Agent, AgentOutcome};
pub use assembly::{Assembly, EntryKind, OnFailure, Plan, PlannedStep, Scope};
pub use compaction::{Compaction, Summarizer};
pub use content::PackContent;
pub use context::{Injection, InjectionEngine, InjectionEntry, Position};
pub use dylib::{DynamicPlugin, HostApi, PluginMeta, ToolSpec, ABI_VERSION};
pub use fetch::{Fetched, Fetcher, Source};
pub use git::{GitSnapshot, SnapshotRecord};
pub use hotreload::{
    AppliedChange, ApplyOutcome, CachePolicy, Change, ChangeKind, ChangePayload, ChangeScope,
    Consent, ModelRoute, PendingChange, Runtime, Snapshot,
};
pub use ledger::{Ledger, LedgerEntry};
pub use llm::{LlmClient, LlmConfig, LlmEvent};
pub use loader::{LoadReport, Loader, Outcome};
pub use mcp::{McpClient, McpSpec, McpTool};
pub use message::{Message, Role, ToolCall};
pub use network::{NetworkPolicy, NetworkSettings};
pub use pack::{InstalledPack, PackManifest, VerifyReport};
pub use plugin::{
    Action, Contributions, Decision, FiberState, Kernel, PermissionProvider, PermissionStack,
    Plugin, PluginCtx, RealmId, RealmMap, ReloadEvent,
};
pub use request::{ExtraFields, RequestShaper};
pub use settings::Settings;
pub use source::{FilesystemSource, PackSource};
pub use model::{ModelAccess, ModelService};
pub use theme::{Theme, ThemeRegistry};
pub use ui::{UiPanel, UiRegistry, UiSlot};
pub use size::{format_size, parse_size};
pub use window::{ContextPolicy, ContextWindow, WindowSource};
pub use session::{Session, SessionStore};
pub use skills::{Skill, SkillRegistry};
pub use tools::{ToolDef, ToolRegistry};
