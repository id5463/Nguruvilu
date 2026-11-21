//! 加载层的数据模型：一个条目加载成什么、失败了怎么记、加载器本身握着什么。
//!
//! 这里是纯数据。真正的加载动作在 [`super::apply`]、[`super::entries`]
//! 和 [`super::process`] 里，各自负责一个阶段。

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;

use crate::ledger::Ledger;
use crate::mcp::McpClient;
use crate::plugin::Kernel;
use crate::skills::SkillRegistry;

pub enum Outcome {
    /// Loaded for the first time in this process.
    Loaded,
    /// Already present; reused.
    Reused,
    /// Deliberately left out.
    Skipped(String),
}

/// An entry that failed while its policy allowed the rest to continue.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FailedEntry {
    /// Entry id.
    pub id: String,
    /// Entry kind.
    pub kind: String,
    /// Why it failed.
    pub error: String,
}

/// Result of applying one plan.
#[derive(Debug, Clone, Default)]
pub struct LoadReport {
    /// Entries loaded.
    pub loaded: Vec<String>,
    /// Entries reused from a previous load.
    pub reused: Vec<String>,
    /// Entries skipped, with reasons.
    pub skipped: Vec<SkippedEntry>,
    /// Entries that failed under a non-aborting policy.
    pub failed: Vec<FailedEntry>,
}

impl LoadReport {
    /// Whether every entry either loaded or was reused.
    pub fn is_clean(&self) -> bool {
        self.failed.is_empty()
    }
}

/// Owns the kernel plus everything the loader manages alongside it.
/// An MCP server that has not been started yet.
///
/// The tools it declares are already in the table, so the model sees exactly
/// what it would see after a start. The first call starts the server; later ones
/// reuse it. A failed start is not cached — the next call tries again, because a
/// server that was missing a moment ago may have been installed since.
struct LazyMcp {
    spec: McpSpec,
    client: tokio::sync::Mutex<Option<Arc<McpClient>>>,
}

impl LazyMcp {
    async fn get(&self) -> Result<Arc<McpClient>> {
        let mut slot = self.client.lock().await;
        if let Some(client) = slot.as_ref() {
            return Ok(Arc::clone(client));
        }
        let started = Arc::new(McpClient::connect(&self.spec).await?);
        crate::mcp::track(&started);
        *slot = Some(Arc::clone(&started));
        Ok(started)
    }
}

pub struct Loader {
    kernel: Kernel,
    skills: SkillRegistry,
    mcp_clients: Vec<Arc<McpClient>>,
    ledger: Ledger,
    ledger_path: PathBuf,
    pack: String,
    pack_dir: Option<PathBuf>,
    /// Plugin processes this loader started, kept alive for as long as it is.
    ///
    /// Held rather than dropped because dropping one stops it: the process is
    /// the capability, and a plugin whose process ended contributes nothing.
    processes: Vec<Arc<crate::protocol::Process>>,
    /// MCP servers declared but not started, kept so their pipes are not dropped
    /// between calls.
    lazy_mcp: Vec<Arc<LazyMcp>>,
}

