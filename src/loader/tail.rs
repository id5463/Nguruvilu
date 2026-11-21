//! 加载器的收尾与几处查询：包里设置的形状、源码从哪解析、插件在哪个隔离域、
//! 以及交回它握着的东西。
//!
//! `finish` 与 `shutdown` 是两种不同的结束：前者把内核和工具表交出去、MCP
//! server 继续活着；后者把 server 停掉。分成两个方法是因为“加载结束”不等于
//! “会话结束”。

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;

use crate::assembly::PlannedStep;
use crate::mcp::McpClient;
use crate::plugin::{Kernel, RealmMap};
use crate::skills::SkillRegistry;

use super::model::Loader;

impl Loader {
    /// Every pack's settings, as the table a plugin reads.
    ///
    /// Serialised here rather than in `protocol` because the loader is what has
    /// a process to hand it to; the protocol module never reads the settings
    /// file.
    fn pack_settings(&self) -> std::collections::BTreeMap<String, serde_json::Value> {
        crate::settings::Settings::load()
            .unwrap_or_default()
            .packs
    }



    fn resolve_source(&self, raw: &str) -> PathBuf {
        let candidate = PathBuf::from(raw);
        if candidate.is_absolute() {
            return candidate;
        }
        match &self.pack_dir {
            Some(dir) => {
                let joined = dir.join(&candidate);
                if joined.exists() {
                    joined
                } else {
                    candidate
                }
            }
            None => candidate,
        }
    }

    /// Resolve the realm for an entry from its scope.
    ///
    /// A session-scoped entry gets a fresh realm, which is what lets two packs
    /// each own a service of the same name. A global entry lands in the root
    /// realm where every session sees it.
    fn realm_for(&self, step: &PlannedStep) -> RealmMap {
        match step.scope {
            // Global entries land in the root realm, where every session sees
            // them.
            Scope::Global => RealmMap::new(),
            // Session entries default to a fresh realm, so everything this pack
            // contributes is isolated from every other session's copy.
            Scope::Session => RealmMap::new().with_default(self.kernel.new_realm()),
        }
    }

    /// Take the kernel back, handing over the skill roots this pack added and
    /// the MCP servers it started.
    ///
    /// The MCP servers are deliberately **not** stopped here: their tools are
    /// in this kernel and stay callable for as long as the conversation runs;
    /// stopping them here killed every MCP tool the moment loading ended. The
    /// caller keeps them — and stops them on the way out, because a server
    /// still running when the runtime tears down is what kept the process alive
    /// after everything it had to say was printed.
    ///
    /// The skills come back rather than staying inside the loader for the same
    /// reason: the host owns the standing catalog, and a pack's roots join it
    /// instead of replacing it.
    pub async fn finish(self) -> (Kernel, SkillRegistry, Vec<Arc<McpClient>>) {
        let Loader { kernel, skills, mcp_clients, .. } = self;
        (kernel, skills, mcp_clients)
    }

    /// Stop every MCP server this loader started.
    ///
    /// For a preview that is about to exit, not for a session: see
    /// [`Loader::finish`].
    pub async fn shutdown(&mut self) {
        for client in self.mcp_clients.drain(..) {
            client.shutdown().await;
        }
    }
}
