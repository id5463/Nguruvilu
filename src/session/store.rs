//! Durable session storage: the trait a backend implements, and the service
//! that makes a store reachable from a plugin.

use std::path::Path;

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::message::Message;

use super::model::{Session, SessionSummary, TurnStats};

/// Durable session storage.
pub trait SessionStore: Send + Sync {
    /// Write a new session (header plus any messages).
    fn create(&self, session: &Session) -> Result<()>;
    /// Append messages to an existing session.
    fn append(&self, id: &str, messages: &[Message]) -> Result<()>;
    /// Record that a leading run of messages was replaced by a summary.
    ///
    /// Defaulted to a no-op so a store that does not model compaction still
    /// satisfies the trait; the JSONL store implements it.
    fn append_compaction(&self, _id: &str, _compaction: &crate::compaction::Compaction) -> Result<()> {
        Ok(())
    }
    /// Record what was injected into a request, so the log explains what the
    /// model was actually sent.
    fn append_injection(&self, _id: &str, _record: &InjectionRecord) -> Result<()> {
        Ok(())
    }
    /// Record the full set of packs this session has loaded. The latest set
    /// recorded is the current one; a store without pack tracking keeps none.
    fn append_packs(&self, _id: &str, _packs: &[String]) -> Result<()> {
        Ok(())
    }
    /// Record one turn's usage. Defaulted to a no-op, like the other optional
    /// records: a store that keeps no ledger is still a valid store.
    fn append_stats(&self, _id: &str, _stats: &TurnStats) -> Result<()> {
        Ok(())
    }
    /// Load a session, or `None` when it does not exist.
    fn load(&self, id: &str) -> Result<Option<Session>>;
    /// List sessions, most recently updated first.
    fn list(&self) -> Result<Vec<SessionSummary>>;
    /// Delete a session. Succeeds when it is already absent.
    fn delete(&self, id: &str) -> Result<()>;
    /// The most recently updated session.
    fn latest(&self) -> Result<Option<SessionSummary>> {
        Ok(self.list()?.into_iter().next())
    }
    /// Root directory holding the sessions.
    fn root(&self) -> &Path;
}

/// One injection that happened, for the session log.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InjectionRecord {
    /// Ids of the entries that were placed.
    pub activated: Vec<String>,
    /// Entries that were moved because the cache policy refused their position.
    #[serde(default)]
    pub relocated: Vec<String>,
    /// Estimated tokens the fragments cost.
    #[serde(default)]
    pub budget_used: usize,
    /// The text that was placed, so a reader can see it verbatim.
    #[serde(default)]
    pub texts: Vec<String>,
}

// ---------------------------------------------------------------- the service
//
// A plugin that wants to start its own conversation — a subagent with a real
// id, a background worker, a second view — needs somewhere to put it. The store
// is that somewhere, and it used to be reachable only from the shell.
//
// Registered as a service rather than passed around: the kernel already hands
// services to plugins through `PluginCtx`, and a store is exactly the shape
// that fits — one shared object, many readers, no ownership question.

/// Name the store answers to in the service table.
pub const SERVICE: &str = "sessions";

/// The store as a plugin sees it.
///
/// A trait object rather than the concrete `JsonlStore` because the host owns
/// the path and may swap the backend; a plugin only ever needs these four
/// operations.
pub trait SessionService: std::any::Any + Send + Sync {
    /// Write a new session.
    fn create(&self, session: &Session) -> Result<()>;
    /// Append messages to an existing session.
    fn append(&self, id: &str, messages: &[Message]) -> Result<()>;
    /// Read one session back.
    fn load(&self, id: &str) -> Result<Option<Session>>;
    /// Every session id the store holds, newest first as the store defines it.
    fn ids(&self) -> Result<Vec<String>>;
}

/// The plugin that puts the store in the service table.
struct StoreService {
    store: std::sync::Arc<dyn SessionService>,
}

impl StoreService {
    const NAME: &str = "builtin:sessions";
}

impl crate::plugin::Plugin for StoreService {
    fn name(&self) -> &str {
        Self::NAME
    }

    fn provide(&self) -> Vec<String> {
        vec![SERVICE.to_string()]
    }

    fn apply(&self, _ctx: &crate::plugin::PluginCtx) -> Result<crate::plugin::Contributions> {
        let handle: std::sync::Arc<dyn std::any::Any + Send + Sync> = self.store.clone() as _;
        Ok(crate::plugin::Contributions::new().service(SERVICE, handle))
    }
}

/// Register the session store so plugins can reach it.
pub fn install_store(
    kernel: &mut crate::plugin::Kernel,
    store: std::sync::Arc<dyn SessionService>,
) -> Result<u64> {
    kernel.define(std::sync::Arc::new(StoreService { store }));
    kernel.load(StoreService::NAME, crate::plugin::RealmMap::new(), serde_json::Value::Null)
}
