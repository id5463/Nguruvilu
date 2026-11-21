//! The queue a load or unload waits in until the turn ends.
//!
//! A `pack` call runs while the loop owns the kernel, so it cannot change the
//! tool table itself. It records the request here instead, and the host drains
//! the queue at the end of the turn — the next moment the table is read anyway.
//!
//! Entries carry the session that asked, because one kernel serves several
//! conversations: a drain that took the whole queue would load a pack another
//! conversation requested into its own tool table.
//!
//! The registration lives here too, because registering the tool and naming
//! the queue it reports to are one decision: a `pack` tool with no queue can
//! only change state on disk, and the only reason to pass one is so the
//! load/unload actions have somewhere to go.

use std::sync::Arc;

use anyhow::Result;
use serde_json::json;

use crate::tools::{ConflictPolicy, ToolDef, ToolFuture, ToolRegistry};

use super::action::run_with;

/// A load or unload this conversation asked for.
///
/// Queued rather than done inside the call: the tool runs while the loop owns
/// the kernel, and the host applies the change at the end of the turn — which
/// is the next moment the tool table is read anyway.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Pending {
    /// Bring an installed pack into this conversation. The session is the one
    /// that asked, so the host can record the pack against it.
    Load { name: String, session: Option<String> },
    /// Take it out of this conversation, keeping its files. Removed from the
    /// kernel only when no other conversation still uses the pack.
    Unload { name: String, session: Option<String> },
    /// An archive just landed on disk. The tool installs it itself — the
    /// kernel is not touched — but the host's pack list is a snapshot the page
    /// took at load time, and without this notice a pack the agent installed
    /// reads as "the agent said installed, the list shows nothing".
    Installed { name: String, version: String },
}

/// What the tool hands the host, drained once the turn is over.
#[derive(Default)]
pub struct PendingQueue {
    items: std::sync::Mutex<Vec<Pending>>,
}

impl PendingQueue {
    /// An empty queue.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one request.
    pub fn push(&self, item: Pending) {
        self.items.lock().expect("pending queue lock").push(item);
    }

    /// Take every request recorded so far, leaving the queue empty.
    ///
    /// This is the process-wide sweep, and it is what a host running a single
    /// conversation wants: the CLI serves one session per process, so every
    /// entry in its queue belongs to the turn that just ended. A host serving
    /// several conversations has to use [`PendingQueue::drain_for`] instead,
    /// because a turn that takes the whole queue loads a pack another
    /// conversation asked for into its own tool table.
    pub fn drain(&self) -> Vec<Pending> {
        std::mem::take(&mut *self.items.lock().expect("pending queue lock"))
    }

    /// Take the requests recorded for `session`, leaving the rest queued.
    ///
    /// An entry that names no conversation is addressed to all of them — an
    /// install changes a pack list every conversation reads — so it goes to
    /// whoever asks first. An entry some conversation queued goes only to that
    /// conversation, which is why asking with `None` takes the session-free
    /// entries and nothing else: an identity-less caller has no claim on work
    /// another conversation is waiting for.
    pub fn drain_for(&self, session: Option<&str>) -> Vec<Pending> {
        let mut items = self.items.lock().expect("pending queue lock");
        let (taken, kept): (Vec<Pending>, Vec<Pending>) = std::mem::take(&mut *items)
            .into_iter()
            .partition(|item| match (session, queued_for(item)) {
                (_, None) => true,
                (Some(asking), Some(queued)) => asking == queued,
                (None, Some(_)) => false,
            });
        *items = kept;
        taken
    }
}

/// The conversation an entry was queued for, or `None` when it is for everyone.
fn queued_for(item: &Pending) -> Option<&str> {
    match item {
        Pending::Load { session, .. } | Pending::Unload { session, .. } => session.as_deref(),
        // An install is news about the installed set, which every conversation
        // reads, so it does not belong to one of them.
        Pending::Installed { .. } => None,
    }
}

/// Register the pack tool with no host attached.
///
/// Enough for tests and for a caller that only wants build/verify/list: the
/// load and unload actions then change state on disk and say so, rather than
/// pretending a kernel was reached.
pub fn register(registry: &mut ToolRegistry) -> Result<()> {
    register_with(registry, None)
}

/// Register the pack tool, handing its load and unload requests to `pending`.
pub fn register_with(
    registry: &mut ToolRegistry,
    pending: Option<Arc<PendingQueue>>,
) -> Result<()> {
    registry.register(
        ToolDef::new(
            "pack",
            "Build, inspect, install, load, and unload `.ngupack` archives (the older `.dshpack` spelling is read). A pack is a zip \
             holding `ngu.index.json` (identity: name, version, license) plus a few small \
             document files, and nothing else — everything it needs is referenced from the \
             manifest. Actions: \
             `build` an archive from a directory, `verify` an archive without installing it, \
             `list` installed packs with whether each is loaded, `install` an archive, \
             `load` an installed pack into this conversation (files are kept, and an unloaded \
             pack comes back), `unload` an installed pack out of this conversation (its tools \
             stop being callable; the files stay), `apply` an installed pack's assembly. \
             Building needs the directory to already contain `ngu.index.json` — write that \
             file first.",
            json!({
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["build", "verify", "list", "install", "load", "unload", "apply"],
                        "description": "What to do."
                    },
                    "path": {
                        "type": "string",
                        "description": "build: the pack directory. verify/install: the .dshpack archive. apply: the installed pack's assembly.yaml. Not used by list, load, or unload."
                    },
                    "name": {
                        "type": "string",
                        "description": "load/unload: the installed pack's name, optionally name@version. Not used by the other actions."
                    },
                    "out": {
                        "type": "string",
                        "description": "build: where to write the archive. Defaults to <name>-<version>.dshpack beside the directory."
                    }
                },
                "required": ["action"]
            }),
            "kernel",
            move |args| {
                let pending = pending.clone();
                Box::pin(async move { run_with(args, pending).await }) as ToolFuture
            },
        ),
        ConflictPolicy::Error,
    )?;
    Ok(())
}
