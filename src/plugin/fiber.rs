//! One plugin instance: its state, its epoch, and what it contributed.
//!
//! A [`Fiber`] records which sessions asked for it and what it installed; its
//! [`Epoch`] is the identity of everything it depends on, so a replacement
//! upstream is noticed without anyone asking.

use crate::tools::ToolDef;
use serde_json::Value;
use std::any::Any;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use super::ctx::Disposer;
use super::realm::{RealmId, RealmMap};

/// Lifecycle state of one plugin instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FiberState {
    /// Waiting for a required service.
    Pending,
    /// Loaded and contributing.
    Active,
    /// Load failed; the fiber stays for diagnostics.
    Failed,
}

/// Epoch of a fiber: the identity of everything it depends on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Epoch {
    /// At least one dependency is missing.
    Inactive,
    /// All dependencies present; the string changes when any is replaced.
    Active(String),
}

/// One plugin instance.
pub struct Fiber {
    /// Stable id.
    pub id: u64,
    /// Plugin name.
    pub plugin: String,
    /// Which pack activated this fiber, when a pack did.
    ///
    /// Display provenance: [`Kernel::plugin_origins`] pairs it with
    /// [`Kernel::plugin_names`] so the panel can read a capability back to
    /// the pack that brought it. A load with no pack behind it (kernel code,
    /// a test) keeps `None`.
    pub origin: Option<String>,
    /// Services this fiber needs.
    pub inject: Vec<String>,
    /// Services this fiber declared it provides.
    pub provide: Vec<String>,
    /// Isolation map in force.
    pub realm: RealmMap,
    /// Current state.
    pub state: FiberState,
    /// Failure detail, when `state` is `Failed`, or the reason the last rebuild
    /// was refused.
    pub error: Option<String>,
    /// Configuration the plugin is applied with.
    ///
    /// Kept on the fiber rather than passed in once, because the kernel rebuilds
    /// fibers on its own — [`Kernel::refresh`] re-applies a dependent when its
    /// dependency is replaced. Carrying the configuration here means that
    /// rebuild re-applies with what the host asked for instead of falling back
    /// to `Null` and quietly dropping it.
    pub config: Value,
    /// Interface panels this fiber contributes.
    pub ui: Vec<crate::ui::UiPanel>,
    /// Theme layers this fiber contributes.
    pub themes: Vec<crate::theme::Theme>,
    /// Interface labels this fiber contributes, by element id.
    pub strings: std::collections::BTreeMap<String, String>,
    /// Stylesheet rules this fiber contributes, appended after the built-in
    /// sheet. Kept on the fiber so they unload with it.
    pub css: Vec<String>,
    /// Interface points this fiber hangs code on. Kept here for the same
    /// reason: unloading the pack must take its hooks with it.
    pub hooks: std::collections::BTreeMap<String, String>,
    /// Panel kinds this fiber defines, unloaded with it.
    pub panel_types: std::collections::BTreeMap<String, crate::pack::PanelType>,
    /// Sessions that have asked for this instance.
    ///
    /// The reference count that makes session isolation real: an instance is
    /// torn down when its last user releases it, not when the first one asks.
    pub(crate) users: BTreeSet<String>,
    pub(crate) effects: Vec<Disposer>,
    pub(crate) epoch: Epoch,
}

impl Fiber {
    /// The current epoch, for diagnostics.
    pub fn epoch_label(&self) -> String {
        match &self.epoch {
            Epoch::Inactive => "inactive".to_string(),
            Epoch::Active(s) => s.clone(),
        }
    }

    /// Which sessions hold this instance, sorted.
    ///
    /// The session plane needs this to answer "who would lose this capability if
    /// I unloaded it" before doing so.
    pub fn users(&self) -> Vec<String> {
        self.users.iter().cloned().collect()
    }

    /// How many sessions hold this instance.
    pub fn user_count(&self) -> usize {
        self.users.len()
    }

    /// Whether `session` holds this instance.
    pub fn is_used_by(&self, session: &str) -> bool {
        self.users.contains(session)
    }
}

/// What a [`Kernel::release`] did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReleaseOutcome {
    /// No session still holds the instance: it was torn down and forgotten.
    Unloaded,
    /// The caller's claim was dropped, and this many sessions still hold it.
    StillUsed(usize),
    /// No such fiber: already unloaded.
    Absent,
}

/// One contribution set that passed every check the kernel can make before it
/// touches its own tables.
///
/// This is the whole content of "build the replacement before tearing down the
/// running version": a `Staged` value means `apply` succeeded and nothing it
/// claims clashes with the kernel, so installing it cannot fail.
pub(crate) struct Staged {
    pub(crate) fiber: u64,
    pub(crate) plugin: String,
    pub(crate) epoch: Epoch,
    pub(crate) ui: Vec<crate::ui::UiPanel>,
    pub(crate) themes: Vec<crate::theme::Theme>,
    pub(crate) strings: std::collections::BTreeMap<String, String>,
    pub(crate) css: Vec<String>,
    pub(crate) hooks: std::collections::BTreeMap<String, String>,
    pub(crate) panel_types: std::collections::BTreeMap<String, crate::pack::PanelType>,
    pub(crate) services: Vec<((String, RealmId), Arc<dyn Any + Send + Sync>)>,
    pub(crate) tools: Vec<ToolDef>,
    pub(crate) readers: Vec<crate::tools::readers::ReaderSpec>,
    pub(crate) effects: Vec<Disposer>,
}

/// Names already claimed by fibers staged in the same batch.
///
/// Staging validates against the kernel's live tables, which do not yet contain
/// the batch's own earlier entries; without this, two fibers reloaded together
/// could each look free and then collide at install time — past the point where
/// a rollback is possible.
#[derive(Default)]
pub(crate) struct StagedClaims {
    pub(crate) services: BTreeMap<(String, RealmId), String>,
    pub(crate) tools: BTreeMap<String, String>,
}
