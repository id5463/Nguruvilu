//! One injectable fragment, and the resolved injection built from them.

use serde::{Deserialize, Serialize};

use crate::message::Role;

use super::position::{default_role, default_true, default_weight, PositionKind};

/// One injectable fragment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InjectionEntry {
    /// Identifier, used in reports and group tie-breaking.
    pub id: String,
    /// The text to inject.
    pub content: String,
    /// Always inject, regardless of triggers.
    #[serde(default)]
    pub constant: bool,
    /// Substrings that activate this entry. Matching is case-insensitive.
    #[serde(default)]
    pub triggers: Vec<String>,
    /// Where to place it.
    #[serde(default)]
    pub position: PositionKind,
    /// Messages back from the end, for `position: at-depth`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub depth: Option<usize>,
    /// Role to inject under, for history positions.
    #[serde(default = "default_role")]
    pub role: Role,
    /// Sort key; higher is considered first.
    #[serde(default)]
    pub order: i32,
    /// Group name. At most one member of a group is injected.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub group: Option<String>,
    /// Weight within the group; higher wins.
    #[serde(default = "default_weight")]
    pub group_weight: u32,
    /// Only scan this many of the most recent messages for triggers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scan_depth: Option<usize>,
    /// Exempt from the budget.
    #[serde(default)]
    pub ignore_budget: bool,
    /// Whether the entry may trigger further entries.
    #[serde(default = "default_true")]
    pub recursive: bool,
    /// Whether the entry is active at all.
    #[serde(default = "default_true")]
    pub enabled: bool,
}

impl InjectionEntry {
    /// A constant entry placed in the history tail.
    pub fn constant(id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            content: content.into(),
            constant: true,
            triggers: Vec::new(),
            position: PositionKind::HistoryTail,
            depth: None,
            role: Role::System,
            order: 0,
            group: None,
            group_weight: default_weight(),
            scan_depth: None,
            ignore_budget: false,
            recursive: true,
            enabled: true,
        }
    }

    /// A trigger-activated entry.
    pub fn triggered(
        id: impl Into<String>,
        content: impl Into<String>,
        triggers: Vec<String>,
    ) -> Self {
        Self {
            constant: false,
            triggers,
            ..Self::constant(id, content)
        }
    }

    /// Place this entry at a position.
    pub fn at(mut self, position: PositionKind) -> Self {
        self.position = position;
        self
    }

    /// Place this entry a number of messages back from the end.
    pub fn at_depth(mut self, depth: usize) -> Self {
        self.position = PositionKind::AtDepth;
        self.depth = Some(depth);
        self
    }

    /// Give this entry an order.
    pub fn ordered(mut self, order: i32) -> Self {
        self.order = order;
        self
    }

    /// Put this entry in a group.
    pub fn in_group(mut self, group: impl Into<String>, weight: u32) -> Self {
        self.group = Some(group.into());
        self.group_weight = weight;
        self
    }

    /// Exempt this entry from the budget.
    pub fn ignoring_budget(mut self) -> Self {
        self.ignore_budget = true;
        self
    }

    /// Restrict trigger scanning to the most recent `depth` messages.
    pub fn scanning(mut self, depth: usize) -> Self {
        self.scan_depth = Some(depth);
        self
    }

    /// Whether this entry matches `haystack`.
    pub(super) fn matches(&self, haystack: &str) -> bool {
        if self.constant {
            return true;
        }
        if self.triggers.is_empty() {
            return false;
        }
        let lowered = haystack.to_lowercase();
        self.triggers
            .iter()
            .any(|trigger| lowered.contains(&trigger.to_lowercase()))
    }
}

/// The resolved injection for one request.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Injection {
    /// Text for the prompt prefix, in order.
    pub prefix: Vec<String>,
    /// Fragments placed inside the history: `(depth from end, role, text)`.
    pub at_depth: Vec<(usize, Role, String)>,
    /// Fragments appended at the history tail, in order.
    pub tail: Vec<String>,
    /// Ids of the entries that were injected.
    pub activated: Vec<String>,
    /// Estimated tokens used.
    pub budget_used: usize,
    /// Whether the budget stopped further activation.
    pub overflowed: bool,
    /// Entries that were activated but moved because the cache policy refused
    /// their preferred position.
    pub relocated: Vec<String>,
}

impl Injection {
    /// Whether anything was injected.
    pub fn is_empty(&self) -> bool {
        self.prefix.is_empty() && self.at_depth.is_empty() && self.tail.is_empty()
    }
}
