//! Where plugins put things on screen.
//!
//! The desktop shell renders one page. Without a registration point, a plugin
//! that wants to show anything has to modify that page — which means modifying
//! the kernel, which is the thing plugins exist to avoid.
//!
//! A [`UiPanel`] is a named fragment of HTML placed in one of a fixed set of
//! [`UiSlot`]s. The shell renders the slots; plugins fill them. Slots are a
//! closed set on purpose: an open set would let a plugin place content anywhere
//! and make the layout unpredictable.
//!
//! # What a panel may contain
//!
//! HTML, and optionally a script. Both come from a plugin, so both are trusted
//! to the same degree the plugin is — a plugin that can register a tool can
//! already run code. The panel is inserted with `innerHTML` and its script is
//! evaluated once, after insertion, so it can find its own elements.
//!
//! # Why not a full component model
//!
//! There is no virtual DOM here and no reactive binding. A panel that needs to
//! update itself calls back into the shell with `toShell({cmd: ...})`, which is
//! the same channel the built-in panels use. That keeps one update path instead
//! of two.

use serde::{Deserialize, Serialize};

/// The built-in interface, as a string.
///
/// Written out by the eject command so a pack author has a worked example to
/// start from rather than having to handle every event from nothing. It is the
/// same string the desktop shell compiles in, so the two cannot drift.
pub const BUILTIN_HTML: &str = include_str!("../desktop/ui/index.html");

/// A place on screen a plugin may fill.
/// Where a panel goes.
///
/// Named places first, then anywhere: `Slot::At` names a real element id and a
/// position relative to it, which is what lets a pack put a button next to the
/// Send button instead of in one of six fixed spots. The named variants stay
/// because they are the common case and they survive a page restructure; an
/// anchored one is exactly as stable as the id it names.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum UiSlot {
    /// Top of the left column, above the session list.
    SidebarTop,
    /// Bottom of the left column, below the session list.
    SidebarBottom,
    /// Top of the right column, above Settings.
    DetailsTop,
    /// Bottom of the right column, below Tool output.
    DetailsBottom,
    /// Right-hand side of the status bar.
    StatusBar,
    /// Above the composer, spanning the conversation column.
    AboveComposer,
    /// Any element on the page, named by a selector.
    ///
    /// Written in a manifest as `"at": {"select": "#composer .row", "place":
    /// "before"}`. `select` is any CSS selector, not just an id: the page has a
    /// handful of ids and hundreds of class-named elements, so an id-only
    /// anchor could only ever reach a few fixed spots — which is the limit this
    /// replaces.
    ///
    /// Any selector is accepted. A pack already runs what it likes: loading one
    /// can start an MCP server, which is an arbitrary process that can read
    /// `settings.json` and everything else. Refusing a selector would restrict a
    /// capability the pack does not need restricted while changing nothing about
    /// what it can reach — and it would silently break a legitimate panel.
    ///
    /// `all: true` places one host per match instead of only the first, which is
    /// what makes "beside every turn" possible.
    At {
        /// A CSS selector for the anchor element(s).
        #[serde(rename = "select", alias = "id")]
        select: String,
        /// Where, relative to each match.
        place: UiPlacement,
        /// Whether to anchor to every match or only the first.
        #[serde(default)]
        all: bool,
    },
}

/// Where an anchored panel sits relative to its id.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum UiPlacement {
    /// As a sibling immediately before the element.
    Before,
    /// As a sibling immediately after the element.
    After,
    /// As the first child.
    Prepend,
    /// As the last child.
    Append,
}


/// A stable sort key for a slot.
///
/// The old code cast the enum to `u8`, which a variant carrying data cannot do.
/// An anchored slot sorts last and carries its anchor's name into the key, so
/// two panels on the same id keep a deterministic order.
fn slot_rank(slot: &UiSlot) -> (u8, &str) {
    match slot {
        UiSlot::SidebarTop => (0, ""),
        UiSlot::SidebarBottom => (1, ""),
        UiSlot::DetailsTop => (2, ""),
        UiSlot::DetailsBottom => (3, ""),
        UiSlot::StatusBar => (4, ""),
        UiSlot::AboveComposer => (5, ""),
        UiSlot::At { select, .. } => (6, select.as_str()),
    }
}


impl UiSlot {
    /// Every slot, so the shell can create a container for each.
    pub fn all() -> &'static [UiSlot] {
        &[
            UiSlot::SidebarTop,
            UiSlot::SidebarBottom,
            UiSlot::DetailsTop,
            UiSlot::DetailsBottom,
            UiSlot::StatusBar,
            UiSlot::AboveComposer,
        ]
    }

    /// The element id the shell gives this slot's container.
    pub fn element_id(&self) -> &'static str {
        match self {
            UiSlot::SidebarTop => "slot-sidebar-top",
            UiSlot::SidebarBottom => "slot-sidebar-bottom",
            UiSlot::DetailsTop => "slot-details-top",
            UiSlot::DetailsBottom => "slot-details-bottom",
            UiSlot::StatusBar => "slot-status-bar",
            UiSlot::AboveComposer => "slot-above-composer",
            // An anchored panel has no pre-made container: the page creates its
            // host at insertion time, next to the element the panel named.
            UiSlot::At { .. } => "",
        }
    }

    /// The name a plugin writes in its manifest.
    pub fn as_str(&self) -> &'static str {
        match self {
            UiSlot::SidebarTop => "sidebar-top",
            UiSlot::SidebarBottom => "sidebar-bottom",
            UiSlot::DetailsTop => "details-top",
            UiSlot::DetailsBottom => "details-bottom",
            UiSlot::StatusBar => "status-bar",
            UiSlot::AboveComposer => "above-composer",
            UiSlot::At { .. } => "at",
        }
    }
}

/// The panel kind used when a pack does not name one.
fn default_kind() -> String {
    "html".to_string()
}

/// A fragment of interface contributed by a plugin.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UiPanel {
    /// Unique id. A second panel with the same id **in the same namespace**
    /// replaces the first, so a reload does not stack duplicates.
    ///
    /// Ids are scoped: `stats/row` from one pack and `stats/row` from another
    /// are two panels, not one overwriting the other. Before namespaces, two
    /// packs that happened to pick the same id silently clobbered each other —
    /// which is exactly the "my pack's panel vanished" report that has no
    /// visible cause.
    pub id: String,
    /// Which pack's space this id lives in.
    ///
    /// Empty means "unscoped", which keeps every panel written before
    /// namespaces existed working unchanged. A pack that wants to sit beside
    /// another pack's panels in one group gives the same namespace and a
    /// different id; a pack that wants to replace its own panel keeps the same
    /// pair.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub namespace: String,
    /// What kind of panel this is.
    ///
    /// `html` (the default) renders the fragment in a sandboxed frame. Any
    /// other name must be registered by a pack through `look.json`'s `types`
    /// table, which is how a pack defines a *new* kind of panel instead of only
    /// filling a slot — the `pageTypes` registry Patchouli uses for the same
    /// reason.
    ///
    /// An unknown type is reported rather than rendered as raw HTML: silently
    /// falling back would make a typo look like a styling problem.
    #[serde(default = "default_kind", rename = "type")]
    pub kind: String,
    /// Where it goes.
    pub slot: UiSlot,
    /// Optional heading rendered above the fragment.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// The HTML fragment.
    pub html: String,
    /// Script evaluated once after the fragment is inserted. It runs after
    /// insertion so it can query the elements it just added.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub script: Option<String>,
    /// Higher sorts earlier within a slot.
    #[serde(default)]
    pub order: i32,
}

impl UiPanel {
    /// A panel with no title and no script.
    pub fn new(id: impl Into<String>, slot: UiSlot, html: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            namespace: String::new(),
            kind: default_kind(),
            slot,
            title: None,
            html: html.into(),
            script: None,
            order: 0,
        }
    }

    /// Give it a heading.
    pub fn titled(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Attach a script to run after insertion.
    pub fn with_script(mut self, script: impl Into<String>) -> Self {
        self.script = Some(script.into());
        self
    }

    /// Sort earlier within its slot.
    pub fn ordered(mut self, order: i32) -> Self {
        self.order = order;
        self
    }
}

/// Every panel a set of plugins contributed.
///
/// Panels are kept in one place so the shell asks once per turn rather than
/// walking the plugin graph, and so a duplicate id is resolved the same way
/// every time.
#[derive(Debug, Clone, Default)]
pub struct UiRegistry {
    panels: Vec<UiPanel>,
}

impl UiRegistry {
    /// An empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add or replace a panel by (namespace, id).
    ///
    /// The pair is the identity, not the id alone: two packs may both call
    /// their panel `stats`, and the second one should appear beside the first
    /// rather than erase it. A pack replacing its *own* panel keeps the same
    /// pair, so a reload still updates in place.
    pub fn add(&mut self, panel: UiPanel) {
        match self
            .panels
            .iter_mut()
            .find(|p| p.id == panel.id && p.namespace == panel.namespace)
        {
            Some(existing) => *existing = panel,
            None => self.panels.push(panel),
        }
    }

    /// Add every panel in a list.
    pub fn extend(&mut self, panels: impl IntoIterator<Item = UiPanel>) {
        for panel in panels {
            self.add(panel);
        }
    }

    /// Remove a panel by id.
    pub fn remove(&mut self, id: &str) {
        self.panels.retain(|panel| panel.id != id);
    }

    /// Remove by the full identity, leaving same-named panels elsewhere alone.
    pub fn remove_scoped(&mut self, namespace: &str, id: &str) {
        self.panels
            .retain(|panel| !(panel.id == id && panel.namespace == namespace));
    }

    /// Every panel, ordered for rendering: by slot, then by order, then by id.
    pub fn panels(&self) -> Vec<&UiPanel> {
        let mut panels: Vec<&UiPanel> = self.panels.iter().collect();
        panels.sort_by(|a, b| {
            slot_rank(&a.slot)
                .cmp(&slot_rank(&b.slot))
                .then_with(|| b.order.cmp(&a.order))
                .then_with(|| a.id.cmp(&b.id))
        });
        panels
    }

    /// How many panels there are.
    pub fn len(&self) -> usize {
        self.panels.len()
    }

    /// Whether there are none.
    pub fn is_empty(&self) -> bool {
        self.panels.is_empty()
    }
}

