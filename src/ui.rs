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

/// A place on screen a plugin may fill.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
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
        }
    }
}

/// A fragment of interface contributed by a plugin.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UiPanel {
    /// Unique id. A second panel with the same id replaces the first, so a
    /// reload does not stack duplicates.
    pub id: String,
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

    /// Add or replace a panel by id.
    pub fn add(&mut self, panel: UiPanel) {
        match self.panels.iter_mut().find(|p| p.id == panel.id) {
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

    /// Every panel, ordered for rendering: by slot, then by order, then by id.
    pub fn panels(&self) -> Vec<&UiPanel> {
        let mut panels: Vec<&UiPanel> = self.panels.iter().collect();
        panels.sort_by(|a, b| {
            (a.slot as u8)
                .cmp(&(b.slot as u8))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_panel_lands_in_its_slot() {
        let mut registry = UiRegistry::new();
        registry.add(UiPanel::new("p", UiSlot::DetailsTop, "<div>hi</div>").titled("Hello"));

        let panels = registry.panels();
        assert_eq!(panels.len(), 1);
        assert_eq!(panels[0].slot, UiSlot::DetailsTop);
        assert_eq!(panels[0].title.as_deref(), Some("Hello"));
    }

    #[test]
    fn a_duplicate_id_replaces_rather_than_stacks() {
        // A reload must not leave two copies of the same panel on screen.
        let mut registry = UiRegistry::new();
        registry.add(UiPanel::new("same", UiSlot::DetailsTop, "<div>first</div>"));
        registry.add(UiPanel::new("same", UiSlot::DetailsTop, "<div>second</div>"));

        assert_eq!(registry.len(), 1);
        assert!(registry.panels()[0].html.contains("second"));
    }

    #[test]
    fn panels_are_ordered_within_a_slot() {
        let mut registry = UiRegistry::new();
        registry.add(UiPanel::new("low", UiSlot::DetailsTop, "").ordered(1));
        registry.add(UiPanel::new("high", UiSlot::DetailsTop, "").ordered(10));
        registry.add(UiPanel::new("mid", UiSlot::DetailsTop, "").ordered(5));

        let ids: Vec<&str> = registry.panels().iter().map(|p| p.id.as_str()).collect();
        assert_eq!(ids, vec!["high", "mid", "low"], "higher order renders first");
    }

    #[test]
    fn a_tie_is_broken_by_id_so_the_order_is_stable() {
        let mut registry = UiRegistry::new();
        registry.add(UiPanel::new("b", UiSlot::DetailsTop, "").ordered(5));
        registry.add(UiPanel::new("a", UiSlot::DetailsTop, "").ordered(5));

        let ids: Vec<&str> = registry.panels().iter().map(|p| p.id.as_str()).collect();
        assert_eq!(ids, vec!["a", "b"]);
    }

    #[test]
    fn removing_a_panel_works() {
        let mut registry = UiRegistry::new();
        registry.add(UiPanel::new("gone", UiSlot::DetailsTop, ""));
        registry.add(UiPanel::new("kept", UiSlot::DetailsTop, ""));
        registry.remove("gone");

        assert_eq!(registry.len(), 1);
        assert_eq!(registry.panels()[0].id, "kept");
    }

    #[test]
    fn every_slot_has_a_distinct_element_id_and_name() {
        let mut ids: Vec<&str> = UiSlot::all().iter().map(|s| s.element_id()).collect();
        let mut names: Vec<&str> = UiSlot::all().iter().map(|s| s.as_str()).collect();
        ids.sort();
        names.sort();
        ids.dedup();
        names.dedup();
        assert_eq!(ids.len(), UiSlot::all().len());
        assert_eq!(names.len(), UiSlot::all().len());
    }

    #[test]
    fn a_slot_round_trips_through_its_name() {
        for slot in UiSlot::all() {
            let json = serde_json::to_string(slot).unwrap();
            let back: UiSlot = serde_json::from_str(&json).unwrap();
            assert_eq!(*slot, back, "{json}");
        }
    }

    #[test]
    fn a_panel_serialises_for_the_page() {
        let panel = UiPanel::new("x", UiSlot::StatusBar, "<b>1</b>")
            .titled("Count")
            .with_script("document.title = 'ok';");

        let json = serde_json::to_value(&panel).unwrap();
        assert_eq!(json["id"], "x");
        assert_eq!(json["slot"], "status-bar");
        assert_eq!(json["title"], "Count");
        assert_eq!(json["script"], "document.title = 'ok';");
    }

    #[test]
    fn optional_fields_stay_out_of_the_payload_when_unset() {
        let panel = UiPanel::new("bare", UiSlot::DetailsTop, "<i></i>");
        let json = serde_json::to_value(&panel).unwrap();
        assert!(json.get("title").is_none());
        assert!(json.get("script").is_none());
    }
}
