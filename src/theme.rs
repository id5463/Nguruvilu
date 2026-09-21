//! Themes: the interface's colours as named tokens.
//!
//! The page is styled entirely through CSS custom properties, so a theme is a
//! mapping from token name to value. A plugin ships one, the kernel collects it,
//! and the shell applies it — no plugin has to know the page's markup, and no
//! plugin can break the layout by replacing a stylesheet.
//!
//! # Partial themes
//!
//! A theme names only the tokens it wants to change. Everything else keeps the
//! built-in value, so "make it light" is six lines rather than a copy of the
//! whole palette that goes stale the moment the page gains a token.
//!
//! # Why tokens and not raw CSS
//!
//! A plugin that could inject arbitrary CSS could restyle the shell into
//! something unusable — hide the composer, move the send button off screen —
//! and the user would have no way to recover from inside the interface. A token
//! list bounds what a theme can do to changing colours and fonts.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Every token the interface understands, with its built-in value.
///
/// This is the contract: a token here can be themed, and a token not here
/// cannot. The list is short on purpose — each entry is a promise that changing
/// it looks right.
pub const TOKENS: &[(&str, &str)] = &[
    ("--bg", "#14161a"),
    ("--panel", "#171a1f"),
    ("--line", "#262b33"),
    ("--text", "#d8dee9"),
    ("--dim", "#7c8798"),
    ("--accent", "#6ea8fe"),
    ("--ok", "#7ddc9a"),
    ("--bad", "#ff8f8f"),
    ("--warn", "#e8c98a"),
    ("--mono", "ui-monospace, \"Cascadia Mono\", Consolas, monospace"),
];

/// Whether a token name is one the interface knows.
pub fn is_known_token(name: &str) -> bool {
    TOKENS.iter().any(|(token, _)| *token == name)
}

/// A set of token overrides.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Theme {
    /// A name, for diagnostics and for the theme picker.
    #[serde(default)]
    pub name: String,
    /// Token name to value. Only the tokens being changed need appear.
    #[serde(default)]
    pub tokens: BTreeMap<String, String>,
}

impl Theme {
    /// An empty theme.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            tokens: BTreeMap::new(),
        }
    }

    /// Set one token.
    pub fn set(mut self, token: impl Into<String>, value: impl Into<String>) -> Self {
        self.tokens.insert(token.into(), value.into());
        self
    }

    /// The built-in dark palette, spelled out.
    pub fn builtin() -> Self {
        let mut theme = Self::new("builtin");
        for (token, value) in TOKENS {
            theme.tokens.insert((*token).to_string(), (*value).to_string());
        }
        theme
    }

    /// A light palette, as the obvious worked example.
    pub fn light() -> Self {
        Self::new("light")
            .set("--bg", "#ffffff")
            .set("--panel", "#f5f6f8")
            .set("--line", "#d8dce3")
            .set("--text", "#1b1f24")
            .set("--dim", "#6b7480")
            .set("--accent", "#1a6fd4")
            .set("--ok", "#1a7f4b")
            .set("--bad", "#c0392b")
            .set("--warn", "#8a6100")
    }

    /// Overlay another theme on top of this one.
    ///
    /// Later layers win, so a plugin can adjust a theme rather than replace it.
    pub fn merge(&mut self, other: &Theme) {
        for (token, value) in &other.tokens {
            self.tokens.insert(token.clone(), value.clone());
        }
    }

    /// Drop tokens the interface does not know.
    ///
    /// Reported rather than applied silently: a typo in a token name is
    /// otherwise a theme that quietly does nothing.
    pub fn sanitize(&mut self) -> Vec<String> {
        let mut rejected = Vec::new();
        self.tokens.retain(|token, _| {
            if is_known_token(token) {
                true
            } else {
                rejected.push(token.clone());
                false
            }
        });
        rejected
    }

    /// Whether this theme changes nothing.
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// How many tokens it sets.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }
}

/// The themes plugins contributed, layered in order.
///
/// Kept as an ordered list rather than one winner so that "a light theme" and
/// "bigger monospace font" compose instead of one silently discarding the
/// other.
#[derive(Debug, Clone, Default)]
pub struct ThemeRegistry {
    layers: Vec<Theme>,
}

impl ThemeRegistry {
    /// An empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a layer. A layer with the same name replaces the earlier one, so a
    /// reload does not stack.
    pub fn add(&mut self, theme: Theme) {
        match self.layers.iter_mut().find(|t| t.name == theme.name) {
            Some(existing) => *existing = theme,
            None => self.layers.push(theme),
        }
    }

    /// Remove a layer by name.
    pub fn remove(&mut self, name: &str) {
        self.layers.retain(|theme| theme.name != name);
    }

    /// The effective token set: built-in values with every layer applied in
    /// registration order.
    pub fn resolve(&self) -> Theme {
        let mut resolved = Theme::builtin();
        for layer in &self.layers {
            resolved.merge(layer);
        }
        resolved.name = if self.layers.is_empty() {
            "builtin".to_string()
        } else {
            self.layers
                .iter()
                .map(|t| t.name.as_str())
                .collect::<Vec<_>>()
                .join("+")
        };
        resolved
    }

    /// The layers, in order.
    pub fn layers(&self) -> &[Theme] {
        &self.layers
    }

    /// Whether no plugin contributed a theme.
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_builtin_palette_matches_the_token_list() {
        let builtin = Theme::builtin();
        assert_eq!(builtin.len(), TOKENS.len());
        for (token, value) in TOKENS {
            assert_eq!(builtin.tokens.get(*token).map(String::as_str), Some(*value));
        }
    }

    #[test]
    fn a_light_theme_changes_only_the_colours_it_names() {
        let light = Theme::light();
        assert_eq!(light.tokens.get("--bg").map(String::as_str), Some("#ffffff"));

        let mut registry = ThemeRegistry::new();
        registry.add(light);
        let resolved = registry.resolve();

        assert_eq!(resolved.tokens["--bg"], "#ffffff", "the theme won");
        assert_eq!(
            resolved.tokens["--mono"],
            TOKENS.iter().find(|(t, _)| *t == "--mono").unwrap().1,
            "an untouched token keeps its built-in value"
        );
    }

    #[test]
    fn a_theme_naming_only_one_token_is_valid() {
        let mut registry = ThemeRegistry::new();
        registry.add(Theme::new("just-bg").set("--bg", "#000000"));

        let resolved = registry.resolve();
        assert_eq!(resolved.tokens["--bg"], "#000000");
        assert_eq!(resolved.len(), TOKENS.len(), "the rest stay built-in");
    }

    #[test]
    fn layers_compose_rather_than_compete() {
        let mut registry = ThemeRegistry::new();
        registry.add(Theme::light());
        registry.add(Theme::new("mono").set("--mono", "Consolas, monospace"));

        let resolved = registry.resolve();
        assert_eq!(resolved.tokens["--bg"], "#ffffff", "the light layer survived");
        assert_eq!(resolved.tokens["--mono"], "Consolas, monospace", "and so did the font");
        assert_eq!(resolved.name, "light+mono");
    }

    #[test]
    fn a_later_layer_wins_on_a_shared_token() {
        let mut registry = ThemeRegistry::new();
        registry.add(Theme::light());
        registry.add(Theme::new("darker").set("--bg", "#f0f0f0"));

        assert_eq!(registry.resolve().tokens["--bg"], "#f0f0f0");
    }

    #[test]
    fn a_reload_replaces_a_layer_instead_of_stacking_it() {
        let mut registry = ThemeRegistry::new();
        registry.add(Theme::new("t").set("--bg", "#111111"));
        registry.add(Theme::new("t").set("--bg", "#222222"));

        assert_eq!(registry.layers().len(), 1);
        assert_eq!(registry.resolve().tokens["--bg"], "#222222");
    }

    #[test]
    fn removing_a_layer_restores_what_was_under_it() {
        let mut registry = ThemeRegistry::new();
        registry.add(Theme::light());
        registry.add(Theme::new("override").set("--bg", "#000000"));

        registry.remove("override");
        assert_eq!(registry.resolve().tokens["--bg"], "#ffffff");
    }

    #[test]
    fn an_unknown_token_is_rejected_and_named() {
        // A typo would otherwise be a theme that silently does nothing.
        let mut theme = Theme::new("typo")
            .set("--bg", "#fff")
            .set("--bakground", "#fff")
            .set("position", "fixed");

        let rejected = theme.sanitize();
        assert_eq!(rejected.len(), 2);
        assert!(rejected.contains(&"--bakground".to_string()));
        assert!(rejected.contains(&"position".to_string()));
        assert_eq!(theme.len(), 1, "the valid token survived");
    }

    #[test]
    fn an_empty_registry_resolves_to_the_builtin() {
        let registry = ThemeRegistry::new();
        assert!(registry.is_empty());
        let resolved = registry.resolve();
        assert_eq!(resolved.name, "builtin");
        assert_eq!(resolved.tokens["--bg"], "#14161a");
    }

    #[test]
    fn a_theme_round_trips_through_json() {
        let theme = Theme::light();
        let json = serde_json::to_string(&theme).unwrap();
        let back: Theme = serde_json::from_str(&json).unwrap();
        assert_eq!(back, theme);
    }

    #[test]
    fn a_token_name_is_recognised() {
        assert!(is_known_token("--bg"));
        assert!(is_known_token("--mono"));
        assert!(!is_known_token("--nope"));
        assert!(!is_known_token("bg"));
    }
}
