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
    // 奶白（warm cream）。原来是纯黑底配蓝灰字：深色底上所有次要文字都往
    // 灰里挤，长对话读着眼睛累。奶白不是纯白 —— 纯白配深字的对比过强，
    // 大面积看反而刺眼；带一点暖的米白才耐看。
    ("--bg", "#f2ede3"),
    ("--panel", "#e9e3d7"),
    ("--line", "#d5cbb9"),
    ("--text", "#3a2e24"),
    ("--dim", "#7d7264"),
    // 陶土橙做主色：配奶白最稳，也不和红/绿两个状态色打架。
    ("--accent", "#b4632a"),
    ("--ok", "#3f7d4e"),
    ("--bad", "#b3402f"),
    ("--warn", "#9a6b16"),
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

