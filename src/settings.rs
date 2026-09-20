//! Durable settings: where the API key, base URL, and model actually live.
//!
//! These are the values a user has to be able to change without editing
//! environment variables or rebuilding anything. They come from three places,
//! in this order:
//!
//! 1. **Environment** (`NGU_API_KEY`, `NGU_BASE_URL`, `NGU_MODEL`) — for
//!    scripts and CI, where a machine-level setting would be wrong.
//! 2. **The settings file** — what the desktop shell's settings panel writes,
//!    and what `ngu config set` writes.
//! 3. **Defaults** — a model name and no endpoint.
//!
//! The base URL deliberately has **no default endpoint**. Guessing one is how a
//! user in a region the guessed provider does not serve ends up staring at a
//! 403 they cannot explain. An empty endpoint is reported as "not configured",
//! which is actionable; a wrong endpoint is not.

use std::path::PathBuf;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// Default model when nothing is configured.
pub const DEFAULT_MODEL: &str = "deepseek-v4.1-flash";

/// Settings as stored on disk.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Settings {
    /// API base URL including the version segment, e.g. `https://host/v1`.
    #[serde(default)]
    pub base_url: String,
    /// API key.
    #[serde(default)]
    pub api_key: String,
    /// Model id.
    #[serde(default)]
    pub model: String,
    /// Reasoning effort: `minimal`, `low`, `medium`, `high`, or empty to let
    /// the provider decide.
    ///
    /// This is the main cost lever on a reasoning model. Measured against a
    /// real endpoint, `minimal` cut one answer from 137 output tokens to 69.
    #[serde(default)]
    pub reasoning_effort: String,
    /// Proxy for all requests, e.g. `http://127.0.0.1:7890`.
    ///
    /// Empty means direct, and — more importantly — it means the process's
    /// `HTTP_PROXY`/`HTTPS_PROXY` variables are ignored. Leaving those to be
    /// picked up implicitly is how a proxy configured for some other tool
    /// silently reroutes model traffic and produces a `tls handshake eof` that
    /// mentions nothing about proxies.
    #[serde(default)]
    pub proxy: String,
    /// Context window in tokens.
    ///
    /// Empty means "work it out": the provider's own number when it reports one,
    /// a built-in table for well-known models, otherwise a conservative floor.
    /// Setting it explicitly always wins.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_window: Option<usize>,
    /// Compact once this percentage of the window is in use.
    #[serde(default = "default_compact_percent")]
    pub compact_percent: u32,
    /// Recent messages kept verbatim when compacting.
    #[serde(default = "default_keep_recent")]
    pub compact_keep_recent: usize,

    /// Extra fields merged into every request body.
    ///
    /// This is how a provider difference stays a settings change: an endpoint
    /// that wants enable_thinking, top_k, or any other field gets it here.
    /// A null value removes a field the kernel would otherwise send.
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub extra_body: serde_json::Map<String, serde_json::Value>,
}

fn default_compact_percent() -> u32 {
    75
}

fn default_keep_recent() -> usize {
    8
}

impl Settings {
    /// Whether the settings are complete enough to make a request.
    pub fn is_configured(&self) -> bool {
        !self.base_url.trim().is_empty() && !self.api_key.trim().is_empty()
    }

    /// What is missing, in the words a settings panel can show.
    pub fn missing(&self) -> Vec<&'static str> {
        let mut out = Vec::new();
        if self.base_url.trim().is_empty() {
            out.push("base_url");
        }
        if self.api_key.trim().is_empty() {
            out.push("api_key");
        }
        if self.model.trim().is_empty() {
            out.push("model");
        }
        out
    }

    /// The model to use, falling back to the default.
    pub fn model_or_default(&self) -> String {
        if self.model.trim().is_empty() {
            DEFAULT_MODEL.to_string()
        } else {
            self.model.clone()
        }
    }

    /// Build the context policy these settings describe.
    ///
    /// This is the built-in policy; a plugin that provides `context.policy`
    /// replaces it at runtime without touching these settings.
    pub fn context_policy(&self, provider_window: Option<usize>) -> crate::window::DefaultContextPolicy {
        crate::window::DefaultContextPolicy {
            configured: self.context_window.filter(|t| *t > 0),
            threshold_percent: self.compact_percent,
            keep_recent: self.compact_keep_recent.max(1),
            provider_window: provider_window.filter(|t| *t > 0),
        }
    }

    /// The key with everything but its first and last few characters hidden.
    ///
    /// The settings panel has to show *something* so a user can tell whether a
    /// key is present, without putting the whole secret on screen.
    pub fn masked_key(&self) -> String {
        let key = self.api_key.trim();
        if key.is_empty() {
            return String::new();
        }
        let chars: Vec<char> = key.chars().collect();
        if chars.len() <= 10 {
            return "*".repeat(chars.len());
        }
        let head: String = chars.iter().take(6).collect();
        let tail: String = chars.iter().skip(chars.len() - 4).collect();
        format!("{head}…{tail}")
    }

    /// Where the settings file lives.
    ///
    /// `$NGU_HOME/settings.json` when set, else `<home>/.nguruvilu/settings.json`.
    pub fn path() -> PathBuf {
        if let Ok(home) = std::env::var("NGU_HOME") {
            if !home.trim().is_empty() {
                return PathBuf::from(home).join("settings.json");
            }
        }
        home_dir().join(".nguruvilu").join("settings.json")
    }

    /// Read the settings file, or an empty set when there is none.
    pub fn load() -> Result<Self> {
        let path = Self::path();
        if !path.is_file() {
            return Ok(Self::default());
        }
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        // A settings file a user hand-edited should not brick the app: fall
        // back to defaults and let the panel show empty fields.
        Ok(serde_json::from_str(&text).unwrap_or_default())
    }

    /// Write the settings file, creating its directory.
    pub fn save(&self) -> Result<()> {
        let path = Self::path();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating {}", parent.display()))?;
        }
        let text = serde_json::to_string_pretty(self)?;
        std::fs::write(&path, format!("{text}\n"))
            .with_context(|| format!("writing {}", path.display()))?;
        Ok(())
    }

    /// The effective settings: environment overrides the file, which overrides
    /// the defaults.
    pub fn resolve() -> Self {
        let mut settings = Self::load().unwrap_or_default();

        if let Ok(value) = std::env::var("NGU_BASE_URL") {
            if !value.trim().is_empty() {
                settings.base_url = value;
            }
        } else if let Ok(value) = std::env::var("OPENAI_BASE_URL") {
            if !value.trim().is_empty() {
                settings.base_url = value;
            }
        }

        if let Ok(value) = std::env::var("NGU_API_KEY") {
            if !value.trim().is_empty() {
                settings.api_key = value;
            }
        } else if let Ok(value) = std::env::var("OPENAI_API_KEY") {
            if !value.trim().is_empty() {
                settings.api_key = value;
            }
        }

        if let Ok(value) = std::env::var("NGU_MODEL") {
            if !value.trim().is_empty() {
                settings.model = value;
            }
        }

        if let Ok(value) = std::env::var("NGU_REASONING_EFFORT") {
            if !value.trim().is_empty() {
                settings.reasoning_effort = value;
            }
        }

        // `NGU_PROXY` is read, but `HTTP_PROXY`/`HTTPS_PROXY` deliberately are
        // not: those are set for other tools and rerouting model traffic through
        // an unreachable proxy yields an error that never mentions proxies.
        if let Ok(value) = std::env::var("NGU_PROXY") {
            settings.proxy = value.trim().to_string();
        }

        if let Ok(value) = std::env::var("NGU_CONTEXT_WINDOW") {
            if let Ok(tokens) = value.trim().parse::<usize>() {
                settings.context_window = Some(tokens);
            }
        }

        if settings.model.trim().is_empty() {
            settings.model = DEFAULT_MODEL.to_string();
        }
        settings
    }
}

/// The user's home directory.
pub fn home_dir() -> PathBuf {
    if let Ok(profile) = std::env::var("USERPROFILE") {
        if !profile.trim().is_empty() {
            return PathBuf::from(profile);
        }
    }
    if let Ok(home) = std::env::var("HOME") {
        if !home.trim().is_empty() {
            return PathBuf::from(home);
        }
    }
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_empty_settings_set_is_not_configured() {
        let settings = Settings::default();
        assert!(!settings.is_configured());
        assert_eq!(settings.missing(), vec!["base_url", "api_key", "model"]);
    }

    #[test]
    fn a_complete_settings_set_is_configured() {
        let settings = Settings {
            base_url: "https://example.test/v1".into(),
            api_key: "sk-test".into(),
            model: "m".into(),
            reasoning_effort: String::new(),
            proxy: String::new(),
            context_window: None,
            compact_percent: 75,
            compact_keep_recent: 8,
            extra_body: serde_json::Map::new(),
        };
        assert!(settings.is_configured());
        assert!(settings.missing().is_empty());
    }

    #[test]
    fn a_missing_model_still_yields_a_usable_one() {
        let settings = Settings {
            base_url: "https://example.test/v1".into(),
            api_key: "sk-test".into(),
            model: String::new(),
            reasoning_effort: String::new(),
            proxy: String::new(),
            context_window: None,
            compact_percent: 75,
            compact_keep_recent: 8,
            extra_body: serde_json::Map::new(),
        };
        assert_eq!(settings.model_or_default(), DEFAULT_MODEL);
    }

    #[test]
    fn keys_are_masked_for_display() {
        let settings = Settings {
            api_key: "sk-abcdefghijklmnop".into(),
            ..Default::default()
        };
        let masked = settings.masked_key();
        assert!(masked.starts_with("sk-abc"), "{masked}");
        assert!(masked.ends_with("mnop"), "{masked}");
        assert!(!masked.contains("defghijkl"), "the middle is hidden: {masked}");
    }

    #[test]
    fn a_short_key_is_hidden_entirely() {
        let settings = Settings {
            api_key: "short".into(),
            ..Default::default()
        };
        assert_eq!(settings.masked_key(), "*****");
    }

    #[test]
    fn an_absent_key_masks_to_nothing() {
        assert_eq!(Settings::default().masked_key(), "");
    }

    #[test]
    fn settings_round_trip_through_a_file() {
        let dir = std::env::temp_dir().join(format!("ngu-settings-{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("settings.json");

        let settings = Settings {
            base_url: "https://example.test/v1".into(),
            api_key: "sk-round-trip".into(),
            model: "test-model".into(),
            reasoning_effort: "low".into(),
            proxy: String::new(),
            context_window: None,
            compact_percent: 75,
            compact_keep_recent: 8,
            extra_body: serde_json::Map::new(),
        };
        let text = serde_json::to_string_pretty(&settings).unwrap();
        std::fs::write(&path, &text).unwrap();

        let read: Settings = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(read.base_url, settings.base_url);
        assert_eq!(read.api_key, settings.api_key);
        assert_eq!(read.model, settings.model);
    }

    #[test]
    fn a_corrupt_settings_file_falls_back_to_defaults() {
        let parsed: Settings = serde_json::from_str("{ this is not json }").unwrap_or_default();
        assert!(!parsed.is_configured());
    }
}
