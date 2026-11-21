//! Named API profiles: more than one provider, switched without overwriting.
//!
//! The live settings hold exactly one endpoint, key, and model — which is what
//! a turn routes through. Profiles are the bookmark layer above that: the
//! current values saved under a name, recallable at any time. A switch rewrites
//! the live settings from the profile and saves them, so every consumer of the
//! settings (the panel, a pack's recompute baseline, a subagent's route) sees
//! one consistent view — the same path `save_settings` takes, never a parallel
//! truth.
//!
//! The store is its own file (`routes.json`) rather than new fields in
//! `settings.json` on purpose: an older binary reading a settings file it does
//! not fully recognize falls back to defaults, and a profile list parked there
//! would then silently erase the user's API key on the next save. A separate
//! file keeps the compatibility surface exactly where it is added.

use std::path::PathBuf;

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};

use crate::settings::Settings;

/// One saved API line: everything that says *where the request goes*.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct RouteProfile {
    /// User-visible name, also the key it is switched by.
    pub name: String,
    #[serde(default)]
    pub base_url: String,
    #[serde(default)]
    pub api_key: String,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub proxy: String,
    #[serde(default)]
    pub reasoning_effort: String,
    #[serde(default)]
    pub delivery: String,
    #[serde(default)]
    pub max_output_tokens: Option<usize>,
}

/// The saved profiles and which one is currently applied.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Routes {
    #[serde(default)]
    pub active: String,
    #[serde(default)]
    pub profiles: Vec<RouteProfile>,
}

impl Routes {
    /// Where the profile list lives.
    pub fn path() -> PathBuf {
        crate::settings::data_dir().join("routes.json")
    }

    /// Read the file; absent or unreadable means "no profiles yet", never an
    /// error that stops a conversation from starting.
    pub fn load() -> Self {
        let path = Self::path();
        let Ok(text) = std::fs::read_to_string(&path) else {
            return Self::default();
        };
        serde_json::from_str(&text).unwrap_or_default()
    }

    /// Write the file, creating its directory.
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

    /// The profile with this name, if it exists.
    pub fn find(&self, name: &str) -> Option<&RouteProfile> {
        self.profiles.iter().find(|profile| profile.name == name)
    }

    /// Replace the profile of the same name, or add it. Keeps list order for
    /// an existing name so the panel does not shuffle on every save.
    pub fn upsert(&mut self, profile: RouteProfile) {
        match self.profiles.iter_mut().find(|p| p.name == profile.name) {
            Some(existing) => *existing = profile,
            None => self.profiles.push(profile),
        }
    }

    /// Remove a profile. Removing the active one leaves the live settings
    /// untouched — the user is deleting a bookmark, not their running line.
    pub fn remove(&mut self, name: &str) -> Result<RouteProfile> {
        let at = self
            .profiles
            .iter()
            .position(|p| p.name == name)
            .ok_or_else(|| anyhow!("no saved profile named '{name}'"))?;
        if self.active == name {
            self.active.clear();
        }
        Ok(self.profiles.remove(at))
    }

    /// Snapshot the live settings as a profile under `name`.
    pub fn capture(settings: &Settings, name: &str) -> RouteProfile {
        RouteProfile {
            name: name.trim().to_string(),
            base_url: settings.base_url.trim().to_string(),
            api_key: settings.api_key.clone(),
            model: settings.model.trim().to_string(),
            proxy: settings.proxy.trim().to_string(),
            reasoning_effort: settings.reasoning_effort.trim().to_string(),
            delivery: settings.delivery.trim().to_string(),
            max_output_tokens: settings.max_output_tokens,
        }
    }

    /// Copy a profile onto the live settings. The caller saves the settings and
    /// pushes them to the kernel — this only moves the values.
    pub fn apply(&self, name: &str, settings: &mut Settings) -> Result<()> {
        let profile = self
            .find(name)
            .ok_or_else(|| anyhow!("no saved profile named '{name}'"))?;
        settings.base_url = profile.base_url.clone();
        settings.api_key = profile.api_key.clone();
        settings.model = profile.model.clone();
        settings.proxy = profile.proxy.clone();
        settings.reasoning_effort = profile.reasoning_effort.clone();
        settings.delivery = profile.delivery.clone();
        settings.max_output_tokens = profile.max_output_tokens;
        Ok(())
    }
}

