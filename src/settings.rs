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

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// Default model when nothing is configured.
pub const DEFAULT_MODEL: &str = "deepseek-v4.1-flash";

/// Settings as stored on disk.
///
/// Unknown keys are **kept**, not rejected. `deny_unknown_fields` used to make
/// one unrecognised key — a hand-typed one, a plugin's, or one written by a
/// newer build — fail the whole parse, and [`Settings::load`] answered a failed
/// parse with [`Settings::default`]. The next `ngu config set` then wrote that
/// default back over the file, so a single typo cost the user their `api_key`,
/// model, network tuning, search and judge credentials at once. The unknown keys
/// land in [`Settings::unknown`] and are written back out unchanged.
#[derive(Clone, Default, Serialize, Deserialize)]
#[serde(default)]
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
    /// How the answer is delivered: `stream`, `buffered`, or empty for `stream`.
    ///
    /// `buffered` exists for gateways that accept a streaming request, open the
    /// response, and then never write the answer — see the module docs in
    /// [`crate::llm`]. On such an endpoint the only alternative is a turn that
    /// hangs until the whole-request timeout.
    #[serde(default)]
    pub delivery: String,
    /// Whether a stalled stream switches this route to buffered delivery.
    ///
    /// Absent means yes: one stalled step is enough evidence to stop paying the
    /// stall bound on every later step. Set to `false` to keep asking the stream
    /// regardless.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub degrade_on_stall: Option<bool>,
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
    /// Most model steps one turn may take before the loop stops and reports.
    ///
    /// Defaults to the loop's own 50 rather than to `0`: a file that never
    /// mentions this key must not describe a turn that is not allowed to run.
    #[serde(default = "default_max_steps")]
    pub max_steps: u32,

    /// Cloud relay URL the local app dials out to for remote control.
    ///
    /// Empty leaves remote control off. No default host is assumed.
    #[serde(default)]
    pub relay_url: String,
    /// Bearer token presented to the relay. Empty leaves remote control off.
    #[serde(default)]
    pub relay_token: String,

    /// Output token ceiling, when the provider accepts one.
    ///
    /// Sent as max_completion_tokens. Reasoning tokens count against it, so
    /// a value that looks generous for the answer can still be tight for a
    /// model that thinks first.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<usize>,

    /// How requests are sent: timeouts, pooling, retries.
    ///
    /// A pack's `models.json` may set these, and a plugin publishing the
    /// `network` service overrides both.
    #[serde(default)]
    pub network: crate::network::NetworkSettings,

    /// Web search, when configured.
    ///
    /// Absent means no web_search tool is registered at all — a tool that
    /// always fails is worse than one that is not there.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub search: Option<crate::tools::search::SearchSettings>,

    /// The decision judge (Jev / TypeSafe System One), when configured.
    ///
    /// Absent means no `judge` tool — the same rule as search: a tool that
    /// cannot authenticate costs a turn every time the model tries it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub judge: Option<crate::tools::judge::JudgeSettings>,

    /// Extra fields merged into every request body.
    ///
    /// This is how a provider difference stays a settings change: an endpoint
    /// that wants enable_thinking, top_k, or any other field gets it here.
    /// A null value removes a field the kernel would otherwise send.
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub extra_body: serde_json::Map<String, serde_json::Value>,

    /// Keys this build does not know about, kept verbatim.
    ///
    /// Round-trip rather than validation: a field written by a newer build, a
    /// plugin, or a hand edit must survive being read and written by an older
    /// one. This is also what replaces `deny_unknown_fields` — an unknown key is
    /// stored here instead of being a parse error.
    #[serde(flatten, skip_serializing_if = "BTreeMap::is_empty")]
    pub unknown: BTreeMap<String, serde_json::Value>,
}

impl std::fmt::Debug for Settings {
    /// Written by hand so a secret never reaches a log.
    ///
    /// `#[derive(Debug)]` would print `api_key` and `relay_token` in full, and
    /// the first `eprintln!("{settings:?}")` added while chasing a route problem
    /// would put both into a terminal scrollback and every bug report pasted
    /// from it. Everything a diagnostic actually needs is here; the credentials
    /// are not. Nested settings print as presence only, so this impl never
    /// depends on how those types choose to render themselves.
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Settings")
            .field("base_url", &self.base_url)
            .field("api_key", &mask(&self.api_key))
            .field("model", &self.model)
            .field("reasoning_effort", &self.reasoning_effort)
            .field("delivery", &self.delivery)
            .field("degrade_on_stall", &self.degrade_on_stall)
            .field("proxy", &self.proxy)
            .field("context_window", &self.context_window)
            .field("compact_percent", &self.compact_percent)
            .field("compact_keep_recent", &self.compact_keep_recent)
            .field("max_steps", &self.max_steps)
            .field("relay_url", &self.relay_url)
            .field("relay_token", &mask(&self.relay_token))
            .field("max_output_tokens", &self.max_output_tokens)
            .field("search", &self.search.is_some())
            .field("judge", &self.judge.is_some())
            .field("extra_body", &self.extra_body.keys().collect::<Vec<_>>())
            .field("unknown", &self.unknown.keys().collect::<Vec<_>>())
            .finish()
    }
}

/// Everything but the first and last few characters, for display.
fn mask(secret: &str) -> String {
    let secret = secret.trim();
    if secret.is_empty() {
        return String::new();
    }
    let characters: Vec<char> = secret.chars().collect();
    if characters.len() <= 10 {
        return "*".repeat(characters.len());
    }
    let head: String = characters.iter().take(6).collect();
    let tail: String = characters.iter().skip(characters.len() - 4).collect();
    format!("{head}…{tail}")
}

/// Write a file that may hold credentials.
///
/// On Unix the file is created `0o600`: `settings.json` carries the API key, the
/// relay token, and the search/judge credentials, and `0o644` in a `0o755`
/// directory means every other local user can read them. `sync_all` is called
/// before the rename so a crash just after it cannot leave the renamed file
/// empty.
fn write_private(path: &Path, text: &str) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::io::Write;
        use std::os::unix::fs::OpenOptionsExt;

        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600)
            .open(path)?;
        file.write_all(text.as_bytes())?;
        file.sync_all()
    }
    #[cfg(not(unix))]
    {
        std::fs::write(path, text)
    }
}

fn default_compact_percent() -> u32 {
    75
}

fn default_keep_recent() -> usize {
    8
}

fn default_max_steps() -> u32 {
    50
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
            // The threshold is clamped here as well as in the policy: a
            // hand-edited `600` would otherwise mean "never compact" (and a `0`
            // "compact on every step"), and both are written by hand in a file
            // this program does not prompt for.
            threshold_percent: self.compact_percent.clamp(1, 95),
            keep_recent: self.compact_keep_recent.max(1),
            provider_window: provider_window.filter(|t| *t > 0),
            // The output ceiling is part of the window: a prompt that fills the
            // whole window leaves no room for the answer, and the provider
            // rejects the request instead of compacting.
            reserve: self.max_output_tokens.unwrap_or(0),
        }
    }

    /// The step cap one turn may use, in the range the loop can honour.
    ///
    /// `0` reads as "not set" and means the default, not "no steps": this type
    /// derives `Default`, so every numeric field starts at zero and the field
    /// defaults only exist inside serde — a fresh install with no file would
    /// otherwise get a cap of one step, i.e. a turn that cannot finish a single
    /// tool round. Above the ceiling the value is clamped, because a turn bills
    /// for every step it takes and the loop's continuation behaviour already
    /// multiplies whatever is asked for here.
    pub fn max_steps(&self) -> usize {
        match self.max_steps {
            0 => default_max_steps() as usize,
            steps => (steps as usize).min(200),
        }
    }

    /// The key with everything but its first and last few characters hidden.
    ///
    /// The settings panel has to show *something* so a user can tell whether a
    /// key is present, without putting the whole secret on screen.
    pub fn masked_key(&self) -> String {
        mask(&self.api_key)
    }

    /// Where the settings file lives.
    ///
    /// `$NGU_HOME/settings.json` when set, else `<home>/.nguruvilu/settings.json`.
    pub fn path() -> PathBuf {
        data_dir().join("settings.json")
    }

    /// Read the settings file.
    ///
    /// Three outcomes, and the caller is meant to tell them apart:
    ///
    /// * no file → defaults (a first run, not a failure);
    /// * a file that parses → those settings, including keys this build does not
    ///   know (they are kept in [`Settings::unknown`] and written back);
    /// * a file that does **not** parse → an error naming the file and the
    ///   serde position. It is never answered with defaults, because the caller
    ///   of a load is usually about to [`Settings::save`], and saving the
    ///   defaults is how one broken key destroys a whole working configuration.
    pub fn load() -> Result<Self> {
        Self::load_from(&Self::path())
    }

    /// [`Settings::load`] against an explicit path.
    pub fn load_from(path: &Path) -> Result<Self> {
        if !path.is_file() {
            return Ok(Self::default());
        }
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading {}", path.display()))?;
        serde_json::from_str(&text).with_context(|| {
            format!(
                "{} is not valid settings; it was left untouched — fix it or move it aside \
                 (nothing is written over an unreadable file)",
                path.display()
            )
        })
    }

    /// Write the settings file, creating its directory.
    pub fn save(&self) -> Result<()> {
        self.save_to(&Self::path())
    }

    /// [`Settings::save`] against an explicit path.
    ///
    /// Refuses to replace a file that does not parse as settings. That check is
    /// the real protection behind [`Settings::load`]: a caller that ignored the
    /// load error, or built its [`Settings`] from defaults, cannot then flatten
    /// the user's file — the write fails and the bytes stay as they were.
    ///
    /// The write itself is atomic (temp file, then rename) and leaves one
    /// `settings.json.bak` from before the first overwrite, so a
    /// half-written file is not a state this program can produce and a good
    /// configuration that was once on disk stays recoverable.
    pub fn save_to(&self, path: &Path) -> Result<()> {
        // A settings file is where a clear-text endpoint gets chosen, and a
        // hand-edit or a restored profile is where one usually comes from, so the
        // route is checked before it can be written rather than after it has been
        // used (see `crate::model::check_base_url`).
        crate::model::check_base_url(&self.base_url)?;

        if path.is_file() {
            let existing = std::fs::read_to_string(path)
                .with_context(|| format!("reading {}", path.display()))?;
            if let Err(error) = serde_json::from_str::<Self>(&existing) {
                return Err(anyhow::anyhow!(
                    "refusing to overwrite {}: it is not valid settings ({error}). \
                     Fix it or move it aside first — nothing was written, so the file is \
                     still exactly as it was",
                    path.display()
                ));
            }
        }

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating {}", parent.display()))?;
        }

        // One backup, taken before the first overwrite and kept from then on:
        // the second save must not overwrite the copy of the file as the user
        // last had it working.
        let backup = path.with_extension("json.bak");
        if path.is_file() && !backup.is_file() {
            std::fs::copy(path, &backup).with_context(|| {
                format!("backing up {} to {}", path.display(), backup.display())
            })?;
        }

        let text = format!("{}\n", serde_json::to_string_pretty(self)?);
        let temporary = path.with_extension("json.tmp");
        write_private(&temporary, &text)
            .with_context(|| format!("writing {}", temporary.display()))?;
        std::fs::rename(&temporary, path)
            .with_context(|| format!("replacing {}", path.display()))?;
        Ok(())
    }

    /// The effective settings: environment overrides the file, which overrides
    /// the defaults.
    pub fn resolve() -> Self {
        let mut settings = Self::load().unwrap_or_else(|error| {
            // A read path may carry on with defaults, but not quietly: these two
            // lines are the only thing that explains why every field suddenly
            // reads as empty, and they keep the blame from landing on the
            // provider. Writes stay refused (see `save_to`), so this cannot
            // turn into data loss.
            eprintln!("[settings] {error:#}");
            eprintln!("[settings] continuing with defaults; the file on disk is untouched");
            Self::default()
        });

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

        if let Ok(value) = std::env::var("NGU_DELIVERY") {
            if !value.trim().is_empty() {
                settings.delivery = value;
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

        if let Some(from_env) = crate::tools::search::from_env() {
            settings.search = Some(from_env);
        }

        if let Some(from_env) = crate::tools::judge::from_env() {
            settings.judge = Some(from_env);
        }

        if let Ok(value) = std::env::var("NGU_MAX_OUTPUT_TOKENS") {
            if let Ok(tokens) = crate::size::parse_size(&value) {
                settings.max_output_tokens = Some(tokens);
            }
        }

        if let Ok(value) = std::env::var("NGU_CONTEXT_WINDOW") {
            // Accepts 128K, 1M, or a plain number.
            if let Ok(tokens) = crate::size::parse_size(&value) {
                settings.context_window = Some(tokens);
            }
        }

        if settings.model.trim().is_empty() {
            settings.model = DEFAULT_MODEL.to_string();
        }
        settings
    }
}

/// Where this program keeps everything: settings, sessions, packs, cache.
///
/// `$NGU_HOME` when set, else `<home>/.nguruvilu`.
///
/// Separate from [`home_dir`] on purpose. That one is the *user's* home, which
/// belongs to the operating system; this one is ours. Conflating them puts a
/// `packs/` directory in the user's home folder, which is what happened before
/// this existed.
pub fn data_dir() -> PathBuf {
    if let Ok(home) = std::env::var("NGU_HOME") {
        if !home.trim().is_empty() {
            return PathBuf::from(home);
        }
    }
    home_dir().join(".nguruvilu")
}

/// The user's home directory.
pub fn home_dir() -> PathBuf {    if let Ok(profile) = std::env::var("USERPROFILE") {
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
            delivery: String::new(),
            degrade_on_stall: None,
            proxy: String::new(),
            context_window: None,
            compact_percent: 75,
            compact_keep_recent: 8,
            max_steps: 50,
            relay_url: String::new(),
            relay_token: String::new(),
            max_output_tokens: None,
            network: crate::network::NetworkSettings::default(),
            search: None,
            judge: None,
            extra_body: serde_json::Map::new(),
            ..Default::default()
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
            delivery: String::new(),
            degrade_on_stall: None,
            proxy: String::new(),
            context_window: None,
            compact_percent: 75,
            compact_keep_recent: 8,
            max_steps: 50,
            relay_url: String::new(),
            relay_token: String::new(),
            max_output_tokens: None,
            network: crate::network::NetworkSettings::default(),
            search: None,
            judge: None,
            extra_body: serde_json::Map::new(),
            ..Default::default()
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

    /// A scratch directory plus the settings path inside it.
    fn scratch(tag: &str) -> (PathBuf, PathBuf) {
        let dir = std::env::temp_dir().join(format!(
            "ngu-settings-{tag}-{}",
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("settings.json");
        (dir, path)
    }

    fn configured() -> Settings {
        Settings {
            base_url: "https://example.test/v1".into(),
            api_key: "sk-round-trip".into(),
            model: "test-model".into(),
            reasoning_effort: "low".into(),
            delivery: "buffered".into(),
            degrade_on_stall: Some(false),
            max_output_tokens: Some(4096),
            ..Default::default()
        }
    }

    #[test]
    fn settings_round_trip_through_a_file() {
        let (_dir, path) = scratch("round-trip");
        let settings = configured();
        settings.save_to(&path).unwrap();

        let read = Settings::load_from(&path).unwrap();
        assert_eq!(read.base_url, settings.base_url);
        assert_eq!(read.api_key, settings.api_key);
        assert_eq!(read.model, settings.model);
        assert_eq!(read.delivery, settings.delivery);
        assert_eq!(read.max_output_tokens, settings.max_output_tokens);
    }

    #[test]
    fn an_unknown_key_survives_a_load_and_save_round_trip() {
        // The defect this pins down: `deny_unknown_fields` turned one
        // unrecognised key into a failed parse, `load()` answered that with
        // defaults, and the next save wrote the defaults over everything.
        let (_dir, path) = scratch("unknown-key");
        let original = serde_json::json!({
            "base_url": "https://example.test/v1",
            "api_key": "sk-keep-me",
            "model": "test-model",
            "a_field_from_a_newer_build": {"nested": [1, 2, 3]},
            "another_unknown": "kept",
        });
        std::fs::write(&path, serde_json::to_string_pretty(&original).unwrap()).unwrap();

        let mut settings = Settings::load_from(&path).unwrap();
        assert_eq!(settings.api_key, "sk-keep-me", "the known fields load");
        assert_eq!(settings.unknown.len(), 2, "{:?}", settings.unknown);

        settings.model = "another-model".into();
        settings.save_to(&path).unwrap();

        let written: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(written["model"], "another-model");
        assert_eq!(
            written["a_field_from_a_newer_build"],
            serde_json::json!({"nested": [1, 2, 3]}),
            "an unknown key is written back, not eaten"
        );
        assert_eq!(written["another_unknown"], "kept");
        assert_eq!(written["api_key"], "sk-keep-me", "and nothing else moved");
    }

    #[test]
    fn a_corrupt_file_is_an_error_rather_than_defaults() {
        let (_dir, path) = scratch("corrupt-load");
        std::fs::write(&path, "{ this is not json }").unwrap();

        let error = Settings::load_from(&path).expect_err("a broken file must not read as defaults");
        let text = format!("{error:#}");
        assert!(text.contains("settings.json"), "the path is named: {text}");
        assert!(text.contains("line"), "the serde position is named: {text}");
    }

    #[test]
    fn save_refuses_to_overwrite_a_file_that_does_not_parse() {
        // The core of the fix: a caller that ignored the load error (or built
        // its settings from defaults) must not be able to flatten the file.
        let (_dir, path) = scratch("corrupt-save");
        let original = "{\"api_key\": \"sk-still-here\", oops";
        std::fs::write(&path, original).unwrap();

        let error = Settings::default()
            .save_to(&path)
            .expect_err("an unreadable file must not be overwritten");
        assert!(
            format!("{error:#}").contains("nothing was written"),
            "{error:#}"
        );
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            original,
            "the bytes are exactly as they were"
        );
        assert!(
            !path.with_extension("json.bak").exists(),
            "and no backup was invented for a file that was never read"
        );
    }

    #[test]
    fn save_leaves_one_backup_from_before_the_first_overwrite() {
        let (_dir, path) = scratch("backup");
        let mut settings = configured();
        settings.save_to(&path).unwrap();
        assert!(
            !path.with_extension("json.bak").exists(),
            "the first write creates the file, nothing to back up"
        );

        let first = std::fs::read_to_string(&path).unwrap();
        settings.model = "second".into();
        settings.save_to(&path).unwrap();
        settings.model = "third".into();
        settings.save_to(&path).unwrap();

        let backup = path.with_extension("json.bak");
        assert_eq!(
            std::fs::read_to_string(&backup).unwrap(),
            first,
            "the backup is the file as it was before the first overwrite"
        );
        assert_eq!(
            Settings::load_from(&path).unwrap().model,
            "third",
            "and the live file is the newest write"
        );
    }

    #[test]
    fn setting_one_field_leaves_every_other_field_alone() {
        // The shape of `ngu config set`: load, change one field, save. Every
        // field the command does not touch has to come back identical.
        let (_dir, path) = scratch("set-one");
        let before = configured();
        before.save_to(&path).unwrap();

        let mut after = Settings::load_from(&path).unwrap();
        after.model = "switched-model".into();
        after.save_to(&path).unwrap();

        let reloaded = Settings::load_from(&path).unwrap();
        assert_eq!(reloaded.model, "switched-model");
        assert_eq!(reloaded.base_url, before.base_url);
        assert_eq!(reloaded.api_key, before.api_key, "the key survives a set");
        assert_eq!(reloaded.reasoning_effort, before.reasoning_effort);
        assert_eq!(reloaded.delivery, before.delivery);
        assert_eq!(reloaded.degrade_on_stall, before.degrade_on_stall);
        assert_eq!(reloaded.max_output_tokens, before.max_output_tokens);
        assert_eq!(reloaded.compact_percent, before.compact_percent);
        assert_eq!(reloaded.compact_keep_recent, before.compact_keep_recent);
        assert_eq!(reloaded.max_steps, before.max_steps, "a set keeps the step cap");
        assert_eq!(reloaded.network.retry_attempts, before.network.retry_attempts);
    }

    #[test]
    fn the_debug_output_masks_the_secret() {
        let settings = Settings {
            api_key: "sk-abcdefghijklmnop".into(),
            relay_token: "relay-0123456789abcdef".into(),
            ..Default::default()
        };
        let printed = format!("{settings:?}");
        assert!(!printed.contains("sk-abcdefghijklmnop"), "{printed}");
        assert!(!printed.contains("relay-0123456789abcdef"), "{printed}");
        assert!(printed.contains("sk-abc"), "enough to identify it: {printed}");
    }

    #[test]
    fn the_threshold_and_the_output_reserve_reach_the_policy() {
        let settings = Settings {
            compact_percent: 500,
            max_output_tokens: Some(8192),
            ..Default::default()
        };
        let policy = settings.context_policy(None);
        assert_eq!(policy.threshold_percent, 95, "100% would mean never compact");
        assert_eq!(policy.reserve, 8192, "the answer needs room inside the window");
    }

    #[test]
    fn the_step_cap_defaults_to_fifty() {
        // The product decision, in one line: a file that says nothing about
        // steps behaves exactly as the desktop always has.
        assert_eq!(Settings::default().max_steps(), 50);
    }

    #[test]
    fn the_step_cap_is_read_from_the_file() {
        let (_dir, path) = scratch("step-cap");
        std::fs::write(&path, r#"{"model": "m", "max_steps": 12}"#).unwrap();
        let settings = Settings::load_from(&path).unwrap();
        assert_eq!(settings.max_steps, 12);
        assert_eq!(settings.max_steps(), 12);
    }

    #[test]
    fn an_absurd_step_cap_is_clamped_rather_than_trusted() {
        // 0 reads as "not set", not as "no steps": `Settings` derives `Default`,
        // so a build with no settings file has zero here and would otherwise
        // start every turn with a cap of one.
        assert_eq!(
            Settings { max_steps: 0, ..Default::default() }.max_steps(),
            50
        );
        // The other end is a runaway turn: a file edited to a huge number is
        // cheap to write and expensive to watch, so the ceiling wins.
        assert_eq!(
            Settings { max_steps: 100_000, ..Default::default() }.max_steps(),
            200
        );
    }
}
