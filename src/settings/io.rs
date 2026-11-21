//! Reading and writing the settings file, and masking what must not be logged.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use super::model::{Settings, DEFAULT_MODEL};
use super::paths::data_dir;

/// Everything but the first and last few characters, for display.
pub(super) fn mask(secret: &str) -> String {
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
/// relay token, and the judge credentials, and `0o644` in a `0o755`
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

// The file-facing half of the same `impl`: where it lives and how it round-trips.
impl Settings {
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
        // Before anything reads a pack setting: a file written by an older build
        // keeps its values at the top level, and they belong under pack names.
        settings.migrate_legacy_pack_keys();

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

        // The judge's credentials come from the environment like everything
        // else, and land under the pack's own key: the kernel does not know what
        // a `judge` is, it only knows the pack asked for a place to keep things.
        if let Some(from_env) = crate::tools::judge::from_env() {
            if let Ok(value) = serde_json::to_value(from_env) {
                settings.set_pack("judge", value);
            }
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
