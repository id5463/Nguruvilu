//! Where this program keeps its files.

use std::path::PathBuf;

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
