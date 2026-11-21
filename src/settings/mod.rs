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

mod io;
mod model;
mod paths;

pub use model::{Settings, ToolOutputSettings, DEFAULT_MODEL};
pub use paths::{data_dir, home_dir};
