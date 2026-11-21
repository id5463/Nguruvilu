//! One seam: a value a plugin may rewrite on its way through.
//!
//! [`Seam`] is named by whoever asks about it rather than being a variant per
//! seam, so a new point is not a kernel change. [`TurnContext`] and
//! [`ToolCallView`] are what the plugin is told about the turn and the call it
//! is answering for.

use serde_json::Value;

/// One seam, named by whoever asks about it.
///
/// This used to be an `enum` with a variant per seam, and that was the
/// unfreedom worth removing: a new variant is a kernel change, and a plugin
/// could only ever care about the seams the kernel had thought of. A name and
/// a JSON body means the set of seams is open — the loop asks about the ones it
/// knows, a plugin may answer about any of them, and a pack that invents a name
/// of its own is not refused.
///
/// The body type is `Value` rather than `String` so the ordinary seams stay
/// typed: `Prompt`, `Reply` and `Tools` are plain JSON values, and a plugin that
/// only cares about one of them matches on the name and ignores the rest.
pub struct Seam {
    /// What is being passed, e.g. `prompt`, `reply`, `tools`, `turn`.
    ///
    /// Lowercase, no spaces: it travels through the dylib ABI as JSON and
    /// through a page as an attribute, so a name that needs escaping would be
    /// trouble in two places for no gain.
    pub name: String,
    /// The data itself.
    pub body: Value,
}

impl Seam {
    /// A seam carrying one string — the shape of `prompt` and `reply`.
    pub fn text(name: &str, text: impl Into<String>) -> Self {
        Self {
            name: name.to_string(),
            body: Value::String(text.into()),
        }
    }

    /// A seam carrying a list of names — the shape of `tools`.
    pub fn names(name: &str, names: Vec<String>) -> Self {
        Self {
            name: name.to_string(),
            body: Value::Array(names.into_iter().map(Value::String).collect()),
        }
    }

    /// The body as a string, when it is one.
    pub fn as_text(&self) -> Option<&str> {
        self.body.as_str()
    }

    /// The body as a list of strings, when it is one.
    pub fn as_names(&self) -> Option<Vec<String>> {
        self.body.as_array().map(|items| {
            items
                .iter()
                .filter_map(|item| item.as_str().map(str::to_string))
                .collect()
        })
    }
}

/// What a plugin is told about the turn it is being asked about.
///
/// Passed by reference and read-only: a plugin answers a question, it does not
/// own the turn. Everything here is a snapshot for this one call.
pub struct TurnContext {
    /// The conversation this turn belongs to.
    pub session: String,
    /// The user's message that started it.
    pub input: String,
    /// The model in force, so a plugin can answer differently per route.
    pub model: String,
    /// Skill ids with their one-line descriptions.
    ///
    /// Here rather than behind a service lookup because every prompt-side
    /// question starts by looking at what skills exist.
    pub skills: Vec<(String, String)>,
}

/// One tool call, as a plugin sees it.
pub struct ToolCallView {
    /// Tool name.
    pub name: String,
    /// Raw JSON arguments.
    pub arguments: String,
}
