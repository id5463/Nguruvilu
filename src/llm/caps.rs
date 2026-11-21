use super::*;

pub fn refused_key(base_url: &str, effort: &str) -> String {
    format!(
        "{}\u{1}{}",
        base_url.trim().trim_end_matches('/').to_ascii_lowercase(),
        effort.trim()
    )
}

/// Whether this process has already seen this endpoint refuse this effort.
pub fn effort_already_refused(base_url: &str, effort: &str) -> bool {
    let key = refused_key(base_url, effort);
    REFUSED_EFFORTS
        .lock()
        .expect("refused table")
        .as_ref()
        .map(|seen| seen.contains(&key))
        .unwrap_or(false)
}

/// Record a refusal for the rest of the process.
// ---------------------------------------------------------------- capabilities
//
// Whether a route accepts a given shape is not something the OpenAI format
// answers: providers that claim compatibility disagree about optional fields,
// and the disagreement arrives as a 400 on the whole request rather than a note
// about one field. This file uses both ways of knowing:
//
//   * A static table for what was measured by hand (`MODEL_CAPS`).
//   * Runtime refusal for everything else (`TOOL_IMAGE_REFUSED`), the same
//     pattern `REFUSED_EFFORTS` already follows.
//
// The table wins when it has an opinion, because it was measured deliberately;
// the runtime memory fills the rest, so a route nobody tested still works after
// one wasted round trip instead of never working at all.

/// One measured capability of a model.
///
/// `None` means "not measured", which is not the same as `false` ("measured, and
/// it does not work"). A model missing from the table is unknown, not broken.
#[derive(Debug, Clone, Copy, Default)]
pub struct ModelCaps {
    /// The model reads images at all. `Some(false)` means the provider accepts
    /// the request and drops the picture silently — DeepSeek's flash routes do
    /// exactly that, which is worse than a rejection because nothing fails.
    pub vision: Option<bool>,
    /// A `role: "tool"` message may carry an `image_url` part. DeepSeek's docs
    /// say yes ("for image input"); the OpenAI wire format says no, because a
    /// tool message's content is a string. Measured per route.
    pub image_in_tool_result: Option<bool>,
}

/// Capabilities measured by hand, matched by model substring.
///
/// Substring rather than exact id: a route is usually a family
/// (`deepseek/deepseek-v4.1-flash`, `-fast`, `-ultrafast`), and the measured
/// behaviour has been the family's rather than the variant's.
pub const MODEL_CAPS: &[(&str, ModelCaps)] = &[
    // Measured: accepts the request, reports `image_tokens: 0`, answers nothing.
    // Format-legal, useless.
    (
        "deepseek-v4.1-flash",
        ModelCaps { vision: Some(false), image_in_tool_result: Some(true) },
    ),
    (
        "deepseek-v4-flash-vision-exp",
        ModelCaps { vision: Some(false), image_in_tool_result: Some(true) },
    ),
    // Measured: read a 40x40 red square correctly.
    ("mimo-v2.6-flash", ModelCaps { vision: Some(true), image_in_tool_result: Some(true) }),
    ("minimax-m3", ModelCaps { vision: Some(true), image_in_tool_result: Some(true) }),
];

/// What is known about a model, static table first.
pub fn model_caps(model: &str) -> ModelCaps {
    let lower = model.to_ascii_lowercase();
    for (needle, caps) in MODEL_CAPS {
        if lower.contains(&needle.to_ascii_lowercase()) {
            return *caps;
        }
    }
    ModelCaps::default()
}

/// Routes that rejected an image inside a tool result.
pub static TOOL_IMAGE_REFUSED: std::sync::Mutex<Option<std::collections::HashSet<String>>> =
    std::sync::Mutex::new(None);

pub fn route_key(base_url: &str, model: &str) -> String {
    format!(
        "{}\u{1}{}",
        base_url.trim().trim_end_matches('/').to_ascii_lowercase(),
        model.trim()
    )
}

pub fn remember_tool_image_refused(base_url: &str, model: &str) {
    let key = route_key(base_url, model);
    let mut table = TOOL_IMAGE_REFUSED.lock().expect("tool image table");
    table
        .get_or_insert_with(std::collections::HashSet::new)
        .insert(key);
}

/// Whether this route may carry an image inside a tool result.
///
/// The static table answers when it has an opinion; otherwise a route that has
/// not refused before gets the benefit of the doubt, because DeepSeek's shape
/// keeps the transcript clean and every other route falls back after one 400.
pub fn tool_image_allowed(base_url: &str, model: &str) -> bool {
    if let Some(measured) = model_caps(model).image_in_tool_result {
        return measured;
    }
    let key = route_key(base_url, model);
    let table = TOOL_IMAGE_REFUSED.lock().expect("tool image table");
    !table.as_ref().map(|set| set.contains(&key)).unwrap_or(false)
}

/// Whether this model reads images, when that has been measured.
///
/// `None` means unknown, and callers should try rather than refuse: guessing
/// against a capable model costs more than one wasted request.
pub fn model_reads_images(model: &str) -> Option<bool> {
    model_caps(model).vision
}

pub fn remember_refused_effort(base_url: &str, effort: &str) {
    let key = refused_key(base_url, effort);
    let mut table = REFUSED_EFFORTS.lock().expect("refused table");
    table
        .get_or_insert_with(std::collections::HashSet::new)
        .insert(key);
}

/// One entry from a provider's model catalog.
///
/// Providers report more than an id: `owned_by` says which vendor a model
/// actually belongs to (useful when a gateway fronts several), and
/// `supported_endpoint_types` says which API dialects it serves.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Model id, as passed in a request.
    pub id: String,
    /// Vendor the provider attributes it to.
    #[serde(default)]
    pub owned_by: Option<String>,
    /// Creation timestamp, when reported.
    #[serde(default)]
    pub created: Option<i64>,
    /// Endpoint families the model serves, e.g. `["openai", "anthropic"]`.
    #[serde(default, alias = "supported_endpoint_types")]
    pub endpoint_types: Vec<String>,
}

impl ModelInfo {
    /// A short label for a picker: the id, plus the vendor when it is known.
    pub fn label(&self) -> String {
        match &self.owned_by {
            Some(owner) if !owner.is_empty() && owner != "unknown" => {
                format!("{}  ({})", self.id, owner)
            }
            _ => self.id.clone(),
        }
    }
}

/// A model catalog, as reported by the provider.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelCatalog {
    /// Entries.
    #[serde(default)]
    pub data: Vec<ModelInfo>,
}
