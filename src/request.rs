//! Shaping the request body: the seam for everything provider-specific.
//!
//! The kernel builds a standard OpenAI Chat Completions body. Real endpoints
//! deviate from that standard constantly — a field spelled differently, an
//! extra parameter, a capability that must be opted into — and those
//! differences are not the kernel's business to guess.
//!
//! [`RequestShaper`] is where they belong. The kernel ships a no-op and one
//! generic implementation driven by configuration, so a new provider quirk is a
//! settings change rather than a code change; a plugin can replace the shaper
//! entirely when it needs real logic.
//!
//! Shaping happens once, immediately before the body is sent, and never touches
//! the recorded conversation: what the model was asked is not what the session
//! log stores, and a shaper cannot corrupt history.

use std::sync::Arc;

use serde_json::{Map, Value};

/// Rewrites a request body just before it is sent.
pub trait RequestShaper: Send + Sync {
    /// A name for diagnostics.
    fn name(&self) -> &str;

    /// Adjust the body in place.
    ///
    /// Called with the finished body, after the kernel has set the model,
    /// messages, stream flag, tools, and sampling fields. A shaper may add,
    /// replace, or remove anything — including fields the kernel just set.
    fn shape(&self, body: &mut Value);
}

/// A shaper that changes nothing.
///
/// The default. An endpoint that follows the standard needs no adjustment, and
/// silently "fixing" a body that is already correct is how a client breaks
/// providers it has never been tested against.
#[derive(Debug, Clone, Default)]
pub struct Passthrough;

impl RequestShaper for Passthrough {
    fn name(&self) -> &str {
        "passthrough"
    }

    fn shape(&self, _body: &mut Value) {}
}

/// A shaper that merges configured fields into every request body.
///
/// This is what makes provider differences a configuration matter: an endpoint
/// that wants `enable_thinking`, `thinking_budget`, `top_k`, or a vendor header
/// carried in the body gets it from settings.
///
/// Values are merged at the top level, so a configured key **replaces** whatever
/// the kernel set. That is deliberate — a user who writes
/// `"max_completion_tokens": 8192` means it, even though the kernel already
/// chose a value from the same settings.
#[derive(Debug, Clone, Default)]
pub struct ExtraFields {
    /// Fields to merge into every request body.
    pub fields: Map<String, Value>,
}

impl ExtraFields {
    /// Build from a map of extra fields.
    pub fn new(fields: Map<String, Value>) -> Self {
        Self { fields }
    }

    /// Whether there is anything to apply.
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }
}

impl RequestShaper for ExtraFields {
    fn name(&self) -> &str {
        "extra-fields"
    }

    fn shape(&self, body: &mut Value) {
        let Some(object) = body.as_object_mut() else {
            return;
        };
        for (key, value) in &self.fields {
            // `null` removes a field, which is the only way to say "never send
            // this" without also saying "send it as null".
            if value.is_null() {
                object.remove(key);
            } else {
                object.insert(key.clone(), value.clone());
            }
        }
    }
}

/// Compose several shapers, applied in order.
pub struct Chained {
    /// Shapers, applied front to back.
    pub shapers: Vec<Arc<dyn RequestShaper>>,
}

impl Chained {
    /// Build from a list.
    pub fn new(shapers: Vec<Arc<dyn RequestShaper>>) -> Self {
        Self { shapers }
    }
}

impl RequestShaper for Chained {
    fn name(&self) -> &str {
        "chained"
    }

    fn shape(&self, body: &mut Value) {
        for shaper in &self.shapers {
            shaper.shape(body);
        }
    }
}

/// The shaper a set of extra fields describes, or the no-op when there are none.
pub fn from_extra_fields(fields: Map<String, Value>) -> Arc<dyn RequestShaper> {
    let extra = ExtraFields::new(fields);
    if extra.is_empty() {
        Arc::new(Passthrough)
    } else {
        Arc::new(extra)
    }
}

