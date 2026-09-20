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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn passthrough_changes_nothing() {
        let mut body = json!({ "model": "m", "messages": [] });
        let before = body.clone();
        Passthrough.shape(&mut body);
        assert_eq!(body, before);
        assert_eq!(Passthrough.name(), "passthrough");
    }

    #[test]
    fn extra_fields_are_merged_in() {
        let mut fields = Map::new();
        fields.insert("enable_thinking".into(), json!(true));
        fields.insert("top_k".into(), json!(40));

        let mut body = json!({ "model": "m" });
        ExtraFields::new(fields).shape(&mut body);

        assert_eq!(body["enable_thinking"], true);
        assert_eq!(body["top_k"], 40);
        assert_eq!(body["model"], "m", "existing fields survive");
    }

    #[test]
    fn a_configured_field_replaces_what_the_kernel_set() {
        // A user who writes a value means it, even if the kernel chose one.
        let mut fields = Map::new();
        fields.insert("max_completion_tokens".into(), json!(8192));

        let mut body = json!({ "max_completion_tokens": 512 });
        ExtraFields::new(fields).shape(&mut body);
        assert_eq!(body["max_completion_tokens"], 8192);
    }

    #[test]
    fn a_null_value_removes_a_field() {
        let mut fields = Map::new();
        fields.insert("tool_choice".into(), Value::Null);

        let mut body = json!({ "model": "m", "tool_choice": "auto" });
        ExtraFields::new(fields).shape(&mut body);

        assert!(body.get("tool_choice").is_none(), "removed, not set to null");
    }

    #[test]
    fn a_chain_applies_in_order() {
        let mut first = Map::new();
        first.insert("a".into(), json!(1));
        let mut second = Map::new();
        second.insert("a".into(), json!(2));
        second.insert("b".into(), json!(3));

        let chained = Chained::new(vec![
            Arc::new(ExtraFields::new(first)),
            Arc::new(ExtraFields::new(second)),
        ]);

        let mut body = json!({});
        chained.shape(&mut body);
        assert_eq!(body["a"], 2, "later shapers win");
        assert_eq!(body["b"], 3);
    }

    #[test]
    fn an_empty_field_set_yields_the_no_op() {
        assert_eq!(from_extra_fields(Map::new()).name(), "passthrough");
    }

    #[test]
    fn a_populated_field_set_yields_the_extra_fields_shaper() {
        let mut fields = Map::new();
        fields.insert("x".into(), json!(1));
        assert_eq!(from_extra_fields(fields).name(), "extra-fields");
    }

    #[test]
    fn shaping_a_non_object_body_is_ignored() {
        let mut fields = Map::new();
        fields.insert("x".into(), json!(1));
        let mut body = json!("not an object");
        ExtraFields::new(fields).shape(&mut body);
        assert_eq!(body, json!("not an object"));
    }
}
