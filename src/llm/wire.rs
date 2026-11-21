use super::*;

/// Convert one neutral message to the OpenAI wire shape.
pub fn to_wire(message: &Message) -> Value {
    to_wire_with(message, false)
}

/// Convert one message, choosing whether a tool result may carry images.
///
/// `tool_images` comes from [`tool_image_allowed`]. When false, the tool message
/// keeps its text and any images it carried are dropped here — the caller is
/// responsible for re-attaching them as a user message, which is what the agent
/// loop already does for every attachment.
pub fn to_wire_with(message: &Message, tool_images: bool) -> Value {
    let mut obj = json!({ "role": message.role.as_wire() });

    match message.role {
        crate::message::Role::Assistant if !message.tool_calls.is_empty() => {
            obj["content"] = match &message.content {
                Some(text) if !text.is_empty() => json!(text),
                _ => Value::Null,
            };
            let calls: Vec<Value> = message
                .tool_calls
                .iter()
                .map(|c| {
                    json!({
                        "id": c.id,
                        "type": "function",
                        "function": { "name": c.name, "arguments": c.arguments },
                    })
                })
                .collect();
            obj["tool_calls"] = json!(calls);
        }
        crate::message::Role::Tool => {
            // DeepSeek's docs allow an image part here ("for image input"),
            // while the OpenAI format allows only a string. Which one applies is
            // per route, so it is decided by the capability table and, when that
            // has no opinion, by whether the route refused before.
            if tool_images && !message.images.is_empty() {
                let mut parts: Vec<Value> = Vec::with_capacity(message.images.len() + 1);
                let text = message.text();
                if !text.is_empty() {
                    parts.push(json!({ "type": "text", "text": text }));
                }
                for image in &message.images {
                    let mut url = json!({ "url": image.url });
                    if let Some(detail) = &image.detail {
                        url["detail"] = json!(detail);
                    }
                    parts.push(json!({ "type": "image_url", "image_url": url }));
                }
                obj["content"] = json!(parts);
            } else {
                obj["content"] = json!(message.text());
            }
            if let Some(id) = &message.tool_call_id {
                obj["tool_call_id"] = json!(id);
            }
        }
        _ => {
            if message.images.is_empty() {
                obj["content"] = json!(message.text());
            } else {
                // An image forces the content to become an array of parts; the
                // provider reads a plain string and a part list differently.
                let mut parts: Vec<Value> = Vec::with_capacity(message.images.len() + 1);
                let text = message.text();
                if !text.is_empty() {
                    parts.push(json!({ "type": "text", "text": text }));
                }
                for image in &message.images {
                    let mut url = json!({ "url": image.url });
                    if let Some(detail) = &image.detail {
                        url["detail"] = json!(detail);
                    }
                    parts.push(json!({ "type": "image_url", "image_url": url }));
                }
                obj["content"] = json!(parts);
            }
        }
    }

    obj
}

/// The text of a `content` field, which arrives as a string or as a part list.
pub fn content_text(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::Array(parts) => {
            let mut out = String::new();
            for part in parts {
                if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                    out.push_str(text);
                }
            }
            Some(out)
        }
        _ => None,
    }
}
