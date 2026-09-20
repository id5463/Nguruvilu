//! Neutral message format.
//!
//! Messages are stored and passed around in this provider-independent form.
//! Conversion to a provider's wire format happens only at the request
//! boundary (`llm::to_wire`), which is what makes switching models mid-session
//! possible: history never binds to one provider's shape.

use serde::{Deserialize, Serialize};

/// Who produced a message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// System instruction. May appear in the middle of history for in-history
    /// prompt updates that keep the cached prefix stable.
    System,
    /// User input.
    User,
    /// Model output, possibly carrying tool calls.
    Assistant,
    /// Result of a tool call.
    Tool,
}

impl Role {
    /// Wire spelling used by OpenAI-compatible APIs.
    pub fn as_wire(self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        }
    }
}

/// A tool call requested by the model.
///
/// `arguments` stays a raw JSON string because that is what the wire format
/// carries and what streaming delivers in fragments; it is parsed only when
/// the call is executed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Provider-assigned call id, echoed back with the result.
    pub id: String,
    /// Tool name.
    pub name: String,
    /// Raw JSON arguments as a string.
    pub arguments: String,
}

/// One message in a conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// Who produced it.
    pub role: Role,
    /// Text content. Absent for an assistant message that only calls tools.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// Tool calls requested by an assistant message.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    /// For `Role::Tool`, the id of the call being answered.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Optional display name (tool name, participant name).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Images carried by this message.
    ///
    /// Only a user message may carry them: the OpenAI format restricts a tool
    /// message's content to text, so an image a tool produced travels in a user
    /// message placed after the tool results.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub images: Vec<ImageAttachment>,
}

/// An image attached to a message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageAttachment {
    /// A URL, or a `data:` URI holding the bytes.
    pub url: String,
    /// Detail hint the provider understands: `low`, `high`, `auto`, `original`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
    /// What the image is, for the model and for a transcript reader. Not sent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}

impl ImageAttachment {
    /// Attach an image by URL or data URI.
    pub fn url(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            detail: None,
            label: None,
        }
    }

    /// Describe what the image is.
    pub fn labelled(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Set the detail hint.
    pub fn with_detail(mut self, detail: impl Into<String>) -> Self {
        self.detail = Some(detail.into());
        self
    }
}

/// Image formats the kernel will attach.
///
/// Checked by extension and by magic bytes, because a file named `.png` that is
/// not one should be reported rather than sent to a provider as a broken image.
pub fn image_mime(bytes: &[u8], path: &str) -> Option<&'static str> {
    // Magic bytes win: they describe what the file actually is.
    if bytes.starts_with(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]) {
        return Some("image/png");
    }
    if bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
        return Some("image/jpeg");
    }
    if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
        return Some("image/gif");
    }
    if bytes.len() > 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
        return Some("image/webp");
    }
    if bytes.starts_with(b"BM") {
        return Some("image/bmp");
    }

    // Fall back to the extension for formats without a reliable signature.
    let lowered = path.to_ascii_lowercase();
    if lowered.ends_with(".svg") {
        return Some("image/svg+xml");
    }
    None
}

/// Build a `data:` URI for an image, so a local file can be sent to a provider.
pub fn data_uri(mime: &str, bytes: &[u8]) -> String {
    use base64::Engine as _;
    let encoded = base64::engine::general_purpose::STANDARD.encode(bytes);
    format!("data:{mime};base64,{encoded}")
}

impl Message {
    /// A message of the given role carrying only text.
    ///
    /// Used for injected fragments, whose role is a placement decision rather
    /// than a statement about who said something.
    pub fn of_role(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: Some(content.into()),
            tool_calls: Vec::new(),
            tool_call_id: None,
            name: None,
            images: Vec::new(),
        }
    }

    /// A user message.
    pub fn user(text: impl Into<String>) -> Self {
        Self { role: Role::User, content: Some(text.into()), tool_calls: Vec::new(), tool_call_id: None, name: None, images: Vec::new() }
    }

    /// A user message carrying images.
    ///
    /// This is how an image reaches the model: the OpenAI format restricts a
    /// tool message to text, so a tool's images travel here, after its results.
    pub fn user_with_images(text: impl Into<String>, images: Vec<ImageAttachment>) -> Self {
        Self { role: Role::User, content: Some(text.into()), tool_calls: Vec::new(), tool_call_id: None, name: None, images }
    }

    /// A system message.
    pub fn system(text: impl Into<String>) -> Self {
        Self { role: Role::System, content: Some(text.into()), tool_calls: Vec::new(), tool_call_id: None, name: None, images: Vec::new() }
    }

    /// An assistant message with text only.
    pub fn assistant(text: impl Into<String>) -> Self {
        Self { role: Role::Assistant, content: Some(text.into()), tool_calls: Vec::new(), tool_call_id: None, name: None, images: Vec::new() }
    }

    /// An assistant message that requests tool calls.
    pub fn assistant_tools(calls: Vec<ToolCall>, text: Option<String>) -> Self {
        Self { role: Role::Assistant, content: text, tool_calls: calls, tool_call_id: None, name: None, images: Vec::new() }
    }

    /// A tool result message.
    pub fn tool_result(call_id: impl Into<String>, name: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: Role::Tool,
            content: Some(content.into()),
            tool_calls: Vec::new(),
            tool_call_id: Some(call_id.into()),
            name: Some(name.into()),
            images: Vec::new(),
        }
    }

    /// Text content, or an empty string.
    pub fn text(&self) -> &str {
        self.content.as_deref().unwrap_or("")
    }
}
