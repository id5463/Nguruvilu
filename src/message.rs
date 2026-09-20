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
        }
    }

    /// A user message.
    pub fn user(text: impl Into<String>) -> Self {
        Self { role: Role::User, content: Some(text.into()), tool_calls: Vec::new(), tool_call_id: None, name: None }
    }

    /// A system message.
    pub fn system(text: impl Into<String>) -> Self {
        Self { role: Role::System, content: Some(text.into()), tool_calls: Vec::new(), tool_call_id: None, name: None }
    }

    /// An assistant message with text only.
    pub fn assistant(text: impl Into<String>) -> Self {
        Self { role: Role::Assistant, content: Some(text.into()), tool_calls: Vec::new(), tool_call_id: None, name: None }
    }

    /// An assistant message that requests tool calls.
    pub fn assistant_tools(calls: Vec<ToolCall>, text: Option<String>) -> Self {
        Self { role: Role::Assistant, content: text, tool_calls: calls, tool_call_id: None, name: None }
    }

    /// A tool result message.
    pub fn tool_result(call_id: impl Into<String>, name: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: Role::Tool,
            content: Some(content.into()),
            tool_calls: Vec::new(),
            tool_call_id: Some(call_id.into()),
            name: Some(name.into()),
        }
    }

    /// Text content, or an empty string.
    pub fn text(&self) -> &str {
        self.content.as_deref().unwrap_or("")
    }
}
