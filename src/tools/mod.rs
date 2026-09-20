//! The tool registry.
//!
//! The registry is the kernel's mutable tool table. Plugins contribute to it
//! through `register`; the four base tools are registered at boot.
//!
//! Name collisions fail loud by default. Silent override is the most
//! dangerous failure mode a plugin system can have: one plugin quietly
//! replacing another's safety check leaves no trace. A contribution that
//! genuinely intends to replace an existing tool says so explicitly.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use anyhow::{anyhow, Result};
use serde_json::{json, Value};

pub mod base;

/// What a tool produced.
///
/// Text is what the model reads. Images cannot travel in a tool message — the
/// OpenAI format restricts that content to text — so the loop collects them and
/// attaches them to a user message placed after the tool results.
#[derive(Debug, Clone, Default)]
pub struct ToolOutput {
    /// Text the model reads.
    pub text: String,
    /// Images the model should see.
    pub images: Vec<crate::message::ImageAttachment>,
}

impl ToolOutput {
    /// Text only.
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            images: Vec::new(),
        }
    }

    /// Attach an image.
    pub fn with_image(mut self, image: crate::message::ImageAttachment) -> Self {
        self.images.push(image);
        self
    }

    /// Whether this output carries anything for the model to look at.
    pub fn has_images(&self) -> bool {
        !self.images.is_empty()
    }
}

impl From<String> for ToolOutput {
    fn from(text: String) -> Self {
        Self::text(text)
    }
}

impl From<&str> for ToolOutput {
    fn from(text: &str) -> Self {
        Self::text(text)
    }
}

/// Boxed future returned by a tool handler.
pub type ToolFuture = Pin<Box<dyn Future<Output = Result<ToolOutput>> + Send>>;

/// A tool implementation.
pub type ToolHandler = Arc<dyn Fn(Value) -> ToolFuture + Send + Sync>;

/// What to do when a tool name is already registered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConflictPolicy {
    /// Refuse the registration and report both owners (default).
    #[default]
    Error,
    /// Replace the existing tool. The replaced owner is recorded.
    Override,
}

/// One registered tool.
#[derive(Clone)]
pub struct ToolDef {
    /// Name the model calls.
    pub name: String,
    /// Description shown to the model.
    pub description: String,
    /// JSON Schema for the arguments object.
    pub parameters: Value,
    /// Implementation.
    pub handler: ToolHandler,
    /// Name of the plugin that contributed this tool (`"kernel"` for base tools).
    pub owner: String,
}

impl ToolDef {
    /// Build a tool definition from a handler closure.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: Value,
        owner: impl Into<String>,
        handler: impl Fn(Value) -> ToolFuture + Send + Sync + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
            handler: Arc::new(handler),
            owner: owner.into(),
        }
    }

    /// The OpenAI `tools` entry for this tool.
    pub fn schema(&self) -> Value {
        json!({
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        })
    }
}

/// Record of one override, kept so conflicts stay inspectable.
#[derive(Debug, Clone)]
pub struct OverrideRecord {
    /// Tool name.
    pub name: String,
    /// Owner that was replaced.
    pub previous_owner: String,
    /// Owner that replaced it.
    pub new_owner: String,
}

/// The kernel's mutable tool table.
#[derive(Default, Clone)]
pub struct ToolRegistry {
    tools: HashMap<String, ToolDef>,
    overrides: Vec<OverrideRecord>,
}

impl ToolRegistry {
    /// An empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// A registry with the four base tools registered.
    pub fn with_base_tools() -> Result<Self> {
        let mut registry = Self::new();
        base::register_all(&mut registry)?;
        Ok(registry)
    }

    /// Register a tool.
    ///
    /// Fails when the name is taken and the policy is [`ConflictPolicy::Error`].
    pub fn register(&mut self, def: ToolDef, policy: ConflictPolicy) -> Result<()> {
        if let Some(existing) = self.tools.get(&def.name) {
            match policy {
                ConflictPolicy::Error => {
                    return Err(anyhow!(
                        "tool name conflict: '{}' is already registered by '{}' (requested by '{}')",
                        def.name,
                        existing.owner,
                        def.owner
                    ));
                }
                ConflictPolicy::Override => {
                    self.overrides.push(OverrideRecord {
                        name: def.name.clone(),
                        previous_owner: existing.owner.clone(),
                        new_owner: def.owner.clone(),
                    });
                }
            }
        }
        self.tools.insert(def.name.clone(), def);
        Ok(())
    }

    /// Remove a tool, returning whether it existed.
    pub fn unregister(&mut self, name: &str) -> bool {
        self.tools.remove(name).is_some()
    }

    /// Remove every tool contributed by `owner`, returning how many went.
    ///
    /// Unloading a plugin must not leave its tools callable.
    pub fn retain_owner(&mut self, owner: &str) -> usize {
        let before = self.tools.len();
        self.tools.retain(|_, def| def.owner != owner);
        before - self.tools.len()
    }

    /// Look up a tool.
    pub fn get(&self, name: &str) -> Option<&ToolDef> {
        self.tools.get(name)
    }

    /// All tool names, sorted.
    pub fn names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.tools.keys().cloned().collect();
        names.sort();
        names
    }

    /// Number of registered tools.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Owner of a registered tool, if any.
    pub fn owner(&self, name: &str) -> Option<&str> {
        self.tools.get(name).map(|t| t.owner.as_str())
    }

    /// Every recorded override.
    pub fn overrides(&self) -> &[OverrideRecord] {
        &self.overrides
    }

    /// Tool schemas in a stable order, for the model request.
    pub fn schemas(&self) -> Vec<Value> {
        let mut entries: Vec<(&String, &ToolDef)> = self.tools.iter().collect();
        entries.sort_by(|a, b| a.0.cmp(b.0));
        entries.into_iter().map(|(_, def)| def.schema()).collect()
    }

    /// Execute a tool by name with raw JSON arguments.
    pub async fn execute(&self, name: &str, arguments: &str) -> Result<ToolOutput> {
        let def = self
            .tools
            .get(name)
            .ok_or_else(|| anyhow!("unknown tool '{name}'"))?;

        let args: Value = if arguments.trim().is_empty() {
            json!({})
        } else {
            serde_json::from_str(arguments)
                .map_err(|e| anyhow!("invalid JSON arguments for '{name}': {e}"))?
        };

        (def.handler)(args).await
    }
}
