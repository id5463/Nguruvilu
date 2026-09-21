//! A plugin that exercises every extension point at once.
//!
//! This is the worked example for "what can a plugin do": it contributes a tool,
//! a theme layer, a UI panel, and a subagent. Read it as the answer to that
//! question rather than as something to install — it exists to be copied.
//!
//! Build it as a dynamic library and load it with a pack:
//!
//! ```sh
//! cargo build --release --manifest-path examples/demo-plugin/Cargo.toml
//! ```

use std::sync::Arc;

use anyhow::Result;
use serde_json::{json, Value};

use nguruvilu::agent::{Agent, StaticConfig};
use nguruvilu::message::Message;
use nguruvilu::model::{ModelAccess, ModelHandle, SERVICE};
use nguruvilu::plugin::{Contributions, Plugin, PluginCtx};
use nguruvilu::theme::Theme;
use nguruvilu::tools::{ConflictPolicy, ToolDef, ToolFuture, ToolOutput};
use nguruvilu::ui::{UiPanel, UiSlot};

/// Name used to load the plugin.
pub const NAME: &str = "example";

/// Contributes a tool, a theme, and a panel.
pub struct ExamplePlugin;

impl Plugin for ExamplePlugin {
    fn name(&self) -> &str {
        NAME
    }

    fn apply(&self, ctx: &PluginCtx) -> Result<Contributions> {
        let access = ModelHandle::from_view(&ctx.services);

        // A panel that reports whether the model service was found. Without it
        // a subagent tool that silently does nothing looks identical to one that
        // works and returns nothing.
        let status = match &access {
            Some(_) => ready_panel(),
            None => unavailable_panel(),
        };

        Ok(Contributions::new()
            .tool(subagent_tool(access))
            .tool(echo_tool())
            .theme(light_theme())
            .ui(status_panel())
            .ui(status))
    }
}

/// A light theme, for a user who finds the default too dark.
fn light_theme() -> Theme {
    Theme::new("example-light")
        .set("--bg", "#ffffff")
        .set("--panel", "#f4f6f9")
        .set("--line", "#d9dee6")
        .set("--text", "#1a1f26")
        .set("--dim", "#667080")
        .set("--accent", "#1a6fd4")
}

/// A panel in the right column.
fn status_panel() -> UiPanel {
    UiPanel::new(
        "example-status",
        UiSlot::DetailsTop,
        "<div id='example-count' style='font-size:11.5px'>0 subagent runs</div>\
         <button id='example-reset' style='margin-top:4px'>Reset</button>",
    )
    .titled("Example")
    .with_script(
        "let runs = 0;\
         const out = panel.querySelector('#example-count');\
         panel.querySelector('#example-reset').onclick = () => { runs = 0; out.textContent = '0 subagent runs'; };\
         // The panel is a plain fragment: it updates itself through the same\
         // channel the built-in panels use.\
         window.__exampleCount = () => { runs += 1; out.textContent = runs + ' subagent runs'; };",
    )
}

fn ready_panel() -> UiPanel {
    UiPanel::new(
        "example-model",
        UiSlot::StatusBar,
        "<span style='color:#7ddc9a'>model service ready</span>",
    )
}

fn unavailable_panel() -> UiPanel {
    UiPanel::new(
        "example-model",
        UiSlot::StatusBar,
        "<span style='color:#e8c98a'>no model service: subagent tool disabled</span>",
    )
}

/// A tool that echoes its input. The smallest possible tool.
fn echo_tool() -> ToolDef {
    ToolDef::new(
        "example_echo",
        "Echo the given text back. Used to prove the plugin loaded.",
        json!({
            "type": "object",
            "properties": { "text": { "type": "string" } },
            "required": ["text"]
        }),
        NAME,
        |args| Box::pin(async move {
            let text = args.get("text").and_then(|v| v.as_str()).unwrap_or("");
            Ok(ToolOutput::text(format!("example plugin heard: {text}")))
        }) as ToolFuture,
    )
}

/// A subagent: a nested agent loop on the session's own route.
///
/// This is the whole of what "subagent support" means here. There is no
/// separate registry or scheduler — the plugin builds an [`Agent`] with the
/// session's client, tools, and prompt, gives it a fresh history, and returns
/// its answer. Being a tool, it inherits the same permission checks and the same
/// recording as any other tool call.
fn subagent_tool(access: Option<Arc<dyn ModelAccess>>) -> ToolDef {
    ToolDef::new(
        "example_subagent",
        "Run a subtask in a fresh agent with no memory of this conversation, and return \
         its answer. Use it to keep a large side investigation out of this transcript. \
         The subagent sees the same tools and the same model route.",
        json!({
            "type": "object",
            "properties": {
                "task": { "type": "string", "description": "What the subagent should do." },
                "model": { "type": "string", "description": "Optional model override; defaults to the session's." }
            },
            "required": ["task"]
        }),
        NAME,
        move |args| {
            let access = access.clone();
            Box::pin(async move { run_subagent(access, args).await }) as ToolFuture
        },
    )
}

async fn run_subagent(access: Option<Arc<dyn ModelAccess>>, args: Value) -> Result<ToolOutput> {
    // Failing loudly beats a tool that quietly returns nothing: a model that
    // gets an empty answer will retry, and retrying forever looks like a hang.
    let Some(access) = access else {
        anyhow::bail!(
            "the model service is not available, so no subagent can be started. \
             It is published by the host as the '{SERVICE}' service."
        );
    };

    let task = args
        .get("task")
        .and_then(|t| t.as_str())
        .ok_or_else(|| anyhow::anyhow!("missing required argument: task"))?;

    let client = match args.get("model").and_then(|m| m.as_str()) {
        Some(model) => access.client_for(model)?,
        None => access.client()?,
    };

    // A fresh history: the point of a subagent is that it does not carry the
    // parent's context, so the parent's transcript stays small. The policy comes
    // from the session so a subagent compacts the same way its parent does.
    let config: Arc<dyn nguruvilu::agent::TurnConfig> = Arc::new(
        StaticConfig::new(access.system_prompt(), access.tools(), client.model())
            .with_policy(access.context_policy()),
    );

    let mut agent = Agent::new(client, config, Vec::new());

    let outcome = agent.run(task).await?;

    Ok(ToolOutput::text(format!(
        "subagent finished after {} step(s), {} tool call(s):\n\n{}",
        outcome.steps, outcome.tool_calls, outcome.text
    )))
}

/// Registering through the kernel's own table, for a statically linked build.
pub fn register(registry: &mut nguruvilu::tools::ToolRegistry) -> Result<()> {
    registry.register(echo_tool(), ConflictPolicy::Error)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use nguruvilu::plugin::{Kernel, RealmMap};
    use nguruvilu::tools::ToolRegistry;

    /// A stand-in for a host's access, so the tests need no network.
    struct FakeAccess {
        tools: Arc<ToolRegistry>,
    }

    impl ModelAccess for FakeAccess {
        fn route(&self) -> nguruvilu::hotreload::ModelRoute {
            nguruvilu::hotreload::ModelRoute {
                provider: "openai".into(),
                base_url: "http://localhost:1/v1".into(),
                api_key: "k".into(),
                model: "m".into(),
                ..Default::default()
            }
        }

        fn client(&self) -> Result<nguruvilu::llm::LlmClient> {
            self.client_for("m")
        }

        fn client_for(&self, model: &str) -> Result<nguruvilu::llm::LlmClient> {
            Ok(nguruvilu::llm::LlmClient::new(
                nguruvilu::llm::LlmConfig::new("http://localhost:1/v1", "k", model),
            )?)
        }

        fn tools(&self) -> Arc<ToolRegistry> {
            Arc::clone(&self.tools)
        }

        fn system_prompt(&self) -> String {
            "you are a subagent".into()
        }

        fn context_policy(&self) -> Arc<dyn nguruvilu::window::ContextPolicy> {
            Arc::new(nguruvilu::window::DefaultContextPolicy::default())
        }
    }

    fn contributions() -> Contributions {
        let tools = Arc::new(ToolRegistry::with_base_tools().unwrap());
        let ctx = PluginCtx {
            plugin: NAME.into(),
            fiber: 1,
            realm: RealmMap::new(),
            services: nguruvilu::plugin::ServiceView::default(),
            config: Value::Null,
        };
        let _ = tools;
        Plugin::apply(&ExamplePlugin, &ctx).unwrap()
    }

    #[test]
    fn the_plugin_contributes_all_four_kinds_of_thing() {
        let c = contributions();
        assert_eq!(c.tools.len(), 2, "echo and subagent");
        assert!(c.theme.is_some(), "a theme layer");
        assert!(c.ui.len() >= 2, "a panel and a status indicator");
    }

    #[test]
    fn the_theme_only_names_real_tokens() {
        let mut theme = light_theme();
        let rejected = theme.sanitize();
        assert!(rejected.is_empty(), "these would be dead tokens: {rejected:?}");
        assert!(theme.len() >= 5);
    }

    #[test]
    fn the_theme_changes_the_background_to_light() {
        let mut registry = nguruvilu::theme::ThemeRegistry::new();
        registry.add(light_theme());
        let resolved = registry.resolve();
        assert_eq!(resolved.tokens["--bg"], "#ffffff");
        assert_eq!(
            resolved.tokens["--mono"],
            nguruvilu::theme::TOKENS
                .iter()
                .find(|(t, _)| *t == "--mono")
                .unwrap()
                .1,
            "an unthemed token keeps the built-in value"
        );
    }

    #[test]
    fn the_panels_land_in_their_slots() {
        let c = contributions();
        let slots: Vec<UiSlot> = c.ui.iter().map(|p| p.slot).collect();
        assert!(slots.contains(&UiSlot::DetailsTop));
        assert!(slots.contains(&UiSlot::StatusBar));
    }

    #[test]
    fn without_the_model_service_the_panel_says_so() {
        // The contributions here are built with an empty service view, so the
        // subagent tool is present but unusable and the page must show that.
        let c = contributions();
        let status = c
            .ui
            .iter()
            .find(|p| p.slot == UiSlot::StatusBar)
            .expect("a status panel");
        assert!(status.html.contains("no model service"), "{}", status.html);
    }

    #[tokio::test]
    async fn the_subagent_tool_refuses_clearly_when_the_service_is_missing() {
        let error = run_subagent(None, json!({ "task": "anything" }))
            .await
            .expect_err("must refuse");
        let text = format!("{error:#}");
        assert!(text.contains("model service"), "{text}");
        assert!(text.contains(SERVICE), "it should name the service: {text}");
    }

    #[tokio::test]
    async fn the_subagent_tool_requires_a_task() {
        let access: Arc<dyn ModelAccess> = Arc::new(FakeAccess {
            tools: Arc::new(ToolRegistry::with_base_tools().unwrap()),
        });
        let error = run_subagent(Some(access), json!({}))
            .await
            .expect_err("must refuse");
        assert!(format!("{error:#}").contains("task"));
    }

    #[tokio::test]
    async fn the_echo_tool_works() {
        let tool = echo_tool();
        let output = (tool.handler)(json!({ "text": "hi" })).await.unwrap();
        assert!(output.text.contains("hi"));
    }

    #[test]
    fn a_kernel_with_the_service_publishes_it_to_the_plugin() {
        let mut kernel = Kernel::new();
        let access: Arc<dyn ModelAccess> = Arc::new(FakeAccess {
            tools: Arc::new(ToolRegistry::with_base_tools().unwrap()),
        });
        nguruvilu::model::install(&mut kernel, access).unwrap();

        let view = kernel.service_view(RealmMap::new());
        assert!(
            ModelHandle::from_view(&view).is_some(),
            "the plugin's lookup path works against a real kernel"
        );
    }
}
