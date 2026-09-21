//! Applying what a pack carries: persona, route, context, appearance, rules.
//!
//! A pack's manifest names content files. This turns them into things the
//! running kernel uses, and hands back the parts the *host* owns — the persona
//! text, the model route, the context policy — because those live in the
//! session's settings rather than in the kernel's registries.
//!
//! # Appearance is applied here; the rest is returned
//!
//! Themes and panels belong to the kernel's plugin registries, so they are
//! installed directly as one built-in plugin. Everything else describes *this
//! session's* configuration, and the kernel does not own that — the CLI and the
//! desktop shell do, differently. Returning them keeps that split honest
//! instead of teaching the kernel about a settings file.

use std::path::Path;

use anyhow::{Context, Result};

use crate::pack::{ContextFile, LookFile, McpFile, ModelsFile, PackManifest};

/// What a pack contributes that the host has to apply.
#[derive(Debug, Clone, Default)]
pub struct PackContent {
    /// Persona text, from `soul.md`.
    pub soul: Option<String>,
    /// Model route, from `models.json`.
    pub models: Option<ModelsFile>,
    /// Context policy, from `context.json`.
    pub context: Option<ContextFile>,
    /// Injection rules, from `injections.json`.
    pub injections: Option<crate::context::InjectionEngine>,
    /// MCP servers, from `mcp.json`.
    pub mcp: Option<McpFile>,
}

impl PackContent {
    /// Whether the pack carried nothing.
    pub fn is_empty(&self) -> bool {
        self.soul.is_none()
            && self.models.is_none()
            && self.context.is_none()
            && self.injections.is_none()
            && self.mcp.is_none()
    }

    /// One line per part, for diagnostics.
    pub fn summary(&self) -> Vec<String> {
        let mut out = Vec::new();
        if let Some(soul) = &self.soul {
            out.push(format!("persona ({} chars)", soul.chars().count()));
        }
        if let Some(models) = &self.models {
            let mut parts = Vec::new();
            if let Some(model) = &models.model {
                parts.push(format!("model {model}"));
            }
            if let Some(url) = &models.base_url {
                parts.push(format!("endpoint {url}"));
            }
            if !models.extra_body.is_empty() {
                parts.push(format!("{} shaped field(s)", models.extra_body.len()));
            }
            out.push(format!("route ({})", parts.join(", ")));
        }
        if let Some(context) = &self.context {
            let mut parts = Vec::new();
            if let Some(window) = &context.window {
                parts.push(format!("window {window}"));
            }
            if let Some(policy) = &context.cache_policy {
                parts.push(format!("cache {policy}"));
            }
            out.push(format!("context ({})", parts.join(", ")));
        }
        if let Some(injections) = &self.injections {
            out.push(format!("{} injection rule(s)", injections.len()));
        }
        if let Some(mcp) = &self.mcp {
            out.push(format!("{} MCP server(s)", mcp.servers.len()));
        }
        out
    }
}

/// Read every content file a pack names, and install its appearance.
///
/// Returns what the host still has to apply. A file that is named but missing
/// is an error: a pack that silently loses its persona because of a typo is
/// worse than one that refuses to load.
pub fn apply(
    kernel: &mut crate::plugin::Kernel,
    dir: &Path,
    manifest: &PackManifest,
) -> Result<PackContent> {
    let mut content = PackContent::default();

    if let Some(file) = &manifest.soul {
        let path = dir.join(file);
        content.soul = Some(
            std::fs::read_to_string(&path)
                .with_context(|| format!("reading the persona from {}", path.display()))?,
        );
    }

    if let Some(file) = &manifest.models {
        let path = dir.join(file);
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        content.models = Some(
            serde_json::from_str(&text)
                .with_context(|| format!("{} is not a valid models file", path.display()))?,
        );
    }

    if let Some(file) = &manifest.context {
        let path = dir.join(file);
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        content.context = Some(
            serde_json::from_str(&text)
                .with_context(|| format!("{} is not a valid context file", path.display()))?,
        );
    }

    if let Some(file) = &manifest.injections {
        content.injections = Some(crate::pack::read_injections(&dir.join(file))?);
    }

    if let Some(file) = &manifest.mcp {
        let path = dir.join(file);
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        content.mcp = Some(
            serde_json::from_str(&text)
                .with_context(|| format!("{} is not a valid mcp file", path.display()))?,
        );
    }

    if let Some(file) = &manifest.look {
        let path = dir.join(file);
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        let look: LookFile = serde_json::from_str(&text)
            .with_context(|| format!("{} is not a valid look file", path.display()))?;
        install_appearance(kernel, &manifest.name, look)?;
    }

    Ok(content)
}

/// Install themes and panels as one built-in plugin.
///
/// A plugin rather than a direct registry write, so they land on a fiber and
/// unload with the pack. Writing them into the kernel's tables directly would
/// leave an appearance behind that nothing owns and nothing can remove.
fn install_appearance(
    kernel: &mut crate::plugin::Kernel,
    pack_name: &str,
    look: LookFile,
) -> Result<()> {
    if look.themes.is_empty() && look.panels.is_empty() {
        return Ok(());
    }

    // An `activeTheme` naming a theme that is not in the file would otherwise
    // fall back to the first, which is a different theme than the author asked
    // for and looks like it worked.
    let choice_problems = look.problems();
    if !choice_problems.is_empty() {
        anyhow::bail!("pack '{pack_name}': {}", choice_problems.join("; "));
    }

    // A theme naming a token the interface does not have is a theme that
    // quietly does nothing, so it is reported rather than dropped.
    for theme in &look.themes {
        let mut probe = theme.clone();
        let rejected = probe.sanitize();
        if !rejected.is_empty() {
            anyhow::bail!(
                "theme '{}' in pack '{pack_name}' sets unknown token(s): {}",
                theme.name,
                rejected.join(", ")
            );
        }
    }

    struct Appearance {
        name: String,
        look: LookFile,
    }

    impl crate::plugin::Plugin for Appearance {
        fn name(&self) -> &str {
            &self.name
        }

        fn apply(
            &self,
            _ctx: &crate::plugin::PluginCtx,
        ) -> Result<crate::plugin::Contributions> {
            let mut contributions = crate::plugin::Contributions::new();
            // Variants, not layers: exactly one theme is in force. Applying
            // every theme in the file would let the last win and hide the
            // choice the author offered.
            if let Some(theme) = self.look.active() {
                contributions = contributions.theme(theme.clone());
            }
            for panel in &self.look.panels {
                contributions = contributions.ui(panel.clone());
            }
            Ok(contributions)
        }
    }

    let name = format!("pack:{pack_name}");
    kernel.define(std::sync::Arc::new(Appearance {
        name: name.clone(),
        look,
    }));
    kernel.load(&name, crate::plugin::RealmMap::new(), serde_json::Value::Null)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pack::{PackManifest, write_manifest};
    use std::path::PathBuf;

    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir()
            .join(format!("ngu-content-{tag}-{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn a_pack_with_no_content_files_yields_nothing() {
        let dir = scratch("empty");
        let manifest = PackManifest::new("demo", "1.0.0");
        write_manifest(&dir, &manifest).unwrap();

        let mut kernel = crate::plugin::Kernel::new();
        let content = apply(&mut kernel, &dir, &manifest).unwrap();
        assert!(content.is_empty());
        assert!(content.summary().is_empty());
    }

    #[test]
    fn a_named_but_missing_file_is_an_error() {
        // Losing a persona to a typo is worse than refusing to load.
        let dir = scratch("missing");
        let mut manifest = PackManifest::new("demo", "1.0.0");
        manifest.soul = Some("soul.md".into());
        write_manifest(&dir, &manifest).unwrap();

        let mut kernel = crate::plugin::Kernel::new();
        let error = apply(&mut kernel, &dir, &manifest).expect_err("must refuse");
        assert!(format!("{error:#}").contains("soul.md"));
    }

    #[test]
    fn a_persona_is_returned_for_the_host_to_apply() {
        let dir = scratch("soul");
        let mut manifest = PackManifest::new("demo", "1.0.0");
        manifest.soul = Some("soul.md".into());
        write_manifest(&dir, &manifest).unwrap();
        std::fs::write(dir.join("soul.md"), "Be terse.\n").unwrap();

        let mut kernel = crate::plugin::Kernel::new();
        let content = apply(&mut kernel, &dir, &manifest).unwrap();
        assert_eq!(content.soul.as_deref(), Some("Be terse.\n"));
        assert!(content.summary()[0].contains("persona"));
    }

    #[test]
    fn a_theme_lands_on_a_fiber_and_resolves() {
        let dir = scratch("theme");
        let mut manifest = PackManifest::new("look", "1.0.0");
        manifest.look = Some("look.json".into());
        write_manifest(&dir, &manifest).unwrap();

        let mut look = LookFile::default();
        look.themes
            .push(crate::theme::Theme::new("light").set("--bg", "#ffffff"));
        std::fs::write(dir.join("look.json"), serde_json::to_string(&look).unwrap()).unwrap();

        let mut kernel = crate::plugin::Kernel::new();
        apply(&mut kernel, &dir, &manifest).unwrap();

        let resolved = kernel.resolved_theme();
        assert_eq!(resolved.tokens["--bg"], "#ffffff");
        assert!(
            resolved.tokens["--text"].starts_with('#'),
            "unthemed tokens keep their built-in value"
        );
    }

    #[test]
    fn a_panel_lands_on_a_fiber() {
        let dir = scratch("panel");
        let mut manifest = PackManifest::new("look", "1.0.0");
        manifest.look = Some("look.json".into());
        write_manifest(&dir, &manifest).unwrap();

        let mut look = LookFile::default();
        look.panels.push(crate::ui::UiPanel::new(
            "clock",
            crate::ui::UiSlot::StatusBar,
            "<span></span>",
        ));
        std::fs::write(dir.join("look.json"), serde_json::to_string(&look).unwrap()).unwrap();

        let mut kernel = crate::plugin::Kernel::new();
        apply(&mut kernel, &dir, &manifest).unwrap();

        let panels = kernel.ui_panels();
        assert_eq!(panels.len(), 1);
        assert_eq!(panels[0].id, "clock");
    }

    #[test]
    fn a_theme_with_an_unknown_token_is_refused_by_name() {
        let dir = scratch("bad-token");
        let mut manifest = PackManifest::new("look", "1.0.0");
        manifest.look = Some("look.json".into());
        write_manifest(&dir, &manifest).unwrap();

        let mut look = LookFile::default();
        look.themes
            .push(crate::theme::Theme::new("typo").set("--bakground", "#fff"));
        std::fs::write(dir.join("look.json"), serde_json::to_string(&look).unwrap()).unwrap();

        let mut kernel = crate::plugin::Kernel::new();
        let error = apply(&mut kernel, &dir, &manifest).expect_err("must refuse");
        let text = format!("{error:#}");
        assert!(text.contains("--bakground"), "{text}");
        assert!(text.contains("typo"), "{text}");
    }

    #[test]
    fn a_pack_offering_several_themes_applies_the_chosen_one() {
        // Variants, not layers. Applying all of them would let the last win and
        // hide the choice the author offered.
        let dir = scratch("variants");
        let mut manifest = PackManifest::new("look", "1.0.0");
        manifest.look = Some("look.json".into());
        write_manifest(&dir, &manifest).unwrap();

        let mut look = LookFile {
            active_theme: Some("light".into()),
            ..Default::default()
        };
        look.themes.push(
            crate::theme::Theme::new("light")
                .set("--bg", "#ffffff")
                .set("--text", "#000000"),
        );
        look.themes
            .push(crate::theme::Theme::new("dark").set("--bg", "#000000"));
        std::fs::write(dir.join("look.json"), serde_json::to_string(&look).unwrap()).unwrap();

        let mut kernel = crate::plugin::Kernel::new();
        apply(&mut kernel, &dir, &manifest).unwrap();

        let resolved = kernel.resolved_theme();
        assert_eq!(resolved.tokens["--bg"], "#ffffff", "the chosen one");
        assert_eq!(resolved.name, "light", "and only it");
    }

    #[test]
    fn an_absent_choice_takes_the_first_theme() {
        let dir = scratch("first");
        let mut manifest = PackManifest::new("look", "1.0.0");
        manifest.look = Some("look.json".into());
        write_manifest(&dir, &manifest).unwrap();

        let mut look = LookFile::default();
        look.themes
            .push(crate::theme::Theme::new("first").set("--bg", "#111111"));
        look.themes
            .push(crate::theme::Theme::new("second").set("--bg", "#222222"));
        std::fs::write(dir.join("look.json"), serde_json::to_string(&look).unwrap()).unwrap();

        let mut kernel = crate::plugin::Kernel::new();
        apply(&mut kernel, &dir, &manifest).unwrap();
        assert_eq!(kernel.resolved_theme().tokens["--bg"], "#111111");
    }

    #[test]
    fn choosing_a_theme_that_is_not_in_the_file_is_refused_by_name() {
        let dir = scratch("bad-choice");
        let mut manifest = PackManifest::new("look", "1.0.0");
        manifest.look = Some("look.json".into());
        write_manifest(&dir, &manifest).unwrap();

        let mut look = LookFile {
            active_theme: Some("nope".into()),
            ..Default::default()
        };
        look.themes
            .push(crate::theme::Theme::new("light").set("--bg", "#ffffff"));
        std::fs::write(dir.join("look.json"), serde_json::to_string(&look).unwrap()).unwrap();

        let mut kernel = crate::plugin::Kernel::new();
        let error = apply(&mut kernel, &dir, &manifest).expect_err("must refuse");
        let text = format!("{error:#}");
        assert!(text.contains("nope"), "{text}");
        assert!(text.contains("light"), "it should list the real ones: {text}");
    }

    #[test]
    fn injection_rules_are_parsed_from_the_packs_own_format() {
        let dir = scratch("injections");
        let mut manifest = PackManifest::new("rules", "1.0.0");
        manifest.injections = Some("injections.json".into());
        write_manifest(&dir, &manifest).unwrap();
        std::fs::write(
            dir.join("injections.json"),
            "{\"budget_percent\":10,\"entries\":[]}",
        )
        .unwrap();

        let mut kernel = crate::plugin::Kernel::new();
        let content = apply(&mut kernel, &dir, &manifest).unwrap();
        let engine = content.injections.expect("parsed");
        assert_eq!(engine.budget_percent, 10);
    }

    #[test]
    fn the_models_file_parses_and_reports_itself() {
        let dir = scratch("models");
        let mut manifest = PackManifest::new("route", "1.0.0");
        manifest.models = Some("models.json".into());
        write_manifest(&dir, &manifest).unwrap();
        std::fs::write(
            dir.join("models.json"),
            "{\"baseUrl\":\"https://x/v1\",\"apiKeyEnv\":\"K\",\"model\":\"m\",\"extraBody\":{\"a\":1}}",
        )
        .unwrap();

        let mut kernel = crate::plugin::Kernel::new();
        let content = apply(&mut kernel, &dir, &manifest).unwrap();
        let summary = content.summary();
        let models = content.models.expect("parsed");
        assert_eq!(models.model.as_deref(), Some("m"));
        assert_eq!(models.extra_body.len(), 1);
        assert!(summary[0].contains("route"));
    }

    #[test]
    fn the_mcp_file_parses_into_servers() {
        let dir = scratch("mcp");
        let mut manifest = PackManifest::new("servers", "1.0.0");
        manifest.mcp = Some("mcp.json".into());
        write_manifest(&dir, &manifest).unwrap();
        std::fs::write(
            dir.join("mcp.json"),
            "{\"servers\":[{\"id\":\"fs\",\"command\":\"npx\",\"args\":[\"-y\",\"x\"]}]}",
        )
        .unwrap();

        let mut kernel = crate::plugin::Kernel::new();
        let content = apply(&mut kernel, &dir, &manifest).unwrap();
        let mcp = content.mcp.expect("parsed");
        assert_eq!(mcp.servers.len(), 1);
        assert_eq!(mcp.servers[0].id, "fs");
        assert_eq!(mcp.servers[0].transport, "stdio", "the default");
    }

    #[test]
    fn a_malformed_content_file_names_itself() {
        let dir = scratch("malformed");
        let mut manifest = PackManifest::new("demo", "1.0.0");
        manifest.context = Some("context.json".into());
        write_manifest(&dir, &manifest).unwrap();
        std::fs::write(dir.join("context.json"), "{ not json").unwrap();

        let mut kernel = crate::plugin::Kernel::new();
        let error = apply(&mut kernel, &dir, &manifest).expect_err("must refuse");
        assert!(format!("{error:#}").contains("context.json"));
    }
}
