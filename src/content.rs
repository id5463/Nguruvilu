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

    /// Fold the settings-side content of this pack onto `settings`.
    ///
    /// Route and context window — the host-owned fields a pack may
    /// declare over the user's own. A field the pack does not carry keeps the
    /// value already there, so packs compose in the order they are applied.
    /// Persona and injections are deliberately *not* handled here: the persona
    /// is a join over every enabled pack and the injection engine is replaced
    /// as one unit, so both need the whole set rather than one pack.
    pub fn fold_settings(&self, settings: &mut crate::settings::Settings) {
        if let Some(models) = &self.models {
            if let Some(url) = &models.base_url {
                settings.base_url = url.clone();
            }
            if let Some(name) = &models.model {
                settings.model = name.clone();
            }
            if let Some(effort) = &models.reasoning_effort {
                settings.reasoning_effort = effort.clone();
            }
            if let Some(delivery) = &models.delivery {
                settings.delivery = delivery.clone();
            }
            if let Some(degrade) = models.degrade_on_stall {
                settings.degrade_on_stall = Some(degrade);
            }
            if let Some(proxy) = &models.proxy {
                settings.proxy = proxy.clone();
            }
            match models
                .max_output_tokens
                .as_deref()
                .map(crate::size::parse_size)
            {
                Some(Ok(tokens)) => settings.max_output_tokens = Some(tokens),
                Some(Err(error)) => eprintln!("[pack] maxOutputTokens: {error}"),
                None => {}
            }
            for (key, value) in &models.extra_body {
                settings.extra_body.insert(key.clone(), value.clone());
            }
            if let Some(network) = &models.network {
                settings.network = network.clone();
            }
            // The key itself is never in a pack; only the name of the variable
            // holding it.
            if let Some(var) = &models.api_key_env {
                if let Ok(value) = std::env::var(var) {
                    if !value.trim().is_empty() {
                        settings.api_key = value;
                    }
                }
            }
        }

        if let Some(context) = &self.context {
            if let Some(window) = &context.window {
                settings.context_window = crate::size::parse_size(window).ok();
            }
            if let Some(percent) = context.compact_percent {
                settings.compact_percent = percent.min(100);
            }
            if let Some(keep) = context.compact_keep_recent {
                settings.compact_keep_recent = keep.max(1);
            }
        }
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

/// Read every content file a pack names, without touching the kernel.
///
/// This is the read-only half of [`apply`]: persona, route, context, rules,
/// MCP declarations — everything the *host* owns. Appearance is the
/// one kernel-owned part and is deliberately left to `apply`, so a recomputation
/// pass over the installed packs can re-read what they contribute without
/// reinstalling appearance fibers that are already in place.
pub fn read(dir: &Path, manifest: &PackManifest) -> Result<PackContent> {
    let mut content = PackContent::default();

    if let Some(file) = &manifest.soul {
        let path = dir.join(file.path());
        content.soul = Some(
            std::fs::read_to_string(&path)
                .with_context(|| format!("reading the persona from {}", path.display()))?,
        );
    }

    if let Some(file) = &manifest.models {
        let path = dir.join(file.path());
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        content.models = Some(
            serde_json::from_str(&text)
                .with_context(|| format!("{} is not a valid models file", path.display()))?,
        );
    }

    if let Some(file) = &manifest.context {
        let path = dir.join(file.path());
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        content.context = Some(
            serde_json::from_str(&text)
                .with_context(|| format!("{} is not a valid context file", path.display()))?,
        );
    }

    if let Some(file) = &manifest.injections {
        content.injections = Some(crate::pack::read_injections(&dir.join(file.path()))?);
    }

    if let Some(file) = &manifest.mcp {
        let path = dir.join(file.path());
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        content.mcp = Some(
            serde_json::from_str(&text)
                .with_context(|| format!("{} is not a valid mcp file", path.display()))?,
        );
    }

    Ok(content)
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
    let content = read(dir, manifest)?;

    if let Some(file) = &manifest.look {
        let path = dir.join(file.path());
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        let mut look: LookFile = serde_json::from_str(&text)
            .with_context(|| format!("{} is not a valid look file", path.display()))?;

        // `hooks` holds file names, not script text: a hook is code, and code
        // in a JSON string is unreadable and uneditable. Read each one relative
        // to the pack, the same way `look` itself is resolved — a name that is
        // not there is an error, because a pack whose hook silently vanished is
        // worse than one that refuses to load.
        for (point, file) in look.hooks.clone() {
            let script_path = dir.join(&file);
            match std::fs::read_to_string(&script_path) {
                Ok(script) => {
                    look.hooks.insert(point, script);
                }
                Err(error) => {
                    anyhow::bail!(
                        "pack '{}': hook '{}' names {}, which cannot be read: {error}",
                        manifest.name,
                        point,
                        script_path.display()
                    );
                }
            }
        }

        eprintln!(
            "[look] {}: {} theme(s), {} panel(s), {} label(s), css={}, {} hook point(s)",
            manifest.name,
            look.themes.len(),
            look.panels.len(),
            look.strings.len(),
            look.css.as_deref().map(str::len).unwrap_or(0),
            look.hooks.len(),
        );
        install_appearance(kernel, &manifest.name, look)?;
    }

    Ok(content)
}

/// The content every **installed, enabled** pack declares, in name order.
///
/// Recomputing a host's adopted state (after a load or an unload) needs exactly
/// this set: what the packs currently say, not what each individual load last
/// happened to fold in. Name order is `installed`'s own order, so the pass
/// reproduces the sequence a boot performs.
pub fn installed_content(packs_dir: &Path) -> Result<Vec<(String, PackContent)>> {
    let mut packs = crate::pack::installed(packs_dir)?;
    packs.sort_by(|a, b| a.manifest.name.cmp(&b.manifest.name));
    let mut out = Vec::new();
    for pack in packs.iter().filter(|pack| pack.enabled) {
        let Ok(manifest) = crate::pack::read_manifest(&pack.path) else {
            // A manifest that cannot be read is reported by listing paths
            // elsewhere; here it contributes nothing rather than aborting
            // the recomputation of every other pack.
            continue;
        };
        match read(&pack.path, &manifest) {
            Ok(content) => out.push((manifest.name, content)),
            Err(error) => eprintln!("[pack] {} content: {error:#}", manifest.name),
        }
    }
    Ok(out)
}

/// The state a host adopts from packs, recomputed from the enabled set.
///
/// [`apply`] folds one pack's content into a host incrementally, which means an
/// unload has no inverse: the pack that brought a route is gone but its route
/// stays. This is the set-shaped answer — recompute everything from *every*
/// enabled pack at once, so load and unload converge on the same state a fresh
/// boot would produce:
///
/// * `settings` starts at `baseline` (the user's own stored settings) and each
///   enabled pack folds over it in name order, exactly as boot does.
/// * `injections` starts at the user's standing rules
///   ([`InjectionEngine::load_default`]) and each pack's engine replaces it —
///   the same last-in-name-order-wins rule a boot arrives at.
/// * `persona` is the join over all enabled packs (see [`installed_persona`]).
#[derive(Debug, Clone)]
pub struct AdoptedContent {
    /// Settings with every enabled pack's fold applied over the baseline.
    pub settings: crate::settings::Settings,
    /// Injection rules: the standing file with each pack's engine applied over it.
    pub injections: crate::context::InjectionEngine,
    /// Joined persona of every enabled pack, or `None` when none carries one.
    pub persona: Option<String>,
}

impl AdoptedContent {
    /// Recompute the adopted state from `baseline` and the enabled packs.
    pub fn recompute(
        baseline: &crate::settings::Settings,
        packs_dir: &Path,
    ) -> Result<Self> {
        let mut settings = baseline.clone();
        let mut injections = crate::context::InjectionEngine::load_default();
        for (_, content) in installed_content(packs_dir)? {
            content.fold_settings(&mut settings);
            // Merged, not replaced. Assigning here meant only the last pack in
            // name order had its rules in force, and every other pack's
            // `injections.json` was silently dead — the file was present, the
            // pack was enabled, and the text never reached the model.
            if let Some(rules) = content.injections {
                injections.absorb(rules);
            }
        }
        let persona = installed_persona(packs_dir)?;
        Ok(Self {
            settings,
            injections,
            persona,
        })
    }
}

/// The persona every installed pack contributes, joined into one replace.///
/// The runtime holds a **single** persona slot. Applying each pack's soul
/// one by one would make load order decide who the model is — the alphabet
/// would decide it. The join is over the *set* of installed, enabled packs
/// instead, sorted by name, so every boot rebuilds the same text, installing
/// a pack changes exactly its own line, and unloading one removes its
/// contribution at the next adoption (the conversation history keeps what it
/// already absorbed — that part is the user's).
///
/// `None` when no pack carries a soul: hosts leave the standing persona
/// (including an explicit `--persona`) alone.
pub fn installed_persona(packs_dir: &std::path::Path) -> Result<Option<String>> {
    let mut packs = crate::pack::installed(packs_dir)?;
    packs.sort_by(|a, b| a.manifest.name.cmp(&b.manifest.name));
    let mut parts = Vec::new();
    for pack in packs {
        if !pack.enabled {
            continue;
        }
        let Some(file) = &pack.manifest.soul else {
            continue;
        };
        let path = pack.path.join(file.path());
        if !path.is_file() {
            // A declared-but-missing soul is the pack's own error; `apply`
            // reports it when content loads. Here it simply contributes
            // nothing rather than failing everyone's persona.
            continue;
        }
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("reading the persona from {}", path.display()))?;
        if !text.trim().is_empty() {
            parts.push(text.trim().to_string());
        }
    }
    if parts.is_empty() {
        Ok(None)
    } else {
        Ok(Some(parts.join("\n\n---\n\n")))
    }
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
    if look.themes.is_empty()
        && look.panels.is_empty()
        && look.strings.is_empty()
        && look.css.as_deref().map_or(true, str::is_empty)
        && look.hooks.is_empty()
        && look.types.is_empty()
    {
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
            // Labels rather than colours: applied by the shell, which does not
            // need to know which language they are in.
            if !self.look.strings.is_empty() {
                contributions = contributions.strings(self.look.strings.clone());
            }
            // Rules after the interface's own, so a pack can restyle without
            // replacing the page it restyles.
            if let Some(rules) = self.look.css.as_deref() {
                if !rules.trim().is_empty() {
                    contributions = contributions.css(rules);
                }
            }
            // Scripts on named points: the pack extends a rendering step
            // instead of replacing the page.
            if !self.look.hooks.is_empty() {
                contributions = contributions.hooks(self.look.hooks.clone());
            }
            // Panel kinds: the pack defines a name, panels elsewhere ask for it
            // by name instead of shipping their own renderer each time.
            if !self.look.types.is_empty() {
                contributions = contributions.panel_types(self.look.types.clone());
            }
            Ok(contributions)
        }
    }

    let name = format!("pack:{pack_name}");
    kernel.define(std::sync::Arc::new(Appearance {
        name: name.clone(),
        look,
    }));
    // Originated by the pack itself: this fiber *is* the pack's appearance,
    // so `plugin_origins` points it straight back at the manifest name.
    kernel.load_owned(
        &name,
        crate::plugin::RealmMap::new(),
        serde_json::Value::Null,
        Some(pack_name.to_string()),
    )?;
    Ok(())
}
