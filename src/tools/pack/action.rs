//! The `pack` tool: one tool with an `action` argument.
//!
//! One tool with actions rather than five tools, for the same reason the skill
//! tool works that way: a model choosing between `pack_build` and
//! `pack_install` and `pack_verify` spends attention on the choice, and the
//! actions share almost all of their description.
//!
//! # Why packing is a tool but installing is guarded
//!
//! Building an archive writes a file in the workspace. Installing one places
//! code where the kernel will load it, so it is the step that can change what
//! the agent itself can do. It is still available — refusing outright would make
//! the tool useless for its stated purpose — but it is a separate action with
//! its own description, and every install is reported with the path it landed
//! at, so the transcript shows what happened.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use serde_json::Value;
use crate::tools::{current_session, current_workspace, ToolOutput};

use super::pending::{Pending, PendingQueue};

/// Test helper: a call with no host attached (disk-only actions).

pub(super) async fn run_with(args: Value, pending: Option<Arc<PendingQueue>>) -> Result<ToolOutput> {
    let action = args
        .get("action")
        .and_then(|a| a.as_str())
        .ok_or_else(|| anyhow!("missing required argument: action"))?;
    let path = args.get("path").and_then(|p| p.as_str());
    let name = args.get("name").and_then(|n| n.as_str());

    match action {
        "build" => build(path, args.get("out").and_then(|o| o.as_str())),
        "verify" => verify(path),
        "list" => list(),
        "install" => install(path, pending.as_ref()).await,
        "load" => set_loaded(name, true, pending.as_ref()),
        "unload" => set_loaded(name, false, pending.as_ref()),
        "apply" => apply(path, pending.as_ref()),
        other => Err(anyhow!(
            "unknown action '{other}'; expected build, verify, list, install, load, unload, or apply"
        )),
    }
}

/// The path a `pack` action was given, resolved the way every other tool
/// resolves one: against the session's workspace, not the process directory.
///
/// Every other file-facing tool starts from `tools::current_workspace()` (see
/// `base::default_base_dir`). This tool used to hand `PathBuf::from` straight to
/// the installer, so a relative archive path meant "wherever the server happened
/// to be started" — the one tool that ignored the identity the rest of them
/// obey, which is exactly how a served session ends up reading a pack file out
/// of the host's own directory.
fn resolve(raw: &str) -> PathBuf {
    let path = PathBuf::from(raw);
    if path.is_absolute() {
        return path;
    }
    current_workspace()
        .map(|workspace| workspace.join(&path))
        .unwrap_or(path)
}

fn build(dir: Option<&str>, out: Option<&str>) -> Result<ToolOutput> {
    let dir = dir.ok_or_else(|| anyhow!("build needs a path: the pack directory"))?;
    let dir = resolve(dir);

    // Report a missing manifest as what it is, with the fix, rather than as a
    // bare filesystem error: this is the mistake a first attempt makes.
    if !dir.join(crate::pack::MANIFEST_NAME).is_file() {
        return Err(anyhow!(
            "{} has no {}; a pack needs one before it can be built. \
             Write it with at least: {{\"format_version\":1,\"game\":\"dsh\",\
             \"name\":\"...\",\"version_id\":\"...\",\"license\":\"...\",\
             \"kernel_version\":\"...\"}}",
            dir.display(),
            crate::pack::MANIFEST_NAME
        ));
    }

    let manifest = crate::pack::read_manifest(&dir)?;
    let archive = match out {
        Some(out) => PathBuf::from(out),
        None => dir.join(format!("{}-{}.dshpack", manifest.name, manifest.version_id)),
    };

    // Counting goes through the same exclusion pack uses, so a rebuild does
    // not report the previous archive as a member.
    let contents = crate::pack::packable_files(&dir, &archive)?;
    let packed = crate::pack::pack(&dir, &archive)?;

    let mut text = format!(
        "Built {} {} → {} ({} files)",
        packed.name,
        packed.version_id,
        archive.display(),
        contents.files.len()
    );
    if !contents.has_assembly {
        text.push_str(&format!(
            "\nNote: no {} in the pack, so it identifies itself but loads nothing.",
            crate::pack::ASSEMBLY_NAME
        ));
    }
    Ok(ToolOutput::text(text))
}

fn verify(path: Option<&str>) -> Result<ToolOutput> {
    let path = path.ok_or_else(|| anyhow!("verify needs a path: the .dshpack archive"))?;
    let report = crate::pack::verify(&resolve(path))?;

    let mut text = format!(
        "{} {} — license {}, {} files, assembly {}",
        report.manifest.name,
        report.manifest.version_id,
        report.manifest.license,
        report.contents.files.len(),
        if report.contents.has_assembly { "present" } else { "absent" }
    );
    text.push_str(&format!(
        "\nrequires nguruvilu {}",
        report.manifest.dependencies.nguruvilu
    ));
    for warning in &report.warnings {
        text.push_str(&format!("\nwarning: {warning}"));
    }
    if report.warnings.is_empty() {
        text.push_str("\nno problems found");
    }
    Ok(ToolOutput::text(text))
}

fn list() -> Result<ToolOutput> {
    let source = FilesystemSource::default_root();
    let packs = source.list()?;
    if packs.is_empty() {
        return Ok(ToolOutput::text(format!(
            "No packs installed in {}",
            source.root().map(|p| p.display().to_string()).unwrap_or_default()
        )));
    }

    let loaded = packs.iter().filter(|p| p.enabled).count();
    let mut text = format!("{} installed ({loaded} loaded):", packs.len());
    for pack in packs {
        text.push_str(&format!(
            "\n- {} {} ({}){}{}",
            pack.manifest.name,
            pack.manifest.version_id,
            pack.manifest.license,
            if pack.enabled { " [loaded]" } else { " [unloaded, files kept]" },
            match &pack.assembly {
                Some(assembly) => format!("\n  assembly: {}", assembly.display()),
                None => "  (no assembly)".to_string(),
            }
        ));
    }
    Ok(ToolOutput::text(text))
}

/// Installed packs matching `name` or `name@version`.
fn installed_matching(wanted: &str) -> Result<Vec<crate::pack::InstalledPack>> {
    let (base, version) = match wanted.split_once('@') {
        Some((base, version)) => (base, Some(version)),
        None => (wanted, None),
    };
    let packs = FilesystemSource::default_root().list()?;
    let matches: Vec<_> = packs
        .into_iter()
        .filter(|pack| pack.manifest.name == base)
        .filter(|pack| version.map_or(true, |v| pack.manifest.version_id == v))
        .collect();

    if matches.is_empty() {
        // Naming what is installed turns a dead end into a next step, the same
        // way the CLI's uninstall does.
        let installed = FilesystemSource::default_root().list()?;
        anyhow::bail!(
            "no installed pack matches '{wanted}'.{}",
            if installed.is_empty() {
                " Nothing is installed.".to_string()
            } else {
                format!(
                    " Installed: {}",
                    installed
                        .iter()
                        .map(|p| format!("{}@{}", p.manifest.name, p.manifest.version_id))
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            }
        );
    }
    Ok(matches)
}

/// Load or unload an installed pack for this conversation.
///
/// Unloading keeps the files: it takes the pack *out of the conversation*,
/// which is a different act from deleting it.
fn set_loaded(
    wanted: Option<&str>,
    enable: bool,
    pending: Option<&Arc<PendingQueue>>,
) -> Result<ToolOutput> {
    let wanted = wanted.ok_or_else(|| {
        anyhow!(
            "{} needs a `name`: which installed pack to {}",
            if enable { "load" } else { "unload" },
            if enable { "load" } else { "unload" }
        )
    })?;

    let matches = installed_matching(wanted)?;
    let mut text = format!(
        "{} {wanted}:",
        if enable { "loaded" } else { "unloaded" }
    );
    for pack in &matches {
        let was = crate::pack::set_enabled(&pack.path, enable)?;
        text.push_str(&format!(
            "\n- {}@{} {} → {}",
            pack.manifest.name,
            pack.manifest.version_id,
            if was == enable { "was already" } else { "now" },
            if enable { "loaded" } else { "unloaded" }
        ));
        text.push_str(&format!("\n  files: {}", pack.path.display()));
    }

    let label = matches[0].manifest.name.clone();
    match pending {
        Some(queue) => {
            let session = current_session();
            queue.push(if enable {
                Pending::Load { name: label, session }
            } else {
                Pending::Unload { name: label, session }
            });
            text.push_str(if enable {
                "\nThis conversation picks it up at the end of this turn."
            } else {
                "\nThis conversation drops what it brought at the end of this turn: \
                 its tools stop being callable from then on."
            });
        }
        None => {
            text.push_str(
                "\nNo host is attached to this tool, so only the state changed; \
                 it takes effect from the next start.",
            );
        }
    }
    Ok(ToolOutput::text(text))
}

fn apply(path: Option<&str>, pending: Option<&Arc<PendingQueue>>) -> Result<ToolOutput> {
    let path = path.ok_or_else(|| anyhow!("apply needs a path: the installed assembly.yaml"))?;
    let assembly = resolve(path);

    if !assembly.is_file() {
        return Err(anyhow!(
            "{} does not exist. Use action `list` to see installed packs and the assembly \
             path for each.",
            assembly.display()
        ));
    }

    // With a host attached, applying is a real load: the pack is enabled and
    // the host brings it in at the end of this turn. Hot loading is the whole
    // point of the action existing as a verb rather than as an instruction.
    if let Some(queue) = pending {
        let dir = assembly
            .parent()
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("."));
        let name = crate::pack::read_manifest(&dir)
            .map(|manifest| manifest.name)
            .unwrap_or_else(|_| {
                dir.file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_else(|| "pack".to_string())
            });
        if let Ok(was) = crate::pack::set_enabled(&dir, true) {
            let _ = was;
        }
        queue.push(Pending::Load { name: name.clone(), session: current_session() });
        return Ok(ToolOutput::text(format!(
            "Queued {name}: it loads into this conversation at the end of this turn."
        )));
    }

    // Without one, the honest answer is to say what to run instead of
    // pretending.
    Ok(ToolOutput::text(format!(
        "To load {}: the kernel loads packs at startup or through the CLI, not from inside \
         a tool call. Run:\n\n    ngu apply \"{}\"\n\nor start the session with:\n\n    \
         ngu --assembly \"{}\"\n\nThe pack is installed and its assembly is valid; loading it \
         is the one step that has to happen outside this conversation.",
        assembly.display(),
        assembly.display(),
        assembly.display()
    )))
}

async fn install(path: Option<&str>, pending: Option<&Arc<PendingQueue>>) -> Result<ToolOutput> {
    let path = path.ok_or_else(|| anyhow!("install needs a path: the .dshpack archive"))?;
    let archive = resolve(path);

    // Verify before unpacking, so a corrupt archive is reported rather than
    // scattered across the packs directory.
    let report = crate::pack::verify(&archive)?;

    let source = FilesystemSource::default_root();
    let placed = source
        .install(&archive)
        .await
        .with_context(|| format!("installing {}", archive.display()))?;

    // Installing is a load: a pack the user had unloaded and then reinstalled
    // is a pack the user just asked for.
    crate::pack::set_enabled(&placed.path, true)?;

    // The host's pack list is a snapshot; without this it keeps showing the
    // state from before the install, which reads as an install that never
    // happened.
    if let Some(queue) = pending {
        queue.push(Pending::Installed {
            name: placed.manifest.name.clone(),
            version: placed.manifest.version_id.clone(),
        });
    }

    let mut text = format!(
        "Installed {} {} → {}",
        placed.manifest.name,
        placed.manifest.version_id,
        placed.path.display()
    );
    if pending.is_some() {
        text.push_str("\nThe host was told; its pack list now includes it.");
    }
    match &placed.assembly {
        Some(assembly) => {
            text.push_str(&format!(
                "\nAssembly: {}\nRun `pack` with action `apply` on that path to load it.",
                assembly.display()
            ));
        }
        None => text.push_str("\nThis pack has no assembly.yaml, so it loads nothing."),
    }
    for warning in &report.warnings {
        text.push_str(&format!("\nwarning: {warning}"));
    }
    Ok(ToolOutput::text(text))
}

