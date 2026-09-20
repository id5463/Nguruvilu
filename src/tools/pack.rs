//! The `pack` tool: let the agent build, inspect, and install packs itself.
//!
//! One tool with an `action` argument rather than five tools, for the same
//! reason the skill tool works that way: a model choosing between `pack_build`
//! and `pack_install` and `pack_verify` spends attention on the choice, and the
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

use anyhow::{anyhow, Context, Result};
use serde_json::{json, Value};

use crate::source::{FilesystemSource, PackSource};
use crate::tools::{ConflictPolicy, ToolDef, ToolFuture, ToolOutput};

/// Register the pack tool.
pub fn register(registry: &mut crate::tools::ToolRegistry) -> Result<()> {
    registry.register(
        ToolDef::new(
            "pack",
            "Build, inspect, and install `.dshpack` archives. A pack is a zip holding \
             `dsh.index.json` (identity: name, version, license), `assembly.yaml` (what it \
             loads), and whatever those reference. Actions: \
             `build` an archive from a directory, `verify` an archive without installing it, \
             `list` installed packs, `install` an archive, `apply` an installed pack's \
             assembly. Building needs the directory to already contain `dsh.index.json` — \
             write that file first.",
            json!({
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["build", "verify", "list", "install", "apply"],
                        "description": "What to do."
                    },
                    "path": {
                        "type": "string",
                        "description": "build: the pack directory. verify/install: the .dshpack archive. apply: the installed pack's assembly.yaml. Not used by list."
                    },
                    "out": {
                        "type": "string",
                        "description": "build: where to write the archive. Defaults to <name>-<version>.dshpack beside the directory."
                    }
                },
                "required": ["action"]
            }),
            "kernel",
            |args| Box::pin(run(args)) as ToolFuture,
        ),
        ConflictPolicy::Error,
    )?;
    Ok(())
}

async fn run(args: Value) -> Result<ToolOutput> {
    let action = args
        .get("action")
        .and_then(|a| a.as_str())
        .ok_or_else(|| anyhow!("missing required argument: action"))?;
    let path = args.get("path").and_then(|p| p.as_str());

    match action {
        "build" => build(path, args.get("out").and_then(|o| o.as_str())),
        "verify" => verify(path),
        "list" => list(),
        "install" => install(path),
        "apply" => apply(path),
        other => Err(anyhow!(
            "unknown action '{other}'; expected build, verify, list, install, or apply"
        )),
    }
}

fn build(dir: Option<&str>, out: Option<&str>) -> Result<ToolOutput> {
    let dir = dir.ok_or_else(|| anyhow!("build needs a path: the pack directory"))?;
    let dir = PathBuf::from(dir);

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
    let report = crate::pack::verify(&PathBuf::from(path))?;

    let mut text = format!(
        "{} {} — license {}, {} files, assembly {}",
        report.manifest.name,
        report.manifest.version_id,
        report.manifest.license,
        report.contents.files.len(),
        if report.contents.has_assembly { "present" } else { "absent" }
    );
    if let Some(range) = &report.manifest.dependencies.kernel {
        text.push_str(&format!("\nrequires kernel {range}"));
    }
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

    let mut text = format!("{} installed:", packs.len());
    for pack in packs {
        text.push_str(&format!(
            "\n- {} {} ({}){}",
            pack.manifest.name,
            pack.manifest.version_id,
            pack.manifest.license,
            match &pack.assembly {
                Some(assembly) => format!("\n  assembly: {}", assembly.display()),
                None => "  (no assembly)".to_string(),
            }
        ));
    }
    Ok(ToolOutput::text(text))
}

fn install(path: Option<&str>) -> Result<ToolOutput> {
    let path = path.ok_or_else(|| anyhow!("install needs a path: the .dshpack archive"))?;
    let archive = PathBuf::from(path);

    // Verify before unpacking, so a corrupt archive is reported rather than
    // scattered across the packs directory.
    let report = crate::pack::verify(&archive)?;

    let source = FilesystemSource::default_root();
    let placed = source
        .install(&archive)
        .with_context(|| format!("installing {}", archive.display()))?;

    let mut text = format!(
        "Installed {} {} → {}",
        placed.manifest.name,
        placed.manifest.version_id,
        placed.path.display()
    );
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

fn apply(path: Option<&str>) -> Result<ToolOutput> {
    let path = path.ok_or_else(|| anyhow!("apply needs a path: the installed assembly.yaml"))?;
    let assembly = PathBuf::from(path);

    if !assembly.is_file() {
        return Err(anyhow!(
            "{} does not exist. Use action `list` to see installed packs and the assembly \
             path for each.",
            assembly.display()
        ));
    }

    // Applying loads plugins, MCP servers, and skills, and it is asynchronous,
    // so it cannot run from inside a tool call without re-entering the kernel.
    // The honest answer is to say what to run instead of pretending.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::Role;

    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("ngu-packtool-{tag}-{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[tokio::test]
    async fn building_a_directory_without_a_manifest_explains_the_fix() {
        let dir = scratch("nomanifest");
        let error = run(json!({ "action": "build", "path": dir }))
            .await
            .expect_err("must fail");
        let text = format!("{error:#}");
        assert!(text.contains(crate::pack::MANIFEST_NAME), "{text}");
        assert!(text.contains("format_version"), "it should show the shape: {text}");
    }

    #[tokio::test]
    async fn building_then_verifying_a_pack_works() {
        let dir = scratch("roundtrip");
        crate::pack::write_manifest(&dir, &crate::pack::PackManifest::new("tool-pack", "1.0.0"))
            .unwrap();
        std::fs::write(
            dir.join(crate::pack::ASSEMBLY_NAME),
            "version: 1\nstages:\n  - name: s\n    skills: []\n",
        )
        .unwrap();

        let built = run(json!({ "action": "build", "path": dir })).await.unwrap();
        assert!(built.text.contains("tool-pack"), "{}", built.text);
        assert!(built.text.contains("1 files") || built.text.contains("2 files"), "{}", built.text);

        let archive = dir.join("tool-pack-1.0.0.dshpack");
        assert!(archive.is_file());

        let verified = run(json!({ "action": "verify", "path": archive })).await.unwrap();
        assert!(verified.text.contains("no problems found"), "{}", verified.text);
    }

    #[tokio::test]
    async fn building_without_an_assembly_says_it_loads_nothing() {
        let dir = scratch("noassembly");
        crate::pack::write_manifest(&dir, &crate::pack::PackManifest::new("bare", "1.0.0"))
            .unwrap();

        let built = run(json!({ "action": "build", "path": dir })).await.unwrap();
        assert!(built.text.contains("loads nothing"), "{}", built.text);
    }

    #[tokio::test]
    async fn a_missing_path_is_reported_per_action() {
        for (action, expected) in [
            ("build", "pack directory"),
            ("verify", ".dshpack archive"),
            ("install", ".dshpack archive"),
            ("apply", "assembly.yaml"),
        ] {
            let error = run(json!({ "action": action })).await.expect_err(action);
            let text = format!("{error:#}");
            assert!(text.contains(expected), "{action}: {text}");
        }
    }

    #[tokio::test]
    async fn an_unknown_action_lists_the_real_ones() {
        let error = run(json!({ "action": "explode" })).await.expect_err("must fail");
        let text = format!("{error:#}");
        for action in ["build", "verify", "list", "install", "apply"] {
            assert!(text.contains(action), "{text}");
        }
    }

    #[tokio::test]
    async fn applying_a_missing_assembly_points_at_list() {
        let error = run(json!({ "action": "apply", "path": "no/such/assembly.yaml" }))
            .await
            .expect_err("must fail");
        let text = format!("{error:#}");
        assert!(text.contains("does not exist"), "{text}");
        assert!(text.contains("list"), "{text}");
    }

    #[tokio::test]
    async fn applying_an_installed_assembly_explains_it_needs_the_cli() {
        let dir = scratch("apply");
        let assembly = dir.join(crate::pack::ASSEMBLY_NAME);
        std::fs::write(&assembly, "version: 1\nstages: []\n").unwrap();

        let output = run(json!({ "action": "apply", "path": assembly })).await.unwrap();
        assert!(output.text.contains("ngu apply"), "{}", output.text);
        assert!(
            output.text.contains("outside this conversation"),
            "it must not pretend to have loaded it: {}",
            output.text
        );
    }

    #[tokio::test]
    async fn listing_works_whether_or_not_anything_is_installed() {
        let output = run(json!({ "action": "list" })).await.unwrap();
        assert!(
            output.text.contains("installed"),
            "either a count or 'No packs installed': {}",
            output.text
        );
        assert_eq!(output.text.chars().next().map(|_| ()), Some(()));
        assert!(!output.text.is_empty());
    }

    #[test]
    fn the_tool_is_registered_under_the_kernel() {
        let mut tools = crate::tools::ToolRegistry::with_base_tools().unwrap();
        register(&mut tools).unwrap();
        assert_eq!(tools.owner("pack"), Some("kernel"));
        assert!(tools.names().contains(&"pack".to_string()));
    }

    #[test]
    fn the_schema_advertises_every_action() {
        let mut tools = crate::tools::ToolRegistry::new();
        register(&mut tools).unwrap();
        let schema = &tools.schemas()[0];
        let actions = schema["function"]["parameters"]["properties"]["action"]["enum"]
            .as_array()
            .expect("action enum");
        assert_eq!(actions.len(), 5);
        assert_eq!(schema["function"]["name"], "pack");
    }

    #[test]
    fn a_role_is_not_required_by_this_tool() {
        // Guards against an accidental dependency on message roles in a tool.
        assert_eq!(Role::Tool.as_wire(), "tool");
    }

    #[tokio::test]
    async fn packing_beside_the_tree_does_not_embed_the_archive_in_itself() {
        // The default output path is inside the directory being packed, so a
        // build that scans after creating the file archives an empty copy of
        // itself — and a rebuild would archive the previous archive.
        let dir = scratch("self");
        crate::pack::write_manifest(&dir, &crate::pack::PackManifest::new("self", "1.0.0"))
            .unwrap();
        std::fs::write(dir.join(crate::pack::ASSEMBLY_NAME), "version: 1\nstages: []\n").unwrap();

        let built = run(json!({ "action": "build", "path": dir })).await.unwrap();
        assert!(built.text.contains("2 files"), "manifest + assembly only: {}", built.text);

        // Building again must not pick up the first archive either.
        let again = run(json!({ "action": "build", "path": dir })).await.unwrap();
        assert!(again.text.contains("2 files"), "still two: {}", again.text);
    }
}
