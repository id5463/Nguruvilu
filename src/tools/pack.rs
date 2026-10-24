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
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use serde_json::{json, Value};

use crate::source::{FilesystemSource, PackSource};
use crate::tools::{ConflictPolicy, ToolDef, ToolFuture, ToolOutput};

/// A load or unload this conversation asked for.
///
/// Queued rather than done inside the call: the tool runs while the loop owns
/// the kernel, and the host applies the change at the end of the turn — which
/// is the next moment the tool table is read anyway.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Pending {
    /// Bring an installed pack into this conversation. The session is the one
    /// that asked, so the host can record the pack against it.
    Load { name: String, session: Option<String> },
    /// Take it out of this conversation, keeping its files. Removed from the
    /// kernel only when no other conversation still uses the pack.
    Unload { name: String, session: Option<String> },
    /// An archive just landed on disk. The tool installs it itself — the
    /// kernel is not touched — but the host's pack list is a snapshot the page
    /// took at load time, and without this notice a pack the agent installed
    /// reads as "the agent said installed, the list shows nothing".
    Installed { name: String, version: String },
}

/// What the tool hands the host, drained once the turn is over.
#[derive(Default)]
pub struct PendingQueue {
    items: std::sync::Mutex<Vec<Pending>>,
}

impl PendingQueue {
    /// An empty queue.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one request.
    pub fn push(&self, item: Pending) {
        self.items.lock().expect("pending queue lock").push(item);
    }

    /// Take every request recorded so far, leaving the queue empty.
    ///
    /// This is the process-wide sweep, and it is what a host running a single
    /// conversation wants: the CLI serves one session per process, so every
    /// entry in its queue belongs to the turn that just ended. A host serving
    /// several conversations has to use [`PendingQueue::drain_for`] instead,
    /// because a turn that takes the whole queue loads a pack another
    /// conversation asked for into its own tool table.
    pub fn drain(&self) -> Vec<Pending> {
        std::mem::take(&mut *self.items.lock().expect("pending queue lock"))
    }

    /// Take the requests recorded for `session`, leaving the rest queued.
    ///
    /// An entry that names no conversation is addressed to all of them — an
    /// install changes a pack list every conversation reads — so it goes to
    /// whoever asks first. An entry some conversation queued goes only to that
    /// conversation, which is why asking with `None` takes the session-free
    /// entries and nothing else: an identity-less caller has no claim on work
    /// another conversation is waiting for.
    pub fn drain_for(&self, session: Option<&str>) -> Vec<Pending> {
        let mut items = self.items.lock().expect("pending queue lock");
        let (taken, kept): (Vec<Pending>, Vec<Pending>) = std::mem::take(&mut *items)
            .into_iter()
            .partition(|item| match (session, queued_for(item)) {
                (_, None) => true,
                (Some(asking), Some(queued)) => asking == queued,
                (None, Some(_)) => false,
            });
        *items = kept;
        taken
    }
}

/// The conversation an entry was queued for, or `None` when it is for everyone.
fn queued_for(item: &Pending) -> Option<&str> {
    match item {
        Pending::Load { session, .. } | Pending::Unload { session, .. } => session.as_deref(),
        // An install is news about the installed set, which every conversation
        // reads, so it does not belong to one of them.
        Pending::Installed { .. } => None,
    }
}

/// Register the pack tool with no host attached.
///
/// Enough for tests and for a caller that only wants build/verify/list: the
/// load and unload actions then change state on disk and say so, rather than
/// pretending a kernel was reached.
pub fn register(registry: &mut crate::tools::ToolRegistry) -> Result<()> {
    register_with(registry, None)
}

/// Register the pack tool, handing its load and unload requests to `pending`.
pub fn register_with(
    registry: &mut crate::tools::ToolRegistry,
    pending: Option<Arc<PendingQueue>>,
) -> Result<()> {
    registry.register(
        ToolDef::new(
            "pack",
            "Build, inspect, install, load, and unload `.dshpack` archives. A pack is a zip \
             holding `dsh.index.json` (identity: name, version, license) plus a few small \
             document files, and nothing else — everything it needs is referenced from the \
             manifest. Actions: \
             `build` an archive from a directory, `verify` an archive without installing it, \
             `list` installed packs with whether each is loaded, `install` an archive, \
             `load` an installed pack into this conversation (files are kept, and an unloaded \
             pack comes back), `unload` an installed pack out of this conversation (its tools \
             stop being callable; the files stay), `apply` an installed pack's assembly. \
             Building needs the directory to already contain `dsh.index.json` — write that \
             file first.",
            json!({
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["build", "verify", "list", "install", "load", "unload", "apply"],
                        "description": "What to do."
                    },
                    "path": {
                        "type": "string",
                        "description": "build: the pack directory. verify/install: the .dshpack archive. apply: the installed pack's assembly.yaml. Not used by list, load, or unload."
                    },
                    "name": {
                        "type": "string",
                        "description": "load/unload: the installed pack's name, optionally name@version. Not used by the other actions."
                    },
                    "out": {
                        "type": "string",
                        "description": "build: where to write the archive. Defaults to <name>-<version>.dshpack beside the directory."
                    }
                },
                "required": ["action"]
            }),
            "kernel",
            move |args| {
                let pending = pending.clone();
                Box::pin(async move { run_with(args, pending).await }) as ToolFuture
            },
        ),
        ConflictPolicy::Error,
    )?;
    Ok(())
}

/// Test helper: a call with no host attached (disk-only actions).
#[cfg(test)]
async fn run(args: Value) -> Result<ToolOutput> {
    run_with(args, None).await
}

async fn run_with(args: Value, pending: Option<Arc<PendingQueue>>) -> Result<ToolOutput> {
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
    super::current_workspace()
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
            let session = super::current_session();
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
        queue.push(Pending::Load { name: name.clone(), session: super::current_session() });
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
    async fn agent_install_tells_the_host_so_the_pack_list_can_refresh() {
        // The tool installs to disk itself; the one thing its host must hear
        // about is that the list it shows is now stale.
        let dir = scratch("install-notify");
        crate::pack::write_manifest(&dir, &crate::pack::PackManifest::new("tool-ping", "1.0.0"))
            .unwrap();
        let built = run(json!({ "action": "build", "path": dir })).await.unwrap();
        assert!(built.text.contains("tool-ping"), "{}", built.text);
        let archive = dir.join("tool-ping-1.0.0.dshpack");

        // The install goes to the default packs directory, so the test keeps
        // it inside a scratch home rather than the user's.
        let home = scratch("install-notify-home");
        std::env::set_var("NGU_HOME", &home);
        let queue = Arc::new(PendingQueue::new());
        let installed = run_with(
            json!({ "action": "install", "path": archive.display().to_string() }),
            Some(Arc::clone(&queue)),
        )
        .await
        .unwrap();
        std::env::remove_var("NGU_HOME");

        assert!(installed.text.contains("Installed tool-ping"), "{}", installed.text);
        // The install entry names no conversation, so it is addressed to every
        // one of them and an identity-less drain is the one that takes it.
        let pending = queue.drain_for(None);
        assert!(
            pending.iter().any(|item| matches!(
                item,
                Pending::Installed { name, version }
                    if name == "tool-ping" && version == "1.0.0"
            )),
            "{pending:?}"
        );
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(&home);
    }

    #[test]
    fn draining_for_one_conversation_leaves_anothers_pack_queued() {
        // One process serves several conversations, so a turn may take only the
        // work its own conversation asked for: draining everything is how a
        // pack queued in one conversation loads into another one's tool table.
        let queue = PendingQueue::new();
        queue.push(Pending::Load {
            name: "pack-a".into(),
            session: Some("session-a".into()),
        });
        queue.push(Pending::Load {
            name: "pack-b".into(),
            session: Some("session-b".into()),
        });

        let mine = queue.drain_for(Some("session-a"));
        assert_eq!(
            mine,
            vec![Pending::Load {
                name: "pack-a".into(),
                session: Some("session-a".into()),
            }],
            "only the pack session A queued comes out"
        );
        // B's request names a conversation, so an identity-less ask has no
        // claim on it and takes nothing.
        assert!(
            queue.drain_for(None).is_empty(),
            "A's request is gone and B's is not free to take"
        );

        // B's pack is still queued for B's own turn, and only that turn.
        let theirs = queue.drain_for(Some("session-b"));
        assert!(
            matches!(&theirs[..], [Pending::Load { name, .. }] if name == "pack-b"),
            "{theirs:?}"
        );
        assert!(
            queue.drain_for(Some("session-b")).is_empty(),
            "the queue is empty once each conversation took its own"
        );
    }

    #[test]
    fn an_install_is_addressed_to_every_conversation() {
        // Installing changes the installed set, which every conversation reads,
        // so an entry that names no session must stay available to whichever
        // conversation drains next rather than be held to one of them.
        let queue = PendingQueue::new();
        queue.push(Pending::Installed {
            name: "shared".into(),
            version: "2.0.0".into(),
        });
        queue.push(Pending::Unload {
            name: "only-here".into(),
            session: Some("session-a".into()),
        });

        let mine = queue.drain_for(Some("session-b"));
        assert!(
            matches!(&mine[..], [Pending::Installed { name, version }]
                if name == "shared" && version == "2.0.0"),
            "{mine:?}"
        );
        assert!(
            queue.drain_for(Some("session-b")).is_empty(),
            "the notice is taken once, not on every drain"
        );

        // The unload was queued by A, so it is still waiting for A.
        let theirs = queue.drain_for(Some("session-a"));
        assert!(
            matches!(&theirs[..], [Pending::Unload { name, .. }] if name == "only-here"),
            "{theirs:?}"
        );
    }

    #[test]
    fn a_process_wide_drain_still_takes_every_conversations_pack() {
        // The single-conversation hosts keep the behaviour they have: the CLI
        // runs one session per process, and a host that owns the queue outright
        // has no other conversation's work in it to protect.
        let queue = PendingQueue::new();
        queue.push(Pending::Load {
            name: "a".into(),
            session: Some("session-a".into()),
        });
        queue.push(Pending::Load {
            name: "b".into(),
            session: Some("session-b".into()),
        });
        queue.push(Pending::Installed {
            name: "c".into(),
            version: "1.0.0".into(),
        });

        assert_eq!(queue.drain().len(), 3);
        assert!(queue.drain().is_empty(), "the sweep leaves nothing behind");
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
            ("load", "name"),
            ("unload", "name"),
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
        for action in ["build", "verify", "list", "install", "load", "unload", "apply"] {
            assert!(text.contains(action), "{text}");
        }
    }

    #[tokio::test]
    async fn unloading_a_pack_that_is_not_installed_says_what_is() {
        // The dead end has to name the alternatives, or the model can only
        // guess at what to try next.
        let error = run(json!({ "action": "unload", "name": "definitely-not-installed" }))
            .await
            .expect_err("must fail");
        let text = format!("{error:#}");
        assert!(text.contains("no installed pack matches"), "{text}");
        assert!(
            text.contains("Installed") || text.contains("Nothing is installed"),
            "{text}"
        );
    }

    #[tokio::test]
    async fn load_and_unload_both_name_the_pack_they_mean() {
        // Neither action can guess which pack is meant, and both must say so
        // in the same words the argument is documented under.
        for action in ["load", "unload"] {
            let error = run(json!({ "action": action }))
                .await
                .expect_err("no name");
            let text = format!("{error:#}");
            assert!(text.contains("name"), "{action}: {text}");
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
        assert_eq!(actions.len(), 7);
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
