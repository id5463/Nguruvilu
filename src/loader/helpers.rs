//! 加载层用到的独立工具函数：路径解析、进程启动、命名与标签。
//!
//! 这些不挂在 [`super::model::Loader`] 上，因为它们不持有加载器的状态：
//! 一个路径怎么拼、一个程序去哪找、一步在报告里叫什么名字，都是纯函数。

use std::collections::{BTreeMap, HashMap};
use std::ffi::OsString;
use std::path::{Path, PathBuf};

use anyhow::Result;

use crate::assembly::Scope;
use crate::protocol::{self, Spec};
use crate::skills::{Skill, SkillRegistry};

use super::model::Loader;


/// What an unload took out of the kernel.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct UnloadReport {
    /// Plugin instances unloaded, by plugin name.
    pub plugins: Vec<String>,
    /// MCP servers taken out of the conversation: their tools are gone, and
    /// their processes were stopped here rather than left to whatever drops
    /// last. A server that could not be reached is in [`Self::notes`], with the
    /// reason for it.
    pub mcp: Vec<String>,
    /// Skill roots removed from the catalog.
    pub skills: Vec<String>,
    /// Anything that did not match, so an unload that did nothing says so
    /// rather than looking like it worked.
    pub notes: Vec<String>,
}

/// Unload everything one pack contributed, leaving its files where they are.
///
/// Unload and delete are different acts: this takes the pack *out of the
/// conversation* — its tools stop being callable — and keeps the installed
/// directory, so it can be loaded again without fetching anything.
///
/// `label` is the ledger's pack label (`name-version`); `name` is the
/// manifest's name, which is what the appearance fiber is registered under.
/// The ledger entries for the pack are removed as they are undone, so a later
/// load sees them as new work rather than as something already in place.
pub fn unload_pack(
    kernel: &mut Kernel,
    skills: &mut SkillRegistry,
    ledger_path: &Path,
    label: &str,
    name: &str,
) -> Result<UnloadReport> {
    let mut ledger = Ledger::load(ledger_path)?;
    let entries: Vec<LedgerEntry> = ledger
        .entries
        .iter()
        .filter(|entry| entry.pack == label)
        .cloned()
        .collect();

    let mut report = UnloadReport::default();

    for entry in &entries {
        match entry.kind.as_str() {
            "plugin" => {
                let plugin = plugin_name(&entry.source, &entry.id);
                let ids: Vec<u64> = kernel
                    .fibers()
                    .iter()
                    .filter(|fiber| fiber.plugin == plugin)
                    .map(|fiber| fiber.id)
                    .collect();
                if ids.is_empty() {
                    report
                        .notes
                        .push(format!("plugin '{plugin}' was not loaded here"));
                }
                for id in ids {
                    // Pack-plane unload is deliberately global and
                    // ledger-driven: a pack's resources are not per-session
                    // references, so this tears the fiber down for every session
                    // at once. `Kernel::release` is the other shape — a teardown
                    // counted against the sessions still holding a reference —
                    // and it is not what a ledger row means.
                    kernel.unload(id)?;
                    report.plugins.push(plugin.clone());
                }
                ledger.remove("plugin", &entry.id);
            }
            "mcp" => {
                // An MCP server has no fiber: its tools are registered
                // directly, so they are removed directly. Removing them drops
                // *a* handle on the client — not the last one, whatever the
                // old comment here assumed: whoever started the server keeps
                // one, and a call in flight keeps another. So the process is
                // stopped here, explicitly, through the bounded kill path, and
                // what it could not reach is reported rather than hoped about.
                let owner = format!("mcp:{}", entry.id);
                let tools: Vec<String> = kernel
                    .tools()
                    .names()
                    .into_iter()
                    .filter(|tool| kernel.tools().owner(tool).as_deref() == Some(owner.as_str()))
                    .collect();
                for tool in tools {
                    kernel.tools_mut().unregister(&tool);
                }
                let closed = crate::mcp::close_server(&entry.id);
                for failure in &closed.failed {
                    // A process this could not stop is said out loud, where the
                    // user sees the unload report, rather than left to be found
                    // later in a process list.
                    report
                        .notes
                        .push(format!("an MCP server was not stopped — {failure}"));
                }
                report.mcp.push(entry.id.clone());
                ledger.remove("mcp", &entry.id);
            }
            "skill" => {
                if let Some(path) = &entry.path {
                    skills.remove_root(Path::new(path));
                    report.skills.push(entry.id.clone());
                }
                ledger.remove("skill", &entry.id);
            }
            other => report
                .notes
                .push(format!("entry '{other}:{}' has no unload path", entry.id)),
        }
    }

    // The appearance a pack's look.json installs is a fiber of its own,
    // registered under the manifest name rather than the ledger label.
    let appearance = format!("pack:{name}");
    let ids: Vec<u64> = kernel
        .fibers()
        .iter()
        .filter(|fiber| fiber.plugin == appearance)
        .map(|fiber| fiber.id)
        .collect();
    for id in ids {
        // Same rule as the ledger rows above, for the appearance fiber: pack
        // unload is global and ledger-driven, not reference-counted per session,
        // so this one teardown takes the look away from every session at once.
        // (`Kernel::release` is the per-session form, and a ledger row is not a
        // session's claim.)
        kernel.unload(id)?;
        report.plugins.push(appearance.clone());
    }

    // The catalog tool holds a snapshot of the registry, so removing a root
    // only reaches the model once the tool is rebuilt over what is left — and
    // when nothing is left, no tool is the right answer rather than one that
    // advertises an empty catalog.
    kernel.tools_mut().unregister("skill");
    skills.scan()?;
    if !skills.is_empty() {
        crate::skills::register_skill_tool(kernel.tools_mut(), std::sync::Arc::new(skills.clone()))?;
    }

    if entries.is_empty() && report.plugins.is_empty() {
        report
            .notes
            .push(format!("nothing in the ledger came from '{label}'"));
    }

    // The pack's appearance is not in the ledger — `content::apply` loads it as
    // a built-in plugin named `pack:<name>`, which is provenance the ledger
    // never records. Unloading it here is what makes a pack that was copied
    // into the directory by hand (no install, so no ledger entry) still lose
    // its stylesheet, labels and hooks: without this the interface kept them
    // forever, because the ledger loop above had nothing to iterate over.
    //
    // Done last so the ledger's own unloads keep their order in the report.
    let appearance = format!("pack:{name}");
    let appearance_fibers: Vec<u64> = kernel
        .fibers()
        .iter()
        .filter(|fiber| fiber.plugin == appearance)
        .map(|fiber| fiber.id)
        .collect();
    for id in appearance_fibers {
        if kernel.unload(id).is_ok() {
            report.plugins.push(appearance.clone());
        }
    }

    ledger.save(ledger_path)?;
    Ok(report)
}

/// Load one pack's assembly into a kernel, returning the roots it added.
///
/// Takes the kernel by reference and gives it back on **every** path, including
/// a failure: a load that loses the kernel would leave the session with no
/// tools at all, which is far worse than a pack that failed to load. The
/// skills come back rather than being merged here so the caller never holds
/// its catalog across the awaits in this function; hand them straight to
/// [`merge_skills`].
pub async fn load_pack(
    kernel: &mut Kernel,
    ledger_path: &Path,
    assembly: &Path,
    label: &str,
) -> Result<(SkillRegistry, LoadReport, Vec<Arc<McpClient>>)> {
    let pack_dir = assembly
        .parent()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));

    let taken = std::mem::replace(kernel, Kernel::new());
    let mut loader = Loader::new(taken, ledger_path.to_path_buf(), label).with_pack_dir(&pack_dir);

    let loaded = async {
        let document = crate::assembly::Assembly::from_file(assembly)?;
        let plan = document.plan(crate::fetch::platform_tag())?;
        loader.apply(&plan).await
    }
    .await;

    let (returned, pack_skills, servers) = loader.finish().await;
    *kernel = returned;
    let report = loaded?;
    Ok((pack_skills, report, servers))
}

/// Merge a pack's skill roots into the standing catalog and rebuild the tool.
///
/// A path with its `.` components dropped.
///
/// Only `.` is removed. `..` is left alone on purpose: resolving it is a decision
/// about where a pack is allowed to point, and that decision is not this
/// function's to make — a `..` that escapes the pack directory has to stay
/// visible for whatever checks it rather than being silently folded here.
fn tidy(path: PathBuf) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            std::path::Component::CurDir => {}
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// The `skill` tool holds a snapshot of the registry, so a root that arrives
/// or leaves only reaches the model once the tool is registered over what is
/// there now. Without this, a pack's skills would *replace* the host's instead
/// of joining them.
pub fn merge_skills(
    kernel: &mut Kernel,
    host: &mut SkillRegistry,
    pack: SkillRegistry,
) -> Result<()> {
    for root in pack.roots() {
        host.add_root(root.clone());
    }
    host.scan()?;
    kernel.tools_mut().unregister("skill");
    if !host.is_empty() {
        crate::skills::register_skill_tool(kernel.tools_mut(), Arc::new(host.clone()))?;
    }
    Ok(())
}

/// Resolve a plugin's registered name from its source string.
fn plugin_name(source: &str, fallback: &str) -> String {
    if let Some(rest) = source.strip_prefix("builtin:") {
        return rest.trim().to_string();
    }
    if source.trim().is_empty() {
        return fallback.to_string();
    }
    // A bare name that is not a path or URL is taken as the plugin name.
    if !source.contains('/') && !source.contains(':') && !source.contains('@') {
        return source.trim().to_string();
    }
    fallback.to_string()
}

/// The environment a stdio MCP server starts with.
///
/// Declared entries win; `NGU_EXE` is added when the pack did not name one,
/// because the host's own path is not otherwise discoverable from inside a
/// plugin — and a pack that shells out to `ngu` (starting a worker, checking a
/// flag) needs exactly that. `${VAR}` references are left untouched on purpose:
/// substitution happens where the values are known.
/// Lines handed to each proxy, keyed by proxy id.
///
/// Process-wide because a proxy's target list is not per-session: the proxy is
/// one child process however many conversations exist, so a per-conversation list
/// would have to be merged back into one anyway.
static PROXY_TARGETS: std::sync::Mutex<
    Option<std::collections::BTreeMap<String, Vec<String>>>,
> = std::sync::Mutex::new(None);

/// The command line a proxy should run for one declared server.
///
/// The shape is the proxy's own, not the kernel's: `name=command args`, with a
/// `cwd=<dir>::` prefix when the server is a script named relative to its pack.
/// A pack that declares `node bin/x.mjs` means *its* `bin`, and the proxy starts
/// its children elsewhere — the prefix is what carries that.
fn proxy_target_line(
    step: &PlannedStep,
    entry: &crate::assembly::McpEntry,
    pack_dir: Option<&std::path::Path>,
) -> Result<String> {
    let command = entry
        .command
        .clone()
        .ok_or_else(|| anyhow!("mcp '{}' is handed to a proxy but declares no command", step.id))?;
    // The pack's own directory, when the server is a script that came with it.
    // Same rule the direct-spawn path uses: an argument that resolves to a file
    // inside the pack means the pack's directory is where the process belongs.
    let cwd = {
        let inside = |candidate: &str| {
            pack_dir
                .map(|dir| {
                    let path = std::path::Path::new(candidate);
                    !path.is_absolute() && dir.join(path).is_file()
                })
                .unwrap_or(false)
        };
        if inside(&command) || entry.args.iter().any(|arg| inside(arg)) {
            pack_dir
                .map(|dir| format!("cwd={}::", dir.display()))
                .unwrap_or_default()
        } else {
            String::new()
        }
    };
    let mut line = format!("{}={cwd}{command}", step.id);
    for arg in &entry.args {
        line.push(' ');
        line.push_str(arg);
    }
    Ok(line)
}

/// Record one server for a proxy to start.
fn register_proxy_target(proxy_id: &str, line: String) {
    let mut guard = PROXY_TARGETS.lock().expect("proxy target table");
    let table = guard.get_or_insert_with(Default::default);
    let entry = table.entry(proxy_id.to_string()).or_default();
    if !entry.contains(&line) {
        entry.push(line);
    }
}

/// The targets a proxy should start this process, one per line.
///
/// Read when the proxy itself is launched. Settings' own `mcp_proxy_targets` are
/// not consulted here: those are the user's hand-written list for servers no pack
/// declares, and mixing the two would make it impossible to tell which line came
/// from where when one of them fails to start.
fn proxy_targets_for(proxy_id: &str) -> Vec<String> {
    PROXY_TARGETS
        .lock()
        .expect("proxy target table")
        .as_ref()
        .and_then(|table| table.get(proxy_id).cloned())
        .unwrap_or_default()
}

fn mcp_env(declared: &std::collections::BTreeMap<String, String>) -> std::collections::HashMap<String, String> {
    let mut env: std::collections::HashMap<String, String> =
        declared.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
    if !env.contains_key("NGU_EXE") {
        if let Ok(exe) = std::env::current_exe() {
            env.insert("NGU_EXE".to_string(), exe.display().to_string());
        }
    }
    // Every pack's settings, as one JSON object, under one name.
    //
    // Before this, each pack that wanted a setting the user had chosen needed a
    // kernel-side reader: `mcp-proxy` had `NGU_MCP_MODE` and `NGU_MCP_TARGETS`
    // written out here by name, which made the kernel know that a pack called
    // `mcp-proxy` exists and that it has settings called `mode` and `targets`.
    // That is the shape this whole change is trying to remove — a pack cannot be
    // added without editing this function.
    //
    // One variable instead: the whole `packs` table, verbatim. A server that
    // wants its own settings reads `NGU_PACK_SETTINGS` and takes its own key;
    // one that wants nothing ignores it. The kernel's involvement ends at
    // "serialize the map".
    //
    // Declared entries still win, for the ordinary reason: a pack that hardcodes
    // an environment variable is saying it does not want this control.
    if !env.contains_key("NGU_PACK_SETTINGS") {
        let settings = crate::settings::Settings::load().unwrap_or_default();
        if !settings.packs.is_empty() {
            if let Ok(text) = serde_json::to_string(&settings.packs) {
                env.insert("NGU_PACK_SETTINGS".to_string(), text);
            }
        }
    }
    env
}

/// Start a plugin process from a synchronous caller.
///
/// `Loader` runs inside the host's async runtime most of the time, but
/// `load_plugin` is a synchronous method reached from `apply`, and the process
/// start is a `tokio` operation. Building a current-thread runtime here rather
/// than making every caller async keeps one boundary: everything above this line
/// is sync, everything below is not.
/// The cargo home implied by wherever `cargo` was found.
///
/// `cargo` lives in `<CARGO_HOME>/bin`, so the parent of its directory is the
/// home — which is the directory holding `registry/`. A machine that installed
/// Rust anywhere but the default has its cache there and nowhere else, and the
/// default (`~/.cargo`) does not exist at all.
fn cargo_home_from_path() -> Option<std::path::PathBuf> {
    let cargo = resolve_program("cargo")?;
    let home = cargo.parent()?.parent()?;
    // Only when it actually looks like a cargo home: a `cargo` on a plain
    // `PATH` with no registry beside it is not one, and setting `CARGO_HOME` to
    // nonsense replaces a clear error with a confusing one.
    home.join("registry").is_dir().then(|| home.to_path_buf())
}

/// Find an executable by name, looking where a shell would.
///
/// `None` when it is not on any of those paths, which leaves the caller to try a
/// bare name and report the operating system's own error.
fn resolve_program(command: &str) -> Option<std::path::PathBuf> {
    let direct = std::path::Path::new(command);
    if direct.components().count() > 1 {
        return direct.is_file().then(|| direct.to_path_buf());
    }
    let path = toolchain_path()?;
    let exts: Vec<&str> = if cfg!(windows) {
        vec!["", ".exe", ".cmd", ".bat"]
    } else {
        vec![""]
    };
    for dir in std::env::split_paths(&path) {
        for ext in &exts {
            let candidate = dir.join(format!("{command}{ext}"));
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    None
}

/// A `PATH` that includes the usual toolchain locations, whether or not the
/// process was started with one.
///
/// The cheap part is the process's own `PATH`, extended with `CARGO_HOME`/`.cargo`
/// when they are set. The expensive part — asking the OS for the *user's*
/// configured `PATH` rather than the inherited one — is only worth it on Windows,
/// where a program started from Explorer or a shortcut gets a stale environment
/// and a developer's toolchain is on a drive the default `PATH` has never heard
/// of.
///
/// Returns `None` when there is nothing to add, so the caller can leave the
/// environment alone: copying a `PATH` that is already right is a way to be wrong
/// about it later.
fn toolchain_path() -> Option<std::ffi::OsString> {
    let mut parts: Vec<String> = std::env::var("PATH")
        .unwrap_or_default()
        .split(if cfg!(windows) { ';' } else { ':' })
        .filter(|part| !part.trim().is_empty())
        .map(str::to_string)
        .collect();
    let before = parts.len();

    // The user's own environment, which on Windows is what their shell would
    // have had. Read only here, and only when a build is about to run.
    #[cfg(windows)]
    {
        for (var, scope) in [("Path", "User"), ("Path", "Machine")] {
            let script = format!("[Environment]::GetEnvironmentVariable('{var}','{scope}')");
            // Absolute path, not a bare name: this runs precisely when the
            // inherited `PATH` is the broken thing, so a lookup through it would
            // fail for the reason we are here to fix.
            let shell = std::path::Path::new(&std::env::var("SystemRoot").unwrap_or_else(|_| r"C:\Windows".into()))
                .join("System32")
                .join("WindowsPowerShell")
                .join("v1.0")
                .join("powershell.exe");
            if let Ok(output) = std::process::Command::new(&shell)
                .args(["-NoProfile", "-NonInteractive", "-Command", &script])
                .output()
            {
                if output.status.success() {
                    let text = String::from_utf8_lossy(&output.stdout);
                    for part in text.split(';') {
                        let part = part.trim();
                        if !part.is_empty() && !parts.iter().any(|known| known == part) {
                            parts.push(part.to_string());
                        }
                    }
                }
            }
        }
    }

    // `CARGO_HOME` names where cargo keeps its binaries; a machine that set it
    // did so for a reason and `cargo` is almost certainly in `<it>/bin`.
    if let Ok(home) = std::env::var("CARGO_HOME") {
        let candidate = std::path::Path::new(&home).join("bin");
        let text = candidate.display().to_string();
        if !parts.iter().any(|known| known == &text) {
            parts.push(text);
        }
    }
    // `~/.cargo/bin`, spelled out.
    //
    // This is where rustup puts `cargo`, and a server started by systemd or by a
    // shell that never sourced `~/.cargo/env` does not have it on `PATH`. The
    // variable above only helps a machine that exports `CARGO_HOME`; most do
    // not, and the default location is the one that is right. Measured on the
    // Ubuntu server: `cargo` at `/home/ubuntu/.cargo/bin/cargo`, absent from the
    // service's `PATH`, and every pack build failed with "program not found".
    if let Some(home) = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")) {
        let candidate = std::path::Path::new(&home).join(".cargo").join("bin");
        let text = candidate.display().to_string();
        if candidate.is_dir() && !parts.iter().any(|known| known == &text) {
            parts.push(text);
        }
    }

    // On Windows the user's `PATH` is consulted even when nothing was added:
    // the inherited value is not the one a developer's shell has, and the whole
    // point of this function is to bridge that gap rather than to be frugal.
    if parts.len() == before && !cfg!(windows) {
        return None;
    }
    if parts.is_empty() {
        return None;
    }
    Some(std::ffi::OsString::from(parts.join(if cfg!(windows) { ";" } else { ":" })))
}

/// The same path, made absolute against the process directory.
///
/// A pack directory handed to the loader is often relative (`packs/restart`),
/// and a relative executable path is resolved against whatever directory the
/// child ends up in — which for a process plugin is the pack itself. The two
/// disagree, and the symptom is "No such file or directory" for a file that is
/// right there.
fn absolute(path: &std::path::Path) -> PathBuf {
    if path.is_absolute() {
        return path.to_path_buf();
    }
    std::env::current_dir()
        .map(|cwd| cwd.join(path))
        .unwrap_or_else(|_| path.to_path_buf())
}

/// The suffixes an executable may carry on this platform, best first.
///
/// Empty string first so an exact name always wins: a pack that says
/// `bin/ngu-restart.exe` gets that file, not `bin/ngu-restart.exe.exe`.
fn platform_suffixes() -> &'static [&'static str] {
    if cfg!(windows) {
        &["", ".exe"]
    } else {
        &[""]
    }
}

fn start_process_with_state_for(spec: &crate::protocol::Spec) -> Result<crate::protocol::Process> {
    // The provider captures nothing: it reads the settings file and the thread's
    // workspace at the moment the plugin asks, so a plugin that asks twice sees
    // a settings save that landed between the two.
    let provider: std::sync::Arc<dyn Fn(&str) -> Option<serde_json::Value> + Send + Sync> =
        std::sync::Arc::new(|key: &str| plugin_state_of(key));
    match tokio::runtime::Handle::try_current() {
        Ok(_) => std::thread::scope(|scope| {
            scope
                .spawn(|| {
                    tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                        .map_err(|error| anyhow!("building a runtime: {error}"))
                        .and_then(|runtime| {
                            runtime.block_on(crate::protocol::Process::start_with_state(
                                spec,
                                Some(provider),
                            ))
                        })
                })
                .join()
                .map_err(|_| anyhow!("the plugin start thread panicked"))?
        }),
        Err(_) => tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|error| anyhow!("building a runtime: {error}"))
            .and_then(|runtime| {
                runtime.block_on(crate::protocol::Process::start_with_state(
                    spec,
                    Some(provider),
                ))
            }),
    }
}

/// The state keys, answered from module-level state rather than from a loader.
///
/// Split out of `Loader::plugin_state` because the provider handed to a process
/// must not borrow the loader: the process outlives the load call, and a borrow
/// would tie its lifetime to a stack frame that is gone by the time a plugin
/// asks.
fn plugin_state_of(key: &str) -> Option<serde_json::Value> {
    use crate::protocol::*;
    match key {
        KERNEL_PID => Some(serde_json::json!(crate::protocol::kernel_pid())),
        KERNEL_EXE => std::env::current_exe()
            .ok()
            .map(|exe| serde_json::json!(exe.display().to_string())),
        SETTINGS => serde_json::to_value(
            crate::settings::Settings::load()
                .unwrap_or_default()
                .packs,
        )
        .ok(),
        WORKSPACE => crate::tools::current_workspace()
            .map(|dir| serde_json::json!(dir.display().to_string())),
        SESSION => crate::tools::current_session()
            .map(|id| serde_json::json!(id)),
        _ => None,
    }
}

fn start_process(spec: &crate::protocol::Spec) -> Result<crate::protocol::Process> {
    match tokio::runtime::Handle::try_current() {
        // Already inside a runtime: `block_on` there panics. A dedicated thread
        // with its own runtime is the only correct way to wait.
        Ok(_) => std::thread::scope(|scope| {
            scope
                .spawn(|| {
                    tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                        .map_err(|error| anyhow!("building a runtime: {error}"))
                        .and_then(|runtime| runtime.block_on(crate::protocol::Process::start(spec)))
                })
                .join()
                .map_err(|_| anyhow!("the plugin start thread panicked"))?
        }),
        Err(_) => tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|error| anyhow!("building a runtime: {error}"))
            .and_then(|runtime| runtime.block_on(crate::protocol::Process::start(spec))),
    }
}

fn scope_name(scope: Scope) -> &'static str {
    match scope {
        Scope::Session => "session",
        Scope::Global => "global",
    }
}

fn label(step: &PlannedStep) -> String {
    format!("{}:{}", step.kind.as_str(), step.id)
}

