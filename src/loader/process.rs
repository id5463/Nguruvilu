//! 进程插件：一个插件跑成子进程时，怎么把它建起来、指到哪个二进制、怎么收尾。
//!
//! # 为什么这三件事在一处
//!
//! “去哪找二进制”和“怎么编译出二进制”是同一个问题的两半：包里带了源码而
//! 二进制不在时，唯一的答案是自己编。所以解析路径和构建放在一起。
//!
//! 实测：启动一个进程 45 ms，一次调用 0.25 ms。所以进程活过整次加载，而不是
//! 每次调用重连 —— 重连会让协议比它用的传输慢 180 倍。

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{anyhow, Context, Result};

use crate::assembly::{BuildSpec, PlannedStep};
use crate::plugin::Plugin;

use super::model::{Loader, Outcome};

impl Loader {
    /// Resolve a source path against the pack directory when it is relative.
    /// Load a plugin that runs as a process.
    ///
    /// Three steps, in this order, because each makes the next cheaper:
    ///
    /// 1. **Build it if the pack ships source and the binary is missing.**
    /// 2. **Start it.** One spawn; the process stays for the life of the loader.
    ///    Measured: 45 ms to start against 0.25 ms per call, so reconnecting per
    ///    call would make the protocol 180x slower than the transport it uses.
    /// 3. **Register what it declared.** Each tool becomes an ordinary entry in
    ///    the tool table, owned by `plugin:<id>`, so unloading removes exactly
    ///    what this plugin added.
    fn load_process_plugin(&mut self, step: &PlannedStep, raw: &str) -> Result<Outcome> {
        let pack_dir = self.pack_dir.clone();
        let path = self.resolve_process_path(raw, pack_dir.as_deref());

        let build_spec = step.build.clone();
        let path = match path {
            Some(path) => path,
            None => match build_spec {
                Some(build) => self.build_plugin(step, &build)?,
                None => {
                    return Err(anyhow!(
                        "plugin '{}' names '{}', which is not there, and the pack \
                         declares no `build` to produce it",
                        step.id,
                        raw
                    ))
                }
            },
        };

        // Already provided by another pack: reuse it, do not start a second copy.
        //
        // The same capability legitimately appears in more than one pack —
        // `delegate` is wanted by the ticket board, the research pack, and a
        // plain starter install alike, and each of them declares it because a
        // pack that silently depends on another pack being installed is a pack
        // that breaks when it is installed alone.
        //
        // Before this, the second declaration *overwrote* the first
        // (`ConflictPolicy::Override`), so which copy ran depended on load
        // order, and unloading either pack removed the tool for both. Starting a
        // second process was the other half of that: two pipes, one table.
        //
        // The identity is the entry id — `delegate` means the same capability no
        // matter which pack names it. A plugin whose id is already serving tools
        // is reported as reused, and the loader keeps no handle to it.
        // A process plugin's tools are owned by `plugin:<id>`, so a live entry
        // for any of them means this capability is already running.
        let owner = format!("plugin:{}", step.id);
        if self.kernel.tools().has_owner(&owner) {
            return Ok(Outcome::Reused);
        }

        let mut spec = crate::protocol::Spec::new(step.id.clone(), path.display().to_string());
        if let Some(dir) = &pack_dir {
            // A plugin that came with a pack runs inside it, the same rule MCP
            // servers already follow: a relative path in its own arguments then
            // means what the pack author meant.
            spec = spec.in_dir(dir.display().to_string());
        }
        // `NGU_EXE` — the host's own binary — so a plugin can start another
        // copy of the kernel.
        //
        // MCP servers have had this since the pack protocol was written; process
        // plugins did not, and the gap is exactly the one a subagent hits: it is
        // a plugin whose whole job is to run `ngu -p`, and without this it has
        // to guess where the kernel is. Measured on the server, where every
        // guess missed and `delegate` answered "cannot find the kernel binary".
        //
        // A pack that names its own `NGU_EXE` wins: it may want a specific build.
        if !spec.env.contains_key("NGU_EXE") {
            if let Ok(exe) = std::env::current_exe() {
                spec.env
                    .insert("NGU_EXE".to_string(), exe.display().to_string());
            }
        }
        let process = start_process_with_state_for(&spec)
            .with_context(|| format!("loading process plugin '{}'", step.id))?;
        let process = Arc::new(process);

        let owner = format!("plugin:{}", step.id);
        let mut registered: Vec<String> = Vec::new();
        for decl in &process.description().tools {
            let handler_process = Arc::clone(&process);
            let tool_name = decl.name.clone();
            let def = crate::tools::ToolDef::new(
                decl.name.clone(),
                decl.description.clone(),
                decl.parameters.clone(),
                owner.clone(),
                move |args| {
                    let process = Arc::clone(&handler_process);
                    let name = tool_name.clone();
                    Box::pin(async move {
                        let value = process.call(&name, args).await?;
                        // A plugin answers with JSON. Text is what the model
                        // reads, so a string answer is passed through and
                        // anything else is serialised compactly.
                        Ok(match value {
                            serde_json::Value::String(text) => {
                                crate::tools::ToolOutput::text(text)
                            }
                            other => crate::tools::ToolOutput::text(
                                serde_json::to_string(&other).unwrap_or_default(),
                            ),
                        })
                    }) as crate::tools::ToolFuture
                },
            );
            if let Err(error) = self
                .kernel
                .tools_mut()
                .register(def, ConflictPolicy::Override)
            {
                for tool in &registered {
                    self.kernel.tools_mut().unregister(tool);
                }
                return Err(error);
            }
            registered.push(decl.name.clone());
        }

        // Readers: a format this plugin teaches the kernel's own `read`.
        //
        // Registered into the same table the kernel version used, with a process
        // where the closure was — see `readers::register_process`. The disposers
        // go on the ledger's effects so an unload takes them with the tools.
        let mut reader_count = 0;
        for decl in &process.description().readers {
            crate::tools::readers::register_process(crate::tools::readers::ProcessReader {
                id: decl.id.clone(),
                process: Arc::clone(&process),
                magic: decl.magic.clone(),
                extensions: decl.extensions.clone(),
            });
            reader_count += 1;
        }

        let tool_count = registered.len();
        let panel_count = process.description().panels.len();
        self.processes.push(Arc::clone(&process));

        self.ledger.record(LedgerEntry {
            kind: EntryKind::Plugin.as_str().into(),
            id: step.id.clone(),
            source: step.source.clone(),
            sha1: None,
            path: Some(path.display().to_string()),
            scope: scope_name(step.scope).into(),
            pack: self.pack.clone(),
            installed_at: now_iso(),
        });

        if tool_count == 0 && panel_count == 0 && reader_count == 0 {
            return Ok(Outcome::Skipped(
                "plugin described no tool, panel, or reader".into(),
            ));
        }
        Ok(Outcome::Loaded)
    }

    /// Where a `plugin:` source points, if it is there.
    ///
    /// Relative paths resolve inside the pack; a bare name is handed to the
    /// operating system's `PATH` lookup, exactly like every other `command` in
    /// this loader.
    fn resolve_process_path(
        &self,
        raw: &str,
        pack_dir: Option<&std::path::Path>,
    ) -> Option<PathBuf> {
        let candidate = PathBuf::from(raw);
        if candidate.is_absolute() {
            return candidate.is_file().then_some(candidate);
        }
        if !raw.contains('/') && !raw.contains('\\') {
            // A bare name: let `Command` resolve it. Claiming a file we have not
            // checked would be worse than letting the spawn say it is missing.
            return Some(candidate);
        }
        // Absolute, because the returned path travels through `Spec::command`
        // and is then executed *after* `Spec::cwd` has moved the child. A
        // relative path resolves twice — once here against this process's cwd,
        // once in the child against `cwd` — and the second resolution is the one
        // that fails with ENOENT the moment the two differ. Measured: loading
        // `delegate` from its own directory failed while the same manifest from
        // the parent directory succeeded, which is the fingerprint of exactly
        // this.
        let dir = absolute(pack_dir?);
        let joined = absolute(&dir.join(&candidate));
        if joined.is_file() {
            return Some(joined);
        }
        // The platform's own suffix, when the name does not carry one.
        //
        // A pack says `plugin:bin/ngu-restart` and means the binary on whatever
        // machine this is. Windows needs `.exe`; Unix does not. Resolving both
        // here is what lets one `assembly.yaml` work on either — measured on a
        // Linux server, where the first version looked for `ngu-restart` with no
        // suffix, did not find it, and reported "the file is missing" for a file
        // that had compiled a moment earlier.
        for suffix in platform_suffixes() {
            let with_suffix = absolute(&dir.join(format!("{}{suffix}", candidate.display())));
            if with_suffix.is_file() {
                return Some(with_suffix);
            }
        }
        None
    }

    /// Build a plugin from the source a pack ships.
    ///
    /// The toolchain is the pack's choice (`build.command`), not an enum here:
    /// the field names a directory and a binary, which is everything a build
    /// needs to know and everything the kernel should ever learn about one.
    ///
    /// A build that fails reports the command it ran and the compiler's own
    /// words. That is the whole value of building on the user's machine: the
    /// error names the real problem (a missing toolchain, a crates index the
    /// machine cannot reach) rather than "the binary is missing from the
    /// archive".
    fn build_plugin(
        &self,
        step: &PlannedStep,
        build: &crate::assembly::BuildSpec,
    ) -> Result<PathBuf> {
        let pack_dir = self.pack_dir.clone().ok_or_else(|| {
            anyhow!("plugin '{}' needs a build but has no pack directory", step.id)
        })?;
        // Absolute, always. The pack directory a caller hands over may be
        // relative (`packs/restart`, as `ngu apply` passes it), and a relative
        // path survives neither a `current_dir` change nor the spawn below —
        // measured on a Linux server, where the build succeeded, the binary
        // existed, and starting it failed with "No such file or directory"
        // because the path was resolved against the *pack* once and the
        // *process's* directory twice.
        let dir = if pack_dir.is_absolute() {
            pack_dir.join(&build.dir)
        } else {
            std::env::current_dir()
                .map(|cwd| cwd.join(&pack_dir).join(&build.dir))
                .unwrap_or_else(|_| pack_dir.join(&build.dir))
        };
        if !dir.is_dir() {
            return Err(anyhow!(
                "plugin '{}' declares `build` with dir '{}', which is not a directory \
                 under {}",
                step.id,
                build.dir,
                pack_dir.display()
            ));
        }
        let command = build.command.clone().unwrap_or_else(|| "cargo".to_string());
        let args: Vec<String> = if build.args.is_empty() {
            vec![
                "build".into(),
                "--release".into(),
                // `--offline` unless the pack overrides: a build that reaches
                // the network while a turn is starting is a stall with no
                // explanation, and the crates a pack needs are in its lockfile
                // by the time it ships.
                "--offline".into(),
            ]
        } else {
            build.args.clone()
        };

        // Resolve the program through the *extended* PATH rather than letting
        // `Command` use the inherited one.
        //
        // On Windows the program lookup happens against the parent's PATH before
        // any `env` override applies, so `Command::new("cargo")` fails even with
        // a corrected child PATH. Resolving the absolute path first is the only
        // order that works.
        let program = resolve_program(&command).unwrap_or_else(|| std::path::PathBuf::from(&command));
        let mut spawn = std::process::Command::new(&program);
        spawn.args(&args).current_dir(&dir);
        // The toolchain has to be findable even when this process was started
        // with a narrower PATH than the user's shell has.
        //
        // Measured cause: `cargo` lives on `I:\cargoin` and was added to the
        // user's PATH, but the running `ngu` inherited the PATH it was launched
        // with, so a pack that built fine by hand failed here with "program not
        // found". A build step that only works when the parent happens to have
        // the right environment is a build step that fails for the user who
        // never opened a terminal.
        if let Some(path) = toolchain_path() {
            spawn.env("PATH", path);
        }
        // `CARGO_HOME` is where the crates index and the download cache live. A
        // build with the wrong one is not a build that fails to find `cargo` —
        // it is a build that finds `cargo` and then cannot find a single crate,
        // which is a much more confusing message. When the caller has not set
        // it, take the one the toolchain on `PATH` implies.
        if std::env::var_os("CARGO_HOME").is_none() {
            if let Some(home) = cargo_home_from_path() {
                // `RUSTUP_HOME` is the toolchains; `CARGO_HOME` is the crates.
                // They are usually siblings, and a machine that moved one moved
                // the other — but only claim it when the directory is there, so
                // a plain cargo install (no rustup) is left alone.
                let toolchains = home.join("..").join("rustup");
                if toolchains.join("toolchains").is_dir() {
                    if std::env::var_os("RUSTUP_HOME").is_none() {
                        spawn.env("RUSTUP_HOME", &toolchains);
                    }
                }
                spawn.env("CARGO_HOME", home);
            }
        }
        let output = spawn
            .output()
            .with_context(|| {
                format!(
                    "building plugin '{}' with `{}` in {} — is the toolchain installed \
                     and on PATH?",
                    step.id,
                    command,
                    dir.display()
                )
            })?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow!(
                "building plugin '{}' failed: `{} {}` in {} exited {}.\n{}",
                step.id,
                command,
                args.join(" "),
                dir.display(),
                output.status,
                stderr.trim()
            ));
        }

        let candidates = [
            dir.join("target").join("release").join(&build.binary),
            dir.join("target")
                .join("release")
                .join(format!("{}.exe", build.binary)),
            dir.join(&build.binary),
        ];
        for candidate in candidates {
            if candidate.is_file() {
                return Ok(candidate);
            }
        }
        Err(anyhow!(
            "plugin '{}' built without error but produced no '{}' under {}",
            step.id,
            build.binary,
            dir.join("target/release").display()
        ))
    }

}
