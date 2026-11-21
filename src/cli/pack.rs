//! `ngu pack` 家族：打包、安装、卸载、校验、列出。
//!
//! 这五件事共用一个问题：一个包从哪来、到哪去、装成什么样。所以它们在一
//! 个文件里；但它们各自是一条独立的路由，因为想知道"装了什么"的人，和想
//! 换掉"怎么装"的人，不是同一个。
//!
//! 这一层只做命令行的事：读参数、调 `nguruvilu::pack` 与 `nguruvilu::fetch`、
//! 打印结果。包格式本身不在这里。

use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};

use nguruvilu::fetch::Fetcher;
use nguruvilu::pack::{self, InstalledPack, PackManifest, VerifyReport};

use super::args::*;

async fn pack_command(
    dir: &PathBuf,
    out: Option<&PathBuf>,
    pin: bool,
    offline: bool,
    as_json: bool,
) -> Result<()> {
    let mut manifest = nguruvilu::pack::read_manifest(dir)?;
    let archive = match out {
        Some(path) => path.clone(),
        None => PathBuf::from(format!(
            "{}-{}.{}",
            manifest.name,
            manifest.version_id,
            nguruvilu::pack::ARCHIVE_EXT
        )),
    };

    if pin {
        let fetcher = nguruvilu::fetch::Fetcher::new()?;
        // Collected first: pinning mutates the manifest, so the borrow that
        // produced these references cannot still be live.
        let todo: Vec<(String, String, nguruvilu::fetch::Source, Option<String>)> = manifest
            .references()?
            .into_iter()
            .filter(|reference| reference.source.is_remote())
            .map(|reference| {
                (
                    reference.kind.to_string(),
                    reference.id.to_string(),
                    reference.source.clone(),
                    reference.sha256.map(str::to_string),
                )
            })
            .collect();

        let mut pinned = 0;
        for (kind, id, source, expected) in todo {
            let fetched = fetcher
                .fetch(&source, expected.as_deref())
                .await
                .with_context(|| format!("pinning {kind} '{id}'"))?;
            manifest.set_hash(&kind, &id, &fetched.sha256)?;
            pinned += 1;
            if !as_json {
                eprintln!(
                    "pinned {kind} {id} → {}",
                    &fetched.sha256[..12.min(fetched.sha256.len())]
                );
            }
        }
        // Written back so the directory and the archive agree; a pinned pack
        // that only existed inside the archive would be re-pinned on every
        // build, which is the opposite of pinning.
        nguruvilu::pack::write_manifest(dir, &manifest)?;
        if !as_json && pinned == 0 {
            eprintln!("nothing to pin: every reference is local or built in");
        }
    }

    let packed = if offline {
        nguruvilu::pack::pack_offline(dir, &archive).await?
    } else {
        nguruvilu::pack::pack(dir, &archive)?
    };
    // The same exclusion the archive used, so the count is what was packed
    // rather than what happens to be in the directory.
    let contents = nguruvilu::pack::packable_files(dir, &archive)?;

    if as_json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "ok": true,
                "archive": archive.display().to_string(),
                "name": packed.name,
                "version": packed.version_id,
                "license": packed.license,
                "files": contents.files.len(),
                "has_assembly": contents.has_assembly,
            }))?
        );
    } else {
        println!(
            "packed {} {} → {}",
            packed.name,
            packed.version_id,
            archive.display()
        );
        println!("{} files", contents.files.len());
        if !contents.has_assembly {
            println!("note: no assembly.yaml — the pack loads nothing yet");
        }
    }
    Ok(())
}

async fn install_command(file: &PathBuf, into: Option<&PathBuf>, as_json: bool) -> Result<()> {
    let packs_dir = into.cloned().unwrap_or_else(nguruvilu::pack::default_packs_dir);

    // Two ways in, both the same command: a spec downloads the archive first,
    // a path installs what is already on disk — the offline half. The content
    // inside has always had both forms (`Carried` vs `Fetched`); this is the
    // pack itself catching up.
    let raw = file.to_string_lossy();
    let raw = raw.trim();
    let local = if is_remote_spec(raw) {
        resolve_pack_source(raw).await?
    } else {
        file.clone()
    };

    let placed = nguruvilu::pack::install(&local, &packs_dir).await?;

    if as_json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "ok": true,
                "name": placed.manifest.name,
                "version": placed.manifest.version_id,
                "path": placed.path.display().to_string(),
                "assembly": placed.assembly.as_ref().map(|p| p.display().to_string()),
            }))?
        );
    } else {
        println!(
            "installed {} {} → {}",
            placed.manifest.name,
            placed.manifest.version_id,
            placed.path.display()
        );
        match &placed.assembly {
            Some(assembly) => println!("assembly: {}", assembly.display()),
            None => println!("no assembly.yaml in this pack"),
        }
    }
    Ok(())
}

/// A download spec rather than a path: `github:…`, `https://…`, `http://…`.
///
/// A prefix check rather than `Source::parse` on purpose: a Windows path
/// (`C:\…`) contains a colon too, and a typo in a path must fail as a path
/// instead of being reinterpreted as a source.
fn is_remote_spec(raw: &str) -> bool {
    raw.starts_with("github:") || raw.starts_with("https://") || raw.starts_with("http://")
}

/// Resolve a download spec to a local archive to install.
///
/// A URL that *is* the archive lands as a file and goes straight through; a
/// `github:` spec lands as a directory (the pack's own directory, or a repo),
/// and the archive inside it is picked.
async fn resolve_pack_source(spec: &str) -> Result<PathBuf> {
    let source = nguruvilu::fetch::Source::parse(spec)
        .with_context(|| format!("reading '{spec}' as a download source"))?;
    let fetcher = nguruvilu::fetch::Fetcher::new()?;
    let fetched = fetcher
        .fetch(&source, None)
        .await
        .with_context(|| format!("fetching '{spec}'"))?;
    if fetched.path.is_file() {
        return Ok(fetched.path);
    }
    pack_archive_in(&fetched.path)
}

/// The one pack archive inside a fetched directory.
///
/// A pack's own directory contains one archive; a repository contains many,
/// and guessing which was meant would install the wrong pack — so it lists
/// them instead of choosing.
fn pack_archive_in(dir: &std::path::Path) -> Result<PathBuf> {
    let mut found = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(current) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&current) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path
                .extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| nguruvilu::pack::ARCHIVE_EXTS.contains(&ext.to_ascii_lowercase().as_str()))
                .unwrap_or(false)
            {
                found.push(path);
            }
        }
    }
    match found.len() {
        0 => Err(anyhow::anyhow!(
            "no .{} (or .{}) archive under {}",
            nguruvilu::pack::ARCHIVE_EXT,
            nguruvilu::pack::LEGACY_ARCHIVE_EXT,
            dir.display()
        )),
        1 => Ok(found.pop().expect("exactly one match")),
        _ => {
            found.sort();
            let names = found
                .iter()
                .map(|path| {
                    path.strip_prefix(dir)
                        .unwrap_or(path)
                        .display()
                        .to_string()
                })
                .collect::<Vec<_>>()
                .join(", ");
            Err(anyhow::anyhow!(
                "{} archives under {}; point at the pack's own directory, one of: {names}",
                found.len(),
                dir.display()
            ))
        }
    }
}

/// Import MCP servers from a Claude Code config file into a new pack directory.
///
/// The classic config shape is `{"mcpServers": {name: {command, args, env}}}`
/// (Claude Code's `.mcp.json`, Claude Desktop's config file); our own
/// `{"servers": [...]}` passes through the strict reader. Entries without a
/// command — the url-based transports — are reported and *not* written to the
/// pack: this kernel connects stdio servers today, and a pack claiming
/// otherwise would fail at load instead of at import.
/// A file-derived pack name: lowercase, `[a-z0-9-]`, no edge dashes.
fn sanitize_pack_name(raw: &str) -> String {
    let lowered = raw.trim().to_ascii_lowercase();
    let mut out = String::new();
    let mut dash = false;
    for ch in lowered.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch);
            dash = false;
        } else if !dash && !out.is_empty() {
            out.push('-');
            dash = true;
        }
    }
    out.trim_end_matches('-').to_string()
}

/// The generated pack's README: what was imported, what was skipped, how to
/// pack and install it — and that the runtime and licenses stay with the user.
fn generated_pack_readme(
    pack_name: &str,
    source_file: &Path,
    license: &str,
    servers: &[nguruvilu::pack::McpServerDecl],
    skipped: &[(String, String)],
) -> String {
    let mut out = String::new();
    out.push_str(&format!("# {pack_name} — imported MCP servers\n\n"));
    out.push_str(&format!(
        "由 `ngu mcp import {}` 生成:身份清单 + `mcp.json` + 这份说明。\n\n",
        source_file.display()
    ));
    out.push_str(
        "**只搬运声明,不搬运程序** —— 每个服务器的运行时(npx 需要 Node、各自的\n\
         npm 包或可执行文件)与它自己的许可证,由你提供与核对。环境变量的**值**\n\
         不会出现在这份文件里。\n\n",
    );
    out.push_str("## 服务器\n\n| id | 命令 | 参数 | 环境变量名 |\n|---|---|---|---|\n");
    for server in servers {
        let command = server.command.as_deref().unwrap_or("");
        let args = server
            .args
            .iter()
            .map(|arg| format!("`{arg}`"))
            .collect::<Vec<_>>()
            .join(" ");
        let args = if args.is_empty() { "—".into() } else { args };
        let keys = server.env.keys().cloned().collect::<Vec<_>>().join(", ");
        let keys = if keys.is_empty() { "—".into() } else { keys };
        out.push_str(&format!("| {} | `{command}` | {args} | {keys} |\n", server.id));
    }
    if !skipped.is_empty() {
        out.push_str("\n## 未导入(本内核暂不支持)\n\n");
        for (id, reason) in skipped {
            out.push_str(&format!("- `{id}`:{reason}\n"));
        }
    }
    out.push_str(&format!(
        "\n## 用法\n\n```bash\nngu pack {name} --out {name}-1.0.0.ngupack\n\
         ngu install {name}-1.0.0.ngupack\nngu packs\n```\n\n\
         装载后每个服务器的工具以 `mcp__<id>__*` 出现;卸载留文件,`--delete` 才删。\n\n\
         ## 许可\n\n本包清单声明 **{license}**(这是这个*包*的许可);各 MCP 服务器\n\
         本身的许可以其上游为准。\n",
        name = pack_name
    ));
    out
}

/// `ngu events` — read back the audit trail the hosts append to.
///
/// The file is the source of truth (the bus keeps no history), so this is a
/// file read with a shape, not a query into a running process.
fn list_packs(as_json: bool) -> Result<()> {
    let packs_dir = nguruvilu::pack::default_packs_dir();
    let packs = nguruvilu::pack::installed(&packs_dir)?;

    if as_json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "packs_dir": packs_dir.display().to_string(),
                "packs": packs.iter().map(|p| serde_json::json!({
                    "name": p.manifest.name,
                    "description": p.manifest.description,
                    "version": p.manifest.version_id,
                    "license": p.manifest.license,
                    "path": p.path.display().to_string(),
                    "has_assembly": p.assembly.is_some(),
                    "loaded": p.enabled,
                })).collect::<Vec<_>>(),
            }))?
        );
        return Ok(());
    }

    if packs.is_empty() {
        println!("no packs installed in {}", packs_dir.display());
        return Ok(());
    }
    println!(
        "{:<24} {:<12} {:<10} {:<8} {}",
        "name", "version", "license", "loaded", "assembly"
    );
    for pack in packs {
        println!(
            "{:<24} {:<12} {:<10} {:<8} {}",
            pack.manifest.name,
            pack.manifest.version_id,
            pack.manifest.license,
            if pack.enabled { "yes" } else { "no" },
            if pack.assembly.is_some() { "yes" } else { "no" }
        );
        if !pack.manifest.description.is_empty() {
            println!("  {}", pack.manifest.description);
        }
    }
    Ok(())
}

/// Write a copy of the built-in interface to a directory.

/// Take an installed pack out of every conversation, or delete its files.
///
/// Two different acts with one command on purpose: the ordinary one is
/// *unload* — the pack stops being loaded into new conversations, and its
/// files stay, so it can come back without downloading anything. `--delete` is
/// the one that removes files, because deleting is not how you stop using
/// something.
///
/// Named rather than pathed: the user knows which pack they installed, and
/// asking them to find the directory is asking them to know how install
/// arranges things. `name@version` picks one when several are installed.
fn uninstall_command(
    name: &str,
    all: bool,
    delete: bool,
    into: Option<&PathBuf>,
    as_json: bool,
) -> Result<()> {
    let root = into
        .cloned()
        .unwrap_or_else(nguruvilu::pack::default_packs_dir);
    let source = nguruvilu::source::FilesystemSource::new(&root);

    let (wanted_name, wanted_version) = match name.split_once('@') {
        Some((pack, version)) => (pack, Some(version)),
        None => (name, None),
    };

    let matches: Vec<nguruvilu::pack::InstalledPack> = source
        .list()?
        .into_iter()
        .filter(|pack| pack.manifest.name == wanted_name)
        .filter(|pack| match wanted_version {
            Some(version) => pack.manifest.version_id == version,
            None => true,
        })
        .collect();

    if matches.is_empty() {
        // Naming what is installed turns a dead end into a next step.
        let installed = source.list()?;
        let names: Vec<String> = installed
            .iter()
            .map(|pack| format!("{}@{}", pack.manifest.name, pack.manifest.version_id))
            .collect();
        anyhow::bail!(
            "no installed pack matches '{name}'.{}",
            if names.is_empty() {
                format!(" Nothing is installed in {}.", root.display())
            } else {
                format!(" Installed: {}", names.join(", "))
            }
        );
    }

    // Several versions at once is what `--all` is for; without it, ask rather
    // than guessing which one was meant.
    if matches.len() > 1 && !all {
        let versions: Vec<String> = matches
            .iter()
            .map(|pack| pack.manifest.version_id.clone())
            .collect();
        anyhow::bail!(
            "'{wanted_name}' has {} versions installed: {}. \
             Name one as '{wanted_name}@<version>', or pass --all.",
            versions.len(),
            versions.join(", ")
        );
    }

    let mut affected = Vec::new();
    for pack in &matches {
        let label = format!("{}@{}", pack.manifest.name, pack.manifest.version_id);
        if delete {
            source.remove(&pack.path)?;
        } else {
            nguruvilu::pack::set_enabled(&pack.path, false)?;
        }
        affected.push(label);
    }

    if as_json {
        let payload = if delete {
            serde_json::json!({
                "ok": true,
                "removed": affected,
                "files_kept": false,
                "dir": root.display().to_string(),
            })
        } else {
            serde_json::json!({
                "ok": true,
                "unloaded": affected,
                "files_kept": true,
                "dir": root.display().to_string(),
            })
        };
        println!("{}", serde_json::to_string_pretty(&payload)?);
    } else if delete {
        for entry in &affected {
            println!("deleted {entry}");
        }
        println!("Its files are gone; reinstall to use it again.");
    } else {
        for entry in &affected {
            println!("unloaded {entry} (files kept)");
        }
        println!(
            "New conversations will not load it. To bring it back: ngu install <archive>, \
             or load it from inside a conversation with the pack tool."
        );
        println!(
            "A session already running keeps what it loaded; ask it to unload, or restart it."
        );
    }
    Ok(())
}

fn verify_command(file: &PathBuf, as_json: bool) -> Result<()> {
    let report = nguruvilu::pack::verify(file)?;

    if as_json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "ok": report.warnings.is_empty(),
                "manifest": {
                    "name": report.manifest.name,
                    "version": report.manifest.version_id,
                    "license": report.manifest.license,
                    "kernel_version": report.manifest.kernel_version,
                    "requires": report.manifest.dependencies.nguruvilu,
                },
                "files": report.contents.files.len(),
                "has_assembly": report.contents.has_assembly,
                "warnings": report.warnings,
            }))?
        );
        return Ok(());
    }

    println!(
        "{} {} (license {}, built against kernel {})",
        report.manifest.name,
        report.manifest.version_id,
        report.manifest.license,
        report.manifest.kernel_version
    );
    println!("requires nguruvilu {}", report.manifest.dependencies.nguruvilu);


    println!("{} files", report.contents.files.len());
    if report.warnings.is_empty() {
        println!("no problems found");
    } else {
        for warning in &report.warnings {
            println!("warning: {warning}");
        }
    }
    Ok(())
}



/// 注册 `ngu pack` 家族的默认实现：`cli.pack.pack`、`.install`、`.uninstall`、
/// `.verify`、`.list`。
///
/// 五条路由而不是一条：这五件事被替换的理由不一样。「怎么打包」是一个构建
/// 问题（换压缩、换归档布局），「装了什么」是一个查询问题（换数据源），
/// 合成一条会让换其中任何一个都得重写其余四个。
///
/// 副本留在链尾：包可以包一层先做事再调 `next`，也可以整条换掉。
pub fn register(dispatch: &nguruvilu::dispatch::Dispatch) {
    use nguruvilu::dispatch::{Args, Chain, Fut};
    use std::sync::Arc;

    let json_flag = |args: &Args| args.get("json").and_then(|v| v.as_bool()).unwrap_or(false);
    let path_of = |args: &Args, key: &str| {
        args.get(key)
            .and_then(|v| v.as_str())
            .map(PathBuf::from)
    };

    dispatch.register(
        "cli.pack.pack",
        Arc::new(|args: Args, _next: Chain| -> Fut {
            Box::pin(async move {
                let dir = path_of(&args, "dir").ok_or_else(|| "cli.pack.pack needs a dir".to_string())?;
                let out = path_of(&args, "out");
                let pin = args.get("pin").and_then(|v| v.as_bool()).unwrap_or(false);
                let offline = args.get("offline").and_then(|v| v.as_bool()).unwrap_or(false);
                pack_command(&dir, out.as_ref(), pin, offline, json_flag(&args))
                    .await
                    .map_err(|e| format!("{e:#}"))?;
                Ok(serde_json::json!({ "ok": true }))
            })
        }),
    );

    dispatch.register(
        "cli.pack.install",
        Arc::new(|args: Args, _next: Chain| -> Fut {
            Box::pin(async move {
                let file = path_of(&args, "file").ok_or_else(|| "cli.pack.install needs a file".to_string())?;
                let into = path_of(&args, "into");
                install_command(&file, into.as_ref(), json_flag(&args))
                    .await
                    .map_err(|e| format!("{e:#}"))?;
                Ok(serde_json::json!({ "ok": true }))
            })
        }),
    );

    dispatch.register(
        "cli.pack.list",
        Arc::new(|args: Args, _next: Chain| -> Fut {
            Box::pin(async move {
                list_packs(json_flag(&args)).map_err(|e| format!("{e:#}"))?;
                Ok(serde_json::json!({ "ok": true }))
            })
        }),
    );

    dispatch.register(
        "cli.pack.uninstall",
        Arc::new(|args: Args, _next: Chain| -> Fut {
            Box::pin(async move {
                let name = args
                    .get("name")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| "cli.pack.uninstall needs a name".to_string())?;
                let all = args.get("all").and_then(|v| v.as_bool()).unwrap_or(false);
                let delete = args.get("delete").and_then(|v| v.as_bool()).unwrap_or(false);
                let into = path_of(&args, "into");
                uninstall_command(name, all, delete, into.as_ref(), json_flag(&args))
                    .map_err(|e| format!("{e:#}"))?;
                Ok(serde_json::json!({ "ok": true }))
            })
        }),
    );

    dispatch.register(
        "cli.pack.verify",
        Arc::new(|args: Args, _next: Chain| -> Fut {
            Box::pin(async move {
                let file = path_of(&args, "file").ok_or_else(|| "cli.pack.verify needs a file".to_string())?;
                verify_command(&file, json_flag(&args)).map_err(|e| format!("{e:#}"))?;
                Ok(serde_json::json!({ "ok": true }))
            })
        }),
    );
}
