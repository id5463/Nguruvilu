//! 校验一个归档，而不安装它：内容齐不齐、内核版本合不合、平台对不对。
//!
//! `verify` 跑的是与 `install` 同一条解包路径，只是解到临时目录、检查完就扔。
//! 这样「校验通过」和「装得上」说的是同一件事。

use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};

use super::archive::unpack;
use super::format::VerifyReport;
use super::manifest::{McpServerDecl, PackManifest, PlatformSource};

pub fn verify(pack_path: &Path) -> Result<VerifyReport> {
    let scratch = std::env::temp_dir().join(format!(
        "ngu-verify-{}",
        uuid::Uuid::new_v4().simple()
    ));
    let result = (|| -> Result<VerifyReport> {
        let manifest = unpack(pack_path, &scratch)?;
        let contents = inspect(&scratch)?;

        // What the archive holds, described rather than judged. erify exists
        // so a person can see inside a pack; it does not decide whether the pack
        // is acceptable, because that is the person's call.
        let mut warnings = Vec::new();
        for (field, _) in CONTENT_FILES {
            let Some(reference) = manifest.content_file(field) else {
                continue;
            };
            // A fetched file is not in the archive; it arrives at install.
            if reference.source().is_some() {
                continue;
            }
            if !scratch.join(reference.path()).is_file() {
                warnings.push(format!(
                    "{field} names {}, which is not in the archive",
                    reference.path()
                ));
            }
        }
        let kernel_compatible = manifest.accepts_kernel(kernel_version());
        if !kernel_compatible {
            warnings.push(format!(
                "this pack accepts nguruvilu {}, this kernel is {}",
                manifest.dependencies.nguruvilu,
                kernel_version()
            ));
        }
        if contents.has_assembly && manifest.references().map(|r| !r.is_empty()).unwrap_or(false) {
            // An offline pack embeds payload; a manifest pack must not.
            warnings.push(
                "the archive carries an assembly and also names references; \
                 it will install as an offline pack"
                    .into(),
            );
        }

        Ok(VerifyReport {
            manifest,
            contents,
            kernel_compatible,
            warnings,
        })
    })();

    let _ = std::fs::remove_dir_all(&scratch);
    result
}

/// The entry this machine's platform maps to, when a server declares one per
/// platform.
///
/// `Ok(None)` means the single-source form, with nothing platform-specific to
/// choose. A map without this platform's tag is an error rather than a
/// fallback: fetching another platform's binary would install something that
/// cannot run. The error names the tag it looked for, the tags the map does
/// carry, and every tag the format knows — that is the whole of what the
/// author needs to fix it.
fn platform_entry(server: &McpServerDecl) -> Result<Option<&PlatformSource>> {
    if server.platforms.is_empty() {
        return Ok(None);
    }
    let tag = crate::fetch::platform_tag();
    match server.platforms.get(tag) {
        Some(entry) => Ok(Some(entry)),
        None => Err(anyhow!(
            "mcp '{}' declares no source for platform '{tag}'; it has {}. Supported platforms: {}",
            server.id,
            server.platforms.keys().map(String::as_str).collect::<Vec<_>>().join(", "),
            crate::fetch::PLATFORMS.join(", ")
        )),
    }
}

/// The single file a one-file fetch produced, which is what an archive
/// download stores under the cache entry.
fn only_file(dir: &Path) -> Result<PathBuf> {
    let mut files = Vec::new();
    for entry in
        std::fs::read_dir(dir).with_context(|| format!("listing {}", dir.display()))?
    {
        let entry = entry?;
        if entry.file_type()?.is_file() {
            files.push(entry.path());
        }
    }
    match files.len() {
        1 => Ok(files.pop().expect("one file")),
        count => Err(anyhow!(
            "{} holds {count} files; an archive fetch stores exactly one",
            dir.display()
        )),
    }
}

/// Install an `npm:`-sourced MCP server into its directory under the pack and
/// point `command` at the shim npm generated.
///
/// `npm install --prefix <dir>` builds the dependency tree inside the server's
/// own directory, so nothing is installed globally and the pack runs offline
/// afterwards. The command is written back absolute, exactly like a fetched
/// server's: the loader spawns stdio commands against the caller's working
/// directory rather than the pack's.
///
/// `bin_hint` is the pack's declared `command`, which selects among the
/// executables the package exposes; when it matches none, the package's own
/// name decides, and a single-executable package needs no hint at all.
async fn install_npm_server(
    local: &Path,
    mcp_root: &Path,
    server: &mut McpServerDecl,
    package: &str,
    version: Option<&str>,
    bin_hint: Option<&str>,
) -> Result<()> {
    std::fs::create_dir_all(local)
        .with_context(|| format!("creating {}", local.display()))?;

    let spec = match version {
        Some(version) => format!("{package}@{version}"),
        None => package.to_string(),
    };

    // npm resolves through the same lookup `npx` does — on Windows it is
    // `npm.cmd`, which `Command` needs spelled out.
    let npm = crate::mcp::McpClient::resolve_program("npm");
    let mut command = tokio::process::Command::new(&npm);
    command
        .arg("install")
        .arg("--prefix")
        .arg(local)
        .arg("--no-audit")
        .arg("--no-fund")
        .arg(&spec)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true);
    let output = tokio::time::timeout(
        std::time::Duration::from_secs(600),
        command.output(),
    )
    .await
    .map_err(|_| anyhow!("npm install {spec} timed out after 10 minutes"))?
    .with_context(|| {
        format!(
            "running {} — an 'npm:' source needs Node.js/npm on PATH; \
             install Node.js, or declare the server another way",
            npm.display()
        )
    })?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let tail: Vec<&str> = stderr.lines().rev().take(15).collect();
        let tail: Vec<&str> = tail.into_iter().rev().collect();
        return Err(anyhow!(
            "npm install {spec} failed ({}): {}",
            output.status,
            tail.join("\n")
        ));
    }

    // Which executable the package exposes: `bin` is either one path (the
    // command takes the package's unscoped name) or a map of names to paths.
    let manifest_path = local
        .join("node_modules")
        .join(package)
        .join("package.json");
    let text = std::fs::read_to_string(&manifest_path).with_context(|| {
        format!(
            "reading {}; npm reported success installing {spec} but its \
             package.json is missing",
            manifest_path.display()
        )
    })?;
    let manifest: serde_json::Value = serde_json::from_str(&text)
        .with_context(|| format!("parsing {}", manifest_path.display()))?;
    let unscoped = package.rsplit('/').next().unwrap_or(package);
    let bin_name = match manifest.get("bin") {
        Some(serde_json::Value::String(_)) => unscoped.to_string(),
        Some(serde_json::Value::Object(map)) => {
            let chosen = bin_hint
                .filter(|hint| map.contains_key(*hint))
                .map(str::to_string)
                .or_else(|| map.contains_key(unscoped).then(|| unscoped.to_string()))
                .or_else(|| {
                    (map.len() == 1)
                        .then(|| map.keys().next().expect("one key").clone())
                })
                .ok_or_else(|| {
                    anyhow!(
                        "npm package {spec} exposes several commands ({}) \
                         and mcp '{}' declares no \"command\" naming which \
                         one to run",
                        map.keys().map(String::as_str).collect::<Vec<_>>().join(", "),
                        server.id
                    )
                })?;
            chosen
        }
        _ => {
            return Err(anyhow!(
                "npm package {spec} declares no \"bin\"; it cannot be run \
                 as an MCP server"
            ))
        }
    };

    // The `.bin` shim: on Windows npm writes `<name>.cmd`, elsewhere an
    // extensionless script with a shebang. Both launch directly.
    #[cfg(windows)]
    let shim = format!("{bin_name}.cmd");
    #[cfg(not(windows))]
    let shim = bin_name.clone();

    let installed = local
        .join("node_modules")
        .join(".bin")
        .join(&shim);
    if !installed.is_file() {
        let available = std::fs::read_dir(local.join("node_modules").join(".bin"))
            .map(|entries| {
                entries
                    .filter_map(Result::ok)
                    .map(|entry| entry.file_name().to_string_lossy().to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            })
            .unwrap_or_else(|_| "(nothing)".into());
        return Err(anyhow!(
            "npm installed {spec} but produced no .bin/{shim}; it provides: {available}"
        ));
    }

    server.command = Some(
        mcp_root
            .join(&server.id)
            .join("node_modules")
            .join(".bin")
            .join(&shim)
            .to_string_lossy()
            .replace('\\', "/"),
    );
    Ok(())
}

// ------------------------------------------------------------------ install
