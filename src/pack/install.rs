//! 安装一个包：解开它、取它点名的东西、写它要的那份装配清单。
//!
//! 安装写的是 `assembly.yaml` 而不是包里带的那份，所以包作者从来不用维护
//! 同一件事的两个描述。已经装过的包按账本里的哈希跳过，所以装两次是廉价的。

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use serde_json::Value;

use super::archive::{packable_files, unpack};
use super::enabled::{is_enabled, read_manifest, write_manifest};
use super::format::{ASSEMBLY_NAME, FILES_DIR, InstalledPack, PackContents};
use super::manifest::{McpServerDecl, PackManifest, PlatformSource};

pub async fn install(pack_path: &Path, packs_dir: &Path) -> Result<InstalledPack> {
    let staging = std::env::temp_dir().join(format!(
        "ngu-install-{}",
        uuid::Uuid::new_v4().simple()
    ));
    let result = install_inner(pack_path, packs_dir, &staging).await;
    let _ = std::fs::remove_dir_all(&staging);
    result
}

async fn install_inner(
    pack_path: &Path,
    packs_dir: &Path,
    staging: &Path,
) -> Result<InstalledPack> {
    let manifest = unpack(pack_path, staging)?;

    // The declared kernel range is not enforced. A pack that names a range this
    // kernel is outside of installs anyway, and whatever the mismatch actually
    // breaks — a tool that never appears, a field that reads as absent — is
    // reported where it happens. Refusing here would be this program deciding
    // for the user that they may not try it.

    // Fetch what the pack names. Anything already in the cache is not
    // re-downloaded, and an offline pack carries its own copy.
    let embedded = staging.join(FILES_DIR);
    let fetcher = Fetcher::new()?;
    for reference in manifest.references()? {
        let Reference { kind, id, source, sha256 } = reference;
        let (source, expected) = (&source, sha256);
        let local = embedded.join(id);
        if local.is_dir() {
            continue;
        }
        if !source.is_remote() {
            continue;
        }
        let fetched = fetcher.fetch(source, expected).await.with_context(|| {
            format!("fetching {kind} '{id}' from {}", source.describe())
        })?;
        copy_tree(&fetched.path, &local)?;
    }

    // Fetch content that lives elsewhere, then rewrite the manifest so the
    // installed pack is self-contained: everything downstream reads a path
    // inside it, and nothing needs to know the content arrived over the wire.
    let mut manifest = manifest;
    let mut rewritten = false;
    for (field, _) in CONTENT_FILES {
        let Some(reference) = manifest.content_file(field).cloned() else {
            continue;
        };
        let Some(source) = reference.source() else {
            continue;
        };
        let parsed = Source::parse(source)
            .with_context(|| format!("{field} source"))?;
        let fetched = fetcher
            .fetch(&parsed, reference.sha256())
            .await
            .with_context(|| format!("fetching {field} from {source}"))?;
        let local = staging.join(FILES_DIR).join(field);
        copy_tree(&fetched.path, &local)?;
        manifest.set_content(
            field,
            ContentRef::carried(format!("{FILES_DIR}/{field}/{}", reference.path())),
        );
        rewritten = true;
    }
    if rewritten {
        write_manifest(staging, &manifest)?;
    }

    // MCP servers named by a source are fetched alongside skills and plugins.
    // A server only reachable through npx is a server the user has to install a
    // runtime for.
    // Where this pack will sit once placed. A platform-selected server's
    // command is written back as an absolute path under it: the loader
    // spawns stdio commands against the caller's working directory rather
    // than the pack's, so a relative `files/mcp/...` would resolve only when
    // `ngu` happened to start in the right place.
    let mcp_root = packs_dir
        .join(format!("{}-{}", manifest.name, manifest.version_id))
        .join(FILES_DIR)
        .join("mcp");
    let mut mcp_decls = match manifest.mcp.as_ref() {
        Some(reference) => {
            let path = staging.join(reference.path());
            if path.is_file() {
                let text = std::fs::read_to_string(&path)?;
                serde_json::from_str::<McpFile>(&text)
                    .with_context(|| format!("{} is not a valid mcp file", path.display()))?
                    .servers
            } else {
                Vec::new()
            }
        }
        None => Vec::new(),
    };
    // A declaration with `platforms` fetches this machine's entry. An
    // archive URL is unpacked straight into the server's directory; every
    // other source fetches and copies, like the single-source loop below.
    // The chosen source and command are written back into the mcp file after
    // both loops, which is what lets the assembly render the command from
    // inside the pack.
    let mut mcp_changed = false;
    for server in &mut mcp_decls {
        let Some(entry) = platform_entry(server)? else {
            continue;
        };
        let source = entry.source.clone();
        let sha256 = entry.sha256.clone();
        let command = entry.command.clone();
        let parsed = Source::parse(&source)
            .with_context(|| format!("mcp '{}' source", server.id))?;
        // Fetch and unpack into staging — install replaces the destination
        // wholesale afterwards, so anything meant to survive must be here.
        let local = staging.join(FILES_DIR).join("mcp").join(&server.id);
        match &parsed {
            // A dependency, not a download: npm builds the whole tree inside
            // the server's own directory and the shim path is written back
            // absolute, like every other fetched command. Handled here rather
            // than fetched, because a tarball alone has no dependency tree.
            Source::Npm { package, version } => {
                install_npm_server(&local, &mcp_root, server, package, version.as_deref(), command.as_deref())
                    .await
                    .with_context(|| format!("installing mcp '{}' from {source}", server.id))?;
                server.source = Some(source);
                mcp_changed = true;
                continue;
            }
            Source::Url { url } if crate::fetch::is_archive_name(url) => {
                let expect = sha256.as_deref().ok_or_else(|| {
                    anyhow!(
                        "mcp '{}' names the archive {source} but declares no sha256; \
                         an archive is fetched by the hash of the file",
                        server.id
                    )
                })?;
                let fetched = fetcher
                    .fetch_file(url, expect)
                    .await
                    .with_context(|| format!("fetching mcp '{}' from {source}", server.id))?;
                let archive = only_file(&fetched.path)?;
                crate::fetch::extract_archive(&archive, &local)
                    .with_context(|| format!("unpacking mcp '{}' from {source}", server.id))?;
            }
            _ => {
                let fetched = fetcher
                    .fetch(&parsed, sha256.as_deref())
                    .await
                    .with_context(|| format!("fetching mcp '{}' from {source}", server.id))?;
                copy_tree(&fetched.path, &local)?;
            }
        }
        server.source = Some(source);
        // The effective command is the platform's override, else the base
        // name — either way an executable the fetch just placed, so it is
        // resolved to an absolute path under the installed pack.
        if let Some(name) = command.or_else(|| server.command.clone()) {
            let absolute = mcp_root.join(&server.id).join(name);
            server.command = Some(absolute.to_string_lossy().replace('\\', "/"));
        }
        mcp_changed = true;
    }
    for server in &mut mcp_decls {
        if !server.platforms.is_empty() {
            continue; // fetched above, with this platform's source
        }
        let Some(source) = server.source.clone() else {
            continue;
        };
        let parsed = Source::parse(&source)
            .with_context(|| format!("mcp '{}' source", server.id))?;
        let local = staging.join(FILES_DIR).join("mcp").join(&server.id);
        // Same rule as the platform loop above: an npm source is installed by
        // npm, never handed to the fetcher.
        if let Source::Npm { package, version } = &parsed {
            let hint = server.command.clone();
            install_npm_server(&local, &mcp_root, server, package, version.as_deref(), hint.as_deref())
                .await
                .with_context(|| format!("installing mcp '{}' from {source}", server.id))?;
            mcp_changed = true;
            continue;
        }
        let fetched = fetcher
            .fetch(&parsed, server.sha256.as_deref())
            .await
            .with_context(|| format!("fetching mcp '{}' from {source}", server.id))?;
        copy_tree(&fetched.path, &local)?;
    }

    // The assembly is rendered from this file and reads each server's
    // `source` and `command`, so the platform choice made above is written
    // back here — once, after both loops. From here on the installed pack
    // looks exactly like a single-source declaration for this machine.
    if mcp_changed {
        let path = staging
            .join(
                manifest
                    .mcp
                    .as_ref()
                    .expect("declared: a server was fetched")
                    .path(),
            );
        let text = serde_json::to_string_pretty(&McpFile {
            servers: mcp_decls,
        })?;
        std::fs::write(&path, format!("{text}\n"))
            .with_context(|| format!("writing {}", path.display()))?;
    }

    // A fetched interface lands under files/ui/, and the declaration is
    // rewritten to point inside the pack so the shell reads one layout whether
    // the interface arrived over the wire or was carried.
    if let Some(ui) = manifest.ui.clone() {
        if let Some(source) = &ui.source {
            let parsed = Source::parse(source)
                .with_context(|| format!("ui '{}' source", ui.id))?;
            let fetched = fetcher
                .fetch(&parsed, ui.sha256.as_deref())
                .await
                .with_context(|| format!("fetching ui '{}' from {source}", ui.id))?;
            let local = staging.join(FILES_DIR).join("ui");
            copy_tree(&fetched.path, &local)?;
            let mut carried = ui.clone();
            carried.source = None;
            carried.sha256 = None;
            carried.entry = format!("{FILES_DIR}/ui/{}", ui.entry.trim_start_matches('/'));
            manifest.ui = Some(carried);
            write_manifest(staging, &manifest)?;
        }
    }

    let destination = packs_dir.join(format!("{}-{}", manifest.name, manifest.version_id));
    if destination.exists() {
        std::fs::remove_dir_all(&destination)
            .with_context(|| format!("replacing {}", destination.display()))?;
    }
    std::fs::create_dir_all(packs_dir)
        .with_context(|| format!("creating {}", packs_dir.display()))?;

    let assembly = render_assembly(&manifest, staging)?;
    std::fs::write(staging.join(ASSEMBLY_NAME), &assembly)?;
    copy_tree(staging, &destination)?;

    let assembly_path = destination.join(ASSEMBLY_NAME);
    Ok(InstalledPack {
        manifest,
        path: destination,
        assembly: assembly_path.is_file().then_some(assembly_path),
        enabled: true,
    })
}

/// Render the assembly the loader reads, from the manifest and content files.
///
/// Install writes it rather than shipping it, so a pack author never has to
/// maintain two descriptions of the same thing — and so the loader keeps
/// reading one format while the distribution format evolves.
fn render_assembly(manifest: &PackManifest, staging: &Path) -> Result<String> {
    #[derive(Serialize)]
    struct OutBase {
        id: String,
        source: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        scope: Option<String>,
    }
    #[derive(Serialize)]
    struct OutPlugin {
        #[serde(flatten)]
        base: OutBase,
        #[serde(skip_serializing_if = "Option::is_none")]
        build: Option<crate::assembly::BuildSpec>,
    }
    #[derive(Serialize)]
    struct OutMcp {
        #[serde(flatten)]
        base: OutBase,
        transport: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        command: Option<String>,
        #[serde(skip_serializing_if = "Vec::is_empty")]
        args: Vec<String>,
        #[serde(skip_serializing_if = "BTreeMap::is_empty")]
        env: BTreeMap<String, String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        url: Option<String>,
        #[serde(skip_serializing_if = "BTreeMap::is_empty")]
        headers: BTreeMap<String, String>,
        #[serde(skip_serializing_if = "Vec::is_empty")]
        tools: Vec<McpToolDecl>,
    }
    #[derive(Serialize)]
    struct OutSkill {
        #[serde(flatten)]
        base: OutBase,
    }
    #[derive(Serialize)]
    struct OutStage {
        name: String,
        #[serde(skip_serializing_if = "Vec::is_empty")]
        plugins: Vec<OutPlugin>,
        #[serde(skip_serializing_if = "Vec::is_empty")]
        mcp: Vec<OutMcp>,
        #[serde(skip_serializing_if = "Vec::is_empty")]
        skills: Vec<OutSkill>,
    }
    #[derive(Serialize)]
    struct OutDefaults {
        scope: String,
        on_failure: String,
    }
    #[derive(Serialize)]
    struct Out {
        version: u32,
        name: String,
        defaults: OutDefaults,
        stages: Vec<OutStage>,
    }
    let platform = crate::fetch::platform_tag();

    let plugins: Vec<OutPlugin> = manifest
        .plugins
        .iter()
        .map(|plugin| {
            // A plugin the kernel already carries is *named*, not fetched: the
            // pack decides that it loads, the kernel decides what it is. This
            // is how a capability the kernel ships (a subagent, say) becomes a
            // pack's to enable without shipping a binary anywhere.
            //
            // Everything else is fetched at install time and lands in
            // `files/<id>`, so the loader gets a path inside the pack.
            // A source that names something already in reach is kept as
            // written: `builtin:delegate` (compiled in), `plugin:bin/x` (a
            // process the pack built), `dylib:…` (a library inside the pack),
            // and a path inside the pack. Only an entry that has to be *fetched*
            // gets rewritten to point at `files/<id>`, because that is where a
            // fetch puts it.
            let named = matches!(
                crate::fetch::Source::parse(&plugin.source),
                Ok(crate::fetch::Source::Builtin { .. })
                    | Ok(crate::fetch::Source::Local { .. })
            ) || plugin.source.starts_with("plugin:");
            let source = if named {
                plugin.source.clone()
            } else {
                let file = plugin
                    .platforms
                    .get(platform)
                    .map(|artifact| artifact.file.clone())
                    .unwrap_or_else(|| plugin.id.clone());
                format!("dylib:{FILES_DIR}/{}/{file}", plugin.id)
            };
            OutPlugin {
                base: OutBase {
                    id: plugin.id.clone(),
                    source,
                    scope: Some("global".into()),
                },
                // The build recipe travels with the entry, otherwise an
                // installed pack loses the one thing that turns its source into
                // an executable — and a raw install on a machine without the
                // binary fails with "the file is missing" for a pack that
                // carries everything it needs.
                build: plugin.build.clone(),
            }
        })
        .collect();

    // Skill ids must not collide with the plugin and MCP ids beside them.
    //
    // An assembly is one namespace: `duplicate entry id` is the loader refusing
    // exactly what this rendered. The manifest's author writes a skill id for
    // their own use (`web-search`), and it is entirely reasonable for that to be
    // the pack's own name — which the MCP entry beside it also uses. Measured on
    // `packs/web-search`, where the rendered assembly could not be applied at
    // all: the first time the pack was installed from an archive rather than run
    // from its source directory.
    // Every id already spoken for in this assembly, before the skills are
    // rendered. MCP server ids are read from `mcp.json` rather than assumed:
    // the file is the authority on what they are called, and a set built from
    // anything else would not notice a collision with one of them.
    let mcp_ids: Vec<String> = match manifest.mcp.as_ref() {
        Some(file) => {
            let path = staging.join(file.path());
            match std::fs::read_to_string(&path)
                .ok()
                .and_then(|text| serde_json::from_str::<McpFile>(&text).ok())
            {
                Some(parsed) => parsed.servers.into_iter().map(|s| s.id).collect(),
                // Unreadable here is reported by the MCP branch below, with the
                // path in the message; this pass just has nothing to reserve.
                None => Vec::new(),
            }
        }
        None => Vec::new(),
    };
    let mut taken: std::collections::BTreeSet<String> = manifest
        .plugins
        .iter()
        .map(|plugin| plugin.id.clone())
        .chain(mcp_ids.iter().cloned())
        .collect();
    let skills: Vec<OutSkill> = manifest
        .skills
        .iter()
        .map(|skill| OutSkill {
            base: OutBase {
                id: {
                    let mut id = skill.id.clone();
                    if taken.contains(&id) {
                        // The suffix names what it is, so the rendered file says
                        // why two entries exist rather than numbering them.
                        id = format!("{id}-skill");
                    }
                    taken.insert(id.clone());
                    id
                },
                // The authored source, unchanged — the same rule the plugin
                // branch applies to what it *names*. A carried skill ships
                // inside the archive (`skills/…`) and the loader resolves a
                // relative source against the pack directory; rewriting every
                // source to `files/<id>` pointed the loader at a path nothing
                // ever creates, because no skill fetch exists yet. Keeping the
                // authored source also leaves `github:` its own honest
                // "not implemented" message instead of "does not exist".
                source: skill.source.clone(),
                scope: Some("session".into()),
            },
        })
        .collect();

    let mcp: Vec<OutMcp> = match manifest.mcp.as_ref() {
        Some(file) => {
            let path = staging.join(file.path());
            let text = std::fs::read_to_string(&path)
                .with_context(|| format!("reading {}", path.display()))?;
            let parsed: McpFile = serde_json::from_str(&text)
                .with_context(|| format!("{} is not a valid mcp file", path.display()))?;
            // An endpoint transport with no endpoint cannot connect, and an
            // assembly that carried one would defer the complaint to load.
            for server in &parsed.servers {
                if server.transport == "streamable-http" && server.url.is_none() {
                    return Err(anyhow!(
                        "mcp '{}' uses streamable-http but declares no url",
                        server.id
                    ));
                }
            }
            parsed
                .servers
                .into_iter()
                .map(|server| {
                    if server.transport == "streamable-http" {
                        // The URL is the source: everything the loader needs
                        // to reach the server is the endpoint plus headers.
                        return OutMcp {
                            base: OutBase {
                                id: server.id.clone(),
                                source: format!(
                                    "streamable-http:{}",
                                    server.url.as_deref().unwrap_or_default()
                                ),
                                scope: server.scope.clone(),
                            },
                            transport: server.transport.clone(),
                            command: None,
                            args: Vec::new(),
                            env: BTreeMap::new(),
                            url: server.url.clone(),
                            headers: server.headers.clone(),
                            tools: server.tools.clone(),
                        };
                    }
                    // A fetched server runs from where it landed, so the
                    // command is resolved inside that directory. Without this
                    // the loader would look for it on PATH.
                    let command = match (&server.source, &server.command) {
                        (Some(_), Some(command)) => Some(
                            Path::new(FILES_DIR)
                                .join("mcp")
                                .join(&server.id)
                                .join(command)
                                .to_string_lossy()
                                .replace('\\', "/"),
                        ),
                        (_, command) => command.clone(),
                    };
                    OutMcp {
                        base: OutBase {
                            id: server.id.clone(),
                            source: command
                                .clone()
                                .map(|c| format!("stdio:{c} {}", server.args.join(" ")))
                                .unwrap_or_default()
                                .trim()
                                .to_string(),
                            scope: server.scope.clone(),
                        },
                        transport: server.transport.clone(),
                        command,
                        args: server.args.clone(),
                        env: server.env.clone(),
                        url: None,
                        headers: BTreeMap::new(),
                        tools: server.tools.clone(),
                    }
                })
                .collect()
        }
        None => Vec::new(),
    };

    let stages = if plugins.is_empty() && skills.is_empty() && mcp.is_empty() {
        Vec::new()
    } else {
        vec![OutStage {
            name: "pack".into(),
            plugins,
            mcp,
            skills,
        }]
    };

    let assembly = Out {
        version: 1,
        name: manifest.name.clone(),
        defaults: OutDefaults {
            scope: "session".into(),
            on_failure: "skip".into(),
        },
        stages,
    };
    Ok(serde_yaml::to_string(&assembly)?)
}

/// Every installed pack, newest name first.
pub fn installed(packs_dir: &Path) -> Result<Vec<InstalledPack>> {
    let mut out = Vec::new();
    let entries = match std::fs::read_dir(packs_dir) {
        Ok(entries) => entries,
        // A missing directory means nothing is installed, which is not an error.
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(out),
        Err(error) => {
            return Err(error).with_context(|| format!("listing {}", packs_dir.display()))
        }
    };

    for entry in entries {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let path = entry.path();
        // Rename an old manifest spelling in place, now, on sight. Doing it
        // here rather than in a startup pass means every path that lists packs
        // migrates them — the shell, `pack list`, a test — so no caller has to
        // remember. A failure to rename is not fatal: the pack still reads under
        // its old name, and the next listing tries again.
        match migrate_manifest_name(&path) {
            Ok(true) => eprintln!(
                "[pack] {} 的清单已改名为 {MANIFEST_NAME}",
                path.display()
            ),
            Ok(false) => {}
            Err(error) => eprintln!(
                "[pack] {} 的清单改名失败（仍可读取）：{error:#}",
                path.display()
            ),
        }
        // A directory carrying either spelling is a pack; `read_manifest`
        // prefers the current name when a migration left both behind.
        //
        // No manifest at all is not "not a pack" — it is usually a pack that
        // lost one. The difference matters because this is exactly the shape a
        // half-placed or hand-copied pack takes: a directory full of files,
        // no manifest, and therefore no entry in any list. It used to be
        // skipped without a word, so the only symptom was an absence, and an
        // absence explains nothing. Named here so the log says which directory
        // and what name it was wearing.
        if manifest_path(&path).is_none() {
            eprintln!(
                "[pack] {} 不是一个完整的包：目录里没有 {}（有文件但没有清单）",
                path.display(),
                MANIFEST_NAME
            );
            continue;
        }
        // A pack with a broken manifest is reported rather than skipped: a
        // silent omission looks like the pack was never installed. Reported
        // means printed and left out of the list — failing the whole listing
        // would let one leftover directory stop every other pack from loading.
        let manifest = match read_manifest(&path) {
            Ok(manifest) => manifest,
            Err(error) => {
                eprintln!("[pack] ignoring {}: {error:#}", path.display());
                continue;
            }
        };
        let assembly = path.join(ASSEMBLY_NAME);
        let enabled = is_enabled(&path);
        out.push(InstalledPack {
            manifest,
            path,
            assembly: assembly.is_file().then_some(assembly),
            enabled,
        });
    }
    out.sort_by(|a, b| a.manifest.name.cmp(&b.manifest.name));
    Ok(out)
}

/// Where packs are installed: `$NGU_HOME/packs`, else `~/.nguruvilu/packs`.
pub fn default_packs_dir() -> PathBuf {
    crate::settings::data_dir().join("packs")
}

/// Read a pack's assembly, when it has one.
pub fn read_assembly(dir: &Path) -> Result<Option<crate::assembly::Assembly>> {
    let path = dir.join(ASSEMBLY_NAME);
    if !path.is_file() {
        return Ok(None);
    }
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("reading {}", path.display()))?;
    let assembly: crate::assembly::Assembly = serde_yaml::from_str(&text)
        .with_context(|| format!("{} is not a valid assembly", path.display()))?;
    Ok(Some(assembly))
}

/// The pack's assembly, rendering one when the directory carries none.
///
/// Install writes `assembly.yaml`; a directory placed by hand or by an older
/// install has its manifest and content but no assembly, so every load skips
/// it — the pack lists fine and does nothing. Rendering the document install
/// would render (from the same manifest and content files) makes such a pack
/// loadable without reinstalling it, and the repair is logged. A directory
/// with no manifest has nothing to render from and keeps its `None`.
pub fn ensure_assembly(dir: &Path) -> Result<Option<PathBuf>> {
    let path = dir.join(ASSEMBLY_NAME);
    if path.is_file() {
        return Ok(Some(path));
    }
    let Ok(manifest) = read_manifest(dir) else {
        return Ok(None);
    };
    let text = render_assembly(&manifest, dir)?;
    std::fs::write(&path, &text).with_context(|| format!("writing {}", path.display()))?;
    eprintln!(
        "[pack] {} had no {}; rendered one from its manifest",
        manifest.name, ASSEMBLY_NAME
    );
    Ok(Some(path))
}

/// A manifest as displayable pairs.
pub fn describe(manifest: &PackManifest) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    out.insert("name".into(), manifest.name.clone());
    if !manifest.description.is_empty() {
        out.insert("description".into(), manifest.description.clone());
    }
    out.insert("version".into(), manifest.version_id.clone());
    out.insert("license".into(), manifest.license.clone());
    out.insert("built against".into(), manifest.kernel_version.clone());
    out.insert("accepts".into(), manifest.dependencies.nguruvilu.clone());
    out.insert("skills".into(), manifest.skills.len().to_string());
    out.insert("plugins".into(), manifest.plugins.len().to_string());
    out
}

