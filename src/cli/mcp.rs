//! `ngu mcp import`：把一个 MCP server 的声明转成一个包的清单片段。
//!
//! 从别的工具搬过来的 MCP 声明形状各异，`classify_server` 认出它属于哪一类，
//! `convert_mcp_servers` 把它翻成 `ngu.index.json` 里那段。翻不了的如实说明，
//! 让调用方能落到下一个 reader。

use std::path::PathBuf;

use anyhow::{anyhow, Context, Result};

fn mcp_import_command(
    file: &Path,
    out: Option<&Path>,
    name: Option<&str>,
    license: &str,
) -> Result<()> {
    let raw =
        std::fs::read_to_string(file).with_context(|| format!("reading {}", file.display()))?;
    let doc: serde_json::Value = serde_json::from_str(&raw)
        .with_context(|| format!("parsing {} as JSON", file.display()))?;

    let (servers, skipped) = convert_mcp_servers(&doc)?;

    let pack_name = match name {
        Some(name) => name.trim().to_string(),
        None => sanitize_pack_name(
            &file
                .file_stem()
                .map(|stem| stem.to_string_lossy().to_string())
                .unwrap_or_else(|| "mcp".into()),
        ),
    };
    if pack_name.is_empty() {
        anyhow::bail!("the pack name is empty after sanitizing; pass --name");
    }

    let dir = out.map(Path::to_path_buf).unwrap_or_else(|| PathBuf::from(&pack_name));
    if dir.exists() {
        let non_empty = std::fs::read_dir(&dir)?.next().transpose()?.is_some();
        if non_empty {
            anyhow::bail!(
                "{} exists and is not empty — refusing to overwrite; pass --out for another directory",
                dir.display()
            );
        }
    }
    std::fs::create_dir_all(&dir)?;

    let manifest = serde_json::json!({
        "formatVersion": 1,
        "game": "nguruvilu",
        "name": pack_name,
        "versionId": "1.0.0",
        "license": license,
        "kernelVersion": "0.1.0",
        "dependencies": { "nguruvilu": ">=0.1.0" },
        "mcp": "mcp.json",
    });
    std::fs::write(
        dir.join(nguruvilu::pack::MANIFEST_NAME),
        serde_json::to_string_pretty(&manifest)?,
    )?;
    std::fs::write(
        dir.join("mcp.json"),
        serde_json::to_string_pretty(&nguruvilu::pack::McpFile {
            servers: servers.clone(),
        })?,
    )?;
    std::fs::write(
        dir.join("README.md"),
        generated_pack_readme(&pack_name, file, license, &servers, &skipped),
    )?;

    println!("imported {} server(s) into {}", servers.len(), dir.display());
    for server in &servers {
        println!(
            "  + {}  ({})",
            server.id,
            server.command.as_deref().unwrap_or("(no command)")
        );
    }
    for (id, reason) in &skipped {
        println!("  - {id}: {reason}");
    }
    println!();
    println!(
        "next: ngu pack {dir} --out {name}-1.0.0.ngupack   then   ngu install <the archive>",
        dir = dir.display(),
        name = pack_name
    );
    println!("note: each server's runtime (node, …) and its own license are yours to provide.");
    Ok(())
}

/// Convert a config document into our servers plus what had to be left out.
fn convert_mcp_servers(
    doc: &serde_json::Value,
) -> Result<(Vec<nguruvilu::pack::McpServerDecl>, Vec<(String, String)>)> {
    let mut servers = Vec::new();
    let mut skipped: Vec<(String, String)> = Vec::new();

    if let Some(map) = doc.get("mcpServers").and_then(|v| v.as_object()) {
        for (id, entry) in map {
            match classify_server(id, entry) {
                Ok(server) => servers.push(server),
                Err(reason) => skipped.push((id.clone(), reason)),
            }
        }
        return Ok((servers, skipped));
    }
    if let Some(list) = doc.get("servers").and_then(|v| v.as_array()) {
        // Our own format: pass it through the same strict reader an install
        // would use, so an invalid file fails here rather than at load.
        let ours: nguruvilu::pack::McpFile = serde_json::from_value(
            serde_json::json!({ "servers": list.clone() }),
        )
        .context("the servers array does not match our mcp.json shape")?;
        return Ok((ours.servers, skipped));
    }
    anyhow::bail!(
        "no servers found: expected an object with 'mcpServers' (Claude Code / \
         Claude Desktop) or 'servers' (our mcp.json)"
    );
}

/// One entry → one declaration, or the reason it was left out.
fn classify_server(
    id: &str,
    entry: &serde_json::Value,
) -> Result<nguruvilu::pack::McpServerDecl, String> {
    let obj = entry
        .as_object()
        .ok_or_else(|| "entry is not an object".to_string())?;
    let kind = obj
        .get("type")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_else(|| {
            if obj.contains_key("url") { "http" } else { "unknown" }
        });
    let command = obj
        .get("command")
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|c| !c.is_empty());
    let Some(command) = command else {
        // No command: either an endpoint (a URL every message posts to) or
        // something with nothing to run. An endpoint imports into our
        // streamable-http form so it connects instead of being dropped.
        let url = obj
            .get("url")
            .and_then(serde_json::Value::as_str)
            .map(str::trim)
            .filter(|url| !url.is_empty());
        if let (true, Some(url)) = ((kind == "http" || kind == "unknown"), url) {
            let headers = obj
                .get("headers")
                .and_then(serde_json::Value::as_object)
                .map(|map| {
                    map.iter()
                        .filter_map(|(key, value)| {
                            value
                                .as_str()
                                .map(|text| (key.clone(), text.to_string()))
                        })
                        .collect::<std::collections::BTreeMap<String, String>>()
                })
                .unwrap_or_default();
            return Ok(nguruvilu::pack::McpServerDecl {
                tools: Vec::new(),
                id: id.to_string(),
                transport: "streamable-http".into(),
                source: None,
                sha256: None,
                command: None,
                args: Vec::new(),
                env: std::collections::BTreeMap::new(),
                scope: Some("session".into()),
                via_proxy: None,
                platforms: std::collections::BTreeMap::new(),
                url: Some(url.to_string()),
                headers,
            });
        }
        if kind == "http" {
            return Err("http transport declares no url".to_string());
        }
        return Err(format!(
            "'{kind}' transport is not supported (stdio and streamable-http are)"
        ));
    };
    let args = obj
        .get("args")
        .and_then(serde_json::Value::as_array)
        .map(|list| {
            list.iter()
                .map(|value| match value {
                    serde_json::Value::String(text) => text.clone(),
                    other => other.to_string(),
                })
                .collect()
        })
        .unwrap_or_default();
    let env = obj
        .get("env")
        .and_then(serde_json::Value::as_object)
        .map(|map| {
            map.iter()
                .map(|(key, value)| match value {
                    serde_json::Value::String(text) => (key.clone(), text.clone()),
                    other => (key.clone(), other.to_string()),
                })
                .collect::<std::collections::BTreeMap<String, String>>()
        })
        .unwrap_or_default();
    Ok(nguruvilu::pack::McpServerDecl {
        tools: Vec::new(),
        id: id.to_string(),
        transport: "stdio".into(),
        source: None,
        sha256: None,
        command: Some(command.to_string()),
        args,
        env,
        scope: Some("session".into()),
        via_proxy: None,
        platforms: std::collections::BTreeMap::new(),
        url: None,
        headers: std::collections::BTreeMap::new(),
    })
}

