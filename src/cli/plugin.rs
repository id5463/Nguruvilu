//! `ngu plugin`：直接加载一个插件二进制，或调它一个工具。
//!
//! 用于开发插件时不必先把插件打成包。它走的仍是内核那条加载路径：
//! 加载、问能力、调一个工具，三步与从包里加载时相同。

use std::path::PathBuf;

use anyhow::{anyhow, Context, Result};

use super::args::PluginAction;

fn plugin_command(action: &PluginAction, as_json: bool) -> Result<()> {
    match action {
        PluginAction::Load { path } => plugin_load(path, as_json),
        PluginAction::Call { path, tool, args } => plugin_call(path, tool, args, as_json),
    }
}

/// Load a plugin and report what it contributes.
///
/// Loading runs native code in this process, so this command is the explicit
/// opt-in: nothing loads a library implicitly.
fn plugin_load(path: &PathBuf, as_json: bool) -> Result<()> {
    let plugin = unsafe { nguruvilu::dylib::DynamicPlugin::load(path)? };

    // Tools are registered through a callback during `apply`, so the plugin has
    // to be applied once before its contributions are visible.
    let ctx = nguruvilu::plugin::PluginCtx::detached(plugin.name());
    let contributions = PluginTrait::apply(&plugin, &ctx)?;
    let tools: Vec<String> = contributions.tools.iter().map(|t| t.name.clone()).collect();
    let logs = plugin.log_lines();
    // The library stays mapped for the life of the process on purpose, and a
    // second load of the *same path* returns the module already in memory —
    // `DllMain` does not run again, so the plugin's statics carry over. Both
    // facts used to be invisible from outside; they are the reason a reload can
    // behave differently from a fresh start.
    let lib_ref = plugin.lib_ref();
    let leaked = nguruvilu::dylib::leaked_libraries();
    let mine = leaked.iter().find(|entry| entry.path == *path);
    let loads = mine.map(|entry| entry.loads).unwrap_or(0);

    if as_json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "ok": true,
                "path": plugin.path().display().to_string(),
                "name": plugin.name(),
                "version": plugin.meta().version,
                "description": plugin.meta().description,
                "abi_version": plugin.meta().abi_version,
                "inject": plugin.inject(),
                "provide": plugin.provide(),
                "tools": tools,
                "library": {
                    "lib_ref": lib_ref,
                    "loads": loads,
                    "leaked_total": leaked.len(),
                },
                "log": logs.iter().map(|(level, text)| serde_json::json!({"level": level, "text": text})).collect::<Vec<_>>(),
            }))?
        );
        return Ok(());
    }

    println!("loaded {} from {}", plugin.name(), plugin.path().display());
    if let Some(version) = &plugin.meta().version {
        println!("version:     {version}");
    }
    if let Some(description) = &plugin.meta().description {
        println!("description: {description}");
    }
    if let Some(abi) = plugin.meta().abi_version {
        println!("abi:         {abi}");
    }
    if !plugin.inject().is_empty() {
        println!("injects:     {}", plugin.inject().join(", "));
    }
    if !plugin.provide().is_empty() {
        println!("provides:    {}", plugin.provide().join(", "));
    }
    println!("tools ({}):", tools.len());
    for tool in &tools {
        println!("  {tool}");
    }
    println!("library:     ref {lib_ref}, {loads} load(s) of this path here");
    if loads > 1 {
        println!(
            "note:        this module was already mapped, so it was not loaded again — its \
             statics carry over from the previous load"
        );
    }
    if leaked.len() > 1 {
        println!("leaked:      {} librar(y|ies) kept for the life of this process", leaked.len());
    }
    for (level, text) in logs {
        let name = match level {
            0 => "debug",
            1 => "info",
            2 => "warn",
            _ => "error",
        };
        println!("[{name}] {text}");
    }
    Ok(())
}

/// Load a plugin and call one of its tools, for verification.
fn plugin_call(path: &PathBuf, tool: &str, args: &str, as_json: bool) -> Result<()> {
    let plugin = unsafe { nguruvilu::dylib::DynamicPlugin::load(path)? };

    let ctx = nguruvilu::plugin::PluginCtx::detached(plugin.name());
    PluginTrait::apply(&plugin, &ctx)?;

    let arguments: serde_json::Value =
        serde_json::from_str(args).with_context(|| format!("parsing arguments {args:?}"))?;
    let result = plugin.call(tool, arguments);

    if as_json {
        let payload = match &result {
            Ok(content) => serde_json::json!({"ok": true, "tool": tool, "content": content}),
            Err(error) => serde_json::json!({"ok": false, "tool": tool, "error": format!("{error:#}")}),
        };
        println!("{}", serde_json::to_string_pretty(&payload)?);
        // A plugin-reported failure is a failed call, and the exit code says so.
        if result.is_err() {
            std::process::exit(1);
        }
        return Ok(());
    }

    match result {
        Ok(content) => {
            println!("{content}");
            Ok(())
        }
        Err(error) => Err(error),
    }
}
