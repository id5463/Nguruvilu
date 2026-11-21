//! `ngu injections`：列注入规则，或问"这段文字会触发哪些规则"。
//!
//! 规则文件由包提供（`injections.json`），这一层只读它并回答，
//! 不修改任何状态。

use anyhow::Result;

fn show_injections(for_text: Option<&str>, as_json: bool) -> Result<()> {
    let path = nguruvilu::context::InjectionEngine::default_path();
    let engine = nguruvilu::context::InjectionEngine::load_default();

    // What would activate for the given text, if any.
    let activated: Vec<String> = match for_text {
        Some(text) => {
            let probe = vec![nguruvilu::message::Message::user(text)];
            nguruvilu::context::InjectionSource::inject(
                &engine,
                &probe,
                128_000,
                nguruvilu::hotreload::CachePolicy::Balanced,
            )
                .activated
        }
        None => Vec::new(),
    };

    if as_json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "path": path.display().to_string(),
                "exists": path.is_file(),
                "budget_percent": engine.budget_percent,
                "budget_cap": engine.budget_cap,
                "max_recursion": engine.max_recursion,
                "entries": engine.entries.iter().map(|entry| serde_json::json!({
                    "id": entry.id,
                    "constant": entry.constant,
                    "triggers": entry.triggers,
                    "position": format!("{:?}", entry.position),
                    "depth": entry.depth,
                    "order": entry.order,
                    "group": entry.group,
                    "ignore_budget": entry.ignore_budget,
                    "enabled": entry.enabled,
                })).collect::<Vec<_>>(),
                "activated": activated,
            }))?
        );
        return Ok(());
    }

    println!("file: {}", path.display());
    if !path.is_file() {
        println!("(no file yet — write one to add standing rules or triggered hints)");
        return Ok(());
    }
    println!(
        "budget: {}% of the window{}, up to {} recursion passes",
        engine.budget_percent,
        match engine.budget_cap {
            Some(cap) => format!(" capped at {cap} tokens"),
            None => String::new(),
        },
        engine.max_recursion
    );
    println!("entries: {}", engine.len());
    for entry in &engine.entries {
        let when = if entry.constant {
            "always".to_string()
        } else if entry.triggers.is_empty() {
            "never (no triggers)".to_string()
        } else {
            format!("on {}", entry.triggers.join(", "))
        };
        let where_to = match entry.position {
            nguruvilu::context::PositionKind::AtDepth => {
                format!("at-depth {}", entry.depth.unwrap_or(1))
            }
            other => other.as_str().to_string(),
        };
        let mark = if activated.contains(&entry.id) { " *" } else { "  " };
        println!("{mark} {:<20} {:<28} {when}", entry.id, where_to);
    }
    if for_text.is_some() {
        println!("\n* = activates for the given text");
    }
    Ok(())
}

