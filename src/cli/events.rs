//! `ngu events`：读审计日志，按时间倒序列出发生了什么。
//!
//! 日志是 JSONL，每行一个事件。这一层只负责读和排版，不产生事件。

use anyhow::Result;

fn events_command(last: usize, as_json: bool) -> Result<()> {
    let path = nguruvilu::events::audit_path();
    let Ok(raw) = std::fs::read_to_string(&path) else {
        if as_json {
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "ok": true,
                    "path": path.display().to_string(),
                    "lines": [],
                }))?
            );
        } else {
            println!("no events yet — the log is {}", path.display());
        }
        return Ok(());
    };
    let lines: Vec<&str> = raw.lines().filter(|l| !l.trim().is_empty()).collect();
    let start = lines.len().saturating_sub(last.max(1));
    if as_json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "ok": true,
                "path": path.display().to_string(),
                "count": lines.len(),
                "lines": &lines[start..],
            }))?
        );
        return Ok(());
    }
    println!(
        "{} — {} event(s), showing the last {}",
        path.display(),
        lines.len(),
        lines.len() - start
    );
    for line in &lines[start..] {
        match serde_json::from_str::<serde_json::Value>(line) {
            Ok(value) => println!(
                "{:>13}  {:<15}  {}",
                value["ts"].as_u64().unwrap_or(0),
                value["event"].as_str().unwrap_or("?"),
                nguruvilu::events::truncate(&value["payload"].to_string(), 200)
            ),
            Err(_) => println!("  (unparseable) {line}"),
        }
    }
    Ok(())
}

