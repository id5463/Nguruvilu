//! `ngu runtime`：内核自己现在是什么样。
//!
//! 报告的是**装配完成之后**的状态：模型路由、缓存策略、工具表、生效的人格。
//! 它需要活的 `Runtime`，所以拿不到它时命令行仍然直接调用，不走路由。

use anyhow::Result;

use nguruvilu::hotreload::{CachePolicy, ModelRoute, Runtime};
use nguruvilu::skills::SkillRegistry;

/// Describe the session that would run.
///
/// Answered from the assembled kernel and the runtime a turn reads, so the
/// tools and services reported here are the ones a model would actually be
/// handed — a fresh table with only the base tools would describe a session
/// that does not exist.
fn show_runtime(
    route: &ModelRoute,
    runtime: &Runtime,
    cache_policy: CachePolicy,
    skills: &SkillRegistry,
    as_json: bool,
    kernel: &Kernel,
) -> Result<()> {
    let tools = kernel.tools().names();
    let services: Vec<serde_json::Value> = kernel
        .service_list()
        .iter()
        .map(|(name, realm, owner)| {
            serde_json::json!({ "name": name, "realm": realm.to_string(), "owner": owner })
        })
        .collect();

    let policy: Vec<serde_json::Value> = runtime
        .policy_table()
        .into_iter()
        .map(|(kind, consent)| {
            serde_json::json!({
                "kind": kind.as_str(),
                "consent": match consent {
                    Consent::AutoAllow => "auto-allow",
                    Consent::Ask => "ask",
                    Consent::Deny => "deny",
                },
            })
        })
        .collect();

    let network = serde_json::json!({
        "request_timeout_secs": route.network.request_timeout_secs,
        "pool_idle_timeout_secs": route.network.pool_idle_timeout_secs,
        "retry_attempts": route.network.retry_attempts,
        "retry_backoff_ms": route.network.retry_backoff_ms,
    });

    if as_json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "model": route.model,
                "base_url": route.base_url,
                "api_key": if route.api_key.is_empty() { "(not set)" } else { "(set)" },
                "cache_policy": format!("{cache_policy:?}"),
                "network": network,
                "tools": tools,
                "services": services,
                "policy": policy,
                "skill_roots": skills.roots().iter().map(|r| r.display().to_string()).collect::<Vec<_>>(),
                "skill_count": skills.len(),
                "change_kinds": ChangeKind::all().iter().map(|k| k.as_str()).collect::<Vec<_>>(),
            }))?
        );
        return Ok(());
    }

    println!("model:        {}", route.model);
    println!("base url:     {}", route.base_url);
    println!(
        "api key:      {}",
        if route.api_key.is_empty() { "(not set)" } else { "(set)" }
    );
    println!("cache policy: {cache_policy:?}");
    println!(
        "network:      request {}s, pool idle {}s, {} retry(ies) from {}ms",
        route.network.request_timeout_secs,
        route.network.pool_idle_timeout_secs,
        route.network.retry_attempts,
        route.network.retry_backoff_ms,
    );
    println!("\ntools ({}):", tools.len());
    println!("  {}", tools.join(", "));
    println!("\nservices:");
    if services.is_empty() {
        println!("  (none)");
    } else {
        for service in &services {
            println!(
                "  {:<12} {} [{}]",
                service["name"].as_str().unwrap_or(""),
                service["owner"].as_str().unwrap_or(""),
                service["realm"].as_str().unwrap_or(""),
            );
        }
    }
    println!("\nconsent policy (global changes):");
    for entry in &policy {
        println!(
            "  {:<14} {}",
            entry["kind"].as_str().unwrap_or(""),
            entry["consent"].as_str().unwrap_or("")
        );
    }
    println!("\nskill roots:");
    if skills.roots().is_empty() {
        println!("  (none configured)");
    } else {
        for root in skills.roots() {
            println!("  {}", root.display());
        }
    }
    println!("\nskills found: {}", skills.len());
    Ok(())
}
