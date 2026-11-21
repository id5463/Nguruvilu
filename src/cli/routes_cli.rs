//! `ngu routes`：直接调一条内核路由，参数走 JSON。
//!
//! 这是"一切皆调用"那条约定的命令行出口：任何注册过的路由，都能在这里被
//! 按名字调一次，不需要先有一个包。

use anyhow::{anyhow, Result};

/// `ngu routes` —— 列出路由表，或调一条。
///
/// 这是"一切皆调用"的可验证入口：表里有什么、能不能调通，一条命令看清。
fn routes_command(call: Option<&str>, raw_args: &str, json: bool) -> Result<()> {
    let dispatch = nguruvilu::kernel_routes::dispatch();

    let Some(name) = call else {
        let names = dispatch.names();
        if json {
            println!("{}", serde_json::to_string_pretty(&names)?);
        } else {
            for line in &names {
                println!("{line}");
            }
            println!();
            println!("{} 条路由。加 --call <名字> --args '<JSON>' 真的调一条。", names.len());
        }
        return Ok(());
    };

    let args: serde_json::Value =
        serde_json::from_str(raw_args).with_context(|| format!("--args 不是 JSON：{raw_args}"))?;

    // 已经在 tokio 的 runtime 里了（`run` 整个跑在 block_on 之内），
    // 再 `block_on` 一次就是 "Cannot start a runtime from within a runtime"。
    // 所以这里用 `block_in_place` + `Handle::block_on`，让出当前 worker 去等，
    // 而不是另起一个 runtime。
    let call = tokio::task::block_in_place(|| {
        tokio::runtime::Handle::current().block_on(dispatch.call(name, args))
    });

    match call {
        Ok(value) => {
            if json {
                println!("{}", serde_json::to_string_pretty(&serde_json::json!({
                    "route": name,
                    "result": value,
                }))?);
            } else {
                println!("{}", serde_json::to_string_pretty(&value)?);
            }
            Ok(())
        }
        Err(e) => {
            if json {
                println!("{}", serde_json::to_string_pretty(&serde_json::json!({
                    "route": name,
                    "error": e,
                }))?);
                std::process::exit(1);
            }
            anyhow::bail!("路由 {name} 失败：{e}")
        }
    }
}
