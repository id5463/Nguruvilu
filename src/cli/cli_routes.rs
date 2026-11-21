//! 命令行自己那两条路由的注册：`cli.runtime.skills`、`cli.runtime.assembly`。
//!
//! # 为什么只有这两条
//!
//! `show_runtime` 要一个已经装好的 `Runtime` 和 `ModelRoute` —— 那是启动过程
//! 中才有的东西，不是一条路由的入参能表达的。把它压成 JSON 参数会让"换掉内
//! 核运行时报告"变成"模拟启动一遍"，比直接改源码更难。
//!
//! 留下的是这两条：技能目录怎么列、装配清单怎么读。

use std::path::PathBuf;

use nguruvilu::skills::SkillRegistry;

use super::{list_skills, show_assembly};


// ---------------------------------------------------------------- the routes

/// 注册 `ngu skills` 与 `ngu assembly` 这两条能自成一体的命令行路径。
///
/// # 为什么只有两条
///
/// `show_runtime` 要拿一个已经装好的 `Runtime` 和 `ModelRoute` —— 那是启动
/// 过程中才有的东西，不是一条路由的入参能表达的。把它压成 JSON 参数会让
/// 「换掉内核运行时报告」变成「模拟启动一遍」，那比直接改源码更难。
///
/// 所以留在这里直接调用；能被替换的是这两条：技能目录怎么列、装配清单怎么读。
pub fn register(dispatch: &nguruvilu::dispatch::Dispatch) {
    use nguruvilu::dispatch::{Args, Chain, Fut};
    use std::sync::Arc;

    let json_flag = |args: &Args| args.get("json").and_then(|v| v.as_bool()).unwrap_or(false);

    dispatch.register(
        "cli.runtime.skills",
        Arc::new(|args: Args, _next: Chain| -> Fut {
            Box::pin(async move {
                let verbose = args.get("verbose").and_then(|v| v.as_bool()).unwrap_or(false);
                let mut skills = SkillRegistry::default();
                skills.scan().map_err(|e| format!("{e:#}"))?;
                list_skills(&skills, verbose, json_flag(&args)).map_err(|e| format!("{e:#}"))?;
                Ok(serde_json::json!({ "ok": true }))
            })
        }),
    );

    dispatch.register(
        "cli.runtime.assembly",
        Arc::new(|args: Args, _next: Chain| -> Fut {
            Box::pin(async move {
                let file = args
                    .get("file")
                    .and_then(|v| v.as_str())
                    .map(PathBuf::from)
                    .ok_or_else(|| "cli.runtime.assembly needs a file".to_string())?;
                let platform = args
                    .get("platform")
                    .and_then(|v| v.as_str())
                    .unwrap_or("linux-x64")
                    .to_string();
                show_assembly(&file, &platform, json_flag(&args)).map_err(|e| format!("{e:#}"))?;
                Ok(serde_json::json!({ "ok": true }))
            })
        }),
    );
}

