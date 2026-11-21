//! `ngu export`：把一条会话写成别人能自己读的文件。
//!
//! 盘上的会话是 JSONL 回放日志 —— 对存储是对的，对拿到文件的人是错的。
//! 所以导出会把同一份内容重述成带解释的形态：Markdown 给人看，JSON 给
//! 另一个 agent 读。
//!
//! 与 `ngu stats` 的区别在这里：`stats` 对不存在的 id 报零，因为「什么都没记」
//! 是关于文件的事实；`export` 对不存在的 id 报错，因为导出一个不存在的会话
//! 是失败，不是空报告。

use std::path::PathBuf;

use anyhow::{anyhow, Result};

use nguruvilu::session::{JsonlStore, SessionStore};

use super::args::*;

/// `ngu export` — write a stored session out in a shape that can be shared.
///
/// A missing session is an error here, unlike `ngu stats`, which reports zeros
/// for an id with no file: a report about nothing is still a report, while an
/// export of a session that does not exist would be an empty file that looks
/// exactly like a real one.
fn export_command(
    store: &JsonlStore,
    id: &str,
    out: Option<&PathBuf>,
    requested: Option<&str>,
    as_json: bool,
) -> Result<()> {
    let session = store
        .load(id)?
        .ok_or_else(|| anyhow!("session not found: {id}"))?;
    let options = nguruvilu::export::ExportOptions {
        tool_output_cap: nguruvilu::export::DEFAULT_TOOL_OUTPUT_CAP,
        format: export_format(requested, as_json)?,
    };
    let rendered = nguruvilu::export::render(&session, &totals_of(&session), &options);

    match out {
        // `--out` turns the export into a file. Missing directories are created
        // rather than demanded, because `--out exports/session.md` failing
        // because `exports/` did not exist yet is a worse answer than making it.
        Some(path) => {
            if let Some(parent) = path.parent().filter(|parent| !parent.as_os_str().is_empty()) {
                std::fs::create_dir_all(parent)
                    .with_context(|| format!("creating {}", parent.display()))?;
            }
            std::fs::write(path, rendered.as_bytes())
                .with_context(|| format!("writing {}", path.display()))?;
            // One line on stdout, saying where the file went and how large it
            // is: the size is the only feedback that distinguishes a complete
            // export from one of a session that turned out to be nearly empty.
            println!("wrote {} ({} bytes)", path.display(), rendered.len());
        }
        // `print!` rather than `println!`: the rendered text already ends with
        // exactly one newline, so the bytes reaching a pipe here are the same
        // bytes `--out` writes to a file.
        None => print!("{rendered}"),
    }
    Ok(())
}

/// Which format an export should use, from `--format` and the global `--json`.
///
/// `--json` is answered first, before `--format` is even read: a script driving
/// this program passes `--json` to every subcommand, and one that quietly
/// ignored it because the subcommand's own flag defaulted to Markdown would hand
/// that script prose to parse as JSON.
fn export_format(requested: Option<&str>, as_json: bool) -> Result<nguruvilu::export::Format> {
    if as_json {
        return Ok(nguruvilu::export::Format::Json);
    }
    match requested.map(str::trim) {
        None => Ok(nguruvilu::export::Format::Markdown),
        Some(raw) => match raw.to_ascii_lowercase().as_str() {
            "md" | "markdown" => Ok(nguruvilu::export::Format::Markdown),
            "json" => Ok(nguruvilu::export::Format::Json),
            other => Err(anyhow!(
                "unknown export format '{other}'; expected md, markdown, or json"
            )),
        },
    }
}

// ---------------------------------------------------------------- the route

/// `cli.export` —— 把一条会话写成可分享的文件。
///
/// 参数：`{"id": "...", "out": "...", "format": "...", "json": bool}`
/// 结果：`{"exported": true}`。
///
/// 一条路由的必要性在这里最明显：导出格式是**给别人看的**，所以想换它的人
/// 往往不是改内核的人。一个包想让导出多一节、或者换成别的序列化，覆盖这条
/// 路由即可，内核那份留在链尾。
pub fn register(dispatch: &nguruvilu::dispatch::Dispatch) {
    use nguruvilu::dispatch::{Args, Chain, Fut};
    use std::sync::Arc;

    dispatch.register(
        "cli.export",
        Arc::new(|args: Args, _next: Chain| -> Fut {
            Box::pin(async move {
                let id = args
                    .get("id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| "cli.export needs an id".to_string())?
                    .to_string();
                let out = args.get("out").and_then(|v| v.as_str()).map(PathBuf::from);
                let format = args.get("format").and_then(|v| v.as_str()).map(str::to_string);
                let as_json = args.get("json").and_then(|v| v.as_bool()).unwrap_or(false);

                let root = args
                    .get("home")
                    .and_then(|v| v.as_str())
                    .map(PathBuf::from)
                    .unwrap_or_else(JsonlStore::default_root);
                let store = JsonlStore::open(root).map_err(|e| format!("{e:#}"))?;

                export_command(&store, &id, out.as_ref(), format.as_deref(), as_json)
                    .map_err(|e| format!("{e:#}"))?;

                Ok(serde_json::json!({ "exported": true }))
            })
        }),
    );
}
