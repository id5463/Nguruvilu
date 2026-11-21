//! `ngu stats`：一条会话花了多少，从它记录的回合里读出来。
//!
//! 没有用量行的会话（账本存在之前写下的那些）汇总成零而不是报错；id 对应
//! 的文件不在了也一样 ——「什么都没记」是关于文件的事实，不是读取失败。
//!
//! 这一层只读：它把存下来的数字加起来、乘上用户给的费率、打印。它不写任何
//! 统计，写入发生在 `run()` 里（`record_turn_stats`）。

use anyhow::Result;

use nguruvilu::pricing::{self, Cost};
use nguruvilu::session::{JsonlStore, SessionStore, SessionTotals, TurnStats};

use super::args::*;

/// the same struct, so a script reading either frontend reads the same shape.
/// `cache_hit_ratio` is a fraction (0..=1), so no consumer repeats the division.
fn stats_totals_json(totals: &SessionTotals) -> serde_json::Value {
    serde_json::json!({
        "turns": totals.turns,
        "steps": totals.steps,
        "tool_calls": totals.tool_calls,
        "input": totals.input,
        "output": totals.output,
        "cached": totals.cached,
        "cache_created": totals.cache_created,
        "reasoning": totals.reasoning,
        "model_ms": totals.model_ms,
        "tools_ms": totals.tools_ms,
        "cache_hit_ratio": totals.cache_hit_ratio(),
    })
}

/// One recorded turn, as the JSON payload's per-turn entries.
fn stats_turn_json(stats: &TurnStats) -> serde_json::Value {
    serde_json::json!({
        "ts": stats.ts,
        "steps": stats.steps,
        "tool_calls": stats.tool_calls,
        "retries": stats.retries,
        "input": stats.input,
        "output": stats.output,
        "cached": stats.cached,
        "cache_created": stats.cache_created,
        "reasoning": stats.reasoning,
        "model_ms": stats.model_ms,
        "tools_ms": stats.tools_ms,
    })
}

/// How many turns `ngu stats` shows per session.
///
/// A screenful for tracing where a spike came from, not a ledger: the totals
/// cover the whole session.
const STATS_TAIL: usize = 20;

fn print_stats_block(id: &str, totals: &SessionTotals, cost: Option<&nguruvilu::pricing::Cost>, found: bool) {
    println!("session {id}{}", if found { "" } else { " (not found)" });
    if totals.turns == 0 {
        println!("  no turns recorded");
        return;
    }
    println!(
        "  turns {}  steps {}  tool calls {}",
        totals.turns, totals.steps, totals.tool_calls
    );
    println!(
        "  tokens  in {}  out {}  cached {}  written {}  hit {:.1}%  reasoning {}",
        totals.input,
        totals.output,
        totals.cached,
        totals.cache_created,
        totals.cache_hit_ratio() * 100.0,
        totals.reasoning
    );
    println!(
        "  time    model {} ms  tools {} ms",
        totals.model_ms, totals.tools_ms
    );
    // Money is printed as a fact or as a refusal, never as a zero: "unpriced"
    // and "cost nothing" read the same on a receipt, and only one is true.
    if let Some(cost) = cost {
        match (&cost.reason, cost.priced) {
            (_, true) => println!("  cost    {:.4} ({})", cost.amount, cost.model),
            (Some(reason), false) => println!("  cost    unpriced: {reason}"),
            (None, false) => println!("  cost    unpriced"),
        }
    }
}

/// `ngu stats` — what stored sessions cost, read from their recorded turns.
///
/// A session with no usage rows — every session written before the ledger
/// existed — totals to zeros rather than erroring, and so does an id whose file
/// is gone: "nothing recorded" is a fact about the file, not a failed read.
#[allow(clippy::too_many_arguments)]
fn stats_command(
    store: &JsonlStore,
    id: Option<&str>,
    all: bool,
    priced: bool,
    prices: Option<&std::path::Path>,
    as_json: bool,
) -> Result<()> {
    // The table is only read when money was asked for: a report without `--cost`
    // must not fail because a price file somewhere has a typo in it.
    let table = if priced {
        match prices {
            Some(path) => nguruvilu::pricing::PriceTable::from_path(path)?,
            None => {
                nguruvilu::pricing::PriceTable::from_path(&nguruvilu::pricing::PriceTable::default_path())?
            }
        }
    } else {
        None
    };
    let cost_of = |model: Option<&str>, totals: &SessionTotals| -> Option<nguruvilu::pricing::Cost> {
        if !priced {
            return None;
        }
        // A session that records no model has nothing to price against. Saying
        // so is better than pricing it at the wildcard rate as if it were known.
        let model = model.unwrap_or("");
        Some(if model.is_empty() {
            nguruvilu::pricing::Cost {
                amount: 0.0,
                model: String::new(),
                priced: false,
                reason: Some("the session records no model".to_string()),
                fresh_input: 0,
                cached: 0,
                cache_created: 0,
                output: 0,
            }
        } else {
            nguruvilu::pricing::cost(
                table.as_ref(),
                model,
                nguruvilu::pricing::Tokens::of_totals(totals),
            )
        })
    };

    // Which sessions to report, in the listing's recency order.
    let ids: Vec<String> = if all {
        store.list()?.into_iter().map(|summary| summary.id).collect()
    } else if let Some(id) = id {
        vec![id.to_string()]
    } else {
        store.latest()?.map(|summary| summary.id).into_iter().collect()
    };

    if ids.is_empty() {
        if as_json {
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "sessions": [],
                    "total": stats_totals_json(&SessionTotals::default()),
                }))?
            );
        } else {
            println!("no sessions in {}", store.root().display());
        }
        return Ok(());
    }

    // Every session is loaded up front so the report can borrow them: the
    // per-turn tail reads the same in-memory rows the totals were summed from.
    // A missing file is a zero row, not an error — a typo and a deleted session
    // should both answer "nothing recorded".
    let mut loaded: Vec<(String, Option<Session>)> = Vec::with_capacity(ids.len());
    for id in ids {
        let session = store.load(&id)?;
        loaded.push((id, session));
    }

    let total: SessionTotals = loaded
        .iter()
        .map(|(_, session)| session.as_ref().map(totals_of).unwrap_or_default())
        .sum();
    // The grand total's money is the sum of the sessions' money, not the money
    // of the summed tokens: sessions can run on different models at different
    // rates, and one rate applied to a mixed total would invent a number.
    let total_cost = if priced {
        Some(nguruvilu::pricing::total(loaded.iter().filter_map(
            |(_, session)| {
                let totals = session.as_ref().map(totals_of).unwrap_or_default();
                cost_of(session.as_ref().and_then(|s| s.model.as_deref()), &totals)
            },
        )))
    } else {
        None
    };

    if as_json {
        let entries: Vec<serde_json::Value> = loaded
            .iter()
            .map(|(id, session)| {
                let totals = session.as_ref().map(totals_of).unwrap_or_default();
                let mut entry = serde_json::json!({
                    "session": id,
                    "found": session.is_some(),
                    "totals": stats_totals_json(&totals),
                });
                if let Some(cost) = cost_of(session.as_ref().and_then(|s| s.model.as_deref()), &totals) {
                    entry["cost"] = cost.json();
                }
                // The per-turn tail is for tracing one session; an `--all`
                // listing would only bury the totals in a wall of rows.
                if !all {
                    if let Some(session) = session {
                        let tail: Vec<serde_json::Value> = session
                            .stats
                            .iter()
                            .rev()
                            .take(STATS_TAIL)
                            .rev()
                            .map(stats_turn_json)
                            .collect();
                        entry["turns"] = serde_json::Value::Array(tail);
                    }
                }
                entry
            })
            .collect();
        let mut payload = serde_json::json!({
            "sessions": entries,
            "total": stats_totals_json(&total),
        });
        if let Some(cost) = &total_cost {
            payload["total_cost"] = cost.json();
        }
        println!("{}", serde_json::to_string_pretty(&payload)?);
        return Ok(());
    }

    // A grand total is what `--all` asked for, so it is printed even when one
    // session is all there is: a report whose last line appears sometimes is a
    // report a script cannot read.
    let show_total = all || loaded.len() > 1;
    for (id, session) in &loaded {
        let totals = session.as_ref().map(totals_of).unwrap_or_default();
        let cost = cost_of(session.as_ref().and_then(|s| s.model.as_deref()), &totals);
        print_stats_block(id, &totals, cost.as_ref(), session.is_some());
        if show_total {
            println!();
        }
    }
    if show_total {
        print_stats_block("total", &total, total_cost.as_ref(), true);
    }
    Ok(())
}

// ---------------------------------------------------------------- the route

/// `cli.stats` —— 打印一条会话的用量，或者全部会话加一个总计。
///
/// 参数：`{"id": "...", "all": bool, "cost": bool, "prices": "...", "json": bool}`
/// 结果：`{"printed": true}`。
///
/// # 为什么这是一条路由
///
/// `ngu stats` 打印什么，是这一个功能说了算，不是命令行的其余部分说了算。
/// 注册成路由之后，一个包可以覆盖它：想在统计里加一列、换个表头、把小费
/// 算进去，注册同名路由即可，内核的那份留在链尾当回退。
///
/// 所以这里注册的是**内核的默认实现**；`next` 不会被调用，因为它是链尾。
pub fn register(dispatch: &nguruvilu::dispatch::Dispatch) {
    use nguruvilu::dispatch::{Args, Chain, Fut};
    use std::sync::Arc;

    dispatch.register(
        "cli.stats",
        Arc::new(|args: Args, _next: Chain| -> Fut {
            Box::pin(async move {
                let id = args.get("id").and_then(|v| v.as_str()).map(str::to_string);
                let all = args.get("all").and_then(|v| v.as_bool()).unwrap_or(false);
                let show_cost = args.get("cost").and_then(|v| v.as_bool()).unwrap_or(false);
                let prices = args
                    .get("prices")
                    .and_then(|v| v.as_str())
                    .map(std::path::PathBuf::from);
                let as_json = args.get("json").and_then(|v| v.as_bool()).unwrap_or(false);

                // 路由不带 store：它从参数里的 `home` 自己开一个，或者用默认根。
                // 这样包覆盖这条路由时，不必知道内核把会话存在哪。
                let root = args
                    .get("home")
                    .and_then(|v| v.as_str())
                    .map(std::path::PathBuf::from)
                    .unwrap_or_else(JsonlStore::default_root);
                let store = JsonlStore::open(root).map_err(|e| format!("{e:#}"))?;

                stats_command(&store, id.as_deref(), all, show_cost, prices.as_deref(), as_json)
                    .map_err(|e| format!("{e:#}"))?;

                Ok(serde_json::json!({ "printed": true }))
            })
        }),
    );
}
