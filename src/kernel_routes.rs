//! 内核自己的默认路由：把"内核原本就有的能力"注册进 [`crate::dispatch::Dispatch`]。
//!
//! # 为什么要有这一层
//!
//! 目标是"一切皆调用"——包调内核、内核调包、内核调内核，都走同一条路。
//! 如果内核内部仍然直接 `compaction::plan(...)`，那就是两套接口：
//! 一套对内（函数调用），一套对外（路由）。**两套接口意味着"暴露"是假的**，
//! 因为包永远碰不到那条对内路径上发生的事。
//!
//! 所以内核的默认实现也要进表。包注册同名路由就覆盖它，调 `next(1, args)` 就回退到它。
//!
//! # 这一层不认识任何具体能力
//!
//! 下面注册的名字（`tool.*` / `compact.*` / `state.*` / `window.*`）描述的是**做什么**，
//! 不是**哪个包**。没有 `judge`、没有 `search`、没有 `browser`——加一个新包不需要动这个文件。
//!
//! # 数据形状
//!
//! 进出都是 JSON。`Arc<ToolRegistry>` 这类活对象**不进**路由参数：
//! 它们从 [`crate::catalog::Catalog::get_live`] 按名字取，路由内部用。
//! 这正是"数据进树、活对象进 live"的分工——把注册表压成 JSON 会丢共享语义。

use std::sync::Arc;

use serde_json::{json, Value};

use crate::catalog::Catalog;
use crate::dispatch::{Args, Chain, Dispatch, Fut, Handler};

/// 内核默认路由用到的活对象名。
///
/// 这些名字是**槽位**，不是能力名：`tools` 就是工具表，谁来提供都叫这个名字。
pub mod slot {
    /// 工具表（`Arc<ToolRegistry>`）。
    pub const TOOLS: &str = "tools";
    /// 会话标识，摄入 `tool.execute` 时用。
    pub const IDENTITY: &str = "identity";
}

/// 把默认实现全部注册进 `dispatch`，并把配套的活对象放进 `catalog`。
///
/// 幂等：重复调用只是多叠一层（同一份实现），不会报错。
/// 顺序上每个名字只注册一次默认，包后续注册会插到链头。
pub fn install(dispatch: &Dispatch, catalog: &Arc<Catalog>) {
    register_tool(dispatch);
    register_compact(dispatch);
    register_window(dispatch);
    register_state(dispatch, catalog);
}

/// `tool.execute` —— 调一个工具，参数和结果都是 JSON。
///
/// 参数：`{"name": "...", "arguments": {...}}`（`arguments` 也可以是 JSON 字符串）。
/// 结果：`{"ok": true, "content": "...", "error": null}` 或 `{"ok": false, ...}`。
///
/// 找不到工具名时**返回错误而不是静默空结果**——工具名打错是最常见的失败，
/// 静默的话模型会以为工具做了事。
fn register_tool(dispatch: &Dispatch) {
    dispatch.register("tool.execute", Arc::new(|args: Args, _next: Chain| -> Fut {
        Box::pin(async move { tool_execute(args).await })
    }));
    dispatch.register("tool.names", Arc::new(|_a: Args, _n: Chain| -> Fut {
        Box::pin(async { Err("no tools registered: tool.names needs a registry in the catalog".into()) })
    }));
}

/// `tool.execute` 的实际实现。
///
/// 单独成一个 async fn 而不是写在闭包里，是为了让它能被测试直接调到——
/// 闭包里的逻辑只能通过 `dispatch.call` 间接测，出错时分不清是路由错了还是工具错了。
async fn tool_execute(args: Value) -> Result<Value, String> {
    let name = args
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| "tool.execute needs a 'name'".to_string())?;

    // 工具表是活对象，从这个进程级的单例表里取。
    // 注意：这里不持有注册表的强引用超过一次调用——注册表可能在调用期间被替换。
    let tools = crate::kernel_routes::registry()
        .ok_or_else(|| "no tool registry installed".to_string())?;

    let arguments = match args.get("arguments") {
        Some(Value::String(s)) => s.clone(),
        Some(v) => v.to_string(),
        None => "{}".to_string(),
    };

    match tools.execute(name, &arguments).await {
        // 工具自己说"我失败了"是 `Err`，不是结果里的一个标志位 ——
        // 所以成功分支里没有 `is_error` 可读。
        Ok(out) => Ok(json!({
            "ok": true,
            "text": out.text,
            "images": out.images.len(),
        })),
        Err(e) => Err(format!("{e}")),
    }
}

/// `compact.plan` —— 算出要压缩多少、留多少。
///
/// 参数：`{"messages": [...], "keep_recent": N}`。
/// 结果：`{"summarize": N, "keep": M}` 或 `null`（不值得压缩）。
///
/// 返回 `null` 而不是错误：压缩不必要是正常结果，不是失败。
fn register_compact(dispatch: &Dispatch) {
    dispatch.register("compact.plan", Arc::new(|args: Args, _next: Chain| -> Fut {
        Box::pin(async move {
            let messages: Vec<crate::message::Message> = args
                .get("messages")
                .cloned()
                .map(serde_json::from_value)
                .transpose()
                .map_err(|e| format!("compact.plan got unreadable messages: {e}"))?
                .unwrap_or_default();

            let keep = args
                .get("keep_recent")
                .and_then(Value::as_u64)
                .unwrap_or(1) as usize;

            Ok(match crate::compaction::plan(&messages, keep) {
                Some(p) => json!({ "summarize": p.summarize, "keep": p.keep }),
                None => Value::Null,
            })
        })
    }));
}

/// `window.estimate` —— 估一段对话占多少 token。
///
/// 参数：`{"messages": [...]}`。结果：`{"tokens": N}`。
fn register_window(dispatch: &Dispatch) {
    dispatch.register("window.estimate", Arc::new(|args: Args, _next: Chain| -> Fut {
        Box::pin(async move {
            let messages: Vec<crate::message::Message> = args
                .get("messages")
                .cloned()
                .map(serde_json::from_value)
                .transpose()
                .map_err(|e| format!("window.estimate got unreadable messages: {e}"))?
                .unwrap_or_default();
            Ok(json!({ "tokens": crate::window::estimate_history(&messages) }))
        })
    }));
}

/// `state.read` / `state.write` / `state.keys` / `state.live` —— 状态的全部出入口。
///
/// 参数形状直接对应 [`Catalog`] 的方法。写回带上版本号，供乐观并发用。
fn register_state(dispatch: &Dispatch, catalog: &Arc<Catalog>) {
    let c = Arc::clone(catalog);
    dispatch.register("state.read", Arc::new(move |args: Args, _n: Chain| -> Fut {
        let c = c.clone();
        Box::pin(async move {
            let path = args.get("path").and_then(Value::as_str).unwrap_or("");
            Ok(c.read(path))
        })
    }));

    let c = Arc::clone(catalog);
    dispatch.register("state.write", Arc::new(move |args: Args, _n: Chain| -> Fut {
        let c = c.clone();
        Box::pin(async move {
            let path = args
                .get("path")
                .and_then(Value::as_str)
                .ok_or_else(|| "state.write needs a 'path'".to_string())?;
            let value = args.get("value").cloned().unwrap_or(Value::Null);
            let version = c.write(path, value);
            Ok(json!({ "version": version }))
        })
    }));

    let c = Arc::clone(catalog);
    dispatch.register("state.keys", Arc::new(move |args: Args, _n: Chain| -> Fut {
        let c = c.clone();
        Box::pin(async move {
            let path = args.get("path").and_then(Value::as_str).unwrap_or("");
            Ok(json!(c.keys(path)))
        })
    }));

    let c = Arc::clone(catalog);
    dispatch.register("state.live", Arc::new(move |_a: Args, _n: Chain| -> Fut {
        let c = c.clone();
        Box::pin(async move { Ok(json!({ "names": c.live_names() })) })
    }));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn installed() -> Dispatch {
        let d = Dispatch::new();
        let c = Arc::new(Catalog::new());
        install(&d, &c);
        d
    }

    #[tokio::test]
    async fn the_default_routes_are_all_present() {
        let d = installed();
        for name in [
            "tool.execute",
            "compact.plan",
            "window.estimate",
            "state.read",
            "state.write",
            "state.keys",
            "state.live",
        ] {
            assert!(d.has(name), "missing default route {name}");
            assert_eq!(d.layers(name), 1, "{name} should have exactly one default");
        }
    }

    #[tokio::test]
    async fn a_package_can_wrap_a_default_and_call_next() {
        let d = installed();

        // 一个包：只想看看到底调了什么，其余照旧。
        let seen = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
        let probe = seen.clone();
        d.register(
            "compact.plan",
            Arc::new(move |args: Args, next: Chain| -> Fut {
                let probe = probe.clone();
                Box::pin(async move {
                    probe.lock().unwrap().push("pack".into());
                    next.next(1, args).await
                })
            }),
        );
        assert_eq!(d.layers("compact.plan"), 2);

        // 一条不值得压缩的短对话：默认实现回 null，包放行后拿到 null。
        let out = d
            .call("compact.plan", json!({ "messages": [], "keep_recent": 1 }))
            .await
            .unwrap();
        assert_eq!(out, Value::Null);
        assert_eq!(*seen.lock().unwrap(), vec!["pack".to_string()]);
    }

    #[tokio::test]
    async fn a_package_can_stop_the_chain() {
        let d = installed();
        d.register(
            "compact.plan",
            Arc::new(|_a: Args, _n: Chain| -> Fut {
                Box::pin(async { Ok(json!({ "refused": true })) })
            }),
        );
        let out = d
            .call("compact.plan", json!({ "messages": [], "keep_recent": 1 }))
            .await
            .unwrap();
        assert_eq!(out, json!({ "refused": true }));
    }

    #[tokio::test]
    async fn state_reads_and_writes_go_through_the_route() {
        let d = installed();
        let v = d
            .call("state.write", json!({ "path": "settings.model", "value": "m1" }))
            .await
            .unwrap();
        assert_eq!(v["version"], 1);

        let got = d.call("state.read", json!({ "path": "settings.model" })).await.unwrap();
        assert_eq!(got, json!("m1"));

        let keys = d.call("state.keys", json!({ "path": "settings" })).await.unwrap();
        assert_eq!(keys, json!(["model"]));
    }

    #[tokio::test]
    async fn an_unknown_route_is_named_not_silent() {
        let d = installed();
        let err = d.call("nothing.here", json!({})).await.unwrap_err();
        assert!(err.contains("nothing.here"), "error should name the route: {err}");
    }

    #[tokio::test]
    async fn tool_execute_without_a_name_says_so() {
        let d = installed();
        let err = d.call("tool.execute", json!({})).await.unwrap_err();
        assert!(err.contains("name"), "{err}");
    }

    #[tokio::test]
    async fn compact_plan_parses_messages_and_answers() {
        let d = installed();
        // 三条消息、keep_recent=1：够了就给出一个计划。
        let msgs = json!([
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
            {"role": "assistant", "content": "d"},
        ]);
        let out = d
            .call("compact.plan", json!({ "messages": msgs, "keep_recent": 1 }))
            .await
            .unwrap();
        // 要么给出计划，要么 null —— 但绝不能是 Err。
        assert!(out.is_null() || out.get("keep").is_some(), "unexpected: {out}");
    }

    #[tokio::test]
    async fn window_estimate_counts_something() {
        let d = installed();
        let out = d
            .call(
                "window.estimate",
                json!({ "messages": [{"role": "user", "content": "hello there"}] }),
            )
            .await
            .unwrap();
        assert!(out["tokens"].as_u64().unwrap_or(0) > 0, "{out}");
    }
}

/// 进程级的工具表：`tool.execute` 从这里取。
///
/// 做成进程级单例而不是路由参数，是因为 `Arc<ToolRegistry>` 不能 JSON 化
/// （见模块注释）。内核启动时装一次，`set_registry` 可以换掉它——
/// 换掉之后所有路由调用都用新的表，不需要重新注册路由。
static REGISTRY: std::sync::OnceLock<std::sync::RwLock<Option<Arc<crate::tools::ToolRegistry>>>> =
    std::sync::OnceLock::new();

fn registry_cell() -> &'static std::sync::RwLock<Option<Arc<crate::tools::ToolRegistry>>> {
    REGISTRY.get_or_init(|| std::sync::RwLock::new(None))
}

/// 装上工具表。启动序列调这个一次。
pub fn set_registry(registry: Arc<crate::tools::ToolRegistry>) {
    *registry_cell().write().expect("tool registry cell") = Some(registry);
}

/// 进程级的路由表。
///
/// 和工具表一样是单例：路由在启动时装一次，之后包往里插层。
/// 做成静态而不是沿参数传递，是因为调用点在 `Agent::run` 深处，
/// 为了一个查表操作把 `Arc<Dispatch>` 穿过十几层签名不划算 ——
/// 而路由表本身是全局的，没有"两个内核各有一张表"的语义。
static DISPATCH: std::sync::OnceLock<Arc<Dispatch>> = std::sync::OnceLock::new();

/// 当前的进程级路由表，第一次调用时建好。
pub fn dispatch() -> &'static Arc<Dispatch> {
    DISPATCH.get_or_init(|| {
        let d = Arc::new(Dispatch::new());
        let catalog = Arc::new(Catalog::new());
        // 状态容器也要活到进程结束：`state.*` 路由拿着它的 Arc。
        let _ = CATALOG.set(catalog.clone());
        install(&d, &catalog);
        d
    })
}

/// 进程级的状态容器。`dispatch()` 里建的那一个。
static CATALOG: std::sync::OnceLock<Arc<Catalog>> = std::sync::OnceLock::new();

/// 当前的状态容器。**必须先调过 `dispatch()`**，否则 panic ——
/// 因为两者是一起建的，单独取它说明接线顺序错了。
pub fn catalog() -> &'static Arc<Catalog> {
    let _ = dispatch();
    CATALOG.get().expect("catalog is built with dispatch")
}

/// 启动序列的一行接线：把工具表和路由表都装好。
///
/// 在 `Kernel` 建完之后调一次。幂等 —— 重复调用只是把同一份工具表再装一遍。
pub fn route_everything(kernel: &crate::plugin::Kernel) {
    // 工具表要能被 `tool.execute` 拿到。Kernel 自己的那份是 `&ToolRegistry`，
    // 而路由是在别的线程上跑的，所以拷一份 Arc 出来 —— 注册表内部全是 Arc，
    // 这次拷贝不复制任何 handler。
    let registry = Arc::new(kernel.tools().clone());
    set_registry(registry);
    let _ = dispatch();
}

/// 当前的工具表，没装时 `None`。
pub fn registry() -> Option<Arc<crate::tools::ToolRegistry>> {
    registry_cell().read().expect("tool registry cell").clone()
}
