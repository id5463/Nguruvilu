//! 这一轮用哪个模型：命令行参数 > 包提供的路由 > 设置文件。
//!
//! 一个函数而不是一条路由：它读 `Cli` 和 `Settings`，都是启动时已有的东西，
//! 压成 JSON 参数只会让"换个模型"变成"模拟一遍启动"。

use nguruvilu::hotreload::ModelRoute;
use nguruvilu::plugin::Kernel;
use nguruvilu::settings::Settings;

use super::args::Cli;

/// The route this session runs on, built from the resolved settings.
///
/// Command-line flags win over the stored settings, and a plugin that
/// published a network policy wins over both: it is the more specific choice,
/// made by the pack the user installed for this provider. Everything a client
/// needs is on the route, so the session, a one-off command, and a subagent
/// built later all read the same values.
fn model_route(cli: &Cli, settings: &Settings, kernel: Option<&Kernel>) -> ModelRoute {
    let mut route = ModelRoute::from_settings(settings);
    // The proxy stays explicit: when it is empty, the process's `HTTP_PROXY`
    // and `HTTPS_PROXY` are deliberately ignored. Picking those up silently is
    // how a proxy set for another tool reroutes model traffic and fails with a
    // TLS handshake error that never mentions proxies.
    if let Some(url) = &cli.base_url {
        route.base_url = url.clone();
    }
    if let Some(key) = &cli.api_key {
        route.api_key = key.clone();
    }
    if let Some(name) = &cli.model {
        route.model = name.clone();
    }
    if let Some(effort) = &cli.reasoning_effort {
        route.reasoning_effort = if effort.trim().is_empty() {
            None
        } else {
            Some(effort.clone())
        };
    }
    if let Some(delivery) = &cli.delivery {
        // Checked here rather than passed through as a string: a mistyped mode
        // on the command line should say so before the first request, not after
        // a turn has hung for the whole request timeout.
        let word = delivery.trim();
        if !word.is_empty() && nguruvilu::llm::Delivery::parse(word).is_none() {
            eprintln!(
                "warning: unknown delivery mode {word:?}; expected one of {}. Keeping the configured mode.",
                nguruvilu::llm::Delivery::names().join(", ")
            );
        } else {
            route.delivery = if word.is_empty() { None } else { Some(word.to_string()) };
        }
    }
    if let Some(kernel) = kernel {
        let view = kernel.service_view(nguruvilu::plugin::RealmMap::new());
        if let Some(network) = nguruvilu::network::NetworkHandle::from_view(&view) {
            route.network = network;
        }
    }
    route
}


