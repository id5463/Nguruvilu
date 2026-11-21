//! `ngu config`：看和改存下来的设置。
//!
//! 一个设置项一条路：`config <key>` 读一个，`config set <key> <value>` 写一个，
//! `config path` 说文件在哪。没有 "config dump everything"，因为那份输出会变成
//! 一个别人依赖的格式，而设置项本身还会变。
//!
//! 这一层只读写设置文件与订阅表；它不碰内核状态 —— 改了设置之后要生效，
//! 是下一次加载或下一次回合的事。

use anyhow::{anyhow, Context, Result};

use nguruvilu::settings::Settings;

use super::args::*;

/// Show or change the stored settings.
fn config_command(action: Option<&ConfigAction>, as_json: bool) -> Result<()> {
    match action {
        Some(ConfigAction::Path) => {
            println!("{}", Settings::path().display());
            Ok(())
        }
        Some(ConfigAction::Set {
            base_url,
            api_key,
            model,
            reasoning_effort,
            delivery,
            proxy,
            context_window,
            compact_percent,
            compact_keep_recent,
            max_steps,
            max_output_tokens,
            tool_timeout,
            mcp_timeout,
            subagent_steps,
            subagent_wall,
            subagent_depth,
            job_wait,
            network,
            request_timeout,
            pool_idle_timeout,
            retry_attempts,
            retry_backoff_ms,
            judge_endpoint,
            judge_key,
            judge_model,
        }) => {
            let mut settings = Settings::load()?;
            if let Some(value) = base_url {
                settings.base_url = value.trim().to_string();
            }
            if let Some(value) = api_key {
                settings.api_key = value.trim().to_string();
            }
            if let Some(value) = model {
                settings.model = value.trim().to_string();
            }
            if let Some(value) = reasoning_effort {
                settings.reasoning_effort = value.trim().to_string();
            }
            if let Some(value) = delivery {
                // Refused at the point of writing: a mode the client cannot
                // parse would otherwise sit in the file until a turn hung on it.
                let word = value.trim();
                if word.is_empty() {
                    settings.delivery = String::new();
                } else {
                    let parsed = nguruvilu::llm::Delivery::parse(word).ok_or_else(|| {
                        anyhow!(
                            "unknown delivery mode '{word}'; expected one of {}",
                            nguruvilu::llm::Delivery::names().join(", ")
                        )
                    })?;
                    // Stored as the canonical word, so what `config show` prints
                    // is what the client reads back.
                    settings.delivery = parsed.as_str().to_string();
                }
            }
            if let Some(value) = proxy {
                settings.proxy = value.trim().to_string();
            }
            if let Some(value) = context_window {
                // Accepts `128K`, `1M`, or a plain count. Zero means "go back
                // to working it out".
                let tokens = nguruvilu::size::parse_size(value)?;
                settings.context_window = if tokens == 0 { None } else { Some(tokens) };
            }
            if let Some(value) = compact_percent {
                settings.compact_percent = (*value).min(100);
            }
            if let Some(value) = compact_keep_recent {
                settings.compact_keep_recent = (*value).max(1);
            }
            if let Some(value) = max_steps {
                // Stored as written: the clamping belongs to `Settings::max_steps`,
                // which is where every reader goes, so a value that can never be
                // honoured is reported once rather than silently rewritten here.
                settings.max_steps = *value;
            }
            if let Some(value) = max_output_tokens {
                let tokens = nguruvilu::size::parse_size(value)?;
                settings.max_output_tokens = if tokens == 0 { None } else { Some(tokens) };
            }
            // The budgets. Stored as written and clamped on the way out (in
            // `Settings::budgets`), so `config show` reports what was set while
            // no reader can ever see a zero-length deadline.
            if let Some(value) = tool_timeout {
                settings.tool_timeout_secs = *value;
            }
            if let Some(value) = mcp_timeout {
                settings.mcp_timeout_secs = *value;
            }
            if let Some(value) = subagent_steps {
                settings.subagent_max_steps = *value;
            }
            if let Some(value) = subagent_wall {
                settings.subagent_wall_secs = *value;
            }
            if let Some(value) = subagent_depth {
                settings.subagent_max_depth = *value;
            }
            if let Some(value) = job_wait {
                settings.job_wait_secs = *value;
            }
            if let Some(name) = network {
                settings.network = nguruvilu::network::NetworkSettings::preset(name)
                    .ok_or_else(|| {
                        anyhow!(
                            "unknown network preset '{name}'; expected one of {}",
                            nguruvilu::network::NetworkSettings::presets().join(", ")
                        )
                    })?;
            }
            if let Some(value) = request_timeout {
                settings.network.request_timeout_secs = *value;
            }
            if let Some(value) = pool_idle_timeout {
                settings.network.pool_idle_timeout_secs = *value;
            }
            if let Some(value) = retry_attempts {
                settings.network.retry_attempts = *value;
            }
            if let Some(value) = retry_backoff_ms {
                settings.network.retry_backoff_ms = *value;
            }
            // A value that would behave unlike its name is reported here rather
            // than at the first failed request.
            //
            // The judge: one of three flags means "touch it",
            // the key alone decides whether the tool exists, and an endpoint
            // must be a URL because a typo here becomes a failed call later.
            if judge_endpoint.is_some() || judge_key.is_some() || judge_model.is_some() {
                let mut judge = nguruvilu::tools::judge::from_settings(&settings)
                    .unwrap_or_else(nguruvilu::tools::judge::JudgeSettings::default);
                if let Some(endpoint) = judge_endpoint {
                    let endpoint = endpoint.trim().trim_end_matches('/').to_string();
                    if !(endpoint.starts_with("http://") || endpoint.starts_with("https://")) {
                        anyhow::bail!("judge endpoint must be an http(s) URL, got '{endpoint}'");
                    }
                    judge.endpoint = endpoint;
                }
                if let Some(key) = judge_key {
                    judge.api_key = key.clone();
                }
                if let Some(model) = judge_model {
                    judge.model = model.trim().to_string();
                }
                nguruvilu::tools::judge::into_settings(&mut settings, Some(judge))?;
            }

            let network_problems = settings.network.problems();
            if !network_problems.is_empty() {
                anyhow::bail!("{}", network_problems.join("; "));
            }
            settings.save()?;

            if as_json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({
                        "ok": true,
                        "path": Settings::path().display().to_string(),
                        "base_url": settings.base_url,
                        "api_key": settings.masked_key(),
                        "model": settings.model_or_default(),
                        "configured": settings.is_configured(),
                    }))?
                );
            } else {
                println!("saved to {}", Settings::path().display());
                println!("  endpoint: {}", if settings.base_url.is_empty() { "(not set)" } else { &settings.base_url });
                println!("  key:      {}", if settings.api_key.is_empty() { "(not set)" } else { "(stored)" });
                println!("  model:    {}", settings.model_or_default());
                println!(
                    "  delivery: {}",
                    nguruvilu::llm::Delivery::parse(&settings.delivery).unwrap_or_default().as_str()
                );
                println!(
                    "  thinking: {}",
                    if settings.reasoning_effort.trim().is_empty() {
                        "(provider default)"
                    } else {
                        &settings.reasoning_effort
                    }
                );
                if let Some(judge) = nguruvilu::tools::judge::from_settings(&settings) {
                    println!(
                        "  judge:    {} ({})",
                        judge.endpoint,
                        if judge.is_configured() {
                            "key set — judge tool on"
                        } else {
                            "no key — judge tool off"
                        }
                    );
                }
                if !settings.is_configured() {
                    println!("\nstill missing: {}", settings.missing().join(", "));
                }
            }
            Ok(())
        }
        Some(ConfigAction::Routes) => {
            let routes = nguruvilu::routes::Routes::load();
            if as_json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({
                        "path": nguruvilu::routes::Routes::path().display().to_string(),
                        "active": routes.active,
                        "profiles": routes.profiles.iter().map(|p| serde_json::json!({
                            "name": p.name, "base_url": p.base_url, "model": p.model,
                            "has_key": !p.api_key.trim().is_empty(),
                        })).collect::<Vec<_>>(),
                    }))?
                );
                return Ok(());
            }
            if routes.profiles.is_empty() {
                println!(
                    "no saved profiles. `ngu config save-route <name>` stores the current settings as one."
                );
                return Ok(());
            }
            for profile in &routes.profiles {
                let mark = if routes.active == profile.name { "*" } else { " " };
                println!(
                    "{mark} {:<18} {:<28} {}",
                    profile.name, profile.model, profile.base_url
                );
            }
            Ok(())
        }
        Some(ConfigAction::SaveRoute { name }) => {
            let name = name.trim();
            anyhow::ensure!(!name.is_empty(), "save-route needs a profile name");
            let mut routes = nguruvilu::routes::Routes::load();
            let stored = Settings::load()?;
            routes.upsert(nguruvilu::routes::Routes::capture(&stored, name));
            // The stored settings are exactly this profile, so it is the active
            // line until something else is switched to.
            routes.active = name.to_string();
            routes.save()?;
            println!(
                "saved profile '{name}' → {} [{}]",
                stored.base_url,
                stored.model_or_default()
            );
            Ok(())
        }
        Some(ConfigAction::Use { name }) => {
            let mut routes = nguruvilu::routes::Routes::load();
            let mut settings = Settings::load()?;
            routes.apply(name, &mut settings)?;
            routes.active = name.trim().to_string();
            routes.save()?;
            settings.save()?;
            println!(
                "settings now point at profile '{}' → {} [{}]",
                name, settings.base_url, settings.model
            );
            Ok(())
        }
        None | Some(ConfigAction::Show) => {
            // `resolve` is what a request would actually use, so this is the
            // honest answer to "what is configured right now".
            let effective = Settings::resolve();
            let stored = Settings::load()?;

            if as_json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({
                        "path": Settings::path().display().to_string(),
                        "base_url": effective.base_url,
                        "api_key": effective.masked_key(),
                        "model": effective.model_or_default(),
                        "configured": effective.is_configured(),
                        "missing": effective.missing(),
                        "stored_base_url": stored.base_url,
                        "stored_model": stored.model,
                    }))?
                );
                return Ok(());
            }

            println!("settings file: {}", Settings::path().display());
            println!(
                "endpoint:      {}",
                if effective.base_url.is_empty() { "(not set)" } else { &effective.base_url }
            );
            println!(
                "api key:       {}",
                if effective.api_key.is_empty() { "(not set)".to_string() } else { effective.masked_key() }
            );
            println!("model:         {}", effective.model_or_default());
            println!(
                "thinking:      {}",
                if effective.reasoning_effort.trim().is_empty() {
                    "(provider default)"
                } else {
                    &effective.reasoning_effort
                }
            );
            println!(
                "proxy:         {}",
                if effective.proxy.trim().is_empty() {
                    "(direct; HTTP_PROXY/HTTPS_PROXY ignored)"
                } else {
                    &effective.proxy
                }
            );
            // Read back as the mode a request will actually use: an empty
            // setting is the default rather than a third state, and the auto
            // fallback is worth naming because it changes what a stall costs.
            let mode = nguruvilu::llm::Delivery::parse(&effective.delivery).unwrap_or_default();
            println!(
                "delivery:      {}{}",
                mode.as_str(),
                if effective.degrade_on_stall == Some(false) {
                    ""
                } else {
                    "  (a stalled stream switches this route to buffered)"
                }
            );
            let window = effective
                .context_policy(None)
                .resolve(&effective.model_or_default());
            println!(
                "context:       {} tokens ({}){}",
                nguruvilu::size::format_size(window.tokens),
                window.source.as_str(),
                if window.source.is_guess() {
                    "  <- a guess; set --context-window if you know better"
                } else {
                    ""
                }
            );
            println!(
                "compaction:    at {}% of the window, keeping {} recent messages",
                effective.compact_percent, effective.compact_keep_recent
            );
            // The second lever on context size, and the one that does not wait
            // for the window: what an old tool result or screenshot still
            // travels in the request. Full copies live on disk either way.
            // Read through `policy()` — the effective numbers, not the stored
            // ones: a count left at 0 means "not written", and printing the raw
            // file would show a machine thinning everything when it keeps eight.
            let tool_output = effective.tool_output.policy();
            println!(
                "tool output:   {}",
                if tool_output.enabled {
                    format!(
                        "older results keep {} head + {} tail chars (newest {} exact), newest {} image(s) inline; originals in {}",
                        tool_output.head_chars,
                        tool_output.tail_chars,
                        tool_output.keep_recent,
                        tool_output.keep_images,
                        nguruvilu::tools::spill::root_dir().display()
                    )
                } else {
                    format!(
                        "kept whole — thinning is off (copies still land in {})",
                        nguruvilu::tools::spill::root_dir().display()
                    )
                }
            );
            println!(
                "steps:         at most {} model step(s) per batch; the cap is asked about {} more time(s), each granting another batch",
                effective.max_steps(),
                nguruvilu::agent::CONTINUATION_BATCHES
            );
            println!(
                "budgets:       bash {}s, MCP {}s, subagent {} steps / {}s / depth {}, job wait {}s",
                effective.budgets().tool_deadline_ms / 1000,
                effective.budgets().mcp_timeout_ms / 1000,
                effective.budgets().subagent_max_steps,
                effective.budgets().subagent_wall_ms / 1000,
                effective.budgets().subagent_max_depth,
                effective.budgets().job_wait_max_ms / 1000,
            );
            println!(
                "network:       request {}s, pool idle {}s, {} retry(ies) from {}ms",
                effective.network.request_timeout_secs,
                effective.network.pool_idle_timeout_secs,
                effective.network.retry_attempts,
                effective.network.retry_backoff_ms
            );
            println!(
                "max output:    {}",
                match effective.max_output_tokens {
                    Some(tokens) => format!("{} tokens", nguruvilu::size::format_size(tokens)),
                    None => "(provider default)".to_string(),
                }
            );
            if !effective.is_configured() {
                println!("\nnot configured — set it with:");
                println!("  ngu config set --base-url https://your-host/v1 --api-key sk-...");
            } else if effective.base_url != stored.base_url || effective.api_key != stored.api_key {
                println!("\n(an environment variable is overriding the stored file)");
            }
            Ok(())
        }
    }
}

// ---------------------------------------------------------------- the route

/// `cli.config` —— 看和改设置，以及设置里那几项路由。
///
/// 参数：`{"action": "...", "json": bool}`，`action` 是 `path` / `routes` / `show`。
/// 结果：`{"ok": true}`。
///
/// # 为什么只有三个动作进路由
///
/// `ConfigAction` 有九个变体，但 `set` 的每个字段是一个独立的 flag（`--base-url`、
/// `--api-key`…），把它压成 JSON 只是换一种拼 CLI 的方式，没有换来可替换性。
/// 而 `path`/`routes`/`show` 三个动作读的是**同一个问题**：设置从哪来、里面有什么。
/// 这三件事被人替换才有意义（换配置源、换路由表、换展示），所以它们进路由，
/// 其余的留在命令行里。
///
/// 认不出的动作调 `next`：链上其余实现可能认别的名字，回退是这条链的用法。
pub fn register(dispatch: &nguruvilu::dispatch::Dispatch) {
    use nguruvilu::dispatch::{Args, Chain, Fut};
    use std::sync::Arc;

    dispatch.register(
        "cli.config",
        Arc::new(|args: Args, _next: Chain| -> Fut {
            Box::pin(async move {
                let as_json = args.get("json").and_then(|v| v.as_bool()).unwrap_or(false);
                let action = match args.get("action").and_then(|v| v.as_str()).unwrap_or("show") {
                    "path" => ConfigAction::Path,
                    "routes" => ConfigAction::Routes,
                    "show" => ConfigAction::Show,
                    other => return Err(format!("cli.config cannot do '{other}'")),
                };

                config_command(Some(&action), as_json).map_err(|e| format!("{e:#}"))?;
                Ok(serde_json::json!({ "ok": true }))
            })
        }),
    );
}

/// 一个非空的 API key，或者一条说清缺什么的错误。
///
/// 独立的函数因为它被两处用：启动时（要不要发请求）和 `config`（看现在配了什么）。
/// 空 key 的错误信息点名了两个环境变量，因为那是用户唯一能立刻动手改的地方。
pub(crate) fn require_key(api_key: &str) -> anyhow::Result<()> {
    if api_key.trim().is_empty() {
        return Err(anyhow::anyhow!(
            "no API key: pass --api-key or set NGU_API_KEY (or OPENAI_API_KEY)"
        ));
    }
    Ok(())
}
