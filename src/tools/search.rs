//! Web search as a tool.
//!
//! Search is not one API. The three that publish for agents disagree about the
//! HTTP method, the auth header, where the results live in the response, and
//! what the snippet field is called:
//!
//! | | Tavily | Brave | Exa |
//! |---|---|---|---|
//! | method | `POST /search` | `GET /res/v1/web/search?q=` | `POST /search` |
//! | auth | `api_key` in the body | `X-Subscription-Token` | `Authorization: Bearer` |
//! | results at | `results[]` | `web.results[]` | `results[]` |
//! | snippet | `content` | `description` | `text` |
//!
//! So "point the endpoint somewhere else" is not a thing that works. The
//! differences are data — every dialect is a request shape and a set of JSON
//! paths — so they are a table rather than three types with their own state.
//!
//! # Why the tool is absent until it is configured
//!
//! A tool that appears and then fails on every call is worse than one that is
//! not there: a model that sees `web_search` will use it, retry when it fails,
//! and spend the turn on something it cannot do. So the tool is registered only
//! when a provider and a key are both present.

use std::collections::BTreeMap;

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::tools::{ToolDef, ToolFuture, ToolOutput};

/// A search provider's request shape and response layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Dialect {
    /// Aggregated results, cleaned for models.
    Tavily,
    /// An independent index; keyword search.
    Brave,
    /// A neural index; `find pages like this`.
    Exa,
}

impl Dialect {
    /// Every dialect, for diagnostics.
    pub fn all() -> &'static [Dialect] {
        &[Dialect::Tavily, Dialect::Brave, Dialect::Exa]
    }

    /// The name a configuration file writes.
    pub fn as_str(&self) -> &'static str {
        match self {
            Dialect::Tavily => "tavily",
            Dialect::Brave => "brave",
            Dialect::Exa => "exa",
        }
    }

    /// The endpoint when none is configured.
    pub fn default_endpoint(&self) -> &'static str {
        match self {
            Dialect::Tavily => "https://api.tavily.com/search",
            Dialect::Brave => "https://api.search.brave.com/res/v1/web/search",
            Dialect::Exa => "https://api.exa.ai/search",
        }
    }

    /// Parse a configured name.
    pub fn parse(name: &str) -> Option<Dialect> {
        Dialect::all()
            .iter()
            .copied()
            .find(|d| d.as_str().eq_ignore_ascii_case(name.trim()))
    }

    /// Where the results array lives in the response.
    fn results_path(&self) -> &'static [&'static str] {
        match self {
            // Tavily's envelope is flat.
            Dialect::Tavily | Dialect::Exa => &["results"],
            // Brave nests web results under `web`.
            Dialect::Brave => &["web", "results"],
        }
    }
}

/// One result, normalised across dialects.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Hit {
    /// Page title.
    pub title: String,
    /// Page URL.
    pub url: String,
    /// The excerpt the provider chose.
    pub snippet: String,
}

/// How to reach a search provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SearchSettings {
    /// Which provider.
    pub provider: Dialect,
    /// The API key.
    #[serde(default)]
    pub api_key: String,
    /// Override the provider's endpoint.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub endpoint: Option<String>,
    /// How many results to ask for.
    #[serde(default = "default_limit")]
    pub max_results: usize,
}

fn default_limit() -> usize {
    5
}

impl SearchSettings {
    /// Whether this is usable: a provider and a key.
    ///
    /// The key is what makes it usable. A provider without one would register a
    /// tool that fails on every call.
    pub fn is_configured(&self) -> bool {
        !self.api_key.trim().is_empty()
    }

    /// The endpoint to call.
    pub fn url(&self) -> String {
        self.endpoint
            .clone()
            .unwrap_or_else(|| self.provider.default_endpoint().to_string())
    }
}

impl Default for SearchSettings {
    /// A starting point for a merge, before a pack or the user names anything.
    ///
    /// Tavily is first only because it is the first dialect listed, not because
    /// it is preferred: any file that says otherwise overwrites this, and with
    /// no key the tool does not register at all, so a wrong guess here cannot
    /// reach the network.
    fn default() -> Self {
        Self {
            provider: Dialect::Tavily,
            api_key: String::new(),
            endpoint: None,
            max_results: default_limit(),
        }
    }
}

/// Build the request for a query.
///
/// Returns the method, the URL, the headers, and the body. Public so a test can
/// check the shape without a network.
pub fn build_request(
    settings: &SearchSettings,
    query: &str,
    http: &reqwest::Client,
) -> (reqwest::RequestBuilder, bool) {
    let limit = settings.max_results.max(1);
    let key = settings.api_key.trim();

    match settings.provider {
        Dialect::Tavily => (
            http.post(settings.url())
                .bearer_auth(key)
                .json(&json!({
                    "query": query,
                    "max_results": limit,
                    // The raw page text is what a model can use; without this
                    // Tavily returns short snippets only.
                    "include_raw_content": false,
                })),
            true,
        ),
        Dialect::Brave => (
            http.get(settings.url())
                .header("X-Subscription-Token", key)
                .header("Accept", "application/json")
                .query(&[("q", query), ("count", &limit.to_string())]),
            false,
        ),
        Dialect::Exa => (
            http.post(settings.url())
                .header("Authorization", format!("Bearer {key}"))
                .json(&json!({
                    "query": query,
                    "numResults": limit,
                    "contents": { "text": { "maxCharacters": 1200 } },
                })),
            true,
        ),
    }
}

/// Read the results out of a provider's response.
pub fn parse_response(dialect: Dialect, body: &Value) -> Vec<Hit> {
    let mut entries = body;
    for key in dialect.results_path() {
        match entries.get(*key) {
            Some(next) => entries = next,
            None => return Vec::new(),
        }
    }
    let Some(array) = entries.as_array() else {
        return Vec::new();
    };

    array
        .iter()
        .filter_map(|entry| {
            let url = entry.get("url").and_then(|u| u.as_str())?;
            let title = entry
                .get("title")
                .and_then(|t| t.as_str())
                .unwrap_or(url)
                .to_string();
            // Each provider names the excerpt differently, and a missing one is
            // not a reason to drop the result: the URL alone is useful.
            let snippet = ["content", "description", "text", "snippet"]
                .iter()
                .find_map(|key| entry.get(*key).and_then(|s| s.as_str()))
                .unwrap_or_default()
                .to_string();
            Some(Hit {
                title,
                url: url.to_string(),
                snippet,
            })
        })
        .collect()
}

/// The tool name.
///
/// **Not `web_search`.** That name is reserved on the provider this was tested
/// against: a *function* tool called `web_search` is silently dropped from the
/// request, and the model then reports that it has no such tool. Everything
/// else about it was correct — the tool registered, reached the tool table, and
/// its schema was sent — and the only visible symptom was the model denying the
/// tool existed, with no error anywhere to explain it.
///
/// So the name is `search_web`, and a test pins it.
pub const TOOL: &str = "search_web";

/// The plugin's name: what a pack asks for as `builtin:search`, and what the
/// kernel stamps on the tool it contributes.
pub const PLUGIN: &str = "search";

/// The host's copy of the search settings, shared with the plugin.
///
/// The plugin never reaches into `Settings` — the host owns those and pushes
/// them here; the plugin reads the cell when its fibers apply.
pub type SettingsCell = std::sync::Arc<std::sync::RwLock<Option<SearchSettings>>>;

/// A cell holding `search` for the given settings.
pub fn cell(search: Option<SearchSettings>) -> SettingsCell {
    std::sync::Arc::new(std::sync::RwLock::new(search))
}

/// Make the code available: a pack's assembly asking for `builtin:search`
/// decides whether a conversation actually has search.
///
/// Defining is not loading, the same split as `builtin:delegate` — the code
/// ships inside this binary, *whether it is in use* is the pack's decision.
pub fn install(kernel: &mut crate::plugin::Kernel, settings: SettingsCell) {
    kernel.define(std::sync::Arc::new(Search { settings }));
}

/// Push new settings and re-apply every fiber of the plugin.
///
/// This is how a panel save or a pack's content reaches the tool table: the
/// cell changes, the fibers re-read it, and the tool appears or disappears
/// with the key. With no pack asking for the plugin there are no fibers to
/// re-apply — the settings are still stored, and the next pack load reads
/// them.
pub fn configure(
    kernel: &mut crate::plugin::Kernel,
    settings: &SettingsCell,
    search: Option<SearchSettings>,
) -> Result<()> {
    *settings.write().expect("search settings cell") = search;
    kernel.reload_plugin(PLUGIN)?;
    Ok(())
}

/// The plugin behind [`TOOL`].
pub struct Search {
    settings: SettingsCell,
}

impl crate::plugin::Plugin for Search {
    fn name(&self) -> &str {
        PLUGIN
    }

    fn apply(&self, _ctx: &crate::plugin::PluginCtx) -> Result<crate::plugin::Contributions> {
        // Present only while a key is configured: a search tool that cannot
        // authenticate costs a turn every time the model tries it, while an
        // absent one costs nothing. The pack decides the plugin is there; the
        // settings decide whether it has a tool.
        let current = self
            .settings
            .read()
            .map(|guard| guard.clone())
            .unwrap_or(None);
        match current {
            Some(search) if search.is_configured() => {
                Ok(crate::plugin::Contributions::new().tool(tool(search)?))
            }
            _ => Ok(crate::plugin::Contributions::new()),
        }
    }
}

/// Build the tool: client, schema, handler.
fn tool(settings: SearchSettings) -> Result<ToolDef> {
    let mut builder = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(45))
        .user_agent(concat!("nguruvilu/", env!("CARGO_PKG_VERSION")));
    // Same rule as everywhere else: the environment's proxy variables are
    // ignored unless this program was told to use a proxy.
    if let Ok(proxy) = std::env::var("NGU_PROXY") {
        if !proxy.trim().is_empty() {
            builder = builder.proxy(reqwest::Proxy::all(proxy.trim())?);
        }
    } else {
        builder = builder.no_proxy();
    }
    let http = builder.build().context("building the search client")?;

    let settings = std::sync::Arc::new(settings);
    let client = std::sync::Arc::new(http);

    Ok(ToolDef::new(
        TOOL,
        format!(
            "Search the web with {}. Returns titles, URLs, and excerpts. \
             Use it when you need information you do not have, and cite the \
             URLs you used.",
            settings.provider.as_str()
        ),
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search for. Be specific."
                }
            },
            "required": ["query"]
        }),
        PLUGIN,
        move |args| {
            let settings = std::sync::Arc::clone(&settings);
            let client = std::sync::Arc::clone(&client);
            Box::pin(async move { search(&settings, &client, args).await }) as ToolFuture
        },
    ))
}

async fn search(
    settings: &SearchSettings,
    http: &reqwest::Client,
    args: Value,
) -> Result<ToolOutput> {
    let query = args
        .get("query")
        .and_then(|q| q.as_str())
        .ok_or_else(|| anyhow!("missing required argument: query"))?;

    let (request, _has_body) = build_request(settings, query, http);
    let response = request.send().await.context("calling the search provider")?;

    let status = response.status();
    let text = response.text().await.unwrap_or_default();
    if !status.is_success() {
        return Err(anyhow!(
            "{} returned {status}: {}",
            settings.provider.as_str(),
            text.chars().take(400).collect::<String>()
        ));
    }

    let body: Value =
        serde_json::from_str(&text).context("the search provider returned something else")?;
    let hits = parse_response(settings.provider, &body);

    if hits.is_empty() {
        // Said plainly rather than as an empty result: an empty list reads as
        // "nothing matched", which is a different claim.
        return Ok(ToolOutput::text(format!(
            "No results for '{query}'. The provider returned a response with no \
             result entries; the query may be too narrow, or the account may be \
             out of quota."
        )));
    }

    let mut out = format!("{} result(s) for '{query}':\n", hits.len());
    for (index, hit) in hits.iter().enumerate() {
        out.push_str(&format!(
            "\n{}. {}\n   {}\n   {}\n",
            index + 1,
            hit.title,
            hit.url,
            hit.snippet.trim()
        ));
    }
    Ok(ToolOutput::text(out))
}

/// Read search settings from the environment, when set.
///
/// `NGU_SEARCH_PROVIDER` and `NGU_SEARCH_API_KEY`. The settings file is the
/// normal path; this exists so a shell can try the tool without writing config.
pub fn from_env() -> Option<SearchSettings> {
    let provider = std::env::var("NGU_SEARCH_PROVIDER").ok()?;
    let provider = Dialect::parse(&provider)?;
    Some(SearchSettings {
        provider,
        api_key: std::env::var("NGU_SEARCH_API_KEY").unwrap_or_default(),
        endpoint: std::env::var("NGU_SEARCH_ENDPOINT").ok(),
        max_results: default_limit(),
    })
}

/// A map of dialect name to where its key comes from, for diagnostics.
pub fn describe(settings: &SearchSettings) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    out.insert("provider".into(), settings.provider.as_str().into());
    out.insert("endpoint".into(), settings.url());
    out.insert("max results".into(), settings.max_results.to_string());
    out.insert(
        "key".into(),
        if settings.is_configured() {
            "set".into()
        } else {
            "(absent; the tool is not registered)".into()
        },
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settings(provider: Dialect) -> SearchSettings {
        SearchSettings {
            provider,
            api_key: "test-key".into(),
            endpoint: None,
            max_results: 3,
        }
    }

    #[test]
    fn a_dialect_round_trips_through_its_name() {
        for dialect in Dialect::all() {
            assert_eq!(Dialect::parse(dialect.as_str()), Some(*dialect));
        }
        assert_eq!(Dialect::parse("TAVILY"), Some(Dialect::Tavily));
        assert_eq!(Dialect::parse("google"), None);
    }

    #[test]
    fn each_dialect_has_its_own_default_endpoint() {
        let urls: Vec<&str> = Dialect::all().iter().map(|d| d.default_endpoint()).collect();
        assert_eq!(urls.len(), 3);
        let unique: std::collections::BTreeSet<&&str> = urls.iter().collect();
        assert_eq!(unique.len(), 3, "the three are different services");
    }

    #[test]
    fn a_configured_endpoint_overrides_the_default() {
        let mut settings = settings(Dialect::Tavily);
        settings.endpoint = Some("https://internal/search".into());
        assert_eq!(settings.url(), "https://internal/search");
    }

    #[test]
    fn search_is_unusable_without_a_key() {
        // The tool is not registered at all in that case, rather than
        // registered and failing on every call.
        let mut settings = settings(Dialect::Brave);
        assert!(settings.is_configured());
        settings.api_key = String::new();
        assert!(!settings.is_configured());
        settings.api_key = "   ".into();
        assert!(!settings.is_configured(), "whitespace is not a key");
    }

    #[tokio::test]
    async fn the_requests_differ_in_method_and_header() {
        let http = reqwest::Client::new();

        let (brave, has_body) = build_request(&settings(Dialect::Brave), "rust", &http);
        let brave = brave.build().unwrap();
        assert_eq!(brave.method(), reqwest::Method::GET);
        assert!(
            brave.headers().contains_key("X-Subscription-Token"),
            "Brave authenticates with its own header"
        );
        assert!(!has_body, "a GET carries no body");
        // The query goes in the URL for Brave.
        assert!(brave.url().query().unwrap_or_default().contains("rust"));

        let (tavily, _) = build_request(&settings(Dialect::Tavily), "rust", &http);
        let tavily = tavily.build().unwrap();
        assert_eq!(tavily.method(), reqwest::Method::POST);
        assert!(tavily.headers().contains_key("authorization"));

        let (exa, _) = build_request(&settings(Dialect::Exa), "rust", &http);
        let exa = exa.build().unwrap();
        assert_eq!(exa.method(), reqwest::Method::POST);
        assert!(exa.headers().contains_key("authorization"));
    }

    #[test]
    fn results_are_read_from_each_providers_own_path() {
        let tavily = json!({ "results": [
            { "title": "T", "url": "https://t", "content": "tavily snippet" } ] });
        let hits = parse_response(Dialect::Tavily, &tavily);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].snippet, "tavily snippet");

        // Brave nests one level deeper.
        let brave = json!({ "web": { "results": [
            { "title": "B", "url": "https://b", "description": "brave snippet" } ] } });
        let hits = parse_response(Dialect::Brave, &brave);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].snippet, "brave snippet");

        let exa = json!({ "results": [
            { "title": "E", "url": "https://e", "text": "exa snippet" } ] });
        let hits = parse_response(Dialect::Exa, &exa);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].snippet, "exa snippet");
    }

    #[test]
    fn a_result_without_a_snippet_is_kept() {
        // The URL alone is useful; dropping the result would lose it.
        let body = json!({ "results": [ { "url": "https://x" } ] });
        let hits = parse_response(Dialect::Tavily, &body);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].title, "https://x", "the URL stands in for a title");
        assert_eq!(hits[0].snippet, "");
    }

    #[test]
    fn a_result_without_a_url_is_dropped() {
        // Nothing to cite, so nothing to report.
        let body = json!({ "results": [ { "title": "no url", "content": "x" } ] });
        assert!(parse_response(Dialect::Tavily, &body).is_empty());
    }

    #[test]
    fn a_response_of_the_wrong_shape_yields_nothing_rather_than_panicking() {
        for body in [
            json!({}),
            json!({ "results": "not an array" }),
            json!({ "web": {} }),
            json!([]),
            json!(null),
        ] {
            assert!(parse_response(Dialect::Brave, &body).is_empty());
            assert!(parse_response(Dialect::Tavily, &body).is_empty());
        }
    }

    /// The kernel a pack that asks for `builtin:search` leaves behind: the
    /// code defined, one fiber loaded, and the host's cell to configure it.
    fn loaded(search: Option<SearchSettings>) -> (crate::plugin::Kernel, SettingsCell) {
        let mut kernel = crate::plugin::Kernel::new();
        let cell = cell(search);
        install(&mut kernel, cell.clone());
        kernel
            .load(PLUGIN, crate::plugin::RealmMap::new(), serde_json::Value::Null)
            .unwrap();
        (kernel, cell)
    }

    #[test]
    fn defining_alone_puts_no_tool_in_the_table() {
        // The same split as `builtin:delegate`: the kernel holds the code,
        // `packs/search` is what asks for it. Without the pack's assembly the
        // kernel must not appear to have search.
        let mut kernel = crate::plugin::Kernel::new();
        install(&mut kernel, cell(Some(settings(Dialect::Tavily))));

        assert!(kernel.tools().get(TOOL).is_none());
        assert!(kernel.plugin(PLUGIN).is_some(), "but the code is available");
    }

    #[test]
    fn the_tool_is_absent_without_a_key_and_present_with_one() {
        let mut without = settings(Dialect::Tavily);
        without.api_key = String::new();
        let (kernel, _cell) = loaded(Some(without));
        assert!(kernel.tools().get(TOOL).is_none(), "no key, no tool");

        let (kernel, _cell) = loaded(Some(settings(Dialect::Tavily)));
        assert!(kernel.tools().get(TOOL).is_some());
        assert_eq!(
            kernel.tools().owner(TOOL),
            Some(PLUGIN),
            "the plugin owns the tool it contributes"
        );
    }

    #[test]
    fn configure_makes_the_tool_appear_and_vanish() {
        // The panel's path: settings go into the cell, the fibers re-apply.
        let (mut kernel, cell) = loaded(None);
        assert!(kernel.tools().get(TOOL).is_none());

        configure(&mut kernel, &cell, Some(settings(Dialect::Tavily))).unwrap();
        assert!(kernel.tools().get(TOOL).is_some(), "a key lands the tool");

        configure(&mut kernel, &cell, None).unwrap();
        assert!(kernel.tools().get(TOOL).is_none(), "off takes it away");
    }

    #[test]
    fn without_the_pack_configuring_changes_nothing() {
        // The capability belongs to the pack: settings alone must not smuggle
        // the tool into a conversation that never loaded it.
        let mut kernel = crate::plugin::Kernel::new();
        let cell = cell(None);
        install(&mut kernel, cell.clone());

        configure(&mut kernel, &cell, Some(settings(Dialect::Tavily))).unwrap();
        assert!(kernel.tools().get(TOOL).is_none());
    }

    #[test]
    fn the_tool_description_names_the_provider() {
        let (kernel, _cell) = loaded(Some(settings(Dialect::Brave)));
        let schema = kernel
            .tools()
            .schemas()
            .into_iter()
            .find(|schema| schema["function"]["name"] == TOOL)
            .expect("the search tool has a schema");
        assert!(schema["function"]["description"].as_str().unwrap().contains("brave"));
    }

    #[test]
    fn the_settings_round_trip_through_json() {
        let settings = SearchSettings {
            provider: Dialect::Exa,
            api_key: "k".into(),
            endpoint: Some("https://x".into()),
            max_results: 9,
        };
        let text = serde_json::to_string(&settings).unwrap();
        let back: SearchSettings = serde_json::from_str(&text).unwrap();
        assert_eq!(back.provider, Dialect::Exa);
        assert_eq!(back.max_results, 9);
    }

    #[test]
    fn an_unknown_field_is_refused() {
        let text = "{\"provider\":\"tavily\",\"apiKey\":\"k\",\"maxResult\":5}";
        let error = serde_json::from_str::<SearchSettings>(text).expect_err("must refuse");
        assert!(format!("{error}").contains("maxResult"), "{error}");
    }

    #[test]
    fn the_default_result_count_is_reasonable() {
        let text = "{\"provider\":\"tavily\",\"apiKey\":\"k\"}";
        let settings: SearchSettings = serde_json::from_str(text).unwrap();
        assert_eq!(settings.max_results, default_limit());
    }

    #[test]
    fn a_description_says_whether_the_key_is_there() {
        let described = describe(&settings(Dialect::Tavily));
        assert_eq!(described["key"], "set");
        let mut absent = settings(Dialect::Tavily);
        absent.api_key = String::new();
        assert!(describe(&absent)["key"].contains("not registered"));
    }

    #[test]
    fn the_tool_is_not_named_web_search() {
        // A function tool called `web_search` is dropped by the provider this
        // was tested against — the model then denies the tool exists, with no
        // error anywhere. Renaming it was the fix, so the name is pinned.
        assert_ne!(TOOL, "web_search");
        assert_eq!(TOOL, "search_web");
    }
}
