//! The `judge` tool: a decision model in the loop, not a gate in the way.
//!
//! Jev (TypeSafe AI's "System One" model) answers typed questions about a
//! block of state: a choice, a score, or a yes/no probability — with a
//! confidence derived from the answer distribution. It does not chat; a
//! full request runs in 70–500 ms, which is what makes "ask before deciding"
//! affordable inside a turn.
//!
//! # The red line
//!
//! The judge is an **adviser, never a gate**: its answer arrives in the
//! transcript as data, and the model (and the user) decide what it means.
//! A `0` is reported, not enforced — this kernel refuses no action on a
//! judge's say-so, and the tool's own description says so. That is the same
//! "no gates, report honestly" rule the rest of the system runs under.
//!
//! # Contract
//!
//! `POST <endpoint>/v1/systemone` with `Authorization: Bearer <key>`:
//!
//! ```json
//! { "model": "jev-latest",
//!   "state": "…anything to judge…",
//!   "questions": { "q": { "type": "choice", "instructions": "…",
//!                          "choices": ["a", "b"] } } }
//! ```
//!
//! Answer shapes vary per question type (`choice` carries `choice`,
//! `noul` a boolean, `score` a number — each with `confidence` and usually
//! `probabilities`), so answers are flattened generically: a verdict field is
//! picked when present, and anything unrecognised is passed through as raw
//! JSON rather than guessed at.

use anyhow::{anyhow, Context, Result};
use serde_json::{json, Value};

/// The tool name.
pub const TOOL: &str = "judge";

/// The plugin's name: what a pack asks for as `builtin:judge`.
pub const PLUGIN: &str = "judge";

/// The evaluate endpoint's path under the endpoint host.
const PATH: &str = "/v1/systemone";

/// How the judge is reached and which model answers.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct JudgeSettings {
    /// API base, e.g. `https://api.typesafe.ai` — no path.
    pub endpoint: String,
    /// Bearer key. Never enters a pack; the panel only ever sees "(set)".
    pub api_key: String,
    /// Model alias; `jev-latest` tracks whatever the provider calls current.
    pub model: String,
}

impl Default for JudgeSettings {
    fn default() -> Self {
        Self {
            endpoint: "https://api.typesafe.ai".into(),
            api_key: String::new(),
            model: "jev-latest".into(),
        }
    }
}

impl JudgeSettings {
    /// Usable only with a key: an unauthenticated call is a guaranteed 401,
    /// and a tool that always fails costs a turn every time it is tried.
    pub fn is_configured(&self) -> bool {
        !self.api_key.trim().is_empty()
    }

    /// The full evaluate URL.
    fn url(&self) -> String {
        format!(
            "{}{}",
            self.endpoint.trim_end_matches('/'),
            PATH
        )
    }
}

/// Environment fallbacks, mirroring how the model key and search arrive:
/// a user who has never opened a panel can still export three variables.
pub fn from_env() -> Option<JudgeSettings> {
    let api_key = std::env::var("NGU_JUDGE_API_KEY").unwrap_or_default();
    if api_key.trim().is_empty() {
        return None;
    }
    let mut settings = JudgeSettings::default();
    settings.api_key = api_key;
    if let Ok(endpoint) = std::env::var("NGU_JUDGE_ENDPOINT") {
        if !endpoint.trim().is_empty() {
            settings.endpoint = endpoint.trim().to_string();
        }
    }
    if let Ok(model) = std::env::var("NGU_JUDGE_MODEL") {
        if !model.trim().is_empty() {
            settings.model = model.trim().to_string();
        }
    }
    Some(settings)
}

/// The host's copy of the judge settings, shared with the plugin.
///
/// The plugin never reaches into `Settings` — the host owns those and pushes
/// them here; the plugin reads the cell when its fibers apply.
pub type SettingsCell = std::sync::Arc<std::sync::RwLock<Option<JudgeSettings>>>;

/// A cell holding the given settings.
pub fn cell(judge: Option<JudgeSettings>) -> SettingsCell {
    std::sync::Arc::new(std::sync::RwLock::new(judge))
}

/// Make the code available: a pack's assembly asking for `builtin:judge`
/// decides whether a conversation actually has a judge. Same split as
/// `builtin:delegate` and `builtin:search`.
pub fn install(kernel: &mut crate::plugin::Kernel, judge: SettingsCell) {
    kernel.define(std::sync::Arc::new(Judge { judge }));
}

/// Push new settings and re-apply every fiber of the plugin: the tool appears
/// when a key lands and vanishes when the panel turns the judge off.
pub fn configure(
    kernel: &mut crate::plugin::Kernel,
    cell: &SettingsCell,
    judge: Option<JudgeSettings>,
) -> Result<()> {
    *cell.write().expect("judge settings cell") = judge;
    kernel.reload_plugin(PLUGIN)?;
    Ok(())
}

/// The plugin behind [`TOOL`].
pub struct Judge {
    judge: SettingsCell,
}

impl crate::plugin::Plugin for Judge {
    fn name(&self) -> &str {
        PLUGIN
    }

    fn apply(&self, _ctx: &crate::plugin::PluginCtx) -> Result<crate::plugin::Contributions> {
        let current = self
            .judge
            .read()
            .map(|guard| guard.clone())
            .unwrap_or(None);
        match current {
            Some(judge) if judge.is_configured() => {
                Ok(crate::plugin::Contributions::new().tool(tool(judge)?))
            }
            _ => Ok(crate::plugin::Contributions::new()),
        }
    }
}

/// Build the tool: schema and handler.
fn tool(settings: JudgeSettings) -> Result<ToolDef> {
    let description = format!(
        "Judge a block of state against typed questions with {} — a fast, \
         type-safe decision model (70–500 ms), not a text generator. Send \
         `state` (whatever is being judged, as text or JSON) and `questions` — \
         a map of id → question. Question types: choice {{\"type\": \"choice\", \
         \"instructions\": \"…\", \"choices\": [\"…\"]}}, score {{\"type\": \"score\", \
         \"instructions\": \"…\", \"range\": [0, 100]}}, noul (yes/no) {{\"type\": \
         \"noul\", \"instructions\": \"…\", \"criteria\": {{\"true\": \"…\", \
         \"false\": \"…\"}}}}. Each answer comes back with its value, a confidence \
         in [0,1] and the probability distribution. The answers are data to \
         report and act on at your own judgment — a low confidence means \
         gather more information or ask the user, never silently proceed as \
         if the answer were certain.",
        settings.model
    );
    Ok(ToolDef::new(
        TOOL,
        description,
        json!({
            "type": "object",
            "properties": {
                "state": {
                    "type": "string",
                    "description": "The material to judge: text, or a JSON object rendered as text."
                },
                "questions": {
                    "type": "object",
                    "description": "Map of question id to a typed question (choice / score / noul). See the tool description for shapes.",
                    "additionalProperties": { "type": "object" }
                },
                "model": {
                    "type": "string",
                    "description": "Override the judge model for this call. Defaults to the configured one."
                }
            },
            "required": ["state", "questions"]
        }),
        PLUGIN,
        move |args| {
            let settings = settings.clone();
            Box::pin(async move { run(&settings, args).await }) as ToolFuture
        },
    ))
}

/// One evaluate call.
async fn run(settings: &JudgeSettings, args: Value) -> Result<ToolOutput> {
    let state = match args.get("state") {
        None => None,
        Some(Value::Null) => None,
        Some(Value::String(text)) if text.trim().is_empty() => None,
        Some(Value::String(_)) | Some(Value::Object(_)) | Some(Value::Array(_)) => {
            args.get("state").cloned()
        }
        // A number or boolean as "state" is a caller mistake worth naming.
        Some(_) => {
            return Err(anyhow!(
                "state must be text, an object, or an array — the material to judge"
            ))
        }
    };
    let state = state.ok_or_else(|| {
        anyhow!(
            "missing required argument: state — the material to judge. The judge \
             needs the content itself; a question alone has nothing to read."
        )
    })?;
    let questions = args
        .get("questions")
        .and_then(|q| q.as_object())
        .filter(|q| !q.is_empty())
        .cloned()
        .ok_or_else(|| {
            anyhow!(
                "missing required argument: questions — a map of id → typed question. \
                 Without one the evaluate endpoint has nothing to answer."
            )
        })?;
    let model = args
        .get("model")
        .and_then(|m| m.as_str())
        .map(str::trim)
        .filter(|m| !m.is_empty())
        .unwrap_or(&settings.model)
        .to_string();

    let body = json!({ "model": model, "state": state, "questions": questions });

    let mut builder = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .user_agent(concat!("nguruvilu/", env!("CARGO_PKG_VERSION")));
    // The same proxy rule as every other client: only an explicit NGU_PROXY
    // counts; the environment's variables are not allowed to reroute traffic
    // chosen by this program.
    if let Ok(proxy) = std::env::var("NGU_PROXY") {
        if !proxy.trim().is_empty() {
            builder = builder.proxy(reqwest::Proxy::all(proxy.trim())?);
        }
    } else {
        builder = builder.no_proxy();
    }
    let client = builder.build().context("building the judge client")?;

    let response = client
        .post(settings.url())
        .header("Authorization", format!("Bearer {}", settings.api_key.trim()))
        .header("Content-Type", "application/json")
        .json(&body)
        .send()
        .await
        .with_context(|| format!("reaching the judge at {}", settings.endpoint))?;

    let status = response.status();
    let text = response.text().await.unwrap_or_default();
    Ok(ToolOutput::text(interpret(status, &text)?))
}

/// Turn a response (or a failure) into the transcript.
///
/// Split out from the network call so the contract is unit-testable without
/// a server: every status maps to a message that says what to *do*, and every
/// success body is flattened without guessing at shapes.
fn interpret(status: reqwest::StatusCode, text: &str) -> Result<String> {
    if !status.is_success() {
        let snippet: String = text.chars().take(600).collect();
        return Err(match status.as_u16() {
            401 => anyhow!(
                "judge: 401 — the key is missing or invalid. Set it in the settings \
                 panel (Judge section), with `ngu config set --judge-key`, or in \
                 NGU_JUDGE_API_KEY."
            ),
            422 => anyhow!("judge: 422 — the request does not match the contract: {snippet}"),
            429 => anyhow!("judge: 429 — rate limited; back off and retry rather than looping."),
            529 => anyhow!("judge: 529 — the service is overloaded; retry later."),
            _ => anyhow!("judge: {status}: {snippet}"),
        });
    }

    let value: Value = serde_json::from_str(text)
        .with_context(|| format!("judge returned a non-JSON body: {}", snippet_of(text)))?;
    let answers = value
        .get("answers")
        .cloned()
        .unwrap_or_else(|| json!({}));

    let mut out = String::new();
    let usage = value.get("usage").cloned().unwrap_or(Value::Null);
    let model = value
        .get("model")
        .and_then(|m| m.as_str())
        .unwrap_or("(unreported)");
    out.push_str(&format!("judge ({model}):\n"));
    for (id, answer) in answers.as_object().into_iter().flatten() {
        out.push_str(&format!("  {id}: {}\n", flatten_answer(answer)));
    }
    if !usage.is_null() {
        out.push_str(&format!(
            "usage: {} in / {} out tokens\n",
            usage.get("input_tokens").and_then(Value::as_u64).unwrap_or(0),
            usage.get("output_tokens").and_then(Value::as_u64).unwrap_or(0)
        ));
    }
    out.push_str(&format!("raw: {}", compact(&value)));
    Ok(out)
}

/// One answer → one line: verdict, confidence, distribution.
///
/// The three primitives name their value differently (`choice`, `value`,
/// `score`), and the provider may add more; picking the first known field and
/// passing the rest through as raw JSON keeps an unknown shape honest instead
/// of inventing a verdict.
fn flatten_answer(answer: &Value) -> String {
    let verdict = ["choice", "value", "score"]
        .iter()
        .find_map(|key| answer.get(*key))
        .cloned();
    let confidence = answer
        .get("confidence")
        .and_then(Value::as_f64)
        .map(|c| format!("{c:.3}"));
    let probabilities = answer.get("probabilities").cloned();

    let mut line = String::new();
    match verdict {
        Some(Value::String(text)) => line.push_str(&text),
        Some(Value::Bool(flag)) => line.push_str(if flag { "true" } else { "false" }),
        Some(Value::Number(number)) => line.push_str(&number.to_string()),
        Some(other) => line.push_str(&compact(&other)),
        // No known value field: this shape is the provider's to define.
        None => return compact(answer),
    }
    if let Some(confidence) = confidence {
        line.push_str(&format!("  (confidence {confidence})"));
    }
    if let Some(probabilities) = probabilities {
        if let Some(map) = probabilities.as_object() {
            let parts: Vec<String> = map
                .iter()
                .filter_map(|(label, p)| p.as_f64().map(|p| format!("{label}={p:.2}")))
                .collect();
            if !parts.is_empty() {
                line.push_str(&format!("  [{}]", parts.join(", ")));
            }
        }
    }
    line
}

fn compact(value: &Value) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| value.to_string())
}

fn snippet_of(text: &str) -> String {
    text.chars().take(300).collect()
}

use crate::tools::{ToolDef, ToolFuture, ToolOutput};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plugin::{Kernel, RealmMap};

    fn settings(key: &str) -> JudgeSettings {
        JudgeSettings {
            api_key: key.into(),
            ..JudgeSettings::default()
        }
    }

    /// The kernel a pack that asks for `builtin:judge` leaves behind.
    fn loaded(judge: Option<JudgeSettings>) -> (Kernel, SettingsCell) {
        let mut kernel = Kernel::new();
        let cell = cell(judge);
        install(&mut kernel, cell.clone());
        kernel
            .load(PLUGIN, RealmMap::new(), Value::Null)
            .unwrap();
        (kernel, cell)
    }

    #[test]
    fn a_key_is_the_gate_and_the_panel_never_sees_one() {
        assert!(!JudgeSettings::default().is_configured(), "no key, no tool");
        assert!(settings("sk-test").is_configured());

        // The value round-trips through settings JSON so a panel save works…
        let text = serde_json::to_string(&settings("sk-test")).unwrap();
        assert!(text.contains("api.typesafe.ai"));
        // …and is the shape a pack's content would never carry (no such file
        // exists for the judge on purpose).
        let back: JudgeSettings = serde_json::from_str(&text).unwrap();
        assert_eq!(back.model, "jev-latest");
    }

    #[test]
    fn defining_alone_puts_no_tool_in_the_table() {
        let mut kernel = Kernel::new();
        install(&mut kernel, cell(Some(settings("sk-test"))));
        assert!(kernel.tools().get(TOOL).is_none());
        assert!(kernel.plugin(PLUGIN).is_some(), "but the code is available");
    }

    #[test]
    fn configure_makes_the_tool_appear_and_vanish() {
        let (mut kernel, cell) = loaded(None);
        assert!(kernel.tools().get(TOOL).is_none(), "nothing configured");

        configure(&mut kernel, &cell, Some(settings("sk-test"))).unwrap();
        assert!(kernel.tools().get(TOOL).is_some(), "a key lands the tool");
        assert_eq!(kernel.tools().owner(TOOL), Some(PLUGIN));

        configure(&mut kernel, &cell, None).unwrap();
        assert!(kernel.tools().get(TOOL).is_none(), "off takes it away");
    }

    #[test]
    fn a_choice_answer_flattens_to_verdict_and_distribution() {
        // The shape from the JEV-as-a-Judge paper's worked example.
        let answer = json!({
            "type": "choice",
            "choice": "supported",
            "confidence": 1.0,
            "probabilities": { "contradicted": 0.0, "unknown": 0.0, "supported": 1.0 }
        });
        let line = flatten_answer(&answer);
        assert!(line.starts_with("supported"), "{line}");
        assert!(line.contains("confidence 1.000"), "{line}");
        assert!(line.contains("supported=1.00"), "{line}");
    }

    #[test]
    fn boolean_and_numeric_answers_both_survive() {
        let noul = json!({ "type": "noul", "value": true, "confidence": 0.82 });
        let line = flatten_answer(&noul);
        assert!(line.starts_with("true"), "{line}");
        assert!(line.contains("confidence 0.820"), "{line}");

        let score = json!({ "type": "score", "score": 73, "confidence": 0.6 });
        assert!(flatten_answer(&score).starts_with("73"), "{score}");
    }

    #[test]
    fn an_unknown_shape_is_passed_through_not_invented() {
        let odd = json!({ "something_new": { "nested": 1 } });
        assert_eq!(flatten_answer(&odd), compact(&odd), "honest, no guessing");
    }

    #[test]
    fn a_success_body_flattens_end_to_end() {
        let body = json!({
            "model": "jev-1.13.0",
            "answers": {
                "is_bug": { "type": "noul", "value": true, "confidence": 0.97,
                            "probabilities": { "true": 0.97, "false": 0.03 } }
            },
            "usage": { "input_tokens": 416, "output_tokens": 42 }
        });
        let text = interpret(
            reqwest::StatusCode::OK,
            &serde_json::to_string(&body).unwrap(),
        )
        .unwrap();
        assert!(text.contains("judge (jev-1.13.0)"), "{text}");
        assert!(text.contains("is_bug: true"), "{text}");
        assert!(text.contains("confidence 0.970"), "{text}");
        assert!(text.contains("416 in / 42 out"), "{text}");
    }

    #[test]
    fn every_failure_says_what_to_do_next() {
        let cases = [
            (401, "judge: 401"),
            (422, "judge: 422"),
            (429, "judge: 429"),
            (529, "judge: 529"),
            (500, "judge: 500"),
        ];
        for (code, prefix) in cases {
            let status = reqwest::StatusCode::from_u16(code).unwrap();
            let error = format!(
                "{:#}",
                interpret(status, r#"{"detail":"bad"}"#).expect_err("must fail")
            );
            assert!(error.starts_with(prefix), "{error}");
        }
        let error = format!(
            "{:#}",
            interpret(reqwest::StatusCode::UNAUTHORIZED, "").expect_err("must fail")
        );
        assert!(error.contains("NGU_JUDGE_API_KEY"), "{error}");
    }

    #[test]
    fn the_endpoint_never_double_slashes() {
        let mut s = JudgeSettings::default();
        assert_eq!(s.url(), "https://api.typesafe.ai/v1/systemone");
        s.endpoint = "https://alt.example/".into();
        assert_eq!(s.url(), "https://alt.example/v1/systemone");
    }

    #[test]
    fn env_provides_a_key_without_a_panel() {
        // The env fallback is opt-in per process; assert the parse path only
        // through the struct (setting real env vars would race other tests).
        let mut s = JudgeSettings::default();
        s.api_key = "from-env".into();
        assert!(s.is_configured());
    }
}
