//! The `judge` plugin: a decision model in the loop, as a process.
//!
//! It was `src/tools/judge.rs` (773 lines compiled into the kernel). Nothing
//! about the judge needed to be in the kernel: it makes one HTTP call and
//! flattens the answer. It is now an executable the kernel starts, and the
//! kernel has no idea a judge exists.
//!
//! # What it needs from the kernel, and gets
//!
//! Its endpoint and key, which live in the settings file under the `judge` key
//! of the pack table. The kernel passes the whole table in
//! `NGU_PACK_SETTINGS`; this takes its own key and never asks for anything else.
//! That is what lets a settings save take effect on the very next call without
//! the kernel having to know a judge exists — there is nothing to push.
//!
//! # The red line, unchanged
//!
//! The judge is an **adviser, never a gate**: its answer arrives in the
//! transcript as data, and the model (and the user) decide what it means. A `0`
//! is reported, not enforced. This plugin cannot make the kernel refuse
//! anything — it has no channel for that, by design — so the red line is now
//! structural rather than a rule somebody has to remember.
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
//! Answer shapes vary per question type (`choice` carries `choice`, `noul` a
//! boolean, `score` a number — each with `confidence` and usually
//! `probabilities`), so answers are flattened generically: a verdict field is
//! picked when present, and anything unrecognised is passed through as raw JSON
//! rather than guessed at.

use std::io::{Read, Write};
use std::time::Duration;

use serde_json::{json, Value};

/// The tool's name.
const TOOL: &str = "judge";
/// The key this plugin's settings live under in the pack table.
const SETTINGS_KEY: &str = "judge";
/// The evaluate endpoint's path under the endpoint host.
const PATH: &str = "/v1/systemone";
/// The most response body one call reads into memory.
const MAX_BODY_BYTES: usize = 4 * 1024 * 1024;

fn main() {
    let mut stdin = std::io::stdin();
    loop {
        let frame = match read_frame(&mut stdin) {
            Some(frame) => frame,
            None => return,
        };
        let message: Value = match serde_json::from_slice(&frame) {
            Ok(value) => value,
            Err(error) => {
                write_frame(&json!({ "error": format!("not JSON: {error}") }));
                continue;
            }
        };

        let answer = match message.get("op").and_then(|op| op.as_str()) {
            Some("describe") => describe(),
            Some("call") if message.get("tool").and_then(|t| t.as_str()) == Some(TOOL) => {
                run(message.get("args").cloned().unwrap_or(Value::Null))
            }
            Some("call") => json!({
                "error": format!(
                    "no such tool: {}",
                    message.get("tool").unwrap_or(&Value::Null)
                )
            }),
            Some(other) => json!({ "error": format!("unsupported op: {other}") }),
            None => json!({ "error": "missing op" }),
        };
        write_frame(&answer);
    }
}

/// The judge's settings, or `None` when the user has not configured one.
struct Settings {
    endpoint: String,
    api_key: String,
    model: String,
}

impl Settings {
    /// Read this plugin's own entry out of the pack settings table.
    ///
    /// `None` covers three cases the caller does not need to tell apart for the
    /// tool to be absent — no table, no entry, nothing usable in it — but the
    /// `describe` reply states which one, because "the tool is missing" with no
    /// reason is the failure mode this whole project keeps trying to remove.
    fn load() -> Result<Option<Self>, String> {
        let raw = match std::env::var("NGU_PACK_SETTINGS") {
            Ok(raw) => raw,
            // No table at all: the kernel did not pass one. Not an error — a
            // plugin loaded outside a kernel has no settings and no tool.
            Err(_) => return Ok(None),
        };
        let all: Value = match serde_json::from_str(&raw) {
            Ok(value) => value,
            Err(error) => return Err(format!("NGU_PACK_SETTINGS is not JSON: {error}")),
        };
        let entry = match all.get(SETTINGS_KEY) {
            Some(entry) if !entry.is_null() => entry,
            _ => return Ok(None),
        };
        let endpoint = entry
            .get("endpoint")
            .and_then(|v| v.as_str())
            .unwrap_or("https://api.typesafe.ai")
            .trim()
            .trim_end_matches('/')
            .to_string();
        let api_key = entry
            .get("api_key")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_string();
        let model = entry
            .get("model")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|m| !m.is_empty())
            .unwrap_or("jev-latest")
            .to_string();
        if api_key.is_empty() {
            // A judge with no key is a guaranteed 401, and a tool that always
            // fails costs a turn every time the model tries it.
            return Ok(None);
        }
        Ok(Some(Self {
            endpoint,
            api_key,
            model,
        }))
    }

    /// The full evaluate URL.
    fn url(&self) -> String {
        format!("{}{PATH}", self.endpoint)
    }
}

/// The `describe` reply.
///
/// With no key configured it describes **no tools**, which is how the kernel
/// learns the judge is off: the tool does not appear in the table at all, rather
/// than appearing and failing every time it is tried. The reason is in
/// `settings_key` so a panel can say what to fill in.
fn describe() -> Value {
    match Settings::load() {
        Ok(Some(_)) => json!({
            "tools": [{
                "name": TOOL,
                "description": DESCRIPTION,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "state": { "description": "The material to judge: text, an object, or an array." },
                        "questions": { "type": "object", "description":
                            "A map of id → typed question. Types: choice {instructions, choices}, \
                             score {instructions, range}, noul {instructions, criteria}." },
                        "model": { "type": "string", "description": "Override the model alias for this call." }
                    },
                    "required": ["state", "questions"]
                }
            }],
            "settings_key": SETTINGS_KEY
        }),
        Ok(None) => json!({
            "tools": [],
            "settings_key": SETTINGS_KEY,
            "note": "no judge configured: set `judge.endpoint` and `judge.api_key` in the \
                     settings file (panel: 判官 three fields) — an unauthenticated judge \
                     costs a turn every time it is tried"
        }),
        Err(error) => json!({ "tools": [], "error": error, "settings_key": SETTINGS_KEY }),
    }
}

const DESCRIPTION: &str = "Judge a block of state against typed questions with a fast, \
     type-safe decision model (70-500 ms), not a text generator. Send the material and a \
     map of questions; the answer is a verdict per question plus confidence. It ADVISES: \
     its answer arrives as data for you to weigh — nothing is blocked or allowed by it. \
     Low confidence means ask again with more material, or ask the user.";

/// One `call`.
fn run(args: Value) -> Value {
    let settings = match Settings::load() {
        Ok(Some(settings)) => settings,
        Ok(None) => {
            return json!({
                "error": "no judge configured. Set `judge.endpoint` and `judge.api_key` \
                          in ~/.nguruvilu/settings.json (or the panel's three 判官 fields), \
                          then the tool becomes available."
            })
        }
        Err(error) => return json!({ "error": error }),
    };

    let state = match args.get("state") {
        None | Some(Value::Null) => None,
        Some(Value::String(text)) if text.trim().is_empty() => None,
        Some(Value::String(_)) | Some(Value::Object(_)) | Some(Value::Array(_)) => {
            args.get("state").cloned()
        }
        // A number or boolean as "state" is a caller mistake worth naming.
        Some(_) => {
            return json!({
                "error": "state must be text, an object, or an array — the material to judge"
            })
        }
    };
    let Some(state) = state else {
        return json!({
            "error": "missing required argument: state — the material to judge. The judge \
                      needs the content itself; a question alone has nothing to read."
        });
    };
    let questions = match args
        .get("questions")
        .and_then(|q| q.as_object())
        .filter(|q| !q.is_empty())
        .cloned()
    {
        Some(questions) => questions,
        None => {
            return json!({
                "error": "missing required argument: questions — a map of id → typed \
                          question. Without one the evaluate endpoint has nothing to answer."
            })
        }
    };
    let model = args
        .get("model")
        .and_then(|m| m.as_str())
        .map(str::trim)
        .filter(|m| !m.is_empty())
        .unwrap_or(&settings.model)
        .to_string();

    let body = json!({ "model": model, "state": state, "questions": questions });

    let mut builder = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(30))
        .user_agent(concat!("nguruvilu-judge/", env!("CARGO_PKG_VERSION")));
    // The same proxy rule as every other client in this project: only an explicit
    // NGU_PROXY counts; the environment's proxy variables are not allowed to
    // reroute traffic this program chose.
    match std::env::var("NGU_PROXY") {
        Ok(proxy) if !proxy.trim().is_empty() => match reqwest::Proxy::all(proxy.trim()) {
            Ok(proxy) => builder = builder.proxy(proxy),
            Err(error) => return json!({ "error": format!("bad NGU_PROXY: {error}") }),
        },
        _ => builder = builder.no_proxy(),
    }
    let client = match builder.build() {
        Ok(client) => client,
        Err(error) => return json!({ "error": format!("building the judge client: {error}") }),
    };

    let response = client
        .post(settings.url())
        .header("Authorization", format!("Bearer {}", settings.api_key))
        .header("Content-Type", "application/json")
        .json(&body)
        .send();

    let response = match response {
        Ok(response) => response,
        Err(error) => {
            return json!({ "error": format!("reaching the judge at {}: {error}", settings.endpoint) })
        }
    };

    let status = response.status();
    // Bounded read: a misbehaving endpoint must fail at the cap rather than
    // after being buffered in full.
    let text = match read_bounded(response) {
        Ok(text) => text,
        Err(error) => return json!({ "error": error }),
    };
    match interpret(status.as_u16(), &text) {
        Ok(result) => json!({ "result": result }),
        Err(error) => json!({ "error": error }),
    }
}

/// Read a response body, refusing an oversized one.
fn read_bounded(response: reqwest::blocking::Response) -> Result<String, String> {
    if response.content_length().unwrap_or(0) > MAX_BODY_BYTES as u64 {
        return Err(format!(
            "the judge announced more than {} MiB of body; refusing to read it",
            MAX_BODY_BYTES / (1024 * 1024)
        ));
    }
    let mut reader = response;
    let mut bytes = Vec::new();
    let mut chunk = [0u8; 8192];
    loop {
        match reader.read(&mut chunk) {
            Ok(0) => break,
            Ok(n) => {
                if bytes.len() + n > MAX_BODY_BYTES {
                    return Err(format!(
                        "the judge sent more than {} MiB of body; refusing to read it",
                        MAX_BODY_BYTES / (1024 * 1024)
                    ));
                }
                bytes.extend_from_slice(&chunk[..n]);
            }
            Err(error) => return Err(format!("the judge response could not be read: {error}")),
        }
    }
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

/// Turn a response (or a failure) into the transcript.
///
/// Every status maps to a message that says what to **do**, because a judge that
/// answers "401" leaves the model with nothing to try next.
fn interpret(status: u16, text: &str) -> Result<String, String> {
    match status {
        200..=299 => interpret_body(text),
        401 | 403 => Err("the judge rejected the key (401/403). Set it again: \
                          `judge.api_key` in the settings file, the panel's 判官 key field, \
                          or NGU_JUDGE_API_KEY in the environment."
            .to_string()),
        404 => Err(format!(
            "no evaluate endpoint at that address (404). Check `judge.endpoint` — it \
             should be the host only, with no path. Body: {}",
            snippet(text)
        )),
        422 => Err(format!(
            "the judge rejected the request's shape (422). This is a contract problem, \
             not something to retry: the questions must be typed objects. Body: {}",
            snippet(text)
        )),
        429 => Err("rate limited (429). Wait, or send fewer questions per call.".to_string()),
        529 | 503 => Err(format!(
            "the judge is unavailable ({status}). Retry later; this is not a problem with \
             the request. Body: {}",
            snippet(text)
        )),
        other => Err(format!(
            "the judge answered HTTP {other}. Body: {}",
            snippet(text)
        )),
    }
}

/// Flatten a successful body into what the model reads.
fn interpret_body(text: &str) -> Result<String, String> {
    let value: Value = serde_json::from_str(text)
        .map_err(|_| format!("the judge returned a non-JSON body: {}", snippet(text)))?;
    let answers = value.get("answers").and_then(|a| a.as_object());
    let Some(answers) = answers else {
        // No `answers` at all: pass the body through rather than inventing a
        // verdict. A shape this build does not know is still information.
        return Ok(format!("judge (raw)\n{}", compact(&value)));
    };
    let model = value
        .get("model")
        .and_then(|m| m.as_str())
        .unwrap_or("judge");
    let mut out = format!("judge ({model})\n");
    for (id, answer) in answers {
        out.push_str("  ");
        out.push_str(id);
        out.push_str(": ");
        out.push_str(&flatten_answer(answer));
        out.push('\n');
    }
    Ok(out.trim_end().to_string())
}

/// One answer as a single line: verdict, then confidence and distribution.
fn flatten_answer(answer: &Value) -> String {
    if let Some(text) = answer.as_str() {
        return text.to_string();
    }
    // A verdict field is picked when present; the order matters only when an
    // answer carries two, which none of the documented types do.
    let verdict = ["choice", "value", "score", "noul"]
        .iter()
        .find_map(|key| answer.get(*key))
        .map(compact);
    let confidence = answer
        .get("confidence")
        .and_then(|c| c.as_f64())
        .map(|c| format!(" (confidence {c:.3})"))
        .unwrap_or_default();
    let spread = answer
        .get("probabilities")
        .and_then(|p| p.as_object())
        .map(|map| {
            let parts: Vec<String> = map
                .iter()
                .map(|(key, value)| {
                    format!(
                        "{key}={:.2}",
                        value.as_f64().unwrap_or(0.0)
                    )
                })
                .collect();
            format!(" [{}]", parts.join(" "))
        })
        .unwrap_or_default();
    match verdict {
        Some(verdict) => format!("{verdict}{confidence}{spread}"),
        // Nothing recognised: the raw JSON, not a guess.
        None => compact(answer),
    }
}

/// Compact JSON, for a single line of transcript.
fn compact(value: &Value) -> String {
    serde_json::to_string(value).unwrap_or_default()
}

/// A short excerpt of a body, for an error message.
fn snippet(text: &str) -> String {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return "(empty)".to_string();
    }
    let taken: String = trimmed.chars().take(400).collect();
    if trimmed.chars().count() > 400 {
        format!("{taken}…")
    } else {
        taken
    }
}

/* ---------------------------------------------------------------- framing */

fn read_frame(input: &mut impl Read) -> Option<Vec<u8>> {
    let mut header = [0u8; 4];
    if input.read_exact(&mut header).is_err() {
        return None;
    }
    let len = u32::from_be_bytes(header) as usize;
    let mut body = vec![0u8; len];
    input.read_exact(&mut body).ok()?;
    Some(body)
}

fn write_frame(value: &Value) {
    let body = match serde_json::to_vec(value) {
        Ok(body) => body,
        Err(_) => return,
    };
    let mut out = std::io::stdout();
    let mut header = [0u8; 4];
    header.copy_from_slice(&(body.len() as u32).to_be_bytes());
    let _ = out.write_all(&header);
    let _ = out.write_all(&body);
    let _ = out.flush();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_choice_answer_flattens_to_verdict_and_distribution() {
        let answer = json!({
            "type": "choice",
            "choice": "supported",
            "confidence": 1.0,
            "probabilities": { "contradicted": 0.0, "supported": 1.0 }
        });
        let line = flatten_answer(&answer);
        assert!(line.starts_with("supported"), "{line}");
        assert!(line.contains("confidence 1.000"), "{line}");
        assert!(line.contains("supported=1.00"), "{line}");
    }

    #[test]
    fn an_unknown_shape_is_passed_through_not_invented() {
        let odd = json!({ "something_new": { "nested": 1 } });
        assert_eq!(flatten_answer(&odd), compact(&odd));
    }

    #[test]
    fn a_401_names_the_three_places_the_key_can_live() {
        let text = interpret(401, "").expect_err("must fail");
        assert!(text.contains("judge.api_key"), "{text}");
        assert!(text.contains("NGU_JUDGE_API_KEY"), "{text}");
        assert!(text.contains("panel"), "{text}");
    }

    #[test]
    fn a_422_says_not_to_retry() {
        let text = interpret(422, "{\"detail\":\"bad\"}").expect_err("must fail");
        assert!(text.contains("not something to retry"), "{text}");
    }

    #[test]
    fn no_key_means_no_tools_rather_than_a_tool_that_always_fails() {
        // The rule the kernel version had, now expressed by an empty `describe`.
        std::env::remove_var("NGU_PACK_SETTINGS");
        let described = describe();
        assert_eq!(described["tools"].as_array().unwrap().len(), 0);
    }

    #[test]
    fn the_settings_survive_the_environment_round_trip() {
        std::env::set_var(
            "NGU_PACK_SETTINGS",
            json!({
                "judge": {
                    "endpoint": "https://api.typesafe.ai/",
                    "api_key": "sk-test",
                    "model": "jev-latest"
                }
            })
            .to_string(),
        );
        let settings = Settings::load().unwrap().expect("configured");
        assert_eq!(settings.endpoint, "https://api.typesafe.ai");
        assert_eq!(settings.model, "jev-latest");
        assert!(settings.url().ends_with("/v1/systemone"));
        std::env::remove_var("NGU_PACK_SETTINGS");
    }
}
