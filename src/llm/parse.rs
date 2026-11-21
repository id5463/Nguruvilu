use super::*;

/// A token count a provider wrote as a number or a string, when it is plausible.
///
/// Anything else — an object, a negative, a number far outside what a context
/// window can be — reads as "not reported" rather than as an error: the catalog
/// is still useful without it.
pub fn plausible_window(value: &Value) -> Option<usize> {
    let tokens = match value {
        Value::Number(number) => number.as_u64().map(|n| n as usize)?,
        Value::String(text) => text.trim().replace(['_', ','], "").parse::<usize>().ok()?,
        _ => return None,
    };
    (crate::window::MIN_PLAUSIBLE_WINDOW..=crate::window::MAX_PLAUSIBLE_WINDOW)
        .contains(&tokens)
        .then_some(tokens)
}


/// A failure a provider reported inside an otherwise successful answer.
///
/// Shared by both modes: a stream carries it in a frame, a buffered answer
/// carries it as the whole body, and the retry decision cannot be allowed to
/// depend on which one the route happens to use.
pub fn error_request(error: &Value) -> RequestError {
    let message = error
        .get("message")
        .and_then(|m| m.as_str())
        .unwrap_or("provider reported an error mid-stream")
        .to_string();

    // Providers label the same failure three ways: `{"code": 429}`,
    // `{"code": "429"}` (gateways behind a JSON string), or a name such as
    // `{"type": "rate_limit_error"}`. Reading only the number loses the first
    // case to the message heuristic and every name entirely.
    let status = error
        .get("code")
        .or_else(|| error.get("status"))
        .or_else(|| error.get("type"))
        .and_then(status_of);

    // Everything the error says about itself, lower-cased: the type, the raw
    // code, and the message. Classification has to read the words, because
    // that is all a provider that never sends a number gives us.
    let words = ["type", "code", "message", "status"]
        .iter()
        .filter_map(|key| error.get(*key))
        .filter_map(|value| value.as_str())
        .collect::<Vec<&str>>()
        .join(" ")
        .to_lowercase();

    let retryable = match status {
        Some(code) => retryable_status(code),
        // With no status to go on, the word list decides — and the default is
        // "do not retry". It used to be the opposite (anything that did not
        // read as one of the fatal phrases was retried), which turned a
        // provider's invalid parameter into three more identical round trips
        // and, on a metered route, three more bills.
        None => !looks_fatal(&words) && says_retryable(&words),
    };
    RequestError {
        status,
        retryable,
        message,
    }
}

/// The HTTP status an error field carries, however it is spelled.
///
/// A number is the normal form; a quoted number (`"429"`) is what a gateway
/// built on stringly-typed errors sends; anything else — `rate_limit_error`,
/// `overloaded` — is a name, answered by [`says_retryable`] instead.
pub fn status_of(value: &Value) -> Option<u16> {
    let code = match value {
        Value::Number(number) => number.as_u64(),
        Value::String(text) => text.trim().parse::<u64>().ok(),
        _ => None,
    }?;
    // A "status" that is not a status (a name, a 0, a 200) is not usable:
    // treating 200 as a retryable-worthy code would be worse than ignoring it.
    (100..=599).contains(&code).then_some(code as u16)
}

/// The retryable families a provider names in words instead of numbers.
///
/// Anthropic types the error `rate_limit_error` or `overloaded_error`; OpenAI
/// uses `rate_limit_exceeded`; the gateways in between invent `server_error`,
/// `upstream_error`, `capacity_error` and plain "please try again". Matching is
/// a lowercase substring because the spellings multiply and an exact list ages
/// badly — and because a false negative here costs one failed turn, while a
/// false positive costs a retry loop.
pub fn says_retryable(words: &str) -> bool {
    const WORDS: [&str; 18] = [
        "rate_limit",
        "rate limit",
        "ratelimit",
        "too many requests",
        "overload",
        "server_error",
        "server error",
        "internal error",
        "internal_error",
        "bad gateway",
        "service unavailable",
        "gateway timeout",
        "timed out",
        "temporarily",
        "try again",
        "capacity",
        "upstream",
        "busy",
    ];
    WORDS.iter().any(|word| words.contains(word))
}

/// The context window a provider stated in an error message, when it stated one.
///
/// An error that rejects an oversized prompt is the provider stating the limit
/// it actually enforces, and for the route in front of us that is the most
/// trustworthy source there is: the built-in table is a guess about a model id,
/// while this is a statement about this deployment. Real messages read like
///
/// * `This model's maximum context length is 8192 tokens`
/// * `context length is 128000`
/// * `maximum context window: 65536`
/// * `max_model_len=32768`
///
/// So the rule is a phrase about a context length or window followed, within a
/// few words, by a number — not "the first large number in the message". The
/// same message usually also names the prompt size that was rejected, and
/// measuring against that number would compact history down to a size the
/// provider refuses just as firmly.
pub fn reported_window(message: &str) -> Option<usize> {
    if message.trim().is_empty() {
        return None;
    }
    // ASCII-lowercased rather than `to_lowercase`: the byte offsets found in the
    // lowered copy are used to slice that same copy, and a full lowercase can
    // change the byte length of a non-ASCII character, which would shift every
    // offset after it and could cut a character in half.
    let lowered = message.to_ascii_lowercase();

    // The earliest phrase that is followed by a number answers the question,
    // because that is the first place in the message that states a window; a
    // later mention is a restatement or a different limit.
    let mut found: Option<(usize, usize)> = None;
    for &phrase in CONTEXT_WINDOW_PHRASES {
        let mut from = 0;
        while let Some(offset) = lowered[from..].find(phrase) {
            let at = from + offset;
            // Resume past this occurrence, so a phrase that is not followed by a
            // number does not stop the search for a later one.
            from = at + phrase.len();
            if let Some(tokens) = number_near(&lowered[from..]) {
                if found.map_or(true, |(earliest, _)| at < earliest) {
                    found = Some((at, tokens));
                }
            }
        }
    }
    found.map(|(_, tokens)| tokens)
}

/// The words providers use to introduce a context window in an error message.
///
/// Both spacings are listed because both appear on the wire: Chat Completions
/// writes prose ("maximum context length is …"), while vLLM and the servers
/// around it name their own field ("max_model_len=…"). Everything is lowercase
/// because it is matched against a lowercased message.
const CONTEXT_WINDOW_PHRASES: &[&str] = &[
    "context length",
    "context_length",
    "context window",
    "context_window",
    "context limit",
    "context_limit",
    "max_model_len",
];

/// The first plausible window written within a few words after a phrase.
///
/// The number follows the phrase but rarely immediately: a verb, a colon, or an
/// `=` sits in between ("is 8192", ": 65536", "=32768"). Four words covers every
/// shape seen in the wild and is short enough that a number belonging to the
/// next clause — the rejected prompt's `tokens: 123456` — is left to that clause
/// instead of being read as the limit.
pub fn number_near(tail: &str) -> Option<usize> {
    const NEARBY_WORDS: usize = 4;
    let plausible =
        crate::window::MIN_PLAUSIBLE_WINDOW..=crate::window::MAX_PLAUSIBLE_WINDOW;
    tail.split_whitespace()
        .take(NEARBY_WORDS)
        .filter_map(number_in_word)
        .find(|tokens| plausible.contains(tokens))
}

/// The number one word holds, if it is a plain count with optional separators.
///
/// Surrounding punctuation is trimmed because the number sits inside prose and
/// inside `key=value`: `(8192`, `8192,`, `:65536` and `=32768` all mean the same
/// number. A word that still holds anything except digits and underscores after
/// that trimming — `4.5`, `1.2e6`, `v1.8192` — is refused rather than parsed
/// loosely, because a decimal, an exponent, or a version is not a window.
pub fn number_in_word(word: &str) -> Option<usize> {
    let digits = word
        .trim_matches(|character: char| !character.is_ascii_digit() && character != '_');
    let plain = !digits.is_empty()
        && digits
            .chars()
            .all(|character| character.is_ascii_digit() || character == '_');
    if !plain {
        return None;
    }
    // Underscores are the thousands separators a provider writes when it prints
    // a number it read from its own configuration (`max_model_len=32_768`).
    digits.replace('_', "").parse::<usize>().ok()
}

/// Read token accounting, tolerating every spelling in the wild.
///
/// There is no single field name for cache hits: OpenAI reports
/// `prompt_tokens_details.cached_tokens`, DeepSeek and SiliconFlow report
/// `prompt_cache_hit_tokens`, Together has a flat `cached_tokens`, xAI nests it
/// under `input_tokens_details`, and Anthropic-shaped gateways use
/// `cache_read_input_tokens`. Reading only one spelling silently reports zero,
/// with no error to notice.
///
/// `prompt_tokens` is inclusive of cached tokens on every provider checked, so
/// `input` is reported as-is rather than having the cached count subtracted.
///
/// A cache *write* is a different number from a cache read and is billed
/// differently: Anthropic reports `cache_creation_input_tokens` and prices it
/// above a normal prompt token, so a turn that paid for one and recorded
/// nothing was also the most expensive turn in the accounting.
pub fn extract_usage(usage: &Value) -> TokenUsage {
    let input = first_count(usage, &[&["prompt_tokens"][..], &["input_tokens"][..]]).unwrap_or(0);
    let output = first_count(usage, &[&["completion_tokens"][..], &["output_tokens"][..]])
        .unwrap_or(0);

    let cached = first_count(
        usage,
        &[
            &["prompt_tokens_details", "cached_tokens"][..],
            &["input_tokens_details", "cached_tokens"][..],
            &["prompt_cache_hit_tokens"][..],
            &["cached_tokens"][..],
            &["cache_read_input_tokens"][..],
        ],
    )
    .unwrap_or(0);

    let cache_created = first_count(
        usage,
        &[
            &["cache_creation_input_tokens"][..],
            &["cache_creation_tokens"][..],
        ],
    )
    .unwrap_or(0);

    let reasoning = first_count(
        usage,
        &[
            &["completion_tokens_details", "reasoning_tokens"][..],
            &["output_tokens_details", "reasoning_tokens"][..],
        ],
    )
    .unwrap_or(0);

    TokenUsage { input, output, cached, cache_created, reasoning }
}

/// The count a single usage field holds, however the gateway wrote it.
///
/// The normal form is a JSON integer, but the same API is fronted by gateways
/// that re-encode numbers: one sends `"123"` because its own struct declared a
/// string, another sends `0.0` because an upstream database column is a float.
/// Both are the number they look like, so both are read. Anything else is
/// reported as absent rather than as zero, because a count of zero and a count
/// nobody could read are different facts and a provider sending nonsense must
/// not look like a provider sending nothing.
pub fn count_of(value: &Value) -> Option<u64> {
    match value {
        Value::Number(number) => number.as_u64().or_else(|| {
            // A float only counts if it is a whole, non-negative one: `45.0` is
            // the integer a float column could not keep, while `-1` and `4.5`
            // are not token counts and rounding them would invent a number.
            number
                .as_f64()
                .filter(|float| float.is_finite() && *float >= 0.0 && float.fract() == 0.0)
                .map(|float| float as u64)
        }),
        Value::String(text) => text.trim().parse::<u64>().ok(),
        _ => None,
    }
}

/// Walk key paths and return the first one that holds a readable count.
pub fn first_count(value: &Value, paths: &[&[&str]]) -> Option<u64> {
    paths.iter().find_map(|path| count_of(at_path(value, *path)?))
}

/// Follow a key path, answering only when every key on it exists.
pub fn at_path<'a>(value: &'a Value, path: &[&str]) -> Option<&'a Value> {
    let mut current = value;
    for key in path {
        current = current.get(*key)?;
    }
    Some(current)
}

pub fn truncate(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_string();
    }
    let mut out: String = text.chars().take(max).collect();
    out.push_str("… (truncated)");
    out
}
