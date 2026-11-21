//! Human-sized numbers: `128K`, `1M`, `200000`.
//!
//! A token count is a number nobody holds in their head. Writing `128K` and
//! having it mean 131072 is the difference between a setting someone can change
//! and one they leave alone because they are not sure what 131072 should be.
//!
//! Parsing is deliberately narrow: an optional decimal number followed by an
//! optional single-letter suffix. It accepts what a person would type and
//! refuses everything else, rather than guessing at `1.5e6` or `1,000`.

use anyhow::{anyhow, Result};

/// Parse a token count written the way a person writes one.
///
/// Accepts `200000`, `200k`, `128K`, `1M`, `1m`, `2g`, and a decimal such as
/// `1.5M`. Whitespace is ignored. Underscores are allowed as separators
/// (`128_000`), because they are how a long number stays readable.
pub fn parse_size(raw: &str) -> Result<usize> {
    let text = raw.trim().replace('_', "").replace(' ', "");
    if text.is_empty() {
        return Err(anyhow!("empty size; write it as 128K, 1M, or 200000"));
    }

    // Split the numeric part from the suffix.
    let split = text
        .find(|c: char| !c.is_ascii_digit() && c != '.')
        .unwrap_or(text.len());
    let (number, suffix) = text.split_at(split);

    if number.is_empty() {
        return Err(anyhow!(
            "'{raw}' has no number; write it as 128K, 1M, or 200000"
        ));
    }

    let value: f64 = number
        .parse()
        .map_err(|_| anyhow!("'{raw}' is not a number; write it as 128K, 1M, or 200000"))?;
    if !value.is_finite() || value < 0.0 {
        return Err(anyhow!("'{raw}' is not a usable size"));
    }

    let multiplier: f64 = match suffix.trim().to_ascii_lowercase().as_str() {
        "" => 1.0,
        "k" => 1024.0,
        "m" => 1024.0 * 1024.0,
        "g" => 1024.0 * 1024.0 * 1024.0,
        other => {
            return Err(anyhow!(
                "unknown suffix '{other}' in '{raw}'; use K, M, or G (or no suffix for tokens)"
            ))
        }
    };

    let tokens = value * multiplier;
    if tokens > usize::MAX as f64 {
        return Err(anyhow!("'{raw}' is larger than this machine can count"));
    }
    Ok(tokens.round() as usize)
}

/// Write a token count the way a person reads one: `128K`, `1M`, `1500`.
///
/// **Lossless.** A suffix is used only when the value divides evenly by it, so
/// parsing the result returns exactly what was formatted. A friendlier `1.4K`
/// for 1500 would read back as 1434, and this value travels through a settings
/// field the user may save without touching — silently changing the number they
/// configured is worse than showing one they find slightly less pretty.
pub fn format_size(tokens: usize) -> String {
    const K: usize = 1024;
    const M: usize = 1024 * 1024;
    const G: usize = 1024 * 1024 * 1024;

    let (divisor, suffix) = if tokens >= G {
        (G, "G")
    } else if tokens >= M {
        (M, "M")
    } else if tokens >= K {
        (K, "K")
    } else {
        return tokens.to_string();
    };

    if tokens % divisor == 0 {
        return format!("{}{suffix}", tokens / divisor);
    }

    // A single decimal is allowed when it is exact, so 1572864 shows as 1.5M
    // and still parses back to 1572864. Anything coarser is left as the plain
    // number: a rounded rendering would read back as a different value.
    let tenths = tokens.saturating_mul(10);
    if tenths % divisor == 0 {
        let scaled = tenths / divisor;
        return format!("{}.{}{suffix}", scaled / 10, scaled % 10);
    }

    tokens.to_string()
}

