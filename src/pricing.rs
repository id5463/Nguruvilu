//! What recorded tokens cost, in money.
//!
//! The ledger counts tokens because tokens can be counted exactly. Money is the
//! reason anyone looks at those counts, and money is the part the kernel cannot
//! know: every provider, plan and region prices a token differently, and a
//! built-in table would be wrong the week after it was written. So rates come
//! from a file the user writes, and a model with no rate is reported as
//! *unpriced* rather than guessed at. A guessed price is worse than no price:
//! it is a number somebody would put in an invoice.
//!
//! The file is JSON, rates are per 1,000,000 tokens, and `*` is the fallback:
//!
//! ```json
//! {
//!   "tierflow": { "input": 0.55, "output": 2.20, "cache_read": 0.055, "cache_write": 0.69 },
//!   "*":        { "input": 1.00, "output": 4.00 }
//! }
//! ```

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::session::{SessionTotals, TurnStats};

/// The key that answers for every model the file does not name.
pub const ANY_MODEL: &str = "*";

/// What the rates are quoted against.
const PER_TOKENS: f64 = 1_000_000.0;

/// Token counts a rate applies to.
///
/// Four numbers rather than the ledger's whole row, because a price only cares
/// about these: steps and milliseconds cost nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Tokens {
    /// Prompt tokens, provider semantics — cache reads and writes included.
    pub input: u64,
    /// Completion tokens.
    pub output: u64,
    /// Prompt tokens the provider served from its cache.
    pub cached: u64,
    /// Prompt tokens the provider wrote into its cache.
    pub cache_created: u64,
}

impl Tokens {
    /// The countable part of a session's totals.
    pub fn of_totals(totals: &SessionTotals) -> Self {
        Self {
            input: totals.input,
            output: totals.output,
            cached: totals.cached,
            cache_created: totals.cache_created,
        }
    }

    /// The countable part of one recorded turn.
    pub fn of_turn(stats: &TurnStats) -> Self {
        Self {
            input: stats.input,
            output: stats.output,
            cached: stats.cached,
            cache_created: stats.cache_created,
        }
    }
}

/// One model's rates, per 1,000,000 tokens.
///
/// Only `input` is required: a price list that does not price the prompt is not
/// a price list. The other three fall back to the input rate, which is the
/// conservative reading — the alternative (treating an unstated cache rate as
/// free) would under-report the bill, and under-reporting a bill is the one
/// error nobody forgives.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize, Serialize)]
pub struct Rate {
    /// Prompt tokens the provider had to compute.
    pub input: f64,
    /// Completion tokens; defaults to the input rate.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<f64>,
    /// Prompt tokens served from cache; defaults to the input rate.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_read: Option<f64>,
    /// Prompt tokens written into cache; defaults to the input rate.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_write: Option<f64>,
}

impl Rate {
    /// The rate for completion tokens.
    pub fn output_rate(&self) -> f64 {
        self.output.unwrap_or(self.input)
    }

    /// The rate for prompt tokens served from cache.
    pub fn cache_read_rate(&self) -> f64 {
        self.cache_read.unwrap_or(self.input)
    }

    /// The rate for prompt tokens written into cache.
    pub fn cache_write_rate(&self) -> f64 {
        self.cache_write.unwrap_or(self.input)
    }
}

/// Every rate a price file names.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PriceTable {
    models: BTreeMap<String, Rate>,
}

impl PriceTable {
    /// Read a price file's text.
    ///
    /// A negative or non-finite rate is refused: `NaN` would poison every sum it
    /// touches, and a negative price is either a typo or a refund, and a refund
    /// is not something a usage report should invent.
    pub fn parse(text: &str) -> Result<Self> {
        let models: BTreeMap<String, Rate> =
            serde_json::from_str(text).context("parsing the price table")?;
        for (model, rate) in &models {
            for (name, value) in [
                ("input", Some(rate.input)),
                ("output", rate.output),
                ("cache_read", rate.cache_read),
                ("cache_write", rate.cache_write),
            ] {
                if let Some(value) = value {
                    if !value.is_finite() || value < 0.0 {
                        anyhow::bail!(
                            "price table: {name} for '{model}' is {value}, which is not a rate"
                        );
                    }
                }
            }
        }
        Ok(Self { models })
    }

    /// Read a price file, or `None` when there is no file.
    ///
    /// A file that exists but cannot be read or parsed is an error: silence
    /// there would turn "I misconfigured my prices" into "these turns cost
    /// nothing", and the second answer is indistinguishable from the truth.
    pub fn from_path(path: &Path) -> Result<Option<Self>> {
        match std::fs::read_to_string(path) {
            Ok(text) => Ok(Some(
                Self::parse(&text).with_context(|| format!("reading {}", path.display()))?,
            )),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(error).with_context(|| format!("reading {}", path.display())),
        }
    }

    /// Where the price file is looked for when nothing says otherwise.
    ///
    /// Beside the settings file, so one directory holds everything a machine
    /// has configured.
    pub fn default_path() -> PathBuf {
        if let Ok(home) = std::env::var("NGU_HOME") {
            if !home.trim().is_empty() {
                return PathBuf::from(home).join("prices.json");
            }
        }
        match crate::settings::Settings::path().parent() {
            Some(dir) => dir.join("prices.json"),
            None => crate::settings::home_dir()
                .join(".nguruvilu")
                .join("prices.json"),
        }
    }

    /// The rate for a model: its own entry first, then the fallback.
    pub fn rate_for(&self, model: &str) -> Option<Rate> {
        self.models
            .get(model)
            .copied()
            .or_else(|| self.models.get(ANY_MODEL).copied())
    }

    /// Whether the file named anything at all.
    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }

    /// The model ids the file prices, fallback excluded.
    pub fn models(&self) -> Vec<&str> {
        self.models
            .keys()
            .map(String::as_str)
            .filter(|model| *model != ANY_MODEL)
            .collect()
    }
}

/// What one set of token counts cost under one rate.
#[derive(Debug, Clone, PartialEq)]
pub struct Cost {
    /// Money, in whatever unit the rates were written in.
    pub amount: f64,
    /// The model the rate came from — a fallback rate reports the model it was
    /// applied to, not `*`, because that is what the reader is asking about.
    pub model: String,
    /// Whether a rate was found. False means the amount is 0 because nothing
    /// was priced, not because nothing was spent.
    pub priced: bool,
    /// Why the tokens are unpriced, when they are.
    pub reason: Option<String>,
    /// Prompt tokens charged at the input rate.
    pub fresh_input: u64,
    /// Prompt tokens charged at the cache-read rate.
    pub cached: u64,
    /// Prompt tokens charged at the cache-write rate.
    pub cache_created: u64,
    /// Completion tokens.
    pub output: u64,
}

impl Cost {
    /// The cost as the JSON a script reads.
    ///
    /// `priced: false` with a reason rather than an absent field: a script that
    /// sums `amount` over sessions must be able to tell "nothing recorded" from
    /// "nobody priced this".
    pub fn json(&self) -> serde_json::Value {
        let mut value = serde_json::json!({
            "amount": self.amount,
            "priced": self.priced,
            "model": self.model,
            "fresh_input": self.fresh_input,
            "cached": self.cached,
            "cache_created": self.cache_created,
            "output": self.output,
        });
        if let Some(reason) = &self.reason {
            value["reason"] = serde_json::Value::String(reason.clone());
        }
        value
    }
}

/// Split the prompt tokens into the three buckets a bill distinguishes.
///
/// `input` counts the whole prompt under provider semantics, cache reads and
/// writes included, so both cached parts come out before the rest is charged at
/// the input rate — otherwise a cache hit would be paid for twice.
fn split(tokens: Tokens) -> (u64, u64, u64) {
    let cached = tokens.cached.min(tokens.input);
    let written = tokens.cache_created.min(tokens.input - cached);
    (tokens.input - cached - written, cached, written)
}

/// Price one set of token counts.
///
/// `table` is `None` when the machine has no price file, which is the ordinary
/// case: the report still prints, with the amount left unstated.
pub fn cost(table: Option<&PriceTable>, model: &str, tokens: Tokens) -> Cost {
    let (fresh_input, cached, cache_created) = split(tokens);
    let mut cost = Cost {
        amount: 0.0,
        model: model.to_string(),
        priced: false,
        reason: None,
        fresh_input,
        cached,
        cache_created,
        output: tokens.output,
    };

    let Some(table) = table else {
        cost.reason = Some(format!(
            "no price table: write {} to see money",
            PriceTable::default_path().display()
        ));
        return cost;
    };
    let Some(rate) = table.rate_for(model) else {
        cost.reason = Some(format!("no rate for model '{model}'"));
        return cost;
    };

    cost.amount = (fresh_input as f64 * rate.input
        + cached as f64 * rate.cache_read_rate()
        + cache_created as f64 * rate.cache_write_rate()
        + tokens.output as f64 * rate.output_rate())
        / PER_TOKENS;
    cost.priced = true;
    cost
}

/// Price a set of sessions together.
///
/// The amounts add up, but `priced` is only true when *every* part was priced:
/// a grand total that silently left out the sessions nobody priced would be a
/// smaller number presented as the whole bill. The token counts are summed
/// regardless, so the report still says how much was spent even when it cannot
/// say what it cost.
pub fn total(costs: impl IntoIterator<Item = Cost>) -> Cost {
    let mut total = Cost {
        amount: 0.0,
        model: String::new(),
        priced: true,
        reason: None,
        fresh_input: 0,
        cached: 0,
        cache_created: 0,
        output: 0,
    };
    let mut seen = 0usize;
    let mut unpriced = 0usize;
    let mut last_reason = None;
    for cost in costs {
        seen += 1;
        total.fresh_input += cost.fresh_input;
        total.cached += cost.cached;
        total.cache_created += cost.cache_created;
        total.output += cost.output;
        if cost.priced {
            total.amount += cost.amount;
        } else {
            unpriced += 1;
            last_reason = cost.reason.clone().or(last_reason);
        }
    }
    if seen == 0 {
        // Nothing to price is not an unpriced bill: there is simply no work.
        return total;
    }
    if unpriced > 0 {
        total.priced = false;
        total.reason = Some(match (unpriced, last_reason) {
            (count, Some(reason)) if count == seen => reason,
            (count, _) => format!("{count} of {seen} part(s) unpriced"),
        });
    }
    total
}
