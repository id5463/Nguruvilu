//! Reading one `SKILL.md`: its frontmatter, and the id it gets.
//!
//! The frontmatter is a small flat `key: value` block between two `---` lines.
//! No YAML parser: the two shapes real skills use beyond flat pairs — block
//! scalars and quoted values — are handled explicitly, and a full parser would
//! be more machinery than the format deserves.
//!
//! De-duplication lives here too: two roots may offer the same id, and neither
//! copy may be discarded (see [`unique_id`]).

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result};

use super::model::Skill;

/// Parse one `SKILL.md`.
pub(super) fn parse_skill(manifest: &Path, dir: &Path, root: &Path) -> Result<Skill> {
    let raw = std::fs::read_to_string(manifest)
        .with_context(|| format!("reading {}", manifest.display()))?;
    let (fields, body) = split_frontmatter(&raw);

    let fallback_id = dir
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "skill".into());

    let id = fields
        .get("name")
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or(fallback_id);

    let description = fields
        .get("description")
        .map(|s| s.trim().to_string())
        .unwrap_or_default();

    // `triggers:` may be a comma-separated string or a YAML list. The flat
    // frontmatter reader only sees `key: value`, so a list arrives as
    // "- one\n- two" in the value; both forms are split here rather than in
    // the matcher, so every reader agrees on what the words are.
    let triggers = fields
        .get("triggers")
        .map(|raw| {
            raw.split(['\n', ','])
                .map(|t| t.trim().trim_start_matches('-').trim().trim_matches('"').trim_matches('\''))
                .filter(|t| !t.is_empty())
                .map(str::to_string)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    Ok(Skill {
        name: id.clone(),
        id,
        description,
        path: manifest.to_path_buf(),
        dir: dir.to_path_buf(),
        body: body.trim().to_string(),
        source: root.to_path_buf(),
        triggers,
    })
}

/// Split a `---` frontmatter block from the body.
///
/// Only flat `key: value` pairs are read; skills do not need a full YAML
/// parser and pulling one in would be more machinery than the format deserves.
/// The two shapes real skills use beyond flat pairs — block scalars and
/// quoted values — are handled explicitly (see inside).
///
/// The walk runs over the **original** text and advances by the bytes it really
/// consumed. `str::lines()` is deliberately not used for offsets: it strips the
/// `\r` of a CRLF file while a caller has to guess the newline width, so every
/// frontmatter line in a Windows `SKILL.md` (CRLF is the default there) shifted
/// the body one byte early. The body then began inside the closing `---`, and
/// when the drift landed mid-character `str::get` returned `None` — which the
/// old `.unwrap_or("")` turned into a **silently empty body**. A skill that
/// loads with no instructions is the worst of the three outcomes.
pub(super) fn split_frontmatter(raw: &str) -> (BTreeMap<String, String>, String) {
    let mut fields = BTreeMap::new();
    let trimmed = raw.strip_prefix('\u{feff}').unwrap_or(raw);

    // The opening delimiter has to be the first line, exactly `---`.
    let Some(first_newline) = trimmed.find('\n') else {
        // No newline at all: nothing can be frontmatter, so all of it is body.
        return (fields, trimmed.to_string());
    };
    if trimmed[..first_newline].trim() != "---" {
        return (fields, trimmed.to_string());
    }

    // A YAML block scalar (`>` folded or `|` literal, optionally chomped with
    // -/+) keeps its value on the following indented lines. Treating those as
    // pairs would silently drop a long description and smuggle its sentences
    // in as fake keys — Claude Code skills use this shape routinely, so
    // reading their frontmatter needs it.
    let mut block: Option<(String, bool)> = None; // (key, folded?)
    let mut cursor = first_newline + 1;
    while cursor < trimmed.len() {
        let rest = &trimmed[cursor..];
        // Byte-exact: the line, plus whatever newline actually ended it.
        let (line, next) = match rest.find('\n') {
            Some(newline) => (&rest[..newline], cursor + newline + 1),
            None => (rest, trimmed.len()),
        };
        let line = line.strip_suffix('\r').unwrap_or(line);
        cursor = next;

        if line.trim() == "---" {
            // A slice of the original text: it cannot drift, and it cannot be
            // cut mid-character.
            return (fields, trimmed[cursor..].to_string());
        }

        if let Some((key, folded)) = block.clone() {
            let indented = line.starts_with([' ', '\t']) || line.trim().is_empty();
            if indented {
                let text = line.trim();
                if !text.is_empty() {
                    let slot = fields.entry(key).or_default();
                    if folded {
                        if !slot.is_empty() {
                            slot.push(' ');
                        }
                    } else if !slot.is_empty() {
                        slot.push('\n');
                    }
                    slot.push_str(text);
                }
                continue;
            }
            // Not indented any more: the block ended.
            block = None;
        }

        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        let key = key.trim().to_ascii_lowercase();
        let value = value.trim();
        // `>`, `|-`, `|+` … — an indicator, never a value.
        let scalar_shape = !value.is_empty()
            && matches!(value.as_bytes()[0], b'>' | b'|')
            && value.bytes().all(|b| matches!(b, b'>' | b'|' | b'-' | b'+'));
        if scalar_shape {
            fields.entry(key.clone()).or_default();
            block = Some((key, value.as_bytes()[0] == b'>'));
            continue;
        }
        fields.insert(key, unquote(value).to_string());
    }

    // Unterminated frontmatter: treat the whole file as body.
    (BTreeMap::new(), trimmed.to_string())
}

/// A key that is not taken yet: `id`, else `id-2`, `id-3`, …
///
/// Used when two roots offer the same skill id. Neither copy is discarded: a
/// workspace that happens to check in `.nguruvilu/skills/pdf` must not be able
/// to make the user's own `pdf` skill disappear from the catalog.
pub(super) fn unique_id(wanted: &str, taken: &BTreeMap<String, Skill>) -> String {
    if !taken.contains_key(wanted) {
        return wanted.to_string();
    }
    for suffix in 2.. {
        let candidate = format!("{wanted}-{suffix}");
        if !taken.contains_key(&candidate) {
            return candidate;
        }
    }
    unreachable!("a free suffix always exists")
}

/// Strip one matching pair of surrounding quotes.
///
/// Quoted values are how a description carries a colon without the flat
/// parser splitting on it, so the quotes have to go after the split.
fn unquote(value: &str) -> &str {
    let bytes = value.as_bytes();
    if value.len() >= 2 {
        let (first, last) = (bytes[0], bytes[value.len() - 1]);
        if (first == b'"' && last == b'"') || (first == b'\'' && last == b'\'') {
            return &value[1..value.len() - 1];
        }
    }
    value
}
