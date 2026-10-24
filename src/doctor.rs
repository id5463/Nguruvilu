//! `ngu doctor` — one read-only pass over everything this machine is set up to do.
//!
//! The pieces of this program live in separate places: settings in one JSON
//! file, sessions in another directory, packs, a ledger, skills under several
//! roots, prices in a file that may not exist. Each piece already reports its own
//! problems when it is used, which means a user only learns about the ones they
//! happened to exercise — and the failure they hit first is usually a *load*
//! failure, where nothing downstream can say anything at all.
//!
//! This module asks every part how it is doing without touching the network and
//! without writing anything. It answers three levels rather than pass/fail,
//! because "there is no price file" and "the settings file does not parse" are
//! not the same kind of news.

use std::path::{Path, PathBuf};

use anyhow::Result;
use serde_json::Value;

use crate::ledger::{default_ledger_path, Ledger};
use crate::pack;
use crate::pricing::PriceTable;
use crate::session::{JsonlStore, SessionStore};
use crate::settings::{Settings, DEFAULT_MODEL};
use crate::skills::SkillRegistry;

/// How bad one finding is.
///
/// Ordered, so [`Level::max`] folds a report into its worst finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Level {
    /// Working as configured.
    Ok,
    /// Worth knowing: a default is in force, a file is absent, or something was
    /// quietly skipped. Not necessarily wrong.
    Warn,
    /// Broken: this part will not answer, and whatever depends on it will fail.
    Fail,
}

impl Level {
    /// How the level is spelled in output.
    pub fn as_str(&self) -> &'static str {
        match self {
            Level::Ok => "ok",
            Level::Warn => "warning",
            Level::Fail => "FAIL",
        }
    }
}

/// One finding.
#[derive(Debug, Clone, PartialEq)]
pub struct Check {
    /// What was examined: `settings`, `sessions`, `skills`, and so on.
    pub name: &'static str,
    /// How it came out.
    pub level: Level,
    /// What was found, in the words a user can act on.
    pub detail: String,
    /// What to do about it, when there is something to do.
    pub hint: Option<String>,
}

/// Everything one pass found.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Report {
    /// Findings, in the order they were examined.
    pub checks: Vec<Check>,
}

impl Report {
    fn add(&mut self, level: Level, name: &'static str, detail: impl Into<String>) {
        self.checks.push(Check {
            name,
            level,
            detail: detail.into(),
            hint: None,
        });
    }

    fn add_with_hint(
        &mut self,
        level: Level,
        name: &'static str,
        detail: impl Into<String>,
        hint: impl Into<String>,
    ) {
        self.checks.push(Check {
            name,
            level,
            detail: detail.into(),
            hint: Some(hint.into()),
        });
    }

    /// The worst level any finding reached.
    ///
    /// An empty report is `Ok`: nothing examined is not a failure.
    pub fn worst(&self) -> Level {
        self.checks
            .iter()
            .map(|check| check.level)
            .max()
            .unwrap_or(Level::Ok)
    }

    /// How many findings reached `level`.
    pub fn count(&self, level: Level) -> usize {
        self.checks
            .iter()
            .filter(|check| check.level == level)
            .count()
    }

    /// The one-line verdict.
    pub fn summary(&self) -> String {
        format!(
            "doctor: {} ok, {} warning(s), {} failure(s)",
            self.count(Level::Ok),
            self.count(Level::Warn),
            self.count(Level::Fail)
        )
    }

    /// The report as one line per finding, then the verdict.
    pub fn lines(&self) -> Vec<String> {
        let mut lines: Vec<String> = self
            .checks
            .iter()
            .map(|check| {
                let mut line = format!("[{}] {}: {}", check.level.as_str(), check.name, check.detail);
                if let Some(hint) = &check.hint {
                    line.push_str(&format!("\n        hint: {hint}"));
                }
                line
            })
            .collect();
        lines.push(self.summary());
        lines
    }

    /// The report as the JSON a script reads.
    pub fn json(&self) -> Value {
        let checks: Vec<Value> = self
            .checks
            .iter()
            .map(|check| {
                let mut entry = serde_json::json!({
                    "name": check.name,
                    "level": check.level.as_str(),
                    "detail": check.detail,
                });
                if let Some(hint) = &check.hint {
                    entry["hint"] = Value::String(hint.clone());
                }
                entry
            })
            .collect();
        serde_json::json!({
            "checks": checks,
            "ok": self.count(Level::Ok),
            "warnings": self.count(Level::Warn),
            "failures": self.count(Level::Fail),
            "worst": self.worst().as_str(),
        })
    }
}

/// Where everything lives, and what else to look at.
///
/// Explicit rather than read from the environment on each check, so a test can
/// point the whole examination at a temporary directory instead of mutating
/// process-wide variables that other tests are reading at the same time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Options {
    /// Settings file.
    pub settings: PathBuf,
    /// Session store root: the directory holding `<id>.jsonl` files.
    pub sessions: PathBuf,
    /// Installed packs.
    pub packs: PathBuf,
    /// Install ledger.
    pub ledger: PathBuf,
    /// Price table.
    pub prices: PathBuf,
    /// Extra skill roots, before the built-in ones.
    pub skill_dirs: Vec<PathBuf>,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            settings: Settings::path(),
            sessions: JsonlStore::default_root(),
            packs: pack::default_packs_dir(),
            ledger: default_ledger_path(),
            prices: PriceTable::default_path(),
            skill_dirs: Vec::new(),
        }
    }
}

impl Options {
    /// The default places, plus extra skill roots.
    pub fn with_skill_dirs(skill_dirs: impl IntoIterator<Item = PathBuf>) -> Self {
        Self {
            skill_dirs: skill_dirs.into_iter().collect(),
            ..Self::default()
        }
    }
}

/// Examine everything `options` names.
///
/// Every check is independent: one broken part must not hide the state of the
/// rest, which is the whole reason for running this instead of waiting for the
/// next thing to fail.
pub fn run(options: &Options) -> Report {
    let mut report = Report::default();
    check_settings(options, &mut report);
    check_sessions(options, &mut report);
    check_skills(options, &mut report);
    check_packs(options, &mut report);
    check_ledger(options, &mut report);
    check_prices(options, &mut report);
    report
}

fn check_settings(options: &Options, report: &mut Report) {
    let settings = match Settings::load_from(&options.settings) {
        Ok(settings) => settings,
        Err(error) => {
            report.add_with_hint(
                Level::Fail,
                "settings",
                format!("{error:#}"),
                "fix the JSON, or move the file aside; nothing is written over a file that does not parse",
            );
            return;
        }
    };
    if !options.settings.is_file() {
        report.add_with_hint(
            Level::Warn,
            "settings",
            format!(
                "no settings file at {}; defaults are in force",
                options.settings.display()
            ),
            "run `ngu config set base_url <url>` (and `api_key`) to write one",
        );
        return;
    }

    // Never the key itself: a diagnostic is meant to be pasted into a chat.
    let route = if settings.base_url.trim().is_empty() {
        "no base_url".to_string()
    } else {
        settings.base_url.trim().to_string()
    };
    let key = if settings.api_key.trim().is_empty() {
        "no api_key"
    } else {
        "api_key set"
    };
    let model = if settings.model.trim().is_empty() {
        format!("default model {DEFAULT_MODEL}")
    } else {
        settings.model.clone()
    };
    let detail = format!("{route}  {model}  {key}");
    if settings.base_url.trim().is_empty() || settings.api_key.trim().is_empty() {
        report.add_with_hint(
            Level::Warn,
            "settings",
            detail,
            "a request needs both a base_url and an api_key; environment variables are also read at run time, and this check reads the file only",
        );
    } else {
        report.add(Level::Ok, "settings", detail);
    }
}

fn check_sessions(options: &Options, report: &mut Report) {
    let store = match JsonlStore::open(&options.sessions) {
        Ok(store) => store,
        Err(error) => {
            report.add_with_hint(
                Level::Fail,
                "sessions",
                format!("{error:#}"),
                "the session directory must be writable; it is where every turn is recorded",
            );
            return;
        }
    };
    let sessions = match store.list() {
        Ok(sessions) => sessions,
        Err(error) => {
            report.add(Level::Fail, "sessions", format!("{error:#}"));
            return;
        }
    };
    if sessions.is_empty() {
        report.add(
            Level::Ok,
            "sessions",
            format!("no sessions yet in {}", options.sessions.display()),
        );
        return;
    }

    // The newest session is the one a user is most likely to open, so it is the
    // one worth reading: a record it cannot make sense of will look like lost
    // history the moment it is opened.
    let newest = sessions.first().map(|summary| summary.id.clone());
    let mut detail = format!(
        "{} session(s) in {}",
        sessions.len(),
        options.sessions.display()
    );
    let mut level = Level::Ok;
    let mut hint = None;
    if let Some(id) = newest {
        match store.load(&id) {
            Err(error) => {
                return report.add_with_hint(
                    Level::Fail,
                    "sessions",
                    format!("the newest session {id} cannot be read: {error:#}"),
                    "check that the file is readable and not truncated mid-line",
                )
            }
            Ok(None) => {
                return report.add_with_hint(
                    Level::Warn,
                    "sessions",
                    format!("the newest session {id} disappeared while it was being read"),
                    "another process may be compacting sessions; run this again",
                )
            }
            Ok(Some(session)) => {
                detail.push_str(&format!(
                    "; newest {id}: {} message(s), {} recorded turn(s)",
                    session.messages.len(),
                    session.stats.len()
                ));
                if !session.drift.is_empty() {
                    level = Level::Warn;
                    detail.push_str(&format!(
                        "; {} compaction record(s) did not line up with the history",
                        session.drift.len()
                    ));
                    hint = Some(format!(
                        "the history is intact — nothing was folded — but the summary in the file \
                         no longer describes it: {}",
                        session.drift[0]
                    ));
                }
            }
        }
    }
    match hint {
        Some(hint) => report.add_with_hint(level, "sessions", detail, hint),
        None => report.add(level, "sessions", detail),
    }
}

fn check_skills(options: &Options, report: &mut Report) {
    let mut roots: Vec<PathBuf> = options.skill_dirs.clone();
    for root in SkillRegistry::default_roots() {
        if !roots.contains(&root) {
            roots.push(root);
        }
    }
    let mut registry = SkillRegistry::with_roots(roots);
    match registry.scan() {
        Err(error) => report.add(
            Level::Fail,
            "skills",
            format!("the skill scan failed: {error:#}"),
        ),
        Ok(scan) => {
            let level = if scan.skipped.is_empty() && scan.collisions.is_empty() {
                Level::Ok
            } else {
                Level::Warn
            };
            let mut detail = scan.summary();
            if !scan.missing_roots.is_empty() {
                // Absent roots are the ordinary case, so they are a count rather
                // than a warning: a checkout with no skill directory is normal.
                detail.push_str(&format!(
                    " ({} root(s) absent)",
                    scan.missing_roots.len()
                ));
            }
            if level == Level::Warn {
                report.add_with_hint(
                    level,
                    "skills",
                    detail,
                    "a directory counts as a skill only when it holds a SKILL.md that parses",
                );
            } else {
                report.add(level, "skills", detail);
            }
        }
    }
}

fn check_packs(options: &Options, report: &mut Report) {
    let packs = match pack::installed(&options.packs) {
        Ok(packs) => packs,
        Err(error) => {
            return report.add_with_hint(
                Level::Warn,
                "packs",
                format!("{}: {error:#}", options.packs.display()),
                "the packs directory could not be listed; anything installed there will not load",
            )
        }
    };
    if packs.is_empty() {
        report.add(
            Level::Ok,
            "packs",
            format!("no packs installed in {}", options.packs.display()),
        );
        return;
    }
    let enabled = packs.iter().filter(|installed| installed.enabled).count();
    // A manifest that names an assembly which is not there is the one pack
    // failure that looks like success: the pack loads and contributes nothing.
    let broken: Vec<String> = packs
        .iter()
        .filter_map(|installed| match &installed.assembly {
            Some(path) if !path.is_file() => Some(format!(
                "{} names an assembly that is not there",
                installed.manifest.name
            )),
            _ => None,
        })
        .collect();
    let detail = format!(
        "{} pack(s) installed, {enabled} enabled, {} unloaded",
        packs.len(),
        packs.len() - enabled
    );
    if broken.is_empty() {
        report.add(Level::Ok, "packs", detail);
    } else {
        report.add_with_hint(
            Level::Warn,
            "packs",
            format!("{detail}; {}", broken.join("; ")),
            "reinstall the pack, or remove its directory so it stops being listed",
        );
    }
}

fn check_ledger(options: &Options, report: &mut Report) {
    let ledger = match Ledger::load(&options.ledger) {
        Ok(ledger) => ledger,
        Err(error) => {
            return report.add_with_hint(
                Level::Fail,
                "ledger",
                format!("{error:#}"),
                "the ledger records what is installed; a file that does not parse makes every \
                 install look missing, so fix it or move it aside",
            )
        }
    };
    if ledger.is_empty() {
        report.add(
            Level::Ok,
            "ledger",
            format!("nothing recorded in {}", options.ledger.display()),
        );
        return;
    }
    // Only skill entries carry a path, and a recorded path that is gone means an
    // install that will silently do nothing the next time it is asked to load.
    let missing: Vec<String> = ledger
        .entries
        .iter()
        .filter_map(|entry| match &entry.path {
            Some(path) if !Path::new(path).exists() => {
                Some(format!("{} {} is gone ({path})", entry.kind, entry.id))
            }
            _ => None,
        })
        .collect();
    let detail = format!(
        "{} install(s) recorded in {}",
        ledger.entries.len(),
        options.ledger.display()
    );
    if missing.is_empty() {
        report.add(Level::Ok, "ledger", detail);
    } else {
        report.add_with_hint(
            Level::Warn,
            "ledger",
            format!("{detail}; {}", missing.join("; ")),
            "reinstall what is missing, or remove the entry so it stops being counted as installed",
        );
    }
}

fn check_prices(options: &Options, report: &mut Report) {
    match PriceTable::from_path(&options.prices) {
        Err(error) => report.add_with_hint(
            Level::Warn,
            "prices",
            format!("{error:#}"),
            "`ngu stats --cost` will refuse to answer until this file parses; delete it to run \
             unpriced instead",
        ),
        Ok(None) => report.add_with_hint(
            Level::Ok,
            "prices",
            format!("no price file at {}", options.prices.display()),
            "optional: write it to make `ngu stats --cost` report money as well as tokens",
        ),
        Ok(Some(table)) if table.is_empty() => report.add_with_hint(
            Level::Warn,
            "prices",
            format!("{} names no rates", options.prices.display()),
            "an empty price file prices nothing; add a model entry or delete the file",
        ),
        Ok(Some(table)) => {
            let models = table.models();
            report.add(
                Level::Ok,
                "prices",
                format!(
                    "{} rate(s) in {}{}",
                    models.len(),
                    options.prices.display(),
                    if table.rate_for("anything").is_some() {
                        ", with a fallback"
                    } else {
                        ", no fallback"
                    }
                ),
            );
        }
    }
}

/// Whether a diagnostic's exit code should be non-zero.
///
/// Warnings do not fail the command: a missing price file is not a broken
/// installation, and a script that treats it as one would cry wolf at every run.
pub fn exit_code(report: &Report) -> i32 {
    match report.worst() {
        Level::Fail => 1,
        _ => 0,
    }
}

/// Run the checks and print them the way `ngu doctor` prints.
pub fn print_report(options: &Options, as_json: bool) -> Result<i32> {
    let report = run(options);
    if as_json {
        println!("{}", serde_json::to_string_pretty(&report.json())?);
    } else {
        for line in report.lines() {
            println!("{line}");
        }
    }
    Ok(exit_code(&report))
}

#[cfg(test)]
mod doctor_tests {
    use super::{exit_code, run, Level, Options};
    use crate::message::Message;
    // The store's writing methods live on the trait, so a test that builds a
    // session has to bring it into scope as well.
    use crate::session::{JsonlStore, Session, SessionStore};

    /// A private directory, cleaned up by the caller.
    fn temp_dir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("ngu-doctor-{tag}-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        dir
    }

    /// Options that point entirely inside `dir`, so no check reads the real
    /// machine or the process environment.
    fn options_under(dir: &std::path::Path) -> Options {
        Options {
            settings: dir.join("settings.json"),
            sessions: dir.join("sessions"),
            packs: dir.join("packs"),
            ledger: dir.join("ledger.json"),
            prices: dir.join("prices.json"),
            skill_dirs: vec![dir.join("skills")],
        }
    }

    fn check<'a>(report: &'a super::Report, name: &str) -> &'a super::Check {
        report
            .checks
            .iter()
            .find(|check| check.name == name)
            .unwrap_or_else(|| panic!("no {name} check in {:?}", report.checks))
    }

    #[test]
    fn a_fresh_home_warns_but_fails_nothing() {
        let dir = temp_dir("fresh");
        let report = run(&options_under(&dir));
        assert_eq!(report.worst(), Level::Warn, "{:?}", report.checks);
        assert_eq!(report.count(Level::Fail), 0);
        // A missing settings file is a warning with a way out of it.
        let settings = check(&report, "settings");
        assert_eq!(settings.level, Level::Warn);
        assert!(settings.hint.is_some());
        // Absent directories are ordinary, not failures.
        assert_eq!(check(&report, "sessions").level, Level::Ok);
        assert_eq!(check(&report, "packs").level, Level::Ok);
        assert_eq!(check(&report, "ledger").level, Level::Ok);
        assert_eq!(check(&report, "prices").level, Level::Ok);
        assert_eq!(exit_code(&report), 0, "warnings do not fail the command");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_settings_file_that_does_not_parse_is_a_failure_that_names_it() {
        let dir = temp_dir("settings");
        let options = options_under(&dir);
        std::fs::write(&options.settings, "{ not json").expect("write");
        let report = run(&options);
        let settings = check(&report, "settings");
        assert_eq!(settings.level, Level::Fail, "{:?}", settings);
        assert!(
            settings.detail.contains("settings.json"),
            "the finding names the file: {}",
            settings.detail
        );
        assert_eq!(exit_code(&report), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_route_is_described_without_ever_printing_the_key() {
        let dir = temp_dir("route");
        let options = options_under(&dir);
        std::fs::write(
            &options.settings,
            r#"{"base_url": "https://example.test/v1", "api_key": "sk-secret-value", "model": "m1"}"#,
        )
        .expect("write");
        let report = run(&options);
        let settings = check(&report, "settings");
        assert_eq!(settings.level, Level::Ok, "{:?}", settings);
        assert!(settings.detail.contains("https://example.test/v1"), "{}", settings.detail);
        assert!(settings.detail.contains("api_key set"), "{}", settings.detail);
        let printed = report.lines().join("\n") + &report.json().to_string();
        assert!(
            !printed.contains("sk-secret-value"),
            "a diagnostic must never carry the key: {printed}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_skill_directory_that_yields_no_skill_is_reported_with_its_reason() {
        let dir = temp_dir("skills");
        let options = options_under(&dir);
        // A directory under a skill root that has no SKILL.md: the scan sees it
        // and cannot make a skill of it, which is exactly what a user who
        // "installed" a skill by copying files reports as "my skill vanished".
        let root = dir.join("skills");
        std::fs::create_dir_all(root.join("half-a-skill")).expect("skill dir");
        std::fs::write(root.join("half-a-skill").join("notes.md"), "not a skill").expect("write");
        let report = run(&options);
        let skills = check(&report, "skills");
        assert_eq!(skills.level, Level::Warn, "{:?}", skills);
        assert!(skills.detail.contains("half-a-skill"), "{}", skills.detail);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_compaction_that_no_longer_lines_up_is_surfaced_from_the_newest_session() {
        let dir = temp_dir("drift");
        let options = options_under(&dir);
        let store = JsonlStore::open(&options.sessions).expect("open store");
        let session = Session::new(Some("m".into()));
        store.create(&session).expect("create");
        store
            .append(
                &session.id,
                &[Message::user("one"), Message::assistant("two"), Message::user("three")],
            )
            .expect("append");
        store
            .append_compaction(
                &session.id,
                &crate::compaction::Compaction {
                    replaced: 2,
                    summary: "one two".into(),
                },
            )
            .expect("compaction");
        // Something outside the program edits a message the summary covered.
        let path = options.sessions.join(format!("{}.jsonl", session.id));
        let raw = std::fs::read_to_string(&path).expect("raw");
        std::fs::write(&path, raw.replace("\"one\"", "\"one, edited\"")).expect("write");

        let report = run(&options);
        let sessions = check(&report, "sessions");
        assert_eq!(sessions.level, Level::Warn, "{:?}", sessions);
        assert!(
            sessions.detail.contains("did not line up"),
            "{}",
            sessions.detail
        );
        assert_eq!(exit_code(&report), 0, "drift is news, not a broken install");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_ledger_entry_whose_path_is_gone_is_reported() {
        let dir = temp_dir("ledger");
        let options = options_under(&dir);
        std::fs::write(
            &options.ledger,
            r#"{"version": 1, "entries": [
                {"kind": "skill", "id": "gone", "source": "pack:x", "path": "C:/nowhere/at/all",
                 "scope": "user", "pack": "x", "installed_at": "2026-01-01T00:00:00Z"}
            ]}"#,
        )
        .expect("write");
        let report = run(&options);
        let ledger = check(&report, "ledger");
        assert_eq!(ledger.level, Level::Warn, "{:?}", ledger);
        assert!(ledger.detail.contains("gone"), "{}", ledger.detail);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn every_check_appears_in_the_text_and_in_the_json() {
        let dir = temp_dir("shape");
        let report = run(&options_under(&dir));
        let json = report.json();
        assert_eq!(json["checks"].as_array().map(Vec::len), Some(report.checks.len()));
        assert_eq!(json["ok"], report.count(Level::Ok));
        assert_eq!(json["warnings"], report.count(Level::Warn));
        assert_eq!(json["failures"], report.count(Level::Fail));
        assert_eq!(json["worst"], report.worst().as_str());
        let text = report.lines();
        assert_eq!(text.len(), report.checks.len() + 1, "one line each, then the verdict");
        assert_eq!(text.last().map(String::as_str), Some(report.summary().as_str()));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
