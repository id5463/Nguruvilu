//! Workspace snapshots: a git commit per agent action.
//!
//! This is the safety net that makes "the kernel only reads and writes files"
//! survivable. Every tool that can change the workspace is bracketed by a
//! snapshot, so a bad edit is one command away from being undone, and each
//! session's history maps onto a commit.
//!
//! Deliberately **not** a global git hook. A hook would fire for every
//! repository on the machine, including ones this agent never touches, and it
//! cannot know when a snapshot is meaningful. The kernel knows, so the kernel
//! does it.
//!
//! Identity is passed per-invocation rather than read from global config, so
//! snapshots work on a machine with no git identity configured and never write
//! to the user's gitconfig.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{anyhow, Context, Result};

/// Marker prefixed to every snapshot commit message.
pub const SNAPSHOT_PREFIX: &str = "[ngu]";

/// Identity used for snapshot commits.
const SNAPSHOT_NAME: &str = "nguruvilu";
const SNAPSHOT_EMAIL: &str = "ngu@localhost";

/// One recorded snapshot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SnapshotRecord {
    /// Full commit hash.
    pub commit: String,
    /// Commit subject.
    pub subject: String,
}

impl SnapshotRecord {
    /// Short hash for display.
    pub fn short(&self) -> &str {
        &self.commit[..self.commit.len().min(8)]
    }
}

/// A git-backed snapshot service for one workspace.
#[derive(Debug, Clone)]
pub struct GitSnapshot {
    workspace: PathBuf,
    available: bool,
}

impl GitSnapshot {
    /// Open a workspace, initializing a repository when there is none.
    ///
    /// When `git` is not installed the service reports itself unavailable and
    /// every operation becomes a no-op rather than an error: snapshots are a
    /// safety net, and their absence must not stop the agent from working.
    pub fn open(workspace: impl Into<PathBuf>) -> Result<Self> {
        let workspace = workspace.into();
        if !workspace.exists() {
            std::fs::create_dir_all(&workspace)
                .with_context(|| format!("creating workspace {}", workspace.display()))?;
        }

        let available = git_available();
        if available && !is_repo(&workspace) {
            init_repo(&workspace)?;
        }

        Ok(Self { workspace, available })
    }

    /// Whether git is usable here.
    pub fn is_available(&self) -> bool {
        self.available
    }

    /// The workspace root.
    pub fn workspace(&self) -> &Path {
        &self.workspace
    }

    /// Whether the workspace is a repository.
    pub fn is_repo(&self) -> bool {
        self.available && is_repo(&self.workspace)
    }

    /// Commit the current state.
    ///
    /// Returns `None` when there is nothing to commit, so a no-op action does
    /// not litter the history.
    pub fn snapshot(&self, label: &str) -> Result<Option<SnapshotRecord>> {
        if !self.is_repo() {
            return Ok(None);
        }

        run(&self.workspace, &["add", "-A"])?;

        // An empty index means nothing changed since the last snapshot.
        let status = run(&self.workspace, &["status", "--porcelain"])?;
        if status.trim().is_empty() {
            return Ok(None);
        }

        let message = format!("{SNAPSHOT_PREFIX} {label}");
        run_commit(&self.workspace, &message)?;

        let commit = run(&self.workspace, &["rev-parse", "HEAD"])?.trim().to_string();
        Ok(Some(SnapshotRecord { commit, subject: message }))
    }

    /// The current commit, if any.
    pub fn head(&self) -> Result<Option<String>> {
        if !self.is_repo() {
            return Ok(None);
        }
        match run(&self.workspace, &["rev-parse", "HEAD"]) {
            Ok(out) => Ok(Some(out.trim().to_string())),
            // A repository with no commits has no HEAD; that is not an error.
            Err(_) => Ok(None),
        }
    }

    /// Recent snapshots, newest first.
    pub fn history(&self, limit: usize) -> Result<Vec<SnapshotRecord>> {
        if !self.is_repo() {
            return Ok(Vec::new());
        }
        let format = "--format=%H%x00%s";
        let count = format!("--max-count={}", limit.max(1));
        let output = match run(&self.workspace, &["log", format, &count]) {
            Ok(out) => out,
            Err(_) => return Ok(Vec::new()),
        };

        let mut records = Vec::new();
        for line in output.lines() {
            let Some((commit, subject)) = line.split_once('\0') else {
                continue;
            };
            records.push(SnapshotRecord {
                commit: commit.to_string(),
                subject: subject.to_string(),
            });
        }
        Ok(records)
    }

    /// Restore the working tree to a snapshot, keeping history intact.
    ///
    /// A new commit records the restore, so the rollback is itself auditable
    /// and can be undone.
    pub fn restore(&self, commit: &str) -> Result<SnapshotRecord> {
        if !self.is_repo() {
            return Err(anyhow!("workspace is not a git repository"));
        }

        // Verify the target exists before touching anything.
        run(&self.workspace, &["cat-file", "-e", &format!("{commit}^{{commit}}")])
            .with_context(|| format!("snapshot {commit} does not exist"))?;

        // `read-tree --reset -u` rewrites the index and working tree to match
        // the snapshot without moving HEAD, so the rollback is itself recorded
        // rather than erasing the commits that led to it.
        //
        // `checkout <commit> -- .` is not enough: it restores files the
        // snapshot knew about but leaves later additions in place, which is
        // exactly the case a rollback usually needs to undo.
        //
        // Files the snapshot never tracked and that the ignore rules do not
        // cover are left alone. Deleting a user's untracked work to satisfy a
        // rollback would be a worse failure than an incomplete restore.
        run(&self.workspace, &["read-tree", "--reset", "-u", commit])?;
        run(&self.workspace, &["add", "-A"])?;

        let short = &commit[..commit.len().min(8)];
        let status = run(&self.workspace, &["status", "--porcelain"])?;
        if status.trim().is_empty() {
            // The tree already matched; record that instead of failing on an
            // empty commit.
            let head = self
                .head()?
                .unwrap_or_else(|| commit.to_string());
            return Ok(SnapshotRecord {
                commit: head,
                subject: format!("{SNAPSHOT_PREFIX} already at {short}"),
            });
        }

        let message = format!("{SNAPSHOT_PREFIX} restore to {short}");
        run_commit(&self.workspace, &message)?;

        let head = run(&self.workspace, &["rev-parse", "HEAD"])?.trim().to_string();
        Ok(SnapshotRecord { commit: head, subject: message })
    }

    /// Files changed since a snapshot.
    pub fn changed_since(&self, commit: &str) -> Result<Vec<String>> {
        if !self.is_repo() {
            return Ok(Vec::new());
        }
        let mut files: BTreeSet<String> = BTreeSet::new();

        // Tracked files that differ from the snapshot.
        for line in run(&self.workspace, &["diff", "--name-only", commit, "--"])?.lines() {
            let line = line.trim();
            if !line.is_empty() {
                files.insert(line.to_string());
            }
        }

        // Files the snapshot never knew about. `git diff` does not report
        // these, and "what did the agent create" is the more useful half of
        // the question.
        for line in run(&self.workspace, &["ls-files", "--others", "--exclude-standard"])?.lines() {
            let line = line.trim();
            if !line.is_empty() {
                files.insert(line.to_string());
            }
        }

        Ok(files.into_iter().collect())
    }
}

/// Whether a `git` binary is on PATH.
fn git_available() -> bool {
    Command::new("git")
        .arg("--version")
        .output()
        .map(|out| out.status.success())
        .unwrap_or(false)
}

/// Whether `dir` is itself the root of a git work tree.
///
/// This checks for `.git` in `dir` rather than asking whether `dir` is *inside*
/// a work tree. A workspace nested under another repository — a scratch
/// directory under a user's project, for instance — must get its own snapshots
/// instead of silently operating on the outer repository's index.
fn is_repo(dir: &Path) -> bool {
    dir.join(".git").exists()
}

/// Initialize a repository with a baseline ignore file.
fn init_repo(dir: &Path) -> Result<()> {
    run(dir, &["init", "--quiet"])?;

    // Without an ignore file a first snapshot would swallow build output and
    // dependency trees, which is both slow and useless.
    let ignore = dir.join(".gitignore");
    if !ignore.exists() {
        std::fs::write(
            &ignore,
            "# Created by nguruvilu so workspace snapshots stay meaningful.\n\
             node_modules/\n\
             target/\n\
             dist/\n\
             build/\n\
             .nguruvilu/\n\
             *.log\n",
        )
        .with_context(|| format!("writing {}", ignore.display()))?;

        // Commit the baseline immediately, so the ignore file is not mistaken
        // for the agent's first change and a no-op action really is a no-op.
        run(dir, &["add", ".gitignore"])?;
        run_commit(dir, &format!("{SNAPSHOT_PREFIX} initialize workspace"))?;
    }
    Ok(())
}

/// Run a git command in `dir` and return stdout.
fn run(dir: &Path, args: &[&str]) -> Result<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(args)
        .output()
        .with_context(|| format!("running git {}", args.join(" ")))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(anyhow!(
            "git {} failed: {}",
            args.join(" "),
            stderr.trim()
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

/// Commit with an explicit identity, so global config is never consulted.
fn run_commit(dir: &Path, message: &str) -> Result<()> {
    let output = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args([
            "-c",
            &format!("user.name={SNAPSHOT_NAME}"),
            "-c",
            &format!("user.email={SNAPSHOT_EMAIL}"),
            "-c",
            "commit.gpgsign=false",
            "commit",
            "--quiet",
            "--no-verify",
            "-m",
            message,
        ])
        .output()
        .context("running git commit")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(anyhow!("git commit failed: {}", stderr.trim()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_workspace(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("ngu-git-{tag}-{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn opening_initializes_a_repository() {
        let dir = temp_workspace("init");
        let snapshot = GitSnapshot::open(&dir).unwrap();
        if !snapshot.is_available() {
            // git is not installed; the service degrades instead of failing.
            assert!(!snapshot.is_repo());
            return;
        }
        assert!(snapshot.is_repo());
        assert!(dir.join(".git").exists());
        assert!(dir.join(".gitignore").exists(), "a baseline ignore file is written");
    }

    #[test]
    fn a_snapshot_records_a_commit_and_a_no_op_records_nothing() {
        let dir = temp_workspace("commit");
        let snapshot = GitSnapshot::open(&dir).unwrap();
        if !snapshot.is_available() {
            return;
        }

        // Nothing changed yet: no commit, so history stays clean.
        assert!(snapshot.snapshot("empty").unwrap().is_none());

        std::fs::write(dir.join("a.txt"), "first").unwrap();
        let record = snapshot.snapshot("wrote a.txt").unwrap().expect("a commit");
        assert!(record.subject.starts_with(SNAPSHOT_PREFIX));
        assert!(record.subject.contains("wrote a.txt"));
        assert_eq!(record.commit.len(), 40);
        assert_eq!(record.short().len(), 8);

        // The same content again is still a no-op.
        assert!(snapshot.snapshot("unchanged").unwrap().is_none());

        // A real change produces a new commit.
        std::fs::write(dir.join("a.txt"), "second").unwrap();
        let second = snapshot.snapshot("edited a.txt").unwrap().expect("a commit");
        assert_ne!(second.commit, record.commit);
    }

    #[test]
    fn history_is_newest_first_and_bounded() {
        let dir = temp_workspace("history");
        let snapshot = GitSnapshot::open(&dir).unwrap();
        if !snapshot.is_available() {
            return;
        }

        for i in 0..3 {
            std::fs::write(dir.join("f.txt"), format!("v{i}")).unwrap();
            snapshot.snapshot(&format!("change {i}")).unwrap();
        }

        // The baseline commit from `open` plus the three changes above.
        let all = snapshot.history(10).unwrap();
        assert!(all.len() >= 3, "history holds the changes: {all:?}");
        assert!(all[0].subject.contains("change 2"), "newest first");
        assert!(all[1].subject.contains("change 1"));
        assert!(all[2].subject.contains("change 0"));

        let limited = snapshot.history(2).unwrap();
        assert_eq!(limited.len(), 2);
    }

    #[test]
    fn restoring_brings_back_earlier_content() {
        let dir = temp_workspace("restore");
        let snapshot = GitSnapshot::open(&dir).unwrap();
        if !snapshot.is_available() {
            return;
        }

        std::fs::write(dir.join("f.txt"), "good").unwrap();
        let good = snapshot.snapshot("good state").unwrap().expect("commit");

        std::fs::write(dir.join("f.txt"), "broken").unwrap();
        snapshot.snapshot("bad state").unwrap();

        let restored = snapshot.restore(&good.commit).unwrap();
        assert!(restored.subject.contains("restore to"));
        assert_eq!(std::fs::read_to_string(dir.join("f.txt")).unwrap(), "good");

        // The restore is itself a commit, so it is auditable and reversible.
        let head = snapshot.head().unwrap().unwrap();
        assert_eq!(head, restored.commit);
    }

    #[test]
    fn restoring_removes_files_the_snapshot_never_had() {
        let dir = temp_workspace("restore-new");
        let snapshot = GitSnapshot::open(&dir).unwrap();
        if !snapshot.is_available() {
            return;
        }

        std::fs::write(dir.join("kept.txt"), "kept").unwrap();
        let base = snapshot.snapshot("base").unwrap().expect("commit");

        // A file the agent creates after the snapshot.
        std::fs::write(dir.join("added.txt"), "added").unwrap();
        snapshot.snapshot("added").unwrap();
        assert!(dir.join("added.txt").exists());

        snapshot.restore(&base.commit).unwrap();

        assert!(
            !dir.join("added.txt").exists(),
            "a rollback must undo files created after the snapshot"
        );
        assert!(dir.join("kept.txt").exists(), "and keep what the snapshot had");
        assert_eq!(std::fs::read_to_string(dir.join("kept.txt")).unwrap(), "kept");
    }

    #[test]
    fn restoring_a_missing_snapshot_is_an_error() {
        let dir = temp_workspace("missing");
        let snapshot = GitSnapshot::open(&dir).unwrap();
        if !snapshot.is_available() {
            return;
        }
        let error = snapshot
            .restore("0000000000000000000000000000000000000000")
            .expect_err("unknown commit");
        assert!(format!("{error:#}").contains("does not exist"));
    }

    #[test]
    fn changed_files_are_reported_since_a_snapshot() {
        let dir = temp_workspace("changed");
        let snapshot = GitSnapshot::open(&dir).unwrap();
        if !snapshot.is_available() {
            return;
        }

        std::fs::write(dir.join("one.txt"), "1").unwrap();
        let base = snapshot.snapshot("base").unwrap().expect("commit");

        std::fs::write(dir.join("one.txt"), "changed").unwrap();
        std::fs::write(dir.join("two.txt"), "new").unwrap();

        let mut changed = snapshot.changed_since(&base.commit).unwrap();
        changed.sort();
        assert_eq!(changed, vec!["one.txt", "two.txt"]);
    }

    #[test]
    fn a_snapshot_never_consults_global_git_identity() {
        let dir = temp_workspace("identity");
        let snapshot = GitSnapshot::open(&dir).unwrap();
        if !snapshot.is_available() {
            return;
        }
        std::fs::write(dir.join("x.txt"), "x").unwrap();
        snapshot.snapshot("x").unwrap();

        // The commit carries the snapshot identity, not whatever the machine
        // has configured.
        let author = run(&dir, &["log", "-1", "--format=%an <%ae>"]).unwrap();
        assert!(author.contains(SNAPSHOT_NAME), "author is {author:?}");
        assert!(author.contains(SNAPSHOT_EMAIL), "author is {author:?}");
    }

    #[test]
    fn the_generated_ignore_file_keeps_build_output_out() {
        let dir = temp_workspace("ignore");
        let snapshot = GitSnapshot::open(&dir).unwrap();
        if !snapshot.is_available() {
            return;
        }

        std::fs::create_dir_all(dir.join("node_modules")).unwrap();
        std::fs::write(dir.join("node_modules/big.js"), "x").unwrap();
        std::fs::write(dir.join("keep.txt"), "kept").unwrap();

        snapshot.snapshot("with node_modules").unwrap();
        let tracked = run(&dir, &["ls-files"]).unwrap();
        assert!(tracked.contains("keep.txt"));
        assert!(!tracked.contains("node_modules"), "ignored tree: {tracked}");
    }
}
