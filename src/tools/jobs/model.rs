//! The job model: what a background job is, and how it reports itself.
//!
//! A job is a command, an owner, a log file, an index entry and the process
//! behind them. This file holds the data and the behaviour that reads it — not
//! the process plumbing that produces it (see [`super::process`]) and not the
//! registry that owns it (see [`super::registry`]).
//!
//! Liveness is never stored: [`Job::state`] asks the process, because a record
//! that says "running" is a claim about the past. This is the file where that
//! rule is implemented, so it is the file to change if it ever needs to move.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::{json, Value};
use tokio::io::AsyncReadExt;

use super::process::{kill_pid, process_alive, verify_target, KillOutcome};

/// Output bytes one reader is handed: the tail of the log, newest kept.
const MAX_JOB_OUTPUT: usize = 256 * 1024;

/// Bytes one job's log file may reach. Past this the job keeps running and the
/// file records that it stopped growing — the alternative is a runaway
/// printer filling the data disk.
const MAX_JOB_LOG: u64 = 32 * 1024 * 1024;

/// What a job is doing now.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JobState {
    /// Still running.
    Running,
    /// Finished on its own; carries the exit code when the OS reported one.
    Exited(Option<i32>),
    /// Stopped by `job_kill`.
    Killed,
    /// Not running any more, and nothing recorded how it ended — a job from a
    /// previous process whose pid is gone. Said plainly instead of guessing an
    /// exit code nobody saw.
    Gone,
}

impl JobState {
    /// One line a model reads instead of the enum.
    pub fn label(&self, age: Duration) -> String {
        match self {
            JobState::Running => format!("running for {}", format_age(age)),
            JobState::Exited(Some(code)) => format!("exited with code {code}"),
            JobState::Exited(None) => "exited".to_string(),
            JobState::Killed => "killed".to_string(),
            JobState::Gone => "finished (exit code not recorded; the process that started it is gone)"
                .to_string(),
        }
    }

    /// The short tag the index file stores.
    fn tag(&self) -> &'static str {
        match self {
            JobState::Running => "running",
            JobState::Exited(_) => "exited",
            JobState::Killed => "killed",
            JobState::Gone => "gone",
        }
    }

    /// Rebuild from an index entry, if it names a settled state.
    pub(super) fn from_index(tag: &str, code: Option<i32>) -> Option<Self> {
        match tag {
            "exited" => Some(JobState::Exited(code)),
            "killed" => Some(JobState::Killed),
            "gone" => Some(JobState::Gone),
            // `running` is deliberately not a state: liveness is asked of the
            // process, never read off the record.
            _ => None,
        }
    }
}

/// One background command.
pub struct Job {
    /// Stable id the model uses: `job-1`, `job-2`, …
    pub id: String,
    /// The command line as it was started.
    pub command: String,
    /// The session whose turn spawned this job (`None` if the caller set no
    /// session — CLI/headless, or a spawn outside any execute).
    ///
    /// Attribution lives here, where spawn happens, because that is the only
    /// moment the caller identity is unambiguous: `current_session()` reads it
    /// from the execute-pinned thread-local.
    pub(super)owner: Option<String>,
    /// When it started, for age reporting.
    pub started: Instant,
    /// The same moment as wall-clock, which is what the index stores.
    pub(super)started_ms: u64,
    /// Whether the settlement hook should wake the model when this job ends.
    pub(super)notify: AtomicU8,
    /// Settled state, once observed. `None` means "not settled as far as this
    /// process knows" — which is not the same as running; see [`Job::state`].
    pub(super)settled: Mutex<Option<JobState>>,
    /// Set once `job_kill` asked; the waiter reports `Killed` instead of the
    /// exit code a forced termination happens to produce.
    pub(super)killed: AtomicU8,
    /// stdout and stderr readers still draining.
    pub(super)readers: AtomicU8,
    /// The child itself, held for `job_kill`; never across an await. `None`
    /// for an entry rebuilt from the index, whose process this one never owned.
    pub(super)child: Mutex<Option<tokio::process::Child>>,
    /// The pid, kept in the index so liveness can be asked after a restart.
    pub(super)pid: Option<u32>,
    /// The process image this job started, lowercased and without extension
    /// (`cmd`, `bash`), recorded at spawn.
    ///
    /// It is the identity half of a kill: a pid read off the index may have
    /// been recycled by the OS, and only a fingerprint recorded when the job
    /// started can tell "that process" from "whatever holds pid 4321 now".
    pub(super)image: Option<String>,
    /// Whether this entry was rebuilt from the index rather than spawned here.
    /// A process this one never owned cannot report an exit code, and saying
    /// "exited" for it would invent one.
    pub(super)legacy: bool,
    /// Where the log and the index entry live.
    pub(super)dir: PathBuf,
    /// The append handle the pumps write through. The file is the output.
    pub(super)log: Mutex<std::fs::File>,
    /// Bytes written so far, against [`MAX_JOB_LOG`].
    pub(super)written: AtomicU64,
}

impl Job {
    /// The job's current state, asked of the process rather than the record.
    pub fn state(&self) -> JobState {
        if let Some(settled) = self.settled.lock().unwrap_or_else(|e| e.into_inner()).clone() {
            return settled;
        }
        self.ask_the_process().unwrap_or(JobState::Running)
    }

    /// Ask reality: the child handle when this process owns it, the pid
    /// otherwise. `None` means still running.
    pub(super) fn ask_the_process(&self) -> Option<JobState> {
        {
            let mut guard = self.child.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(child) = guard.as_mut() {
                return match child.try_wait() {
                    Ok(Some(status)) => Some(self.settle(status.code())),
                    Ok(None) => None,
                    // The handle cannot say; fall through to the pid.
                    Err(_) => Some(self.settle(None)),
                };
            }
        }
        match self.pid {
            Some(pid) if process_alive(pid) => None,
            // No pid (never recorded) or a pid that is gone: finished, and
            // this process never saw the exit code.
            _ => Some(self.settle(None)),
        }
    }

    /// Record a settled state once, and write it into the index.
    fn settle(&self, code: Option<i32>) -> JobState {
        let state = if self.killed.load(Ordering::Relaxed) == 1 {
            JobState::Killed
        } else if self.legacy {
            // A job from a previous process: its exit code was never this
            // process's to observe, so it is not invented.
            JobState::Gone
        } else {
            JobState::Exited(code)
        };
        let mut guard = self.settled.lock().unwrap_or_else(|e| e.into_inner());
        if guard.is_none() {
            *guard = Some(state.clone());
            drop(guard);
            self.write_index();
            return state;
        }
        guard.clone().expect("settled")
    }

    /// The session that spawned this job, when known.
    pub fn owner(&self) -> Option<&str> {
        self.owner.as_deref()
    }

    /// Whether a session may see this job: its own jobs, plus any job spawned
    /// without an owner (nothing to attribute — CLI, or a call outside execute).
    pub fn belongs_to(&self, session: &str) -> bool {
        self.owner.as_deref().map_or(true, |o| o == session)
    }

    /// Whether the job has not settled yet, by the process's own account.
    pub fn is_running(&self) -> bool {
        matches!(self.state(), JobState::Running)
    }

    /// Where this job's output is kept. The file exists from the first byte on.
    pub fn log_path(&self) -> PathBuf {
        self.dir.join(format!("{}.log", self.id))
    }

    /// Where this job's index entry is kept.
    fn index_path(&self) -> PathBuf {
        self.dir.join(format!("{}.json", self.id))
    }

    /// Append what the job printed to its log.
    ///
    /// Writes stop at [`MAX_JOB_LOG`], with one line in the file saying so:
    /// the job keeps running, the disk does not fill.
    fn push_output(&self, chunk: &[u8]) {
        let written = self.written.load(Ordering::Relaxed);
        if written >= MAX_JOB_LOG {
            return;
        }
        let room = (MAX_JOB_LOG - written) as usize;
        let (slice, capped) = if chunk.len() > room {
            (&chunk[..room], true)
        } else {
            (chunk, false)
        };
        let mut file = self.log.lock().unwrap_or_else(|e| e.into_inner());
        use std::io::Write;
        if file.write_all(slice).is_err() {
            return;
        }
        let mut total = written + slice.len() as u64;
        if capped {
            let note = format!(
                "\n… output capped at {} MB; the job keeps running …\n",
                MAX_JOB_LOG / (1024 * 1024)
            );
            let _ = file.write_all(note.as_bytes());
            total = MAX_JOB_LOG + note.len() as u64;
        }
        let _ = file.flush();
        self.written.store(total, Ordering::Relaxed);
    }

    /// The tail of the job's output, as text.
    ///
    /// Read from the log file — the file is the output, so what a previous
    /// process's job printed is still readable here.
    pub fn output_text(&self) -> String {
        let path = self.log_path();
        let Ok(metadata) = std::fs::metadata(&path) else {
            return String::new();
        };
        let len = metadata.len();
        let cap = MAX_JOB_OUTPUT as u64;
        let mut file = match std::fs::File::open(&path) {
            Ok(file) => file,
            Err(_) => return String::new(),
        };
        if len > cap {
            use std::io::{Read, Seek, SeekFrom};
            if file.seek(SeekFrom::Start(len - cap)).is_err() {
                return String::new();
            }
            let mut tail = Vec::with_capacity(cap as usize);
            if file.read_to_end(&mut tail).is_err() {
                return String::new();
            }
            format!(
                "… ({} bytes omitted; showing the last {}) …\n{}",
                len - cap,
                format_bytes(cap),
                String::from_utf8_lossy(&tail)
            )
        } else {
            std::fs::read(&path)
                .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
                .unwrap_or_default()
        }
    }

    /// Write the index entry: what this job is, and the last state seen.
    pub(super) fn write_index(&self) {
        let state = self.settled.lock().unwrap_or_else(|e| e.into_inner()).clone();
        let payload = json!({
            "id": self.id,
            "command": self.command,
            "pid": self.pid,
            "image": self.image,
            "owner": self.owner,
            "started_ms": self.started_ms,
            "notify": self.notify.load(Ordering::Relaxed) == 1,
            "log": self.log_path().display().to_string(),
            "state": state.as_ref().map(|s| s.tag()).unwrap_or("running"),
            "exit_code": match state {
                Some(JobState::Exited(code)) => code.map(Value::from).unwrap_or(Value::Null),
                _ => Value::Null,
            },
        });
        if let Ok(text) = serde_json::to_string_pretty(&payload) {
            let _ = std::fs::write(self.index_path(), text);
        }
    }

    /// Stop the job, if it is still this job's process to stop.
    ///
    /// The kill itself is quick; the short wait afterwards is only for the
    /// settle path to observe the exit and record it, so the caller can report
    /// the outcome instead of "killed, probably".
    ///
    /// Two targets, two standards of proof:
    ///
    /// * **A child handle this process still owns** — the handle *is* the
    ///   proof. Windows and Unix both keep a process object alive while a
    ///   handle to it is open, so its pid cannot be recycled behind our back.
    ///   An already-reaped child is not signalled at all: after the reap that
    ///   pid may belong to anything.
    /// * **A pid read off the index** — nothing but a claim. Before signalling
    ///   it, [`verify_target`] has to show it is still the process this job
    ///   started; if it cannot, the kill is refused and the caller is told.
    ///   `taskkill /T /F` on a recycled pid kills a stranger's whole tree.
    pub async fn kill(&self) -> KillOutcome {
        // What to signal, and whether its identity is already proven. Decided
        // before anything is signalled, because deciding *whether* to signal is
        // the part that must not touch the process.
        let target = {
            let mut guard = self.child.lock().unwrap_or_else(|e| e.into_inner());
            match guard.as_mut() {
                Some(child) => match child.try_wait() {
                    // Reaped here already: its pid is somebody else's now.
                    Ok(Some(_)) => None,
                    // Held handle: the OS keeps this pid from being recycled,
                    // so it is safe to signal without asking further.
                    Ok(None) | Err(_) => child.id().map(|pid| (true, pid)),
                },
                None => self.pid.map(|pid| (false, pid)),
            }
        };
        let Some((owned, pid)) = target else {
            // No handle, no recorded pid: nothing to signal. Report what the
            // process says about itself.
            return KillOutcome::Stopped(self.state());
        };

        if !owned {
            if let Err(reason) = verify_target(self, pid) {
                // Nothing was signalled and nothing was recorded: the job is
                // still exactly as it was.
                return KillOutcome::Refused(reason);
            }
        }

        self.killed.store(1, Ordering::Relaxed);
        // The async-safe signal first (it also lets the runtime reap), then the
        // tree sweep that reaches what the shell forked.
        {
            let mut guard = self.child.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(child) = guard.as_mut() {
                let _ = child.start_kill();
            }
        }
        kill_pid(pid);

        // A kill is reported only once the process is observed gone: the
        // settle path may need its next poll, and the pipes take a moment to
        // close. Three seconds outlasts the whole settle window; a healthy
        // kill returns in ~200ms. A process that survived the signal is
        // reported as still running, which is the truth.
        for _ in 0..150 {
            if !self.is_running() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        KillOutcome::Stopped(self.state())
    }
}

/// Drain one pipe into the job's log until EOF.
///
/// Generic because `ChildStdout` and `ChildStderr` are distinct types: on
/// Windows they happen to collapse into one, which is how an array holding
/// both compiled there and failed everywhere else.
pub(super) fn pump<R>(mut pipe: R, job: Arc<Job>)
where
    R: tokio::io::AsyncRead + Send + Unpin + 'static,
{
    job.readers.fetch_add(1, Ordering::Relaxed);
    tokio::spawn(async move {
        let mut buf = [0u8; 8192];
        loop {
            match pipe.read(&mut buf).await {
                Ok(0) | Err(_) => break,
                Ok(read) => job.push_output(&buf[..read]),
            }
        }
        job.readers.fetch_sub(1, Ordering::Relaxed);
    });
}

/// `12s`, `3m 04s`, `1h 12m` — an age a model can compare at a glance.
fn format_age(age: Duration) -> String {
    let secs = age.as_secs();
    if secs < 60 {
        format!("{secs}s")
    } else if secs < 3600 {
        format!("{}m {:02}s", secs / 60, secs % 60)
    } else {
        format!("{}h {}m", secs / 3600, (secs % 3600) / 60)
    }
}

/// `512 KB`, `2 MB` — for the truncation marker.
fn format_bytes(bytes: u64) -> String {
    if bytes >= 1024 * 1024 {
        format!("{} MB", bytes / (1024 * 1024))
    } else {
        format!("{} KB", bytes / 1024)
    }
}
