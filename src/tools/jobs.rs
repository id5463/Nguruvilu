//! Background jobs: commands that outlive the step that started them.
//!
//! A `bash` command that compiles for ten minutes should not pin the turn for
//! ten minutes. Run it with `background: true` — it becomes a job, the model
//! gets an id back immediately, and `job_output` collects the result whenever
//! it wants it (optionally waiting). `job_list` shows what is running;
//! `job_kill` stops one.
//!
//! # Output lives on disk
//!
//! Every job appends to `<data>/jobs/<id>.log` as it runs; `job_output` reads
//! the tail of that file. Nothing the job printed exists only in this
//! process's memory, so a long run's early output is still there later, and a
//! restart does not erase what a job said. One view is capped (256 KB handed
//! to the model per call) and the file itself stops growing at 32 MB so a
//! runaway printer cannot fill the disk — both caps are about reading, not
//! about what the job produced up to that point.
//!
//! # The registry is an index; the process is the truth
//!
//! `<data>/jobs/<id>.json` records what a job *is* — command, pid, owner,
//! start time, log path — and the last state anyone observed. The in-memory
//! map is only a cache of that index, rebuilt from it when the process starts:
//! a job started before a restart still lists, and its output still reads.
//! Whether it is *running* is never taken from the record — it is asked of the
//! process (the child handle when this process owns it, the pid otherwise), so
//! a job whose process died reads as finished with no bookkeeping required.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, Result};
use serde_json::{json, Value};
use tokio::io::AsyncReadExt;
use tokio::process::Command;

use super::{ConflictPolicy, ToolDef, ToolFuture, ToolRegistry};

/// Output bytes one reader is handed: the tail of the log, newest kept.
const MAX_JOB_OUTPUT: usize = 256 * 1024;

/// Bytes one job's log file may reach. Past this the job keeps running and the
/// file records that it stopped growing — the alternative is a runaway
/// printer filling the data disk.
const MAX_JOB_LOG: u64 = 32 * 1024 * 1024;

/// Longest `job_output` will block waiting for a job to finish.
const MAX_WAIT_MS: u64 = 30_000;

/// How often a waiter re-checks whether the child has exited.
const WAIT_POLL_MS: u64 = 100;

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
    fn from_index(tag: &str, code: Option<i32>) -> Option<Self> {
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
    owner: Option<String>,
    /// When it started, for age reporting.
    pub started: Instant,
    /// The same moment as wall-clock, which is what the index stores.
    started_ms: u64,
    /// Whether the settlement hook should wake the model when this job ends.
    notify: AtomicU8,
    /// Settled state, once observed. `None` means "not settled as far as this
    /// process knows" — which is not the same as running; see [`Job::state`].
    settled: Mutex<Option<JobState>>,
    /// Set once `job_kill` asked; the waiter reports `Killed` instead of the
    /// exit code a forced termination happens to produce.
    killed: AtomicU8,
    /// stdout and stderr readers still draining.
    readers: AtomicU8,
    /// The child itself, held for `job_kill`; never across an await. `None`
    /// for an entry rebuilt from the index, whose process this one never owned.
    child: Mutex<Option<tokio::process::Child>>,
    /// The pid, kept in the index so liveness can be asked after a restart.
    pid: Option<u32>,
    /// Whether this entry was rebuilt from the index rather than spawned here.
    /// A process this one never owned cannot report an exit code, and saying
    /// "exited" for it would invent one.
    legacy: bool,
    /// Where the log and the index entry live.
    dir: PathBuf,
    /// The append handle the pumps write through. The file is the output.
    log: Mutex<std::fs::File>,
    /// Bytes written so far, against [`MAX_JOB_LOG`].
    written: AtomicU64,
}

impl Job {
    /// The job's current state, asked of the process rather than the record.
    pub fn state(&self) -> JobState {
        if let Some(settled) = self.settled.lock().expect("job state").clone() {
            return settled;
        }
        self.ask_the_process().unwrap_or(JobState::Running)
    }

    /// Ask reality: the child handle when this process owns it, the pid
    /// otherwise. `None` means still running.
    fn ask_the_process(&self) -> Option<JobState> {
        {
            let mut guard = self.child.lock().expect("job child");
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
        let mut guard = self.settled.lock().expect("job state");
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
        let mut file = self.log.lock().expect("job log");
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
    fn write_index(&self) {
        let state = self.settled.lock().expect("job state").clone();
        let payload = json!({
            "id": self.id,
            "command": self.command,
            "pid": self.pid,
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

    /// Stop the job and return its settled state.
    ///
    /// The kill itself is instant; the short wait is only for the waiter
    /// thread to observe the exit and record it, so the caller can report the
    /// outcome instead of "killed, probably". A job from a previous process
    /// has no child handle here: its pid is signalled directly.
    pub async fn kill(&self) -> JobState {
        self.killed.store(1, Ordering::Relaxed);
        {
            let mut guard = self.child.lock().expect("job child");
            if let Some(child) = guard.as_mut() {
                kill_tree(child);
            } else if let Some(pid) = self.pid {
                kill_pid(pid);
            }
        }
        // The settle path may observe the exit only after its 100ms poll and
        // then wait up to a full second for the pipes to drain -- so a 1s
        // budget here (50 x 20ms) lost that race every time and reported a
        // stopped job as still running. Three seconds outlasts the whole
        // settle window; a healthy kill returns in ~200ms.
        for _ in 0..150 {
            if !self.is_running() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        self.state()
    }
}

/// Drain one pipe into the job's log until EOF.
///
/// Generic because `ChildStdout` and `ChildStderr` are distinct types: on
/// Windows they happen to collapse into one, which is how an array holding
/// both compiled there and failed everywhere else.
fn pump<R>(mut pipe: R, job: Arc<Job>)
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

/// Whether a process is still there. Signal 0 asks without delivering.
fn process_alive(pid: u32) -> bool {
    #[cfg(unix)]
    {
        // A process this one started can be a zombie it has not reaped; the
        // pid then still "exists". The child handle path answers for those,
        // so a pid probe only ever sees somebody else's process here.
        unsafe { libc::kill(pid as i32, 0) == 0 }
    }
    #[cfg(windows)]
    {
        // No winapi dependency in this crate; `tasklist` is the portable
        // question and this runs only when a job is inspected, never in a
        // loop.
        std::process::Command::new("tasklist")
            .args(["/FI", &format!("PID eq {pid}"), "/NH"])
            .output()
            .map(|output| {
                let text = String::from_utf8_lossy(&output.stdout).to_lowercase();
                text.contains(&pid.to_string())
            })
            .unwrap_or(false)
    }
}

/// Stop a pid and the group it leads — for a job whose child handle this
/// process does not hold (one rebuilt from the index).
fn kill_pid(pid: u32) {
    #[cfg(unix)]
    {
        let pid = pid as i32;
        unsafe {
            // The same guard `kill_tree` uses: only a group this code created
            // is signalled as a group.
            if libc::getpgid(pid) == pid {
                libc::kill(-pid, libc::SIGKILL);
            } else {
                libc::kill(pid, libc::SIGKILL);
            }
        }
    }
    #[cfg(windows)]
    {
        let _ = std::process::Command::new("taskkill")
            .args(["/PID", &pid.to_string(), "/T", "/F"])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn();
    }
}

/// Stop a process and everything it started.
///
/// The shell is never the only process that matters: a loop forks a child per
/// iteration, and a `nohup ... &` inside the command leaves a grandchild
/// holding the output pipes. Killing only the shell left those running -- the
/// pipes stayed open past the settle window, and `job_kill` answered "running
/// for 23s" about a job it had just stopped. One signal reaches the whole
/// tree because the caller puts the child in its own process group at spawn;
/// `getpgid` guards the negative pid so a group this code did not create can
/// never be signalled. Windows reaches the same end with `taskkill /T`.
pub fn kill_tree(child: &mut tokio::process::Child) {
    if let Some(pid) = child.id() {
        kill_pid(pid);
    }
    let _ = child.start_kill();
}

/// Every background job this process knows, indexed by `<data>/jobs`.
pub struct JobRegistry {
    next: AtomicU64,
    dir: PathBuf,
    jobs: Mutex<HashMap<String, Arc<Job>>>,
}

impl JobRegistry {
    /// Open the registry over a directory, rebuilding the index it finds.
    ///
    /// Entries from earlier processes come back as jobs: their output is still
    /// on disk, and whether they run is asked of their pid. The counter starts
    /// above every id already there, so a restart never reuses `job-3` and
    /// overwrites its log.
    fn open(dir: PathBuf) -> Self {
        let mut jobs: HashMap<String, Arc<Job>> = HashMap::new();
        let mut next = 1u64;
        if let Ok(entries) = std::fs::read_dir(&dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) != Some("json") {
                    continue;
                }
                let Some(job) = load_index(&path, &dir) else {
                    continue;
                };
                if let Some(number) = job
                    .id
                    .strip_prefix("job-")
                    .and_then(|n| n.parse::<u64>().ok())
                {
                    next = next.max(number + 1);
                }
                jobs.insert(job.id.clone(), Arc::new(job));
            }
        }
        Self {
            next: AtomicU64::new(next),
            dir,
            jobs: Mutex::new(jobs),
        }
    }

    /// Start `cmd` as a job labelled with `command`.
    ///
    /// The caller builds the command (it owns shell detection); this owns the
    /// job's lifetime: the log file, the index entry, two reader tasks, and one
    /// waiter that observes the exit. `notify` marks the job so the shell's
    /// settlement hook — when one is installed — wakes the model on completion.
    pub fn spawn(&self, command: String, mut cmd: Command, notify: bool) -> Result<Arc<Job>> {
        let _ = std::fs::create_dir_all(&self.dir);
        // The job leads its own process group: everything it forks inherits
        // the group, so one signal in `kill_tree` stops the whole tree.
        #[cfg(unix)]
        cmd.process_group(0);
        let mut child = cmd
            .spawn()
            .map_err(|error| anyhow!("spawning background command: {error}"))?;
        let stdout = child.stdout.take();
        let stderr = child.stderr.take();
        // The pid goes into the index now: it is how a later process asks
        // whether this job is still running.
        let pid = child.id();

        let id = format!("job-{}", self.next.fetch_add(1, Ordering::Relaxed));
        let owner = super::current_session();
        let started_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        let log_path = self.dir.join(format!("{id}.log"));
        let log = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&log_path)
            .map_err(|error| anyhow!("opening {}: {error}", log_path.display()))?;

        let job = Arc::new(Job {
            id: id.clone(),
            command,
            owner,
            started: Instant::now(),
            started_ms,
            notify: AtomicU8::new(u8::from(notify)),
            settled: Mutex::new(None),
            killed: AtomicU8::new(0),
            readers: AtomicU8::new(0),
            child: Mutex::new(Some(child)),
            pid,
            legacy: false,
            dir: self.dir.clone(),
            log: Mutex::new(log),
            written: AtomicU64::new(0),
        });
        job.write_index();

        if let Some(pipe) = stdout {
            pump(pipe, Arc::clone(&job));
        }
        if let Some(pipe) = stderr {
            pump(pipe, Arc::clone(&job));
        }

        let waiter_job = Arc::clone(&job);
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_millis(WAIT_POLL_MS)).await;
                // Ask the process, not the record. `ask_the_process` settles
                // and writes the index entry the moment the child is gone.
                let Some(settled) = waiter_job.ask_the_process() else {
                    continue;
                };
                // Give the readers a moment to drain what the process wrote
                // before the pipes reported EOF.
                for _ in 0..50 {
                    if waiter_job.readers.load(Ordering::Relaxed) == 0 {
                        break;
                    }
                    tokio::time::sleep(Duration::from_millis(20)).await;
                }
                let label = settled.label(waiter_job.started.elapsed());
                *waiter_job.child.lock().expect("job child") = None;
                if waiter_job.notify.load(Ordering::Relaxed) == 1 {
                    if let Some(hook) = SETTLE.get() {
                        hook(&waiter_job.id, &label);
                    }
                }
                break;
            }
        });

        self.jobs.lock().expect("job registry").insert(id, Arc::clone(&job));
        Ok(job)
    }

    /// Look a job up by id.
    pub fn get(&self, id: &str) -> Option<Arc<Job>> {
        self.jobs.lock().expect("job registry").get(id).cloned()
    }

    /// Look a job up by id, but only if `session` may see it.
    pub fn get_visible(&self, id: &str, session: Option<&str>) -> Option<Arc<Job>> {
        let job = self.get(id)?;
        match session {
            None => Some(job),
            Some(s) if job.belongs_to(s) => Some(job),
            _ => None,
        }
    }

    /// Every job `session` may see (its own, plus ownerless), oldest first.
    pub fn list_visible(&self, session: Option<&str>) -> Vec<Arc<Job>> {
        let mut jobs: Vec<Arc<Job>> = self
            .jobs
            .lock()
            .expect("job registry")
            .values()
            .filter(|job| session.map_or(true, |s| job.belongs_to(s)))
            .cloned()
            .collect();
        jobs.sort_by_key(|job| job.started);
        jobs
    }

    /// Every job, oldest first.
    pub fn list(&self) -> Vec<Arc<Job>> {
        let mut jobs: Vec<Arc<Job>> = self.jobs.lock().expect("job registry").values().cloned().collect();
        jobs.sort_by_key(|job| job.started);
        jobs
    }
}

/// Rebuild one job from its index entry. `None` when the entry is unreadable.
fn load_index(path: &Path, dir: &Path) -> Option<Job> {
    let text = std::fs::read_to_string(path).ok()?;
    let value: Value = serde_json::from_str(&text).ok()?;
    let id = value.get("id")?.as_str()?.to_string();
    if !id.starts_with("job-") {
        return None;
    }
    let command = value
        .get("command")
        .and_then(|c| c.as_str())
        .unwrap_or("")
        .to_string();
    let owner = value
        .get("owner")
        .and_then(|o| o.as_str())
        .map(str::to_string);
    let pid = value
        .get("pid")
        .and_then(|p| p.as_u64())
        .map(|p| p as u32);
    let notify = value.get("notify").and_then(|n| n.as_bool()).unwrap_or(false);
    let started_ms = value
        .get("started_ms")
        .and_then(|s| s.as_u64())
        .unwrap_or(0);
    // A settled state in the record is history worth keeping; `running` is not
    // a state at all here — liveness is asked of the pid on every read.
    let settled = value
        .get("state")
        .and_then(|s| s.as_str())
        .and_then(|tag| {
            let code = value.get("exit_code").and_then(|c| c.as_i64()).map(|c| c as i32);
            JobState::from_index(tag, code)
        });
    // The log must exist for an entry to mean anything: an index whose log was
    // cleaned away is not a job anyone can read.
    let log_path = dir.join(format!("{id}.log"));
    let log = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .ok()?;
    let written = std::fs::metadata(&log_path).map(|m| m.len()).unwrap_or(0);
    // Age is reported from the wall-clock start; `Instant` cannot be built for
    // a past time, so an entry from an earlier run is anchored to now and its
    // age comes from the stored timestamp.
    let age = started_ms.max(1);
    let started = Instant::now()
        .checked_sub(Duration::from_millis(
            now_ms().saturating_sub(age).min(u64::from(u32::MAX)),
        ))
        .unwrap_or_else(Instant::now);

    Some(Job {
        id,
        command,
        owner,
        started,
        started_ms,
        notify: AtomicU8::new(u8::from(notify)),
        settled: Mutex::new(settled),
        killed: AtomicU8::new(0),
        readers: AtomicU8::new(0),
        child: Mutex::new(None),
        pid,
        legacy: true,
        dir: dir.to_path_buf(),
        log: Mutex::new(log),
        written: AtomicU64::new(written),
    })
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Where this process's jobs live: `<data>/jobs`.
///
/// Tests get their own directory: they spawn real processes, and writing those
/// logs into a user's data directory would litter it.
fn jobs_dir() -> PathBuf {
    #[cfg(test)]
    {
        std::env::temp_dir().join(format!("ngu-jobs-test-{}", std::process::id()))
    }
    #[cfg(not(test))]
    {
        crate::settings::data_dir().join("jobs")
    }
}

/// The shell process's job registry.
fn registry() -> &'static JobRegistry {
    static REGISTRY: OnceLock<JobRegistry> = OnceLock::new();
    REGISTRY.get_or_init(|| JobRegistry::open(jobs_dir()))
}

/// What runs when a job marked `notify` settles.
///
/// Installed once per process (the registry is process-wide). The CLI installs
/// nothing, where a notify flag simply does nothing.
static SETTLE: OnceLock<Arc<dyn Fn(&str, &str) + Send + Sync>> = OnceLock::new();

/// Install the hook that wakes the model when a `notify` job finishes.
///
/// Arguments are the job id and its settled state label (`exited with code 0`,
/// `killed`, …). Later installs are ignored: one shell, one hook.
pub fn set_settlement_hook(hook: Arc<dyn Fn(&str, &str) + Send + Sync>) {
    let _ = SETTLE.set(hook);
}

/// Start a prepared command as a background job.
///
/// Handed to `bash` so shell detection stays in one place.
pub fn spawn_background(command: &str, cmd: Command, notify: bool) -> Result<Arc<Job>> {
    registry().spawn(command.to_string(), cmd, notify)
}

/// Look up a job by id, restricted to the caller's session when one is pinned.
pub fn get(id: &str) -> Option<Arc<Job>> {
    registry().get_visible(id, super::current_session().as_deref())
}

/// Every job the caller's session may see (its own, plus ownerless).
pub fn list() -> Vec<Arc<Job>> {
    registry().list_visible(super::current_session().as_deref())
}

/// Register the job tools next to the base tools they extend.
pub fn install(registry: &mut ToolRegistry) -> Result<()> {
    registry.register(
        ToolDef::new(
            "job_list",
            "List background jobs: id, state, age, and command. Jobs come from \
             bash calls with background: true. State is read from the process \
             itself, so a job whose process is gone reads as finished.",
            json!({ "type": "object", "properties": {} }),
            "kernel",
            |args| Box::pin(async move { list_tool(args).await.map(Into::into) }) as ToolFuture,
        ),
        ConflictPolicy::Error,
    )?;
    registry.register(
        ToolDef::new(
            "job_output",
            "Read a background job's output and state. Set wait_ms to block until \
             it finishes (or the wait elapses) instead of reading once. Output is \
             read from the job's log file, so it survives a restart.",
            json!({
                "type": "object",
                "properties": {
                    "id": { "type": "string", "description": "Job id from bash, e.g. job-1" },
                    "wait_ms": { "type": "integer", "description": "Block up to this long for the job to finish (default 0, max 30000)" }
                },
                "required": ["id"]
            }),
            "kernel",
            |args| Box::pin(async move { output_tool(args).await.map(Into::into) }) as ToolFuture,
        ),
        ConflictPolicy::Error,
    )?;
    registry.register(
        ToolDef::new(
            "job_kill",
            "Stop a running background job.",
            json!({
                "type": "object",
                "properties": {
                    "id": { "type": "string", "description": "Job id, e.g. job-1" }
                },
                "required": ["id"]
            }),
            "kernel",
            |args| Box::pin(async move { kill_tool(args).await.map(Into::into) }) as ToolFuture,
        ),
        ConflictPolicy::Error,
    )?;
    Ok(())
}

async fn list_tool(_args: Value) -> Result<String> {
    let jobs = list();
    if jobs.is_empty() {
        return Ok("(no background jobs)".to_string());
    }
    let mut out = String::new();
    for job in jobs {
        out.push_str(&format!(
            "{}  {}  {}\n",
            job.id,
            job.state().label(job.started.elapsed()),
            truncate(&job.command, 100),
        ));
    }
    Ok(out.trim_end().to_string())
}

async fn output_tool(args: Value) -> Result<String> {
    let id = args
        .get("id")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("missing required argument: id"))?;
    let job = get(id).ok_or_else(|| anyhow!("no job '{id}' — call job_list to see the ids"))?;

    let wait_ms = args
        .get("wait_ms")
        .and_then(|v| v.as_u64())
        .unwrap_or(0)
        .min(MAX_WAIT_MS);
    let deadline = Instant::now() + Duration::from_millis(wait_ms);
    while job.is_running() && Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    let output = job.output_text();
    let state = job.state().label(job.started.elapsed());
    Ok(format!(
        "{} ({state})\nlog: {}\n{}",
        job.id,
        job.log_path().display(),
        if output.trim().is_empty() { "(no output)" } else { &output }
    ))
}

async fn kill_tool(args: Value) -> Result<String> {
    let id = args
        .get("id")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("missing required argument: id"))?;
    let job = get(id).ok_or_else(|| anyhow!("no job '{id}' — call job_list to see the ids"))?;

    if !job.is_running() {
        return Ok(format!(
            "{} already finished: {}",
            job.id,
            job.state().label(job.started.elapsed())
        ));
    }
    let state = job.kill().await;
    Ok(format!("{} {}", job.id, state.label(job.started.elapsed())))
}

/// Shorten for a one-line listing without cutting mid-character.
fn truncate(text: &str, max: usize) -> String {
    let mut out: String = text.chars().take(max).collect();
    if text.chars().count() > max {
        out.push('…');
    }
    out
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

#[cfg(test)]
mod tests {
    use super::*;

    async fn bash(args: Value) -> Result<String> {
        let registry = ToolRegistry::with_base_tools()?;
        let output = registry.execute("bash", &args.to_string()).await?;
        Ok(output.text)
    }

    async fn tool(name: &str, args: Value) -> Result<String> {
        let registry = ToolRegistry::with_base_tools()?;
        let output = registry.execute(name, &args.to_string()).await?;
        Ok(output.text)
    }

    fn id_from(text: &str) -> String {
        text.split_whitespace()
            .find(|token| token.starts_with("job-"))
            .expect("the reply names a job id")
            .to_string()
    }

    #[tokio::test]
    async fn a_background_command_outlives_the_call_that_started_it() {
        let started = bash(json!({ "command": "echo hello-from-job", "background": true }))
            .await
            .expect("background start");
        assert!(started.contains("background"), "{started}");
        let id = id_from(&started);

        let output = tool(
            "job_output",
            json!({ "id": id, "wait_ms": 10_000 }),
        )
        .await
        .expect("output");
        assert!(output.contains("hello-from-job"), "{output}");
        assert!(output.contains("exited"), "{output}");
        // The answer names where the output lives: the file *is* the output.
        assert!(output.contains("log:"), "{output}");
    }

    #[tokio::test]
    async fn output_is_on_disk_and_a_rebuilt_registry_still_reads_it() {
        let started = bash(json!({
            "command": "echo persisted-line-1 && echo persisted-line-2",
            "background": true
        }))
        .await
        .expect("background start");
        let id = id_from(&started);
        // Let it finish.
        let _ = tool("job_output", json!({ "id": &id, "wait_ms": 10_000 })).await;

        let job = get(&id).expect("the job");
        let log = job.log_path();
        assert!(log.is_file(), "the log file exists: {}", log.display());
        let text = std::fs::read_to_string(&log).expect("read the log");
        assert!(text.contains("persisted-line-1"), "{text}");
        assert!(text.contains("persisted-line-2"), "{text}");

        // A second registry over the same directory is what a restart sees:
        // the job is listed, its state is asked of the process, and its output
        // still reads from the file.
        let rebuilt = JobRegistry::open(jobs_dir());
        let recovered = rebuilt.get(&id).expect("recovered from the index");
        assert_eq!(recovered.command, job.command);
        assert!(matches!(
            recovered.state(),
            JobState::Exited(_) | JobState::Gone
        ));
        assert!(recovered.output_text().contains("persisted-line-2"));
        // And its id is never handed out again.
        assert!(
            recovered
                .id
                .strip_prefix("job-")
                .and_then(|n| n.parse::<u64>().ok())
                .map(|n| n < rebuilt.next.load(Ordering::Relaxed))
                .unwrap_or(false),
            "the counter starts above every id already on disk"
        );
    }

    #[tokio::test]
    async fn a_running_jobs_state_comes_from_the_process_not_the_record() {
        let command = if cfg!(windows) {
            "ping -n 60 127.0.0.1 >nul"
        } else {
            "sleep 60"
        };
        let started = bash(json!({ "command": command, "background": true }))
            .await
            .expect("background start");
        let id = id_from(&started);
        let job = get(&id).expect("the job");

        assert!(job.is_running(), "the process is alive, so it is running");
        // The index says running too — but the *check* is the process: kill it
        // behind the registry's back and the state must follow.
        let pid = job.pid.expect("the index records the pid");
        kill_pid(pid);
        let mut settled = false;
        for _ in 0..100 {
            if !job.is_running() {
                settled = true;
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        assert!(settled, "a killed process is not 'running'");

        let listed = tool("job_list", json!({})).await.expect("list");
        assert!(!listed.contains(&format!("{id}  running")), "{listed}");
    }

    #[tokio::test]
    async fn job_kill_stops_a_running_job() {
        // Long enough that it cannot finish first on any machine.
        let command = if cfg!(windows) {
            "ping -n 60 127.0.0.1 >nul"
        } else {
            "sleep 60"
        };
        let started = bash(json!({ "command": command, "background": true }))
            .await
            .expect("background start");
        let id = id_from(&started);

        let killed = tool("job_kill", json!({ "id": id })).await.expect("kill");
        assert!(killed.contains("killed"), "{killed}");

        let listed = tool("job_list", json!({})).await.expect("list");
        assert!(listed.contains(&format!("{id}  killed")), "{listed}");
    }

    #[tokio::test]
    async fn job_kill_reports_killed_not_a_stale_running_state() {
        // The live failure this fixes: the shell exits, a forked child keeps
        // the output pipes open, the settle window waits for them -- and
        // `kill`'s own budget expired first, answering "running for 23s"
        // about a job it had just stopped. The group is killed, the pipes
        // close, and the state settles well inside the budget.
        let command = if cfg!(windows) {
            "ping -n 60 127.0.0.1 >nul & ping -n 60 127.0.0.1 >nul"
        } else {
            "sleep 60 & sleep 60"
        };
        let started = bash(json!({ "command": command, "background": true }))
            .await
            .expect("background start");
        let id = id_from(&started);

        let killed = tool("job_kill", json!({ "id": id })).await.expect("kill");
        assert!(killed.contains("killed"), "{killed}");

        let listed = tool("job_list", json!({})).await.expect("list");
        assert!(listed.contains(&format!("{id}  killed")), "{listed}");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn job_kill_stops_grandchildren_not_just_the_shell() {
        // The second live failure: killing the shell orphaned whatever it
        // had forked (a `sleep 45` outlived `job_kill` with PPID 1, still
        // holding the pipes). The job leads its own process group, so the
        // kill reaches every member.
        let started = bash(json!({
            "command": "sleep 60 & echo grandchild=$!; wait",
            "background": true
        }))
        .await
        .expect("background start");
        let id = id_from(&started);

        // Wait for the grandchild pid to be printed, then kill the job.
        let mut pid = None;
        for _ in 0..50 {
            let out = tool("job_output", json!({ "id": &id })).await.expect("output");
            if let Some(rest) = out.split("grandchild=").nth(1) {
                pid = rest.trim().split_whitespace().next().and_then(|s| s.parse::<i32>().ok());
                if pid.is_some() {
                    break;
                }
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        let pid = pid.expect("the grandchild pid was printed");

        let killed = tool("job_kill", json!({ "id": id })).await.expect("kill");
        assert!(killed.contains("killed"), "{killed}");

        // The grandchild must be gone, not reparented and still running.
        let mut gone = false;
        for _ in 0..50 {
            // Signal 0 probes existence; ESRCH (errno 3) means reaped.
            let alive = unsafe { libc::kill(pid, 0) } == 0;
            if !alive {
                gone = true;
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        assert!(gone, "grandchild {pid} survived the job kill");
    }

    #[tokio::test]
    async fn an_unknown_job_says_so() {
        let error = tool("job_output", json!({ "id": "job-999" }))
            .await
            .expect_err("there is no such job");
        assert!(error.to_string().contains("job_list"), "{error}");
    }

    #[tokio::test]
    async fn a_notify_job_fires_the_settlement_hook_when_it_finishes() {
        let seen: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let recorder = Arc::clone(&seen);
        set_settlement_hook(Arc::new(move |id, label| {
            recorder.lock().expect("seen").push(format!("{id}|{label}"));
        }));

        let started = bash(json!({
            "command": "echo wake-test-done",
            "background": true,
            "notify": true
        }))
        .await
        .expect("background start");
        let id = id_from(&started);

        // The hook runs from the job's waiter — no waiting on job_output here:
        // waking must work even when nobody ever collects.
        let mut fired = false;
        for _ in 0..100 {
            if seen.lock().expect("seen").iter().any(|entry| entry.starts_with(&format!("{id}|"))) {
                fired = true;
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        assert!(fired, "the settlement hook never fired for {id}");

        let entry = seen
            .lock()
            .expect("seen")
            .iter()
            .find(|entry| entry.starts_with(&format!("{id}|")))
            .cloned()
            .expect("entry");
        assert!(entry.contains("exited"), "{entry}");
    }
}
