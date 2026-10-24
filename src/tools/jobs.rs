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

/// Longest a waiter waits between two looks at its child. Short jobs settle on
/// the first poll; a job that runs for hours is not polled 36 000 times an hour.
const WAIT_POLL_MAX_MS: u64 = 1_000;

/// How long a job's waiter keeps watching before it gives up and exits.
///
/// The waiter exists to fire the settlement hook and to release the child
/// handle. Without a bound, every job — including one whose process outlives
/// the shell by days — leaked a polling task for the entire life of the
/// process. Six hours is far past any background command a turn waits on, and
/// a job that outruns it keeps working: liveness is asked of the process on
/// every read, never taken from the waiter's bookkeeping.
const WAITER_LIFETIME: Duration = Duration::from_secs(6 * 60 * 60);

/// How far a pid's creation time may sit from a job's recorded start before
/// the pid is treated as recycled. The two are milliseconds apart in practice
/// (the job stamps its start right after the spawn); the tolerance is for
/// clock granularity, and it is far narrower than the window a busy machine
/// needs to hand the same pid to another process.
/// `#[cfg(windows)]` so the constant is never read on Unix builds: the Unix
/// verifier takes the child handle's word for it and needs no clock at all.
#[cfg_attr(not(windows), allow(dead_code))]
const PID_START_TOLERANCE_MS: u64 = 60_000;

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
    /// The process image this job started, lowercased and without extension
    /// (`cmd`, `bash`), recorded at spawn.
    ///
    /// It is the identity half of a kill: a pid read off the index may have
    /// been recycled by the OS, and only a fingerprint recorded when the job
    /// started can tell "that process" from "whatever holds pid 4321 now".
    image: Option<String>,
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
        if let Some(settled) = self.settled.lock().unwrap_or_else(|e| e.into_inner()).clone() {
            return settled;
        }
        self.ask_the_process().unwrap_or(JobState::Running)
    }

    /// Ask reality: the child handle when this process owns it, the pid
    /// otherwise. `None` means still running.
    fn ask_the_process(&self) -> Option<JobState> {
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
    fn write_index(&self) {
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

/// How a kill request ended.
#[derive(Debug, Clone)]
pub enum KillOutcome {
    /// The process was signalled (or had already ended); this is the state it
    /// settled into.
    Stopped(JobState),
    /// **Nothing was signalled.** The pid could not be proven to still be the
    /// process this job started, and a recycled pid belongs to a stranger.
    Refused(String),
}

/// Prove that `pid` is still the process this job started, before signalling
/// it. `Err` carries the reason the kill must not happen.
///
/// Windows: this process may have no handle on the job (the entry was rebuilt
/// from the index after a restart), so identity comes from what the job
/// recorded at spawn — the image name — cross-checked against the OS's own
/// answer for that pid, plus its creation time, which is what actually tells a
/// recycled pid apart from the original. Anything unreadable is a refusal:
/// failing closed leaves a stray process running, failing open kills a
/// stranger's tree.
///
/// Unix: the recorded pid is signalled directly, so a recycled pid's blast
/// radius is one process rather than a tree, and [`kill_pid`] never upgrades a
/// pid to a process group unless that pid leads the group.
fn verify_target(job: &Job, pid: u32) -> Result<(), String> {
    #[cfg(unix)]
    {
        let _ = (job, pid);
        Ok(())
    }
    #[cfg(windows)]
    {
        let recorded = job.image.as_deref().unwrap_or_default();
        if recorded.is_empty() {
            return Err(format!(
                "job '{}' recorded no process image when it started, so pid {pid} cannot \
                 be proven to be its process; nothing was signalled",
                job.id
            ));
        }
        let Some(current) = probe_process(pid) else {
            return Err(format!(
                "pid {pid} is not running any more, so there is nothing to stop; \
                 nothing was signalled"
            ));
        };
        if !current.image.eq_ignore_ascii_case(recorded) {
            return Err(format!(
                "pid {pid} is now '{}', not the '{recorded}' this job started — the pid \
                 was recycled; nothing was signalled",
                current.image
            ));
        }
        match current.created_ms {
            Some(created) => {
                let delta = created.abs_diff(job.started_ms);
                if delta > PID_START_TOLERANCE_MS {
                    return Err(format!(
                        "pid {pid} is called '{}' but started {delta} ms away from this \
                         job's recorded start, so it is not provably the same process; \
                         nothing was signalled",
                        current.image
                    ));
                }
            }
            None => {
                return Err(format!(
                    "the start time of pid {pid} could not be read, so it cannot be \
                     proven to be this job's process; nothing was signalled"
                ))
            }
        }
        Ok(())
    }
}

/// What the OS says a pid is right now.
#[cfg(windows)]
#[derive(Debug, Clone)]
struct ProcIdentity {
    /// Image name as Windows reports it, lowercased and extension-less.
    image: String,
    /// Creation time in ms since the Unix epoch, when it could be read.
    created_ms: Option<u64>,
}

/// Ask Windows what a pid is: its image name and its creation time.
///
/// `Get-Process` is the only way to answer both without a Win32 dependency in
/// this crate. It is called on the kill path only — never in a poll loop — and
/// an unanswerable question returns `None`, which the caller turns into a
/// refusal rather than a blind kill.
#[cfg(windows)]
fn probe_process(pid: u32) -> Option<ProcIdentity> {
    let script = format!(
        "$p = Get-Process -Id {pid} -ErrorAction Stop; \
         $t = 0; try {{ $t = $p.StartTime.ToFileTimeUtc() }} catch {{ }}; \
         Write-Output ($p.ProcessName + '|' + $t)"
    );
    let output = std::process::Command::new("powershell")
        .args(["-NoProfile", "-NonInteractive", "-Command", &script])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&output.stdout);
    let (image, created) = text.trim().split_once('|')?;
    let image = image.trim().trim_end_matches(".exe").to_lowercase();
    if image.is_empty() {
        return None;
    }
    let created_ms = match created.trim().parse::<u64>() {
        Ok(filetime) if filetime > 0 => Some(filetime_to_ms(filetime)),
        _ => None,
    };
    Some(ProcIdentity { image, created_ms })
}

/// Windows FILETIME (100ns ticks since 1601-01-01) to ms since the Unix epoch.
#[cfg(windows)]
fn filetime_to_ms(filetime: u64) -> u64 {
    /// 100ns ticks between 1601-01-01 and 1970-01-01.
    const EPOCH_DIFF: u64 = 116_444_736_000_000_000;
    filetime
        .checked_sub(EPOCH_DIFF)
        .map(|ticks| ticks / 10_000)
        .unwrap_or(0)
}

/// The image name `cmd` will start, as Windows reports it: lowercased, no
/// extension. `None` when the command names nothing usable.
fn program_image(cmd: &Command) -> Option<String> {
    let program = cmd.as_std().get_program().to_string_lossy().to_lowercase();
    let stem = Path::new(&program)
        .file_stem()?
        .to_string_lossy()
        .trim_end_matches(".exe")
        .to_string();
    (!stem.is_empty()).then_some(stem)
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

// ------------------------------------------------------ the command's tree

/// One command's whole process tree, held as a single unit so "stop
/// everything this command started" is one call.
///
/// Unix needs nothing here: `process_group(0)` at spawn plus the negative-pid
/// signal in `kill_pid` already stop the group.
///
/// Windows needs it, and not only for tidiness. `taskkill /T` walks a
/// parent/child snapshot *asynchronously*, so a grandchild can still be alive
/// when the call returns — and a live grandchild usually holds an inherited
/// copy of the command's output pipes. A pipe with a writer left open never
/// reports EOF, so the pump's in-flight `read` cannot finish; aborting the task
/// does not cancel a read the OS has already been handed. That is the real
/// shape of the "60-second tail": the turn returns at its deadline while the
/// runtime teardown waits for the descendant's own lifetime (observed: 30+
/// minutes of `CPU=0` on a test binary, 105 minutes on a real turn).
///
/// A Job Object terminated with one call closes every write end at the same
/// instant, so those reads complete immediately.
#[cfg(unix)]
pub struct ProcessTree;

#[cfg(unix)]
impl ProcessTree {
    /// Nothing to adopt: the process group came with `process_group(0)`, and a
    /// no-op keeps the caller free of `cfg` noise.
    pub fn adopt(_pid: Option<u32>) -> Option<Self> {
        None
    }

    /// Unreachable: [`ProcessTree::adopt`] is always `None` on Unix.
    pub fn terminate(&self) -> bool {
        false
    }

    /// Nothing to hand back.
    pub fn release(self) {}
}

#[cfg(windows)]
pub use job_object::ProcessTree;

#[cfg(windows)]
mod job_object {
    //! The minimum of `kernel32` needed to own one process tree.
    //!
    //! Six declarations beat a `windows-sys` dependency that would enlarge every
    //! build for one feature; `libc` does not exist on Windows, so this is a raw
    //! `extern "system"` block.

    use std::ffi::c_void;

    type Handle = *mut c_void;

    /// `JOBOBJECT_EXTENDED_LIMIT_INFORMATION`'s class id.
    const EXTENDED_LIMIT_INFORMATION_CLASS: u32 = 9;
    /// Kill every process in the job when its last handle closes.
    const LIMIT_KILL_ON_JOB_CLOSE: u32 = 0x0000_2000;
    /// Rights needed for the two job operations below.
    const PROCESS_TERMINATE: u32 = 0x0001;
    const PROCESS_SET_QUOTA: u32 = 0x0100;

    #[link(name = "kernel32")]
    #[allow(non_snake_case)]
    extern "system" {
        fn CreateJobObjectW(attributes: *mut c_void, name: *const u16) -> Handle;
        fn SetInformationJobObject(
            job: Handle,
            class: u32,
            info: *const c_void,
            length: u32,
        ) -> i32;
        fn AssignProcessToJobObject(job: Handle, process: Handle) -> i32;
        fn TerminateJobObject(job: Handle, exit_code: u32) -> i32;
        fn OpenProcess(access: u32, inherit: i32, pid: u32) -> Handle;
        fn CloseHandle(handle: Handle) -> i32;
    }

    /// `JOBOBJECT_BASIC_LIMIT_INFORMATION`, field for field. Only `LimitFlags`
    /// is ever written; the rest exists so the OS reads the size it expects.
    #[repr(C)]
    #[derive(Default)]
    #[allow(dead_code)]
    struct BasicLimitInformation {
        per_process_user_time_limit: i64,
        per_job_user_time_limit: i64,
        limit_flags: u32,
        minimum_working_set_size: usize,
        maximum_working_set_size: usize,
        active_process_limit: u32,
        affinity: usize,
        priority_class: u32,
        scheduling_class: u32,
    }

    /// `IO_COUNTERS`, field for field (layout only).
    #[repr(C)]
    #[derive(Default)]
    #[allow(dead_code)]
    struct IoCounters {
        read_operation_count: u64,
        write_operation_count: u64,
        other_operation_count: u64,
        read_transfer_count: u64,
        write_transfer_count: u64,
        other_transfer_count: u64,
    }

    /// `JOBOBJECT_EXTENDED_LIMIT_INFORMATION`, field for field (layout only).
    #[repr(C)]
    #[derive(Default)]
    #[allow(dead_code)]
    struct ExtendedLimitInformation {
        basic_limit_information: BasicLimitInformation,
        io_info: IoCounters,
        process_memory_limit: usize,
        job_memory_limit: usize,
        peak_process_memory_used: usize,
        peak_job_memory_used: usize,
    }

    /// A job holding one command's tree.
    ///
    /// Adopt **immediately** after `spawn`: descendants inherit the job as they
    /// are created, so a tree that grows before the assignment leaves those
    /// early processes outside it. The window is the microseconds between
    /// `CreateProcess` returning and this call — far shorter than the time a
    /// shell needs to fork anything — and a straggler is still covered by the
    /// caller's bounded wait, so it costs latency, not correctness.
    pub struct ProcessTree(Handle);

    // A Win32 kernel handle is an opaque, process-wide value: it names a
    // reference-counted kernel object and carries no thread affinity, so moving
    // the guard to another thread is not just allowed, it is what the handle is
    // for. The alternative — keeping the guard out of every async state machine
    // — is impossible here: the whole point is to still hold it *after* the
    // awaits that waited for the command, so that the tree can be stopped.
    //
    // `&ProcessTree` is shared-safe for the same reason: `TerminateJobObject`
    // and `SetInformationJobObject` are thread-safe on a handle the caller keeps
    // open, and no thread can close it behind another's back because closing
    // happens in `Drop`, which needs ownership.
    unsafe impl Send for ProcessTree {}
    unsafe impl Sync for ProcessTree {}

    impl ProcessTree {
        /// Create a kill-on-close job and put `pid` in it.
        ///
        /// `None` when the platform will not cooperate (no job, no rights, the
        /// process already inside a job that forbids nesting): the caller falls
        /// back to signalling, exactly as before this existed.
        pub fn adopt(pid: Option<u32>) -> Option<Self> {
            let pid = pid?;
            unsafe {
                let job = CreateJobObjectW(std::ptr::null_mut(), std::ptr::null());
                if job.is_null() {
                    return None;
                }
                let mut tree = ProcessTree(job);
                // Set before any member exists: from here on, losing the handle
                // (panic, early return, process death) takes the tree with it.
                if !tree.set_kill_on_close(true) {
                    return None;
                }
                let process = OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE, 0, pid);
                if process.is_null() {
                    return None;
                }
                let assigned = AssignProcessToJobObject(job, process) != 0;
                CloseHandle(process);
                if !assigned {
                    return None;
                }
                Some(tree)
            }
        }

        /// Stop every process in the job at once — descendants included, and
        /// without walking a snapshot that is stale the moment it is taken.
        pub fn terminate(&self) -> bool {
            unsafe { TerminateJobObject(self.0, 1) != 0 }
        }

        /// Hand the tree back unharmed.
        ///
        /// Kill-on-close is cleared first, so dropping the handle no longer
        /// stops members: a `nohup ... &` descendant is deliberate work that
        /// must outlive the command that started it.
        pub fn release(mut self) {
            let _ = self.set_kill_on_close(false);
            // The handle is closed by `Drop`.
        }

        fn set_kill_on_close(&mut self, kill: bool) -> bool {
            let mut info = ExtendedLimitInformation::default();
            if kill {
                info.basic_limit_information.limit_flags = LIMIT_KILL_ON_JOB_CLOSE;
            }
            unsafe {
                SetInformationJobObject(
                    self.0,
                    EXTENDED_LIMIT_INFORMATION_CLASS,
                    &info as *const _ as *const c_void,
                    std::mem::size_of::<ExtendedLimitInformation>() as u32,
                ) != 0
            }
        }
    }

    impl Drop for ProcessTree {
        /// Exception safety: while kill-on-close is still set, this is the
        /// difference between an abandoned tree and a leaked one — an unwind
        /// past the guard stops the command's processes instead of leaving them
        /// running with nobody holding their pipes.
        fn drop(&mut self) {
            if !self.0.is_null() {
                unsafe {
                    CloseHandle(self.0);
                }
                self.0 = std::ptr::null_mut();
            }
        }
    }
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
        // question and this runs only when a job is inspected. `contains(pid)`
        // on the whole output false-positives on a memory column ("4,1234 K"),
        // so the pid is matched as a whole field and the "no tasks match"
        // sentinel is believed when it appears.
        std::process::Command::new("tasklist")
            .args(["/FI", &format!("PID eq {pid}"), "/NH"])
            .output()
            .map(|output| {
                let text = String::from_utf8_lossy(&output.stdout).to_lowercase();
                if text.contains("no tasks") {
                    return false;
                }
                let pid = pid.to_string();
                text.split_whitespace().any(|field| field == pid)
            })
            .unwrap_or(false)
    }
}

/// Stop a pid and the group it leads — for a job whose child handle this
/// process does not hold (one rebuilt from the index), and for the tree sweep
/// after a timeout.
///
/// **Callers must have proven the pid still means the process they want gone.**
/// This function only signals; it does not ask whose pid it is. `kill_tree`
/// gets that proof from the child handle it holds, and `Job::kill` from
/// [`verify_target`] before it gets here.
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
        // `taskkill /T /F` walks the tree, so it must be *waited for*: the old
        // code spawned it and moved on, which both leaked the process handle
        // and let `job_kill` answer "killed" before anything had been killed.
        let spawned = std::process::Command::new("taskkill")
            .args(["/PID", &pid.to_string(), "/T", "/F"])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn();
        let Ok(mut killer) = spawned else { return };
        // A taskkill that stops answering is abandoned, not waited on for ever:
        // the caller's settle window judges the outcome from the process itself
        // either way.
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            match killer.try_wait() {
                Ok(Some(_)) => break,
                Ok(None) if Instant::now() < deadline => {
                    std::thread::sleep(Duration::from_millis(20));
                }
                _ => {
                    let _ = killer.kill();
                    break;
                }
            }
        }
        // Reap it whatever happened above: an un-waited child is a handle that
        // outlives the kill that made it.
        let _ = killer.wait();
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
        // What the OS will call this process, recorded before it exists so a
        // later `job_kill` from another process can prove a pid is still it.
        let image = program_image(&cmd);
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
            image,
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

        // The waiter is deliberately *detached*: it holds an `Arc<Job>`, so
        // storing its handle on the job would be a reference cycle keeping
        // every job alive forever, and storing it in a registry-wide list
        // would be just as unbounded. What bounds it instead is its own
        // lifetime — `WAITER_LIFETIME` — after which it takes one last look
        // and exits, so a single job cannot leave a poller running for the
        // rest of the process's life.
        let waiter_job = Arc::clone(&job);
        tokio::spawn(async move {
            let watching = Instant::now();
            let mut interval = Duration::from_millis(WAIT_POLL_MS);
            loop {
                tokio::time::sleep(interval).await;
                // Ask the process, not the record. `ask_the_process` settles
                // and writes the index entry the moment the child is gone.
                if let Some(settled) = waiter_job.ask_the_process() {
                    announce_settling(&waiter_job, settled).await;
                    break;
                }
                if watching.elapsed() >= WAITER_LIFETIME {
                    // Past its usefulness. One last probe, then self-cleanup:
                    // the job keeps running, and `job_list`/`job_output` still
                    // read the truth from the process itself.
                    if let Some(settled) = waiter_job.ask_the_process() {
                        announce_settling(&waiter_job, settled).await;
                    }
                    break;
                }
                // Back off: a short job settles on the first poll, a
                // hours-long one should not be polled every 100 ms.
                interval = (interval * 2).min(Duration::from_millis(WAIT_POLL_MAX_MS));
            }
        });

        self.jobs
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(id, Arc::clone(&job));
        Ok(job)
    }

    /// Look a job up by id.
    pub fn get(&self, id: &str) -> Option<Arc<Job>> {
        self.jobs.lock().unwrap_or_else(|e| e.into_inner()).get(id).cloned()
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
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .filter(|job| session.map_or(true, |s| job.belongs_to(s)))
            .cloned()
            .collect();
        jobs.sort_by_key(|job| job.started);
        jobs
    }

    /// Every job, oldest first.
    pub fn list(&self) -> Vec<Arc<Job>> {
        let mut jobs: Vec<Arc<Job>> = self
            .jobs
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .cloned()
            .collect();
        jobs.sort_by_key(|job| job.started);
        jobs
    }
}

/// What a waiter does the moment it sees its job end.
///
/// Dropping the child handle releases the OS process object; the settlement
/// hook — when the job asked for one — wakes the model. The readers get a
/// moment first, so a job's last line is in its log before the wake-up names
/// it as finished.
async fn announce_settling(job: &Arc<Job>, settled: JobState) {
    for _ in 0..50 {
        if job.readers.load(Ordering::Relaxed) == 0 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let label = settled.label(job.started.elapsed());
    *job.child.lock().unwrap_or_else(|e| e.into_inner()) = None;
    if job.notify.load(Ordering::Relaxed) == 1 {
        if let Some(hook) = SETTLE.get() {
            hook(&job.id, &label);
        }
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
    // The identity half of a kill. An index written by an older version has
    // none, and a pid without one is never signalled (see `verify_target`).
    let image = value
        .get("image")
        .and_then(|i| i.as_str())
        .map(|i| i.trim().trim_end_matches(".exe").to_lowercase())
        .filter(|i| !i.is_empty());
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
        image,
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
    match job.kill().await {
        KillOutcome::Stopped(state) => {
            let label = state.label(job.started.elapsed());
            if matches!(state, JobState::Running) {
                // Asked for a kill and the process is still there: say that
                // instead of answering "killed" about a live process.
                Ok(format!(
                    "{} did not stop: {label}. Nothing else was attempted — \
                     check the log with job_output, or stop it by hand.\nlog: {}",
                    job.id,
                    job.log_path().display()
                ))
            } else {
                Ok(format!("{} {label}", job.id))
            }
        }
        KillOutcome::Refused(reason) => Ok(format!(
            "{} was NOT killed: {reason}\n\
             Nothing was signalled, so if its process is really still running you \
             will need to stop it by hand (the log is at {}).",
            job.id,
            job.log_path().display()
        )),
    }
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

    /// A scratch index directory no other test shares.
    fn temp_dir(label: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("ngu-jobs-{label}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("scratch dir");
        dir
    }

    /// One index entry, shaped like the record a previous process leaves.
    fn write_index(dir: &Path, id: &str, pid: u32, started_ms: u64, image: &str) {
        let payload = json!({
            "id": id,
            "command": "someone else's process",
            "pid": pid,
            "image": image,
            "owner": Value::Null,
            "started_ms": started_ms,
            "notify": false,
            "log": dir.join(format!("{id}.log")).display().to_string(),
            "state": "running",
            "exit_code": Value::Null,
        });
        std::fs::write(dir.join(format!("{id}.json")), payload.to_string()).expect("index entry");
    }

    /// A pid read off the index is a claim, not proof. This job claims a pid
    /// that belongs to a process nobody here started: `job_kill` must refuse,
    /// say so, and leave the victim alone — `taskkill /T /F` on a recycled pid
    /// kills a stranger's whole tree.
    #[cfg(windows)]
    #[tokio::test]
    async fn a_pid_that_cannot_be_proven_is_never_killed() {
        let mut victim = std::process::Command::new("ping")
            .args(["-n", "30", "127.0.0.1"])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .expect("a live process to impersonate");
        let pid = victim.id();

        let dir = temp_dir("verify-refuse");
        write_index(&dir, "job-900", pid, now_ms(), "definitely-not-this");
        let registry = JobRegistry::open(dir.clone());
        let job = registry.get("job-900").expect("rebuilt from the index");
        assert!(job.is_running(), "the foreign pid is alive");

        match job.kill().await {
            KillOutcome::Refused(reason) => {
                assert!(
                    reason.contains("signalled") || reason.contains("proven"),
                    "a refusal explains itself: {reason}"
                );
            }
            KillOutcome::Stopped(state) => panic!("must refuse, not kill: {state:?}"),
        }
        assert!(process_alive(pid), "the stranger is untouched");
        assert!(job.is_running(), "a refused kill changes nothing");

        let _ = victim.kill();
        let _ = victim.wait();
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The same path when the identity *does* check out: the recorded image and
    /// the process's own creation time agree, so the pid is stopped and the
    /// state that comes back is `killed` — not a hopeful "killed, probably".
    #[cfg(windows)]
    #[tokio::test]
    async fn a_proven_pid_is_killed_and_reported_killed() {
        let mut victim = std::process::Command::new("ping")
            .args(["-n", "60", "127.0.0.1"])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .expect("a process to stop");
        let pid = victim.id();
        // A machine that cannot answer the identity question at all must
        // refuse (the test above) — that is correct behaviour, not a failure.
        if probe_process(pid).is_none() {
            let _ = victim.kill();
            let _ = victim.wait();
            return;
        }

        let dir = temp_dir("verify-kill");
        write_index(&dir, "job-901", pid, now_ms(), "ping");
        let registry = JobRegistry::open(dir.clone());
        let job = registry.get("job-901").expect("rebuilt from the index");

        match job.kill().await {
            KillOutcome::Stopped(state) => assert_eq!(state, JobState::Killed),
            KillOutcome::Refused(reason) => panic!("a provable pid must be killed: {reason}"),
        }
        assert!(!process_alive(pid), "the recorded process is gone");
        let _ = victim.wait();
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A job's index now carries the image it started, so a kill from a later
    /// process has something to compare a pid against.
    #[tokio::test]
    async fn the_index_records_the_process_image_the_kill_will_check() {
        let started = bash(json!({ "command": "echo identify-me", "background": true }))
            .await
            .expect("background start");
        let id = id_from(&started);
        let _ = tool("job_output", json!({ "id": &id, "wait_ms": 10_000 })).await;

        let path = jobs_dir().join(format!("{id}.json"));
        let text = std::fs::read_to_string(&path).expect("the index entry");
        let value: Value = serde_json::from_str(&text).expect("json");
        let image = value.get("image").and_then(|i| i.as_str()).unwrap_or("");
        assert!(!image.is_empty(), "the image is recorded: {text}");
        assert_eq!(image, image.to_lowercase(), "normalized: {text}");
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
