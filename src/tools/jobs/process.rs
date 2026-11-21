//! The process side of a job: what a pid is, and how to stop it.
//!
//! Nothing here knows about logs, owners or tool calls. It answers two
//! questions and nothing else: *is this pid still the process we started*, and
//! *stop that process and everything it forked*.
//!
//! Both questions are asked of the OS rather than of our records, because a pid
//! is recycled and a record is only a claim. A kill that cannot be proven to
//! have the right target is refused — failing closed leaves a stray process
//! running; failing open kills a stranger's whole tree.

use std::path::Path;

use anyhow::Result;
use tokio::process::Command;

#[cfg(windows)]
use std::time::{Duration, Instant};

use super::model::{Job, JobState};

/// How far a pid's creation time may sit from a job's recorded start before
/// the pid is treated as recycled. The two are milliseconds apart in practice
/// (the job stamps its start right after the spawn); the tolerance is for
/// clock granularity, and it is far narrower than the window a busy machine
/// needs to hand the same pid to another process.
/// `#[cfg(windows)]` so the constant is never read on Unix builds: the Unix
/// verifier takes the child handle's word for it and needs no clock at all.
#[cfg_attr(not(windows), allow(dead_code))]
const PID_START_TOLERANCE_MS: u64 = 60_000;

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
pub(super) fn verify_target(job: &Job, pid: u32) -> Result<(), String> {
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
pub(super) fn probe_process(pid: u32) -> Option<ProcIdentity> {
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
pub(super) fn filetime_to_ms(filetime: u64) -> u64 {
    /// 100ns ticks between 1601-01-01 and 1970-01-01.
    const EPOCH_DIFF: u64 = 116_444_736_000_000_000;
    filetime
        .checked_sub(EPOCH_DIFF)
        .map(|ticks| ticks / 10_000)
        .unwrap_or(0)
}

/// The image name `cmd` will start, as Windows reports it: lowercased, no
/// extension. `None` when the command names nothing usable.
pub(super) fn program_image(cmd: &Command) -> Option<String> {
    let program = cmd.as_std().get_program().to_string_lossy().to_lowercase();
    let stem = Path::new(&program)
        .file_stem()?
        .to_string_lossy()
        .trim_end_matches(".exe")
        .to_string();
    (!stem.is_empty()).then_some(stem)
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
pub(super) fn process_alive(pid: u32) -> bool {
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
pub(super) fn kill_pid(pid: u32) {
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
