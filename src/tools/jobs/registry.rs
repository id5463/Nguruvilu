//! The job registry: every background job this process knows, and its index.
//!
//! `<data>/jobs/<id>.json` records what a job *is* — command, pid, owner,
//! start time, log path — and the last state anyone observed. The in-memory map
//! is only a cache of that index, rebuilt from it when the process starts: a
//! job started before a restart still lists, and its output still reads.
//!
//! Whether it is *running* is never taken from the record — it is asked of the
//! process (the child handle when this process owns it, the pid otherwise), so
//! a job whose process died reads as finished with no bookkeeping required.
//!
//! This is also where the three job tools are registered, and where their
//! replies are written.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, Result};
use serde_json::{json, Value};
use tokio::process::Command;

use crate::tools::{budgets, current_session, ConflictPolicy, ToolDef, ToolFuture, ToolRegistry};

use super::model::{pump, Job, JobState};
use super::process::{program_image, KillOutcome};

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
        let owner = current_session();
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
    registry().get_visible(id, current_session().as_deref())
}

/// Every job the caller's session may see (its own, plus ownerless).
pub fn list() -> Vec<Arc<Job>> {
    registry().list_visible(current_session().as_deref())
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
                    "wait_ms": { "type": "integer", "description": "Block up to this long for the job to finish (default 0, max 30000 unless the settings say otherwise: `job_wait_secs`)" }
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
        .min(budgets().job_wait_max_ms);
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
