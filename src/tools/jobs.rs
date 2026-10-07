//! Background jobs: commands that outlive the step that started them.
//!
//! A `bash` command that compiles for ten minutes should not pin the turn for
//! ten minutes. Run it with `background: true` — it becomes a job, the model
//! gets an id back immediately, and `job_output` collects the result whenever
//! it wants it (optionally waiting). `job_list` shows what is running;
//! `job_kill` stops one.
//!
//! One registry per shell process. A job *is* a child process of this
//! process: its id, output, and handle all die here, so a process-global
//! registry is the honest scope — there is no second reader a per-session
//! registry would serve.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};
use serde_json::{json, Value};
use tokio::io::AsyncReadExt;
use tokio::process::Command;

use super::{ConflictPolicy, ToolDef, ToolFuture, ToolRegistry};

/// Output bytes kept per job. Past this the oldest half is dropped, so a
/// runaway printer cannot grow without bound.
const MAX_JOB_OUTPUT: usize = 256 * 1024;

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
}

impl JobState {
    /// One line a model reads instead of the enum.
    pub fn label(&self, age: Duration) -> String {
        match self {
            JobState::Running => format!("running for {}", format_age(age)),
            JobState::Exited(Some(code)) => format!("exited with code {code}"),
            JobState::Exited(None) => "exited".to_string(),
            JobState::Killed => "killed".to_string(),
        }
    }
}

/// One background command.
pub struct Job {
    /// Stable id the model uses: `job-1`, `job-2`, …
    pub id: String,
    /// The command line as it was started.
    pub command: String,
    /// When it started, for age reporting.
    pub started: Instant,
    state: Mutex<JobState>,
    output: Mutex<Vec<u8>>,
    /// Set once `job_kill` asked; the waiter reports `Killed` instead of the
    /// exit code a forced termination happens to produce.
    killed: AtomicU8,
    /// stdout and stderr readers still draining.
    readers: AtomicU8,
    /// The child itself, held for `job_kill`; never across an await.
    child: Mutex<Option<tokio::process::Child>>,
}

impl Job {
    /// The job's current state.
    pub fn state(&self) -> JobState {
        self.state.lock().expect("job state").clone()
    }

    /// Whether the job has not settled yet.
    pub fn is_running(&self) -> bool {
        matches!(self.state(), JobState::Running)
    }

    fn push_output(&self, chunk: &[u8]) {
        let mut buffer = self.output.lock().expect("job output");
        buffer.extend_from_slice(chunk);
        if buffer.len() > MAX_JOB_OUTPUT {
            let keep = MAX_JOB_OUTPUT / 2;
            let drop = buffer.len() - keep;
            buffer.drain(..drop);
        }
    }

    /// Everything the job has printed so far, as text.
    pub fn output_text(&self) -> String {
        let buffer = self.output.lock().expect("job output");
        String::from_utf8_lossy(&buffer).into_owned()
    }

    /// Stop the job and return its settled state.
    ///
    /// The kill itself is instant; the short wait is only for the waiter
    /// thread to observe the exit and record it, so the caller can report the
    /// outcome instead of "killed, probably".
    pub async fn kill(&self) -> JobState {
        self.killed.store(1, Ordering::Relaxed);
        {
            let mut guard = self.child.lock().expect("job child");
            if let Some(child) = guard.as_mut() {
                let _ = child.start_kill();
            }
        }
        for _ in 0..50 {
            if !self.is_running() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        self.state()
    }
}

/// Drain one pipe into the job’s buffer until EOF.

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

/// Every background job this shell process owns.
pub struct JobRegistry {
    next: AtomicU64,
    jobs: Mutex<HashMap<String, Arc<Job>>>,
}

impl JobRegistry {
    fn new() -> Self {
        Self {
            next: AtomicU64::new(0),
            jobs: Mutex::new(HashMap::new()),
        }
    }

    /// Start `cmd` as a job labelled with `command`.
    ///
    /// The caller builds the command (it owns shell detection); this owns the
    /// job's lifetime: two reader tasks drain the pipes into the job's buffer,
    /// and one waiter polls for exit so the state settles without anyone
    /// holding a lock across an await.
    pub fn spawn(&self, command: String, mut cmd: Command) -> Result<Arc<Job>> {
        let mut child = cmd
            .spawn()
            .map_err(|error| anyhow!("spawning background command: {error}"))?;
        let stdout = child.stdout.take();
        let stderr = child.stderr.take();

        let id = format!("job-{}", self.next.fetch_add(1, Ordering::Relaxed) + 1);
        let job = Arc::new(Job {
            id: id.clone(),
            command,
            started: Instant::now(),
            state: Mutex::new(JobState::Running),
            output: Mutex::new(Vec::new()),
            killed: AtomicU8::new(0),
            readers: AtomicU8::new(0),
            child: Mutex::new(Some(child)),
        });

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
                let exited = {
                    let mut guard = waiter_job.child.lock().expect("job child");
                    match guard.as_mut() {
                        Some(child) => match child.try_wait() {
                            Ok(Some(status)) => Some(status.code()),
                            Ok(None) => None,
                            Err(_) => Some(None),
                        },
                        None => Some(None),
                    }
                };
                let Some(code) = exited else { continue };
                // Give the readers a moment to drain what the process wrote
                // before the pipes reported EOF.
                for _ in 0..50 {
                    if waiter_job.readers.load(Ordering::Relaxed) == 0 {
                        break;
                    }
                    tokio::time::sleep(Duration::from_millis(20)).await;
                }
                let state = if waiter_job.killed.load(Ordering::Relaxed) == 1 {
                    JobState::Killed
                } else {
                    JobState::Exited(code)
                };
                *waiter_job.state.lock().expect("job state") = state;
                *waiter_job.child.lock().expect("job child") = None;
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

    /// Every job, oldest first.
    pub fn list(&self) -> Vec<Arc<Job>> {
        let mut jobs: Vec<Arc<Job>> = self.jobs.lock().expect("job registry").values().cloned().collect();
        jobs.sort_by_key(|job| job.started);
        jobs
    }
}

/// The shell process's job registry.
fn registry() -> &'static JobRegistry {
    static REGISTRY: OnceLock<JobRegistry> = OnceLock::new();
    REGISTRY.get_or_init(JobRegistry::new)
}

/// Start a prepared command as a background job.
///
/// Handed to `bash` so shell detection stays in one place.
pub fn spawn_background(command: &str, cmd: Command) -> Result<Arc<Job>> {
    registry().spawn(command.to_string(), cmd)
}

/// Look up a job by id.
pub fn get(id: &str) -> Option<Arc<Job>> {
    registry().get(id)
}

/// Every job, oldest first.
pub fn list() -> Vec<Arc<Job>> {
    registry().list()
}

/// Register the job tools next to the base tools they extend.
pub fn install(registry: &mut ToolRegistry) -> Result<()> {
    registry.register(
        ToolDef::new(
            "job_list",
            "List background jobs: id, state, age, and command. Jobs come from \
             bash calls with background: true.",
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
             it finishes (or the wait elapses) instead of reading once.",
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
        "{} ({state})\n{}",
        job.id,
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
    async fn an_unknown_job_says_so() {
        let error = tool("job_output", json!({ "id": "job-999" }))
            .await
            .expect_err("there is no such job");
        assert!(error.to_string().contains("job_list"), "{error}");
    }
}
