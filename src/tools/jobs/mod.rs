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
//!
//! # Layout
//!
//! One file per function:
//!
//! * [`model`] — what a job is ([`Job`], [`JobState`]): its log, its index
//!   entry, and the answer "state is asked of the process".
//! * [`process`] — the OS side ([`verify_target`], [`kill_pid`],
//!   [`ProcessTree`]): proving a pid is still ours, and stopping a tree.
//! * [`registry`] — [`JobRegistry`], the index on disk, and the three tools
//!   (`job_list`, `job_output`, `job_kill`).
//!
//! Every item this module exported before is re-exported here, so
//! `crate::tools::jobs::X` still names the same item.

mod model;
mod process;
mod registry;

pub use model::{Job, JobState};
pub use process::{kill_tree, KillOutcome, ProcessTree};
pub use registry::{get, install, list, set_settlement_hook, spawn_background, JobRegistry};
