//! Compaction: reducing history when it approaches the window.
//!
//! Built into the kernel, not a plugin, because a conversation that cannot be
//! continued is a broken kernel rather than a missing feature. What *is* a
//! seam is how the summary is written — see [`Summarizer`](crate::compaction::Summarizer).
//!
//! # What the kernel guarantees
//!
//! Two invariants hold no matter which summarizer is installed, because the
//! kernel owns the mechanics:
//!
//! * **Tool pairs stay together.** A `tool` result is only valid next to the
//!   assistant message that requested it. Splitting them produces a request the
//!   provider rejects, so the split point is always moved to a user message.
//! * **The newest messages are never touched.** Compaction only ever removes a
//!   leading run of messages; what the user just said survives verbatim.
//!
//! # How it is recorded
//!
//! A compaction is written to the session as its own record rather than by
//! rewriting the file. The store is append-only precisely so a crash cannot
//! corrupt it, and rewriting would give that up. On load, a compaction record
//! replaces the run it covers, so a reloaded session matches what the model saw.
//!
//! # How the record finds its run again
//!
//! A count alone — "the first N messages" — says where the run *started*, not
//! which messages it held. N is exactly what a missing line changes, and a
//! session file can lose a line for reasons this program never sees: an
//! interrupted write, a hand edit, a merge between machines, an unparsable line
//! skipped by the reader. So a record also stores [`anchors_of`](crate::compaction::anchors_of) the run it
//! replaced, and the store finds the span by those ids. When they are gone, the
//! store says so instead of folding at the old offset. The count stays as well:
//! it is the fallback for records written before anchors existed, and it is the
//! check the anchors are measured against.
//!
//! # Layout
//!
//! One file per function: `record` (the compaction and its summary message),
//! `anchor` (finding the replaced run), `plan` (what to summarize), `summarize`
//! (the replaceable summarizer), `apply` (rewriting a history) and `prune`
//! (thinning old tool output). Every item this module exported before is
//! re-exported here, so `crate::compaction::X` still names the same item.

mod anchor;
mod apply;
mod plan;
mod prune;
mod record;
mod summarize;

pub use anchor::{anchor_of, anchors_of, locate_span, ANCHOR_HEX};
pub use apply::{apply, replay};
pub use plan::{plan, Plan, MIN_SUMMARIZE};
pub use prune::{prune_images, prune_tool_results, PrunePolicy};
pub use record::Compaction;
pub use summarize::{summarize, summary_request, ModelSummarizer, Summarizer};
