//! The `documents` plugin: readers that teach `read` PDF and Office formats.
//!
//! Available to the kernel as `builtin:documents`; a pack declaring it turns
//! document reading on (**读不了就装包** — the pack provides the format, the
//! kernel only provides the asking). Two readers:
//!
//! * **PDF** — the machine's Python engines first (PyMuPDF carries the font
//!   tables a CJK PDF needs; markitdown behind it). No PDF is guessed at
//!   natively: a hand-rolled parser showing confident garbage is worse than a
//!   command that installs the real thing.
//! * **Office** (`pptx`/`docx`/`xlsx`) — engines first, then the built-in
//!   zip+XML walk, so a fresh machine reads a deck with nothing installed at
//!   all (that walk is the offline floor; engines remain the quality path).
//!
//! # Failure is an instruction (缺环境就自己装)
//!
//! Every unreadable outcome names the command that fixes the machine: install
//! Python, install the engines, or "the engines ran and found no text" (an
//! image-scanned PDF needs OCR, which is honestly out of scope). The model has
//! `bash` and no gate to ask — one command, then retry the read.

use std::path::Path;
use std::sync::Arc;

use crate::plugin::{Contributions, Kernel, Plugin, PluginCtx};
use crate::tools::readers::{ReadOutcome, ReaderSpec};

/// The plugin's name: what a pack asks for as `builtin:documents`.
pub const PLUGIN: &str = "documents";

/// Make the readers available: a pack's assembly asking for
/// `builtin:documents` decides whether a conversation can read documents.
pub fn define(kernel: &mut Kernel) {
    kernel.define(Arc::new(Documents));
}

struct Documents;

impl Plugin for Documents {
    fn name(&self) -> &str {
        PLUGIN
    }

    fn apply(&self, _ctx: &PluginCtx) -> anyhow::Result<Contributions> {
        Ok(Contributions::new()
            .reader(ReaderSpec {
                id: "pdf".into(),
                claims: Arc::new(|bytes, _| bytes.starts_with(b"%PDF")),
                convert: Arc::new(|_, path| pdf_read(path)),
            })
            .reader(ReaderSpec {
                id: "office".into(),
                claims: Arc::new(|bytes, path| is_office_zip(bytes, path)),
                convert: Arc::new(|bytes, path| office_read(bytes, path)),
            }))
    }
}

/// The install-command cards. Shown verbatim on failure — the whole point
/// is that the model can copy one into `bash` and the next read works.
pub const PYTHON_MISSING: &str = "Python is not installed on this machine. \
    Install it with one command — Windows: `winget install Python.Python.3.13`; \
    Linux: `sudo apt install -y python3`; macOS: `brew install python` — \
    then retry this read.";
pub const ENGINES_MISSING: &str = "Python is present but the document engines \
    are not. Run `python -m pip install --user pymupdf markitdown` via bash, \
    then retry this read.";
pub const NO_TEXT_FOUND: &str = "the engines ran but produced no text \
    (image-scanned PDFs need OCR — that is out of this reader's scope), and \
    the built-in fallback found nothing readable either.";

/// Magic + name: the zip signature AND a container markitdown reads. A plain
/// `.zip` is not a document; a `.txt` that is a PDF still reaches the PDF
/// reader (magic-first, the rule every check here follows).
fn is_office_zip(bytes: &[u8], path: &Path) -> bool {
    if !(bytes.len() > 4 && bytes[..4] == [0x50, 0x4B, 0x03, 0x04]) {
        return false;
    }
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().to_lowercase())
        .unwrap_or_default();
    ["pptx", "docx", "xlsx", "odt", "odp", "ods"]
        .iter()
        .any(|ext| name.ends_with(&format!(".{ext}")))
}

/// PDF through the engines; no native guessing.
fn pdf_read(path: &Path) -> ReadOutcome {
    match engine_convert(path, true) {
        Ok(text) => ReadOutcome::Text(text),
        Err(reason) => ReadOutcome::Unreadable(reason),
    }
}

/// Office through the engines, falling back to the built-in zip+XML walk:
/// engines are the quality path, the walk is the "nothing installed" floor.
fn office_read(bytes: &[u8], path: &Path) -> ReadOutcome {
    match engine_convert(path, false) {
        Ok(text) => ReadOutcome::Text(text),
        Err(reason) => {
            if let Some(text) = ooxml_text(bytes) {
                return ReadOutcome::Text(text);
            }
            if reason == PYTHON_MISSING {
                // No engines AND the built-in walk could not parse this
                // container: the install command is the only way forward.
                ReadOutcome::Unreadable(PYTHON_MISSING.to_string())
            } else {
                ReadOutcome::Unreadable(format!(
                    "the built-in reader could not extract this container; {ENGINES_MISSING}"
                ))
            }
        }
    }
}

/// One engine chain: `Err` is always an actionable reason.
///
/// `spawn failed` is told apart from `engine ran, nothing came out` because
/// the two lead to different commands — that distinction is the whole value
/// of the message.
fn engine_convert(path: &Path, pdf: bool) -> Result<String, String> {
    let attempts: Vec<Vec<String>> = if pdf {
        vec![
            vec!["python".into(), "-c".into(), PDF_SCRIPT.into(), path.display().to_string()],
            vec!["python".into(), "-m".into(), "markitdown".into(), path.display().to_string()],
        ]
    } else {
        vec![vec!["python".into(), "-m".into(), "markitdown".into(), path.display().to_string()]]
    };

    let mut spawned_at_least_once = false;
    let mut last_spawn_missing = false;
    for argv in attempts {
        let (program, args) = argv.split_first().expect("argv non-empty");
        // Blocking is correct here: `read` calls readers inside its own
        // spawn_blocking (see base.rs), so this already runs on a dedicated
        // thread — no executor worker is parked by the poll loop below.
        match run_blocking(program, args) {
            Ok(text) if !text.trim().is_empty() => return Ok(text),
            Ok(_) => {}
            Err(error) => {
                if error.contains("spawning") {
                    last_spawn_missing = true;
                } else {
                    spawned_at_least_once = true;
                }
            }
        }
    }
    if last_spawn_missing && !spawned_at_least_once {
        Err(PYTHON_MISSING.to_string())
    } else {
        Err(ENGINES_MISSING.to_string())
    }
}

/// PDF text, extracted by PyMuPDF — printed on stdout as UTF-8.
const PDF_SCRIPT: &str = r#"
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import fitz
doc = fitz.open(sys.argv[1])
for i, page in enumerate(doc):
    print(f"--- page {i + 1} ---")
    print(page.get_text())
"#;

// ------------------------------------------------------------ bounded engine run

/// Bytes one engine run may accumulate before its reader stops growing the
/// buffer. The reader keeps draining the pipe past the cap — a child that
/// cannot write is a child that never exits — but the text is then known to
/// be a prefix, not the whole conversion.
const ENGINE_OUTPUT_LIMIT: usize = 4 * 1024 * 1024;

/// How long one engine attempt gets, end to end: spawn, run, and close its
/// output pipe.
const ENGINE_BUDGET: std::time::Duration = std::time::Duration::from_secs(30);

/// How much of the budget a reader may still use after its process is gone.
/// A killed tree owes an EOF at once; this is slack, never a second timeout.
const DRAIN_GRACE: std::time::Duration = std::time::Duration::from_secs(2);

/// What one engine's reader thread has collected so far.
///
/// Shared with the waiting thread on purpose. The drain can be abandoned at
/// the deadline — and the bytes that did arrive are still the answer the user
/// asked for, minus whatever the wedged writer never sent.
struct Collected {
    bytes: std::sync::Mutex<Vec<u8>>,
    /// Set once the cap was hit: the text above is a prefix.
    overflowed: std::sync::atomic::AtomicBool,
    /// Set when the reader reached EOF (or its pipe broke trying).
    done: std::sync::atomic::AtomicBool,
}

impl Collected {
    fn new() -> Self {
        use std::sync::atomic::AtomicBool;
        use std::sync::Mutex;
        Self {
            bytes: Mutex::new(Vec::new()),
            overflowed: AtomicBool::new(false),
            done: AtomicBool::new(false),
        }
    }

    /// Append what just arrived, keeping at most [`ENGINE_OUTPUT_LIMIT`].
    fn push(&self, chunk: &[u8]) {
        use std::sync::atomic::Ordering;
        // A poisoned mutex still holds a perfectly good `Vec<u8>`: recovering
        // the inner data beats panicking every later read of this file.
        let mut buf = self
            .bytes
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let room = ENGINE_OUTPUT_LIMIT.saturating_sub(buf.len());
        if chunk.len() > room {
            buf.extend_from_slice(&chunk[..room]);
            self.overflowed.store(true, Ordering::Relaxed);
        } else {
            buf.extend_from_slice(chunk);
        }
    }

    /// Everything read so far, taken out of the shared buffer.
    fn take(&self) -> Vec<u8> {
        std::mem::take(&mut *self.bytes.lock().unwrap_or_else(|error| error.into_inner()))
    }

    fn finished(&self) -> bool {
        use std::sync::atomic::Ordering;
        self.done.load(Ordering::Acquire)
    }
}

/// Read a pipe into `sink` until EOF, on its own thread.
///
/// The thread is deliberately never joined: see [`await_drain`].
fn drain_pipe<R>(mut pipe: R, sink: Arc<Collected>)
where
    R: std::io::Read + Send + 'static,
{
    use std::sync::atomic::Ordering;
    // Detached on purpose: the handle is never joined, so there is nothing to
    // do with it (`await_drain` is how a caller gives up on this thread).
    let _ = std::thread::spawn(move || {
        let mut chunk = [0u8; 8192];
        loop {
            match std::io::Read::read(&mut pipe, &mut chunk) {
                Ok(0) | Err(_) => break,
                Ok(read) => sink.push(&chunk[..read]),
            }
        }
        sink.done.store(true, Ordering::Release);
    });
}

/// Wait up to `budget` for the reader to reach EOF; `false` means "still
/// open", and what was collected is as good as it gets.
fn await_drain(sink: &Collected, budget: std::time::Duration) -> bool {
    let deadline = std::time::Instant::now() + budget;
    while !sink.finished() {
        if std::time::Instant::now() >= deadline {
            return false;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    true
}

/// `30s`, `1200ms` — the budget as the message states it.
fn budget_label(budget: std::time::Duration) -> String {
    if budget.as_millis() % 1000 == 0 {
        format!("{}s", budget.as_secs())
    } else {
        format!("{}ms", budget.as_millis())
    }
}

/// Spawn, collect stdout with a bound, kill on timeout.
///
/// A parser wedged on a corrupt file must not wedge the turn that tried to
/// read it — and neither may a *pipe* the process left open. The old code
/// bounded only the wait for the process, then joined the reader unbounded:
/// a grandchild that inherited stdout (the exact thing the comment here
/// claimed to defend against) kept `read_to_end` from ever returning, leaking
/// the thread and hanging the whole read. Both waits now run to one deadline,
/// and the timeout path returns the partial text with a note instead of
/// hanging or discarding it.
/// `PYTHONIOENCODING=utf-8` fixes the one encoding bug that would sink this
/// path: a piped stdout otherwise follows the console code page, and a GBK
/// console dies on exactly the CJK text this exists for.
fn run_blocking(program: &str, args: &[String]) -> Result<String, String> {
    run_blocking_within(program, args, ENGINE_BUDGET)
}

/// [`run_blocking`] with the budget in hand, so a test can prove the bound
/// without waiting thirty seconds for the production one.
fn run_blocking_within(
    program: &str,
    args: &[String],
    budget: std::time::Duration,
) -> Result<String, String> {
    use std::process::Stdio;
    use std::sync::atomic::Ordering;

    let mut child = std::process::Command::new(program)
        .args(args)
        .env("PYTHONIOENCODING", "utf-8")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|error| format!("spawning {program}: {error}"))?;

    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| format!("{program}: stdout was not piped"))?;
    let sink = Arc::new(Collected::new());
    drain_pipe(stdout, Arc::clone(&sink));

    let deadline = std::time::Instant::now() + budget;
    let mut timed_out = false;
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) => {
                if std::time::Instant::now() >= deadline {
                    timed_out = true;
                    let _ = child.kill();
                    let _ = child.wait();
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(25));
            }
            Err(error) => {
                // Never leave an un-reaped child behind on the error path.
                let _ = child.kill();
                let _ = child.wait();
                return Err(format!("waiting for {program}: {error}"));
            }
        }
    }

    // The process is gone, but the pipe only owes an EOF once every writer is
    // gone with it — a descendant can hold it open forever. Same deadline;
    // after a kill, a short grace is enough for the EOF a killed tree owes.
    let drain_budget = if timed_out {
        DRAIN_GRACE
    } else {
        deadline.saturating_duration_since(std::time::Instant::now())
    };
    let complete = await_drain(&sink, drain_budget);
    let truncated = sink.overflowed.load(Ordering::Relaxed);
    let bytes = sink.take();

    let mut text = String::from_utf8_lossy(&bytes).into_owned();
    if truncated {
        text.push('…');
    }
    let mut notes: Vec<String> = Vec::new();
    if timed_out {
        notes.push(format!(
            "conversion timed out after {} and was killed",
            budget_label(budget)
        ));
    }
    if !complete {
        notes.push(format!(
            "a process other than the engine still holds its output pipe, so the \
             text above stops at {} and may be incomplete",
            budget_label(budget)
        ));
    }

    if text.trim().is_empty() {
        // Nothing readable came out. Without a note that is the honest
        // "engine produced no text"; with one, the reason is the whole answer.
        return if notes.is_empty() {
            Ok(String::new())
        } else {
            Err(format!("{}: {}", program, notes.join("; ")))
        };
    }
    if notes.is_empty() {
        return Ok(text);
    }
    Ok(format!("{text}\n\n[{}]", notes.join("; ")))
}

// ---------------------------------------------------------------- built-in
// The offline floor for Office: pure zip + string walks, no interpreter.

/// Extract OOXML text: per-slide text with each slide's notes for a deck,
/// paragraphs of `word/document.xml`, shared strings as a spreadsheet's last
/// resort. `None` when nothing readable comes out — garbage is never the
/// answer.
fn ooxml_text(bytes: &[u8]) -> Option<String> {
    use std::io::Read;

    let mut archive = zip::ZipArchive::new(std::io::Cursor::new(bytes)).ok()?;

    let mut slides: Vec<(u32, String)> = Vec::new();
    let mut notes: std::collections::BTreeMap<u32, String> = std::collections::BTreeMap::new();
    let mut document: Option<String> = None;
    let mut shared: Option<String> = None;

    for index in 0..archive.len() {
        // Scoped per iteration: each borrow of the archive ends before the
        // next `by_index`, and everything kept is owned.
        let (name, xml) = {
            let mut file = archive.by_index(index).ok()?;
            let name = file.name().to_string();
            let mut xml = String::new();
            if file.read_to_string(&mut xml).is_err() {
                continue; // a binary part (media, embeddings): not our text
            }
            (name, xml)
        };

        if let Some(rest) = name
            .strip_prefix("ppt/slides/slide")
            .and_then(|r| r.strip_suffix(".xml"))
        {
            if let Ok(number) = rest.parse::<u32>() {
                slides.push((number, xml_paragraphs(&xml, "</a:p>", "a:t")));
            }
        } else if let Some(rest) = name
            .strip_prefix("ppt/notesSlides/notesSlide")
            .and_then(|r| r.strip_suffix(".xml"))
        {
            if let Ok(number) = rest.parse::<u32>() {
                notes.insert(number, xml_paragraphs(&xml, "</a:p>", "a:t"));
            }
        } else if name == "word/document.xml" {
            document = Some(xml_paragraphs(&xml, "</w:p>", "w:t"));
        } else if name == "xl/sharedStrings.xml" {
            shared = Some(xml_paragraphs(&xml, "</si>", "t"));
        }
    }

    if !slides.is_empty() {
        slides.sort_by_key(|(number, _)| *number);
        let mut out = String::new();
        for (number, text) in slides {
            out.push_str(&format!("--- slide {number} ---\n"));
            if !text.trim().is_empty() {
                out.push_str(&text);
                out.push('\n');
            }
            if let Some(note) = notes.get(&number) {
                if !note.trim().is_empty() {
                    out.push_str(&format!("[notes {number}] {note}\n"));
                }
            }
        }
        return Some(out);
    }
    if let Some(text) = document.filter(|t| !t.trim().is_empty()) {
        return Some(text);
    }
    shared.filter(|t| !t.trim().is_empty())
}

/// Split one OOXML part into paragraphs and pull the text runs out of each.
///
/// `close_tag` ends a paragraph (`</a:p>`, `</w:p>`, `</si>`); `tag` names the
/// run element (`a:t`, `w:t`, `t` — boundary-checked so `<a:tbl>` is not
/// mistaken for `<a:t>`). Entities are decoded because the XML is walked as
/// bytes, not parsed by a library.
fn xml_paragraphs(xml: &str, close_tag: &str, tag: &str) -> String {
    let open = format!("<{tag}");
    let close = format!("</{tag}>");
    let mut out = Vec::new();
    for paragraph in xml.split(close_tag) {
        let mut line = String::new();
        let mut rest = paragraph;
        while let Some(start) = rest.find(&open) {
            let after_tag = &rest[start + open.len()..];
            let Some(greater) = after_tag.find('>') else { break };
            // Boundary: `<a:t>` or `<a:t attr…>` — not `<a:tbl>`.
            if !after_tag[..greater].is_empty() && !after_tag.starts_with(' ') {
                rest = &after_tag[greater..];
                continue;
            }
            let body_start = &after_tag[greater + 1..];
            let Some(end) = body_start.find(&close) else { break };
            line.push_str(&xml_unescape(&body_start[..end]));
            rest = &body_start[end + close.len()..];
        }
        if !line.trim().is_empty() {
            out.push(line);
        }
    }
    out.join("\n")
}

/// Decode the five XML entities plus numeric references.
fn xml_unescape(text: &str) -> String {
    if !text.contains('&') {
        return text.to_string();
    }
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find('&') {
        out.push_str(&rest[..start]);
        rest = &rest[start..];
        let Some(end) = rest.find(';') else {
            out.push_str(rest);
            return out;
        };
        let entity = &rest[1..end];
        match entity {
            "amp" => out.push('&'),
            "lt" => out.push('<'),
            "gt" => out.push('>'),
            "quot" => out.push('"'),
            "apos" => out.push('\''),
            _ => {
                let code = if let Some(hex) = entity.strip_prefix("#x").or_else(|| entity.strip_prefix("#X")) {
                    u32::from_str_radix(hex, 16).ok()
                } else {
                    entity.strip_prefix('#').and_then(|d| d.parse().ok())
                };
                match code.and_then(char::from_u32) {
                    Some(ch) => out.push(ch),
                    None => {
                        out.push('&');
                        out.push_str(entity);
                        out.push(';');
                    }
                }
            }
        }
        rest = &rest[end + 1..];
    }
    out.push_str(rest);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn claims_need_magic_and_an_office_name() {
        let ppt = std::path::PathBuf::from("deck.PPTX");
        let zip = std::path::PathBuf::from("bundle.zip");
        let txt = std::path::PathBuf::from("notes.txt");
        let office = [0x50u8, 0x4B, 0x03, 0x04, 0x14];

        assert!(is_office_zip(&office, &ppt), "zip magic + office name");
        assert!(!is_office_zip(&office, &zip), "a plain zip is not a document");
        assert!(!is_office_zip(b"plain", &ppt), "the name alone claims nothing");
        assert!(!is_office_zip(&office, &txt));
    }

    #[test]
    fn entities_and_boundaries_survive_the_walk() {
        assert_eq!(xml_unescape("a &amp; b &lt;c&gt; &#65;&#x42; &unknown;"), "a & b <c> AB &unknown;");

        // `<a:tbl>` must not be swallowed as an `<a:t>` run…
        let xml = "<p><a:tbl>TABLE</a:tbl><a:t>real &amp; text</a:t></p>";
        assert_eq!(xml_paragraphs(xml, "</p>", "a:t"), "real & text");

        // …and an attribute-bearing run counts.
        let spaced = "<p><a:t xml:space=\"preserve\">kept</a:t></p>";
        assert_eq!(xml_paragraphs(spaced, "</p>", "a:t"), "kept");
    }

    #[test]
    fn a_deck_reads_slide_by_slide_with_notes_in_numeric_order() {
        let mut zip = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
        let options = zip::write::SimpleFileOptions::default();
        let slide = |text: &str| {
            format!("<p:sld><a:p><a:r><a:t>{text}</a:t></a:r></a:p></p:sld>")
        };
        for (name, text) in [
            ("ppt/slides/slide2.xml", "second"),
            ("ppt/slides/slide10.xml", "tenth"),
            ("ppt/notesSlides/notesSlide2.xml", "note for two"),
        ] {
            zip.start_file(name, options).unwrap();
            std::io::Write::write_all(&mut zip, slide(text).as_bytes()).unwrap();
        }
        // `ZipWriter::into_inner` finishes the archive: the central directory
        // it writes is what `ZipArchive` reads back.
        let bytes = zip.finish().expect("finishing the archive").into_inner();
        let text = ooxml_text(&bytes).expect("a readable deck");

        let second = text.find("--- slide 2 ---").expect("slide 2 present");
        let tenth = text.find("--- slide 10 ---").expect("slide 10 present");
        assert!(second < tenth, "numeric order, not lexicographic: {text}");
        assert!(text.contains("second") && text.contains("tenth"), "{text}");
        assert!(text.contains("[notes 2] note for two"), "notes attach: {text}");
    }

    #[test]
    fn a_word_document_reads_its_paragraphs() {
        let mut zip = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
        let options = zip::write::SimpleFileOptions::default();
        zip.start_file("word/document.xml", options).unwrap();
        std::io::Write::write_all(
            &mut zip,
            br#"<?xml version="1.0"?><w:document><w:body>
                <w:p><w:r><w:t>First &amp; foremost</w:t></w:r></w:p>
                <w:p><w:r><w:t>Second paragraph</w:t></w:r></w:p>
            </w:body></w:document>"#,
        )
        .unwrap();
        let bytes = zip.finish().expect("finishing the archive").into_inner();

        let text = ooxml_text(&bytes).expect("a readable document");
        assert_eq!(text, "First & foremost\nSecond paragraph");
    }

    #[test]
    fn unreadable_outcomes_carry_installable_commands() {
        // The cards are the feature: an agent with bash must be able to fix
        // the machine from the error alone.
        assert!(PYTHON_MISSING.contains("winget install Python.Python.3.13"));
        assert!(PYTHON_MISSING.contains("apt install -y python3"));
        assert!(PYTHON_MISSING.contains("brew install python"));
        assert!(ENGINES_MISSING.contains("pip install --user pymupdf markitdown"));
        assert!(NO_TEXT_FOUND.contains("OCR"));
    }

    /// The wedged pipe, in-process: a `Read` that hands over some bytes and
    /// then never reaches EOF is exactly what a grandchild holding the
    /// engine's stdout looks like. Joining that reader is the unbounded wait
    /// the 30s deadline used to miss — so the drain gets a bounded grace, and
    /// the bytes that did arrive survive it.
    #[test]
    fn a_wedged_drain_is_abandoned_without_losing_what_it_read() {
        use std::io::Read;
        use std::time::{Duration, Instant};

        /// Some text, then a writer that will not close: two seconds of
        /// blocking stands in for "forever" at test speed.
        struct Wedged {
            served: bool,
        }

        impl Read for Wedged {
            fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
                if !self.served {
                    self.served = true;
                    let text = b"partial-conversion";
                    buf[..text.len()].copy_from_slice(text);
                    return Ok(text.len());
                }
                // The writer nobody is waiting for: eventually it does close,
                // but an unbounded join would wait exactly that long.
                for _ in 0..100 {
                    std::thread::sleep(Duration::from_millis(20));
                }
                Ok(0)
            }
        }

        let sink = Arc::new(Collected::new());
        let reading = Arc::clone(&sink);
        let _ = std::thread::spawn(move || drain_pipe(Wedged { served: false }, reading));

        let started = Instant::now();
        let complete = await_drain(&sink, Duration::from_millis(200));
        let elapsed = started.elapsed();

        assert!(!complete, "a pipe that never closes must be abandoned");
        assert!(
            elapsed < Duration::from_secs(1),
            "the wait ends at its own budget, not at the writer's pleasure: {elapsed:?}"
        );
        let bytes = sink.take();
        assert_eq!(
            String::from_utf8_lossy(&bytes).as_ref(),
            "partial-conversion",
            "what was read before the deadline is the answer, not garbage"
        );
    }

    /// The same cap the reader shares: past it the text is a prefix and the
    /// flag says so, while the pipe keeps draining (a child that cannot write
    /// never exits, and the deadline would then only hide the hang).
    #[test]
    fn the_drain_stops_growing_but_keeps_reading() {
        use std::io::Read as _;

        /// An endless pipe: always ready, never EOF.
        struct Endless;
        impl std::io::Read for Endless {
            fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
                let filled = buf.len().min(64);
                for byte in &mut buf[..filled] {
                    *byte = b'x';
                }
                Ok(filled)
            }
        }

        let sink = Arc::new(Collected::new());
        // Pushed directly rather than through a thread: what is under test is
        // the cap, and a thread would only make the assertion racy.
        let mut source = Endless;
        let mut chunk = [0u8; 64];
        let mut pushed = 0usize;
        while pushed <= ENGINE_OUTPUT_LIMIT + 4096 {
            let read = source.read(&mut chunk).expect("endless source");
            sink.push(&chunk[..read]);
            pushed += read;
        }

        assert_eq!(
            sink.take().len(),
            ENGINE_OUTPUT_LIMIT,
            "the buffer is capped, not unbounded"
        );
        assert!(
            sink.overflowed.load(std::sync::atomic::Ordering::Relaxed),
            "the truncation is recorded, so the caller can say so"
        );
    }

    /// End to end, with a real process: the engine exits at once, a background
    /// descendant inherits its stdout and keeps the pipe open. The old code
    /// joined that reader with no bound and hung for as long as the descendant
    /// lived; now the call returns at its deadline with the text it did get.
    ///
    /// The Windows command carries no double quotes on purpose. Rust passes an
    /// argument to `cmd` in MSVCRT escaping (`"` inside an argument becomes
    /// `\"`), and cmd.exe has no backslash escape — a nested `cmd /C "…"` gets
    /// mangled, its `echo` never runs, and the test fails for a reason that has
    /// nothing to do with the drain. Bare words only; `start /b` hands the
    /// grandchild our stdout, and `ping` holds it for ~15 s.
    #[test]
    fn a_grandchild_holding_stdout_returns_partial_output_at_the_deadline() {
        use std::time::{Duration, Instant};

        let budget = Duration::from_millis(1200);
        // Long enough that an unbounded join would blow through the assertion
        // below, short enough not to leave a stray process for long.
        let (program, args) = if cfg!(windows) {
            (
                "cmd".to_string(),
                vec![
                    "/C".to_string(),
                    "start /b ping -n 16 127.0.0.1 & echo engine-done".to_string(),
                ],
            )
        } else {
            (
                "sh".to_string(),
                vec!["-c".to_string(), "sleep 15 & echo engine-done".to_string()],
            )
        };

        let started = Instant::now();
        let outcome = run_blocking_within(&program, &args, budget);
        let elapsed = started.elapsed();
        assert!(
            elapsed < Duration::from_secs(8),
            "the call must end at its deadline, not when the orphan closes the pipe: {elapsed:?}"
        );

        let text = match outcome {
            Ok(text) => text,
            Err(reason) => reason,
        };
        assert!(text.contains("engine-done"), "partial output is kept: {text}");
        assert!(
            text.contains("output pipe") || text.contains("timed out"),
            "the reason the text may be short is named: {text}"
        );
    }

    /// The whole chain, end to end: kernel loads the plugin, `read` consults
    /// the registry, the PDF reader runs. Either the engines produce text or
    /// the error is an install-command card — both outcomes prove the
    /// contract ("content, or precisely why not, plus the command that fixes
    /// it"), so the test accepts either instead of assuming this machine's
    /// Python state forever.
    #[tokio::test]
    async fn the_read_tool_reads_a_pdf_through_the_registered_reader() {
        use serde_json::json;

        // A minimal one-page PDF: enough for a repairing parser to find the
        // text operator, and enough for the raw path if it ever exists.
        let pdf = b"%PDF-1.4\n1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n\
            2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n\
            3 0 obj << /Type /Page /Parent 2 0 R /Contents 4 0 R /MediaBox [0 0 612 792] >> endobj\n\
            4 0 obj << /Length 44 >> stream\n\
            BT /F1 12 Tf 72 720 Td (Hello PDF) Tj ET\n\
            endstream endobj\ntrailer << /Root 1 0 R >>\n%%EOF\n";
        let dir = std::env::temp_dir().join(format!(
            "ngu-doc-read-{}",
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("fixture.pdf");
        std::fs::write(&path, pdf).unwrap();

        let mut kernel = crate::plugin::Kernel::new();
        define(&mut kernel);
        kernel
            .load(
                PLUGIN,
                crate::plugin::RealmMap::new(),
                serde_json::Value::Null,
            )
            .unwrap();

        let registry = kernel.tools().clone();
        let outcome = registry
            .execute(
                "read",
                &json!({ "path": path.display().to_string() }).to_string(),
            )
            .await;
        match outcome {
            Ok(output) => assert!(
                output.text.contains("Hello PDF"),
                "the engine read the page: {}",
                output.text.chars().take(200).collect::<String>()
            ),
            Err(error) => {
                let text = format!("{error:#}");
                assert!(
                    text.contains("winget install") || text.contains("pip install"),
                    "a failure must be the command card, got: {text}"
                );
            }
        }

        let _ = std::fs::remove_dir_all(&dir);
    }
}
