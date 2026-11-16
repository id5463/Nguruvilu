//! The `documents` plugin: teach `read` the formats the kernel does not parse.
//!
//! 读不了就装包 — the pack provides the format, the kernel only provides the
//! asking. This was `src/tools/documents.rs` (843 lines) registering two
//! `Arc<dyn Fn>` closures into the kernel's reader registry, which is why the
//! capability could not leave: a closure cannot cross a process boundary.
//!
//! Now the kernel asks two things over the protocol:
//!
//! ```json
//! {"op":"describe"}                               → {"readers":[{"id":"pdf","magic":["%PDF"],"extensions":["pdf"]}]}
//! {"op":"convert","reader":"pdf","path":"…","bytes_base64":"…"} → {"text":"…"} | {"not_mine":true} | {"error":"…"}
//! ```
//!
//! The declaration travels as **data** (`magic`, `extensions`) rather than as a
//! probe function, because the kernel has to decide whether to call at all on
//! every binary `read` — and a round trip to ask "is this yours?" would cost
//! 0.25 ms on every file the plugin does not claim. The kernel checks the cheap
//! declaration locally; this process only starts when it matches.
//!
//! # The conversion itself is unchanged
//!
//! The engines, the fallbacks, the install-command cards — all of it is the code
//! that was in the kernel, moved. A PDF is still handed to PyMuPDF, Office files
//! still go through a zip+XML walk when no engine answers, and a failure still
//! names the command that fixes the machine.

use std::io::{Read, Write};
use std::sync::Arc;
use std::path::Path;

use serde_json::{json, Value};

/// What one reader claims, as data the kernel can check without asking.
struct Reader {
    id: &'static str,
    /// Byte prefixes that mean "this is mine".
    magic: &'static [&'static str],
    /// File extensions, lowercase and without the dot.
    extensions: &'static [&'static str],
    /// The conversion.
    run: fn(&[u8], &Path) -> Outcome,
}

/// The result of a conversion attempt.
enum Outcome {
    /// Readable text.
    Text(String),
    /// These bytes are not this reader's format after all.
    NotMine,
    /// The format is mine, but extraction failed — with a reason worth showing
    /// (an install-command card belongs here).
    Unreadable(String),
}

fn main() {
    let readers = readers();
    let mut stdin = std::io::stdin();
    loop {
        let frame = match read_frame(&mut stdin) {
            Some(frame) => frame,
            None => return,
        };
        let message: Value = match serde_json::from_slice(&frame) {
            Ok(value) => value,
            Err(error) => {
                write_frame(&json!({ "error": format!("not JSON: {error}") }));
                continue;
            }
        };
        let answer = match message.get("op").and_then(|op| op.as_str()) {
            Some("describe") => json!({
                "tools": [],
                // Readers, not tools: this plugin exists to make the kernel's
                // own `read` work on more formats, and `read` is the tool.
                "readers": readers
                    .iter()
                    .map(|reader| json!({
                        "id": reader.id,
                        "magic": reader.magic,
                        "extensions": reader.extensions,
                    }))
                    .collect::<Vec<_>>(),
            }),
            Some("convert") => convert(&readers, &message),
            Some(other) => json!({ "error": format!("unsupported op: {other}") }),
            None => json!({ "error": "missing op" }),
        };
        write_frame(&answer);
    }
}

/// Run one conversion.
fn convert(readers: &[Reader], message: &Value) -> Value {
    use base64::Engine as _;

    let id = message.get("reader").and_then(|v| v.as_str()).unwrap_or("");
    let path = message
        .get("path")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let encoded = message
        .get("bytes_base64")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let bytes = match base64::engine::general_purpose::STANDARD.decode(encoded) {
        Ok(bytes) => bytes,
        Err(error) => return json!({ "error": format!("bytes_base64 did not decode: {error}") }),
    };

    let Some(reader) = readers.iter().find(|reader| reader.id == id) else {
        return json!({ "error": format!("no reader called '{id}'") });
    };
    match (reader.run)(&bytes, Path::new(&path)) {
        Outcome::Text(text) => json!({ "text": text }),
        Outcome::NotMine => json!({ "not_mine": true }),
        Outcome::Unreadable(reason) => json!({ "error": reason }),
    }
}

/// The two readers, unchanged from the kernel version.
fn readers() -> Vec<Reader> {
    vec![
        Reader {
            id: "pdf",
            magic: &["%PDF"],
            extensions: &["pdf"],
            run: |_, path| pdf_read(path),
        },
        Reader {
            id: "office",
            magic: &["PK\x03\x04"],
            extensions: &["pptx", "docx", "xlsx"],
            run: office_read,
        },
    ]
}

fn read_frame(input: &mut impl Read) -> Option<Vec<u8>> {
    let mut header = [0u8; 4];
    input.read_exact(&mut header).ok()?;
    let len = u32::from_be_bytes(header) as usize;
    let mut body = vec![0u8; len];
    input.read_exact(&mut body).ok()?;
    Some(body)
}

fn write_frame(value: &Value) {
    let body = match serde_json::to_vec(value) {
        Ok(body) => body,
        Err(_) => return,
    };
    let mut out = std::io::stdout();
    out.write_all(&(body.len() as u32).to_be_bytes()).ok();
    out.write_all(&body).ok();
    out.flush().ok();
}

// ------------------------------------------------------------ the conversions
// Everything below is the code that was in `src/tools/documents.rs`, with
// `ReadOutcome` replaced by this plugin's own `Outcome`.

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
fn pdf_read(path: &Path) -> Outcome {
    match engine_convert(path, true) {
        Ok(text) => Outcome::Text(text),
        Err(reason) => Outcome::Unreadable(reason),
    }
}

/// Office through the engines, falling back to the built-in zip+XML walk:
/// engines are the quality path, the walk is the "nothing installed" floor.
fn office_read(bytes: &[u8], path: &Path) -> Outcome {
    match engine_convert(path, false) {
        Ok(text) => Outcome::Text(text),
        Err(reason) => {
            if let Some(text) = ooxml_text(bytes) {
                return Outcome::Text(text);
            }
            if reason == PYTHON_MISSING {
                // No engines AND the built-in walk could not parse this
                // container: the install command is the only way forward.
                Outcome::Unreadable(PYTHON_MISSING.to_string())
            } else {
                Outcome::Unreadable(format!(
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

