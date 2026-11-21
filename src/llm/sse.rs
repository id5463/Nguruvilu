use super::*;

/// The largest partial frame the reader will hold before calling the stream
/// broken.
///
/// A frame is a small JSON object — the largest one any provider sends is a
/// tool-call fragment of a few kilobytes. Eight mebibytes is orders of
/// magnitude above that and orders of magnitude below what it takes to stop
/// this process: without a ceiling, an endpoint that writes bytes forever
/// without a newline grows the buffer until the machine refuses.
pub const MAX_FRAME_BYTES: usize = 8 * 1024 * 1024;

/// What the reader should do after one line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Step {
    /// Keep reading frames.
    Continue,
    /// The sequence is complete: `Finished` has been sent.
    End,
    /// Stop reading and send nothing more.
    Stop,
}

/// Split a byte stream into complete SSE lines.
///
/// Two properties matter here, and neither is free:
///
/// * A line is decoded only when all of its bytes have arrived. Decoding each
///   network chunk on its own replaces a UTF-8 character that straddles two
///   chunks with U+FFFD and reports nothing, so a Chinese or emoji answer has a
///   silent chance of arriving mangled — and the mangling is then written to the
///   session file as the answer.
/// * The last line of a stream is not newline-terminated. `next()` returning
///   `None` says the bytes are over, not that the buffer was empty, so whatever
///   is left has to be read as a line rather than dropped.
///
/// It also bounds what it holds: a peer that never sends a newline must not be
/// able to grow the buffer without limit.
pub struct SseDecoder {
    /// Bytes of a line that has not been terminated yet.
    pending: Vec<u8>,
}

impl SseDecoder {
    pub fn new() -> Self {
        Self { pending: Vec::new() }
    }

    /// The complete lines in `chunk`, keeping any unterminated tail for later.
    ///
    /// `Err` means the unterminated tail has outgrown [`MAX_FRAME_BYTES`]; the
    /// message says so in the words a user needs to understand the stream was
    /// abandoned rather than finished.
    pub fn push(&mut self, chunk: &[u8]) -> std::result::Result<Vec<String>, String> {
        self.pending.extend_from_slice(chunk);

        let mut lines = Vec::new();
        let mut start = 0usize;
        while let Some(at) = self.pending[start..].iter().position(|byte| *byte == b'\n') {
            let end = start + at;
            lines.push(decode_line(&self.pending[start..end]));
            start = end + 1;
        }
        if start > 0 {
            self.pending.drain(..start);
        }

        if self.pending.len() > MAX_FRAME_BYTES {
            return Err(format!(
                "the provider sent more than {} MiB without ending a stream frame; \
                 the stream is abandoned rather than buffered",
                MAX_FRAME_BYTES / (1024 * 1024)
            ));
        }
        Ok(lines)
    }

    /// The last line of a stream that ended without a newline after it.
    pub fn finish(&mut self) -> Option<String> {
        if self.pending.is_empty() {
            return None;
        }
        let line = decode_line(&self.pending);
        self.pending.clear();
        Some(line)
    }
}

/// Decode one frame line, dropping the carriage return of a CRLF ending.
///
/// Lossy is right here and only here: the bytes of one line may still contain
/// invalid UTF-8, and a frame that cannot be decoded is a frame to skip rather
/// than a stream to fail. The point is that a *valid* character is never split
/// across the decode.
pub fn decode_line(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).trim_end_matches('\r').to_string()
}

/// Send one event, reporting whether anyone is still listening.
///
/// A closed receiver means the turn was abandoned — the user cancelled, or the
/// loop moved on. Ignoring that and reading on is how a cancelled turn keeps
/// pulling frames and paying for tokens nobody will see.
pub async fn deliver(tx: &mpsc::Sender<Result<LlmEvent>>, event: Result<LlmEvent>) -> bool {
    tx.send(event).await.is_ok()
}

/// Read one complete SSE line and deliver what it carries.
///
/// The parsing lives here rather than in the read loop because the final line
/// arrives from [`SseDecoder::finish`] instead of from the chunk loop: both
/// paths must read a line the same way, or the last frame of every answer would
/// be read by a second, weaker parser.
pub async fn handle_line(
    line: &str,
    tx: &mpsc::Sender<Result<LlmEvent>>,
    calls: &mut Vec<PartialCall>,
    finish_reason: &mut Option<String>,
    saw_output: &mut bool,
) -> Step {
    let Some(data) = line.strip_prefix("data:") else { return Step::Continue };
    let data = data.trim();
    if data.is_empty() {
        return Step::Continue;
    }
    if data == "[DONE]" {
        // The terminal event of the sequence: whether or not anyone is still
        // listening, this task has nothing left to send.
        let _ = deliver(tx, Ok(LlmEvent::Finished { reason: finish_reason.clone() })).await;
        return Step::End;
    }

    let parsed: Value = match serde_json::from_str(data) {
        Ok(v) => v,
        Err(_) => return Step::Continue,
    };

    // A provider that fails mid-answer reports it as an `error` object inside an
    // otherwise normal frame. Reading only `choices` would drop that on the
    // floor: the stream ends, the turn looks complete, and half an answer reads
    // as a finished one.
    if let Some(error) = parsed.get("error").filter(|value| !value.is_null()) {
        let _ = deliver(tx, Err(error_request(error).into())).await;
        return Step::Stop;
    }

    if let Some(usage) = parsed.get("usage").filter(|u| !u.is_null()) {
        if !deliver(tx, Ok(LlmEvent::Usage(extract_usage(usage)))).await {
            return Step::Stop;
        }
    }

    let Some(choice) = parsed.get("choices").and_then(|c| c.get(0)) else { return Step::Continue };

    if let Some(reason) = choice.get("finish_reason").and_then(|r| r.as_str()) {
        *finish_reason = Some(reason.to_string());
    }

    let Some(delta) = choice.get("delta") else { return Step::Continue };

    if let Some(text) = delta.get("content").and_then(|c| c.as_str()) {
        if !text.is_empty() {
            *saw_output = true;
            if !deliver(tx, Ok(LlmEvent::TextDelta(text.to_string()))).await {
                return Step::Stop;
            }
        }
    }

    // Providers spell the reasoning channel differently, and reading
    // only one spelling loses it silently: DeepSeek, Moonshot, Zhipu,
    // DashScope and llama.cpp use `reasoning_content`, while vLLM,
    // Groq, Together and Ollama use `reasoning`. OpenRouter adds a
    // typed `reasoning_details` array on top of both.
    for key in ["reasoning_content", "reasoning"] {
        if let Some(text) = delta.get(key).and_then(|c| c.as_str()) {
            if !text.is_empty() {
                *saw_output = true;
                if !deliver(tx, Ok(LlmEvent::ReasoningDelta(text.to_string()))).await {
                    return Step::Stop;
                }
            }
        }
    }
    if let Some(parts) = delta.get("reasoning_details").and_then(|d| d.as_array()) {
        for part in parts {
            if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                if !text.is_empty() {
                    *saw_output = true;
                    if !deliver(tx, Ok(LlmEvent::ReasoningDelta(text.to_string()))).await {
                        return Step::Stop;
                    }
                }
            }
        }
    }

    if let Some(tool_calls) = delta.get("tool_calls").and_then(|t| t.as_array()) {
        for call in tool_calls {
            *saw_output = true;
            let index = call.get("index").and_then(|i| i.as_u64()).unwrap_or(0) as usize;
            while calls.len() <= index {
                calls.push(PartialCall::default());
            }
            let slot = &mut calls[index];

            let id = call.get("id").and_then(|i| i.as_str()).map(str::to_string);
            let name = call
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(|n| n.as_str())
                .map(str::to_string);
            let args = call
                .get("function")
                .and_then(|f| f.get("arguments"))
                .and_then(|a| a.as_str())
                .map(str::to_string);

            if let Some(id) = &id {
                slot.id = id.clone();
            }
            if let Some(name) = &name {
                slot.name = name.clone();
            }
            if let Some(args) = &args {
                slot.arguments.push_str(args);
            }

            if !deliver(tx, Ok(LlmEvent::ToolCallDelta { index, id, name, arguments: args })).await {
                return Step::Stop;
            }
        }
    }

    Step::Continue
}
