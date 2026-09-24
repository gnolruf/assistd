//! Minimal SSE line parser for `/v1/chat/completions` streams.

use super::error::ChatClientError;

/// A parsed SSE event from a `/v1/chat/completions` stream: the payload
/// of a `data:` line, or the terminal `data: [DONE]`.
#[derive(Debug, PartialEq, Eq)]
pub enum SseEvent {
    Data(String),
    Done,
}

/// Byte-buffered SSE line parser. Buffering raw bytes keeps a chunk
/// boundary inside a multi-byte UTF-8 character safe; decoding happens
/// per completed line. Only `data:` lines yield events; other fields,
/// comments and blank lines are skipped.
#[derive(Debug, Default)]
pub struct SseLineReader {
    buf: Vec<u8>,
}

impl SseLineReader {
    /// Create a reader with an empty buffer.
    pub fn new() -> Self {
        Self { buf: Vec::new() }
    }

    /// Append a chunk of raw stream bytes to the buffer.
    pub fn feed(&mut self, chunk: &[u8]) {
        self.buf.extend_from_slice(chunk);
    }

    /// Extract the next complete SSE event from the buffer, if one is
    /// available. Returns `Ok(None)` if more bytes are needed.
    pub fn next_event(&mut self) -> Result<Option<SseEvent>, ChatClientError> {
        loop {
            let Some(nl_pos) = self.buf.iter().position(|b| *b == b'\n') else {
                return Ok(None);
            };

            let mut end = nl_pos;
            if end > 0 && self.buf[end - 1] == b'\r' {
                end -= 1;
            }
            let line_bytes = self.buf[..end].to_vec();
            self.buf.drain(..=nl_pos);

            if line_bytes.is_empty() {
                continue;
            }
            if line_bytes[0] == b':' {
                continue;
            }

            let line = std::str::from_utf8(&line_bytes)
                .map_err(|e| ChatClientError::Sse(format!("line is not valid UTF-8: {e}")))?;

            let Some(rest) = line.strip_prefix("data:") else {
                continue;
            };
            let payload = rest.strip_prefix(' ').unwrap_or(rest);

            if payload == "[DONE]" {
                return Ok(Some(SseEvent::Done));
            }
            return Ok(Some(SseEvent::Data(payload.to_string())));
        }
    }
}

#[cfg(test)]
mod tests;
