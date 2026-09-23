//! Minimal byte-buffered SSE line parser for `/v1/chat/completions` streams.
//!
//! llama.cpp emits `data: {json}\n\n` frames followed by a terminal
//! `data: [DONE]\n\n`. We only care about `data:` lines; `event:`, `id:`,
//! `retry:`, comments, and blank lines are ignored. The buffer is raw bytes
//! so that chunk boundaries landing inside a multi-byte UTF-8 character are
//! safe; decoding happens per completed line, not per chunk.

use super::error::ChatClientError;

/// A parsed SSE event from a `/v1/chat/completions` stream.
#[derive(Debug, PartialEq, Eq)]
pub enum SseEvent {
    Data(String),
    Done,
}

/// Byte-buffered SSE line parser that handles chunk boundaries safely.
#[derive(Debug, Default)]
pub struct SseLineReader {
    buf: Vec<u8>,
}

impl SseLineReader {
    pub fn new() -> Self {
        Self { buf: Vec::new() }
    }

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
