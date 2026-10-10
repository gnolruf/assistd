//! Minimal SSE line parser for `/v1/chat/completions` streams.

use super::error::ChatClientError;

/// A parsed SSE event from a `/v1/chat/completions` stream: the payload
/// of a `data:` line, or the terminal `data: [DONE]`.
#[derive(Debug, PartialEq, Eq)]
pub enum SseEvent {
    Data(String),
    Done,
}

/// Longest incomplete line the reader buffers before failing the stream.
pub const LINE_MAX: usize = 4 * 1024 * 1024;

/// Byte-buffered SSE line parser. Buffering raw bytes keeps a chunk
/// boundary inside a multi-byte UTF-8 character safe; decoding happens
/// per completed line. Only `data:` lines yield events; other fields,
/// comments and blank lines are skipped.
#[derive(Debug, Default)]
pub struct SseLineReader {
    buf: Vec<u8>,
    /// Bytes at the front of `buf` already returned as lines.
    consumed: usize,
    /// Bytes of `buf` known to hold no newline past `consumed`.
    scanned: usize,
}

impl SseLineReader {
    /// Create a reader with an empty buffer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a chunk of raw stream bytes to the buffer.
    pub fn feed(&mut self, chunk: &[u8]) {
        self.buf.drain(..self.consumed);
        self.scanned -= self.consumed;
        self.consumed = 0;
        self.buf.extend_from_slice(chunk);
    }

    /// Extract the next complete SSE event from the buffer, if one is
    /// available. Returns `Ok(None)` if more bytes are needed; fails on
    /// a non-UTF-8 line or an incomplete one longer than [`LINE_MAX`].
    pub fn next_event(&mut self) -> Result<Option<SseEvent>, ChatClientError> {
        loop {
            let Some(offset) = self.buf[self.scanned..].iter().position(|b| *b == b'\n') else {
                self.scanned = self.buf.len();
                return self.check_pending_line().map(|()| None);
            };
            let line_start = self.consumed;
            let newline = self.scanned + offset;
            self.consumed = newline + 1;
            self.scanned = self.consumed;
            if let Some(event) = parse_line(&self.buf[line_start..newline])? {
                return Ok(Some(event));
            }
        }
    }

    fn check_pending_line(&self) -> Result<(), ChatClientError> {
        let pending = self.buf.len() - self.consumed;
        if pending > LINE_MAX {
            return Err(ChatClientError::Sse(format!(
                "line exceeds {LINE_MAX} bytes without a newline"
            )));
        }
        Ok(())
    }
}

/// The event carried by one line, without its `\n`; `None` for lines
/// that are not `data:` fields.
fn parse_line(line: &[u8]) -> Result<Option<SseEvent>, ChatClientError> {
    let line = line.strip_suffix(b"\r").unwrap_or(line);
    if line.first().is_none_or(|b| *b == b':') {
        return Ok(None);
    }
    let line = std::str::from_utf8(line)
        .map_err(|e| ChatClientError::Sse(format!("line is not valid UTF-8: {e}")))?;
    let Some(rest) = line.strip_prefix("data:") else {
        return Ok(None);
    };
    let payload = rest.strip_prefix(' ').unwrap_or(rest);
    if payload == "[DONE]" {
        return Ok(Some(SseEvent::Done));
    }
    Ok(Some(SseEvent::Data(payload.to_string())))
}

#[cfg(test)]
mod tests;
