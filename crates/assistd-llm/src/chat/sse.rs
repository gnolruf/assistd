//! Minimal byte-buffered SSE line parser for `/v1/chat/completions` streams.
//!
//! llama.cpp emits `data: {json}\n\n` frames followed by a terminal
//! `data: [DONE]\n\n`. We only care about `data:` lines; `event:`, `id:`,
//! `retry:`, comments, and blank lines are ignored. The buffer is raw bytes
//! so that chunk boundaries landing inside a multi-byte UTF-8 character are
//! safe — decoding happens per completed line, not per chunk.

use super::error::ChatClientError;

#[derive(Debug, PartialEq, Eq)]
pub enum SseEvent {
    Data(String),
    Done,
}

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
mod tests {
    use super::*;

    fn drain_all(reader: &mut SseLineReader) -> Vec<SseEvent> {
        let mut out = Vec::new();
        while let Some(ev) = reader.next_event().expect("parse ok") {
            out.push(ev);
        }
        out
    }

    #[test]
    fn parses_single_data_event() {
        let mut r = SseLineReader::new();
        r.feed(b"data: {\"hello\":\"world\"}\n\n");
        let events = drain_all(&mut r);
        assert_eq!(
            events,
            vec![SseEvent::Data("{\"hello\":\"world\"}".to_string())]
        );
    }

    #[test]
    fn parses_multiple_events_in_order() {
        let mut r = SseLineReader::new();
        r.feed(b"data: one\ndata: two\ndata: [DONE]\n\n");
        let events = drain_all(&mut r);
        assert_eq!(
            events,
            vec![
                SseEvent::Data("one".into()),
                SseEvent::Data("two".into()),
                SseEvent::Done,
            ]
        );
    }

    #[test]
    fn handles_chunk_boundary_mid_line() {
        let mut r = SseLineReader::new();
        r.feed(b"data: {\"ch");
        assert_eq!(r.next_event().unwrap(), None);
        r.feed(b"unk\":1}\n\n");
        let events = drain_all(&mut r);
        assert_eq!(events, vec![SseEvent::Data("{\"chunk\":1}".into())]);
    }

    #[test]
    fn handles_chunk_boundary_mid_utf8_codepoint() {
        let mut r = SseLineReader::new();
        let bytes = "data: 世界\n\n".as_bytes();
        let split = bytes.iter().position(|&b| b == 0xe4).unwrap() + 1;
        r.feed(&bytes[..split]);
        assert_eq!(r.next_event().unwrap(), None);
        r.feed(&bytes[split..]);
        let events = drain_all(&mut r);
        assert_eq!(events, vec![SseEvent::Data("世界".into())]);
    }

    #[test]
    fn parses_done_marker() {
        let mut r = SseLineReader::new();
        r.feed(b"data: [DONE]\n\n");
        let events = drain_all(&mut r);
        assert_eq!(events, vec![SseEvent::Done]);
    }

    #[test]
    fn handles_crlf_line_endings() {
        let mut r = SseLineReader::new();
        r.feed(b"data: {}\r\ndata: [DONE]\r\n\r\n");
        let events = drain_all(&mut r);
        assert_eq!(events, vec![SseEvent::Data("{}".into()), SseEvent::Done]);
    }

    #[test]
    fn skips_comment_lines() {
        let mut r = SseLineReader::new();
        r.feed(b": keep-alive\ndata: {}\n");
        let events = drain_all(&mut r);
        assert_eq!(events, vec![SseEvent::Data("{}".into())]);
    }

    #[test]
    fn skips_blank_lines() {
        let mut r = SseLineReader::new();
        r.feed(b"\n\ndata: {}\n\n");
        let events = drain_all(&mut r);
        assert_eq!(events, vec![SseEvent::Data("{}".into())]);
    }

    #[test]
    fn skips_unknown_fields() {
        let mut r = SseLineReader::new();
        r.feed(b"event: foo\nid: 42\nretry: 100\ndata: {}\n");
        let events = drain_all(&mut r);
        assert_eq!(events, vec![SseEvent::Data("{}".into())]);
    }

    #[test]
    fn data_without_leading_space_still_parses() {
        let mut r = SseLineReader::new();
        r.feed(b"data:{}\n");
        let events = drain_all(&mut r);
        assert_eq!(events, vec![SseEvent::Data("{}".into())]);
    }

    #[test]
    fn leaves_incomplete_line_in_buffer() {
        let mut r = SseLineReader::new();
        r.feed(b"data: incomplete");
        assert_eq!(r.next_event().unwrap(), None);
    }

    #[test]
    fn invalid_utf8_line_surfaces_error() {
        let mut r = SseLineReader::new();
        r.feed(&[b'd', b'a', b't', b'a', b':', b' ', 0xff, 0xfe, b'\n']);
        let err = r.next_event().unwrap_err();
        match err {
            ChatClientError::Sse(msg) => assert!(msg.contains("UTF-8")),
            other => panic!("expected Sse error, got {other:?}"),
        }
    }
}
