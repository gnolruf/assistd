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
