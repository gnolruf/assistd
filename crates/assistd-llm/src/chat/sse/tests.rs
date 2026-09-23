use super::*;

fn drain_all(reader: &mut SseLineReader) -> Vec<SseEvent> {
    let mut out = Vec::new();
    while let Some(ev) = reader.next_event().expect("parse ok") {
        out.push(ev);
    }
    out
}

#[test]
fn parses_complete_input() {
    let cases: [(&str, &[u8], Vec<SseEvent>); 8] = [
        (
            "single data event",
            b"data: {\"hello\":\"world\"}\n\n",
            vec![SseEvent::Data("{\"hello\":\"world\"}".into())],
        ),
        (
            "events in order, then done",
            b"data: one\ndata: two\ndata: [DONE]\n\n",
            vec![
                SseEvent::Data("one".into()),
                SseEvent::Data("two".into()),
                SseEvent::Done,
            ],
        ),
        ("done marker", b"data: [DONE]\n\n", vec![SseEvent::Done]),
        (
            "crlf line endings",
            b"data: {}\r\ndata: [DONE]\r\n\r\n",
            vec![SseEvent::Data("{}".into()), SseEvent::Done],
        ),
        (
            "comment lines skipped",
            b": keep-alive\ndata: {}\n",
            vec![SseEvent::Data("{}".into())],
        ),
        (
            "blank lines skipped",
            b"\n\ndata: {}\n\n",
            vec![SseEvent::Data("{}".into())],
        ),
        (
            "non-data fields skipped",
            b"event: foo\nid: 42\nretry: 100\ndata: {}\n",
            vec![SseEvent::Data("{}".into())],
        ),
        (
            "no space after colon",
            b"data:{}\n",
            vec![SseEvent::Data("{}".into())],
        ),
    ];
    for (label, input, expected) in cases {
        let mut r = SseLineReader::new();
        r.feed(input);
        assert_eq!(drain_all(&mut r), expected, "{label}");
    }
}

#[test]
fn holds_a_partial_line_until_its_newline_arrives() {
    let mut r = SseLineReader::new();
    r.feed(b"data: {\"ch");
    assert_eq!(r.next_event().unwrap(), None);
    r.feed(b"unk\":1}\n\n");
    assert_eq!(
        drain_all(&mut r),
        vec![SseEvent::Data("{\"chunk\":1}".into())]
    );
}

#[test]
fn handles_chunk_boundary_mid_utf8_codepoint() {
    let mut r = SseLineReader::new();
    let bytes = "data: 世界\n\n".as_bytes();
    let split = bytes.iter().position(|&b| b == 0xe4).unwrap() + 1;
    r.feed(&bytes[..split]);
    assert_eq!(r.next_event().unwrap(), None);
    r.feed(&bytes[split..]);
    assert_eq!(drain_all(&mut r), vec![SseEvent::Data("世界".into())]);
}

#[test]
fn invalid_utf8_line_surfaces_error() {
    let mut r = SseLineReader::new();
    r.feed(b"data: \xff\xfe\n");
    let err = r.next_event().unwrap_err();
    assert!(
        matches!(&err, ChatClientError::Sse(msg) if msg.contains("UTF-8")),
        "{err:?}"
    );
}
