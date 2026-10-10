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
    let cases: [(&str, &[u8], Vec<SseEvent>); 4] = [
        (
            "events in order, then done",
            b"data: one\ndata: two\ndata: [DONE]\n\n",
            vec![
                SseEvent::Data("one".into()),
                SseEvent::Data("two".into()),
                SseEvent::Done,
            ],
        ),
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
            "non-data fields skipped",
            b"event: foo\nid: 42\nretry: 100\ndata: {}\n",
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
fn rejects_unterminated_line_past_cap_across_feeds() {
    let mut r = SseLineReader::new();
    let chunk = vec![b'x'; 64 * 1024];
    let mut fed = 0;
    let err = loop {
        r.feed(&chunk);
        fed += chunk.len();
        match r.next_event() {
            Ok(None) => assert!(fed <= LINE_MAX, "cap not enforced"),
            Ok(Some(ev)) => panic!("unexpected event {ev:?}"),
            Err(e) => break e,
        }
    };
    assert!(matches!(err, ChatClientError::Sse(_)));
}
