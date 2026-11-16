use super::*;

fn run(chunks: &[&str]) -> Vec<Segment> {
    let mut s = ThinkSplitter::default();
    let mut out = Vec::new();
    for chunk in chunks {
        out.extend(s.feed(chunk));
    }
    if let Some(tail) = s.finish() {
        out.push(tail);
    }
    out
}

fn visible(text: &str) -> Segment {
    Segment::Visible(text.into())
}

fn reasoning(text: &str) -> Segment {
    Segment::Reasoning(text.into())
}

#[test]
fn classifies_streamed_chunks() {
    let cases: [(&str, &[&str], Vec<Segment>); 6] = [
        (
            "text around a block",
            &["pre <think>cogitate</think> post"],
            vec![visible("pre "), reasoning("cogitate"), visible(" post")],
        ),
        (
            "open tag split across chunks",
            &["<thi", "nk>hello</think>done"],
            vec![reasoning("hello"), visible("done")],
        ),
        (
            "close tag split across chunks",
            &["<think>foo</thi", "nk>bar"],
            vec![reasoning("foo"), visible("bar")],
        ),
        (
            "unrelated tag passes through",
            &["<x>foo"],
            vec![visible("<x>foo")],
        ),
        (
            "dangling open-tag prefix is flushed as visible on finish",
            &["trailing<thi"],
            vec![visible("trailing"), visible("<thi")],
        ),
        (
            "close tag outside a block is literal text",
            &["a</think>b"],
            vec![visible("a</think>b")],
        ),
    ];
    for (label, chunks, expected) in cases {
        assert_eq!(run(chunks), expected, "{label}");
    }
}
