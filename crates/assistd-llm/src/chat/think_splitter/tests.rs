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
    let cases: [(&str, &[&str], Vec<Segment>); 12] = [
        (
            "visible only",
            &["hello world"],
            vec![visible("hello world")],
        ),
        (
            "reasoning block",
            &["<think>maybe</think>4"],
            vec![reasoning("maybe"), visible("4")],
        ),
        (
            "text around a block",
            &["pre <think>cogitate</think> post"],
            vec![visible("pre "), reasoning("cogitate"), visible(" post")],
        ),
        (
            "adjacent blocks",
            &["a<think>b</think>c<think>d</think>e"],
            vec![
                visible("a"),
                reasoning("b"),
                visible("c"),
                reasoning("d"),
                visible("e"),
            ],
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
            "tags split byte by byte",
            &[
                "<", "t", "h", "i", "n", "k", ">", "x", "<", "/", "t", "h", "i", "n", "k", ">", "y",
            ],
            vec![reasoning("x"), visible("y")],
        ),
        (
            "plain text is emitted per feed, not held",
            &["hello", " ", "world"],
            vec![visible("hello"), visible(" "), visible("world")],
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
        (
            "multibyte body",
            &["<think>héllo 🌍</think>!"],
            vec![reasoning("héllo 🌍"), visible("!")],
        ),
    ];
    for (label, chunks, expected) in cases {
        assert_eq!(run(chunks), expected, "{label}");
    }
}

#[test]
fn empty_feeds_are_no_ops() {
    let mut s = ThinkSplitter::default();
    assert!(s.feed("").is_empty());
    assert_eq!(s.feed("hi"), vec![visible("hi")]);
    assert!(s.feed("").is_empty());
    assert_eq!(s.finish(), None);
}
