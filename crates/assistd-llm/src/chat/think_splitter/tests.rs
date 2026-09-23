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

#[test]
fn passes_through_visible_only() {
    assert_eq!(
        run(&["hello world"]),
        vec![Segment::Visible("hello world".into())]
    );
}

#[test]
fn extracts_reasoning_block() {
    assert_eq!(
        run(&["<think>maybe</think>4"]),
        vec![
            Segment::Reasoning("maybe".into()),
            Segment::Visible("4".into())
        ]
    );
}

#[test]
fn handles_text_before_and_after_block() {
    assert_eq!(
        run(&["pre <think>cogitate</think> post"]),
        vec![
            Segment::Visible("pre ".into()),
            Segment::Reasoning("cogitate".into()),
            Segment::Visible(" post".into()),
        ]
    );
}

#[test]
fn tolerates_open_tag_split_across_chunks() {
    assert_eq!(
        run(&["<thi", "nk>hello</think>done"]),
        vec![
            Segment::Reasoning("hello".into()),
            Segment::Visible("done".into()),
        ]
    );
}

#[test]
fn tolerates_close_tag_split_across_chunks() {
    assert_eq!(
        run(&["<think>foo</thi", "nk>bar"]),
        vec![
            Segment::Reasoning("foo".into()),
            Segment::Visible("bar".into()),
        ]
    );
}

#[test]
fn tolerates_tag_split_byte_by_byte() {
    let chunks: Vec<&str> = vec![
        "<", "t", "h", "i", "n", "k", ">", "x", "<", "/", "t", "h", "i", "n", "k", ">", "y",
    ];
    assert_eq!(
        run(&chunks),
        vec![Segment::Reasoning("x".into()), Segment::Visible("y".into())]
    );
}

#[test]
fn coalesces_adjacent_visible_chunks() {
    // Visible content arriving across several chunks (no tags)
    // should coalesce into one Visible segment per feed call.
    let mut s = ThinkSplitter::default();
    let mut out = Vec::new();
    out.extend(s.feed("hello"));
    out.extend(s.feed(" "));
    out.extend(s.feed("world"));
    // No coalescing across feed calls because each call returns
    // fresh segments; callers concatenate by appending.
    assert_eq!(
        out,
        vec![
            Segment::Visible("hello".into()),
            Segment::Visible(" ".into()),
            Segment::Visible("world".into()),
        ]
    );
}

#[test]
fn adjacent_blocks_are_classified_separately() {
    assert_eq!(
        run(&["a<think>b</think>c<think>d</think>e"]),
        vec![
            Segment::Visible("a".into()),
            Segment::Reasoning("b".into()),
            Segment::Visible("c".into()),
            Segment::Reasoning("d".into()),
            Segment::Visible("e".into()),
        ]
    );
}

#[test]
fn unrelated_lt_fragment_passes_through_as_visible() {
    // `<x>foo` is not a tag we recognise: the `<` initially holds
    // back the rest until we can confirm it's not `<think>`.
    // Verify the whole string ends up Visible.
    assert_eq!(run(&["<x>foo"]), vec![Segment::Visible("<x>foo".into())]);
}

#[test]
fn dangling_open_tag_is_held_then_flushed_as_visible() {
    // Open-tag prefix that never completes ends up as Visible
    // (current state at finish is OutsideThink, so we don't
    // silently swallow content).
    let mut s = ThinkSplitter::default();
    let mut out = s.feed("trailing<thi");
    out.extend(s.finish());
    assert_eq!(
        out,
        vec![
            Segment::Visible("trailing".into()),
            Segment::Visible("<thi".into()),
        ]
    );
}

#[test]
fn unmatched_close_tag_flips_to_visible_on_finish() {
    // Defensive: if a `</think>` appears while OutsideThink,
    // we treat the literal text as Visible since we never
    // entered InsideThink.
    let mut s = ThinkSplitter::default();
    let mut out = s.feed("a</think>b");
    out.extend(s.finish());
    // The splitter sees no `<think>` to switch state, so
    // `</think>` is not recognised as a tag in the OutsideThink
    // state — emit the whole string as Visible.
    let concatenated: String = out
        .iter()
        .map(|s| match s {
            Segment::Visible(t) | Segment::Reasoning(t) => t.as_str(),
        })
        .collect();
    assert_eq!(concatenated, "a</think>b");
    assert!(out.iter().all(|s| matches!(s, Segment::Visible(_))));
}

#[test]
fn handles_utf8_body_inside_block() {
    assert_eq!(
        run(&["<think>héllo 🌍</think>!"]),
        vec![
            Segment::Reasoning("héllo 🌍".into()),
            Segment::Visible("!".into()),
        ]
    );
}

#[test]
fn empty_feeds_are_no_ops() {
    let mut s = ThinkSplitter::default();
    assert!(s.feed("").is_empty());
    assert_eq!(s.feed("hi"), vec![Segment::Visible("hi".into())]);
    assert!(s.feed("").is_empty());
    assert!(s.finish().is_none());
}
