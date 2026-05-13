//! Stateful classifier that separates `<think>...</think>` reasoning
//! from visible content as `delta.content` chunks stream in.
//!
//! Some llama.cpp builds and some reasoning models (Qwen3, DeepSeek-R1)
//! emit reasoning inline as raw `<think>...</think>` tags inside the
//! `content` field rather than via the separated `reasoning_content`
//! channel. We can't depend on operators flipping `--reasoning-format`
//! on the server, so the SSE handler runs every `content` chunk through
//! this splitter and forwards the resulting segments to the correct
//! `LlmEvent` variant.
//!
//! The splitter tolerates tag splits across SSE chunks: an incoming
//! `"<thi"` parks in `pending`, and the next chunk's `"nk>hello"`
//! completes the tag and emits `Reasoning("hello")`.
//!
//! The recognised tags are ASCII-only (`<`, `/`, `t`, `h`, `i`, `n`,
//! `k`, `>`), so byte-slice arithmetic on `pending` never lands inside
//! a multibyte UTF-8 character.

/// One classified slice emitted by [`ThinkSplitter::feed`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Segment {
    /// Content that should reach the user as part of the reply.
    Visible(String),
    /// Content that should reach the user as part of a Thinking block.
    Reasoning(String),
}

#[derive(Debug, Default, PartialEq, Eq)]
enum State {
    /// Looking for `<think>`; emit incoming bytes as [`Segment::Visible`].
    #[default]
    OutsideThink,
    /// Looking for `</think>`; emit incoming bytes as [`Segment::Reasoning`].
    InsideThink,
}

const OPEN_TAG: &str = "<think>";
const CLOSE_TAG: &str = "</think>";

/// Stateful `<think>` / `</think>` tag tracker for streamed `content`
/// chunks. Holds at most `CLOSE_TAG.len() - 1 == 7` bytes of partial
/// trailing-tag prefix between calls.
#[derive(Debug, Default)]
pub struct ThinkSplitter {
    state: State,
    /// Carryover bytes from the previous chunk's tail that might be the
    /// start of a tag completing on the next chunk.
    pending: String,
}

impl ThinkSplitter {
    /// Create a fresh splitter that starts in the "outside reasoning"
    /// state — the common case at the start of a turn.
    pub fn new() -> Self {
        Self::default()
    }

    /// Push the next `content` chunk and return zero or more
    /// classified segments in order. Pending bytes carried over from
    /// the previous call are prepended transparently.
    pub fn feed(&mut self, chunk: &str) -> Vec<Segment> {
        let mut out = Vec::new();
        if chunk.is_empty() && self.pending.is_empty() {
            return out;
        }
        // Combine carryover prefix with new bytes; cheap because
        // `pending` is at most 7 chars.
        let mut buf = std::mem::take(&mut self.pending);
        buf.push_str(chunk);

        let mut cursor = 0usize;
        loop {
            let target = match self.state {
                State::OutsideThink => OPEN_TAG,
                State::InsideThink => CLOSE_TAG,
            };
            let haystack = &buf[cursor..];
            if let Some(rel) = haystack.find(target) {
                let abs = cursor + rel;
                if abs > cursor {
                    push_segment(
                        &mut out,
                        self.state == State::InsideThink,
                        &buf[cursor..abs],
                    );
                }
                cursor = abs + target.len();
                self.state = match self.state {
                    State::OutsideThink => State::InsideThink,
                    State::InsideThink => State::OutsideThink,
                };
                continue;
            }
            // No complete tag in the remainder. Check whether the
            // tail is a non-empty prefix of `target`; if so, hold it
            // back for the next `feed()`.
            let tail = &buf[cursor..];
            let mut hold = 0usize;
            for n in (1..target.len()).rev() {
                if n <= tail.len() && tail.ends_with(&target[..n]) {
                    hold = n;
                    break;
                }
            }
            let emit_end = tail.len() - hold;
            if emit_end > 0 {
                push_segment(
                    &mut out,
                    self.state == State::InsideThink,
                    &tail[..emit_end],
                );
            }
            if hold > 0 {
                self.pending.push_str(&tail[emit_end..]);
            }
            break;
        }
        out
    }

    /// Drain on stream end. Emits any pending bytes as a segment of
    /// the current state (defensive: models rarely leave an open tag
    /// dangling; if they do, classifying by current state keeps the
    /// content addressable rather than silently swallowed).
    pub fn finish(&mut self) -> Option<Segment> {
        if self.pending.is_empty() {
            return None;
        }
        let text = std::mem::take(&mut self.pending);
        Some(match self.state {
            State::OutsideThink => Segment::Visible(text),
            State::InsideThink => Segment::Reasoning(text),
        })
    }
}

fn push_segment(out: &mut Vec<Segment>, inside_think: bool, text: &str) {
    if text.is_empty() {
        return;
    }
    // Coalesce with the previous segment when classifications match,
    // so callers see one `Visible("hello world")` instead of two.
    if let Some(last) = out.last_mut() {
        match (last, inside_think) {
            (Segment::Visible(s), false) | (Segment::Reasoning(s), true) => {
                s.push_str(text);
                return;
            }
            _ => {}
        }
    }
    if inside_think {
        out.push(Segment::Reasoning(text.to_string()));
    } else {
        out.push(Segment::Visible(text.to_string()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(chunks: &[&str]) -> Vec<Segment> {
        let mut s = ThinkSplitter::new();
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
        let mut s = ThinkSplitter::new();
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
        let mut s = ThinkSplitter::new();
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
        let mut s = ThinkSplitter::new();
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
        let mut s = ThinkSplitter::new();
        assert!(s.feed("").is_empty());
        assert_eq!(s.feed("hi"), vec![Segment::Visible("hi".into())]);
        assert!(s.feed("").is_empty());
        assert!(s.finish().is_none());
    }
}
