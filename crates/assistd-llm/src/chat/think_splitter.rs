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

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
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
                    push_segment(&mut out, self.state, &buf[cursor..abs]);
                }
                cursor = abs + target.len();
                self.state = match self.state {
                    State::OutsideThink => State::InsideThink,
                    State::InsideThink => State::OutsideThink,
                };
                continue;
            }
            // Hold back a tail that could be the start of `target`.
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
                push_segment(&mut out, self.state, &tail[..emit_end]);
            }
            if hold > 0 {
                self.pending.push_str(&tail[emit_end..]);
            }
            break;
        }
        out
    }

    /// Drain on stream end, classifying any held-back bytes by the
    /// current state so a dangling partial tag is never swallowed.
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

/// Append `text` classified by `state`, coalescing with the previous
/// segment when the classification matches.
fn push_segment(out: &mut Vec<Segment>, state: State, text: &str) {
    if text.is_empty() {
        return;
    }
    if let Some(last) = out.last_mut() {
        match (last, state) {
            (Segment::Visible(prev), State::OutsideThink)
            | (Segment::Reasoning(prev), State::InsideThink) => {
                prev.push_str(text);
                return;
            }
            _ => {}
        }
    }
    out.push(match state {
        State::OutsideThink => Segment::Visible(text.to_string()),
        State::InsideThink => Segment::Reasoning(text.to_string()),
    });
}

#[cfg(test)]
mod tests;
