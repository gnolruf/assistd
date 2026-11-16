//! Stateful classifier that separates `<think>...</think>` reasoning
//! from visible content as `delta.content` chunks stream in.

/// One classified slice emitted by [`ThinkSplitter::feed`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Segment {
    /// Text outside any `<think>` block.
    Visible(String),
    /// Text inside a `<think>` block.
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
/// chunks. A tag split across chunks is held back until it completes, so
/// at most `CLOSE_TAG.len() - 1` bytes carry over between calls.
#[derive(Debug, Default)]
pub struct ThinkSplitter {
    state: State,
    /// Tail of the previous chunk that may begin a tag.
    pending: String,
}

impl ThinkSplitter {
    /// Push the next `content` chunk and return its classified segments in
    /// order, prepending any bytes held back from the previous call.
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
            let tail = &buf[cursor..];
            let hold = partial_tag_len(tail, target);
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

/// Length of the longest proper prefix of `tag` that `tail` ends with.
fn partial_tag_len(tail: &str, tag: &str) -> usize {
    (1..tag.len())
        .rev()
        .find(|&n| n <= tail.len() && tail.ends_with(&tag[..n]))
        .unwrap_or(0)
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
