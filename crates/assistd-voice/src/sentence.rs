//! Streaming sentence segmenter between the LLM token stream and TTS.
//! Strips markdown, handles fenced code blocks per [`CodeBlockMode`],
//! and emits whole sentences at prosody boundaries.
//!
//! Boundary priority, highest first: paragraph break `\n\n`; bullet
//! marker `\n- ` or `\n* `; strong terminator `[.!?]` followed by
//! whitespace and an uppercase letter, digit, or newline, with
//! abbreviation and decimal guards; the `max_len` safety net. A
//! terminator at the end of the buffer is never a boundary, because
//! it may be an abbreviation awaiting context; `finish` flushes it.

use std::collections::VecDeque;

pub use assistd_config::CodeBlockMode;

const MAX_LANG_LEN: usize = 32;

const ABBREVIATIONS: &[&str] = &[
    "Mr", "Mrs", "Ms", "Dr", "St", "Sr", "Jr", "Inc", "Ltd", "Co", "etc", "vs", "e.g", "i.e",
    "U.S", "U.K", "approx", "Prof", "Gen", "Capt",
];

/// Streaming sentence segmenter. Feed deltas with [`push`](Self::push),
/// call [`finish`](Self::finish) when the stream ends, and
/// [`flush_idle`](Self::flush_idle) when it pauses.
pub struct SentenceBuffer {
    buf: String,
    in_code_fence: bool,
    pending: String,
    max_len: usize,
    mode: CodeBlockMode,
    capturing_lang: bool,
    lang_buf: String,
}

impl SentenceBuffer {
    /// [`new_with_mode`](Self::new_with_mode) with [`CodeBlockMode::Skip`].
    pub fn new(max_len: usize) -> Self {
        Self::new_with_mode(max_len, CodeBlockMode::Skip)
    }

    pub fn new_with_mode(max_len: usize, mode: CodeBlockMode) -> Self {
        Self {
            buf: String::new(),
            in_code_fence: false,
            pending: String::new(),
            max_len: max_len.max(50),
            mode,
            capturing_lang: false,
            lang_buf: String::new(),
        }
    }

    /// Append a delta and return any sentences that finished as a result.
    pub fn push(&mut self, delta: &str) -> Vec<String> {
        let mut out = Vec::new();
        for ch in delta.chars() {
            self.feed_char(ch, &mut out);
        }
        self.scan_boundaries(&mut out);
        out
    }

    /// Flush the remaining tail as one sentence. An unterminated code
    /// fence is dropped silently.
    pub fn finish(&mut self) -> Option<String> {
        if !self.in_code_fence && !self.pending.is_empty() {
            let pending = std::mem::take(&mut self.pending);
            self.buf.push_str(&strip_inline(&pending));
        }
        self.in_code_fence = false;
        self.pending.clear();
        self.capturing_lang = false;
        self.lang_buf.clear();
        let raw = std::mem::take(&mut self.buf);
        let speech = postprocess_for_speech(&raw);
        if speech.is_empty() {
            None
        } else {
            Some(speech)
        }
    }

    /// Flush up to the last whitespace, leaving a trailing partial
    /// word buffered. `None` inside a code fence, when there is no
    /// whitespace to cut on, or when the prefix is empty after
    /// postprocessing. Fence state is preserved.
    pub fn flush_idle(&mut self) -> Option<String> {
        if self.in_code_fence {
            return None;
        }
        let cut = self.buf.rfind(char::is_whitespace)?;
        let end = cut + self.buf[cut..].chars().next()?.len_utf8();
        let prefix: String = self.buf.drain(..end).collect();
        let speech = postprocess_for_speech(&prefix);
        if speech.is_empty() {
            None
        } else {
            Some(speech)
        }
    }

    fn feed_char(&mut self, ch: char, out: &mut Vec<String>) {
        if ch == '`' {
            self.feed_backtick(out);
            return;
        }
        if self.in_code_fence {
            self.feed_fenced(ch);
            return;
        }

        if !self.pending.is_empty() {
            let pending = std::mem::take(&mut self.pending);
            self.buf.push_str(&strip_inline(&pending));
        }

        let at_line_start = self.buf.is_empty() || self.buf.ends_with('\n');
        if (ch == '#' || ch == '>') && at_line_start {
            return;
        }
        self.buf.push(ch);
        self.scan_boundaries(out);
    }

    /// Backticks accumulate in `pending`; the third in a row toggles
    /// the fence.
    fn feed_backtick(&mut self, out: &mut Vec<String>) {
        self.pending.push('`');
        if !self.pending.ends_with("```") {
            return;
        }
        self.pending.truncate(self.pending.len() - 3);
        if self.in_code_fence {
            self.close_fence(out);
        } else {
            self.open_fence(out);
        }
    }

    fn open_fence(&mut self, out: &mut Vec<String>) {
        self.in_code_fence = true;
        self.flush_buf_to_out(out);
        self.capturing_lang = matches!(self.mode, CodeBlockMode::Summarize);
        self.lang_buf.clear();
    }

    fn close_fence(&mut self, out: &mut Vec<String>) {
        self.in_code_fence = false;
        self.pending.clear();
        if matches!(self.mode, CodeBlockMode::Summarize) {
            let phrase = if self.lang_buf.is_empty() {
                "Code block.".to_string()
            } else {
                format!("Code block in {}.", self.lang_buf)
            };
            out.push(phrase);
        }
        self.capturing_lang = false;
        self.lang_buf.clear();
    }

    /// Inside a fence only the language tag after the opener is kept.
    fn feed_fenced(&mut self, ch: char) {
        if self.capturing_lang {
            if ch.is_whitespace() {
                self.capturing_lang = false;
            } else if self.lang_buf.len() < MAX_LANG_LEN
                && (ch.is_ascii_alphanumeric() || ch == '+' || ch == '-' || ch == '_')
            {
                self.lang_buf.push(ch);
            } else {
                self.capturing_lang = false;
            }
        }
        self.pending.clear();
    }

    fn flush_buf_to_out(&mut self, out: &mut Vec<String>) {
        if self.buf.trim().is_empty() {
            self.buf.clear();
            return;
        }
        let raw = std::mem::take(&mut self.buf);
        let speech = postprocess_for_speech(&raw);
        if !speech.is_empty() {
            out.push(speech);
        }
    }

    fn scan_boundaries(&mut self, out: &mut Vec<String>) {
        while let Some(idx) = find_boundary(&self.buf, self.max_len) {
            let raw = self.buf[..idx].to_string();
            let rest = self.buf[idx..].trim_start().to_string();
            self.buf = rest;
            let sentence = postprocess_for_speech(&raw);
            if !sentence.is_empty() {
                out.push(sentence);
            }
            if self.buf.is_empty() {
                break;
            }
        }
    }
}

/// Byte offset of the first boundary in `buf`, or `None`.
fn find_boundary(buf: &str, max_len: usize) -> Option<usize> {
    if let Some(i) = buf.find("\n\n") {
        return Some(i + 2);
    }
    find_bullet_marker(buf)
        .or_else(|| find_terminator(buf))
        .or_else(|| length_cutoff(buf, max_len))
}

/// Offset just past the whitespace following a `.`, `!`, or `?` that
/// ends a sentence. A terminator at the very end of the buffer never
/// qualifies: it may be an abbreviation awaiting its next word.
fn find_terminator(buf: &str) -> Option<usize> {
    let bytes = buf.as_bytes();
    for (i, &c) in bytes.iter().enumerate() {
        if !matches!(c, b'.' | b'!' | b'?') {
            continue;
        }
        let &next = bytes.get(i + 1)?;
        if !next.is_ascii_whitespace() {
            continue;
        }
        if c == b'.' && (is_decimal_point(bytes, i) || is_abbreviation_at(buf, i)) {
            continue;
        }
        let after_ws = bytes[i + 1..]
            .iter()
            .position(|b| !b.is_ascii_whitespace())
            .map(|off| i + 1 + off)?;
        let succ = bytes[after_ws];
        if succ.is_ascii_uppercase() || succ.is_ascii_digit() || next == b'\n' {
            return Some(after_ws);
        }
    }
    None
}

fn is_decimal_point(bytes: &[u8], i: usize) -> bool {
    i > 0 && bytes[i - 1].is_ascii_digit() && bytes.get(i + 2).is_some_and(u8::is_ascii_digit)
}

/// Cut at the last whitespace before `max_len`, or hard-cut at
/// `max_len` when there is none.
fn length_cutoff(buf: &str, max_len: usize) -> Option<usize> {
    if buf.len() < max_len {
        return None;
    }
    let window = floor_char_boundary(buf, max_len);
    let cut = buf[..window]
        .char_indices()
        .rev()
        .find(|(_, c)| c.is_whitespace())
        .map_or(window, |(i, c)| i + c.len_utf8());
    Some(cut)
}

fn floor_char_boundary(s: &str, mut idx: usize) -> usize {
    idx = idx.min(s.len());
    while !s.is_char_boundary(idx) {
        idx -= 1;
    }
    idx
}

fn find_bullet_marker(buf: &str) -> Option<usize> {
    let bytes = buf.as_bytes();
    let mut i = 1;
    while i + 2 < bytes.len() {
        if bytes[i] == b'\n'
            && (bytes[i + 1] == b'-' || bytes[i + 1] == b'*')
            && bytes[i + 2] == b' '
        {
            return Some(i + 1);
        }
        i += 1;
    }
    None
}

fn is_abbreviation_at(buf: &str, i: usize) -> bool {
    let bytes = buf.as_bytes();
    let mut start = i;
    while start > 0 {
        let prev = bytes[start - 1];
        if prev.is_ascii_whitespace() {
            break;
        }
        start -= 1;
    }
    let token = &buf[start..=i];
    let stem = &token[..token.len() - 1];
    ABBREVIATIONS
        .iter()
        .any(|abbr| stem.eq_ignore_ascii_case(abbr) || stem.ends_with(&format!(".{abbr}")))
}

fn postprocess_for_speech(s: &str) -> String {
    let s = strip_links(s);
    let s = strip_emphasis(&s);
    let s = replace_urls(&s);
    let s = strip_inline(&s);
    collapse_whitespace(&s)
}

fn strip_inline(s: &str) -> String {
    s.replace('`', "")
}

fn strip_links(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '[' {
            let mut text = String::new();
            let mut found_close_bracket = false;
            for inner in chars.by_ref() {
                if inner == ']' {
                    found_close_bracket = true;
                    break;
                }
                text.push(inner);
            }
            if found_close_bracket && chars.peek() == Some(&'(') {
                chars.next();
                let mut consumed_paren = false;
                for url_c in chars.by_ref() {
                    if url_c == ')' {
                        consumed_paren = true;
                        break;
                    }
                }
                if consumed_paren {
                    out.push_str(&text);
                    continue;
                } else {
                    out.push('[');
                    out.push_str(&text);
                    out.push_str("](");
                    continue;
                }
            }
            out.push('[');
            out.push_str(&text);
            if found_close_bracket {
                out.push(']');
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// Drop runs of one to three `*` or `_`.
fn strip_emphasis(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut buf = VecDeque::<char>::new();
    for c in s.chars() {
        if c == '*' || c == '_' {
            buf.push_back(c);
            if buf.len() >= 3 {
                buf.clear();
            }
        } else {
            if !buf.is_empty() {
                buf.clear();
            }
            out.push(c);
        }
    }
    out
}

fn replace_urls(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let scheme = if s[i..].starts_with("https://") {
            Some(8)
        } else if s[i..].starts_with("http://") {
            Some(7)
        } else {
            None
        };
        if let Some(skip) = scheme {
            let prev_ok = i == 0 || !bytes[i - 1].is_ascii_alphanumeric();
            if prev_ok {
                let mut j = i + skip;
                while j < bytes.len() && !bytes[j].is_ascii_whitespace() {
                    j += 1;
                }
                out.push_str("link");
                i = j;
                continue;
            }
        }
        let ch = s[i..]
            .chars()
            .next()
            .expect("non-empty since i < bytes.len()");
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

fn collapse_whitespace(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut prev_ws = false;
    for c in s.chars() {
        if c.is_whitespace() {
            if !prev_ws {
                out.push(' ');
            }
            prev_ws = true;
        } else {
            out.push(c);
            prev_ws = false;
        }
    }
    out.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn push_all(buf: &mut SentenceBuffer, deltas: &[&str]) -> Vec<String> {
        let mut out = Vec::new();
        for d in deltas {
            out.extend(buf.push(d));
        }
        out
    }

    #[test]
    fn flushes_on_period_then_capital() {
        let mut b = SentenceBuffer::new(400);
        let s = b.push("Hello world. Then more.");
        assert_eq!(s, vec!["Hello world."]);
        let tail = b.finish();
        assert_eq!(tail.as_deref(), Some("Then more."));
    }

    #[test]
    fn streamed_chunks_assemble_to_sentences() {
        let mut b = SentenceBuffer::new(400);
        let out = push_all(&mut b, &["Hel", "lo wo", "rld. ", "Then ", "more."]);
        assert_eq!(out, vec!["Hello world."]);
        assert_eq!(b.finish().as_deref(), Some("Then more."));
    }

    #[test]
    fn does_not_split_on_abbreviation() {
        let mut b = SentenceBuffer::new(400);
        let s = b.push("Dr. Smith arrived. Then he left.");
        assert_eq!(s, vec!["Dr. Smith arrived."]);
        assert_eq!(b.finish().as_deref(), Some("Then he left."));
    }

    #[test]
    fn does_not_split_on_eg() {
        let mut b = SentenceBuffer::new(400);
        let s = b.push("Use a tool, e.g. grep. It works.");
        assert_eq!(s, vec!["Use a tool, e.g. grep."]);
        assert_eq!(b.finish().as_deref(), Some("It works."));
    }

    #[test]
    fn does_not_split_on_decimal() {
        let mut b = SentenceBuffer::new(400);
        let s = b.push("Pi is about 3.14159 here. End.");
        assert_eq!(s, vec!["Pi is about 3.14159 here."]);
        assert_eq!(b.finish().as_deref(), Some("End."));
    }

    #[test]
    fn flushes_on_paragraph_break() {
        let mut b = SentenceBuffer::new(400);
        let s = b.push("First paragraph\n\nSecond starts");
        assert_eq!(s, vec!["First paragraph"]);
        assert_eq!(b.finish().as_deref(), Some("Second starts"));
    }

    #[test]
    fn drops_fenced_code_block() {
        let mut b = SentenceBuffer::new(400);
        let s = push_all(
            &mut b,
            &[
                "Here is code: ",
                "```rust\nfn main() { println!(\"hi\"); }\n```",
                " That was a snippet.",
            ],
        );
        let joined = s.join(" | ");
        assert!(
            !joined.contains("println"),
            "code leaked into TTS: {joined:?}"
        );
        let tail = b.finish().unwrap_or_default();
        assert!(!tail.contains("println"), "code leaked into tail: {tail:?}");
    }

    #[test]
    fn strips_markdown_link_to_text() {
        let mut b = SentenceBuffer::new(400);
        let s = b.push("See [the docs](https://example.com/docs) please.");
        let tail = b.finish().unwrap_or_default();
        let joined = s.join(" ") + " " + &tail;
        assert!(joined.contains("the docs"));
        assert!(!joined.contains("example.com"));
        assert!(!joined.contains('['));
    }

    #[test]
    fn replaces_bare_url_with_link() {
        let mut b = SentenceBuffer::new(400);
        let _ = b.push("Visit https://example.com for more. Bye.");
        let s = b.push("");
        assert!(s.is_empty() || s.iter().any(|t| t.contains("link")));
        let tail = b.finish().unwrap_or_default();
        let joined = format!("{} {}", s.join(" "), tail);
        assert!(!joined.contains("example.com"), "got {joined:?}");
    }

    #[test]
    fn handles_multibyte_chars_around_no_url() {
        let mut b = SentenceBuffer::new(400);
        let mut all = b.push("That’s a famous line from John F. Kennedy. ");
        if let Some(t) = b.finish() {
            all.push(t);
        }
        let joined = all.join(" ");
        assert!(joined.contains("That’s"), "got {joined:?}");
        assert!(joined.contains("Kennedy"), "got {joined:?}");
    }

    #[test]
    fn handles_multibyte_chars_with_url() {
        let mut b = SentenceBuffer::new(400);
        let mut all = b.push("It’s at https://example.com, really. ");
        if let Some(t) = b.finish() {
            all.push(t);
        }
        let joined = all.join(" ");
        assert!(joined.contains("It’s"), "got {joined:?}");
        assert!(!joined.contains("example.com"), "got {joined:?}");
    }

    #[test]
    fn strips_emphasis_markers() {
        let mut b = SentenceBuffer::new(400);
        let _ = b.push("This is *important* and **very urgent**.");
        let tail = b.finish().unwrap_or_default();
        assert!(!tail.contains('*'));
        assert!(tail.contains("important"));
        assert!(tail.contains("very urgent"));
    }

    #[test]
    fn strips_heading_markers_at_line_start() {
        let mut b = SentenceBuffer::new(400);
        let mut all = b.push("# A Heading\n\nContent here.");
        if let Some(t) = b.finish() {
            all.push(t);
        }
        let joined = all.join(" ");
        assert!(!joined.contains('#'), "got {joined:?}");
        assert!(joined.contains("A Heading"), "got {joined:?}");
        assert!(joined.contains("Content here."), "got {joined:?}");
    }

    #[test]
    fn length_cap_flushes_at_whitespace() {
        let mut b = SentenceBuffer::new(50);
        let s = b.push("aaaaaaaaa bbbbbbbbb ccccccccc ddddddddd eeeeeeeee fffffffff");
        assert!(
            !s.is_empty(),
            "length cap should have produced at least one flush"
        );
        assert!(
            !s[0].ends_with(['a', 'b', 'c']),
            "should not split mid-word: {:?}",
            s[0]
        );
        let tail = b.finish().unwrap_or_default();
        let total: usize = s.iter().map(|x| x.len()).sum::<usize>() + tail.len();
        assert!(total > 0, "should have spoken something total");
    }

    #[test]
    fn flushes_remaining_on_finish() {
        let mut b = SentenceBuffer::new(400);
        let s = b.push("Unfinished thought");
        assert!(s.is_empty());
        assert_eq!(b.finish().as_deref(), Some("Unfinished thought"));
    }

    #[test]
    fn finish_returns_none_if_only_fence_left_open() {
        let mut b = SentenceBuffer::new(400);
        let _ = b.push("```rust\nfn main() {");
        assert_eq!(b.finish(), None);
    }

    #[test]
    fn handles_question_mark_terminator() {
        let mut b = SentenceBuffer::new(400);
        let s = b.push("Are you sure? Yes I am.");
        assert_eq!(s, vec!["Are you sure?"]);
        assert_eq!(b.finish().as_deref(), Some("Yes I am."));
    }

    #[test]
    fn handles_exclamation_terminator() {
        let mut b = SentenceBuffer::new(400);
        let s = b.push("Wow! That works.");
        assert_eq!(s, vec!["Wow!"]);
        assert_eq!(b.finish().as_deref(), Some("That works."));
    }

    #[test]
    fn collapses_inner_whitespace() {
        let mut b = SentenceBuffer::new(400);
        let s = b.push("a   b\t\tc. End.");
        assert_eq!(s, vec!["a b c.".to_string()]);
        assert_eq!(b.finish().as_deref(), Some("End."));
    }

    #[test]
    fn does_not_split_mid_word_on_period_then_lowercase() {
        let mut b = SentenceBuffer::new(400);
        let _ = b.push("file.txt is here. Done.");
        let tail = b.finish().unwrap_or_default();
        assert!(tail == "Done." || tail.is_empty());
    }

    #[test]
    fn no_split_when_period_followed_by_lowercase() {
        let mut b = SentenceBuffer::new(400);
        let s = b.push("End. then continue.");
        assert!(s.is_empty(), "should not split on lowercase succ: {s:?}");
        let tail = b.finish().unwrap_or_default();
        assert_eq!(tail, "End. then continue.");
    }

    #[test]
    fn newline_after_period_counts_as_boundary() {
        let mut b = SentenceBuffer::new(400);
        let s = b.push("Item one.\nItem two.");
        assert!(!s.is_empty());
    }

    // ---- flush_idle ----

    #[test]
    fn flush_idle_emits_at_last_whitespace() {
        let mut b = SentenceBuffer::new(400);
        let _ = b.push("I am writ");
        let out = b.flush_idle();
        assert_eq!(out.as_deref(), Some("I am"));
        let s = b.push("ing now. Done.");
        assert_eq!(s, vec!["writing now."]);
        assert_eq!(b.finish().as_deref(), Some("Done."));
    }

    #[test]
    fn flush_idle_returns_none_on_pure_whitespace() {
        let mut b = SentenceBuffer::new(400);
        let _ = b.push("   \t  ");
        assert!(b.flush_idle().is_none());
    }

    #[test]
    fn flush_idle_returns_none_in_code_fence() {
        let mut b = SentenceBuffer::new(400);
        let _ = b.push("```rust\nfn main");
        assert!(b.flush_idle().is_none());
    }

    #[test]
    fn flush_idle_returns_none_on_single_partial_word() {
        let mut b = SentenceBuffer::new(400);
        let _ = b.push("writ");
        assert!(b.flush_idle().is_none());
    }

    #[test]
    fn flush_idle_can_be_called_repeatedly_without_loss() {
        let mut b = SentenceBuffer::new(400);
        let _ = b.push("Hello world ");
        let first = b.flush_idle();
        assert_eq!(first.as_deref(), Some("Hello world"));
        assert!(b.flush_idle().is_none());
        let s = b.push("again. End.");
        assert_eq!(s, vec!["again.".to_string()]);
        assert_eq!(b.finish().as_deref(), Some("End."));
    }

    // ---- code-block mode ----

    #[test]
    fn code_block_skip_drops_content_default() {
        let mut b = SentenceBuffer::new(400);
        let s = push_all(
            &mut b,
            &["Prelude. ", "```rust\nfn main() {}\n```", " Tail end."],
        );
        let joined = s.join(" | ");
        let tail = b.finish().unwrap_or_default();
        let all = format!("{joined} {tail}");
        assert!(!all.contains("fn main"), "code leaked: {all:?}");
        assert!(!all.contains("Code block"), "summary leaked: {all:?}");
    }

    #[test]
    fn code_block_summarize_emits_phrase_with_lang() {
        let mut b = SentenceBuffer::new_with_mode(400, CodeBlockMode::Summarize);
        let s = push_all(
            &mut b,
            &["Prelude. ", "```rust\nfn main() {}\n```", " Tail end."],
        );
        let joined = s.join(" | ");
        let tail = b.finish().unwrap_or_default();
        let all = format!("{joined} {tail}");
        assert!(!all.contains("fn main"), "code leaked: {all:?}");
        assert!(
            all.contains("Code block in rust"),
            "expected lang-tagged summary: {all:?}"
        );
        // Order: prelude before summary before tail.
        let prelude_pos = all.find("Prelude").expect("prelude present");
        let summary_pos = all.find("Code block").expect("summary present");
        let tail_pos = all.find("Tail end").expect("tail present");
        assert!(prelude_pos < summary_pos, "got: {all:?}");
        assert!(summary_pos < tail_pos, "got: {all:?}");
    }

    #[test]
    fn code_block_summarize_no_lang_tag() {
        let mut b = SentenceBuffer::new_with_mode(400, CodeBlockMode::Summarize);
        let s = push_all(&mut b, &["```\nopaque content\n```", " After."]);
        assert!(
            s.iter().any(|x| x == "Code block."),
            "expected exactly \"Code block.\": {s:?}"
        );
        assert!(
            !s.iter().any(|x| x.contains("Code block in")),
            "should not have a lang suffix: {s:?}"
        );
    }

    #[test]
    fn code_block_summarize_lang_tag_capped_no_panic() {
        let mut b = SentenceBuffer::new_with_mode(400, CodeBlockMode::Summarize);
        let long_lang: String = "a".repeat(100);
        let _ = b.push(&format!("```{long_lang}\nfoo\n```"));
        let _ = b.finish();
    }

    #[test]
    fn hard_cuts_input_with_no_boundary() {
        let mut b = SentenceBuffer::new(50);
        let blob: String = std::iter::repeat_n('a', 600).collect();
        let out = b.push(&blob);
        assert!(
            !out.is_empty(),
            "expected hard-cut sentence(s), got nothing"
        );
        let total_emitted: usize = out.iter().map(|s| s.len()).sum();
        assert!(
            total_emitted > 0,
            "hard cut emitted only empty strings: {out:?}"
        );
    }

    #[test]
    fn natural_boundaries_split_normal_input() {
        let mut b = SentenceBuffer::new(400);
        let out = b.push("First. Second. Third. Fourth. Fifth. Sixth.");
        let tail = b.finish().unwrap_or_default();
        let combined = out.join(" ") + " " + &tail;
        assert_eq!(
            combined.matches('.').count(),
            6,
            "natural-boundary input was not split on terminators: {combined:?}"
        );
    }

    #[test]
    fn code_block_summarize_emits_only_after_close() {
        let mut b = SentenceBuffer::new_with_mode(400, CodeBlockMode::Summarize);
        let _ = b.push("```python\nprint('hi')\n");
        let s = b.push("");
        assert!(
            s.iter().all(|x| !x.contains("Code block")),
            "summary emitted prematurely: {s:?}"
        );
        let s2 = b.push("```");
        assert!(
            s2.iter().any(|x| x.contains("Code block in python")),
            "expected summary on close: {s2:?}"
        );
    }

    #[test]
    fn length_safety_net_handles_multibyte_chars() {
        let mut b = SentenceBuffer::new(50);
        let blob: String = std::iter::repeat_n('\u{1F600}', 13).collect();
        let out = b.push(&blob);
        let combined = out.join("") + &b.finish().unwrap_or_default();
        assert_eq!(combined.matches('\u{1F600}').count(), 13);
    }

    #[test]
    fn length_safety_net_splits_multibyte_words_on_whitespace() {
        let mut b = SentenceBuffer::new(50);
        let blob: String = std::iter::repeat_n("\u{1F600}\u{1F600} ", 20).collect();
        let out = b.push(&blob);
        assert!(!out.is_empty());
        let combined = out.join("") + &b.finish().unwrap_or_default();
        assert_eq!(combined.matches('\u{1F600}').count(), 40);
    }

    #[test]
    fn length_safety_net_cuts_after_multibyte_whitespace() {
        for ws in ['\u{3000}', '\u{a0}'] {
            let mut b = SentenceBuffer::new(50);
            let blob = format!("{}{ws}{}", "a".repeat(45), "b".repeat(10));
            let out = b.push(&blob);
            assert_eq!(out, vec!["a".repeat(45)], "whitespace {ws:?}");
            assert_eq!(b.finish().as_deref(), Some("bbbbbbbbbb"));
        }
    }
}
