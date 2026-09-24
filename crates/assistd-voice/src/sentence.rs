//! Streaming sentence segmenter between the LLM token stream and TTS.
//! Strips markdown, handles fenced code blocks per [`CodeBlockMode`],
//! and emits whole sentences at prosody boundaries.

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
///
/// Boundary priority, highest first: paragraph break `\n\n`; bullet
/// marker `\n- ` or `\n* `; strong terminator `[.!?]` followed by
/// whitespace and an uppercase letter, digit, or newline, with
/// abbreviation and decimal guards; the `max_len` safety net. A
/// terminator at the end of the buffer is never a boundary, because it
/// may be an abbreviation awaiting context; `finish` flushes it.
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

    /// A segmenter that handles fenced code blocks per `mode`. `max_len`
    /// is raised to at least 50 bytes.
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
/// ends a sentence.
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

/// Drops every `*` and `_`.
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

    fn run(mut b: SentenceBuffer, deltas: &[&str]) -> (Vec<String>, Option<String>) {
        let emitted = deltas.iter().flat_map(|d| b.push(d)).collect();
        (emitted, b.finish())
    }

    /// `(label, deltas, emitted sentences, finish tail)`, each run
    /// through a fresh buffer from `make`.
    type Case<'a> = (&'a str, &'a [&'a str], &'a [&'a str], Option<&'a str>);

    fn check(make: impl Fn() -> SentenceBuffer, cases: &[Case<'_>]) {
        for &(name, deltas, emitted, tail) in cases {
            let (got, got_tail) = run(make(), deltas);
            assert_eq!(got, emitted, "{name}: emitted");
            assert_eq!(got_tail.as_deref(), tail, "{name}: tail");
        }
    }

    #[test]
    fn splits_at_sentence_boundaries() {
        check(
            || SentenceBuffer::new(400),
            &[
                (
                    "period then capital",
                    &["Hello world. Then more."],
                    &["Hello world."],
                    Some("Then more."),
                ),
                (
                    "streamed chunks",
                    &["Hel", "lo wo", "rld. ", "Then ", "more."],
                    &["Hello world."],
                    Some("Then more."),
                ),
                (
                    "several sentences in one delta",
                    &["First. Second. Third. Fourth. Fifth. Sixth."],
                    &["First.", "Second.", "Third.", "Fourth.", "Fifth."],
                    Some("Sixth."),
                ),
                (
                    "question mark",
                    &["Are you sure? Yes I am."],
                    &["Are you sure?"],
                    Some("Yes I am."),
                ),
                (
                    "exclamation mark",
                    &["Wow! That works."],
                    &["Wow!"],
                    Some("That works."),
                ),
                (
                    "abbreviation",
                    &["Dr. Smith arrived. Then he left."],
                    &["Dr. Smith arrived."],
                    Some("Then he left."),
                ),
                (
                    "dotted abbreviation",
                    &["Use a tool, e.g. grep. It works."],
                    &["Use a tool, e.g. grep."],
                    Some("It works."),
                ),
                (
                    "decimal",
                    &["Pi is about 3.14159 here. End."],
                    &["Pi is about 3.14159 here."],
                    Some("End."),
                ),
                (
                    "period inside a word",
                    &["file.txt is here. Done."],
                    &["file.txt is here."],
                    Some("Done."),
                ),
                (
                    "period then lowercase",
                    &["End. then continue."],
                    &[],
                    Some("End. then continue."),
                ),
                (
                    "period then newline",
                    &["Item one.\nItem two."],
                    &["Item one."],
                    Some("Item two."),
                ),
                (
                    "paragraph break",
                    &["First paragraph\n\nSecond starts"],
                    &["First paragraph"],
                    Some("Second starts"),
                ),
                (
                    "no terminator",
                    &["Unfinished thought"],
                    &[],
                    Some("Unfinished thought"),
                ),
            ],
        );
    }

    #[test]
    fn rewrites_markdown_for_speech() {
        check(
            || SentenceBuffer::new(400),
            &[
                (
                    "link keeps its text",
                    &["See [the docs](https://example.com/docs) please."],
                    &[],
                    Some("See the docs please."),
                ),
                (
                    "bare url becomes link",
                    &["Visit https://example.com for more. Bye."],
                    &["Visit link for more."],
                    Some("Bye."),
                ),
                (
                    "multibyte text",
                    &["That’s a famous line from Kennedy. "],
                    &[],
                    Some("That’s a famous line from Kennedy."),
                ),
                (
                    "multibyte text with url",
                    &["It’s at https://example.com, really. "],
                    &[],
                    Some("It’s at link really."),
                ),
                (
                    "emphasis",
                    &["This is *important* and **very urgent**."],
                    &[],
                    Some("This is important and very urgent."),
                ),
                (
                    "heading marker",
                    &["# A Heading\n\nContent here."],
                    &["A Heading"],
                    Some("Content here."),
                ),
                (
                    "inner whitespace",
                    &["a   b\t\tc. End."],
                    &["a b c."],
                    Some("End."),
                ),
            ],
        );
    }

    #[test]
    fn skip_mode_drops_code_blocks() {
        check(
            || SentenceBuffer::new(400),
            &[
                (
                    "closed fence",
                    &[
                        "Here is code: ",
                        "```rust\nfn main() { println!(\"hi\"); }\n```",
                        " That was a snippet.",
                    ],
                    &["Here is code:"],
                    Some("That was a snippet."),
                ),
                ("unterminated fence", &["```rust\nfn main() {"], &[], None),
            ],
        );
    }

    #[test]
    fn summarize_mode_replaces_code_blocks_with_a_phrase() {
        check(
            || SentenceBuffer::new_with_mode(400, CodeBlockMode::Summarize),
            &[
                (
                    "with language",
                    &["Prelude. ", "```rust\nfn main() {}\n```", " Tail end."],
                    &["Prelude.", "Code block in rust."],
                    Some("Tail end."),
                ),
                (
                    "without language",
                    &["```\nopaque content\n```", " After."],
                    &["Code block."],
                    Some("After."),
                ),
            ],
        );
    }

    #[test]
    fn summarize_mode_caps_the_language_tag() {
        let fence = format!("```{}\nfoo\n```", "a".repeat(100));
        let (emitted, tail) = run(
            SentenceBuffer::new_with_mode(400, CodeBlockMode::Summarize),
            &[&fence],
        );
        assert_eq!(
            emitted,
            [format!("Code block in {}.", "a".repeat(MAX_LANG_LEN))]
        );
        assert_eq!(tail, None);
    }

    #[test]
    fn summarize_mode_emits_on_close() {
        let mut b = SentenceBuffer::new_with_mode(400, CodeBlockMode::Summarize);
        assert!(b.push("```python\nprint('hi')\n").is_empty());
        assert_eq!(b.push("```"), ["Code block in python."]);
    }

    #[test]
    fn length_safety_net_cuts_at_last_whitespace() {
        let (emitted, tail) = run(
            SentenceBuffer::new(50),
            &["aaaaaaaaaa bbbbbbbbbb cccccccccc dddddddddd eeeeeeeeee ffff"],
        );
        assert_eq!(emitted, ["aaaaaaaaaa bbbbbbbbbb cccccccccc dddddddddd"]);
        assert_eq!(tail.as_deref(), Some("eeeeeeeeee ffff"));
    }

    #[test]
    fn length_safety_net_hard_cuts_on_a_char_boundary_without_whitespace() {
        let (emitted, tail) = run(SentenceBuffer::new(50), &[&"a".repeat(600)]);
        assert_eq!(emitted, vec!["a".repeat(50); 12]);
        assert_eq!(tail, None);

        let (emitted, tail) = run(SentenceBuffer::new(50), &[&"😀".repeat(13)]);
        assert_eq!(emitted, ["😀".repeat(12)]);
        assert_eq!(tail.as_deref(), Some("😀"));
    }

    #[test]
    fn length_safety_net_keeps_multibyte_words_whole() {
        let (mut out, tail) = run(SentenceBuffer::new(50), &[&"😀😀 ".repeat(20)]);
        assert!(!out.is_empty());
        out.extend(tail);
        for s in &out {
            assert!(s.split(' ').all(|w| w == "😀😀"), "split mid-word: {s:?}");
        }
        assert_eq!(out.concat().matches('😀').count(), 40);
    }

    #[test]
    fn length_safety_net_cuts_after_multibyte_whitespace() {
        for ws in ['\u{3000}', '\u{a0}'] {
            let blob = format!("{}{ws}{}", "a".repeat(45), "b".repeat(10));
            let (emitted, tail) = run(SentenceBuffer::new(50), &[&blob]);
            assert_eq!(emitted, ["a".repeat(45)], "whitespace {ws:?}");
            assert_eq!(tail.as_deref(), Some("bbbbbbbbbb"), "whitespace {ws:?}");
        }
    }

    #[test]
    fn flush_idle_emits_at_last_whitespace() {
        let mut b = SentenceBuffer::new(400);
        let _ = b.push("I am writ");
        assert_eq!(b.flush_idle().as_deref(), Some("I am"));
        assert_eq!(b.push("ing now. Done."), ["writing now."]);
        assert_eq!(b.finish().as_deref(), Some("Done."));
    }

    #[test]
    fn flush_idle_returns_none_when_nothing_is_speakable() {
        for input in ["   \t  ", "```rust\nfn main", "writ"] {
            let mut b = SentenceBuffer::new(400);
            let _ = b.push(input);
            assert_eq!(b.flush_idle(), None, "{input:?}");
        }
    }

    #[test]
    fn flush_idle_can_be_called_repeatedly_without_loss() {
        let mut b = SentenceBuffer::new(400);
        let _ = b.push("Hello world ");
        assert_eq!(b.flush_idle().as_deref(), Some("Hello world"));
        assert_eq!(b.flush_idle(), None);
        assert_eq!(b.push("again. End."), ["again."]);
        assert_eq!(b.finish().as_deref(), Some("End."));
    }
}
