//! Streaming sentence segmenter between the LLM token stream and TTS: strips
//! markdown and drops fenced code blocks.

const ABBREVIATIONS: &[&str] = &[
    "Mr", "Mrs", "Ms", "Dr", "St", "Sr", "Jr", "Inc", "Ltd", "Co", "etc", "vs", "e.g", "i.e",
    "U.S", "U.K", "approx", "Prof", "Gen", "Capt",
];

/// Terminators that end a sentence without needing whitespace or a capital after them.
const SELF_DELIMITING_TERMINATORS: &[char] = &[
    '。', '！', '？', '｡', '।', '॥', '؟', '۔', '።', '፧', '။', '։',
];

/// Streaming sentence segmenter: [`push`](Self::push) deltas, then
/// [`finish`](Self::finish) at end of stream or [`flush_idle`](Self::flush_idle) on a pause.
///
/// Boundaries, highest priority first: `\n\n`; a `\n- `/`\n* ` bullet; a terminator
/// such as `。` or `।`, or `[.!?]` plus whitespace and a non-lowercase letter, digit, or
/// newline (abbreviations and decimals excluded); the `max_len` cap in chars. A terminator
/// at the buffer's end waits for context.
#[derive(Debug)]
pub struct SentenceBuffer {
    buf: String,
    in_code_fence: bool,
    pending: String,
    max_len: usize,
}

impl SentenceBuffer {
    /// A segmenter cutting sentences at `max_len` chars, raised to at least 50.
    pub fn new(max_len: usize) -> Self {
        Self {
            buf: String::new(),
            in_code_fence: false,
            pending: String::new(),
            max_len: max_len.max(50),
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
        let raw = std::mem::take(&mut self.buf);
        let speech = postprocess_for_speech(&raw);
        (!speech.is_empty()).then_some(speech)
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
        (!speech.is_empty()).then_some(speech)
    }

    fn feed_char(&mut self, ch: char, out: &mut Vec<String>) {
        if ch == '`' {
            self.feed_backtick(out);
            return;
        }
        if self.in_code_fence {
            self.pending.clear();
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
            self.close_fence();
        } else {
            self.open_fence(out);
        }
    }

    fn open_fence(&mut self, out: &mut Vec<String>) {
        self.in_code_fence = true;
        self.flush_buf_to_out(out);
    }

    fn close_fence(&mut self) {
        self.in_code_fence = false;
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

/// Offset just past a sentence terminator and the whitespace following it.
fn find_terminator(buf: &str) -> Option<usize> {
    let mut chars = buf.char_indices().peekable();
    while let Some((i, c)) = chars.next() {
        let &(next_at, next) = chars.peek()?;
        if SELF_DELIMITING_TERMINATORS.contains(&c) {
            return Some(next_at);
        }
        if !matches!(c, '.' | '!' | '?') || !next.is_whitespace() {
            continue;
        }
        if c == '.' && (is_decimal_point(buf.as_bytes(), i) || is_abbreviation_at(buf, i)) {
            continue;
        }
        let (after_ws, successor) = buf[next_at..]
            .char_indices()
            .find(|(_, ch)| !ch.is_whitespace())
            .map(|(off, ch)| (next_at + off, ch))?;
        if starts_sentence(successor) || next == '\n' {
            return Some(after_ws);
        }
    }
    None
}

/// Uppercase or uncased letters (CJK, Devanagari, Arabic, ...) and digits.
fn starts_sentence(c: char) -> bool {
    c.is_ascii_digit() || (c.is_alphabetic() && !c.is_lowercase())
}

fn is_decimal_point(bytes: &[u8], i: usize) -> bool {
    i > 0 && bytes[i - 1].is_ascii_digit() && bytes.get(i + 2).is_some_and(u8::is_ascii_digit)
}

/// Cut at the last whitespace within the first `max_len` chars, or
/// hard-cut after them when there is none.
fn length_cutoff(buf: &str, max_len: usize) -> Option<usize> {
    let window = buf
        .char_indices()
        .map(|(i, _)| i)
        .chain(std::iter::once(buf.len()))
        .nth(max_len)?;
    let cut = buf[..window]
        .char_indices()
        .rev()
        .find(|(_, c)| c.is_whitespace())
        .map_or(window, |(i, c)| i + c.len_utf8());
    Some(cut)
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
                }
                out.push('[');
                out.push_str(&text);
                out.push_str("](");
                continue;
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

fn strip_emphasis(s: &str) -> String {
    s.chars().filter(|&c| c != '*' && c != '_').collect()
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

    fn run(mut buffer: SentenceBuffer, deltas: &[&str]) -> (Vec<String>, Option<String>) {
        let emitted = deltas.iter().flat_map(|delta| buffer.push(delta)).collect();
        (emitted, buffer.finish())
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
                    "abbreviation",
                    &["Dr. Smith arrived. Then he left."],
                    &["Dr. Smith arrived."],
                    Some("Then he left."),
                ),
                (
                    "decimal",
                    &["Pi is about 3.14159 here. End."],
                    &["Pi is about 3.14159 here."],
                    Some("End."),
                ),
                (
                    "period then lowercase",
                    &["End. then continue."],
                    &[],
                    Some("End. then continue."),
                ),
                (
                    "unspaced CJK terminators",
                    &["你好。今天", "天气很好！你呢？"],
                    &["你好。", "今天天气很好！"],
                    Some("你呢？"),
                ),
                (
                    "uncased script after period",
                    &["It is done. यह ठीक है। Next one."],
                    &["It is done.", "यह ठीक है।"],
                    Some("Next one."),
                ),
                (
                    "paragraph break",
                    &["First paragraph\n\nSecond starts"],
                    &["First paragraph"],
                    Some("Second starts"),
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
            ],
        );
    }

    #[test]
    fn code_blocks_are_dropped() {
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

        let (emitted, tail) = run(SentenceBuffer::new(50), &[&"字".repeat(60)]);
        assert_eq!(emitted, ["字".repeat(50)]);
        assert_eq!(tail, Some("字".repeat(10)));
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
    fn flush_idle_can_be_called_repeatedly_without_loss() {
        let mut buffer = SentenceBuffer::new(400);
        let _ = buffer.push("Hello world ");
        assert_eq!(buffer.flush_idle().as_deref(), Some("Hello world"));
        assert_eq!(buffer.flush_idle(), None);
        assert_eq!(buffer.push("again. End."), ["again."]);
        assert_eq!(buffer.finish().as_deref(), Some("End."));
    }
}
