//! Cuts text into the chunks that are embedded: overlapping character windows for
//! messages, whole paragraphs under their headings for documents.

use std::ops::Range;

use sha2::{Digest, Sha256};

/// The line prefixes that open or close a Markdown fenced code block.
const CODE_FENCES: [&str; 2] = ["```", "~~~"];

/// Bytes of a text's SHA-256 kept as its [`TextHash`].
const TEXT_HASH_BYTES: usize = 16;

/// What a chunk's vector is stored under: the first bytes of the SHA-256 of its text.
pub(crate) type TextHash = [u8; TEXT_HASH_BYTES];

/// How a text is cut into chunks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Chunker {
    /// Windows of `chunk_chars` characters, each sharing `overlap_chars` with the next.
    Windows,
    /// Whole paragraphs packed up to `chunk_chars`, a Markdown heading always opening
    /// a chunk; only a paragraph longer than that is cut into windows.
    Paragraphs,
}

/// Chunking policy; construction guarantees `overlap_chars < chunk_chars`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ChunkingConfig {
    chunker: Chunker,
    /// Maximum number of Unicode characters per chunk.
    chunk_chars: usize,
    /// Number of characters shared between consecutive windows.
    overlap_chars: usize,
}

impl ChunkingConfig {
    /// The policy applied to stored user and assistant messages.
    pub(crate) const MESSAGES: Self = Self::new(Chunker::Windows, 512, 64);

    /// The policy applied to indexed documents.
    pub(crate) const DOCUMENTS: Self = Self::new(Chunker::Paragraphs, 512, 64);

    /// Panics, at compile time when used in a const, unless `overlap_chars < chunk_chars`.
    pub(crate) const fn new(chunker: Chunker, chunk_chars: usize, overlap_chars: usize) -> Self {
        assert!(
            overlap_chars < chunk_chars,
            "chunk overlap must be shorter than the chunk"
        );
        Self {
            chunker,
            chunk_chars,
            overlap_chars,
        }
    }

    pub(crate) const fn chunker(self) -> Chunker {
        self.chunker
    }

    pub(crate) const fn chunk_chars(self) -> usize {
        self.chunk_chars
    }

    pub(crate) const fn overlap_chars(self) -> usize {
        self.overlap_chars
    }

    fn fits(self, text: &str) -> bool {
        char_offset(text, self.chunk_chars).is_none()
    }
}

/// A paragraph together with the Markdown headings directly above it.
struct Block {
    bytes: Range<usize>,
    under_heading: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum LineKind {
    Blank,
    Heading,
    Text,
}

pub(crate) fn text_hash(text: &[u8]) -> TextHash {
    let digest: [u8; 32] = Sha256::digest(text).into();
    let mut hash = TextHash::default();
    hash.copy_from_slice(&digest[..TEXT_HASH_BYTES]);
    hash
}

/// Byte ranges of `content`'s chunks under `cfg`, ordered by start and never cutting
/// inside a character. Whitespace-only input has none.
pub(crate) fn chunk_ranges(content: &str, cfg: ChunkingConfig) -> Vec<Range<usize>> {
    match cfg.chunker {
        Chunker::Windows => window_ranges(content, cfg).collect(),
        Chunker::Paragraphs => paragraph_ranges(content, cfg),
    }
}

/// Byte ranges of `content`'s windows of `chunk_chars` characters, advancing by
/// `chunk_chars - overlap_chars` and dropping whitespace-only windows. Short inputs
/// yield the whole input as one range.
fn window_ranges(content: &str, cfg: ChunkingConfig) -> impl Iterator<Item = Range<usize>> + '_ {
    let step = cfg.chunk_chars - cfg.overlap_chars;
    let mut next_start = Some(0);
    std::iter::from_fn(move || {
        let start = next_start?;
        let rest = &content[start..];
        let window_len = char_offset(rest, cfg.chunk_chars);
        next_start = window_len
            .and_then(|_| char_offset(rest, step))
            .map(|advance| start + advance);
        Some(start..start + window_len.unwrap_or(rest.len()))
    })
    .filter(|bytes| !content[bytes.clone()].trim().is_empty())
}

/// Byte ranges that each run from the start of one block of `content` to the end of
/// the same or a later one, as many blocks as fit. A block under a heading always
/// starts a range, and a block that does not fit alone is cut into windows.
fn paragraph_ranges(content: &str, cfg: ChunkingConfig) -> Vec<Range<usize>> {
    let mut chunks = Vec::new();
    let mut packing: Option<Range<usize>> = None;
    for block in blocks(content) {
        let extended = packing
            .as_ref()
            .filter(|_| !block.under_heading)
            .map(|chunk| chunk.start..block.bytes.end)
            .filter(|extended| cfg.fits(&content[extended.clone()]));
        if extended.is_some() {
            packing = extended;
            continue;
        }
        chunks.extend(packing.take());
        let Range { start, end } = block.bytes;
        if cfg.fits(&content[start..end]) {
            packing = Some(start..end);
        } else {
            let windows = window_ranges(&content[start..end], cfg);
            chunks.extend(windows.map(|window| start + window.start..start + window.end));
        }
    }
    chunks.extend(packing);
    chunks
}

/// `content`'s paragraphs in order, each a run of non-blank lines that starts at any
/// headings directly above it and ends before its last line's trailing whitespace.
fn blocks(content: &str) -> Vec<Block> {
    let mut blocks = Vec::new();
    let mut open: Option<Block> = None;
    let mut open_fence = None;
    let mut previous = LineKind::Blank;
    let mut next_line_start = 0;
    for line in content.split_inclusive('\n') {
        let start = next_line_start;
        next_line_start += line.len();
        let line = line.trim_end();
        let kind = line_kind(line, &mut open_fence);
        if previous == LineKind::Text && kind != LineKind::Text {
            blocks.extend(open.take());
        }
        if kind != LineKind::Blank {
            let block = open.get_or_insert(Block {
                bytes: start..start,
                under_heading: kind == LineKind::Heading,
            });
            block.bytes.end = start + line.len();
        }
        previous = kind;
    }
    blocks.extend(open);
    blocks
}

/// What `line`, already without trailing whitespace, is to a paragraph. `open_fence`
/// is the fence of the code block the line sits in, where nothing is a heading, and
/// is updated when the line opens or closes one.
fn line_kind(line: &str, open_fence: &mut Option<&'static str>) -> LineKind {
    let fence = CODE_FENCES
        .into_iter()
        .find(|fence| line.trim_start().starts_with(fence));
    match (*open_fence, fence) {
        _ if line.is_empty() => LineKind::Blank,
        (None, Some(_)) => {
            *open_fence = fence;
            LineKind::Text
        }
        (Some(open), Some(closing)) if open == closing => {
            *open_fence = None;
            LineKind::Text
        }
        (None, None) if is_heading(line) => LineKind::Heading,
        _ => LineKind::Text,
    }
}

/// Whether `line` is a Markdown ATX heading: up to three spaces, one to six `#`,
/// then a space, a tab or the end of the line.
fn is_heading(line: &str) -> bool {
    let marker = line.trim_start_matches(' ');
    let title = marker.trim_start_matches('#');
    line.len() - marker.len() <= 3
        && (1..=6).contains(&(marker.len() - title.len()))
        && (title.is_empty() || title.starts_with([' ', '\t']))
}

/// The byte offset of `text`'s character number `chars`; `None` when it has no more
/// than `chars` characters.
fn char_offset(text: &str, chars: usize) -> Option<usize> {
    text.char_indices().nth(chars).map(|(offset, _)| offset)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(chunk_chars: usize, overlap: usize) -> ChunkingConfig {
        ChunkingConfig::new(Chunker::Windows, chunk_chars, overlap)
    }

    fn paragraphs(chunk_chars: usize, overlap: usize) -> ChunkingConfig {
        ChunkingConfig::new(Chunker::Paragraphs, chunk_chars, overlap)
    }

    fn alphabet(len: usize) -> String {
        (b'a'..=b'z').cycle().take(len).map(char::from).collect()
    }

    fn chunks_of<'a>(content: &'a str, cfg: &ChunkingConfig) -> Vec<&'a str> {
        chunk_ranges(content, *cfg)
            .into_iter()
            .map(|bytes| &content[bytes])
            .collect()
    }

    fn char_window(s: &str, start: usize, end: usize) -> String {
        s.chars().skip(start).take(end - start).collect()
    }

    #[test]
    fn inputs_that_fit_come_back_whole_or_empty() {
        let cases = [
            ("whitespace only", "   \n\t  ", cfg(64, 8), vec![]),
            ("short", "hello world", cfg(64, 8), vec!["hello world"]),
        ];
        for (label, input, cfg, expected) in cases {
            assert_eq!(chunks_of(input, &cfg), expected, "{label}");
        }
    }

    #[test]
    fn long_inputs_slide_an_overlapping_window() {
        let mixed_width: String = "a\u{e9}\u{4e16}".repeat(10);
        let cases = [
            (
                "short tail kept",
                alphabet(110),
                cfg(40, 10),
                vec![(0, 40), (30, 70), (60, 100), (90, 110)],
            ),
            (
                "multi-byte chars never split",
                mixed_width,
                cfg(10, 2),
                vec![(0, 10), (8, 18), (16, 26), (24, 30)],
            ),
        ];
        for (label, input, cfg, windows) in cases {
            let expected: Vec<String> = windows
                .into_iter()
                .map(|(start, end)| char_window(&input, start, end))
                .collect();
            assert_eq!(chunks_of(&input, &cfg), expected, "{label}");
        }
    }

    #[test]
    fn paragraphs_pack_whole_and_a_heading_opens_a_chunk() {
        let content = "\
intro one

intro two\r
still two

intro three does not fit with them
# Title

## Setup
install it

```sh
# not a heading
make
```

trailing text
#hashtag is no heading

# Ending";
        assert_eq!(
            chunks_of(content, &paragraphs(40, 8)),
            [
                "intro one\n\nintro two\r\nstill two",
                "intro three does not fit with them",
                "# Title\n\n## Setup\ninstall it",
                "```sh\n# not a heading\nmake\n```",
                "trailing text\n#hashtag is no heading",
                "# Ending",
            ]
        );
    }

    #[test]
    fn a_paragraph_over_the_limit_is_windowed_without_splitting_a_character() {
        let long: String = "a\u{e9}\u{4e16}".repeat(8);
        let content = format!("short before\n\n{long}\n\nshort after");
        let windows =
            [(0, 12), (10, 22), (20, 24)].map(|(start, end)| char_window(&long, start, end));
        assert_eq!(
            chunks_of(&content, &paragraphs(12, 2)),
            [
                "short before",
                &windows[0],
                &windows[1],
                &windows[2],
                "short after"
            ]
        );
    }
}
