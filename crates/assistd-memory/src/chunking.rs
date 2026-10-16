//! Overlapping character-window chunker applied to messages before embedding.

use std::ops::Range;

/// Chunking policy; construction guarantees `overlap_chars < chunk_chars`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ChunkingConfig {
    /// Maximum number of Unicode characters per chunk.
    chunk_chars: usize,
    /// Number of characters shared between consecutive chunks.
    overlap_chars: usize,
}

impl ChunkingConfig {
    /// The policy applied to stored user and assistant messages.
    pub(crate) const MESSAGES: Self = Self::new(512, 64);

    /// Panics, at compile time when used in a const, unless `overlap_chars < chunk_chars`.
    pub(crate) const fn new(chunk_chars: usize, overlap_chars: usize) -> Self {
        assert!(
            overlap_chars < chunk_chars,
            "chunk overlap must be shorter than the chunk"
        );
        Self {
            chunk_chars,
            overlap_chars,
        }
    }
}

/// Byte ranges of `content`'s windows of `chunk_chars` characters, advancing by
/// `chunk_chars - overlap_chars` and dropping whitespace-only windows. Short inputs
/// yield the whole input as one range.
pub(crate) fn chunk_ranges(content: &str, cfg: &ChunkingConfig) -> Vec<Range<usize>> {
    if content.trim().is_empty() {
        return Vec::new();
    }
    let whole_message = 0..content.len();

    let boundaries: Vec<usize> = content
        .char_indices()
        .map(|(byte_idx, _)| byte_idx)
        .chain(std::iter::once(content.len()))
        .collect();
    let total_chars = boundaries.len() - 1;
    if total_chars <= cfg.chunk_chars {
        return vec![whole_message];
    }

    let step = cfg.chunk_chars - cfg.overlap_chars;
    let mut chunks = Vec::new();
    let mut start = 0usize;
    while start < total_chars {
        let end = (start + cfg.chunk_chars).min(total_chars);
        let bytes = boundaries[start]..boundaries[end];
        if !content[bytes.clone()].trim().is_empty() {
            chunks.push(bytes);
        }
        if end == total_chars {
            break;
        }
        start += step;
    }
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(chunk_chars: usize, overlap: usize) -> ChunkingConfig {
        ChunkingConfig::new(chunk_chars, overlap)
    }

    fn alphabet(len: usize) -> String {
        (b'a'..=b'z').cycle().take(len).map(char::from).collect()
    }

    fn chunks_of<'a>(content: &'a str, cfg: &ChunkingConfig) -> Vec<&'a str> {
        chunk_ranges(content, cfg)
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
}
