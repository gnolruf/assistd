//! Overlapping character-window chunker applied to messages before embedding.

use std::ops::Range;

/// Chunking policy; `overlap_chars` must be less than `chunk_chars`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ChunkingConfig {
    /// Maximum number of Unicode characters per chunk.
    pub(crate) chunk_chars: usize,
    /// Number of characters shared between consecutive chunks.
    pub(crate) overlap_chars: usize,
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            chunk_chars: 512,
            overlap_chars: 64,
        }
    }
}

/// Byte ranges of `content`'s windows of `chunk_chars` characters, advancing by
/// `chunk_chars - overlap_chars` and dropping whitespace-only windows. Short inputs
/// and invalid configs yield the whole input as one range.
pub(crate) fn chunk_ranges(content: &str, cfg: &ChunkingConfig) -> Vec<Range<usize>> {
    if content.trim().is_empty() {
        return Vec::new();
    }
    let whole_message = 0..content.len();
    if cfg.chunk_chars == 0 || cfg.overlap_chars >= cfg.chunk_chars {
        return vec![whole_message];
    }

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
        ChunkingConfig {
            chunk_chars,
            overlap_chars: overlap,
        }
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
    fn inputs_that_fit_or_cannot_be_split_come_back_whole_or_empty() {
        let a50 = "a".repeat(50);
        let cases = [
            ("whitespace only", "   \n\t  ", cfg(64, 8), vec![]),
            ("short", "hello world", cfg(64, 8), vec!["hello world"]),
            (
                "overlap >= chunk",
                a50.as_str(),
                cfg(10, 10),
                vec![a50.as_str()],
            ),
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
