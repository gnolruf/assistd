//! Overlapping character-window chunker applied to messages before embedding.

use serde::{Deserialize, Serialize};

/// Chunking policy; `overlap_chars` must be less than `chunk_chars`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChunkingConfig {
    /// Maximum number of Unicode characters per chunk.
    pub chunk_chars: usize,
    /// Number of characters shared between consecutive chunks.
    pub overlap_chars: usize,
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            chunk_chars: 512,
            overlap_chars: 64,
        }
    }
}

/// Split `content` into windows of `chunk_chars` characters advancing by
/// `chunk_chars - overlap_chars`, dropping whitespace-only windows. Short inputs and
/// invalid configs yield the whole input as one chunk.
pub fn chunk_message(content: &str, cfg: &ChunkingConfig) -> Vec<String> {
    if content.trim().is_empty() {
        return Vec::new();
    }
    if cfg.chunk_chars == 0 || cfg.overlap_chars >= cfg.chunk_chars {
        return vec![content.to_string()];
    }

    let boundaries: Vec<usize> = content
        .char_indices()
        .map(|(byte_idx, _)| byte_idx)
        .chain(std::iter::once(content.len()))
        .collect();
    let total_chars = boundaries.len() - 1;
    if total_chars <= cfg.chunk_chars {
        return vec![content.to_string()];
    }

    let step = cfg.chunk_chars - cfg.overlap_chars;
    let mut chunks = Vec::new();
    let mut start = 0usize;
    while start < total_chars {
        let end = (start + cfg.chunk_chars).min(total_chars);
        let byte_start = boundaries[start];
        let byte_end = boundaries[end];
        let slice = &content[byte_start..byte_end];
        if !slice.trim().is_empty() {
            chunks.push(slice.to_string());
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
        (0..len).map(|i| (b'a' + (i % 26) as u8) as char).collect()
    }

    fn char_window(s: &str, start: usize, end: usize) -> String {
        s.chars().skip(start).take(end - start).collect()
    }

    #[test]
    fn inputs_that_fit_or_cannot_be_split_come_back_whole_or_empty() {
        let a64 = "a".repeat(64);
        let a50 = "a".repeat(50);
        let spaces = " ".repeat(30);
        let cases = [
            ("empty", "", cfg(64, 8), vec![]),
            ("whitespace only", "   \n\t  ", cfg(64, 8), vec![]),
            ("long whitespace only", spaces.as_str(), cfg(10, 2), vec![]),
            ("short", "hello world", cfg(64, 8), vec!["hello world"]),
            (
                "exactly at the limit",
                a64.as_str(),
                cfg(64, 8),
                vec![a64.as_str()],
            ),
            (
                "overlap >= chunk",
                a50.as_str(),
                cfg(10, 10),
                vec![a50.as_str()],
            ),
        ];
        for (label, input, cfg, expected) in cases {
            assert_eq!(chunk_message(input, &cfg), expected, "{label}");
        }
    }

    #[test]
    fn long_inputs_slide_an_overlapping_window() {
        let mixed_width: String = "a\u{e9}\u{4e16}".repeat(10);
        let cases = [
            (
                "even split",
                alphabet(100),
                cfg(40, 10),
                vec![(0, 40), (30, 70), (60, 100)],
            ),
            (
                "short tail kept",
                alphabet(110),
                cfg(40, 10),
                vec![(0, 40), (30, 70), (60, 100), (90, 110)],
            ),
            (
                "small overlap",
                alphabet(50),
                cfg(20, 5),
                vec![(0, 20), (15, 35), (30, 50)],
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
            assert_eq!(chunk_message(&input, &cfg), expected, "{label}");
        }
    }

    #[test]
    fn whitespace_only_windows_are_dropped() {
        let input = format!("{}{}{}", "a".repeat(10), " ".repeat(20), "b".repeat(10));
        let chunks = chunk_message(&input, &cfg(10, 2));
        assert_eq!(
            chunks,
            [
                "a".repeat(10),
                format!("aa{}", " ".repeat(8)),
                format!("{}bbbb", " ".repeat(6)),
                "b".repeat(8),
            ]
        );
    }
}
