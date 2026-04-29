//! Char-window chunker used by the persistence path before embedding.
//!
//! Goals (in priority order):
//! 1. **UTF-8 safety.** Splitting must never produce invalid string slices,
//!    even with multi-byte characters (CJK, emoji, etc.).
//! 2. **Bounded chunks.** Each chunk has a stable upper-bound on character
//!    count so embedding requests stay predictably-sized.
//! 3. **Boundary overlap.** Consecutive chunks overlap so semantically
//!    coherent runs (a sentence that straddles a window boundary) are
//!    visible in *both* chunks. The retrieval ranker can then surface the
//!    one with stronger overlap with the query.
//!
//! Trade-offs vs. sentence-aware chunking: this is a deliberately simple
//! window-based split. The repo already has `assistd-voice/src/sentence.rs`
//! but that's TTS-shaped (markdown stripping, code-fence handling) — it
//! drops content semantic search needs and applies transforms search
//! doesn't want. A char-window suffices for v1; if retrieval quality
//! suffers we can revisit.

use serde::{Deserialize, Serialize};

/// Chunking policy shared between the persistence hook and any future
/// backfill pass. `chunk_chars` is the upper bound; `overlap_chars` is
/// strictly less than it (validated at config load time).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChunkingConfig {
    pub chunk_chars: usize,
    pub overlap_chars: usize,
}

/// Split `content` into one or more chunks bounded by `cfg.chunk_chars`.
///
/// - For inputs at or below the limit, returns one chunk (whole input).
/// - For longer inputs, slides a window of `chunk_chars` characters,
///   advancing by `chunk_chars - overlap_chars` each step. The last
///   window is included even when shorter than the limit.
/// - Pure-whitespace chunks are dropped (an empty input or a tail that
///   is fully overlapped by the previous window).
///
/// Iteration is via `char_indices()` so all index math is on UTF-8
/// boundaries; multi-byte chars are never split.
pub fn chunk_message(content: &str, cfg: &ChunkingConfig) -> Vec<String> {
    if content.trim().is_empty() {
        return Vec::new();
    }
    if cfg.chunk_chars == 0 || cfg.overlap_chars >= cfg.chunk_chars {
        // Defensive: validation rejects this at load, but never split-by-zero.
        return vec![content.to_string()];
    }

    // Char-aware window. Collect a slice of (char_idx, byte_offset)
    // pairs so we can grab UTF-8 substrings by byte range without
    // re-walking the string for each window.
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
    let mut out = Vec::new();
    let mut start = 0usize;
    while start < total_chars {
        let end = (start + cfg.chunk_chars).min(total_chars);
        let byte_start = boundaries[start];
        let byte_end = boundaries[end];
        let slice = &content[byte_start..byte_end];
        if !slice.trim().is_empty() {
            out.push(slice.to_string());
        }
        if end == total_chars {
            break;
        }
        start += step;
    }
    out
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

    #[test]
    fn empty_input_returns_empty() {
        assert!(chunk_message("", &cfg(64, 8)).is_empty());
        assert!(chunk_message("   \n\t  ", &cfg(64, 8)).is_empty());
    }

    #[test]
    fn short_input_returns_single_chunk() {
        let r = chunk_message("hello world", &cfg(64, 8));
        assert_eq!(r, vec!["hello world".to_string()]);
    }

    #[test]
    fn exact_boundary_returns_single_chunk() {
        let s = "a".repeat(64);
        let r = chunk_message(&s, &cfg(64, 8));
        assert_eq!(r.len(), 1);
        assert_eq!(r[0], s);
    }

    #[test]
    fn long_input_splits_with_overlap() {
        // 100 chars, chunk=40, overlap=10 → step=30 → starts at 0, 30, 60.
        // Window 2 (start=60) reaches end=100 and the loop breaks, so the
        // last chunk is exactly chunk_chars wide; no separate tail.
        let s: String = (0..100).map(|i| (b'a' + (i % 26) as u8) as char).collect();
        let r = chunk_message(&s, &cfg(40, 10));
        assert_eq!(
            r.len(),
            3,
            "{:?}",
            r.iter().map(|c| c.len()).collect::<Vec<_>>()
        );
        for chunk in &r {
            assert_eq!(chunk.chars().count(), 40);
        }
    }

    #[test]
    fn long_input_with_uneven_tail_keeps_tail_chunk() {
        // 110 chars, chunk=40, overlap=10 → starts at 0, 30, 60, 90.
        // Window 3 covers 90..110 = 20 chars (a real partial tail).
        let s: String = (0..110).map(|i| (b'a' + (i % 26) as u8) as char).collect();
        let r = chunk_message(&s, &cfg(40, 10));
        assert_eq!(r.len(), 4);
        assert_eq!(r[0].chars().count(), 40);
        assert_eq!(r[1].chars().count(), 40);
        assert_eq!(r[2].chars().count(), 40);
        assert_eq!(r[3].chars().count(), 20);
    }

    #[test]
    fn overlap_is_actually_present() {
        // Make a string where each chunk's last `overlap` chars equal the
        // next chunk's first `overlap` chars.
        let s: String = (0..50).map(|i| (b'a' + (i % 26) as u8) as char).collect();
        let r = chunk_message(&s, &cfg(20, 5));
        // step=15, starts at 0, 15, 30, 45.
        let chunk0_tail: String = r[0]
            .chars()
            .rev()
            .take(5)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        let chunk1_head: String = r[1].chars().take(5).collect();
        assert_eq!(
            chunk0_tail, chunk1_head,
            "expected last 5 of chunk0 to equal first 5 of chunk1"
        );
    }

    #[test]
    fn multi_byte_utf8_never_panics_or_splits_chars() {
        // Each char takes 3 bytes. 30 chars = 90 bytes. With chunk=10,
        // overlap=2, step=8: starts at chars 0, 8, 16, 24; lengths 10, 10, 10, 6.
        let s: String = (0..30).map(|_| '世').collect();
        let r = chunk_message(&s, &cfg(10, 2));
        for c in &r {
            // Every chunk must be a valid UTF-8 string (Rust guarantees) and
            // contain only whole `世` characters (no boundary splits).
            assert!(c.chars().all(|ch| ch == '世'));
        }
        assert!(r.len() >= 3);
    }

    #[test]
    fn pure_whitespace_chunk_is_dropped() {
        // 30 chars of spaces. Whole thing becomes a single chunk; trim
        // detects it as whitespace and the function returns empty.
        let s = " ".repeat(30);
        let r = chunk_message(&s, &cfg(10, 2));
        assert!(r.is_empty());
    }

    #[test]
    fn defensive_against_overlap_geq_chunk() {
        // Validation should prevent this, but if it slips through we
        // return one chunk rather than loop forever.
        let s = "a".repeat(50);
        let r = chunk_message(&s, &cfg(10, 10));
        assert_eq!(r, vec![s]);
    }
}
