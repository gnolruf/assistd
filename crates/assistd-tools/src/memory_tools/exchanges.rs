//! Session-scoped `reminisce` hits laid out as runs of consecutive chunks, so
//! hits that share a neighbour read as one exchange.

use std::iter;

use assistd_memory::{AdjacentChunks, ChunkId, ChunkLocator, ContextChunk, EmbeddingHit};

use super::{locator_tag, one_line};

/// One printed chunk; `similarity` is set when the chunk was itself a hit.
#[derive(Debug, Clone)]
struct ExchangeLine {
    chunk_id: ChunkId,
    locator: ChunkLocator,
    text: String,
    similarity: Option<f32>,
}

impl ExchangeLine {
    fn hit(hit: &EmbeddingHit) -> Self {
        Self {
            chunk_id: hit.chunk_id,
            locator: hit.locator.clone(),
            text: hit.text.clone(),
            similarity: Some(hit.similarity),
        }
    }

    fn context(chunk: ContextChunk) -> Self {
        Self {
            chunk_id: chunk.chunk_id,
            locator: chunk.locator,
            text: chunk.text,
            similarity: None,
        }
    }

    fn render(&self) -> String {
        let score = self
            .similarity
            .map(|similarity| format!(" sim={:.0}%", similarity * 100.0))
            .unwrap_or_default();
        format!(
            "{}{score}] {}",
            locator_tag(&self.locator),
            one_line(&self.text)
        )
    }
}

/// Runs of consecutive chunks, ordered by when each run's first hit was added.
/// Runs never share a chunk, so no chunk prints twice.
#[derive(Debug, Clone, Default)]
pub(super) struct Exchanges {
    runs: Vec<Vec<ExchangeLine>>,
}

impl Exchanges {
    /// Add `hit` between its neighbours, folding every run it shares a chunk with
    /// into one run at the position of the earliest of them.
    pub(super) fn add(&mut self, hit: &EmbeddingHit, adjacent: AdjacentChunks) {
        let mut run: Vec<ExchangeLine> = adjacent
            .before
            .map(ExchangeLine::context)
            .into_iter()
            .chain(iter::once(ExchangeLine::hit(hit)))
            .chain(adjacent.after.map(ExchangeLine::context))
            .collect();
        let mut slot = None;
        let mut index = 0;
        while index < self.runs.len() {
            if shares_chunk(&self.runs[index], &run) {
                run = merge_runs(self.runs.remove(index), run);
                slot.get_or_insert(index);
            } else {
                index += 1;
            }
        }
        self.runs.insert(slot.unwrap_or(self.runs.len()), run);
    }

    /// Every run's lines, each run led by a blank line.
    pub(super) fn lines(&self) -> Vec<String> {
        self.runs
            .iter()
            .flat_map(|run| iter::once(String::new()).chain(run.iter().map(ExchangeLine::render)))
            .collect()
    }
}

fn shares_chunk(run: &[ExchangeLine], other: &[ExchangeLine]) -> bool {
    run.iter()
        .any(|line| other.iter().any(|o| o.chunk_id == line.chunk_id))
}

/// Two overlapping runs as one, in source order: the run holding the other's first
/// chunk leads, and the other's chunks it lacks follow. A chunk in both keeps its score.
fn merge_runs(first: Vec<ExchangeLine>, second: Vec<ExchangeLine>) -> Vec<ExchangeLine> {
    let first_leads = second
        .first()
        .is_some_and(|start| first.iter().any(|line| line.chunk_id == start.chunk_id));
    let (mut merged, tail) = if first_leads {
        (first, second)
    } else {
        (second, first)
    };
    for line in tail {
        match merged
            .iter_mut()
            .find(|kept| kept.chunk_id == line.chunk_id)
        {
            Some(kept) => kept.similarity = kept.similarity.or(line.similarity),
            None => merged.push(line),
        }
    }
    merged
}
