//! Handler for the `Search` request: one ranking of sessions, memories, documents and
//! the files and pages sessions touched.

use assistd_config::Config;
use assistd_ipc::{Event, EventBody, SearchKind};
use assistd_memory::{ChunkSearch, SearchRanking, SessionScope};
use assistd_tools::commands::search::{MAX_HITS, chunk_source, enabling_setting, wire_place};

use super::{AppState, DispatchError, FailedStep};
use crate::bus::EventSink;

impl AppState {
    pub(super) async fn handle_search(
        &self,
        id: String,
        query: String,
        limit: u32,
        kinds: Vec<SearchKind>,
        tx: EventSink,
    ) -> Result<(), DispatchError> {
        let kinds = searched_kinds(&self.config, &kinds)?;
        let top_k = match limit {
            0 => self.config.embedding.top_k.get(),
            limit => limit,
        };
        let embedder = self.memory.embedder.get().context("search failed")?;
        let query_vector = embedder.embed_query(&query).await.context("embed failed")?;
        let search = ChunkSearch {
            query,
            query_vector,
            kinds: kinds.into_iter().map(chunk_source).collect(),
            sessions: SessionScope::All,
            top_k: MAX_HITS.min(top_k) as usize,
            ranking: SearchRanking::VectorAndKeyword,
        };
        let hits = self
            .memory
            .recall
            .search(search)
            .await
            .context("search failed")?;
        for hit in hits {
            let body = EventBody::SearchHit {
                place: wire_place(&hit.locator),
                content: hit.text,
                similarity: hit.similarity,
                held_elsewhere: hit.held_elsewhere as u64,
            };
            let _ = tx.send(Event::new(id.clone(), body)).await;
        }
        Ok(())
    }
}

/// The kinds a search asking for `named` ranks, each once: every kind turned on when
/// none is named. Errors when a named kind, or every kind, is turned off.
fn searched_kinds(config: &Config, named: &[SearchKind]) -> Result<Vec<SearchKind>, DispatchError> {
    let turned_on = |kind: &SearchKind| match kind {
        SearchKind::Sessions | SearchKind::Memories | SearchKind::Accesses => config.memory.enabled,
        SearchKind::Documents => config.documents.enabled,
    };
    if let Some(off) = named.iter().find(|kind| !turned_on(kind)) {
        return Err(DispatchError::SearchKindOff {
            kind: *off,
            setting: enabling_setting(*off),
        });
    }
    let kinds: Vec<SearchKind> = SearchKind::ALL
        .into_iter()
        .filter(|kind| turned_on(kind) && (named.is_empty() || named.contains(kind)))
        .collect();
    if kinds.is_empty() {
        return Err(DispatchError::NothingSearched);
    }
    Ok(kinds)
}
