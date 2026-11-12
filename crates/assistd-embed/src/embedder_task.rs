//! Background worker that embeds the rows the semantic store lists as missing an
//! embedding, a batch at a time, each time it is woken.

use std::future::Future;
use std::sync::Arc;

use assistd_memory::{EmbeddingTarget, MemoryError, SemanticStore};
use assistd_utils::backoff::backoff_delay;
use thiserror::Error;
use tokio::sync::{Notify, watch};
use tokio::task::JoinHandle;

use crate::{BATCH_SIZE, EmbedError, Embedder, embed_rows};

/// Why a pass over the missing rows stopped before reaching the end.
#[derive(Debug, Error)]
enum PassError {
    #[error(transparent)]
    Embed(#[from] EmbedError),
    #[error(transparent)]
    Store(#[from] MemoryError),
}

/// Spawn the worker. It embeds every row missing an embedding at once and again after
/// each `wake`, retrying a pass that stopped early with backoff. It exits on `shutdown`
/// between batches; the database behind `store` must stay open until it exits.
pub fn spawn_embedder_task(
    embedder: Arc<dyn Embedder>,
    store: Arc<dyn SemanticStore>,
    wake: Arc<Notify>,
    mut shutdown: watch::Receiver<bool>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut failed_passes = 0u32;
        loop {
            let retry_in = match embed_missing_rows(&*embedder, &*store, &shutdown).await {
                Ok(()) => {
                    failed_passes = 0;
                    None
                }
                Err(err) => {
                    let delay = backoff_delay(failed_passes);
                    log_stopped_pass(&err, failed_passes);
                    failed_passes = failed_passes.saturating_add(1);
                    Some(delay)
                }
            };
            tokio::select! {
                biased;
                _ = shutdown.wait_for(|stop| *stop) => break,
                () = wake.notified() => {}
                () = tokio::time::sleep(retry_in.unwrap_or_default()), if retry_in.is_some() => {}
            }
        }
        tracing::debug!(target: "assistd::embed", "embed worker exiting");
    })
}

fn log_stopped_pass(err: &PassError, earlier_failures: u32) {
    if earlier_failures == 0 {
        tracing::warn!(
            target: "assistd::embed",
            error = %err,
            "embedding pass stopped early; rows stay unindexed until a retry succeeds"
        );
    } else {
        tracing::debug!(
            target: "assistd::embed",
            error = %err,
            earlier_failures,
            "embedding pass stopped early again"
        );
    }
}

/// Embed and store every chunk, then every memory, missing an embedding.
async fn embed_missing_rows(
    embedder: &dyn Embedder,
    store: &dyn SemanticStore,
    shutdown: &watch::Receiver<bool>,
) -> Result<(), PassError> {
    embed_missing_of_kind(
        embedder,
        store,
        shutdown,
        |after, limit| store.chunks_missing_embedding(after, limit),
        EmbeddingTarget::Chunk,
    )
    .await?;
    embed_missing_of_kind(
        embedder,
        store,
        shutdown,
        |after, limit| store.memories_missing_embedding(after, limit),
        EmbeddingTarget::Memory,
    )
    .await
}

/// Page through `list` in id order, embedding and storing each page, until it runs
/// out, `shutdown` flips, or a page has rows left for a later attempt.
async fn embed_missing_of_kind<Id, L, LFut>(
    embedder: &dyn Embedder,
    store: &dyn SemanticStore,
    shutdown: &watch::Receiver<bool>,
    list: L,
    to_target: fn(Id) -> EmbeddingTarget,
) -> Result<(), PassError>
where
    Id: Copy,
    L: Fn(Option<Id>, usize) -> LFut,
    LFut: Future<Output = Result<Vec<(Id, String)>, MemoryError>>,
{
    let mut after = None;
    while !*shutdown.borrow() {
        let page = list(after, BATCH_SIZE).await?;
        let Some(&(last_id, _)) = page.last() else {
            break;
        };
        after = Some(last_id);
        let rows = page
            .into_iter()
            .map(|(id, text)| (to_target(id), text))
            .collect();
        let embedded = embed_rows(embedder, rows).await;
        store.store_embeddings(embedded.embeddings).await?;
        if let Some(err) = embedded.deferred {
            return Err(err.into());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
