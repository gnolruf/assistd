//! Fire-and-forget message persistence and its bounded drain.

use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::{mpsc, oneshot};
use tracing::{debug, warn};

use assistd_embed::EmbedJob;
use assistd_memory::{
    ChunkingConfig, PersistedMessage, PersistedRole, SqliteHandle, TurnId, chunk_message,
};

use super::AppState;

const PERSISTENCE_DRAIN_TIMEOUT: Duration = Duration::from_millis(500);
const PERSISTENCE_DRAIN_POLL: Duration = Duration::from_millis(5);

/// Row id [`assistd_memory::NoConversationStore`] returns for a message it
/// did not store.
const UNSTORED_ROW_ID: i64 = 0;

impl AppState {
    /// Persist one message on a background task, then chunk and queue
    /// user or assistant text for embedding (dropped if the queue is full).
    ///
    /// Writes land in call order: each task awaits its predecessor's
    /// completion signal from `runtime.persist_chain` before appending,
    /// because the store assigns `seq` in arrival order.
    pub(super) fn persist_message_fire_and_forget(
        &self,
        turn: Option<TurnId>,
        msg: PersistedMessage,
    ) {
        let conversations = self.memory.conversations.clone();
        let conversation_ctx = self.runtime.conversation_ctx.clone();
        let embed_tx = self.memory.embed_tx.clone();
        let chunk_target = self.chunk_target(&msg);
        let (landed_tx, landed_rx) = oneshot::channel();
        let previous = self.runtime.persist_chain.lock().replace(landed_rx);
        self.runtime.persistence_tracker.spawn(async move {
            if let Some(previous) = previous {
                let _ = previous.await;
            }
            let (session, branch) = conversation_ctx.current().await;
            let append = conversations
                .append_message_to_branch(&session, branch, turn, msg)
                .await;
            let _ = landed_tx.send(());
            let row_id = match append {
                Ok(id) => id,
                Err(e) => {
                    warn!(
                        target: "assistd::memory",
                        error = %e,
                        "failed to persist message (continuing)"
                    );
                    return;
                }
            };
            if let Some((chunks, content)) = chunk_target
                && row_id != UNSTORED_ROW_ID
            {
                store_and_queue_chunks(&chunks, &embed_tx, row_id, &content).await;
            }
        });
    }

    /// The chunk store and text to chunk for `msg`, or `None` when it
    /// should not be embedded.
    fn chunk_target(&self, msg: &PersistedMessage) -> Option<(Arc<SqliteHandle>, String)> {
        let embeddable = self.memory.embedding_cfg.enabled
            && matches!(msg.role, PersistedRole::User | PersistedRole::Assistant)
            && !msg.content.is_empty();
        let chunks = self.memory.chunks.clone().filter(|_| embeddable)?;
        Some((chunks, msg.content.clone()))
    }

    /// Wait, up to 500ms, for every queued persistence task to land.
    /// Hold `agent_turn_lock` across the call so no new tasks spawn
    /// during the wait.
    pub(super) async fn drain_persistence_inflight(&self) {
        let deadline = Instant::now() + PERSISTENCE_DRAIN_TIMEOUT;
        while !self.runtime.persistence_tracker.is_empty() && Instant::now() < deadline {
            tokio::time::sleep(PERSISTENCE_DRAIN_POLL).await;
        }
        if !self.runtime.persistence_tracker.is_empty() {
            warn!(
                target: "assistd::state",
                "persistence drain timed out; in-flight writes may race branch op"
            );
        }
    }
}

async fn store_and_queue_chunks(
    chunks: &SqliteHandle,
    embed_tx: &mpsc::Sender<EmbedJob>,
    row_id: i64,
    content: &str,
) {
    for (idx, chunk) in chunk_message(content, &ChunkingConfig::default())
        .into_iter()
        .enumerate()
    {
        match chunks
            .store_chunk(row_id, idx as i64, chunk.clone(), None)
            .await
        {
            Ok(chunk_id) => {
                if embed_tx
                    .try_send(EmbedJob::Chunk {
                        chunk_id,
                        text: chunk,
                    })
                    .is_err()
                {
                    debug!(
                        target: "assistd::embed",
                        chunk_id,
                        "embed queue full or closed; dropping job"
                    );
                }
            }
            Err(e) => warn!(
                target: "assistd::memory",
                conversation_id = row_id,
                chunk_index = idx,
                error = %e,
                "failed to persist chunk (continuing)"
            ),
        }
    }
}

#[cfg(test)]
mod tests;
