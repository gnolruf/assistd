//! Fire-and-forget message persistence and its bounded drain.

use super::AppState;
use assistd_embed::EmbedJob;
use assistd_memory::{ChunkingConfig, PersistedMessage, PersistedRole, TurnId, chunk_message};

impl AppState {
    /// Persist one message on a background task, then chunk and queue it
    /// for embedding when it is user or assistant text. A full embed
    /// queue drops the job; the chunk row stays unindexed until reindex.
    ///
    /// Writes land in call order: each task takes the previous task's
    /// completion signal off `runtime.persist_chain` in a synchronous
    /// swap and awaits it before appending. The store assigns `seq` in
    /// arrival order, so without the chain a tool result could be
    /// sequenced ahead of the call that produced it. Chunking and
    /// embedding run after the signal fires and stay off the chain.
    pub(super) fn persist_message_fire_and_forget(
        &self,
        turn: Option<TurnId>,
        msg: PersistedMessage,
    ) {
        let conv = self.memory.conversations.clone();
        let conversation_ctx = self.runtime.conversation_ctx.clone();
        let chunks_handle = self.memory.chunks.clone();
        let embed_tx = self.memory.embed_tx.clone();
        let embedding_enabled = self.memory.embedding_cfg.enabled;
        let chunking_cfg = ChunkingConfig::default();
        let should_embed = embedding_enabled
            && chunks_handle.is_some()
            && matches!(msg.role, PersistedRole::User | PersistedRole::Assistant)
            && !msg.content.is_empty();
        let content_for_chunks = if should_embed {
            Some(msg.content.clone())
        } else {
            None
        };
        let (landed_tx, landed_rx) = tokio::sync::oneshot::channel();
        let previous = self.runtime.persist_chain.lock().replace(landed_rx);
        self.runtime.persistence_tracker.spawn(async move {
            // An `Err` here means the predecessor's task was dropped
            // without writing; there is nothing left to wait for.
            if let Some(previous) = previous {
                let _ = previous.await;
            }
            let (session, branch) = conversation_ctx.current().await;
            let append = conv
                .append_message_to_branch(&session, branch, turn, msg)
                .await;
            let _ = landed_tx.send(());
            let row_id = match append {
                Ok(id) => id,
                Err(e) => {
                    tracing::warn!(
                        target: "assistd::memory",
                        error = %e,
                        "failed to persist message (continuing)"
                    );
                    return;
                }
            };
            let Some(content) = content_for_chunks else {
                return;
            };
            let Some(chunks_handle) = chunks_handle else {
                return;
            };
            // NoConversationStore returns 0: nothing to chunk.
            if row_id == 0 {
                return;
            }
            for (idx, chunk) in chunk_message(&content, &chunking_cfg)
                .into_iter()
                .enumerate()
            {
                match chunks_handle
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
                            tracing::debug!(
                                target: "assistd::embed",
                                chunk_id,
                                "embed queue full or closed; dropping job"
                            );
                        }
                    }
                    Err(e) => tracing::warn!(
                        target: "assistd::memory",
                        conversation_id = row_id,
                        chunk_index = idx,
                        error = %e,
                        "failed to persist chunk (continuing)"
                    ),
                }
            }
        });
    }

    /// Wait, up to 500ms, for every queued persistence task to land.
    /// Hold `agent_turn_lock` across the call so no new tasks spawn
    /// during the wait.
    pub(super) async fn drain_persistence_inflight(&self) {
        let deadline = std::time::Instant::now() + std::time::Duration::from_millis(500);
        while !self.runtime.persistence_tracker.is_empty() && std::time::Instant::now() < deadline {
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
        if !self.runtime.persistence_tracker.is_empty() {
            tracing::warn!(
                target: "assistd::state",
                "persistence drain timed out; in-flight writes may race branch op"
            );
        }
    }
}

#[cfg(test)]
mod tests;
