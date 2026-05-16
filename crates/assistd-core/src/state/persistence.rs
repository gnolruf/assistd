//! Fire-and-forget persistence pipeline and in-flight drain helper.
//!
//! `persist_message_fire_and_forget` spawns the writer task; every
//! handler that mutates branches (`/fork`, `/switch`, `/undo`,
//! `/resume`, `/new`) calls `drain_persistence_inflight` while holding
//! the agent turn lock so previously-spawned writes have a chance to
//! land before the next branch op.

use super::AppState;
use assistd_embed::EmbedJob;
use assistd_memory::{ChunkingConfig, PersistedMessage, PersistedRole, TurnId, chunk_message};

impl AppState {
    /// Fire-and-forget persist of one message. Spawns a task that:
    /// 1. Writes the row via the conversation store (returns row id).
    /// 2. If the row is a User/Assistant text message and embedding is
    ///    enabled, splits the content into chunks, persists each chunk
    ///    (returns chunk id), and `try_send`s an `EmbedJob::Chunk` for
    ///    each so the embedder task can index it.
    ///
    /// The whole pipeline is on a `tokio::spawn`'d task so the dispatch
    /// loop never waits on disk or the embed queue. `try_send` (not
    /// `send`) so a wedged embedder doesn't backpressure persistence;
    /// dropped jobs just leave the chunk row unindexed for the next
    /// backfill pass.
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
        let chunking_cfg = ChunkingConfig {
            chunk_chars: self.memory.embedding_cfg.chunk_chars,
            overlap_chars: self.memory.embedding_cfg.chunk_overlap_chars,
        };
        let should_embed = embedding_enabled
            && chunks_handle.is_some()
            && matches!(msg.role, PersistedRole::User | PersistedRole::Assistant)
            && !msg.content.is_empty()
            && msg.tool_calls.is_none();
        let content_for_chunks = if should_embed {
            Some(msg.content.clone())
        } else {
            None
        };
        self.runtime.persistence_tracker.spawn(async move {
            let (session, branch) = conversation_ctx.current().await;
            let row_id = match conv
                .append_message_to_branch(&session, branch, turn, msg)
                .await
            {
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
            // NoConversationStore returns 0: nothing to chunk.
            let Some(content) = content_for_chunks else {
                return;
            };
            let Some(chunks_handle) = chunks_handle else {
                return;
            };
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

    /// Block until every previously-spawned `persist_message_fire_and_forget`
    /// task has landed. Held inside `agent_turn_lock` so no new tasks
    /// can spawn during the wait.
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
