//! Fire-and-forget message persistence and its bounded drain.

use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::{mpsc, oneshot};
use tracing::warn;

use assistd_embed::{EmbedJob, enqueue_embed_job};
use assistd_memory::{
    BranchId, ChunkingConfig, MessageId, MessageKind, PersistedMessage, SessionId, SqliteHandle,
    TurnId, chunk_ranges,
};

use super::AppState;

const PERSISTENCE_DRAIN_TIMEOUT: Duration = Duration::from_millis(500);
const PERSISTENCE_DRAIN_POLL: Duration = Duration::from_millis(5);

/// Id [`assistd_memory::NoTranscriptStore`] returns for a message it did not store.
const UNSTORED_MESSAGE_ID: MessageId = MessageId(0);

impl AppState {
    /// The active session and branch, inserting the session and its `main`
    /// branch first when the session is still unsaved.
    pub(super) async fn saved_session(&self) -> assistd_memory::Result<(Arc<SessionId>, BranchId)> {
        let (session, branch) = self.runtime.session_ctx.current().await;
        if let Some(branch) = branch {
            return Ok((session, branch));
        }
        let branch = self
            .memory
            .transcripts
            .begin_session_with_main_branch(&session, std::process::id())
            .await?;
        self.runtime
            .session_ctx
            .replace(session.clone(), Some(branch))
            .await;
        Ok((session, branch))
    }

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
        let transcripts = self.memory.transcripts.clone();
        let session_ctx = self.runtime.session_ctx.clone();
        let embed_tx = self.memory.embed_tx.clone();
        let chunk_target = self.chunk_target(&msg);
        let (previous, landed_tx) = self.join_persist_chain();
        self.runtime.persistence_tracker.spawn(async move {
            if let Some(previous) = previous {
                let _ = previous.await;
            }
            let (session, branch) = session_ctx.current().await;
            let Some(branch) = branch else {
                warn!(
                    target: "assistd::memory",
                    "session is unsaved; dropping message"
                );
                return;
            };
            let append = transcripts
                .append_message_to_branch(&session, branch, turn, msg)
                .await;
            let _ = landed_tx.send(());
            let message_id = match append {
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
                && message_id != UNSTORED_MESSAGE_ID
            {
                store_and_queue_chunks(&chunks, &embed_tx, message_id, &content).await;
            }
        });
    }

    /// Record `summary` as the active branch's compaction summary on a
    /// background task, ordered after every message persisted before it.
    pub(super) fn persist_summary_fire_and_forget(
        &self,
        summary: String,
        kept_turns: NonZeroUsize,
    ) {
        let transcripts = self.memory.transcripts.clone();
        let session_ctx = self.runtime.session_ctx.clone();
        let (previous, landed_tx) = self.join_persist_chain();
        self.runtime.persistence_tracker.spawn(async move {
            if let Some(previous) = previous {
                let _ = previous.await;
            }
            let (_, branch) = session_ctx.current().await;
            let Some(branch) = branch else {
                warn!(
                    target: "assistd::memory",
                    "session is unsaved; dropping compaction summary"
                );
                return;
            };
            let record = transcripts
                .record_branch_summary(branch, summary, kept_turns)
                .await;
            let _ = landed_tx.send(());
            if let Err(e) = record {
                warn!(
                    target: "assistd::memory",
                    error = %e,
                    "failed to persist compaction summary (continuing)"
                );
            }
        });
    }

    /// Take the next place in `runtime.persist_chain`: the predecessor's
    /// completion signal to await, and this write's own to fire once it lands.
    fn join_persist_chain(&self) -> (Option<oneshot::Receiver<()>>, oneshot::Sender<()>) {
        let (landed_tx, landed_rx) = oneshot::channel();
        let previous = self.runtime.persist_chain.lock().replace(landed_rx);
        (previous, landed_tx)
    }

    /// The chunk store and text to chunk for `msg`, or `None` when it
    /// should not be embedded.
    fn chunk_target(&self, msg: &PersistedMessage) -> Option<(Arc<SqliteHandle>, String)> {
        let embeddable = self.memory.embedding_cfg.enabled
            && matches!(msg.kind, MessageKind::User | MessageKind::Assistant { .. })
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
    message_id: MessageId,
    content: &str,
) {
    for bytes in chunk_ranges(content, &ChunkingConfig::default()) {
        let text = content[bytes.clone()].to_string();
        match chunks.store_chunk(message_id, bytes.clone()).await {
            Ok(chunk_id) => enqueue_embed_job(embed_tx, EmbedJob::Chunk { chunk_id, text }),
            Err(e) => warn!(
                target: "assistd::memory",
                %message_id,
                byte_start = bytes.start,
                error = %e,
                "failed to persist chunk (continuing)"
            ),
        }
    }
}

#[cfg(test)]
mod tests;
