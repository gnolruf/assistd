//! Fire-and-forget message persistence and its drain.

use std::num::NonZeroUsize;

use tokio::sync::{mpsc, oneshot};
use tracing::warn;

use assistd_embed::{EmbedJob, enqueue_embed_job};
use assistd_memory::{BranchId, ChunkId, EmbeddingTarget, PersistedMessage, SessionId, TurnId};

use super::AppState;

/// Where a turn's messages are written: the branch current when the turn
/// opened, and the turn's row unless creating it failed.
#[derive(Debug, Clone, Copy)]
pub(super) struct PersistTarget {
    pub(super) branch: BranchId,
    pub(super) turn: Option<TurnId>,
}

impl AppState {
    /// The active session and branch, inserting the session and its `main`
    /// branch first when the session is still unsaved.
    pub(super) async fn saved_session(&self) -> assistd_memory::Result<(SessionId, BranchId)> {
        let (session, branch) = self.runtime.session_ctx.current().await;
        if let Some(branch) = branch {
            return Ok((session, branch));
        }
        let branch = self
            .memory
            .transcripts
            .begin_session_with_main_branch(session, std::process::id())
            .await?;
        self.runtime
            .session_ctx
            .replace(session, Some(branch))
            .await;
        Ok((session, branch))
    }

    /// Persist one message to `target` on a background task, then queue the
    /// chunks the store cut from it for embedding (dropped if the queue is full).
    ///
    /// Writes land in call order: each task awaits its predecessor's
    /// completion signal from `runtime.persist_chain` before appending,
    /// because the store assigns `seq` in arrival order.
    pub(super) fn persist_message_fire_and_forget(
        &self,
        target: PersistTarget,
        msg: PersistedMessage,
    ) {
        let transcripts = self.memory.transcripts.clone();
        let embed_tx = self
            .memory
            .embedding_cfg
            .enabled
            .then(|| self.memory.embed_tx.clone());
        let (previous, landed_tx) = self.join_persist_chain();
        self.runtime.persistence_tracker.spawn(async move {
            if let Some(previous) = previous {
                let _ = previous.await;
            }
            let append = transcripts
                .append_message_to_branch(target.branch, target.turn, msg)
                .await;
            let _ = landed_tx.send(());
            let appended = match append {
                Ok(appended) => appended,
                Err(e) => {
                    warn!(
                        target: "assistd::memory",
                        error = %e,
                        "failed to persist message (continuing)"
                    );
                    return;
                }
            };
            if let Some(embed_tx) = embed_tx {
                queue_chunk_embeddings(&embed_tx, appended.chunks);
            }
        });
    }

    /// Record `summary` as `branch`'s compaction summary on a background
    /// task, ordered after every message persisted before it.
    pub(super) fn persist_summary_fire_and_forget(
        &self,
        branch: BranchId,
        summary: String,
        kept_turns: NonZeroUsize,
    ) {
        let transcripts = self.memory.transcripts.clone();
        let (previous, landed_tx) = self.join_persist_chain();
        self.runtime.persistence_tracker.spawn(async move {
            if let Some(previous) = previous {
                let _ = previous.await;
            }
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

    /// Wait for every queued message and summary write to land. Hold
    /// `agent_turn_lock` across the call so no new writes queue during the wait.
    pub(super) async fn drain_persistence_inflight(&self) {
        let (previous, landed_tx) = self.join_persist_chain();
        if let Some(previous) = previous {
            let _ = previous.await;
        }
        let _ = landed_tx.send(());
    }
}

fn queue_chunk_embeddings(embed_tx: &mpsc::Sender<EmbedJob>, chunks: Vec<(ChunkId, String)>) {
    for (chunk_id, text) in chunks {
        enqueue_embed_job(
            embed_tx,
            EmbedJob {
                target: EmbeddingTarget::Chunk(chunk_id),
                text,
            },
        );
    }
}

#[cfg(test)]
mod tests;
