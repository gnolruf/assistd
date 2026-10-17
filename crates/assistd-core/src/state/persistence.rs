//! Fire-and-forget message persistence and its drain.

use std::num::NonZeroUsize;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use tokio::sync::{mpsc, oneshot};
use tracing::{debug, warn};

use assistd_embed::{EmbedJob, enqueue_embed_job};
use assistd_memory::{BranchId, ChunkId, EmbeddingTarget, PersistedMessage, SessionId, TurnId};

use super::AppState;

/// Where a turn's messages are written: the branch current when the turn
/// opened, and the turn's row once the write opening it has landed.
#[derive(Debug, Clone)]
pub(super) struct PersistTarget {
    pub(super) branch: BranchId,
    turn: Arc<OnceLock<TurnId>>,
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

    /// Open a turn on `branch` with `prompt` as its first message on a
    /// background task, returning the target the turn's later messages go to.
    pub(super) fn open_turn_fire_and_forget(
        &self,
        branch: BranchId,
        prompt: PersistedMessage,
    ) -> PersistTarget {
        let target = PersistTarget {
            branch,
            turn: Arc::default(),
        };
        let transcripts = self.memory.transcripts.clone();
        let embed_tx = self.embed_tx();
        let turn = Arc::clone(&target.turn);
        self.spawn_in_persist_chain(async move {
            match transcripts.begin_turn(branch, prompt).await {
                Ok(opened) => {
                    let _ = turn.set(opened.turn);
                    queue_chunk_embeddings(embed_tx.as_ref(), opened.message.chunks);
                }
                Err(e) => warn!(
                    target: "assistd::memory",
                    error = %e,
                    "failed to open the turn; it will not be persisted (continuing)"
                ),
            }
        });
        target
    }

    /// Persist one message to `target` on a background task, then queue the
    /// chunks the store cut from it for embedding (dropped if the queue is full).
    /// Dropped when the turn's opening write failed.
    pub(super) fn persist_message_fire_and_forget(
        &self,
        target: PersistTarget,
        msg: PersistedMessage,
    ) {
        let transcripts = self.memory.transcripts.clone();
        let embed_tx = self.embed_tx();
        self.spawn_in_persist_chain(async move {
            let Some(&turn) = target.turn.get() else {
                debug!(
                    target: "assistd::memory",
                    "turn was never opened; dropping message"
                );
                return;
            };
            match transcripts
                .append_message_to_branch(target.branch, turn, msg)
                .await
            {
                Ok(appended) => queue_chunk_embeddings(embed_tx.as_ref(), appended.chunks),
                Err(e) => warn!(
                    target: "assistd::memory",
                    error = %e,
                    "failed to persist message (continuing)"
                ),
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
        elapsed: Duration,
    ) {
        let transcripts = self.memory.transcripts.clone();
        let duration_ms = u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX);
        self.spawn_in_persist_chain(async move {
            if let Err(e) = transcripts
                .record_branch_summary(branch, summary, kept_turns, duration_ms)
                .await
            {
                warn!(
                    target: "assistd::memory",
                    error = %e,
                    "failed to persist compaction summary (continuing)"
                );
            }
        });
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

    /// Run `write` on a background task once every write queued before it has
    /// finished, since the store assigns `seq` in arrival order.
    fn spawn_in_persist_chain(&self, write: impl Future<Output = ()> + Send + 'static) {
        let (previous, landed_tx) = self.join_persist_chain();
        self.runtime.persistence_tracker.spawn(async move {
            if let Some(previous) = previous {
                let _ = previous.await;
            }
            write.await;
            let _ = landed_tx.send(());
        });
    }

    /// Take the next place in `runtime.persist_chain`: the predecessor's
    /// completion signal to await, and this write's own to fire once it lands.
    fn join_persist_chain(&self) -> (Option<oneshot::Receiver<()>>, oneshot::Sender<()>) {
        let (landed_tx, landed_rx) = oneshot::channel();
        let previous = self.runtime.persist_chain.lock().replace(landed_rx);
        (previous, landed_tx)
    }

    /// The embedding queue, when embedding is enabled.
    fn embed_tx(&self) -> Option<mpsc::Sender<EmbedJob>> {
        self.memory
            .embedding_cfg
            .enabled
            .then(|| self.memory.embed_tx.clone())
    }
}

fn queue_chunk_embeddings(
    embed_tx: Option<&mpsc::Sender<EmbedJob>>,
    chunks: Vec<(ChunkId, String)>,
) {
    let Some(embed_tx) = embed_tx else {
        return;
    };
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
