//! Fire-and-forget message persistence and its bounded drain.

use std::num::NonZeroUsize;
use std::time::{Duration, Instant};

use tokio::sync::{mpsc, oneshot};
use tracing::warn;

use assistd_embed::{EmbedJob, enqueue_embed_job};
use assistd_memory::{BranchId, ChunkId, EmbeddingTarget, PersistedMessage, SessionId, TurnId};

use super::AppState;

const PERSISTENCE_DRAIN_TIMEOUT: Duration = Duration::from_millis(500);
const PERSISTENCE_DRAIN_POLL: Duration = Duration::from_millis(5);

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

    /// Persist one message on a background task, then queue the chunks the
    /// store cut from it for embedding (dropped if the queue is full).
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
            let (_, branch) = session_ctx.current().await;
            let Some(branch) = branch else {
                warn!(
                    target: "assistd::memory",
                    "session is unsaved; dropping message"
                );
                return;
            };
            let append = transcripts
                .append_message_to_branch(branch, turn, msg)
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
