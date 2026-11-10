//! Queueing transcript writes for the transcript writer, and its drain.

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{Notify, mpsc, oneshot};
use tracing::warn;

use assistd_ipc::Component;
use assistd_llm::CompactionCut;
use assistd_memory::{BranchId, PersistedMessage, PersistedRelayStep, SessionId, TurnStats};
use assistd_utils::time::saturating_millis;

use super::AppState;
use super::seq_ledger::SharedLedger;
use super::transcript_writer::{
    PersistTarget, TranscriptWriter, WRITE_QUEUE_CAPACITY, WriteCommand,
};
use crate::recovery::spawn_supervised;

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
            .begin_session_with_main_branch(session)
            .await?;
        self.runtime
            .session_ctx
            .replace(session, Some(branch))
            .await;
        Ok((session, branch))
    }

    /// Queue the write opening a turn of the user's conversation on `branch`
    /// with `prompt`, returning the target the turn's later messages go to.
    pub(super) async fn open_user_turn(
        &self,
        branch: BranchId,
        prompt: PersistedMessage,
    ) -> PersistTarget {
        let ledger = self.runtime.user_ledger.clone();
        self.open_turn(PersistTarget::new(branch, ledger), prompt)
            .await
    }

    /// [`Self::open_user_turn`] for a task run's turn on its new, empty `branch`.
    pub(super) async fn open_task_run_turn(
        &self,
        branch: BranchId,
        prompt: PersistedMessage,
    ) -> PersistTarget {
        let ledger = SharedLedger::of_new_branch();
        self.open_turn(PersistTarget::new(branch, ledger), prompt)
            .await
    }

    async fn open_turn(&self, target: PersistTarget, prompt: PersistedMessage) -> PersistTarget {
        let command = WriteCommand::OpenTurn {
            target: target.clone(),
            prompt,
        };
        self.queue_write(command).await;
        target
    }

    /// Queue `msg` for `target`'s turn, with the sub-agent `relay_steps`
    /// behind it when it is a `relay` tool result.
    pub(super) async fn persist_message(
        &self,
        target: PersistTarget,
        msg: PersistedMessage,
        relay_steps: Option<Vec<PersistedRelayStep>>,
    ) {
        let command = WriteCommand::Message {
            target,
            msg,
            relay_steps,
        };
        self.queue_write(command).await;
    }

    /// Queue `stats` for `target`'s turn, ordered after its messages.
    pub(super) async fn persist_turn_stats(&self, target: PersistTarget, stats: TurnStats) {
        self.queue_write(WriteCommand::TurnStats { target, stats })
            .await;
    }

    /// Queue `summary` as the compaction summary of `target`'s branch, standing
    /// in for what `cut` leaves out of every message queued before it.
    pub(super) async fn persist_summary(
        &self,
        target: PersistTarget,
        summary: String,
        cut: CompactionCut,
        elapsed: Duration,
    ) {
        let command = WriteCommand::Summary {
            target,
            summary,
            cut,
            duration_ms: saturating_millis(elapsed),
        };
        self.queue_write(command).await;
    }

    /// Wait for every queued transcript write to land. Hold `user_turn_lock`
    /// across the call so no new user-turn writes queue during the wait.
    pub(super) async fn drain_persistence_inflight(&self) {
        let Some(writes) = self.runtime.transcript_writes.get() else {
            return;
        };
        let (landed_tx, landed_rx) = oneshot::channel();
        if writes.send(WriteCommand::Barrier(landed_tx)).await.is_ok() {
            let _ = landed_rx.await;
        }
    }

    /// Wait for every queued transcript write and every background
    /// persistence task; none may be started afterwards.
    pub async fn finish_persistence(&self) {
        self.runtime.persistence_tracker.close();
        self.drain_persistence_inflight().await;
        self.runtime.persistence_tracker.wait().await;
    }

    /// Hand `command` to the transcript writer, waiting while its queue is full.
    async fn queue_write(&self, command: WriteCommand) {
        if self.transcript_writes().send(command).await.is_err() {
            warn!(
                target: "assistd::memory",
                "the transcript writer has stopped; dropping a write"
            );
        }
    }

    /// The transcript writer's queue, starting the writer on first use.
    fn transcript_writes(&self) -> &mpsc::Sender<WriteCommand> {
        self.runtime.transcript_writes.get_or_init(|| {
            let (writes_tx, writes_rx) = mpsc::channel(WRITE_QUEUE_CAPACITY);
            let writer = TranscriptWriter::new(self.memory.transcripts.clone(), self.embed_wake());
            spawn_supervised(
                "transcript_writer",
                Component::Memory,
                writer.run(writes_rx),
            );
            writes_tx
        })
    }

    /// The embedding worker's wake signal, when embedding is enabled.
    fn embed_wake(&self) -> Option<Arc<Notify>> {
        self.memory
            .embedding_cfg
            .enabled
            .then(|| Arc::clone(&self.memory.embed_wake))
    }
}

#[cfg(test)]
mod tests;
