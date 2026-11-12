//! Queueing transcript writes for the transcript writer, and its drain.

use std::sync::{Arc, OnceLock};
use std::time::Duration;

use tokio::sync::{Notify, mpsc, oneshot};
use tokio_util::task::TaskTracker;
use tracing::warn;

use assistd_ipc::Component;
use assistd_llm::CompactionCut;
use assistd_memory::{
    BranchId, PersistedMessage, PersistedRelayStep, SessionId, TranscriptStore, TurnStats,
};
use assistd_utils::time::saturating_millis;

use super::seq_ledger::SharedLedger;
use super::session::SessionContext;
use super::transcript_writer::{
    PersistTarget, TranscriptWriter, WRITE_QUEUE_CAPACITY, WriteCommand,
};
use crate::recovery::spawn_supervised;

/// The transcript of every conversation: queues writes for the single writer task and
/// tracks the background work that follows a turn.
#[derive(Debug)]
pub(super) struct TranscriptLog {
    transcripts: Arc<dyn TranscriptStore>,
    /// The embedding worker's wake signal, when embedding is enabled.
    embed_wake: Option<Arc<Notify>>,
    session_ctx: Arc<SessionContext>,
    /// The transcript writer's queue, once its first write started the writer.
    writes: OnceLock<mpsc::Sender<WriteCommand>>,
    /// Seqs of the rows behind the user's conversation.
    pub(super) user_ledger: SharedLedger,
    /// Background session-title tasks, drained at daemon shutdown.
    pub(super) tracker: TaskTracker,
}

impl TranscriptLog {
    /// A log over `transcripts` for the conversation `session_ctx` points at, waking
    /// the embedding worker through `embed_wake` after each write.
    pub(super) fn new(
        transcripts: Arc<dyn TranscriptStore>,
        embed_wake: Option<Arc<Notify>>,
        session_ctx: Arc<SessionContext>,
    ) -> Self {
        Self {
            transcripts,
            embed_wake,
            session_ctx,
            writes: OnceLock::new(),
            user_ledger: SharedLedger::default(),
            tracker: TaskTracker::new(),
        }
    }

    /// The store the log writes to.
    pub(super) fn store(&self) -> &Arc<dyn TranscriptStore> {
        &self.transcripts
    }

    /// The active session and branch, inserting the session and its `main`
    /// branch first when the session is still unsaved.
    pub(super) async fn saved_session(&self) -> assistd_memory::Result<(SessionId, BranchId)> {
        let (session, branch) = self.session_ctx.current().await;
        if let Some(branch) = branch {
            return Ok((session, branch));
        }
        let branch = self
            .transcripts
            .begin_session_with_main_branch(session)
            .await?;
        self.session_ctx.replace(session, Some(branch)).await;
        Ok((session, branch))
    }

    /// Queue the write opening a turn of the user's conversation on `branch`
    /// with `prompt`, returning the target the turn's later messages go to.
    pub(super) async fn open_user_turn(
        &self,
        branch: BranchId,
        prompt: PersistedMessage,
    ) -> PersistTarget {
        let ledger = self.user_ledger.clone();
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

    /// Wait for every queued transcript write to land. Hold the user-turn lock
    /// across the call so no new user-turn writes queue during the wait.
    pub(super) async fn drain_inflight(&self) {
        let Some(writes) = self.writes.get() else {
            return;
        };
        let (landed_tx, landed_rx) = oneshot::channel();
        if writes.send(WriteCommand::Barrier(landed_tx)).await.is_ok() {
            let _ = landed_rx.await;
        }
    }

    /// Wait for every queued transcript write and every background
    /// persistence task; none may be started afterwards.
    pub(super) async fn finish(&self) {
        self.tracker.close();
        self.drain_inflight().await;
        self.tracker.wait().await;
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
        self.writes.get_or_init(|| {
            let (writes_tx, writes_rx) = mpsc::channel(WRITE_QUEUE_CAPACITY);
            let writer = TranscriptWriter::new(self.transcripts.clone(), self.embed_wake.clone());
            spawn_supervised(
                "transcript_writer",
                Component::Memory,
                writer.run(writes_rx),
            );
            writes_tx
        })
    }
}

#[cfg(test)]
mod tests;
