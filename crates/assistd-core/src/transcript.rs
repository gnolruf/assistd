//! `TranscriptLog`: queues every turn's transcript writes for the single writer task,
//! and drains them.

use std::sync::{Arc, OnceLock};
use std::time::Duration;

use tokio::sync::{mpsc, oneshot};
use tokio_util::task::TaskTracker;
use tracing::warn;

use assistd_ipc::Component;
use assistd_llm::CompactionCut;
use assistd_memory::{
    BranchId, PersistedMessage, SessionId, ToolTrace, TranscriptStore, TurnStats,
};
use assistd_utils::time::saturating_millis;

use self::seq_ledger::SharedLedger;
use self::writer::{TranscriptWriter, WRITE_QUEUE_CAPACITY, WriteCommand};
use crate::recovery::spawn_supervised;

pub(crate) mod agent_runs;
pub(crate) mod messages;
mod seq_ledger;
mod session;
mod writer;

pub use self::session::SessionContext;
pub(crate) use self::writer::PersistTarget;

/// The transcript of every turn: queues writes for the single writer task and
/// tracks the background work that follows a turn.
#[derive(Debug)]
pub(crate) struct TranscriptLog {
    transcripts: Arc<dyn TranscriptStore>,
    session_ctx: Arc<SessionContext>,
    /// The transcript writer's queue, once its first write started the writer.
    writes: OnceLock<mpsc::Sender<WriteCommand>>,
    /// Seqs of the rows behind the main backend's history.
    user_ledger: SharedLedger,
    /// Background session-title tasks, drained at daemon shutdown.
    tracker: TaskTracker,
}

impl TranscriptLog {
    /// A log over `transcripts` for the session `session_ctx` points at.
    pub(crate) fn new(
        transcripts: Arc<dyn TranscriptStore>,
        session_ctx: Arc<SessionContext>,
    ) -> Self {
        Self {
            transcripts,
            session_ctx,
            writes: OnceLock::new(),
            user_ledger: SharedLedger::default(),
            tracker: TaskTracker::new(),
        }
    }

    /// The store the log writes to.
    pub(crate) fn store(&self) -> &Arc<dyn TranscriptStore> {
        &self.transcripts
    }

    /// The active session and branch, inserting the session and its `main`
    /// branch first when the session is still unsaved.
    pub(crate) async fn saved_session(&self) -> assistd_memory::Result<(SessionId, BranchId)> {
        let (session, branch) = self.session_ctx.current().await;
        if let Some(branch) = branch {
            return Ok((session, branch));
        }
        let branch = self.transcripts.begin_active_session(session).await?;
        self.session_ctx.replace(session, Some(branch)).await;
        Ok((session, branch))
    }

    /// Queue the write opening a user turn on `branch`
    /// with `prompt`, returning the target the turn's later messages go to.
    pub(crate) async fn open_user_turn(
        &self,
        branch: BranchId,
        prompt: PersistedMessage,
    ) -> PersistTarget {
        let ledger = self.user_ledger.clone();
        self.open_turn(PersistTarget::new(branch, ledger), prompt)
            .await
    }

    /// [`Self::open_user_turn`] for a task run's turn on its new, empty `branch`.
    pub(crate) async fn open_task_run_turn(
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

    /// Queue `msg` for `target`'s turn, with the `trace` its call left when it is a
    /// tool result.
    pub(crate) async fn persist_message(
        &self,
        target: PersistTarget,
        msg: PersistedMessage,
        trace: Option<ToolTrace>,
    ) {
        let command = WriteCommand::Message { target, msg, trace };
        self.queue_write(command).await;
    }

    /// Queue `stats` for `target`'s turn, ordered after its messages.
    pub(crate) async fn persist_turn_stats(&self, target: PersistTarget, stats: TurnStats) {
        self.queue_write(WriteCommand::TurnStats { target, stats })
            .await;
    }

    /// Queue `summary` as the compaction summary of `target`'s branch, standing
    /// in for what `cut` leaves out of every message queued before it.
    pub(crate) async fn persist_summary(
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
    pub(crate) async fn drain_inflight(&self) {
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
    pub(crate) async fn finish(&self) {
        self.tracker.close();
        self.drain_inflight().await;
        self.tracker.wait().await;
    }

    /// Run `work` in the background; [`Self::finish`] waits for it.
    pub(crate) fn spawn_tracked(&self, work: impl Future<Output = ()> + Send + 'static) {
        self.tracker.spawn(work);
    }

    /// Follow an undo of the user's newest turn.
    pub(crate) fn forget_last_user_turn(&self) {
        self.user_ledger.forget_last_turn();
    }

    /// Follow the main backend's history being replaced from the store.
    pub(crate) fn forget_user_history(&self) {
        self.user_ledger.invalidate();
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
            let writer = TranscriptWriter::new(self.transcripts.clone());
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
mod tests {
    use std::collections::HashMap;
    use std::num::NonZeroUsize;
    use std::sync::Arc;
    use std::time::Duration;

    use chrono::{DateTime, Utc};
    use parking_lot::Mutex;

    use assistd_llm::CompactionCut;
    use assistd_memory::{
        AppendedMessage, BranchHistory, BranchId, BranchInfo, BranchSummary, HistoryScope,
        MemoryError, MessageId, OpenedTurn, PersistedAgentRun, PersistedMessage, PersistedToolCall,
        ResumeCandidate, Seq, SessionId, TranscriptStore, TurnId, TurnStats, UndoOutcome,
    };

    use super::*;

    /// Records what reaches it, makes each write finish faster than the one
    /// before it, and fails any message whose content is `lost`.
    #[derive(Debug)]
    struct RecordingStore {
        arrivals: Mutex<Vec<String>>,
        remaining_delay_ms: Mutex<u64>,
        incomplete_turns: Mutex<Vec<TurnId>>,
        summaries: Mutex<Vec<BranchSummary>>,
    }

    impl RecordingStore {
        fn new(first_delay_ms: u64) -> Self {
            Self {
                arrivals: Mutex::new(Vec::new()),
                remaining_delay_ms: Mutex::new(first_delay_ms),
                incomplete_turns: Mutex::new(Vec::new()),
                summaries: Mutex::new(Vec::new()),
            }
        }

        async fn arrive(&self, msg: PersistedMessage) -> assistd_memory::Result<AppendedMessage> {
            let delay = {
                let mut d = self.remaining_delay_ms.lock();
                let taken = *d;
                *d = d.saturating_sub(20);
                taken
            };
            tokio::time::sleep(Duration::from_millis(delay)).await;
            if msg.content == "lost" {
                return Err(MemoryError::BranchNotFound(BranchId(1)));
            }
            let mut arrivals = self.arrivals.lock();
            let seq = Seq(i64::try_from(arrivals.len()).unwrap());
            arrivals.push(msg.content);
            Ok(AppendedMessage {
                id: MessageId(0),
                seq,
            })
        }
    }

    #[async_trait::async_trait]
    impl TranscriptStore for RecordingStore {
        async fn append_message_to_branch(
            &self,
            _b: BranchId,
            _t: TurnId,
            m: PersistedMessage,
        ) -> assistd_memory::Result<AppendedMessage> {
            self.arrive(m).await
        }

        async fn begin_active_session(&self, _s: SessionId) -> assistd_memory::Result<BranchId> {
            Ok(BranchId(0))
        }
        async fn begin_turn(
            &self,
            _b: BranchId,
            prompt: PersistedMessage,
        ) -> assistd_memory::Result<OpenedTurn> {
            Ok(OpenedTurn {
                turn: TurnId(1),
                message: self.arrive(prompt).await?,
            })
        }
        async fn begin_session_with_main_branch(
            &self,
            _s: SessionId,
        ) -> assistd_memory::Result<BranchId> {
            Ok(BranchId(0))
        }
        async fn activate_branch(&self, _s: SessionId, _b: BranchId) -> assistd_memory::Result<()> {
            Ok(())
        }
        async fn clear_active_session(&self) -> assistd_memory::Result<()> {
            Ok(())
        }
        async fn current_branch(&self, _s: SessionId) -> assistd_memory::Result<Option<BranchId>> {
            Ok(None)
        }
        async fn append_tool_result(
            &self,
            _b: BranchId,
            _t: TurnId,
            m: PersistedMessage,
            _trace: ToolTrace,
        ) -> assistd_memory::Result<AppendedMessage> {
            self.arrive(m).await
        }
        async fn load_agent_runs(
            &self,
            _b: BranchId,
        ) -> assistd_memory::Result<HashMap<MessageId, Vec<PersistedAgentRun>>> {
            Ok(HashMap::new())
        }
        async fn list_branches(&self) -> assistd_memory::Result<Vec<BranchInfo>> {
            Ok(Vec::new())
        }
        async fn branch_info(&self, _b: BranchId) -> assistd_memory::Result<Option<BranchInfo>> {
            Ok(None)
        }
        async fn resolve_branch(
            &self,
            _t: &str,
            _p: Option<SessionId>,
        ) -> assistd_memory::Result<Option<(SessionId, BranchId)>> {
            Ok(None)
        }
        async fn resolve_session(&self, _p: &str) -> assistd_memory::Result<Option<SessionId>> {
            Ok(None)
        }
        async fn fork_and_switch(
            &self,
            _src: BranchId,
            _name: &str,
        ) -> assistd_memory::Result<BranchId> {
            Ok(BranchId(0))
        }
        async fn load_branch(
            &self,
            _b: BranchId,
            _scope: HistoryScope,
        ) -> assistd_memory::Result<BranchHistory> {
            Ok(BranchHistory::default())
        }
        async fn record_branch_summary(
            &self,
            _b: BranchId,
            summary: BranchSummary,
        ) -> assistd_memory::Result<()> {
            self.summaries.lock().push(summary);
            Ok(())
        }

        async fn mark_turn_incomplete(&self, turn: TurnId) -> assistd_memory::Result<()> {
            self.incomplete_turns.lock().push(turn);
            Ok(())
        }

        async fn record_turn_stats(
            &self,
            _turn: TurnId,
            _stats: TurnStats,
        ) -> assistd_memory::Result<()> {
            Ok(())
        }
        async fn latest_branch_activity(
            &self,
            _b: BranchId,
        ) -> assistd_memory::Result<Option<DateTime<Utc>>> {
            Ok(None)
        }
        async fn undo_last_turn(&self, _b: BranchId) -> assistd_memory::Result<UndoOutcome> {
            Ok(UndoOutcome::default())
        }
        async fn active_session(&self) -> assistd_memory::Result<Option<ResumeCandidate>> {
            Ok(None)
        }
        async fn session_title(&self, _s: SessionId) -> assistd_memory::Result<Option<String>> {
            Ok(None)
        }
        async fn set_session_title(&self, _s: SessionId, _t: &str) -> assistd_memory::Result<()> {
            Ok(())
        }
    }

    fn log_over(store: Arc<dyn TranscriptStore>) -> TranscriptLog {
        let session = Arc::new(SessionContext::new(SessionId::new(), Some(BranchId(1))));
        TranscriptLog::new(store, session)
    }

    fn tool_step(content: &str) -> PersistedMessage {
        let call = PersistedToolCall {
            id: "call".into(),
            name: "run".into(),
            arguments: serde_json::Value::Null,
        };
        PersistedMessage::assistant_tool_calls(content, vec![call], None)
    }

    #[tokio::test(start_paused = true)]
    async fn messages_reach_the_store_in_queue_order_past_an_abandoned_drain() {
        let store = Arc::new(RecordingStore::new(60));
        let log = log_over(store.clone());

        let target = log
            .open_user_turn(BranchId(1), PersistedMessage::user("first"))
            .await;
        let reply = |text| PersistedMessage::assistant_text(text, None);
        log.persist_message(target.clone(), reply("second"), None)
            .await;
        let abandoned = Duration::from_millis(10);
        tokio::time::timeout(abandoned, log.drain_inflight())
            .await
            .unwrap_err();
        for text in ["third", "fourth"] {
            log.persist_message(target.clone(), reply(text), None).await;
        }
        log.drain_inflight().await;

        assert_eq!(
            *store.arrivals.lock(),
            ["first", "second", "third", "fourth"],
            "a slower early write must not let a later message take its seq"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn a_failed_write_marks_its_turn_and_still_counts_toward_the_summary_cut() {
        let store = Arc::new(RecordingStore::new(0));
        let log = log_over(store.clone());

        let target = log
            .open_user_turn(BranchId(1), PersistedMessage::user("question"))
            .await;
        for step in ["step one", "lost", "step three"] {
            log.persist_message(target.clone(), tool_step(step), None)
                .await;
        }
        let cut = CompactionCut {
            kept_turns: NonZeroUsize::MIN,
            kept_steps: NonZeroUsize::new(2),
        };
        log.persist_summary(target, "summary".into(), cut, Duration::ZERO)
            .await;
        log.drain_inflight().await;

        assert_eq!(*store.incomplete_turns.lock(), [TurnId(1)]);
        assert_eq!(
            *store.arrivals.lock(),
            ["question", "step one", "step three"]
        );
        assert_eq!(
            *store.summaries.lock(),
            [BranchSummary {
                kept_from_seq: Seq(0),
                steps_folded_through: Some(Seq(1)),
                compacted_after_seq: Seq(2),
                content: "summary".into(),
                duration_ms: 0,
            }],
            "the model kept the unsaved step, so only the first step is folded"
        );
    }
}
