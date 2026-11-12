use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Utc};
use parking_lot::Mutex;

use assistd_llm::CompactionCut;
use assistd_memory::{
    AppendedMessage, BranchHistory, BranchId, BranchInfo, BranchSummary, HistoryScope, MemoryError,
    MessageId, OpenedTurn, PersistedMessage, PersistedRelayStep, PersistedToolCall,
    ResumeCandidate, Seq, SessionId, TranscriptStore, TurnId, TurnStats, UndoOutcome,
};

use super::*;
use crate::state::session::SessionContext;

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
            chunks: Vec::new(),
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
    async fn append_relay_result(
        &self,
        _b: BranchId,
        _t: TurnId,
        m: PersistedMessage,
        _steps: Vec<PersistedRelayStep>,
    ) -> assistd_memory::Result<AppendedMessage> {
        self.arrive(m).await
    }
    async fn load_relay_traces(
        &self,
        _b: BranchId,
    ) -> assistd_memory::Result<HashMap<MessageId, Vec<PersistedRelayStep>>> {
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
    TranscriptLog::new(store, None, session)
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
