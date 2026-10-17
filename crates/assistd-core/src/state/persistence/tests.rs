use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Utc};
use parking_lot::Mutex;

use assistd_config::Config;
use assistd_ipc::PresenceTarget;
use assistd_llm::EchoBackend;
use assistd_memory::{
    AppendedMessage, BranchId, BranchInfo, BranchSummary, HistoryRow, MessageId, OpenedTurn,
    ResumeCandidate, SessionId, TranscriptStore, UndoOutcome,
};
use assistd_tools::ToolRegistry;

use super::*;
use crate::presence::PresenceManager;
use crate::state::memory_stack::MemoryStack;
use crate::state::runtime::{RuntimeState, SessionContext};
use crate::state::subsystems::Subsystems;

/// Records the order writes arrive in, and makes each write finish
/// faster than the one before it. Unchained, that inverts the order
/// the daemon queued the messages in.
#[derive(Debug)]
struct SlowingStore {
    arrivals: Mutex<Vec<String>>,
    remaining_delay_ms: Mutex<u64>,
}

impl SlowingStore {
    fn new(first_delay_ms: u64) -> Self {
        Self {
            arrivals: Mutex::new(Vec::new()),
            remaining_delay_ms: Mutex::new(first_delay_ms),
        }
    }

    async fn arrive(&self, msg: PersistedMessage) -> AppendedMessage {
        let delay = {
            let mut d = self.remaining_delay_ms.lock();
            let taken = *d;
            *d = d.saturating_sub(20);
            taken
        };
        tokio::time::sleep(Duration::from_millis(delay)).await;
        self.arrivals.lock().push(msg.content);
        AppendedMessage {
            id: MessageId(0),
            chunks: Vec::new(),
        }
    }
}

#[async_trait::async_trait]
impl TranscriptStore for SlowingStore {
    async fn append_message_to_branch(
        &self,
        _b: BranchId,
        _t: TurnId,
        m: PersistedMessage,
    ) -> assistd_memory::Result<AppendedMessage> {
        Ok(self.arrive(m).await)
    }

    async fn end_session(&self, _s: SessionId) -> assistd_memory::Result<()> {
        Ok(())
    }
    async fn begin_turn(
        &self,
        _b: BranchId,
        prompt: PersistedMessage,
    ) -> assistd_memory::Result<OpenedTurn> {
        Ok(OpenedTurn {
            turn: TurnId(1),
            message: self.arrive(prompt).await,
        })
    }
    async fn begin_session_with_main_branch(
        &self,
        _s: SessionId,
        _pid: u32,
    ) -> assistd_memory::Result<BranchId> {
        Ok(BranchId(0))
    }
    async fn set_current_branch(&self, _s: SessionId, _b: BranchId) -> assistd_memory::Result<()> {
        Ok(())
    }
    async fn current_branch(&self, _s: SessionId) -> assistd_memory::Result<Option<BranchId>> {
        Ok(None)
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
    async fn fork_and_switch(
        &self,
        _src: BranchId,
        _name: &str,
    ) -> assistd_memory::Result<BranchId> {
        Ok(BranchId(0))
    }
    async fn load_branch_history(&self, _b: BranchId) -> assistd_memory::Result<Vec<HistoryRow>> {
        Ok(Vec::new())
    }
    async fn record_branch_summary(
        &self,
        _b: BranchId,
        _summary: String,
        _cut: SummaryCut,
        _duration_ms: u64,
    ) -> assistd_memory::Result<()> {
        Ok(())
    }
    async fn load_branch_summary(
        &self,
        _b: BranchId,
    ) -> assistd_memory::Result<Option<BranchSummary>> {
        Ok(None)
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
    async fn find_resumable_session(&self) -> assistd_memory::Result<Option<ResumeCandidate>> {
        Ok(None)
    }
    async fn session_title(&self, _s: SessionId) -> assistd_memory::Result<Option<String>> {
        Ok(None)
    }
    async fn set_session_title(&self, _s: SessionId, _t: &str) -> assistd_memory::Result<()> {
        Ok(())
    }
}

fn state_with_store(store: Arc<dyn TranscriptStore>) -> Arc<AppState> {
    let config = Config::default();
    let memory = MemoryStack::disabled(config.embedding.clone()).with_transcripts(store);
    Arc::new(AppState {
        subsystems: Subsystems::new(
            Arc::new(EchoBackend::new()),
            PresenceManager::stub(PresenceTarget::Active),
            Arc::new(ToolRegistry::default()),
            assistd_voice::VoiceManager::new(true),
        ),
        memory,
        runtime: RuntimeState::new().with_session_ctx(Arc::new(SessionContext::new(
            SessionId::new(),
            Some(BranchId(1)),
        ))),
        config,
    })
}

#[tokio::test(start_paused = true)]
async fn messages_reach_the_store_in_the_order_they_were_queued() {
    let store = Arc::new(SlowingStore::new(60));
    let state = state_with_store(store.clone());

    let target = state.open_turn_fire_and_forget(BranchId(1), PersistedMessage::user("first"));
    for text in ["second", "third", "fourth"] {
        state.persist_message_fire_and_forget(
            target.clone(),
            PersistedMessage::assistant_text(text, None),
        );
    }
    state.runtime.persistence_tracker.close();
    state.runtime.persistence_tracker.wait().await;

    assert_eq!(
        *store.arrivals.lock(),
        ["first", "second", "third", "fourth"],
        "a slower early write must not let a later message take its seq"
    );
}
