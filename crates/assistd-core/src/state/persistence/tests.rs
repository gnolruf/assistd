use parking_lot::Mutex;

use assistd_config::Config;
use assistd_ipc::PresenceState;
use assistd_llm::EchoBackend;
use assistd_memory::{
    BranchId, BranchInfo, ConversationStore, HistoryRow, ResumeCandidate, SessionId, TurnSummary,
    UndoOutcome,
};
use assistd_tools::ToolRegistry;

use super::*;
use crate::VoiceOutputController;
use crate::presence::PresenceManager;
use crate::state::memory_stack::MemoryStack;
use crate::state::runtime::RuntimeState;
use crate::state::subsystems::Subsystems;

/// Records the order writes arrive in, and makes each write finish
/// faster than the one before it. Unchained, that inverts the order
/// the daemon queued the messages in.
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
}

#[async_trait::async_trait]
impl ConversationStore for SlowingStore {
    async fn append_message_to_branch(
        &self,
        _s: &SessionId,
        _b: BranchId,
        _t: Option<TurnId>,
        m: PersistedMessage,
    ) -> assistd_memory::Result<i64> {
        let delay = {
            let mut d = self.remaining_delay_ms.lock();
            let taken = *d;
            *d = d.saturating_sub(20);
            taken
        };
        tokio::time::sleep(Duration::from_millis(delay)).await;
        self.arrivals.lock().push(m.content);
        Ok(0)
    }

    async fn end_session(&self, _id: &SessionId) -> assistd_memory::Result<()> {
        Ok(())
    }
    async fn begin_turn(&self, _s: &SessionId, _t: &str) -> assistd_memory::Result<TurnId> {
        Ok(TurnId(0))
    }
    async fn end_turn(&self, _t: TurnId) -> assistd_memory::Result<()> {
        Ok(())
    }
    async fn recent_turns(&self, _l: usize) -> assistd_memory::Result<Vec<TurnSummary>> {
        Ok(Vec::new())
    }
    async fn begin_session_with_main_branch(
        &self,
        _pid: u32,
    ) -> assistd_memory::Result<(SessionId, BranchId)> {
        Ok((SessionId::new(), BranchId(0)))
    }
    async fn set_current_branch(&self, _s: &SessionId, _b: BranchId) -> assistd_memory::Result<()> {
        Ok(())
    }
    async fn get_current_branch(&self, _s: &SessionId) -> assistd_memory::Result<Option<BranchId>> {
        Ok(None)
    }
    async fn list_branches(&self) -> assistd_memory::Result<Vec<BranchInfo>> {
        Ok(Vec::new())
    }
    async fn resolve_branch(
        &self,
        _t: &str,
        _p: Option<&SessionId>,
    ) -> assistd_memory::Result<Option<(SessionId, BranchId)>> {
        Ok(None)
    }
    async fn fork_branch(&self, _src: BranchId, _name: &str) -> assistd_memory::Result<BranchId> {
        Ok(BranchId(0))
    }
    async fn load_branch_history(&self, _b: BranchId) -> assistd_memory::Result<Vec<HistoryRow>> {
        Ok(Vec::new())
    }
    async fn latest_branch_activity(&self, _b: BranchId) -> assistd_memory::Result<Option<String>> {
        Ok(None)
    }
    async fn undo_last_turn(&self, _b: BranchId) -> assistd_memory::Result<UndoOutcome> {
        Ok(UndoOutcome::default())
    }
    async fn find_resumable_session(&self) -> assistd_memory::Result<Option<ResumeCandidate>> {
        Ok(None)
    }
    async fn get_session_title(&self, _s: &SessionId) -> assistd_memory::Result<Option<String>> {
        Ok(None)
    }
    async fn set_session_title(&self, _s: &SessionId, _t: &str) -> assistd_memory::Result<()> {
        Ok(())
    }
}

fn state_with_store(store: Arc<dyn ConversationStore>) -> Arc<AppState> {
    let config = Config::default();
    let memory = MemoryStack::disabled(config.embedding.clone()).with_conversations(store);
    Arc::new(AppState {
        subsystems: Subsystems::new(
            Arc::new(EchoBackend::new()),
            PresenceManager::stub(PresenceState::Active),
            Arc::new(ToolRegistry::default()),
            Arc::new(assistd_voice::NoVoiceInput::new()),
            Arc::new(assistd_voice::NoContinuousListener::new()),
            VoiceOutputController::new(Arc::new(assistd_voice::NoVoiceOutput), true),
        ),
        memory,
        runtime: RuntimeState::new(),
        config,
    })
}

#[tokio::test(start_paused = true)]
async fn messages_reach_the_store_in_the_order_they_were_queued() {
    let store = Arc::new(SlowingStore::new(60));
    let state = state_with_store(store.clone());

    for text in ["first", "second", "third", "fourth"] {
        state.persist_message_fire_and_forget(
            None,
            PersistedMessage::assistant_text(text.to_string()),
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
