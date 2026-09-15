//! Fire-and-forget persistence pipeline and in-flight drain helper.
//!
//! `persist_message_fire_and_forget` spawns the writer task; every
//! handler that mutates branches (`/fork`, `/switch`, `/undo`,
//! `/resume`, `/new`) calls `drain_persistence_inflight` while holding
//! the agent turn lock so previously-spawned writes have a chance to
//! land before the next branch op.

use super::AppState;
use assistd_embed::EmbedJob;
use assistd_memory::{ChunkingConfig, PersistedMessage, PersistedRole, TurnId, chunk_message};

impl AppState {
    /// Fire-and-forget persist of one message. Spawns a task that:
    /// 1. Writes the row via the conversation store (returns row id).
    /// 2. If the row is a User/Assistant text message and embedding is
    ///    enabled, splits the content into chunks, persists each chunk
    ///    (returns chunk id), and `try_send`s an `EmbedJob::Chunk` for
    ///    each so the embedder task can index it.
    ///
    /// The whole pipeline is on a `tokio::spawn`'d task so the dispatch
    /// loop never waits on disk or the embed queue. `try_send` (not
    /// `send`) so a wedged embedder doesn't backpressure persistence;
    /// dropped jobs just leave the chunk row unindexed for the next
    /// backfill pass.
    ///
    /// Writes land in call order: each task takes the previous task's
    /// completion signal off `runtime.persist_chain` — a synchronous
    /// swap, before any await — and waits on it before appending. The
    /// store assigns `seq` in arrival order, so without the chain a
    /// tool result could be sequenced ahead of the call that produced
    /// it and `/switch` would replay a jumbled transcript. Chunking and
    /// embedding stay off the chain; they run after the signal fires.
    pub(super) fn persist_message_fire_and_forget(
        &self,
        turn: Option<TurnId>,
        msg: PersistedMessage,
    ) {
        let conv = self.memory.conversations.clone();
        let conversation_ctx = self.runtime.conversation_ctx.clone();
        let chunks_handle = self.memory.chunks.clone();
        let embed_tx = self.memory.embed_tx.clone();
        let embedding_enabled = self.memory.embedding_cfg.enabled;
        let chunking_cfg = ChunkingConfig {
            chunk_chars: self.memory.embedding_cfg.chunk_chars,
            overlap_chars: self.memory.embedding_cfg.chunk_overlap_chars,
        };
        let should_embed = embedding_enabled
            && chunks_handle.is_some()
            && matches!(msg.role, PersistedRole::User | PersistedRole::Assistant)
            && !msg.content.is_empty()
            && msg.tool_calls.is_none();
        let content_for_chunks = if should_embed {
            Some(msg.content.clone())
        } else {
            None
        };
        let (landed_tx, landed_rx) = tokio::sync::oneshot::channel();
        let previous = self.runtime.persist_chain.lock().replace(landed_rx);
        self.runtime.persistence_tracker.spawn(async move {
            // An `Err` here means the predecessor's task was dropped
            // without writing; there is nothing left to wait for.
            if let Some(previous) = previous {
                let _ = previous.await;
            }
            let (session, branch) = conversation_ctx.current().await;
            let append = conv
                .append_message_to_branch(&session, branch, turn, msg)
                .await;
            let _ = landed_tx.send(());
            let row_id = match append {
                Ok(id) => id,
                Err(e) => {
                    tracing::warn!(
                        target: "assistd::memory",
                        error = %e,
                        "failed to persist message (continuing)"
                    );
                    return;
                }
            };
            // NoConversationStore returns 0: nothing to chunk.
            let Some(content) = content_for_chunks else {
                return;
            };
            let Some(chunks_handle) = chunks_handle else {
                return;
            };
            if row_id == 0 {
                return;
            }
            for (idx, chunk) in chunk_message(&content, &chunking_cfg)
                .into_iter()
                .enumerate()
            {
                match chunks_handle
                    .store_chunk(row_id, idx as i64, chunk.clone(), None)
                    .await
                {
                    Ok(chunk_id) => {
                        if embed_tx
                            .try_send(EmbedJob::Chunk {
                                chunk_id,
                                text: chunk,
                            })
                            .is_err()
                        {
                            tracing::debug!(
                                target: "assistd::embed",
                                chunk_id,
                                "embed queue full or closed; dropping job"
                            );
                        }
                    }
                    Err(e) => tracing::warn!(
                        target: "assistd::memory",
                        conversation_id = row_id,
                        chunk_index = idx,
                        error = %e,
                        "failed to persist chunk (continuing)"
                    ),
                }
            }
        });
    }

    /// Block until every previously-spawned `persist_message_fire_and_forget`
    /// task has landed. Held inside `agent_turn_lock` so no new tasks
    /// can spawn during the wait.
    pub(super) async fn drain_persistence_inflight(&self) {
        let deadline = std::time::Instant::now() + std::time::Duration::from_millis(500);
        while !self.runtime.persistence_tracker.is_empty() && std::time::Instant::now() < deadline {
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
        if !self.runtime.persistence_tracker.is_empty() {
            tracing::warn!(
                target: "assistd::state",
                "persistence drain timed out; in-flight writes may race branch op"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::VoiceOutputController;
    use crate::presence::PresenceManager;
    use crate::state::memory_stack::MemoryStack;
    use crate::state::runtime::RuntimeState;
    use crate::state::subsystems::Subsystems;
    use assistd_config::Config;
    use assistd_ipc::PresenceState;
    use assistd_llm::EchoBackend;
    use assistd_memory::{
        BranchId, BranchInfo, ConversationStore, HistoryRow, ResumeCandidate, SearchHit, SessionId,
        TurnSummary, UndoOutcome,
    };
    use assistd_tools::ToolRegistry;
    use parking_lot::Mutex;
    use std::sync::Arc;
    use std::time::Duration;

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
        ) -> anyhow::Result<i64> {
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

        async fn begin_session(&self, _pid: u32) -> anyhow::Result<SessionId> {
            Ok(SessionId::new())
        }
        async fn end_session(&self, _id: &SessionId) -> anyhow::Result<()> {
            Ok(())
        }
        async fn begin_turn(&self, _s: &SessionId, _t: &str) -> anyhow::Result<TurnId> {
            Ok(TurnId(0))
        }
        async fn end_turn(&self, _t: TurnId) -> anyhow::Result<()> {
            Ok(())
        }
        async fn append_message(
            &self,
            _s: &SessionId,
            _t: Option<TurnId>,
            _m: PersistedMessage,
        ) -> anyhow::Result<i64> {
            Ok(0)
        }
        async fn search(&self, _q: &str, _l: usize) -> anyhow::Result<Vec<SearchHit>> {
            Ok(Vec::new())
        }
        async fn recent_turns(&self, _l: usize) -> anyhow::Result<Vec<TurnSummary>> {
            Ok(Vec::new())
        }
        async fn begin_session_with_main_branch(
            &self,
            _pid: u32,
        ) -> anyhow::Result<(SessionId, BranchId)> {
            Ok((SessionId::new(), BranchId(0)))
        }
        async fn create_branch(
            &self,
            _s: &SessionId,
            _name: &str,
            _parent: Option<BranchId>,
            _fp: Option<i64>,
        ) -> anyhow::Result<BranchId> {
            Ok(BranchId(0))
        }
        async fn set_current_branch(&self, _s: &SessionId, _b: BranchId) -> anyhow::Result<()> {
            Ok(())
        }
        async fn get_current_branch(&self, _s: &SessionId) -> anyhow::Result<Option<BranchId>> {
            Ok(None)
        }
        async fn list_branches(&self) -> anyhow::Result<Vec<BranchInfo>> {
            Ok(Vec::new())
        }
        async fn resolve_branch(
            &self,
            _t: &str,
            _p: Option<&SessionId>,
        ) -> anyhow::Result<Option<(SessionId, BranchId)>> {
            Ok(None)
        }
        async fn fork_branch(&self, _src: BranchId, _name: &str) -> anyhow::Result<BranchId> {
            Ok(BranchId(0))
        }
        async fn load_branch_history(&self, _b: BranchId) -> anyhow::Result<Vec<HistoryRow>> {
            Ok(Vec::new())
        }
        async fn latest_branch_activity(&self, _b: BranchId) -> anyhow::Result<Option<String>> {
            Ok(None)
        }
        async fn undo_last_turn(&self, _b: BranchId) -> anyhow::Result<UndoOutcome> {
            Ok(UndoOutcome::default())
        }
        async fn find_resumable_session(&self) -> anyhow::Result<Option<ResumeCandidate>> {
            Ok(None)
        }
        async fn get_session_title(&self, _s: &SessionId) -> anyhow::Result<Option<String>> {
            Ok(None)
        }
        async fn set_session_title(&self, _s: &SessionId, _t: &str) -> anyhow::Result<()> {
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

    #[tokio::test]
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
            vec![
                "first".to_string(),
                "second".to_string(),
                "third".to_string(),
                "fourth".to_string()
            ],
            "a slower early write must not let a later message take its seq"
        );
    }
}
