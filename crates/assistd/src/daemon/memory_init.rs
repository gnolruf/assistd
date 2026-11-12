//! Memory subsystem wiring for the daemon: open SQLite, then resume the
//! latest unended session or start an unsaved one.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use assistd_core::branch_replay;
use assistd_llm::HistoryReplay;
use assistd_memory::{
    BranchHistory, BranchId, HistoryScope, MemoryError, MemoryStore, SessionId, SqliteHandle,
    SqliteMemoryStore, SqliteTaskStore, SqliteTranscriptStore, TaskStore, TranscriptStore,
};
use tracing::info;

pub(super) struct MemorySubsystem {
    pub memory_store: Arc<dyn MemoryStore>,
    pub transcript_store: Arc<dyn TranscriptStore>,
    pub task_store: Arc<dyn TaskStore>,
    pub session_id: SessionId,
    /// `None` for an unsaved session, inserted with its first message.
    pub branch_id: Option<BranchId>,
    /// History of the resumed branch, to replay into the LLM.
    pub resumed_history: HistoryReplay,
    pub sqlite_handle: SqliteHandle,
    /// Every saved relay answer the database names; `None` when it cannot say.
    pub relay_answer_files: Option<HashSet<PathBuf>>,
}

impl MemorySubsystem {
    /// End the session row, if it was ever saved, then close the database once its
    /// queued writes have run. Must run after every other database client has stopped.
    pub(super) async fn shutdown(self) {
        match self.transcript_store.end_session(self.session_id).await {
            Ok(()) | Err(MemoryError::SessionNotFound(_)) => {}
            Err(e) => tracing::warn!("memory: end_session failed at shutdown: {e:#}"),
        }
        if let Err(e) = self.sqlite_handle.close().await {
            tracing::warn!("memory: closing the database failed at shutdown: {e:#}");
        }
    }
}

/// Open the memory database and pick the session to continue; messages are cut into
/// chunks only when `embedding_enabled`. Fails when there is no data directory or the
/// database cannot be opened or migrated.
pub(super) async fn init(embedding_enabled: bool) -> Result<MemorySubsystem> {
    let db_path = db_path().context("no data directory for memory.db (is $HOME set?)")?;
    let handle = SqliteHandle::open(&db_path).await.with_context(|| {
        format!(
            "cannot open the memory database; to start with an empty one, move {} aside",
            db_path.display()
        )
    })?;

    let transcripts = Arc::new(transcript_store(handle.clone(), embedding_enabled).await);
    let mem_store = Arc::new(SqliteMemoryStore::new(handle.clone()));
    let (session_id, branch_id, resumed_history) =
        resume_or_start_unsaved_session(transcripts.as_ref(), &db_path).await;
    let relay_answer_files = transcripts
        .relay_answer_files()
        .await
        .inspect_err(|e| tracing::warn!("memory: cannot list saved relay answers ({e:#})"))
        .ok()
        .map(HashSet::from_iter);

    Ok(MemorySubsystem {
        memory_store: mem_store,
        transcript_store: transcripts,
        task_store: Arc::new(SqliteTaskStore::new(handle.clone())),
        session_id,
        branch_id,
        resumed_history,
        sqlite_handle: handle,
        relay_answer_files,
    })
}

/// The transcript store: with embedding on it cuts chunks, first for the messages
/// stored while embedding was off; with it off it cuts none.
async fn transcript_store(handle: SqliteHandle, embedding_enabled: bool) -> SqliteTranscriptStore {
    let transcripts = SqliteTranscriptStore::new(handle);
    if !embedding_enabled {
        return transcripts.without_message_chunks();
    }
    match transcripts.chunk_unchunked_messages().await {
        Ok(0) => {}
        Ok(added) => info!("memory: cut {added} chunks from messages stored without embedding"),
        Err(e) => tracing::warn!(
            "memory: cannot chunk messages stored without embedding ({e:#}); \
             they stay out of semantic search"
        ),
    }
    transcripts
}

/// Resume the most recent unended session, else start an unsaved session
/// that is inserted with its first message.
async fn resume_or_start_unsaved_session(
    transcripts: &SqliteTranscriptStore,
    db_path: &Path,
) -> (SessionId, Option<BranchId>, HistoryReplay) {
    match transcripts.find_resumable_session().await {
        Ok(Some(cand)) => {
            info!(
                "memory: resuming prior session {} (branch={})",
                cand.session_id, cand.current_branch_id
            );
            let history = load_resumed_replay(transcripts, cand.current_branch_id).await;
            (cand.session_id, Some(cand.current_branch_id), history)
        }
        Ok(None) => {
            info!(
                "memory: SQLite ready at {} (new session)",
                db_path.display()
            );
            (SessionId::new(), None, HistoryReplay::default())
        }
        Err(e) => {
            tracing::warn!("memory: find_resumable_session failed: {e:#}; starting fresh");
            (SessionId::new(), None, HistoryReplay::default())
        }
    }
}

/// The resumed branch's unsummarized history and summary as the LLM replays
/// them; a failed load replays nothing in its place.
async fn load_resumed_replay(
    transcripts: &SqliteTranscriptStore,
    branch: BranchId,
) -> HistoryReplay {
    let BranchHistory { summary, rows } = transcripts
        .load_branch(branch, HistoryScope::Unsummarized)
        .await
        .inspect_err(|e| tracing::warn!("memory: load_branch failed for resume ({e:#})"))
        .unwrap_or_default();
    branch_replay(&rows, summary)
}

/// `$XDG_DATA_HOME/assistd/memory.db`; `None` when there is no data directory.
fn db_path() -> Option<PathBuf> {
    dirs::data_dir().map(|data| data.join("assistd").join("memory.db"))
}
