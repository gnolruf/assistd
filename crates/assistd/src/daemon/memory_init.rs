//! Memory subsystem wiring for the daemon: open SQLite, keep saved messages chunked,
//! then resume the active session or start an unsaved one.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use assistd_core::branch_replay;
use assistd_llm::HistoryReplay;
use assistd_memory::{
    BranchHistory, BranchId, HistoryScope, MemoryStore, MessageChunker, SessionId, SqliteHandle,
    SqliteTranscriptStore, TaskStore, TranscriptStore,
};
use tokio::sync::watch;
use tracing::info;

use super::background_service::BackgroundService;

const MESSAGE_CHUNKING: &str = "message chunking";

pub(super) struct MemorySubsystem {
    pub memory_store: Arc<MemoryStore>,
    pub transcript_store: Arc<dyn TranscriptStore>,
    pub task_store: Arc<TaskStore>,
    pub session_id: SessionId,
    /// `None` for an unsaved session, inserted with its first message.
    pub branch_id: Option<BranchId>,
    /// History of the resumed branch, to replay into the LLM.
    pub resumed_history: HistoryReplay,
    pub sqlite_handle: SqliteHandle,
    /// Every saved relay answer the database names; `None` when it cannot say.
    pub relay_answer_files: Option<HashSet<PathBuf>>,
    message_chunking: MessageChunking,
}

impl MemorySubsystem {
    /// Stop cutting messages into chunks, then close the databases once their queued
    /// writes have run. Must run after every other database client has stopped.
    pub(super) async fn shutdown(self) {
        self.message_chunking.stop.send_replace(true);
        self.message_chunking.task.join().await;
        if let Err(e) = self.sqlite_handle.close().await {
            tracing::warn!("memory: closing the database failed at shutdown: {e:#}");
        }
    }
}

/// The task that cuts saved messages into chunks, and what stops it.
struct MessageChunking {
    task: BackgroundService<()>,
    stop: watch::Sender<bool>,
}

impl MessageChunking {
    /// Cut the messages saved without chunks and each one saved from now on when
    /// `embedding_enabled`; otherwise they wait for a start that has it.
    fn start(handle: &SqliteHandle, embedding_enabled: bool) -> Self {
        let (stop, stopped) = watch::channel(false);
        let task = if embedding_enabled {
            let chunker = MessageChunker::new(handle.clone());
            BackgroundService::spawn(MESSAGE_CHUNKING, chunker.keep_chunked(stopped))
        } else {
            BackgroundService::disabled(MESSAGE_CHUNKING)
        };
        Self { task, stop }
    }
}

/// Open the memory database and its index and pick the session to continue; messages
/// are cut into chunks only when `embedding_enabled`. Fails when there is no data
/// directory or the memory database cannot be opened or migrated.
pub(super) async fn init(embedding_enabled: bool) -> Result<MemorySubsystem> {
    let db_path = db_path().context("no data directory for memory.db (is $HOME set?)")?;
    let handle = SqliteHandle::open(&db_path).await.with_context(|| {
        format!(
            "cannot open the memory database; to start with an empty one, move {} aside",
            db_path.display()
        )
    })?;

    let transcripts = Arc::new(SqliteTranscriptStore::new(handle.clone()));
    let mem_store = Arc::new(MemoryStore::new(handle.clone()));
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
        task_store: Arc::new(TaskStore::new(handle.clone())),
        session_id,
        branch_id,
        resumed_history,
        message_chunking: MessageChunking::start(&handle, embedding_enabled),
        sqlite_handle: handle,
        relay_answer_files,
    })
}

/// Resume the active session, else start an unsaved session that is inserted
/// with its first message.
async fn resume_or_start_unsaved_session(
    transcripts: &SqliteTranscriptStore,
    db_path: &Path,
) -> (SessionId, Option<BranchId>, HistoryReplay) {
    match transcripts.active_session().await {
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
            tracing::warn!("memory: active_session failed: {e:#}; starting fresh");
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
