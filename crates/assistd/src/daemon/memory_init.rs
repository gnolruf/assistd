//! Memory subsystem wiring for the daemon: open SQLite, then resume a
//! session whose daemon has died or start an unsaved one.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use assistd_core::{Config, LlmReplay};
use assistd_memory::{
    BranchId, MemoryStore, NoMemoryStore, NoTranscriptStore, SessionId, SqliteHandle,
    SqliteMemoryStore, SqliteTranscriptStore, TranscriptStore,
};
use rustix::io::Errno;
use rustix::process::Pid;
use tracing::info;

pub(super) struct MemorySubsystem {
    pub memory_store: Arc<dyn MemoryStore>,
    pub transcript_store: Arc<dyn TranscriptStore>,
    pub session_id: SessionId,
    /// `None` for an unsaved session, inserted with its first message.
    pub branch_id: Option<BranchId>,
    /// History of the resumed branch, to replay into the LLM.
    pub resumed_history: LlmReplay,
    pub sqlite_handle: Option<SqliteHandle>,
}

impl MemorySubsystem {
    fn disabled() -> Self {
        Self {
            memory_store: Arc::new(NoMemoryStore),
            transcript_store: Arc::new(NoTranscriptStore),
            session_id: SessionId::new(),
            branch_id: None,
            resumed_history: LlmReplay::default(),
            sqlite_handle: None,
        }
    }

    /// End the session row, then close the database once its queued writes
    /// have run. Must run after every other database client has stopped.
    pub(super) async fn shutdown(self) {
        if let Err(e) = self.transcript_store.end_session(self.session_id).await {
            tracing::warn!("memory: end_session failed at shutdown: {e:#}");
        }
        if let Some(handle) = self.sqlite_handle
            && let Err(e) = handle.close().await
        {
            tracing::warn!("memory: closing the database failed at shutdown: {e:#}");
        }
    }
}

/// Degrades to a no-op subsystem when disabled or when the database
/// cannot be opened.
pub(super) async fn init(config: &Config) -> MemorySubsystem {
    if !config.memory.enabled {
        info!("memory: disabled in config (memory.enabled = false)");
        return MemorySubsystem::disabled();
    }

    let db_path = PathBuf::from(&config.memory.db_path);
    let handle = match SqliteHandle::open(&db_path).await {
        Ok(handle) => handle,
        Err(e) => {
            tracing::warn!(
                "memory: failed to open {} ({e:#}); persistence disabled this run",
                db_path.display()
            );
            return MemorySubsystem::disabled();
        }
    };

    let transcripts = Arc::new(SqliteTranscriptStore::new(handle.clone()));
    let mem_store = Arc::new(SqliteMemoryStore::new(handle.clone()));
    let (session_id, branch_id, resumed_history) =
        resume_or_start_unsaved_session(transcripts.as_ref(), &db_path).await;

    MemorySubsystem {
        memory_store: mem_store,
        transcript_store: transcripts,
        session_id,
        branch_id,
        resumed_history,
        sqlite_handle: Some(handle),
    }
}

/// Resume the most recent session whose daemon is dead, else start an
/// unsaved session that is inserted with its first message.
async fn resume_or_start_unsaved_session(
    transcripts: &SqliteTranscriptStore,
    db_path: &Path,
) -> (SessionId, Option<BranchId>, LlmReplay) {
    match transcripts.find_resumable_session().await {
        Ok(Some(cand)) if !pid_is_alive(cand.daemon_pid) => {
            info!(
                "memory: resuming prior session {} (branch={})",
                cand.session_id, cand.current_branch_id.0
            );
            let history = load_resumed_replay(transcripts, cand.current_branch_id).await;
            (cand.session_id, Some(cand.current_branch_id), history)
        }
        Ok(_) => {
            info!(
                "memory: SQLite ready at {} (new session)",
                db_path.display()
            );
            (SessionId::new(), None, LlmReplay::default())
        }
        Err(e) => {
            tracing::warn!("memory: find_resumable_session failed: {e:#}; starting fresh");
            (SessionId::new(), None, LlmReplay::default())
        }
    }
}

/// The resumed branch's history and summary as the LLM replays them; a
/// failed load replays nothing in its place.
async fn load_resumed_replay(transcripts: &SqliteTranscriptStore, branch: BranchId) -> LlmReplay {
    let rows = transcripts
        .load_branch_history(branch)
        .await
        .inspect_err(|e| tracing::warn!("memory: load_branch_history failed for resume ({e:#})"))
        .unwrap_or_default();
    let summary = transcripts
        .load_branch_summary(branch)
        .await
        .inspect_err(|e| tracing::warn!("memory: load_branch_summary failed for resume ({e:#})"))
        .ok()
        .flatten();
    LlmReplay::new(&rows, summary)
}

fn pid_is_alive(pid: u32) -> bool {
    let Some(pid) = i32::try_from(pid).ok().and_then(Pid::from_raw) else {
        return false;
    };
    match rustix::process::test_kill_process(pid) {
        Ok(()) => true,
        Err(Errno::PERM) => true,
        Err(_) => false,
    }
}
