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
use tokio::sync::watch;
use tokio::task::JoinHandle;
use tracing::info;

pub(super) struct MemorySubsystem {
    pub memory_store: Arc<dyn MemoryStore>,
    pub transcript_store: Arc<dyn TranscriptStore>,
    pub writer_handle: Option<JoinHandle<()>>,
    pub session_id: Arc<SessionId>,
    /// `None` for an unsaved session, inserted with its first message.
    pub branch_id: Option<BranchId>,
    /// History of the resumed branch, to replay into the LLM.
    pub resumed_history: LlmReplay,
    pub sqlite_handle: Option<Arc<SqliteHandle>>,
}

impl MemorySubsystem {
    fn disabled() -> Self {
        Self {
            memory_store: Arc::new(NoMemoryStore),
            transcript_store: Arc::new(NoTranscriptStore),
            writer_handle: None,
            session_id: Arc::new(SessionId::new()),
            branch_id: None,
            resumed_history: LlmReplay::default(),
            sqlite_handle: None,
        }
    }

    /// End the session row, then flip `writer_shutdown` and drain the
    /// writer task. Must run after every other writer client has stopped.
    pub(super) async fn shutdown(self, writer_shutdown: &watch::Sender<bool>) {
        if let Err(e) = self.transcript_store.end_session(&self.session_id).await {
            tracing::warn!("memory: end_session failed at shutdown: {e:#}");
        }
        writer_shutdown.send_replace(true);
        if let Some(h) = self.writer_handle {
            let _ = h.await;
        }
    }
}

/// Degrades to a no-op subsystem when disabled or when the database
/// cannot be opened.
pub(super) async fn init(config: &Config, shutdown_tx: &watch::Sender<bool>) -> MemorySubsystem {
    if !config.memory.enabled {
        info!("memory: disabled in config (memory.enabled = false)");
        return MemorySubsystem::disabled();
    }

    let db_path = PathBuf::from(&config.memory.db_path);
    let (handle, writer_handle) = match SqliteHandle::open(&db_path, shutdown_tx.subscribe()).await
    {
        Ok(pair) => pair,
        Err(e) => {
            tracing::warn!(
                "memory: failed to open {} ({e:#}); persistence disabled this run",
                db_path.display()
            );
            return MemorySubsystem::disabled();
        }
    };

    let handle = Arc::new(handle);
    let conv_store = Arc::new(SqliteTranscriptStore::new(handle.clone()));
    let mem_store = Arc::new(SqliteMemoryStore::new(handle.clone()));
    let (session_id, branch_id, resumed_history) =
        resume_or_start_unsaved_session(conv_store.as_ref(), &db_path).await;

    MemorySubsystem {
        memory_store: mem_store,
        transcript_store: conv_store,
        writer_handle: Some(writer_handle),
        session_id,
        branch_id,
        resumed_history,
        sqlite_handle: Some(handle),
    }
}

/// Resume the most recent session whose daemon is dead, else start an
/// unsaved session that is inserted with its first message.
async fn resume_or_start_unsaved_session(
    conv_store: &SqliteTranscriptStore,
    db_path: &Path,
) -> (Arc<SessionId>, Option<BranchId>, LlmReplay) {
    match conv_store.find_resumable_session().await {
        Ok(Some(cand)) if !pid_is_alive(cand.daemon_pid) => {
            info!(
                "memory: resuming prior session {} (branch={})",
                cand.session_id, cand.current_branch_id.0
            );
            let history = load_resumed_replay(conv_store, cand.current_branch_id).await;
            (
                Arc::new(cand.session_id),
                Some(cand.current_branch_id),
                history,
            )
        }
        Ok(_) => {
            info!(
                "memory: SQLite ready at {} (new session)",
                db_path.display()
            );
            (Arc::new(SessionId::new()), None, LlmReplay::default())
        }
        Err(e) => {
            tracing::warn!("memory: find_resumable_session failed: {e:#}; starting fresh");
            (Arc::new(SessionId::new()), None, LlmReplay::default())
        }
    }
}

/// The resumed branch's history and summary as the LLM replays them; a
/// failed load replays nothing in its place.
async fn load_resumed_replay(conv_store: &SqliteTranscriptStore, branch: BranchId) -> LlmReplay {
    let rows = conv_store
        .load_branch_history(branch)
        .await
        .inspect_err(|e| tracing::warn!("memory: load_branch_history failed for resume ({e:#})"))
        .unwrap_or_default();
    let summary = conv_store
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
