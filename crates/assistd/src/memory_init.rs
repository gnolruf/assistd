//! Memory subsystem wiring for the daemon: open SQLite, then resume a
//! session whose daemon has died or begin a fresh one.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use assistd_core::Config;
use assistd_memory::{
    BranchId, ConversationStore, HistoryRow, MemoryStore, NoConversationStore, NoMemoryStore,
    SessionId, SqliteConversationStore, SqliteHandle, SqliteMemoryStore,
};
use rustix::io::Errno;
use rustix::process::Pid;
use tokio::sync::watch;
use tokio::task::JoinHandle;
use tracing::info;

pub struct MemorySubsystem {
    pub memory_store: Arc<dyn MemoryStore>,
    pub conversation_store: Arc<dyn ConversationStore>,
    pub writer_handle: Option<JoinHandle<()>>,
    pub session_id: Arc<SessionId>,
    pub branch_id: BranchId,
    /// History of the resumed branch, to replay into the LLM.
    pub resumed_history: Vec<HistoryRow>,
    pub sqlite_handle: Option<Arc<SqliteHandle>>,
}

impl MemorySubsystem {
    fn disabled() -> Self {
        Self {
            memory_store: Arc::new(NoMemoryStore),
            conversation_store: Arc::new(NoConversationStore),
            writer_handle: None,
            session_id: Arc::new(SessionId::new()),
            branch_id: BranchId(0),
            resumed_history: Vec::new(),
            sqlite_handle: None,
        }
    }

    /// End the session row and drain the writer task. Must run before
    /// presence shutdown so writes accepted earlier still land.
    pub async fn shutdown(self) {
        if let Err(e) = self.conversation_store.end_session(&self.session_id).await {
            tracing::warn!("memory: end_session failed at shutdown: {e:#}");
        }
        if let Some(h) = self.writer_handle {
            let _ = h.await;
        }
    }
}

/// Degrades to a no-op subsystem when disabled or when the database
/// cannot be opened.
pub async fn init(config: &Config, shutdown_tx: &watch::Sender<bool>) -> MemorySubsystem {
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
    let conv_store = Arc::new(SqliteConversationStore::new(handle.clone()));
    let mem_store = Arc::new(SqliteMemoryStore::new(handle.clone()));
    let (session_id, branch_id, resumed_history) =
        resume_or_begin_session(conv_store.as_ref(), &db_path).await;

    MemorySubsystem {
        memory_store: mem_store,
        conversation_store: conv_store,
        writer_handle: Some(writer_handle),
        session_id,
        branch_id,
        resumed_history,
        sqlite_handle: Some(handle),
    }
}

/// Resume the most recent session whose daemon is dead, else begin a
/// fresh one. A failed begin yields an unpersisted placeholder session.
async fn resume_or_begin_session(
    conv_store: &SqliteConversationStore,
    db_path: &Path,
) -> (Arc<SessionId>, BranchId, Vec<HistoryRow>) {
    match conv_store.find_resumable_session().await {
        Ok(Some(cand)) if !pid_is_alive(cand.daemon_pid) => {
            info!(
                "memory: resuming prior session {} (branch={})",
                cand.session_id, cand.current_branch_id.0
            );
            let history = match conv_store.load_branch_history(cand.current_branch_id).await {
                Ok(rows) => rows,
                Err(e) => {
                    tracing::warn!("memory: load_branch_history failed for resume ({e:#})");
                    Vec::new()
                }
            };
            (Arc::new(cand.session_id), cand.current_branch_id, history)
        }
        Ok(_) => match conv_store
            .begin_session_with_main_branch(std::process::id())
            .await
        {
            Ok((s, b)) => {
                info!(
                    "memory: SQLite ready at {} (session={}, branch={})",
                    db_path.display(),
                    s,
                    b.0
                );
                (Arc::new(s), b, Vec::new())
            }
            Err(e) => {
                tracing::warn!(
                    "memory: begin_session_with_main_branch failed: {e:#}; \
                     continuing without session row"
                );
                (Arc::new(SessionId::new()), BranchId(0), Vec::new())
            }
        },
        Err(e) => {
            tracing::warn!("memory: find_resumable_session failed: {e:#}; starting fresh");
            match conv_store
                .begin_session_with_main_branch(std::process::id())
                .await
            {
                Ok((s, b)) => (Arc::new(s), b, Vec::new()),
                Err(e) => {
                    tracing::warn!("memory: begin_session_with_main_branch failed: {e:#}");
                    (Arc::new(SessionId::new()), BranchId(0), Vec::new())
                }
            }
        }
    }
}

fn pid_is_alive(pid: u32) -> bool {
    let Some(pid) = Pid::from_raw(pid as i32) else {
        return false;
    };
    match rustix::process::test_kill_process(pid) {
        Ok(()) => true,
        Err(Errno::PERM) => true,
        Err(_) => false,
    }
}
