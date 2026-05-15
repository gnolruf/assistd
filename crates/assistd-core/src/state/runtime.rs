//! `RuntimeState` groups the per-process bookkeeping that AppState
//! needs but that isn't a "subsystem backend" or "memory stack": the
//! active (session, branch) pointer, the agent-turn lock, the
//! persistence task tracker, and the warmup-join handle.
//!
//! `ConversationContext` also lives here — it's the runtime-mutable
//! pointer that `/switch` and `/fork` swap.

use assistd_memory::{BranchId, SessionId};
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio_util::task::TaskTracker;

/// Active (session, branch) pointer shared by every persistence write
/// site. Held under a `RwLock` because the persistence path takes a
/// read lock on every message append (cheap, lock-free in practice
/// under WAL) while `/switch` takes a write lock once per command. We
/// hold `Arc<SessionId>` inside so reads return a cheap `Arc::clone`
/// without copying the underlying string.
pub struct ConversationContext {
    inner: tokio::sync::RwLock<ConversationContextInner>,
}

#[derive(Clone)]
struct ConversationContextInner {
    session_id: Arc<SessionId>,
    branch_id: BranchId,
}

impl ConversationContext {
    /// Construct a new context, wrapping `session_id` in an `Arc`.
    pub fn new(session_id: SessionId, branch_id: BranchId) -> Self {
        Self {
            inner: tokio::sync::RwLock::new(ConversationContextInner {
                session_id: Arc::new(session_id),
                branch_id,
            }),
        }
    }

    /// Construct a new context from an already-`Arc`-wrapped session id.
    pub fn from_arc(session_id: Arc<SessionId>, branch_id: BranchId) -> Self {
        Self {
            inner: tokio::sync::RwLock::new(ConversationContextInner {
                session_id,
                branch_id,
            }),
        }
    }

    /// Read snapshot under a brief shared lock. The `Arc<SessionId>`
    /// clone is reference-count only.
    pub async fn current(&self) -> (Arc<SessionId>, BranchId) {
        let g = self.inner.read().await;
        (g.session_id.clone(), g.branch_id)
    }

    /// Atomically swap to a different (session, branch). Used by
    /// `/switch` and by daemon-startup resume.
    pub async fn replace(&self, session_id: Arc<SessionId>, branch_id: BranchId) {
        let mut g = self.inner.write().await;
        g.session_id = session_id;
        g.branch_id = branch_id;
    }
}

/// Runtime bookkeeping owned by `AppState`. Added in Step 2 but not
/// yet wired into the struct; Step 3 flips the field set.
pub struct RuntimeState {
    pub conversation_ctx: Arc<ConversationContext>,
    /// Serializes entire agent turns. Concurrent queries each grab this
    /// lock before running `Agent::run_turn`, so one query's tool-call /
    /// tool-result cycle never interleaves with another's.
    /// Not `pub`: kept within `state/` so handlers can reach it but
    /// external code cannot.
    pub(in crate::state) agent_turn_lock: Arc<Mutex<()>>,
    /// Tracks fire-and-forget persistence tasks so daemon shutdown can
    /// drain them before dropping the writer-task channel. Not `pub`
    /// for the same reason as `agent_turn_lock` — handlers reach it
    /// internally; outside code uses
    /// [`Self::persistence_tracker_handle`].
    pub(in crate::state) persistence_tracker: TaskTracker,
    /// Handle to the `presence.ensure_active()` task spawned at
    /// PTT-start. PTT-stop joins it in parallel with Whisper
    /// transcription. Not `pub` so external code can't accidentally
    /// race PTT-start by stuffing in a foreign join handle.
    pub(in crate::state) warmup_handle:
        Arc<Mutex<Option<tokio::task::JoinHandle<anyhow::Result<()>>>>>,
}

impl RuntimeState {
    /// Build a fresh runtime state with default locks and a
    /// brand-new auto-generated conversation context.
    pub fn new() -> Self {
        Self {
            conversation_ctx: Arc::new(ConversationContext::new(SessionId::new(), BranchId(0))),
            agent_turn_lock: Arc::new(Mutex::new(())),
            persistence_tracker: TaskTracker::new(),
            warmup_handle: Arc::new(Mutex::new(None)),
        }
    }

    pub fn with_conversation_ctx(mut self, ctx: Arc<ConversationContext>) -> Self {
        self.conversation_ctx = ctx;
        self
    }

    /// Return a clone of the persistence tracker. Daemon shutdown
    /// owns one and uses it to drain in-flight writes before the
    /// writer-task channel sender drops.
    pub fn persistence_tracker_handle(&self) -> TaskTracker {
        self.persistence_tracker.clone()
    }
}

impl Default for RuntimeState {
    fn default() -> Self {
        Self::new()
    }
}
