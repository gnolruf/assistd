//! `RuntimeState`: per-process bookkeeping owned by `AppState`.

use assistd_ipc::Event;
use assistd_memory::{BranchId, SessionId};
use parking_lot::Mutex as StdMutex;
use std::sync::Arc;
use tokio::sync::{Mutex, broadcast, oneshot, watch};
use tokio_util::sync::CancellationToken;
use tokio_util::task::TaskTracker;

const EVENTS_BUS_CAPACITY: usize = 256;

/// Active (session, branch) pointer shared by every persistence write site.
pub struct ConversationContext {
    inner: tokio::sync::RwLock<ConversationContextInner>,
    /// Lets holders that cannot await the lock read the session
    /// synchronously.
    session: watch::Sender<Arc<SessionId>>,
}

#[derive(Clone)]
struct ConversationContextInner {
    session_id: Arc<SessionId>,
    branch_id: BranchId,
}

impl ConversationContext {
    pub fn new(session_id: SessionId, branch_id: BranchId) -> Self {
        Self::from_arc(Arc::new(session_id), branch_id)
    }

    pub fn from_arc(session_id: Arc<SessionId>, branch_id: BranchId) -> Self {
        let (session, _) = watch::channel(session_id.clone());
        Self {
            inner: tokio::sync::RwLock::new(ConversationContextInner {
                session_id,
                branch_id,
            }),
            session,
        }
    }

    /// Watch the active session id.
    pub fn session_updates(&self) -> watch::Receiver<Arc<SessionId>> {
        self.session.subscribe()
    }

    pub async fn current(&self) -> (Arc<SessionId>, BranchId) {
        let g = self.inner.read().await;
        (g.session_id.clone(), g.branch_id)
    }

    pub async fn replace(&self, session_id: Arc<SessionId>, branch_id: BranchId) {
        let mut g = self.inner.write().await;
        g.session_id = session_id.clone();
        g.branch_id = branch_id;
        self.session.send_replace(session_id);
    }
}

/// Runtime bookkeeping owned by `AppState`.
pub struct RuntimeState {
    pub conversation_ctx: Arc<ConversationContext>,
    /// Serialises whole agent turns.
    pub(in crate::state) agent_turn_lock: Arc<Mutex<()>>,
    /// Fire-and-forget persistence tasks, drained at daemon shutdown.
    pub(in crate::state) persistence_tracker: TaskTracker,
    /// Presence warmup spawned by PTT-start and joined by PTT-stop.
    pub(in crate::state) warmup_handle:
        Arc<Mutex<Option<tokio::task::JoinHandle<anyhow::Result<()>>>>>,
    /// Cancellation token for the running agent turn.
    pub(in crate::state) current_cancel: Arc<Mutex<Option<CancellationToken>>>,
    /// Completion signal of the most recently queued persistence write.
    /// Each new write swaps it out synchronously and awaits it first, so
    /// `seq` follows emission order rather than scheduler order.
    pub(in crate::state) persist_chain: StdMutex<Option<oneshot::Receiver<()>>>,
    events_bus: broadcast::Sender<Event>,
}

impl RuntimeState {
    pub fn new() -> Self {
        let (events_bus, _) = broadcast::channel(EVENTS_BUS_CAPACITY);
        Self {
            conversation_ctx: Arc::new(ConversationContext::new(SessionId::new(), BranchId(0))),
            agent_turn_lock: Arc::new(Mutex::new(())),
            persistence_tracker: TaskTracker::new(),
            warmup_handle: Arc::new(Mutex::new(None)),
            current_cancel: Arc::new(Mutex::new(None)),
            persist_chain: StdMutex::new(None),
            events_bus,
        }
    }

    pub fn with_conversation_ctx(mut self, ctx: Arc<ConversationContext>) -> Self {
        self.conversation_ctx = ctx;
        self
    }

    pub fn persistence_tracker_handle(&self) -> TaskTracker {
        self.persistence_tracker.clone()
    }

    pub fn events_bus(&self) -> &broadcast::Sender<Event> {
        &self.events_bus
    }

    pub fn subscribe_events(&self) -> broadcast::Receiver<Event> {
        self.events_bus.subscribe()
    }
}

impl Default for RuntimeState {
    fn default() -> Self {
        Self::new()
    }
}
