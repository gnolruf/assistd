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
    /// Broadcasts the active session so holders that cannot await the
    /// lock — the `reminisce` tool, which excludes the conversation
    /// already in context — can read it synchronously.
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

    /// Watch the active session id. The current value is readable
    /// synchronously via `borrow()`, and updates land on every
    /// `/switch` or `/new`.
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
    /// Serializes entire agent turns so one query's tool-call /
    /// tool-result cycle never interleaves with another's.
    pub(in crate::state) agent_turn_lock: Arc<Mutex<()>>,
    /// Fire-and-forget persistence tasks, drained at daemon shutdown
    /// before the writer-task channel sender drops.
    pub(in crate::state) persistence_tracker: TaskTracker,
    /// Handle to the `presence.ensure_active()` task PTT-start spawns;
    /// PTT-stop joins it alongside Whisper transcription.
    pub(in crate::state) warmup_handle:
        Arc<Mutex<Option<tokio::task::JoinHandle<anyhow::Result<()>>>>>,
    /// Cancellation token for the currently-running agent turn, taken
    /// by `Request::InterruptTurn` to abort the turn on its next await.
    pub(in crate::state) current_cancel: Arc<Mutex<Option<CancellationToken>>>,
    /// Tail of the persistence write chain: the completion signal of the
    /// most recently queued message. Each new write takes it (a
    /// synchronous swap, so the order is fixed at call time rather than
    /// by the scheduler) and awaits it before touching the store, which
    /// is what keeps `seq` in the order the daemon emitted the messages.
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
