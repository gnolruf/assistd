//! `RuntimeState` groups the per-process bookkeeping that AppState
//! needs but that isn't a "subsystem backend" or "memory stack": the
//! active (session, branch) pointer, the agent-turn lock, the
//! persistence task tracker, the warmup-join handle, and the
//! process-wide events broadcast bus that feeds passive
//! `Request::Subscribe` connections.
//!
//! `ConversationContext` also lives here — it's the runtime-mutable
//! pointer that `/switch` and `/fork` swap.

use assistd_ipc::Event;
use assistd_memory::{BranchId, SessionId};
use std::sync::Arc;
use tokio::sync::{Mutex, broadcast};
use tokio_util::task::TaskTracker;

/// Capacity of the events broadcast bus. Slow subscribers that fall
/// more than this many events behind receive a `RecvError::Lagged`
/// signal and resume from the latest event; the handler logs and
/// keeps the connection open.
const EVENTS_BUS_CAPACITY: usize = 256;

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
    /// Process-wide passive event bus. Connections tee broadcast-
    /// eligible outbound events into this sender so that
    /// `Request::Subscribe` connections can observe each other's
    /// activity; the per-turn delta coalescer also writes
    /// `Event::LastDelta` summaries here directly. External code
    /// obtains a sender / receiver via [`Self::events_bus`] and
    /// [`Self::subscribe_events`].
    events_bus: broadcast::Sender<Event>,
}

impl RuntimeState {
    /// Build a fresh runtime state with default locks and a
    /// brand-new auto-generated conversation context.
    pub fn new() -> Self {
        let (events_bus, _) = broadcast::channel(EVENTS_BUS_CAPACITY);
        Self {
            conversation_ctx: Arc::new(ConversationContext::new(SessionId::new(), BranchId(0))),
            agent_turn_lock: Arc::new(Mutex::new(())),
            persistence_tracker: TaskTracker::new(),
            warmup_handle: Arc::new(Mutex::new(None)),
            events_bus,
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

    /// Borrow the events broadcast sender. Used by the socket fan-
    /// out tee and by the per-turn delta coalescer to publish
    /// events to passive subscribers.
    pub fn events_bus(&self) -> &broadcast::Sender<Event> {
        &self.events_bus
    }

    /// Open a fresh receiver on the events broadcast bus. Used by
    /// the Subscribe handler; equivalent to
    /// `self.events_bus().subscribe()`.
    pub fn subscribe_events(&self) -> broadcast::Receiver<Event> {
        self.events_bus.subscribe()
    }
}

impl Default for RuntimeState {
    fn default() -> Self {
        Self::new()
    }
}
