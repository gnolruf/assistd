//! `RuntimeState`: per-process bookkeeping owned by `AppState`.

use assistd_ipc::{Event, EventKind, SubscribeFilter};
use assistd_memory::{BranchId, SessionId};
use parking_lot::Mutex as StdMutex;
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::broadcast::error::RecvError;
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
    pub(in crate::state) warmup_handle: Arc<Mutex<Option<tokio::task::JoinHandle<()>>>>,
    /// Cancellation token for the running agent turn.
    pub(in crate::state) current_cancel: Arc<Mutex<Option<CancellationToken>>>,
    /// Completion signal of the most recently queued persistence write.
    /// Each new write swaps it out synchronously and awaits it first, so
    /// `seq` follows emission order rather than scheduler order.
    pub(in crate::state) persist_chain: StdMutex<Option<oneshot::Receiver<()>>>,
    events_bus: broadcast::Sender<Event>,
    /// Filters of the live [`BusSubscription`]s.
    bus_interest: Arc<StdMutex<Vec<SubscribeFilter>>>,
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
            bus_interest: Arc::default(),
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

    /// Attach a receiver to the events bus that yields only events
    /// matching `filter`.
    pub fn subscribe_events(&self, filter: SubscribeFilter) -> BusSubscription {
        // Registered before subscribing, so an event published once the
        // receiver exists is never skipped for lack of interest.
        self.bus_interest.lock().push(filter.clone());
        BusSubscription {
            rx: self.events_bus.subscribe(),
            filter,
            interest: Arc::clone(&self.bus_interest),
        }
    }

    /// Whether any attached [`BusSubscription`] wants events of `kind`.
    /// Lets a publisher skip building an event nobody would receive.
    pub fn bus_wants(&self, kind: EventKind) -> bool {
        self.bus_interest.lock().iter().any(|f| f.matches(kind))
    }

    /// Publish a copy of `event` on the events bus if any subscriber
    /// wants its kind. A `ToolResult` copy leaves out the result's
    /// `attachments`: no subscriber reads the base64 images, and every
    /// receiver would otherwise clone them.
    pub fn publish(&self, event: &Event) {
        if event.kind().is_some_and(|kind| self.bus_wants(kind)) {
            let _ = self.events_bus.send(bus_copy(event));
        }
    }
}

fn bus_copy(event: &Event) -> Event {
    match event {
        Event::ToolResult { id, name, result } => Event::ToolResult {
            id: id.clone(),
            name: name.clone(),
            result: match result.as_object() {
                Some(fields) => Value::Object(
                    fields
                        .iter()
                        .filter(|(key, _)| key.as_str() != "attachments")
                        .map(|(key, value)| (key.clone(), value.clone()))
                        .collect(),
                ),
                None => result.clone(),
            },
        },
        other => other.clone(),
    }
}

/// A receiver on the events bus, from [`RuntimeState::subscribe_events`].
/// Its filter counts toward [`RuntimeState::bus_wants`] until dropped.
pub struct BusSubscription {
    rx: broadcast::Receiver<Event>,
    filter: SubscribeFilter,
    interest: Arc<StdMutex<Vec<SubscribeFilter>>>,
}

impl BusSubscription {
    /// The next bus event that matches the filter. Cancel-safe.
    pub async fn recv(&mut self) -> Result<Event, RecvError> {
        loop {
            let event = self.rx.recv().await?;
            if event.kind().is_some_and(|kind| self.filter.matches(kind)) {
                return Ok(event);
            }
        }
    }
}

impl Drop for BusSubscription {
    fn drop(&mut self) {
        let mut filters = self.interest.lock();
        if let Some(i) = filters.iter().position(|f| *f == self.filter) {
            filters.swap_remove(i);
        }
    }
}

impl Default for RuntimeState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn only(kind: EventKind) -> SubscribeFilter {
        SubscribeFilter { kinds: vec![kind] }
    }

    #[test]
    fn bus_interest_follows_live_subscriptions() {
        let runtime = RuntimeState::new();
        assert!(!runtime.bus_wants(EventKind::LastDelta));

        let titles = runtime.subscribe_events(only(EventKind::SessionTitle));
        assert!(!runtime.bus_wants(EventKind::LastDelta));
        assert!(runtime.bus_wants(EventKind::SessionTitle));

        let everything = runtime.subscribe_events(SubscribeFilter::default());
        assert!(runtime.bus_wants(EventKind::LastDelta));

        drop(everything);
        assert!(!runtime.bus_wants(EventKind::LastDelta));
        drop(titles);
        assert!(!runtime.bus_wants(EventKind::SessionTitle));
    }

    #[tokio::test]
    async fn published_tool_results_leave_attachments_to_the_requester() {
        let runtime = RuntimeState::new();
        let mut sub = runtime.subscribe_events(only(EventKind::ToolResult));
        runtime.publish(&Event::Delta {
            id: "q".into(),
            text: "unwanted".into(),
        });
        runtime.publish(&Event::ToolResult {
            id: "q".into(),
            name: "run".into(),
            result: json!({
                "output": "[image]",
                "exit_code": 0,
                "attachments": [{"type": "image", "mime": "image/png", "data": "AAAA"}],
            }),
        });
        assert_eq!(
            sub.recv().await.unwrap(),
            Event::ToolResult {
                id: "q".into(),
                name: "run".into(),
                result: json!({"output": "[image]", "exit_code": 0}),
            }
        );
    }
}
