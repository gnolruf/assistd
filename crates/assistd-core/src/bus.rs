//! `EventBus`, the broadcast bus every subscriber hears the daemon's events on, and
//! `EventSink`, which publishes a request's events there as they are produced.

use std::future::pending;
use std::sync::Arc;

use parking_lot::Mutex;
use thiserror::Error;
use tokio::sync::broadcast::error::RecvError;
use tokio::sync::{broadcast, mpsc};

use assistd_ipc::{Event, EventBody, EventKind, SubscribeFilter, TaskRunId};

const EVENTS_BUS_CAPACITY: usize = 256;

/// The daemon's events bus; clones publish to the same subscribers.
#[derive(Debug, Clone)]
pub struct EventBus {
    events: broadcast::Sender<Event>,
    /// Filters of the live [`BusSubscription`]s.
    interest: Arc<Mutex<Vec<SubscribeFilter>>>,
}

impl EventBus {
    /// A bus with no subscribers.
    pub fn new() -> Self {
        let (events, _) = broadcast::channel(EVENTS_BUS_CAPACITY);
        Self {
            events,
            interest: Arc::default(),
        }
    }

    /// The broadcast sender [`Self::publish`] sends on.
    pub fn sender(&self) -> &broadcast::Sender<Event> {
        &self.events
    }

    /// Attach a receiver that yields only events matching `filter`. The filter is
    /// registered first, so no matching event published after this returns is skipped.
    pub fn subscribe(&self, filter: SubscribeFilter) -> BusSubscription {
        self.interest.lock().push(filter.clone());
        BusSubscription {
            rx: self.events.subscribe(),
            filter,
            interest: Arc::clone(&self.interest),
        }
    }

    /// Whether any attached [`BusSubscription`] wants events of `kind`.
    pub fn wants(&self, kind: EventKind) -> bool {
        self.interest.lock().iter().any(|f| f.matches(kind))
    }

    /// Publish a copy of `event`, an event of task run `run`, wrapped in
    /// [`EventBody::TaskRun`], if any subscriber wants task-run events.
    fn publish_task_run(&self, run: TaskRunId, event: &Event) {
        if self.wants(EventKind::TaskRun) {
            let _ = self.events.send(Event::new(
                event.id.clone(),
                EventBody::TaskRun {
                    run,
                    event: Box::new(bus_copy(event)),
                },
            ));
        }
    }

    /// Publish a copy of `event` if any subscriber wants its kind. A `ToolResult` copy,
    /// wrapped in a `TaskRun` or not, omits the result's `attachments`, which no
    /// subscriber reads.
    pub fn publish(&self, event: &Event) {
        if event.kind().is_some_and(|kind| self.wants(kind)) {
            let _ = self.events.send(bus_copy(event));
        }
    }
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new()
    }
}

/// A receiver on the events bus, from [`EventBus::subscribe`].
/// Its filter counts toward [`EventBus::wants`] until dropped.
#[derive(Debug)]
pub struct BusSubscription {
    rx: broadcast::Receiver<Event>,
    filter: SubscribeFilter,
    interest: Arc<Mutex<Vec<SubscribeFilter>>>,
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
        if let Some(idx) = filters.iter().position(|f| *f == self.filter) {
            filters.swap_remove(idx);
        }
    }
}

/// The client an [`EventSink`] sends to has gone away.
#[derive(Debug, Error)]
#[error("the client has gone away")]
pub struct ClientGone;

/// Where the events of one request or turn go: onto the bus as each is produced, and to
/// the client that asked, when there is one. A slow client never delays the bus.
#[derive(Debug, Clone)]
pub struct EventSink {
    client: Option<mpsc::Sender<Event>>,
    bus: Option<EventBus>,
    /// The task run whose events these are, published wrapped in [`EventBody::TaskRun`].
    run: Option<TaskRunId>,
}

impl EventSink {
    /// Events published on `bus` and sent to `client`.
    pub fn to_client(bus: EventBus, client: mpsc::Sender<Event>) -> Self {
        Self {
            client: Some(client),
            bus: Some(bus),
            run: None,
        }
    }

    /// Events sent to `client` alone, for a client that is itself reading the bus.
    pub fn unpublished(client: mpsc::Sender<Event>) -> Self {
        Self {
            client: Some(client),
            bus: None,
            run: None,
        }
    }

    /// Events nobody asked for, published on `bus`.
    pub fn to_bus(bus: EventBus) -> Self {
        Self {
            client: None,
            bus: Some(bus),
            run: None,
        }
    }

    /// The events of task run `run`, published on `bus` as `TaskRun`.
    pub fn of_task_run(bus: EventBus, run: TaskRunId) -> Self {
        Self {
            client: None,
            bus: Some(bus),
            run: Some(run),
        }
    }

    /// Publish `event`, then hand it to the client. Errors when the client has gone,
    /// by which time the event is already published.
    pub async fn send(&self, event: Event) -> Result<(), ClientGone> {
        self.publish(&event);
        match &self.client {
            Some(client) => client.send(event).await.map_err(|_| ClientGone),
            None => Ok(()),
        }
    }

    /// Publish `event` without sending it to the client.
    pub fn publish(&self, event: &Event) {
        let Some(bus) = &self.bus else {
            return;
        };
        match self.run {
            None => bus.publish(event),
            Some(run) => bus.publish_task_run(run, event),
        }
    }

    /// Whether [`Self::publish`] would reach a subscriber with an event of `kind`.
    pub fn publishes(&self, kind: EventKind) -> bool {
        let kind = if self.run.is_some() {
            EventKind::TaskRun
        } else {
            kind
        };
        self.bus.as_ref().is_some_and(|bus| bus.wants(kind))
    }

    /// Wait until the client goes away; never resolves without a client.
    pub async fn closed(&self) {
        match &self.client {
            Some(client) => client.closed().await,
            None => pending().await,
        }
    }
}

fn bus_copy(event: &Event) -> Event {
    let body = match &event.body {
        EventBody::ToolResult { name, result } => EventBody::ToolResult {
            name: name.clone(),
            result: result.without_attachments(),
        },
        EventBody::TaskRun { run, event } => EventBody::TaskRun {
            run: *run,
            event: Box::new(bus_copy(event)),
        },
        other => other.clone(),
    };
    Event {
        id: event.id.clone(),
        body,
    }
}

#[cfg(test)]
mod tests {
    use std::pin::pin;

    use futures_util::FutureExt;

    use assistd_ipc::{ImageMime, ToolAttachment, ToolReport};

    use super::*;

    fn only(kind: EventKind) -> SubscribeFilter {
        SubscribeFilter { kinds: vec![kind] }
    }

    #[test]
    fn bus_interest_follows_live_subscriptions() {
        let bus = EventBus::new();
        assert!(!bus.wants(EventKind::LastDelta));

        let titles = bus.subscribe(only(EventKind::SessionTitle));
        assert!(!bus.wants(EventKind::LastDelta));
        assert!(bus.wants(EventKind::SessionTitle));

        let everything = bus.subscribe(SubscribeFilter::default());
        assert!(bus.wants(EventKind::LastDelta));

        drop(everything);
        assert!(!bus.wants(EventKind::LastDelta));
        drop(titles);
        assert!(!bus.wants(EventKind::SessionTitle));
    }

    #[tokio::test]
    async fn a_client_that_is_not_reading_does_not_hold_up_the_bus() {
        let bus = EventBus::new();
        let mut sub = bus.subscribe(only(EventKind::Delta));
        let (client, _unread) = mpsc::channel(1);
        let events = EventSink::to_client(bus.clone(), client);
        let delta = |text: &str| Event::new("q", EventBody::Delta { text: text.into() });

        events.send(delta("first")).await.unwrap();
        let mut waiting_on_client = pin!(events.send(delta("second")));
        assert!(waiting_on_client.as_mut().now_or_never().is_none());

        assert_eq!(sub.recv().await.unwrap(), delta("first"));
        assert_eq!(sub.recv().await.unwrap(), delta("second"));
    }

    #[tokio::test]
    async fn published_tool_results_leave_attachments_to_the_requester() {
        let bus = EventBus::new();
        let mut sub = bus.subscribe(only(EventKind::ToolResult));
        let shown = ToolReport {
            output: "[image]".into(),
            ..ToolReport::default()
        };
        bus.publish(&Event::new(
            "q",
            EventBody::Delta {
                text: "unwanted".into(),
            },
        ));
        bus.publish(&Event::new(
            "q",
            EventBody::ToolResult {
                name: "run".into(),
                result: ToolReport {
                    attachments: vec![ToolAttachment::Image {
                        mime: ImageMime::Png,
                        data: "AAAA".into(),
                    }],
                    ..shown.clone()
                },
            },
        ));
        assert_eq!(
            sub.recv().await.unwrap(),
            Event::new(
                "q",
                EventBody::ToolResult {
                    name: "run".into(),
                    result: shown,
                }
            )
        );
    }
}
