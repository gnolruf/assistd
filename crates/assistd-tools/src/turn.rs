//! `TurnContext`: what a tool call is told about the turn it runs in.

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use async_trait::async_trait;
use tokio::sync::watch;

use assistd_ipc::RelayEvent;

use crate::policy::ConfirmRouter;

/// Is shown what a relay running inside a tool call does.
#[async_trait]
pub trait RelayObserver: fmt::Debug + Send + Sync + 'static {
    /// Show `event`; an observer whose turn has ended drops it.
    async fn report(&self, event: RelayEvent);
}

/// The model lane a task run's turn holds, which the user's interaction may ask for.
#[async_trait]
pub trait YieldableLane: fmt::Debug + Send + Sync + 'static {
    /// Wait until the user's interaction wants the lane. Cancel-safe.
    async fn yield_requested(&self);

    /// Give the lane up if it is wanted, then wait to hold one again; `false` when the
    /// run was cancelled first.
    async fn give_way(&self) -> bool;
}

/// Whether tool calls have read this machine into a history, after which anything the model
/// sends out could carry what they read. Clones share the answer.
#[derive(Debug, Clone, Default)]
pub struct HostReads(Arc<AtomicBool>);

impl HostReads {
    /// A history already known to hold what tool calls read from this machine, or to hold none.
    pub fn known(seen: bool) -> Self {
        Self(Arc::new(AtomicBool::new(seen)))
    }

    /// Whether a tool call has read this machine into the history.
    pub fn seen(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }

    /// Note that a tool call read this machine into the history.
    pub fn record(&self) {
        self.0.store(true, Ordering::Release);
    }
}

/// The turn a tool call belongs to: who answers its prompts, who watches its relays, the lane
/// it may have to give up, and what its history has read from this machine. The default belongs
/// to no turn: prompts are denied, nobody watches, it holds no lane, and its history is new.
#[derive(Debug, Clone)]
pub struct TurnContext {
    router: Option<Arc<ConfirmRouter>>,
    relay_observer: Option<Arc<dyn RelayObserver>>,
    lane: Option<Arc<dyn YieldableLane>>,
    /// `None` in a quarantine, whose reads are not recorded.
    host_reads: Option<HostReads>,
}

impl Default for TurnContext {
    fn default() -> Self {
        Self {
            router: None,
            relay_observer: None,
            lane: None,
            host_reads: Some(HostReads::default()),
        }
    }
}

impl TurnContext {
    /// A turn whose prompts `router` answers.
    pub fn answered_by(router: Arc<ConfirmRouter>) -> Self {
        Self {
            router: Some(router),
            ..Self::default()
        }
    }

    /// Show `observer` the relays this turn's tool calls run.
    pub fn observed_by(mut self, observer: Arc<dyn RelayObserver>) -> Self {
        self.relay_observer = Some(observer);
        self
    }

    /// Run on `lane`, giving it up when the user's interaction asks.
    pub fn on_lane(mut self, lane: Arc<dyn YieldableLane>) -> Self {
        self.lane = Some(lane);
        self
    }

    /// Record what the turn's tool calls read from this machine in `host_reads`, which every
    /// turn over the same history shares.
    pub fn reading_into(mut self, host_reads: HostReads) -> Self {
        self.host_reads = Some(host_reads);
        self
    }

    /// The same turn for a sub-agent in a quarantine, which reaches files or the web but
    /// never both, so its reads are not recorded.
    pub fn quarantined(&self) -> Self {
        Self {
            host_reads: None,
            ..self.clone()
        }
    }

    /// Note that a tool call read this machine into the turn's history.
    pub fn record_host_read(&self) {
        if let Some(host_reads) = &self.host_reads {
            host_reads.record();
        }
    }

    /// Whether a tool call has read this machine into the turn's history.
    pub fn has_read_host(&self) -> bool {
        self.host_reads.as_ref().is_some_and(HostReads::seen)
    }

    /// Who answers the turn's prompts; `None` when nobody can.
    pub fn router(&self) -> Option<&Arc<ConfirmRouter>> {
        self.router.as_ref()
    }

    /// The lane the turn gives up on request; `None` when it never has to.
    pub fn lane(&self) -> Option<&Arc<dyn YieldableLane>> {
        self.lane.as_ref()
    }

    /// Whether the turn is planning.
    pub fn planning(&self) -> bool {
        self.router.as_ref().is_some_and(|router| router.planning())
    }

    /// How many asks the turn's router has refused while planning.
    pub fn refused_while_planning(&self) -> usize {
        self.router
            .as_ref()
            .map_or(0, |router| router.refused_while_planning())
    }

    /// Watch whether one of the turn's prompts waits for a later answer; `None` when none can.
    pub fn parked(&self) -> Option<watch::Receiver<bool>> {
        self.router.as_ref().map(|router| router.parked())
    }

    /// Show the turn's relay observer `event`, if it has one.
    pub async fn report_relay(&self, event: RelayEvent) {
        if let Some(observer) = &self.relay_observer {
            observer.report(event).await;
        }
    }
}
