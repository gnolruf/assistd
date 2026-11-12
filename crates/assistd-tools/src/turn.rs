//! `TurnContext`: what a tool call is told about the turn it runs in.

use std::fmt;
use std::sync::Arc;

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

/// The model lane a task run's turn holds, which the user's conversation may ask for.
#[async_trait]
pub trait YieldableLane: fmt::Debug + Send + Sync + 'static {
    /// Wait until the user's conversation wants the lane. Cancel-safe.
    async fn yield_requested(&self);

    /// Give the lane up if it is wanted, then wait to hold one again; `false` when the
    /// run was cancelled first.
    async fn give_way(&self) -> bool;
}

/// The turn a tool call belongs to: who answers its prompts, who watches its relays, and
/// the lane it may have to give up. The default belongs to no turn: its prompts are denied,
/// nobody watches, and it holds no lane.
#[derive(Debug, Clone, Default)]
pub struct TurnContext {
    router: Option<Arc<ConfirmRouter>>,
    relay_observer: Option<Arc<dyn RelayObserver>>,
    lane: Option<Arc<dyn YieldableLane>>,
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

    /// Run on `lane`, giving it up when the user's conversation asks.
    pub fn on_lane(mut self, lane: Arc<dyn YieldableLane>) -> Self {
        self.lane = Some(lane);
        self
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
