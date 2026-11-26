//! `TurnContext`: what a tool call is told about the turn it runs in.

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use async_trait::async_trait;
use parking_lot::Mutex;
use tokio::sync::{mpsc, watch};
use tracing::warn;

use assistd_ipc::{AgentId, LlmEvent, TaskId};

use crate::policy::{Answer, Approval, ConfirmRouter, ConfirmationRequest};

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

/// What the user said in place of approving an agent's prompts, kept until that agent is
/// told. Clones share the messages.
#[derive(Debug, Clone, Default)]
struct DeclineMessages(Arc<Mutex<Vec<String>>>);

/// The ids an agent's loop gives the sub-agents its tool calls start. Clones share the count.
#[derive(Debug, Clone, Default)]
struct AgentIds(Arc<AtomicU32>);

/// The tool call in flight on its agent's event stream.
#[derive(Debug, Clone)]
struct CallStream {
    events: mpsc::Sender<LlmEvent>,
    call_id: String,
}

/// The turn a tool call belongs to: who answers its prompts, its agent's event stream, the lane
/// it may have to give up, and what its history has read from this machine. The default belongs
/// to no turn: prompts are denied, events go nowhere, it holds no lane, and its history is new.
#[derive(Debug, Clone)]
pub struct TurnContext {
    router: Option<Arc<ConfirmRouter>>,
    decline_messages: DeclineMessages,
    agent_ids: AgentIds,
    call: Option<CallStream>,
    lane: Option<Arc<dyn YieldableLane>>,
    /// `None` in a quarantine, whose reads are not recorded.
    host_reads: Option<HostReads>,
}

impl Default for TurnContext {
    fn default() -> Self {
        Self {
            router: None,
            decline_messages: DeclineMessages::default(),
            agent_ids: AgentIds::default(),
            call: None,
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

    /// The same turn for one agent's loop, which alone is told what the user says when
    /// declining the prompts its tool calls raise, numbers its own sub-agents, and has no
    /// call in flight.
    pub fn for_agent(mut self) -> Self {
        self.decline_messages = DeclineMessages::default();
        self.agent_ids = AgentIds::default();
        self.call = None;
        self
    }

    /// The same turn for tool call `call_id` of the agent that streams on `events`.
    pub fn in_call(&self, events: mpsc::Sender<LlmEvent>, call_id: &str) -> Self {
        Self {
            call: Some(CallStream {
                events,
                call_id: call_id.to_owned(),
            }),
            ..self.clone()
        }
    }

    /// Ask whoever answers the turn's prompts, keeping a message given instead of an
    /// approval for [`take_decline_messages`](Self::take_decline_messages). Denied when
    /// nobody answers.
    pub async fn ask(&self, req: ConfirmationRequest) -> Approval {
        let Some(router) = &self.router else {
            warn!(
                target: "assistd::policy",
                tool = %req.tool,
                pattern = %req.matched_pattern,
                "command denied: no IPC client attached to ask"
            );
            return Approval::Deny;
        };
        match router.ask(req).await {
            Answer::Choice(approval) => approval,
            Answer::Message(message) => {
                self.decline_messages.0.lock().push(message);
                Approval::Deny
            }
        }
    }

    /// What the user said when declining prompts since the last call, oldest first.
    pub fn take_decline_messages(&self) -> Vec<String> {
        std::mem::take(&mut self.decline_messages.0.lock())
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

    /// The saved task the turn is a run of, with no user present; `None` for any other turn.
    pub fn task_run(&self) -> Option<TaskId> {
        self.router.as_ref().and_then(|router| router.task_run())
    }

    /// Whether the turn's prompts are asked out loud.
    pub fn asks_aloud(&self) -> bool {
        self.router
            .as_ref()
            .is_some_and(|router| router.asks_aloud())
    }

    /// The web hosts the user granted to the turn out loud; `None` when its prompts are not
    /// asked out loud.
    pub fn granted_hosts(&self) -> Option<Vec<String>> {
        self.router
            .as_ref()
            .and_then(|router| router.granted_hosts())
    }

    /// How many more web hosts a headless turn may ask about before its user must agree to a
    /// plan naming the rest; `None` once they have, and for any other turn.
    pub fn unplanned_sites_left(&self) -> Option<usize> {
        self.router
            .as_ref()
            .and_then(|router| router.unplanned_sites_left())
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

    /// The id of the next sub-agent a tool call of this agent's loop starts.
    pub fn next_agent_id(&self) -> AgentId {
        AgentId(self.agent_ids.0.fetch_add(1, Ordering::Relaxed) + 1)
    }

    /// Stream `event` as the calling agent's own; dropped outside a tool call and once the
    /// turn has ended.
    pub async fn emit(&self, event: LlmEvent) {
        if let Some(call) = &self.call {
            let _ = call.events.send(event).await;
        }
    }

    /// Stream `event` on the calling agent's stream as sub-agent `agent`'s, which the call
    /// in flight started.
    pub async fn emit_from(&self, agent: AgentId, event: LlmEvent) {
        if let Some(call) = &self.call {
            let wrapped = LlmEvent::FromAgent {
                agent,
                call_id: call.call_id.clone(),
                event: Box::new(event),
            };
            let _ = call.events.send(wrapped).await;
        }
    }
}
