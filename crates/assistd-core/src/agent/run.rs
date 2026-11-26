//! The loop: step the model, dispatch the tool calls it requests, feed the results back,
//! and repeat until it answers.

use std::pin::pin;
use std::sync::Arc;

use parking_lot::Mutex;
use serde_json::Value;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::{debug, instrument, warn};

use assistd_ipc::StatusKind;
use assistd_llm::{LlmError, LlmEvent, StepOutcome, ToolCall};
use assistd_tools::{ToolRegistry, TurnContext};

use super::definition::AgentLimits;
use super::outcome::{AgentEnd, AgentOutcome, RunRecord};
use super::recovery::TRUNCATION_NOTE;
use super::{Agent, AgentTask};
use crate::recovery::{Component, StatusSeverity};
use crate::recovery_event;

/// Ends the reply kept for a turn cut off before the model finished.
const INTERRUPTED_MARKER: &str = "[interrupted]";

/// Events one model step may have waiting for the turn.
const STEP_EVENT_BUFFER: usize = 32;

impl Agent {
    /// Run one turn on `task` in `context`, streaming it on `events`, until the model answers,
    /// `events` closes or `cancel` fires. Turns must not run concurrently on one agent. A
    /// backend failure is the outcome's end; an incomplete restart is
    /// [`LlmError::ServerRestarting`].
    #[instrument(skip_all, name = "agent_turn")]
    pub async fn run(
        &self,
        task: impl Into<AgentTask>,
        context: &TurnContext,
        events: mpsc::Sender<LlmEvent>,
        cancel: CancellationToken,
    ) -> AgentOutcome {
        let tools = Arc::clone(self.tools_for(context));
        let mut turn = Turn {
            context: context.clone().for_agent(),
            tx: events,
            cancel,
            schemas: tools.openai_schemas(),
            tools,
            tools_withdrawn: None,
            streak: CallStreak::default(),
            iteration: 0,
            truncated_steps: 0,
            record: Mutex::default(),
        };
        let end = self
            .run_to_end(task.into(), &mut turn)
            .await
            .unwrap_or_else(AgentEnd::Failed);
        turn.record.into_inner().ended(end)
    }

    async fn run_to_end(&self, task: AgentTask, turn: &mut Turn) -> Result<AgentEnd, LlmError> {
        self.backend.push_user(task.text, task.attachments).await?;
        loop {
            if turn.stop_requested() {
                debug!(
                    target: "assistd::agent",
                    iteration = turn.iteration,
                    cancelled = turn.cancel.is_cancelled(),
                    "stopping between iterations (client gone or explicit cancel)"
                );
                return self.record_interruption(turn, "").await;
            }

            let outcome = match self.step_with_replay(turn).await? {
                StepEnd::Outcome(outcome) => outcome,
                StepEnd::Interrupted { streamed } => {
                    return self.record_interruption(turn, &streamed).await;
                }
            };

            match outcome {
                StepOutcome::Final => {
                    turn.emit(LlmEvent::Done).await;
                    return Ok(turn.answered());
                }
                StepOutcome::Truncated => self.retry_truncated_step(turn).await?,
                StepOutcome::ToolCalls(calls) if turn.tools_withdrawn.is_some() => {
                    self.refuse_withdrawn_calls(turn, &calls).await?;
                    return Ok(AgentEnd::ToolsWithdrawn);
                }
                StepOutcome::ToolCalls(calls) => {
                    turn.truncated_steps = 0;
                    let (results, stuck) = self.dispatch_tool_calls(turn, calls).await;
                    self.backend.push_tool_results(results).await?;
                    turn.offer(self.tools_for(&turn.context));
                    turn.iteration += 1;
                    let limits = self.definition.limits;
                    if let Some(why) = ToolBudgetExhausted::check(stuck, turn.iteration, limits) {
                        self.withdraw_tools(turn, why).await;
                    }
                }
            }
        }
    }

    /// The planning tools while `context`'s turn plans, else the acting ones.
    fn tools_for(&self, context: &TurnContext) -> &Arc<ToolRegistry> {
        match &self.definition.planning_tools {
            Some(planning) if context.planning() => planning,
            _ => &self.definition.tools,
        }
    }

    /// Keep what the cut-off step streamed as the turn's reply, marked as
    /// interrupted, in the history and for whoever records the turn.
    async fn record_interruption(&self, turn: &Turn, streamed: &str) -> Result<AgentEnd, LlmError> {
        let reply = interrupted_reply(streamed);
        self.backend.push_interrupted_reply(reply.clone()).await?;
        turn.emit(LlmEvent::Interrupted { reply }).await;
        Ok(AgentEnd::Interrupted)
    }

    /// Run one model step, passing its events on to the turn. The step is
    /// dropped once the turn is cancelled or its consumer leaves.
    pub(super) async fn relay_step(&self, turn: &Turn) -> Result<StepEnd, LlmError> {
        let (step_tx, mut step_rx) = mpsc::channel(STEP_EVENT_BUFFER);
        let mut step = pin!(self.backend.step(turn.schemas.clone(), step_tx));
        let mut streamed = String::new();
        loop {
            tokio::select! {
                biased;
                () = turn.stopped() => {
                    debug!(
                        target: "assistd::agent",
                        iteration = turn.iteration,
                        cancelled = turn.cancel.is_cancelled(),
                        "stopping during LLM step (client gone or explicit cancel)"
                    );
                    return Ok(StepEnd::Interrupted { streamed });
                }
                Some(event) = step_rx.recv() => turn.pass_on(event, &mut streamed).await,
                result = &mut step => {
                    while let Ok(event) = step_rx.try_recv() {
                        turn.pass_on(event, &mut streamed).await;
                    }
                    return result.map(StepEnd::Outcome);
                }
            }
        }
    }

    async fn withdraw_tools(&self, turn: &mut Turn, why: ToolBudgetExhausted) {
        recovery_event!(
            StatusSeverity::Warning,
            Component::Agent,
            "tools_withdrawn",
            iteration = turn.iteration,
            reason = %why.reason(),
            "withdrawing tools; asking model to answer from what it has"
        );
        turn.emit(status_event(
            StatusSeverity::Warning,
            Component::Agent,
            StatusKind::ToolsWithdrawn,
            format!(
                "Agent stopped using tools ({}); answering with what it has",
                why.reason()
            ),
        ))
        .await;
        if let Err(e) = self.backend.set_transient_note(why.model_note()).await {
            warn!(
                target: "assistd::agent",
                error = %e,
                "set_transient_note failed; answering without the note"
            );
        }
        turn.tools_withdrawn = Some(why);
    }
}

pub(super) struct Turn {
    /// The caller's context, as this agent's loop holds it.
    pub(super) context: TurnContext,
    tx: mpsc::Sender<LlmEvent>,
    pub(super) cancel: CancellationToken,
    pub(super) tools: Arc<ToolRegistry>,
    schemas: Vec<Value>,
    tools_withdrawn: Option<ToolBudgetExhausted>,
    pub(super) streak: CallStreak,
    pub(super) iteration: u32,
    pub(super) truncated_steps: u32,
    record: Mutex<RunRecord>,
}

impl Turn {
    /// Send `event` to the turn's consumer, keeping the record of what the run streamed.
    pub(super) async fn emit(&self, event: LlmEvent) {
        self.record.lock().observe(&event);
        let _ = self.tx.send(event).await;
    }

    pub(super) fn stop_requested(&self) -> bool {
        self.tx.is_closed() || self.cancel.is_cancelled()
    }

    /// Resolves once the turn is cancelled or its consumer has left.
    async fn stopped(&self) {
        tokio::select! {
            () = self.cancel.cancelled() => {}
            () = self.tx.closed() => {}
        }
    }

    /// Send a step's `event` to the turn's consumer, keeping `streamed` equal
    /// to the text the step has shown so far.
    async fn pass_on(&self, event: LlmEvent, streamed: &mut String) {
        match &event {
            LlmEvent::Delta { text } => streamed.push_str(text),
            LlmEvent::StepDiscarded => streamed.clear(),
            _ => {}
        }
        self.emit(event).await;
    }

    /// Offer `tools` from the next step on.
    fn offer(&mut self, tools: &Arc<ToolRegistry>) {
        if !Arc::ptr_eq(&self.tools, tools) {
            self.tools = Arc::clone(tools);
            self.schemas = tools.openai_schemas();
        }
    }

    /// The turn's context for tool call `call`, whose sub-agents stream on the turn's events.
    pub(super) fn call_context(&self, call: &ToolCall) -> TurnContext {
        self.context
            .in_call(self.tx.clone(), &call.id, Arc::clone(&self.tools))
    }

    /// How a turn whose model has just answered ended.
    fn answered(&self) -> AgentEnd {
        match self.tools_withdrawn {
            Some(_) => AgentEnd::ToolsWithdrawn,
            None => AgentEnd::Answered,
        }
    }

    /// [`TRUNCATION_NOTE`], restating the withdrawal the retry would otherwise lose.
    pub(super) fn truncation_note(&self) -> String {
        match self.tools_withdrawn {
            Some(why) => format!("{TRUNCATION_NOTE} {}", why.model_note()),
            None => TRUNCATION_NOTE.to_string(),
        }
    }
}

/// How a model step that did not fail ended.
pub(super) enum StepEnd {
    Outcome(StepOutcome),
    /// The turn stopped mid-step, after the step had streamed this text.
    Interrupted {
        streamed: String,
    },
}

/// Why the loop stopped accepting tool calls for the rest of the turn. The schemas
/// stay in the request after withdrawal; without them the server stops parsing
/// tool-call markup and would stream it as text.
#[derive(Debug, Clone, Copy)]
pub(super) enum ToolBudgetExhausted {
    Repeating { times: usize },
    StepCeiling { steps: u32 },
}

impl ToolBudgetExhausted {
    fn check(stuck: bool, iteration: u32, limits: AgentLimits) -> Option<Self> {
        if stuck {
            Some(Self::Repeating {
                times: limits.duplicate_calls,
            })
        } else if iteration >= limits.tool_steps {
            Some(Self::StepCeiling {
                steps: limits.tool_steps,
            })
        } else {
            None
        }
    }

    fn reason(self) -> String {
        match self {
            Self::Repeating { times } => {
                format!("the same tool call was repeated {times} times in a row")
            }
            Self::StepCeiling { steps } => {
                format!("the turn reached its ceiling of {steps} tool steps")
            }
        }
    }

    pub(super) fn model_note(self) -> String {
        format!(
            "Tool access for this turn has been withdrawn because {}. Do not request \
             any more tool calls. Answer the user now using what you have already \
             learned, and say plainly which parts you could not finish.",
            self.reason()
        )
    }
}

/// How many times in a row the model has issued the same tool call.
#[derive(Default)]
pub(super) struct CallStreak {
    last: Option<(String, Value)>,
    count: usize,
}

impl CallStreak {
    pub(super) fn record(&mut self, call: &ToolCall) -> usize {
        let same = self
            .last
            .as_ref()
            .is_some_and(|(name, args)| *name == call.name && *args == call.arguments);
        if same {
            self.count += 1;
        } else {
            self.last = Some((call.name.clone(), call.arguments.clone()));
            self.count = 1;
        }
        self.count
    }
}

/// `streamed` ending in [`INTERRUPTED_MARKER`], or the marker alone when
/// nothing was streamed.
fn interrupted_reply(streamed: &str) -> String {
    match streamed.trim_end() {
        "" => INTERRUPTED_MARKER.to_string(),
        shown => format!("{shown}\n\n{INTERRUPTED_MARKER}"),
    }
}

pub(super) fn status_event(
    severity: StatusSeverity,
    component: Component,
    event: StatusKind,
    message: String,
) -> LlmEvent {
    LlmEvent::Status {
        severity,
        component,
        event,
        message,
    }
}
