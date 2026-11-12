//! The handler for `TaskRefine`: a task prompt explored in a conversation of its own,
//! rewritten as a brief for an unattended run, and revised there on feedback.

use std::sync::Arc;

use parking_lot::Mutex as StdMutex;
use thiserror::Error;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info};

use assistd_ipc::{Event, EventBody, HostName, PermissionMode, TaskPrompt};
use assistd_llm::{ConversationKey, LlmBackend, LlmEvent, SlotAssignment};

use super::task_runs::task_system_prompt;
use super::task_service::TaskService;
use super::{DispatchError, FailedStep};
use crate::bus::EventSink;
use crate::planner::{PlanAsk, PlanError, Refinement, WorkPermissions};

const EVENT_BUFFER: usize = 32;

/// Why a task prompt could not be refined.
#[derive(Debug, Error)]
pub enum TaskPlanError {
    #[error("this prompt has no rewrite to revise; refine it without feedback first")]
    NothingToRevise,
}

/// What a `TaskRefine` asks for: the prompt, the permissions its runs would have, and any
/// feedback on the last rewrite.
pub(super) struct RefineAsk {
    pub prompt: TaskPrompt,
    pub mode: PermissionMode,
    pub allowed_hosts: Vec<HostName>,
    pub feedback: Option<String>,
}

/// The planning conversation of the latest refine since the daemon started.
#[derive(Debug, Default)]
pub(in crate::state) struct TaskPlans {
    latest: StdMutex<Option<PlanningSession>>,
}

/// A planning conversation, the prompt it explored, and its last rewrite of that prompt.
#[derive(Debug, Clone)]
struct PlanningSession {
    conversation: Arc<dyn LlmBackend>,
    prompt: TaskPrompt,
    rewrite: Option<TaskPrompt>,
}

impl TaskPlans {
    fn latest(&self) -> Option<PlanningSession> {
        self.latest.lock().clone()
    }

    fn record(&self, session: PlanningSession) {
        *self.latest.lock() = Some(session);
    }

    /// Drop the latest planning conversation; `false` when there was none.
    fn forget(&self) -> bool {
        self.latest.lock().take().is_some()
    }
}

impl TaskService {
    pub(super) async fn handle_task_refine(
        &self,
        id: String,
        ask: RefineAsk,
        tx: EventSink,
    ) -> Result<(), DispatchError> {
        let cancel = self.deps.admission.turn_cancellation();
        let refining = self.refine_prompt(&id, ask, &tx, &cancel);
        let Refinement {
            prompt,
            plan,
            recommended,
        } = cancel_when_client_leaves(refining, &id, &tx, &cancel)
            .await
            .context("task refine failed")?;
        let refined = Event::new(
            id.clone(),
            EventBody::TaskRefined {
                prompt,
                plan,
                recommended,
            },
        );
        let _ = tx.send(refined).await;
        let _ = tx.send(Event::new(id, EventBody::Done)).await;
        Ok(())
    }

    /// Refine the asked prompt, or revise its last rewrite, on the main lane, as a user
    /// turn takes it, streaming the planning turn to `tx` as `TaskPlan` events.
    async fn refine_prompt(
        &self,
        id: &str,
        ask: RefineAsk,
        tx: &EventSink,
        cancel: &CancellationToken,
    ) -> Result<Refinement, DispatchError> {
        if ask.mode == PermissionMode::Auto && !self.deps.config.tools.allow_auto_mode {
            return Err(DispatchError::AutoModeOff);
        }
        let lane = tokio::select! {
            biased;
            () = cancel.cancelled() => return Err(PlanError::Cancelled.into()),
            lane = Arc::clone(&self.deps.admission.agent_turn_lock).lock_owned() => lane,
        };
        let Some(_awake) = self.wake_for_task_run(id, tx, cancel).await? else {
            return Err(PlanError::Cancelled.into());
        };
        let (session, plan_ask) = self.planning_session(ask.prompt, ask.feedback).await?;
        let permissions = WorkPermissions {
            mode: ask.mode,
            allowed_hosts: ask.allowed_hosts,
        };
        let (llm_tx, llm_rx) = mpsc::channel(EVENT_BUFFER);
        let planner = &self.deps.planner;
        let planning = planner.refine(
            Arc::clone(&session.conversation),
            plan_ask,
            permissions,
            id.to_string(),
            llm_tx,
            cancel.clone(),
        );
        let (refinement, ()) = tokio::join!(planning, forward_planning_events(id, llm_rx, tx));
        drop(lane);
        let refinement = refinement?;
        info!(target: "assistd::tasks", plan = %refinement.plan.id, "task prompt refined");
        self.plans.record(PlanningSession {
            rewrite: Some(refinement.prompt.clone()),
            ..session
        });
        Ok(refinement)
    }

    /// The conversation to refine `prompt` in and what to ask it: with `feedback`, the one
    /// that last rewrote `prompt`; without, a new one in place of the last.
    async fn planning_session(
        &self,
        prompt: TaskPrompt,
        feedback: Option<String>,
    ) -> Result<(PlanningSession, PlanAsk), DispatchError> {
        if let Some(feedback) = feedback {
            let (session, shown) = self
                .plans
                .latest()
                .filter(|session| session.prompt == prompt)
                .and_then(|session| {
                    let shown = session.rewrite.as_ref()?.as_str().to_string();
                    Some((session, shown))
                })
                .ok_or(TaskPlanError::NothingToRevise)?;
            return Ok((session, PlanAsk::Revise { shown, feedback }));
        }
        self.end_planning_conversation().await;
        let conversation = self.deps.llm.isolated_conversation(
            task_system_prompt(&self.deps.config.chat.system_prompt),
            SlotAssignment::Shared(ConversationKey::TaskRefine),
        )?;
        let ask = PlanAsk::Work(prompt.as_str().to_string());
        let session = PlanningSession {
            conversation,
            prompt,
            rewrite: None,
        };
        Ok((session, ask))
    }

    /// Forget the latest planning conversation and its saved slot state.
    async fn end_planning_conversation(&self) {
        if self.plans.forget()
            && let Some(stash) = &self.deps.slot_stash
        {
            stash.end(ConversationKey::TaskRefine).await;
        }
    }
}

/// Run `refining` to completion, cancelling it through `cancel` if the client behind `tx`
/// disconnects first, so an abandoned refine gives up the main lane.
async fn cancel_when_client_leaves<T>(
    refining: impl Future<Output = T>,
    id: &str,
    tx: &EventSink,
    cancel: &CancellationToken,
) -> T {
    tokio::pin!(refining);
    tokio::select! {
        biased;
        outcome = &mut refining => outcome,
        () = tx.closed() => {
            debug!(target: "assistd::tasks", id = %id, "client left mid-refine; cancelling it");
            cancel.cancel();
            refining.await
        }
    }
}

/// Send the planning turn's events to `tx`, each wrapped as `TaskPlan`.
async fn forward_planning_events(id: &str, mut llm_rx: mpsc::Receiver<LlmEvent>, tx: &EventSink) {
    while let Some(event) = llm_rx.recv().await {
        if let Some(event) = planning_wire_event(id, event) {
            let wrapped = Event::new(
                id.to_string(),
                EventBody::TaskPlan {
                    event: Box::new(event),
                },
            );
            let _ = tx.send(wrapped).await;
        }
    }
}

/// `event` as a client is shown it; `None` for what only the agent loop uses.
fn planning_wire_event(id: &str, event: LlmEvent) -> Option<Event> {
    let id = id.to_string();
    Some(match event {
        LlmEvent::Delta { text } => Event::new(id, EventBody::Delta { text }),
        LlmEvent::ReasoningDelta { text } => Event::new(id, EventBody::ReasoningDelta { text }),
        LlmEvent::ToolCall {
            name, arguments, ..
        } => Event::new(
            id,
            EventBody::ToolCall {
                name,
                args: arguments,
            },
        ),
        LlmEvent::ToolResult { name, result, .. } => Event::new(
            id,
            EventBody::ToolResult {
                name,
                result: result.report(),
            },
        ),
        LlmEvent::Relay(event) => Event::new(id, EventBody::Relay { event }),
        LlmEvent::ContextSize { tokens, total } => {
            Event::new(id, EventBody::ContextSize { tokens, total })
        }
        LlmEvent::Status {
            severity,
            component,
            event,
            message,
        } => Event::new(
            id,
            EventBody::Status {
                severity,
                component,
                event,
                message,
            },
        ),
        LlmEvent::StepStarted
        | LlmEvent::StepDiscarded
        | LlmEvent::ToolCallsRequested { .. }
        | LlmEvent::HistoryCompacted { .. }
        | LlmEvent::Interrupted { .. }
        | LlmEvent::Done => {
            return None;
        }
    })
}
