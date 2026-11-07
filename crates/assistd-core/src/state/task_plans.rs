//! The handler for `TaskRefine`: a task prompt explored in a conversation of its own,
//! rewritten as a brief for an unattended run, and revised there on feedback.

use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex as StdMutex;
use thiserror::Error;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::info;

use assistd_ipc::{Event, HostName, PermissionMode, TaskPrompt};
use assistd_llm::{ConversationKey, LlmBackend, LlmEvent, LlmHealthProbe, SlotAssignment};

use super::task_runs::task_system_prompt;
use super::{AppState, DispatchError, send_error};
use crate::planner::{PlanAsk, PlanError, Planner, Refinement, WorkPermissions};
use crate::presence::PresenceLlmHealthProbe;

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

impl AppState {
    pub(super) async fn handle_task_refine(
        self: Arc<Self>,
        id: String,
        ask: RefineAsk,
        tx: mpsc::Sender<Event>,
    ) -> Result<(), DispatchError> {
        let cancel = self.runtime.turn_cancellation();
        match self.refine_prompt(&id, ask, &tx, &cancel).await {
            Ok(Refinement {
                prompt,
                plan,
                recommended,
            }) => {
                let refined = Event::TaskRefined {
                    id: id.clone(),
                    prompt,
                    plan,
                    recommended,
                };
                let _ = tx.send(refined).await;
                let _ = tx.send(Event::Done { id }).await;
                Ok(())
            }
            Err(e) => {
                send_error(&tx, id, format!("task refine failed: {e}")).await;
                Err(e)
            }
        }
    }

    /// Refine the asked prompt, or revise its last rewrite, on the main lane, as a user
    /// turn takes it, streaming the planning turn to `tx` as `TaskPlan` events.
    async fn refine_prompt(
        &self,
        id: &str,
        ask: RefineAsk,
        tx: &mpsc::Sender<Event>,
        cancel: &CancellationToken,
    ) -> Result<Refinement, DispatchError> {
        if ask.mode == PermissionMode::Auto && !self.config.tools.allow_auto_mode {
            return Err(DispatchError::AutoModeOff);
        }
        let lane = tokio::select! {
            biased;
            () = cancel.cancelled() => return Err(PlanError::Cancelled.into()),
            lane = Arc::clone(&self.runtime.agent_turn_lock).lock_owned() => lane,
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
        let planner = self.planner();
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
        self.runtime.task_plans.record(PlanningSession {
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
                .runtime
                .task_plans
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
        let conversation = self.subsystems.llm.isolated_conversation(
            task_system_prompt(&self.config.chat.system_prompt),
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
        if self.runtime.task_plans.forget()
            && let Some(stash) = &self.subsystems.slot_stash
        {
            stash.end(ConversationKey::TaskRefine).await;
        }
    }

    pub(super) fn planner(&self) -> Planner {
        let health: Arc<dyn LlmHealthProbe> = Arc::new(PresenceLlmHealthProbe::new(
            self.subsystems.presence.clone(),
        ));
        Planner::new(
            self.subsystems.planning.clone(),
            Arc::clone(&self.subsystems.tools),
            Some(health),
            Duration::from_secs(self.config.timeouts.tool_call_secs),
            self.config.tools.allow_auto_mode,
        )
    }
}

/// Send the planning turn's events to `tx`, each wrapped as `TaskPlan`.
async fn forward_planning_events(
    id: &str,
    mut llm_rx: mpsc::Receiver<LlmEvent>,
    tx: &mpsc::Sender<Event>,
) {
    while let Some(event) = llm_rx.recv().await {
        if let Some(event) = planning_wire_event(id, event) {
            let wrapped = Event::TaskPlan {
                id: id.to_string(),
                event: Box::new(event),
            };
            let _ = tx.send(wrapped).await;
        }
    }
}

/// `event` as a client is shown it; `None` for what only the agent loop uses.
fn planning_wire_event(id: &str, event: LlmEvent) -> Option<Event> {
    let id = id.to_string();
    Some(match event {
        LlmEvent::Delta { text } => Event::Delta { id, text },
        LlmEvent::ReasoningDelta { text } => Event::ReasoningDelta { id, text },
        LlmEvent::ToolCall {
            name, arguments, ..
        } => Event::ToolCall {
            id,
            name,
            args: arguments,
        },
        LlmEvent::ToolResult { name, result, .. } => Event::ToolResult {
            id,
            name,
            result: result.to_json(),
        },
        LlmEvent::Relay(event) => Event::Relay { id, event },
        LlmEvent::ContextSize { tokens, total } => Event::ContextSize { id, tokens, total },
        LlmEvent::Status {
            severity,
            component,
            event,
            message,
        } => Event::Status {
            id,
            severity,
            component,
            event,
            message,
        },
        LlmEvent::ToolCallsRequested { .. }
        | LlmEvent::HistoryCompacted { .. }
        | LlmEvent::Done => {
            return None;
        }
    })
}
