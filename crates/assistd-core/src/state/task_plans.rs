//! Handlers for `TaskPlan` and `TaskPlanApprove`: a task's prompt planned in a conversation of
//! its own and revised there on feedback, and an approved plan saved on the task with its
//! accepted actions as the task's grants.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex as StdMutex;
use thiserror::Error;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::info;

use assistd_ipc::{Event, Plan, PlanId, TaskGrant, TaskId, TaskPrompt};
use assistd_llm::{ConversationKey, LlmBackend, LlmEvent, LlmHealthProbe, SlotAssignment};
use assistd_memory::TaskRecord;

use super::task_runs::task_system_prompt;
use super::{AppState, DispatchError, send_error};
use crate::planner::{PlanAsk, PlanError, Planner};
use crate::presence::PresenceLlmHealthProbe;

const EVENT_BUFFER: usize = 32;

/// Why a task could not be planned, or its plan approved.
#[derive(Debug, Error)]
pub enum TaskPlanError {
    #[error("task {0} has no plan to revise; plan it without feedback first")]
    NoPlanToRevise(TaskId),
    #[error("plan {plan} is not the latest plan made for task {task}")]
    NotLatest { task: TaskId, plan: PlanId },
    #[error("an accepted grant is not one of the plan's actions: {0:?}")]
    NotInPlan(TaskGrant),
    #[error("task {0}'s prompt changed after it was planned; plan it again")]
    PromptChanged(TaskId),
}

/// The planning conversation of each task planned since the daemon started.
#[derive(Debug, Default)]
pub(in crate::state) struct TaskPlans {
    sessions: StdMutex<HashMap<TaskId, PlanningSession>>,
}

/// One task's planning conversation, the prompt it planned, and the last plan it made.
#[derive(Debug, Clone)]
struct PlanningSession {
    conversation: Arc<dyn LlmBackend>,
    prompt: TaskPrompt,
    latest: Option<Plan>,
}

impl TaskPlans {
    fn get(&self, task: TaskId) -> Option<PlanningSession> {
        self.sessions.lock().get(&task).cloned()
    }

    fn record(&self, task: TaskId, session: PlanningSession) {
        self.sessions.lock().insert(task, session);
    }

    /// Drop task `task`'s planning conversation; `false` when it had none.
    fn forget(&self, task: TaskId) -> bool {
        self.sessions.lock().remove(&task).is_some()
    }
}

impl AppState {
    pub(super) async fn handle_task_plan(
        self: Arc<Self>,
        id: String,
        task: TaskId,
        feedback: Option<String>,
        tx: mpsc::Sender<Event>,
    ) -> Result<(), DispatchError> {
        let cancel = self.runtime.turn_cancellation();
        match self.plan_task(&id, task, feedback, &tx, &cancel).await {
            Ok(plan) => {
                let _ = tx
                    .send(Event::Plan {
                        id: id.clone(),
                        plan,
                    })
                    .await;
                let _ = tx.send(Event::Done { id }).await;
                Ok(())
            }
            Err(e) => {
                send_error(&tx, id, format!("task plan failed: {e}")).await;
                Err(e)
            }
        }
    }

    pub(super) async fn handle_plan_approve(
        self: Arc<Self>,
        id: String,
        task: TaskId,
        plan: PlanId,
        accepted: Vec<TaskGrant>,
        tx: mpsc::Sender<Event>,
    ) -> Result<(), DispatchError> {
        let record = self
            .approve_task_plan(task, plan, accepted)
            .await
            .map(|record| vec![record]);
        self.send_tasks(id, "task plan approve", record, &tx).await
    }

    /// Plan or revise task `task` on the main lane, as a user turn takes it, streaming the
    /// planning turn to `tx` as `TaskPlan` events.
    async fn plan_task(
        &self,
        id: &str,
        task: TaskId,
        feedback: Option<String>,
        tx: &mpsc::Sender<Event>,
        cancel: &CancellationToken,
    ) -> Result<Plan, DispatchError> {
        let record = self.memory.tasks.get(task).await?;
        let lane = tokio::select! {
            biased;
            () = cancel.cancelled() => return Err(PlanError::Cancelled.into()),
            lane = Arc::clone(&self.runtime.agent_turn_lock).lock_owned() => lane,
        };
        let Some(_awake) = self.wake_for_task_run(id, tx, cancel).await? else {
            return Err(PlanError::Cancelled.into());
        };
        let (session, ask) = self.planning_session(&record, feedback).await?;
        let (llm_tx, llm_rx) = mpsc::channel(EVENT_BUFFER);
        let planner = self.planner();
        let planning = planner.plan(
            Arc::clone(&session.conversation),
            ask,
            id.to_string(),
            llm_tx,
            cancel.clone(),
        );
        let (plan, ()) = tokio::join!(planning, forward_planning_events(id, task, llm_rx, tx));
        drop(lane);
        let plan = plan?;
        info!(target: "assistd::tasks", %task, plan = %plan.id, "task planned");
        let latest = Some(plan.clone());
        self.runtime
            .task_plans
            .record(task, PlanningSession { latest, ..session });
        Ok(plan)
    }

    /// The conversation to plan `record` in and what to ask it: with `feedback`, the one
    /// that made the task's last plan for its current prompt; without, a new one.
    async fn planning_session(
        &self,
        record: &TaskRecord,
        feedback: Option<String>,
    ) -> Result<(PlanningSession, PlanAsk), DispatchError> {
        let task = record.id;
        if let Some(feedback) = feedback {
            let session = self
                .runtime
                .task_plans
                .get(task)
                .filter(|session| session.prompt == record.spec.prompt && session.latest.is_some())
                .ok_or(TaskPlanError::NoPlanToRevise(task))?;
            return Ok((session, PlanAsk::Revise(feedback)));
        }
        self.end_planning_conversation(task).await;
        let conversation = self.subsystems.llm.isolated_conversation(
            task_system_prompt(&self.config.chat.system_prompt),
            SlotAssignment::Shared(ConversationKey::TaskPlan(task)),
        )?;
        let session = PlanningSession {
            conversation,
            prompt: record.spec.prompt.clone(),
            latest: None,
        };
        let ask = PlanAsk::Work(record.spec.prompt.as_str().to_string());
        Ok((session, ask))
    }

    /// Save plan `plan` on task `task` with `accepted` as its grants, then drop the task's
    /// planning conversation.
    async fn approve_task_plan(
        &self,
        task: TaskId,
        plan: PlanId,
        accepted: Vec<TaskGrant>,
    ) -> Result<TaskRecord, DispatchError> {
        let not_latest = || TaskPlanError::NotLatest {
            task,
            plan: plan.clone(),
        };
        let session = self.runtime.task_plans.get(task).ok_or_else(not_latest)?;
        let latest = session
            .latest
            .filter(|latest| latest.id == plan)
            .ok_or_else(not_latest)?;
        let approved = approved_plan(latest, &accepted)?;
        let grants = distinct(accepted);
        let record = self
            .memory
            .tasks
            .approve_plan(task, session.prompt, approved, grants)
            .await?
            .ok_or(TaskPlanError::PromptChanged(task))?;
        info!(target: "assistd::tasks", %task, %plan, "task plan approved");
        self.end_planning_conversation(task).await;
        Ok(record)
    }

    /// Forget task `task`'s planning conversation and its saved slot state.
    async fn end_planning_conversation(&self, task: TaskId) {
        if self.runtime.task_plans.forget(task)
            && let Some(stash) = &self.subsystems.slot_stash
        {
            stash.end(ConversationKey::TaskPlan(task)).await;
        }
    }

    fn planner(&self) -> Planner {
        let health: Arc<dyn LlmHealthProbe> = Arc::new(PresenceLlmHealthProbe::new(
            self.subsystems.presence.clone(),
        ));
        Planner::new(
            self.subsystems.planning.clone(),
            Arc::clone(&self.subsystems.tools),
            Some(health),
            Duration::from_secs(self.config.timeouts.tool_call_secs),
        )
    }
}

/// `plan` holding only the actions in `accepted`, each of which must be one of its actions.
fn approved_plan(plan: Plan, accepted: &[TaskGrant]) -> Result<Plan, TaskPlanError> {
    let offered = |grant: &TaskGrant| {
        plan.steps
            .iter()
            .flat_map(|step| &step.actions)
            .any(|action| action.grant == *grant)
    };
    if let Some(stray) = accepted.iter().find(|grant| !offered(grant)) {
        return Err(TaskPlanError::NotInPlan(stray.clone()));
    }
    let steps = plan
        .steps
        .into_iter()
        .map(|mut step| {
            step.actions
                .retain(|action| accepted.contains(&action.grant));
            step
        })
        .collect();
    Ok(Plan { steps, ..plan })
}

fn distinct(grants: Vec<TaskGrant>) -> Vec<TaskGrant> {
    let mut kept = Vec::with_capacity(grants.len());
    for grant in grants {
        if !kept.contains(&grant) {
            kept.push(grant);
        }
    }
    kept
}

/// Send the planning turn's events to `tx`, each wrapped as task `task`'s.
async fn forward_planning_events(
    id: &str,
    task: TaskId,
    mut llm_rx: mpsc::Receiver<LlmEvent>,
    tx: &mpsc::Sender<Event>,
) {
    while let Some(event) = llm_rx.recv().await {
        if let Some(event) = planning_wire_event(id, event) {
            let wrapped = Event::TaskPlan {
                id: id.to_string(),
                task,
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
