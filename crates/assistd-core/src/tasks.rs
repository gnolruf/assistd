//! `TaskService`: saved tasks, their queue, schedule, triggers and unattended runs.

use std::sync::Arc;

use thiserror::Error;
use tokio::task::JoinError;

use assistd_config::Config;
use assistd_llm::{LlmBackend, LlmError, SlotStash};
use assistd_memory::{MemoryError, TaskStore};
use assistd_tools::ToolCatalog;
use assistd_voice::VoiceManager;

use self::confirms::TaskConfirms;
use self::events::TaskEvents;
use self::plans::TaskPlans;
use self::scheduler::TaskScheduler;
use crate::admission::TurnAdmission;
use crate::agent::AgentSpec;
use crate::bus::EventBus;
use crate::planner::{PlanError, Planner};
use crate::presence::{PresenceError, PresenceManager};
use crate::transcript::{SessionContext, TranscriptLog};
use crate::vision::VisionRevalidator;

mod announce;
mod confirms;
mod dispatch;
mod event_triggers;
mod events;
mod fire_times;
mod handlers;
mod path_watch;
mod plans;
mod run_lane;
mod runner;
mod runs;
mod scheduler;
mod session_unlock;

pub use self::confirms::TaskConfirmError;
pub use self::fire_times::TriggerError;
pub use self::plans::TaskPlanError;
pub use self::runner::TaskRunner;
pub use crate::admission::queue::QueueOpError;

/// Why a `Task` request or a task run failed.
#[derive(Debug, Error)]
pub enum TaskError {
    /// Another failure, prefixed with the step of the request it happened in.
    #[error("{context}: {source}")]
    Context {
        context: &'static str,
        source: Box<TaskError>,
    },

    #[error(transparent)]
    QueueOp(#[from] QueueOpError),

    #[error(transparent)]
    Confirm(#[from] TaskConfirmError),

    #[error(transparent)]
    InvalidTrigger(#[from] TriggerError),

    #[error("auto mode is off (tools.allow_auto_mode), so a task cannot run in it")]
    AutoModeOff,

    #[error(transparent)]
    Refine(#[from] TaskPlanError),

    #[error(transparent)]
    Plan(#[from] PlanError),

    #[error(transparent)]
    Presence(#[from] PresenceError),

    #[error(transparent)]
    Llm(#[from] LlmError),

    #[error("llm backend panicked: {0}")]
    AgentPanicked(#[source] JoinError),

    #[error(transparent)]
    Memory(#[from] MemoryError),
}

/// Names the step of a task request a failure happened in.
trait FailedStep<T> {
    fn context(self, context: &'static str) -> Result<T, TaskError>;
}

impl<T, E: Into<TaskError>> FailedStep<T> for Result<T, E> {
    fn context(self, context: &'static str) -> Result<T, TaskError> {
        self.map_err(|e| TaskError::Context {
            context,
            source: Box::new(e.into()),
        })
    }
}

/// What the task service borrows from the rest of the daemon.
#[derive(Debug)]
pub(crate) struct TaskDeps {
    pub(crate) config: Arc<Config>,
    pub(crate) store: Arc<TaskStore>,
    pub(crate) transcript: Arc<TranscriptLog>,
    pub(crate) admission: Arc<TurnAdmission>,
    pub(crate) session_ctx: Arc<SessionContext>,
    pub(crate) bus: EventBus,
    pub(crate) llm: Arc<dyn LlmBackend>,
    pub(crate) presence: Arc<PresenceManager>,
    pub(crate) vision_revalidator: Option<Arc<VisionRevalidator>>,
    pub(crate) tools: Arc<ToolCatalog>,
    pub(crate) agents: AgentSpec,
    pub(crate) planner: Arc<Planner>,
    pub(crate) slot_stash: Option<Arc<SlotStash>>,
    pub(crate) voice: Arc<VoiceManager>,
}

/// Handles every `Task` request and runs saved tasks unattended: owns the runs in flight,
/// their paused prompts, the schedule, the event watches and the planning sessions.
#[derive(Debug)]
pub struct TaskService {
    deps: TaskDeps,
    confirms: TaskConfirms,
    scheduler: TaskScheduler,
    events: TaskEvents,
    plans: TaskPlans,
    runner: Arc<TaskRunner>,
}

impl TaskService {
    pub(crate) fn new(deps: TaskDeps) -> Arc<Self> {
        Arc::new(Self {
            deps,
            confirms: TaskConfirms::default(),
            scheduler: TaskScheduler::default(),
            events: TaskEvents::default(),
            plans: TaskPlans::default(),
            runner: Arc::default(),
        })
    }

    /// The task runs in flight.
    pub fn runner(&self) -> &Arc<TaskRunner> {
        &self.runner
    }
}

#[cfg(test)]
mod tests;
