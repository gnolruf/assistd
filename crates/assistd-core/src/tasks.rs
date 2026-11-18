//! `TaskService`: saved tasks, their queue, schedule, triggers and unattended runs.

use std::sync::Arc;

use assistd_config::Config;
use assistd_llm::{LlmBackend, SlotStash};
use assistd_memory::TaskStore;
use assistd_tools::ToolCatalog;
use assistd_voice::VoiceManager;

use self::confirms::TaskConfirms;
use self::events::TaskEvents;
use self::plans::TaskPlans;
use self::scheduler::TaskScheduler;
use crate::agent::AgentSpec;
use crate::bus::EventBus;
use crate::planner::Planner;
use crate::presence::PresenceManager;
use crate::state::persistence::TranscriptLog;
use crate::state::session::SessionContext;
use crate::state::turn_admission::TurnAdmission;
use crate::vision::VisionRevalidator;

pub(crate) mod announce;
pub(crate) mod confirms;
pub(crate) mod dispatch;
pub(crate) mod event_triggers;
pub(crate) mod events;
pub(crate) mod fire_times;
pub(crate) mod handlers;
pub(crate) mod path_watch;
pub(crate) mod plans;
pub(crate) mod queue;
pub(crate) mod runner;
pub(crate) mod runs;
pub(crate) mod scheduler;
pub(crate) mod session_unlock;

pub use self::confirms::TaskConfirmError;
pub use self::fire_times::TriggerError;
pub use self::plans::TaskPlanError;
pub use self::queue::QueueOpError;
pub use self::runner::TaskRunner;

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
    pub(crate) deps: TaskDeps,
    pub(crate) confirms: TaskConfirms,
    pub(crate) scheduler: TaskScheduler,
    pub(crate) events: TaskEvents,
    pub(crate) plans: TaskPlans,
    pub(crate) runner: Arc<TaskRunner>,
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
