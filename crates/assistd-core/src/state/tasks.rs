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
use self::runner::TaskRunner;
use self::scheduler::TaskScheduler;
use super::persistence::TranscriptLog;
use super::session::SessionContext;
use super::turn_admission::TurnAdmission;
use crate::agent::AgentSpec;
use crate::bus::EventBus;
use crate::planner::Planner;
use crate::presence::PresenceManager;
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

/// What the task service borrows from the rest of the daemon.
#[derive(Debug)]
pub(super) struct TaskDeps {
    pub(super) config: Arc<Config>,
    pub(super) store: Arc<TaskStore>,
    pub(super) transcript: Arc<TranscriptLog>,
    pub(super) admission: Arc<TurnAdmission>,
    pub(super) session_ctx: Arc<SessionContext>,
    pub(super) bus: EventBus,
    pub(super) llm: Arc<dyn LlmBackend>,
    pub(super) presence: Arc<PresenceManager>,
    pub(super) vision_revalidator: Option<Arc<VisionRevalidator>>,
    pub(super) tools: Arc<ToolCatalog>,
    pub(super) agents: AgentSpec,
    pub(super) planner: Arc<Planner>,
    pub(super) slot_stash: Option<Arc<SlotStash>>,
    pub(super) voice: Arc<VoiceManager>,
}

/// Handles every `Task` request and runs saved tasks unattended: owns the runs in flight,
/// their paused prompts, the schedule, the event watches and the planning sessions.
#[derive(Debug)]
pub struct TaskService {
    pub(super) deps: TaskDeps,
    pub(super) confirms: TaskConfirms,
    pub(super) scheduler: TaskScheduler,
    pub(super) events: TaskEvents,
    pub(super) plans: TaskPlans,
    pub(super) runner: Arc<TaskRunner>,
}

impl TaskService {
    pub(super) fn new(deps: TaskDeps) -> Arc<Self> {
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
