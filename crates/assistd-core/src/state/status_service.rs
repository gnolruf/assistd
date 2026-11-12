//! `StatusService`: answers what the daemon is doing and how far it has started.

use std::sync::Arc;

use assistd_config::Config;
use assistd_embed::EmbedderHandle;
use assistd_tools::ToolsDisabled;
use assistd_utils::child_server::ChildServerStatus;
use assistd_utils::readiness::ReadinessCell;
use assistd_voice::VoiceManager;

use super::subsystems::McpServerStatus;
use super::task_service::TaskService;
use super::turn_admission::TurnAdmission;
use crate::bus::EventBus;
use crate::presence::PresenceManager;
use crate::vision::VisionRevalidator;

/// Handles the presence, capability, readiness and `Subscribe` requests: it reads the
/// subsystems' state and owns none of its own.
#[derive(Debug)]
pub struct StatusService {
    pub(super) config: Arc<Config>,
    pub(super) presence: Arc<PresenceManager>,
    pub(super) voice: Arc<VoiceManager>,
    pub(super) vision_revalidator: Option<Arc<VisionRevalidator>>,
    pub(super) mcp_servers: Arc<[McpServerStatus]>,
    pub(super) tools_disabled: Option<ToolsDisabled>,
    pub(super) web_search: Arc<ReadinessCell<ChildServerStatus>>,
    pub(super) embedder: Arc<EmbedderHandle>,
    pub(super) admission: Arc<TurnAdmission>,
    pub(super) tasks: Arc<TaskService>,
    pub(super) bus: EventBus,
}
