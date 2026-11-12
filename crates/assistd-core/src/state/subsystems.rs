//! `Subsystems`: LLM, voice, presence, tools, and WM handles owned by
//! `AppState`.

use std::sync::Arc;

use parking_lot::RwLock;

use assistd_ipc::{ComponentReadiness, McpServerState, McpTransportKind};
use assistd_llm::{LlmBackend, SlotStash};
use assistd_tools::{ToolCatalog, ToolRegistry, ToolsDisabled};
use assistd_utils::child_server::ChildServerStatus;
use assistd_utils::readiness::{Readiness, ReadinessCell};
use assistd_voice::VoiceManager;
use assistd_wm::{NoWindowManager, WindowManager};

use super::mcp_handlers::McpAccounts;
use crate::planner::PlanningTools;
use crate::{PresenceManager, VisionRevalidator};

/// A configured MCP server and where it stands.
#[derive(Debug)]
pub struct McpServerStatus {
    name: String,
    transport: McpTransportKind,
    state: RwLock<McpServerState>,
}

impl McpServerStatus {
    pub fn starting(name: String, transport: McpTransportKind) -> Self {
        Self {
            name,
            transport,
            state: RwLock::new(McpServerState::Starting),
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn transport(&self) -> McpTransportKind {
        self.transport
    }

    pub fn state(&self) -> McpServerState {
        self.state.read().clone()
    }

    pub fn set(&self, state: McpServerState) {
        *self.state.write() = state;
    }

    /// The state as background-subsystem readiness.
    pub fn readiness(&self) -> ComponentReadiness {
        let state = self.state.read();
        match state.unavailable_reason(&self.name) {
            Some(reason) => ComponentReadiness::Unavailable { reason },
            None if *state == McpServerState::Ready => ComponentReadiness::Ready,
            None => ComponentReadiness::Starting,
        }
    }
}

/// Handles to the long-lived daemon subsystems.
#[derive(Debug)]
pub struct Subsystems {
    pub llm: Arc<dyn LlmBackend>,
    pub presence: Arc<PresenceManager>,
    pub tools: Arc<ToolCatalog>,
    /// What planning turns may use.
    pub planning: PlanningTools,
    pub voice: Arc<VoiceManager>,
    pub window_manager: Arc<dyn WindowManager>,
    pub vision_revalidator: Option<Arc<VisionRevalidator>>,
    pub mcp_servers: Arc<[McpServerStatus]>,
    /// Signs OAuth MCP servers in and out; `None` while MCP is off.
    pub mcp_accounts: Option<Arc<dyn McpAccounts>>,
    /// Readiness of the SearXNG instance web search goes through.
    pub web_search: Arc<ReadinessCell<ChildServerStatus>>,
    pub tools_disabled: Option<ToolsDisabled>,
    /// Keeps each conversation's slot state while another uses the slot.
    pub slot_stash: Option<Arc<SlotStash>>,
}

impl Subsystems {
    /// Bundle the required subsystems, with no planning tools, no window manager, no vision
    /// revalidator, no MCP servers, and no web search.
    pub fn new(
        llm: Arc<dyn LlmBackend>,
        presence: Arc<PresenceManager>,
        tools: Arc<ToolRegistry>,
        voice: Arc<VoiceManager>,
    ) -> Self {
        Self {
            llm,
            presence,
            tools: Arc::new(ToolCatalog::new(tools)),
            planning: PlanningTools::default(),
            voice,
            window_manager: Arc::new(NoWindowManager),
            vision_revalidator: None,
            mcp_servers: Arc::default(),
            mcp_accounts: None,
            web_search: Arc::new(ReadinessCell::new(Readiness::Unavailable(
                "not configured".into(),
            ))),
            tools_disabled: None,
            slot_stash: None,
        }
    }

    /// Give planning turns `planning`.
    pub fn with_planning_tools(mut self, planning: PlanningTools) -> Self {
        self.planning = planning;
        self
    }

    /// Replace the window manager.
    pub fn with_window_manager(mut self, window_manager: Arc<dyn WindowManager>) -> Self {
        self.window_manager = window_manager;
        self
    }

    /// Attach a vision revalidator.
    pub fn with_vision_revalidator(mut self, revalidator: Arc<VisionRevalidator>) -> Self {
        self.vision_revalidator = Some(revalidator);
        self
    }

    /// Track the configured MCP servers, signing them in and out through `accounts`; `None`
    /// while MCP is off.
    pub fn with_mcp_servers(
        mut self,
        servers: Vec<McpServerStatus>,
        accounts: Option<Arc<impl McpAccounts + 'static>>,
    ) -> Self {
        self.mcp_servers = servers.into();
        self.mcp_accounts = accounts.map(|accounts| accounts as Arc<dyn McpAccounts>);
        self
    }

    /// Track the web search instance's startup.
    pub fn with_web_search(mut self, web_search: Arc<ReadinessCell<ChildServerStatus>>) -> Self {
        self.web_search = web_search;
        self
    }

    /// Keep each conversation's slot state in `slot_stash` while another uses the slot.
    pub fn with_slot_stash(mut self, slot_stash: Arc<SlotStash>) -> Self {
        self.slot_stash = Some(slot_stash);
        self
    }

    /// Record why the model is offered no tools, if it is not.
    pub fn with_tools_disabled(mut self, disabled: Option<ToolsDisabled>) -> Self {
        self.tools_disabled = disabled;
        self
    }
}
