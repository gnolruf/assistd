//! `Subsystems`: LLM, voice, presence, tools, and WM handles owned by
//! `AppState`.

use std::sync::Arc;

use assistd_llm::LlmBackend;
use assistd_tools::{ToolCatalog, ToolRegistry, ToolsDisabled};
use assistd_utils::child_server::ChildServerStatus;
use assistd_utils::readiness::{NotReady, Readiness, ReadinessCell};
use assistd_voice::VoiceManager;
use assistd_wm::{NoWindowManager, WindowManager};

use crate::{PresenceManager, VisionRevalidator};

/// A configured MCP server and how far its startup has got.
#[derive(Debug)]
pub struct McpServerStatus {
    name: String,
    readiness: ReadinessCell<()>,
}

impl McpServerStatus {
    pub fn starting(name: String) -> Self {
        Self {
            name,
            readiness: ReadinessCell::starting(),
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// Record how far the server's startup has got.
    pub fn set(&self, readiness: Readiness<()>) {
        self.readiness.set(readiness);
    }

    /// `Ok` once the server's tools are offered, else why they are not.
    pub fn readiness(&self) -> Result<(), NotReady> {
        self.readiness.get()
    }
}

/// Handles to the long-lived daemon subsystems.
#[derive(Debug)]
pub struct Subsystems {
    pub llm: Arc<dyn LlmBackend>,
    pub presence: Arc<PresenceManager>,
    pub tools: Arc<ToolCatalog>,
    pub voice: Arc<VoiceManager>,
    pub window_manager: Arc<dyn WindowManager>,
    pub vision_revalidator: Option<Arc<VisionRevalidator>>,
    pub mcp_servers: Vec<McpServerStatus>,
    /// The SearXNG instance web search goes through, once it is serving.
    pub web_search: Arc<ReadinessCell<ChildServerStatus>>,
    pub tools_disabled: Option<ToolsDisabled>,
}

impl Subsystems {
    /// Bundle the required subsystems, with no window manager, no vision
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
            voice,
            window_manager: Arc::new(NoWindowManager),
            vision_revalidator: None,
            mcp_servers: Vec::new(),
            web_search: Arc::new(ReadinessCell::new(Readiness::Unavailable(
                "not configured".into(),
            ))),
            tools_disabled: None,
        }
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

    /// Track the configured MCP servers' startup.
    pub fn with_mcp_servers(mut self, servers: Vec<McpServerStatus>) -> Self {
        self.mcp_servers = servers;
        self
    }

    /// Track the web search instance's startup.
    pub fn with_web_search(mut self, web_search: Arc<ReadinessCell<ChildServerStatus>>) -> Self {
        self.web_search = web_search;
        self
    }

    /// Record why the model is offered no tools, if it is not.
    pub fn with_tools_disabled(mut self, disabled: Option<ToolsDisabled>) -> Self {
        self.tools_disabled = disabled;
        self
    }
}
