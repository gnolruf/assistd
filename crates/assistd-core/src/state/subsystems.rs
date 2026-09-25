//! `Subsystems`: LLM, voice, presence, tools, and WM handles owned by
//! `AppState`.

use std::sync::Arc;

use assistd_llm::LlmBackend;
use assistd_tools::ToolRegistry;
use assistd_voice::{ContinuousListener, VoiceInput, VoiceOutputController};
use assistd_wm::{NoWindowManager, WindowManager};

use crate::{PresenceManager, VisionRevalidator};

/// One MCP server that failed to start during daemon boot.
#[derive(Debug, Clone)]
pub struct McpStartupFailure {
    pub server_name: String,
    pub reason: String,
}

/// Handles to the long-lived daemon subsystems.
pub struct Subsystems {
    pub llm: Arc<dyn LlmBackend>,
    pub presence: Arc<PresenceManager>,
    pub tools: Arc<ToolRegistry>,
    pub voice: Arc<dyn VoiceInput>,
    pub listener: Arc<dyn ContinuousListener>,
    pub voice_output: Arc<VoiceOutputController>,
    pub window_manager: Arc<dyn WindowManager>,
    pub vision_revalidator: Option<Arc<VisionRevalidator>>,
    pub mcp_startup_failures: Vec<McpStartupFailure>,
}

impl Subsystems {
    /// Bundle the required subsystems, with no window manager, no vision
    /// revalidator, and no MCP startup failures.
    pub fn new(
        llm: Arc<dyn LlmBackend>,
        presence: Arc<PresenceManager>,
        tools: Arc<ToolRegistry>,
        voice: Arc<dyn VoiceInput>,
        listener: Arc<dyn ContinuousListener>,
        voice_output: Arc<VoiceOutputController>,
    ) -> Self {
        Self {
            llm,
            presence,
            tools,
            voice,
            listener,
            voice_output,
            window_manager: Arc::new(NoWindowManager),
            vision_revalidator: None,
            mcp_startup_failures: Vec::new(),
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

    /// Record the MCP servers that failed to start.
    pub fn with_mcp_startup_failures(mut self, failures: Vec<McpStartupFailure>) -> Self {
        self.mcp_startup_failures = failures;
        self
    }
}
