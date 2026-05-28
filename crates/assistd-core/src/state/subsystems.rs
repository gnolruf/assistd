//! `Subsystems`: LLM, voice, presence, tools, and WM handles owned by
//! `AppState`.

use crate::PresenceManager;
use assistd_llm::LlmBackend;
use assistd_tools::ToolRegistry;
use assistd_voice::{ContinuousListener, VoiceInput, VoiceOutputController};
use assistd_wm::{NoWindowManager, WindowManager};
use std::sync::Arc;

/// One MCP server that failed to start during daemon boot, surfaced
/// to clients via `Event::Status`.
#[derive(Debug, Clone)]
pub struct McpStartupFailure {
    pub server_name: String,
    pub reason: String,
}

pub struct Subsystems {
    pub llm: Arc<dyn LlmBackend>,
    pub presence: Arc<PresenceManager>,
    pub tools: Arc<ToolRegistry>,
    pub voice: Arc<dyn VoiceInput>,
    pub listener: Arc<dyn ContinuousListener>,
    pub voice_output: Arc<VoiceOutputController>,
    pub window_manager: Arc<dyn WindowManager>,
    pub vision_revalidator: Option<Arc<crate::VisionRevalidator>>,
    pub mcp_startup_failures: Vec<McpStartupFailure>,
}

impl Subsystems {
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

    pub fn with_window_manager(mut self, wm: Arc<dyn WindowManager>) -> Self {
        self.window_manager = wm;
        self
    }

    pub fn with_vision_revalidator(mut self, r: Arc<crate::VisionRevalidator>) -> Self {
        self.vision_revalidator = Some(r);
        self
    }

    pub fn with_mcp_startup_failures(mut self, failures: Vec<McpStartupFailure>) -> Self {
        self.mcp_startup_failures = failures;
        self
    }
}
