//! `Subsystems` groups the LLM-, voice-, presence-, tools-, and
//! window-manager handles that AppState routes requests through.
//!
//! Substructs are added in this step but not yet wired into
//! [`AppState`]; the `#[allow(dead_code)]` annotation keeps clippy
//! quiet until Step 3 flips the field set.

use crate::PresenceManager;
use assistd_llm::LlmBackend;
use assistd_tools::ToolRegistry;
use assistd_voice::{ContinuousListener, VoiceInput, VoiceOutputController};
use assistd_wm::{NoWindowManager, WindowManager};
use std::sync::Arc;

pub struct Subsystems {
    pub llm: Arc<dyn LlmBackend>,
    pub presence: Arc<PresenceManager>,
    pub tools: Arc<ToolRegistry>,
    pub voice: Arc<dyn VoiceInput>,
    pub listener: Arc<dyn ContinuousListener>,
    pub voice_output: Arc<VoiceOutputController>,
    pub window_manager: Arc<dyn WindowManager>,
    pub vision_revalidator: Option<Arc<crate::VisionRevalidator>>,
}

impl Subsystems {
    /// Construct a `Subsystems` with all required backends supplied.
    /// `window_manager` defaults to [`NoWindowManager`] and
    /// `vision_revalidator` to `None`; use the `with_*` builders to
    /// attach those.
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
}
