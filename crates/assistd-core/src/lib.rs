//! Daemon orchestration: the agent loop, presence state machine, IPC socket server and
//! `AppState` dispatcher; re-exports the subsystem crates so dependents need only this one.

pub mod agent;
mod build_tools;
pub mod bus;
pub mod gpu_monitor;
pub mod idle_monitor;
pub mod listen_dispatcher;
pub mod planner;
pub mod presence;
pub mod recovery;
mod relay;
mod scratch;
pub mod socket;
pub mod state;
pub mod subagent;
mod vision;
mod vram;

pub use agent::Agent;
pub use build_tools::{
    BuildToolsDeps, BuildToolsError, BuiltTools, build_tools, mcp_tool_approvals,
    probe_tool_sandbox,
};
pub use bus::{BusSubscription, ClientGone, EventBus, EventSink};
pub use recovery::{
    Component, StatusSeverity, drain_join_set, install_panic_hook, spawn_supervised,
};
pub use relay::{RelayDeps, sweep_relay_answers};

pub use assistd_config as config;
pub use assistd_config::{
    BashSandboxMode, ChatConfig, CompositorConfig, CompositorType, Config, ConfigError, McpConfig,
    McpServerConfig, ModelConfig, SleepConfig, SynthesisConfig, ToolsBashConfig, ToolsConfig,
    ToolsOutputConfig, ToolsScratchConfig, ToolsWriteConfig, VoiceConfig, WakeWordConfig,
};

pub use assistd_ipc as ipc;
pub use assistd_ipc::{ListenPhase, PresenceState, PresenceTarget, VoiceCaptureState};

pub use assistd_tools::{CommandRegistry, ToolRegistry};

pub use assistd_voice::{
    CaptureUnavailable, ContinuousListener, Heard, NoContinuousListener, SpeakDecision,
    VoiceCapture, VoiceInput, VoiceManager, VoiceOutput, VoiceOutputController,
};

pub use assistd_wm::{NoWindowManager, WindowManager};

pub use presence::{PresenceError, PresenceManager, RequestGuard, VramShortfall, WakeRoom};
pub use state::{
    AppState, BranchService, DispatchError, McpAccountError, McpAccounts, McpHost, McpLogin,
    McpServerStatus, McpService, MemoryStack, Query, Requester, SessionContext, StatusService,
    Subsystems, TaskRunner, TaskService, TurnOrigin, TurnService, VoiceService, branch_replay,
};
pub use subagent::{RelayReport, SubAgentError, SubAgentRunner};
pub use vision::VisionRevalidator;
pub use vram::VramLedger;
