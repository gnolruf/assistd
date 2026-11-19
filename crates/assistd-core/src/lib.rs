//! Daemon orchestration: the agent loop, presence state machine, IPC socket server,
//! `AppState` dispatcher and saved-task scheduling.

mod admission;
pub mod agent;
mod build_tools;
pub mod bus;
mod documents;
pub mod gpu_monitor;
pub mod idle_monitor;
mod judge;
pub mod listen_dispatcher;
pub mod planner;
pub mod presence;
pub mod recovery;
mod relay;
mod scratch;
pub mod socket;
pub mod state;
pub mod subagent;
pub mod tasks;
mod transcript;
mod turn_driver;
mod vision;
mod vram;

pub use agent::{Agent, AgentSpec};
pub use build_tools::{
    BuildToolsDeps, BuildToolsError, BuiltTools, build_tools, mcp_tool_approvals,
    probe_tool_sandbox,
};
pub use bus::{BusSubscription, ClientGone, EventBus, EventSink};
pub use documents::{
    DocumentIndex, DocumentsUnavailable, IndexDeps, IndexPasses, IndexProgress,
    MAX_DOCUMENT_CHUNKS, PassCounts, PassEnd, PassProgress, RootPass, SkipCause, document_indexing,
    remove_dropped_roots, system_prompt,
};
pub use recovery::{
    Component, StatusSeverity, drain_join_set, install_panic_hook, spawn_supervised,
};
pub use relay::{RelayDeps, sweep_relay_answers};

pub use presence::{PresenceError, PresenceManager, RequestGuard, VramShortfall, WakeRoom};
pub use state::{
    AppState, BranchService, DispatchError, DocumentsService, McpAccountError, McpAccounts,
    McpHost, McpLogin, McpServerStatus, McpService, MemoryStack, Query, Requester, StatusService,
    Subsystems, TurnOrigin, TurnService, VoiceService, branch_replay,
};
pub use subagent::{RelayReport, SubAgentError, SubAgentRunner};
pub use tasks::{TaskError, TaskRunner, TaskService};
pub use transcript::SessionContext;
pub use vision::VisionRevalidator;
pub use vram::VramLedger;
