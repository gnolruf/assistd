//! Wire types for the daemon's Unix-socket protocol: line-delimited JSON,
//! with [`Request`] lines from the client and, after one [`Hello`], [`Event`] lines back.

use serde::{Deserialize, Serialize};

pub mod attachment;
#[cfg(feature = "client")]
pub mod client;
mod documents;
mod event;
pub mod mcp;
mod permission;
mod presence;
mod readiness;
mod relay;
mod request;
pub mod socket;
mod status;
pub mod task;
pub mod tool_report;
mod voice;
pub mod wakeword;

pub use attachment::{Attachment, ImageAttachment, ImageMime};
#[cfg(feature = "client")]
pub use client::{DialogConnection, EventStream, IpcClient, IpcClientError};
pub use documents::{
    DocumentActivity, DocumentPass, DocumentPassCounts, DocumentPassEnd, DocumentRootStatus,
    DocumentSkipReason, SkippedFiles,
};
pub use event::{Event, EventBody, EventKind, ReindexKind, Role, TurnStats};
pub use mcp::{McpServerState, McpTransportKind, TokenRevocation};
pub use permission::{ConfirmSource, PermissionMode, PlanMode, PlanVerdict};
pub use presence::{DownloadProgress, PresenceState, PresenceTarget};
pub use readiness::{ComponentReadiness, StartupComponent, StartupReadiness};
pub use relay::{
    RelayEvent, RelayStepEnd, RelayStepPlan, RelayStepRecord, RelayStepRun, RelayToolOutcome,
    RelayTraceEntry,
};
pub use request::{Request, RequestBody, SubscribeFilter};
pub use socket::socket_path;
pub use status::{Component, StatusKind, StatusSeverity};
pub use task::{
    ActionOutlook, AnticipatedAction, DEFAULT_EVENT_COOLDOWN_SECS, DEFAULT_PATH_DEBOUNCE_SECS,
    FinishOutcome, HostName, InteractionHold, KeptRuns, LaneHolder, MAX_KEPT_RUNS, MAX_LISTED_RUNS,
    MissedFire, ModeRecommendation, PausedPrompt, Plan, PlanId, PlanStep, PlannedAction, QueueItem,
    SuspendReason, TaskEvent, TaskId, TaskName, TaskPrompt, TaskQueueOp, TaskQueueSnapshot,
    TaskRequest, TaskRunId, TaskRunRecord, TaskRunState, TaskRunSummary, TaskSpec, TaskSpecError,
    TaskTrigger,
};
pub use tool_report::{ToolAttachment, ToolReport, ToolStreams};
pub use voice::{ListenPhase, VoiceCaptureState};
pub use wakeword::WakeWordOp;

/// The assistd version this build speaks the protocol of; a daemon and its clients must
/// share it.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// The line the daemon writes when it accepts a connection, ahead of every [`Event`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Hello {
    /// The daemon's [`VERSION`].
    pub daemon_version: String,
}

impl Hello {
    /// The hello of a daemon from this build.
    pub fn current() -> Self {
        Self {
            daemon_version: VERSION.to_owned(),
        }
    }
}

#[cfg(test)]
mod tests;
