//! Status updates: the subsystem one is attributed to, what it reports and how severe it is.

use std::fmt;

use serde::{Deserialize, Serialize};

/// Severity of an [`EventBody::Status`](crate::EventBody::Status) update; maps 1:1 to a `tracing`
/// level.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum StatusSeverity {
    Info,
    Warning,
    Error,
}

impl StatusSeverity {
    /// The wire spelling.
    pub fn as_str(self) -> &'static str {
        match self {
            StatusSeverity::Info => "info",
            StatusSeverity::Warning => "warning",
            StatusSeverity::Error => "error",
        }
    }
}

impl fmt::Display for StatusSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad(self.as_str())
    }
}

/// Daemon subsystem an [`EventBody::Status`](crate::EventBody::Status) update is attributed to.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Component {
    Agent,
    Llm,
    Mcp,
    Voice,
    Memory,
    Wm,
    Embed,
    Daemon,
    IdleMonitor,
    GpuMonitor,
    ListenDispatcher,
}

impl Component {
    /// The wire spelling.
    pub fn as_str(self) -> &'static str {
        match self {
            Component::Agent => "agent",
            Component::Llm => "llm",
            Component::Mcp => "mcp",
            Component::Voice => "voice",
            Component::Memory => "memory",
            Component::Wm => "wm",
            Component::Embed => "embed",
            Component::Daemon => "daemon",
            Component::IdleMonitor => "idle_monitor",
            Component::GpuMonitor => "gpu_monitor",
            Component::ListenDispatcher => "listen_dispatcher",
        }
    }
}

impl fmt::Display for Component {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad(self.as_str())
    }
}

/// What an [`EventBody::Status`](crate::EventBody::Status) update reports.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StatusKind {
    /// The LLM server died mid-turn and is being restarted.
    Restarting,
    /// The LLM server is back; the interrupted step is being replayed.
    Replaying,
    /// Recovery failed; the turn will not complete.
    Degraded,
    /// The tool schema was withdrawn so the model answers from what it has.
    ToolsWithdrawn,
    /// The model's response hit the output-token limit and was discarded;
    /// the step is being retried.
    OutputTruncated,
    /// The model is still downloading or loading after a wake.
    ModelLoading,
    /// A subsystem failed to start and is unavailable this run.
    StartupFailed,
    /// History is close enough to the token budget that an upcoming step
    /// will summarize it.
    CompactingSoon,
    /// Older history is being summarized to fit the token budget.
    CompactingHistory,
    /// The compaction announced by [`StatusKind::CompactingHistory`] finished.
    HistoryCompacted,
    /// A task run gave the model to the user's interaction; the step it was streaming
    /// was discarded and runs again when the task resumes.
    YieldedToUser,
}
