//! What a `relay` tool call's sub-agents did, stored beside the call's tool
//! result and never part of the conversation the model sees.

use crate::ToolOutcome;

/// One step of a relay's plan as stored, in plan order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PersistedRelayStep {
    pub task: String,
    pub profile: PersistedRelayProfile,
    /// How the step ran; `None` when the plan ended before it started.
    pub run: Option<PersistedRelayRun>,
}

/// What a relay step's sub-agent could reach.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PersistedRelayProfile {
    Read,
    Fetch,
}

/// A relay step that started: how it ended, what it did, and what it cost.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PersistedRelayRun {
    pub end: PersistedRelayEnd,
    pub duration_ms: u64,
    /// Tokens its conversation had held when it ended, once reported.
    pub tokens: Option<u32>,
    pub entries: Vec<PersistedRelayEntry>,
}

/// How a relay step ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PersistedRelayEnd {
    /// It reported; `truncated` when its report was cut to fit.
    Answered {
        truncated: bool,
    },
    Failed {
        message: String,
    },
}

/// One thing a sub-agent did, in the order it happened.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PersistedRelayEntry {
    /// Reply text it streamed; never empty.
    Text(String),
    /// Reasoning it streamed, never empty, and how long that took.
    Reasoning { text: String, duration_ms: u64 },
    /// A tool call; `outcome` is `None` when it never finished.
    Tool {
        name: String,
        arguments: serde_json::Value,
        output: String,
        outcome: Option<ToolOutcome>,
    },
    /// A compaction of its history; `duration_ms` is `None` when it never finished.
    Compaction { duration_ms: Option<u64> },
}
