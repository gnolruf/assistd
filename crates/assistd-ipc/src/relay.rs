//! A relay's plan, the progress of its sub-agent steps, and the stored trace of what each step
//! did.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::ToolReport;

/// One step of a relay's plan, as announced before any step runs.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RelayStepPlan {
    pub task: String,
    /// The name of the kind of sub-agent the step runs, such as `web_research`.
    pub agent: String,
}

/// Progress of the relay inside the turn's running `relay` tool call: `Planned`
/// once, then for each step `Started`, its sub-agent's output, then `Answered`
/// or `Failed`. Every event after `Started` belongs to the step it started.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RelayEvent {
    /// The validated plan; a single-task relay is a plan of one step.
    Planned { steps: Vec<RelayStepPlan> },
    /// Step `step` of the plan, counted from 1, handed its task to a sub-agent.
    Started { step: u32 },
    /// A streamed chunk of the sub-agent's reply.
    Delta { text: String },
    /// A streamed chunk of the sub-agent's reasoning.
    ReasoningDelta { text: String },
    /// Tokens the sub-agent's history has held, counting what compaction removed.
    ContextSize { total: u32 },
    /// The sub-agent invoked a tool.
    ToolCall {
        name: String,
        args: serde_json::Value,
    },
    /// A sub-agent tool finished; `result` carries no attachments.
    ToolResult { name: String, result: ToolReport },
    /// The sub-agent's history outgrew its budget and is being compacted.
    Compacting,
    /// The compaction `Compacting` announced finished after `duration_ms`.
    Compacted { duration_ms: u64 },
    /// The sub-agent answered in `answer_chars` characters, saved to
    /// `answer_file`, after `tool_steps` tool-calling steps.
    Answered {
        answer_chars: u32,
        tool_steps: u32,
        answer_file: PathBuf,
    },
    /// The step ended without a report.
    Failed { message: String },
}

/// One step of a stored relay, replayed with the `relay` tool row it belongs to.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RelayStepRecord {
    pub task: String,
    /// The name of the kind of sub-agent the step ran.
    pub agent: String,
    /// How the step ran; `None` when the plan ended before it started.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run: Option<RelayStepRun>,
}

/// A stored relay step that started: how it ended, its cost, and what it did in order.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RelayStepRun {
    pub end: RelayStepEnd,
    pub duration_ms: u64,
    /// Tokens its history had held when it ended, once reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokens: Option<u32>,
    pub entries: Vec<RelayTraceEntry>,
}

/// How a stored relay step ended.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RelayStepEnd {
    /// It reported.
    Answered,
    Failed {
        message: String,
    },
}

/// One thing a stored relay step's sub-agent did.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RelayTraceEntry {
    Text {
        text: String,
    },
    Reasoning {
        text: String,
        duration_ms: u64,
    },
    /// A tool call; `outcome` is `None` when it never finished.
    Tool {
        name: String,
        args: serde_json::Value,
        output: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        outcome: Option<RelayToolOutcome>,
    },
    /// A history compaction; `duration_ms` is `None` when it never finished.
    Compaction {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        duration_ms: Option<u64>,
    },
}

/// How a stored relay tool call ended.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct RelayToolOutcome {
    pub exit_code: i64,
    pub duration_ms: u64,
}
