//! A relay's plan, how each of its steps ended, and the stored trace of what each step did.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// One step of a relay's plan, as announced before any step runs.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RelayStepPlan {
    pub task: String,
    /// The name of the kind of sub-agent the step runs, such as `web_research`.
    pub agent: String,
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

/// How a relay step ended.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RelayStepEnd {
    /// It reported, and what it wrote is saved whole in `answer_file`.
    Answered { answer_file: PathBuf },
    /// It ended without a report.
    Failed { message: String },
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
