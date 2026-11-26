//! A relay's plan and how each of its steps ended.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// One step of a relay's plan, as announced before any step runs.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RelayStepPlan {
    pub task: String,
    /// The name of the kind of sub-agent the step runs, such as `web_research`.
    pub agent: String,
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
