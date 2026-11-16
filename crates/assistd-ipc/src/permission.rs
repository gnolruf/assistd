//! How a turn's confirmation prompts are answered, whether it plans before acting, and which
//! agent a prompt comes from.

use serde::{Deserialize, Serialize};

use crate::{TaskId, TaskRunId};

/// How a query connection's confirmation prompts are answered.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PermissionMode {
    /// Every prompt goes to the client.
    #[default]
    Manual,
    /// Edits of ordinary paths inside the edit scope run without asking; the rest go to the
    /// client.
    AcceptEdits,
    /// As [`PermissionMode::AcceptEdits`], and the safety judge answers the prompts it may;
    /// offered only when the daemon allows it.
    Auto,
}

/// Whether a query's turn may plan before it acts.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PlanMode {
    /// The turn acts and cannot plan.
    #[default]
    Off,
    /// The turn acts, and the model may ask the user to plan first.
    Offered,
    /// The turn starts planning: it runs only what asks for nothing until the user accepts
    /// its plan.
    Planning,
}

impl PlanMode {
    pub fn is_off(&self) -> bool {
        *self == PlanMode::Off
    }
}

/// The user's answer to an [`EventBody::PlanReview`](crate::EventBody::PlanReview).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "verdict", rename_all = "snake_case")]
pub enum PlanVerdict {
    /// Leave plan mode and carry the plan out, answering prompts under `mode`.
    Accept { mode: PermissionMode },
    /// Stay in plan mode; with `feedback`, revise the plan by it, else wait for the user.
    KeepPlanning {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        feedback: Option<String>,
    },
}

/// Who raised an [`EventBody::ConfirmRequest`](crate::EventBody::ConfirmRequest).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "agent", rename_all = "snake_case")]
pub enum ConfirmSource {
    /// The main backend's agent.
    #[default]
    Main,
    /// A relay's sub-agent, working from untrusted content on `task`, inside task run `run`
    /// when one relayed it.
    Relay {
        task: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        run: Option<TaskRunId>,
    },
    /// Run `run` of saved task `task`, with no user present.
    Task { task: TaskId, run: TaskRunId },
}

impl ConfirmSource {
    pub fn is_main(&self) -> bool {
        matches!(self, ConfirmSource::Main)
    }
}
