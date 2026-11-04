//! Wire types for saved tasks: a prompt run unattended when one of its
//! triggers fires, and the state of each of its runs.

use std::fmt;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Longest task name accepted, in characters.
pub const MAX_TASK_NAME_CHARS: usize = 80;

/// Largest task prompt accepted, in bytes.
pub const MAX_TASK_PROMPT_BYTES: usize = 16 * 1024;

/// A saved task's row id.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TaskId(pub i64);

impl fmt::Display for TaskId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// A task run's row id.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TaskRunId(pub i64);

impl fmt::Display for TaskRunId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// Why a [`TaskName`] or [`TaskPrompt`] was refused.
#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum TaskSpecError {
    #[error("task name is empty")]
    EmptyName,

    #[error("task name is longer than {MAX_TASK_NAME_CHARS} characters")]
    NameTooLong,

    #[error("task prompt is empty")]
    EmptyPrompt,

    #[error("task prompt is {0} bytes; the limit is {MAX_TASK_PROMPT_BYTES}")]
    PromptTooLarge(usize),
}

/// A task's display name: trimmed, non-empty, at most [`MAX_TASK_NAME_CHARS`] characters.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct TaskName(String);

impl TaskName {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for TaskName {
    type Error = TaskSpecError;

    fn try_from(name: String) -> Result<Self, Self::Error> {
        let trimmed = name.trim();
        if trimmed.is_empty() {
            return Err(TaskSpecError::EmptyName);
        }
        if trimmed.chars().count() > MAX_TASK_NAME_CHARS {
            return Err(TaskSpecError::NameTooLong);
        }
        Ok(Self(trimmed.to_string()))
    }
}

impl From<TaskName> for String {
    fn from(name: TaskName) -> Self {
        name.0
    }
}

impl fmt::Display for TaskName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad(&self.0)
    }
}

/// The text a task sends the model: not blank, at most [`MAX_TASK_PROMPT_BYTES`] bytes.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct TaskPrompt(String);

impl TaskPrompt {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for TaskPrompt {
    type Error = TaskSpecError;

    fn try_from(prompt: String) -> Result<Self, Self::Error> {
        if prompt.trim().is_empty() {
            return Err(TaskSpecError::EmptyPrompt);
        }
        if prompt.len() > MAX_TASK_PROMPT_BYTES {
            return Err(TaskSpecError::PromptTooLarge(prompt.len()));
        }
        Ok(Self(prompt))
    }
}

impl From<TaskPrompt> for String {
    fn from(prompt: TaskPrompt) -> Self {
        prompt.0
    }
}

/// A saved task: its prompt, what starts it, and the approvals made for that prompt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskSpec {
    pub name: TaskName,
    pub prompt: TaskPrompt,
    #[serde(default)]
    pub triggers: Vec<TaskTrigger>,
    pub enabled: bool,
    /// Approvals runs of this task may use; cleared whenever the prompt changes.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub grants: Vec<TaskGrant>,
    /// The approved plan; cleared whenever the prompt changes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plan: Option<TaskPlan>,
}

/// A condition that starts a run of its task.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TaskTrigger {
    /// Started on request.
    Manual,
}

/// One approval a task run may use in place of asking.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TaskGrant {
    /// Run `program`, limited to `subcommand` when set.
    Program {
        program: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        subcommand: Option<String>,
    },
    /// Reach exactly `host` over the web.
    WebHost { host: String },
    /// Call the MCP tool with this full name.
    McpTool { name: String },
    /// Launch the app with this desktop entry id.
    App { desktop_entry: String },
    /// Write files under the directory `under`.
    Write { under: PathBuf },
}

/// An approved plan for a task's prompt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskPlan {
    pub summary: String,
    pub steps: Vec<PlanStep>,
}

/// One step of a [`TaskPlan`] and the approvals it expects to need.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanStep {
    pub description: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub actions: Vec<TaskGrant>,
}

/// Where a task run has got; only a finished run carries an outcome.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum TaskRunState {
    Queued,
    Running,
    Suspended {
        reason: SuspendReason,
    },
    /// Paused on a confirmation nobody was there to answer.
    NeedsInput,
    Succeeded,
    Failed {
        reason: String,
    },
    Cancelled,
}

/// The task run a transcript session holds, as listed beside the session.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskRunSummary {
    pub task: TaskId,
    pub run: TaskRunId,
    pub task_name: TaskName,
    pub state: TaskRunState,
}

/// Why a task run is suspended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SuspendReason {
    /// The user's conversation holds the model.
    UserConversation,
    /// Another process needs the GPU.
    GpuBusy,
}
