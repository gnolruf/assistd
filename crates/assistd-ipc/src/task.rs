//! Wire types for saved tasks: a prompt run unattended when one of its
//! triggers fires, and the state of each of its runs.

use std::fmt;
use std::path::PathBuf;

use chrono::{DateTime, NaiveDateTime, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Longest task name accepted, in characters.
pub const MAX_TASK_NAME_CHARS: usize = 80;

/// Largest task prompt accepted, in bytes.
pub const MAX_TASK_PROMPT_BYTES: usize = 16 * 1024;

/// Least time between two fires of one task's event triggers when none is given, in seconds.
pub const DEFAULT_EVENT_COOLDOWN_SECS: u32 = 60;

/// How long a burst of path changes is gathered into one fire when not given, in seconds.
pub const DEFAULT_PATH_DEBOUNCE_SECS: u32 = 2;

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
    /// The approved plan, holding only its accepted actions; cleared whenever the prompt
    /// changes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plan: Option<Plan>,
}

/// A condition that starts a run of its task.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TaskTrigger {
    /// Started on request.
    Manual,
    /// Fires once at local wall-clock time `at`; a time a DST gap skips is read at the
    /// offset in force before the gap.
    At { at: NaiveDateTime },
    /// Recurs at every local time matching the five-field cron expression `cron`.
    Schedule {
        cron: String,
        #[serde(default)]
        missed: MissedFire,
    },
    /// Fires when `event` happens, at most once per `cooldown_secs` seconds for its task.
    Event {
        event: TaskEvent,
        #[serde(default = "default_cooldown_secs")]
        cooldown_secs: u32,
    },
}

/// Something that happened which an [`TaskTrigger::Event`] fires on.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TaskEvent {
    /// The daemon started and its model first became ready.
    DaemonStarted,
    /// logind's `LockedHint` on the user's graphical session turned false.
    SessionUnlocked,
    /// A file was created, modified or moved in at or under `path`, which must exist and
    /// is tilde-expanded at save; a burst within `debounce_secs` seconds fires once.
    PathChanged {
        path: PathBuf,
        #[serde(default = "default_debounce_secs")]
        debounce_secs: u32,
    },
    /// A run of task `task` finished with `outcome`; a chain that loops is refused at save.
    TaskFinished {
        task: TaskId,
        #[serde(default)]
        outcome: FinishOutcome,
    },
}

/// Which endings of the upstream run fire a [`TaskEvent::TaskFinished`]; a cancelled run
/// fires none.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishOutcome {
    Succeeded,
    Failed,
    #[default]
    Either,
}

/// What a schedule does about fire times that passed while the daemon was down
/// or the machine was suspended.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MissedFire {
    /// Fire once on return, however many fire times passed.
    #[default]
    RunOnce,
    Skip,
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

/// A plan's id, fresh for every plan the planner makes.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct PlanId(pub String);

impl fmt::Display for PlanId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad(&self.0)
    }
}

/// How the agent means to do a piece of work, and the approvals it expects to need.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Plan {
    pub id: PlanId,
    pub summary: String,
    pub steps: Vec<PlanStep>,
}

/// One step of a [`Plan`] and the approvals it expects to need.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanStep {
    pub description: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub actions: Vec<AnticipatedAction>,
}

/// An approval a plan expects to need; approving it makes `grant` a grant.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnticipatedAction {
    pub grant: TaskGrant,
    /// Why the daemon expects the grant to cover nothing, such as a program missing from
    /// `PATH`; `None` when it passed every check.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub problem: Option<String>,
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

/// A change to the task queue, sent as [`crate::Request::TaskQueue`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TaskQueueOp {
    /// Place waiting run `run` at index `to` among the waiting runs, 0 starting next;
    /// an index past the end places it last.
    Move { run: TaskRunId, to: u32 },
    /// Drop waiting run `run`, or stop it if it is running.
    Cancel { run: TaskRunId },
    /// Drop the user turn queued by the query with request id `id`, which then ends
    /// with `Done`.
    CancelTurn { id: String },
    /// Start no more runs until [`TaskQueueOp::Resume`]; running ones carry on.
    Pause,
    /// Start runs again, ending the user's conversation if it holds the queue.
    Resume,
}

/// The task queue: its unfinished runs and waiting user turns, what holds each model
/// lane, and whether it is paused or held for the user's conversation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskQueueSnapshot {
    /// Running runs in the order they started, then waiting items in the order they start.
    pub items: Vec<QueueItem>,
    /// One entry per lane the model can decode at once; lane 0 is the main conversation's.
    pub lanes: Vec<LaneHolder>,
    pub paused: bool,
    /// Set while the user's conversation keeps runs from the only lane.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub held: Option<ConversationHold>,
}

/// One unfinished item in a [`TaskQueueSnapshot`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum QueueItem {
    Run {
        run: TaskRunId,
        task: TaskId,
        task_name: TaskName,
        state: TaskRunState,
    },
    /// A user message waiting its turn behind the queue, by its query's request id.
    UserTurn { id: String, preview: String },
}

/// The user's conversation holding the task queue.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConversationHold {
    /// When the queue resumes unless another user turn arrives; `None` while a turn is
    /// in flight.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resumes_at: Option<DateTime<Utc>>,
}

/// What holds one model lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "holder", rename_all = "snake_case")]
pub enum LaneHolder {
    /// The main conversation, whether or not a turn is running in it.
    Main,
    Task {
        run: TaskRunId,
    },
    Idle,
}

fn default_cooldown_secs() -> u32 {
    DEFAULT_EVENT_COOLDOWN_SECS
}

fn default_debounce_secs() -> u32 {
    DEFAULT_PATH_DEBOUNCE_SECS
}
