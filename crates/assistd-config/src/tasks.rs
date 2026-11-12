//! Saved-task configuration: how a user message meets a task holding the model.

use serde::{Deserialize, Serialize};

use crate::defaults::{DEFAULT_TASKS_KEPT_RUNS, DEFAULT_TASKS_RESUME_AFTER_IDLE_SECS};
use crate::settings::{DescribeSettings, choice_setting, describe_fields};

/// What a user message does when a task run holds the model's only lane. With
/// more than one lane the two run at once and this never applies.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TaskInterruption {
    /// The run suspends at its next model step, a tool call in flight finishing
    /// first, and resumes once the conversation ends.
    #[default]
    Interrupt,
    /// The message waits for the running task to finish, then goes before the queue.
    AfterTask,
    /// The message joins the back of the task queue.
    AfterQueue,
}

choice_setting!(TaskInterruption {
    Interrupt => "interrupt",
    AfterTask => "after_task",
    AfterQueue => "after_queue",
});

/// How saved tasks share the model with the user's conversation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct TasksConfig {
    pub interruption: TaskInterruption,
    /// Seconds after a reply finishes, speech included, with no new user turn
    /// before the conversation ends and the queue resumes, `0..=3600`. The
    /// wake word's follow-up window keeps it open at least that long; `0` ends it
    /// as soon as the reply does.
    pub resume_after_idle_secs: u32,
    /// Finished runs kept per task that sets no count of its own, `1..=10000`, with their
    /// transcripts; older runs are deleted at startup and whenever a run ends.
    pub kept_runs: u32,
}

impl DescribeSettings for TasksConfig {
    describe_fields! {
        interruption => (
            "when a message arrives mid-task",
            "interrupt suspends the task; after_task or after_queue makes the message wait"
        ),
        resume_after_idle_secs => (
            "resume tasks after idle (s)",
            "Seconds without a new message before held tasks resume"
        ),
        kept_runs => (
            "finished runs kept per task",
            "Older runs and their transcripts are deleted"
        ),
    }
}

impl Default for TasksConfig {
    fn default() -> Self {
        Self {
            interruption: TaskInterruption::default(),
            resume_after_idle_secs: DEFAULT_TASKS_RESUME_AFTER_IDLE_SECS,
            kept_runs: DEFAULT_TASKS_KEPT_RUNS,
        }
    }
}
