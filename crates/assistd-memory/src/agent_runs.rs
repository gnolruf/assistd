//! A sub-agent's run as stored: how it went, and its history as messages of its own,
//! which belong to the tool result of the call that started it and to no branch.

use std::path::PathBuf;

use assistd_ipc::Access;

use crate::{BranchSummary, PersistedMessage};

/// One sub-agent run a tool call planned, in plan order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PersistedAgentRun {
    pub task: String,
    /// The name of the kind of sub-agent the run is of; never empty.
    pub agent: String,
    /// What the run did; `None` when the plan ended before it started.
    pub record: Option<AgentRunRecord>,
}

/// A sub-agent run that started: how it ended, what it cost, and its history.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentRunRecord {
    pub end: AgentRunEnd,
    pub duration_ms: u64,
    /// Tokens its history had held when it ended, once reported.
    pub tokens: Option<u32>,
    /// Its history in order, opened by the task as it was handed over.
    pub messages: Vec<RunMessage>,
    /// The latest compaction summary of its history; its seqs count `messages` from 0.
    pub summary: Option<BranchSummary>,
}

/// How a sub-agent run ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgentRunEnd {
    /// It reported, its whole answer saved to `answer_file`.
    Answered {
        answer_file: PathBuf,
    },
    Failed {
        message: String,
    },
}

/// One message of a sub-agent's history.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunMessage {
    pub message: PersistedMessage,
    /// What the call behind a tool result touched, as the run's tools saw it.
    pub accesses: Vec<Access>,
}
