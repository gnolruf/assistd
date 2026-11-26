//! What an agent's run streams inside the daemon, and the id that says which agent of a
//! turn an event comes from.

use std::fmt;
use std::num::NonZeroUsize;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    Access, Component, RelayStepEnd, RelayStepPlan, StatusKind, StatusSeverity, ToolReport,
};

/// A sub-agent of one turn, numbered from 1 in the order the turn started them.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(transparent)]
pub struct AgentId(pub u32);

impl fmt::Display for AgentId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// One tool call the model requested during a step.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolCall {
    /// Server-assigned call ID, echoed back on the matching result.
    pub id: String,
    pub name: String,
    pub arguments: Value,
}

/// What a compaction summary leaves of the history: its `kept_turns`
/// newest user turns, the oldest of them cut to its `kept_steps` newest tool
/// steps when its earlier steps were folded into its call log.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompactionCut {
    pub kept_turns: NonZeroUsize,
    pub kept_steps: Option<NonZeroUsize>,
}

/// Events an agent's run streams, from its backend and its loop.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LlmEvent {
    /// A model step is about to stream; what follows up to the step's tool
    /// calls or `Done` is that step's output.
    StepStarted,
    /// What the step in flight streamed so far is void: the step runs again
    /// or the turn ends without it.
    StepDiscarded,
    /// A streamed chunk of model output.
    Delta { text: String },
    /// A streamed chunk of the model's reasoning; never spoken.
    ReasoningDelta { text: String },
    /// Every tool call the model requested in one step, in request
    /// order; emitted once, before the first of them runs.
    ToolCallsRequested { calls: Vec<ToolCall> },
    /// The model asked to invoke a tool; emitted before the tool runs.
    ToolCall {
        id: String,
        name: String,
        arguments: Value,
    },
    /// A tool finished with `result`.
    ToolResult {
        id: String,
        name: String,
        result: Box<ToolReport>,
        /// Whether the call could put data from this machine in `result`.
        read_host: bool,
        /// What the call touched.
        accesses: Vec<Access>,
    },
    /// The `relay` tool call in flight validated its plan; a single-task relay is a plan
    /// of one step.
    RelayPlanned { steps: Vec<RelayStepPlan> },
    /// Step `step` of the running relay's plan, counted from 1, handed its task to
    /// sub-agent `agent`.
    RelayStepStarted { step: u32, agent: AgentId },
    /// Step `step` of the running relay's plan ended, `duration_ms` after it started.
    RelayStepEnded {
        step: u32,
        end: RelayStepEnd,
        duration_ms: u64,
    },
    /// The task a sub-agent's run was handed, streamed before its first step.
    Task { text: String },
    /// `event` as sub-agent `agent` streamed it, inside tool call `call_id` of the agent
    /// whose stream this is.
    FromAgent {
        agent: AgentId,
        call_id: String,
        event: Box<LlmEvent>,
    },
    /// Tokens in the history as sent with the request about to stream,
    /// and `total` the history has held, counting what compaction removed.
    ContextSize { tokens: u32, total: u32 },
    /// Non-terminal recovery or status update.
    Status {
        severity: StatusSeverity,
        component: Component,
        event: StatusKind,
        message: String,
    },
    /// History was compacted: everything `cut` leaves out is now represented
    /// by `summary`, which took `elapsed` to make.
    HistoryCompacted {
        summary: String,
        cut: CompactionCut,
        elapsed: Duration,
    },
    /// The turn was cut off before the model finished; `reply` is what the
    /// history keeps as its answer. No `Done` follows.
    Interrupted { reply: String },
    /// The model has finished generating.
    Done,
}
