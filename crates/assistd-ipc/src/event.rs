//! The lines the daemon sends back: [`Event`], what its [`EventBody`] reports, and the kinds a
//! subscriber selects.

use std::fmt;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::{
    Component, ComponentReadiness, ConfirmSource, DocumentActivity, DocumentPass,
    DocumentPassCounts, DocumentRootStatus, ListenPhase, McpServerState, McpTransportKind,
    ModeRecommendation, PageTranscriptionStatus, PermissionMode, Plan, PresenceState, RelayEvent,
    RelayStepRecord, StartupComponent, StatusKind, StatusSeverity, TaskId, TaskName, TaskPrompt,
    TaskQueueSnapshot, TaskRunId, TaskRunRecord, TaskRunState, TaskRunSummary, TaskSpec,
    TokenRevocation, ToolReport, VoiceCaptureState,
};

/// Kinds of [`Event`] carried on the daemon-wide broadcast bus, selectable by a
/// [`SubscribeFilter`](crate::SubscribeFilter). Dialog-local events have no variant.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum EventKind {
    Delta,
    ReasoningDelta,
    ToolCall,
    ToolResult,
    Presence,
    ListenState,
    WakeWordDetected,
    VoiceState,
    SpeakingState,
    SessionTitle,
    Done,
    Error,
    LastDelta,
    Transcription,
    ChatFocus,
    Readiness,
    ContextSize,
    Relay,
    TaskQueue,
    TaskReply,
    DocumentActivity,
}

/// Author of a persisted transcript message.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
    Tool,
}

impl Role {
    /// The wire spelling.
    pub fn as_str(self) -> &'static str {
        match self {
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        }
    }
}

impl fmt::Display for Role {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad(self.as_str())
    }
}

/// How long a finished turn ran and the tokens it produced.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct TurnStats {
    pub duration_ms: u64,
    /// Tokens the turn added to the history.
    pub tokens: u64,
}

/// Which table a [`EventBody::ReindexProgress`] item belongs to.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReindexKind {
    Chunks,
    Memories,
}

impl ReindexKind {
    /// The wire spelling.
    pub fn as_str(self) -> &'static str {
        match self {
            ReindexKind::Chunks => "chunks",
            ReindexKind::Memories => "memories",
        }
    }
}

impl fmt::Display for ReindexKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad(self.as_str())
    }
}

/// Daemon-to-client event, one JSON line: `id` beside the fields of `body`. A response stream
/// ends with exactly one terminal [`EventBody::Done`] or [`EventBody::Error`]; a
/// [`RequestBody::Subscribe`](crate::RequestBody::Subscribe) stream never ends.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Event {
    /// The id of the request this event responds to.
    pub id: String,
    #[serde(flatten)]
    pub body: EventBody,
}

impl Event {
    pub fn new(id: impl Into<String>, body: EventBody) -> Self {
        Self {
            id: id.into(),
            body,
        }
    }

    /// Returns true if this event terminates a response stream.
    pub fn is_terminal(&self) -> bool {
        matches!(self.body, EventBody::Done | EventBody::Error { .. })
    }

    /// The broadcast kind of this event, or `None` for dialog-local events.
    pub fn kind(&self) -> Option<EventKind> {
        Some(match self.body {
            EventBody::Delta { .. } => EventKind::Delta,
            EventBody::ReasoningDelta { .. } => EventKind::ReasoningDelta,
            EventBody::ToolCall { .. } => EventKind::ToolCall,
            EventBody::ToolResult { .. } => EventKind::ToolResult,
            EventBody::Presence { .. } => EventKind::Presence,
            EventBody::VoiceState { .. } => EventKind::VoiceState,
            EventBody::ListenState { .. } => EventKind::ListenState,
            EventBody::WakeWordDetected => EventKind::WakeWordDetected,
            EventBody::SpeakingState { .. } => EventKind::SpeakingState,
            EventBody::SessionTitle { .. } => EventKind::SessionTitle,
            EventBody::Done => EventKind::Done,
            EventBody::Error { .. } => EventKind::Error,
            EventBody::LastDelta { .. } => EventKind::LastDelta,
            EventBody::Transcription { .. } => EventKind::Transcription,
            EventBody::ChatFocus { .. } => EventKind::ChatFocus,
            EventBody::Readiness { .. } => EventKind::Readiness,
            EventBody::ContextSize { .. } => EventKind::ContextSize,
            EventBody::Relay { .. } => EventKind::Relay,
            EventBody::TaskQueue { .. } => EventKind::TaskQueue,
            EventBody::TaskReply { .. } => EventKind::TaskReply,
            EventBody::DocumentActivity { .. } => EventKind::DocumentActivity,
            EventBody::VoiceOutputState { .. }
            | EventBody::WakeWordStatus { .. }
            | EventBody::WakeWordTest { .. }
            | EventBody::WakeWordSpeech { .. }
            | EventBody::SemanticHit { .. }
            | EventBody::MemoryValue { .. }
            | EventBody::MemoryRow { .. }
            | EventBody::MemoryForgetResult { .. }
            | EventBody::ReindexProgress { .. }
            | EventBody::DocumentIndex { .. }
            | EventBody::DocumentRoot { .. }
            | EventBody::DocumentPassProgress { .. }
            | EventBody::DocumentPassEnded { .. }
            | EventBody::DocumentHit { .. }
            | EventBody::ConfirmRequest { .. }
            | EventBody::PlanReview { .. }
            | EventBody::PlanModeEntered
            | EventBody::Capabilities { .. }
            | EventBody::Status { .. }
            | EventBody::BranchInfo { .. }
            | EventBody::BranchSwitched { .. }
            | EventBody::HistoryEntry { .. }
            | EventBody::UndoApplied { .. }
            | EventBody::Task { .. }
            | EventBody::UndecodableTask { .. }
            | EventBody::TaskRunRecord { .. }
            | EventBody::TaskRunQueued { .. }
            | EventBody::Queued { .. }
            | EventBody::TaskRefined { .. }
            | EventBody::TriggerFires { .. }
            | EventBody::McpAuthorizeUrl { .. }
            | EventBody::McpServer { .. }
            | EventBody::McpLoggedOut { .. } => return None,
        })
    }
}

/// What an [`Event`] reports, tagged by `"type"`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EventBody {
    /// Chunk of reply text.
    Delta { text: String },
    /// Chunk of model reasoning, separate from the reply.
    ReasoningDelta { text: String },
    /// The model invoked a tool.
    ToolCall {
        name: String,
        args: serde_json::Value,
    },
    /// A tool the model invoked finished.
    ToolResult { name: String, result: ToolReport },
    /// Current presence state.
    Presence { state: PresenceState },
    /// Push-to-talk capture state transition.
    VoiceState { state: VoiceCaptureState },
    /// Transcript of a PTT recording; empty means silence, and only `Done` follows.
    Transcription { text: String },
    /// The wake-word listener's phase.
    ListenState { phase: ListenPhase },
    /// The wake phrase was heard; a command's `Transcription` may follow.
    WakeWordDetected,
    /// Whether TTS is enabled.
    VoiceOutputState { enabled: bool },
    /// The configured wake phrase; `listening` is whether the wake-word
    /// listener is running.
    WakeWordStatus { phrase: String, listening: bool },
    /// Whether the wake phrase was spotted in one utterance of a test session.
    WakeWordTest { matched: bool },
    /// Speech started (`true`) or ended (`false`) on the mic of the
    /// wake-word session opened by this connection.
    WakeWordSpeech { speaking: bool },
    /// How far one background subsystem has started; broadcast whenever
    /// one finishes starting.
    Readiness {
        component: StartupComponent,
        state: ComponentReadiness,
    },
    /// Whether a live chat has keyboard focus; broadcast when it changes.
    ChatFocus { focused: bool },
    /// TTS playback for a turn started (`true`) or drained (`false`).
    SpeakingState { speaking: bool },
    /// The session's display title, sent after each successful turn; may arrive after its `Done`.
    SessionTitle { session_id: String, title: String },
    /// One search hit; `content` is the matched chunk of message `message_id`, `similarity` is in
    /// `[0.0, 1.0]`.
    SemanticHit {
        message_id: i64,
        chunk_id: i64,
        session_id: String,
        timestamp: String,
        role: Role,
        content: String,
        similarity: f32,
    },
    /// Result of `MemoryLoad`; `value` is `None` when the key is absent.
    MemoryValue { key: String, value: Option<String> },
    /// One `MemoryListAll` row, in key order; `memory_id` is what `MemoryForget` takes.
    MemoryRow {
        memory_id: i64,
        key: String,
        value: String,
    },
    /// Result of `MemoryForget`; `deleted: false` with no `key` means no row had that id.
    MemoryForgetResult { deleted: bool, key: Option<String> },
    /// Per-item progress of a `MemoryReindex` run.
    ReindexProgress {
        kind: ReindexKind,
        done: u32,
        total: u32,
    },
    /// The document index as a whole, sent first in answer to `DocumentsStatus`.
    DocumentIndex {
        /// Whether the extraction worker that reads the roots' files runs.
        indexing: ComponentReadiness,
        /// Whether the embedder that makes stored chunks searchable runs.
        embedding: ComponentReadiness,
        /// Chunks the stored documents are cut into.
        chunks: u64,
        /// Most chunks the index holds; a pass that reaches it stores nothing more.
        chunk_limit: u64,
        /// The root a pass is walking now.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        running_pass: Option<String>,
        transcription: PageTranscriptionStatus,
    },
    /// One index root, in `documents.roots` order.
    DocumentRoot { status: DocumentRootStatus },
    /// How far the pass a `DocumentsReindex` follows has got through `root`.
    DocumentPassProgress {
        root: String,
        counts: DocumentPassCounts,
    },
    /// The pass a `DocumentsReindex` follows over `root` ended.
    DocumentPassEnded { root: String, pass: DocumentPass },
    /// One `DocumentsSearch` hit: a document and its best-matching passage. `page` counts
    /// from one, the byte range is within that page's text, and `similarity`, in
    /// `[0.0, 1.0]`, is absent for a passage that has no vector yet.
    DocumentHit {
        path: String,
        page: u64,
        page_count: u64,
        byte_start: u64,
        byte_end: u64,
        indexed_at: DateTime<Utc>,
        content: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        similarity: Option<f32>,
    },
    /// What the document index is doing; broadcast when it changes and sent first to every
    /// subscriber whose filter takes it.
    DocumentActivity { activity: DocumentActivity },
    /// Asks to authorize a tool action; the turn waits for a matching `ConfirmResponse`. Never
    /// sent to a write-closed client; disconnect or timeout denies.
    ConfirmRequest {
        confirm_id: String,
        tool: String,
        script: String,
        /// Why confirmation is needed, for display.
        matched_pattern: String,
        /// Programs an "always allow" answer adds to the allowlist; empty when not offered.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        always_allow: Vec<String>,
        /// Which agent asked.
        #[serde(default, skip_serializing_if = "ConfirmSource::is_main")]
        source: ConfirmSource,
    },
    /// Asks the user to review `plan`; the turn waits for a matching `PlanVerdict`, and a
    /// disconnect leaves it unreviewed.
    PlanReview { review_id: String, plan: String },
    /// The user agreed to the model's ask to plan first; the turn now plans.
    PlanModeEntered,
    /// Runtime capabilities; `model_name` is the basename of the configured `model.name`, and
    /// `auto_mode` whether [`PermissionMode::Auto`](crate::PermissionMode::Auto) is honored,
    /// and `default_mode` the mode a chat window starts in.
    Capabilities {
        vision: bool,
        model_name: String,
        #[serde(default)]
        auto_mode: bool,
        #[serde(default)]
        default_mode: PermissionMode,
    },
    /// Progress of a relay running inside the turn's `relay` tool call.
    Relay { event: RelayEvent },
    /// Tokens in the history as sent with the model request about to stream,
    /// and `total` the history has held, counting what compaction removed.
    ContextSize { tokens: u32, total: u32 },
    /// Non-terminal notice of a recoverable condition.
    Status {
        severity: StatusSeverity,
        component: Component,
        event: StatusKind,
        message: String,
    },
    /// One branch listed by `Branches`.
    BranchInfo {
        branch_id: i64,
        session_id: String,
        session_started_at: String,
        #[serde(default)]
        session_title: Option<String>,
        name: String,
        parent_branch_name: Option<String>,
        fork_point_seq: Option<i64>,
        created_at: String,
        message_count: u64,
        is_current_in_session: bool,
        is_active_session: bool,
        /// The later of the branch's creation and its newest message, RFC 3339.
        #[serde(default)]
        last_active_at: String,
        /// The task run the session holds; `None` for a user's session.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        task_run: Option<TaskRunSummary>,
    },
    /// The active branch changed; `session_id` is the now-active session.
    /// `branch_id` is `None` until the session's first message is saved.
    BranchSwitched {
        branch_id: Option<i64>,
        session_id: String,
        session_title: Option<String>,
        name: String,
        parent_branch_name: Option<String>,
        fork_point_seq: Option<i64>,
    },
    /// One message of branch history, with its reasoning or tool call and outcome.
    /// `compaction_ms` is set only on the row the branch's latest compaction followed;
    /// `turn_stats` only on the last row of a finished turn; `relay` only on a `relay` tool row.
    HistoryEntry {
        seq: i64,
        role: Role,
        content: String,
        #[serde(default, skip_serializing_if = "String::is_empty")]
        reasoning: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reasoning_ms: Option<u64>,
        tool_name: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        tool_args: Option<serde_json::Value>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        exit_code: Option<i64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        duration_ms: Option<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        compaction_ms: Option<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        turn_stats: Option<TurnStats>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        relay: Vec<RelayStepRecord>,
    },
    /// Result of `Undo`; `removed_messages == 0` means there was nothing to undo.
    UndoApplied {
        removed_messages: u32,
        last_user_text: Option<String>,
    },
    /// One saved task.
    Task {
        task_id: TaskId,
        spec: TaskSpec,
        /// When its triggers next fire; `None` while it is disabled or nothing is due.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        next_fire: Option<DateTime<Utc>>,
        /// How its latest finished run ended; `None` before any has finished.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        last_run: Option<TaskRunState>,
        /// Why the watch for one of its event triggers last stopped, while it retries;
        /// `None` while every watch runs.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        trigger_failure: Option<String>,
    },
    /// One saved task whose stored form the daemon cannot decode, listed by `TaskList` so
    /// it can be deleted; `name` is as stored and `reason` says what failed.
    UndecodableTask {
        task_id: TaskId,
        name: String,
        reason: String,
    },
    /// The next fire times of the trigger a `TaskCheckTrigger` checked, soonest first, at
    /// most three; empty for a trigger that fires on request or on an event.
    TriggerFires { fires: Vec<DateTime<Utc>> },
    /// One run listed by `TaskRuns`.
    TaskRunRecord { record: TaskRunRecord },
    /// Run `run` of `task` was queued.
    TaskRunQueued { task: TaskId, run: TaskRunId },
    /// The final reply of run `run`, which request `id` queued, of a task that announces its
    /// replies; broadcast only.
    TaskReply {
        run: TaskRunId,
        task_name: TaskName,
        reply: String,
    },
    /// The query `id` waits for the model behind `behind` queue items: a running task, or
    /// the runs and turns ahead of it in the queue.
    Queued { behind: u32 },
    /// The task queue as it now stands; broadcast on every change and sent first to every
    /// subscriber whose filter takes it.
    TaskQueue { snapshot: TaskQueueSnapshot },
    /// The asked prompt rewritten as a brief for an unattended run, the plan of that brief,
    /// each anticipated action checked against the machine and the asked permissions, and the
    /// mode its runs are best given among those the daemon allows.
    TaskRefined {
        prompt: TaskPrompt,
        plan: Plan,
        recommended: ModeRecommendation,
    },
    /// The URL the user opens to sign MCP server `server` in.
    McpAuthorizeUrl { server: String, url: String },
    /// One configured MCP server.
    McpServer {
        server: String,
        transport: McpTransportKind,
        state: McpServerState,
    },
    /// MCP server `server` was signed out.
    McpLoggedOut {
        server: String,
        revocation: TokenRevocation,
    },
    /// Terminal failure.
    Error { message: String },
    /// Terminal success.
    Done,
    /// The running reply so far; sent only to subscribers.
    LastDelta { text: String },
}
