//! Wire types for the daemon's Unix-socket protocol: line-delimited JSON,
//! with [`Request`] lines from the client and, after one [`Hello`], [`Event`] lines back.

use std::env;
use std::fmt;
use std::path::PathBuf;

use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use assistd_utils::readiness::NotReady;

pub mod attachment;
#[cfg(feature = "client")]
pub mod client;
pub mod mcp;
pub mod task;
pub mod tool_report;
pub mod wakeword;
pub use attachment::ImageMime;
#[cfg(feature = "client")]
pub use client::{DialogConnection, EventStream, IpcClient, IpcClientError};
pub use mcp::{McpServerState, McpTransportKind, TokenRevocation};
pub use task::{
    ActionOutlook, AnticipatedAction, ConversationHold, DEFAULT_EVENT_COOLDOWN_SECS,
    DEFAULT_PATH_DEBOUNCE_SECS, FinishOutcome, HostName, KeptRuns, LaneHolder, MAX_KEPT_RUNS,
    MAX_LISTED_RUNS, MissedFire, ModeRecommendation, PausedPrompt, Plan, PlanId, PlanStep,
    PlannedAction, QueueItem, SuspendReason, TaskEvent, TaskId, TaskName, TaskPrompt, TaskQueueOp,
    TaskQueueSnapshot, TaskRequest, TaskRunId, TaskRunRecord, TaskRunState, TaskRunSummary,
    TaskSpec, TaskSpecError, TaskTrigger,
};
pub use tool_report::{ToolAttachment, ToolReport, ToolStreams};
pub use wakeword::WakeWordOp;

/// The assistd version this build speaks the protocol of; a daemon and its clients must
/// share it.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// An image on a [`RequestBody::Query`]; `data_base64` is standard padded base64.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ImageAttachment {
    pub mime: ImageMime,
    pub data_base64: String,
}

impl ImageAttachment {
    /// Encode raw image `bytes` as standard padded base64.
    pub fn from_bytes(mime: ImageMime, bytes: &[u8]) -> Self {
        Self {
            mime,
            data_base64: STANDARD.encode(bytes),
        }
    }

    /// Decode `data_base64`; errors when it is not valid standard base64.
    pub fn decode_bytes(&self) -> Result<Vec<u8>, base64::DecodeError> {
        STANDARD.decode(&self.data_base64)
    }
}

/// Coarse daemon lifecycle state.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PresenceState {
    /// Ready to answer queries.
    Active,
    /// llama-server running with its model weights unloaded.
    Drowsy,
    /// llama-server stopped.
    Sleeping,
    /// llama-server starting or loading the model; queries wait for it.
    Waking,
}

impl PresenceState {
    /// Lower-case name, matching the wire form.
    pub const fn as_str(self) -> &'static str {
        match self {
            PresenceState::Active => "active",
            PresenceState::Drowsy => "drowsy",
            PresenceState::Sleeping => "sleeping",
            PresenceState::Waking => "waking",
        }
    }
}

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

/// The user's answer to an [`EventBody::PlanReview`].
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

/// A presence state the daemon can be driven to and settle in.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PresenceTarget {
    Active,
    Drowsy,
    Sleeping,
}

impl From<PresenceTarget> for PresenceState {
    fn from(target: PresenceTarget) -> Self {
        match target {
            PresenceTarget::Active => PresenceState::Active,
            PresenceTarget::Drowsy => PresenceState::Drowsy,
            PresenceTarget::Sleeping => PresenceState::Sleeping,
        }
    }
}

/// Push-to-talk capture state. `Idle` must stay the first variant.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum VoiceCaptureState {
    Idle,
    Recording,
    Transcribing,
}

/// Wake-word listener phase.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ListenPhase {
    /// The listener is stopped.
    Off,
    /// The mic is open and waiting for the wake phrase.
    Armed,
    /// The wake phrase was heard; the next utterance is the command.
    Capturing,
}

impl ListenPhase {
    /// Whether the listener has the mic open.
    pub const fn is_on(self) -> bool {
        !matches!(self, ListenPhase::Off)
    }

    /// Lower-case name, matching the wire form.
    pub const fn as_str(self) -> &'static str {
        match self {
            ListenPhase::Off => "off",
            ListenPhase::Armed => "armed",
            ListenPhase::Capturing => "capturing",
        }
    }
}

/// A subsystem that starts in the background after the socket opens.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum StartupComponent {
    /// Push-to-talk and continuous listening.
    VoiceInput,
    Speech,
    /// The embedding server behind semantic recall.
    Embedding,
    /// The SearXNG instance behind web search.
    WebSearch,
    Mcp {
        server: String,
    },
}

impl fmt::Display for StartupComponent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StartupComponent::VoiceInput => f.write_str("voice input"),
            StartupComponent::Speech => f.write_str("speech"),
            StartupComponent::Embedding => f.write_str("semantic memory"),
            StartupComponent::WebSearch => f.write_str("web search"),
            StartupComponent::Mcp { server } => write!(f, "MCP {server}"),
        }
    }
}

/// How far one [`StartupComponent`]'s startup has got.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum ComponentReadiness {
    Starting,
    Ready,
    /// Disabled in config or failed to start.
    Unavailable {
        reason: String,
    },
}

impl fmt::Display for ComponentReadiness {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ComponentReadiness::Starting => f.write_str("starting"),
            ComponentReadiness::Ready => f.write_str("ready"),
            ComponentReadiness::Unavailable { reason } => write!(f, "unavailable ({reason})"),
        }
    }
}

/// The latest [`ComponentReadiness`] a client has seen for each
/// [`StartupComponent`], in the order each was first reported.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StartupReadiness {
    components: Vec<(StartupComponent, ComponentReadiness)>,
}

impl StartupReadiness {
    /// Record `component`'s latest `state`.
    pub fn record(&mut self, component: StartupComponent, state: ComponentReadiness) {
        match self
            .components
            .iter_mut()
            .find(|(seen, _)| *seen == component)
        {
            Some((_, current)) => *current = state,
            None => self.components.push((component, state)),
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = &(StartupComponent, ComponentReadiness)> {
        self.components.iter()
    }

    pub fn starting(&self) -> impl Iterator<Item = &StartupComponent> {
        self.components
            .iter()
            .filter(|(_, state)| *state == ComponentReadiness::Starting)
            .map(|(component, _)| component)
    }
}

impl<T> From<Result<T, NotReady>> for ComponentReadiness {
    fn from(ready: Result<T, NotReady>) -> Self {
        match ready {
            Ok(_) => ComponentReadiness::Ready,
            Err(NotReady::Starting) => ComponentReadiness::Starting,
            Err(NotReady::Unavailable(reason)) => ComponentReadiness::Unavailable {
                reason: reason.to_string(),
            },
        }
    }
}

/// Who raised an [`EventBody::ConfirmRequest`].
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "agent", rename_all = "snake_case")]
pub enum ConfirmSource {
    /// The main conversation's agent.
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
    /// Tokens the sub-agent's conversation has held, counting what compaction removed.
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
    /// Tokens its conversation had held when it ended, once reported.
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

/// Kinds of [`Event`] carried on the daemon-wide broadcast bus, selectable by a
/// [`SubscribeFilter`]. Dialog-local events have no variant.
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
    TaskRun,
    TaskQueue,
    TaskReply,
}

/// Event-kind filter for [`RequestBody::Subscribe`]; empty `kinds` matches every kind.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct SubscribeFilter {
    #[serde(default)]
    pub kinds: Vec<EventKind>,
}

impl SubscribeFilter {
    /// True when `kind` should be delivered under this filter.
    pub fn matches(&self, kind: EventKind) -> bool {
        self.kinds.is_empty() || self.kinds.contains(&kind)
    }
}

/// The line the daemon writes when it accepts a connection, ahead of every [`Event`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Hello {
    /// The daemon's [`VERSION`].
    pub daemon_version: String,
}

impl Hello {
    /// The hello of a daemon from this build.
    pub fn current() -> Self {
        Self {
            daemon_version: VERSION.to_owned(),
        }
    }
}

/// Client-to-daemon request, one JSON line: `id` beside the fields of `body`.
/// A client sends one request and closes its write half unless answering
/// [`EventBody::ConfirmRequest`] or [`EventBody::PlanReview`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Request {
    /// Correlation id, echoed on every [`Event`] sent in response.
    pub id: String,
    #[serde(flatten)]
    pub body: RequestBody,
}

impl Request {
    pub fn new(id: impl Into<String>, body: RequestBody) -> Self {
        Self {
            id: id.into(),
            body,
        }
    }

    /// Build a text-only [`RequestBody::Query`].
    pub fn query(id: impl Into<String>, text: impl Into<String>) -> Self {
        Self::query_with_attachments(id, text, Vec::new())
    }

    /// Build a [`RequestBody::Query`] with image attachments.
    pub fn query_with_attachments(
        id: impl Into<String>,
        text: impl Into<String>,
        attachments: Vec<ImageAttachment>,
    ) -> Self {
        Self::new(
            id,
            RequestBody::Query {
                text: text.into(),
                attachments,
                plan: PlanMode::Off,
            },
        )
    }

    /// The variant's wire `"type"` tag.
    pub fn kind(&self) -> &'static str {
        match self.body {
            RequestBody::Query { .. } => "query",
            RequestBody::SetPresence { .. } => "set_presence",
            RequestBody::GetPresence => "get_presence",
            RequestBody::PttStart => "ptt_start",
            RequestBody::PttStop => "ptt_stop",
            RequestBody::ListenStart => "listen_start",
            RequestBody::ListenStop => "listen_stop",
            RequestBody::ListenToggle => "listen_toggle",
            RequestBody::GetListenState => "get_listen_state",
            RequestBody::VoiceToggle => "voice_toggle",
            RequestBody::VoiceSkip => "voice_skip",
            RequestBody::InterruptTurn => "interrupt_turn",
            RequestBody::GetVoiceState => "get_voice_state",
            RequestBody::WakeWord { .. } => "wake_word",
            RequestBody::MemorySave { .. } => "memory_save",
            RequestBody::MemoryLoad { .. } => "memory_load",
            RequestBody::MemoryListAll { .. } => "memory_list_all",
            RequestBody::MemoryDelete { .. } => "memory_delete",
            RequestBody::MemoryForget { .. } => "memory_forget",
            RequestBody::MemorySemanticSearch { .. } => "memory_semantic_search",
            RequestBody::MemoryReindex => "memory_reindex",
            RequestBody::ConfirmResponse { .. } => "confirm_response",
            RequestBody::SetPermissionMode { .. } => "set_permission_mode",
            RequestBody::PlanVerdict { .. } => "plan_verdict",
            RequestBody::GetCapabilities => "get_capabilities",
            RequestBody::Fork { .. } => "fork",
            RequestBody::Branches => "branches",
            RequestBody::Switch { .. } => "switch",
            RequestBody::Undo => "undo",
            RequestBody::ResumeOrNew { .. } => "resume_or_new",
            RequestBody::NewSession => "new_session",
            RequestBody::ChatState { .. } => "chat_state",
            RequestBody::ChatClosed => "chat_closed",
            RequestBody::GetChatFocus => "get_chat_focus",
            RequestBody::GetReadiness => "get_readiness",
            RequestBody::Subscribe { .. } => "subscribe",
            RequestBody::Task(_) => "task",
            RequestBody::McpLogin { .. } => "mcp_login",
            RequestBody::McpLogout { .. } => "mcp_logout",
            RequestBody::McpStatus => "mcp_status",
        }
    }
}

/// What a [`Request`] asks for, tagged by `"type"`. A line is capped at 64 KiB, except that a
/// connection's first line may reach 64 MiB when it is a `Query`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RequestBody {
    /// Run an agent turn on `text`. Streams the turn, then `Done`.
    Query {
        text: String,
        /// Vision inputs for this turn; empty for text-only queries.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        attachments: Vec<ImageAttachment>,
        /// Whether the turn may plan, or starts planning.
        #[serde(default, skip_serializing_if = "PlanMode::is_off")]
        plan: PlanMode,
    },
    /// Drive the daemon to `target`. Emits `Presence`, then `Done`.
    SetPresence { target: PresenceTarget },
    /// Report the presence state. Emits `Presence`, then `Done`.
    GetPresence,
    /// Open the mic and buffer audio until `PttStop`. Emits `Done` once the mic is open.
    PttStart,
    /// Transcribe the recording and run it as a query. Emits `VoiceState`, `Transcription`, the
    /// query's stream, then `Done`.
    PttStop,
    /// Start continuous VAD listening; errors during a PTT recording. Emits `ListenState`, `Done`.
    ListenStart,
    /// Stop continuous listening; idempotent. Emits `ListenState`, then `Done`.
    ListenStop,
    /// Toggle continuous listening. Emits `ListenState`, then `Done`.
    ListenToggle,
    /// Report the wake-word listener's phase. Emits `ListenState`, then `Done`.
    GetListenState,
    /// Toggle TTS; off cancels in-flight playback. Emits `VoiceOutputState`, then `Done`.
    VoiceToggle,
    /// Drop the current TTS response, keeping TTS enabled. Emits `VoiceOutputState`, then `Done`.
    VoiceSkip,
    /// Cancel the in-flight turn and queued TTS audio; idempotent. Emits `Done`.
    InterruptTurn,
    /// Run one wake-phrase operation; see [`WakeWordOp`] for what each emits.
    WakeWord { op: WakeWordOp },
    /// Report whether TTS is enabled and how far voice startup has got.
    /// Emits `VoiceOutputState`, a `Readiness` for voice input and for
    /// speech, then `Done`.
    GetVoiceState,
    /// Store `value` under `key`, overwriting. Emits `Done`.
    MemorySave { key: String, value: String },
    /// Read `key`. Emits `MemoryValue`, then `Done`.
    MemoryLoad { key: String },
    /// Rows whose key starts with `prefix`; `limit = 0` is uncapped. Emits `MemoryRow`s, `Done`.
    MemoryListAll {
        #[serde(default)]
        prefix: String,
        #[serde(default)]
        limit: u32,
    },
    /// Remove `key`; no-op when absent. Emits `Done`.
    MemoryDelete { key: String },
    /// Remove the row with id `memory_id`. Emits `MemoryForgetResult`, then `Done`.
    MemoryForget { memory_id: i64 },
    /// Rank past messages by similarity to `query`; `limit = 0` uses the daemon default.
    /// Emits `SemanticHit`s best-first, then `Done`.
    MemorySemanticSearch {
        query: String,
        #[serde(default)]
        limit: u32,
    },
    /// Embed every row lacking an embedding under the current model. Emits `ReindexProgress`es,
    /// then `Done`.
    MemoryReindex,
    /// Answer the [`EventBody::ConfirmRequest`] with this `confirm_id` on the same connection.
    ConfirmResponse {
        confirm_id: String,
        allow: bool,
        /// With `allow`, permanently allowlist the prompt's `always_allow` programs.
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        always: bool,
    },
    /// Switch how the rest of the turn's prompts on the same connection are answered: under
    /// `mode`, or with `plan` refused while the turn plans; honored only after a `Query`.
    SetPermissionMode {
        mode: PermissionMode,
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        plan: bool,
    },
    /// Answer the [`EventBody::PlanReview`] with this `review_id` on the same connection.
    PlanVerdict {
        review_id: String,
        verdict: PlanVerdict,
    },
    /// Probe runtime capabilities. Emits `Capabilities`, then `Done`.
    GetCapabilities,
    /// Snapshot into new branch `name` and switch to it; errors if `name` is taken or empty.
    /// Emits `BranchSwitched`, then `Done`.
    Fork { name: String },
    /// List every branch, active session first. Emits `BranchInfo`s, then `Done`.
    Branches,
    /// Switch to `target`: a branch name, or `<session_prefix>/<name>` across sessions.
    /// Emits `BranchSwitched`, `HistoryEntry`s, then `Done`.
    Switch { target: String },
    /// Drop the last user message and its reply. Emits `UndoApplied`, then `Done`.
    Undo,
    /// Resume the current branch if written within `recency_secs` (emits `HistoryEntry`s), else
    /// start a new session (emits `BranchSwitched`); then `Done`.
    ResumeOrNew { recency_secs: u64 },
    /// Start a new session, saved with its first message. Emits `BranchSwitched`, then `Done`.
    NewSession,
    /// Report whether the chat has keyboard focus; the daemon keys the chat by the sender's PID.
    /// Emits `Done`.
    ChatState { focused: bool },
    /// The chat is exiting. Emits `Done`.
    ChatClosed,
    /// Report whether a live chat has keyboard focus. Emits `ChatFocus`, then `Done`.
    GetChatFocus,
    /// Report how far every background subsystem has started. Emits one
    /// `Readiness` per subsystem, then `Done`.
    GetReadiness,
    /// Forward broadcast events matching `filter`, tagged with their turn's `id`, until the
    /// client disconnects; no `Done`, and `ToolResult` attachments are stripped. Opens with the
    /// current `TaskQueue`, `Presence`, `ListenState` and `VoiceState`, then sends every
    /// presence change, each tagged with this request's `id`.
    Subscribe {
        #[serde(default)]
        filter: SubscribeFilter,
    },
    /// Read or change saved tasks, their runs and the task queue.
    Task(TaskRequest),
    /// Sign OAuth server `server` in through the browser. Emits `McpAuthorizeUrl`, then once
    /// the redirect is redeemed and the server restarted, its `McpServer` and `Done`; errors
    /// when the server is unknown or not OAuth, or the sign-in fails or times out.
    McpLogin { server: String },
    /// Stop OAuth server `server` and delete its token, revoking it where the authorization
    /// server offers that. Emits `McpLoggedOut`, then `Done`.
    McpLogout { server: String },
    /// Report every configured MCP server. Emits one `McpServer` each, then `Done`.
    McpStatus,
}

/// Severity of an [`EventBody::Status`] update; maps 1:1 to a `tracing` level.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum StatusSeverity {
    Info,
    Warning,
    Error,
}

impl StatusSeverity {
    /// The wire spelling.
    pub fn as_str(self) -> &'static str {
        match self {
            StatusSeverity::Info => "info",
            StatusSeverity::Warning => "warning",
            StatusSeverity::Error => "error",
        }
    }
}

impl fmt::Display for StatusSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad(self.as_str())
    }
}

/// Daemon subsystem an [`EventBody::Status`] update is attributed to.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Component {
    Agent,
    Llm,
    Mcp,
    Voice,
    Memory,
    Wm,
    Embed,
    Daemon,
    IdleMonitor,
    GpuMonitor,
    ListenDispatcher,
}

impl Component {
    /// The wire spelling.
    pub fn as_str(self) -> &'static str {
        match self {
            Component::Agent => "agent",
            Component::Llm => "llm",
            Component::Mcp => "mcp",
            Component::Voice => "voice",
            Component::Memory => "memory",
            Component::Wm => "wm",
            Component::Embed => "embed",
            Component::Daemon => "daemon",
            Component::IdleMonitor => "idle_monitor",
            Component::GpuMonitor => "gpu_monitor",
            Component::ListenDispatcher => "listen_dispatcher",
        }
    }
}

impl fmt::Display for Component {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad(self.as_str())
    }
}

/// What an [`EventBody::Status`] update reports.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StatusKind {
    /// The LLM server died mid-turn and is being restarted.
    Restarting,
    /// The LLM server is back; the interrupted step is being replayed.
    Replaying,
    /// Recovery failed; the turn will not complete.
    Degraded,
    /// The tool schema was withdrawn so the model answers from what it has.
    ToolsWithdrawn,
    /// The model's response hit the output-token limit and was discarded;
    /// the step is being retried.
    OutputTruncated,
    /// The model is still loading after a wake.
    ModelLoading,
    /// A subsystem failed to start and is unavailable this run.
    StartupFailed,
    /// History is close enough to the token budget that an upcoming step
    /// will summarize it.
    CompactingSoon,
    /// Older history is being summarized to fit the token budget.
    CompactingHistory,
    /// The compaction announced by [`StatusKind::CompactingHistory`] finished.
    HistoryCompacted,
    /// A task run gave the model to the user's conversation; the step it was streaming
    /// was discarded and runs again when the task resumes.
    YieldedToUser,
}

/// Author of a persisted transcript message.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

impl Role {
    /// The wire spelling.
    pub fn as_str(self) -> &'static str {
        match self {
            Role::System => "system",
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
    /// Tokens the turn added to the conversation.
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
/// [`RequestBody::Subscribe`] stream never ends.
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
            EventBody::TaskRun { .. } => EventKind::TaskRun,
            EventBody::TaskQueue { .. } => EventKind::TaskQueue,
            EventBody::TaskReply { .. } => EventKind::TaskReply,
            EventBody::VoiceOutputState { .. }
            | EventBody::WakeWordStatus { .. }
            | EventBody::WakeWordTest { .. }
            | EventBody::WakeWordSpeech { .. }
            | EventBody::SemanticHit { .. }
            | EventBody::MemoryValue { .. }
            | EventBody::MemoryRow { .. }
            | EventBody::MemoryForgetResult { .. }
            | EventBody::ReindexProgress { .. }
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
            | EventBody::TaskPlan { .. }
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
    /// One search hit; `content` is the matched chunk of message `message_id`, `similarity` is in `[0.0, 1.0]`.
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
    /// `auto_mode` whether [`PermissionMode::Auto`] is honored.
    Capabilities {
        vision: bool,
        model_name: String,
        #[serde(default)]
        auto_mode: bool,
    },
    /// Progress of a relay running inside the turn's `relay` tool call.
    Relay { event: RelayEvent },
    /// Tokens in the conversation as sent with the model request about to stream,
    /// and `total` the conversation has held, counting what compaction removed.
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
    /// One event of task run `run`'s turn, which `id` queued; never terminal itself.
    TaskRun { run: TaskRunId, event: Box<Event> },
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
    /// One event of the planning turn `id` asked for; never terminal itself.
    TaskPlan { event: Box<Event> },
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

/// The per-user daemon socket: `$XDG_RUNTIME_DIR/assistd.sock`, else
/// `/tmp/assistd-<euid>/assistd.sock`, whose directory the daemon creates owner-only.
pub fn socket_path() -> PathBuf {
    let runtime_dir = env::var_os("XDG_RUNTIME_DIR")
        .map(PathBuf::from)
        .filter(|dir| dir.is_absolute());
    socket_path_for(runtime_dir, rustix::process::geteuid().as_raw())
}

fn socket_path_for(runtime_dir: Option<PathBuf>, euid: u32) -> PathBuf {
    runtime_dir
        .unwrap_or_else(|| PathBuf::from(format!("/tmp/assistd-{euid}")))
        .join("assistd.sock")
}

#[cfg(test)]
mod tests;
