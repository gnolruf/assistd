//! Wire types for the daemon's Unix-socket protocol: line-delimited JSON,
//! with [`Request`] lines from the client and [`Event`] lines back.

use std::ffi::OsString;
use std::fmt;
use std::path::PathBuf;

use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use serde::{Deserialize, Serialize};

#[cfg(feature = "client")]
pub mod attachment;
#[cfg(feature = "client")]
pub mod client;
#[cfg(feature = "client")]
pub use client::{DialogConnection, EventStream, IpcClient, IpcClientError};

/// An image on a [`Request::Query`]: `mime` is PNG, JPEG or WebP; `data_base64` is standard
/// padded base64.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ImageAttachment {
    pub mime: String,
    pub data_base64: String,
}

impl ImageAttachment {
    /// Encode raw image `bytes` as standard padded base64.
    pub fn from_bytes(mime: impl Into<String>, bytes: &[u8]) -> Self {
        Self {
            mime: mime.into(),
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
}

impl PresenceState {
    /// Next state in the manual-cycle order: Active → Drowsy → Sleeping → Active.
    pub fn next(self) -> Self {
        match self {
            PresenceState::Active => PresenceState::Drowsy,
            PresenceState::Drowsy => PresenceState::Sleeping,
            PresenceState::Sleeping => PresenceState::Active,
        }
    }
}

/// Push-to-talk capture state. `Idle` must stay the first variant.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum VoiceCaptureState {
    Idle,
    /// Waiting for the GPU to free up before transcribing.
    Queued,
    Recording,
    Transcribing,
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
    VoiceState,
    SpeakingState,
    SessionTitle,
    Done,
    Error,
    LastDelta,
}

/// Event-kind filter for [`Request::Subscribe`]; empty `kinds` matches every kind.
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

/// Client-to-daemon request, one JSON line tagged by `"type"`. Its `id` is echoed on every
/// [`Event`] in response. A client sends one request and closes its write half unless it must
/// answer an [`Event::ConfirmRequest`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Request {
    /// Run an agent turn on `text`. Streams the turn, then `Done`.
    Query {
        id: String,
        text: String,
        /// Vision inputs for this turn; empty for text-only queries.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        attachments: Vec<ImageAttachment>,
    },
    /// Drive the daemon to `target`. Emits `Presence`, then `Done`.
    SetPresence { id: String, target: PresenceState },
    /// Report the presence state. Emits `Presence`, then `Done`.
    GetPresence { id: String },
    /// Atomically advance one step along `Active → Drowsy → Sleeping → Active`.
    Cycle { id: String },
    /// Open the mic and buffer audio until `PttStop`. Emits `Done` once the mic is open.
    PttStart { id: String },
    /// Transcribe the recording and run it as a query. Emits `VoiceState`, `Transcription`, the
    /// query's stream, then `Done`.
    PttStop { id: String },
    /// Start continuous VAD listening; errors during a PTT recording. Emits `ListenState`, `Done`.
    ListenStart { id: String },
    /// Stop continuous listening; idempotent. Emits `ListenState`, then `Done`.
    ListenStop { id: String },
    /// Toggle continuous listening. Emits `ListenState`, then `Done`.
    ListenToggle { id: String },
    /// Report continuous-listening state. Emits `ListenState`, then `Done`.
    GetListenState { id: String },
    /// Toggle TTS; off cancels in-flight playback. Emits `VoiceOutputState`, then `Done`.
    VoiceToggle { id: String },
    /// Drop the current TTS response, keeping TTS enabled. Emits `VoiceOutputState`, then `Done`.
    VoiceSkip { id: String },
    /// Cancel the in-flight turn and queued TTS audio; idempotent. Emits `Done`.
    InterruptTurn { id: String },
    /// Report whether TTS is enabled. Emits `VoiceOutputState`, then `Done`.
    GetVoiceState { id: String },
    /// Store `value` under `key`, overwriting. Emits `Done`.
    MemorySave {
        id: String,
        key: String,
        value: String,
    },
    /// Read `key`. Emits `MemoryValue`, then `Done`.
    MemoryLoad { id: String, key: String },
    /// List keys starting with `prefix` (empty lists all). Emits `MemoryKeys`, then `Done`.
    MemoryList {
        id: String,
        #[serde(default)]
        prefix: String,
    },
    /// Rows whose key starts with `prefix`; `limit = 0` is uncapped. Emits `MemoryRow`s, `Done`.
    MemoryListAll {
        id: String,
        #[serde(default)]
        prefix: String,
        #[serde(default)]
        limit: u32,
    },
    /// Remove `key`; no-op when absent. Emits `Done`.
    MemoryDelete { id: String, key: String },
    /// Remove the row with id `memory_id`. Emits `MemoryForgetResult`, then `Done`.
    MemoryForget { id: String, memory_id: i64 },
    /// Rank past messages by similarity to `query`; `limit = 0` uses the daemon default.
    /// Emits `SemanticHit`s best-first, then `Done`.
    MemorySemanticSearch {
        id: String,
        query: String,
        #[serde(default)]
        limit: u32,
    },
    /// Embed every row lacking an embedding under the current model. Emits `ReindexProgress`es,
    /// then `Done`.
    MemoryReindex { id: String },
    /// Answer the [`Event::ConfirmRequest`] with this `confirm_id` on the same connection.
    ConfirmResponse {
        id: String,
        confirm_id: String,
        allow: bool,
        /// With `allow`, permanently allowlist the prompt's `always_allow` programs.
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        always: bool,
    },
    /// Probe runtime capabilities. Emits `Capabilities`, then `Done`.
    GetCapabilities { id: String },
    /// Snapshot into new branch `name` and switch to it; errors if `name` is taken or empty.
    /// Emits `BranchSwitched`, then `Done`.
    Fork { id: String, name: String },
    /// List every branch, active session first. Emits `BranchInfo`s, then `Done`.
    Branches { id: String },
    /// Switch to `target`: a branch name, or `<session_prefix>/<name>` across sessions.
    /// Emits `BranchSwitched`, `HistoryEntry`s, then `Done`.
    Switch { id: String, target: String },
    /// Drop the last user message and its reply. Emits `UndoApplied`, then `Done`.
    Undo { id: String },
    /// Resume the current branch if written within `recency_secs` (emits `HistoryEntry`s), else
    /// start a new session (emits `BranchSwitched`); then `Done`.
    ResumeOrNew { id: String, recency_secs: u64 },
    /// Start a new session with an empty `main` branch. Emits `BranchSwitched`, then `Done`.
    NewSession { id: String },
    /// Forward broadcast events matching `filter`, tagged with their turn's `id`, until the
    /// client disconnects; no `Done`, and `ToolResult` attachments are stripped.
    Subscribe {
        id: String,
        #[serde(default)]
        filter: SubscribeFilter,
    },
}

impl Request {
    /// Build a text-only [`Request::Query`].
    pub fn query(id: impl Into<String>, text: impl Into<String>) -> Self {
        Request::Query {
            id: id.into(),
            text: text.into(),
            attachments: Vec::new(),
        }
    }

    /// Build a [`Request::Query`] with one or more image attachments.
    pub fn query_with_attachments(
        id: impl Into<String>,
        text: impl Into<String>,
        attachments: Vec<ImageAttachment>,
    ) -> Self {
        Request::Query {
            id: id.into(),
            text: text.into(),
            attachments,
        }
    }

    /// The correlation id echoed on every [`Event`] sent in response.
    pub fn id(&self) -> &str {
        match self {
            Request::Query { id, .. }
            | Request::SetPresence { id, .. }
            | Request::GetPresence { id }
            | Request::Cycle { id }
            | Request::PttStart { id }
            | Request::PttStop { id }
            | Request::ListenStart { id }
            | Request::ListenStop { id }
            | Request::ListenToggle { id }
            | Request::GetListenState { id }
            | Request::VoiceToggle { id }
            | Request::VoiceSkip { id }
            | Request::InterruptTurn { id }
            | Request::GetVoiceState { id }
            | Request::MemorySave { id, .. }
            | Request::MemoryLoad { id, .. }
            | Request::MemoryList { id, .. }
            | Request::MemoryListAll { id, .. }
            | Request::MemoryDelete { id, .. }
            | Request::MemoryForget { id, .. }
            | Request::MemorySemanticSearch { id, .. }
            | Request::MemoryReindex { id, .. }
            | Request::ConfirmResponse { id, .. }
            | Request::GetCapabilities { id, .. }
            | Request::Fork { id, .. }
            | Request::Branches { id }
            | Request::Switch { id, .. }
            | Request::Undo { id }
            | Request::ResumeOrNew { id, .. }
            | Request::NewSession { id }
            | Request::Subscribe { id, .. } => id,
        }
    }

    /// The variant's wire `"type"` tag.
    pub fn kind(&self) -> &'static str {
        match self {
            Request::Query { .. } => "query",
            Request::SetPresence { .. } => "set_presence",
            Request::GetPresence { .. } => "get_presence",
            Request::Cycle { .. } => "cycle",
            Request::PttStart { .. } => "ptt_start",
            Request::PttStop { .. } => "ptt_stop",
            Request::ListenStart { .. } => "listen_start",
            Request::ListenStop { .. } => "listen_stop",
            Request::ListenToggle { .. } => "listen_toggle",
            Request::GetListenState { .. } => "get_listen_state",
            Request::VoiceToggle { .. } => "voice_toggle",
            Request::VoiceSkip { .. } => "voice_skip",
            Request::InterruptTurn { .. } => "interrupt_turn",
            Request::GetVoiceState { .. } => "get_voice_state",
            Request::MemorySave { .. } => "memory_save",
            Request::MemoryLoad { .. } => "memory_load",
            Request::MemoryList { .. } => "memory_list",
            Request::MemoryListAll { .. } => "memory_list_all",
            Request::MemoryDelete { .. } => "memory_delete",
            Request::MemoryForget { .. } => "memory_forget",
            Request::MemorySemanticSearch { .. } => "memory_semantic_search",
            Request::MemoryReindex { .. } => "memory_reindex",
            Request::ConfirmResponse { .. } => "confirm_response",
            Request::GetCapabilities { .. } => "get_capabilities",
            Request::Fork { .. } => "fork",
            Request::Branches { .. } => "branches",
            Request::Switch { .. } => "switch",
            Request::Undo { .. } => "undo",
            Request::ResumeOrNew { .. } => "resume_or_new",
            Request::NewSession { .. } => "new_session",
            Request::Subscribe { .. } => "subscribe",
        }
    }
}

/// Severity of an [`Event::Status`] update; maps 1:1 to a `tracing` level.
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

/// Daemon subsystem an [`Event::Status`] update is attributed to.
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
    Hotkey,
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
            Component::Hotkey => "hotkey",
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

/// What an [`Event::Status`] update reports.
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
    /// The model is still loading after a wake.
    ModelLoading,
    /// A subsystem failed to start and is unavailable this run.
    StartupFailed,
}

/// Author of a persisted conversation message.
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

/// Which table a [`Event::ReindexProgress`] item belongs to.
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

/// Daemon-to-client event, one JSON line tagged by `"type"`. A response stream ends with exactly
/// one terminal [`Event::Done`] or [`Event::Error`]; a [`Request::Subscribe`] stream never ends.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Event {
    /// Chunk of reply text.
    Delta { id: String, text: String },
    /// Chunk of model reasoning, separate from the reply.
    ReasoningDelta { id: String, text: String },
    /// The model invoked a tool.
    ToolCall {
        id: String,
        name: String,
        args: serde_json::Value,
    },
    ToolResult {
        id: String,
        name: String,
        result: serde_json::Value,
    },
    /// Current presence state.
    Presence { id: String, state: PresenceState },
    /// Push-to-talk capture state transition.
    VoiceState {
        id: String,
        state: VoiceCaptureState,
    },
    /// Transcript of a PTT recording; empty means silence, and only `Done` follows.
    Transcription { id: String, text: String },
    /// Whether continuous listening is active.
    ListenState { id: String, active: bool },
    /// Whether TTS is enabled.
    VoiceOutputState { id: String, enabled: bool },
    /// TTS playback for a turn started (`true`) or drained (`false`).
    SpeakingState { id: String, speaking: bool },
    /// Session display title; may arrive after the triggering turn's `Done`.
    SessionTitle {
        id: String,
        session_id: String,
        title: String,
    },
    /// One search hit; `content` is the full message, `similarity` is in `[0.0, 1.0]`.
    SemanticHit {
        id: String,
        conversation_id: i64,
        chunk_id: i64,
        session_id: String,
        timestamp: String,
        role: Role,
        content: String,
        similarity: f32,
    },
    /// Result of `MemoryLoad`; `value` is `None` when the key is absent.
    MemoryValue {
        id: String,
        key: String,
        value: Option<String>,
    },
    /// Result of `MemoryList`, lexicographically sorted; emitted even when empty.
    MemoryKeys { id: String, keys: Vec<String> },
    /// One `MemoryListAll` row, in key order; `memory_id` is what `MemoryForget` takes.
    MemoryRow {
        id: String,
        memory_id: i64,
        key: String,
        value: String,
    },
    /// Result of `MemoryForget`; `deleted: false` with no `key` means no row had that id.
    MemoryForgetResult {
        id: String,
        deleted: bool,
        key: Option<String>,
    },
    /// Per-item progress of a `MemoryReindex` run.
    ReindexProgress {
        id: String,
        kind: ReindexKind,
        done: u32,
        total: u32,
    },
    /// Asks to authorize a tool action; the turn waits for a matching `ConfirmResponse`. Never
    /// sent to a write-closed client; disconnect or timeout denies.
    ConfirmRequest {
        id: String,
        confirm_id: String,
        tool: String,
        script: String,
        /// Why confirmation is needed, for display.
        matched_pattern: String,
        /// Programs an "always allow" answer adds to the allowlist; empty when not offered.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        always_allow: Vec<String>,
    },
    /// Runtime capabilities; `model_name` is the basename of the configured `model.name`.
    Capabilities {
        id: String,
        vision: bool,
        model_name: String,
    },
    /// Non-terminal notice of a recoverable condition.
    Status {
        id: String,
        severity: StatusSeverity,
        component: Component,
        event: StatusKind,
        message: String,
    },
    /// One branch listed by `Branches`.
    BranchInfo {
        id: String,
        branch_id: i64,
        session_id: String,
        session_started_at: String,
        session_ended_at: Option<String>,
        #[serde(default)]
        session_title: Option<String>,
        name: String,
        parent_branch_name: Option<String>,
        fork_point_seq: Option<i64>,
        created_at: String,
        message_count: i64,
        is_current_in_session: bool,
        is_active_session: bool,
    },
    /// The active branch changed; `session_id` is the now-active session.
    BranchSwitched {
        id: String,
        branch_id: i64,
        session_id: String,
        session_title: Option<String>,
        name: String,
        parent_branch_name: Option<String>,
        fork_point_seq: Option<i64>,
    },
    /// One message of branch history.
    HistoryEntry {
        id: String,
        seq: i64,
        role: Role,
        content: String,
        tool_name: Option<String>,
    },
    /// Result of `Undo`; `removed_messages == 0` means there was nothing to undo.
    UndoApplied {
        id: String,
        removed_messages: u32,
        last_user_text: Option<String>,
    },
    /// Terminal failure.
    Error { id: String, message: String },
    /// Terminal success.
    Done { id: String },
    /// The running reply so far; sent only to subscribers.
    LastDelta { id: String, text: String },
}

impl Event {
    /// Returns true if this event terminates a response stream.
    pub fn is_terminal(&self) -> bool {
        matches!(self, Event::Done { .. } | Event::Error { .. })
    }

    /// The id of the request this event responds to.
    pub fn id(&self) -> &str {
        match self {
            Event::Delta { id, .. }
            | Event::ReasoningDelta { id, .. }
            | Event::ToolCall { id, .. }
            | Event::ToolResult { id, .. }
            | Event::Presence { id, .. }
            | Event::VoiceState { id, .. }
            | Event::Transcription { id, .. }
            | Event::ListenState { id, .. }
            | Event::VoiceOutputState { id, .. }
            | Event::SpeakingState { id, .. }
            | Event::SessionTitle { id, .. }
            | Event::SemanticHit { id, .. }
            | Event::MemoryValue { id, .. }
            | Event::MemoryKeys { id, .. }
            | Event::MemoryRow { id, .. }
            | Event::MemoryForgetResult { id, .. }
            | Event::ReindexProgress { id, .. }
            | Event::ConfirmRequest { id, .. }
            | Event::Capabilities { id, .. }
            | Event::Status { id, .. }
            | Event::BranchInfo { id, .. }
            | Event::BranchSwitched { id, .. }
            | Event::HistoryEntry { id, .. }
            | Event::UndoApplied { id, .. }
            | Event::Error { id, .. }
            | Event::Done { id }
            | Event::LastDelta { id, .. } => id,
        }
    }

    /// The broadcast kind of this event, or `None` for dialog-local events.
    pub fn kind(&self) -> Option<EventKind> {
        Some(match self {
            Event::Delta { .. } => EventKind::Delta,
            Event::ReasoningDelta { .. } => EventKind::ReasoningDelta,
            Event::ToolCall { .. } => EventKind::ToolCall,
            Event::ToolResult { .. } => EventKind::ToolResult,
            Event::Presence { .. } => EventKind::Presence,
            Event::VoiceState { .. } => EventKind::VoiceState,
            Event::ListenState { .. } => EventKind::ListenState,
            Event::SpeakingState { .. } => EventKind::SpeakingState,
            Event::SessionTitle { .. } => EventKind::SessionTitle,
            Event::Done { .. } => EventKind::Done,
            Event::Error { .. } => EventKind::Error,
            Event::LastDelta { .. } => EventKind::LastDelta,
            Event::Transcription { .. }
            | Event::VoiceOutputState { .. }
            | Event::SemanticHit { .. }
            | Event::MemoryValue { .. }
            | Event::MemoryKeys { .. }
            | Event::MemoryRow { .. }
            | Event::MemoryForgetResult { .. }
            | Event::ReindexProgress { .. }
            | Event::ConfirmRequest { .. }
            | Event::Capabilities { .. }
            | Event::Status { .. }
            | Event::BranchInfo { .. }
            | Event::BranchSwitched { .. }
            | Event::HistoryEntry { .. }
            | Event::UndoApplied { .. } => return None,
        })
    }
}

/// The per-user daemon socket: `$XDG_RUNTIME_DIR/assistd.sock`, else `/tmp/assistd-$USER.sock`
/// (`nobody` when `$USER` is unset).
pub fn socket_path() -> PathBuf {
    socket_path_for(
        std::env::var_os("XDG_RUNTIME_DIR"),
        std::env::var_os("USER"),
    )
}

fn socket_path_for(xdg_runtime_dir: Option<OsString>, user: Option<OsString>) -> PathBuf {
    if let Some(dir) = xdg_runtime_dir {
        let mut path = PathBuf::from(dir);
        path.push("assistd.sock");
        return path;
    }
    let user = user
        .and_then(|u| u.into_string().ok())
        .unwrap_or_else(|| "nobody".into());
    PathBuf::from(format!("/tmp/assistd-{user}.sock"))
}

#[cfg(test)]
mod tests;
