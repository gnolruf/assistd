//! Wire types for the daemon's Unix-socket protocol: line-delimited JSON,
//! with [`Request`] lines from the client and [`Event`] lines back.

use base64::Engine;
use serde::{Deserialize, Serialize};
use std::ffi::OsString;
use std::path::PathBuf;

#[cfg(feature = "client")]
pub mod attachment;
#[cfg(feature = "client")]
pub mod client;
#[cfg(feature = "client")]
pub use client::{DialogConnection, EventStream, IpcClient, IpcClientError};

/// An image attachment on a [`Request::Query`]. `data_base64` is
/// standard padded base64; `mime` is `image/png`, `image/jpeg` or
/// `image/webp`.
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
            data_base64: base64::engine::general_purpose::STANDARD.encode(bytes),
        }
    }

    /// Decode `data_base64`; errors when it is not valid standard base64.
    pub fn decode_bytes(&self) -> Result<Vec<u8>, base64::DecodeError> {
        base64::engine::general_purpose::STANDARD.decode(&self.data_base64)
    }
}

/// Coarse daemon lifecycle state. `Sleeping` means llama-server is
/// fully stopped; `Drowsy` keeps the process alive but its model weights
/// unloaded; `Active` is fully ready to answer queries.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PresenceState {
    Active,
    Drowsy,
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

/// Push-to-talk capture state. `Queued` means the transcriber is
/// waiting for the GPU to free up before inference. `Idle` must stay
/// the first variant.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum VoiceCaptureState {
    Idle,
    Queued,
    Recording,
    Transcribing,
}

/// Categories of [`Event`] that pass through the daemon-wide
/// broadcast bus and so can be selected by a [`SubscribeFilter`].
/// Dialog-local events stay scoped to the originating connection
/// and have no [`EventKind`] variant.
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

/// Set-of-kinds filter for [`Request::Subscribe`]. An empty `kinds`
/// vector matches every broadcast-eligible kind.
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

/// Request sent by a client to the daemon, as one JSON line with a
/// `"type"` discriminant. Every variant carries an `id` that is echoed
/// on every [`Event`] emitted in response. A client sends one request
/// and shuts down its write half, except on a connection that must
/// answer an [`Event::ConfirmRequest`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Request {
    /// Submit a text prompt (with optional image attachments) to the daemon.
    Query {
        id: String,
        text: String,
        /// Image attachments to surface as vision inputs on this turn.
        /// Empty for text-only queries.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        attachments: Vec<ImageAttachment>,
    },
    /// Drive the daemon to a specific presence state.
    SetPresence { id: String, target: PresenceState },
    /// Report the daemon's current presence state.
    GetPresence { id: String },
    /// Atomically advance the daemon one step along
    /// `Active → Drowsy → Sleeping → Active`.
    Cycle { id: String },
    /// Begin a push-to-talk recording. Returns immediately with `Done`
    /// once the input device is open; audio is buffered in the daemon
    /// until a matching `PttStop` arrives.
    PttStart { id: String },
    /// End the push-to-talk recording and transcribe what was captured.
    /// Emits `VoiceState::Transcribing`, then `Transcription { text }`,
    /// then the text is dispatched internally as a `Query` whose
    /// streaming `Delta`s flow back on the same connection before `Done`.
    PttStop { id: String },
    /// Enable hands-free continuous listening. The daemon keeps the mic
    /// open and auto-dispatches each VAD-segmented utterance as a
    /// `Query`. Emits `ListenState { active: true }` + `Done`.
    /// Rejects with `Error` when a PTT recording is already in flight.
    ListenStart { id: String },
    /// Disable continuous listening. Emits `ListenState { active: false }`
    /// + `Done`. Idempotent when already stopped.
    ListenStop { id: String },
    /// Flip continuous listening on/off in a single call. Emits the
    /// post-toggle `ListenState` + `Done`.
    ListenToggle { id: String },
    /// Report whether continuous listening is currently active. Emits
    /// `ListenState` + `Done` with no state change.
    GetListenState { id: String },
    /// Flip TTS on/off at runtime. Off cancels in-flight playback and
    /// drains any subsequent sentences for the active query without
    /// speaking them; on resumes for the next sentence delivered. Emits
    /// the post-toggle `VoiceOutputState` + `Done`.
    VoiceToggle { id: String },
    /// Abort the current TTS response: drop the rest of the audio queue
    /// and any pending sentences for the active query. Does not change
    /// the enabled flag. Emits `VoiceOutputState` + `Done`.
    VoiceSkip { id: String },
    /// Cancel the in-flight agent turn (if any) and drop queued TTS
    /// audio. Idempotent. Emits `Done`.
    InterruptTurn { id: String },
    /// Report whether TTS is currently enabled. Emits `VoiceOutputState`
    /// + `Done` with no state change.
    GetVoiceState { id: String },
    /// Persist a string value under `key`. Overwrites any existing
    /// value at the same key. Emits `Done` (no payload) on success.
    MemorySave {
        id: String,
        key: String,
        value: String,
    },
    /// Read the value previously stored at `key`. Emits a single
    /// `MemoryValue` (with `value: None` when the key is absent) and
    /// then `Done`.
    MemoryLoad { id: String, key: String },
    /// Enumerate keys whose name starts with `prefix`. Emits a single
    /// `MemoryKeys` event with the matching keys, then `Done`. An
    /// empty prefix lists every key.
    MemoryList {
        id: String,
        #[serde(default)]
        prefix: String,
    },
    /// Enumerate full `(id, key, value)` rows whose key starts with
    /// `prefix`. Streams one [`Event::MemoryRow`] per match in
    /// lexicographic key order, then a terminal `Done`. `limit = 0`
    /// means "no cap".
    MemoryListAll {
        id: String,
        #[serde(default)]
        prefix: String,
        #[serde(default)]
        limit: u32,
    },
    /// Remove `key` from the memory store. No-op when absent. Emits
    /// `Done` on success.
    MemoryDelete { id: String, key: String },
    /// Remove the memory whose row id is `memory_id`. Emits a single
    /// [`Event::MemoryForgetResult`] (with `deleted: false` when the
    /// id didn't match any row), then a terminal `Done`. Unlike
    /// `MemoryDelete`, addresses the row by id and reports the deleted
    /// key.
    MemoryForget { id: String, memory_id: i64 },
    /// Semantic search over persisted conversation chunks. Embeds the
    /// query and ranks past messages by cosine similarity. Emits zero
    /// or more `SemanticHit` events ordered best-first, then `Done`.
    /// `limit = 0` is treated as the daemon's default cap.
    MemorySemanticSearch {
        id: String,
        query: String,
        #[serde(default)]
        limit: u32,
    },
    /// Re-embed every memory and conversation-chunk row that has no
    /// embedding under the currently configured embedding model, such
    /// as rows written before a model swap or while the embedder was
    /// unavailable. Emits one [`Event::ReindexProgress`] per
    /// kind/transition plus per item processed, then a terminal `Done`
    /// (or `Error` on a fatal embedder failure).
    MemoryReindex { id: String },
    /// Reply to a daemon-issued [`Event::ConfirmRequest`], sent on the
    /// same connection as the originating request and routed by
    /// `confirm_id`.
    ConfirmResponse {
        id: String,
        confirm_id: String,
        allow: bool,
    },
    /// Probe the daemon's runtime capabilities. Emits a single
    /// [`Event::Capabilities`] then `Done`.
    GetCapabilities { id: String },
    /// Snapshot the current conversation state into a new branch named
    /// `name` and switch to it. Emits a single [`Event::BranchSwitched`]
    /// describing the new branch, then `Done`. Errors when the name is
    /// already taken or empty.
    Fork { id: String, name: String },
    /// Enumerate every branch across every session. Emits zero or more
    /// [`Event::BranchInfo`] events (active session first), then `Done`.
    Branches { id: String },
    /// Switch the active conversation to a different branch.
    /// `target` is either a bare branch name (resolved to the current
    /// session's branch first, then most-recent session on collision)
    /// or `<session_prefix>/<name>` for an explicit cross-session jump.
    /// Emits a [`Event::BranchSwitched`] followed by one
    /// [`Event::HistoryEntry`] per loaded message, then `Done`.
    Switch { id: String, target: String },
    /// Drop the most recent user message and the entire assistant
    /// reply that followed it (including any tool-call/result rounds)
    /// from the current branch. Emits a single [`Event::UndoApplied`]
    /// reporting how many messages were removed, then `Done`.
    Undo { id: String },
    /// Resume the current branch or start a fresh conversation. If the
    /// latest message on the current branch was written within
    /// `recency_secs`, the daemon keeps that branch and streams its
    /// history back (one [`Event::HistoryEntry`] per message).
    /// Otherwise it creates a new session with an empty `main` branch,
    /// sets it as current, and emits [`Event::BranchSwitched`]. Either
    /// path terminates with `Done`.
    ResumeOrNew { id: String, recency_secs: u64 },
    /// Unconditionally start a fresh conversation. Creates a new
    /// session with an empty `main` branch, sets it as current, emits
    /// [`Event::BranchSwitched`], then `Done`. Unlike
    /// [`Request::ResumeOrNew`], never keeps the existing branch.
    NewSession { id: String },
    /// Attach a passive subscriber to the daemon-wide events bus.
    /// Forwards every broadcast-eligible event that matches
    /// `filter` until the client disconnects. No terminal `Done`
    /// is emitted for the subscription itself; events carry the
    /// originating turn's `id`, not `Subscribe.id`. A forwarded
    /// [`Event::ToolResult`] omits the result's `attachments`; only the
    /// connection that made the request receives the images.
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

    /// Stable snake_case name of the variant, identical to its wire
    /// `"type"` tag.
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

impl std::fmt::Display for StatusSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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

impl std::fmt::Display for Component {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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

impl std::fmt::Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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

impl std::fmt::Display for ReindexKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.pad(self.as_str())
    }
}

/// Events streamed from the daemon to a client, as JSON lines with a
/// `"type"` discriminant. A response stream ends with exactly one
/// terminal [`Event::Done`] or [`Event::Error`], after which the daemon
/// closes the connection; a [`Request::Subscribe`] stream has no
/// terminal event.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Event {
    /// A streamed chunk of response text.
    Delta { id: String, text: String },
    /// A streamed chunk of the model's reasoning content, distinct from
    /// the reply text carried by `Delta`.
    ReasoningDelta { id: String, text: String },
    /// The model asked to invoke a tool.
    ToolCall {
        id: String,
        name: String,
        args: serde_json::Value,
    },
    /// Result of a tool invocation.
    ToolResult {
        id: String,
        name: String,
        result: serde_json::Value,
    },
    /// Daemon presence state, emitted in response to GetPresence or after a
    /// successful SetPresence transition.
    Presence { id: String, state: PresenceState },
    /// Push-to-talk capture state transition. Emitted when the daemon's
    /// mic pipeline moves between Idle / Queued / Recording / Transcribing.
    VoiceState {
        id: String,
        state: VoiceCaptureState,
    },
    /// Final whisper transcription emitted once on `PttStop`, before the
    /// text is dispatched internally as a `Query`. Empty string means VAD
    /// trimmed the audio down to silence; no `Query` follows in that
    /// case, only a terminal `Done`.
    Transcription { id: String, text: String },
    /// Current state of continuous listening. Emitted in response to
    /// `ListenStart` / `ListenStop` / `ListenToggle` / `GetListenState`.
    ListenState { id: String, active: bool },
    /// Current TTS enabled state. Emitted in response to `VoiceToggle`
    /// / `VoiceSkip` / `GetVoiceState`. Skip leaves the flag unchanged
    /// (true if synthesis was on before the skip).
    VoiceOutputState { id: String, enabled: bool },
    /// TTS playback state for a turn: `speaking: true` on the first
    /// enqueued sentence, `false` once the playback queue drains.
    SpeakingState { id: String, speaking: bool },
    /// A display title for a session, broadcast whenever one is
    /// generated or loaded. Generation runs in the background, so this
    /// can arrive well after the triggering turn's `Done`.
    SessionTitle {
        id: String,
        session_id: String,
        title: String,
    },
    /// One semantic-search hit, best first. `content` is the full
    /// parent message, not a snippet; `similarity` is in `[0.0, 1.0]`.
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
    /// Result of a `MemoryLoad`. `value` is `None` when the key was
    /// absent; the daemon still emits the event so the client knows
    /// the lookup completed.
    MemoryValue {
        id: String,
        key: String,
        value: Option<String>,
    },
    /// Result of a `MemoryList`. Keys are returned in lexicographic
    /// order. Always emitted exactly once before the terminal `Done`,
    /// even when empty.
    MemoryKeys { id: String, keys: Vec<String> },
    /// One `(id, key, value)` row emitted by `MemoryListAll`. The
    /// daemon streams these in lexicographic key order, then a
    /// terminal `Done`. `memory_id` is the row id accepted by
    /// [`Request::MemoryForget`].
    MemoryRow {
        id: String,
        memory_id: i64,
        key: String,
        value: String,
    },
    /// Result of a `MemoryForget`, emitted exactly once before `Done`.
    /// `deleted = false` with `key = None` means no row had that id.
    MemoryForgetResult {
        id: String,
        deleted: bool,
        key: Option<String>,
    },
    /// Progress of a `MemoryReindex` run, one per item.
    ReindexProgress {
        id: String,
        kind: ReindexKind,
        done: u32,
        total: u32,
    },
    /// Mid-stream prompt to authorize a destructive tool action. The
    /// turn is parked until a [`Request::ConfirmResponse`] with the same
    /// `confirm_id` arrives on this connection. A client that has closed
    /// its write side never receives the prompt; a dropped connection or
    /// an unanswered prompt past the daemon's confirmation timeout
    /// denies.
    ConfirmRequest {
        id: String,
        confirm_id: String,
        tool: String,
        script: String,
        matched_pattern: String,
    },
    /// Response to [`Request::GetCapabilities`]. `vision` is true when
    /// the loaded model accepts images; `model_name` is the basename of
    /// the configured `model.name`.
    Capabilities {
        id: String,
        vision: bool,
        model_name: String,
    },
    /// Non-terminal status update for a recoverable condition; a `Done`
    /// or `Error` still follows.
    Status {
        id: String,
        severity: StatusSeverity,
        component: Component,
        event: StatusKind,
        message: String,
    },
    /// One branch emitted by [`Request::Branches`], active session
    /// first.
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
    /// Emitted by [`Request::Fork`] and [`Request::Switch`] to confirm
    /// the active branch changed. After a cross-session switch,
    /// `session_id` is the session the daemon is now active in.
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
    /// Emitted by [`Request::Undo`]. `removed_messages` is the count of
    /// rows dropped from the current branch; 0 means "nothing to undo"
    /// (the branch had no real user turn). `last_user_text` echoes the
    /// undone prompt.
    UndoApplied {
        id: String,
        removed_messages: u32,
        last_user_text: Option<String>,
    },
    /// Terminal error event; the stream is over.
    Error { id: String, message: String },
    /// Terminal success event; the stream is over.
    Done { id: String },
    /// The running reply so far, sent only to [`Request::Subscribe`]
    /// connections.
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

    /// The broadcast kind of this event, or `None` for dialog-local
    /// events. Exhaustive so a new variant must decide its eligibility.
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

/// Return the assistd daemon socket path for the current user.
///
/// Prefers `$XDG_RUNTIME_DIR/assistd.sock`; falls back to
/// `/tmp/assistd-$USER.sock` (or `/tmp/assistd-nobody.sock` when `$USER`
/// is unset).
pub fn socket_path() -> PathBuf {
    socket_path_for(
        std::env::var_os("XDG_RUNTIME_DIR"),
        std::env::var_os("USER"),
    )
}

fn socket_path_for(xdg_runtime_dir: Option<OsString>, user: Option<OsString>) -> PathBuf {
    if let Some(dir) = xdg_runtime_dir {
        let mut p = PathBuf::from(dir);
        p.push("assistd.sock");
        return p;
    }
    let user = user
        .and_then(|u| u.into_string().ok())
        .unwrap_or_else(|| "nobody".into());
    PathBuf::from(format!("/tmp/assistd-{user}.sock"))
}

#[cfg(test)]
mod tests;
