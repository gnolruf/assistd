//! The lines a client sends: [`Request`], what its [`RequestBody`] asks for, and the bus filter.

use serde::{Deserialize, Serialize};

use crate::{
    EventKind, ImageAttachment, PermissionMode, PlanMode, PlanVerdict, PresenceTarget, SearchKind,
    TaskRequest, WakeWordOp,
};

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

/// Client-to-daemon request, one JSON line: `id` beside the fields of `body`. A client sends one
/// request and closes its write half unless answering
/// [`EventBody::ConfirmRequest`](crate::EventBody::ConfirmRequest) or
/// [`EventBody::PlanReview`](crate::EventBody::PlanReview).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Request {
    /// Correlation id, echoed on every [`Event`](crate::Event) sent in response.
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
            RequestBody::MemoryReindex => "memory_reindex",
            RequestBody::DocumentsStatus => "documents_status",
            RequestBody::DocumentsReindex => "documents_reindex",
            RequestBody::Search { .. } => "search",
            RequestBody::ConfirmResponse { .. } => "confirm_response",
            RequestBody::SetPermissionMode { .. } => "set_permission_mode",
            RequestBody::PlanVerdict { .. } => "plan_verdict",
            RequestBody::GetCapabilities => "get_capabilities",
            RequestBody::Fork { .. } => "fork",
            RequestBody::Branches => "branches",
            RequestBody::Switch { .. } => "switch",
            RequestBody::Undo => "undo",
            RequestBody::ResumeOrNew { .. } => "resume_or_new",
            RequestBody::ReplayBranch => "replay_branch",
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
    /// Have the embedding worker retry the rows it marked as refused. Emits `ReindexProgress`es
    /// until no row waits to be embedded, then `Done`.
    MemoryReindex,
    /// Report the document index and every index root. Emits `DocumentIndex`, one
    /// `DocumentRoot` per `documents.roots` entry, then `Done`.
    DocumentsStatus,
    /// Start a pass over every index root and follow it. Emits `DocumentPassProgress`es and
    /// one `DocumentPassEnded` per root, then `Done`; errors when no extraction worker runs.
    DocumentsReindex,
    /// Rank the stored text of `kinds` by meaning and by the words of `query`; no kind
    /// means every kind turned on, and `limit = 0` uses `embedding.top_k`. Emits `SearchHit`s
    /// best-first, then `Done`; errors when a kind named is turned off.
    Search {
        query: String,
        #[serde(default)]
        limit: u32,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        kinds: Vec<SearchKind>,
    },
    /// Answer the [`EventBody::ConfirmRequest`](crate::EventBody::ConfirmRequest) with this
    /// `confirm_id` on the same connection.
    ConfirmResponse {
        confirm_id: String,
        allow: bool,
        /// With `allow`, permanently allowlist the prompt's `always_allow` programs.
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        always: bool,
        /// Words for the agent whose command this is, shown in the command's result. A
        /// message that is not blank denies the command whatever `allow` says.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        message: Option<String>,
    },
    /// Switch how the rest of the turn's prompts on the same connection are answered: under
    /// `mode`, or with `plan` refused while the turn plans; honored only after a `Query`.
    SetPermissionMode {
        mode: PermissionMode,
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        plan: bool,
    },
    /// Answer the [`EventBody::PlanReview`](crate::EventBody::PlanReview) with this `review_id` on
    /// the same connection.
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
    /// Replay the current branch however old it is. Emits `BranchSwitched`, `HistoryEntry`s,
    /// then `Done`.
    ReplayBranch,
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
    /// current `TaskQueue`, `Presence`, `ListenState`, `VoiceState` and `DocumentActivity`,
    /// then sends every presence change, each tagged with this request's `id`.
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
