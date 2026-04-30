#![allow(unsafe_code)] // libc / env / fd primitives — each unsafe block is locally justified
#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::print_stdout,
        clippy::print_stderr
    )
)]

//! Wire-level IPC types shared between the assistd daemon and its clients.
//!
//! Kept in its own crate so a client-only build can depend on just these
//! types without pulling in the daemon's subsystem crates.
//!
//! # Protocol
//!
//! The daemon listens on a Unix domain socket and speaks a line-delimited
//! JSON protocol. A client sends exactly one [`Request`] line and shuts
//! down its write half, then reads zero or more [`Event`] lines until the
//! daemon sends [`Event::Done`] (success) or [`Event::Error`] (failure) and
//! closes the connection.
//!
//! Every event carries the originating request's `id`, so a future
//! multiplexing transport can correlate concurrent in-flight requests.

use base64::Engine;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Wire-protocol version. Bumped when an existing field's semantics
/// change in a non-additive way; new optional fields don't bump this.
/// Servers log a warning and continue when they see a version they
/// don't understand, so additive evolution stays compatible.
pub const PROTOCOL_VERSION: u32 = 1;

/// An image attachment carried over the wire alongside a [`Request::Query`].
/// `data_base64` is standard base64 (with padding); the daemon decodes
/// it back into raw bytes before handing it to the LLM. `mime` is one of
/// the values `assistd-tools::attachment` accepts (image/png, image/jpeg,
/// image/webp).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ImageAttachment {
    pub mime: String,
    pub data_base64: String,
}

impl ImageAttachment {
    /// Build from raw bytes. Allocates the base64-encoded string.
    pub fn from_bytes(mime: impl Into<String>, bytes: &[u8]) -> Self {
        Self {
            mime: mime.into(),
            data_base64: base64::engine::general_purpose::STANDARD.encode(bytes),
        }
    }

    /// Decode `data_base64` back to bytes. Returns `Err` if the field is
    /// not valid base64.
    pub fn decode_bytes(&self) -> Result<Vec<u8>, base64::DecodeError> {
        base64::engine::general_purpose::STANDARD.decode(&self.data_base64)
    }
}

/// Coarse daemon lifecycle state exposed on the wire so clients and the
/// daemon can agree on resource usage. `Sleeping` means llama-server is
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

/// Push-to-talk capture state exposed on the wire so the TUI can render a
/// four-state indicator. `Transcribing` is distinct from `Recording` because
/// whisper inference takes 1–3 s on a few seconds of audio and users
/// otherwise keep talking into dead air. `Queued` sits between them when the
/// GPU is busy with an LLM stream — the transcriber is briefly waiting for
/// the GPU before starting inference (or deciding to fall back to CPU).
///
/// `Idle` is pinned to discriminant 0; a unit test in
/// `assistd-voice::mic` guards the invariant so reordering the variants
/// without updating TUI defaults triggers a build failure.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum VoiceCaptureState {
    Idle,
    Queued,
    Recording,
    Transcribing,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Request {
    Query {
        id: String,
        text: String,
        /// Image attachments to surface as vision inputs on this turn.
        /// Empty for text-only queries; deserializes from missing fields
        /// for backward compat.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        attachments: Vec<ImageAttachment>,
        /// Wire-protocol version the client speaks. None means a
        /// legacy client (pre-versioning); the daemon accepts these
        /// for now but logs a warning.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        version: Option<u32>,
    },
    /// Drive the daemon to a specific presence state.
    SetPresence { id: String, target: PresenceState },
    /// Report the daemon's current presence state.
    GetPresence { id: String },
    /// Atomically advance the daemon one step along
    /// `Active → Drowsy → Sleeping → Active`.
    Cycle { id: String },
    /// Begin a push-to-talk recording. Returns immediately with `Done`
    /// once cpal has opened the device; audio is buffered in the daemon
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
    /// id didn't match any row), then a terminal `Done`. Distinct from
    /// `MemoryDelete` so the CLI can take an integer id and the daemon
    /// can echo back the deleted key.
    MemoryForget { id: String, memory_id: i64 },
    /// Semantic search over persisted conversation chunks. Embeds the
    /// query and ranks past messages by cosine similarity. Emits zero
    /// or more `SemanticHit` events ordered best-first, then `Done`.
    /// `limit = 0` is treated as the daemon's default cap. Backs the
    /// `assistd memory reminisce` CLI subcommand and the LLM-callable
    /// `reminisce` tool.
    MemorySemanticSearch {
        id: String,
        query: String,
        #[serde(default)]
        limit: u32,
    },
    /// Re-embed every memory and conversation-chunk row that has no
    /// embedding under the daemon's currently-configured embedding
    /// model. Used to recover after a model swap or after a run where
    /// the embedding subsystem was unavailable. Emits one
    /// [`Event::ReindexProgress`] per kind/transition plus per item
    /// processed, then a terminal `Done` (or `Error` on a fatal
    /// embedder failure). Backs `assistd memory reindex`.
    MemoryReindex { id: String },
}

impl Request {
    /// Build a text-only [`Request::Query`] tagged with the current
    /// [`PROTOCOL_VERSION`]. Use this in clients to keep call sites
    /// short; legacy struct-literal construction still works for
    /// callers that need to set `attachments` explicitly.
    pub fn query(id: impl Into<String>, text: impl Into<String>) -> Self {
        Request::Query {
            id: id.into(),
            text: text.into(),
            attachments: Vec::new(),
            version: Some(PROTOCOL_VERSION),
        }
    }

    /// Build a [`Request::Query`] with one or more image attachments,
    /// tagged with the current [`PROTOCOL_VERSION`].
    pub fn query_with_attachments(
        id: impl Into<String>,
        text: impl Into<String>,
        attachments: Vec<ImageAttachment>,
    ) -> Self {
        Request::Query {
            id: id.into(),
            text: text.into(),
            attachments,
            version: Some(PROTOCOL_VERSION),
        }
    }

    /// Returns the request id every variant carries.
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
            | Request::GetVoiceState { id }
            | Request::MemorySave { id, .. }
            | Request::MemoryLoad { id, .. }
            | Request::MemoryList { id, .. }
            | Request::MemoryListAll { id, .. }
            | Request::MemoryDelete { id, .. }
            | Request::MemoryForget { id, .. }
            | Request::MemorySemanticSearch { id, .. }
            | Request::MemoryReindex { id, .. } => id,
        }
    }

    /// Short, stable name for the variant — used as a span field so
    /// concurrent requests can be filtered by kind in trace output.
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
            Request::GetVoiceState { .. } => "get_voice_state",
            Request::MemorySave { .. } => "memory_save",
            Request::MemoryLoad { .. } => "memory_load",
            Request::MemoryList { .. } => "memory_list",
            Request::MemoryListAll { .. } => "memory_list_all",
            Request::MemoryDelete { .. } => "memory_delete",
            Request::MemoryForget { .. } => "memory_forget",
            Request::MemorySemanticSearch { .. } => "memory_semantic_search",
            Request::MemoryReindex { .. } => "memory_reindex",
        }
    }
}

/// Note: `Eq` is intentionally not derived. `Event::SemanticHit`
/// carries an `f32` similarity score and `f32: !Eq`. Tests compare
/// events with `PartialEq` (sufficient for `assert_eq!`) — an `Eq`
/// bound is only needed for hashing/Set membership, neither of which
/// the IPC layer requires.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Event {
    /// A streamed chunk of response text.
    Delta { id: String, text: String },
    /// The model asked to invoke a tool. (Reserved for future milestones.)
    ToolCall {
        id: String,
        name: String,
        args: serde_json::Value,
    },
    /// Result of a tool invocation. (Reserved for future milestones.)
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
    /// trimmed the audio down to silence — no `Query` follows in that
    /// case, only a terminal `Done`.
    Transcription { id: String, text: String },
    /// Current state of continuous listening. Emitted in response to
    /// `ListenStart` / `ListenStop` / `ListenToggle` / `GetListenState`.
    ListenState { id: String, active: bool },
    /// Current TTS enabled state. Emitted in response to `VoiceToggle`
    /// / `VoiceSkip` / `GetVoiceState`. Skip leaves the flag unchanged
    /// (true if synthesis was on before the skip).
    VoiceOutputState { id: String, enabled: bool },
    /// One semantic-search hit emitted by `MemorySemanticSearch`. The
    /// daemon emits zero or more of these ranked by cosine similarity
    /// (best-first), then a terminal `Done`. `content` is the *full*
    /// parent message text (not a snippet) — chunks may cut
    /// mid-sentence, so the surface message is the useful unit for the
    /// model. `similarity` is in `[0.0, 1.0]`.
    SemanticHit {
        id: String,
        conversation_id: i64,
        chunk_id: i64,
        session_id: String,
        timestamp: String,
        role: String,
        content: String,
        similarity: f32,
    },
    /// Result of a `MemoryLoad`. `value` is `None` when the key was
    /// absent — the daemon still emits the event so the client knows
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
    /// terminal `Done`. `memory_id` is the SQLite row id — the CLI
    /// uses it as the argument to `assistd memory forget <id>`.
    MemoryRow {
        id: String,
        memory_id: i64,
        key: String,
        value: String,
    },
    /// Result of a `MemoryForget`. Always emitted exactly once before
    /// the terminal `Done`. `deleted = false` (with `key = None`)
    /// signals that no row with the given id existed — the CLI maps
    /// this to a "no memory with id=N" stderr message and exit 2.
    /// `key = Some(k)` carries the deleted row's key so the CLI can
    /// echo `forgot id=N key=k`.
    MemoryForgetResult {
        id: String,
        deleted: bool,
        key: Option<String>,
    },
    /// Progress update for a `MemoryReindex` run. `kind` is `"chunks"`
    /// or `"memories"`; `done` is how many of `total` rows of that kind
    /// have been embedded so far. The daemon emits these incrementally
    /// (one per item) so the CLI can render a progress meter. The
    /// stream terminates with `Done` after both kinds finish.
    ReindexProgress {
        id: String,
        kind: String,
        done: u32,
        total: u32,
    },
    /// Terminal error event — the stream is over.
    Error { id: String, message: String },
    /// Terminal success event — the stream is over.
    Done { id: String },
}

impl Event {
    /// Returns true if this event terminates a response stream.
    pub fn is_terminal(&self) -> bool {
        matches!(self, Event::Done { .. } | Event::Error { .. })
    }

    /// Returns the request id this event is associated with.
    pub fn id(&self) -> &str {
        match self {
            Event::Delta { id, .. }
            | Event::ToolCall { id, .. }
            | Event::ToolResult { id, .. }
            | Event::Presence { id, .. }
            | Event::VoiceState { id, .. }
            | Event::Transcription { id, .. }
            | Event::ListenState { id, .. }
            | Event::VoiceOutputState { id, .. }
            | Event::SemanticHit { id, .. }
            | Event::MemoryValue { id, .. }
            | Event::MemoryKeys { id, .. }
            | Event::MemoryRow { id, .. }
            | Event::MemoryForgetResult { id, .. }
            | Event::ReindexProgress { id, .. }
            | Event::Error { id, .. }
            | Event::Done { id } => id,
        }
    }
}

pub fn socket_path() -> PathBuf {
    if let Some(dir) = std::env::var_os("XDG_RUNTIME_DIR") {
        let mut p = PathBuf::from(dir);
        p.push("assistd.sock");
        return p;
    }
    let user = std::env::var("USER").unwrap_or_else(|_| "nobody".into());
    PathBuf::from(format!("/tmp/assistd-{user}.sock"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_roundtrip() {
        // Legacy struct-literal form: empty attachments + no version are
        // both skipped on serialize so the wire shape is unchanged from
        // pre-multimodality clients.
        let req = Request::Query {
            id: "req-1".into(),
            text: "ping".into(),
            attachments: Vec::new(),
            version: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        assert_eq!(json, r#"{"type":"query","id":"req-1","text":"ping"}"#);
        let parsed: Request = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, req);
    }

    #[test]
    fn request_query_with_version_and_attachments_roundtrip() {
        let req = Request::query_with_attachments(
            "req-2",
            "describe this",
            vec![ImageAttachment::from_bytes(
                "image/png",
                &[0xDE, 0xAD, 0xBE, 0xEF],
            )],
        );
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains(r#""version":1"#));
        assert!(json.contains(r#""data_base64":"3q2+7w==""#));
        let parsed: Request = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, req);
    }

    #[test]
    fn request_query_helper_emits_current_version() {
        let req = Request::query("req-3", "hi");
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains(&format!(r#""version":{PROTOCOL_VERSION}"#)));
    }

    #[test]
    fn legacy_query_payload_deserializes_with_default_attachments() {
        // A pre-multimodality client emits the original two-field shape.
        // Older daemons keep accepting it: attachments defaults to empty,
        // version defaults to None.
        let json = r#"{"type":"query","id":"req-4","text":"hello"}"#;
        let parsed: Request = serde_json::from_str(json).unwrap();
        match parsed {
            Request::Query {
                id,
                text,
                attachments,
                version,
            } => {
                assert_eq!(id, "req-4");
                assert_eq!(text, "hello");
                assert!(attachments.is_empty());
                assert_eq!(version, None);
            }
            _ => panic!("expected Query"),
        }
    }

    #[test]
    fn image_attachment_round_trips_through_base64() {
        let payload = b"\x89PNG\r\n\x1a\n";
        let att = ImageAttachment::from_bytes("image/png", payload);
        assert_eq!(att.decode_bytes().unwrap(), payload);
    }

    #[test]
    fn delta_event_roundtrip() {
        let evt = Event::Delta {
            id: "req-1".into(),
            text: "pong".into(),
        };
        let json = serde_json::to_string(&evt).unwrap();
        assert_eq!(json, r#"{"type":"delta","id":"req-1","text":"pong"}"#);
        let parsed: Event = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, evt);
    }

    #[test]
    fn done_event_roundtrip() {
        let evt = Event::Done { id: "req-1".into() };
        let json = serde_json::to_string(&evt).unwrap();
        assert_eq!(json, r#"{"type":"done","id":"req-1"}"#);
        let parsed: Event = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, evt);
    }

    #[test]
    fn error_event_roundtrip() {
        let evt = Event::Error {
            id: "req-1".into(),
            message: "boom".into(),
        };
        let json = serde_json::to_string(&evt).unwrap();
        assert_eq!(json, r#"{"type":"error","id":"req-1","message":"boom"}"#);
        let parsed: Event = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, evt);
    }

    #[test]
    fn tool_call_event_roundtrip() {
        let evt = Event::ToolCall {
            id: "req-1".into(),
            name: "echo".into(),
            args: serde_json::json!({"text": "hi"}),
        };
        let parsed: Event = serde_json::from_str(&serde_json::to_string(&evt).unwrap()).unwrap();
        assert_eq!(parsed, evt);
    }

    #[test]
    fn memory_save_load_list_delete_request_roundtrip() {
        let cases = vec![
            Request::MemorySave {
                id: "r1".into(),
                key: "k".into(),
                value: "v".into(),
            },
            Request::MemoryLoad {
                id: "r2".into(),
                key: "k".into(),
            },
            Request::MemoryList {
                id: "r3".into(),
                prefix: "pref:".into(),
            },
            Request::MemoryDelete {
                id: "r4".into(),
                key: "k".into(),
            },
            Request::MemoryListAll {
                id: "r5".into(),
                prefix: "fact:".into(),
                limit: 0,
            },
            Request::MemoryForget {
                id: "r6".into(),
                memory_id: 42,
            },
        ];
        for r in cases {
            let parsed: Request =
                serde_json::from_str(&serde_json::to_string(&r).unwrap()).unwrap();
            assert_eq!(parsed, r);
        }
    }

    #[test]
    fn memory_list_all_request_omits_optional_fields() {
        // Both `prefix` and `limit` use `#[serde(default)]`, so a
        // pre-feature client that only sends `id` should still parse
        // (additive evolution).
        let json = r#"{"type":"memory_list_all","id":"r"}"#;
        let parsed: Request = serde_json::from_str(json).unwrap();
        match parsed {
            Request::MemoryListAll { id, prefix, limit } => {
                assert_eq!(id, "r");
                assert_eq!(prefix, "");
                assert_eq!(limit, 0);
            }
            _ => panic!("expected MemoryListAll"),
        }
    }

    #[test]
    fn memory_forget_request_carries_id() {
        let req = Request::MemoryForget {
            id: "r".into(),
            memory_id: 7,
        };
        assert_eq!(req.id(), "r");
        assert_eq!(req.kind(), "memory_forget");
    }

    #[test]
    fn memory_reindex_request_round_trips() {
        let req = Request::MemoryReindex { id: "r".into() };
        let json = serde_json::to_string(&req).unwrap();
        assert_eq!(json, r#"{"type":"memory_reindex","id":"r"}"#);
        let parsed: Request = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, req);
        assert_eq!(req.kind(), "memory_reindex");
    }

    #[test]
    fn reindex_progress_event_round_trips() {
        let ev = Event::ReindexProgress {
            id: "r".into(),
            kind: "chunks".into(),
            done: 3,
            total: 10,
        };
        let json = serde_json::to_string(&ev).unwrap();
        let parsed: Event = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, ev);
        assert_eq!(ev.id(), "r");
        assert!(!ev.is_terminal());
    }

    #[test]
    fn memory_event_roundtrip() {
        let cases = vec![
            Event::SemanticHit {
                id: "r".into(),
                conversation_id: 42,
                chunk_id: 7,
                session_id: "s".into(),
                timestamp: "2026-04-28T00:00:00Z".into(),
                role: "user".into(),
                content: "the rust embeddings daemon".into(),
                similarity: 0.87,
            },
            Event::MemoryValue {
                id: "r".into(),
                key: "k".into(),
                value: Some("v".into()),
            },
            Event::MemoryValue {
                id: "r".into(),
                key: "absent".into(),
                value: None,
            },
            Event::MemoryKeys {
                id: "r".into(),
                keys: vec!["a".into(), "b".into()],
            },
            Event::MemoryRow {
                id: "r".into(),
                memory_id: 17,
                key: "fact:user.name".into(),
                value: "Ben".into(),
            },
            Event::MemoryForgetResult {
                id: "r".into(),
                deleted: true,
                key: Some("fact:user.name".into()),
            },
            Event::MemoryForgetResult {
                id: "r".into(),
                deleted: false,
                key: None,
            },
        ];
        for e in cases {
            let parsed: Event = serde_json::from_str(&serde_json::to_string(&e).unwrap()).unwrap();
            assert_eq!(parsed, e);
        }
    }

    #[test]
    fn memory_row_and_forget_result_are_not_terminal() {
        let row = Event::MemoryRow {
            id: "r".into(),
            memory_id: 1,
            key: "k".into(),
            value: "v".into(),
        };
        assert!(!row.is_terminal());
        assert_eq!(row.id(), "r");

        let forget = Event::MemoryForgetResult {
            id: "f".into(),
            deleted: true,
            key: Some("k".into()),
        };
        assert!(!forget.is_terminal());
        assert_eq!(forget.id(), "f");
    }

    #[test]
    fn memory_semantic_search_request_roundtrip() {
        let req = Request::MemorySemanticSearch {
            id: "ms-1".into(),
            query: "the rust thing we discussed".into(),
            limit: 5,
        };
        let parsed: Request = serde_json::from_str(&serde_json::to_string(&req).unwrap()).unwrap();
        assert_eq!(parsed, req);
        assert_eq!(req.id(), "ms-1");
        assert_eq!(req.kind(), "memory_semantic_search");
    }

    #[test]
    fn is_terminal_identifies_done_and_error() {
        assert!(Event::Done { id: "x".into() }.is_terminal());
        assert!(
            Event::Error {
                id: "x".into(),
                message: "e".into()
            }
            .is_terminal()
        );
        assert!(
            !Event::Delta {
                id: "x".into(),
                text: "t".into()
            }
            .is_terminal()
        );
    }

    #[test]
    fn set_presence_request_roundtrip() {
        let req = Request::SetPresence {
            id: "p-1".into(),
            target: PresenceState::Drowsy,
        };
        let json = serde_json::to_string(&req).unwrap();
        assert_eq!(
            json,
            r#"{"type":"set_presence","id":"p-1","target":"drowsy"}"#
        );
        let parsed: Request = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, req);
    }

    #[test]
    fn get_presence_request_roundtrip() {
        let req = Request::GetPresence { id: "p-2".into() };
        let json = serde_json::to_string(&req).unwrap();
        assert_eq!(json, r#"{"type":"get_presence","id":"p-2"}"#);
        let parsed: Request = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, req);
    }

    #[test]
    fn presence_event_roundtrip() {
        let evt = Event::Presence {
            id: "p-1".into(),
            state: PresenceState::Sleeping,
        };
        let json = serde_json::to_string(&evt).unwrap();
        assert_eq!(json, r#"{"type":"presence","id":"p-1","state":"sleeping"}"#);
        let parsed: Event = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, evt);
    }

    #[test]
    fn presence_event_is_not_terminal() {
        let evt = Event::Presence {
            id: "p-1".into(),
            state: PresenceState::Active,
        };
        assert!(!evt.is_terminal());
        assert_eq!(evt.id(), "p-1");
    }

    #[test]
    fn cycle_request_roundtrip() {
        let req = Request::Cycle { id: "c-1".into() };
        let json = serde_json::to_string(&req).unwrap();
        assert_eq!(json, r#"{"type":"cycle","id":"c-1"}"#);
        let parsed: Request = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, req);
    }

    #[test]
    fn presence_state_next_cycles() {
        assert_eq!(PresenceState::Active.next(), PresenceState::Drowsy);
        assert_eq!(PresenceState::Drowsy.next(), PresenceState::Sleeping);
        assert_eq!(PresenceState::Sleeping.next(), PresenceState::Active);
    }

    #[test]
    fn ptt_start_request_roundtrip() {
        let req = Request::PttStart { id: "v-1".into() };
        let json = serde_json::to_string(&req).unwrap();
        assert_eq!(json, r#"{"type":"ptt_start","id":"v-1"}"#);
        let parsed: Request = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, req);
    }

    #[test]
    fn ptt_stop_request_roundtrip() {
        let req = Request::PttStop { id: "v-2".into() };
        let json = serde_json::to_string(&req).unwrap();
        assert_eq!(json, r#"{"type":"ptt_stop","id":"v-2"}"#);
        let parsed: Request = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, req);
    }

    #[test]
    fn voice_state_event_roundtrip() {
        let evt = Event::VoiceState {
            id: "v-1".into(),
            state: VoiceCaptureState::Recording,
        };
        let json = serde_json::to_string(&evt).unwrap();
        assert_eq!(
            json,
            r#"{"type":"voice_state","id":"v-1","state":"recording"}"#
        );
        let parsed: Event = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, evt);
    }

    #[test]
    fn voice_state_queued_roundtrip() {
        let evt = Event::VoiceState {
            id: "v-2".into(),
            state: VoiceCaptureState::Queued,
        };
        let json = serde_json::to_string(&evt).unwrap();
        assert_eq!(
            json,
            r#"{"type":"voice_state","id":"v-2","state":"queued"}"#
        );
        let parsed: Event = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, evt);
    }

    #[test]
    fn transcription_event_roundtrip() {
        let evt = Event::Transcription {
            id: "v-1".into(),
            text: "hello world".into(),
        };
        let json = serde_json::to_string(&evt).unwrap();
        assert_eq!(
            json,
            r#"{"type":"transcription","id":"v-1","text":"hello world"}"#
        );
        let parsed: Event = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, evt);
    }

    #[test]
    fn voice_state_and_transcription_are_not_terminal() {
        let rec = Event::VoiceState {
            id: "x".into(),
            state: VoiceCaptureState::Recording,
        };
        assert!(!rec.is_terminal());
        assert_eq!(rec.id(), "x");
        let txt = Event::Transcription {
            id: "y".into(),
            text: "z".into(),
        };
        assert!(!txt.is_terminal());
        assert_eq!(txt.id(), "y");
    }

    #[test]
    fn listen_start_request_roundtrip() {
        let req = Request::ListenStart { id: "l-1".into() };
        let json = serde_json::to_string(&req).unwrap();
        assert_eq!(json, r#"{"type":"listen_start","id":"l-1"}"#);
        let parsed: Request = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, req);
    }

    #[test]
    fn listen_stop_request_roundtrip() {
        let req = Request::ListenStop { id: "l-2".into() };
        let json = serde_json::to_string(&req).unwrap();
        assert_eq!(json, r#"{"type":"listen_stop","id":"l-2"}"#);
        let parsed: Request = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, req);
    }

    #[test]
    fn listen_toggle_request_roundtrip() {
        let req = Request::ListenToggle { id: "l-3".into() };
        let json = serde_json::to_string(&req).unwrap();
        assert_eq!(json, r#"{"type":"listen_toggle","id":"l-3"}"#);
        let parsed: Request = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, req);
    }

    #[test]
    fn get_listen_state_request_roundtrip() {
        let req = Request::GetListenState { id: "l-4".into() };
        let json = serde_json::to_string(&req).unwrap();
        assert_eq!(json, r#"{"type":"get_listen_state","id":"l-4"}"#);
        let parsed: Request = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, req);
    }

    #[test]
    fn listen_state_event_roundtrip() {
        let evt = Event::ListenState {
            id: "l-1".into(),
            active: true,
        };
        let json = serde_json::to_string(&evt).unwrap();
        assert_eq!(json, r#"{"type":"listen_state","id":"l-1","active":true}"#);
        let parsed: Event = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, evt);
    }

    #[test]
    fn listen_state_is_not_terminal() {
        let evt = Event::ListenState {
            id: "l-1".into(),
            active: false,
        };
        assert!(!evt.is_terminal());
        assert_eq!(evt.id(), "l-1");
    }

    #[test]
    fn voice_toggle_request_roundtrip() {
        let req = Request::VoiceToggle { id: "vt-1".into() };
        let json = serde_json::to_string(&req).unwrap();
        assert_eq!(json, r#"{"type":"voice_toggle","id":"vt-1"}"#);
        let parsed: Request = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, req);
    }

    #[test]
    fn voice_skip_request_roundtrip() {
        let req = Request::VoiceSkip { id: "vs-1".into() };
        let json = serde_json::to_string(&req).unwrap();
        assert_eq!(json, r#"{"type":"voice_skip","id":"vs-1"}"#);
        let parsed: Request = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, req);
    }

    #[test]
    fn get_voice_state_request_roundtrip() {
        let req = Request::GetVoiceState { id: "vg-1".into() };
        let json = serde_json::to_string(&req).unwrap();
        assert_eq!(json, r#"{"type":"get_voice_state","id":"vg-1"}"#);
        let parsed: Request = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, req);
    }

    #[test]
    fn voice_output_state_event_roundtrip() {
        let evt = Event::VoiceOutputState {
            id: "vt-1".into(),
            enabled: true,
        };
        let json = serde_json::to_string(&evt).unwrap();
        assert_eq!(
            json,
            r#"{"type":"voice_output_state","id":"vt-1","enabled":true}"#
        );
        let parsed: Event = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, evt);
    }

    #[test]
    fn voice_output_state_is_not_terminal() {
        let evt = Event::VoiceOutputState {
            id: "vt-1".into(),
            enabled: false,
        };
        assert!(!evt.is_terminal());
        assert_eq!(evt.id(), "vt-1");
    }

    #[test]
    fn socket_path_uses_xdg_runtime_dir() {
        // Safe to set env in this single-threaded test function.
        unsafe { std::env::set_var("XDG_RUNTIME_DIR", "/run/user/1234") };
        let path = socket_path();
        assert_eq!(path, PathBuf::from("/run/user/1234/assistd.sock"));
    }
}
