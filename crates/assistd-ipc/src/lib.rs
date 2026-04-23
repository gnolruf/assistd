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

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

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
/// three-state indicator (idle / recording / transcribing). `Transcribing`
/// is distinct from `Recording` because whisper inference takes 1–3 s on a
/// few seconds of audio and users otherwise keep talking into dead air.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum VoiceCaptureState {
    Idle,
    Recording,
    Transcribing,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Request {
    Query {
        id: String,
        text: String,
    },
    /// Drive the daemon to a specific presence state.
    SetPresence {
        id: String,
        target: PresenceState,
    },
    /// Report the daemon's current presence state.
    GetPresence {
        id: String,
    },
    /// Atomically advance the daemon one step along
    /// `Active → Drowsy → Sleeping → Active`.
    Cycle {
        id: String,
    },
    /// Begin a push-to-talk recording. Returns immediately with `Done`
    /// once cpal has opened the device; audio is buffered in the daemon
    /// until a matching `PttStop` arrives.
    PttStart {
        id: String,
    },
    /// End the push-to-talk recording and transcribe what was captured.
    /// Emits `VoiceState::Transcribing`, then `Transcription { text }`,
    /// then the text is dispatched internally as a `Query` whose
    /// streaming `Delta`s flow back on the same connection before `Done`.
    PttStop {
        id: String,
    },
    /// Enable hands-free continuous listening. The daemon keeps the mic
    /// open and auto-dispatches each VAD-segmented utterance as a
    /// `Query`. Emits `ListenState { active: true }` + `Done`.
    /// Rejects with `Error` when a PTT recording is already in flight.
    ListenStart {
        id: String,
    },
    /// Disable continuous listening. Emits `ListenState { active: false }`
    /// + `Done`. Idempotent when already stopped.
    ListenStop {
        id: String,
    },
    /// Flip continuous listening on/off in a single call. Emits the
    /// post-toggle `ListenState` + `Done`.
    ListenToggle {
        id: String,
    },
    /// Report whether continuous listening is currently active. Emits
    /// `ListenState` + `Done` with no state change.
    GetListenState {
        id: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
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
    /// mic pipeline moves between Idle / Recording / Transcribing.
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
        let req = Request::Query {
            id: "req-1".into(),
            text: "ping".into(),
        };
        let json = serde_json::to_string(&req).unwrap();
        assert_eq!(json, r#"{"type":"query","id":"req-1","text":"ping"}"#);
        let parsed: Request = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, req);
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
    fn socket_path_uses_xdg_runtime_dir() {
        // Safe to set env in this single-threaded test function.
        unsafe { std::env::set_var("XDG_RUNTIME_DIR", "/run/user/1234") };
        let path = socket_path();
        assert_eq!(path, PathBuf::from("/run/user/1234/assistd.sock"));
    }
}
