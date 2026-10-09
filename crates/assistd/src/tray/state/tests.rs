use super::*;

fn delta(id: &str) -> Event {
    Event::Delta {
        id: id.into(),
        text: String::new(),
    }
}

fn tool_call(id: &str) -> Event {
    Event::ToolCall {
        id: id.into(),
        name: "x".into(),
        args: serde_json::Value::Null,
    }
}

fn done(id: &str) -> Event {
    Event::Done { id: id.into() }
}

fn error(id: &str) -> Event {
    Event::Error {
        id: id.into(),
        message: "boom".into(),
    }
}

fn listen(active: bool) -> Event {
    Event::ListenState {
        id: "x".into(),
        active,
    }
}

fn ptt(state: VoiceCaptureState) -> Event {
    Event::VoiceState {
        id: "p".into(),
        state,
    }
}

#[test]
fn config_error_outranks_everything() {
    let mut t = TrayTracker::new(Some("unknown field `foo`".into()));
    assert_eq!(t.current(), TrayState::ConfigError);
    assert_eq!(t.config_error(), Some("unknown field `foo`"));

    assert!(!t.set_connected());
    assert!(t.connected());
    assert!(!t.ingest(&listen(true)));
    assert!(!t.ingest(&delta("a")));
    assert_eq!(t.current(), TrayState::ConfigError);

    assert!(!t.set_disconnected());
    assert_eq!(t.current(), TrayState::ConfigError);
}

#[test]
fn generating_outranks_listening_and_presence() {
    let mut t = TrayTracker::default();
    t.set_connected();
    t.ingest(&listen(true));
    t.ingest(&delta("a"));
    assert_eq!(t.current(), TrayState::Generating);
}

#[test]
fn concurrent_turns_keep_generating_until_all_resolve() {
    let mut t = TrayTracker::default();
    t.set_connected();
    t.ingest(&delta("a"));
    t.ingest(&tool_call("b"));
    assert_eq!(t.current(), TrayState::Generating);
    t.ingest(&done("a"));
    assert_eq!(t.current(), TrayState::Generating);
    t.ingest(&error("b"));
    assert_eq!(t.current(), TrayState::Active);
}

#[test]
fn disconnect_clears_in_flight_and_listening() {
    let mut t = TrayTracker::default();
    t.set_connected();
    t.ingest(&listen(true));
    t.ingest(&delta("a"));
    assert!(t.set_disconnected());
    assert_eq!(t.current(), TrayState::Disconnected);

    let changed = t.set_connected();
    assert!(changed);
    assert_eq!(t.current(), TrayState::Active);
}

#[test]
fn push_to_talk_shows_listening_until_transcription_ends() {
    let mut t = TrayTracker::default();
    t.set_connected();
    assert!(t.ingest(&ptt(VoiceCaptureState::Recording)));
    assert_eq!(t.current(), TrayState::Listening);
    assert!(!t.ingest(&ptt(VoiceCaptureState::Transcribing)));
    assert!(t.ingest(&ptt(VoiceCaptureState::Idle)));
    assert_eq!(t.current(), TrayState::Active);
}
