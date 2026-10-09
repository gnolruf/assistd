use assistd_ipc::{ComponentReadiness, StartupComponent};

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

fn presence(state: PresenceState) -> Event {
    Event::Presence {
        id: "x".into(),
        state,
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
fn disconnected_outranks_daemon_activity() {
    let mut t = TrayTracker::default();
    assert_eq!(t.current(), TrayState::Disconnected);
    assert!(!t.ingest(&listen(true)));
    assert!(!t.ingest(&delta("a")));
    assert_eq!(t.current(), TrayState::Disconnected);
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
fn listening_outranks_presence() {
    let mut t = TrayTracker::default();
    t.set_connected();
    t.ingest(&presence(PresenceState::Active));
    t.ingest(&listen(true));
    assert_eq!(t.current(), TrayState::Listening);
}

#[test]
fn drowsy_and_sleeping_both_render_as_sleeping() {
    let mut t = TrayTracker::default();
    t.set_connected();
    t.ingest(&presence(PresenceState::Drowsy));
    assert_eq!(t.current(), TrayState::Sleeping);
    t.ingest(&presence(PresenceState::Sleeping));
    assert_eq!(t.current(), TrayState::Sleeping);
    t.ingest(&presence(PresenceState::Active));
    assert_eq!(t.current(), TrayState::Active);
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
fn done_for_unknown_id_is_harmless() {
    let mut t = TrayTracker::default();
    t.set_connected();
    let changed = t.ingest(&done("never-seen"));
    assert!(!changed);
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

#[test]
fn disconnect_ends_a_push_to_talk_capture() {
    let mut t = TrayTracker::default();
    t.set_connected();
    t.ingest(&ptt(VoiceCaptureState::Recording));
    t.set_disconnected();
    t.set_connected();
    assert_eq!(t.current(), TrayState::Active);
}

#[test]
fn ingest_returns_change_flag() {
    let mut t = TrayTracker::default();
    t.set_connected();
    assert!(t.ingest(&delta("a")));
    assert!(!t.ingest(&delta("a")));
    assert!(t.ingest(&done("a")));
}

#[test]
fn unrelated_events_do_not_change_state() {
    let mut t = TrayTracker::default();
    t.set_connected();
    let changed = t.ingest(&Event::ToolResult {
        id: "a".into(),
        name: "x".into(),
        result: serde_json::Value::Null,
    });
    assert!(!changed);
    let changed = t.ingest(&Event::LastDelta {
        id: "a".into(),
        text: "x".into(),
    });
    assert!(!changed);
}

#[test]
fn every_state_maps_to_a_distinct_default_icon() {
    let icons = TrayIconsConfig::default();
    let names: Vec<_> = [
        TrayState::ConfigError,
        TrayState::Disconnected,
        TrayState::Generating,
        TrayState::Listening,
        TrayState::Active,
        TrayState::Sleeping,
    ]
    .into_iter()
    .map(|state| icon_name_for(state, &icons))
    .collect();
    let unique: HashSet<_> = names.iter().collect();
    assert_eq!(unique.len(), names.len(), "{names:?}");
}

#[test]
fn configured_icons_replace_the_defaults_except_for_config_errors() {
    let icons = TrayIconsConfig {
        generating: "my-busy-icon".into(),
        ..TrayIconsConfig::default()
    };
    assert_eq!(icon_name_for(TrayState::Generating, &icons), "my-busy-icon");
    assert_eq!(
        icon_name_for(TrayState::ConfigError, &icons),
        "dialog-error"
    );
}

#[test]
fn startup_summary_lists_each_subsystem_until_disconnect() {
    let mut tracker = TrayTracker::default();
    tracker.set_connected();
    assert_eq!(tracker.startup_summary(), None, "not reported yet");

    for (component, state) in [
        (StartupComponent::VoiceInput, ComponentReadiness::Ready),
        (StartupComponent::Embedding, ComponentReadiness::Starting),
        (
            StartupComponent::Mcp {
                server: "fs".into(),
            },
            ComponentReadiness::Unavailable {
                reason: "failed to start".into(),
            },
        ),
    ] {
        tracker.ingest(&Event::Readiness {
            id: "startup".into(),
            component,
            state,
        });
    }
    assert_eq!(
        tracker.startup_summary().as_deref(),
        Some(
            "voice input: ready\nsemantic memory: starting\nMCP fs: unavailable (failed to start)"
        )
    );

    tracker.set_disconnected();
    assert_eq!(tracker.startup_summary(), None);
}
