use super::*;
use serde_json::json;

fn id() -> String {
    "r".into()
}

/// One of every [`Request`] variant with its exact wire form.
fn request_cases() -> Vec<(Request, &'static str)> {
    vec![
        (
            Request::query("r", "hi"),
            r#"{"type":"query","id":"r","text":"hi"}"#,
        ),
        (
            Request::query_with_attachments(
                "r",
                "describe this",
                vec![ImageAttachment::from_bytes(
                    "image/png",
                    &[0xDE, 0xAD, 0xBE, 0xEF],
                )],
            ),
            r#"{"type":"query","id":"r","text":"describe this","attachments":[{"mime":"image/png","data_base64":"3q2+7w=="}]}"#,
        ),
        (
            Request::SetPresence {
                id: id(),
                target: PresenceState::Drowsy,
            },
            r#"{"type":"set_presence","id":"r","target":"drowsy"}"#,
        ),
        (
            Request::GetPresence { id: id() },
            r#"{"type":"get_presence","id":"r"}"#,
        ),
        (Request::Cycle { id: id() }, r#"{"type":"cycle","id":"r"}"#),
        (
            Request::PttStart { id: id() },
            r#"{"type":"ptt_start","id":"r"}"#,
        ),
        (
            Request::PttStop { id: id() },
            r#"{"type":"ptt_stop","id":"r"}"#,
        ),
        (
            Request::ListenStart { id: id() },
            r#"{"type":"listen_start","id":"r"}"#,
        ),
        (
            Request::ListenStop { id: id() },
            r#"{"type":"listen_stop","id":"r"}"#,
        ),
        (
            Request::ListenToggle { id: id() },
            r#"{"type":"listen_toggle","id":"r"}"#,
        ),
        (
            Request::GetListenState { id: id() },
            r#"{"type":"get_listen_state","id":"r"}"#,
        ),
        (
            Request::VoiceToggle { id: id() },
            r#"{"type":"voice_toggle","id":"r"}"#,
        ),
        (
            Request::VoiceSkip { id: id() },
            r#"{"type":"voice_skip","id":"r"}"#,
        ),
        (
            Request::InterruptTurn { id: id() },
            r#"{"type":"interrupt_turn","id":"r"}"#,
        ),
        (
            Request::GetVoiceState { id: id() },
            r#"{"type":"get_voice_state","id":"r"}"#,
        ),
        (
            Request::MemorySave {
                id: id(),
                key: "k".into(),
                value: "v".into(),
            },
            r#"{"type":"memory_save","id":"r","key":"k","value":"v"}"#,
        ),
        (
            Request::MemoryLoad {
                id: id(),
                key: "k".into(),
            },
            r#"{"type":"memory_load","id":"r","key":"k"}"#,
        ),
        (
            Request::MemoryList {
                id: id(),
                prefix: "pref:".into(),
            },
            r#"{"type":"memory_list","id":"r","prefix":"pref:"}"#,
        ),
        (
            Request::MemoryListAll {
                id: id(),
                prefix: "fact:".into(),
                limit: 3,
            },
            r#"{"type":"memory_list_all","id":"r","prefix":"fact:","limit":3}"#,
        ),
        (
            Request::MemoryDelete {
                id: id(),
                key: "k".into(),
            },
            r#"{"type":"memory_delete","id":"r","key":"k"}"#,
        ),
        (
            Request::MemoryForget {
                id: id(),
                memory_id: 42,
            },
            r#"{"type":"memory_forget","id":"r","memory_id":42}"#,
        ),
        (
            Request::MemorySemanticSearch {
                id: id(),
                query: "the rust thing".into(),
                limit: 5,
            },
            r#"{"type":"memory_semantic_search","id":"r","query":"the rust thing","limit":5}"#,
        ),
        (
            Request::MemoryReindex { id: id() },
            r#"{"type":"memory_reindex","id":"r"}"#,
        ),
        (
            Request::ConfirmResponse {
                id: id(),
                confirm_id: "c-abc".into(),
                allow: true,
                always: false,
            },
            r#"{"type":"confirm_response","id":"r","confirm_id":"c-abc","allow":true}"#,
        ),
        (
            Request::ConfirmResponse {
                id: id(),
                confirm_id: "c-abc".into(),
                allow: true,
                always: true,
            },
            r#"{"type":"confirm_response","id":"r","confirm_id":"c-abc","allow":true,"always":true}"#,
        ),
        (
            Request::GetCapabilities { id: id() },
            r#"{"type":"get_capabilities","id":"r"}"#,
        ),
        (
            Request::Fork {
                id: id(),
                name: "experiment".into(),
            },
            r#"{"type":"fork","id":"r","name":"experiment"}"#,
        ),
        (
            Request::Branches { id: id() },
            r#"{"type":"branches","id":"r"}"#,
        ),
        (
            Request::Switch {
                id: id(),
                target: "abc12345/main".into(),
            },
            r#"{"type":"switch","id":"r","target":"abc12345/main"}"#,
        ),
        (Request::Undo { id: id() }, r#"{"type":"undo","id":"r"}"#),
        (
            Request::ResumeOrNew {
                id: id(),
                recency_secs: 600,
            },
            r#"{"type":"resume_or_new","id":"r","recency_secs":600}"#,
        ),
        (
            Request::NewSession { id: id() },
            r#"{"type":"new_session","id":"r"}"#,
        ),
        (
            Request::Subscribe {
                id: id(),
                filter: SubscribeFilter::default(),
            },
            r#"{"type":"subscribe","id":"r","filter":{"kinds":[]}}"#,
        ),
        (
            Request::Subscribe {
                id: id(),
                filter: SubscribeFilter {
                    kinds: vec![
                        EventKind::Presence,
                        EventKind::ListenState,
                        EventKind::LastDelta,
                    ],
                },
            },
            r#"{"type":"subscribe","id":"r","filter":{"kinds":["presence","listen_state","last_delta"]}}"#,
        ),
    ]
}

#[test]
fn requests_match_pinned_wire_format() {
    for (req, wire) in request_cases() {
        assert_eq!(serde_json::to_string(&req).unwrap(), wire);
        let parsed: Request = serde_json::from_str(wire).unwrap();
        assert_eq!(parsed, req, "{wire}");
        let tag: serde_json::Value = serde_json::from_str(wire).unwrap();
        assert_eq!(tag["type"], req.kind(), "{wire}");
        assert_eq!(req.id(), "r", "{wire}");
    }
}

#[test]
fn request_optional_fields_default_when_absent() {
    let cases = [
        (
            r#"{"type":"memory_list","id":"r"}"#,
            Request::MemoryList {
                id: id(),
                prefix: String::new(),
            },
        ),
        (
            r#"{"type":"memory_list_all","id":"r"}"#,
            Request::MemoryListAll {
                id: id(),
                prefix: String::new(),
                limit: 0,
            },
        ),
        (
            r#"{"type":"memory_semantic_search","id":"r","query":"q"}"#,
            Request::MemorySemanticSearch {
                id: id(),
                query: "q".into(),
                limit: 0,
            },
        ),
        (
            r#"{"type":"subscribe","id":"r"}"#,
            Request::Subscribe {
                id: id(),
                filter: SubscribeFilter::default(),
            },
        ),
    ];
    for (wire, expected) in cases {
        let parsed: Request = serde_json::from_str(wire).unwrap();
        assert_eq!(parsed, expected, "{wire}");
    }
}

/// One of every [`Event`] variant with its exact wire form and its
/// broadcast kind.
fn event_cases() -> Vec<(Event, &'static str, Option<EventKind>)> {
    vec![
        (
            Event::Delta {
                id: id(),
                text: "pong".into(),
            },
            r#"{"type":"delta","id":"r","text":"pong"}"#,
            Some(EventKind::Delta),
        ),
        (
            Event::ReasoningDelta {
                id: id(),
                text: "hmm".into(),
            },
            r#"{"type":"reasoning_delta","id":"r","text":"hmm"}"#,
            Some(EventKind::ReasoningDelta),
        ),
        (
            Event::ToolCall {
                id: id(),
                name: "echo".into(),
                args: json!({"text": "hi"}),
            },
            r#"{"type":"tool_call","id":"r","name":"echo","args":{"text":"hi"}}"#,
            Some(EventKind::ToolCall),
        ),
        (
            Event::ToolResult {
                id: id(),
                name: "echo".into(),
                result: json!("hi"),
            },
            r#"{"type":"tool_result","id":"r","name":"echo","result":"hi"}"#,
            Some(EventKind::ToolResult),
        ),
        (
            Event::Presence {
                id: id(),
                state: PresenceState::Sleeping,
            },
            r#"{"type":"presence","id":"r","state":"sleeping"}"#,
            Some(EventKind::Presence),
        ),
        (
            Event::VoiceState {
                id: id(),
                state: VoiceCaptureState::Recording,
            },
            r#"{"type":"voice_state","id":"r","state":"recording"}"#,
            Some(EventKind::VoiceState),
        ),
        (
            Event::VoiceState {
                id: id(),
                state: VoiceCaptureState::Queued,
            },
            r#"{"type":"voice_state","id":"r","state":"queued"}"#,
            Some(EventKind::VoiceState),
        ),
        (
            Event::Transcription {
                id: id(),
                text: "hello world".into(),
            },
            r#"{"type":"transcription","id":"r","text":"hello world"}"#,
            None,
        ),
        (
            Event::ListenState {
                id: id(),
                active: true,
            },
            r#"{"type":"listen_state","id":"r","active":true}"#,
            Some(EventKind::ListenState),
        ),
        (
            Event::VoiceOutputState {
                id: id(),
                enabled: true,
            },
            r#"{"type":"voice_output_state","id":"r","enabled":true}"#,
            None,
        ),
        (
            Event::SpeakingState {
                id: id(),
                speaking: true,
            },
            r#"{"type":"speaking_state","id":"r","speaking":true}"#,
            Some(EventKind::SpeakingState),
        ),
        (
            Event::SessionTitle {
                id: id(),
                session_id: "s".into(),
                title: "cats".into(),
            },
            r#"{"type":"session_title","id":"r","session_id":"s","title":"cats"}"#,
            Some(EventKind::SessionTitle),
        ),
        (
            Event::SemanticHit {
                id: id(),
                conversation_id: 42,
                chunk_id: 7,
                session_id: "s".into(),
                timestamp: "2026-04-28T00:00:00Z".into(),
                role: Role::User,
                content: "the rust embeddings daemon".into(),
                similarity: 0.5,
            },
            r#"{"type":"semantic_hit","id":"r","conversation_id":42,"chunk_id":7,"session_id":"s","timestamp":"2026-04-28T00:00:00Z","role":"user","content":"the rust embeddings daemon","similarity":0.5}"#,
            None,
        ),
        (
            Event::MemoryValue {
                id: id(),
                key: "absent".into(),
                value: None,
            },
            r#"{"type":"memory_value","id":"r","key":"absent","value":null}"#,
            None,
        ),
        (
            Event::MemoryKeys {
                id: id(),
                keys: vec!["a".into(), "b".into()],
            },
            r#"{"type":"memory_keys","id":"r","keys":["a","b"]}"#,
            None,
        ),
        (
            Event::MemoryRow {
                id: id(),
                memory_id: 17,
                key: "fact:user.name".into(),
                value: "Ben".into(),
            },
            r#"{"type":"memory_row","id":"r","memory_id":17,"key":"fact:user.name","value":"Ben"}"#,
            None,
        ),
        (
            Event::MemoryForgetResult {
                id: id(),
                deleted: true,
                key: Some("fact:user.name".into()),
            },
            r#"{"type":"memory_forget_result","id":"r","deleted":true,"key":"fact:user.name"}"#,
            None,
        ),
        (
            Event::ReindexProgress {
                id: id(),
                kind: ReindexKind::Chunks,
                done: 3,
                total: 10,
            },
            r#"{"type":"reindex_progress","id":"r","kind":"chunks","done":3,"total":10}"#,
            None,
        ),
        (
            Event::ConfirmRequest {
                id: id(),
                confirm_id: "c-abc".into(),
                tool: "bash".into(),
                script: "rm -rf /tmp/foo".into(),
                matched_pattern: "rm -rf".into(),
                always_allow: Vec::new(),
            },
            r#"{"type":"confirm_request","id":"r","confirm_id":"c-abc","tool":"bash","script":"rm -rf /tmp/foo","matched_pattern":"rm -rf"}"#,
            None,
        ),
        (
            Event::ConfirmRequest {
                id: id(),
                confirm_id: "c-abc".into(),
                tool: "bash".into(),
                script: "cargo build".into(),
                matched_pattern: "not on the allowlist: cargo".into(),
                always_allow: vec!["cargo".into()],
            },
            r#"{"type":"confirm_request","id":"r","confirm_id":"c-abc","tool":"bash","script":"cargo build","matched_pattern":"not on the allowlist: cargo","always_allow":["cargo"]}"#,
            None,
        ),
        (
            Event::Capabilities {
                id: id(),
                vision: true,
                model_name: "Qwen3-14B-GGUF:Q4_K_M".into(),
            },
            r#"{"type":"capabilities","id":"r","vision":true,"model_name":"Qwen3-14B-GGUF:Q4_K_M"}"#,
            None,
        ),
        (
            Event::Status {
                id: id(),
                severity: StatusSeverity::Warning,
                component: Component::IdleMonitor,
                event: StatusKind::ToolsWithdrawn,
                message: "m".into(),
            },
            r#"{"type":"status","id":"r","severity":"warning","component":"idle_monitor","event":"tools_withdrawn","message":"m"}"#,
            None,
        ),
        (
            Event::BranchInfo {
                id: id(),
                branch_id: 7,
                session_id: "s".into(),
                session_started_at: "2026-01-01T00:00:00Z".into(),
                session_ended_at: None,
                session_title: Some("cats".into()),
                name: "main".into(),
                parent_branch_name: None,
                fork_point_seq: None,
                created_at: "2026-01-01T00:00:00Z".into(),
                message_count: 4,
                is_current_in_session: true,
                is_active_session: false,
            },
            r#"{"type":"branch_info","id":"r","branch_id":7,"session_id":"s","session_started_at":"2026-01-01T00:00:00Z","session_ended_at":null,"session_title":"cats","name":"main","parent_branch_name":null,"fork_point_seq":null,"created_at":"2026-01-01T00:00:00Z","message_count":4,"is_current_in_session":true,"is_active_session":false}"#,
            None,
        ),
        (
            Event::BranchSwitched {
                id: id(),
                branch_id: 9,
                session_id: "s".into(),
                session_title: None,
                name: "experiment".into(),
                parent_branch_name: Some("main".into()),
                fork_point_seq: Some(5),
            },
            r#"{"type":"branch_switched","id":"r","branch_id":9,"session_id":"s","session_title":null,"name":"experiment","parent_branch_name":"main","fork_point_seq":5}"#,
            None,
        ),
        (
            Event::HistoryEntry {
                id: id(),
                seq: 3,
                role: Role::Assistant,
                content: "hello".into(),
                tool_name: None,
            },
            r#"{"type":"history_entry","id":"r","seq":3,"role":"assistant","content":"hello","tool_name":null}"#,
            None,
        ),
        (
            Event::UndoApplied {
                id: id(),
                removed_messages: 2,
                last_user_text: Some("hi".into()),
            },
            r#"{"type":"undo_applied","id":"r","removed_messages":2,"last_user_text":"hi"}"#,
            None,
        ),
        (
            Event::Error {
                id: id(),
                message: "boom".into(),
            },
            r#"{"type":"error","id":"r","message":"boom"}"#,
            Some(EventKind::Error),
        ),
        (
            Event::Done { id: id() },
            r#"{"type":"done","id":"r"}"#,
            Some(EventKind::Done),
        ),
        (
            Event::LastDelta {
                id: id(),
                text: "Hello world".into(),
            },
            r#"{"type":"last_delta","id":"r","text":"Hello world"}"#,
            Some(EventKind::LastDelta),
        ),
    ]
}

#[test]
fn events_match_pinned_wire_format() {
    for (ev, wire, _) in event_cases() {
        assert_eq!(serde_json::to_string(&ev).unwrap(), wire);
        let parsed: Event = serde_json::from_str(wire).unwrap();
        assert_eq!(parsed, ev, "{wire}");
        assert_eq!(ev.id(), "r", "{wire}");
    }
}

#[test]
fn only_done_and_error_are_terminal() {
    for (ev, wire, _) in event_cases() {
        let expected = matches!(ev, Event::Done { .. } | Event::Error { .. });
        assert_eq!(ev.is_terminal(), expected, "{wire}");
    }
}

#[test]
fn event_kind_matches_broadcast_eligibility() {
    for (ev, wire, kind) in event_cases() {
        assert_eq!(ev.kind(), kind, "{wire}");
    }
}

#[test]
fn wire_spelling_helpers_agree_with_serde() {
    fn check<T: Serialize + fmt::Display>(value: T, as_str: &str) {
        assert_eq!(serde_json::to_value(&value).unwrap(), as_str);
        assert_eq!(value.to_string(), as_str);
    }
    for v in [
        StatusSeverity::Info,
        StatusSeverity::Warning,
        StatusSeverity::Error,
    ] {
        check(v, v.as_str());
    }
    for v in [
        Component::Agent,
        Component::Llm,
        Component::Mcp,
        Component::Voice,
        Component::Memory,
        Component::Wm,
        Component::Embed,
        Component::Hotkey,
        Component::Daemon,
        Component::IdleMonitor,
        Component::GpuMonitor,
        Component::ListenDispatcher,
    ] {
        check(v, v.as_str());
    }
    for v in [Role::System, Role::User, Role::Assistant, Role::Tool] {
        check(v, v.as_str());
    }
    for v in [ReindexKind::Chunks, ReindexKind::Memories] {
        check(v, v.as_str());
    }
}

#[test]
fn image_attachment_round_trips_through_base64() {
    let payload = b"\x89PNG\r\n\x1a\n";
    let att = ImageAttachment::from_bytes("image/png", payload);
    assert_eq!(att.decode_bytes().unwrap(), payload);
}

#[test]
fn presence_state_next_cycles() {
    assert_eq!(PresenceState::Active.next(), PresenceState::Drowsy);
    assert_eq!(PresenceState::Drowsy.next(), PresenceState::Sleeping);
    assert_eq!(PresenceState::Sleeping.next(), PresenceState::Active);
}

#[test]
fn socket_path_prefers_xdg_runtime_dir_then_user() {
    let cases = [
        (
            Some("/run/user/1234"),
            Some("alice"),
            "/run/user/1234/assistd.sock",
        ),
        (None, Some("alice"), "/tmp/assistd-alice.sock"),
        (None, None, "/tmp/assistd-nobody.sock"),
    ];
    for (xdg, user, expected) in cases {
        let path = socket_path_for(xdg.map(OsString::from), user.map(OsString::from));
        assert_eq!(path, PathBuf::from(expected), "xdg={xdg:?} user={user:?}");
    }
}

#[test]
fn subscribe_filter_default_matches_all() {
    let filter = SubscribeFilter::default();
    for kind in [
        EventKind::Delta,
        EventKind::ReasoningDelta,
        EventKind::ToolCall,
        EventKind::ToolResult,
        EventKind::Presence,
        EventKind::ListenState,
        EventKind::VoiceState,
        EventKind::SpeakingState,
        EventKind::SessionTitle,
        EventKind::Done,
        EventKind::Error,
        EventKind::LastDelta,
    ] {
        assert!(filter.matches(kind), "default filter should match {kind:?}");
    }
}

#[test]
fn subscribe_filter_matches_listed_only() {
    let filter = SubscribeFilter {
        kinds: vec![EventKind::Presence, EventKind::LastDelta],
    };
    assert!(filter.matches(EventKind::Presence));
    assert!(filter.matches(EventKind::LastDelta));
    assert!(!filter.matches(EventKind::Delta));
    assert!(!filter.matches(EventKind::ToolCall));
    assert!(!filter.matches(EventKind::Done));
}
