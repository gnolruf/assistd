use super::*;

#[test]
fn request_roundtrip() {
    let req = Request::Query {
        id: "req-1".into(),
        text: "ping".into(),
        attachments: Vec::new(),
    };
    let json = serde_json::to_string(&req).unwrap();
    assert_eq!(json, r#"{"type":"query","id":"req-1","text":"ping"}"#);
    let parsed: Request = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed, req);
}

#[test]
fn request_query_with_attachments_roundtrip() {
    let req = Request::query_with_attachments(
        "req-2",
        "describe this",
        vec![ImageAttachment::from_bytes(
            "image/png",
            &[0xDE, 0xAD, 0xBE, 0xEF],
        )],
    );
    let json = serde_json::to_string(&req).unwrap();
    assert!(json.contains(r#""data_base64":"3q2+7w==""#));
    let parsed: Request = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed, req);
}

#[test]
fn text_only_query_omits_attachments_on_the_wire() {
    let json = serde_json::to_string(&Request::query("req-3", "hi")).unwrap();
    assert_eq!(json, r#"{"type":"query","id":"req-3","text":"hi"}"#);
}

#[test]
fn fork_request_round_trips() {
    let req = Request::Fork {
        id: "r-1".into(),
        name: "experiment".into(),
    };
    let json = serde_json::to_string(&req).unwrap();
    let parsed: Request = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed, req);
    assert_eq!(req.kind(), "fork");
    assert_eq!(req.id(), "r-1");
}

#[test]
fn branches_request_round_trips() {
    let req = Request::Branches { id: "r-2".into() };
    let json = serde_json::to_string(&req).unwrap();
    let parsed: Request = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed, req);
    assert_eq!(req.kind(), "branches");
}

#[test]
fn switch_request_round_trips() {
    let req = Request::Switch {
        id: "r-3".into(),
        target: "abc12345/main".into(),
    };
    let json = serde_json::to_string(&req).unwrap();
    let parsed: Request = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed, req);
    assert_eq!(req.kind(), "switch");
}

#[test]
fn undo_request_round_trips() {
    let req = Request::Undo { id: "r-4".into() };
    let json = serde_json::to_string(&req).unwrap();
    let parsed: Request = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed, req);
    assert_eq!(req.kind(), "undo");
}

#[test]
fn branch_info_event_round_trips() {
    let ev = Event::BranchInfo {
        id: "r-1".into(),
        branch_id: 7,
        session_id: "abc12345-...".into(),
        session_started_at: "2026-01-01T00:00:00Z".into(),
        session_ended_at: None,
        session_title: Some("a chat about cats".into()),
        name: "main".into(),
        parent_branch_name: None,
        fork_point_seq: None,
        created_at: "2026-01-01T00:00:00Z".into(),
        message_count: 4,
        is_current_in_session: true,
        is_active_session: true,
    };
    let json = serde_json::to_string(&ev).unwrap();
    let parsed: Event = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed, ev);
    assert_eq!(ev.id(), "r-1");
}

#[test]
fn branch_switched_event_round_trips() {
    let ev = Event::BranchSwitched {
        id: "r-2".into(),
        branch_id: 9,
        session_id: "sess".into(),
        session_title: Some("a chat about cats".into()),
        name: "experiment".into(),
        parent_branch_name: Some("main".into()),
        fork_point_seq: Some(5),
    };
    let json = serde_json::to_string(&ev).unwrap();
    let parsed: Event = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed, ev);
}

#[test]
fn history_entry_event_round_trips() {
    let ev = Event::HistoryEntry {
        id: "r-3".into(),
        seq: 3,
        role: Role::Assistant,
        content: "hello".into(),
        tool_name: None,
    };
    let json = serde_json::to_string(&ev).unwrap();
    let parsed: Event = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed, ev);
}

#[test]
fn undo_applied_event_round_trips() {
    let ev = Event::UndoApplied {
        id: "r-4".into(),
        removed_messages: 2,
        last_user_text: Some("hi".into()),
    };
    let json = serde_json::to_string(&ev).unwrap();
    let parsed: Event = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed, ev);
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
fn reasoning_delta_event_roundtrip() {
    let evt = Event::ReasoningDelta {
        id: "req-1".into(),
        text: "let me think...".into(),
    };
    let json = serde_json::to_string(&evt).unwrap();
    assert_eq!(
        json,
        r#"{"type":"reasoning_delta","id":"req-1","text":"let me think..."}"#
    );
    let parsed: Event = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed, evt);
    assert!(!evt.is_terminal());
    assert_eq!(evt.id(), "req-1");
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
        let parsed: Request = serde_json::from_str(&serde_json::to_string(&r).unwrap()).unwrap();
        assert_eq!(parsed, r);
    }
}

#[test]
fn memory_list_all_request_omits_optional_fields() {
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
        kind: ReindexKind::Chunks,
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
            role: Role::User,
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
fn interrupt_turn_request_roundtrip() {
    let req = Request::InterruptTurn { id: "it-1".into() };
    let json = serde_json::to_string(&req).unwrap();
    assert_eq!(json, r#"{"type":"interrupt_turn","id":"it-1"}"#);
    let parsed: Request = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed, req);
    assert_eq!(req.kind(), "interrupt_turn");
    assert_eq!(req.id(), "it-1");
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
fn speaking_state_event_roundtrip() {
    let evt = Event::SpeakingState {
        id: "q-1".into(),
        speaking: true,
    };
    let json = serde_json::to_string(&evt).unwrap();
    assert_eq!(
        json,
        r#"{"type":"speaking_state","id":"q-1","speaking":true}"#
    );
    let parsed: Event = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed, evt);
}

#[test]
fn speaking_state_is_bus_eligible_and_not_terminal() {
    let evt = Event::SpeakingState {
        id: "q-1".into(),
        speaking: false,
    };
    assert!(!evt.is_terminal());
    assert_eq!(evt.id(), "q-1");
    assert_eq!(evt.kind(), Some(EventKind::SpeakingState));
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
fn confirm_request_event_roundtrip() {
    let evt = Event::ConfirmRequest {
        id: "req-1".into(),
        confirm_id: "c-abc".into(),
        tool: "bash".into(),
        script: "rm -rf /tmp/foo".into(),
        matched_pattern: "rm -rf".into(),
    };
    let parsed: Event = serde_json::from_str(&serde_json::to_string(&evt).unwrap()).unwrap();
    assert_eq!(parsed, evt);
    assert_eq!(evt.id(), "req-1");
    assert!(!evt.is_terminal());
}

#[test]
fn confirm_response_request_roundtrip() {
    let req = Request::ConfirmResponse {
        id: "req-1".into(),
        confirm_id: "c-abc".into(),
        allow: true,
    };
    let parsed: Request = serde_json::from_str(&serde_json::to_string(&req).unwrap()).unwrap();
    assert_eq!(parsed, req);
    assert_eq!(req.id(), "req-1");
    assert_eq!(req.kind(), "confirm_response");
}

#[test]
fn get_capabilities_request_roundtrip() {
    let req = Request::GetCapabilities { id: "cap-1".into() };
    let json = serde_json::to_string(&req).unwrap();
    assert_eq!(json, r#"{"type":"get_capabilities","id":"cap-1"}"#);
    let parsed: Request = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed, req);
    assert_eq!(req.kind(), "get_capabilities");
}

#[test]
fn capabilities_event_roundtrip() {
    let evt = Event::Capabilities {
        id: "cap-1".into(),
        vision: true,
        model_name: "Qwen3-14B-GGUF:Q4_K_M".into(),
    };
    let parsed: Event = serde_json::from_str(&serde_json::to_string(&evt).unwrap()).unwrap();
    assert_eq!(parsed, evt);
    assert!(!evt.is_terminal());
}

#[test]
fn socket_path_uses_xdg_runtime_dir() {
    let path = socket_path_for(Some(OsString::from("/run/user/1234")), None);
    assert_eq!(path, PathBuf::from("/run/user/1234/assistd.sock"));
}

#[test]
fn socket_path_falls_back_to_tmp_with_user() {
    let path = socket_path_for(None, Some(OsString::from("alice")));
    assert_eq!(path, PathBuf::from("/tmp/assistd-alice.sock"));
}

#[test]
fn socket_path_falls_back_to_nobody_without_user() {
    let path = socket_path_for(None, None);
    assert_eq!(path, PathBuf::from("/tmp/assistd-nobody.sock"));
}

#[test]
fn subscribe_request_roundtrip_empty_filter() {
    let req = Request::Subscribe {
        id: "s-1".into(),
        filter: SubscribeFilter::default(),
    };
    let json = serde_json::to_string(&req).unwrap();
    assert_eq!(
        json,
        r#"{"type":"subscribe","id":"s-1","filter":{"kinds":[]}}"#
    );
    let parsed: Request = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed, req);
}

#[test]
fn subscribe_request_accepts_missing_filter_field() {
    let parsed: Request = serde_json::from_str(r#"{"type":"subscribe","id":"s-1"}"#).unwrap();
    assert_eq!(
        parsed,
        Request::Subscribe {
            id: "s-1".into(),
            filter: SubscribeFilter::default(),
        }
    );
}

#[test]
fn subscribe_request_roundtrip_populated_filter() {
    let req = Request::Subscribe {
        id: "s-2".into(),
        filter: SubscribeFilter {
            kinds: vec![
                EventKind::Presence,
                EventKind::ListenState,
                EventKind::LastDelta,
            ],
        },
    };
    let json = serde_json::to_string(&req).unwrap();
    assert_eq!(
        json,
        r#"{"type":"subscribe","id":"s-2","filter":{"kinds":["presence","listen_state","last_delta"]}}"#
    );
    let parsed: Request = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed, req);
}

#[test]
fn subscribe_request_id_and_kind() {
    let req = Request::Subscribe {
        id: "s-3".into(),
        filter: SubscribeFilter::default(),
    };
    assert_eq!(req.id(), "s-3");
    assert_eq!(req.kind(), "subscribe");
}

#[test]
fn subscribe_filter_default_matches_all() {
    let f = SubscribeFilter::default();
    for k in [
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
        assert!(f.matches(k), "default filter should match {k:?}");
    }
}

#[test]
fn subscribe_filter_matches_listed_only() {
    let f = SubscribeFilter {
        kinds: vec![EventKind::Presence, EventKind::LastDelta],
    };
    assert!(f.matches(EventKind::Presence));
    assert!(f.matches(EventKind::LastDelta));
    assert!(!f.matches(EventKind::Delta));
    assert!(!f.matches(EventKind::ToolCall));
    assert!(!f.matches(EventKind::Done));
}

#[test]
fn last_delta_event_roundtrip() {
    let ev = Event::LastDelta {
        id: "q-7".into(),
        text: "Hello world".into(),
    };
    let json = serde_json::to_string(&ev).unwrap();
    assert_eq!(
        json,
        r#"{"type":"last_delta","id":"q-7","text":"Hello world"}"#
    );
    let parsed: Event = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed, ev);
}

#[test]
fn last_delta_is_not_terminal() {
    let ev = Event::LastDelta {
        id: "q-1".into(),
        text: "snapshot".into(),
    };
    assert!(!ev.is_terminal());
    assert_eq!(ev.id(), "q-1");
}

#[test]
fn event_kind_classifies_broadcast_eligible_variants() {
    let cases: Vec<(Event, EventKind)> = vec![
        (
            Event::Delta {
                id: "q".into(),
                text: "t".into(),
            },
            EventKind::Delta,
        ),
        (
            Event::ReasoningDelta {
                id: "q".into(),
                text: "t".into(),
            },
            EventKind::ReasoningDelta,
        ),
        (
            Event::ToolCall {
                id: "q".into(),
                name: "bash".into(),
                args: serde_json::json!({}),
            },
            EventKind::ToolCall,
        ),
        (
            Event::ToolResult {
                id: "q".into(),
                name: "bash".into(),
                result: serde_json::json!({}),
            },
            EventKind::ToolResult,
        ),
        (
            Event::Presence {
                id: "q".into(),
                state: PresenceState::Active,
            },
            EventKind::Presence,
        ),
        (
            Event::VoiceState {
                id: "q".into(),
                state: VoiceCaptureState::Idle,
            },
            EventKind::VoiceState,
        ),
        (
            Event::ListenState {
                id: "q".into(),
                active: false,
            },
            EventKind::ListenState,
        ),
        (
            Event::SessionTitle {
                id: "q".into(),
                session_id: "s".into(),
                title: "cats and dogs".into(),
            },
            EventKind::SessionTitle,
        ),
        (Event::Done { id: "q".into() }, EventKind::Done),
        (
            Event::Error {
                id: "q".into(),
                message: "m".into(),
            },
            EventKind::Error,
        ),
        (
            Event::LastDelta {
                id: "q".into(),
                text: "t".into(),
            },
            EventKind::LastDelta,
        ),
    ];
    for (ev, expected) in cases {
        assert_eq!(ev.kind(), Some(expected), "wrong kind for {ev:?}");
    }
}

#[test]
fn event_kind_returns_none_for_dialog_local_variants() {
    let dialog_local = vec![
        Event::Transcription {
            id: "q".into(),
            text: "t".into(),
        },
        Event::VoiceOutputState {
            id: "q".into(),
            enabled: true,
        },
        Event::MemoryValue {
            id: "q".into(),
            key: "k".into(),
            value: None,
        },
        Event::MemoryKeys {
            id: "q".into(),
            keys: Vec::new(),
        },
        Event::MemoryRow {
            id: "q".into(),
            memory_id: 0,
            key: "k".into(),
            value: "v".into(),
        },
        Event::MemoryForgetResult {
            id: "q".into(),
            deleted: false,
            key: None,
        },
        Event::ReindexProgress {
            id: "q".into(),
            kind: ReindexKind::Chunks,
            done: 0,
            total: 0,
        },
        Event::ConfirmRequest {
            id: "q".into(),
            confirm_id: "c".into(),
            tool: "bash".into(),
            script: "ls".into(),
            matched_pattern: "ls".into(),
        },
        Event::Capabilities {
            id: "q".into(),
            vision: false,
            model_name: "m".into(),
        },
        Event::Status {
            id: "q".into(),
            severity: StatusSeverity::Info,
            component: Component::Llm,
            event: StatusKind::Restarting,
            message: "".into(),
        },
    ];
    for ev in dialog_local {
        assert_eq!(ev.kind(), None, "expected None for {ev:?}");
    }
}
