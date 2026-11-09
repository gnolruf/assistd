use super::*;
use chrono::NaiveDate;
use serde_json::json;

use crate::task::{MAX_TASK_NAME_CHARS, MAX_TASK_PROMPT_BYTES};

fn id() -> String {
    "r".into()
}

/// One of every [`Request`] variant with its exact wire form.
fn request_cases() -> Vec<(Request, &'static str)> {
    [
        voice_and_presence_request_cases(),
        memory_request_cases(),
        session_request_cases(),
        wake_word_request_cases(),
        task_request_cases(),
        task_confirm_request_cases(),
        task_refine_request_cases(),
        mcp_request_cases(),
    ]
    .concat()
}

fn wake_word_request_cases() -> Vec<(Request, &'static str)> {
    vec![
        (
            Request::WakeWord {
                id: id(),
                op: WakeWordOp::TestStart,
            },
            r#"{"type":"wake_word","id":"r","op":{"action":"test_start"}}"#,
        ),
        (
            Request::WakeWord {
                id: id(),
                op: WakeWordOp::TestStop,
            },
            r#"{"type":"wake_word","id":"r","op":{"action":"test_stop"}}"#,
        ),
    ]
}

fn voice_and_presence_request_cases() -> Vec<(Request, &'static str)> {
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
                    ImageMime::Png,
                    &[0xDE, 0xAD, 0xBE, 0xEF],
                )],
            ),
            r#"{"type":"query","id":"r","text":"describe this","attachments":[{"mime":"image/png","data_base64":"3q2+7w=="}]}"#,
        ),
        (
            Request::Query {
                id: "r".into(),
                text: "tidy the repo".into(),
                attachments: Vec::new(),
                plan: PlanMode::Planning,
            },
            r#"{"type":"query","id":"r","text":"tidy the repo","plan":"planning"}"#,
        ),
        (
            Request::PlanVerdict {
                id: id(),
                review_id: "v".into(),
                verdict: PlanVerdict::Accept {
                    mode: PermissionMode::AcceptEdits,
                },
            },
            r#"{"type":"plan_verdict","id":"r","review_id":"v","verdict":{"verdict":"accept","mode":"accept_edits"}}"#,
        ),
        (
            Request::PlanVerdict {
                id: id(),
                review_id: "v".into(),
                verdict: PlanVerdict::KeepPlanning {
                    feedback: Some("skip the tests".into()),
                },
            },
            r#"{"type":"plan_verdict","id":"r","review_id":"v","verdict":{"verdict":"keep_planning","feedback":"skip the tests"}}"#,
        ),
        (
            Request::SetPresence {
                id: id(),
                target: PresenceTarget::Drowsy,
            },
            r#"{"type":"set_presence","id":"r","target":"drowsy"}"#,
        ),
        (
            Request::ChatState {
                id: id(),
                focused: false,
            },
            r#"{"type":"chat_state","id":"r","focused":false}"#,
        ),
        (
            Request::PttStart { id: id() },
            r#"{"type":"ptt_start","id":"r"}"#,
        ),
        (
            Request::ListenStart { id: id() },
            r#"{"type":"listen_start","id":"r"}"#,
        ),
        (
            Request::VoiceToggle { id: id() },
            r#"{"type":"voice_toggle","id":"r"}"#,
        ),
    ]
}

fn memory_request_cases() -> Vec<(Request, &'static str)> {
    vec![
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
    ]
}

fn session_request_cases() -> Vec<(Request, &'static str)> {
    vec![
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
            Request::Fork {
                id: id(),
                name: "experiment".into(),
            },
            r#"{"type":"fork","id":"r","name":"experiment"}"#,
        ),
        (
            Request::Switch {
                id: id(),
                target: "abc12345/main".into(),
            },
            r#"{"type":"switch","id":"r","target":"abc12345/main"}"#,
        ),
        (
            Request::ResumeOrNew {
                id: id(),
                recency_secs: 600,
            },
            r#"{"type":"resume_or_new","id":"r","recency_secs":600}"#,
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

/// One of every [`Event`] variant with its exact wire form.
fn event_cases() -> Vec<(Event, &'static str)> {
    [
        turn_event_cases(),
        voice_and_presence_event_cases(),
        memory_event_cases(),
        confirmation_and_status_event_cases(),
        session_event_cases(),
        relay_history_event_cases(),
        wake_word_event_cases(),
        task_event_cases(),
        task_queue_event_cases(),
        mcp_event_cases(),
    ]
    .concat()
}

/// A `relay` history row with every kind of stored sub-agent entry.
fn relay_history_event_cases() -> Vec<(Event, &'static str)> {
    vec![(
        Event::HistoryEntry {
            id: id(),
            seq: 5,
            role: Role::Tool,
            content: "report".into(),
            reasoning: String::new(),
            reasoning_ms: None,
            tool_name: Some("relay".into()),
            tool_args: None,
            exit_code: Some(0),
            duration_ms: Some(9000),
            compaction_ms: None,
            turn_stats: None,
            relay: vec![
                RelayStepRecord {
                    task: "find it".into(),
                    agent: "web_research".into(),
                    run: Some(RelayStepRun {
                        end: RelayStepEnd::Answered,
                        duration_ms: 8000,
                        tokens: Some(4100),
                        entries: vec![
                            RelayTraceEntry::Reasoning {
                                text: "search".into(),
                                duration_ms: 300,
                            },
                            RelayTraceEntry::Tool {
                                name: "run".into(),
                                args: serde_json::json!({"command": "web search it"}),
                                output: "hit".into(),
                                outcome: Some(RelayToolOutcome {
                                    exit_code: 0,
                                    duration_ms: 40,
                                }),
                            },
                            RelayTraceEntry::Compaction { duration_ms: None },
                            RelayTraceEntry::Text {
                                text: "Found.".into(),
                            },
                        ],
                    }),
                },
                RelayStepRecord {
                    task: "read it".into(),
                    agent: "file_review".into(),
                    run: None,
                },
            ],
        },
        r#"{"type":"history_entry","id":"r","seq":5,"role":"tool","content":"report","tool_name":"relay","exit_code":0,"duration_ms":9000,"relay":[{"task":"find it","agent":"web_research","run":{"end":{"kind":"answered"},"duration_ms":8000,"tokens":4100,"entries":[{"kind":"reasoning","text":"search","duration_ms":300},{"kind":"tool","name":"run","args":{"command":"web search it"},"output":"hit","outcome":{"exit_code":0,"duration_ms":40}},{"kind":"compaction"},{"kind":"text","text":"Found."}]}},{"task":"read it","agent":"file_review"}]}"#,
    )]
}

fn wake_word_event_cases() -> Vec<(Event, &'static str)> {
    vec![
        (
            Event::WakeWordStatus {
                id: id(),
                phrase: "hey jarvis".into(),
                listening: true,
            },
            r#"{"type":"wake_word_status","id":"r","phrase":"hey jarvis","listening":true}"#,
        ),
        (
            Event::WakeWordTest {
                id: id(),
                matched: true,
            },
            r#"{"type":"wake_word_test","id":"r","matched":true}"#,
        ),
        (
            Event::WakeWordSpeech {
                id: id(),
                speaking: true,
            },
            r#"{"type":"wake_word_speech","id":"r","speaking":true}"#,
        ),
    ]
}

fn turn_event_cases() -> Vec<(Event, &'static str)> {
    vec![
        (
            Event::Delta {
                id: id(),
                text: "pong".into(),
            },
            r#"{"type":"delta","id":"r","text":"pong"}"#,
        ),
        (
            Event::ReasoningDelta {
                id: id(),
                text: "hmm".into(),
            },
            r#"{"type":"reasoning_delta","id":"r","text":"hmm"}"#,
        ),
        (
            Event::ToolCall {
                id: id(),
                name: "echo".into(),
                args: json!({"text": "hi"}),
            },
            r#"{"type":"tool_call","id":"r","name":"echo","args":{"text":"hi"}}"#,
        ),
        (
            Event::ToolResult {
                id: id(),
                name: "echo".into(),
                result: json!("hi"),
            },
            r#"{"type":"tool_result","id":"r","name":"echo","result":"hi"}"#,
        ),
        (
            Event::Error {
                id: id(),
                message: "boom".into(),
            },
            r#"{"type":"error","id":"r","message":"boom"}"#,
        ),
        (Event::Done { id: id() }, r#"{"type":"done","id":"r"}"#),
        (
            Event::LastDelta {
                id: id(),
                text: "Hello world".into(),
            },
            r#"{"type":"last_delta","id":"r","text":"Hello world"}"#,
        ),
    ]
}

fn voice_and_presence_event_cases() -> Vec<(Event, &'static str)> {
    vec![
        (
            Event::Presence {
                id: id(),
                state: PresenceState::Sleeping,
            },
            r#"{"type":"presence","id":"r","state":"sleeping"}"#,
        ),
        (
            Event::VoiceState {
                id: id(),
                state: VoiceCaptureState::Recording,
            },
            r#"{"type":"voice_state","id":"r","state":"recording"}"#,
        ),
        (
            Event::Transcription {
                id: id(),
                text: "hello world".into(),
            },
            r#"{"type":"transcription","id":"r","text":"hello world"}"#,
        ),
        (
            Event::ListenState {
                id: id(),
                phase: ListenPhase::Armed,
            },
            r#"{"type":"listen_state","id":"r","phase":"armed"}"#,
        ),
        (
            Event::WakeWordDetected { id: id() },
            r#"{"type":"wake_word_detected","id":"r"}"#,
        ),
        (
            Event::VoiceOutputState {
                id: id(),
                enabled: true,
            },
            r#"{"type":"voice_output_state","id":"r","enabled":true}"#,
        ),
        (
            Event::Readiness {
                id: id(),
                component: StartupComponent::Mcp {
                    server: "fs".into(),
                },
                state: ComponentReadiness::Unavailable {
                    reason: "no npx".into(),
                },
            },
            r#"{"type":"readiness","id":"r","component":{"kind":"mcp","server":"fs"},"state":{"state":"unavailable","reason":"no npx"}}"#,
        ),
        (
            Event::Readiness {
                id: id(),
                component: StartupComponent::VoiceInput,
                state: ComponentReadiness::Starting,
            },
            r#"{"type":"readiness","id":"r","component":{"kind":"voice_input"},"state":{"state":"starting"}}"#,
        ),
        (
            Event::SpeakingState {
                id: id(),
                speaking: true,
            },
            r#"{"type":"speaking_state","id":"r","speaking":true}"#,
        ),
        (
            Event::ChatFocus {
                id: id(),
                focused: true,
            },
            r#"{"type":"chat_focus","id":"r","focused":true}"#,
        ),
    ]
}

fn memory_event_cases() -> Vec<(Event, &'static str)> {
    vec![
        (
            Event::SemanticHit {
                id: id(),
                message_id: 42,
                chunk_id: 7,
                session_id: "s".into(),
                timestamp: "2026-04-28T00:00:00Z".into(),
                role: Role::User,
                content: "the rust embeddings daemon".into(),
                similarity: 0.5,
            },
            r#"{"type":"semantic_hit","id":"r","message_id":42,"chunk_id":7,"session_id":"s","timestamp":"2026-04-28T00:00:00Z","role":"user","content":"the rust embeddings daemon","similarity":0.5}"#,
        ),
        (
            Event::MemoryValue {
                id: id(),
                key: "absent".into(),
                value: None,
            },
            r#"{"type":"memory_value","id":"r","key":"absent","value":null}"#,
        ),
        (
            Event::MemoryKeys {
                id: id(),
                keys: vec!["a".into(), "b".into()],
            },
            r#"{"type":"memory_keys","id":"r","keys":["a","b"]}"#,
        ),
        (
            Event::MemoryRow {
                id: id(),
                memory_id: 17,
                key: "fact:user.name".into(),
                value: "Ben".into(),
            },
            r#"{"type":"memory_row","id":"r","memory_id":17,"key":"fact:user.name","value":"Ben"}"#,
        ),
        (
            Event::MemoryForgetResult {
                id: id(),
                deleted: true,
                key: Some("fact:user.name".into()),
            },
            r#"{"type":"memory_forget_result","id":"r","deleted":true,"key":"fact:user.name"}"#,
        ),
        (
            Event::ReindexProgress {
                id: id(),
                kind: ReindexKind::Chunks,
                done: 3,
                total: 10,
            },
            r#"{"type":"reindex_progress","id":"r","kind":"chunks","done":3,"total":10}"#,
        ),
    ]
}

fn confirmation_and_status_event_cases() -> Vec<(Event, &'static str)> {
    vec![
        (
            Event::ConfirmRequest {
                id: id(),
                confirm_id: "c-abc".into(),
                tool: "bash".into(),
                script: "rm -rf /tmp/foo".into(),
                matched_pattern: "rm -rf".into(),
                always_allow: Vec::new(),
                source: ConfirmSource::Main,
            },
            r#"{"type":"confirm_request","id":"r","confirm_id":"c-abc","tool":"bash","script":"rm -rf /tmp/foo","matched_pattern":"rm -rf"}"#,
        ),
        (
            Event::ConfirmRequest {
                id: id(),
                confirm_id: "c-abc".into(),
                tool: "bash".into(),
                script: "cargo build".into(),
                matched_pattern: "not on the allowlist: cargo".into(),
                always_allow: vec!["cargo".into()],
                source: ConfirmSource::Main,
            },
            r#"{"type":"confirm_request","id":"r","confirm_id":"c-abc","tool":"bash","script":"cargo build","matched_pattern":"not on the allowlist: cargo","always_allow":["cargo"]}"#,
        ),
        (
            Event::ConfirmRequest {
                id: id(),
                confirm_id: "c-abc".into(),
                tool: "bash".into(),
                script: "curl example.com".into(),
                matched_pattern: "not on the allowlist: curl".into(),
                always_allow: Vec::new(),
                source: ConfirmSource::Relay {
                    task: "summarise the page".into(),
                    run: None,
                },
            },
            r#"{"type":"confirm_request","id":"r","confirm_id":"c-abc","tool":"bash","script":"curl example.com","matched_pattern":"not on the allowlist: curl","source":{"agent":"relay","task":"summarise the page"}}"#,
        ),
        (
            Event::ConfirmRequest {
                id: id(),
                confirm_id: "c-abc".into(),
                tool: "web".into(),
                script: "web fetch https://example.com".into(),
                matched_pattern: "fetches from example.com".into(),
                always_allow: vec!["example.com".into()],
                source: ConfirmSource::Relay {
                    task: "summarise the page".into(),
                    run: Some(TaskRunId(9)),
                },
            },
            r#"{"type":"confirm_request","id":"r","confirm_id":"c-abc","tool":"web","script":"web fetch https://example.com","matched_pattern":"fetches from example.com","always_allow":["example.com"],"source":{"agent":"relay","task":"summarise the page","run":9}}"#,
        ),
        (
            Event::Relay {
                id: id(),
                event: RelayEvent::Planned {
                    steps: vec![RelayStepPlan {
                        task: "find the release".into(),
                        agent: "web_research".into(),
                    }],
                },
            },
            r#"{"type":"relay","id":"r","event":{"kind":"planned","steps":[{"task":"find the release","agent":"web_research"}]}}"#,
        ),
        (
            Event::Capabilities {
                id: id(),
                vision: true,
                model_name: "Qwen3-14B-GGUF:Q4_K_M".into(),
                auto_mode: true,
            },
            r#"{"type":"capabilities","id":"r","vision":true,"model_name":"Qwen3-14B-GGUF:Q4_K_M","auto_mode":true}"#,
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
        ),
    ]
}

fn session_event_cases() -> Vec<(Event, &'static str)> {
    vec![
        (
            Event::SessionTitle {
                id: id(),
                session_id: "s".into(),
                title: "cats".into(),
            },
            r#"{"type":"session_title","id":"r","session_id":"s","title":"cats"}"#,
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
                last_active_at: "2026-01-02T00:00:00Z".into(),
                task_run: Some(TaskRunSummary {
                    task: TaskId(4),
                    run: TaskRunId(9),
                    task_name: TaskName::try_from("digest".to_string()).unwrap(),
                    state: TaskRunState::Failed {
                        reason: "timed out".into(),
                    },
                }),
            },
            r#"{"type":"branch_info","id":"r","branch_id":7,"session_id":"s","session_started_at":"2026-01-01T00:00:00Z","session_ended_at":null,"session_title":"cats","name":"main","parent_branch_name":null,"fork_point_seq":null,"created_at":"2026-01-01T00:00:00Z","message_count":4,"is_current_in_session":true,"is_active_session":false,"last_active_at":"2026-01-02T00:00:00Z","task_run":{"task":4,"run":9,"task_name":"digest","state":{"state":"failed","reason":"timed out"}}}"#,
        ),
        (
            Event::BranchSwitched {
                id: id(),
                branch_id: Some(9),
                session_id: "s".into(),
                session_title: None,
                name: "experiment".into(),
                parent_branch_name: Some("main".into()),
                fork_point_seq: Some(5),
            },
            r#"{"type":"branch_switched","id":"r","branch_id":9,"session_id":"s","session_title":null,"name":"experiment","parent_branch_name":"main","fork_point_seq":5}"#,
        ),
        (
            Event::HistoryEntry {
                id: id(),
                seq: 3,
                role: Role::Assistant,
                content: "hello".into(),
                reasoning: "greet back".into(),
                reasoning_ms: Some(1500),
                tool_name: None,
                tool_args: None,
                exit_code: None,
                duration_ms: None,
                compaction_ms: None,
                turn_stats: None,
                relay: Vec::new(),
            },
            r#"{"type":"history_entry","id":"r","seq":3,"role":"assistant","content":"hello","reasoning":"greet back","reasoning_ms":1500,"tool_name":null}"#,
        ),
        (
            Event::HistoryEntry {
                id: id(),
                seq: 4,
                role: Role::Tool,
                content: "out".into(),
                reasoning: String::new(),
                reasoning_ms: None,
                tool_name: Some("run".into()),
                tool_args: Some(serde_json::json!({"command": "ls"})),
                exit_code: Some(1),
                duration_ms: Some(7),
                compaction_ms: Some(4200),
                turn_stats: Some(TurnStats {
                    duration_ms: 5300,
                    tokens: 812,
                }),
                relay: Vec::new(),
            },
            r#"{"type":"history_entry","id":"r","seq":4,"role":"tool","content":"out","tool_name":"run","tool_args":{"command":"ls"},"exit_code":1,"duration_ms":7,"compaction_ms":4200,"turn_stats":{"duration_ms":5300,"tokens":812}}"#,
        ),
        (
            Event::UndoApplied {
                id: id(),
                removed_messages: 2,
                last_user_text: Some("hi".into()),
            },
            r#"{"type":"undo_applied","id":"r","removed_messages":2,"last_user_text":"hi"}"#,
        ),
    ]
}

#[test]
fn events_match_pinned_wire_format() {
    for (ev, wire) in event_cases() {
        assert_eq!(serde_json::to_string(&ev).unwrap(), wire);
        let parsed: Event = serde_json::from_str(wire).unwrap();
        assert_eq!(parsed, ev, "{wire}");
        assert_eq!(ev.id(), "r", "{wire}");
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
fn subscribe_filter_matches_everything_by_default_else_listed_only() {
    assert!(SubscribeFilter::default().matches(EventKind::Delta));
    let filter = SubscribeFilter {
        kinds: vec![EventKind::Presence, EventKind::LastDelta],
    };
    assert!(filter.matches(EventKind::Presence));
    assert!(filter.matches(EventKind::LastDelta));
    assert!(!filter.matches(EventKind::Delta));
    assert!(!filter.matches(EventKind::ToolCall));
    assert!(!filter.matches(EventKind::Done));
}

#[test]
fn startup_readiness_keeps_the_latest_state_in_first_seen_order() {
    let mut readiness = StartupReadiness::default();
    readiness.record(StartupComponent::Embedding, ComponentReadiness::Starting);
    readiness.record(StartupComponent::VoiceInput, ComponentReadiness::Ready);
    assert_eq!(
        readiness.starting().collect::<Vec<_>>(),
        [&StartupComponent::Embedding]
    );

    readiness.record(StartupComponent::Embedding, ComponentReadiness::Ready);
    assert_eq!(readiness.starting().count(), 0);
    assert_eq!(
        readiness.iter().map(|(c, _)| c.clone()).collect::<Vec<_>>(),
        [StartupComponent::Embedding, StartupComponent::VoiceInput]
    );
}

fn sample_task_spec() -> TaskSpec {
    TaskSpec {
        name: TaskName::try_from("digest".to_string()).unwrap(),
        prompt: TaskPrompt::try_from("summarize my inbox".to_string()).unwrap(),
        triggers: vec![
            TaskTrigger::Manual,
            TaskTrigger::At {
                at: NaiveDate::from_ymd_opt(2026, 11, 4)
                    .and_then(|day| day.and_hms_opt(23, 0, 0))
                    .unwrap(),
            },
            TaskTrigger::Schedule {
                cron: "0 9 * * 1-5".into(),
                missed: MissedFire::Skip,
            },
            TaskTrigger::Event {
                event: TaskEvent::DaemonStarted,
                cooldown_secs: 0,
            },
            TaskTrigger::Event {
                event: TaskEvent::PathChanged {
                    path: PathBuf::from("/home/u/inbox"),
                    debounce_secs: 5,
                },
                cooldown_secs: 300,
            },
            TaskTrigger::Event {
                event: TaskEvent::TaskFinished {
                    task: TaskId(3),
                    outcome: FinishOutcome::Failed,
                },
                cooldown_secs: 60,
            },
        ],
        enabled: true,
        mode: PermissionMode::AcceptEdits,
        allowed_hosts: vec!["Example.org".to_string().try_into().unwrap()],
        announce: true,
    }
}

fn sample_plan() -> Plan {
    Plan {
        id: PlanId("p1".into()),
        summary: "read and summarize".into(),
        steps: vec![PlanStep {
            description: "list mail".into(),
            actions: vec![
                AnticipatedAction {
                    action: PlannedAction::Program {
                        program: "git".into(),
                        subcommand: Some("status".into()),
                    },
                    problem: None,
                    outlook: ActionOutlook::JudgeDecides,
                },
                AnticipatedAction {
                    action: PlannedAction::WebHost {
                        host: "example.org".into(),
                    },
                    problem: None,
                    outlook: ActionOutlook::Unasked,
                },
                AnticipatedAction {
                    action: PlannedAction::McpTool {
                        name: "mcp__mail__list".into(),
                    },
                    problem: Some("no such tool".into()),
                    outlook: ActionOutlook::Asks,
                },
                AnticipatedAction {
                    action: PlannedAction::App {
                        desktop_entry: "firefox.desktop".into(),
                    },
                    problem: None,
                    outlook: ActionOutlook::Asks,
                },
                AnticipatedAction {
                    action: PlannedAction::Write {
                        under: PathBuf::from("/home/u/notes"),
                    },
                    problem: None,
                    outlook: ActionOutlook::Unasked,
                },
            ],
        }],
    }
}

const SAMPLE_PLAN_WIRE: &str = r#"{"id":"p1","summary":"read and summarize","steps":[{"description":"list mail","actions":[{"action":{"kind":"program","program":"git","subcommand":"status"},"outlook":"judge_decides"},{"action":{"kind":"web_host","host":"example.org"},"outlook":"unasked"},{"action":{"kind":"mcp_tool","name":"mcp__mail__list"},"problem":"no such tool","outlook":"asks"},{"action":{"kind":"app","desktop_entry":"firefox.desktop"},"outlook":"asks"},{"action":{"kind":"write","under":"/home/u/notes"},"outlook":"unasked"}]}]}"#;

const SAMPLE_TASK_SPEC_WIRE: &str = r#"{"name":"digest","prompt":"summarize my inbox","triggers":[{"kind":"manual"},{"kind":"at","at":"2026-11-04T23:00:00"},{"kind":"schedule","cron":"0 9 * * 1-5","missed":"skip"},{"kind":"event","event":{"kind":"daemon_started"},"cooldown_secs":0},{"kind":"event","event":{"kind":"path_changed","path":"/home/u/inbox","debounce_secs":5},"cooldown_secs":300},{"kind":"event","event":{"kind":"task_finished","task":3,"outcome":"failed"},"cooldown_secs":60}],"enabled":true,"mode":"accept_edits","allowed_hosts":["example.org"],"announce":true}"#;

fn task_request_cases() -> Vec<(Request, &'static str)> {
    let save_wire =
        format!(r#"{{"type":"task_save","id":"r","task":4,"spec":{SAMPLE_TASK_SPEC_WIRE}}}"#);
    vec![
        (
            Request::TaskList { id: id() },
            r#"{"type":"task_list","id":"r"}"#,
        ),
        (
            Request::TaskGet {
                id: id(),
                task: TaskId(4),
            },
            r#"{"type":"task_get","id":"r","task":4}"#,
        ),
        (
            Request::TaskSave {
                id: id(),
                task: Some(TaskId(4)),
                spec: sample_task_spec(),
            },
            save_wire.leak(),
        ),
        (
            Request::TaskSave {
                id: id(),
                task: None,
                spec: TaskSpec {
                    mode: PermissionMode::Manual,
                    allowed_hosts: Vec::new(),
                    announce: false,
                    triggers: Vec::new(),
                    ..sample_task_spec()
                },
            },
            r#"{"type":"task_save","id":"r","spec":{"name":"digest","prompt":"summarize my inbox","triggers":[],"enabled":true,"mode":"manual"}}"#,
        ),
        (
            Request::TaskDelete {
                id: id(),
                task: TaskId(4),
            },
            r#"{"type":"task_delete","id":"r","task":4}"#,
        ),
        (
            Request::TaskSetEnabled {
                id: id(),
                task: TaskId(4),
                enabled: false,
            },
            r#"{"type":"task_set_enabled","id":"r","task":4,"enabled":false}"#,
        ),
        (
            Request::TaskRun {
                id: id(),
                task: TaskId(4),
            },
            r#"{"type":"task_run","id":"r","task":4}"#,
        ),
        (
            Request::TaskCancel {
                id: id(),
                run: TaskRunId(9),
            },
            r#"{"type":"task_cancel","id":"r","run":9}"#,
        ),
        (
            Request::TaskQueue {
                id: id(),
                op: TaskQueueOp::Move {
                    run: TaskRunId(9),
                    to: 0,
                },
            },
            r#"{"type":"task_queue","id":"r","op":{"kind":"move","run":9,"to":0}}"#,
        ),
        (
            Request::TaskQueue {
                id: id(),
                op: TaskQueueOp::Pause,
            },
            r#"{"type":"task_queue","id":"r","op":{"kind":"pause"}}"#,
        ),
        (
            Request::TaskQueue {
                id: id(),
                op: TaskQueueOp::CancelTurn { id: "q".into() },
            },
            r#"{"type":"task_queue","id":"r","op":{"kind":"cancel_turn","id":"q"}}"#,
        ),
    ]
}

fn task_confirm_request_cases() -> Vec<(Request, &'static str)> {
    vec![
        (
            Request::TaskConfirm {
                id: id(),
                run: TaskRunId(9),
                confirm_id: "c".into(),
                allow: true,
                always: false,
            },
            r#"{"type":"task_confirm","id":"r","run":9,"confirm_id":"c","allow":true}"#,
        ),
        (
            Request::TaskConfirm {
                id: id(),
                run: TaskRunId(9),
                confirm_id: "c".into(),
                allow: true,
                always: true,
            },
            r#"{"type":"task_confirm","id":"r","run":9,"confirm_id":"c","allow":true,"always":true}"#,
        ),
        (
            Request::TaskRuns {
                id: id(),
                task: TaskId(4),
            },
            r#"{"type":"task_runs","id":"r","task":4}"#,
        ),
    ]
}

fn task_refine_request_cases() -> Vec<(Request, &'static str)> {
    vec![
        (
            Request::TaskRefine {
                id: id(),
                prompt: "summarize my inbox".to_string().try_into().unwrap(),
                mode: PermissionMode::Manual,
                allowed_hosts: Vec::new(),
                feedback: None,
            },
            r#"{"type":"task_refine","id":"r","prompt":"summarize my inbox","mode":"manual"}"#,
        ),
        (
            Request::TaskRefine {
                id: id(),
                prompt: "summarize my inbox".to_string().try_into().unwrap(),
                mode: PermissionMode::AcceptEdits,
                allowed_hosts: vec!["example.org".to_string().try_into().unwrap()],
                feedback: Some("skip the mail".into()),
            },
            r#"{"type":"task_refine","id":"r","prompt":"summarize my inbox","mode":"accept_edits","allowed_hosts":["example.org"],"feedback":"skip the mail"}"#,
        ),
        (
            Request::TaskCheckTrigger {
                id: id(),
                task: None,
                trigger: TaskTrigger::Manual,
            },
            r#"{"type":"task_check_trigger","id":"r","trigger":{"kind":"manual"}}"#,
        ),
    ]
}

fn mcp_request_cases() -> Vec<(Request, &'static str)> {
    vec![
        (
            Request::McpLogin {
                id: id(),
                server: "drive".into(),
            },
            r#"{"type":"mcp_login","id":"r","server":"drive"}"#,
        ),
        (
            Request::McpLogout {
                id: id(),
                server: "drive".into(),
            },
            r#"{"type":"mcp_logout","id":"r","server":"drive"}"#,
        ),
        (
            Request::McpStatus { id: id() },
            r#"{"type":"mcp_status","id":"r"}"#,
        ),
    ]
}

fn mcp_event_cases() -> Vec<(Event, &'static str)> {
    vec![
        (
            Event::McpAuthorizeUrl {
                id: id(),
                server: "drive".into(),
                url: "https://auth.example/authorize?state=s".into(),
            },
            r#"{"type":"mcp_authorize_url","id":"r","server":"drive","url":"https://auth.example/authorize?state=s"}"#,
        ),
        (
            Event::McpServer {
                id: id(),
                server: "drive".into(),
                transport: McpTransportKind::Http,
                state: McpServerState::NeedsAuthorization {
                    scopes: vec!["drive.readonly".into()],
                },
            },
            r#"{"type":"mcp_server","id":"r","server":"drive","transport":"http","state":{"state":"needs_authorization","scopes":["drive.readonly"]}}"#,
        ),
        (
            Event::McpServer {
                id: id(),
                server: "fs".into(),
                transport: McpTransportKind::Stdio,
                state: McpServerState::Unavailable {
                    reason: "missing".into(),
                },
            },
            r#"{"type":"mcp_server","id":"r","server":"fs","transport":"stdio","state":{"state":"unavailable","reason":"missing"}}"#,
        ),
        (
            Event::McpLoggedOut {
                id: id(),
                server: "drive".into(),
                revocation: TokenRevocation::Failed {
                    reason: "HTTP 503".into(),
                },
            },
            r#"{"type":"mcp_logged_out","id":"r","server":"drive","revocation":{"outcome":"failed","reason":"HTTP 503"}}"#,
        ),
    ]
}

fn task_queue_event_cases() -> Vec<(Event, &'static str)> {
    vec![
        (
            Event::TaskQueue {
                id: id(),
                snapshot: TaskQueueSnapshot {
                    items: vec![
                        QueueItem::Run {
                            run: TaskRunId(9),
                            task: TaskId(4),
                            task_name: "digest".to_string().try_into().unwrap(),
                            state: TaskRunState::Suspended {
                                reason: SuspendReason::GpuBusy,
                            },
                            prompt: None,
                            started_at: None,
                        },
                        QueueItem::Run {
                            run: TaskRunId(8),
                            task: TaskId(5),
                            task_name: "fetch".to_string().try_into().unwrap(),
                            state: TaskRunState::NeedsInput,
                            started_at: DateTime::from_timestamp(1_793_887_200, 0),
                            prompt: Some(PausedPrompt {
                                confirm_id: "c".into(),
                                tool: "web".into(),
                                script: "web fetch https://example.org/".into(),
                                matched_pattern: "fetches from example.org".into(),
                                host: Some("example.org".to_string().try_into().unwrap()),
                                source: ConfirmSource::Task {
                                    task: TaskId(5),
                                    run: TaskRunId(8),
                                },
                            }),
                        },
                        QueueItem::UserTurn {
                            id: "q".into(),
                            preview: "hi".into(),
                        },
                    ],
                    lanes: vec![
                        LaneHolder::Main,
                        LaneHolder::Task { run: TaskRunId(8) },
                        LaneHolder::Idle,
                    ],
                    paused: false,
                    held: Some(ConversationHold {
                        resumes_at: Some(
                            NaiveDate::from_ymd_opt(2026, 11, 5)
                                .unwrap()
                                .and_hms_opt(9, 30, 0)
                                .unwrap()
                                .and_utc(),
                        ),
                    }),
                },
            },
            r#"{"type":"task_queue","id":"r","snapshot":{"items":[{"kind":"run","run":9,"task":4,"task_name":"digest","state":{"state":"suspended","reason":"gpu_busy"}},{"kind":"run","run":8,"task":5,"task_name":"fetch","state":{"state":"needs_input"},"prompt":{"confirm_id":"c","tool":"web","script":"web fetch https://example.org/","matched_pattern":"fetches from example.org","host":"example.org","source":{"agent":"task","task":5,"run":8}},"started_at":"2026-11-05T14:00:00Z"},{"kind":"user_turn","id":"q","preview":"hi"}],"lanes":[{"holder":"main"},{"holder":"task","run":8},{"holder":"idle"}],"paused":false,"held":{"resumes_at":"2026-11-05T09:30:00Z"}}}"#,
        ),
        (
            Event::TaskReply {
                id: id(),
                run: TaskRunId(9),
                task_name: "inbox".to_string().try_into().unwrap(),
                reply: "3 new messages".into(),
            },
            r#"{"type":"task_reply","id":"r","run":9,"task_name":"inbox","reply":"3 new messages"}"#,
        ),
        (
            Event::Queued {
                id: id(),
                behind: 2,
            },
            r#"{"type":"queued","id":"r","behind":2}"#,
        ),
    ]
}

fn task_event_cases() -> Vec<(Event, &'static str)> {
    let wire = format!(
        r#"{{"type":"task","id":"r","task_id":4,"spec":{SAMPLE_TASK_SPEC_WIRE},"next_fire":"2026-11-05T14:00:00Z","last_run":{{"state":"failed","reason":"timed out"}},"trigger_failure":"watching /tmp/in stopped: No such file or directory"}}"#
    );
    vec![
        (
            Event::Task {
                id: id(),
                task_id: TaskId(4),
                spec: sample_task_spec(),
                next_fire: DateTime::from_timestamp(1_793_887_200, 0),
                last_run: Some(TaskRunState::Failed {
                    reason: "timed out".into(),
                }),
                trigger_failure: Some(
                    "watching /tmp/in stopped: No such file or directory".into(),
                ),
            },
            wire.leak(),
        ),
        (
            Event::TriggerFires {
                id: id(),
                fires: DateTime::from_timestamp(1_793_887_200, 0)
                    .into_iter()
                    .collect(),
            },
            r#"{"type":"trigger_fires","id":"r","fires":["2026-11-05T14:00:00Z"]}"#,
        ),
        (
            Event::TaskRunRecord {
                id: id(),
                record: TaskRunRecord {
                    run: TaskRunId(9),
                    trigger: TaskTrigger::Manual,
                    state: TaskRunState::Succeeded,
                    queued_at: DateTime::from_timestamp(1_793_887_200, 0).unwrap(),
                    started_at: DateTime::from_timestamp(1_793_887_201, 0),
                    ended_at: DateTime::from_timestamp(1_793_887_260, 0),
                    reply: Some("done".into()),
                    session_id: Some("s".into()),
                },
            },
            r#"{"type":"task_run_record","id":"r","record":{"run":9,"trigger":{"kind":"manual"},"state":{"state":"succeeded"},"queued_at":"2026-11-05T14:00:00Z","started_at":"2026-11-05T14:00:01Z","ended_at":"2026-11-05T14:01:00Z","reply":"done","session_id":"s"}}"#,
        ),
        (
            Event::TaskRunQueued {
                id: id(),
                task: TaskId(4),
                run: TaskRunId(9),
            },
            r#"{"type":"task_run_queued","id":"r","task":4,"run":9}"#,
        ),
        (
            Event::TaskRun {
                id: id(),
                run: TaskRunId(9),
                event: Box::new(Event::ConfirmRequest {
                    id: id(),
                    confirm_id: "c".into(),
                    tool: "bash".into(),
                    script: "rm x".into(),
                    matched_pattern: "rm".into(),
                    always_allow: Vec::new(),
                    source: ConfirmSource::Task {
                        task: TaskId(4),
                        run: TaskRunId(9),
                    },
                }),
            },
            r#"{"type":"task_run","id":"r","run":9,"event":{"type":"confirm_request","id":"r","confirm_id":"c","tool":"bash","script":"rm x","matched_pattern":"rm","source":{"agent":"task","task":4,"run":9}}}"#,
        ),
        (
            Event::TaskRefined {
                id: id(),
                prompt: "summarize my inbox".to_string().try_into().unwrap(),
                plan: sample_plan(),
                recommended: ModeRecommendation {
                    mode: PermissionMode::AcceptEdits,
                    covers: vec![PlannedAction::Write {
                        under: "/home/u/notes".into(),
                    }],
                    still_asks: vec![PlannedAction::WebHost {
                        host: "example.org".into(),
                    }],
                },
            },
            format!(r#"{{"type":"task_refined","id":"r","prompt":"summarize my inbox","plan":{SAMPLE_PLAN_WIRE},"recommended":{{"mode":"accept_edits","covers":[{{"kind":"write","under":"/home/u/notes"}}],"still_asks":[{{"kind":"web_host","host":"example.org"}}]}}}}"#).leak(),
        ),
        (
            Event::TaskPlan {
                id: id(),
                event: Box::new(Event::Delta {
                    id: id(),
                    text: "hi".into(),
                }),
            },
            r#"{"type":"task_plan","id":"r","event":{"type":"delta","id":"r","text":"hi"}}"#,
        ),
    ]
}

#[test]
fn task_run_states_match_pinned_wire_format() {
    let cases = [
        (TaskRunState::Queued, r#"{"state":"queued"}"#),
        (TaskRunState::Running, r#"{"state":"running"}"#),
        (
            TaskRunState::Suspended {
                reason: SuspendReason::UserConversation,
            },
            r#"{"state":"suspended","reason":"user_conversation"}"#,
        ),
        (
            TaskRunState::Suspended {
                reason: SuspendReason::GpuBusy,
            },
            r#"{"state":"suspended","reason":"gpu_busy"}"#,
        ),
        (TaskRunState::NeedsInput, r#"{"state":"needs_input"}"#),
        (TaskRunState::Succeeded, r#"{"state":"succeeded"}"#),
        (
            TaskRunState::Failed {
                reason: "daemon stopped".into(),
            },
            r#"{"state":"failed","reason":"daemon stopped"}"#,
        ),
        (TaskRunState::Cancelled, r#"{"state":"cancelled"}"#),
    ];
    for (state, wire) in cases {
        assert_eq!(serde_json::to_string(&state).unwrap(), wire);
        assert_eq!(serde_json::from_str::<TaskRunState>(wire).unwrap(), state);
    }
}

#[test]
fn task_save_refuses_blank_names_and_oversized_prompts_at_parse() {
    let save = |name: &str, prompt: &str| {
        serde_json::to_string(&json!({
            "type": "task_save",
            "id": "r",
            "spec": {"name": name, "prompt": prompt, "enabled": true},
        }))
        .unwrap()
    };
    let parse = |line: String| serde_json::from_str::<Request>(&line);

    assert!(parse(save("   ", "hi")).is_err());
    assert!(parse(save(&"n".repeat(MAX_TASK_NAME_CHARS + 1), "hi")).is_err());
    assert!(parse(save("digest", " \n ")).is_err());
    assert!(parse(save("digest", &"p".repeat(MAX_TASK_PROMPT_BYTES + 1))).is_err());

    let Request::TaskSave { spec, .. } = parse(save("  digest  ", "hi")).unwrap() else {
        panic!("not a task_save");
    };
    assert_eq!(spec.name.as_str(), "digest");
}

#[test]
fn task_save_takes_only_exact_host_names_as_allowed_hosts() {
    let save = |host: &str| {
        serde_json::to_string(&json!({
            "type": "task_save",
            "id": "r",
            "spec": {"name": "n", "prompt": "p", "enabled": true, "allowed_hosts": [host]},
        }))
        .unwrap()
    };
    let parse = |line: String| serde_json::from_str::<Request>(&line);

    for refused in [
        "",
        "*.example.org",
        "https://example.org",
        "example.org/path",
        "example..org",
        "-example.org",
        "example.org.",
        "exa mple.org",
    ] {
        assert!(parse(save(refused)).is_err(), "{refused:?} was taken");
    }
    let Request::TaskSave { spec, .. } = parse(save(" API.Example.org ")).unwrap() else {
        panic!("not a task_save");
    };
    assert_eq!(spec.allowed_hosts[0].as_str(), "api.example.org");
}
