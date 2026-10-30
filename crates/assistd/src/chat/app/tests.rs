use std::path::{Path, PathBuf};

use assistd_tools::{ConfirmSource, Risk};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt};
use tokio::net::UnixListener;
use tokio::task::JoinHandle;

use super::attach::longest_common_prefix;
use super::*;

fn test_app() -> (App, mpsc::Receiver<ChatEvent>) {
    test_app_at(PathBuf::from("/tmp/assistd-test-nonexistent.sock"))
}

fn test_app_at(socket: PathBuf) -> (App, mpsc::Receiver<ChatEvent>) {
    let (tx, rx) = mpsc::channel::<ChatEvent>(16);
    let ipc = Arc::new(IpcClient::with_path(socket));
    let app = App::new(
        ipc,
        tx,
        "test-model".into(),
        true,
        None,
        PathBuf::from("/tmp/assistd-test-nonexistent.toml"),
    );
    (app, rx)
}

fn typed(c: char) -> KeyEvent {
    KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE)
}

fn delta(text: &str) -> Event {
    Event::Delta {
        id: "r".into(),
        text: text.into(),
    }
}

fn done() -> Event {
    Event::Done { id: "r".into() }
}

fn delta_for(id: &str, text: &str) -> Event {
    Event::Delta {
        id: id.into(),
        text: text.into(),
    }
}

fn transcription_for(id: &str, text: &str) -> Event {
    Event::Transcription {
        id: id.into(),
        text: text.into(),
    }
}

fn rendered(app: &mut App) -> Vec<String> {
    let (lines, _) = app.output.render_view(80, 80);
    lines
        .iter()
        .map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect())
        .collect()
}

fn line_index(lines: &[String], needle: &str) -> usize {
    lines
        .iter()
        .position(|l| l.contains(needle))
        .unwrap_or_else(|| panic!("{needle:?} missing from {lines:#?}"))
}

fn start_typed_turn(app: &mut App, id: &str, prompt: &str) {
    app.begin_turn(prompt, &[]);
    app.active_reply = Some(ActiveReply {
        id: id.into(),
        writer: None,
    });
}

fn reply(event: Event) -> ChatEvent {
    ChatEvent::Wire {
        stream: WireStream::Reply,
        event,
    }
}

fn status(event: Event) -> ChatEvent {
    ChatEvent::Wire {
        stream: WireStream::Status,
        event,
    }
}

/// Accept one dialog connection, read its request line, then stream
/// `events` back after a pause in which the query driver sees its writer
/// channel close with nothing readable.
fn mock_daemon(socket: &Path, events: Vec<Event>) -> JoinHandle<()> {
    let listener = UnixListener::bind(socket).unwrap();
    tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let (read, mut write) = stream.into_split();
        let mut reader = tokio::io::BufReader::new(read);
        let mut line = String::new();
        reader.read_line(&mut line).await.unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;
        for ev in events {
            let mut out = serde_json::to_string(&ev).unwrap();
            out.push('\n');
            write.write_all(out.as_bytes()).await.unwrap();
        }
    })
}

#[test]
fn branch_wire_error_releases_the_in_flight_op() {
    let (mut app, _rx) = test_app();
    app.reply_started_at = Some(Instant::now());
    app.in_flight_branch_op = Some(BranchOp::Fork);
    app.on_chat_event(ChatEvent::WireError {
        stream: WireStream::Branch,
        message: "branch connect: no such file".into(),
    });
    assert!(app.in_flight_branch_op.is_none());
    assert!(app.generating(), "a branch failure must not end the reply");
}

#[test]
fn status_stream_done_does_not_end_the_reply() {
    let (mut app, _rx) = test_app();
    app.reply_started_at = Some(Instant::now());
    app.on_chat_event(reply(delta("hi")));
    app.on_chat_event(status(done()));
    assert!(app.generating());
    app.on_chat_event(reply(done()));
    assert!(!app.generating());
}

#[test]
fn voice_turn_spoken_over_a_typed_reply_waits_its_turn() {
    let (mut app, _rx) = test_app();
    start_typed_turn(&mut app, "typed", "typed question");
    app.on_chat_event(reply(delta_for("typed", "typed ")));
    app.on_chat_event(reply(transcription_for("voice", "spoken question")));
    app.on_chat_event(reply(delta_for("typed", "answer")));
    let lines = rendered(&mut app);
    assert!(
        lines.iter().any(|l| l.contains("typed answer")),
        "the transcript must not split the typed block: {lines:#?}",
    );

    app.on_chat_event(reply(Event::Done { id: "typed".into() }));
    assert!(!app.generating());

    app.on_chat_event(reply(delta_for("voice", "spoken answer")));
    assert!(
        app.generating(),
        "the voice turn runs once the typed one ends"
    );
    let lines = rendered(&mut app);
    let typed = line_index(&lines, "typed answer");
    let spoken_q = line_index(&lines, "spoken question");
    let spoken_a = line_index(&lines, "spoken answer");
    assert!(typed < spoken_q, "transcript must not split the reply");
    assert!(spoken_q < spoken_a);

    app.on_chat_event(reply(Event::Done { id: "voice".into() }));
    assert!(!app.generating());
}

#[test]
fn another_turns_terminal_events_leave_the_owner_alone() {
    let (mut app, _rx) = test_app();
    start_typed_turn(&mut app, "typed", "typed question");
    app.on_chat_event(reply(delta_for("typed", "half ")));

    app.on_chat_event(reply(Event::Done { id: "other".into() }));
    assert!(app.generating(), "a foreign Done must not end the reply");
    app.on_chat_event(reply(Event::Error {
        id: "other".into(),
        message: "voice turn rejected".into(),
    }));
    assert!(app.generating(), "a foreign Error must not end the reply");
    assert_eq!(app.notice(), Some("voice turn rejected"));

    app.on_chat_event(reply(delta_for("typed", "written")));
    let lines = rendered(&mut app);
    assert!(
        lines.iter().any(|l| l.contains("half written")),
        "the owner's block stayed open: {lines:#?}",
    );
}

#[test]
fn a_transcript_is_dropped_when_its_turn_never_runs() {
    let (mut app, _rx) = test_app();
    start_typed_turn(&mut app, "typed", "typed question");
    app.on_chat_event(reply(transcription_for("voice", "spoken question")));
    app.on_chat_event(reply(Event::Error {
        id: "voice".into(),
        message: "no capacity".into(),
    }));
    app.on_chat_event(reply(Event::Done { id: "typed".into() }));

    app.on_chat_event(reply(delta_for("later", "unrelated")));
    let lines = rendered(&mut app);
    assert!(
        !lines.iter().any(|l| l.contains("spoken question")),
        "a turn that never ran must not replay its transcript: {lines:#?}",
    );
}

#[test]
fn bus_turn_from_another_client_renders_with_its_transcript() {
    let (mut app, _rx) = test_app();
    app.on_chat_event(ChatEvent::Bus(transcription_for("ptt", "spoken question")));
    app.on_chat_event(ChatEvent::Bus(delta_for("ptt", "spoken answer")));
    assert!(app.generating());
    let lines = rendered(&mut app);
    assert!(line_index(&lines, "spoken question") < line_index(&lines, "spoken answer"));

    app.on_chat_event(ChatEvent::Bus(Event::Done { id: "ptt".into() }));
    assert!(!app.generating());
}

#[test]
fn bus_copies_of_this_chats_query_are_ignored() {
    let (mut app, _rx) = test_app();
    app.remember_own_turn("typed");
    start_typed_turn(&mut app, "typed", "typed question");
    app.on_chat_event(reply(delta_for("typed", "answer")));
    app.on_chat_event(ChatEvent::Bus(delta_for("typed", "echo")));
    app.on_chat_event(ChatEvent::Bus(Event::Done { id: "typed".into() }));

    assert!(
        app.generating(),
        "only the dialog's own Done ends the reply"
    );
    let lines = rendered(&mut app);
    assert!(
        !lines.iter().any(|l| l.contains("echo")),
        "the bus copy must not be drawn twice: {lines:#?}",
    );
}

#[test]
fn bus_terminal_events_of_unseen_requests_are_ignored() {
    let (mut app, _rx) = test_app();
    app.on_chat_event(ChatEvent::Bus(Event::Done { id: "poll".into() }));
    app.on_chat_event(ChatEvent::Bus(Event::Error {
        id: "wake".into(),
        message: "already active".into(),
    }));
    assert!(app.active_reply.is_none());
    assert_eq!(app.notice(), None);
}

#[tokio::test]
async fn query_driver_outlives_its_writer_channel() {
    let dir = tempfile::tempdir().unwrap();
    let socket = dir.path().join("mock.sock");
    let server = mock_daemon(&socket, vec![delta("hi"), done()]);

    let (mut app, mut rx) = test_app_at(socket);
    app.spawn_query("hi".into(), Vec::new());
    app.active_reply = None;

    let mut terminal = false;
    while !terminal {
        match tokio::time::timeout(Duration::from_secs(5), rx.recv())
            .await
            .expect("query driver stalled once the writer channel closed")
        {
            Some(ChatEvent::Wire { event, .. }) => terminal = event.is_terminal(),
            other => panic!("unexpected chat event: {other:?}"),
        }
    }
    server.await.unwrap();
}

#[tokio::test]
async fn capabilities_are_refetched_only_when_the_model_comes_up() {
    let (mut app, _rx) = test_app();
    let presence = |state| {
        status(Event::Presence {
            id: "p".into(),
            state,
        })
    };
    app.on_chat_event(presence(PresenceState::Active));
    assert!(app.tasks.is_empty(), "first report is not a transition");
    app.on_chat_event(presence(PresenceState::Active));
    assert!(app.tasks.is_empty(), "still active");
    app.on_chat_event(presence(PresenceState::Waking));
    assert!(app.tasks.is_empty(), "not up yet");
    app.on_chat_event(presence(PresenceState::Active));
    assert_eq!(app.tasks.len(), 1, "came up from waking");
}

fn open_prompt_offering(app: &mut App, always_allow: Vec<String>) {
    app.open_confirmation(
        "c1".into(),
        ConfirmationRequest {
            tool: "bash".into(),
            script: "rm -rf /tmp/junk".into(),
            matched_pattern: "rm -rf".into(),
            always_allow,
            risk: Risk::Elevated,
            source: ConfirmSource::Main,
        },
    );
}

/// An app with an open prompt offering "always" for `always_allow`, whose
/// answers land on the returned writer.
fn app_with_prompt(
    always_allow: Vec<String>,
) -> (App, mpsc::Receiver<ChatEvent>, mpsc::Receiver<Request>) {
    let (mut app, rx) = test_app();
    let (writer_tx, writer_rx) = mpsc::channel(4);
    app.active_reply = Some(ActiveReply {
        id: "r".into(),
        writer: Some(writer_tx),
    });
    open_prompt_offering(&mut app, always_allow);
    (app, rx, writer_rx)
}

fn key(code: KeyCode) -> KeyEvent {
    KeyEvent::new(code, KeyModifiers::NONE)
}

/// The `(allow, always)` of the next answer on the writer.
async fn confirm_answer(writer_rx: &mut mpsc::Receiver<Request>) -> (bool, bool) {
    match tokio::time::timeout(Duration::from_secs(5), writer_rx.recv()).await {
        Ok(Some(Request::ConfirmResponse {
            confirm_id,
            allow,
            always,
            ..
        })) => {
            assert_eq!(confirm_id, "c1");
            (allow, always)
        }
        other => panic!("expected a ConfirmResponse, got {other:?}"),
    }
}

#[tokio::test]
async fn prompt_answers_match_the_numbered_choices_on_screen() {
    let offering_cargo = || vec!["cargo".to_string()];
    for (always_allow, keys, expected) in [
        (Vec::new(), vec![key(KeyCode::Enter)], (false, false)),
        (Vec::new(), vec![typed('1')], (true, false)),
        (Vec::new(), vec![typed('2')], (false, false)),
        (
            Vec::new(),
            vec![key(KeyCode::Up), key(KeyCode::Enter)],
            (true, false),
        ),
        (offering_cargo(), vec![typed('2')], (true, true)),
        (offering_cargo(), vec![typed('3')], (false, false)),
        (
            offering_cargo(),
            vec![key(KeyCode::Up), key(KeyCode::Enter)],
            (true, true),
        ),
        (offering_cargo(), vec![key(KeyCode::Esc)], (false, false)),
        (
            offering_cargo(),
            vec![KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL)],
            (false, false),
        ),
    ] {
        let (mut app, _rx, mut writer_rx) = app_with_prompt(always_allow.clone());
        for ev in &keys {
            app.on_key(*ev);
        }
        assert!(app.confirmation.is_none(), "{always_allow:?} {keys:?}");
        assert_eq!(
            confirm_answer(&mut writer_rx).await,
            expected,
            "{always_allow:?} {keys:?}"
        );
    }
}

#[tokio::test]
async fn prompt_swallows_typing_without_answering() {
    let (mut app, _rx, mut writer_rx) = app_with_prompt(vec!["cargo".into()]);
    for c in ['y', 'a', 'x', '0', '4'] {
        app.on_key(typed(c));
    }
    app.on_key(key(KeyCode::Tab));
    assert!(app.confirmation.as_ref().is_some_and(|p| p.expanded));
    assert!(app.input.buffer().is_empty());
    assert!(writer_rx.try_recv().is_err());
}

#[tokio::test]
async fn shift_tab_switches_the_open_turn_to_accept_commands_without_answering_its_prompt() {
    let (mut app, _rx, mut writer_rx) = app_with_prompt(Vec::new());
    app.on_key(key(KeyCode::BackTab));
    assert_eq!(app.permission_mode, PermissionMode::AcceptCommands);
    assert!(app.confirmation.is_some(), "the shown prompt stays open");
    match tokio::time::timeout(Duration::from_secs(5), writer_rx.recv()).await {
        Ok(Some(Request::SetPermissionMode { mode, .. })) => {
            assert_eq!(mode, PermissionMode::AcceptCommands);
        }
        other => panic!("expected SetPermissionMode, got {other:?}"),
    }
}

#[tokio::test]
async fn second_prompt_is_denied_while_one_is_open() {
    let (mut app, _rx, mut writer_rx) = app_with_prompt(Vec::new());
    open_prompt_offering(&mut app, Vec::new());
    assert_eq!(confirm_answer(&mut writer_rx).await, (false, false));
    assert!(app.confirmation.is_some(), "the first prompt stays open");
}

#[test]
fn prompt_closes_when_the_turn_ends() {
    let (mut app, _rx) = test_app();
    app.on_chat_event(reply(delta("hi")));
    open_prompt_offering(&mut app, Vec::new());
    app.on_chat_event(reply(done()));
    assert!(
        app.confirmation.is_none(),
        "a finished turn cannot be answered"
    );
}

#[test]
fn tool_call_then_result_creates_one_block_with_command() {
    let (mut app, _rx) = test_app();
    app.on_chat_event(reply(Event::ToolCall {
        id: "c1".into(),
        name: "run".into(),
        args: serde_json::json!({"command": "ls /tmp"}),
    }));
    let running = rendered(&mut app);
    assert!(running.contains(&"▎ $ ls /tmp".to_string()), "{running:#?}");
    assert!(
        running.contains(&"▎ running… (0s)".to_string()),
        "{running:#?}"
    );
    app.on_chat_event(reply(Event::ToolResult {
        id: "c1".into(),
        name: "run".into(),
        result: serde_json::json!({
            "output": "a\nb\n[exit:0 | 5ms]",
            "exit_code": 0,
            "truncated": false,
            "duration_ms": 5,
        }),
    }));
    let lines = rendered(&mut app);
    assert!(!lines.iter().any(|l| l.contains("running…")), "{lines:#?}");
    assert_eq!(lines.iter().filter(|l| l.contains("$ ls /tmp")).count(), 1);
    assert!(lines.contains(&"▎ $ ls /tmp".to_string()), "{lines:#?}");
    assert!(
        lines.iter().any(|l| l.ends_with("[exit:0 | 5ms]")),
        "{lines:#?}"
    );
}

fn type_str(app: &mut App, s: &str) {
    for c in s.chars() {
        app.on_key(typed(c));
    }
}

#[test]
fn tab_accepts_selection_and_fills_buffer() {
    let (mut app, _rx) = test_app();
    type_str(&mut app, "/f");
    app.on_key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE));
    assert_eq!(app.input.buffer(), "/fork");
    assert!(app.slash_suggestions().is_empty());
}

fn picker_entry(name: &str, session: &str, title: &str, active: bool) -> BranchListEntry {
    BranchListEntry {
        name: name.into(),
        parent_branch_name: None,
        fork_point_seq: None,
        message_count: 0,
        is_current_in_session: active,
        is_active_session: active,
        session_short: session.into(),
        session_title: Some(title.into()),
        last_active_at: None,
    }
}

#[test]
fn resume_search_targets_the_selected_match_not_the_unfiltered_row() {
    let (mut app, _rx) = test_app();
    app.branches_buffer = vec![
        picker_entry("main", "aaaaaaaa", "Cat names", false),
        picker_entry("main", "bbbbbbbb", "Tax return", true),
        picker_entry("fix", "cccccccc", "Tax return", false),
    ];
    app.in_flight_branch_op = Some(BranchOp::ResumePicker);
    app.on_chat_event(ChatEvent::Wire {
        stream: WireStream::Branch,
        event: done(),
    });
    let target = |app: &App| {
        app.resume_picker
            .as_ref()
            .and_then(ResumePicker::current_target)
    };
    assert_eq!(target(&app).as_deref(), Some("bbbbbbbb/main"));
    type_str(&mut app, "TAX fi");
    assert_eq!(target(&app).as_deref(), Some("cccccccc/fix"));
    app.on_key(key(KeyCode::Down));
    assert_eq!(target(&app).as_deref(), Some("cccccccc/fix"));
}

#[test]
fn longest_common_prefix_stops_at_char_boundaries() {
    assert_eq!(longest_common_prefix(&["é1.txt", "è2.txt"]), "");
    assert_eq!(longest_common_prefix(&["日本a", "日本b"]), "日本");
    assert_eq!(longest_common_prefix(&["日本", "日本語"]), "日本");
    assert_eq!(longest_common_prefix(&["abc"]), "abc");
    assert_eq!(longest_common_prefix(&[]), "");
}
