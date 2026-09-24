use super::*;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers, MouseEvent, MouseEventKind};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt};

fn test_sleep_cfg() -> SleepConfig {
    let mut cfg = assistd_core::Config::default().sleep;
    cfg.idle_to_drowsy_mins = 0;
    cfg.idle_to_sleep_mins = 0;
    cfg
}

fn test_app() -> (App, mpsc::Receiver<ChatEvent>) {
    test_app_with(true)
}

fn test_app_with(vision_enabled: bool) -> (App, mpsc::Receiver<ChatEvent>) {
    test_app_at(
        std::path::PathBuf::from("/tmp/assistd-test-nonexistent.sock"),
        vision_enabled,
    )
}

fn test_app_at(
    socket: std::path::PathBuf,
    vision_enabled: bool,
) -> (App, mpsc::Receiver<ChatEvent>) {
    let (tx, rx) = mpsc::channel::<ChatEvent>(16);
    let ipc = Arc::new(IpcClient::with_path(socket));
    let app = App::new(
        ipc,
        tx,
        "test-model".into(),
        test_sleep_cfg(),
        vision_enabled,
        None,
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
    app.begin_submit(prompt, &[]);
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
/// `events` back. The delay before the first write gives the query
/// driver time to observe its closed writer channel while nothing is
/// readable, so a driver that parks in that state hangs the test.
async fn mock_daemon(
    socket: std::path::PathBuf,
    events: Vec<Event>,
) -> tokio::task::JoinHandle<()> {
    let listener = tokio::net::UnixListener::bind(&socket).unwrap();
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
fn reply_wire_error_clears_generating() {
    let (mut app, _rx) = test_app();
    app.generating = true;
    app.on_chat_event(ChatEvent::WireError {
        stream: WireStream::Reply,
        message: "boom".into(),
    });
    assert!(!app.generating);
}

#[test]
fn branch_wire_error_releases_the_in_flight_op() {
    let (mut app, _rx) = test_app();
    app.generating = true;
    app.in_flight_branch_op = Some(BranchOp::Fork);
    app.on_chat_event(ChatEvent::WireError {
        stream: WireStream::Branch,
        message: "branch connect: no such file".into(),
    });
    assert!(app.in_flight_branch_op.is_none());
    assert!(app.generating, "a branch failure must not end the reply");
}

#[test]
fn session_title_event_lands_in_the_status_bar() {
    let (mut app, _rx) = test_app();
    assert!(app.session_title.is_none());
    app.on_chat_event(status(Event::SessionTitle {
        id: "q-1".into(),
        session_id: "s-1".into(),
        title: "Cats And Dogs".into(),
    }));
    assert_eq!(app.session_title.as_deref(), Some("Cats And Dogs"));
}

#[test]
fn status_stream_done_does_not_end_the_reply() {
    let (mut app, _rx) = test_app();
    app.generating = true;
    app.on_chat_event(reply(delta("hi")));
    app.on_chat_event(status(done()));
    assert!(app.generating);
    app.on_chat_event(reply(done()));
    assert!(!app.generating);
}

#[test]
fn voice_turn_spoken_over_a_typed_reply_waits_its_turn() {
    let (mut app, _rx) = test_app();
    start_typed_turn(&mut app, "typed", "typed question");
    app.on_chat_event(reply(delta_for("typed", "typed ")));

    // The user speaks while the typed reply is still streaming; the
    // daemon transcribes it, then blocks on its agent-turn lock.
    app.on_chat_event(reply(transcription_for("voice", "spoken question")));
    app.on_chat_event(reply(delta_for("typed", "answer")));
    let lines = rendered(&mut app);
    assert!(
        lines.iter().any(|l| l.contains("typed answer")),
        "the transcript must not split the typed block: {lines:#?}",
    );

    app.on_chat_event(reply(Event::Done { id: "typed".into() }));
    assert!(!app.generating);

    // Only now does the voice turn run, and it opens with its own
    // transcript rather than appending to the finished reply.
    app.on_chat_event(reply(delta_for("voice", "spoken answer")));
    assert!(app.generating);
    let lines = rendered(&mut app);
    let typed = line_index(&lines, "typed answer");
    let spoken_q = line_index(&lines, "spoken question");
    let spoken_a = line_index(&lines, "spoken answer");
    assert!(typed < spoken_q, "transcript must not split the reply");
    assert!(spoken_q < spoken_a);

    app.on_chat_event(reply(Event::Done { id: "voice".into() }));
    assert!(!app.generating);
}

#[test]
fn another_turns_terminal_events_leave_the_owner_alone() {
    let (mut app, _rx) = test_app();
    start_typed_turn(&mut app, "typed", "typed question");
    app.on_chat_event(reply(delta_for("typed", "half ")));

    app.on_chat_event(reply(Event::Done { id: "other".into() }));
    assert!(app.generating, "a foreign Done must not end the reply");
    app.on_chat_event(reply(Event::Error {
        id: "other".into(),
        message: "voice turn rejected".into(),
    }));
    assert!(app.generating, "a foreign Error must not end the reply");
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

#[tokio::test]
async fn query_driver_outlives_its_writer_channel() {
    let dir = tempfile::tempdir().unwrap();
    let socket = dir.path().join("mock.sock");
    let server = mock_daemon(socket.clone(), vec![delta("hi"), done()]).await;

    let (mut app, mut rx) = test_app_at(socket, true);
    app.spawn_query("hi".into(), Vec::new());
    app.active_reply = None;

    let mut terminal = false;
    while !terminal {
        match tokio::time::timeout(Duration::from_secs(5), rx.recv()).await {
            Ok(Some(ChatEvent::Wire { event, .. })) => terminal = event.is_terminal(),
            Ok(other) => panic!("unexpected chat event: {other:?}"),
            Err(_) => panic!("query driver stalled once the writer channel closed"),
        }
    }
    server.await.unwrap();
}

#[test]
fn page_up_increments_scroll() {
    let (mut app, _rx) = test_app();
    app.last_output_height = 10;
    app.on_key(KeyEvent::new(KeyCode::PageUp, KeyModifiers::NONE));
    assert_eq!(app.output.scroll_offset(), 5);
}

fn wheel(kind: MouseEventKind) -> MouseEvent {
    MouseEvent {
        kind,
        column: 0,
        row: 0,
        modifiers: KeyModifiers::NONE,
    }
}

#[test]
fn mouse_wheel_scrolls_by_a_fixed_step() {
    let (mut app, _rx) = test_app();
    for (kind, expected) in [
        (MouseEventKind::ScrollUp, MOUSE_WHEEL_STEP),
        (MouseEventKind::ScrollUp, 2 * MOUSE_WHEEL_STEP),
        (MouseEventKind::Moved, 2 * MOUSE_WHEEL_STEP),
        (MouseEventKind::ScrollDown, MOUSE_WHEEL_STEP),
    ] {
        app.on_mouse(wheel(kind));
        assert_eq!(
            app.output.scroll_offset(),
            usize::from(expected),
            "after {kind:?}"
        );
    }
}

#[test]
fn ctrl_c_on_empty_input_sets_quitting() {
    let (mut app, _rx) = test_app();
    app.on_key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL));
    assert!(app.should_quit());
}

#[test]
fn enter_while_generating_sets_notice() {
    let (mut app, _rx) = test_app();
    app.generating = true;
    app.on_key(typed('h'));
    app.on_key(typed('i'));
    app.on_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
    assert_eq!(app.notice(), Some("still generating, please wait"));
}

#[test]
fn on_tick_clears_stale_notice() {
    let (mut app, _rx) = test_app();
    app.notice = Some(("old".into(), Instant::now() - Duration::from_secs(10)));
    app.on_tick();
    assert!(app.notice().is_none());
}

#[test]
fn spinner_char_cycles() {
    let (mut app, _rx) = test_app();
    let c0 = app.spinner_char();
    app.on_tick();
    let c1 = app.spinner_char();
    assert_ne!(c0, c1);
}

#[test]
fn presence_event_updates_state() {
    let (mut app, _rx) = test_app();
    assert_eq!(app.presence_state, None);
    app.on_chat_event(status(Event::Presence {
        id: "p".into(),
        state: PresenceState::Drowsy,
    }));
    assert_eq!(app.presence_state, Some(PresenceState::Drowsy));
    app.on_chat_event(status(Event::Presence {
        id: "p".into(),
        state: PresenceState::Active,
    }));
    assert_eq!(app.presence_state, Some(PresenceState::Active));
}

fn open_test_modal(app: &mut App) {
    app.open_confirmation_modal(
        "c1".into(),
        "bash".into(),
        "rm -rf /tmp/junk".into(),
        "rm -rf".into(),
    );
}

/// An app with an open modal whose answers land on the returned writer.
fn app_with_modal() -> (App, mpsc::Receiver<ChatEvent>, mpsc::Receiver<Request>) {
    let (mut app, rx) = test_app();
    let (writer_tx, writer_rx) = mpsc::channel(4);
    app.active_reply = Some(ActiveReply {
        id: "r".into(),
        writer: Some(writer_tx),
    });
    open_test_modal(&mut app);
    (app, rx, writer_rx)
}

async fn confirm_answer(writer_rx: &mut mpsc::Receiver<Request>) -> bool {
    match tokio::time::timeout(Duration::from_secs(5), writer_rx.recv()).await {
        Ok(Some(Request::ConfirmResponse {
            confirm_id, allow, ..
        })) => {
            assert_eq!(confirm_id, "c1");
            allow
        }
        other => panic!("expected a ConfirmResponse, got {other:?}"),
    }
}

#[tokio::test]
async fn modal_approves_on_y_once_armed() {
    let (mut app, _rx, mut writer_rx) = app_with_modal();
    app.arm_modal();
    app.on_key(typed('y'));
    assert!(!app.has_modal());
    assert!(confirm_answer(&mut writer_rx).await);
}

#[tokio::test]
async fn modal_ignores_approval_before_armed() {
    let (mut app, _rx, mut writer_rx) = app_with_modal();
    app.on_key(typed('y'));
    app.on_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
    assert!(app.has_modal(), "keys still in flight must not approve");
    assert!(writer_rx.try_recv().is_err());
}

#[tokio::test]
async fn modal_enter_never_approves() {
    let (mut app, _rx, _writer_rx) = app_with_modal();
    app.arm_modal();
    app.on_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
    assert!(app.has_modal());
}

#[tokio::test]
async fn modal_denies_on_n_or_esc_even_before_armed() {
    for key in [KeyCode::Char('n'), KeyCode::Char('N'), KeyCode::Esc] {
        let (mut app, _rx, mut writer_rx) = app_with_modal();
        app.on_key(KeyEvent::new(key, KeyModifiers::NONE));
        assert!(!app.has_modal(), "{key:?}");
        assert!(!confirm_answer(&mut writer_rx).await, "{key:?}");
    }
}

#[test]
fn modal_closes_when_the_turn_ends() {
    let (mut app, _rx) = test_app();
    app.on_chat_event(reply(delta("hi")));
    open_test_modal(&mut app);
    app.on_chat_event(reply(done()));
    assert!(!app.has_modal(), "a finished turn cannot be answered");
}

#[tokio::test]
async fn modal_swallows_unrelated_keys() {
    let (mut app, _rx, _writer_rx) = app_with_modal();
    app.arm_modal();
    app.on_key(typed('x'));
    app.on_key(typed('z'));
    assert!(app.has_modal());
    assert!(app.input.buffer().is_empty());
}

#[test]
fn tool_call_then_result_creates_one_block_with_command() {
    let (mut app, _rx) = test_app();
    app.on_chat_event(reply(Event::ToolCall {
        id: "c1".into(),
        name: "run".into(),
        args: serde_json::json!({"command": "ls /tmp"}),
    }));
    assert!(app.pending_tool_call.is_some());
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
    assert!(app.pending_tool_call.is_none());
    let lines = rendered(&mut app);
    assert!(lines.contains(&"▎ $ ls /tmp".to_string()), "{lines:#?}");
    assert!(
        lines.iter().any(|l| l.ends_with("[exit:0 | 5ms]")),
        "{lines:#?}"
    );
}

#[test]
fn confirm_request_event_opens_modal() {
    let (mut app, _rx) = test_app();
    app.on_chat_event(reply(Event::ConfirmRequest {
        id: "r".into(),
        confirm_id: "c-xyz".into(),
        tool: "bash".into(),
        script: "rm -rf /tmp/foo".into(),
        matched_pattern: "rm -rf".into(),
    }));
    let modal = app.modal.as_ref().expect("modal opened");
    assert_eq!(modal.confirm_id, "c-xyz");
    assert_eq!(modal.request.script, "rm -rf /tmp/foo");
}

#[test]
fn capabilities_event_updates_vision_and_model_name() {
    let (mut app, _rx) = test_app_with(false);
    assert!(!app.vision_enabled);
    app.on_chat_event(status(Event::Capabilities {
        id: "c".into(),
        vision: true,
        model_name: "Qwen".into(),
    }));
    assert!(app.vision_enabled);
    assert_eq!(app.model_name, "Qwen");
}

fn type_str(app: &mut App, s: &str) {
    for c in s.chars() {
        app.on_key(typed(c));
    }
}

#[test]
fn slash_popup_shows_after_slash() {
    let (mut app, _rx) = test_app();
    assert!(app.slash_suggestions().is_empty());
    type_str(&mut app, "/");
    let s = app.slash_suggestions();
    assert_eq!(s.len(), SLASH_COMMANDS.len());
}

#[test]
fn slash_popup_filters_by_prefix() {
    let (mut app, _rx) = test_app();
    type_str(&mut app, "/fo");
    assert_eq!(app.slash_suggestions(), [&("/fork", "<name>")]);
}

#[test]
fn slash_popup_hides_after_whitespace() {
    let (mut app, _rx) = test_app();
    type_str(&mut app, "/attach ");
    assert!(app.slash_suggestions().is_empty());
}

#[test]
fn tab_accepts_selection_and_fills_buffer() {
    let (mut app, _rx) = test_app();
    type_str(&mut app, "/f");
    app.on_key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE));
    assert_eq!(app.input.buffer(), "/fork");
    assert!(app.slash_suggestions().is_empty());
}

#[test]
fn down_moves_selection_when_popup_active() {
    let (mut app, _rx) = test_app();
    type_str(&mut app, "/");
    assert_eq!(app.slash_selected(), 0);
    app.on_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));
    assert_eq!(app.slash_selected(), 1);
}

#[test]
fn esc_dismisses_popup_without_clearing_buffer() {
    let (mut app, _rx) = test_app();
    type_str(&mut app, "/at");
    assert!(!app.slash_suggestions().is_empty());
    app.on_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
    assert!(app.slash_suggestions().is_empty());
    assert_eq!(app.input.buffer(), "/at");
}

fn picker_entry(name: &str, session: &str, current: bool, active_sess: bool) -> BranchListEntry {
    BranchListEntry {
        name: name.into(),
        parent_branch_name: None,
        fork_point_seq: None,
        message_count: 0,
        is_current_in_session: current,
        is_active_session: active_sess,
        session_short: session.into(),
        session_title: None,
    }
}

#[test]
fn open_branch_picker_highlights_active_current() {
    let (mut app, _rx) = test_app();
    app.branches_buffer = vec![
        picker_entry("main", "aaaaaaaa", false, false),
        picker_entry("main", "bbbbbbbb", true, true),
        picker_entry("feat", "bbbbbbbb", false, true),
    ];
    app.open_branch_picker();
    let picker = app.picker_modal.as_ref().expect("picker opened");
    assert_eq!(picker.selected, 1);
    assert_eq!(picker.entries.len(), 3);
}

#[test]
fn open_branch_picker_with_no_entries_sets_notice() {
    let (mut app, _rx) = test_app();
    app.open_branch_picker();
    assert!(app.picker_modal.is_none());
    assert_eq!(app.notice(), Some("no branches to resume"));
}

#[test]
fn picker_current_target_is_session_qualified() {
    let modal = BranchPickerModal {
        entries: vec![picker_entry("feature-x", "deadbeef", false, false)],
        selected: 0,
    };
    assert_eq!(
        modal.current_target().as_deref(),
        Some("deadbeef/feature-x")
    );
}

#[test]
fn picker_arrow_keys_move_selection() {
    let (mut app, _rx) = test_app();
    app.picker_modal = Some(BranchPickerModal {
        entries: vec![
            picker_entry("a", "11111111", false, false),
            picker_entry("b", "11111111", false, false),
            picker_entry("c", "11111111", false, false),
        ],
        selected: 0,
    });
    app.on_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));
    app.on_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));
    assert_eq!(app.picker_modal.as_ref().unwrap().selected, 2);
    app.on_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE));
    assert_eq!(app.picker_modal.as_ref().unwrap().selected, 1);
}

#[test]
fn picker_esc_cancels_without_dispatching() {
    let (mut app, _rx) = test_app();
    app.picker_modal = Some(BranchPickerModal {
        entries: vec![picker_entry("a", "11111111", false, false)],
        selected: 0,
    });
    app.on_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
    assert!(app.picker_modal.is_none());
    assert!(app.in_flight_branch_op.is_none());
}

#[test]
fn dismissal_resets_after_buffer_clears() {
    let (mut app, _rx) = test_app();
    type_str(&mut app, "/at");
    app.on_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
    for _ in 0..3 {
        app.on_key(KeyEvent::new(KeyCode::Backspace, KeyModifiers::NONE));
    }
    type_str(&mut app, "/");
    assert!(!app.slash_suggestions().is_empty());
}

#[test]
fn longest_common_prefix_stops_at_char_boundaries() {
    assert_eq!(longest_common_prefix(&["é1.txt", "è2.txt"]), "");
    assert_eq!(longest_common_prefix(&["日本a", "日本b"]), "日本");
    assert_eq!(longest_common_prefix(&["日本", "日本語"]), "日本");
    assert_eq!(longest_common_prefix(&["abc"]), "abc");
    assert_eq!(longest_common_prefix(&[]), "");
}
