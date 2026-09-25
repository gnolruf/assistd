use tokio::sync::oneshot;

use super::*;
use crate::{Config, PresenceManager, PresenceState};

#[test]
fn fd_exhaustion_predicate_matches_only_emfile_and_enfile() {
    let cases = [
        ("EMFILE", io::Error::from_raw_os_error(libc::EMFILE), true),
        ("ENFILE", io::Error::from_raw_os_error(libc::ENFILE), true),
        (
            "ECONNRESET",
            io::Error::from_raw_os_error(libc::ECONNRESET),
            false,
        ),
        (
            "ECONNABORTED",
            io::Error::from_raw_os_error(libc::ECONNABORTED),
            false,
        ),
        ("no errno", io::Error::other("synthetic"), false),
    ];
    for (label, err, expected) in cases {
        assert_eq!(is_fd_exhaustion(&err), expected, "{label}");
    }
}

fn test_state() -> Arc<AppState> {
    state_with_backend_and_grace(Arc::new(assistd_llm::EchoBackend::new()), 5)
}

fn state_with_backend_and_grace(
    backend: Arc<dyn assistd_llm::LlmBackend>,
    grace_secs: u64,
) -> Arc<AppState> {
    let mut config = Config::default();
    config.daemon.shutdown_grace_secs = grace_secs;
    Arc::new(AppState::new(
        config,
        backend,
        PresenceManager::stub(PresenceState::Active),
        Arc::new(assistd_tools::ToolRegistry::default()),
        Arc::new(assistd_voice::NoVoiceInput::new()),
        Arc::new(assistd_voice::NoContinuousListener::new()),
        assistd_voice::VoiceOutputController::new(Arc::new(assistd_voice::NoVoiceOutput), true),
    ))
}

/// A `serve_at` task on a socket in a fresh temp dir.
struct TestServer {
    _dir: tempfile::TempDir,
    path: PathBuf,
    shutdown: oneshot::Sender<()>,
    task: tokio::task::JoinHandle<()>,
}

impl TestServer {
    async fn start(state: Arc<AppState>) -> Self {
        Self::start_with(state, |_| {}).await
    }

    /// `prepare` runs against the socket path before the server binds.
    async fn start_with(state: Arc<AppState>, prepare: impl FnOnce(&Path)) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("assistd.sock");
        prepare(&path);
        let (shutdown, rx) = oneshot::channel::<()>();
        let server_path = path.clone();
        let task = tokio::spawn(async move {
            serve_at(&server_path, state, async {
                let _ = rx.await;
            })
            .await
            .unwrap();
        });
        wait_for_listener(&path).await;
        Self {
            _dir: dir,
            path,
            shutdown,
            task,
        }
    }

    async fn stop(self) {
        self.shutdown.send(()).unwrap();
        self.task.await.unwrap();
    }
}

async fn with_server<F, Fut, R>(state: Arc<AppState>, body: F) -> R
where
    F: FnOnce(PathBuf) -> Fut,
    Fut: std::future::Future<Output = R>,
{
    let server = TestServer::start(state).await;
    let out = body(server.path.clone()).await;
    server.stop().await;
    out
}

async fn wait_for_listener(path: &Path) {
    for _ in 0..200 {
        if UnixStream::connect(path).await.is_ok() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("listener at {} did not become ready", path.display());
}

async fn read_event(reader: &mut BufReader<OwnedReadHalf>) -> Option<Event> {
    let mut line = String::new();
    let n = reader.read_line(&mut line).await.unwrap();
    (n > 0).then(|| serde_json::from_str(line.trim()).unwrap())
}

async fn read_until_terminal(reader: &mut BufReader<OwnedReadHalf>) -> Vec<Event> {
    let mut events = Vec::new();
    while let Some(event) = read_event(reader).await {
        let terminal = event.is_terminal();
        events.push(event);
        if terminal {
            break;
        }
    }
    events
}

/// Write `lines` as newline-delimited frames, optionally close the write
/// side, and return both halves.
async fn open_connection(
    path: &Path,
    lines: &[&str],
    close_write: bool,
) -> (OwnedWriteHalf, BufReader<OwnedReadHalf>) {
    let stream = UnixStream::connect(path).await.unwrap();
    let (read, mut write) = stream.into_split();
    for line in lines {
        write.write_all(line.as_bytes()).await.unwrap();
        write.write_all(b"\n").await.unwrap();
    }
    if close_write {
        write.shutdown().await.unwrap();
    } else {
        write.flush().await.unwrap();
    }
    (write, BufReader::new(read))
}

async fn send_request_collect_events(path: &Path, body: &str) -> Vec<Event> {
    let (_write, mut reader) = open_connection(path, &[body], true).await;
    read_until_terminal(&mut reader).await
}

/// Wait until `n` subscribers are attached to the event bus, so events
/// published afterwards are guaranteed to reach them.
async fn wait_for_subscribers(state: &AppState, n: usize) {
    tokio::time::timeout(Duration::from_secs(5), async {
        while state.runtime.events_bus().receiver_count() < n {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("subscriber never attached to the event bus");
}

fn delta(id: &str, text: &str) -> Event {
    Event::Delta {
        id: id.into(),
        text: text.into(),
    }
}

fn done(id: &str) -> Event {
    Event::Done { id: id.into() }
}

#[tokio::test]
async fn echoes_query_text_as_delta_then_done() {
    let server = TestServer::start(test_state()).await;
    let events = send_request_collect_events(
        &server.path,
        r#"{"type":"query","id":"req-1","text":"ping"}"#,
    )
    .await;
    assert_eq!(events, [delta("req-1", "ping"), done("req-1")]);

    let path = server.path.clone();
    server.stop().await;
    assert!(!path.exists(), "socket file should be removed on shutdown");
}

#[tokio::test]
async fn concurrent_connections_sharing_a_request_id_stay_isolated() {
    with_server(test_state(), |path| async move {
        let mut handles = Vec::new();
        for i in 0..16 {
            let p = path.clone();
            handles.push(tokio::spawn(async move {
                let text = format!("msg-{i}");
                let body = format!(r#"{{"type":"query","id":"shared","text":"{text}"}}"#);
                let events = send_request_collect_events(&p, &body).await;
                assert_eq!(events, [delta("shared", &text), done("shared")], "conn {i}");
            }));
        }
        for h in handles {
            h.await.unwrap();
        }
    })
    .await;
}

#[tokio::test]
async fn oversize_request_is_rejected_without_oom() {
    with_server(test_state(), |path| async move {
        let stream = UnixStream::connect(&path).await.unwrap();
        let (read, mut write) = stream.into_split();

        let chunk = vec![b'a'; 1024 * 1024];
        let mut remaining = MAX_REQUEST_BYTES as usize + 1;
        while remaining > 0 {
            let n = remaining.min(chunk.len());
            if write.write_all(&chunk[..n]).await.is_err() {
                break;
            }
            remaining -= n;
        }
        let _ = write.shutdown().await;

        let event = read_event(&mut BufReader::new(read))
            .await
            .expect("expected an Error event before EOF");
        assert_eq!(
            event,
            Event::Error {
                id: String::new(),
                message: format!("request exceeded {MAX_REQUEST_BYTES}-byte limit"),
            }
        );
    })
    .await;
}

#[tokio::test]
async fn malformed_request_returns_error_event() {
    with_server(test_state(), |path| async move {
        let events = send_request_collect_events(&path, "not json").await;
        assert!(
            matches!(
                events.as_slice(),
                [Event::Error { id, message }]
                    if id.is_empty() && message.starts_with("invalid request: ")
            ),
            "{events:?}"
        );
    })
    .await;
}

#[tokio::test]
async fn second_serve_refuses_to_clobber_live_socket() {
    let server = TestServer::start(test_state()).await;

    let err = serve_at(&server.path, test_state(), std::future::pending::<()>())
        .await
        .expect_err("second serve_at must fail while first is alive");
    assert!(
        matches!(&err, SocketError::AlreadyRunning { path } if *path == server.path),
        "{err:?}"
    );
    assert!(
        server.path.exists(),
        "live socket must remain after a refused second-bind attempt"
    );

    let events =
        send_request_collect_events(&server.path, r#"{"type":"query","id":"q","text":"ok"}"#).await;
    assert_eq!(events.last(), Some(&done("q")));

    server.stop().await;
}

#[tokio::test]
async fn removes_stale_socket_file_on_bind() {
    let server = TestServer::start_with(test_state(), |path| {
        std::fs::write(path, b"stale").unwrap();
    })
    .await;

    let events = send_request_collect_events(
        &server.path,
        r#"{"type":"query","id":"req-ok","text":"ok"}"#,
    )
    .await;
    assert_eq!(events.last(), Some(&done("req-ok")));

    server.stop().await;
}

/// Backend that emits N deltas with a fixed pause between each, then
/// Done.
struct SlowBackend {
    deltas: usize,
    pause: std::time::Duration,
}

#[async_trait::async_trait]
impl assistd_llm::LlmBackend for SlowBackend {
    async fn generate(
        &self,
        _prompt: String,
        tx: tokio::sync::mpsc::Sender<assistd_llm::LlmEvent>,
    ) -> assistd_llm::LlmResult<()> {
        for i in 0..self.deltas {
            let _ = tx
                .send(assistd_llm::LlmEvent::Delta {
                    text: format!("d{i}"),
                })
                .await;
            tokio::time::sleep(self.pause).await;
        }
        let _ = tx.send(assistd_llm::LlmEvent::Done).await;
        Ok(())
    }

    async fn push_user(
        &self,
        _text: String,
        _attachments: Vec<assistd_tools::Attachment>,
    ) -> assistd_llm::LlmResult<()> {
        Ok(())
    }

    async fn push_tool_results(
        &self,
        _results: Vec<assistd_llm::ToolResultPayload>,
    ) -> assistd_llm::LlmResult<()> {
        Ok(())
    }

    async fn step(
        &self,
        _tools: Vec<serde_json::Value>,
        tx: tokio::sync::mpsc::Sender<assistd_llm::LlmEvent>,
    ) -> assistd_llm::LlmResult<assistd_llm::StepOutcome> {
        for i in 0..self.deltas {
            let _ = tx
                .send(assistd_llm::LlmEvent::Delta {
                    text: format!("d{i}"),
                })
                .await;
            tokio::time::sleep(self.pause).await;
        }
        Ok(assistd_llm::StepOutcome::Final)
    }
}

/// Backend that emits a single delta then blocks indefinitely on an
/// un-awoken channel.
struct StuckBackend;

#[async_trait::async_trait]
impl assistd_llm::LlmBackend for StuckBackend {
    async fn generate(
        &self,
        _prompt: String,
        tx: tokio::sync::mpsc::Sender<assistd_llm::LlmEvent>,
    ) -> assistd_llm::LlmResult<()> {
        let _ = tx
            .send(assistd_llm::LlmEvent::Delta {
                text: "stuck".into(),
            })
            .await;
        std::future::pending::<()>().await;
        Ok(())
    }

    async fn push_user(
        &self,
        _text: String,
        _attachments: Vec<assistd_tools::Attachment>,
    ) -> assistd_llm::LlmResult<()> {
        Ok(())
    }

    async fn push_tool_results(
        &self,
        _results: Vec<assistd_llm::ToolResultPayload>,
    ) -> assistd_llm::LlmResult<()> {
        Ok(())
    }

    async fn step(
        &self,
        _tools: Vec<serde_json::Value>,
        tx: tokio::sync::mpsc::Sender<assistd_llm::LlmEvent>,
    ) -> assistd_llm::LlmResult<assistd_llm::StepOutcome> {
        let _ = tx
            .send(assistd_llm::LlmEvent::Delta {
                text: "stuck".into(),
            })
            .await;
        std::future::pending::<()>().await;
        Ok(assistd_llm::StepOutcome::Final)
    }
}

#[tokio::test]
async fn graceful_shutdown_waits_for_in_flight_stream() {
    let state = state_with_backend_and_grace(
        Arc::new(SlowBackend {
            deltas: 3,
            pause: Duration::from_millis(50),
        }),
        5,
    );
    let server = TestServer::start(state).await;
    let (_write, mut reader) = open_connection(
        &server.path,
        &[r#"{"type":"query","id":"s1","text":"go"}"#],
        true,
    )
    .await;

    let first = read_event(&mut reader).await.expect("first Delta");
    assert_eq!(first, delta("s1", "d0"));

    let stopping = tokio::spawn(server.stop());
    let rest = read_until_terminal(&mut reader).await;
    assert_eq!(rest, [delta("s1", "d1"), delta("s1", "d2"), done("s1")]);

    stopping.await.unwrap();
}

#[tokio::test]
async fn graceful_shutdown_aborts_after_grace_timeout() {
    let state = state_with_backend_and_grace(Arc::new(StuckBackend), 0);
    let server = TestServer::start(state).await;
    let (_write, mut reader) = open_connection(
        &server.path,
        &[r#"{"type":"query","id":"s2","text":"hang"}"#],
        true,
    )
    .await;
    read_event(&mut reader)
        .await
        .expect("expected first Delta from StuckBackend");

    tokio::time::timeout(Duration::from_secs(2), server.stop())
        .await
        .expect("server did not exit within 2s of shutdown");
}

#[tokio::test]
async fn unmatched_mid_stream_confirm_response_does_not_crash() {
    with_server(test_state(), |path| async move {
        let (_write, mut reader) = open_connection(
            &path,
            &[
                r#"{"type":"query","id":"q1","text":"ping"}"#,
                r#"{"type":"confirm_response","id":"cr-x","confirm_id":"missing","allow":false}"#,
            ],
            true,
        )
        .await;
        let events = read_until_terminal(&mut reader).await;
        assert_eq!(events, [delta("q1", "ping"), done("q1")]);
    })
    .await;
}

#[tokio::test]
async fn concurrent_set_presence_to_sleeping_is_idempotent_under_fanout() {
    with_server(test_state(), |path| async move {
        let mut handles = Vec::new();
        for i in 0..32 {
            let p = path.clone();
            handles.push(tokio::spawn(async move {
                let id = format!("sp-{i}");
                let body = format!(r#"{{"type":"set_presence","id":"{id}","target":"sleeping"}}"#);
                let events = send_request_collect_events(&p, &body).await;
                assert_eq!(
                    events,
                    [
                        Event::Presence {
                            id: id.clone(),
                            state: PresenceState::Sleeping
                        },
                        done(&id)
                    ],
                    "conn {i}"
                );
            }));
        }
        for h in handles {
            h.await.unwrap();
        }
    })
    .await;
}

#[tokio::test]
async fn mixed_query_and_set_presence_does_not_drop_events() {
    let state = state_with_backend_and_grace(
        Arc::new(SlowBackend {
            deltas: 3,
            pause: Duration::from_millis(50),
        }),
        5,
    );
    with_server(state, |path| async move {
        let mut handles = Vec::new();
        for i in 0..16 {
            let p = path.clone();
            handles.push(tokio::spawn(async move {
                let id = format!("mix-{i}");
                let body = if i % 2 == 0 {
                    format!(r#"{{"type":"query","id":"{id}","text":"q{i}"}}"#)
                } else {
                    format!(r#"{{"type":"set_presence","id":"{id}","target":"active"}}"#)
                };
                let events = send_request_collect_events(&p, &body).await;
                assert_eq!(events.last(), Some(&done(&id)), "conn {i}: {events:?}");
                assert!(
                    !events.iter().any(|e| matches!(e, Event::Error { .. })),
                    "conn {i}: unexpected Error in {events:?}"
                );
            }));
        }
        for h in handles {
            h.await.unwrap();
        }
    })
    .await;
}

/// Collect subscriber events until the stream goes quiet for `idle`.
/// Stops early past a cap so a feedback loop fails instead of hanging.
async fn drain_subscribe(reader: &mut BufReader<OwnedReadHalf>, idle: Duration) -> Vec<Event> {
    const CAP: usize = 64;
    let mut out = Vec::new();
    while out.len() < CAP {
        match tokio::time::timeout(idle, read_event(reader)).await {
            Ok(Some(ev)) => out.push(ev),
            Ok(None) | Err(_) => break,
        }
    }
    out
}

#[tokio::test]
async fn subscribe_receives_each_query_event_from_other_client_once() {
    let state = test_state();
    with_server(state.clone(), |path| async move {
        let (_write_a, mut read_a) = open_connection(
            &path,
            &[r#"{"type":"subscribe","id":"sub-1","filter":{"kinds":[]}}"#],
            false,
        )
        .await;
        wait_for_subscribers(&state, 1).await;

        let b_events =
            send_request_collect_events(&path, r#"{"type":"query","id":"q-1","text":"hello"}"#)
                .await;
        assert_eq!(b_events, [delta("q-1", "hello"), done("q-1")]);

        let a_events = drain_subscribe(&mut read_a, Duration::from_millis(200)).await;
        let count = |want: &Event| a_events.iter().filter(|e| *e == want).count();
        assert_eq!(count(&delta("q-1", "hello")), 1, "{a_events:?}");
        assert_eq!(count(&done("q-1")), 1, "{a_events:?}");
        assert!(
            a_events.contains(&Event::LastDelta {
                id: "q-1".into(),
                text: "hello".into()
            }),
            "subscriber missed LastDelta: {a_events:?}"
        );
    })
    .await;
}

#[tokio::test]
async fn subscribe_filter_rejects_unmatched_kinds() {
    let state = test_state();
    with_server(state.clone(), |path| async move {
        let (_write_a, mut read_a) = open_connection(
            &path,
            &[r#"{"type":"subscribe","id":"sub-2","filter":{"kinds":["presence"]}}"#],
            false,
        )
        .await;
        wait_for_subscribers(&state, 1).await;

        send_request_collect_events(&path, r#"{"type":"query","id":"q-2","text":"hi"}"#).await;
        send_request_collect_events(
            &path,
            r#"{"type":"set_presence","id":"sp-1","target":"sleeping"}"#,
        )
        .await;

        let a_events = drain_subscribe(&mut read_a, Duration::from_millis(200)).await;
        assert_eq!(
            a_events,
            [Event::Presence {
                id: "sp-1".into(),
                state: PresenceState::Sleeping
            }]
        );
    })
    .await;
}

#[tokio::test]
async fn client_sees_eof_promptly_after_terminal_event() {
    with_server(test_state(), |path| async move {
        let (_write, mut reader) = open_connection(
            &path,
            &[r#"{"type":"query","id":"eof-1","text":"ping"}"#],
            true,
        )
        .await;
        let events = read_until_terminal(&mut reader).await;
        assert_eq!(events.last(), Some(&done("eof-1")));

        let next = tokio::time::timeout(Duration::from_secs(1), read_event(&mut reader))
            .await
            .expect("connection stayed open after terminal event");
        assert_eq!(next, None);
    })
    .await;
}

#[tokio::test]
async fn shutdown_closes_idle_subscriber_without_waiting_out_grace() {
    let state = test_state();
    let server = TestServer::start(state.clone()).await;
    let (_write, mut reader) = open_connection(
        &server.path,
        &[r#"{"type":"subscribe","id":"sub-drain","filter":{"kinds":[]}}"#],
        false,
    )
    .await;
    wait_for_subscribers(&state, 1).await;

    tokio::time::timeout(Duration::from_secs(2), server.stop())
        .await
        .expect("server held by idle subscriber past drain");

    let next = tokio::time::timeout(Duration::from_secs(1), read_event(&mut reader))
        .await
        .expect("subscriber not released on shutdown");
    assert_eq!(next, None);
}

/// Backend whose first step asks for the `gated` tool and whose second
/// step ends the turn.
struct GatedToolBackend {
    stepped: std::sync::atomic::AtomicBool,
}

#[async_trait::async_trait]
impl assistd_llm::LlmBackend for GatedToolBackend {
    async fn generate(
        &self,
        _prompt: String,
        tx: tokio::sync::mpsc::Sender<assistd_llm::LlmEvent>,
    ) -> assistd_llm::LlmResult<()> {
        let _ = tx.send(assistd_llm::LlmEvent::Done).await;
        Ok(())
    }

    async fn push_user(
        &self,
        _text: String,
        _attachments: Vec<assistd_tools::Attachment>,
    ) -> assistd_llm::LlmResult<()> {
        Ok(())
    }

    async fn push_tool_results(
        &self,
        _results: Vec<assistd_llm::ToolResultPayload>,
    ) -> assistd_llm::LlmResult<()> {
        Ok(())
    }

    async fn step(
        &self,
        _tools: Vec<serde_json::Value>,
        _tx: tokio::sync::mpsc::Sender<assistd_llm::LlmEvent>,
    ) -> assistd_llm::LlmResult<assistd_llm::StepOutcome> {
        if self.stepped.swap(true, std::sync::atomic::Ordering::SeqCst) {
            return Ok(assistd_llm::StepOutcome::Final);
        }
        Ok(assistd_llm::StepOutcome::ToolCalls(vec![
            assistd_llm::ToolCall {
                id: "call-1".into(),
                name: "gated".into(),
                arguments: serde_json::json!({}),
            },
        ]))
    }
}

/// Tool that runs the production IPC gate and reports its verdict.
struct GatedTool;

#[async_trait::async_trait]
impl assistd_tools::Tool for GatedTool {
    fn name(&self) -> &str {
        "gated"
    }

    fn description(&self) -> &str {
        "asks for confirmation"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({"type": "object"})
    }

    async fn invoke(
        &self,
        _args: serde_json::Value,
    ) -> Result<serde_json::Value, assistd_tools::ToolError> {
        let approved = assistd_tools::ConfirmationGate::confirm(
            &assistd_tools::IpcConfirmationGate,
            assistd_tools::ConfirmationRequest {
                tool: "bash".into(),
                script: "rm -rf /tmp/x".into(),
                matched_pattern: "rm -rf".into(),
                always_allow: Vec::new(),
            },
        )
        .await;
        let approved = approved != assistd_tools::Approval::Deny;
        Ok(serde_json::json!({
            "output": if approved { "approved" } else { "denied" },
            "exit_code": 0,
        }))
    }
}

fn gated_tool_state() -> Arc<AppState> {
    let mut tools = assistd_tools::ToolRegistry::new();
    tools.register(GatedTool);
    Arc::new(AppState::new(
        Config::default(),
        Arc::new(GatedToolBackend {
            stepped: std::sync::atomic::AtomicBool::new(false),
        }),
        PresenceManager::stub(PresenceState::Active),
        Arc::new(tools),
        Arc::new(assistd_voice::NoVoiceInput::new()),
        Arc::new(assistd_voice::NoContinuousListener::new()),
        assistd_voice::VoiceOutputController::new(Arc::new(assistd_voice::NoVoiceOutput), true),
    ))
}

fn tool_output(ev: &Event) -> Option<&str> {
    match ev {
        Event::ToolResult { result, .. } => result.get("output").and_then(|v| v.as_str()),
        _ => None,
    }
}

#[tokio::test]
async fn dialog_client_answer_reaches_gate_in_spawned_agent_turn() {
    with_server(gated_tool_state(), |path| async move {
        let (mut write, mut reader) =
            open_connection(&path, &[r#"{"type":"query","id":"q1","text":"go"}"#], false).await;

        let mut outputs = Vec::new();
        let run = async {
            while let Some(ev) = read_event(&mut reader).await {
                if let Event::ConfirmRequest { id, confirm_id, .. } = &ev {
                    assert_eq!(id, "q1");
                    let answer = format!(
                        "{{\"type\":\"confirm_response\",\"id\":\"cr\",\"confirm_id\":\"{confirm_id}\",\"allow\":true}}\n"
                    );
                    write.write_all(answer.as_bytes()).await.unwrap();
                }
                if let Some(out) = tool_output(&ev) {
                    outputs.push(out.to_string());
                }
                if ev.is_terminal() {
                    break;
                }
            }
        };
        tokio::time::timeout(Duration::from_secs(10), run)
            .await
            .expect("turn must finish once the prompt is answered");
        assert_eq!(outputs, ["approved"]);
    })
    .await;
}

#[tokio::test]
async fn one_shot_client_eof_denies_prompt_without_waiting() {
    with_server(gated_tool_state(), |path| async move {
        let (_write, mut reader) =
            open_connection(&path, &[r#"{"type":"query","id":"q1","text":"go"}"#], true).await;
        let events =
            tokio::time::timeout(Duration::from_secs(10), read_until_terminal(&mut reader))
                .await
                .expect("a client that closed its write side must not park the turn");
        let outputs: Vec<&str> = events.iter().filter_map(tool_output).collect();
        assert_eq!(outputs, ["denied"]);
    })
    .await;
}

/// Backend that streams 1 KiB deltas until its channel closes.
struct FloodBackend;

impl FloodBackend {
    async fn flood(tx: &tokio::sync::mpsc::Sender<assistd_llm::LlmEvent>) {
        let text = "x".repeat(1024);
        while tx
            .send(assistd_llm::LlmEvent::Delta { text: text.clone() })
            .await
            .is_ok()
        {}
    }
}

#[async_trait::async_trait]
impl assistd_llm::LlmBackend for FloodBackend {
    async fn generate(
        &self,
        _prompt: String,
        tx: tokio::sync::mpsc::Sender<assistd_llm::LlmEvent>,
    ) -> assistd_llm::LlmResult<()> {
        Self::flood(&tx).await;
        Ok(())
    }

    async fn push_user(
        &self,
        _text: String,
        _attachments: Vec<assistd_tools::Attachment>,
    ) -> assistd_llm::LlmResult<()> {
        Ok(())
    }

    async fn push_tool_results(
        &self,
        _results: Vec<assistd_llm::ToolResultPayload>,
    ) -> assistd_llm::LlmResult<()> {
        Ok(())
    }

    async fn step(
        &self,
        _tools: Vec<serde_json::Value>,
        tx: tokio::sync::mpsc::Sender<assistd_llm::LlmEvent>,
    ) -> assistd_llm::LlmResult<assistd_llm::StepOutcome> {
        Self::flood(&tx).await;
        Ok(assistd_llm::StepOutcome::Final)
    }
}

/// Sends a query and returns the connection after its first event has
/// arrived, so the caller controls when the client goes away.
async fn open_query_and_read_first_event(
    path: &Path,
    id: &str,
) -> (Event, BufReader<OwnedReadHalf>, OwnedWriteHalf) {
    let body = format!(r#"{{"type":"query","id":"{id}","text":"go"}}"#);
    let (write, mut reader) = open_connection(path, &[&body], false).await;
    let event = read_event(&mut reader)
        .await
        .unwrap_or_else(|| panic!("expected a first event for {id}"));
    (event, reader, write)
}

#[tokio::test]
async fn client_disconnect_mid_turn_releases_the_turn_lock() {
    with_server(
        state_with_backend_and_grace(Arc::new(FloodBackend), 0),
        |path| async move {
            let (first, reader, write) = open_query_and_read_first_event(&path, "gone").await;
            assert!(matches!(first, Event::Delta { .. }), "got {first:?}");
            drop(reader);
            drop(write);

            let (next, _reader, _write) = tokio::time::timeout(
                Duration::from_secs(5),
                open_query_and_read_first_event(&path, "after"),
            )
            .await
            .expect("second query queued behind a turn whose client had disconnected");
            assert!(matches!(next, Event::Delta { .. }), "got {next:?}");
        },
    )
    .await;
}

#[tokio::test(start_paused = true)]
async fn write_event_times_out_when_the_client_stops_reading() {
    let (server, _client_never_reads) = UnixStream::pair().unwrap();
    let (_server_read, mut write_half) = server.into_split();
    let event = Event::Delta {
        id: "stalled".into(),
        text: "x".repeat(1024),
    };

    let err = tokio::time::timeout(EVENT_WRITE_TIMEOUT * 4, async {
        loop {
            if let Err(e) = write_event(&mut write_half, &event).await {
                break e;
            }
        }
    })
    .await
    .expect("write to a stalled client never failed");
    assert!(matches!(err, SocketError::WriteTimeout(_)), "got {err:?}");
}
