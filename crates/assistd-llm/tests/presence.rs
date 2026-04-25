//! End-to-end tests for the presence state machine.
//!
//! These exercise `PresenceManager` against a real `fake_llama_server` child
//! process. Gated behind the `test-support` feature because they need the
//! fake binary. Run with:
//!     cargo test -p assistd-llm --features test-support --test presence

#![cfg(feature = "test-support")]

use std::sync::Arc;
use std::sync::Once;
use std::time::Duration;

use assistd_config::{LlamaServerConfig, ModelConfig};
use assistd_core::{
    AppState, Config, NoContinuousListener, NoVoiceInput, NoVoiceOutput, PresenceManager,
    PresenceState, ToolRegistry, VoiceOutputController,
};
use assistd_ipc::{Event, Request};
use assistd_llm::{EchoBackend, LlmBackend, LlmEvent};
use async_trait::async_trait;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, UnixStream};
use tokio::sync::{Mutex, mpsc, oneshot, watch};

const FAKE_BIN: &str = env!("CARGO_BIN_EXE_fake_llama_server");

// Test serialization guard: the fake binary reads FAKE_LLAMA_MODE from the
// environment, so tests that set it must not run concurrently. Held for the
// duration of each test.
static MODE_LOCK: Mutex<()> = Mutex::const_new(());

fn init_tracing() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let _ = tracing_subscriber::fmt()
            .with_env_filter(
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
            )
            .with_test_writer()
            .try_init();
    });
}

fn set_mode(mode: &str) {
    // SAFETY: MODE_LOCK serializes access so this is single-threaded.
    unsafe { std::env::set_var("FAKE_LLAMA_MODE", mode) };
}

async fn grab_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    drop(listener);
    port
}

fn server_spec(port: u16) -> LlamaServerConfig {
    LlamaServerConfig {
        binary_path: FAKE_BIN.to_string(),
        host: "127.0.0.1".to_string(),
        port,
        gpu_layers: 0,
        ready_timeout_secs: 60,
        alias: None,
        override_tensor: None,
        flash_attn: None,
        cache_type_k: None,
        cache_type_v: None,
        threads: None,
        batch_size: None,
        ubatch_size: None,
        n_cpu_moe: None,
        cache_ram_mib: None,
        mlock: None,
        mmproj_offload: None,
    }
}

fn model_spec() -> ModelConfig {
    ModelConfig {
        name: "test/fake-model-GGUF:Q4_K_M".to_string(),
        context_length: 2048,
    }
}

async fn new_active_manager(port: u16) -> (Arc<PresenceManager>, watch::Sender<bool>) {
    set_mode("normal");
    let (tx, rx) = watch::channel(false);
    let m = PresenceManager::new_active(server_spec(port), model_spec(), rx)
        .await
        .expect("cold-start wake failed");
    (m, tx)
}

fn pid_alive(pid: u32) -> bool {
    // `kill(pid, 0)` returns 0 if the process exists and we may signal it.
    unsafe { libc::kill(pid as i32, 0) == 0 }
}

async fn wait_for_pid_gone(pid: u32, timeout: Duration) -> bool {
    let deadline = std::time::Instant::now() + timeout;
    while std::time::Instant::now() < deadline {
        if !pid_alive(pid) {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    false
}

async fn get_counters(port: u16) -> (u32, u32, Option<String>) {
    let url = format!("http://127.0.0.1:{port}/debug/counters");
    let body = reqwest::get(&url)
        .await
        .expect("GET /debug/counters")
        .text()
        .await
        .expect("counters body");
    let v: serde_json::Value = serde_json::from_str(&body).expect("counters json");
    let load = v["load_count"].as_u64().expect("load_count") as u32;
    let unload = v["unload_count"].as_u64().expect("unload_count") as u32;
    let loaded = v["loaded_model"].as_str().map(|s| s.to_string());
    (load, unload, loaded)
}

#[tokio::test]
async fn cold_start_puts_manager_in_active_and_loads_model() {
    let _g = MODE_LOCK.lock().await;
    init_tracing();
    let port = grab_port().await;
    let (m, _shutdown) = new_active_manager(port).await;

    assert_eq!(m.state(), PresenceState::Active);
    let pid = m.llama_pid().await.expect("child running after wake");
    assert!(pid_alive(pid));

    // Cold start loads the model via POST /models/load.
    let (load_count, unload_count, loaded) = get_counters(port).await;
    assert_eq!(load_count, 1, "cold start should call /models/load once");
    assert_eq!(unload_count, 0);
    assert_eq!(loaded.as_deref(), Some(model_spec().name.as_str()));

    m.sleep().await.unwrap();
}

#[tokio::test]
async fn sleep_stops_supervisor_and_kills_child() {
    let _g = MODE_LOCK.lock().await;
    init_tracing();
    let port = grab_port().await;
    let (m, _shutdown) = new_active_manager(port).await;
    let pid = m.llama_pid().await.expect("child running");
    assert!(pid_alive(pid));

    m.sleep().await.expect("sleep should succeed");
    assert_eq!(m.state(), PresenceState::Sleeping);
    assert!(
        m.llama_pid().await.is_none(),
        "llama handle should be taken"
    );

    assert!(
        wait_for_pid_gone(pid, Duration::from_secs(5)).await,
        "child pid {pid} should be reaped after sleep"
    );
}

#[tokio::test]
async fn sleep_from_sleeping_is_idempotent() {
    let _g = MODE_LOCK.lock().await;
    init_tracing();
    let port = grab_port().await;
    let (m, _shutdown) = new_active_manager(port).await;

    m.sleep().await.unwrap();
    assert_eq!(m.state(), PresenceState::Sleeping);
    // Second sleep must not panic, error, or spawn anything.
    m.sleep().await.expect("second sleep should be a no-op");
    assert_eq!(m.state(), PresenceState::Sleeping);
    assert!(m.llama_pid().await.is_none());
}

#[tokio::test]
async fn drowse_calls_unload_and_keeps_process_alive() {
    let _g = MODE_LOCK.lock().await;
    init_tracing();
    let port = grab_port().await;
    let (m, _shutdown) = new_active_manager(port).await;
    let pid_before = m.llama_pid().await.expect("child running");
    let (_, unload_before, _) = get_counters(port).await;

    m.drowse().await.expect("drowse should succeed");
    assert_eq!(m.state(), PresenceState::Drowsy);

    // Same PID, still alive.
    let pid_after = m.llama_pid().await.expect("child still running");
    assert_eq!(pid_before, pid_after, "drowse must not respawn the process");
    assert!(pid_alive(pid_after));

    // /models/unload was called exactly once more, and server reports no
    // loaded model.
    let (_, unload_after, loaded) = get_counters(port).await;
    assert_eq!(unload_after, unload_before + 1);
    assert!(
        loaded.is_none(),
        "expected no loaded model after unload, got {loaded:?}"
    );

    m.sleep().await.unwrap();
}

#[tokio::test]
async fn drowse_from_sleeping_errors_and_preserves_state() {
    let _g = MODE_LOCK.lock().await;
    init_tracing();
    let port = grab_port().await;
    let (m, _shutdown) = new_active_manager(port).await;
    m.sleep().await.unwrap();

    let err = m
        .drowse()
        .await
        .expect_err("drowse from Sleeping must error");
    assert!(
        err.to_string().to_lowercase().contains("sleeping"),
        "error should mention Sleeping: {err}"
    );
    assert_eq!(m.state(), PresenceState::Sleeping);
}

#[tokio::test]
async fn wake_from_drowsy_reuses_process_and_only_loads_model() {
    let _g = MODE_LOCK.lock().await;
    init_tracing();
    let port = grab_port().await;
    let (m, _shutdown) = new_active_manager(port).await;
    let pid_initial = m.llama_pid().await.expect("child running");
    let (load_initial, _, _) = get_counters(port).await;

    m.drowse().await.unwrap();
    assert_eq!(m.state(), PresenceState::Drowsy);

    m.wake().await.expect("wake from Drowsy should succeed");
    assert_eq!(m.state(), PresenceState::Active);

    // Same PID — no respawn.
    let pid_after = m.llama_pid().await.expect("child still running");
    assert_eq!(
        pid_initial, pid_after,
        "wake from Drowsy must not respawn the process"
    );

    // /models/load was called once more (no extra spawn means the supervisor
    // wasn't restarted, so the ready-then-load sequence ran only a second time).
    let (load_after, _, loaded) = get_counters(port).await;
    assert_eq!(load_after, load_initial + 1);
    assert_eq!(loaded.as_deref(), Some(model_spec().name.as_str()));

    m.sleep().await.unwrap();
}

#[tokio::test]
async fn wake_from_sleeping_cold_starts_and_returns_active() {
    let _g = MODE_LOCK.lock().await;
    init_tracing();
    let port = grab_port().await;
    let (m, _shutdown) = new_active_manager(port).await;
    let pid_before = m.llama_pid().await.expect("child running");

    m.sleep().await.unwrap();
    assert!(wait_for_pid_gone(pid_before, Duration::from_secs(5)).await);

    m.wake().await.expect("wake from Sleeping should succeed");
    assert_eq!(m.state(), PresenceState::Active);
    let pid_after = m.llama_pid().await.expect("child running after wake");
    assert_ne!(
        pid_before, pid_after,
        "cold-start wake must spawn a new child"
    );
    assert!(pid_alive(pid_after));

    m.sleep().await.unwrap();
}

#[tokio::test]
async fn query_during_sleeping_triggers_auto_wake() {
    let _g = MODE_LOCK.lock().await;
    init_tracing();
    let port = grab_port().await;
    let (m, _shutdown) = new_active_manager(port).await;

    // Put the manager to Sleeping before serving any queries.
    m.sleep().await.unwrap();
    assert_eq!(m.state(), PresenceState::Sleeping);

    // AppState uses EchoBackend — we're testing the auto-wake hook in
    // handle_query, not the llama chat path.
    let state = Arc::new(AppState::new(
        Config::default(),
        Arc::new(EchoBackend::new()),
        m.clone(),
        Arc::new(ToolRegistry::default()),
        Arc::new(NoVoiceInput::new()),
        Arc::new(NoContinuousListener::new()),
        VoiceOutputController::new(Arc::new(NoVoiceOutput), true),
    ));

    let dir = tempfile::tempdir().unwrap();
    let sock_path = dir.path().join("assistd.sock");
    let (stop_tx, stop_rx) = oneshot::channel::<()>();
    let server_path = sock_path.clone();
    let server = tokio::spawn(async move {
        assistd_core::socket::serve_at(&server_path, state, async {
            let _ = stop_rx.await;
        })
        .await
        .unwrap();
    });

    // Wait for the socket to come up.
    for _ in 0..200 {
        if UnixStream::connect(&sock_path).await.is_ok() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    let req = Request::Query {
        id: "q1".into(),
        text: "hello".into(),
    };
    let stream = UnixStream::connect(&sock_path).await.unwrap();
    let (read, mut write) = stream.into_split();
    let mut body = serde_json::to_string(&req).unwrap();
    body.push('\n');
    write.write_all(body.as_bytes()).await.unwrap();
    write.shutdown().await.unwrap();

    let mut reader = BufReader::new(read);
    let mut events = Vec::new();
    loop {
        let mut line = String::new();
        let n = reader.read_line(&mut line).await.unwrap();
        if n == 0 {
            break;
        }
        let e: Event = serde_json::from_str(line.trim()).unwrap();
        let terminal = e.is_terminal();
        events.push(e);
        if terminal {
            break;
        }
    }

    assert!(
        events
            .iter()
            .any(|e| matches!(e, Event::Delta { text, .. } if text == "hello"))
    );
    assert!(matches!(events.last(), Some(Event::Done { .. })));
    assert_eq!(
        m.state(),
        PresenceState::Active,
        "auto-wake must leave manager in Active"
    );

    let _ = stop_tx.send(());
    server.await.unwrap();
    m.sleep().await.unwrap();
}

/// Backend that emits one Delta, sleeps for a configurable duration,
/// then emits Done. Used to put a predictable gap between the first
/// visible token and the terminal Done event so we can assert that
/// `sleep()` blocks through the gap rather than killing the generator.
struct DelayBackend {
    delay: Duration,
    last_user: tokio::sync::Mutex<String>,
}

#[async_trait]
impl LlmBackend for DelayBackend {
    async fn generate(&self, prompt: String, tx: mpsc::Sender<LlmEvent>) -> anyhow::Result<()> {
        let _ = tx.send(LlmEvent::Delta { text: prompt }).await;
        tokio::time::sleep(self.delay).await;
        let _ = tx.send(LlmEvent::Done).await;
        Ok(())
    }

    async fn push_user(&self, text: String) -> anyhow::Result<()> {
        *self.last_user.lock().await = text;
        Ok(())
    }

    async fn push_tool_results(
        &self,
        _results: Vec<assistd_llm::ToolResultPayload>,
    ) -> anyhow::Result<()> {
        Ok(())
    }

    async fn step(
        &self,
        _tools: Vec<serde_json::Value>,
        tx: mpsc::Sender<LlmEvent>,
    ) -> anyhow::Result<assistd_llm::StepOutcome> {
        let text = self.last_user.lock().await.clone();
        let _ = tx.send(LlmEvent::Delta { text }).await;
        tokio::time::sleep(self.delay).await;
        Ok(assistd_llm::StepOutcome::Final)
    }
}

async fn connect_and_send(sock: &std::path::Path, req: &Request) -> Vec<Event> {
    let stream = UnixStream::connect(sock).await.unwrap();
    let (read, mut write) = stream.into_split();
    let mut body = serde_json::to_string(req).unwrap();
    body.push('\n');
    write.write_all(body.as_bytes()).await.unwrap();
    write.shutdown().await.unwrap();

    let mut reader = BufReader::new(read);
    let mut events = Vec::new();
    loop {
        let mut line = String::new();
        let n = reader.read_line(&mut line).await.unwrap();
        if n == 0 {
            break;
        }
        let e: Event = serde_json::from_str(line.trim()).unwrap();
        let terminal = e.is_terminal();
        events.push(e);
        if terminal {
            break;
        }
    }
    events
}

#[tokio::test]
async fn sleep_defers_until_inflight_query_done() {
    let _g = MODE_LOCK.lock().await;
    init_tracing();
    let port = grab_port().await;
    let (m, _shutdown) = new_active_manager(port).await;

    let state = Arc::new(AppState::new(
        Config::default(),
        Arc::new(DelayBackend {
            delay: Duration::from_millis(500),
            last_user: tokio::sync::Mutex::new(String::new()),
        }),
        m.clone(),
        Arc::new(ToolRegistry::default()),
        Arc::new(NoVoiceInput::new()),
        Arc::new(NoContinuousListener::new()),
        VoiceOutputController::new(Arc::new(NoVoiceOutput), true),
    ));

    let dir = tempfile::tempdir().unwrap();
    let sock_path = dir.path().join("assistd.sock");
    let (stop_tx, stop_rx) = oneshot::channel::<()>();
    let server_path = sock_path.clone();
    let server = tokio::spawn(async move {
        assistd_core::socket::serve_at(&server_path, state, async {
            let _ = stop_rx.await;
        })
        .await
        .unwrap();
    });

    for _ in 0..200 {
        if UnixStream::connect(&sock_path).await.is_ok() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    // Kick off the query; it'll emit a Delta ~immediately, then sleep 500ms,
    // then emit Done.
    let req = Request::Query {
        id: "q1".into(),
        text: "hello".into(),
    };
    let sock_for_client = sock_path.clone();
    let client_task = tokio::spawn(async move { connect_and_send(&sock_for_client, &req).await });

    // Give the query a head start so the request guard is taken before
    // we issue the sleep.
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Request sleep while the query is still streaming. The sleep must
    // not complete until the generator finishes.
    let m_for_sleep = m.clone();
    let sleep_started = std::time::Instant::now();
    let sleep_task = tokio::spawn(async move { m_for_sleep.sleep().await });

    // Wait for the client to drain its full event stream.
    let events = client_task.await.unwrap();
    assert!(
        events
            .iter()
            .any(|e| matches!(e, Event::Delta { text, .. } if text == "hello")),
        "expected Delta event in stream, got {events:?}"
    );
    assert!(
        matches!(events.last(), Some(Event::Done { .. })),
        "expected terminal Done, got {events:?}"
    );
    assert!(
        !events.iter().any(|e| matches!(e, Event::Error { .. })),
        "no Error events expected, got {events:?}"
    );

    // Now sleep is free to proceed; it must not have completed before
    // the generator finished.
    sleep_task.await.unwrap().expect("sleep returned Err");
    let sleep_elapsed = sleep_started.elapsed();
    assert!(
        sleep_elapsed >= Duration::from_millis(350),
        "sleep finished in {sleep_elapsed:?}, expected >=350ms (block on the 500ms DelayBackend \
         minus the 100ms head start and a little slack)"
    );
    assert_eq!(m.state(), PresenceState::Sleeping);

    let _ = stop_tx.send(());
    server.await.unwrap();
}

#[tokio::test]
async fn multiple_queries_during_wake_complete_in_order() {
    let _g = MODE_LOCK.lock().await;
    init_tracing();
    let port = grab_port().await;
    let (m, _shutdown) = new_active_manager(port).await;

    let state = Arc::new(AppState::new(
        Config::default(),
        Arc::new(EchoBackend::new()),
        m.clone(),
        Arc::new(ToolRegistry::default()),
        Arc::new(NoVoiceInput::new()),
        Arc::new(NoContinuousListener::new()),
        VoiceOutputController::new(Arc::new(NoVoiceOutput), true),
    ));

    let dir = tempfile::tempdir().unwrap();
    let sock_path = dir.path().join("assistd.sock");
    let (stop_tx, stop_rx) = oneshot::channel::<()>();
    let server_path = sock_path.clone();
    let server = tokio::spawn(async move {
        assistd_core::socket::serve_at(&server_path, state, async {
            let _ = stop_rx.await;
        })
        .await
        .unwrap();
    });

    for _ in 0..200 {
        if UnixStream::connect(&sock_path).await.is_ok() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    // Put the daemon to sleep so the next queries have to wake it.
    m.sleep().await.unwrap();
    assert_eq!(m.state(), PresenceState::Sleeping);

    // Fire 5 concurrent queries with distinct ids. They race into
    // acquire_request_guard → ensure_active → transition lock; one
    // wins the wake, the rest queue on the transition mutex.
    let mut handles = Vec::new();
    for i in 0..5 {
        let id = format!("q{i}");
        let text = format!("msg{i}");
        let sock = sock_path.clone();
        let req = Request::Query {
            id: id.clone(),
            text: text.clone(),
        };
        handles.push((
            id,
            text,
            tokio::spawn(async move { connect_and_send(&sock, &req).await }),
        ));
    }

    for (id, text, h) in handles {
        let events = h.await.unwrap();
        assert!(
            !events.iter().any(|e| matches!(e, Event::Error { .. })),
            "query {id} received Error events: {events:?}"
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, Event::Delta { text: t, .. } if *t == text)),
            "query {id} missing expected Delta {text:?}: {events:?}"
        );
        assert!(
            matches!(events.last(), Some(Event::Done { id: done_id }) if *done_id == id),
            "query {id} missing terminal Done: {events:?}"
        );
    }

    assert_eq!(
        m.state(),
        PresenceState::Active,
        "wake must leave manager Active"
    );

    let _ = stop_tx.send(());
    server.await.unwrap();
    m.sleep().await.unwrap();
}
