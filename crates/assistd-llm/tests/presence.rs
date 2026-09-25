//! End-to-end tests for the presence state machine, driving
//! `PresenceManager` against a real `fake_llama_server` child process.

#![cfg(feature = "test-support")]

use std::net::Ipv4Addr;
use std::num::NonZeroU16;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Once};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use rustix::io::Errno;
use rustix::process::{Pid, test_kill_process};
use serde_json::Value;
use tempfile::TempDir;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, Lines};
use tokio::net::unix::OwnedReadHalf;
use tokio::net::{TcpListener, TcpStream, UnixStream};
use tokio::sync::{Mutex, mpsc, oneshot, watch};
use tokio::task::JoinHandle;

use assistd_config::defaults::{nz32, nz64};
use assistd_config::{LlamaServerConfig, ModelConfig, TimeoutsConfig};
use assistd_core::{
    AppState, Config, NoContinuousListener, NoVoiceInput, NoVoiceOutput, PresenceError,
    PresenceManager, PresenceState, ToolRegistry, VoiceOutputController,
};
use assistd_ipc::{Event, Request};
use assistd_llm::{EchoBackend, LlmBackend, LlmEvent, LlmResult, StepOutcome, ToolResultPayload};
use assistd_tools::Attachment;

use common::FakeLlama;

mod common;

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

async fn grab_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    drop(listener);
    port
}

fn server_spec(fake: &FakeLlama, port: u16) -> LlamaServerConfig {
    LlamaServerConfig {
        binary_path: fake.binary_path(),
        host: Ipv4Addr::LOCALHOST.into(),
        port: NonZeroU16::new(port).expect("bound port is never 0"),
        gpu_layers: 0,
        ready_timeout_secs: nz64(60),
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
        context_length: nz32(2048),
    }
}

async fn new_active_manager(
    fake: &FakeLlama,
    port: u16,
) -> (Arc<PresenceManager>, watch::Sender<bool>) {
    let (tx, rx) = watch::channel(false);
    let manager = PresenceManager::new_active(
        server_spec(fake, port),
        model_spec(),
        TimeoutsConfig::default(),
        rx,
    )
    .await
    .expect("cold-start wake failed");
    (manager, tx)
}

/// `kill(pid, 0)` existence probe; EPERM still means the process exists.
fn pid_alive(pid: u32) -> bool {
    let Some(pid) = Pid::from_raw(pid as i32) else {
        return false;
    };
    matches!(test_kill_process(pid), Ok(()) | Err(Errno::PERM))
}

async fn wait_for_pid_gone(pid: u32, timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if !pid_alive(pid) {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    false
}

async fn wait_for_port_closed(port: u16, timeout: Duration) -> bool {
    let addr = format!("127.0.0.1:{port}");
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if TcpStream::connect(&addr).await.is_err() {
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
    let counters: Value = serde_json::from_str(&body).expect("counters json");
    let load = counters["load_count"].as_u64().expect("load_count") as u32;
    let unload = counters["unload_count"].as_u64().expect("unload_count") as u32;
    let loaded = counters["loaded_model"].as_str().map(|s| s.to_string());
    (load, unload, loaded)
}

#[tokio::test]
async fn cold_start_puts_manager_in_active_and_loads_model() {
    let fake = FakeLlama::new("normal");
    init_tracing();
    let port = grab_port().await;
    let (manager, _shutdown) = new_active_manager(&fake, port).await;

    assert_eq!(manager.state(), PresenceState::Active);
    let pid = manager.llama_pid().await.expect("child running after wake");
    assert!(pid_alive(pid));

    let (load_count, unload_count, loaded) = get_counters(port).await;
    assert_eq!(load_count, 1, "cold start should call /models/load once");
    assert_eq!(unload_count, 0);
    assert_eq!(loaded.as_deref(), Some(model_spec().name.as_str()));

    manager.sleep().await.unwrap();
}

#[tokio::test]
async fn sleep_stops_supervisor_and_kills_child() {
    let fake = FakeLlama::new("normal");
    init_tracing();
    let port = grab_port().await;
    let (manager, _shutdown) = new_active_manager(&fake, port).await;
    let pid = manager.llama_pid().await.expect("child running");
    assert!(pid_alive(pid));

    manager.sleep().await.expect("sleep should succeed");
    assert_eq!(manager.state(), PresenceState::Sleeping);
    assert!(
        manager.llama_pid().await.is_none(),
        "llama handle should be taken"
    );

    assert!(
        wait_for_pid_gone(pid, Duration::from_secs(5)).await,
        "child pid {pid} should be reaped after sleep"
    );
}

#[tokio::test]
async fn drowse_calls_unload_and_keeps_process_alive() {
    let fake = FakeLlama::new("normal");
    init_tracing();
    let port = grab_port().await;
    let (manager, _shutdown) = new_active_manager(&fake, port).await;
    let pid_before = manager.llama_pid().await.expect("child running");
    let (_, unload_before, _) = get_counters(port).await;

    manager.drowse().await.expect("drowse should succeed");
    assert_eq!(manager.state(), PresenceState::Drowsy);

    let pid_after = manager.llama_pid().await.expect("child still running");
    assert_eq!(pid_before, pid_after, "drowse must not respawn the process");
    assert!(pid_alive(pid_after));

    let (_, unload_after, loaded) = get_counters(port).await;
    assert_eq!(unload_after, unload_before + 1);
    assert!(
        loaded.is_none(),
        "expected no loaded model after unload, got {loaded:?}"
    );

    manager.sleep().await.unwrap();
}

#[tokio::test]
async fn wake_from_drowsy_reuses_process_and_only_loads_model() {
    let fake = FakeLlama::new("normal");
    init_tracing();
    let port = grab_port().await;
    let (manager, _shutdown) = new_active_manager(&fake, port).await;
    let pid_initial = manager.llama_pid().await.expect("child running");
    let (load_initial, _, _) = get_counters(port).await;

    manager.drowse().await.unwrap();
    assert_eq!(manager.state(), PresenceState::Drowsy);

    manager
        .wake()
        .await
        .expect("wake from Drowsy should succeed");
    assert_eq!(manager.state(), PresenceState::Active);

    let pid_after = manager.llama_pid().await.expect("child still running");
    assert_eq!(
        pid_initial, pid_after,
        "wake from Drowsy must not respawn the process"
    );

    let (load_after, _, loaded) = get_counters(port).await;
    assert_eq!(load_after, load_initial + 1);
    assert_eq!(loaded.as_deref(), Some(model_spec().name.as_str()));

    manager.sleep().await.unwrap();
}

#[tokio::test]
async fn wake_from_sleeping_cold_starts_and_returns_active() {
    let fake = FakeLlama::new("normal");
    init_tracing();
    let port = grab_port().await;
    let (manager, _shutdown) = new_active_manager(&fake, port).await;
    let pid_before = manager.llama_pid().await.expect("child running");

    manager.sleep().await.unwrap();
    assert!(wait_for_pid_gone(pid_before, Duration::from_secs(5)).await);

    manager
        .wake()
        .await
        .expect("wake from Sleeping should succeed");
    assert_eq!(manager.state(), PresenceState::Active);
    let pid_after = manager.llama_pid().await.expect("child running after wake");
    assert_ne!(
        pid_before, pid_after,
        "cold-start wake must spawn a new child"
    );
    assert!(pid_alive(pid_after));

    manager.sleep().await.unwrap();
}

#[tokio::test]
async fn failed_wake_leaves_nothing_behind_for_sleep_to_miss() {
    let fake = FakeLlama::new("normal");
    init_tracing();
    let port = grab_port().await;
    let (manager, _shutdown) = new_active_manager(&fake, port).await;
    let pid_before = manager.llama_pid().await.expect("child running");

    manager.sleep().await.unwrap();
    assert!(wait_for_pid_gone(pid_before, Duration::from_secs(5)).await);

    fake.set_mode("load-failure");
    let err = manager
        .wake()
        .await
        .expect_err("wake must fail when /models/load errors");
    assert!(matches!(err, PresenceError::Load { .. }), "{err:?}");

    assert_eq!(manager.state(), PresenceState::Sleeping);
    assert!(
        manager.llama_pid().await.is_none(),
        "a failed wake must not leave a handle the Sleeping manager cannot reach"
    );
    assert!(
        wait_for_port_closed(port, Duration::from_secs(5)).await,
        "the child started by the failed wake must be torn down, not stranded"
    );

    manager.sleep().await.expect("sleep after a failed wake");
    fake.set_mode("normal");
    manager.wake().await.expect("wake after a failed wake");
    assert_eq!(manager.state(), PresenceState::Active);

    manager.sleep().await.unwrap();
}

#[tokio::test]
async fn sleep_that_cannot_join_the_supervisor_still_commits_sleeping() {
    init_tracing();
    let port = grab_port().await;
    // Outlives SIGTERM past the 1s sleep budget, so the join times out.
    let fake = FakeLlama::new("slow-term=3");
    let (_shutdown_tx, shutdown_rx) = watch::channel(false);
    let timeouts = TimeoutsConfig {
        presence_sleep_secs: 1,
        ..TimeoutsConfig::default()
    };
    let manager = PresenceManager::new_active(
        server_spec(&fake, port),
        model_spec(),
        timeouts,
        shutdown_rx,
    )
    .await
    .expect("cold-start wake failed");
    let pid = manager.llama_pid().await.expect("child running");

    let err = manager
        .sleep()
        .await
        .expect_err("sleep must report a shutdown it could not join");
    assert!(
        matches!(err, PresenceError::ShutdownTimeout { secs: 1 }),
        "{err:?}"
    );

    assert_eq!(
        manager.state(),
        PresenceState::Sleeping,
        "the teardown signal cannot be recalled, so the state must follow it"
    );
    assert!(
        manager.llama_pid().await.is_none(),
        "an Active manager with an empty slot can never be woken or slept again"
    );
    assert!(
        wait_for_pid_gone(pid, Duration::from_secs(10)).await,
        "the child should still wind down after the budget expires"
    );
}

/// A daemon socket served over `AppState` for the duration of a test.
struct Daemon {
    sock_path: PathBuf,
    stop_tx: oneshot::Sender<()>,
    server: JoinHandle<()>,
    _dir: TempDir,
}

impl Daemon {
    async fn serve(manager: &Arc<PresenceManager>, backend: Arc<dyn LlmBackend>) -> Self {
        let state = Arc::new(AppState::new(
            Config::default(),
            backend,
            Arc::clone(manager),
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
        Self {
            sock_path,
            stop_tx,
            server,
            _dir: dir,
        }
    }

    async fn stop(self) {
        let _ = self.stop_tx.send(());
        self.server.await.unwrap();
    }
}

type EventLines = Lines<BufReader<OwnedReadHalf>>;

async fn send_query(sock: &Path, id: &str, text: &str) -> EventLines {
    let (read, mut write) = UnixStream::connect(sock).await.unwrap().into_split();
    let req = Request::Query {
        id: id.into(),
        text: text.into(),
        attachments: Vec::new(),
    };
    let mut body = serde_json::to_string(&req).unwrap();
    body.push('\n');
    write.write_all(body.as_bytes()).await.unwrap();
    write.shutdown().await.unwrap();
    BufReader::new(read).lines()
}

/// Read events into `events` until a terminal one or the end of the stream.
async fn read_to_terminal(lines: &mut EventLines, events: &mut Vec<Event>) {
    while let Some(line) = lines.next_line().await.unwrap() {
        let event: Event = serde_json::from_str(&line).unwrap();
        let terminal = event.is_terminal();
        events.push(event);
        if terminal {
            break;
        }
    }
}

async fn query(sock: &Path, id: &str, text: &str) -> Vec<Event> {
    let mut lines = send_query(sock, id, text).await;
    let mut events = Vec::new();
    read_to_terminal(&mut lines, &mut events).await;
    events
}

#[tokio::test]
async fn query_during_sleeping_triggers_auto_wake() {
    let fake = FakeLlama::new("normal");
    init_tracing();
    let port = grab_port().await;
    let (manager, _shutdown) = new_active_manager(&fake, port).await;

    manager.sleep().await.unwrap();
    assert_eq!(manager.state(), PresenceState::Sleeping);

    let daemon = Daemon::serve(&manager, Arc::new(EchoBackend::new())).await;
    let events = query(&daemon.sock_path, "q1", "hello").await;

    assert!(
        events
            .iter()
            .any(|e| matches!(e, Event::Delta { text, .. } if text == "hello")),
        "{events:?}"
    );
    assert!(
        matches!(events.last(), Some(Event::Done { .. })),
        "{events:?}"
    );
    assert_eq!(
        manager.state(),
        PresenceState::Active,
        "auto-wake must leave manager in Active"
    );

    daemon.stop().await;
    manager.sleep().await.unwrap();
}

/// Backend that leaves a `delay` gap between its one Delta and the end of
/// the turn.
struct DelayBackend {
    delay: Duration,
    last_user: Mutex<String>,
}

#[async_trait]
impl LlmBackend for DelayBackend {
    async fn generate(&self, prompt: String, tx: mpsc::Sender<LlmEvent>) -> LlmResult<()> {
        let _ = tx.send(LlmEvent::Delta { text: prompt }).await;
        tokio::time::sleep(self.delay).await;
        let _ = tx.send(LlmEvent::Done).await;
        Ok(())
    }

    async fn push_user(&self, text: String, _attachments: Vec<Attachment>) -> LlmResult<()> {
        *self.last_user.lock().await = text;
        Ok(())
    }

    async fn push_tool_results(&self, _results: Vec<ToolResultPayload>) -> LlmResult<()> {
        Ok(())
    }

    async fn step(&self, _tools: Vec<Value>, tx: mpsc::Sender<LlmEvent>) -> LlmResult<StepOutcome> {
        let text = self.last_user.lock().await.clone();
        let _ = tx.send(LlmEvent::Delta { text }).await;
        tokio::time::sleep(self.delay).await;
        Ok(StepOutcome::Final)
    }
}

#[tokio::test]
async fn sleep_defers_until_inflight_query_done() {
    let fake = FakeLlama::new("normal");
    init_tracing();
    let port = grab_port().await;
    let (manager, _shutdown) = new_active_manager(&fake, port).await;

    let backend = Arc::new(DelayBackend {
        delay: Duration::from_millis(500),
        last_user: Mutex::new(String::new()),
    });
    let daemon = Daemon::serve(&manager, backend).await;

    let mut lines = send_query(&daemon.sock_path, "q1", "hello").await;
    let mut events = Vec::new();
    while !events.iter().any(|e| matches!(e, Event::Delta { .. })) {
        let line = lines
            .next_line()
            .await
            .unwrap()
            .expect("stream closed before any Delta");
        events.push(serde_json::from_str::<Event>(&line).unwrap());
    }

    let sleeper = Arc::clone(&manager);
    let sleep_started = Instant::now();
    let sleep_task = tokio::spawn(async move { sleeper.sleep().await });

    read_to_terminal(&mut lines, &mut events).await;
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

    sleep_task.await.unwrap().expect("sleep returned Err");
    let sleep_elapsed = sleep_started.elapsed();
    assert!(
        sleep_elapsed >= Duration::from_millis(350),
        "sleep finished in {sleep_elapsed:?}, expected it to wait out most of the \
         500ms still left in the generation"
    );
    assert_eq!(manager.state(), PresenceState::Sleeping);

    daemon.stop().await;
}

#[tokio::test]
async fn concurrent_queries_during_wake_all_complete() {
    let fake = FakeLlama::new("normal");
    init_tracing();
    let port = grab_port().await;
    let (manager, _shutdown) = new_active_manager(&fake, port).await;
    let daemon = Daemon::serve(&manager, Arc::new(EchoBackend::new())).await;

    manager.sleep().await.unwrap();
    assert_eq!(manager.state(), PresenceState::Sleeping);

    let handles: Vec<_> = (0..5)
        .map(|i| {
            let id = format!("q{i}");
            let text = format!("msg{i}");
            let sock = daemon.sock_path.clone();
            let (task_id, task_text) = (id.clone(), text.clone());
            let handle = tokio::spawn(async move { query(&sock, &task_id, &task_text).await });
            (id, text, handle)
        })
        .collect();

    for (id, text, handle) in handles {
        let events = handle.await.unwrap();
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
        manager.state(),
        PresenceState::Active,
        "wake must leave manager Active"
    );

    daemon.stop().await;
    manager.sleep().await.unwrap();
}
