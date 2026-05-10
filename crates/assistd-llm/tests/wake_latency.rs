#![allow(unsafe_code)] // libc / env / fd primitives; each unsafe block is locally justified

//! Latency benchmarks for the auto-wake-on-query path. Each test
//! measures `Instant`-to-first-Delta latency end-to-end through the
//! full daemon stack: Unix socket → AppState → ensure_active() →
//! LlamaChatClient → fake_llama_server. Thresholds are deliberately
//! generous so transient CI jitter doesn't cause flakes; the goal is
//! to fail loudly on a 10× regression. Actual durations are printed
//! via `eprintln!` so logs surface drift over time.
//!
//! Run with: `cargo test -p assistd-llm --features test-support --test wake_latency`

#![cfg(feature = "test-support")]

use std::sync::Arc;
use std::sync::Once;
use std::time::{Duration, Instant};

use assistd_config::{ChatConfig, Config, LlamaServerConfig, ModelConfig, TimeoutsConfig};
use assistd_core::{
    AppState, NoContinuousListener, NoVoiceInput, NoVoiceOutput, PresenceManager, PresenceState,
    ToolRegistry, VoiceOutputController,
};
use assistd_ipc::{Event, Request};
use assistd_llm::{LlamaChatClient, LlmBackend};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, UnixStream};
use tokio::sync::{Mutex, oneshot, watch};

const FAKE_BIN: &str = env!("CARGO_BIN_EXE_fake_llama_server");

static MODE_LOCK: Mutex<()> = Mutex::const_new(());

fn init_tracing() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let _ = tracing_subscriber::fmt()
            .with_env_filter(
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
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
    let m = PresenceManager::new_active(
        server_spec(port),
        model_spec(),
        TimeoutsConfig::default(),
        rx,
    )
    .await
    .expect("cold-start wake failed");
    (m, tx)
}

/// Set up an AppState backed by a real LlamaChatClient against the
/// fake server, expose it on a temp Unix socket, and return:
/// - the manager (so the test can drive sleep/drowse)
/// - the socket path
/// - the stop sender + server task handle (so the test can clean up)
async fn build_running_daemon(
    port: u16,
) -> (
    Arc<PresenceManager>,
    std::path::PathBuf,
    oneshot::Sender<()>,
    tokio::task::JoinHandle<()>,
    tempfile::TempDir,
) {
    let (m, _shutdown) = new_active_manager(port).await;
    let chat_cfg = ChatConfig {
        request_timeout_secs: 10,
        ..ChatConfig::default()
    };
    let server_cfg = server_spec(port);

    let client = LlamaChatClient::new(
        &chat_cfg,
        &server_cfg,
        &model_spec(),
        &TimeoutsConfig::default(),
        None,
    )
    .expect("build chat client");
    let mut config = Config::default();
    // Default grace is 5s; that adds 5s × 3 tests to the suite for no
    // gain in CI. Cap at 1s so test wall time tracks the actual work.
    config.daemon.shutdown_grace_secs = 1;
    let state = Arc::new(AppState::new(
        config,
        Arc::new(client) as Arc<dyn LlmBackend>,
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
    (m, sock_path, stop_tx, server, dir)
}

/// Send a Query to the daemon and return (latency to first Delta,
/// total event count, terminal event). The connection is dropped at
/// the end so tests don't accumulate open sockets.
async fn measure_query_latency(sock_path: &std::path::Path, id: &str) -> (Duration, usize, Event) {
    let stream = UnixStream::connect(sock_path).await.unwrap();
    let (read, mut write) = stream.into_split();
    let req = Request::Query {
        id: id.to_string(),
        text: "ping".into(),
        attachments: Vec::new(),
        version: None,
    };
    let mut body = serde_json::to_string(&req).unwrap();
    body.push('\n');

    let t0 = Instant::now();
    write.write_all(body.as_bytes()).await.unwrap();
    write.shutdown().await.unwrap();

    let mut reader = BufReader::new(read);
    let mut count = 0usize;
    let mut first_delta_at: Option<Duration> = None;
    let mut terminal: Option<Event> = None;
    loop {
        let mut line = String::new();
        let n = reader.read_line(&mut line).await.unwrap();
        if n == 0 {
            break;
        }
        count += 1;
        let e: Event = serde_json::from_str(line.trim()).unwrap();
        if matches!(e, Event::Delta { .. }) && first_delta_at.is_none() {
            first_delta_at = Some(t0.elapsed());
        }
        let is_terminal = matches!(e, Event::Done { .. } | Event::Error { .. });
        if is_terminal {
            terminal = Some(e);
            break;
        }
    }
    let latency = first_delta_at.expect("never received a Delta");
    (latency, count, terminal.expect("no terminal event"))
}

#[tokio::test]
async fn active_query_baseline_under_200ms() {
    let _g = MODE_LOCK.lock().await;
    init_tracing();
    let port = grab_port().await;
    let (m, sock_path, stop_tx, server, _dir) = build_running_daemon(port).await;
    assert_eq!(m.state(), PresenceState::Active);

    let (latency, _count, terminal) = measure_query_latency(&sock_path, "active-baseline").await;
    eprintln!("active baseline first-Delta latency = {latency:?}");
    assert!(
        matches!(terminal, Event::Done { .. }),
        "expected terminal Done"
    );
    assert!(
        latency < Duration::from_millis(200),
        "active baseline regressed: first Delta took {latency:?}, expected <200ms"
    );

    let _ = stop_tx.send(());
    server.await.unwrap();
    m.sleep().await.unwrap();
}

#[tokio::test]
async fn wake_from_drowsy_first_delta_under_1s() {
    let _g = MODE_LOCK.lock().await;
    init_tracing();
    let port = grab_port().await;
    let (m, sock_path, stop_tx, server, _dir) = build_running_daemon(port).await;

    m.drowse().await.expect("drowse");
    assert_eq!(m.state(), PresenceState::Drowsy);

    let (latency, _count, terminal) = measure_query_latency(&sock_path, "wake-from-drowsy").await;
    eprintln!("wake-from-Drowsy first-Delta latency = {latency:?}");
    assert!(
        matches!(terminal, Event::Done { .. }),
        "expected terminal Done"
    );
    assert!(
        latency < Duration::from_secs(1),
        "wake-from-Drowsy regressed: first Delta took {latency:?}, expected <1s"
    );
    // Auto-wake must have left the daemon Active.
    assert_eq!(m.state(), PresenceState::Active);

    let _ = stop_tx.send(());
    server.await.unwrap();
    m.sleep().await.unwrap();
}

#[tokio::test]
async fn wake_from_sleeping_first_delta_under_5s() {
    let _g = MODE_LOCK.lock().await;
    init_tracing();
    let port = grab_port().await;
    let (m, sock_path, stop_tx, server, _dir) = build_running_daemon(port).await;

    m.sleep().await.expect("sleep");
    assert_eq!(m.state(), PresenceState::Sleeping);

    let (latency, _count, terminal) = measure_query_latency(&sock_path, "wake-from-sleeping").await;
    eprintln!("wake-from-Sleeping first-Delta latency = {latency:?}");
    assert!(
        matches!(terminal, Event::Done { .. }),
        "expected terminal Done"
    );
    assert!(
        latency < Duration::from_secs(5),
        "wake-from-Sleeping regressed: first Delta took {latency:?}, expected <5s"
    );
    assert_eq!(m.state(), PresenceState::Active);

    let _ = stop_tx.send(());
    server.await.unwrap();
    m.sleep().await.unwrap();
}
