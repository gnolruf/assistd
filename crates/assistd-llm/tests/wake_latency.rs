//! First-Delta latency benchmarks for auto-wake-on-query through the full
//! daemon stack. Thresholds catch a 10× regression, not CI jitter.

#![cfg(feature = "test-support")]

use std::net::Ipv4Addr;
use std::num::NonZeroU16;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Once};
use std::time::{Duration, Instant};

use tempfile::TempDir;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, UnixStream};
use tokio::sync::{oneshot, watch};
use tokio::task::JoinHandle;

use assistd_config::defaults::{nz32, nz64};
use assistd_config::{ChatConfig, Config, LlamaServerConfig, ModelConfig, TimeoutsConfig};
use assistd_core::{
    AppState, NoContinuousListener, NoVoiceInput, NoVoiceOutput, PresenceManager, PresenceState,
    ToolRegistry, VoiceOutputController,
};
use assistd_ipc::{Event, Request};
use assistd_llm::{LlamaChatClient, LlmBackend};

use common::FakeLlama;

mod common;

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

/// Serve an `AppState` backed by a real `LlamaChatClient` on a temp Unix
/// socket, returning the manager, socket path, stop sender, task and tempdir.
async fn build_running_daemon(
    fake: &FakeLlama,
    port: u16,
) -> (
    Arc<PresenceManager>,
    PathBuf,
    oneshot::Sender<()>,
    JoinHandle<()>,
    TempDir,
) {
    let (manager, _shutdown) = new_active_manager(fake, port).await;
    let chat_cfg = ChatConfig {
        request_timeout_secs: nz64(10),
        ..ChatConfig::default()
    };
    let server_cfg = server_spec(fake, port);

    let client = LlamaChatClient::new(
        &chat_cfg,
        &server_cfg,
        &model_spec(),
        &TimeoutsConfig::default(),
        None,
    )
    .expect("build chat client");
    let mut config = Config::default();
    config.daemon.shutdown_grace_secs = 1;
    let state = Arc::new(AppState::new(
        config,
        Arc::new(client) as Arc<dyn LlmBackend>,
        manager.clone(),
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
    (manager, sock_path, stop_tx, server, dir)
}

/// Send a Query and return the latency to its first Delta plus its terminal
/// event.
async fn measure_query_latency(sock_path: &Path, id: &str) -> (Duration, Event) {
    let stream = UnixStream::connect(sock_path).await.unwrap();
    let (read, mut write) = stream.into_split();
    let req = Request::Query {
        id: id.to_string(),
        text: "ping".into(),
        attachments: Vec::new(),
    };
    let mut body = serde_json::to_string(&req).unwrap();
    body.push('\n');

    let sent_at = Instant::now();
    write.write_all(body.as_bytes()).await.unwrap();
    write.shutdown().await.unwrap();

    let mut reader = BufReader::new(read);
    let mut first_delta_at: Option<Duration> = None;
    let mut terminal: Option<Event> = None;
    loop {
        let mut line = String::new();
        let n = reader.read_line(&mut line).await.unwrap();
        if n == 0 {
            break;
        }
        let event: Event = serde_json::from_str(line.trim()).unwrap();
        if matches!(event, Event::Delta { .. }) && first_delta_at.is_none() {
            first_delta_at = Some(sent_at.elapsed());
        }
        if matches!(event, Event::Done { .. } | Event::Error { .. }) {
            terminal = Some(event);
            break;
        }
    }
    let latency = first_delta_at.expect("never received a Delta");
    (latency, terminal.expect("no terminal event"))
}

#[tokio::test]
async fn active_query_baseline_under_200ms() {
    let fake = FakeLlama::new("normal");
    init_tracing();
    let port = grab_port().await;
    let (manager, sock_path, stop_tx, server, _dir) = build_running_daemon(&fake, port).await;
    assert_eq!(manager.state(), PresenceState::Active);

    let (latency, terminal) = measure_query_latency(&sock_path, "active-baseline").await;
    tracing::info!(?latency, "active baseline first-Delta latency");
    assert!(matches!(terminal, Event::Done { .. }), "{terminal:?}");
    assert!(
        latency < Duration::from_millis(200),
        "active baseline regressed: first Delta took {latency:?}, expected <200ms"
    );

    let _ = stop_tx.send(());
    server.await.unwrap();
    manager.sleep().await.unwrap();
}

#[tokio::test]
async fn wake_from_drowsy_first_delta_under_1s() {
    let fake = FakeLlama::new("normal");
    init_tracing();
    let port = grab_port().await;
    let (manager, sock_path, stop_tx, server, _dir) = build_running_daemon(&fake, port).await;

    manager.drowse().await.expect("drowse");
    assert_eq!(manager.state(), PresenceState::Drowsy);

    let (latency, terminal) = measure_query_latency(&sock_path, "wake-from-drowsy").await;
    tracing::info!(?latency, "wake-from-Drowsy first-Delta latency");
    assert!(matches!(terminal, Event::Done { .. }), "{terminal:?}");
    assert!(
        latency < Duration::from_secs(1),
        "wake-from-Drowsy regressed: first Delta took {latency:?}, expected <1s"
    );
    assert_eq!(manager.state(), PresenceState::Active);

    let _ = stop_tx.send(());
    server.await.unwrap();
    manager.sleep().await.unwrap();
}

#[tokio::test]
async fn wake_from_sleeping_first_delta_under_5s() {
    let fake = FakeLlama::new("normal");
    init_tracing();
    let port = grab_port().await;
    let (manager, sock_path, stop_tx, server, _dir) = build_running_daemon(&fake, port).await;

    manager.sleep().await.expect("sleep");
    assert_eq!(manager.state(), PresenceState::Sleeping);

    let (latency, terminal) = measure_query_latency(&sock_path, "wake-from-sleeping").await;
    tracing::info!(?latency, "wake-from-Sleeping first-Delta latency");
    assert!(matches!(terminal, Event::Done { .. }), "{terminal:?}");
    assert!(
        latency < Duration::from_secs(5),
        "wake-from-Sleeping regressed: first Delta took {latency:?}, expected <5s"
    );
    assert_eq!(manager.state(), PresenceState::Active);

    let _ = stop_tx.send(());
    server.await.unwrap();
    manager.sleep().await.unwrap();
}
