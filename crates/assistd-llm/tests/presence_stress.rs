//! Heavyweight `#[ignore]`d stress tests for the presence state machine and
//! the request-guard / chat-client interaction.

#![cfg(feature = "test-support")]

use std::net::Ipv4Addr;
use std::num::NonZeroU16;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Once};
use std::time::{Duration, Instant};

use serde_json::Value;
use tempfile::TempDir;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::unix::OwnedReadHalf;
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

async fn get_counters(port: u16) -> (u32, u32, u32, Option<String>) {
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
    let chat = counters["chat_completions_count"]
        .as_u64()
        .expect("chat_completions_count") as u32;
    let pid = counters["pid"].as_u64().map(|n| n.to_string());
    (load, unload, chat, pid)
}

/// Queue a scripted reply for the fake server's next chat completion.
async fn push_chat_script(port: u16, deltas: Vec<&str>, delay_ms_between: u64) {
    let body = serde_json::json!({
        "deltas": deltas,
        "delay_ms_between": delay_ms_between,
    });
    let url = format!("http://127.0.0.1:{port}/test/script");
    let resp = reqwest::Client::new()
        .post(&url)
        .header("X-Test-Control", "1")
        .json(&body)
        .send()
        .await
        .expect("POST /test/script");
    assert!(resp.status().is_success(), "push script failed: {resp:?}");
}

async fn serve_daemon(
    manager: &Arc<PresenceManager>,
    backend: Arc<dyn LlmBackend>,
) -> (PathBuf, oneshot::Sender<()>, JoinHandle<()>, TempDir) {
    let state = Arc::new(AppState::new(
        Config::default(),
        backend,
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
    (sock_path, stop_tx, server, dir)
}

async fn send_query(sock_path: &Path, id: &str, text: &str) -> BufReader<OwnedReadHalf> {
    let stream = UnixStream::connect(sock_path).await.unwrap();
    let (read, mut write) = stream.into_split();
    let req = Request::Query {
        id: id.into(),
        text: text.into(),
        attachments: Vec::new(),
    };
    let mut body = serde_json::to_string(&req).unwrap();
    body.push('\n');
    write.write_all(body.as_bytes()).await.unwrap();
    write.shutdown().await.unwrap();
    BufReader::new(read)
}

/// The next event on the stream, or `None` at EOF.
async fn read_event(reader: &mut BufReader<OwnedReadHalf>) -> Option<Event> {
    let mut line = String::new();
    let n = reader.read_line(&mut line).await.unwrap();
    (n != 0).then(|| serde_json::from_str(line.trim()).unwrap())
}

#[tokio::test]
#[ignore = "spawns 10 real fake_llama_server cold-starts; ~10s; run with --ignored"]
async fn ten_cold_start_cycles_no_deadlock() {
    let fake = FakeLlama::new("normal");
    init_tracing();
    let port = grab_port().await;
    let (manager, _shutdown) = new_active_manager(&fake, port).await;

    let initial_pid = manager.llama_pid().await.expect("active after cold start");
    let mut last_pid = initial_pid;

    for i in 0..10 {
        manager
            .sleep()
            .await
            .unwrap_or_else(|e| panic!("sleep cycle {i}: {e}"));
        assert_eq!(manager.state(), PresenceState::Sleeping);

        manager
            .wake()
            .await
            .unwrap_or_else(|e| panic!("wake cycle {i}: {e}"));
        assert_eq!(manager.state(), PresenceState::Active);

        let pid = manager
            .llama_pid()
            .await
            .unwrap_or_else(|| panic!("no llama PID after wake cycle {i}"));
        assert_ne!(
            pid, last_pid,
            "cycle {i}: cold-start respawn must produce a fresh PID, but pid={pid} == previous={last_pid}"
        );
        last_pid = pid;
    }

    let (load_count, _, _, _) = get_counters(port).await;
    assert_eq!(
        load_count, 1,
        "the final fresh child should have seen only its own cold-start load"
    );

    manager.sleep().await.unwrap();
}

#[tokio::test]
#[ignore = "drives a real LlamaChatClient against fake_llama_server with a slow scripted stream; run with --ignored"]
async fn sleep_defers_until_inflight_real_chat_stream_done() {
    let fake = FakeLlama::new("normal");
    init_tracing();

    let port = grab_port().await;
    let (manager, _shutdown) = new_active_manager(&fake, port).await;

    push_chat_script(port, vec!["one ", "two ", "three ", "four ", "five"], 200).await;

    let chat_cfg = ChatConfig {
        request_timeout_secs: nz64(10),
        ..ChatConfig::default()
    };
    let client = LlamaChatClient::new(
        &chat_cfg,
        &server_spec(&fake, port),
        &model_spec(),
        &TimeoutsConfig::default(),
        None,
    )
    .expect("build chat client");
    let (sock_path, stop_tx, server, _dir) = serve_daemon(&manager, Arc::new(client)).await;

    let mut reader = send_query(&sock_path, "q-stream", "hello").await;
    let mut events: Vec<Event> = Vec::new();
    loop {
        let event = read_event(&mut reader)
            .await
            .expect("connection closed before any Delta");
        let is_delta = matches!(event, Event::Delta { .. });
        events.push(event);
        if is_delta {
            break;
        }
    }

    let sleeper = manager.clone();
    let sleep_started = Instant::now();
    let sleep_task = tokio::spawn(async move { sleeper.sleep().await });

    while let Some(event) = read_event(&mut reader).await {
        let terminal = matches!(event, Event::Done { .. } | Event::Error { .. });
        events.push(event);
        if terminal {
            break;
        }
    }

    let delta_count = events
        .iter()
        .filter(|e| matches!(e, Event::Delta { .. }))
        .count();
    assert_eq!(
        delta_count, 5,
        "expected 5 deltas through the chat stream, got {delta_count} (events: {events:?})"
    );
    assert!(
        matches!(events.last(), Some(Event::Done { .. })),
        "expected terminal Done, got {events:?}"
    );
    assert!(
        !events.iter().any(|e| matches!(e, Event::Error { .. })),
        "no Error events expected: {events:?}"
    );

    sleep_task.await.unwrap().expect("sleep returned Err");
    let elapsed = sleep_started.elapsed();
    assert!(
        elapsed >= Duration::from_millis(600),
        "sleep finished in {elapsed:?}; expected >=600ms (RequestGuard should block on the in-flight stream)"
    );
    assert_eq!(manager.state(), PresenceState::Sleeping);

    let _ = stop_tx.send(());
    server.await.unwrap();
}
