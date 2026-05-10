#![allow(unsafe_code)] // libc / env / fd primitives; each unsafe block is locally justified

//! Heavyweight stress tests for the presence state machine and the
//! request-guard / chat-client interaction. Each test spawns a real
//! `fake_llama_server` child process; they are gated behind `#[ignore]`
//! so the default `cargo test` run stays fast.
//!
//! Run with: `cargo test -p assistd-llm --features test-support --test presence_stress -- --ignored`

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

async fn get_counters(port: u16) -> (u32, u32, u32, Option<String>) {
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
    let chat = v["chat_completions_count"]
        .as_u64()
        .expect("chat_completions_count") as u32;
    let pid = v["pid"].as_u64().map(|n| n.to_string());
    (load, unload, chat, pid)
}

/// Push a scripted chat reply onto the fake server's queue. Each
/// subsequent /v1/chat/completions request pops the next entry.
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

#[tokio::test]
#[ignore = "spawns 10 real fake_llama_server cold-starts; ~10s; run with --ignored"]
async fn ten_cold_start_cycles_no_deadlock() {
    let _g = MODE_LOCK.lock().await;
    init_tracing();
    let port = grab_port().await;
    let (m, _shutdown) = new_active_manager(port).await;

    let initial_pid = m.llama_pid().await.expect("active after cold start");
    let mut last_pid = initial_pid;

    for i in 0..10 {
        m.sleep()
            .await
            .unwrap_or_else(|e| panic!("sleep cycle {i}: {e}"));
        assert_eq!(m.state(), PresenceState::Sleeping);

        m.wake()
            .await
            .unwrap_or_else(|e| panic!("wake cycle {i}: {e}"));
        assert_eq!(m.state(), PresenceState::Active);

        let pid = m
            .llama_pid()
            .await
            .unwrap_or_else(|| panic!("no llama PID after wake cycle {i}"));
        assert_ne!(
            pid, last_pid,
            "cycle {i}: cold-start respawn must produce a fresh PID, but pid={pid} == previous={last_pid}"
        );
        last_pid = pid;
    }

    // Final teardown: each cycle calls /models/load on the new child, so
    // the most recent load_count is exactly 1 (a fresh child = fresh
    // counter). Just sanity-check it's >= 1.
    let (load_count, _, _, _) = get_counters(port).await;
    assert!(load_count >= 1, "expected at least one load on final child");

    m.sleep().await.unwrap();
}

#[tokio::test]
#[ignore = "drives a real LlamaChatClient against fake_llama_server with a slow scripted stream; run with --ignored"]
async fn sleep_defers_until_inflight_real_chat_stream_done() {
    let _g = MODE_LOCK.lock().await;
    init_tracing();

    let port = grab_port().await;
    let (m, _shutdown) = new_active_manager(port).await;

    // Push a slow 5-delta script: 200 ms between deltas → ~800 ms stream.
    push_chat_script(port, vec!["one ", "two ", "three ", "four ", "five"], 200).await;

    let chat_cfg = ChatConfig {
        request_timeout_secs: 10,
        ..ChatConfig::default()
    };
    let mut server_cfg = server_spec(port);
    // The chat client connects directly via reqwest; binary_path is
    // unused here but keep it for any future probe code.
    server_cfg.binary_path = FAKE_BIN.to_string();

    let client = LlamaChatClient::new(
        &chat_cfg,
        &server_cfg,
        &model_spec(),
        &TimeoutsConfig::default(),
        None,
    )
    .expect("build chat client");

    let state = Arc::new(AppState::new(
        Config::default(),
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

    // Open the query connection and stream events.
    let stream = UnixStream::connect(&sock_path).await.unwrap();
    let (read, mut write) = stream.into_split();
    let req = Request::Query {
        id: "q-stream".into(),
        text: "hello".into(),
        attachments: Vec::new(),
        version: None,
    };
    let mut body = serde_json::to_string(&req).unwrap();
    body.push('\n');
    write.write_all(body.as_bytes()).await.unwrap();
    write.shutdown().await.unwrap();

    let mut reader = BufReader::new(read);
    let mut events: Vec<Event> = Vec::new();

    // Read the first Delta to confirm the stream is mid-flight.
    let mut first_delta_seen = false;
    while !first_delta_seen {
        let mut line = String::new();
        let n = reader.read_line(&mut line).await.unwrap();
        if n == 0 {
            panic!("connection closed before any Delta");
        }
        let e: Event = serde_json::from_str(line.trim()).unwrap();
        if matches!(e, Event::Delta { .. }) {
            first_delta_seen = true;
        }
        events.push(e);
    }

    // Stream is in flight. Trigger sleep concurrently; it must defer
    // until all 5 deltas + Done are forwarded.
    let m_for_sleep = m.clone();
    let sleep_started = Instant::now();
    let sleep_task = tokio::spawn(async move { m_for_sleep.sleep().await });

    // Drain the rest of the stream until Done.
    loop {
        let mut line = String::new();
        let n = reader.read_line(&mut line).await.unwrap();
        if n == 0 {
            break;
        }
        let e: Event = serde_json::from_str(line.trim()).unwrap();
        let terminal = matches!(e, Event::Done { .. } | Event::Error { .. });
        events.push(e);
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

    // sleep should have blocked on the in-flight RequestGuard until the
    // stream completed. The script runs ~800ms (4 gaps of 200ms each);
    // accept >= 600ms to give a bit of slack for scheduler jitter.
    sleep_task.await.unwrap().expect("sleep returned Err");
    let elapsed = sleep_started.elapsed();
    eprintln!("sleep_started.elapsed() = {elapsed:?}");
    assert!(
        elapsed >= Duration::from_millis(600),
        "sleep finished in {elapsed:?}; expected >=600ms (RequestGuard should block on the in-flight stream)"
    );
    assert_eq!(m.state(), PresenceState::Sleeping);

    let _ = stop_tx.send(());
    server.await.unwrap();
}
