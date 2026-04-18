//! Integration tests for the llama-server supervisor.
//!
//! These tests are gated behind the `test-support` feature because they need
//! the `fake_llama_server` helper binary. Run with:
//!     cargo test -p assistd-llm --features test-support

#![cfg(feature = "test-support")]

use assistd_config::{LlamaServerConfig, ModelConfig};
use assistd_llm::{LlamaService, ReadyState};
use std::sync::Once;
use std::time::{Duration, Instant};
use tokio::net::TcpListener;
use tokio::sync::watch;

const FAKE_BIN: &str = env!("CARGO_BIN_EXE_fake_llama_server");

fn init_tracing() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let _ = tracing_subscriber::fmt()
            .with_env_filter(
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("debug")),
            )
            .with_test_writer()
            .try_init();
    });
}

/// Grab an ephemeral port by binding and dropping. Small race window, good
/// enough for tests.
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
    }
}

fn model_spec() -> ModelConfig {
    ModelConfig {
        name: "test/fake-model-GGUF:Q4_K_M".to_string(),
        context_length: 2048,
    }
}

/// Appends `--mode <mode>` by wrapping the fake binary via a shell invocation.
/// We need this because LlamaServerConfig doesn't expose extra args — but the fake
/// binary parses args itself. Instead we rely on the binary tolerating the
/// llama-server arg shape and parsing its own mode flag out of env.
///
/// Simpler: use an env var the fake binary reads.
///
/// Even simpler: make the fake binary look at its own args — `--mode <m>` is
/// already parsed out of the command line. We put it at the end so the real
/// llama-server-shaped args come first.
///
/// Since LlamaServerConfig just runs `binary_path` directly, we need a way to
/// inject the mode. Two options: wrapper script, or env var.
///
/// Environment variable is cleanest.
fn set_mode(mode: &str) {
    // SAFETY: tests are single-threaded per-test, but integration tests run in
    // parallel by default. We serialize test execution via the MODE_LOCK mutex
    // below to make sure set_mode/spawn happen atomically.
    unsafe { std::env::set_var("FAKE_LLAMA_MODE", mode) };
}

// Tests share a process, so set_mode / start must be serialized. Use tokio's
// async-aware mutex so the guard is legal to hold across `.await` points.
use tokio::sync::Mutex;
static MODE_LOCK: Mutex<()> = Mutex::const_new(());

async fn start_service(mode: &str, port: u16) -> (LlamaService, watch::Sender<bool>) {
    let (shutdown_tx, shutdown_rx) = watch::channel(false);
    set_mode(mode);
    let service = LlamaService::start(server_spec(port), model_spec(), shutdown_rx)
        .await
        .expect("service should start");
    (service, shutdown_tx)
}

#[tokio::test]
async fn brings_up_fake_server_and_reports_ready() {
    let _guard = MODE_LOCK.lock().await;
    let port = grab_port().await;
    let (service, shutdown_tx) = start_service("normal", port).await;

    assert!(service.is_ready());
    assert!(service.pid().is_some());

    let _ = shutdown_tx.send(true);
    service.shutdown().await.unwrap();
}

#[tokio::test]
async fn restarts_after_external_kill() {
    let _guard = MODE_LOCK.lock().await;
    let port = grab_port().await;
    let (service, shutdown_tx) = start_service("normal", port).await;

    let first_pid = service.pid().expect("first pid");
    // SAFETY: libc::kill with a valid pid and signal.
    unsafe {
        libc::kill(first_pid as i32, libc::SIGKILL);
    }

    // Wait for the supervisor to notice and respawn.
    let deadline = Instant::now() + Duration::from_secs(8);
    loop {
        if Instant::now() >= deadline {
            panic!("supervisor did not restart llama-server within the deadline");
        }
        if let Some(pid) = service.pid()
            && pid != first_pid
            && service.is_ready()
        {
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    let _ = shutdown_tx.send(true);
    service.shutdown().await.unwrap();
}

#[tokio::test]
async fn enters_degraded_after_five_failures() {
    init_tracing();
    let _guard = MODE_LOCK.lock().await;
    let port = grab_port().await;
    let (_shutdown_tx, shutdown_rx) = watch::channel(false);
    set_mode("bind-fail");

    let start_at = Instant::now();
    let result = LlamaService::start(server_spec(port), model_spec(), shutdown_rx).await;
    let elapsed = start_at.elapsed();

    let err = result.err().expect("start should fail");
    let msg = format!("{err}");
    assert!(
        msg.contains("startup failed") || msg.contains("after"),
        "unexpected error message: {msg}"
    );
    // Backoff budget: 1 + 2 + 4 + 8 = 15s of sleeps between 4 retries. Add
    // generous slack for scheduler jitter + spawn time.
    assert!(
        elapsed >= Duration::from_secs(14),
        "start returned too quickly: {elapsed:?}"
    );
    assert!(
        elapsed < Duration::from_secs(40),
        "start took too long: {elapsed:?}"
    );
}

#[tokio::test]
async fn respects_shutdown_during_backoff() {
    let _guard = MODE_LOCK.lock().await;
    let port = grab_port().await;
    let (shutdown_tx, shutdown_rx) = watch::channel(false);
    set_mode("bind-fail");

    // Flip the shutdown watch after ~3 seconds — enough time to hit the first
    // backoff sleep but far short of the full 5-failure timeline.
    let flip_tx = shutdown_tx.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_secs(3)).await;
        let _ = flip_tx.send(true);
    });

    let start_at = Instant::now();
    let result = LlamaService::start(server_spec(port), model_spec(), shutdown_rx).await;
    let elapsed = start_at.elapsed();

    // start() should either error (ShutdownDuringHealth) or return quickly.
    assert!(
        result.is_err(),
        "expected an error, got {:?}",
        result.is_ok()
    );
    assert!(
        elapsed < Duration::from_secs(10),
        "start did not respect shutdown: {elapsed:?}"
    );
    let _ = shutdown_tx.send(true);
}

#[tokio::test]
async fn shutdown_kills_running_child() {
    let _guard = MODE_LOCK.lock().await;
    let port = grab_port().await;
    let (service, shutdown_tx) = start_service("normal", port).await;

    let pid = service.pid().expect("running child");
    assert!(
        std::path::Path::new(&format!("/proc/{pid}")).exists(),
        "fake child should be alive before shutdown"
    );

    let _ = shutdown_tx.send(true);
    service.shutdown().await.unwrap();

    // Give the kernel a beat to reap the process.
    for _ in 0..50 {
        if !std::path::Path::new(&format!("/proc/{pid}")).exists() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("fake child {pid} still alive after shutdown");
}

#[tokio::test]
async fn reaches_ready_even_when_state_transitions() {
    let _guard = MODE_LOCK.lock().await;
    let port = grab_port().await;
    let (service, shutdown_tx) = start_service("normal", port).await;

    assert_eq!(service.state(), ReadyState::Ready);

    let _ = shutdown_tx.send(true);
    service.shutdown().await.unwrap();
}
