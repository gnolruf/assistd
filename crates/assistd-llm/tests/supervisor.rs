//! Integration tests for the llama-server supervisor against the
//! `fake_llama_server` helper binary.

#![cfg(feature = "test-support")]

mod common;

use assistd_config::defaults::{nz32, nz64};
use assistd_config::{LlamaServerConfig, ModelConfig};
use assistd_llm::{LlamaServerError, LlamaService, ReadyState};
use common::FakeLlama;
use std::net::Ipv4Addr;
use std::num::NonZeroU16;
use std::sync::Once;
use std::time::{Duration, Instant};
use tokio::net::TcpListener;
use tokio::sync::watch;

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

async fn start_service(fake: &FakeLlama, port: u16) -> (LlamaService, watch::Sender<bool>) {
    let (shutdown_tx, shutdown_rx) = watch::channel(false);
    let service = LlamaService::start(server_spec(fake, port), model_spec(), shutdown_rx)
        .await
        .expect("service should start");
    (service, shutdown_tx)
}

#[tokio::test]
async fn brings_up_fake_server_and_reports_ready() {
    let fake = FakeLlama::new("normal");
    let port = grab_port().await;
    let (service, shutdown_tx) = start_service(&fake, port).await;

    assert_eq!(service.state(), ReadyState::Ready);
    assert!(service.is_ready());
    assert!(service.pid().is_some());

    let _ = shutdown_tx.send(true);
    service.shutdown().await.unwrap();
}

#[tokio::test]
async fn restarts_after_external_kill() {
    let fake = FakeLlama::new("normal");
    let port = grab_port().await;
    let (service, shutdown_tx) = start_service(&fake, port).await;

    let first_pid = service.pid().expect("first pid");
    let pid = rustix::process::Pid::from_raw(first_pid as i32).expect("nonzero pid");
    rustix::process::kill_process(pid, rustix::process::Signal::KILL)
        .expect("SIGKILL on test child");

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
    let fake = FakeLlama::new("bind-fail");
    let port = grab_port().await;
    let (_shutdown_tx, shutdown_rx) = watch::channel(false);

    let start_at = Instant::now();
    let result = LlamaService::start(server_spec(&fake, port), model_spec(), shutdown_rx).await;
    let elapsed = start_at.elapsed();

    let err = result.err().expect("start should fail");
    assert!(
        matches!(err, LlamaServerError::StartupFailed { attempts: 5 }),
        "{err:?}"
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
    let fake = FakeLlama::new("bind-fail");
    let port = grab_port().await;
    let (shutdown_tx, shutdown_rx) = watch::channel(false);

    // Flip the shutdown watch after ~3 seconds: enough time to hit the first
    // backoff sleep but far short of the full 5-failure timeline.
    let flip_tx = shutdown_tx.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_secs(3)).await;
        let _ = flip_tx.send(true);
    });

    let start_at = Instant::now();
    let result = LlamaService::start(server_spec(&fake, port), model_spec(), shutdown_rx).await;
    let elapsed = start_at.elapsed();

    let err = result.err().expect("start should fail once shut down");
    assert!(
        matches!(err, LlamaServerError::ShutdownDuringHealth),
        "{err:?}"
    );
    assert!(
        elapsed < Duration::from_secs(10),
        "start did not respect shutdown: {elapsed:?}"
    );
    let _ = shutdown_tx.send(true);
}

#[tokio::test]
async fn shutdown_kills_running_child() {
    let fake = FakeLlama::new("normal");
    let port = grab_port().await;
    let (service, shutdown_tx) = start_service(&fake, port).await;

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
