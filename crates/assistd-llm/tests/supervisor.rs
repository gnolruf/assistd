//! Integration tests for the llama-server supervisor against the
//! `fake_llama_server` helper binary.

#![cfg(feature = "test-support")]

use std::net::Ipv4Addr;
use std::num::NonZeroU16;
use std::time::{Duration, Instant};

use rustix::process::{Pid, Signal, kill_process};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::watch;

use assistd_config::ModelConfig;
use assistd_config::defaults::{nz32, nz64};
use assistd_llm::LlamaServerSpec;
use assistd_utils::child_server::{ChildServer, ChildServerError};

use common::FakeLlama;

mod common;

/// Grab an ephemeral port by binding and dropping it.
async fn grab_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    drop(listener);
    port
}

fn model_spec(fake: &FakeLlama, port: u16) -> ModelConfig {
    ModelConfig {
        name: "test/fake-model-GGUF:Q4_K_M".to_string(),
        context_length: nz32(2048),
        server_binary: fake.binary_path(),
        host: Ipv4Addr::LOCALHOST.into(),
        port: NonZeroU16::new(port).expect("bound port is never 0"),
        gpu_layers: 0,
        ready_timeout_secs: nz64(60),
        ..ModelConfig::default()
    }
}

/// Answer every connection on `listener` with an empty 200.
async fn answer_every_request_with_ok(listener: TcpListener) {
    while let Ok((mut stream, _)) = listener.accept().await {
        let mut request = [0u8; 1024];
        let _ = stream.read(&mut request).await;
        let _ = stream
            .write_all(b"HTTP/1.1 200 OK\r\ncontent-length: 0\r\nconnection: close\r\n\r\n")
            .await;
    }
}

async fn start_service(fake: &FakeLlama, port: u16) -> (ChildServer, watch::Sender<bool>) {
    let (shutdown_tx, shutdown_rx) = watch::channel(false);
    let service = ChildServer::start(LlamaServerSpec::new(model_spec(fake, port)), shutdown_rx)
        .await
        .expect("service should start");
    (service, shutdown_tx)
}

/// Pid of the grandchild a `with-orphan` fake recorded beside its binary.
fn orphan_pid(fake: &FakeLlama) -> u32 {
    std::fs::read_to_string(fake.binary_path().with_file_name("orphan.pid"))
        .expect("fake wrote its orphan's pid")
        .trim()
        .parse()
        .expect("pid file holds a pid")
}

/// True once `pid` no longer exists or is a zombie awaiting its reaper.
fn process_is_gone(pid: u32) -> bool {
    match std::fs::read_to_string(format!("/proc/{pid}/stat")) {
        Err(_) => true,
        Ok(stat) => stat
            .rsplit(')')
            .next()
            .is_none_or(|fields| fields.trim_start().starts_with('Z')),
    }
}

async fn wait_until_process_is_gone(pid: u32) -> bool {
    tokio::time::timeout(Duration::from_secs(5), async {
        while !process_is_gone(pid) {
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    })
    .await
    .is_ok()
}

#[tokio::test]
async fn restarts_after_external_kill() {
    let fake = FakeLlama::new("normal");
    let port = grab_port().await;
    let (service, shutdown_tx) = start_service(&fake, port).await;

    let first_pid = service.pid().expect("first pid");
    let pid = i32::try_from(first_pid)
        .ok()
        .and_then(Pid::from_raw)
        .expect("valid pid");
    kill_process(pid, Signal::KILL).expect("SIGKILL on test child");

    let deadline = Instant::now() + Duration::from_secs(8);
    loop {
        assert!(
            Instant::now() < deadline,
            "supervisor did not restart llama-server within the deadline"
        );
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
async fn health_from_a_squatter_on_the_port_is_not_ready() {
    let squatter = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = squatter.local_addr().unwrap().port();
    let squatter_task = tokio::spawn(answer_every_request_with_ok(squatter));
    let fake = FakeLlama::new("normal");
    let (shutdown_tx, shutdown_rx) = watch::channel(false);

    let flip_tx = shutdown_tx.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_secs(3)).await;
        let _ = flip_tx.send(true);
    });

    let result =
        ChildServer::start(LlamaServerSpec::new(model_spec(&fake, port)), shutdown_rx).await;
    squatter_task.abort();

    let err = result.expect_err("a 200 from a foreign listener must not count as ready");
    assert!(
        matches!(err, ChildServerError::ShutdownDuringHealth),
        "{err:?}"
    );
}

#[tokio::test]
async fn crash_kills_the_crashed_servers_process_group() {
    let fake = FakeLlama::new("with-orphan");
    let port = grab_port().await;
    let (service, shutdown_tx) = start_service(&fake, port).await;
    let orphan = orphan_pid(&fake);
    fake.set_mode("normal");

    let leader = service
        .pid()
        .and_then(|pid| i32::try_from(pid).ok())
        .and_then(Pid::from_raw)
        .expect("running child");
    kill_process(leader, Signal::KILL).expect("SIGKILL on test child");

    assert!(
        wait_until_process_is_gone(orphan).await,
        "grandchild {orphan} must be killed with the crashed server's process group"
    );
    let _ = shutdown_tx.send(true);
    service.shutdown().await.unwrap();
}

#[tokio::test]
async fn shutdown_kills_grandchildren_that_ignore_sigterm() {
    let fake = FakeLlama::new("with-orphan");
    let port = grab_port().await;
    let (service, shutdown_tx) = start_service(&fake, port).await;
    let orphan = orphan_pid(&fake);

    let _ = shutdown_tx.send(true);
    service.shutdown().await.unwrap();

    assert!(
        wait_until_process_is_gone(orphan).await,
        "grandchild {orphan} must not outlive its server's shutdown"
    );
}
