//! End-to-end tests for saving and restoring the main slot around a group of
//! sub-agent requests, against a presence-managed `fake_llama_server`.

#![cfg(feature = "test-support")]

use std::net::{Ipv4Addr, SocketAddr};
use std::num::NonZeroU16;
use std::sync::Arc;
use std::time::{Duration, Instant};

use rustix::process::{Pid, Signal, kill_process};
use serde_json::Value;
use tokio::net::TcpListener;
use tokio::sync::{mpsc, watch};

use assistd_config::defaults::{nz32, nz64};
use assistd_config::{ChatConfig, ModelConfig, TimeoutsConfig};
use assistd_core::PresenceManager;
use assistd_core::presence::PresenceLlmHealthProbe;
use assistd_llm::{LlamaChatClient, LlmBackend, LlmHealthProbe, MainSlot};

use common::FakeLlama;

mod common;

struct Harness {
    fake: FakeLlama,
    port: u16,
    manager: Arc<PresenceManager>,
    _shutdown: watch::Sender<bool>,
}

impl Harness {
    async fn start() -> Self {
        let fake = FakeLlama::new("normal");
        let port = grab_port().await;
        let (shutdown, shutdown_rx) = watch::channel(false);
        let manager = PresenceManager::new_active(
            model_spec(&fake, port),
            TimeoutsConfig::default(),
            fake.api_key().clone(),
            fake.slot_dir(),
            shutdown_rx,
        )
        .await
        .expect("presence manager");
        Self {
            fake,
            port,
            manager,
            _shutdown: shutdown,
        }
    }

    fn probe(&self) -> Arc<dyn LlmHealthProbe> {
        Arc::new(PresenceLlmHealthProbe::new(Arc::clone(&self.manager)))
    }

    fn main_slot(&self) -> MainSlot {
        MainSlot::new(
            SocketAddr::new(Ipv4Addr::LOCALHOST.into(), self.port),
            Some(self.fake.api_key()),
            model_spec(&self.fake, self.port).name,
            self.fake.slot_dir(),
            Some(self.probe()),
        )
        .expect("main slot")
    }

    fn sub_agent_client(&self) -> LlamaChatClient {
        LlamaChatClient::new(
            &ChatConfig::default(),
            &model_spec(&self.fake, self.port),
            &TimeoutsConfig::default(),
            Some(self.fake.api_key()),
            Some(self.probe()),
        )
        .expect("chat client")
    }

    /// `(save_count, restore_count)` of the child serving now.
    async fn slot_actions(&self) -> (u64, u64) {
        let url = format!("http://127.0.0.1:{}/debug/counters", self.port);
        let counters: Value = reqwest::get(&url)
            .await
            .expect("GET /debug/counters")
            .json()
            .await
            .expect("counters json");
        (
            counters["save_count"].as_u64().expect("save_count"),
            counters["restore_count"].as_u64().expect("restore_count"),
        )
    }

    fn has_slot_file(&self) -> bool {
        std::fs::read_dir(self.fake.slot_dir().path())
            .expect("slot dir exists")
            .next()
            .is_some()
    }

    async fn restart_child(&self) {
        let old = self.manager.llama_pid().expect("child running");
        let pid = i32::try_from(old)
            .ok()
            .and_then(Pid::from_raw)
            .expect("valid pid");
        kill_process(pid, Signal::KILL).expect("SIGKILL the llama-server child");
        let deadline = Instant::now() + Duration::from_secs(10);
        while !self
            .manager
            .llama_health()
            .is_some_and(|health| health.is_serving() && health.pid != Some(old))
        {
            assert!(Instant::now() < deadline, "supervisor never restarted");
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }
}

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

async fn sub_agent_request(client: &LlamaChatClient) {
    let (tx, mut rx) = mpsc::channel(64);
    client
        .generate("sub-agent task".into(), tx)
        .await
        .expect("sub-agent request");
    while rx.recv().await.is_some() {}
}

#[tokio::test]
async fn a_cancelled_group_saves_once_before_and_restores_once_after() {
    let harness = Harness::start().await;
    let slot = harness.main_slot();
    let client = harness.sub_agent_client();

    let sub_agents = async {
        for _ in 0..2 {
            sub_agent_request(&client).await;
            assert_eq!(harness.slot_actions().await, (1, 0));
        }
        std::future::pending::<()>().await;
    };
    let outcome = slot
        .preserve_across(tokio::time::timeout(Duration::from_secs(2), sub_agents))
        .await;

    assert!(outcome.is_err(), "the group should have been cut off");
    assert_eq!(harness.slot_actions().await, (1, 1));
    assert!(
        !harness.has_slot_file(),
        "the saved slot outlived its group"
    );
}

#[tokio::test]
async fn a_restart_inside_the_group_skips_the_restore() {
    let harness = Harness::start().await;
    let slot = harness.main_slot();

    let outcome = slot
        .preserve_across(async {
            assert_eq!(harness.slot_actions().await, (1, 0));
            harness.restart_child().await;
            "finished"
        })
        .await;

    assert_eq!(outcome, "finished");
    assert_eq!(harness.slot_actions().await, (0, 0));
}
