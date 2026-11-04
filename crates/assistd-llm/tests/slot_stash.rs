//! End-to-end tests for stashing each conversation's slot state while others
//! use the slot, against a presence-managed `fake_llama_server`.

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
use assistd_ipc::TaskRunId;
use assistd_llm::{
    ConversationKey, LlamaChatClient, LlmBackend, LlmHealthProbe, SlotAssignment, SlotStash,
};

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

    fn slot_stash(&self) -> SlotStash {
        SlotStash::new(
            SocketAddr::new(Ipv4Addr::LOCALHOST.into(), self.port),
            Some(self.fake.api_key()),
            model_spec(&self.fake, self.port).name,
            self.fake.slot_dir(),
            Some(self.probe()),
        )
        .expect("slot stash")
    }

    fn chat_client(&self) -> LlamaChatClient {
        LlamaChatClient::new(
            &ChatConfig::default(),
            &model_spec(&self.fake, self.port),
            &TimeoutsConfig::default(),
            Some(self.fake.api_key()),
            Some(self.probe()),
        )
        .expect("chat client")
    }

    /// The files the child serving now saved and restored, in order.
    async fn slot_actions(&self) -> (Vec<String>, Vec<String>) {
        let url = format!("http://127.0.0.1:{}/debug/counters", self.port);
        let counters: Value = reqwest::get(&url)
            .await
            .expect("GET /debug/counters")
            .json()
            .await
            .expect("counters json");
        (
            file_names(&counters["saved"]),
            file_names(&counters["restored"]),
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

fn file_names(list: &Value) -> Vec<String> {
    list.as_array()
        .expect("a list of file names")
        .iter()
        .map(|name| name.as_str().expect("a file name").to_string())
        .collect()
}

fn names(files: &[&str]) -> Vec<String> {
    files.iter().map(ToString::to_string).collect()
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

async fn chat_request(client: &dyn LlmBackend) {
    let (tx, mut rx) = mpsc::channel(64);
    client
        .generate("task".into(), tx)
        .await
        .expect("chat request");
    while rx.recv().await.is_some() {}
}

#[tokio::test]
async fn a_cancelled_group_saves_once_before_and_restores_once_after() {
    let harness = Harness::start().await;
    let slot = harness.slot_stash();
    let client = harness.chat_client();
    slot.switch_to(ConversationKey::Main).await;

    let sub_agents = async {
        for _ in 0..2 {
            chat_request(&client).await;
            assert_eq!(
                harness.slot_actions().await,
                (names(&["main.bin"]), names(&[]))
            );
        }
        std::future::pending::<()>().await;
    };
    let outcome = slot
        .preserve_across(tokio::time::timeout(Duration::from_secs(2), sub_agents))
        .await;

    assert!(outcome.is_err(), "the group should have been cut off");
    assert_eq!(
        harness.slot_actions().await,
        (names(&["main.bin"]), names(&["main.bin"]))
    );
    assert!(
        !harness.has_slot_file(),
        "the saved slot outlived its group"
    );
}

#[tokio::test]
async fn a_restart_inside_the_group_skips_the_restore() {
    let harness = Harness::start().await;
    let slot = harness.slot_stash();
    slot.switch_to(ConversationKey::Main).await;

    let outcome = slot
        .preserve_across(async {
            assert_eq!(
                harness.slot_actions().await,
                (names(&["main.bin"]), names(&[]))
            );
            harness.restart_child().await;
            "finished"
        })
        .await;

    assert_eq!(outcome, "finished");
    assert_eq!(harness.slot_actions().await, (names(&[]), names(&[])));
}

#[tokio::test]
async fn a_dropped_group_is_restored_before_the_next_main_request() {
    let harness = Harness::start().await;
    let slot = Arc::new(harness.slot_stash());
    let main = harness.chat_client().with_slot_stash(Arc::clone(&slot));
    chat_request(&main).await;

    let dropped = tokio::time::timeout(
        Duration::from_secs(1),
        slot.preserve_across(std::future::pending::<()>()),
    )
    .await;
    assert!(dropped.is_err(), "the group should have been dropped");
    assert_eq!(
        harness.slot_actions().await,
        (names(&["main.bin"]), names(&[]))
    );

    chat_request(&main).await;
    assert_eq!(
        harness.slot_actions().await,
        (names(&["main.bin"]), names(&["main.bin"]))
    );
    assert!(
        !harness.has_slot_file(),
        "the saved slot outlived its restore"
    );
}

#[tokio::test]
async fn each_conversation_gets_back_its_own_slot_until_the_child_restarts() {
    let harness = Harness::start().await;
    let slot = Arc::new(harness.slot_stash());
    let main = harness.chat_client().with_slot_stash(Arc::clone(&slot));
    let run = main
        .isolated_conversation(
            "task prompt".into(),
            SlotAssignment::Shared(ConversationKey::TaskRun(TaskRunId(7))),
        )
        .expect("task run conversation");

    chat_request(&main).await;
    chat_request(run.as_ref()).await;
    chat_request(&main).await;
    chat_request(run.as_ref()).await;
    assert_eq!(
        harness.slot_actions().await,
        (
            names(&["main.bin", "task-run-7.bin", "main.bin"]),
            names(&["main.bin", "task-run-7.bin"])
        )
    );

    harness.restart_child().await;
    chat_request(&main).await;
    assert_eq!(
        harness.slot_actions().await,
        (names(&["task-run-7.bin"]), names(&[]))
    );
}
