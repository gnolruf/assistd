//! Integration tests for the slot stash against a hand-rolled in-process
//! server that records each slot action.

use std::fs;
use std::future::Future;
use std::io;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use serde_json::Value;
use tempfile::TempDir;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Notify;

use assistd_ipc::TaskRunId;
use assistd_llm::{ConversationKey, SlotDir, SlotStash};

const RUN: ConversationKey = ConversationKey::TaskRun(TaskRunId(1));

/// Records every slot action as `"<action> <filename>"` and answers it, a save
/// writing its file into `dir` first. The action named in `stalled` is held
/// until [`FakeSlots::release`].
#[derive(Clone)]
struct FakeSlots {
    dir: PathBuf,
    actions: Arc<Mutex<Vec<String>>>,
    stalled: Arc<Mutex<Option<&'static str>>>,
    stall_began: Arc<Notify>,
    released: Arc<Notify>,
}

impl FakeSlots {
    fn saving_into(dir: &TempDir) -> Self {
        Self {
            dir: dir.path().to_path_buf(),
            actions: Arc::default(),
            stalled: Arc::default(),
            stall_began: Arc::default(),
            released: Arc::default(),
        }
    }

    async fn spawn(&self) -> SocketAddr {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let fake = self.clone();
        tokio::spawn(async move {
            while let Ok((sock, _)) = listener.accept().await {
                let fake = fake.clone();
                tokio::spawn(async move {
                    let _ = fake.serve(sock).await;
                });
            }
        });
        addr
    }

    async fn serve(&self, mut sock: TcpStream) -> io::Result<()> {
        let (action, filename) = read_slot_action(&mut sock).await?;
        self.actions
            .lock()
            .unwrap()
            .push(format!("{action} {filename}"));
        let stalled = self
            .stalled
            .lock()
            .unwrap()
            .take_if(|stalled| *stalled == action);
        if stalled.is_some() {
            self.stall_began.notify_one();
            self.released.notified().await;
        }
        if action == "save" {
            fs::write(self.dir.join(&filename), b"slot state")?;
        }
        sock.write_all(
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}",
        )
        .await
    }

    /// Hold the next request for `action` until [`Self::release`].
    fn stall_next(&self, action: &'static str) {
        *self.stalled.lock().unwrap() = Some(action);
    }

    fn release(&self) {
        self.released.notify_one();
    }

    /// Drop `switch` once the server holds its stalled request.
    async fn drop_mid_request(&self, switch: impl Future<Output = ()>) {
        tokio::select! {
            () = switch => panic!("the switch finished though its request was stalled"),
            () = self.stall_began.notified() => {}
        }
    }

    fn actions(&self) -> Vec<String> {
        self.actions.lock().unwrap().clone()
    }
}

/// The `action` query value and body `filename` of one slot request.
async fn read_slot_action(sock: &mut TcpStream) -> io::Result<(String, String)> {
    let mut request = Vec::new();
    let body_start = loop {
        if let Some(end) = request.windows(4).position(|w| w == b"\r\n\r\n") {
            break end + 4;
        }
        request.push(sock.read_u8().await?);
    };
    let head = String::from_utf8_lossy(&request).into_owned();
    let content_length = head
        .lines()
        .filter_map(|line| line.split_once(':'))
        .find(|(name, _)| name.eq_ignore_ascii_case("content-length"))
        .and_then(|(_, value)| value.trim().parse().ok())
        .unwrap_or(0);
    request.resize(body_start + content_length, 0);
    sock.read_exact(&mut request[body_start..]).await?;
    let action = head
        .split_once("action=")
        .and_then(|(_, rest)| rest.split_whitespace().next())
        .unwrap_or_default()
        .to_string();
    let body: Value = serde_json::from_slice(&request[body_start..])?;
    Ok((action, body["filename"].as_str().unwrap_or_default().into()))
}

fn stash_at(addr: SocketAddr, dir: &TempDir) -> SlotStash {
    let slot_dir = SlotDir::new(dir.path().to_path_buf());
    SlotStash::new(addr, None, "test-model".into(), slot_dir, None).unwrap()
}

#[tokio::test]
async fn a_switch_whose_caller_is_gone_finishes_before_its_conversation_ends() {
    let dir = tempfile::tempdir().unwrap();
    let fake = FakeSlots::saving_into(&dir);
    let stash = stash_at(fake.spawn().await, &dir);
    stash.switch_to(RUN).await;

    fake.stall_next("save");
    fake.drop_mid_request(stash.switch_to(ConversationKey::Main))
        .await;
    tokio::join!(stash.end(RUN), async { fake.release() });

    assert!(!dir.path().join("task-run-1.bin").exists());
    stash.switch_to(RUN).await;
    assert_eq!(fake.actions(), ["save task-run-1.bin", "save main.bin"]);
}

#[tokio::test]
async fn ending_a_conversation_removes_a_slot_file_the_stash_has_no_record_of() {
    let dir = tempfile::tempdir().unwrap();
    let fake = FakeSlots::saving_into(&dir);
    let stash = stash_at(fake.spawn().await, &dir);
    let unrecorded = dir.path().join("task-run-1.bin");
    fs::write(&unrecorded, b"slot state").unwrap();

    stash.end(RUN).await;

    assert!(!unrecorded.exists());
}
