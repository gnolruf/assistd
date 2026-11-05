//! The private directory llama-server saves and restores slot state in, and
//! the stash that keeps each conversation's slot state while others use the slot.

use std::fs::{self, DirBuilder};
use std::future::Future;
use std::io::{self, ErrorKind};
use std::net::SocketAddr;
use std::num::NonZeroU32;
use std::os::unix::fs::DirBuilderExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex, watch};
use tracing::{debug, warn};

use assistd_ipc::{TaskId, TaskRunId};
use assistd_utils::child_server::ApiKey;
use assistd_utils::http::{local_client_builder, read_capped};
use assistd_utils::text::human_size;

use super::error::LlamaServerError;
use crate::LlmHealthProbe;

const OWNER_ONLY: u32 = 0o700;
/// The main conversation's slot, the only one with `-np 1`.
const SLOT_PATH: &str = "/slots/0";
/// Task conversations whose slot files are kept beside the main conversation's.
const MAX_SUSPENDED_RUNS: usize = 2;
const SLOT_ACTION_TIMEOUT: Duration = Duration::from_secs(60);
const SLOT_RESPONSE_MAX: usize = 64 * 1024;

/// A mode-0700 directory passed as `--slot-save-path`, emptied before every
/// server spawn since a slot file is only valid for the model and settings
/// that wrote it.
#[derive(Debug, Clone)]
pub struct SlotDir {
    path: Arc<Path>,
}

impl SlotDir {
    /// Use `path` as the slot directory; nothing is touched until [`Self::reset`].
    pub fn new(path: PathBuf) -> Self {
        Self { path: path.into() }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Replace whatever is at the path with an empty owner-only directory.
    pub fn reset(&self) -> io::Result<()> {
        if let Err(e) = fs::remove_dir_all(&self.path)
            && e.kind() != ErrorKind::NotFound
        {
            return Err(e);
        }
        DirBuilder::new()
            .recursive(true)
            .mode(OWNER_ONLY)
            .create(&self.path)
    }
}

/// A conversation whose slot state the [`SlotStash`] keeps while another uses the slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConversationKey {
    Main,
    TaskRun(TaskRunId),
    /// The planning conversation of a saved task.
    TaskPlan(TaskId),
}

impl ConversationKey {
    fn file_name(self) -> String {
        match self {
            Self::Main => "main.bin".to_string(),
            Self::TaskRun(run) => format!("task-run-{run}.bin"),
            Self::TaskPlan(task) => format!("task-plan-{task}.bin"),
        }
    }
}

/// Which of the chat server's slots a conversation's requests use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotAssignment {
    /// Whichever slot the server picks; nothing is kept for the conversation.
    Unassigned,
    /// The main conversation's slot, the conversation's state stashed under this key
    /// while another uses it.
    Shared(ConversationKey),
    /// The slot with this index, which no other conversation is given.
    Pinned(u32),
}

/// Keeps each conversation's state of the main conversation's slot in a file of
/// its own while other conversations use the slot, so switching back skips
/// re-processing its context. Every failure is logged and costs only that.
#[derive(Debug)]
pub struct SlotStash {
    client: reqwest::Client,
    url: String,
    model: String,
    slot_dir: SlotDir,
    health: Option<Arc<dyn LlmHealthProbe>>,
    /// The server's slot count; one until told otherwise.
    slots: watch::Receiver<NonZeroU32>,
    state: Mutex<StashState>,
}

impl SlotStash {
    /// A stash for `model`'s slot on the router at `addr`, whose files land in
    /// `slot_dir`. With `health`, a restore is skipped once the serving child
    /// has changed since the save.
    pub fn new(
        addr: SocketAddr,
        api_key: Option<&ApiKey>,
        model: String,
        slot_dir: SlotDir,
        health: Option<Arc<dyn LlmHealthProbe>>,
    ) -> Result<Self, LlamaServerError> {
        let client = local_client_builder()
            .default_headers(
                api_key
                    .map(ApiKey::authorization_headers)
                    .unwrap_or_default(),
            )
            .timeout(SLOT_ACTION_TIMEOUT)
            .build()?;
        Ok(Self {
            client,
            url: format!("http://{addr}{SLOT_PATH}"),
            model,
            slot_dir,
            health,
            slots: watch::channel(NonZeroU32::MIN).1,
            state: Mutex::new(StashState::default()),
        })
    }

    /// Follow `slots`, the server's slot count, instead of assuming one slot.
    pub fn with_slot_count(mut self, slots: watch::Receiver<NonZeroU32>) -> Self {
        self.slots = slots;
        self
    }

    /// Hand the slot to `key`: save it for the conversation holding it, then
    /// restore `key`'s saved state, if any. Call before every request `key` makes.
    pub async fn switch_to(&self, key: ConversationKey) {
        let mut state = self.state.lock().await;
        if state.occupant == Some(key) {
            return;
        }
        self.set_aside(&mut state).await;
        if let Some(saved) = state.take(key) {
            self.restore(key, saved).await;
            self.remove_file(key).await;
        }
        state.occupant = Some(key);
    }

    /// Run `group` of unkeyed requests between a save and a restore of the
    /// conversation holding the slot. If this future is dropped, that
    /// conversation's next [`Self::switch_to`] restores it. Hold the agent turn
    /// lock across the call so no other conversation lands in between. With more
    /// than one slot the main conversation keeps its slot, so `group` just runs.
    pub async fn preserve_across<F: Future>(&self, group: F) -> F::Output {
        if self.slots.borrow().get() > 1 {
            return group.await;
        }
        let owner = {
            let mut state = self.state.lock().await;
            let owner = state.occupant;
            self.set_aside(&mut state).await;
            owner
        };
        let output = group.await;
        if let Some(owner) = owner {
            self.switch_to(owner).await;
        }
        output
    }

    /// Forget `key`'s conversation, which has ended, deleting its saved state.
    pub async fn end(&self, key: ConversationKey) {
        let mut state = self.state.lock().await;
        if state.occupant == Some(key) {
            state.occupant = None;
        }
        if state.take(key).is_some() {
            self.remove_file(key).await;
        }
    }

    /// Save the slot for its occupant, which then no longer holds it, evicting
    /// the oldest suspended task conversation past [`MAX_SUSPENDED_RUNS`].
    async fn set_aside(&self, state: &mut StashState) {
        let Some(owner) = state.occupant.take() else {
            return;
        };
        if let Some(saved) = self.save(owner).await {
            state.saved.push((owner, saved));
        }
        if let Some(evicted) = state.evict_past_limit() {
            debug!(target: "assistd::slots", ?evicted, "dropped the oldest suspended slot");
            self.remove_file(evicted).await;
        }
    }

    async fn save(&self, key: ConversationKey) -> Option<SavedSlot> {
        let Ok(pid) = self.serving_pid().await else {
            warn!(target: "assistd::slots", ?key, "llama-server is not serving; slot not saved");
            return None;
        };
        let started = Instant::now();
        match self.post_action("save", key).await {
            Ok(response) => {
                debug!(
                    target: "assistd::slots",
                    ?key,
                    tokens = response.n_saved.unwrap_or_default(),
                    size = %human_size(response.n_written.unwrap_or_default()),
                    elapsed = ?started.elapsed(),
                    "saved slot"
                );
                Some(SavedSlot { pid })
            }
            Err(e) => {
                warn!(target: "assistd::slots", ?key, "saving the slot failed: {e}");
                None
            }
        }
    }

    async fn restore(&self, key: ConversationKey, saved: SavedSlot) {
        if self.serving_pid().await.ok() != Some(saved.pid) {
            warn!(
                target: "assistd::slots",
                ?key,
                "llama-server restarted since the slot was saved; restore skipped"
            );
            return;
        }
        let started = Instant::now();
        match self.post_action("restore", key).await {
            Ok(response) => debug!(
                target: "assistd::slots",
                ?key,
                tokens = response.n_restored.unwrap_or_default(),
                size = %human_size(response.n_read.unwrap_or_default()),
                elapsed = ?started.elapsed(),
                "restored slot"
            ),
            Err(e) => warn!(target: "assistd::slots", ?key, "restoring the slot failed: {e}"),
        }
    }

    async fn remove_file(&self, key: ConversationKey) {
        let path = self.slot_dir.path().join(key.file_name());
        if let Err(e) = tokio::fs::remove_file(&path).await
            && e.kind() != ErrorKind::NotFound
        {
            warn!(
                target: "assistd::slots",
                "removing the saved slot {} failed: {e}",
                path.display()
            );
        }
    }

    /// The serving child's pid, `None` without a probe; `Err` when the
    /// probed child is not serving.
    async fn serving_pid(&self) -> Result<Option<u32>, LlamaServerError> {
        let Some(probe) = &self.health else {
            return Ok(None);
        };
        match probe.snapshot().await {
            Some(snapshot) if snapshot.is_serving() => Ok(snapshot.pid),
            _ => Err(LlamaServerError::NotReady),
        }
    }

    async fn post_action(
        &self,
        action: &str,
        key: ConversationKey,
    ) -> Result<SlotActionResponse, LlamaServerError> {
        let filename = key.file_name();
        let body = SlotActionRequest {
            model: &self.model,
            filename: &filename,
        };
        let resp = self
            .client
            .post(&self.url)
            .query(&[("action", action)])
            .json(&body)
            .send()
            .await?;
        let status = resp.status();
        if !status.is_success() {
            return Err(LlamaServerError::ControlHttp {
                method: "POST",
                path: SLOT_PATH,
                status: status.as_u16(),
            });
        }
        let body = read_capped(resp, SLOT_RESPONSE_MAX).await?;
        Ok(serde_json::from_slice(&body)?)
    }
}

/// Which conversation holds the slot, `None` while unkeyed requests use it, and
/// the saved state of those set aside, oldest first; a key is held or saved, never both.
#[derive(Debug, Default)]
struct StashState {
    occupant: Option<ConversationKey>,
    saved: Vec<(ConversationKey, SavedSlot)>,
}

impl StashState {
    fn take(&mut self, key: ConversationKey) -> Option<SavedSlot> {
        let index = self.saved.iter().position(|(saved, _)| *saved == key)?;
        Some(self.saved.remove(index).1)
    }

    fn evict_past_limit(&mut self) -> Option<ConversationKey> {
        let mut runs = self
            .saved
            .iter()
            .enumerate()
            .filter(|(_, (key, _))| *key != ConversationKey::Main);
        let (oldest, _) = runs.next()?;
        (runs.count() >= MAX_SUSPENDED_RUNS).then(|| self.saved.remove(oldest).0)
    }
}

/// A slot saved while the child with `pid` was serving.
#[derive(Debug, Clone, Copy)]
struct SavedSlot {
    pid: Option<u32>,
}

/// The router picks the model child from `model`.
#[derive(Serialize)]
struct SlotActionRequest<'a> {
    model: &'a str,
    filename: &'a str,
}

#[derive(Deserialize)]
struct SlotActionResponse {
    #[serde(default)]
    n_saved: Option<u64>,
    #[serde(default)]
    n_written: Option<u64>,
    #[serde(default)]
    n_restored: Option<u64>,
    #[serde(default)]
    n_read: Option<u64>,
}
