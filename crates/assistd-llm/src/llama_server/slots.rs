//! The private directory llama-server saves and restores slot state in, and
//! the stash that keeps each holder's slot state while others use the slot.

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
use tokio::sync::{mpsc, oneshot, watch};
use tokio_util::task::AbortOnDropHandle;
use tracing::{debug, warn};

use assistd_ipc::TaskRunId;
use assistd_utils::child_server::ApiKey;
use assistd_utils::http::{local_client_builder, read_capped};
use assistd_utils::text::human_size;

use super::error::LlamaServerError;
use crate::LlmHealthProbe;

const OWNER_ONLY: u32 = 0o700;
/// The main holder's slot, the only one with `-np 1`.
const SLOT_PATH: &str = "/slots/0";
/// Task runs whose slot files are kept beside the main holder's.
const MAX_SUSPENDED_RUNS: usize = 2;
const SLOT_ACTION_TIMEOUT: Duration = Duration::from_secs(60);
const SLOT_RESPONSE_MAX: usize = 64 * 1024;
const TRANSITION_QUEUE_CAPACITY: usize = 8;

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

/// A backend whose slot state the [`SlotStash`] keeps while another uses the slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SlotHolder {
    Main,
    TaskRun(TaskRunId),
    /// The planning turn refining a task prompt.
    TaskRefine,
}

impl SlotHolder {
    fn file_name(self) -> String {
        match self {
            Self::Main => "main.bin".to_string(),
            Self::TaskRun(run) => format!("task-run-{run}.bin"),
            Self::TaskRefine => "task-refine.bin".to_string(),
        }
    }
}

/// Which of the model server's slots a backend's requests use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotAssignment {
    /// Whichever slot the server picks; nothing is kept for the backend.
    Unassigned,
    /// The main holder's slot, its state stashed for this holder while another uses it.
    Shared(SlotHolder),
    /// The slot with this index, which no other backend is given.
    Pinned(u32),
}

/// Keeps each holder's state of the main holder's slot in a file of
/// its own while other holders use the slot, so switching back skips
/// re-processing its context. Every failure is logged and costs only that.
#[derive(Debug)]
pub struct SlotStash {
    /// The server's slot count; one until told otherwise.
    slots: watch::Receiver<NonZeroU32>,
    transitions: mpsc::Sender<QueuedTransition>,
    _worker: AbortOnDropHandle<()>,
}

impl SlotStash {
    /// A stash for `model`'s slot on the router at `addr`, whose files land in
    /// `slot_dir`. With `health`, a restore is skipped once the serving child
    /// has changed since the save. Call inside a Tokio runtime.
    pub fn new(
        addr: SocketAddr,
        api_key: Option<&ApiKey>,
        model: String,
        slot_dir: SlotDir,
        health: Option<Arc<dyn LlmHealthProbe>>,
    ) -> Result<Self, LlamaServerError> {
        let client = local_client_builder()
            .default_headers(api_key.map_or_default(ApiKey::authorization_headers))
            .timeout(SLOT_ACTION_TIMEOUT)
            .build()?;
        let worker = SlotWorker {
            client,
            url: format!("http://{addr}{SLOT_PATH}"),
            model,
            slot_dir,
            health,
            state: StashState::default(),
        };
        let (transitions, queued) = mpsc::channel(TRANSITION_QUEUE_CAPACITY);
        Ok(Self {
            slots: watch::channel(NonZeroU32::MIN).1,
            transitions,
            _worker: AbortOnDropHandle::new(tokio::spawn(worker.run(queued))),
        })
    }

    /// Follow `slots`, the server's slot count, instead of assuming one slot.
    pub fn with_slot_count(mut self, slots: watch::Receiver<NonZeroU32>) -> Self {
        self.slots = slots;
        self
    }

    /// Hand the slot to `holder`: save it for whoever holds it, then
    /// restore `holder`'s saved state, if any. Call before every request `holder` makes.
    /// A switch that has begun finishes even if this future is dropped.
    pub async fn switch_to(&self, holder: SlotHolder) {
        self.apply(Transition::SwitchTo(holder)).await;
    }

    /// Run `group` of unkeyed requests between a save and a restore of the
    /// slot's holder. If this future is dropped, that holder's next
    /// [`Self::switch_to`] restores it. Hold the agent turn lock across the
    /// call so no other holder lands in between. With more than one slot the
    /// main holder keeps its slot, so `group` just runs.
    pub async fn preserve_across<F: Future>(&self, group: F) -> F::Output {
        if self.slots.borrow().get() > 1 {
            return group.await;
        }
        let owner = self.apply(Transition::SetAside).await;
        let output = group.await;
        if let Some(owner) = owner {
            self.switch_to(owner).await;
        }
        output
    }

    /// Forget `holder`, whose work has ended, deleting its saved state
    /// once every transition begun earlier has finished.
    pub async fn end(&self, holder: SlotHolder) {
        self.apply(Transition::End(holder)).await;
    }

    /// Queue `transition` behind those already begun and wait for it, returning
    /// whoever held the slot before it.
    async fn apply(&self, transition: Transition) -> Option<SlotHolder> {
        let (done, applied) = oneshot::channel();
        let queued = QueuedTransition { transition, done };
        self.transitions.send(queued).await.ok()?;
        applied.await.ok().flatten()
    }
}

#[derive(Debug)]
enum Transition {
    SwitchTo(SlotHolder),
    /// Save the slot for its occupant and leave it to unkeyed requests.
    SetAside,
    End(SlotHolder),
}

#[derive(Debug)]
struct QueuedTransition {
    transition: Transition,
    /// Sent whoever held the slot before the transition.
    done: oneshot::Sender<Option<SlotHolder>>,
}

/// Applies the stash's transitions one at a time on a task of its own, so each
/// runs to its end whether or not its caller still waits for it.
struct SlotWorker {
    client: reqwest::Client,
    url: String,
    model: String,
    slot_dir: SlotDir,
    health: Option<Arc<dyn LlmHealthProbe>>,
    state: StashState,
}

impl SlotWorker {
    async fn run(mut self, mut queued: mpsc::Receiver<QueuedTransition>) {
        while let Some(QueuedTransition { transition, done }) = queued.recv().await {
            let previous_occupant = self.state.occupant;
            match transition {
                Transition::SwitchTo(holder) => self.switch_to(holder).await,
                Transition::SetAside => self.set_aside().await,
                Transition::End(holder) => self.end(holder).await,
            }
            let _ = done.send(previous_occupant);
        }
    }

    async fn switch_to(&mut self, holder: SlotHolder) {
        if self.state.occupant == Some(holder) {
            return;
        }
        self.set_aside().await;
        if let Some(saved) = self.state.saved_for(holder) {
            self.restore(holder, saved).await;
            self.state.forget(holder);
        }
        self.state.occupant = Some(holder);
        self.remove_file(holder).await;
    }

    async fn end(&mut self, holder: SlotHolder) {
        if self.state.occupant == Some(holder) {
            self.state.occupant = None;
        }
        self.state.forget(holder);
        self.remove_file(holder).await;
    }

    /// Save the slot for its occupant, which holds it until the save returns,
    /// evicting the oldest suspended task run past [`MAX_SUSPENDED_RUNS`].
    async fn set_aside(&mut self) {
        let Some(owner) = self.state.occupant else {
            return;
        };
        let saved = self.save(owner).await;
        self.state.occupant = None;
        self.state.saved.extend(saved.map(|saved| (owner, saved)));
        if let Some(evicted) = self.state.evict_past_limit() {
            debug!(target: "assistd::slots", ?evicted, "dropped the oldest suspended slot");
            self.remove_file(evicted).await;
        }
    }

    async fn save(&self, holder: SlotHolder) -> Option<SavedSlot> {
        let Ok(pid) = self.serving_pid().await else {
            warn!(target: "assistd::slots", ?holder, "llama-server is not serving; slot not saved");
            return None;
        };
        let started = Instant::now();
        match self.post_action("save", holder).await {
            Ok(response) => {
                debug!(
                    target: "assistd::slots",
                    ?holder,
                    tokens = response.n_saved.unwrap_or_default(),
                    size = %human_size(response.n_written.unwrap_or_default()),
                    elapsed = ?started.elapsed(),
                    "saved slot"
                );
                Some(SavedSlot { pid })
            }
            Err(e) => {
                warn!(target: "assistd::slots", ?holder, "saving the slot failed: {e}");
                None
            }
        }
    }

    async fn restore(&self, holder: SlotHolder, saved: SavedSlot) {
        if self.serving_pid().await.ok() != Some(saved.pid) {
            warn!(
                target: "assistd::slots",
                ?holder,
                "llama-server restarted since the slot was saved; restore skipped"
            );
            return;
        }
        let started = Instant::now();
        match self.post_action("restore", holder).await {
            Ok(response) => debug!(
                target: "assistd::slots",
                ?holder,
                tokens = response.n_restored.unwrap_or_default(),
                size = %human_size(response.n_read.unwrap_or_default()),
                elapsed = ?started.elapsed(),
                "restored slot"
            ),
            Err(e) => warn!(target: "assistd::slots", ?holder, "restoring the slot failed: {e}"),
        }
    }

    async fn remove_file(&self, holder: SlotHolder) {
        let path = self.slot_dir.path().join(holder.file_name());
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
        holder: SlotHolder,
    ) -> Result<SlotActionResponse, LlamaServerError> {
        let filename = holder.file_name();
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

/// Who holds the slot, `None` while unkeyed requests use it, and the saved state
/// of those set aside, oldest first; a holder has the slot or a saved state, never both.
#[derive(Debug, Default)]
struct StashState {
    occupant: Option<SlotHolder>,
    saved: Vec<(SlotHolder, SavedSlot)>,
}

impl StashState {
    fn saved_for(&self, holder: SlotHolder) -> Option<SavedSlot> {
        self.saved
            .iter()
            .find_map(|(saved_holder, saved)| (*saved_holder == holder).then_some(*saved))
    }

    fn forget(&mut self, holder: SlotHolder) {
        self.saved
            .retain(|(saved_holder, _)| *saved_holder != holder);
    }

    fn evict_past_limit(&mut self) -> Option<SlotHolder> {
        let mut runs = self
            .saved
            .iter()
            .enumerate()
            .filter(|(_, (holder, _))| *holder != SlotHolder::Main);
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
