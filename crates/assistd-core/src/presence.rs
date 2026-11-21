//! Daemon presence state machine (`Active`, `Drowsy`, `Sleeping`), which
//! frees GPU resources on demand and auto-wakes when a query arrives.

use std::fmt;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use parking_lot::Mutex as StdMutex;
use thiserror::Error;
use tokio::sync::{
    Mutex as AsyncMutex, MutexGuard as AsyncMutexGuard, OwnedRwLockReadGuard, RwLock, oneshot,
    watch,
};
use tokio::time::timeout;
use tokio_util::sync::CancellationToken;
use tokio_util::task::{AbortOnDropHandle, TaskTracker};
use tracing::{Instrument, debug, info, warn};

use crate::bus::EventSink;

use assistd_config::{ModelConfig, TimeoutsConfig};
use assistd_ipc::{
    Component, DownloadProgress, Event, EventBody, PresenceState, PresenceTarget, StatusKind,
    StatusSeverity,
};
use assistd_llm::{
    HealthSnapshot, HealthWaitError, LlamaServerControl, LlamaServerError, LlamaServerSpec,
    LlmHealthProbe, ModelPresets, ReadyState, SlotDir,
};
use assistd_utils::child_server::{ApiKey, ChildServer, ChildServerError};
use assistd_utils::time::saturating_millis;

use crate::recovery::spawn_supervised;

const GUARD_ACQUIRE_RETRIES: usize = 3;
const LOAD_POLL_INTERVAL: Duration = Duration::from_millis(200);
const LOAD_PROGRESS_INTERVAL: Duration = Duration::from_secs(3);
const DOWNLOAD_PUBLISH_INTERVAL: Duration = Duration::from_secs(1);
const TASK_DRAIN_GRACE: Duration = Duration::from_secs(5);

type InnerShutdownSlot = Arc<StdMutex<Option<watch::Sender<bool>>>>;

/// Why a presence transition, or a request guard waiting on one, failed.
#[derive(Debug, Error)]
pub enum PresenceError {
    #[error("failed to construct llama-server control client: {0}")]
    Control(#[source] LlamaServerError),

    #[error("cannot drowse from Sleeping: call wake() first")]
    DrowseFromSleeping,

    #[error("llama-server cold-start failed during wake: {0}")]
    ColdStart(#[source] ChildServerError),

    #[error("the model could not be downloaded: {0}")]
    Download(#[source] LlamaServerError),

    #[error("llama-server /models/load failed for {model}: {source}")]
    Load {
        model: String,
        #[source]
        source: LlamaServerError,
    },

    #[error("llama-server did not finish loading {model} within the {secs}s backstop: {source}")]
    LoadWait {
        model: String,
        secs: u64,
        #[source]
        source: LlamaServerError,
    },

    /// The supervisor restarted or gave up on the child mid-download or mid-load.
    #[error("llama-server left the ready state while bringing up {model}")]
    LeftReady { model: String },

    #[error("llama-server /models/unload failed for {model}: {source}")]
    Unload {
        model: String,
        #[source]
        source: LlamaServerError,
    },

    #[error("llama-server /models/unload timed out after {secs}s during drowse")]
    UnloadTimeout { secs: u64 },

    #[error("llama-server shutdown failed: {0}")]
    Shutdown(#[source] ChildServerError),

    /// The child may outlive the daemon.
    #[error("llama-server shutdown timed out after {secs}s")]
    ShutdownTimeout { secs: u64 },

    #[error("wake task panicked")]
    WakeTaskPanicked,

    /// Sleep/wake churn kept the daemon out of `Active` on every retry.
    #[error("failed to acquire active request guard after {attempts} retries")]
    GuardRetriesExhausted { attempts: usize },

    #[error(transparent)]
    NoRoom(#[from] VramShortfall),

    #[error("the daemon is not Drowsy with its llama-server serving")]
    NotDrowsy,

    #[error("another model already borrows the drowsy llama-server")]
    AlreadyLent,

    #[error("the daemon is shutting down")]
    ShuttingDown,
}

/// Loading the model now would leave less free VRAM than the configured headroom.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[error("not enough free VRAM to load the model: {free_mb} MiB free, {needed_mb} MiB needed")]
pub struct VramShortfall {
    pub free_mb: u64,
    pub needed_mb: u64,
}

/// A model loaded on the llama-server in the main model's place while the daemon
/// is `Drowsy`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BorrowedModel {
    /// The Hugging Face name the server downloads it by.
    pub source: String,
    /// The name the server loads and serves it under.
    pub served_as: String,
}

/// Says, before a wake loads the model, whether the GPU has room for it.
pub trait WakeRoom: fmt::Debug + Send + Sync {
    /// `Err` when the model would not fit now. `llama_pid` is the running
    /// llama-server, whose VRAM already counts toward the model.
    fn check(&self, llama_pid: Option<u32>) -> Result<(), VramShortfall>;
}

/// Owner of the llama-server handle and the daemon-wide presence state.
/// Transitions are serialised, so an auto-wake cannot race an explicit
/// `sleep`.
#[derive(Debug)]
pub struct PresenceManager {
    /// The state the last transition left; `state_tx` overlays `Waking`.
    /// Never held across an await.
    lifecycle: StdMutex<Lifecycle>,
    transition: AsyncMutex<()>,
    model: ModelConfig,
    timeouts: TimeoutsConfig,
    api_key: ApiKey,
    slot_dir: SlotDir,
    presets: ModelPresets,
    control: LlamaServerControl,
    /// The shutdown forwarder and detached wakes, joined by [`Self::shutdown`].
    tasks: TaskTracker,
    /// Flipped by `sleep()` to stop the current supervisor only.
    current_inner_shutdown: InnerShutdownSlot,
    state_tx: watch::Sender<PresenceState>,
    /// Once flipped, no wake starts; holding it keeps the watch open for the forwarder.
    daemon_shutdown: watch::Receiver<bool>,
    /// Last user-initiated interaction; automatic monitors don't bump it.
    last_activity: StdMutex<Instant>,
    /// Read by request guards, written by `sleep`/`drowse` to drain them.
    /// Writer-preferring, so queued requests cannot starve a transition.
    inflight: Arc<RwLock<()>>,
    /// `Some(started_at)` while a `wake` is executing.
    wake_started: StdMutex<Option<Instant>>,
    /// Run before every wake; unset when VRAM cannot be measured.
    wake_room: OnceLock<Arc<dyn WakeRoom>>,
    /// Whether the daemon sleeps because another process needs the GPU.
    gpu_hold: watch::Sender<bool>,
    /// Held while a [`BorrowedModel`] is on the drowsy server; `wake` and `sleep`
    /// cancel `borrower` and take this before they load or stop anything.
    loan: AsyncMutex<()>,
    borrower: StdMutex<CancellationToken>,
}

impl PresenceManager {
    /// Create a manager in `Sleeping` with no llama-server started. Every server it starts
    /// accepts only `api_key`, saves slots only into `slot_dir` and reads its models'
    /// presets from `presets`. Flipping `daemon_shutdown` cancels any wake in flight
    /// and fails every later one with [`PresenceError::ShuttingDown`].
    pub fn new_sleeping(
        model: ModelConfig,
        timeouts: TimeoutsConfig,
        api_key: ApiKey,
        slot_dir: SlotDir,
        presets: ModelPresets,
        daemon_shutdown: watch::Receiver<bool>,
    ) -> Result<Arc<Self>, PresenceError> {
        let control = LlamaServerControl::new(model.listen_addr(), Some(&api_key), None)
            .map_err(PresenceError::Control)?;

        let current_inner_shutdown: InnerShutdownSlot = Arc::new(StdMutex::new(None));
        let tasks = TaskTracker::new();
        spawn_shutdown_forwarder(
            &tasks,
            daemon_shutdown.clone(),
            Arc::clone(&current_inner_shutdown),
        );

        let (state_tx, _) = watch::channel(PresenceState::Sleeping);
        Ok(Arc::new(Self {
            lifecycle: StdMutex::new(Lifecycle::Sleeping),
            transition: AsyncMutex::new(()),
            model,
            timeouts,
            api_key,
            slot_dir,
            presets,
            control,
            tasks,
            current_inner_shutdown,
            state_tx,
            daemon_shutdown,
            last_activity: StdMutex::new(Instant::now()),
            inflight: Arc::new(RwLock::new(())),
            wake_started: StdMutex::new(None),
            wake_room: OnceLock::new(),
            gpu_hold: watch::channel(false).0,
            loan: AsyncMutex::new(()),
            borrower: StdMutex::new(CancellationToken::new()),
        }))
    }

    /// Check `room` before every wake from now on; a second call is ignored.
    pub fn set_wake_room(&self, room: Arc<dyn WakeRoom>) {
        let _ = self.wake_room.set(room);
    }

    /// Record whether the daemon sleeps because another process needs the GPU.
    pub fn hold_for_gpu(&self, held: bool) {
        self.gpu_hold.send_if_modified(|current| {
            let changed = *current != held;
            *current = held;
            changed
        });
    }

    /// Watch whether the daemon sleeps because another process needs the GPU.
    pub fn gpu_hold(&self) -> watch::Receiver<bool> {
        self.gpu_hold.subscribe()
    }

    /// The current presence state; `Waking`, or `Downloading` with how far
    /// the model's files have got, while a wake runs.
    pub fn state(&self) -> PresenceState {
        *self.state_tx.borrow()
    }

    fn settled(&self) -> PresenceTarget {
        self.lifecycle.lock().target()
    }

    fn mark_activity(&self) {
        *self.last_activity.lock() = Instant::now();
    }

    fn publish_settled(&self) {
        self.state_tx.send_replace(self.settled().into());
    }

    /// Time since the last user-initiated interaction.
    pub fn idle_duration(&self) -> Duration {
        self.last_activity.lock().elapsed()
    }

    /// Subscribe to presence-state changes, starting at the current state.
    pub fn subscribe(&self) -> watch::Receiver<PresenceState> {
        self.state_tx.subscribe()
    }

    /// PID of the managed llama-server child, or `None` if `Sleeping` or
    /// not yet spawned.
    pub fn llama_pid(&self) -> Option<u32> {
        self.lifecycle.lock().server().and_then(ChildServer::pid)
    }

    /// Whether a llama-server child is attached and serving.
    pub fn llama_serving(&self) -> bool {
        self.lifecycle
            .lock()
            .server()
            .is_some_and(ChildServer::is_serving)
    }

    /// The llama-server supervisor's state and child pid, read together.
    /// `None` when no service is attached.
    pub fn llama_health(&self) -> Option<HealthSnapshot> {
        self.lifecycle
            .lock()
            .server()
            .map(|service| HealthSnapshot {
                state: service.state(),
                pid: service.pid(),
            })
    }

    /// Wait until llama-server reports `ReadyState::Ready` or `budget`
    /// elapses. Fails immediately with `Degraded` while the supervisor is
    /// slow-retrying, and with `NoService` when nothing is attached.
    pub async fn wait_llama_ready(&self, budget: Duration) -> Result<(), HealthWaitError> {
        let mut rx = self
            .lifecycle
            .lock()
            .server()
            .map(ChildServer::subscribe_ready)
            .ok_or(HealthWaitError::NoService)?;

        if matches!(*rx.borrow(), ReadyState::Ready) {
            return Ok(());
        }
        if matches!(*rx.borrow(), ReadyState::Degraded) {
            return Err(HealthWaitError::Degraded);
        }

        timeout(budget, async {
            loop {
                match rx.changed().await {
                    Ok(()) => match *rx.borrow() {
                        ReadyState::Ready => return Ok::<(), HealthWaitError>(()),
                        ReadyState::Degraded => return Err(HealthWaitError::Degraded),
                        ReadyState::Starting | ReadyState::BackingOff { .. } => continue,
                    },
                    Err(_) => return Err(HealthWaitError::NoService),
                }
            }
        })
        .await
        .unwrap_or(Err(HealthWaitError::Timeout))
    }

    /// Wake unless already `Active`. Racing callers serialise on the
    /// transition lock, so only one wake runs.
    #[tracing::instrument(skip(self), fields(from = ?self.state()))]
    pub async fn ensure_active(&self) -> Result<(), PresenceError> {
        self.mark_activity();
        if self.state() == PresenceState::Active {
            return Ok(());
        }
        self.wake().await
    }

    /// Acquire a [`RequestGuard`] that holds the daemon `Active`, waking it first if
    /// needed. Cancel-safe: dropping it leaves the wake running.
    pub async fn acquire_request_guard(self: &Arc<Self>) -> Result<RequestGuard, PresenceError> {
        self.acquire_request_guard_inner(None).await
    }

    /// [`Self::acquire_request_guard`], emitting `ModelLoading` status on `tx` every few
    /// seconds while a wake runs.
    pub async fn acquire_request_guard_with_progress(
        self: &Arc<Self>,
        request_id: String,
        tx: EventSink,
    ) -> Result<RequestGuard, PresenceError> {
        self.acquire_request_guard_inner(Some((request_id, tx)))
            .await
    }

    /// The guard is taken before the state check, so an observed `Active`
    /// holds for its lifetime; the retry covers sleep/wake churn.
    async fn acquire_request_guard_inner(
        self: &Arc<Self>,
        progress: Option<(String, EventSink)>,
    ) -> Result<RequestGuard, PresenceError> {
        self.mark_activity();
        for _ in 0..GUARD_ACQUIRE_RETRIES {
            let guard = self.inflight.clone().read_owned().await;
            if self.state() == PresenceState::Active {
                return Ok(RequestGuard { _guard: guard });
            }
            drop(guard);
            let progress_task = progress.as_ref().map(|(request_id, tx)| {
                spawn_load_progress_emitter(Arc::clone(self), request_id.clone(), tx.clone())
            });
            let result = self.ensure_active_detached().await;
            if let Some(task) = progress_task {
                task.abort();
                let _ = task.await;
            }
            result?;
        }
        Err(PresenceError::GuardRetriesExhausted {
            attempts: GUARD_ACQUIRE_RETRIES,
        })
    }

    /// A [`RequestGuard`] if the daemon is `Active` now; `None` rather than
    /// waking it. Does not count as activity for the idle timer.
    pub async fn request_guard_if_active(&self) -> Option<RequestGuard> {
        let guard = self.inflight.clone().read_owned().await;
        (self.state() == PresenceState::Active).then_some(RequestGuard { _guard: guard })
    }

    /// [`Self::ensure_active`] on a supervised task, so a dropped waiter
    /// stops waiting without aborting a half-done wake.
    async fn ensure_active_detached(self: &Arc<Self>) -> Result<(), PresenceError> {
        let (result_tx, result_rx) = oneshot::channel();
        let presence = Arc::clone(self);
        spawn_supervised(
            "request_wake",
            Component::Llm,
            self.tasks.track_future(
                async move {
                    let _ = result_tx.send(presence.ensure_active().await);
                }
                .in_current_span(),
            ),
        );
        result_rx
            .await
            .map_err(|_| PresenceError::WakeTaskPanicked)?
    }

    fn wake_in_progress(&self) -> Option<Instant> {
        *self.wake_started.lock()
    }

    /// Drive the manager to `target`. Cancel-safe for `Active`: dropping it
    /// leaves the wake running.
    #[tracing::instrument(skip(self), fields(from = ?self.state()))]
    pub async fn set_presence(
        self: &Arc<Self>,
        target: PresenceTarget,
    ) -> Result<(), PresenceError> {
        self.mark_activity();
        match target {
            PresenceTarget::Active => self.ensure_active_detached().await,
            PresenceTarget::Drowsy => self.drowse().await,
            PresenceTarget::Sleeping => self.sleep().await,
        }
    }

    /// `Active|Drowsy → Sleeping`. Idempotent from `Sleeping`. Waits for
    /// every outstanding [`RequestGuard`] to drop first.
    pub async fn sleep(&self) -> Result<(), PresenceError> {
        let _guard = self.transition.lock().await;
        let prior = self.settled();
        if prior == PresenceTarget::Sleeping {
            debug!(target: "assistd::presence", "sleep: already Sleeping, no-op");
            return Ok(());
        }

        self.reclaim_drowsy_server().await;
        let _inflight = self.inflight.write().await;

        let started = Instant::now();
        let service = self.detach_llama_as_sleeping();
        let outcome = self.join_llama_shutdown(service).await;

        info!(
            target: "assistd::presence",
            prior = ?prior,
            new = ?PresenceTarget::Sleeping,
            duration_ms = saturating_millis(started.elapsed()),
            "transitioned {prior:?} → Sleeping"
        );
        outcome
    }

    /// Sleep for daemon shutdown, then wait up to five seconds for the
    /// shutdown forwarder and any detached wake to finish.
    pub async fn shutdown(&self) -> Result<(), PresenceError> {
        let slept = self.sleep().await;
        self.tasks.close();
        if timeout(TASK_DRAIN_GRACE, self.tasks.wait()).await.is_err() {
            warn!(
                target: "assistd::presence",
                remaining = self.tasks.len(),
                "presence tasks still running after the shutdown grace"
            );
        }
        slept
    }

    fn detach_llama_as_sleeping(&self) -> Option<ChildServer> {
        let service = std::mem::take(&mut *self.lifecycle.lock()).into_server();
        self.signal_inner_shutdown();
        self.publish_settled();
        service
    }

    fn signal_inner_shutdown(&self) {
        let tx = self.current_inner_shutdown.lock().take();
        if let Some(tx) = tx {
            let _ = tx.send(true);
        }
    }

    async fn join_llama_shutdown(&self, service: Option<ChildServer>) -> Result<(), PresenceError> {
        let Some(service) = service else {
            return Ok(());
        };

        let secs = self.timeouts.presence_sleep_secs;
        match timeout(Duration::from_secs(secs), service.shutdown()).await {
            Ok(result) => result.map_err(PresenceError::Shutdown),
            Err(_) => {
                warn!(
                    target: "assistd::presence",
                    timeout_secs = secs,
                    "llama-server shutdown timed out; the child may outlive the daemon"
                );
                Err(PresenceError::ShutdownTimeout { secs })
            }
        }
    }

    /// `Active → Drowsy`. Idempotent from `Drowsy`; errors from `Sleeping`.
    /// Waits for every outstanding [`RequestGuard`] to drop first.
    pub async fn drowse(&self) -> Result<(), PresenceError> {
        let _guard = self.transition.lock().await;
        let prior = self.settled();
        match prior {
            PresenceTarget::Drowsy => return Ok(()),
            PresenceTarget::Sleeping => return Err(PresenceError::DrowseFromSleeping),
            PresenceTarget::Active => {}
        }

        let _inflight = self.inflight.write().await;

        let started = Instant::now();
        self.unload_model().await?;

        self.lifecycle.lock().doze();
        self.publish_settled();
        info!(
            target: "assistd::presence",
            prior = ?prior,
            new = ?PresenceTarget::Drowsy,
            duration_ms = saturating_millis(started.elapsed()),
            "transitioned Active → Drowsy"
        );
        Ok(())
    }

    async fn unload_model(&self) -> Result<(), PresenceError> {
        if !self.llama_serving() {
            return Err(PresenceError::Unload {
                model: self.model.name.to_string(),
                source: LlamaServerError::NotReady,
            });
        }
        let secs = self.timeouts.presence_drowse_secs;
        let unload = self.control.unload_model(self.model.name.as_str());
        match timeout(Duration::from_secs(secs), unload).await {
            Ok(Ok(())) => Ok(()),
            Ok(Err(source)) => Err(PresenceError::Unload {
                model: self.model.name.to_string(),
                source,
            }),
            Err(_) => {
                warn!(
                    target: "assistd::presence",
                    timeout_secs = secs,
                    "llama-server /models/unload timed out during drowse; transition aborted"
                );
                Err(PresenceError::UnloadTimeout { secs })
            }
        }
    }

    /// `Sleeping|Drowsy → Active`, reporting `Waking` meanwhile.
    /// Idempotent from `Active`; errors once the daemon is shutting down.
    pub async fn wake(&self) -> Result<(), PresenceError> {
        let _guard = self.transition.lock().await;
        let prior = self.settled();
        if prior == PresenceTarget::Active {
            return Ok(());
        }
        self.ensure_not_shutting_down()?;
        self.reclaim_drowsy_server().await;
        self.ensure_room_to_wake()?;

        let _wake_marker = WakeMarker::new(self);

        let started = Instant::now();
        let drowsy_ready_rx = self
            .lifecycle
            .lock()
            .drowsy_server()
            .map(|service| self.serving_ready_rx(service));
        match drowsy_ready_rx {
            Some(ready_rx) => self.reload_model(ready_rx?).await?,
            None => self.cold_start().await?,
        }

        self.publish_settled();
        info!(
            target: "assistd::presence",
            prior = ?prior,
            new = ?PresenceTarget::Active,
            duration_ms = saturating_millis(started.elapsed()),
            "transitioned {prior:?} → Active"
        );
        Ok(())
    }

    fn ensure_not_shutting_down(&self) -> Result<(), PresenceError> {
        if *self.daemon_shutdown.borrow() {
            return Err(PresenceError::ShuttingDown);
        }
        Ok(())
    }

    fn ensure_room_to_wake(&self) -> Result<(), PresenceError> {
        self.room_to_load().map_err(|shortfall| {
            warn!(target: "assistd::presence", "wake refused: {shortfall}");
            PresenceError::NoRoom(shortfall)
        })
    }

    fn room_to_load(&self) -> Result<(), VramShortfall> {
        self.wake_room
            .get()
            .map_or(Ok(()), |room| room.check(self.llama_pid()))
    }

    /// While `Drowsy`, load `model` on the idle server, run `work` and unload `model`.
    /// A `wake` or `sleep` cuts `work` short, `Ok(None)` here, and waits for the unload.
    /// Errors when not `Drowsy`, without room to wake, or when `model` does not load.
    pub async fn lend_while_drowsy<T>(
        &self,
        model: &BorrowedModel,
        work: impl Future<Output = T>,
    ) -> Result<Option<T>, PresenceError> {
        let (_loan, reclaimed, ready_rx) = self.begin_loan().await?;
        let outcome = tokio::select! {
            biased;
            () = reclaimed.cancelled() => Ok(None),
            outcome = self.work_with_borrowed(model, ready_rx, work) => outcome.map(Some),
        };
        self.unload_borrowed(model).await;
        outcome
    }

    /// Take the loan of the drowsy server, with what cuts it short and the
    /// server's readiness feed.
    async fn begin_loan(
        &self,
    ) -> Result<
        (
            AsyncMutexGuard<'_, ()>,
            CancellationToken,
            watch::Receiver<ReadyState>,
        ),
        PresenceError,
    > {
        let _transition = self.transition.lock().await;
        let ready_rx = self
            .lifecycle
            .lock()
            .drowsy_server()
            .filter(|server| server.is_serving())
            .map(ChildServer::subscribe_ready)
            .ok_or(PresenceError::NotDrowsy)?;
        self.room_to_load()?;
        let loan = self
            .loan
            .try_lock()
            .map_err(|_| PresenceError::AlreadyLent)?;
        let reclaimed = CancellationToken::new();
        *self.borrower.lock() = reclaimed.clone();
        Ok((loan, reclaimed, ready_rx))
    }

    async fn work_with_borrowed<T>(
        &self,
        model: &BorrowedModel,
        mut ready_rx: watch::Receiver<ReadyState>,
        work: impl Future<Output = T>,
    ) -> Result<T, PresenceError> {
        self.load_borrowed(model, ready_rx.clone()).await?;
        tokio::select! {
            done = work => Ok(done),
            () = wait_until_not_ready(&mut ready_rx) => Err(PresenceError::LeftReady {
                model: model.served_as.clone(),
            }),
        }
    }

    /// Download `model` if the server lacks it, reporting no `Downloading`, and load
    /// it once the main model's process is gone.
    async fn load_borrowed(
        &self,
        model: &BorrowedModel,
        mut ready_rx: watch::Receiver<ReadyState>,
    ) -> Result<(), PresenceError> {
        self.await_main_model_unloaded().await?;
        let stall = Duration::from_secs(self.model.ready_timeout_secs.get());
        tokio::select! {
            cached = self.control.ensure_cached(&model.source, stall, |_| {}) => {
                cached.map_err(PresenceError::Download)?;
            }
            () = wait_until_not_ready(&mut ready_rx) => {
                return Err(PresenceError::LeftReady {
                    model: model.source.clone(),
                });
            }
        }
        self.load_and_await(&model.served_as, ready_rx).await
    }

    async fn await_main_model_unloaded(&self) -> Result<(), PresenceError> {
        let model = self.model.name.as_str();
        let deadline = Duration::from_secs(self.timeouts.presence_drowse_secs);
        self.control
            .wait_for_unloaded(model, deadline, LOAD_POLL_INTERVAL)
            .await
            .map_err(|source| PresenceError::Unload {
                model: model.to_string(),
                source,
            })
    }

    /// Unload `model` and wait until its process is gone, so its VRAM is free
    /// before anything else loads.
    async fn unload_borrowed(&self, model: &BorrowedModel) {
        let name = model.served_as.as_str();
        let deadline = Duration::from_secs(self.timeouts.presence_drowse_secs);
        if let Err(error) = self.control.unload_model(name).await {
            debug!(target: "assistd::presence", "llama-server did not unload {name}: {error}");
        }
        let gone = self
            .control
            .wait_for_unloaded(name, deadline, LOAD_POLL_INTERVAL);
        if let Err(error) = gone.await {
            warn!(target: "assistd::presence", "{name} may still be loaded: {error}");
        }
    }

    /// Cut short whatever borrows the drowsy server and wait for its model to unload.
    async fn reclaim_drowsy_server(&self) {
        self.borrower.lock().cancel();
        drop(self.loan.lock().await);
    }

    async fn reload_model(
        &self,
        ready_rx: watch::Receiver<ReadyState>,
    ) -> Result<(), PresenceError> {
        self.load_model_and_wait(ready_rx).await?;
        self.lifecycle.lock().rouse();
        Ok(())
    }

    async fn cold_start(&self) -> Result<(), PresenceError> {
        self.signal_inner_shutdown();
        let (inner_tx, inner_rx) = watch::channel(false);
        *self.current_inner_shutdown.lock() = Some(inner_tx);
        if let Err(shutting_down) = self.ensure_not_shutting_down() {
            *self.current_inner_shutdown.lock() = None;
            return Err(shutting_down);
        }

        let spec = LlamaServerSpec::new(
            self.model.clone(),
            self.api_key.clone(),
            self.slot_dir.clone(),
            self.presets.clone(),
        );
        let service = match ChildServer::start(spec, inner_rx).await {
            Ok(service) => service,
            Err(e) => {
                *self.current_inner_shutdown.lock() = None;
                warn!(target: "assistd::presence", "wake cold-start failed: {e}");
                return Err(PresenceError::ColdStart(e));
            }
        };

        let loaded = match self.serving_ready_rx(&service) {
            Ok(ready_rx) => self.load_model_and_wait(ready_rx).await,
            Err(e) => Err(e),
        };
        match loaded {
            Ok(()) => {
                *self.lifecycle.lock() = Lifecycle::Active(service);
                Ok(())
            }
            Err(e) => {
                self.signal_inner_shutdown();
                if let Err(teardown) = self.join_llama_shutdown(Some(service)).await {
                    warn!(
                        target: "assistd::presence",
                        error = %teardown,
                        "teardown after a failed cold start was unclean"
                    );
                }
                Err(e)
            }
        }
    }

    /// `service`'s readiness feed, or a load error when its child is not serving.
    fn serving_ready_rx(
        &self,
        service: &ChildServer,
    ) -> Result<watch::Receiver<ReadyState>, PresenceError> {
        if service.is_serving() {
            return Ok(service.subscribe_ready());
        }
        Err(PresenceError::Load {
            model: self.model.name.to_string(),
            source: LlamaServerError::NotReady,
        })
    }

    async fn load_model_and_wait(
        &self,
        ready_rx: watch::Receiver<ReadyState>,
    ) -> Result<(), PresenceError> {
        self.ensure_model_cached(ready_rx.clone()).await?;
        self.load_and_await(self.model.name.as_str(), ready_rx)
            .await
    }

    async fn load_and_await(
        &self,
        model: &str,
        ready_rx: watch::Receiver<ReadyState>,
    ) -> Result<(), PresenceError> {
        self.control
            .load_model(model)
            .await
            .map_err(|source| PresenceError::Load {
                model: model.to_string(),
                source,
            })?;
        self.await_model_loaded(model, ready_rx).await
    }

    /// Download the model if llama-server lacks it, reporting `Downloading`
    /// meanwhile. `ready_timeout_secs` bounds the time without new bytes, not
    /// the download.
    async fn ensure_model_cached(
        &self,
        mut ready_rx: watch::Receiver<ReadyState>,
    ) -> Result<(), PresenceError> {
        let model = self.model.name.as_str();
        let stall = Duration::from_secs(self.model.ready_timeout_secs.get());
        let mut published_at = None;
        let publish = |progress| {
            if published_at.is_none_or(|at: Instant| at.elapsed() >= DOWNLOAD_PUBLISH_INTERVAL) {
                published_at = Some(Instant::now());
                self.publish_download(progress);
            }
        };
        let downloaded = tokio::select! {
            res = self.control.ensure_cached(model, stall, publish) => {
                res.map_err(PresenceError::Download)
            }
            () = wait_until_not_ready(&mut ready_rx) => Err(PresenceError::LeftReady {
                model: model.to_string(),
            }),
        };
        self.publish_download_over();
        downloaded
    }

    fn publish_download(&self, progress: DownloadProgress) {
        self.state_tx
            .send_replace(PresenceState::Downloading(progress));
    }

    fn publish_download_over(&self) {
        self.state_tx.send_if_modified(|state| {
            let was_downloading = matches!(state, PresenceState::Downloading(_));
            if was_downloading {
                *state = PresenceState::Waking;
            }
            was_downloading
        });
    }

    async fn await_model_loaded(
        &self,
        model: &str,
        mut ready_rx: watch::Receiver<ReadyState>,
    ) -> Result<(), PresenceError> {
        let secs = self.model.ready_timeout_secs.get();
        let backstop = Duration::from_secs(secs);
        tokio::select! {
            res = self.control.wait_for_loaded(model, backstop, LOAD_POLL_INTERVAL) => {
                res.map_err(|source| PresenceError::LoadWait {
                    model: model.to_string(),
                    secs,
                    source,
                })
            }
            () = wait_until_not_ready(&mut ready_rx) => Err(PresenceError::LeftReady {
                model: model.to_string(),
            }),
        }
    }
}

/// Holds the daemon `Active` for a query: [`PresenceManager::sleep`] and
/// [`PresenceManager::drowse`] wait until every guard drops.
#[derive(Debug)]
pub struct RequestGuard {
    _guard: OwnedRwLockReadGuard<()>,
}

/// Exposes a [`PresenceManager`] through the [`LlmHealthProbe`] trait.
#[derive(Debug)]
pub struct PresenceLlmHealthProbe {
    presence: Arc<PresenceManager>,
}

impl PresenceLlmHealthProbe {
    /// Wrap `presence` as a health probe.
    pub fn new(presence: Arc<PresenceManager>) -> Self {
        Self { presence }
    }
}

#[async_trait]
impl LlmHealthProbe for PresenceLlmHealthProbe {
    async fn snapshot(&self) -> Option<HealthSnapshot> {
        self.presence.llama_health()
    }

    async fn wait_for_ready(&self, budget: Duration) -> Result<(), HealthWaitError> {
        self.presence.wait_llama_ready(budget).await
    }
}

/// Where the llama-server stands; the server handle exists exactly while
/// the model is loaded or merely unloaded.
#[derive(Debug, Default)]
enum Lifecycle {
    #[default]
    Sleeping,
    Drowsy(ChildServer),
    Active(ChildServer),
}

impl Lifecycle {
    fn target(&self) -> PresenceTarget {
        match self {
            Lifecycle::Sleeping => PresenceTarget::Sleeping,
            Lifecycle::Drowsy(_) => PresenceTarget::Drowsy,
            Lifecycle::Active(_) => PresenceTarget::Active,
        }
    }

    fn server(&self) -> Option<&ChildServer> {
        match self {
            Lifecycle::Sleeping => None,
            Lifecycle::Drowsy(server) | Lifecycle::Active(server) => Some(server),
        }
    }

    fn drowsy_server(&self) -> Option<&ChildServer> {
        match self {
            Lifecycle::Drowsy(server) => Some(server),
            Lifecycle::Sleeping | Lifecycle::Active(_) => None,
        }
    }

    fn into_server(self) -> Option<ChildServer> {
        match self {
            Lifecycle::Sleeping => None,
            Lifecycle::Drowsy(server) | Lifecycle::Active(server) => Some(server),
        }
    }

    /// `Active → Drowsy`, keeping the server; other states are unchanged.
    fn doze(&mut self) {
        *self = match std::mem::take(self) {
            Lifecycle::Active(server) => Lifecycle::Drowsy(server),
            other => other,
        };
    }

    /// `Drowsy → Active`, keeping the server; other states are unchanged.
    fn rouse(&mut self) {
        *self = match std::mem::take(self) {
            Lifecycle::Drowsy(server) => Lifecycle::Active(server),
            other => other,
        };
    }
}

/// Sets `wake_started` and reports `Waking` while held; on every return
/// path clears the mark and reports the settled state again.
struct WakeMarker<'a> {
    manager: &'a PresenceManager,
}

impl<'a> WakeMarker<'a> {
    fn new(manager: &'a PresenceManager) -> Self {
        *manager.wake_started.lock() = Some(Instant::now());
        manager.state_tx.send_replace(PresenceState::Waking);
        Self { manager }
    }
}

impl Drop for WakeMarker<'_> {
    fn drop(&mut self) {
        *self.manager.wake_started.lock() = None;
        let settled = PresenceState::from(self.manager.settled());
        self.manager.state_tx.send_if_modified(|state| {
            let changed = *state != settled;
            *state = settled;
            changed
        });
    }
}

/// Once `daemon_shutdown` flips, stop whichever supervisor epoch is current.
fn spawn_shutdown_forwarder(
    tasks: &TaskTracker,
    mut daemon_shutdown: watch::Receiver<bool>,
    current_inner_shutdown: InnerShutdownSlot,
) {
    spawn_supervised(
        "presence_shutdown_forwarder",
        Component::Daemon,
        tasks.track_future(async move {
            if daemon_shutdown.wait_for(|v| *v).await.is_err() {
                return;
            }
            let tx = current_inner_shutdown.lock().clone();
            if let Some(tx) = tx {
                let _ = tx.send(true);
            }
        }),
    );
}

async fn wait_until_not_ready(rx: &mut watch::Receiver<ReadyState>) {
    let _ = rx.wait_for(|s| *s != ReadyState::Ready).await;
}

/// Emit a `ModelLoading` status every [`LOAD_PROGRESS_INTERVAL`] while a
/// wake runs, saying how far a download has got or how long the load has
/// taken. The first immediate tick is skipped so short wakes stay quiet.
fn spawn_load_progress_emitter(
    presence: Arc<PresenceManager>,
    request_id: String,
    tx: EventSink,
) -> AbortOnDropHandle<()> {
    AbortOnDropHandle::new(tokio::spawn(async move {
        let mut interval = tokio::time::interval(LOAD_PROGRESS_INTERVAL);
        interval.tick().await;
        loop {
            interval.tick().await;
            let Some(started) = presence.wake_in_progress() else {
                return;
            };
            let message = match presence.state() {
                PresenceState::Downloading(progress) => format!("downloading model: {progress}"),
                _ => format!("loading model ({}s elapsed)", started.elapsed().as_secs()),
            };
            if tx
                .send(Event::new(
                    request_id.clone(),
                    EventBody::Status {
                        severity: StatusSeverity::Info,
                        component: Component::Llm,
                        event: StatusKind::ModelLoading,
                        message,
                    },
                ))
                .await
                .is_err()
            {
                return;
            }
        }
    }))
}

#[cfg(test)]
pub(crate) mod tests {
    use assistd_config::defaults::{nz16, nz32, nz64};
    use assistd_utils::child_server::scripted::scripted_server;

    use super::*;

    /// A manager in a fixed state whose llama-server has no supervisor or child.
    /// Transitions that hit the network (`drowse`, cold-start `wake`) error.
    pub(crate) fn stub_presence(state: PresenceTarget) -> Arc<PresenceManager> {
        let (_tx, rx) = watch::channel(false);
        let (state_tx, _) = watch::channel(state.into());
        let stopped_server = || scripted_server(ReadyState::Starting, None).1;
        let lifecycle = match state {
            PresenceTarget::Active => Lifecycle::Active(stopped_server()),
            PresenceTarget::Drowsy => Lifecycle::Drowsy(stopped_server()),
            PresenceTarget::Sleeping => Lifecycle::Sleeping,
        };
        let model = stub_model();
        let api_key = ApiKey::generate().expect("stub api key");
        let control =
            LlamaServerControl::new(model.listen_addr(), None, None).expect("dummy control");
        Arc::new(PresenceManager {
            lifecycle: StdMutex::new(lifecycle),
            transition: AsyncMutex::new(()),
            model,
            timeouts: TimeoutsConfig::default(),
            api_key,
            slot_dir: SlotDir::new("/does/not/exist/slots".into()),
            presets: ModelPresets::new("/does/not/exist/presets.ini".into(), None),
            control,
            tasks: TaskTracker::new(),
            current_inner_shutdown: Arc::new(StdMutex::new(None)),
            state_tx,
            daemon_shutdown: rx,
            last_activity: StdMutex::new(Instant::now()),
            inflight: Arc::new(RwLock::new(())),
            wake_started: StdMutex::new(None),
            wake_room: OnceLock::new(),
            gpu_hold: watch::channel(false).0,
            loan: AsyncMutex::new(()),
            borrower: StdMutex::new(CancellationToken::new()),
        })
    }

    fn stub_model() -> ModelConfig {
        ModelConfig {
            name: "stub/model".parse().expect("stub model name"),
            context_length: nz32(1024),
            server_binary: "/does/not/exist".parse().expect("stub server binary"),
            port: nz16(1),
            gpu_layers: 1,
            ready_timeout_secs: nz64(1),
            ..ModelConfig::default()
        }
    }

    #[tokio::test]
    async fn transition_to_current_state_is_a_silent_noop() {
        for state in [
            PresenceTarget::Active,
            PresenceTarget::Drowsy,
            PresenceTarget::Sleeping,
        ] {
            let m = stub_presence(state);
            let rx = m.subscribe();
            m.set_presence(state)
                .await
                .unwrap_or_else(|e| panic!("{state:?}: {e:#}"));
            assert_eq!(m.state(), state.into());
            assert!(!rx.has_changed().unwrap(), "{state:?}: broadcast a no-op");
        }
    }

    #[tokio::test]
    async fn ensure_active_resets_activity_timer() {
        let m = stub_presence(PresenceTarget::Active);
        tokio::time::sleep(Duration::from_millis(50)).await;
        let before = m.idle_duration();
        assert!(before >= Duration::from_millis(40));
        m.ensure_active().await.unwrap();
        let after = m.idle_duration();
        assert!(after < before);
        assert!(after < Duration::from_millis(20));
        assert_eq!(m.state(), PresenceState::Active);
    }

    #[tokio::test]
    async fn dropped_guard_waiter_leaves_its_wake_running() {
        let m = stub_presence(PresenceTarget::Drowsy);
        let transition = m.transition.lock().await;
        let waiter = timeout(
            Duration::from_millis(50),
            m.acquire_request_guard_inner(None),
        )
        .await;
        assert!(waiter.is_err(), "acquire finished while a transition ran");
        assert_eq!(Arc::strong_count(&m), 2, "the wake died with its waiter");

        drop(transition);
        timeout(Duration::from_secs(2), async {
            while Arc::strong_count(&m) > 1 {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("the detached wake never finished");
    }

    #[tokio::test]
    async fn sleep_defers_for_inflight_request() {
        let m = stub_presence(PresenceTarget::Active);
        let guard = m.acquire_request_guard_inner(None).await.unwrap();

        let m2 = Arc::clone(&m);
        let sleep_task = tokio::spawn(async move { m2.sleep().await });

        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(
            !sleep_task.is_finished(),
            "sleep completed while request guard held"
        );

        drop(guard);
        timeout(Duration::from_secs(2), sleep_task)
            .await
            .expect("sleep did not complete after guard dropped")
            .expect("sleep task panicked")
            .expect("sleep returned Err");
        assert_eq!(m.state(), PresenceState::Sleeping);
    }

    #[tokio::test]
    async fn a_wake_cancels_the_borrower_and_waits_for_its_model_to_be_unloaded() {
        let m = stub_presence(PresenceTarget::Drowsy);
        let loan = m.loan.lock().await;
        let borrower = m.borrower.lock().clone();
        let waking = tokio::spawn({
            let m = Arc::clone(&m);
            async move { m.wake().await }
        });

        timeout(Duration::from_secs(2), borrower.cancelled())
            .await
            .expect("the wake left the borrower running");
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(!waking.is_finished(), "the wake went on during the loan");
        assert_eq!(m.state(), PresenceState::Drowsy);

        drop(loan);
        timeout(Duration::from_secs(2), waking)
            .await
            .expect("the wake never went on after the loan")
            .expect("wake task panicked")
            .expect_err("the stub llama-server is not serving");
    }

    #[tokio::test]
    async fn failed_wake_settles_back_to_the_prior_state() {
        let m = stub_presence(PresenceTarget::Sleeping);
        m.wake()
            .await
            .expect_err("stub llama-server binary does not exist");
        assert_eq!(m.state(), PresenceState::Sleeping);
        assert!(m.wake_in_progress().is_none());
    }

    #[tokio::test]
    async fn a_wake_queued_when_the_daemon_shuts_down_starts_no_server() {
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let m = PresenceManager::new_sleeping(
            stub_model(),
            TimeoutsConfig::default(),
            ApiKey::generate().expect("stub api key"),
            SlotDir::new("/does/not/exist/slots".into()),
            ModelPresets::new("/does/not/exist/presets.ini".into(), None),
            shutdown_rx,
        )
        .expect("stub manager");
        let transition = m.transition.lock().await;
        let queued = tokio::spawn({
            let m = Arc::clone(&m);
            async move { m.wake().await }
        });
        tokio::time::sleep(Duration::from_millis(50)).await;

        shutdown_tx
            .send(true)
            .expect("the manager holds a receiver");
        drop(transition);
        let outcome = timeout(Duration::from_secs(2), queued)
            .await
            .expect("the queued wake never returned")
            .expect("wake task panicked");
        assert!(
            matches!(outcome, Err(PresenceError::ShuttingDown)),
            "{outcome:?}"
        );
        assert!(m.current_inner_shutdown.lock().is_none());
    }
}
