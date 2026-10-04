//! Daemon presence state machine (`Active`, `Drowsy`, `Sleeping`), which
//! frees GPU resources on demand and auto-wakes when a query arrives.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use parking_lot::Mutex as StdMutex;
use thiserror::Error;
use tokio::sync::{Mutex as AsyncMutex, OwnedRwLockReadGuard, RwLock, mpsc, oneshot, watch};
use tokio::time::timeout;
use tokio_util::task::AbortOnDropHandle;
use tracing::{Instrument, debug, info, warn};

#[cfg(test)]
use assistd_config::defaults::{nz16, nz32, nz64};
use assistd_config::{ModelConfig, TimeoutsConfig};
use assistd_ipc::{Component, Event, PresenceState, StatusKind, StatusSeverity};
use assistd_llm::{
    HealthSnapshot, HealthWaitError, LlamaServerControl, LlamaServerError, LlamaServerSpec,
    LlmHealthProbe, ReadyState,
};
use assistd_utils::child_server::{ChildServer, ChildServerError};

use crate::recovery::spawn_supervised;

const GUARD_ACQUIRE_RETRIES: usize = 3;
const LOAD_POLL_INTERVAL: Duration = Duration::from_millis(200);
const LOAD_PROGRESS_INTERVAL: Duration = Duration::from_secs(3);

type InnerShutdownSlot = Arc<StdMutex<Option<watch::Sender<bool>>>>;

/// Why a presence transition, or a request guard waiting on one, failed.
#[derive(Debug, Error)]
pub enum PresenceError {
    #[error("failed to construct llama-server control client: {0}")]
    Control(#[source] LlamaServerError),

    #[error("initial cold-start wake failed: {0}")]
    InitialWake(#[source] Box<PresenceError>),

    #[error("cannot drowse from Sleeping: call wake() first")]
    DrowseFromSleeping,

    #[error("llama-server cold-start failed during wake: {0}")]
    ColdStart(#[source] ChildServerError),

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

    /// The supervisor restarted or gave up on the child mid-load.
    #[error("llama-server left the ready state while loading {model}")]
    LeftReady { model: String },

    #[error("llama-server handle missing while loading model")]
    ServiceMissing,

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
}

/// Owner of the llama-server handle and the daemon-wide presence state.
/// Transitions are serialised, so an auto-wake cannot race an explicit
/// `sleep`.
#[derive(Debug)]
pub struct PresenceManager {
    state: StdMutex<PresenceState>,
    transition: AsyncMutex<()>,
    model: ModelConfig,
    timeouts: TimeoutsConfig,
    control: LlamaServerControl,
    /// `Some` iff state is `Active` or `Drowsy`.
    llama: AsyncMutex<Option<ChildServer>>,
    /// Flipped by `sleep()` to stop the current supervisor only.
    current_inner_shutdown: InnerShutdownSlot,
    state_tx: watch::Sender<PresenceState>,
    /// Keeps the daemon shutdown watch open so the forwarder cannot miss it.
    _daemon_shutdown_keepalive: watch::Receiver<bool>,
    /// Last user-initiated interaction; automatic monitors don't bump it.
    last_activity: StdMutex<Instant>,
    /// Read by request guards, written by `sleep`/`drowse` to drain them.
    /// Writer-preferring, so queued requests cannot starve a transition.
    inflight: Arc<RwLock<()>>,
    /// `Some(started_at)` while a `wake` is executing.
    wake_started: Arc<StdMutex<Option<Instant>>>,
    /// LLM streams in flight, which can be fewer than live request guards.
    stream_count_tx: watch::Sender<usize>,
}

impl PresenceManager {
    /// Create a manager and cold-start it to `Active`. Flipping
    /// `daemon_shutdown` cancels an in-flight wake.
    pub async fn new_active(
        model: ModelConfig,
        timeouts: TimeoutsConfig,
        daemon_shutdown: watch::Receiver<bool>,
    ) -> Result<Arc<Self>, PresenceError> {
        let control = LlamaServerControl::new(SocketAddr::new(model.host, model.port.get()), None)
            .map_err(PresenceError::Control)?;

        let current_inner_shutdown: InnerShutdownSlot = Arc::new(StdMutex::new(None));
        spawn_shutdown_forwarder(daemon_shutdown.clone(), Arc::clone(&current_inner_shutdown));

        let (state_tx, _) = watch::channel(PresenceState::Sleeping);
        let (stream_count_tx, _) = watch::channel(0usize);
        let manager = Arc::new(Self {
            state: StdMutex::new(PresenceState::Sleeping),
            transition: AsyncMutex::new(()),
            model,
            timeouts,
            control,
            llama: AsyncMutex::new(None),
            current_inner_shutdown,
            state_tx,
            _daemon_shutdown_keepalive: daemon_shutdown,
            last_activity: StdMutex::new(Instant::now()),
            inflight: Arc::new(RwLock::new(())),
            wake_started: Arc::new(StdMutex::new(None)),
            stream_count_tx,
        });

        manager
            .wake()
            .await
            .map_err(|e| PresenceError::InitialWake(Box::new(e)))?;
        Ok(manager)
    }

    /// The current presence state.
    pub fn state(&self) -> PresenceState {
        *self.state.lock()
    }

    fn mark_activity(&self) {
        *self.last_activity.lock() = Instant::now();
    }

    fn publish_state(&self, state: PresenceState) {
        *self.state.lock() = state;
        self.state_tx.send_replace(state);
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
    pub async fn llama_pid(&self) -> Option<u32> {
        self.llama.lock().await.as_ref().and_then(ChildServer::pid)
    }

    /// Non-blocking [`Self::llama_pid`]. Also `None` while a transition
    /// holds the llama slot, so a `None` may be spurious.
    pub fn llama_pid_blocking(&self) -> Option<u32> {
        self.llama
            .try_lock()
            .ok()
            .and_then(|svc| svc.as_ref().and_then(ChildServer::pid))
    }

    /// Whether a llama-server child is attached and serving. Waits for the
    /// slot, so a transition holding it never reads as not serving.
    pub async fn llama_serving(&self) -> bool {
        self.llama
            .lock()
            .await
            .as_ref()
            .is_some_and(ChildServer::is_serving)
    }

    /// The llama-server supervisor's state and child pid, read under one wait
    /// for the slot. `None` when no service is attached.
    pub async fn llama_health(&self) -> Option<HealthSnapshot> {
        self.llama
            .lock()
            .await
            .as_ref()
            .map(|service| HealthSnapshot {
                state: service.state(),
                pid: service.pid(),
            })
    }

    /// Wait until llama-server reports `ReadyState::Ready` or `budget`
    /// elapses. Fails immediately with `Degraded` once the supervisor has
    /// given up, and with `NoService` when nothing is attached.
    pub async fn wait_llama_ready(&self, budget: Duration) -> Result<(), HealthWaitError> {
        let mut rx = {
            let guard = self.llama.lock().await;
            match guard.as_ref() {
                Some(svc) => svc.subscribe_ready(),
                None => return Err(HealthWaitError::NoService),
            }
        };

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

    /// Acquire a [`RequestGuard`] that holds the daemon `Active`, waking it
    /// first if needed. While a wake runs, emits `ModelLoading` status on
    /// `tx` every few seconds. Cancel-safe: dropping it leaves the wake running.
    pub async fn acquire_request_guard_with_progress(
        self: &Arc<Self>,
        request_id: String,
        tx: mpsc::Sender<Event>,
    ) -> Result<RequestGuard, PresenceError> {
        self.acquire_request_guard_inner(Some((request_id, tx)))
            .await
    }

    /// The guard is taken before the state check, so an observed `Active`
    /// holds for its lifetime; the retry covers sleep/wake churn.
    async fn acquire_request_guard_inner(
        self: &Arc<Self>,
        progress: Option<(String, mpsc::Sender<Event>)>,
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

    /// [`Self::ensure_active`] on a supervised task, so a dropped waiter
    /// stops waiting without aborting a half-done wake.
    async fn ensure_active_detached(self: &Arc<Self>) -> Result<(), PresenceError> {
        let (result_tx, result_rx) = oneshot::channel();
        let presence = Arc::clone(self);
        spawn_supervised(
            "request_wake",
            Component::Llm,
            async move {
                let _ = result_tx.send(presence.ensure_active().await);
            }
            .in_current_span(),
        );
        result_rx
            .await
            .map_err(|_| PresenceError::WakeTaskPanicked)?
    }

    fn wake_in_progress(&self) -> Option<Instant> {
        *self.wake_started.lock()
    }

    /// Register an in-flight LLM stream for as long as the guard lives.
    pub fn acquire_stream_guard(&self) -> LlmStreamGuard {
        self.stream_count_tx.send_modify(|n| *n += 1);
        LlmStreamGuard {
            tx: self.stream_count_tx.clone(),
        }
    }

    /// Wait until no LLM stream is in flight. Returns `false` on
    /// timeout. Cancel-safe.
    pub async fn wait_until_llm_idle(&self, timeout: Duration) -> bool {
        if *self.stream_count_tx.borrow() == 0 {
            return true;
        }
        let mut rx = self.stream_count_tx.subscribe();
        tokio::time::timeout(timeout, async {
            let _ = rx.wait_for(|n| *n == 0).await;
        })
        .await
        .is_ok()
    }

    /// Drive the manager to `target`.
    #[tracing::instrument(skip(self), fields(from = ?self.state()))]
    pub async fn set_presence(&self, target: PresenceState) -> Result<(), PresenceError> {
        self.mark_activity();
        match target {
            PresenceState::Active => self.wake().await,
            PresenceState::Drowsy => self.drowse().await,
            PresenceState::Sleeping => self.sleep().await,
        }
    }

    /// Advance one step along `Active → Drowsy → Sleeping → Active`. Of two
    /// racing calls targeting the same state, the loser is a no-op.
    pub async fn cycle(&self) -> Result<PresenceState, PresenceError> {
        self.mark_activity();
        let target = self.state().next();
        self.set_presence(target).await?;
        Ok(target)
    }

    /// `Active|Drowsy → Sleeping`. Idempotent from `Sleeping`. Waits for
    /// every outstanding [`RequestGuard`] to drop first.
    pub async fn sleep(&self) -> Result<(), PresenceError> {
        let _guard = self.transition.lock().await;
        let prior = self.state();
        if prior == PresenceState::Sleeping {
            debug!(target: "assistd::presence", "sleep: already Sleeping, no-op");
            return Ok(());
        }

        let _inflight = self.inflight.write().await;

        let started = Instant::now();
        let service = self.detach_llama_as_sleeping().await;
        let outcome = self.join_llama_shutdown(service).await;

        info!(
            target: "assistd::presence",
            prior = ?prior,
            new = ?PresenceState::Sleeping,
            duration_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX),
            "transitioned {prior:?} → Sleeping"
        );
        outcome
    }

    async fn detach_llama_as_sleeping(&self) -> Option<ChildServer> {
        let mut slot = self.llama.lock().await;
        let service = slot.take();
        self.signal_inner_shutdown();
        self.publish_state(PresenceState::Sleeping);
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
        let prior = self.state();
        match prior {
            PresenceState::Drowsy => return Ok(()),
            PresenceState::Sleeping => return Err(PresenceError::DrowseFromSleeping),
            PresenceState::Active => {}
        }

        let _inflight = self.inflight.write().await;

        let started = Instant::now();
        self.unload_model().await?;

        self.publish_state(PresenceState::Drowsy);
        info!(
            target: "assistd::presence",
            prior = ?prior,
            new = ?PresenceState::Drowsy,
            duration_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX),
            "transitioned Active → Drowsy"
        );
        Ok(())
    }

    async fn unload_model(&self) -> Result<(), PresenceError> {
        if !self.llama_serving().await {
            return Err(PresenceError::Unload {
                model: self.model.name.clone(),
                source: LlamaServerError::NotReady,
            });
        }
        let secs = self.timeouts.presence_drowse_secs;
        let unload = self.control.unload_model(&self.model.name);
        match timeout(Duration::from_secs(secs), unload).await {
            Ok(Ok(())) => Ok(()),
            Ok(Err(source)) => Err(PresenceError::Unload {
                model: self.model.name.clone(),
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

    /// `Sleeping|Drowsy → Active`. Idempotent from `Active`.
    pub async fn wake(&self) -> Result<(), PresenceError> {
        let _guard = self.transition.lock().await;
        let prior = self.state();
        if prior == PresenceState::Active {
            return Ok(());
        }

        let _wake_marker = WakeMarker::new(Arc::clone(&self.wake_started));

        let started = Instant::now();
        match prior {
            PresenceState::Drowsy => self.reload_model().await?,
            PresenceState::Sleeping => self.cold_start().await?,
            PresenceState::Active => unreachable!("short-circuited above"),
        }

        self.publish_state(PresenceState::Active);
        info!(
            target: "assistd::presence",
            prior = ?prior,
            new = ?PresenceState::Active,
            duration_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX),
            "transitioned {prior:?} → Active"
        );
        Ok(())
    }

    async fn reload_model(&self) -> Result<(), PresenceError> {
        let ready_rx = {
            let slot = self.llama.lock().await;
            let service = slot.as_ref().ok_or(PresenceError::ServiceMissing)?;
            self.serving_ready_rx(service)?
        };
        self.load_model_and_wait(ready_rx).await
    }

    async fn cold_start(&self) -> Result<(), PresenceError> {
        self.signal_inner_shutdown();
        let (inner_tx, inner_rx) = watch::channel(false);
        *self.current_inner_shutdown.lock() = Some(inner_tx);

        let service =
            match ChildServer::start(LlamaServerSpec::new(self.model.clone()), inner_rx).await {
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
                *self.llama.lock().await = Some(service);
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
            model: self.model.name.clone(),
            source: LlamaServerError::NotReady,
        })
    }

    async fn load_model_and_wait(
        &self,
        ready_rx: watch::Receiver<ReadyState>,
    ) -> Result<(), PresenceError> {
        self.control
            .load_model(&self.model.name)
            .await
            .map_err(|source| PresenceError::Load {
                model: self.model.name.clone(),
                source,
            })?;

        self.await_model_loaded(ready_rx).await
    }

    async fn await_model_loaded(
        &self,
        mut ready_rx: watch::Receiver<ReadyState>,
    ) -> Result<(), PresenceError> {
        let secs = self.model.ready_timeout_secs.get();
        let backstop = Duration::from_secs(secs);
        tokio::select! {
            res = self
                .control
                .wait_for_loaded(&self.model.name, backstop, LOAD_POLL_INTERVAL) =>
            {
                res.map_err(|source| PresenceError::LoadWait {
                    model: self.model.name.clone(),
                    secs,
                    source,
                })
            }
            () = wait_until_not_ready(&mut ready_rx) => Err(PresenceError::LeftReady {
                model: self.model.name.clone(),
            }),
        }
    }
}

#[cfg(test)]
impl PresenceManager {
    /// Manager in a fixed state with no llama child. Transitions that hit
    /// the network (`drowse`, cold-start `wake`) error.
    pub(crate) fn stub(state: PresenceState) -> Arc<Self> {
        let (_tx, rx) = watch::channel(false);
        let (state_tx, _) = watch::channel(state);
        let model = ModelConfig {
            name: "stub/model".into(),
            context_length: nz32(1024),
            server_binary: "/does/not/exist".into(),
            host: std::net::Ipv4Addr::LOCALHOST.into(),
            port: nz16(1),
            gpu_layers: 1,
            ready_timeout_secs: nz64(1),
            ..ModelConfig::default()
        };
        let control =
            LlamaServerControl::new(SocketAddr::new(model.host, 1), None).expect("dummy control");
        let (stream_count_tx, _) = watch::channel(0usize);
        Arc::new(Self {
            state: StdMutex::new(state),
            transition: AsyncMutex::new(()),
            model,
            timeouts: TimeoutsConfig::default(),
            control,
            llama: AsyncMutex::new(None),
            current_inner_shutdown: Arc::new(StdMutex::new(None)),
            state_tx,
            _daemon_shutdown_keepalive: rx,
            last_activity: StdMutex::new(Instant::now()),
            inflight: Arc::new(RwLock::new(())),
            wake_started: Arc::new(StdMutex::new(None)),
            stream_count_tx,
        })
    }

    pub(crate) fn set_state_for_test(&self, state: PresenceState) {
        self.publish_state(state);
    }
}

/// Holds the daemon `Active` for a query: [`PresenceManager::sleep`] and
/// [`PresenceManager::drowse`] wait until every guard drops.
#[derive(Debug)]
pub struct RequestGuard {
    _guard: OwnedRwLockReadGuard<()>,
}

/// Counts one in-flight LLM stream for as long as it is held. Unlike
/// [`RequestGuard`], does not block sleep/drowse.
#[derive(Debug)]
pub struct LlmStreamGuard {
    tx: watch::Sender<usize>,
}

impl Drop for LlmStreamGuard {
    fn drop(&mut self) {
        self.tx.send_modify(|n| *n = n.saturating_sub(1));
    }
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
        self.presence.llama_health().await
    }

    async fn wait_for_ready(&self, budget: Duration) -> Result<(), HealthWaitError> {
        self.presence.wait_llama_ready(budget).await
    }
}

/// Sets `wake_started` while held and clears it on every return path.
struct WakeMarker {
    slot: Arc<StdMutex<Option<Instant>>>,
}

impl WakeMarker {
    fn new(slot: Arc<StdMutex<Option<Instant>>>) -> Self {
        *slot.lock() = Some(Instant::now());
        Self { slot }
    }
}

impl Drop for WakeMarker {
    fn drop(&mut self) {
        *self.slot.lock() = None;
    }
}

/// Once `daemon_shutdown` flips, stop whichever supervisor epoch is current.
fn spawn_shutdown_forwarder(
    mut daemon_shutdown: watch::Receiver<bool>,
    current_inner_shutdown: InnerShutdownSlot,
) {
    spawn_supervised(
        "presence_shutdown_forwarder",
        Component::Daemon,
        async move {
            if daemon_shutdown.wait_for(|v| *v).await.is_err() {
                return;
            }
            let tx = current_inner_shutdown.lock().clone();
            if let Some(tx) = tx {
                let _ = tx.send(true);
            }
        },
    );
}

async fn wait_until_not_ready(rx: &mut watch::Receiver<ReadyState>) {
    let _ = rx.wait_for(|s| *s != ReadyState::Ready).await;
}

/// Emit a `ModelLoading` status every [`LOAD_PROGRESS_INTERVAL`] while a
/// wake runs. The first immediate tick is skipped so short wakes stay quiet.
fn spawn_load_progress_emitter(
    presence: Arc<PresenceManager>,
    request_id: String,
    tx: mpsc::Sender<Event>,
) -> AbortOnDropHandle<()> {
    AbortOnDropHandle::new(tokio::spawn(async move {
        let mut interval = tokio::time::interval(LOAD_PROGRESS_INTERVAL);
        interval.tick().await;
        loop {
            interval.tick().await;
            let Some(started) = presence.wake_in_progress() else {
                return;
            };
            let elapsed_secs = started.elapsed().as_secs();
            if tx
                .send(Event::Status {
                    id: request_id.clone(),
                    severity: StatusSeverity::Info,
                    component: Component::Llm,
                    event: StatusKind::ModelLoading,
                    message: format!("loading model ({elapsed_secs}s elapsed)"),
                })
                .await
                .is_err()
            {
                return;
            }
        }
    }))
}

#[cfg(test)]
mod tests;
