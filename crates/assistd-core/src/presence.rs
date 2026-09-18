//! Daemon presence state machine: `Active`, `Drowsy`, `Sleeping`. Lets
//! the daemon free GPU resources on demand while the control socket keeps
//! listening in every state; a query that arrives while not `Active`
//! blocks on an automatic wake and then streams as usual.

use std::sync::Arc;

use parking_lot::Mutex as StdMutex;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
#[cfg(test)]
use assistd_config::defaults::{nz16, nz32, nz64};
use assistd_config::{LlamaServerConfig, ModelConfig, TimeoutsConfig};
use assistd_ipc::PresenceState;
use assistd_llm::{HealthWaitError, LlamaServerControl, LlamaService, LlmHealthProbe, ReadyState};
use async_trait::async_trait;
use tokio::sync::{Mutex as AsyncMutex, OwnedRwLockReadGuard, RwLock, watch};
use tokio::time::timeout;
use tracing::{debug, info, warn};

/// Owner of the llama-server handle and the daemon-wide presence state.
///
/// Transitions are serialised by a mutex, so an auto-wake triggered by a
/// query cannot race an explicit `sleep` from an IPC request.
pub struct PresenceManager {
    state: StdMutex<PresenceState>,
    /// Held across awaits, so a tokio mutex rather than std.
    transition: AsyncMutex<()>,
    llama_server: LlamaServerConfig,
    model: ModelConfig,
    timeouts: TimeoutsConfig,
    control: LlamaServerControl,
    /// `Some` iff state is `Active` or `Drowsy`.
    llama: AsyncMutex<Option<LlamaService>>,
    /// Per-epoch watch that `sleep()` flips to tear down the current
    /// supervisor without disturbing the daemon-wide shutdown watch.
    current_inner_shutdown: Arc<StdMutex<Option<watch::Sender<bool>>>>,
    state_tx: watch::Sender<PresenceState>,
    /// Never read. The shutdown forwarder spawned in `new_active` needs
    /// the daemon shutdown watch to stay open for the manager's whole
    /// lifetime; without this receiver it can miss the signal.
    _daemon_shutdown_keepalive: watch::Receiver<bool>,
    /// Last user-initiated interaction. Automatic monitors (GPU, idle)
    /// deliberately do not update it, so their own transitions don't
    /// defer further idle progress.
    last_activity: StdMutex<Instant>,
    /// Request handlers hold the read side for a generation; `sleep` and
    /// `drowse` take the write side to wait for them to drain. The lock
    /// is writer-preferring, so requests queued behind a pending
    /// transition wait for it rather than starving it.
    inflight: Arc<RwLock<()>>,
    /// `Some(started_at)` while a `wake` transition is executing.
    wake_started: Arc<StdMutex<Option<Instant>>>,
    /// LLM streams in flight. Separate from `inflight` because some
    /// request paths hold a request guard without streaming on the GPU
    /// (presence queries, cycles), and those must not push Whisper off it.
    stream_count_tx: watch::Sender<usize>,
}

/// Held by request handlers for the duration of a query. While any guard
/// is alive, [`PresenceManager::sleep`] and [`PresenceManager::drowse`]
/// block, so a streaming response is never torn down mid-generation.
pub struct RequestGuard {
    _guard: OwnedRwLockReadGuard<()>,
}

/// Exposes a [`PresenceManager`] through the [`LlmHealthProbe`] trait.
pub struct PresenceLlmHealthProbe {
    presence: Arc<PresenceManager>,
}

impl PresenceLlmHealthProbe {
    pub fn new(presence: Arc<PresenceManager>) -> Self {
        Self { presence }
    }
}

#[async_trait]
impl LlmHealthProbe for PresenceLlmHealthProbe {
    fn pid(&self) -> Option<u32> {
        self.presence.llama_pid_blocking()
    }

    fn state(&self) -> Option<ReadyState> {
        self.presence.llama_state_blocking()
    }

    async fn wait_for_ready(&self, budget: Duration) -> Result<(), HealthWaitError> {
        self.presence.wait_llama_ready(budget).await
    }
}

/// Counts one in-flight LLM stream for as long as it is held. Unlike
/// [`RequestGuard`], does not block sleep/drowse.
pub struct LlmStreamGuard {
    tx: watch::Sender<usize>,
}

impl Drop for LlmStreamGuard {
    fn drop(&mut self) {
        self.tx.send_modify(|n| *n = n.saturating_sub(1));
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

impl PresenceManager {
    /// Create a manager and perform the initial cold-start wake, so the
    /// daemon is `Active` before it serves the socket. Flipping
    /// `daemon_shutdown` cancels an in-flight wake.
    pub async fn new_active(
        llama_server: LlamaServerConfig,
        model: ModelConfig,
        timeouts: TimeoutsConfig,
        daemon_shutdown: watch::Receiver<bool>,
    ) -> Result<Arc<Self>> {
        let control =
            LlamaServerControl::new(&llama_server.host.to_string(), llama_server.port.get())
                .context("failed to construct llama-server control client")?;

        let current_inner_shutdown: Arc<StdMutex<Option<watch::Sender<bool>>>> =
            Arc::new(StdMutex::new(None));

        {
            let mut daemon_rx = daemon_shutdown.clone();
            let current = Arc::clone(&current_inner_shutdown);
            crate::recovery::spawn_supervised(
                "presence_shutdown_forwarder",
                crate::recovery::Component::Daemon,
                async move {
                    if daemon_rx.wait_for(|v| *v).await.is_err() {
                        return;
                    }
                    let tx = current.lock().clone();
                    if let Some(tx) = tx {
                        let _ = tx.send(true);
                    }
                },
            );
        }

        let (state_tx, _) = watch::channel(PresenceState::Sleeping);
        let (stream_count_tx, _) = watch::channel(0usize);
        let manager = Arc::new(Self {
            state: StdMutex::new(PresenceState::Sleeping),
            transition: AsyncMutex::new(()),
            llama_server,
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
            .context("initial cold-start wake failed")?;
        Ok(manager)
    }

    pub fn state(&self) -> PresenceState {
        *self.state.lock()
    }

    fn mark_activity(&self) {
        *self.last_activity.lock() = Instant::now();
    }

    /// Time since the last user-initiated interaction.
    pub fn idle_duration(&self) -> Duration {
        self.last_activity.lock().elapsed()
    }

    /// Time until the next idle-based transition given the current
    /// state and config. Returns `None` when idle monitoring is
    /// disabled for the relevant transition or when the daemon is
    /// already `Sleeping`.
    pub fn time_until_next_transition(&self, cfg: &crate::SleepConfig) -> Option<Duration> {
        let idle = self.idle_duration();
        let threshold_mins = match self.state() {
            PresenceState::Active => {
                if cfg.idle_to_drowsy_mins > 0 {
                    cfg.idle_to_drowsy_mins
                } else if cfg.idle_to_sleep_mins > 0 {
                    cfg.idle_to_sleep_mins
                } else {
                    return None;
                }
            }
            PresenceState::Drowsy => {
                if cfg.idle_to_sleep_mins == 0 {
                    return None;
                }
                cfg.idle_to_sleep_mins
            }
            PresenceState::Sleeping => return None,
        };
        Some(Duration::from_secs(threshold_mins * 60).saturating_sub(idle))
    }

    /// Subscribe to presence-state changes. The returned receiver starts at
    /// the current state; each successful transition sends the new value.
    pub fn subscribe(&self) -> watch::Receiver<PresenceState> {
        self.state_tx.subscribe()
    }

    /// PID of the currently-managed llama-server child, or `None` if the
    /// daemon is `Sleeping` or the child has not yet been spawned.
    pub async fn llama_pid(&self) -> Option<u32> {
        self.llama.lock().await.as_ref().and_then(|s| s.pid())
    }

    /// Non-blocking [`Self::llama_pid`]. Also `None` while a transition
    /// holds the llama slot, so periodic callers must tolerate misses.
    pub fn llama_pid_blocking(&self) -> Option<u32> {
        self.llama
            .try_lock()
            .ok()
            .and_then(|svc| svc.as_ref().and_then(|s| s.pid()))
    }

    /// Non-blocking snapshot of the supervisor's [`ReadyState`]. `None`
    /// when no service is attached or a transition holds the slot.
    pub fn llama_state_blocking(&self) -> Option<ReadyState> {
        self.llama
            .try_lock()
            .ok()
            .and_then(|svc| svc.as_ref().map(|s| s.state()))
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

        let res = timeout(budget, async {
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
        .await;

        match res {
            Ok(inner) => inner,
            Err(_) => Err(HealthWaitError::Timeout),
        }
    }

    /// Wake unless already `Active`. Racing callers serialise on the
    /// transition lock, so only one wake runs.
    #[tracing::instrument(skip(self), fields(from = ?self.state()))]
    pub async fn ensure_active(&self) -> Result<()> {
        self.mark_activity();
        if self.state() == PresenceState::Active {
            return Ok(());
        }
        self.wake().await
    }

    /// Acquire a [`RequestGuard`] that holds the daemon `Active`, waking
    /// it first if needed.
    ///
    /// The guard is taken before the state check, so `Active` observed
    /// under it holds for the guard's lifetime: a sleep must first take
    /// the write side of the same lock. A bounded retry covers sleep/wake
    /// churn between the wake and the re-check.
    pub async fn acquire_request_guard(self: &Arc<Self>) -> Result<RequestGuard> {
        self.acquire_request_guard_inner(None).await
    }

    /// [`Self::acquire_request_guard`] that also emits `Event::Status`
    /// progress on `tx` every few seconds while a wake is in progress, so
    /// a long model load is distinguishable from a hang. Wakes shorter
    /// than one tick emit nothing.
    pub async fn acquire_request_guard_with_progress(
        self: &Arc<Self>,
        request_id: String,
        tx: tokio::sync::mpsc::Sender<assistd_ipc::Event>,
    ) -> Result<RequestGuard> {
        self.acquire_request_guard_inner(Some((request_id, tx)))
            .await
    }

    async fn acquire_request_guard_inner(
        self: &Arc<Self>,
        progress: Option<(String, tokio::sync::mpsc::Sender<assistd_ipc::Event>)>,
    ) -> Result<RequestGuard> {
        self.mark_activity();
        const MAX_RETRIES: usize = 3;
        for _ in 0..MAX_RETRIES {
            let guard = self.inflight.clone().read_owned().await;
            if self.state() == PresenceState::Active {
                return Ok(RequestGuard { _guard: guard });
            }
            drop(guard);
            let progress_task = progress.as_ref().map(|(request_id, tx)| {
                spawn_load_progress_emitter(Arc::clone(self), request_id.clone(), tx.clone())
            });
            let result = self.ensure_active().await;
            if let Some(task) = progress_task {
                task.abort();
                let _ = task.await;
            }
            result?;
        }
        bail!("failed to acquire active request guard after {MAX_RETRIES} retries")
    }

    /// `Some(started_at)` while a wake transition is running.
    pub fn wake_in_progress(&self) -> Option<Instant> {
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

    /// Subscribe to changes in the in-flight LLM-stream count.
    pub fn subscribe_llm_streams(&self) -> watch::Receiver<usize> {
        self.stream_count_tx.subscribe()
    }

    /// Drive the manager to `target`.
    #[tracing::instrument(skip(self), fields(from = ?self.state()))]
    pub async fn set_presence(&self, target: PresenceState) -> Result<()> {
        self.mark_activity();
        match target {
            PresenceState::Active => self.wake().await,
            PresenceState::Drowsy => self.drowse().await,
            PresenceState::Sleeping => self.sleep().await,
        }
    }

    /// Advance one step along `Active → Drowsy → Sleeping → Active`.
    ///
    /// Two racing calls can both target the same state, in which case the
    /// loser is a no-op; the transition mutex still guarantees a step is
    /// never skipped or split.
    pub async fn cycle(&self) -> Result<PresenceState> {
        self.mark_activity();
        let target = self.state().next();
        self.set_presence(target).await?;
        Ok(target)
    }

    /// `Active|Drowsy → Sleeping`. Idempotent from `Sleeping`.
    ///
    /// Blocks until every outstanding [`RequestGuard`] has been dropped,
    /// so an in-flight generation is never killed mid-stream.
    pub async fn sleep(&self) -> Result<()> {
        let _guard = self.transition.lock().await;
        let prior = self.state();
        if prior == PresenceState::Sleeping {
            debug!(target: "assistd::presence", "sleep: already Sleeping, no-op");
            return Ok(());
        }

        let _inflight = self.inflight.write().await;

        let started = Instant::now();
        let service = self.llama.lock().await.take();
        let outcome = self.teardown_llama(service).await;

        *self.state.lock() = PresenceState::Sleeping;
        let _ = self.state_tx.send(PresenceState::Sleeping);
        info!(
            target: "assistd::presence",
            prior = ?prior,
            new = ?PresenceState::Sleeping,
            duration_ms = started.elapsed().as_millis() as u64,
            "transitioned {prior:?} → Sleeping"
        );
        outcome
    }

    async fn teardown_llama(&self, service: Option<LlamaService>) -> Result<()> {
        let tx = self.current_inner_shutdown.lock().take();
        if let Some(tx) = tx {
            let _ = tx.send(true);
        }

        let Some(service) = service else {
            return Ok(());
        };

        let budget = Duration::from_secs(self.timeouts.presence_sleep_secs);
        match timeout(budget, service.shutdown()).await {
            Ok(Ok(())) => Ok(()),
            Ok(Err(e)) => Err(anyhow::Error::new(e)).context("llama-server shutdown failed"),
            Err(_) => {
                warn!(
                    target: "assistd::presence",
                    timeout_secs = self.timeouts.presence_sleep_secs,
                    "llama-server shutdown timed out; the child may outlive the daemon"
                );
                Err(anyhow!(
                    "llama-server shutdown timed out after {}s",
                    self.timeouts.presence_sleep_secs
                ))
            }
        }
    }

    /// `Active → Drowsy`. Idempotent from `Drowsy`. Errors from `Sleeping`.
    ///
    /// Blocks until every outstanding [`RequestGuard`] has been dropped,
    /// so an in-flight generation completes before the model weights
    /// are unloaded.
    pub async fn drowse(&self) -> Result<()> {
        let _guard = self.transition.lock().await;
        let prior = self.state();
        match prior {
            PresenceState::Drowsy => return Ok(()),
            PresenceState::Sleeping => {
                bail!("cannot drowse from Sleeping: call wake() first");
            }
            PresenceState::Active => {}
        }

        let _inflight = self.inflight.write().await;

        let started = Instant::now();
        let unload_budget = Duration::from_secs(self.timeouts.presence_drowse_secs);
        match timeout(unload_budget, self.control.unload_model(&self.model.name)).await {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                return Err(e).with_context(|| {
                    format!("llama-server /models/unload failed for {}", self.model.name)
                });
            }
            Err(_) => {
                warn!(
                    target: "assistd::presence",
                    timeout_secs = self.timeouts.presence_drowse_secs,
                    "llama-server /models/unload timed out during drowse; transition aborted"
                );
                return Err(anyhow!(
                    "llama-server /models/unload timed out after {}s during drowse",
                    self.timeouts.presence_drowse_secs
                ));
            }
        }

        *self.state.lock() = PresenceState::Drowsy;
        let _ = self.state_tx.send(PresenceState::Drowsy);
        info!(
            target: "assistd::presence",
            prior = ?prior,
            new = ?PresenceState::Drowsy,
            duration_ms = started.elapsed().as_millis() as u64,
            "transitioned Active → Drowsy"
        );
        Ok(())
    }

    /// `Sleeping|Drowsy → Active`. Idempotent from `Active`. While
    /// running, [`Self::wake_in_progress`] reports the start time.
    pub async fn wake(&self) -> Result<()> {
        let _guard = self.transition.lock().await;
        let prior = self.state();
        if prior == PresenceState::Active {
            return Ok(());
        }

        let _wake_marker = WakeMarker::new(Arc::clone(&self.wake_started));

        let started = Instant::now();
        match prior {
            PresenceState::Drowsy => self.load_model_and_wait().await?,
            PresenceState::Sleeping => self.cold_start().await?,
            PresenceState::Active => unreachable!("short-circuited above"),
        }

        *self.state.lock() = PresenceState::Active;
        let _ = self.state_tx.send(PresenceState::Active);
        info!(
            target: "assistd::presence",
            prior = ?prior,
            new = ?PresenceState::Active,
            duration_ms = started.elapsed().as_millis() as u64,
            "transitioned {prior:?} → Active"
        );
        Ok(())
    }

    async fn cold_start(&self) -> Result<()> {
        let (inner_tx, inner_rx) = watch::channel(false);
        *self.current_inner_shutdown.lock() = Some(inner_tx);

        let service = match LlamaService::start(
            self.llama_server.clone(),
            self.model.clone(),
            inner_rx,
        )
        .await
        {
            Ok(service) => service,
            Err(e) => {
                *self.current_inner_shutdown.lock() = None;
                warn!(target: "assistd::presence", "wake cold-start failed: {e}");
                return Err(anyhow!(e)).context("llama-server cold-start failed during wake");
            }
        };

        *self.llama.lock().await = Some(service);

        let loaded = self.load_model_and_wait().await;
        if loaded.is_err() {
            let service = self.llama.lock().await.take();
            if let Err(e) = self.teardown_llama(service).await {
                warn!(
                    target: "assistd::presence",
                    error = %e,
                    "teardown after a failed cold start was unclean"
                );
            }
        }
        loaded
    }

    async fn load_model_and_wait(&self) -> Result<()> {
        self.control
            .load_model(&self.model.name)
            .await
            .with_context(|| format!("llama-server /models/load failed for {}", self.model.name))?;

        self.await_model_loaded().await
    }

    async fn await_model_loaded(&self) -> Result<()> {
        const LOAD_POLL_INTERVAL: Duration = Duration::from_millis(200);

        let mut ready_rx = {
            let guard = self.llama.lock().await;
            guard
                .as_ref()
                .map(LlamaService::subscribe_ready)
                .context("llama-server handle missing while loading model")?
        };

        let backstop = Duration::from_secs(self.llama_server.ready_timeout_secs.get());
        tokio::select! {
            res = self
                .control
                .wait_for_loaded(&self.model.name, backstop, LOAD_POLL_INTERVAL) =>
            {
                res.with_context(|| {
                    format!(
                        "llama-server did not finish loading {} within the {}s backstop",
                        self.model.name, self.llama_server.ready_timeout_secs
                    )
                })
            }
            () = wait_until_not_ready(&mut ready_rx) => Err(anyhow!(
                "llama-server left the ready state while loading {}",
                self.model.name
            )),
        }
    }
}

async fn wait_until_not_ready(rx: &mut watch::Receiver<ReadyState>) {
    let _ = rx.wait_for(|s| *s != ReadyState::Ready).await;
}

const LOAD_PROGRESS_INTERVAL: Duration = Duration::from_secs(3);

fn spawn_load_progress_emitter(
    presence: Arc<PresenceManager>,
    request_id: String,
    tx: tokio::sync::mpsc::Sender<assistd_ipc::Event>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(LOAD_PROGRESS_INTERVAL);
        // Skip the first immediate tick so sub-interval wakes stay quiet.
        interval.tick().await;
        loop {
            interval.tick().await;
            let Some(started) = presence.wake_in_progress() else {
                return;
            };
            let elapsed_secs = started.elapsed().as_secs();
            if tx
                .send(assistd_ipc::Event::Status {
                    id: request_id.clone(),
                    severity: crate::recovery::RecoverySeverity::Info.as_str().to_string(),
                    component: crate::recovery::Component::Llm.as_str().to_string(),
                    event: "model_loading".to_string(),
                    message: format!("loading model ({elapsed_secs}s elapsed)"),
                })
                .await
                .is_err()
            {
                return;
            }
        }
    })
}

impl PresenceManager {
    /// Manager in a fixed state with no llama child. Transitions that hit
    /// the network (`drowse`, cold-start `wake`) error.
    #[cfg(test)]
    pub(crate) fn stub(state: PresenceState) -> Arc<Self> {
        let (_tx, rx) = watch::channel(false);
        let (state_tx, _) = watch::channel(state);
        let llama_server = LlamaServerConfig {
            binary_path: "/does/not/exist".into(),
            host: std::net::Ipv4Addr::LOCALHOST.into(),
            port: nz16(1),
            gpu_layers: 1,
            ready_timeout_secs: nz64(1),
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
        };
        let model = ModelConfig {
            name: "stub/model".into(),
            context_length: nz32(1024),
        };
        let control =
            LlamaServerControl::new(&llama_server.host.to_string(), 1).expect("dummy control");
        let (stream_count_tx, _) = watch::channel(0usize);
        Arc::new(Self {
            state: StdMutex::new(state),
            transition: AsyncMutex::new(()),
            llama_server,
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

    #[cfg(test)]
    pub(crate) fn set_state_for_test(&self, s: PresenceState) {
        *self.state.lock() = s;
        let _ = self.state_tx.send(s);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn sleep_from_sleeping_is_noop() {
        let m = PresenceManager::stub(PresenceState::Sleeping);
        assert!(m.sleep().await.is_ok());
        assert!(m.sleep().await.is_ok());
        assert_eq!(m.state(), PresenceState::Sleeping);
    }

    #[tokio::test]
    async fn drowse_from_sleeping_errors() {
        let m = PresenceManager::stub(PresenceState::Sleeping);
        let err = m
            .drowse()
            .await
            .expect_err("drowse from Sleeping must error");
        assert!(err.to_string().contains("Sleeping"));
        assert_eq!(m.state(), PresenceState::Sleeping);
    }

    #[tokio::test]
    async fn ensure_active_is_noop_when_active() {
        let m = PresenceManager::stub(PresenceState::Active);
        assert!(m.ensure_active().await.is_ok());
        assert_eq!(m.state(), PresenceState::Active);
    }

    #[tokio::test]
    async fn wake_from_active_is_noop() {
        let m = PresenceManager::stub(PresenceState::Active);
        assert!(m.wake().await.is_ok());
        assert_eq!(m.state(), PresenceState::Active);
    }

    #[tokio::test]
    async fn drowse_from_drowsy_is_noop() {
        let m = PresenceManager::stub(PresenceState::Drowsy);
        assert!(m.drowse().await.is_ok());
        assert_eq!(m.state(), PresenceState::Drowsy);
    }

    #[tokio::test]
    async fn sleep_from_active_broadcasts_sleeping() {
        let m = PresenceManager::stub(PresenceState::Active);
        let mut rx = m.subscribe();
        m.sleep().await.unwrap();
        assert_eq!(*rx.borrow_and_update(), PresenceState::Sleeping);
    }

    #[tokio::test]
    async fn subscribe_sees_test_set_state() {
        let m = PresenceManager::stub(PresenceState::Sleeping);
        let mut rx = m.subscribe();
        m.set_state_for_test(PresenceState::Drowsy);
        rx.changed().await.unwrap();
        assert_eq!(*rx.borrow_and_update(), PresenceState::Drowsy);
    }

    #[tokio::test]
    async fn cycle_from_sleeping_goes_to_active_logically() {
        let start = PresenceState::Sleeping;
        assert_eq!(start.next(), PresenceState::Active);
    }

    fn sleep_cfg(drowsy: u64, sleep: u64) -> crate::SleepConfig {
        let mut cfg = crate::Config::default().sleep;
        cfg.idle_to_drowsy_mins = drowsy;
        cfg.idle_to_sleep_mins = sleep;
        cfg
    }

    #[tokio::test]
    async fn ensure_active_resets_activity_timer() {
        let m = PresenceManager::stub(PresenceState::Active);
        tokio::time::sleep(Duration::from_millis(50)).await;
        let before = m.idle_duration();
        assert!(before >= Duration::from_millis(40));
        m.ensure_active().await.unwrap();
        let after = m.idle_duration();
        assert!(after < before);
        assert!(after < Duration::from_millis(20));
    }

    #[tokio::test]
    async fn set_presence_resets_activity_timer() {
        let m = PresenceManager::stub(PresenceState::Active);
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(m.idle_duration() >= Duration::from_millis(40));
        m.set_presence(PresenceState::Active).await.unwrap();
        assert!(m.idle_duration() < Duration::from_millis(20));
    }

    #[tokio::test]
    async fn cycle_resets_activity_timer() {
        let m = PresenceManager::stub(PresenceState::Drowsy);
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(m.idle_duration() >= Duration::from_millis(40));
        m.cycle().await.unwrap();
        assert!(m.idle_duration() < Duration::from_millis(20));
    }

    #[tokio::test]
    async fn wake_from_active_does_not_reset_activity_timer() {
        let m = PresenceManager::stub(PresenceState::Active);
        tokio::time::sleep(Duration::from_millis(50)).await;
        let before = m.idle_duration();
        m.wake().await.unwrap();
        let after = m.idle_duration();
        assert!(after >= before);
    }

    #[test]
    fn time_until_next_transition_active_counts_down_to_drowsy() {
        let m = PresenceManager::stub(PresenceState::Active);
        let cfg = sleep_cfg(30, 120);
        let d = m.time_until_next_transition(&cfg).unwrap();
        assert!(d <= Duration::from_secs(30 * 60));
        assert!(d >= Duration::from_secs(30 * 60).saturating_sub(Duration::from_secs(5)));
    }

    #[test]
    fn time_until_next_transition_sleeping_returns_none() {
        let m = PresenceManager::stub(PresenceState::Sleeping);
        assert!(m.time_until_next_transition(&sleep_cfg(30, 120)).is_none());
    }

    #[test]
    fn time_until_next_transition_active_with_drowsy_disabled_uses_sleep() {
        let m = PresenceManager::stub(PresenceState::Active);
        let d = m.time_until_next_transition(&sleep_cfg(0, 120)).unwrap();
        assert!(d <= Duration::from_secs(120 * 60));
    }

    #[test]
    fn time_until_next_transition_active_with_both_disabled_returns_none() {
        let m = PresenceManager::stub(PresenceState::Active);
        assert!(m.time_until_next_transition(&sleep_cfg(0, 0)).is_none());
    }

    #[test]
    fn time_until_next_transition_drowsy_with_sleep_disabled_returns_none() {
        let m = PresenceManager::stub(PresenceState::Drowsy);
        assert!(m.time_until_next_transition(&sleep_cfg(30, 0)).is_none());
    }

    #[test]
    fn time_until_next_transition_drowsy_counts_down_to_sleep() {
        let m = PresenceManager::stub(PresenceState::Drowsy);
        let d = m.time_until_next_transition(&sleep_cfg(30, 120)).unwrap();
        assert!(d <= Duration::from_secs(120 * 60));
    }

    #[tokio::test]
    async fn acquire_request_guard_fast_path_when_active() {
        let m = PresenceManager::stub(PresenceState::Active);
        let g = tokio::time::timeout(Duration::from_millis(100), m.acquire_request_guard())
            .await
            .expect("acquire did not complete in time")
            .expect("acquire returned Err");
        drop(g);
        assert_eq!(m.state(), PresenceState::Active);
    }

    #[tokio::test]
    async fn sleep_defers_for_inflight_request() {
        let m = PresenceManager::stub(PresenceState::Active);
        let guard = m.acquire_request_guard().await.unwrap();

        let m2 = Arc::clone(&m);
        let sleep_task = tokio::spawn(async move { m2.sleep().await });

        // Sleep must block while the request guard is alive.
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(
            !sleep_task.is_finished(),
            "sleep completed while request guard held"
        );

        // Drop guard; sleep should now proceed.
        drop(guard);
        let res = tokio::time::timeout(Duration::from_secs(2), sleep_task)
            .await
            .expect("sleep did not complete after guard dropped")
            .expect("sleep task panicked");
        res.expect("sleep returned Err");
        assert_eq!(m.state(), PresenceState::Sleeping);
    }

    #[tokio::test]
    async fn drowse_defers_for_inflight_request() {
        let m = PresenceManager::stub(PresenceState::Active);
        let guard = m.acquire_request_guard().await.unwrap();

        let m2 = Arc::clone(&m);
        let drowse_task = tokio::spawn(async move { m2.drowse().await });

        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(
            !drowse_task.is_finished(),
            "drowse must block while request guard is held"
        );

        drop(guard);
        let _ = tokio::time::timeout(Duration::from_secs(2), drowse_task)
            .await
            .expect("drowse did not unblock after guard dropped");
    }

    #[tokio::test]
    async fn stream_guard_increments_and_decrements_count() {
        let m = PresenceManager::stub(PresenceState::Active);
        let mut rx = m.subscribe_llm_streams();
        assert_eq!(*rx.borrow_and_update(), 0);
        let g1 = m.acquire_stream_guard();
        assert_eq!(*m.subscribe_llm_streams().borrow(), 1);
        let g2 = m.acquire_stream_guard();
        assert_eq!(*m.subscribe_llm_streams().borrow(), 2);
        drop(g1);
        assert_eq!(*m.subscribe_llm_streams().borrow(), 1);
        drop(g2);
        assert_eq!(*m.subscribe_llm_streams().borrow(), 0);
    }

    #[tokio::test]
    async fn wait_until_llm_idle_returns_true_immediately_when_zero() {
        let m = PresenceManager::stub(PresenceState::Active);
        let ok = tokio::time::timeout(
            Duration::from_millis(20),
            m.wait_until_llm_idle(Duration::from_secs(5)),
        )
        .await
        .expect("wait did not complete fast");
        assert!(ok);
    }

    #[tokio::test]
    async fn wait_until_llm_idle_times_out_when_busy() {
        let m = PresenceManager::stub(PresenceState::Active);
        let _g = m.acquire_stream_guard();
        let ok = m.wait_until_llm_idle(Duration::from_millis(30)).await;
        assert!(!ok, "wait should have timed out while a guard is held");
    }

    #[tokio::test]
    async fn wait_until_llm_idle_returns_true_after_guard_dropped() {
        let m = PresenceManager::stub(PresenceState::Active);
        let g = m.acquire_stream_guard();
        let m2 = Arc::clone(&m);
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(30)).await;
            drop(g);
            // Keep the Arc alive for the spawn so the guard drop observes the watch.
            let _ = m2;
        });
        let ok = m.wait_until_llm_idle(Duration::from_millis(500)).await;
        assert!(ok, "wait should have resolved once the guard dropped");
    }

    #[tokio::test]
    async fn stream_guard_does_not_block_sleep() {
        let m = PresenceManager::stub(PresenceState::Active);
        let _stream = m.acquire_stream_guard();
        let m2 = Arc::clone(&m);
        let sleep_task = tokio::spawn(async move { m2.sleep().await });
        let res = tokio::time::timeout(Duration::from_secs(1), sleep_task)
            .await
            .expect("sleep was blocked by an LLM stream guard");
        res.expect("sleep task panicked")
            .expect("sleep returned Err");
        assert_eq!(m.state(), PresenceState::Sleeping);
    }

    #[tokio::test]
    async fn wake_marker_cleared_on_error_path() {
        let m = PresenceManager::stub(PresenceState::Drowsy);
        assert!(m.wake_in_progress().is_none());
        let err = m.wake().await;
        assert!(err.is_err(), "wake must fail against dummy control");
        assert!(
            m.wake_in_progress().is_none(),
            "wake_in_progress must be cleared after wake returns, even on error"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn rapid_sleep_guard_loop_no_crash() {
        let m = PresenceManager::stub(PresenceState::Active);

        let mut readers = Vec::new();
        for _ in 0..4 {
            let m = Arc::clone(&m);
            readers.push(tokio::spawn(async move {
                for _ in 0..50 {
                    match m.acquire_request_guard().await {
                        Ok(g) => {
                            tokio::task::yield_now().await;
                            drop(g);
                        }
                        Err(_) => tokio::task::yield_now().await,
                    }
                }
            }));
        }

        let m2 = Arc::clone(&m);
        let writer = tokio::spawn(async move {
            for _ in 0..100 {
                m2.sleep().await.expect("sleep errored");
                m2.set_state_for_test(PresenceState::Active);
                tokio::task::yield_now().await;
            }
        });

        tokio::time::timeout(Duration::from_secs(30), async move {
            writer.await.expect("writer panicked");
            for r in readers {
                r.await.expect("reader panicked");
            }
        })
        .await
        .expect("rapid toggle workload deadlocked");
    }
}
