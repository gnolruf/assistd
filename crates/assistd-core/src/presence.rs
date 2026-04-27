//! Daemon presence state machine: `Active`, `Drowsy`, `Sleeping`.
//!
//! The presence state lets the daemon free GPU resources on demand while
//! keeping its control plane (the Unix domain socket) listening in every
//! state. A query that arrives while the daemon is not `Active` blocks
//! briefly on an automatic wake and then streams a response as usual.
//!
//! State transitions:
//!
//! - **`sleep()`** — stop llama-server and release all VRAM. Idempotent from
//!   `Sleeping`. Active → Sleeping and Drowsy → Sleeping both run the same
//!   teardown path (flip an inner shutdown watch, wait for the supervisor
//!   task to join). llama-server's VRAM is freed when the child process
//!   exits.
//!
//! - **`drowse()`** — keep the llama-server process alive but unload its
//!   model weights via `POST /models/unload`, dropping VRAM to roughly the
//!   server's own runtime overhead. Only valid from `Active`; idempotent
//!   from `Drowsy`; errors from `Sleeping` (caller must `wake` first).
//!
//! - **`wake()`** — reverse of sleep/drowse. From `Sleeping`, cold-starts
//!   llama-server via [`LlamaService::start`] which blocks until `/health`
//!   returns 200, then loads the model. From `Drowsy`, just reloads the
//!   model. Idempotent from `Active`.
//!
//! The [`PresenceManager`] is held as `Arc<PresenceManager>` inside
//! [`crate::AppState`]; request handlers call [`PresenceManager::ensure_active`]
//! before dispatching work to the LLM backend.

use std::sync::{Arc, Mutex as StdMutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use assistd_config::{LlamaServerConfig, ModelConfig};
use assistd_ipc::PresenceState;
use assistd_llm::{LlamaServerControl, LlamaService};
use tokio::sync::{Mutex as AsyncMutex, OwnedRwLockReadGuard, RwLock, watch};
use tracing::{debug, info, warn};

/// Owner of the llama-server handle and the daemon-wide presence state.
///
/// See the module docs for the transition semantics. All public transition
/// methods are async and safe to call concurrently; a `Mutex` serialises
/// transitions so that, e.g., an auto-wake triggered by a query cannot
/// race with an explicit `sleep` from an IPC request.
pub struct PresenceManager {
    state: StdMutex<PresenceState>,
    // Serialises all transitions. Held across awaits, so must be a
    // tokio mutex rather than std.
    transition: AsyncMutex<()>,
    llama_server: LlamaServerConfig,
    model: ModelConfig,
    control: LlamaServerControl,
    // `Some` iff state is `Active` or `Drowsy`.
    llama: AsyncMutex<Option<LlamaService>>,
    // Per-epoch watch that `sleep()` flips to tear down the current
    // supervisor without disturbing the daemon-wide shutdown watch.
    current_inner_shutdown: Arc<StdMutex<Option<watch::Sender<bool>>>>,
    // Broadcast the current presence state to subscribers (TUI status bar,
    // future clients). Updated after each successful transition, inside the
    // transition lock so ordering is preserved.
    state_tx: watch::Sender<PresenceState>,
    // Holds the daemon shutdown receiver alive so the forwarder task
    // keeps a valid subscription.
    _daemon_shutdown: watch::Receiver<bool>,
    // Monotonic timestamp of the last user-initiated interaction
    // (query, manual presence command, hotkey, cycle). The idle monitor
    // reads this to decide when to drowse or sleep; automatic monitors
    // (GPU, idle) deliberately do not update it so their own
    // transitions don't defer further idle progress.
    last_activity: StdMutex<Instant>,
    // Shared lock over in-flight request tracking. Request handlers take
    // an owned read guard (`RequestGuard`) for the duration of the
    // generation; `sleep`/`drowse` take the write side to block until
    // all outstanding requests drain. `tokio::sync::RwLock` is
    // writer-preferring, so new requests queued after a pending
    // transition wait for that transition plus the subsequent wake
    // rather than starving the writer.
    inflight: Arc<RwLock<()>>,
    // `Some(started_at)` while a `wake` transition is executing — set
    // after the transition mutex is taken and the short-circuit check
    // passes, cleared by RAII when `wake` returns (success or error).
    // The TUI polls this each render tick to drive the "waking up"
    // indicator.
    wake_started: Arc<StdMutex<Option<Instant>>>,
    // Count of LLM streams currently in flight, maintained by
    // [`LlmStreamGuard`] via RAII. Voice transcription subscribes via
    // [`Self::wait_until_llm_idle`] and diverts to a CPU fallback when
    // the count doesn't reach zero within the configured timeout. A
    // separate signal from `inflight` because a few request paths take a
    // request guard without actually streaming on the GPU (presence
    // queries, cycles) and we don't want those to force Whisper off the
    // GPU.
    stream_count_tx: watch::Sender<usize>,
}

/// Held by request handlers for the duration of a query (or other
/// in-flight work). While any guard is alive, [`PresenceManager::sleep`]
/// and [`PresenceManager::drowse`] block — sleep cannot tear down the
/// llama-server while a response is still streaming.
///
/// Created via [`PresenceManager::acquire_request_guard`]; released on
/// drop.
pub struct RequestGuard {
    _guard: OwnedRwLockReadGuard<()>,
}

/// RAII counter for "an LLM stream is currently running". Bumps the
/// shared count on construction and decrements on drop. Held by LLM
/// query handlers for the duration of their streaming lifetime so the
/// voice transcriber can decide whether to queue briefly or fall back
/// to CPU. Does not block sleep/drowse — unlike [`RequestGuard`], which
/// is how the two signals differ.
pub struct LlmStreamGuard {
    tx: watch::Sender<usize>,
}

impl Drop for LlmStreamGuard {
    fn drop(&mut self) {
        self.tx.send_modify(|n| *n = n.saturating_sub(1));
    }
}

/// RAII marker for "a wake transition is in progress". Constructed at
/// the top of [`PresenceManager::wake`] after the short-circuit check;
/// cleared on drop so every return path — `?`-propagated error, panic,
/// success — leaves `wake_started` back at `None`.
struct WakeMarker {
    slot: Arc<StdMutex<Option<Instant>>>,
}

impl WakeMarker {
    fn new(slot: Arc<StdMutex<Option<Instant>>>) -> Self {
        *slot.lock().unwrap_or_else(|e| e.into_inner()) = Some(Instant::now());
        Self { slot }
    }
}

impl Drop for WakeMarker {
    fn drop(&mut self) {
        *self.slot.lock().unwrap_or_else(|e| e.into_inner()) = None;
    }
}

impl PresenceManager {
    /// Creates a new manager and performs an initial cold-start wake so the
    /// daemon is in `Active` before it starts serving the socket.
    ///
    /// `daemon_shutdown` is a subscriber on the daemon's global shutdown
    /// watch; flipping that watch cancels any in-flight cold-start wake
    /// (via a forwarder that mirrors the daemon signal into the current
    /// inner shutdown) so the daemon can exit promptly even if a wake is
    /// blocked on `/health`.
    pub async fn new_active(
        llama_server: LlamaServerConfig,
        model: ModelConfig,
        daemon_shutdown: watch::Receiver<bool>,
    ) -> Result<Arc<Self>> {
        let control = LlamaServerControl::new(&llama_server.host, llama_server.port)
            .context("failed to construct llama-server control client")?;

        let current_inner_shutdown: Arc<StdMutex<Option<watch::Sender<bool>>>> =
            Arc::new(StdMutex::new(None));

        // Forwarder: when daemon shutdown fires, flip whatever inner-shutdown
        // sender is currently active. Lives for the daemon's lifetime.
        //
        // The `JoinHandle` is intentionally discarded: the task is bounded
        // by the daemon-shutdown watch channel. It exits via one of two
        // paths — `wait_for` returns `Ok` once shutdown fires (the normal
        // case), or `Err` if every `watch::Sender` clone for daemon
        // shutdown is dropped (which implies the daemon is already tearing
        // down). `PresenceManager` also holds `_daemon_shutdown:
        // watch::Receiver` below to keep the subscription valid for the
        // manager's lifetime.
        {
            let mut daemon_rx = daemon_shutdown.clone();
            let current = Arc::clone(&current_inner_shutdown);
            tokio::spawn(async move {
                if daemon_rx.wait_for(|v| *v).await.is_err() {
                    return;
                }
                let tx = current.lock().unwrap_or_else(|e| e.into_inner()).clone();
                if let Some(tx) = tx {
                    let _ = tx.send(true);
                }
            });
        }

        let (state_tx, _) = watch::channel(PresenceState::Sleeping);
        let (stream_count_tx, _) = watch::channel(0usize);
        let manager = Arc::new(Self {
            state: StdMutex::new(PresenceState::Sleeping),
            transition: AsyncMutex::new(()),
            llama_server,
            model,
            control,
            llama: AsyncMutex::new(None),
            current_inner_shutdown,
            state_tx,
            _daemon_shutdown: daemon_shutdown,
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

    /// Current presence state. Cheap, lock-protected snapshot.
    pub fn state(&self) -> PresenceState {
        *self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Record that the user just interacted with the daemon. Called at
    /// the top of every user-facing wrapper (`ensure_active`,
    /// `set_presence`, `cycle`). Deliberately NOT called by the
    /// low-level transition methods (`wake`/`drowse`/`sleep`) so that
    /// automatic monitors (GPU, idle) don't defer their own progress.
    fn mark_activity(&self) {
        *self.last_activity.lock().unwrap_or_else(|e| e.into_inner()) = Instant::now();
    }

    /// Time since the last recorded user interaction. Read by the idle
    /// monitor to decide when to drowse/sleep, and by the TUI status
    /// bar for its countdown display.
    pub fn idle_duration(&self) -> Duration {
        self.last_activity
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .elapsed()
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

    /// Non-blocking llama-server PID lookup for fast-path callers that
    /// can't await (e.g. the voice crate's GPU-contention probe running
    /// inside a sync context). Returns `None` when the llama mutex is
    /// currently held by a transition — acceptable for a periodic probe
    /// since it will retry on the next call.
    pub fn llama_pid_blocking(&self) -> Option<u32> {
        self.llama
            .try_lock()
            .ok()
            .and_then(|svc| svc.as_ref().and_then(|s| s.pid()))
    }

    /// Fast path for query handlers: if already `Active`, returns
    /// immediately; otherwise calls [`Self::wake`]. Safe to call under
    /// concurrent queries — racing callers serialise on the transition
    /// lock and only one wake actually runs.
    #[tracing::instrument(skip(self), fields(from = ?self.state()))]
    pub async fn ensure_active(&self) -> Result<()> {
        self.mark_activity();
        if self.state() == PresenceState::Active {
            return Ok(());
        }
        self.wake().await
    }

    /// Request a guard that keeps the daemon `Active` for as long as
    /// the returned [`RequestGuard`] is alive. While any guard exists,
    /// [`Self::sleep`] and [`Self::drowse`] block — this is the
    /// mechanism that prevents a concurrent sleep from tearing down
    /// llama-server mid-generation.
    ///
    /// The guard is acquired *before* the state check, so `state ==
    /// Active` observed under the guard is a stable invariant for the
    /// guard's lifetime (a sleep that wants to flip the state first
    /// needs the write side of the same lock, which this guard blocks).
    ///
    /// If the state is not `Active` when the read guard is taken, the
    /// guard is dropped and [`Self::ensure_active`] runs to wake the
    /// daemon, then a new read guard is acquired and the state
    /// re-checked. A bounded retry guards against pathological
    /// sleep/wake churn.
    pub async fn acquire_request_guard(self: &Arc<Self>) -> Result<RequestGuard> {
        self.mark_activity();
        const MAX_RETRIES: usize = 3;
        for _ in 0..MAX_RETRIES {
            let guard = self.inflight.clone().read_owned().await;
            if self.state() == PresenceState::Active {
                return Ok(RequestGuard { _guard: guard });
            }
            drop(guard);
            self.ensure_active().await?;
        }
        bail!("failed to acquire active request guard after {MAX_RETRIES} retries")
    }

    /// `Some(started_at)` if a wake transition is currently running;
    /// `None` otherwise. Used by the TUI to drive a "waking up"
    /// indicator during cold-start wakes (which can take tens of
    /// seconds to minutes).
    pub fn wake_in_progress(&self) -> Option<Instant> {
        *self.wake_started.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Register an in-flight LLM stream. The returned guard decrements
    /// the shared count on drop. Held by query handlers for the full
    /// streaming lifetime of a response so the voice transcriber can
    /// avoid contending with the LLM on the same GPU. Unlike
    /// [`Self::acquire_request_guard`], this guard does not block
    /// sleep/drowse; the existing request guard still carries that
    /// invariant.
    pub fn acquire_stream_guard(&self) -> LlmStreamGuard {
        self.stream_count_tx.send_modify(|n| *n += 1);
        LlmStreamGuard {
            tx: self.stream_count_tx.clone(),
        }
    }

    /// Waits until the LLM-stream count drops to zero, up to `timeout`.
    /// Returns `true` if the count reached zero within the budget
    /// (including the common case where no stream is running at call
    /// time) and `false` on timeout. Cancel-safe: the internal
    /// `wait_for` releases cleanly if the caller drops the future.
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

    /// Subscribe to changes in the active LLM-stream count. Mostly
    /// useful for diagnostics; the voice transcriber uses
    /// [`Self::wait_until_llm_idle`] directly.
    pub fn subscribe_llm_streams(&self) -> watch::Receiver<usize> {
        self.stream_count_tx.subscribe()
    }

    /// Drive the manager to `target`. Convenience wrapper used by the IPC
    /// `SetPresence` handler.
    #[tracing::instrument(skip(self), fields(from = ?self.state()))]
    pub async fn set_presence(&self, target: PresenceState) -> Result<()> {
        self.mark_activity();
        match target {
            PresenceState::Active => self.wake().await,
            PresenceState::Drowsy => self.drowse().await,
            PresenceState::Sleeping => self.sleep().await,
        }
    }

    /// Advance one step along `Active → Drowsy → Sleeping → Active`. Used by
    /// the global hotkey listener and the `assistd cycle` CLI command.
    ///
    /// Not strictly atomic against concurrent callers: two racing `cycle`s
    /// could both observe the same `current` and both attempt the same
    /// target, in which case the loser is a no-op. That's acceptable — the
    /// transition mutex still serialises the actual state change, so we
    /// never skip or split a step.
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

        // Wait for all in-flight requests to drop their read guards.
        // Writer-preferring: new requests acquiring read guards after
        // this point will block until the write guard is released.
        let _inflight = self.inflight.write().await;

        let started = Instant::now();
        // Flip inner-shutdown to trigger supervisor teardown.
        let tx = self
            .current_inner_shutdown
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take();
        if let Some(tx) = tx {
            let _ = tx.send(true);
        }

        // Take the service out of its slot and join its supervisor task.
        let service = self.llama.lock().await.take();
        if let Some(service) = service {
            service
                .shutdown()
                .await
                .context("llama-server shutdown during sleep")?;
        }

        *self.state.lock().unwrap_or_else(|e| e.into_inner()) = PresenceState::Sleeping;
        let _ = self.state_tx.send(PresenceState::Sleeping);
        info!(
            target: "assistd::presence",
            prior = ?prior,
            new = ?PresenceState::Sleeping,
            duration_ms = started.elapsed().as_millis() as u64,
            "transitioned {prior:?} → Sleeping"
        );
        Ok(())
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

        // Wait for all in-flight requests to drop their read guards
        // before unloading model weights.
        let _inflight = self.inflight.write().await;

        let started = Instant::now();
        self.control
            .unload_model(&self.model.name)
            .await
            .with_context(|| {
                format!("llama-server /models/unload failed for {}", self.model.name)
            })?;

        *self.state.lock().unwrap_or_else(|e| e.into_inner()) = PresenceState::Drowsy;
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

    /// `Sleeping|Drowsy → Active`. Idempotent from `Active`.
    ///
    /// While this is running, [`Self::wake_in_progress`] returns
    /// `Some(started_at)` so the TUI can display a "waking up"
    /// indicator. The marker is installed via RAII after the
    /// short-circuit check and cleared on every return path.
    pub async fn wake(&self) -> Result<()> {
        let _guard = self.transition.lock().await;
        let prior = self.state();
        if prior == PresenceState::Active {
            return Ok(());
        }

        // Mark wake-in-progress for TUI. Placed *after* the short-circuit
        // so idempotent wakes from Active don't briefly flicker the
        // indicator.
        let _wake_marker = WakeMarker::new(Arc::clone(&self.wake_started));

        let started = Instant::now();
        match prior {
            PresenceState::Drowsy => {
                self.control
                    .load_model(&self.model.name)
                    .await
                    .with_context(|| {
                        format!("llama-server /models/load failed for {}", self.model.name)
                    })?;
            }
            PresenceState::Sleeping => {
                let (inner_tx, inner_rx) = watch::channel(false);
                *self
                    .current_inner_shutdown
                    .lock()
                    .unwrap_or_else(|e| e.into_inner()) = Some(inner_tx);

                let service =
                    LlamaService::start(self.llama_server.clone(), self.model.clone(), inner_rx)
                        .await
                        .map_err(|e| {
                            warn!(
                                target: "assistd::presence",
                                "wake cold-start failed: {e}"
                            );
                            anyhow!(e)
                        })
                        .context("llama-server cold-start failed during wake")?;

                *self.llama.lock().await = Some(service);

                self.control
                    .load_model(&self.model.name)
                    .await
                    .with_context(|| {
                        format!("llama-server /models/load failed for {}", self.model.name)
                    })?;
            }
            PresenceState::Active => unreachable!("short-circuited above"),
        }

        *self.state.lock().unwrap_or_else(|e| e.into_inner()) = PresenceState::Active;
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
}

impl PresenceManager {
    /// Test-only constructor: fabricates a manager in a specific state with
    /// dummy server/model specs and no real llama child. Methods that would
    /// hit the network (`drowse`, cold-start `wake`) will error on dummy
    /// connections — use only for unit tests that exercise guards,
    /// idempotency, and state transitions that don't require a live server.
    #[cfg(test)]
    pub(crate) fn stub(state: PresenceState) -> Arc<Self> {
        let (_tx, rx) = watch::channel(false);
        let (state_tx, _) = watch::channel(state);
        let llama_server = LlamaServerConfig {
            binary_path: "/does/not/exist".into(),
            host: "127.0.0.1".into(),
            port: 0,
            gpu_layers: 1,
            ready_timeout_secs: 1,
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
            context_length: 1024,
        };
        let control = LlamaServerControl::new(&llama_server.host, 1).expect("dummy control");
        let (stream_count_tx, _) = watch::channel(0usize);
        Arc::new(Self {
            state: StdMutex::new(state),
            transition: AsyncMutex::new(()),
            llama_server,
            model,
            control,
            llama: AsyncMutex::new(None),
            current_inner_shutdown: Arc::new(StdMutex::new(None)),
            state_tx,
            _daemon_shutdown: rx,
            last_activity: StdMutex::new(Instant::now()),
            inflight: Arc::new(RwLock::new(())),
            wake_started: Arc::new(StdMutex::new(None)),
            stream_count_tx,
        })
    }

    /// Test-only helper: force the observable state without running a
    /// transition. Also broadcasts on `state_tx` so subscribers see the change.
    /// Does not touch the transition mutex or the llama handle.
    #[cfg(test)]
    pub(crate) fn set_state_for_test(&self, s: PresenceState) {
        *self.state.lock().unwrap_or_else(|e| e.into_inner()) = s;
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
        // `borrow()` returns the latest value even if the initial one was
        // missed; the broadcast inside `sleep` must have overwritten it.
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
        // `cycle` from Sleeping computes target=Active, then set_presence(Active)
        // drives `wake`. With the stub, wake from Sleeping attempts a real
        // cold-start and fails — we just assert the target selection by
        // reading the next() helper directly here; full cycle paths are
        // exercised by integration with a live daemon.
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
        // set_presence(Active) from Active short-circuits in wake() but
        // mark_activity still runs as the first line of set_presence.
        m.set_presence(PresenceState::Active).await.unwrap();
        assert!(m.idle_duration() < Duration::from_millis(20));
    }

    #[tokio::test]
    async fn cycle_resets_activity_timer() {
        // Stub starts in Drowsy; cycle() computes target=Sleeping and
        // calls set_presence(Sleeping) → sleep(), which is a pure
        // local-state transition with no network I/O in the stub.
        let m = PresenceManager::stub(PresenceState::Drowsy);
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(m.idle_duration() >= Duration::from_millis(40));
        m.cycle().await.unwrap();
        assert!(m.idle_duration() < Duration::from_millis(20));
    }

    #[tokio::test]
    async fn wake_from_active_does_not_reset_activity_timer() {
        // Low-level transition methods are called directly by automatic
        // monitors (GPU, idle). They must NOT reset last_activity, or
        // those automatic transitions would indefinitely defer their
        // own forward progress.
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

        // Drop guard — sleep should now proceed.
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
        // Same mechanism as sleep: the inflight write lock blocks
        // drowse until all read guards drop. We only verify the block
        // behaviour; the drowse itself will error on the dummy control
        // client once it gets past the inflight wait, which is fine.
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
        // After drop, drowse completes (with Err from the dummy
        // control client, since load/unload hit an unreachable port).
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
        // Stream guards are a separate signal from the inflight RwLock;
        // holding one must not prevent a concurrent sleep() from
        // proceeding. sleep() will still block on a real RequestGuard
        // via inflight, but that's a different path verified elsewhere.
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
        // Wake from Drowsy calls `control.load_model`, which hits an
        // unreachable port in the stub and returns Err. The RAII
        // marker must still clear on the error path.
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
        // Stress the inflight RwLock + transition mutex interaction:
        // many tasks take/release read guards while another loop calls
        // sleep() and restores state via set_state_for_test. The stub
        // can't cold-start wake, so the writer forces state back to
        // Active after each sleep, which lets the retry loop in
        // acquire_request_guard make progress. The point of the test
        // is to confirm no deadlock or state corruption under churn —
        // not to measure throughput. A generous wall-clock timeout
        // wraps the whole workload.
        let m = PresenceManager::stub(PresenceState::Active);

        let mut readers = Vec::new();
        for _ in 0..4 {
            let m = Arc::clone(&m);
            readers.push(tokio::spawn(async move {
                for _ in 0..20 {
                    match m.acquire_request_guard().await {
                        Ok(g) => {
                            tokio::task::yield_now().await;
                            drop(g);
                        }
                        // Reader may observe Sleeping between writer's
                        // sleep() completing and set_state_for_test(Active)
                        // running; ensure_active() then fails in the
                        // stub. Retry exhaustion is fine — just keep
                        // going.
                        Err(_) => tokio::task::yield_now().await,
                    }
                }
            }));
        }

        let m2 = Arc::clone(&m);
        let writer = tokio::spawn(async move {
            for _ in 0..10 {
                m2.sleep().await.expect("sleep errored");
                m2.set_state_for_test(PresenceState::Active);
                tokio::task::yield_now().await;
            }
        });

        // Overall workload cap: if anything deadlocks, this trips and
        // the test fails with a clear message rather than hanging.
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
