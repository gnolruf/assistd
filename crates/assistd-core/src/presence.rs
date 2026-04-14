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
use std::time::Instant;

use anyhow::{Context, Result, anyhow, bail};
use assistd_ipc::PresenceState;
use assistd_llm::{LlamaServerControl, LlamaService, ModelSpec, ServerSpec};
use tokio::sync::{Mutex as AsyncMutex, watch};
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
    server_spec: ServerSpec,
    model_spec: ModelSpec,
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
        server_spec: ServerSpec,
        model_spec: ModelSpec,
        daemon_shutdown: watch::Receiver<bool>,
    ) -> Result<Arc<Self>> {
        let control = LlamaServerControl::new(&server_spec.host, server_spec.port)
            .context("failed to construct llama-server control client")?;

        let current_inner_shutdown: Arc<StdMutex<Option<watch::Sender<bool>>>> =
            Arc::new(StdMutex::new(None));

        // Forwarder: when daemon shutdown fires, flip whatever inner-shutdown
        // sender is currently active. Lives for the daemon's lifetime.
        {
            let mut daemon_rx = daemon_shutdown.clone();
            let current = Arc::clone(&current_inner_shutdown);
            tokio::spawn(async move {
                if daemon_rx.wait_for(|v| *v).await.is_err() {
                    return;
                }
                let tx = current
                    .lock()
                    .expect("inner-shutdown mutex poisoned")
                    .clone();
                if let Some(tx) = tx {
                    let _ = tx.send(true);
                }
            });
        }

        let (state_tx, _) = watch::channel(PresenceState::Sleeping);
        let manager = Arc::new(Self {
            state: StdMutex::new(PresenceState::Sleeping),
            transition: AsyncMutex::new(()),
            server_spec,
            model_spec,
            control,
            llama: AsyncMutex::new(None),
            current_inner_shutdown,
            state_tx,
            _daemon_shutdown: daemon_shutdown,
        });

        manager
            .wake()
            .await
            .context("initial cold-start wake failed")?;
        Ok(manager)
    }

    /// Current presence state. Cheap, lock-protected snapshot.
    pub fn state(&self) -> PresenceState {
        *self.state.lock().expect("presence state mutex poisoned")
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

    /// Fast path for query handlers: if already `Active`, returns
    /// immediately; otherwise calls [`Self::wake`]. Safe to call under
    /// concurrent queries — racing callers serialise on the transition
    /// lock and only one wake actually runs.
    pub async fn ensure_active(&self) -> Result<()> {
        if self.state() == PresenceState::Active {
            return Ok(());
        }
        self.wake().await
    }

    /// Drive the manager to `target`. Convenience wrapper used by the IPC
    /// `SetPresence` handler.
    pub async fn set_presence(&self, target: PresenceState) -> Result<()> {
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
        let target = self.state().next();
        self.set_presence(target).await?;
        Ok(target)
    }

    /// `Active|Drowsy → Sleeping`. Idempotent from `Sleeping`.
    pub async fn sleep(&self) -> Result<()> {
        let _guard = self.transition.lock().await;
        let prior = self.state();
        if prior == PresenceState::Sleeping {
            debug!(target: "assistd::presence", "sleep: already Sleeping, no-op");
            return Ok(());
        }

        let started = Instant::now();
        // Flip inner-shutdown to trigger supervisor teardown.
        let tx = self
            .current_inner_shutdown
            .lock()
            .expect("inner-shutdown mutex poisoned")
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

        *self.state.lock().expect("presence state mutex poisoned") = PresenceState::Sleeping;
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

        let started = Instant::now();
        self.control
            .unload_model(&self.model_spec.name)
            .await
            .with_context(|| {
                format!(
                    "llama-server /models/unload failed for {}",
                    self.model_spec.name
                )
            })?;

        *self.state.lock().expect("presence state mutex poisoned") = PresenceState::Drowsy;
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
    pub async fn wake(&self) -> Result<()> {
        let _guard = self.transition.lock().await;
        let prior = self.state();
        if prior == PresenceState::Active {
            return Ok(());
        }

        let started = Instant::now();
        match prior {
            PresenceState::Drowsy => {
                self.control
                    .load_model(&self.model_spec.name)
                    .await
                    .with_context(|| {
                        format!(
                            "llama-server /models/load failed for {}",
                            self.model_spec.name
                        )
                    })?;
            }
            PresenceState::Sleeping => {
                let (inner_tx, inner_rx) = watch::channel(false);
                *self
                    .current_inner_shutdown
                    .lock()
                    .expect("inner-shutdown mutex poisoned") = Some(inner_tx);

                let service = LlamaService::start(
                    self.server_spec.clone(),
                    self.model_spec.clone(),
                    inner_rx,
                )
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
                    .load_model(&self.model_spec.name)
                    .await
                    .with_context(|| {
                        format!(
                            "llama-server /models/load failed for {}",
                            self.model_spec.name
                        )
                    })?;
            }
            PresenceState::Active => unreachable!("short-circuited above"),
        }

        *self.state.lock().expect("presence state mutex poisoned") = PresenceState::Active;
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
        let server_spec = ServerSpec {
            binary_path: "/does/not/exist".into(),
            host: "127.0.0.1".into(),
            port: 0,
            gpu_layers: 1,
            ready_timeout_secs: 1,
        };
        let model_spec = ModelSpec {
            name: "stub/model".into(),
            context_length: 1024,
        };
        let control = LlamaServerControl::new(&server_spec.host, 1).expect("dummy control");
        Arc::new(Self {
            state: StdMutex::new(state),
            transition: AsyncMutex::new(()),
            server_spec,
            model_spec,
            control,
            llama: AsyncMutex::new(None),
            current_inner_shutdown: Arc::new(StdMutex::new(None)),
            state_tx,
            _daemon_shutdown: rx,
        })
    }

    /// Test-only helper: force the observable state without running a
    /// transition. Also broadcasts on `state_tx` so subscribers see the change.
    /// Does not touch the transition mutex or the llama handle.
    #[cfg(test)]
    pub(crate) fn set_state_for_test(&self, s: PresenceState) {
        *self.state.lock().expect("presence state mutex poisoned") = s;
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
}
