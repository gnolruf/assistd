//! Daemon-level glue between the continuous listener and the LLM
//! agent loop.
//!
//! Spawns two background tasks that share a shutdown channel with the
//! rest of the daemon:
//!
//! 1. **Utterance forwarder** — subscribes to the listener's
//!    broadcast, and for each completed transcript runs
//!    `AppState::handle_query` (the same entry point that socket-side
//!    queries use). Events from that turn are written to a throwaway
//!    mpsc that we drain to `/dev/null` — no IPC client is attached.
//!
//! 2. **Presence-gated toggler** — watches presence transitions and
//!    pauses the listener when the daemon goes `Sleeping`, resumes
//!    when it goes `Active`. Prevents stray room speech from
//!    repeatedly warming up llama-server.
//!
//! On shutdown both tasks observe the daemon's shutdown channel and
//! exit cleanly; the shutdown path does not forcibly stop the
//! listener (that's the daemon's `presence.sleep()` teardown).

use std::sync::Arc;

use assistd_core::{AppState, ContinuousListener, PresenceManager, PresenceState};
use assistd_ipc::Event;
use tokio::sync::{mpsc, watch};
use tokio::task::JoinHandle;
use tracing::{info, warn};

/// Spawn both background tasks. Returns a `JoinHandle` for each so
/// the daemon can await them at shutdown.
pub struct ListenDispatcherHandles {
    pub forwarder: JoinHandle<()>,
    pub presence_gate: JoinHandle<()>,
}

pub fn spawn(
    state: Arc<AppState>,
    listener: Arc<dyn ContinuousListener>,
    presence: Arc<PresenceManager>,
    start_on_launch: bool,
    pause_when_sleeping: bool,
    shutdown: watch::Receiver<bool>,
) -> ListenDispatcherHandles {
    let forwarder = tokio::spawn(run_utterance_forwarder(
        state,
        listener.clone(),
        shutdown.clone(),
    ));
    let presence_gate = tokio::spawn(run_presence_gate(
        listener,
        presence,
        start_on_launch,
        pause_when_sleeping,
        shutdown,
    ));
    ListenDispatcherHandles {
        forwarder,
        presence_gate,
    }
}

async fn run_utterance_forwarder(
    state: Arc<AppState>,
    listener: Arc<dyn ContinuousListener>,
    mut shutdown: watch::Receiver<bool>,
) {
    let mut utterances = listener.subscribe_utterances();
    loop {
        tokio::select! {
            res = utterances.recv() => {
                match res {
                    Ok(text) => {
                        let trimmed = text.trim();
                        if trimmed.is_empty() {
                            continue;
                        }
                        let id = format!("listen-{}", short_id());
                        let state = state.clone();
                        let text = trimmed.to_string();
                        tokio::spawn(async move {
                            // `handle_query` wants a per-request event
                            // channel. There is no IPC client here, so
                            // drain the receiver into /dev/null. Send
                            // failures are already treated as "client
                            // disconnected" by `handle_query`, but the
                            // drain is cheaper than relying on that
                            // fallback.
                            let (tx, mut rx) = mpsc::channel::<Event>(32);
                            tokio::spawn(async move {
                                while rx.recv().await.is_some() {}
                            });
                            if let Err(e) = state.clone().handle_query(id, text, tx).await {
                                warn!(
                                    target: "assistd::listen",
                                    "listen-triggered query failed: {e:#}"
                                );
                            }
                        });
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                        warn!(
                            target: "assistd::listen",
                            dropped = n,
                            "utterance subscriber lagged; dropped transcripts"
                        );
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                        info!(
                            target: "assistd::listen",
                            "utterance broadcast closed; forwarder exiting"
                        );
                        return;
                    }
                }
            }
            _ = shutdown.changed() => {
                if *shutdown.borrow() {
                    return;
                }
            }
        }
    }
}

async fn run_presence_gate(
    listener: Arc<dyn ContinuousListener>,
    presence: Arc<PresenceManager>,
    start_on_launch: bool,
    pause_when_sleeping: bool,
    mut shutdown: watch::Receiver<bool>,
) {
    let mut rx = presence.subscribe();
    // Initial state: either auto-start, or leave off.
    if start_on_launch {
        let initial = *rx.borrow();
        let should_start = !pause_when_sleeping || initial != PresenceState::Sleeping;
        if should_start {
            if let Err(e) = listener.start().await {
                warn!(target: "assistd::listen", "start_on_launch failed: {e:#}");
            } else {
                info!(target: "assistd::listen", "continuous listening auto-started");
            }
        } else {
            info!(
                target: "assistd::listen",
                "start_on_launch deferred: presence is {initial:?}, pause_when_sleeping = true"
            );
        }
    }

    // Set to true when the presence gate itself paused the listener
    // on a Sleeping transition. Used to decide whether the matching
    // wake should auto-resume — we never clobber an explicit
    // user-driven stop.
    let mut paused_by_gate = false;

    loop {
        tokio::select! {
            changed = rx.changed() => {
                if changed.is_err() {
                    return;
                }
                if !pause_when_sleeping {
                    continue;
                }
                let new_state = *rx.borrow_and_update();
                match new_state {
                    PresenceState::Sleeping => {
                        if listener.is_active() {
                            if let Err(e) = listener.stop().await {
                                warn!(
                                    target: "assistd::listen",
                                    "pausing on sleep failed: {e:#}"
                                );
                            } else {
                                paused_by_gate = true;
                                info!(
                                    target: "assistd::listen",
                                    "paused: presence → sleeping"
                                );
                            }
                        }
                    }
                    PresenceState::Active | PresenceState::Drowsy => {
                        if paused_by_gate && !listener.is_active() {
                            match listener.start().await {
                                Ok(()) => info!(
                                    target: "assistd::listen",
                                    "resumed: presence → {new_state:?}"
                                ),
                                Err(e) => warn!(
                                    target: "assistd::listen",
                                    "auto-resume failed: {e:#}"
                                ),
                            }
                            paused_by_gate = false;
                        }
                    }
                }
            }
            _ = shutdown.changed() => {
                if *shutdown.borrow() {
                    return;
                }
            }
        }
    }
}

fn short_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{ts:x}-{n:x}")
}
