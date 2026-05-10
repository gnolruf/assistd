//! Daemon-level glue between the continuous listener and the LLM
//! agent loop.
//!
//! Spawns two background tasks that share a shutdown channel with the
//! rest of the daemon:
//!
//! 1. Utterance forwarder - subscribes to the listener's
//!    broadcast, and for each completed transcript runs
//!    `AppState::handle_query` (the same entry point that socket-side
//!    queries use). Events from that turn are written to a throwaway
//!    mpsc that we drain to `/dev/null`; no IPC client is attached.
//!
//! 2. Presence-gated toggler - watches presence transitions and
//!    pauses the listener when the daemon goes `Sleeping`, resumes
//!    when it goes `Active`. Prevents stray room speech from
//!    repeatedly warming up llama-server.
//!
//! On shutdown both tasks observe the daemon's shutdown channel and
//! exit cleanly; the shutdown path does not forcibly stop the
//! listener (that's the daemon's `presence.sleep()` teardown).

use std::sync::Arc;
use std::time::Duration;

use assistd_core::{AppState, ContinuousListener, PresenceManager, PresenceState};
use assistd_ipc::Event;
use tokio::sync::{mpsc, watch};
use tokio::task::{JoinHandle, JoinSet};
use tracing::{Instrument, error, info, warn};

/// Join handles for the two tasks spawned by [`spawn`].
///
/// The daemon holds these until shutdown and awaits each in order.
pub struct ListenDispatcherHandles {
    /// Utterance-forwarder task: routes transcriptions to [`AppState::handle_query`].
    pub forwarder: JoinHandle<()>,
    /// Presence-gate task: pauses/resumes the listener on sleep transitions.
    pub presence_gate: JoinHandle<()>,
}

/// Spawn the utterance-forwarder and presence-gate background tasks.
///
/// Returns [`ListenDispatcherHandles`] for the daemon to await at shutdown.
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
    let grace = Duration::from_secs(state.config.daemon.shutdown_grace_secs);
    let mut utterances = listener.subscribe_utterances();
    let mut handlers: JoinSet<()> = JoinSet::new();
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
                        let span = tracing::info_span!(
                            "listen",
                            id = %id,
                            req = "query",
                        );
                        handlers.spawn(
                            async move {
                                let (tx, mut rx) = mpsc::channel::<Event>(32);
                                let drain = async {
                                    while rx.recv().await.is_some() {}
                                };
                                let query = async {
                                    if let Err(e) =
                                        state.clone().handle_query(id, text, Vec::new(), tx).await
                                    {
                                        warn!(
                                            target: "assistd::listen",
                                            "listen-triggered query failed: {e:#}"
                                        );
                                    }
                                };
                                tokio::join!(drain, query);
                            }
                            .instrument(span),
                        );
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
                        break;
                    }
                }
            }
            Some(res) = handlers.join_next(), if !handlers.is_empty() => {
                if let Err(e) = res
                    && e.is_panic()
                {
                    error!(
                        target: "assistd::listen",
                        "listen-triggered query task panicked: {e}"
                    );
                }
            }
            _ = shutdown.changed() => {
                if *shutdown.borrow() {
                    break;
                }
            }
        }
    }

    let in_flight = handlers.len();
    if in_flight == 0 {
        return;
    }
    info!(
        target: "assistd::listen",
        grace_secs = grace.as_secs(),
        in_flight,
        "draining in-flight listen-triggered queries"
    );
    let drained = tokio::time::timeout(grace, async {
        while let Some(res) = handlers.join_next().await {
            if let Err(e) = res
                && e.is_panic()
            {
                error!(
                    target: "assistd::listen",
                    "listen-triggered query task panicked: {e}"
                );
            }
        }
    })
    .await;
    if drained.is_err() {
        let remaining = handlers.len();
        warn!(
            target: "assistd::listen",
            remaining,
            "shutdown grace expired; aborting remaining listen handlers"
        );
        handlers.shutdown().await;
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
