//! Routes continuous-listen utterances into the agent loop, and pauses
//! the listener while the daemon sleeps so stray room speech does not
//! keep waking llama-server.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use assistd_core::{
    AppState, Component, ContinuousListener, PresenceManager, PresenceState, drain_join_set,
    spawn_supervised,
};
use assistd_ipc::Event;
use tokio::sync::broadcast::error::RecvError;
use tokio::sync::{mpsc, watch};
use tokio::task::{JoinHandle, JoinSet};
use tracing::{Instrument, error, info, warn};

pub struct ListenDispatcherHandles {
    pub forwarder: JoinHandle<()>,
    pub presence_gate: JoinHandle<()>,
}

pub fn spawn_dispatcher(
    state: Arc<AppState>,
    listener: Arc<dyn ContinuousListener>,
    presence: Arc<PresenceManager>,
    start_on_launch: bool,
    shutdown: watch::Receiver<bool>,
) -> ListenDispatcherHandles {
    let forwarder = spawn_supervised(
        "listen_forwarder",
        Component::ListenDispatcher,
        run_utterance_forwarder(state, listener.clone(), shutdown.clone()),
    );
    let presence_gate = spawn_supervised(
        "listen_presence_gate",
        Component::ListenDispatcher,
        run_presence_gate(listener, presence, start_on_launch, shutdown),
    );
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
                        if !trimmed.is_empty() {
                            spawn_listen_query(&mut handlers, &state, trimmed.to_string());
                        }
                    }
                    Err(RecvError::Lagged(n)) => {
                        warn!(
                            target: "assistd::listen",
                            dropped = n,
                            "utterance subscriber lagged; dropped transcripts"
                        );
                    }
                    Err(RecvError::Closed) => {
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

    drain_join_set(&mut handlers, grace, "listen-triggered query").await;
}

fn spawn_listen_query(handlers: &mut JoinSet<()>, state: &Arc<AppState>, text: String) {
    let id = format!("listen-{}", short_id());
    let span = tracing::info_span!(
        "listen",
        id = %id,
        req = "query",
    );
    handlers.spawn(run_listen_query(state.clone(), id, text).instrument(span));
}

/// Run one utterance as a query turn, publishing its events on the
/// daemon's broadcast bus.
async fn run_listen_query(state: Arc<AppState>, id: String, text: String) {
    let (tx, mut rx) = mpsc::channel::<Event>(32);
    let forward = async {
        while let Some(ev) = rx.recv().await {
            state.runtime.publish(&ev);
        }
    };
    let query = async {
        if let Err(e) = state.clone().handle_query(id, text, Vec::new(), tx).await {
            warn!(
                target: "assistd::listen",
                "listen-triggered query failed: {e:#}"
            );
        }
    };
    tokio::join!(forward, query);
}

async fn run_presence_gate(
    listener: Arc<dyn ContinuousListener>,
    presence: Arc<PresenceManager>,
    start_on_launch: bool,
    mut shutdown: watch::Receiver<bool>,
) {
    let mut rx = presence.subscribe();
    if start_on_launch {
        let initial = *rx.borrow();
        start_unless_sleeping(listener.as_ref(), initial).await;
    }

    let mut paused_by_gate = false;

    loop {
        tokio::select! {
            changed = rx.changed() => {
                if changed.is_err() {
                    return;
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

async fn start_unless_sleeping(listener: &dyn ContinuousListener, initial: PresenceState) {
    if initial == PresenceState::Sleeping {
        info!(
            target: "assistd::listen",
            "start_on_launch deferred: presence is {initial:?}"
        );
    } else if let Err(e) = listener.start().await {
        warn!(target: "assistd::listen", "start_on_launch failed: {e:#}");
    } else {
        info!(target: "assistd::listen", "continuous listening auto-started");
    }
}

fn short_id() -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{ts:x}-{n:x}")
}
