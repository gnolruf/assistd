//! Routes wake-word commands into the agent loop, opens a follow-up window
//! when each reply ends unless push-to-talk took over, and publishes the
//! listener's phase; a command heard while the daemon sleeps wakes it.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use assistd_core::{
    AppState, Component, ContinuousListener, Heard, ListenPhase, TurnOrigin, drain_join_set,
    spawn_supervised,
};
use assistd_ipc::Event;
use tokio::sync::broadcast::error::RecvError;
use tokio::sync::{mpsc, watch};
use tokio::task::{JoinHandle, JoinSet};
use tokio::time::Instant;
use tracing::{Instrument, debug, error, info, warn};

/// Id of the `ListenState` events published when the listener changes phase.
const LISTEN_PHASE_EVENT_ID: &str = "listen";

/// The running listen turn plus at most one utterance waiting behind it;
/// a newer utterance replaces the waiting one.
#[derive(Default)]
struct ListenTurns {
    running: JoinSet<()>,
    /// When the running turn's command was heard.
    running_heard_at: Option<Instant>,
    waiting: Option<HeardCommand>,
}

struct HeardCommand {
    text: String,
    heard_at: Instant,
}

impl ListenTurns {
    fn submit(&mut self, state: &Arc<AppState>, text: String) {
        let command = HeardCommand {
            text,
            heard_at: Instant::now(),
        };
        if self.running.is_empty() {
            self.spawn(state, command);
        } else if self.waiting.replace(command).is_some() {
            debug!(
                target: "assistd::listen",
                "turn still running; dropped the waiting utterance for a newer one"
            );
        }
    }

    /// After the running turn ends, start the waiting utterance or else open
    /// a follow-up window, unless push-to-talk superseded the conversation.
    fn finish_running(&mut self, state: &Arc<AppState>, listener: &dyn ContinuousListener) {
        let superseded = |heard_at: Instant| {
            listener
                .last_push_to_talk()
                .is_some_and(|pressed| pressed >= heard_at)
        };
        let finished_heard_at = self.running_heard_at.take();
        match self.waiting.take() {
            Some(command) if !superseded(command.heard_at) => self.spawn(state, command),
            Some(_) => debug!(
                target: "assistd::listen",
                "push-to-talk superseded the waiting utterance; dropped it"
            ),
            None if finished_heard_at.is_some_and(|heard_at| !superseded(heard_at)) => {
                listener.open_follow_up();
            }
            None => {}
        }
    }

    fn spawn(&mut self, state: &Arc<AppState>, command: HeardCommand) {
        self.running_heard_at = Some(command.heard_at);
        spawn_listen_query(&mut self.running, state, command.text);
    }
}

pub(super) fn spawn_dispatcher(
    state: Arc<AppState>,
    listener: Arc<dyn ContinuousListener>,
    start_on_launch: bool,
    shutdown: watch::Receiver<bool>,
) -> JoinHandle<()> {
    spawn_supervised(
        "listen_forwarder",
        Component::ListenDispatcher,
        run_utterance_forwarder(state, listener, start_on_launch, shutdown),
    )
}

async fn run_utterance_forwarder(
    state: Arc<AppState>,
    listener: Arc<dyn ContinuousListener>,
    start_on_launch: bool,
    mut shutdown: watch::Receiver<bool>,
) {
    let grace = Duration::from_secs(state.config.daemon.shutdown_grace_secs);
    let mut heard = listener.subscribe_heard();
    let mut phases = listener.subscribe_phase();
    if start_on_launch {
        start_listening(listener.as_ref()).await;
    }
    let mut turns = ListenTurns::default();
    loop {
        tokio::select! {
            res = heard.recv() => {
                if !forward_heard(&state, &mut turns, res) {
                    break;
                }
            }
            changed = phases.changed() => {
                if changed.is_err() {
                    break;
                }
                publish_phase(&state, *phases.borrow_and_update());
            }
            Some(res) = turns.running.join_next(), if !turns.running.is_empty() => {
                if let Err(e) = res
                    && e.is_panic()
                {
                    error!(
                        target: "assistd::listen",
                        "listen-triggered query task panicked: {e}"
                    );
                }
                turns.finish_running(&state, listener.as_ref());
            }
            _ = shutdown.changed() => {
                if *shutdown.borrow() {
                    break;
                }
            }
        }
    }

    drain_join_set(&mut turns.running, grace, "listen-triggered query").await;
}

/// Act on one item from the listener; false once its broadcast has closed.
fn forward_heard(
    state: &Arc<AppState>,
    turns: &mut ListenTurns,
    received: Result<Heard, RecvError>,
) -> bool {
    match received {
        Ok(Heard::WakePhrase) => {
            state.runtime.publish(&Event::WakeWordDetected {
                id: format!("listen-{}", short_id()),
            });
        }
        Ok(Heard::Command(text)) => turns.submit(state, text),
        Err(RecvError::Lagged(n)) => {
            warn!(
                target: "assistd::listen",
                dropped = n,
                "listen subscriber lagged; dropped wake phrases or commands"
            );
        }
        Err(RecvError::Closed) => {
            info!(
                target: "assistd::listen",
                "listen broadcast closed; forwarder exiting"
            );
            return false;
        }
    }
    true
}

fn publish_phase(state: &AppState, phase: ListenPhase) {
    state.runtime.publish(&Event::ListenState {
        id: LISTEN_PHASE_EVENT_ID.to_string(),
        phase,
    });
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

/// Run one utterance as a query turn, publishing its transcript and
/// events on the daemon's broadcast bus.
async fn run_listen_query(state: Arc<AppState>, id: String, text: String) {
    state.runtime.publish(&Event::Transcription {
        id: id.clone(),
        text: text.clone(),
    });
    let (tx, mut rx) = mpsc::channel::<Event>(32);
    let forward = async {
        while let Some(ev) = rx.recv().await {
            state.runtime.publish(&ev);
        }
    };
    let query = async {
        if let Err(e) = state
            .clone()
            .handle_query(id, text, Vec::new(), TurnOrigin::Voice, tx)
            .await
        {
            warn!(
                target: "assistd::listen",
                "listen-triggered query failed: {e:#}"
            );
        }
    };
    tokio::join!(forward, query);
}

async fn start_listening(listener: &dyn ContinuousListener) {
    match listener.start().await {
        Ok(()) => info!(target: "assistd::listen", "continuous listening auto-started"),
        Err(e) => warn!(target: "assistd::listen", "start_on_launch failed: {e:#}"),
    }
}

fn short_id() -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX));
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{ts:x}-{n:x}")
}
