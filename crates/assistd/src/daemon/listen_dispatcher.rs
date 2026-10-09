//! Routes continuous-listen utterances into the agent loop, and pauses
//! the listener while the daemon sleeps so stray room speech does not
//! keep waking llama-server.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use assistd_core::{
    AppState, Component, ContinuousListener, PresenceManager, PresenceState, TurnOrigin,
    drain_join_set, spawn_supervised,
};
use assistd_ipc::Event;
use tokio::sync::broadcast::error::RecvError;
use tokio::sync::{mpsc, watch};
use tokio::task::{JoinHandle, JoinSet};
use tracing::{Instrument, error, info, warn};

pub(super) struct ListenDispatcherHandles {
    pub forwarder: JoinHandle<()>,
    pub presence_gate: JoinHandle<()>,
}

pub(super) fn spawn_dispatcher(
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

async fn run_presence_gate(
    listener: Arc<dyn ContinuousListener>,
    presence: Arc<PresenceManager>,
    start_on_launch: bool,
    mut shutdown: watch::Receiver<bool>,
) {
    let mut rx = presence.subscribe();
    let initial = *rx.borrow_and_update();
    let mut paused_by_gate =
        start_on_launch && start_unless_asleep(listener.as_ref(), initial).await;

    loop {
        tokio::select! {
            changed = rx.changed() => {
                if changed.is_err() {
                    return;
                }
                let new_state = *rx.borrow_and_update();
                paused_by_gate =
                    follow_presence(listener.as_ref(), new_state, paused_by_gate).await;
            }
            _ = shutdown.changed() => {
                if *shutdown.borrow() {
                    return;
                }
            }
        }
    }
}

/// Pause the listener when the daemon sleeps and resume it once the model
/// is back, returning whether the gate still holds it paused.
async fn follow_presence(
    listener: &dyn ContinuousListener,
    state: PresenceState,
    paused_by_gate: bool,
) -> bool {
    match state {
        PresenceState::Sleeping => paused_by_gate || pause_for_sleep(listener).await,
        PresenceState::Waking => paused_by_gate,
        PresenceState::Active | PresenceState::Drowsy => {
            if paused_by_gate && !listener.is_active() {
                resume_after_sleep(listener, state).await;
                return false;
            }
            paused_by_gate
        }
    }
}

/// Stop an active listener, returning whether it was stopped.
async fn pause_for_sleep(listener: &dyn ContinuousListener) -> bool {
    if !listener.is_active() {
        return false;
    }
    match listener.stop().await {
        Ok(()) => {
            info!(target: "assistd::listen", "paused: presence → sleeping");
            true
        }
        Err(e) => {
            warn!(target: "assistd::listen", "pausing on sleep failed: {e:#}");
            false
        }
    }
}

async fn resume_after_sleep(listener: &dyn ContinuousListener, state: PresenceState) {
    match listener.start().await {
        Ok(()) => info!(target: "assistd::listen", "resumed: presence → {state:?}"),
        Err(e) => warn!(target: "assistd::listen", "auto-resume failed: {e:#}"),
    }
}

/// Start listening unless the model is asleep or still loading, returning
/// whether the start was deferred until it is back.
async fn start_unless_asleep(listener: &dyn ContinuousListener, initial: PresenceState) -> bool {
    match initial {
        PresenceState::Sleeping | PresenceState::Waking => {
            info!(
                target: "assistd::listen",
                "start_on_launch deferred: presence is {initial:?}"
            );
            true
        }
        PresenceState::Active | PresenceState::Drowsy => {
            match listener.start().await {
                Ok(()) => info!(target: "assistd::listen", "continuous listening auto-started"),
                Err(e) => warn!(target: "assistd::listen", "start_on_launch failed: {e:#}"),
            }
            false
        }
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

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicBool;

    use assistd_voice::ListenError;
    use async_trait::async_trait;
    use tokio::sync::broadcast;

    use super::*;

    #[derive(Debug, Default)]
    struct FakeListener {
        active: AtomicBool,
    }

    #[async_trait]
    impl ContinuousListener for FakeListener {
        async fn start(&self) -> Result<(), ListenError> {
            self.active.store(true, Ordering::SeqCst);
            Ok(())
        }

        async fn stop(&self) -> Result<(), ListenError> {
            self.active.store(false, Ordering::SeqCst);
            Ok(())
        }

        fn is_active(&self) -> bool {
            self.active.load(Ordering::SeqCst)
        }

        fn subscribe_utterances(&self) -> broadcast::Receiver<String> {
            broadcast::channel(1).1
        }

        fn subscribe_state(&self) -> watch::Receiver<bool> {
            watch::channel(false).1
        }
    }

    #[tokio::test]
    async fn launch_start_waits_for_the_model_to_load() {
        let listener = FakeListener::default();
        assert!(start_unless_asleep(&listener, PresenceState::Waking).await);
        assert!(!listener.is_active());

        assert!(follow_presence(&listener, PresenceState::Waking, true).await);
        assert!(!listener.is_active(), "still loading");

        assert!(!follow_presence(&listener, PresenceState::Active, true).await);
        assert!(listener.is_active(), "started once the model is up");
    }

    #[tokio::test]
    async fn sleep_pauses_until_the_next_wake_finishes() {
        let listener = FakeListener::default();
        listener.start().await.unwrap();

        assert!(follow_presence(&listener, PresenceState::Sleeping, false).await);
        assert!(!listener.is_active());
        assert!(follow_presence(&listener, PresenceState::Waking, true).await);
        assert!(!listener.is_active());
        assert!(!follow_presence(&listener, PresenceState::Active, true).await);
        assert!(listener.is_active());
    }

    #[tokio::test]
    async fn listening_the_user_stopped_stays_stopped() {
        let listener = FakeListener::default();
        assert!(!follow_presence(&listener, PresenceState::Sleeping, false).await);
        assert!(!follow_presence(&listener, PresenceState::Active, false).await);
        assert!(!listener.is_active());
    }
}
