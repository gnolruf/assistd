//! The daemon's end of the pipe: calls awaiting replies, and the worker's
//! state mirrored into channels that outlive any one worker process.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use assistd_ipc::{ListenPhase, VoiceCaptureState};
use tokio::sync::{broadcast, mpsc, oneshot, watch};
use tokio::time::Instant;
use tracing::{debug, warn};

use super::protocol::{FromWorker, Outcome, Request, Startup, ToWorker};
use crate::error::VoiceWorkerError;
use crate::listen::{Heard, WakeTest};

/// Wake phrases and commands held for a slow subscriber.
const HEARD_BACKLOG: usize = 16;

/// The running worker's input queue and its calls awaiting a reply.
/// Dropping it fails every one of those calls with [`VoiceWorkerError::Exited`].
#[derive(Debug)]
struct Connection {
    outgoing: mpsc::UnboundedSender<ToWorker>,
    pending: HashMap<u64, oneshot::Sender<Outcome>>,
    /// Fired once, when a call to this worker misses its deadline.
    unresponsive: Option<oneshot::Sender<()>>,
}

/// A component's startup as the worker reported it.
#[derive(Debug)]
pub(super) enum StartupReport {
    Capture(Startup),
    Speech(Startup),
}

/// Shared by the proxies and the supervisor for the daemon's lifetime.
#[derive(Debug)]
pub(super) struct Link {
    connection: parking_lot::Mutex<Option<Connection>>,
    next_id: AtomicU64,
    pub(super) capture_state: watch::Sender<VoiceCaptureState>,
    pub(super) phase: watch::Sender<ListenPhase>,
    pub(super) heard: broadcast::Sender<Heard>,
    /// Where the running wake-word test's reports go.
    pub(super) wake_test: parking_lot::Mutex<Option<mpsc::Sender<WakeTest>>>,
    /// Whether listening was started and not stopped since, so a restarted
    /// worker resumes it.
    pub(super) listen_wanted: AtomicBool,
    pub(super) last_push_to_talk: parking_lot::Mutex<Option<Instant>>,
}

impl Link {
    pub(super) fn new() -> Self {
        Self {
            connection: parking_lot::Mutex::new(None),
            next_id: AtomicU64::new(0),
            capture_state: watch::channel(VoiceCaptureState::Idle).0,
            phase: watch::channel(ListenPhase::Off).0,
            heard: broadcast::channel(HEARD_BACKLOG).0,
            wake_test: parking_lot::Mutex::new(None),
            listen_wanted: AtomicBool::new(false),
            last_push_to_talk: parking_lot::Mutex::new(None),
        }
    }

    /// Route calls to a newly spawned worker through `outgoing`; the returned
    /// receiver resolves once that worker misses a call's deadline.
    pub(super) fn connect(
        &self,
        outgoing: mpsc::UnboundedSender<ToWorker>,
    ) -> oneshot::Receiver<()> {
        let (unresponsive_tx, unresponsive_rx) = oneshot::channel();
        *self.connection.lock() = Some(Connection {
            outgoing,
            pending: HashMap::new(),
            unresponsive: Some(unresponsive_tx),
        });
        unresponsive_rx
    }

    /// The worker is gone: fail its pending calls and reset what it mirrored.
    pub(super) fn disconnect(&self) {
        self.connection.lock().take();
        self.wake_test.lock().take();
        self.capture_state.send_replace(VoiceCaptureState::Idle);
        self.phase.send_replace(ListenPhase::Off);
    }

    /// Send `request` and wait for the worker's answer. A worker that misses
    /// the request's deadline fails it with [`VoiceWorkerError::Unresponsive`]
    /// and is marked for the supervisor to terminate.
    pub(super) async fn call(&self, request: Request) -> Result<Outcome, VoiceWorkerError> {
        let deadline = request.deadline();
        let (id, mut reply) = self.send_call(request)?;
        let Some(deadline) = deadline else {
            return reply.await.map_err(|_| VoiceWorkerError::Exited);
        };
        if let Ok(answered) = tokio::time::timeout(deadline, &mut reply).await {
            return answered.map_err(|_| VoiceWorkerError::Exited);
        }
        if self.abandon(id) {
            return Err(VoiceWorkerError::Unresponsive { deadline });
        }
        reply.await.map_err(|_| VoiceWorkerError::Exited)
    }

    fn send_call(
        &self,
        request: Request,
    ) -> Result<(u64, oneshot::Receiver<Outcome>), VoiceWorkerError> {
        let mut connection = self.connection.lock();
        let connection = connection.as_mut().ok_or(VoiceWorkerError::NotRunning)?;
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (reply_tx, reply_rx) = oneshot::channel();
        connection.pending.insert(id, reply_tx);
        connection
            .outgoing
            .send(ToWorker::Call { id, request })
            .map_err(|_| VoiceWorkerError::Exited)?;
        Ok((id, reply_rx))
    }

    /// Give up on call `id` and flag its worker unresponsive; false when the
    /// call was already answered or its worker already gone.
    fn abandon(&self, id: u64) -> bool {
        let mut connection = self.connection.lock();
        let Some(connection) = connection.as_mut() else {
            return false;
        };
        if connection.pending.remove(&id).is_none() {
            return false;
        }
        warn!(target: "assistd::voice::worker", "voice worker missed a call's deadline");
        if let Some(unresponsive) = connection.unresponsive.take() {
            let _ = unresponsive.send(());
        }
        true
    }

    /// Send a message that has no reply; dropped while no worker runs.
    pub(super) fn notify(&self, message: ToWorker) {
        if let Some(connection) = &*self.connection.lock() {
            let _ = connection.outgoing.send(message);
        }
    }

    /// Mirror one message from the worker; startup reports are handed back
    /// for the supervisor to act on.
    pub(super) fn receive(&self, message: FromWorker) -> Option<StartupReport> {
        match message {
            FromWorker::CaptureReadiness { startup } => {
                return Some(StartupReport::Capture(startup));
            }
            FromWorker::SpeechReadiness { startup } => return Some(StartupReport::Speech(startup)),
            FromWorker::CaptureState { state } => {
                self.capture_state.send_replace(state);
            }
            FromWorker::ListenPhase { phase } => {
                self.phase
                    .send_if_modified(|current| std::mem::replace(current, phase) != phase);
            }
            FromWorker::Heard { heard } => {
                if self.heard.send(heard).is_err() {
                    debug!(target: "assistd::voice::worker", "no listen subscribers; dropped");
                }
            }
            FromWorker::WakeTest { report } => {
                if let Some(sink) = &*self.wake_test.lock() {
                    let _ = sink.try_send(report);
                }
            }
            FromWorker::Reply { id, outcome } => self.answer(id, outcome),
        }
        None
    }

    fn answer(&self, id: u64, outcome: Outcome) {
        let reply = self
            .connection
            .lock()
            .as_mut()
            .and_then(|connection| connection.pending.remove(&id));
        if let Some(reply) = reply {
            let _ = reply.send(outcome);
        }
    }
}
