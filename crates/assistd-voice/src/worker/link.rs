//! The daemon's end of the pipe: calls awaiting replies, and the worker's
//! state mirrored into channels that outlive any one worker process.

use std::sync::atomic::AtomicBool;

use assistd_ipc::{ListenPhase, VoiceCaptureState};
use assistd_utils::pipe_worker::PendingReplies;
use tokio::sync::{broadcast, mpsc, oneshot, watch};
use tokio::time::Instant;
use tracing::debug;

use super::protocol::{FromWorker, Outcome, Request, Startup, ToWorker};
use crate::error::VoiceWorkerError;
use crate::listen::{Heard, WakeTest};

/// Wake phrases and commands held for a slow subscriber.
const HEARD_BACKLOG: usize = 16;

/// A component's startup as the worker reported it.
#[derive(Debug)]
pub(super) enum StartupReport {
    Capture(Startup),
    Speech(Startup),
}

/// Shared by the proxies and the supervisor for the daemon's lifetime.
#[derive(Debug)]
pub(super) struct Link {
    replies: PendingReplies<ToWorker, oneshot::Sender<Outcome>>,
    pub(super) capture_state: watch::Sender<VoiceCaptureState>,
    pub(super) phase: watch::Sender<ListenPhase>,
    pub(super) heard: broadcast::Sender<Heard>,
    /// Where the running wake-word test's reports go.
    pub(super) wake_test: parking_lot::Mutex<Option<mpsc::Sender<WakeTest>>>,
    /// Whether listening was started and not stopped since, so capture that
    /// comes back resumes it.
    pub(super) listen_wanted: AtomicBool,
    pub(super) last_push_to_talk: parking_lot::Mutex<Option<Instant>>,
}

impl Link {
    pub(super) fn new() -> Self {
        Self {
            replies: PendingReplies::new("voice worker"),
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
        self.replies.connect(outgoing)
    }

    /// The worker is gone: fail its pending calls and reset what it mirrored.
    pub(super) fn disconnect(&self) {
        self.replies.disconnect();
        self.wake_test.lock().take();
        self.capture_state.send_replace(VoiceCaptureState::Idle);
        self.phase.send_replace(ListenPhase::Off);
    }

    /// Send `request` and wait for the worker's answer. A worker that misses
    /// the request's deadline fails it with [`VoiceWorkerError::Unresponsive`]
    /// and is marked for the supervisor to terminate.
    pub(super) async fn call(&self, request: Request) -> Result<Outcome, VoiceWorkerError> {
        let deadline = request.deadline();
        let (answer, reply) = oneshot::channel();
        let call = |id| ToWorker::Call { id, request };
        Ok(self.replies.call(call, answer, reply, deadline).await?)
    }

    /// Send a message that has no reply; dropped while no worker runs.
    pub(super) fn notify(&self, message: ToWorker) {
        self.replies.notify(message);
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
            FromWorker::Reply { id, outcome } => {
                if let Some(answer) = self.replies.take(id) {
                    let _ = answer.send(outcome);
                }
            }
        }
        None
    }
}
