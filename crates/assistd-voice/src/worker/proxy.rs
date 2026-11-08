//! [`VoiceInput`], [`VoiceOutput`] and [`ContinuousListener`] for the daemon,
//! each forwarding to the voice worker over its [`Link`].

use std::sync::Arc;
use std::sync::atomic::Ordering;

use assistd_ipc::{ListenPhase, VoiceCaptureState};
use async_trait::async_trait;
use tokio::sync::{broadcast, mpsc, watch};
use tokio::time::Instant;

use super::link::Link;
use super::protocol::{CaptureRequest, Outcome, Request, SpeechRequest, ToWorker};
use crate::error::{VoiceInputError, VoiceOutputError, VoiceWorkerError};
use crate::listen::{ContinuousListener, Heard, ListenError, WakeTest};
use crate::{VoiceInput, VoiceOutput};

/// Push-to-talk on the worker's mic.
#[derive(Debug)]
pub(super) struct WorkerVoiceInput(pub(super) Arc<Link>);

/// Speech through the worker's speaker.
#[derive(Debug)]
pub(super) struct WorkerVoiceOutput(pub(super) Arc<Link>);

/// The worker's wake-word listener.
#[derive(Debug)]
pub(super) struct WorkerListener(pub(super) Arc<Link>);

impl WorkerListener {
    async fn call(&self, request: CaptureRequest) -> Result<(), ListenError> {
        match self.0.call(Request::Capture(request)).await? {
            Outcome::Failed { failure } => Err(failure.into_listen_error()),
            Outcome::Done | Outcome::Transcript { .. } => Ok(()),
        }
    }

    /// Stop listening; with no worker running, or one being killed, there is
    /// nothing to stop.
    async fn stop_with(&self, request: CaptureRequest) -> Result<(), ListenError> {
        self.0.listen_wanted.store(false, Ordering::SeqCst);
        match self.call(request).await {
            Err(ListenError::VoiceWorker(
                VoiceWorkerError::NotRunning
                | VoiceWorkerError::Exited
                | VoiceWorkerError::Unresponsive { .. },
            )) => Ok(()),
            stopped => stopped,
        }
    }
}

#[async_trait]
impl VoiceInput for WorkerVoiceInput {
    async fn start_recording(&self) -> Result<(), VoiceInputError> {
        let outcome = self
            .0
            .call(Request::Capture(CaptureRequest::PttStart))
            .await?;
        unit_outcome(outcome)?;
        Ok(())
    }

    async fn stop_and_transcribe(&self) -> Result<String, VoiceInputError> {
        match self
            .0
            .call(Request::Capture(CaptureRequest::PttStop))
            .await?
        {
            Outcome::Transcript { text } => Ok(text),
            Outcome::Done => Ok(String::new()),
            Outcome::Failed { failure } => Err(failure.into_worker_error().into()),
        }
    }

    fn state(&self) -> VoiceCaptureState {
        *self.0.capture_state.borrow()
    }

    fn subscribe(&self) -> watch::Receiver<VoiceCaptureState> {
        self.0.capture_state.subscribe()
    }
}

#[async_trait]
impl VoiceOutput for WorkerVoiceOutput {
    async fn speak(&self, text: String) -> Result<(), VoiceOutputError> {
        let outcome = self
            .0
            .call(Request::Speech(SpeechRequest::Speak { text }))
            .await?;
        unit_outcome(outcome)?;
        Ok(())
    }

    async fn wait_idle(&self) -> Result<(), VoiceOutputError> {
        let outcome = self
            .0
            .call(Request::Speech(SpeechRequest::WaitIdle))
            .await?;
        unit_outcome(outcome)?;
        Ok(())
    }

    async fn cancel(&self) {
        let _ = self.0.call(Request::Speech(SpeechRequest::Cancel)).await;
    }
}

#[async_trait]
impl ContinuousListener for WorkerListener {
    async fn start(&self) -> Result<(), ListenError> {
        self.call(CaptureRequest::ListenStart).await?;
        self.0.listen_wanted.store(true, Ordering::SeqCst);
        Ok(())
    }

    async fn stop(&self) -> Result<(), ListenError> {
        self.stop_with(CaptureRequest::ListenStop).await
    }

    async fn stop_for_push_to_talk(&self) -> Result<(), ListenError> {
        *self.0.last_push_to_talk.lock() = Some(Instant::now());
        self.stop_with(CaptureRequest::ListenStopForPushToTalk)
            .await
    }

    fn last_push_to_talk(&self) -> Option<Instant> {
        *self.0.last_push_to_talk.lock()
    }

    fn is_active(&self) -> bool {
        *self.0.phase.borrow() != ListenPhase::Off
    }

    fn subscribe_heard(&self) -> broadcast::Receiver<Heard> {
        self.0.heard.subscribe()
    }

    fn subscribe_phase(&self) -> watch::Receiver<ListenPhase> {
        self.0.phase.subscribe()
    }

    fn open_follow_up(&self) {
        self.0.notify(ToWorker::OpenFollowUp);
    }

    fn set_replying(&self, replying: bool) {
        self.0.notify(ToWorker::SetReplying { replying });
    }

    async fn test_wake_word(&self, sink: mpsc::Sender<WakeTest>) -> Result<(), ListenError> {
        *self.0.wake_test.lock() = Some(sink.clone());
        let test = self.call(CaptureRequest::WakeTestStart);
        tokio::pin!(test);
        let tested = tokio::select! {
            tested = &mut test => tested,
            () = sink.closed() => {
                let _ = self.call(CaptureRequest::WakeTestStop).await;
                test.await
            }
        };
        self.0.wake_test.lock().take();
        tested
    }
}

fn unit_outcome(outcome: Outcome) -> Result<(), VoiceWorkerError> {
    match outcome {
        Outcome::Failed { failure } => Err(failure.into_worker_error()),
        Outcome::Done | Outcome::Transcript { .. } => Ok(()),
    }
}
