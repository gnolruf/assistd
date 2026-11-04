//! The worker side: build the voice stack and answer the daemon's calls over
//! stdin and stdout until stdin closes.

use std::io;
use std::sync::{Arc, OnceLock};

use assistd_config::VoiceConfig;
use thiserror::Error;
use tokio::fs::File;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, Lines, Stdin};
use tokio::sync::{broadcast, mpsc, oneshot};
use tokio::task::JoinSet;
use tracing::warn;

use super::protocol::{
    CaptureRequest, Failure, FromWorker, Outcome, ProtocolError, Request, SpeechRequest, Startup,
    ToWorker, decode, encode,
};
use super::stack::{self, ReportedGpu};
use crate::listen::{ContinuousListener, ListenError};
use crate::{SpeakingGuard, VoiceCapture, VoiceOutput, VoiceOutputController};

/// Lowest descriptor the protocol's copy of stdout may take, past stdio.
const FIRST_PRIVATE_FD: rustix::fd::RawFd = 3;

/// Reports queued while the daemon's pipe is busy.
const WAKE_TEST_BACKLOG: usize = 12;

/// Why the worker stopped before stdin closed.
#[derive(Debug, Error)]
pub enum ServeError {
    #[error("voice worker stdio: {0}")]
    Io(#[from] io::Error),
    #[error("stdin closed before the voice settings arrived")]
    Unconfigured,
    #[error("the first line was not the voice settings: {0}")]
    Configure(#[from] ProtocolError),
}

/// The built stack and what the worker tracks across calls.
#[derive(Debug)]
struct Worker {
    out: mpsc::UnboundedSender<FromWorker>,
    gpu: Arc<ReportedGpu>,
    /// Only its recent sentences are used, for barge-in's echo check.
    speech: Arc<VoiceOutputController>,
    output: Result<Arc<dyn VoiceOutput>, String>,
    capture: OnceLock<Result<VoiceCapture, String>>,
    /// Held from a reply's first sentence until its playback drains.
    speaking: parking_lot::Mutex<Option<SpeakingGuard>>,
    wake_test_stop: parking_lot::Mutex<Option<oneshot::Sender<()>>>,
}

impl Worker {
    async fn start_speech(config: &VoiceConfig, out: mpsc::UnboundedSender<FromWorker>) -> Self {
        let output = stack::start_output(&config.synthesis).await;
        let worker = Self {
            out,
            gpu: Arc::new(ReportedGpu::default()),
            speech: VoiceOutputController::new(true),
            output,
            capture: OnceLock::new(),
            speaking: parking_lot::Mutex::new(None),
            wake_test_stop: parking_lot::Mutex::new(None),
        };
        worker.send(FromWorker::SpeechReadiness {
            startup: startup_of(&worker.output),
        });
        worker
    }

    /// Build capture, report it, then forward its state to the daemon for
    /// as long as the worker runs.
    async fn start_capture(self: Arc<Self>, config: VoiceConfig) {
        let capture =
            stack::start_capture(&config, Arc::clone(&self.gpu), self.speech.clone()).await;
        let startup = startup_of(&capture);
        let capture = self.capture.get_or_init(|| capture);
        self.send(FromWorker::CaptureReadiness { startup });
        if let Ok(capture) = capture {
            tokio::join!(
                self.forward_capture_state(capture),
                self.forward_phase(capture.listener.as_ref()),
                self.forward_heard(capture.listener.as_ref()),
            );
        }
    }

    fn send(&self, message: FromWorker) {
        let _ = self.out.send(message);
    }

    fn receive(self: &Arc<Self>, message: ToWorker, calls: &mut JoinSet<()>) {
        match message {
            ToWorker::Configure { .. } => warn!("voice worker: already configured; ignoring"),
            ToWorker::Call { id, request } => {
                let worker = Arc::clone(self);
                calls.spawn(async move {
                    let outcome = worker.answer(request).await;
                    worker.send(FromWorker::Reply { id, outcome });
                });
            }
            ToWorker::SetReplying { replying } => {
                if let Some(Ok(capture)) = self.capture.get() {
                    capture.listener.set_replying(replying);
                }
            }
            ToWorker::OpenFollowUp => {
                if let Some(Ok(capture)) = self.capture.get() {
                    capture.listener.open_follow_up();
                }
            }
            ToWorker::Gpu {
                presence_active,
                foreign_busy,
            } => self.gpu.update(presence_active, foreign_busy),
        }
    }

    async fn answer(&self, request: Request) -> Outcome {
        let answered = match request {
            Request::Speech(request) => self.answer_speech(request).await,
            Request::Capture(request) => match self.ready_capture() {
                Ok(capture) => self.answer_capture(capture, request).await,
                Err(failure) => Err(failure),
            },
        };
        answered.unwrap_or_else(|failure| Outcome::Failed { failure })
    }

    async fn answer_speech(&self, request: SpeechRequest) -> Result<Outcome, Failure> {
        let output = self
            .output
            .as_ref()
            .map_err(|reason| Failure::Other(format!("speech is unavailable: {reason}")))?;
        match request {
            SpeechRequest::Speak { text } => {
                self.speaking
                    .lock()
                    .get_or_insert_with(|| self.speech.begin_speaking());
                self.speech.remember_spoken(&text);
                output.speak(text).await.map_err(Failure::other)?;
            }
            SpeechRequest::Cancel => output.cancel().await,
            SpeechRequest::WaitIdle => {
                let idle = output.wait_idle().await;
                self.speaking.lock().take();
                idle.map_err(Failure::other)?;
            }
        }
        Ok(Outcome::Done)
    }

    /// Answer `request`, reporting the state it changed before the reply so
    /// the daemon sees that state by the time its call returns.
    async fn answer_capture(
        &self,
        capture: &VoiceCapture,
        request: CaptureRequest,
    ) -> Result<Outcome, Failure> {
        let listener = capture.listener.as_ref();
        match request {
            CaptureRequest::PttStart => {
                let started = capture.input.start_recording().await;
                self.send_capture_state(capture);
                started.map_err(Failure::other)?;
                Ok(Outcome::Done)
            }
            CaptureRequest::PttStop => {
                let text = capture.input.stop_and_transcribe().await;
                self.send_capture_state(capture);
                Ok(Outcome::Transcript {
                    text: text.map_err(Failure::other)?,
                })
            }
            CaptureRequest::ListenStart => self.change_listening(listener, listener.start()).await,
            CaptureRequest::ListenStop => self.change_listening(listener, listener.stop()).await,
            CaptureRequest::ListenStopForPushToTalk => {
                let stop = listener.stop_for_push_to_talk();
                self.change_listening(listener, stop).await
            }
            CaptureRequest::WakeTestStart => self.run_wake_test(listener).await,
            CaptureRequest::WakeTestStop => {
                if let Some(stop) = self.wake_test_stop.lock().take() {
                    let _ = stop.send(());
                }
                Ok(Outcome::Done)
            }
        }
    }

    fn ready_capture(&self) -> Result<&VoiceCapture, Failure> {
        match self.capture.get() {
            Some(Ok(capture)) => Ok(capture),
            Some(Err(reason)) => Err(Failure::Other(format!(
                "voice capture is unavailable: {reason}"
            ))),
            None => Err(Failure::Other("voice capture is still starting".into())),
        }
    }

    fn send_capture_state(&self, capture: &VoiceCapture) {
        self.send(FromWorker::CaptureState {
            state: capture.input.state(),
        });
    }

    async fn change_listening(
        &self,
        listener: &dyn ContinuousListener,
        change: impl Future<Output = Result<(), ListenError>>,
    ) -> Result<Outcome, Failure> {
        let changed = change.await;
        self.send(FromWorker::ListenPhase {
            phase: *listener.subscribe_phase().borrow(),
        });
        changed?;
        Ok(Outcome::Done)
    }

    async fn run_wake_test(&self, listener: &dyn ContinuousListener) -> Result<Outcome, Failure> {
        let stop = {
            let mut slot = self.wake_test_stop.lock();
            if slot.is_some() {
                return Err(Failure::ListenTesting);
            }
            let (stop_tx, stop_rx) = oneshot::channel();
            *slot = Some(stop_tx);
            stop_rx
        };
        let (report_tx, report_rx) = mpsc::channel(WAKE_TEST_BACKLOG);
        let (tested, ()) = tokio::join!(
            listener.test_wake_word(report_tx),
            self.forward_wake_test(report_rx, stop),
        );
        self.wake_test_stop.lock().take();
        tested?;
        Ok(Outcome::Done)
    }

    async fn forward_wake_test(
        &self,
        mut reports: mpsc::Receiver<crate::listen::WakeTest>,
        mut stop: oneshot::Receiver<()>,
    ) {
        loop {
            let report = tokio::select! {
                _ = &mut stop => return,
                report = reports.recv() => report,
            };
            let Some(report) = report else {
                return;
            };
            self.send(FromWorker::WakeTest { report });
        }
    }

    async fn forward_capture_state(&self, capture: &VoiceCapture) {
        let mut states = capture.input.subscribe();
        loop {
            let state = *states.borrow_and_update();
            self.send(FromWorker::CaptureState { state });
            if states.changed().await.is_err() {
                return;
            }
        }
    }

    async fn forward_phase(&self, listener: &dyn ContinuousListener) {
        let mut phases = listener.subscribe_phase();
        loop {
            let phase = *phases.borrow_and_update();
            self.send(FromWorker::ListenPhase { phase });
            if phases.changed().await.is_err() {
                return;
            }
        }
    }

    async fn forward_heard(&self, listener: &dyn ContinuousListener) {
        let mut heard = listener.subscribe_heard();
        loop {
            match heard.recv().await {
                Ok(heard) => self.send(FromWorker::Heard { heard }),
                Err(broadcast::error::RecvError::Lagged(dropped)) => {
                    warn!(dropped, "voice worker: dropped what the listener heard");
                }
                Err(broadcast::error::RecvError::Closed) => return,
            }
        }
    }
}

/// Run the voice worker over this process's stdin and stdout, then point fd 1
/// at stderr for the rest of the process. The first line must be the voice
/// settings; returns once stdin closes.
pub async fn serve() -> Result<(), ServeError> {
    let protocol_out = take_stdout_for_protocol()?;
    let mut lines = BufReader::new(tokio::io::stdin()).lines();
    let config = read_config(&mut lines).await?;
    let (out, out_rx) = mpsc::unbounded_channel();
    let mut calls = JoinSet::new();
    let session = async {
        let worker = Arc::new(Worker::start_speech(&config, out).await);
        calls.spawn(Arc::clone(&worker).start_capture(config.clone()));
        read_calls(&worker, &mut lines, &mut calls).await
    };
    let served = tokio::select! {
        served = session => served,
        () = write_messages(protocol_out, out_rx) => Ok(()),
    };
    calls.shutdown().await;
    served
}

async fn read_config(lines: &mut Lines<BufReader<Stdin>>) -> Result<VoiceConfig, ServeError> {
    let line = lines.next_line().await?.ok_or(ServeError::Unconfigured)?;
    match decode(&line)? {
        ToWorker::Configure { config } => Ok(*config),
        _ => Err(ServeError::Unconfigured),
    }
}

async fn read_calls(
    worker: &Arc<Worker>,
    lines: &mut Lines<BufReader<Stdin>>,
    calls: &mut JoinSet<()>,
) -> Result<(), ServeError> {
    while let Some(line) = lines.next_line().await? {
        while calls.try_join_next().is_some() {}
        match decode(&line) {
            Ok(message) => worker.receive(message, calls),
            Err(e) => warn!("voice worker: {e}; line ignored"),
        }
    }
    Ok(())
}

/// Move the protocol off fd 1 and point fd 1 at stderr, so a native
/// library that prints cannot corrupt the daemon's pipe.
fn take_stdout_for_protocol() -> io::Result<File> {
    let protocol = rustix::io::fcntl_dupfd_cloexec(io::stdout(), FIRST_PRIVATE_FD)?;
    rustix::stdio::dup2_stdout(io::stderr())?;
    Ok(File::from_std(std::fs::File::from(protocol)))
}

/// Write each message to the daemon as it comes; returns when the pipe fails.
async fn write_messages(mut stdout: File, mut messages: mpsc::UnboundedReceiver<FromWorker>) {
    while let Some(message) = messages.recv().await {
        let line = encode(&message);
        let written = async {
            stdout.write_all(line.as_bytes()).await?;
            stdout.flush().await
        };
        if let Err(e) = written.await {
            warn!("voice worker: stdout closed: {e}");
            return;
        }
    }
}

fn startup_of<T>(started: &Result<T, String>) -> Startup {
    match started {
        Ok(_) => Startup::Ready,
        Err(reason) => Startup::Unavailable {
            reason: reason.clone(),
        },
    }
}
