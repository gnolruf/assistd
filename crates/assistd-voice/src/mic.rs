//! Push-to-talk microphone capture: cpal callback, SPSC ring buffer,
//! and a blocking consumer that resamples to 16 kHz for the transcriber.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use async_trait::async_trait;
use tokio::sync::{Mutex, watch};
use tokio::task::JoinHandle;
use tracing::{info, warn};

use crate::transcribe::Transcriber;
use crate::{VoiceCaptureState, VoiceInput, VoiceInputError};

pub mod capture;
pub mod consumer;
pub(crate) mod resample;

pub use capture::AudioCaptureError;

/// Push-to-talk voice input backed by cpal and a [`Transcriber`]. The
/// audio device is opened on [`start_recording`](VoiceInput::start_recording),
/// not at construction.
#[derive(Debug)]
pub struct MicVoiceInput {
    transcriber: Arc<dyn Transcriber>,
    mic_device: Option<String>,
    state_tx: watch::Sender<VoiceCaptureState>,
    /// Bumped per press so a stale transition from an aborted press
    /// cannot clobber the state of a newer one.
    active_session_id: Arc<AtomicU64>,
    ptt: Arc<Mutex<PttState>>,
}

#[derive(Debug)]
struct PttState {
    press: Option<Press>,
    forwarder: Option<JoinHandle<()>>,
}

#[derive(Debug)]
struct Press {
    session_id: u64,
    capture: capture::CaptureSession,
}

/// Publishes a press's state only while it is still the active press.
#[derive(Debug)]
struct PressPublisher {
    state_tx: watch::Sender<VoiceCaptureState>,
    active_session_id: Arc<AtomicU64>,
    session_id: u64,
}

impl PressPublisher {
    /// Return whether the press is still active; the check runs under the
    /// channel's write lock, so a newer press's first state always lands last.
    fn publish(&self, state: VoiceCaptureState) -> bool {
        let mut active = false;
        self.state_tx.send_if_modified(|current| {
            active = self.active_session_id.load(Ordering::SeqCst) == self.session_id;
            if active {
                *current = state;
            }
            active
        });
        active
    }
}

impl MicVoiceInput {
    /// `mic_device` selects the cpal input device by name, `None` for
    /// the system default.
    pub fn new(transcriber: Arc<dyn Transcriber>, mic_device: Option<String>) -> Self {
        let (state_tx, _) = watch::channel(VoiceCaptureState::Idle);
        Self {
            transcriber,
            mic_device,
            state_tx,
            active_session_id: Arc::new(AtomicU64::new(0)),
            ptt: Arc::new(Mutex::new(PttState {
                press: None,
                forwarder: None,
            })),
        }
    }

    /// Mirror the transcriber's `Transcribing` state until a newer
    /// press starts; `stop_and_transcribe` owns the terminal `Idle`.
    fn spawn_state_forwarder(
        mut transcriber_state: watch::Receiver<VoiceCaptureState>,
        publisher: PressPublisher,
    ) -> JoinHandle<()> {
        tokio::spawn(async move {
            while transcriber_state.changed().await.is_ok() {
                let state = *transcriber_state.borrow_and_update();
                let forwarded = state == VoiceCaptureState::Transcribing;
                if forwarded && !publisher.publish(state) {
                    return;
                }
            }
        })
    }

    fn publisher(&self, session_id: u64) -> PressPublisher {
        PressPublisher {
            state_tx: self.state_tx.clone(),
            active_session_id: Arc::clone(&self.active_session_id),
            session_id,
        }
    }

    /// Wait for the stopped capture and transcribe what it recorded.
    async fn transcribe_session(
        &self,
        session: capture::CaptureSession,
    ) -> Result<String, VoiceInputError> {
        let pcm = session
            .handle
            .await
            .map_err(VoiceInputError::ConsumerPanic)??;

        let overrun = session.overrun.load(Ordering::Relaxed);
        if overrun > 0 {
            warn!(
                target: "assistd::voice::mic",
                overrun_samples = overrun,
                "ring buffer overrun during PTT capture; audio was truncated"
            );
        }

        if pcm.is_empty() {
            return Ok(String::new());
        }

        log_captured_levels(&pcm);

        let text = self.transcriber.transcribe(&pcm).await?;
        info!(
            target: "assistd::voice::mic",
            pcm_samples = pcm.len(),
            text_chars = text.chars().count(),
            "transcription complete"
        );
        Ok(text)
    }
}

/// Ends a press: aborts its state forwarder and publishes `Idle` unless a
/// newer press has started, on drop too, so a cancelled release cannot
/// leave the state stuck.
struct PressCleanup {
    publisher: PressPublisher,
    forwarder: Option<JoinHandle<()>>,
}

impl PressCleanup {
    /// Wait for the forwarder to exit so it cannot publish after `Idle`.
    async fn finish(mut self) {
        if let Some(forwarder) = self.forwarder.take() {
            forwarder.abort();
            let _ = forwarder.await;
        }
    }
}

impl Drop for PressCleanup {
    fn drop(&mut self) {
        if let Some(forwarder) = self.forwarder.take() {
            forwarder.abort();
        }
        self.publisher.publish(VoiceCaptureState::Idle);
    }
}

#[async_trait]
impl VoiceInput for MicVoiceInput {
    async fn start_recording(&self) -> Result<(), VoiceInputError> {
        let mut ptt = self.ptt.lock().await;
        if ptt.press.is_some() {
            return Ok(());
        }

        let session_id = self.active_session_id.fetch_add(1, Ordering::SeqCst) + 1;

        let press = ptt.press.insert(Press {
            session_id,
            capture: capture::start(self.mic_device.as_deref()),
        });
        if let Err(err) = await_capture_open(&mut press.capture).await {
            ptt.press = None;
            return Err(err);
        }

        if let Some(transcriber_state) = self.transcriber.subscribe_state() {
            ptt.forwarder = Some(Self::spawn_state_forwarder(
                transcriber_state,
                self.publisher(session_id),
            ));
        }

        let _ = self.state_tx.send(VoiceCaptureState::Recording);
        drop(ptt);

        info!(target: "assistd::voice::mic", session_id, "recording started");
        Ok(())
    }

    async fn stop_and_transcribe(&self) -> Result<String, VoiceInputError> {
        let (press, forwarder) = {
            let mut ptt = self.ptt.lock().await;
            match ptt.press.take() {
                Some(press) => (press, ptt.forwarder.take()),
                None => return Ok(String::new()),
            }
        };
        let cleanup = PressCleanup {
            publisher: self.publisher(press.session_id),
            forwarder,
        };

        cleanup.publisher.publish(VoiceCaptureState::Transcribing);
        press.capture.stop_flag.store(true, Ordering::SeqCst);

        let result = self.transcribe_session(press.capture).await;
        cleanup.finish().await;
        result
    }

    fn state(&self) -> VoiceCaptureState {
        *self.state_tx.borrow()
    }

    fn subscribe(&self) -> watch::Receiver<VoiceCaptureState> {
        self.state_tx.subscribe()
    }
}

/// Wait until the capture is recording; its error if the device failed to open.
async fn await_capture_open(session: &mut capture::CaptureSession) -> Result<(), VoiceInputError> {
    if (&mut session.opened).await.is_ok() {
        return Ok(());
    }
    (&mut session.handle)
        .await
        .map_err(VoiceInputError::ConsumerPanic)??;
    Ok(())
}

fn log_captured_levels(pcm: &[i16]) {
    let duration_secs = pcm.len() as f32 / 16_000.0;
    let peak = pcm
        .iter()
        .map(|sample| sample.unsigned_abs())
        .max()
        .unwrap_or(0);
    let peak_dbfs = if peak == 0 {
        f32::NEG_INFINITY
    } else {
        20.0 * (f32::from(peak) / f32::from(i16::MAX)).log10()
    };
    info!(
        target: "assistd::voice::mic",
        pcm_samples = pcm.len(),
        duration_secs,
        peak_dbfs,
        "captured pcm; invoking transcriber"
    );
}
