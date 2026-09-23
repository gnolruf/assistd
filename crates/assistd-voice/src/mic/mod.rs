//! Push-to-talk microphone capture: cpal callback, SPSC ring buffer,
//! and a blocking consumer that resamples to 16 kHz for the transcriber.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use assistd_config::VoiceConfig;
use async_trait::async_trait;
use tokio::sync::{Mutex, watch};
use tokio::task::JoinHandle;
use tracing::{info, warn};

use crate::transcribe::Transcriber;
use crate::whisper::WhisperTranscriberBuilder;
use crate::{VoiceCaptureState, VoiceInput, VoiceInputError};

pub mod capture;
pub mod consumer;
pub(crate) mod resample;

pub use capture::{AudioCaptureError, DeviceValidationError};

/// Push-to-talk voice input backed by cpal and a [`Transcriber`]. The
/// audio device is opened on [`start_recording`](VoiceInput::start_recording),
/// not at construction.
pub struct MicVoiceInput {
    transcriber: Arc<dyn Transcriber>,
    mic_device: Option<String>,
    max_recording_secs: u32,
    state_tx: watch::Sender<VoiceCaptureState>,
    /// Bumped per press so a stale transition from an aborted press
    /// cannot clobber the state of a newer one.
    active_session_id: Arc<AtomicU64>,
    inner: Arc<Mutex<PttState>>,
}

struct PttState {
    session: Option<capture::CaptureSession>,
    forwarder: Option<JoinHandle<()>>,
}

impl MicVoiceInput {
    /// Build from config with a bare [`crate::WhisperTranscriber`] (no
    /// queueing or CPU fallback). Downloads models on first use.
    pub async fn from_config(cfg: &VoiceConfig) -> Result<Self, VoiceInputError> {
        let transcriber = WhisperTranscriberBuilder::from_config(&cfg.transcription)
            .build()
            .await?;
        Ok(Self::new(
            Arc::new(transcriber),
            cfg.mic_device.clone(),
            cfg.max_recording_secs.get(),
        ))
    }

    /// `mic_device` selects the cpal input device by name, `None` for
    /// the system default. `max_recording_secs` caps each session.
    pub fn new(
        transcriber: Arc<dyn Transcriber>,
        mic_device: Option<String>,
        max_recording_secs: u32,
    ) -> Self {
        let (state_tx, _) = watch::channel(VoiceCaptureState::Idle);
        Self {
            transcriber,
            mic_device,
            max_recording_secs,
            state_tx,
            active_session_id: Arc::new(AtomicU64::new(0)),
            inner: Arc::new(Mutex::new(PttState {
                session: None,
                forwarder: None,
            })),
        }
    }

    async fn cleanup_forwarder_and_idle(&self, forwarder: Option<JoinHandle<()>>) {
        if let Some(handle) = forwarder {
            handle.abort();
            let _ = handle.await;
        }
        let _ = self.state_tx.send(VoiceCaptureState::Idle);
    }
}

#[async_trait]
impl VoiceInput for MicVoiceInput {
    async fn start_recording(&self) -> Result<(), VoiceInputError> {
        let mut inner = self.inner.lock().await;
        if inner.session.is_some() {
            return Ok(());
        }

        let session_id = self.active_session_id.fetch_add(1, Ordering::SeqCst) + 1;

        let session = capture::start(self.mic_device.as_deref(), self.max_recording_secs);
        inner.session = Some(session);

        if let Some(mut rx) = self.transcriber.subscribe_state() {
            let state_tx = self.state_tx.clone();
            let session_id_at_spawn = session_id;
            let counter = Arc::clone(&self.active_session_id);
            let handle = tokio::spawn(async move {
                loop {
                    if rx.changed().await.is_err() {
                        return;
                    }
                    let state = *rx.borrow_and_update();
                    if counter.load(Ordering::SeqCst) != session_id_at_spawn {
                        return;
                    }
                    // `stop_and_transcribe` owns the terminal Idle.
                    if matches!(
                        state,
                        VoiceCaptureState::Queued | VoiceCaptureState::Transcribing
                    ) {
                        let _ = state_tx.send(state);
                    }
                }
            });
            inner.forwarder = Some(handle);
        }

        drop(inner);

        let _ = self.state_tx.send(VoiceCaptureState::Recording);
        info!(target: "assistd::voice::mic", session_id, "recording started");
        Ok(())
    }

    async fn stop_and_transcribe(&self) -> Result<String, VoiceInputError> {
        let (session, forwarder) = {
            let mut inner = self.inner.lock().await;
            match inner.session.take() {
                Some(session) => (session, inner.forwarder.take()),
                None => return Ok(String::new()),
            }
        };

        let _ = self.state_tx.send(VoiceCaptureState::Transcribing);
        session.stop_flag.store(true, Ordering::SeqCst);

        let pcm = match session.handle.await {
            Ok(Ok(pcm)) => pcm,
            Ok(Err(e)) => {
                self.cleanup_forwarder_and_idle(forwarder).await;
                return Err(e.into());
            }
            Err(join_err) => {
                self.cleanup_forwarder_and_idle(forwarder).await;
                return Err(VoiceInputError::ConsumerPanic(join_err));
            }
        };

        let overrun = session.overrun.load(Ordering::Relaxed);
        if overrun > 0 {
            warn!(
                target: "assistd::voice::mic",
                overrun_samples = overrun,
                "ring buffer overrun during PTT capture; audio was truncated"
            );
        }

        if pcm.is_empty() {
            self.cleanup_forwarder_and_idle(forwarder).await;
            return Ok(String::new());
        }

        let duration_secs = pcm.len() as f32 / 16_000.0;
        let peak = pcm.iter().map(|s| s.unsigned_abs()).max().unwrap_or(0);
        let peak_dbfs = if peak == 0 {
            f32::NEG_INFINITY
        } else {
            20.0 * (peak as f32 / i16::MAX as f32).log10()
        };
        info!(
            target: "assistd::voice::mic",
            pcm_samples = pcm.len(),
            duration_secs,
            peak_dbfs,
            "captured pcm; invoking transcriber"
        );

        let result = self.transcriber.transcribe(&pcm).await;
        self.cleanup_forwarder_and_idle(forwarder).await;

        let text = result?;

        info!(
            target: "assistd::voice::mic",
            pcm_samples = pcm.len(),
            text_chars = text.chars().count(),
            text = %text,
            "transcription complete"
        );
        Ok(text)
    }

    fn state(&self) -> VoiceCaptureState {
        *self.state_tx.borrow()
    }

    fn subscribe(&self) -> watch::Receiver<VoiceCaptureState> {
        self.state_tx.subscribe()
    }
}
