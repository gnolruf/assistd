//! Push-to-talk microphone capture: cpal callback, SPSC ring buffer,
//! and a blocking consumer that resamples to 16 kHz for the transcriber.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Result, anyhow};
use assistd_config::VoiceConfig;
use async_trait::async_trait;
use thiserror::Error;
use tokio::sync::{Mutex, watch};
use tokio::task::JoinHandle;
use tracing::{info, warn};

use crate::VoiceCaptureState;
use crate::VoiceInput;
use crate::transcribe::{Transcriber, TranscriptionError};
use crate::whisper::WhisperTranscriberBuilder;

pub mod capture;
pub mod consumer;
pub(crate) mod resample;

pub use capture::AudioCaptureError;

/// Errors surfaced by [`MicVoiceInput`].
#[derive(Debug, Error)]
pub enum VoiceInputError {
    #[error("audio capture error: {0}")]
    Capture(#[from] AudioCaptureError),
    #[error("transcription error: {0}")]
    Transcription(#[from] TranscriptionError),
    #[error("capture task panicked: {0}")]
    ConsumerPanic(String),
}

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
    inner: Arc<Mutex<InnerState>>,
}

struct InnerState {
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
            inner: Arc::new(Mutex::new(InnerState {
                session: None,
                forwarder: None,
            })),
        }
    }

    /// Feed pre-recorded PCM straight to the transcriber, publishing the
    /// same `Recording → Transcribing → Idle` sequence as a real press.
    #[cfg(any(test, feature = "test-support"))]
    pub async fn transcribe_pcm_for_test(
        &self,
        pcm_i16_16k_mono: &[i16],
    ) -> Result<String, VoiceInputError> {
        // Yield between transitions so watch subscribers observe each
        // one instead of only the latest.
        let _ = self.state_tx.send(VoiceCaptureState::Recording);
        tokio::task::yield_now().await;
        let _ = self.state_tx.send(VoiceCaptureState::Transcribing);
        tokio::task::yield_now().await;
        let result = self.transcriber.transcribe(pcm_i16_16k_mono).await;
        tokio::task::yield_now().await;
        let _ = self.state_tx.send(VoiceCaptureState::Idle);
        Ok(result?)
    }

    async fn cleanup_forwarder_and_idle(&self, forwarder: Option<JoinHandle<()>>) {
        if let Some(h) = forwarder {
            h.abort();
            let _ = h.await;
        }
        let _ = self.state_tx.send(VoiceCaptureState::Idle);
    }
}

#[async_trait]
impl VoiceInput for MicVoiceInput {
    async fn start_recording(&self) -> Result<()> {
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
                    let s = *rx.borrow_and_update();
                    if counter.load(Ordering::SeqCst) != session_id_at_spawn {
                        return;
                    }
                    // `stop_and_transcribe` owns the terminal Idle.
                    if matches!(
                        s,
                        VoiceCaptureState::Queued | VoiceCaptureState::Transcribing
                    ) {
                        let _ = state_tx.send(s);
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

    async fn stop_and_transcribe(&self) -> Result<String> {
        let (session, forwarder) = {
            let mut inner = self.inner.lock().await;
            match inner.session.take() {
                Some(s) => (s, inner.forwarder.take()),
                None => return Ok(String::new()),
            }
        };

        let _ = self.state_tx.send(VoiceCaptureState::Transcribing);
        session.stop_flag.store(true, Ordering::SeqCst);

        let pcm = match session.handle.await {
            Ok(Ok(pcm)) => pcm,
            Ok(Err(e)) => {
                self.cleanup_forwarder_and_idle(forwarder).await;
                return Err(anyhow!(VoiceInputError::from(e)));
            }
            Err(join_err) => {
                self.cleanup_forwarder_and_idle(forwarder).await;
                return Err(anyhow!(VoiceInputError::ConsumerPanic(
                    join_err.to_string()
                )));
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

        let text = match result {
            Ok(t) => t,
            Err(e) => return Err(anyhow!(VoiceInputError::from(e))),
        };

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn voice_capture_state_pins_idle_default() {
        assert_eq!(VoiceCaptureState::Idle as u8, 0);
    }

    #[test]
    fn error_conversions_compile() {
        fn _from_capture(e: AudioCaptureError) -> VoiceInputError {
            e.into()
        }
        fn _from_transcribe(e: TranscriptionError) -> VoiceInputError {
            e.into()
        }
    }
}
