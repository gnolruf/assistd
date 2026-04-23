//! Microphone capture for push-to-talk voice input.
//!
//! `MicVoiceInput` wires cpal (cross-platform audio I/O), a lock-free
//! SPSC ring buffer (`ringbuf`), and a consumer task that resamples and
//! converts PCM for the existing `WhisperTranscriber`. The three pieces
//! are split across submodules so the audio-thread hot path and the
//! off-thread drain can be read independently.
//!
//! cpal's `Stream` is `!Send` on ALSA, so the stream lives entirely
//! inside a single `spawn_blocking` worker. The outer `MicVoiceInput`
//! holds only the atomics and the `JoinHandle` — all trivially `Send`.

use std::sync::Arc;
use std::sync::atomic::Ordering;

use anyhow::{Result, anyhow};
use assistd_config::VoiceConfig;
use async_trait::async_trait;
use thiserror::Error;
use tokio::sync::{Mutex, watch};
use tracing::{info, warn};

use crate::VoiceCaptureState;
use crate::VoiceInput;
use crate::transcribe::{Transcriber, TranscriptionError};
use crate::whisper::{WhisperTranscriber, WhisperTranscriberBuilder};

pub mod capture;
pub mod consumer;

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

/// Push-to-talk voice input backed by cpal + whisper-rs.
///
/// Construction builds the whisper context (downloads models on first
/// use) but does *not* open the audio device — that happens inside
/// [`start_recording`](VoiceInput::start_recording) so a headless CI
/// server without a mic still builds the daemon successfully.
pub struct MicVoiceInput {
    transcriber: Arc<WhisperTranscriber>,
    mic_device: Option<String>,
    max_recording_secs: u32,
    state_tx: watch::Sender<VoiceCaptureState>,
    inner: Arc<Mutex<InnerState>>,
}

struct InnerState {
    /// Handles for the currently active capture session, `Some`
    /// between `start_recording` and `stop_and_transcribe`.
    session: Option<capture::CaptureSession>,
}

impl MicVoiceInput {
    /// Build a voice input from the user's config. This async step
    /// downloads the whisper/VAD models on first use and probes GPU
    /// availability, but does NOT enumerate or open the audio device.
    pub async fn from_config(cfg: &VoiceConfig) -> Result<Self, VoiceInputError> {
        let transcriber = WhisperTranscriberBuilder::from_config(&cfg.transcription)
            .build()
            .await?;
        Ok(Self::new(
            Arc::new(transcriber),
            cfg.mic_device.clone(),
            cfg.max_recording_secs.max(1),
        ))
    }

    pub fn new(
        transcriber: Arc<WhisperTranscriber>,
        mic_device: Option<String>,
        max_recording_secs: u32,
    ) -> Self {
        let (state_tx, _) = watch::channel(VoiceCaptureState::Idle);
        Self {
            transcriber,
            mic_device,
            max_recording_secs,
            state_tx,
            inner: Arc::new(Mutex::new(InnerState { session: None })),
        }
    }
}

#[async_trait]
impl VoiceInput for MicVoiceInput {
    async fn start_recording(&self) -> Result<()> {
        let mut inner = self.inner.lock().await;
        if inner.session.is_some() {
            // Idempotent re-entry: already recording. A dropped
            // Released event would otherwise leave the user unable to
            // stop — let the first active recording carry on.
            return Ok(());
        }

        let session = capture::start(self.mic_device.as_deref(), self.max_recording_secs);
        inner.session = Some(session);
        drop(inner);

        let _ = self.state_tx.send(VoiceCaptureState::Recording);
        info!(target: "assistd::voice::mic", "recording started");
        Ok(())
    }

    async fn stop_and_transcribe(&self) -> Result<String> {
        let session = {
            let mut inner = self.inner.lock().await;
            match inner.session.take() {
                Some(s) => s,
                None => {
                    // Release without matching press — benign no-op.
                    return Ok(String::new());
                }
            }
        };

        let _ = self.state_tx.send(VoiceCaptureState::Transcribing);

        // Signal stop; the worker will exit its drain loop, drop the
        // cpal stream (on its own thread), and return the final PCM.
        session.stop_flag.store(true, Ordering::SeqCst);

        let pcm = match session.handle.await {
            Ok(Ok(pcm)) => pcm,
            Ok(Err(e)) => {
                let _ = self.state_tx.send(VoiceCaptureState::Idle);
                return Err(anyhow!(VoiceInputError::from(e)));
            }
            Err(join_err) => {
                let _ = self.state_tx.send(VoiceCaptureState::Idle);
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
                "ring buffer overrun during PTT capture — audio was truncated"
            );
        }

        if pcm.is_empty() {
            let _ = self.state_tx.send(VoiceCaptureState::Idle);
            return Ok(String::new());
        }

        let text = match self.transcriber.transcribe(&pcm).await {
            Ok(t) => t,
            Err(e) => {
                let _ = self.state_tx.send(VoiceCaptureState::Idle);
                return Err(anyhow!(VoiceInputError::from(e)));
            }
        };

        let _ = self.state_tx.send(VoiceCaptureState::Idle);
        info!(
            target: "assistd::voice::mic",
            pcm_samples = pcm.len(),
            text_chars = text.chars().count(),
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
        // Guards against accidental reordering of the enum — the
        // TUI treats Idle as the no-indicator default.
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
