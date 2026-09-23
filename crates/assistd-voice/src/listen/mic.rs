//! cpal + webrtc-vad + whisper hands-free listener: ring buffer →
//! resampler → VAD frames → transcriber → broadcast.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Result, anyhow};
use assistd_config::{ContinuousListenConfig, VoiceConfig};
use async_trait::async_trait;
use tokio::sync::mpsc::error::TrySendError;
use tokio::sync::{Mutex, broadcast, mpsc, watch};
use tokio::task::JoinHandle;
use tracing::{debug, info, warn};

use crate::transcribe::Transcriber;

use super::ContinuousListener;
use super::capture::{self, ListenCaptureSession};
use super::vad::{FRAME_SAMPLES, UtteranceVad, VadEvent, VadTuning};

type CaptureJoin = JoinHandle<Result<(), crate::mic::capture::AudioCaptureError>>;

const UTTERANCE_CHANNEL_DEPTH: usize = 16;

/// ~2.5 s of 20 ms frames, enough to ride out scheduling hiccups.
const FRAME_CHANNEL_DEPTH: usize = 128;

/// Utterances awaiting transcription. Each one holds its PCM until the
/// single transcription worker reaches it; beyond this, new utterances
/// are dropped rather than letting a slow transcriber accumulate audio.
const PENDING_UTTERANCE_DEPTH: usize = 4;

/// cpal + webrtc-vad implementation of [`ContinuousListener`].
pub struct MicContinuousListener {
    transcriber: Arc<dyn Transcriber>,
    mic_device: Option<String>,
    tuning: VadTuning,
    active: Arc<AtomicBool>,
    state_tx: watch::Sender<bool>,
    utterances: broadcast::Sender<String>,
    inner: Arc<Mutex<ListenState>>,
}

struct ListenState {
    session: Option<ListenSession>,
}

struct ListenSession {
    capture_stop: Arc<AtomicBool>,
    capture_handle: CaptureJoin,
    vad_handle: JoinHandle<()>,
    transcribe_handle: JoinHandle<()>,
}

impl MicContinuousListener {
    /// The audio device is opened on [`Self::start`], not here.
    pub fn new(transcriber: Arc<dyn Transcriber>, cfg: &VoiceConfig) -> Self {
        let tuning = tuning_from_config(&cfg.continuous);
        let (state_tx, _) = watch::channel(false);
        let (utterances, _) = broadcast::channel(UTTERANCE_CHANNEL_DEPTH);
        Self {
            transcriber,
            mic_device: cfg.mic_device.clone(),
            tuning,
            active: Arc::new(AtomicBool::new(false)),
            state_tx,
            utterances,
            inner: Arc::new(Mutex::new(ListenState { session: None })),
        }
    }

    /// Shared flag that is `true` while listening.
    pub fn active_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.active)
    }
}

fn tuning_from_config(cfg: &ContinuousListenConfig) -> VadTuning {
    VadTuning::from_ms(cfg.silence_ms.get(), cfg.max_utterance_secs.get())
}

#[async_trait]
impl ContinuousListener for MicContinuousListener {
    async fn start(&self) -> Result<()> {
        let mut inner = self.inner.lock().await;
        if inner.session.is_some() {
            return Ok(());
        }
        if self
            .active
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return Err(anyhow!(
                "continuous listening is already active on another subsystem"
            ));
        }

        let (frame_tx, frame_rx) = mpsc::channel::<Box<[i16; FRAME_SAMPLES]>>(FRAME_CHANNEL_DEPTH);

        let ListenCaptureSession {
            stop_flag: capture_stop,
            overrun: _,
            handle: capture_handle,
        } = capture::start(self.mic_device.as_deref(), frame_tx);

        let (pcm_tx, pcm_rx) = mpsc::channel::<Vec<i16>>(PENDING_UTTERANCE_DEPTH);
        let transcribe_handle = tokio::spawn(transcribe_loop(
            self.transcriber.clone(),
            self.utterances.clone(),
            pcm_rx,
        ));
        let tuning = self.tuning;
        let vad_handle = tokio::task::spawn_blocking(move || vad_loop(tuning, frame_rx, pcm_tx));

        inner.session = Some(ListenSession {
            capture_stop,
            capture_handle,
            vad_handle,
            transcribe_handle,
        });
        let _ = self.state_tx.send(true);
        info!(target: "assistd::voice::listen", "continuous listening started");
        Ok(())
    }

    async fn stop(&self) -> Result<()> {
        let session = {
            let mut inner = self.inner.lock().await;
            inner.session.take()
        };
        let Some(session) = session else {
            self.active.store(false, Ordering::SeqCst);
            return Ok(());
        };

        session.capture_stop.store(true, Ordering::SeqCst);
        match session.capture_handle.await {
            Ok(Ok(())) => {}
            Ok(Err(e)) => warn!(target: "assistd::voice::listen", "capture worker error: {e}"),
            Err(e) => warn!(target: "assistd::voice::listen", "capture worker panicked: {e}"),
        }
        if let Err(e) = session.vad_handle.await {
            warn!(target: "assistd::voice::listen", "VAD worker panicked: {e}");
        }
        if let Err(e) = session.transcribe_handle.await {
            warn!(target: "assistd::voice::listen", "transcription worker panicked: {e}");
        }

        self.active.store(false, Ordering::SeqCst);
        let _ = self.state_tx.send(false);
        info!(target: "assistd::voice::listen", "continuous listening stopped");
        Ok(())
    }

    fn is_active(&self) -> bool {
        self.active.load(Ordering::SeqCst)
    }

    fn subscribe_utterances(&self) -> broadcast::Receiver<String> {
        self.utterances.subscribe()
    }

    fn subscribe_state(&self) -> watch::Receiver<bool> {
        self.state_tx.subscribe()
    }
}

/// Blocking because `webrtc_vad::Vad` holds a `!Send` pointer. Never
/// waits on `pcm_tx`: stalling here would back up the frame channel and
/// silently truncate live audio, so a full queue drops the utterance.
fn vad_loop(
    tuning: VadTuning,
    mut frame_rx: mpsc::Receiver<Box<[i16; FRAME_SAMPLES]>>,
    pcm_tx: mpsc::Sender<Vec<i16>>,
) {
    let mut vad = UtteranceVad::new(tuning);
    while let Some(frame) = frame_rx.blocking_recv() {
        let Some(event) = vad.feed(&frame) else {
            continue;
        };
        let pcm = match event {
            VadEvent::UtteranceComplete(p) | VadEvent::Truncated(p) => p,
        };
        match pcm_tx.try_send(pcm) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => warn!(
                target: "assistd::voice::listen",
                "transcription backlog full; dropping utterance"
            ),
            Err(TrySendError::Closed(_)) => return,
        }
    }
}

async fn transcribe_loop(
    transcriber: Arc<dyn Transcriber>,
    utterances: broadcast::Sender<String>,
    mut pcm_rx: mpsc::Receiver<Vec<i16>>,
) {
    while let Some(pcm) = pcm_rx.recv().await {
        match transcriber.transcribe(&pcm).await {
            Ok(text) => {
                let trimmed = text.trim();
                if trimmed.is_empty() {
                    continue;
                }
                if utterances.send(trimmed.to_string()).is_err() {
                    debug!(
                        target: "assistd::voice::listen",
                        "no utterance subscribers; dropping transcript"
                    );
                }
            }
            Err(e) => {
                warn!(target: "assistd::voice::listen", "transcription failed: {e:#}");
            }
        }
    }
}

#[cfg(test)]
mod tests;
