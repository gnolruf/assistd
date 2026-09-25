//! cpal + webrtc-vad + whisper hands-free listener: ring buffer →
//! resampler → VAD frames → transcriber → broadcast.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use assistd_config::{ContinuousListenConfig, VoiceConfig};
use async_trait::async_trait;
use tokio::sync::mpsc::error::TrySendError;
use tokio::sync::{Mutex, broadcast, mpsc, watch};
use tokio::task::JoinHandle;
use tracing::{debug, info, warn};

use super::capture::{self, ListenCaptureSession};
use super::vad::{FRAME_SAMPLES, UtteranceVad, VadEvent, VadTuning};
use super::{ContinuousListener, ListenError};
use crate::mic::capture::AudioCaptureError;
use crate::transcribe::Transcriber;

const UTTERANCE_CHANNEL_DEPTH: usize = 16;

/// ~2.5 s of 20 ms frames, enough to ride out scheduling hiccups.
const FRAME_CHANNEL_DEPTH: usize = 128;

/// Utterances awaiting transcription; beyond this, new utterances are dropped
/// rather than letting a slow transcriber accumulate audio.
const PENDING_UTTERANCE_DEPTH: usize = 4;

type CaptureJoin = JoinHandle<Result<(), AudioCaptureError>>;

/// cpal + webrtc-vad implementation of [`ContinuousListener`].
pub struct MicContinuousListener {
    transcriber: Arc<dyn Transcriber>,
    mic_device: Option<String>,
    tuning: VadTuning,
    active: AtomicBool,
    state_tx: watch::Sender<bool>,
    utterances: broadcast::Sender<String>,
    listen_state: Arc<Mutex<ListenState>>,
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

impl ListenSession {
    /// Signal capture to stop, then wait for each worker in pipeline order.
    async fn shutdown(self) {
        self.capture_stop.store(true, Ordering::SeqCst);
        match self.capture_handle.await {
            Ok(Ok(())) => {}
            Ok(Err(err)) => warn!(target: "assistd::voice::listen", "capture worker error: {err}"),
            Err(err) => warn!(target: "assistd::voice::listen", "capture worker panicked: {err}"),
        }
        if let Err(err) = self.vad_handle.await {
            warn!(target: "assistd::voice::listen", "VAD worker panicked: {err}");
        }
        if let Err(err) = self.transcribe_handle.await {
            warn!(target: "assistd::voice::listen", "transcription worker panicked: {err}");
        }
    }
}

impl MicContinuousListener {
    /// The audio device is opened on [`Self::start`], not here.
    pub fn new(transcriber: Arc<dyn Transcriber>, config: &VoiceConfig) -> Self {
        let tuning = tuning_from_config(&config.continuous);
        let (state_tx, _) = watch::channel(false);
        let (utterances, _) = broadcast::channel(UTTERANCE_CHANNEL_DEPTH);
        Self {
            transcriber,
            mic_device: config.mic_device.clone(),
            tuning,
            active: AtomicBool::new(false),
            state_tx,
            utterances,
            listen_state: Arc::new(Mutex::new(ListenState { session: None })),
        }
    }
}

#[async_trait]
impl ContinuousListener for MicContinuousListener {
    async fn start(&self) -> Result<(), ListenError> {
        let mut listen_state = self.listen_state.lock().await;
        if listen_state.session.is_some() {
            return Ok(());
        }
        self.active.store(true, Ordering::SeqCst);

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

        listen_state.session = Some(ListenSession {
            capture_stop,
            capture_handle,
            vad_handle,
            transcribe_handle,
        });
        let _ = self.state_tx.send(true);
        info!(target: "assistd::voice::listen", "continuous listening started");
        Ok(())
    }

    async fn stop(&self) -> Result<(), ListenError> {
        let session = self.listen_state.lock().await.session.take();
        let Some(session) = session else {
            self.active.store(false, Ordering::SeqCst);
            return Ok(());
        };

        session.shutdown().await;

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

fn tuning_from_config(config: &ContinuousListenConfig) -> VadTuning {
    VadTuning::from_ms(config.silence_ms.get(), config.max_utterance_secs.get())
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
            VadEvent::UtteranceComplete(pcm) | VadEvent::Truncated(pcm) => pcm,
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
            Err(err) => {
                warn!(target: "assistd::voice::listen", "transcription failed: {err:#}");
            }
        }
    }
}

#[cfg(test)]
mod tests;
