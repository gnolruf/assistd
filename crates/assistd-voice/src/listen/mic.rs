//! cpal + webrtc-vad + whisper hands-free listener: ring buffer →
//! resampler → playback gate → VAD frames → transcriber → wake gate → broadcast.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use assistd_config::{VoiceConfig, WakeWordConfig};
use assistd_ipc::ListenPhase;
use async_trait::async_trait;
use tokio::sync::mpsc::error::TrySendError;
use tokio::sync::{Mutex, broadcast, mpsc, watch};
use tokio::task::JoinHandle;
use tokio::time::Instant;
use tracing::{debug, info, trace, warn};

use super::capture::{self, ListenCaptureSession};
use super::enrollment::{CapturedUtterance, SessionAudio};
use super::playback_gate::{PLAYBACK_HANGOVER_FRAMES, PlaybackGate};
use super::vad::{FRAME_SAMPLES, SAMPLE_RATE_HZ, UtteranceVad, VadEvent, VadTuning};
use super::wake_gate::{Admission, WakeGate};
use super::wake_phrase::WakePhrase;
use super::{ContinuousListener, Heard, ListenError};
use crate::mic::capture::AudioCaptureError;
use crate::transcribe::Transcriber;

const UTTERANCE_CHANNEL_DEPTH: usize = 16;

/// ~2.5 s of 20 ms frames, enough to ride out scheduling hiccups.
const FRAME_CHANNEL_DEPTH: usize = 128;

/// Utterances awaiting transcription; beyond this, new utterances are dropped
/// rather than letting a slow transcriber accumulate audio.
const PENDING_UTTERANCE_DEPTH: usize = 4;

type CaptureJoin = JoinHandle<Result<(), AudioCaptureError>>;

#[derive(Debug)]
struct Utterance {
    pcm: Vec<i16>,
    started: Instant,
}

impl Utterance {
    /// Dates `pcm` back from now by its own length.
    fn ending_now(pcm: Vec<i16>) -> Self {
        let millis = pcm.len() as u64 * 1000 / u64::from(SAMPLE_RATE_HZ);
        let now = Instant::now();
        let started = now
            .checked_sub(Duration::from_millis(millis))
            .unwrap_or(now);
        Self { pcm, started }
    }
}

/// cpal + webrtc-vad implementation of [`ContinuousListener`].
#[derive(Debug)]
pub struct MicContinuousListener {
    transcriber: Arc<dyn Transcriber>,
    mic_device: Option<String>,
    tuning: VadTuning,
    output_speaking: Option<watch::Receiver<bool>>,
    wake_phrase: parking_lot::Mutex<WakePhrase>,
    command_timeout: Duration,
    follow_up: Option<Duration>,
    /// The latest signal for the running session; each session subscribes
    /// at start, so signals sent while stopped are ignored.
    conversation_tx: watch::Sender<Option<ConversationSignal>>,
    last_push_to_talk: parking_lot::Mutex<Option<Instant>>,
    active: AtomicBool,
    capturing_utterances: Arc<AtomicBool>,
    phase_tx: watch::Sender<ListenPhase>,
    heard: broadcast::Sender<Heard>,
    listen_state: Arc<Mutex<ListenState>>,
}

/// Steers the running session's conversation from outside it.
#[derive(Debug, Clone, Copy)]
enum ConversationSignal {
    /// A reply to a wake-word command ended at this instant.
    FollowUp(Instant),
    /// Push-to-talk took the mic; act on nothing more this session.
    Abandon,
}

#[derive(Debug)]
struct ListenState {
    session: Option<ListenSession>,
}

#[derive(Debug)]
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
    /// Listens for the built-in wake phrase. The audio device is opened on
    /// [`Self::start`], not here. When the playback gate is configured on,
    /// frames captured while `output_speaking` is true are discarded.
    pub fn new(
        transcriber: Arc<dyn Transcriber>,
        config: &VoiceConfig,
        output_speaking: watch::Receiver<bool>,
    ) -> Self {
        let tuning = tuning_from_config(&config.wakeword);
        let output_speaking = config.wakeword.playback_gate.then_some(output_speaking);
        let (phase_tx, _) = watch::channel(ListenPhase::Off);
        let (heard, _) = broadcast::channel(UTTERANCE_CHANNEL_DEPTH);
        Self {
            transcriber,
            mic_device: config.mic_device.clone(),
            tuning,
            output_speaking,
            wake_phrase: parking_lot::Mutex::new(WakePhrase::builtin(
                config.wakeword.match_similarity,
            )),
            command_timeout: Duration::from_millis(config.wakeword.command_timeout_ms.get().into()),
            follow_up: (config.wakeword.follow_up_ms > 0)
                .then(|| Duration::from_millis(config.wakeword.follow_up_ms.into())),
            conversation_tx: watch::channel(None).0,
            last_push_to_talk: parking_lot::Mutex::new(None),
            active: AtomicBool::new(false),
            capturing_utterances: Arc::new(AtomicBool::new(false)),
            phase_tx,
            heard,
            listen_state: Arc::new(Mutex::new(ListenState { session: None })),
        }
    }

    fn playback_gate(&self) -> Option<PlaybackGate> {
        self.output_speaking
            .clone()
            .map(|speaking| PlaybackGate::new(speaking, PLAYBACK_HANGOVER_FRAMES))
    }

    /// Run capture and the VAD into `sink` until it closes or capture fails.
    async fn stream_utterances(&self, sink: mpsc::Sender<SessionAudio>) -> Result<(), ListenError> {
        let (frame_tx, frame_rx) = mpsc::channel::<Box<[i16; FRAME_SAMPLES]>>(FRAME_CHANNEL_DEPTH);
        let ListenCaptureSession {
            stop_flag,
            overrun: _,
            handle: mut capture_handle,
        } = capture::start(self.mic_device.as_deref(), frame_tx);
        let tuning = self.tuning;
        let playback_gate = self.playback_gate();
        let vad_sink = sink.clone();
        let vad_handle = tokio::task::spawn_blocking(move || {
            vad_loop(
                tuning,
                playback_gate,
                frame_rx,
                |event| forward_captured(&vad_sink, event),
                |speaking| {
                    let _ = vad_sink.try_send(SessionAudio::Speaking(speaking));
                },
            );
        });
        let finished_early = tokio::select! {
            result = &mut capture_handle => Some(result),
            () = sink.closed() => None,
        };
        stop_flag.store(true, Ordering::SeqCst);
        let captured = match finished_early {
            Some(result) => result,
            None => capture_handle.await,
        };
        vad_handle.await?;
        Ok(captured??)
    }
}

#[async_trait]
impl ContinuousListener for MicContinuousListener {
    async fn start(&self) -> Result<(), ListenError> {
        let mut listen_state = self.listen_state.lock().await;
        if listen_state.session.is_some() {
            return Ok(());
        }
        if self.capturing_utterances.load(Ordering::SeqCst) {
            return Err(ListenError::Enrolling);
        }
        self.active.store(true, Ordering::SeqCst);

        let (frame_tx, frame_rx) = mpsc::channel::<Box<[i16; FRAME_SAMPLES]>>(FRAME_CHANNEL_DEPTH);

        let ListenCaptureSession {
            stop_flag: capture_stop,
            overrun: _,
            handle: capture_handle,
        } = capture::start(self.mic_device.as_deref(), frame_tx);

        let (pcm_tx, pcm_rx) = mpsc::channel::<Utterance>(PENDING_UTTERANCE_DEPTH);
        let gate = WakeGate::new(
            self.wake_phrase.lock().clone(),
            self.command_timeout,
            self.follow_up,
        );
        let transcribe_handle = tokio::spawn(transcribe_loop(
            self.transcriber.clone(),
            gate,
            ListenOutputs {
                heard: self.heard.clone(),
                phase: self.phase_tx.clone(),
            },
            TranscribeInputs {
                utterances: pcm_rx,
                signals: self.conversation_tx.subscribe(),
            },
        ));
        let tuning = self.tuning;
        let playback_gate = self.playback_gate();
        let vad_handle = tokio::task::spawn_blocking(move || {
            vad_loop(
                tuning,
                playback_gate,
                frame_rx,
                |event| queue_for_transcription(&pcm_tx, event),
                |_| {},
            );
        });

        listen_state.session = Some(ListenSession {
            capture_stop,
            capture_handle,
            vad_handle,
            transcribe_handle,
        });
        self.phase_tx.send_replace(ListenPhase::Armed);
        info!(target: "assistd::voice::listen", "wake-word listening started");
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
        self.phase_tx.send_replace(ListenPhase::Off);
        info!(target: "assistd::voice::listen", "wake-word listening stopped");
        Ok(())
    }

    async fn stop_for_push_to_talk(&self) -> Result<(), ListenError> {
        *self.last_push_to_talk.lock() = Some(Instant::now());
        self.conversation_tx
            .send_replace(Some(ConversationSignal::Abandon));
        self.stop().await
    }

    fn last_push_to_talk(&self) -> Option<Instant> {
        *self.last_push_to_talk.lock()
    }

    fn is_active(&self) -> bool {
        self.active.load(Ordering::SeqCst)
    }

    fn subscribe_heard(&self) -> broadcast::Receiver<Heard> {
        self.heard.subscribe()
    }

    fn subscribe_phase(&self) -> watch::Receiver<ListenPhase> {
        self.phase_tx.subscribe()
    }

    fn set_wake_phrase(&self, phrase: WakePhrase) {
        *self.wake_phrase.lock() = phrase;
    }

    fn open_follow_up(&self) {
        self.conversation_tx
            .send_replace(Some(ConversationSignal::FollowUp(Instant::now())));
    }

    async fn capture_utterances(
        &self,
        sink: mpsc::Sender<SessionAudio>,
    ) -> Result<(), ListenError> {
        let _claim = {
            let listen_state = self.listen_state.lock().await;
            if listen_state.session.is_some() {
                return Err(ListenError::Listening);
            }
            if self.capturing_utterances.swap(true, Ordering::SeqCst) {
                return Err(ListenError::Enrolling);
            }
            UtteranceCaptureClaim(Arc::clone(&self.capturing_utterances))
        };
        self.stream_utterances(sink).await
    }
}

/// Clears the utterance-capture flag when the capture ends or is dropped.
struct UtteranceCaptureClaim(Arc<AtomicBool>);

impl Drop for UtteranceCaptureClaim {
    fn drop(&mut self) {
        self.0.store(false, Ordering::SeqCst);
    }
}

/// What the transcription worker waits on.
#[derive(Debug)]
struct TranscribeInputs {
    utterances: mpsc::Receiver<Utterance>,
    signals: watch::Receiver<Option<ConversationSignal>>,
}

impl TranscribeInputs {
    /// Whether push-to-talk took the mic since the last signal was applied.
    fn abandoned(&self) -> bool {
        self.signals.has_changed().unwrap_or(false)
            && matches!(*self.signals.borrow(), Some(ConversationSignal::Abandon))
    }

    /// The next utterance, or `None` once its sender is gone or push-to-talk
    /// took the mic. Signals, applied before any queued utterance is
    /// returned, and a closing command window update `gate` while waiting.
    async fn next_utterance(
        &mut self,
        gate: &mut WakeGate,
        outputs: &ListenOutputs,
    ) -> Option<Utterance> {
        loop {
            let window_close = gate
                .command_deadline()
                .filter(|_| *outputs.phase.borrow() == ListenPhase::Capturing);
            tokio::select! {
                biased;
                Ok(()) = self.signals.changed() => {
                    match *self.signals.borrow_and_update() {
                        Some(ConversationSignal::FollowUp(reply_ended)) => {
                            gate.open_follow_up(reply_ended);
                        }
                        Some(ConversationSignal::Abandon) => return None,
                        None => {}
                    }
                    outputs.set_phase(gate.phase(Instant::now()));
                }
                utterance = self.utterances.recv() => return utterance,
                () = sleep_until_some(window_close) => outputs.set_phase(ListenPhase::Armed),
            }
        }
    }
}

#[derive(Debug)]
struct ListenOutputs {
    heard: broadcast::Sender<Heard>,
    phase: watch::Sender<ListenPhase>,
}

impl ListenOutputs {
    fn send_heard(&self, heard: Heard) {
        if self.heard.send(heard).is_err() {
            debug!(
                target: "assistd::voice::listen",
                "no listen subscribers; dropping what was heard"
            );
        }
    }

    fn publish(&self, admission: Admission) {
        match admission {
            Admission::Dropped => {}
            Admission::WakePhrase => self.send_heard(Heard::WakePhrase),
            Admission::WakePhraseAndCommand(command) => {
                self.send_heard(Heard::WakePhrase);
                self.send_heard(Heard::Command(command));
            }
            Admission::Command(command) => self.send_heard(Heard::Command(command)),
        }
    }

    fn set_phase(&self, phase: ListenPhase) {
        self.phase
            .send_if_modified(|current| std::mem::replace(current, phase) != phase);
    }
}

fn tuning_from_config(config: &WakeWordConfig) -> VadTuning {
    VadTuning::from_ms(config.silence_ms.get(), config.max_utterance_secs.get())
}

/// Blocking because `webrtc_vad::Vad` holds a `!Send` pointer. Frames
/// blocked by the playback gate also reset the VAD, since a half-captured
/// utterance is about to be contaminated by the daemon's own speech.
/// `emit` and `speaking_changed`, told when an utterance starts and ends,
/// must never wait: stalling here would back up the frame channel and
/// silently truncate live audio. The loop ends when `emit` returns false.
fn vad_loop(
    tuning: VadTuning,
    mut playback_gate: Option<PlaybackGate>,
    mut frame_rx: mpsc::Receiver<Box<[i16; FRAME_SAMPLES]>>,
    mut emit: impl FnMut(VadEvent) -> bool,
    mut speaking_changed: impl FnMut(bool),
) {
    let mut vad = UtteranceVad::new(tuning);
    let mut speaking = false;
    while let Some(frame) = frame_rx.blocking_recv() {
        let event = if playback_gate
            .as_mut()
            .is_some_and(PlaybackGate::blocks_frame)
        {
            vad.reset();
            None
        } else {
            vad.feed(&frame)
        };
        if vad.in_utterance() != speaking {
            speaking = !speaking;
            speaking_changed(speaking);
        }
        if let Some(event) = event
            && !emit(event)
        {
            return;
        }
    }
}

/// Queue a listening utterance, dropping it when the backlog is full.
/// False once the transcription worker is gone.
fn queue_for_transcription(pcm_tx: &mpsc::Sender<Utterance>, event: VadEvent) -> bool {
    let pcm = match event {
        VadEvent::UtteranceComplete(pcm) | VadEvent::Truncated(pcm) => pcm,
    };
    try_queue(pcm_tx, Utterance::ending_now(pcm))
}

/// Hand an enrollment utterance to the session. False once it closed.
fn forward_captured(sink: &mpsc::Sender<SessionAudio>, event: VadEvent) -> bool {
    let utterance = match event {
        VadEvent::UtteranceComplete(pcm) => CapturedUtterance {
            pcm,
            truncated: false,
        },
        VadEvent::Truncated(pcm) => CapturedUtterance {
            pcm,
            truncated: true,
        },
    };
    try_queue(sink, SessionAudio::Utterance(utterance))
}

fn try_queue<T>(tx: &mpsc::Sender<T>, item: T) -> bool {
    match tx.try_send(item) {
        Ok(()) => true,
        Err(TrySendError::Full(_)) => {
            warn!(
                target: "assistd::voice::listen",
                "utterance backlog full; dropping utterance"
            );
            true
        }
        Err(TrySendError::Closed(_)) => false,
    }
}

/// Transcribes each utterance in turn and gates it on the wake phrase. A
/// bare phrase or a follow-up opens a command window, reported as
/// `Capturing` until it closes.
async fn transcribe_loop(
    transcriber: Arc<dyn Transcriber>,
    mut gate: WakeGate,
    outputs: ListenOutputs,
    mut inputs: TranscribeInputs,
) {
    while let Some(utterance) = inputs.next_utterance(&mut gate, &outputs).await {
        let Some(transcript) = transcribe_utterance(transcriber.as_ref(), &gate, &utterance).await
        else {
            continue;
        };
        if inputs.abandoned() {
            return;
        }
        let admission = gate.admit(&transcript, utterance.started);
        if admission == Admission::Dropped {
            trace!(target: "assistd::voice::listen", %transcript, "no wake phrase; dropped");
        }
        outputs.publish(admission);
        outputs.set_phase(gate.phase(Instant::now()));
    }
}

async fn transcribe_utterance(
    transcriber: &dyn Transcriber,
    gate: &WakeGate,
    utterance: &Utterance,
) -> Option<String> {
    match transcriber
        .transcribe(&utterance.pcm, Some(gate.phrase()))
        .await
    {
        Ok(text) => {
            let trimmed = text.trim();
            (!trimmed.is_empty()).then(|| trimmed.to_string())
        }
        Err(err) => {
            warn!(target: "assistd::voice::listen", "transcription failed: {err:#}");
            None
        }
    }
}

async fn sleep_until_some(deadline: Option<Instant>) {
    match deadline {
        Some(deadline) => tokio::time::sleep_until(deadline).await,
        None => std::future::pending().await,
    }
}

#[cfg(test)]
mod tests;
