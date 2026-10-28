//! cpal + Silero + keyword-spotting hands-free listener: ring buffer → resampler → Silero →
//! utterance segmenter → wake-phrase gate or barge-in check → transcriber → broadcast.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use assistd_config::{VoiceConfig, WakeWordConfig};
use assistd_ipc::ListenPhase;
use async_trait::async_trait;
use tokio::sync::mpsc::error::TrySendError;
use tokio::sync::{Mutex, broadcast, mpsc, oneshot, watch};
use tokio::task::JoinHandle;
use tokio::time::Instant;
use tracing::{debug, error, info, trace, warn};

use super::barge_in::{BargeInWatch, confirms_barge_in};
use super::capture::{self, ListenCaptureSession};
use super::session::{CapturedUtterance, SessionAudio};
use super::silero::{FrameClassifier, SileroVad};
use super::spotter::WakeWordSpotter;
use super::vad::{FRAME_SAMPLES, SAMPLE_RATE_HZ, UtteranceVad, VadEvent, VadTuning};
use super::wake_gate::{Admission, WakeGate};
use super::{ContinuousListener, Heard, ListenError};
use crate::controller::VoiceOutputController;
use crate::mic::capture::AudioCaptureError;
use crate::transcribe::Transcriber;

const UTTERANCE_CHANNEL_DEPTH: usize = 16;

/// ~2.5 s of 20 ms frames, enough to ride out scheduling hiccups.
const FRAME_CHANNEL_DEPTH: usize = 128;

/// Utterances awaiting transcription; beyond this, new utterances are dropped
/// rather than letting a slow transcriber accumulate audio.
const PENDING_UTTERANCE_DEPTH: usize = 4;

/// Shortest speech after the wake phrase worth transcribing as a command:
/// 300 ms at 16 kHz.
const MIN_COMMAND_SAMPLES: usize = 4_800;

type CaptureJoin = JoinHandle<Result<(), AudioCaptureError>>;

#[derive(Debug)]
struct Utterance {
    pcm: Vec<i16>,
    started: Instant,
    /// Began while a reply was under way, so it is a barge-in or nothing.
    during_reply: bool,
}

impl Utterance {
    /// Dates `pcm` back from now by its own length.
    fn ending_now(pcm: Vec<i16>, during_reply: bool) -> Self {
        let millis = pcm.len() as u64 * 1000 / u64::from(SAMPLE_RATE_HZ);
        let now = Instant::now();
        let started = now
            .checked_sub(Duration::from_millis(millis))
            .unwrap_or(now);
        Self {
            pcm,
            started,
            during_reply,
        }
    }
}

/// Audio queued for the transcription worker.
#[derive(Debug)]
enum ListenAudio {
    Utterance(Utterance),
    /// Recent audio of speech still going on over a reply.
    BargeInCheck(Vec<i16>),
}

impl ListenAudio {
    fn pcm(&self) -> &[i16] {
        match self {
            Self::Utterance(utterance) => &utterance.pcm,
            Self::BargeInCheck(pcm) => pcm,
        }
    }
}

/// What the VAD loop hands on.
#[derive(Debug)]
enum Segment {
    Utterance { event: VadEvent, during_reply: bool },
    BargeInCheck(Vec<i16>),
}

/// cpal + Silero implementation of [`ContinuousListener`].
#[derive(Debug)]
pub struct MicContinuousListener {
    transcriber: Arc<dyn Transcriber>,
    /// Shared by listening and utterance capture, which never run together.
    silero: Arc<parking_lot::Mutex<SileroVad>>,
    mic_device: Option<String>,
    tuning: VadTuning,
    speech: Arc<VoiceOutputController>,
    barge_in: bool,
    replying_tx: watch::Sender<bool>,
    spotter: Arc<WakeWordSpotter>,
    command_timeout: Duration,
    follow_up: Option<Duration>,
    /// The latest signal for the running session; each session subscribes
    /// at start, so signals sent while stopped are ignored.
    conversation_tx: watch::Sender<Option<ConversationSignal>>,
    last_push_to_talk: parking_lot::Mutex<Option<Instant>>,
    active: Arc<AtomicBool>,
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

impl ListenState {
    fn is_running(&self) -> bool {
        self.session.as_ref().is_some_and(ListenSession::is_running)
    }
}

#[derive(Debug)]
struct ListenSession {
    capture_stop: CaptureStop,
    pipeline: JoinHandle<()>,
}

impl ListenSession {
    /// False once the pipeline ended, whether stopped or because capture died.
    fn is_running(&self) -> bool {
        !self.pipeline.is_finished()
    }

    /// Signal capture to stop, then wait for the pipeline to drain.
    async fn shutdown(self) {
        self.capture_stop.request();
        if let Err(err) = self.pipeline.await {
            warn!(target: "assistd::voice::listen", "listen pipeline panicked: {err}");
        }
    }
}

/// Sets a capture's stop flag when dropped, so an abandoned session or
/// cancelled caller releases the mic.
#[derive(Debug)]
struct CaptureStop(Arc<AtomicBool>);

impl CaptureStop {
    fn request(&self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

impl Drop for CaptureStop {
    fn drop(&mut self) {
        self.request();
    }
}

/// The workers of one listening session.
#[derive(Debug)]
struct ListenWorkers {
    capture: CaptureJoin,
    vad: JoinHandle<()>,
    transcribe: JoinHandle<()>,
}

impl ListenWorkers {
    /// Wait for each worker in pipeline order; how capture ended.
    async fn join(self) -> Result<(), ListenError> {
        let captured = self.capture.await;
        if let Err(err) = self.vad.await {
            warn!(target: "assistd::voice::listen", "VAD worker panicked: {err}");
        }
        if let Err(err) = self.transcribe.await {
            warn!(target: "assistd::voice::listen", "transcription worker panicked: {err}");
        }
        Ok(captured??)
    }
}

impl MicContinuousListener {
    /// Listens for the wake phrase `spotter` finds; the audio device opens on [`Self::start`].
    /// `speech` supplies reply text that barge-in checks reject as echo; `silero` decides which
    /// frames hold speech.
    pub fn new(
        transcriber: Arc<dyn Transcriber>,
        config: &VoiceConfig,
        speech: Arc<VoiceOutputController>,
        silero: SileroVad,
        spotter: Arc<WakeWordSpotter>,
    ) -> Self {
        let tuning = tuning_from_config(&config.wakeword);
        let (phase_tx, _) = watch::channel(ListenPhase::Off);
        let (heard, _) = broadcast::channel(UTTERANCE_CHANNEL_DEPTH);
        Self {
            transcriber,
            silero: Arc::new(parking_lot::Mutex::new(silero)),
            mic_device: config.mic_device.clone(),
            tuning,
            speech,
            barge_in: config.wakeword.barge_in,
            replying_tx: watch::channel(false).0,
            spotter,
            command_timeout: Duration::from_millis(config.wakeword.command_timeout_ms.get().into()),
            follow_up: (config.wakeword.follow_up_ms > 0)
                .then(|| Duration::from_millis(config.wakeword.follow_up_ms.into())),
            conversation_tx: watch::channel(None).0,
            last_push_to_talk: parking_lot::Mutex::new(None),
            active: Arc::new(AtomicBool::new(false)),
            capturing_utterances: Arc::new(AtomicBool::new(false)),
            phase_tx,
            heard,
            listen_state: Arc::new(Mutex::new(ListenState { session: None })),
        }
    }

    fn barge_in_watch(&self) -> Option<BargeInWatch> {
        self.barge_in
            .then(|| BargeInWatch::new(self.replying_tx.subscribe()))
    }

    /// Spawn the VAD and transcription workers behind a capture that opened.
    fn spawn_workers(
        &self,
        capture: CaptureJoin,
        frame_rx: mpsc::Receiver<Box<[i16; FRAME_SAMPLES]>>,
    ) -> ListenWorkers {
        let (pcm_tx, pcm_rx) = mpsc::channel::<ListenAudio>(PENDING_UTTERANCE_DEPTH);
        let gate = WakeGate::new(self.command_timeout, self.follow_up);
        let transcribe = tokio::spawn(transcribe_loop(
            Listeners {
                transcriber: self.transcriber.clone(),
                spotter: Arc::clone(&self.spotter),
            },
            gate,
            ListenOutputs {
                heard: self.heard.clone(),
                phase: self.phase_tx.clone(),
            },
            TranscribeInputs {
                utterances: pcm_rx,
                signals: self.conversation_tx.subscribe(),
            },
            BargeIn::new(Arc::clone(&self.speech)),
        ));
        let tuning = self.tuning;
        let barge_in = self.barge_in_watch();
        let classifier = FrameClassifier::new(Arc::clone(&self.silero));
        let vad = tokio::task::spawn_blocking(move || {
            vad_loop(
                tuning,
                barge_in,
                classifier,
                frame_rx,
                |segment| queue_for_transcription(&pcm_tx, segment),
                |_| {},
            );
        });
        ListenWorkers {
            capture,
            vad,
            transcribe,
        }
    }

    /// Drain a taken session, if any, and mark listening off.
    async fn shut_down(&self, session: Option<ListenSession>) {
        let Some(session) = session else {
            self.active.store(false, Ordering::SeqCst);
            return;
        };
        session.shutdown().await;
        self.active.store(false, Ordering::SeqCst);
        self.phase_tx.send_replace(ListenPhase::Off);
        info!(target: "assistd::voice::listen", "wake-word listening stopped");
    }

    /// Run capture and the VAD into `sink` until it closes or capture fails.
    async fn stream_utterances(&self, sink: mpsc::Sender<SessionAudio>) -> Result<(), ListenError> {
        let (frame_tx, frame_rx) = mpsc::channel::<Box<[i16; FRAME_SAMPLES]>>(FRAME_CHANNEL_DEPTH);
        let ListenCaptureSession {
            stop_flag,
            overrun: _,
            opened: _,
            handle: mut capture_handle,
        } = capture::start(self.mic_device.as_deref(), frame_tx);
        let stop = CaptureStop(stop_flag);
        let tuning = self.tuning;
        let classifier = FrameClassifier::new(Arc::clone(&self.silero));
        let vad_sink = sink.clone();
        let vad_handle = tokio::task::spawn_blocking(move || {
            vad_loop(
                tuning,
                None,
                classifier,
                frame_rx,
                |segment| forward_captured(&vad_sink, segment),
                |speaking| {
                    let _ = vad_sink.try_send(SessionAudio::Speaking(speaking));
                },
            );
        });
        let finished_early = tokio::select! {
            result = &mut capture_handle => Some(result),
            () = sink.closed() => None,
        };
        stop.request();
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
        if listen_state.is_running() {
            return Ok(());
        }
        if self.capturing_utterances.load(Ordering::SeqCst) {
            return Err(ListenError::Testing);
        }

        let (frame_tx, frame_rx) = mpsc::channel::<Box<[i16; FRAME_SAMPLES]>>(FRAME_CHANNEL_DEPTH);
        let ListenCaptureSession {
            stop_flag,
            overrun: _,
            opened,
            handle: mut capture_handle,
        } = capture::start(self.mic_device.as_deref(), frame_tx);
        let capture_stop = CaptureStop(Arc::clone(&stop_flag));
        await_capture_open(opened, &mut capture_handle).await?;

        self.active.store(true, Ordering::SeqCst);
        self.phase_tx.send_replace(ListenPhase::Armed);
        let workers = self.spawn_workers(capture_handle, frame_rx);
        let pipeline = tokio::spawn(supervise_pipeline(
            workers,
            stop_flag,
            Arc::clone(&self.active),
            self.phase_tx.clone(),
        ));
        listen_state.session = Some(ListenSession {
            capture_stop,
            pipeline,
        });
        info!(target: "assistd::voice::listen", "wake-word listening started");
        Ok(())
    }

    async fn stop(&self) -> Result<(), ListenError> {
        let session = self.listen_state.lock().await.session.take();
        self.shut_down(session).await;
        Ok(())
    }

    async fn stop_for_push_to_talk(&self) -> Result<(), ListenError> {
        *self.last_push_to_talk.lock() = Some(Instant::now());
        let session = self.listen_state.lock().await.session.take();
        if let Some(session) = &session {
            session.capture_stop.request();
        }
        self.conversation_tx
            .send_replace(Some(ConversationSignal::Abandon));
        self.shut_down(session).await;
        Ok(())
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

    fn open_follow_up(&self) {
        self.conversation_tx
            .send_replace(Some(ConversationSignal::FollowUp(Instant::now())));
    }

    fn set_replying(&self, replying: bool) {
        self.replying_tx.send_replace(replying);
    }

    async fn capture_utterances(
        &self,
        sink: mpsc::Sender<SessionAudio>,
    ) -> Result<(), ListenError> {
        let _claim = {
            let listen_state = self.listen_state.lock().await;
            if listen_state.is_running() {
                return Err(ListenError::Listening);
            }
            if self.capturing_utterances.swap(true, Ordering::SeqCst) {
                return Err(ListenError::Testing);
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
    utterances: mpsc::Receiver<ListenAudio>,
    signals: watch::Receiver<Option<ConversationSignal>>,
}

impl TranscribeInputs {
    /// Whether push-to-talk took the mic since the last signal was applied.
    fn abandoned(&self) -> bool {
        self.signals.has_changed().unwrap_or(false)
            && matches!(*self.signals.borrow(), Some(ConversationSignal::Abandon))
    }

    /// The next audio, or `None` once its sender is gone or push-to-talk
    /// took the mic. Signals, applied before any queued audio is returned,
    /// and a closing command window update `gate` while waiting.
    async fn next_audio(
        &mut self,
        gate: &mut WakeGate,
        outputs: &ListenOutputs,
    ) -> Option<ListenAudio> {
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

/// Barge-in progress through the utterance being heard over a reply.
#[derive(Debug)]
struct BargeIn {
    speech: Arc<VoiceOutputController>,
    confirmed: bool,
}

impl BargeIn {
    fn new(speech: Arc<VoiceOutputController>) -> Self {
        Self {
            speech,
            confirmed: false,
        }
    }

    /// Whether `audio` is a check for an utterance already confirmed.
    fn already_confirmed(&self, audio: &ListenAudio) -> bool {
        self.confirmed && matches!(audio, ListenAudio::BargeInCheck(_))
    }

    fn check(&mut self, transcript: &str, outputs: &ListenOutputs) {
        self.confirmed = self.announce_if_confirmed(transcript, outputs);
    }

    /// Settle an utterance heard over a reply: a confirmed barge-in becomes
    /// the next command, anything else is dropped.
    fn finish_utterance(&mut self, transcript: Option<String>, outputs: &ListenOutputs) {
        let confirmed = std::mem::take(&mut self.confirmed);
        if let Some(transcript) = transcript
            && (confirmed || self.announce_if_confirmed(&transcript, outputs))
        {
            outputs.send_heard(Heard::Command(transcript));
        }
    }

    /// Publish [`Heard::BargeIn`] when `transcript` is the user talking
    /// rather than echo of the reply; true if so.
    fn announce_if_confirmed(&self, transcript: &str, outputs: &ListenOutputs) -> bool {
        let confirmed = confirms_barge_in(transcript, &self.speech.recent_speech());
        if confirmed {
            debug!(target: "assistd::voice::listen", %transcript, "barge-in confirmed");
            outputs.send_heard(Heard::BargeIn);
        } else {
            trace!(target: "assistd::voice::listen", %transcript, "not a barge-in");
        }
        confirmed
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

    fn set_phase(&self, phase: ListenPhase) {
        self.phase
            .send_if_modified(|current| std::mem::replace(current, phase) != phase);
    }
}

fn tuning_from_config(config: &WakeWordConfig) -> VadTuning {
    VadTuning::from_ms(config.silence_ms.get(), config.max_utterance_secs.get())
}

/// What to hand on after a frame produced `event`. `barge_in` is `None`
/// when speech over a reply is never checked for barge-in.
fn next_segment(
    barge_in: Option<&mut BargeInWatch>,
    event: Option<VadEvent>,
    vad: &UtteranceVad,
) -> Option<Segment> {
    match (event, barge_in) {
        (Some(event), watch) => Some(Segment::Utterance {
            event,
            during_reply: watch.is_some_and(BargeInWatch::finish_utterance),
        }),
        (None, Some(watch)) => watch.check_due(vad).map(Segment::BargeInCheck),
        (None, None) => None,
    }
}

/// Feeds Silero-labelled frames to the segmenter a stride at a time, blocking on CPU inference.
/// `emit` and `speaking_changed` must never wait, or the frame channel backs up and live audio
/// is silently truncated. The loop ends when `emit` returns false.
fn vad_loop(
    tuning: VadTuning,
    mut barge_in: Option<BargeInWatch>,
    mut classifier: FrameClassifier,
    mut frame_rx: mpsc::Receiver<Box<[i16; FRAME_SAMPLES]>>,
    mut emit: impl FnMut(Segment) -> bool,
    mut speaking_changed: impl FnMut(bool),
) {
    let mut vad = UtteranceVad::new(tuning);
    let mut speaking = false;
    while let Some(frame) = frame_rx.blocking_recv() {
        for (frame, is_speech) in classifier.push(&frame) {
            let event = vad.feed(&frame, is_speech);
            if vad.in_utterance() != speaking {
                speaking = !speaking;
                speaking_changed(speaking);
            }
            if let Some(segment) = next_segment(barge_in.as_mut(), event, &vad)
                && !emit(segment)
            {
                return;
            }
        }
    }
}

/// Queue listening audio, dropping it when the backlog is full; a barge-in
/// check is dropped silently, as another follows. False once the
/// transcription worker is gone.
fn queue_for_transcription(pcm_tx: &mpsc::Sender<ListenAudio>, segment: Segment) -> bool {
    match segment {
        Segment::Utterance {
            event: VadEvent::UtteranceComplete(pcm) | VadEvent::Truncated(pcm),
            during_reply,
        } => try_queue(
            pcm_tx,
            ListenAudio::Utterance(Utterance::ending_now(pcm, during_reply)),
        ),
        Segment::BargeInCheck(pcm) => !matches!(
            pcm_tx.try_send(ListenAudio::BargeInCheck(pcm)),
            Err(TrySendError::Closed(_))
        ),
    }
}

/// Hand a test session's utterance to it. False once it closed.
fn forward_captured(sink: &mpsc::Sender<SessionAudio>, segment: Segment) -> bool {
    let Segment::Utterance { event, .. } = segment else {
        return true;
    };
    let utterance = match event {
        VadEvent::UtteranceComplete(pcm) | VadEvent::Truncated(pcm) => CapturedUtterance { pcm },
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

/// What the transcription worker hears utterances with.
#[derive(Debug)]
struct Listeners {
    transcriber: Arc<dyn Transcriber>,
    spotter: Arc<WakeWordSpotter>,
}

/// Gates each utterance on the wake phrase, or on barge-in when heard over a reply, and
/// transcribes only what is addressed to assistd. A bare phrase or follow-up opens a command
/// window, reported as `Capturing` until it closes.
async fn transcribe_loop(
    listeners: Listeners,
    mut gate: WakeGate,
    outputs: ListenOutputs,
    mut inputs: TranscribeInputs,
    mut barge_in: BargeIn,
) {
    while let Some(audio) = inputs.next_audio(&mut gate, &outputs).await {
        let carry_on = match audio {
            ListenAudio::Utterance(utterance) if !utterance.during_reply => {
                let command = admit_utterance(&listeners, &mut gate, utterance, &outputs).await;
                let abandoned = command.is_some() && inputs.abandoned();
                if let Some(command) = command.filter(|_| !abandoned) {
                    outputs.send_heard(Heard::Command(command));
                }
                !abandoned
            }
            audio => {
                let transcriber = listeners.transcriber.as_ref();
                settle_over_reply(transcriber, audio, &mut barge_in, &outputs, &inputs).await
            }
        };
        if !carry_on {
            return;
        }
        outputs.set_phase(gate.phase(Instant::now()));
    }
}

/// Check speech heard over a reply for barge-in; never wake-gated. False
/// once push-to-talk took the mic.
async fn settle_over_reply(
    transcriber: &dyn Transcriber,
    audio: ListenAudio,
    barge_in: &mut BargeIn,
    outputs: &ListenOutputs,
    inputs: &TranscribeInputs,
) -> bool {
    if barge_in.already_confirmed(&audio) {
        return true;
    }
    let transcript = transcribe_pcm(transcriber, audio.pcm()).await;
    if transcript.is_some() && inputs.abandoned() {
        return false;
    }
    match audio {
        ListenAudio::BargeInCheck(_) => {
            if let Some(transcript) = &transcript {
                barge_in.check(transcript, outputs);
            }
        }
        ListenAudio::Utterance(_) => barge_in.finish_utterance(transcript, outputs),
    }
    true
}

/// Spot the wake phrase in `utterance` and gate it; the command is the
/// speech after the phrase, or a whole utterance inside a command window.
/// A phrase with nothing after it opens the command window instead.
async fn admit_utterance(
    listeners: &Listeners,
    gate: &mut WakeGate,
    utterance: Utterance,
    outputs: &ListenOutputs,
) -> Option<String> {
    let (pcm, phrase_end) = spot(&listeners.spotter, utterance.pcm).await;
    let command_start = match gate.admit(phrase_end.is_some(), utterance.started) {
        Admission::Dropped => {
            trace!(target: "assistd::voice::listen", "no wake phrase; dropped untranscribed");
            return None;
        }
        Admission::WakePhrase => {
            outputs.send_heard(Heard::WakePhrase);
            phrase_end.unwrap_or(0)
        }
        Admission::Command => 0,
    };
    let command_pcm = &pcm[command_start..];
    let command = if command_pcm.len() >= MIN_COMMAND_SAMPLES {
        transcribe_pcm(listeners.transcriber.as_ref(), command_pcm).await
    } else {
        None
    };
    if command.is_none() && phrase_end.is_some() {
        gate.await_command(utterance.started);
    }
    command
}

/// Run the spotter off the runtime; returns `pcm` with where the phrase
/// ends in it, if it is there.
async fn spot(spotter: &Arc<WakeWordSpotter>, pcm: Vec<i16>) -> (Vec<i16>, Option<usize>) {
    let spotter = Arc::clone(spotter);
    tokio::task::spawn_blocking(move || {
        let phrase_end = spotter.find(&pcm);
        (pcm, phrase_end)
    })
    .await
    .unwrap_or_else(|err| {
        warn!(target: "assistd::voice::listen", "wake-phrase spotting failed: {err}");
        (Vec::new(), None)
    })
}

async fn transcribe_pcm(transcriber: &dyn Transcriber, pcm: &[i16]) -> Option<String> {
    match transcriber.transcribe(pcm).await {
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

/// Wait until capture is recording; its error if the device failed to open.
async fn await_capture_open(
    opened: oneshot::Receiver<()>,
    capture: &mut CaptureJoin,
) -> Result<(), ListenError> {
    if opened.await.is_ok() {
        return Ok(());
    }
    capture.await??;
    Ok(())
}

/// Run a session's workers to the end. Capture ending without a stop
/// request turns listening off and reports why.
async fn supervise_pipeline(
    workers: ListenWorkers,
    stop_requested: Arc<AtomicBool>,
    active: Arc<AtomicBool>,
    phase: watch::Sender<ListenPhase>,
) {
    let captured = workers.join().await;
    if stop_requested.load(Ordering::SeqCst) {
        if let Err(err) = captured {
            warn!(target: "assistd::voice::listen", "capture worker error: {err}");
        }
        return;
    }
    active.store(false, Ordering::SeqCst);
    phase.send_replace(ListenPhase::Off);
    match captured {
        Ok(()) => error!(
            target: "assistd::voice::listen",
            "wake-word capture ended unexpectedly; listening is off"
        ),
        Err(err) => error!(
            target: "assistd::voice::listen",
            "wake-word capture failed; listening is off: {err}"
        ),
    }
}

#[cfg(test)]
mod tests;
