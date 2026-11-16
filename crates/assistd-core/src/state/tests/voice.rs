use super::*;

fn state_with_voice(voice: Arc<MockVoice>) -> Arc<AppState> {
    StateParts {
        voice,
        ..StateParts::default()
    }
    .build()
}

#[tokio::test]
async fn dispatch_ptt_stop_with_text_runs_query() {
    let state = state_with_voice(MockVoice::new(Ok(()), Ok("hello world".into())));
    let (res, events) = dispatch(&state, Request::new("p3", RequestBody::PttStop)).await;
    res.unwrap();
    assert_eq!(
        events,
        [
            voice_state("p3", VoiceCaptureState::Transcribing),
            voice_state("p3", VoiceCaptureState::Idle),
            Event::new(
                "p3",
                EventBody::Transcription {
                    text: "hello world".into()
                }
            ),
            Event::new(
                "p3",
                EventBody::Delta {
                    text: "hello world".into()
                }
            ),
            done("p3"),
        ]
    );
}

#[tokio::test]
async fn interrupt_between_ptt_press_and_release_drops_the_prompt() {
    let state = state_with_voice(MockVoice::new(Ok(()), Ok("hello world".into())));
    let (res, _) = dispatch(&state, Request::new("p", RequestBody::PttStart)).await;
    res.unwrap();
    let (res, _) = dispatch(&state, Request::new("int", RequestBody::InterruptTurn)).await;
    res.unwrap();

    let (res, events) = dispatch(&state, Request::new("p", RequestBody::PttStop)).await;
    res.unwrap();
    assert_eq!(
        events,
        [
            voice_state("p", VoiceCaptureState::Transcribing),
            voice_state("p", VoiceCaptureState::Idle),
            done("p"),
        ]
    );
}

/// `VoiceInput` whose mic open blocks until `open` is notified, recording
/// the order calls complete in.
#[derive(Debug, Default)]
struct GatedVoice {
    open: Notify,
    calls: StdMutex<Vec<&'static str>>,
}

#[async_trait::async_trait]
impl VoiceInput for GatedVoice {
    async fn start_recording(&self) -> Result<(), VoiceInputError> {
        self.open.notified().await;
        self.calls.lock().push("start");
        Ok(())
    }
    async fn stop_and_transcribe(&self) -> Result<String, VoiceInputError> {
        self.calls.lock().push("stop");
        Ok(String::new())
    }
    fn state(&self) -> VoiceCaptureState {
        VoiceCaptureState::Idle
    }
    fn subscribe(&self) -> watch::Receiver<VoiceCaptureState> {
        watch::channel(VoiceCaptureState::Idle).1
    }
}

#[tokio::test]
async fn ptt_stop_waits_for_a_press_still_opening_the_mic() {
    let voice = Arc::new(GatedVoice::default());
    let state = StateParts {
        voice: voice.clone(),
        ..StateParts::default()
    }
    .build();
    let press = tokio::spawn({
        let state = state.clone();
        async move { dispatch(&state, Request::new("p", RequestBody::PttStart)).await }
    });
    tokio::task::yield_now().await;
    let release = tokio::spawn({
        let state = state.clone();
        async move { dispatch(&state, Request::new("s", RequestBody::PttStop)).await }
    });
    tokio::task::yield_now().await;
    voice.open.notify_one();

    press.await.unwrap().0.unwrap();
    release.await.unwrap().0.unwrap();
    assert_eq!(*voice.calls.lock(), ["start", "stop"]);
}

/// `ContinuousListener` that tracks whether it is listening.
#[derive(Debug)]
struct MockListener {
    active: AtomicBool,
    phase_tx: watch::Sender<ListenPhase>,
    heard: tokio::sync::broadcast::Sender<Heard>,
    last_push_to_talk: StdMutex<Option<tokio::time::Instant>>,
}

impl MockListener {
    fn new(active: bool) -> Arc<Self> {
        let phase = if active {
            ListenPhase::Armed
        } else {
            ListenPhase::Off
        };
        let (phase_tx, _) = watch::channel(phase);
        let (heard, _) = tokio::sync::broadcast::channel(4);
        Arc::new(Self {
            active: AtomicBool::new(active),
            phase_tx,
            heard,
            last_push_to_talk: StdMutex::new(None),
        })
    }
}

#[async_trait::async_trait]
impl ContinuousListener for MockListener {
    async fn start(&self) -> Result<(), ListenError> {
        self.active.store(true, Ordering::SeqCst);
        self.phase_tx.send_replace(ListenPhase::Armed);
        Ok(())
    }
    async fn stop(&self) -> Result<(), ListenError> {
        self.active.store(false, Ordering::SeqCst);
        self.phase_tx.send_replace(ListenPhase::Off);
        Ok(())
    }
    async fn stop_for_push_to_talk(&self) -> Result<(), ListenError> {
        *self.last_push_to_talk.lock() = Some(tokio::time::Instant::now());
        self.stop().await
    }
    fn last_push_to_talk(&self) -> Option<tokio::time::Instant> {
        *self.last_push_to_talk.lock()
    }
    fn is_active(&self) -> bool {
        self.active.load(Ordering::SeqCst)
    }
    fn subscribe_heard(&self) -> tokio::sync::broadcast::Receiver<Heard> {
        self.heard.subscribe()
    }
    fn subscribe_phase(&self) -> watch::Receiver<ListenPhase> {
        self.phase_tx.subscribe()
    }
    fn open_follow_up(&self) {}
    fn set_replying(&self, _replying: bool) {}
    async fn test_wake_word(&self, sink: mpsc::Sender<WakeTest>) -> Result<(), ListenError> {
        assert!(!self.is_active(), "listening must pause while capturing");
        sink.closed().await;
        Ok(())
    }
}

#[tokio::test]
async fn listen_requests_drive_the_listener() {
    let cases = [
        (
            false,
            Request::new("l", RequestBody::ListenToggle),
            vec![listen_state("l", ListenPhase::Armed), done("l")],
            true,
        ),
        (
            true,
            Request::new("l", RequestBody::ListenToggle),
            vec![listen_state("l", ListenPhase::Off), done("l")],
            false,
        ),
    ];
    for (initially_active, req, expected, active_after) in cases {
        let label = format!("{} from active={initially_active}", req.kind());
        let listener = MockListener::new(initially_active);
        let state = StateParts {
            listener: listener.clone(),
            ..StateParts::default()
        }
        .build();
        let (res, events) = dispatch(&state, req).await;
        res.unwrap_or_else(|e| panic!("{label}: {e:#}"));
        assert_eq!(events, expected, "{label}");
        assert_eq!(listener.is_active(), active_after, "{label}");
    }
}

#[tokio::test]
async fn ptt_takes_the_mic_from_listening_and_returns_it_on_release() {
    let cases = [
        (None, true),
        (Some(Request::new("t", RequestBody::ListenToggle)), false),
    ];
    for (while_held, listening_after_release) in cases {
        let label = format!("{:?} while held", while_held.as_ref().map(Request::kind));
        let listener = MockListener::new(true);
        let state = StateParts {
            voice: MockVoice::new(Ok(()), Ok(String::new())),
            listener: listener.clone(),
            ..StateParts::default()
        }
        .build();

        let (res, events) = dispatch(&state, Request::new("p", RequestBody::PttStart)).await;
        res.unwrap_or_else(|e| panic!("{label}: {e:#}"));
        assert_eq!(
            events,
            [voice_state("p", VoiceCaptureState::Recording), done("p")],
            "{label}"
        );
        assert!(!listener.is_active(), "{label}: listening yields the mic");
        assert!(listener.last_push_to_talk().is_some(), "{label}");

        if let Some(request) = while_held {
            let (res, _) = dispatch(&state, request).await;
            res.unwrap_or_else(|e| panic!("{label}: {e:#}"));
        }
        let (res, _) = dispatch(&state, Request::new("s", RequestBody::PttStop)).await;
        res.unwrap_or_else(|e| panic!("{label}: {e:#}"));
        assert_eq!(listener.is_active(), listening_after_release, "{label}");
    }
}

#[tokio::test]
async fn dispatch_query_speaks_sentences_in_order_and_drains() {
    let recorder = MockSpeechRecorder::new();
    let state = StateParts {
        speech: recorder.clone(),
        ..StateParts::default()
    }
    .build();
    recorder.watch_speaking(state.subsystems.voice.speech());
    let (res, _) = dispatch(&state, query("ord", "First. Second. Third. End.")).await;
    res.unwrap();

    assert_eq!(recorder.calls(), ["First.", "Second.", "Third.", "End."]);
    assert_eq!(
        recorder.wait_idle_calls.load(Ordering::SeqCst),
        1,
        "speech worker must drain before the query returns"
    );
    assert_eq!(
        recorder.unguarded_speaks.load(Ordering::SeqCst),
        0,
        "every sentence is spoken with the speaking signal raised"
    );
    assert!(
        !state.subsystems.voice.speech().is_speaking(),
        "speaking signal drops once playback drains"
    );
}

/// On its first `step`, emits a scripted sequence of deltas with optional
/// pauses between them; always answers `Final`.
#[derive(Debug)]
struct StreamingDeltaBackend {
    script: StdMutex<Option<Vec<DeltaScript>>>,
}

#[derive(Debug)]
enum DeltaScript {
    Text(&'static str),
    Sleep(Duration),
}

impl StreamingDeltaBackend {
    fn new(script: Vec<DeltaScript>) -> Arc<Self> {
        Arc::new(Self {
            script: StdMutex::new(Some(script)),
        })
    }
}

#[async_trait::async_trait]
impl TurnBackend for StreamingDeltaBackend {
    async fn push_user(
        &self,
        _text: String,
        _attachments: Vec<assistd_ipc::Attachment>,
    ) -> assistd_llm::LlmResult<()> {
        Ok(())
    }

    async fn push_tool_results(
        &self,
        _results: Vec<ToolResultPayload>,
    ) -> assistd_llm::LlmResult<()> {
        Ok(())
    }

    async fn step(
        &self,
        _tools: Vec<serde_json::Value>,
        tx: mpsc::Sender<LlmEvent>,
    ) -> assistd_llm::LlmResult<StepOutcome> {
        let script = self.script.lock().take();
        for action in script.into_iter().flatten() {
            match action {
                DeltaScript::Text(s) => {
                    tx.send(LlmEvent::Delta { text: s.into() }).await.ok();
                }
                DeltaScript::Sleep(d) => tokio::time::sleep(d).await,
            }
        }
        Ok(StepOutcome::Final)
    }

    async fn push_interrupted_reply(&self, _reply: String) -> assistd_llm::LlmResult<()> {
        Ok(())
    }

    async fn set_transient_note(&self, _text: String) -> assistd_llm::LlmResult<()> {
        Ok(())
    }
}

impl LlmBackend for StreamingDeltaBackend {}

#[tokio::test(start_paused = true)]
async fn partial_flush_speaks_a_stalled_fragment_only_when_enabled() {
    let cases: [(u32, &[&str]); 2] = [
        (50, &["Half a", "sentence.", "End."]),
        (0, &["Half a sentence.", "End."]),
    ];
    for (flush_ms, expected) in cases {
        let recorder = MockSpeechRecorder::new();
        let state = StateParts {
            config: config_with_partial_flush(flush_ms),
            backend: StreamingDeltaBackend::new(vec![
                DeltaScript::Text("Half a sente"),
                DeltaScript::Sleep(Duration::from_millis(150)),
                DeltaScript::Text("nce. End."),
            ]),
            speech: recorder.clone(),
            ..StateParts::default()
        }
        .build();
        let (res, _) = dispatch(&state, query("pf", "go")).await;
        res.unwrap();
        assert_eq!(recorder.calls(), expected, "partial_flush_ms={flush_ms}");
    }
}
