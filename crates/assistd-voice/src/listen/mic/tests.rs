use std::sync::atomic::AtomicUsize;

use assistd_config::UnitInterval;

use super::*;
use crate::transcribe::TranscriptionError;

/// `(delay ms, transcript)` per utterance, indexed by the utterance's first sample.
type Script = [(u64, &'static str)];

const SCRIPT: [(u64, &str); 6] = [
    (300, "Hey computer, what time is it?"),
    (10, "my computer is slow"),
    (200, "hey computer"),
    (0, ""),
    (10, "open firefox"),
    (10, "open firefox"),
];

/// Plays a script, recording the peak number of concurrent calls.
#[derive(Debug)]
struct Scripted {
    script: &'static Script,
    in_flight: AtomicUsize,
    peak: AtomicUsize,
}

impl Scripted {
    fn new(script: &'static Script) -> Arc<Self> {
        Arc::new(Self {
            script,
            in_flight: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
        })
    }
}

#[async_trait]
impl Transcriber for Scripted {
    async fn transcribe(
        &self,
        pcm: &[i16],
        prompt: Option<&str>,
    ) -> Result<String, TranscriptionError> {
        assert_eq!(prompt, Some("hey computer"));
        let now = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
        self.peak.fetch_max(now, Ordering::SeqCst);
        let (delay_ms, text) = self.script[usize::try_from(pcm[0]).unwrap()];
        tokio::time::sleep(Duration::from_millis(delay_ms)).await;
        self.in_flight.fetch_sub(1, Ordering::SeqCst);
        Ok(text.to_string())
    }
}

#[tokio::test(start_paused = true)]
async fn only_speech_after_the_wake_phrase_is_heard_in_order() {
    let transcriber = Scripted::new(&SCRIPT);
    let (heard, mut rx) = broadcast::channel(UTTERANCE_CHANNEL_DEPTH);
    let (phase, _) = watch::channel(ListenPhase::Armed);
    let (pcm_tx, pcm_rx) = mpsc::channel(SCRIPT.len());
    let gate = WakeGate::new(
        WakePhrase::builtin(UnitInterval::new(0.8).unwrap()),
        Duration::from_secs(4),
        None,
    );

    for index in 0..SCRIPT.len() {
        let pcm = vec![i16::try_from(index).unwrap()];
        let utterance = Utterance::ending_now(pcm, false);
        pcm_tx
            .send(ListenAudio::Utterance(utterance))
            .await
            .unwrap();
    }
    drop(pcm_tx);

    transcribe_loop(
        transcriber.clone(),
        gate,
        ListenOutputs { heard, phase },
        TranscribeInputs {
            utterances: pcm_rx,
            signals: watch::channel(None).1,
        },
        BargeIn::new(VoiceOutputController::new(true)),
    )
    .await;

    let received: Vec<Heard> = std::iter::from_fn(|| rx.try_recv().ok()).collect();
    assert_eq!(
        received,
        [
            Heard::WakePhrase,
            Heard::Command("what time is it?".into()),
            Heard::WakePhrase,
            Heard::Command("open firefox".into()),
        ]
    );
    assert_eq!(transcriber.peak.load(Ordering::SeqCst), 1);
}

const BARGE_IN_SCRIPT: [(u64, &str); 7] = [
    (10, "the forecast for tomorrow"),
    (10, "wait, what about Saturday"),
    (10, "never transcribed once confirmed"),
    (10, "sunny. Wait, what about Saturday?"),
    (10, "The forecast for tomorrow is sunny."),
    (10, "hey computer stop talking"),
    (10, "open firefox"),
];

#[tokio::test(start_paused = true)]
async fn speech_over_a_reply_is_a_barge_in_only_when_it_is_not_echo() {
    let speech = VoiceOutputController::new(true);
    speech.remember_spoken("The forecast for tomorrow is sunny.");
    let (heard, mut rx) = broadcast::channel(UTTERANCE_CHANNEL_DEPTH);
    let (phase, _) = watch::channel(ListenPhase::Armed);
    let (pcm_tx, pcm_rx) = mpsc::channel(BARGE_IN_SCRIPT.len());
    let gate = WakeGate::new(
        WakePhrase::builtin(UnitInterval::new(0.8).unwrap()),
        Duration::from_secs(4),
        None,
    );

    for index in 0..BARGE_IN_SCRIPT.len() {
        let pcm = vec![i16::try_from(index).unwrap()];
        let audio = match index {
            0..=2 => ListenAudio::BargeInCheck(pcm),
            3..=5 => ListenAudio::Utterance(Utterance::ending_now(pcm, true)),
            _ => ListenAudio::Utterance(Utterance::ending_now(pcm, false)),
        };
        pcm_tx.send(audio).await.unwrap();
    }
    drop(pcm_tx);

    transcribe_loop(
        Scripted::new(&BARGE_IN_SCRIPT),
        gate,
        ListenOutputs { heard, phase },
        TranscribeInputs {
            utterances: pcm_rx,
            signals: watch::channel(None).1,
        },
        BargeIn::new(speech),
    )
    .await;

    let received: Vec<Heard> = std::iter::from_fn(|| rx.try_recv().ok()).collect();
    assert_eq!(
        received,
        [
            Heard::BargeIn,
            Heard::Command("sunny. Wait, what about Saturday?".into()),
            Heard::BargeIn,
            Heard::Command("hey computer stop talking".into()),
        ]
    );
}

#[test]
fn speech_begun_over_a_reply_is_settled_as_one_after_the_reply_ends() {
    let (replying_tx, replying_rx) = watch::channel(true);
    let mut barge_in = Some(BargeInWatch::new(replying_rx));
    let mut vad = UtteranceVad::new(VadTuning::from_ms(800, 30));
    let frame = [1000i16; FRAME_SAMPLES];
    for _ in 0..30 {
        let event = vad.feed(&frame, true);
        next_segment(barge_in.as_mut(), event, &vad);
    }

    replying_tx.send_replace(false);
    let during_reply = std::iter::repeat_with(|| {
        let event = vad.feed(&frame, false);
        next_segment(barge_in.as_mut(), event, &vad)
    })
    .take(100)
    .find_map(|segment| match segment {
        Some(Segment::Utterance { during_reply, .. }) => Some(during_reply),
        _ => None,
    });
    assert_eq!(
        during_reply,
        Some(true),
        "the user's utterance stays a barge-in candidate even though the reply ended first"
    );
}
