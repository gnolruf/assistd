use super::*;
use crate::transcribe::TranscriptionError;

/// `(delay ms, transcript)` per utterance, indexed by the utterance's first sample.
type Script = [(u64, &'static str)];

/// Plays a script.
#[derive(Debug)]
struct Scripted {
    script: &'static Script,
}

#[async_trait]
impl Transcriber for Scripted {
    async fn transcribe(
        &self,
        pcm: &[i16],
        prompt: Option<&str>,
    ) -> Result<String, TranscriptionError> {
        assert_eq!(prompt, None);
        let (delay_ms, text) = self.script[usize::try_from(pcm[0]).unwrap()];
        tokio::time::sleep(Duration::from_millis(delay_ms)).await;
        Ok(text.to_string())
    }
}

const BARGE_IN_SCRIPT: [(u64, &str); 6] = [
    (10, "the forecast for tomorrow"),
    (10, "wait, what about Saturday"),
    (10, "never transcribed once confirmed"),
    (10, "sunny. Wait, what about Saturday?"),
    (10, "The forecast for tomorrow is sunny."),
    (10, "hey computer stop talking"),
];

#[tokio::test(start_paused = true)]
async fn speech_over_a_reply_is_a_barge_in_only_when_it_is_not_echo() {
    let speech = VoiceOutputController::new(true);
    speech.remember_spoken("The forecast for tomorrow is sunny.");
    let (heard, mut rx) = broadcast::channel(UTTERANCE_CHANNEL_DEPTH);
    let (phase, _) = watch::channel(ListenPhase::Armed);
    let outputs = ListenOutputs { heard, phase };
    let inputs = TranscribeInputs {
        utterances: mpsc::channel(1).1,
        signals: watch::channel(None).1,
    };
    let transcriber = Scripted {
        script: &BARGE_IN_SCRIPT,
    };
    let mut barge_in = BargeIn::new(speech);

    for index in 0..BARGE_IN_SCRIPT.len() {
        let pcm = vec![i16::try_from(index).unwrap()];
        let audio = match index {
            0..=2 => ListenAudio::BargeInCheck(pcm),
            _ => ListenAudio::Utterance(Utterance::ending_now(pcm, true)),
        };
        assert!(settle_over_reply(&transcriber, audio, &mut barge_in, &outputs, &inputs).await);
    }

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
