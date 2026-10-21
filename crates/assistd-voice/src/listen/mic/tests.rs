use std::sync::atomic::AtomicUsize;

use assistd_config::UnitInterval;

use super::*;
use crate::transcribe::TranscriptionError;

/// `(delay ms, transcript)` per utterance, indexed by the utterance's first sample.
const SCRIPT: [(u64, &str); 6] = [
    (300, "Hey computer, what time is it?"),
    (10, "my computer is slow"),
    (200, "hey computer"),
    (0, ""),
    (10, "open firefox"),
    (10, "open firefox"),
];

/// Plays [`SCRIPT`], recording the peak number of concurrent calls.
#[derive(Debug, Default)]
struct Scripted {
    in_flight: AtomicUsize,
    peak: AtomicUsize,
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
        let (delay_ms, text) = SCRIPT[usize::try_from(pcm[0]).unwrap()];
        tokio::time::sleep(Duration::from_millis(delay_ms)).await;
        self.in_flight.fetch_sub(1, Ordering::SeqCst);
        Ok(text.to_string())
    }
}

#[tokio::test(start_paused = true)]
async fn only_speech_after_the_wake_phrase_is_heard_in_order() {
    let transcriber = Arc::new(Scripted::default());
    let (heard, mut rx) = broadcast::channel(UTTERANCE_CHANNEL_DEPTH);
    let (phase, _) = watch::channel(ListenPhase::Armed);
    let (pcm_tx, pcm_rx) = mpsc::channel(SCRIPT.len());
    let gate = WakeGate::new(
        WakePhrase::builtin(UnitInterval::new(0.8).unwrap()),
        Duration::from_secs(4),
    );

    for index in 0..SCRIPT.len() {
        let pcm = vec![i16::try_from(index).unwrap()];
        let started = Instant::now();
        pcm_tx.send(Utterance { pcm, started }).await.unwrap();
    }
    drop(pcm_tx);

    transcribe_loop(
        transcriber.clone(),
        gate,
        ListenOutputs { heard, phase },
        pcm_rx,
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
