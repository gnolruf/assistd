use std::sync::atomic::AtomicUsize;
use std::time::Duration;

use super::*;
use crate::transcribe::TranscriptionError;

/// Echoes the first sample as text after sleeping for that many ms,
/// recording the peak number of concurrent calls.
#[derive(Default)]
struct SlowEcho {
    in_flight: AtomicUsize,
    peak: AtomicUsize,
}

#[async_trait]
impl Transcriber for SlowEcho {
    async fn transcribe(&self, pcm: &[i16]) -> Result<String, TranscriptionError> {
        let now = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
        self.peak.fetch_max(now, Ordering::SeqCst);
        let ms = pcm.first().copied().unwrap_or_default();
        tokio::time::sleep(Duration::from_millis(ms.unsigned_abs().into())).await;
        self.in_flight.fetch_sub(1, Ordering::SeqCst);
        Ok(if ms == 0 {
            String::new()
        } else {
            format!(" {ms} ")
        })
    }
}

#[tokio::test(start_paused = true)]
async fn transcripts_are_serial_ordered_and_drained() {
    let transcriber = Arc::new(SlowEcho::default());
    let (utterances, mut rx) = broadcast::channel(UTTERANCE_CHANNEL_DEPTH);
    let (pcm_tx, pcm_rx) = mpsc::channel(PENDING_UTTERANCE_DEPTH);

    for ms in [300, 0, 10, 200] {
        pcm_tx.send(vec![ms]).await.unwrap();
    }
    drop(pcm_tx);

    transcribe_loop(transcriber.clone(), utterances, pcm_rx).await;

    let received: Vec<String> = std::iter::from_fn(|| rx.try_recv().ok()).collect();
    assert_eq!(received, ["300", "10", "200"]);
    assert_eq!(transcriber.peak.load(Ordering::SeqCst), 1);
}
