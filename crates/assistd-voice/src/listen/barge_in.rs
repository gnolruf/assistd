//! Barge-in: telling the user talking over a reply to a wake-word command
//! apart from bumps, background noise and the reply's own echo.

use std::collections::HashSet;

use tokio::sync::watch;

use super::vad::{SAMPLE_RATE_HZ, UtteranceVad};
use super::wake_phrase::normalized_words;

/// Voiced speech before an utterance is first checked: 500 ms of 20 ms
/// frames, longer than any bump or click.
const FIRST_CHECK_VOICED_FRAMES: u32 = 25;

/// Further voiced speech between checks while they keep failing: 1 s.
const RECHECK_VOICED_FRAMES: u32 = 50;

/// Audio handed to each check: the utterance's last 2.5 s.
const CHECK_WINDOW_SAMPLES: usize = SAMPLE_RATE_HZ as usize * 5 / 2;

/// Words a transcript needs that the reply did not just say.
const MIN_NOVEL_WORDS: usize = 2;

/// Share of a transcript's words that the reply did not just say; echo
/// that Whisper mishears yields a few novel words among many spoken ones.
const MIN_NOVEL_SHARE: f32 = 0.5;

/// Follows the VAD on the capture thread and says when speech that began
/// during a reply has gone on long enough to check.
#[derive(Debug)]
pub struct BargeInWatch {
    replying: watch::Receiver<bool>,
    utterance: Option<WatchedUtterance>,
}

/// The utterance being recorded; `next_check` is `None` when it began
/// outside a reply and is never checked.
#[derive(Debug, Clone, Copy)]
struct WatchedUtterance {
    next_check: Option<u32>,
}

impl BargeInWatch {
    /// Watch for barge-in while `replying` is true.
    pub fn new(replying: watch::Receiver<bool>) -> Self {
        Self {
            replying,
            utterance: None,
        }
    }

    pub fn replying(&self) -> bool {
        *self.replying.borrow()
    }

    /// After a frame that ended no utterance: the audio to check for
    /// barge-in, when one is due.
    pub fn check_due(&mut self, vad: &UtteranceVad) -> Option<Vec<i16>> {
        if !vad.in_utterance() {
            self.utterance = None;
            return None;
        }
        let replying = self.replying();
        let watched = self.utterance.get_or_insert(WatchedUtterance {
            next_check: replying.then_some(FIRST_CHECK_VOICED_FRAMES),
        });
        let due = watched.next_check?;
        if vad.voiced_frames() < due {
            return None;
        }
        watched.next_check = Some(due + RECHECK_VOICED_FRAMES);
        Some(vad.utterance_tail(CHECK_WINDOW_SAMPLES).to_vec())
    }

    /// After a frame that ended an utterance: whether it began during a reply.
    pub fn finish_utterance(&mut self) -> bool {
        self.utterance
            .take()
            .is_some_and(|watched| watched.next_check.is_some())
    }
}

/// Whether `transcript` is the user talking rather than echo of
/// `recent_speech`, the sentences the reply just spoke.
pub fn confirms_barge_in(transcript: &str, recent_speech: &[String]) -> bool {
    let spoken: HashSet<String> = recent_speech
        .iter()
        .flat_map(|sentence| normalized_words(sentence))
        .collect();
    let heard = normalized_words(transcript);
    let novel = heard.iter().filter(|word| !spoken.contains(*word)).count();
    novel >= MIN_NOVEL_WORDS && novel as f32 >= heard.len() as f32 * MIN_NOVEL_SHARE
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_mostly_new_speech_of_two_words_or_more_confirms() {
        let reply = [
            "The forecast for tomorrow is sunny.".to_string(),
            "Highs will reach twenty degrees.".to_string(),
        ];
        let cases = [
            ("no stop", true),
            ("stop", false),
            ("highs will reach twenty three degrees", false),
            ("the forecast for tomorrow is funny and highs", false),
            ("actually what about tomorrow", true),
        ];
        for (transcript, confirms) in cases {
            assert_eq!(
                confirms_barge_in(transcript, &reply),
                confirms,
                "{transcript:?}"
            );
        }
    }
}
