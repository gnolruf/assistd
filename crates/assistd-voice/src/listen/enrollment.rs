//! Wake-phrase enrollment: screens each recording, then learns the distinct
//! spellings Whisper produces for the phrase as accepted variants.

use assistd_ipc::{EnrollEvent, SampleRejection};

use super::vad::SAMPLE_RATE_HZ;
use super::wake_phrase::{normalized_words, similarity};

/// Accepted samples required before an enrollment can be saved.
pub const MIN_SAMPLES: u8 = 3;

/// Trailing samples that must all repeat a known spelling for the
/// enrollment to count as consistent.
const CONSISTENT_RUN: usize = 3;

const MIN_SPEECH_MS: usize = 250;
const MAX_SPEECH_MS: usize = 4_000;

/// Samples at or above this magnitude count as clipped.
const CLIP_LEVEL: u16 = 32_000;

/// Clipped samples per thousand above which a recording is refused.
const MAX_CLIPPED_PER_THOUSAND: usize = 1;

/// Speech spans from the first to the last sample within this factor of the peak.
const SPEECH_PEAK_DIVISOR: u16 = 10;

/// Below this whole-phrase similarity a transcript is something else said,
/// not a mis-hearing of the phrase.
const MIN_PHRASE_SIMILARITY: f32 = 0.5;

/// One VAD-cut utterance from the listener's capture stack.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapturedUtterance {
    /// 16 kHz mono samples.
    pub pcm: Vec<i16>,
    /// Cut at the configured maximum utterance length while speech went on.
    pub truncated: bool,
}

/// An accepted enrollment recording and what Whisper heard in it.
#[derive(Debug, Clone)]
pub(super) struct Recording {
    pub(super) pcm: Vec<i16>,
    pub(super) heard: String,
}

/// An enrollment in progress: the phrase, the spellings learned so far
/// (the phrase's own first), and the accepted recordings.
#[derive(Debug)]
pub struct Enrollment {
    phrase: String,
    spellings: Vec<String>,
    recordings: Vec<Recording>,
    known_run: usize,
}

impl Enrollment {
    pub fn new(phrase: &str) -> Self {
        Self {
            phrase: phrase.to_string(),
            spellings: vec![spelling(phrase)],
            recordings: Vec::new(),
            known_run: 0,
        }
    }

    /// The phrase as the user typed it.
    pub fn phrase(&self) -> &str {
        &self.phrase
    }

    /// Refuse audio that cannot make a usable sample, before transcribing it.
    pub fn screen(utterance: &CapturedUtterance) -> Result<(), SampleRejection> {
        if utterance.truncated {
            return Err(SampleRejection::TooLong);
        }
        if is_clipped(&utterance.pcm) {
            return Err(SampleRejection::Clipped);
        }
        let speech_ms = speech_span(&utterance.pcm) * 1000 / SAMPLE_RATE_HZ as usize;
        if speech_ms < MIN_SPEECH_MS {
            Err(SampleRejection::TooShort)
        } else if speech_ms > MAX_SPEECH_MS {
            Err(SampleRejection::TooLong)
        } else {
            Ok(())
        }
    }

    /// Keep `pcm` as a sample of the phrase when `heard` plausibly is it,
    /// learning `heard` as a variant if it is a new spelling. Returns the
    /// sample's index, counting from 1.
    pub fn accept(&mut self, pcm: Vec<i16>, heard: &str) -> Result<u8, SampleRejection> {
        let heard_spelling = spelling(heard);
        if heard_spelling.is_empty()
            || similarity(&self.spellings[0], &heard_spelling) < MIN_PHRASE_SIMILARITY
        {
            return Err(SampleRejection::DidNotMatch);
        }
        if self.spellings.contains(&heard_spelling) {
            self.known_run += 1;
        } else {
            self.known_run = 0;
            self.spellings.push(heard_spelling);
        }
        self.recordings.push(Recording {
            pcm,
            heard: heard.to_string(),
        });
        Ok(self.accepted())
    }

    /// Whether enough samples were accepted to save.
    pub fn is_ready(&self) -> bool {
        self.accepted() >= MIN_SAMPLES
    }

    /// The prompt for the next sample, with progress so far.
    pub fn prompt(&self) -> EnrollEvent {
        EnrollEvent::Prompt {
            accepted: self.accepted(),
            minimum: MIN_SAMPLES,
            consistent: self.is_ready() && self.known_run >= CONSISTENT_RUN,
        }
    }

    /// The phrase, its variants, and the accepted recordings.
    pub(super) fn into_parts(self) -> (String, Vec<String>, Vec<Recording>) {
        let variants = self.spellings.into_iter().skip(1).collect();
        (self.phrase, variants, self.recordings)
    }

    fn accepted(&self) -> u8 {
        u8::try_from(self.recordings.len()).unwrap_or(u8::MAX)
    }
}

/// `text` as single-spaced normalized words.
fn spelling(text: &str) -> String {
    normalized_words(text).join(" ")
}

fn is_clipped(pcm: &[i16]) -> bool {
    let clipped = pcm
        .iter()
        .filter(|sample| sample.unsigned_abs() >= CLIP_LEVEL)
        .count();
    clipped * 1000 > pcm.len() * MAX_CLIPPED_PER_THOUSAND
}

/// Samples from the first to the last one loud enough to be speech.
fn speech_span(pcm: &[i16]) -> usize {
    let peak = pcm
        .iter()
        .map(|sample| sample.unsigned_abs())
        .max()
        .unwrap_or(0);
    if peak == 0 {
        return 0;
    }
    let floor = peak / SPEECH_PEAK_DIVISOR;
    let is_loud = |sample: &i16| sample.unsigned_abs() >= floor;
    match (pcm.iter().position(is_loud), pcm.iter().rposition(is_loud)) {
        (Some(first), Some(last)) => last - first + 1,
        _ => 0,
    }
}
