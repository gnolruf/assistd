//! Wake-phrase enrollment: screens each recording, then learns the distinct
//! spellings Whisper produces for the phrase as accepted variants.

use assistd_config::UnitInterval;
use assistd_ipc::{EnrollEvent, SampleRejection};

use super::vad::SAMPLE_RATE_HZ;
use super::wake_phrase::{WakePhrase, normalized_words, similarity};

/// Confident samples an enrollment collects before it is ready to save.
pub const CONFIDENT_SAMPLES: u8 = 5;

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

/// What a wake-word session's capture reports from the mic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionAudio {
    /// An utterance started (`true`) or ended (`false`), kept or not.
    Speaking(bool),
    Utterance(CapturedUtterance),
}

/// An accepted enrollment recording and what Whisper heard in it.
#[derive(Debug, Clone)]
pub(super) struct Recording {
    pub(super) pcm: Vec<i16>,
    pub(super) heard: String,
}

/// An enrollment in progress: the phrase, the spellings learned so far
/// (the phrase's own first), and the accepted recordings, of which
/// `confident` would wake the listener on the phrase alone.
#[derive(Debug)]
pub struct Enrollment {
    phrase: String,
    spellings: Vec<String>,
    recordings: Vec<Recording>,
    bare_phrase: WakePhrase,
    confident: u8,
}

impl Enrollment {
    /// A sample is confident when the phrase, without variants, matches it
    /// at `min_similarity`.
    pub fn new(phrase: &str, min_similarity: UnitInterval) -> Self {
        Self {
            phrase: phrase.to_string(),
            spellings: vec![spelling(phrase)],
            recordings: Vec::new(),
            bare_phrase: WakePhrase::new(phrase, &[], min_similarity),
            confident: 0,
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
    /// `SampleAccepted` event to report.
    pub fn accept(&mut self, pcm: Vec<i16>, heard: &str) -> Result<EnrollEvent, SampleRejection> {
        let heard_spelling = spelling(heard);
        if heard_spelling.is_empty()
            || similarity(&self.spellings[0], &heard_spelling) < MIN_PHRASE_SIMILARITY
        {
            return Err(SampleRejection::DidNotMatch);
        }
        if !self.spellings.contains(&heard_spelling) {
            self.spellings.push(heard_spelling);
        }
        let confident = self.bare_phrase.match_prefix(heard).is_some();
        if confident {
            self.confident = self.confident.saturating_add(1);
        }
        self.recordings.push(Recording {
            pcm,
            heard: heard.to_string(),
        });
        Ok(EnrollEvent::SampleAccepted {
            index: self.accepted(),
            heard: heard.to_string(),
            confident,
        })
    }

    /// Whether [`CONFIDENT_SAMPLES`] confident samples were collected.
    pub fn is_ready(&self) -> bool {
        self.confident >= CONFIDENT_SAMPLES
    }

    /// The prompt for the next sample, with progress so far.
    pub fn prompt(&self) -> EnrollEvent {
        EnrollEvent::Prompt {
            confident: self.confident,
            required: CONFIDENT_SAMPLES,
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
