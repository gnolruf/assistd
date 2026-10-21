//! Wire types for wake-phrase enrollment and testing sessions.

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Most words an enrolled wake phrase may have.
pub const MAX_WAKE_PHRASE_WORDS: usize = 6;

/// A wake phrase as a client asked to enroll it: 1 to
/// [`MAX_WAKE_PHRASE_WORDS`] words of letters and apostrophes, single-spaced.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct WakePhraseText(String);

/// Why a [`WakePhraseText`] was refused.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum InvalidWakePhrase {
    #[error("the wake phrase is empty")]
    Empty,
    #[error("the wake phrase has more than {MAX_WAKE_PHRASE_WORDS} words")]
    TooManyWords,
    #[error("the wake phrase may contain only letters, spaces and apostrophes, not {0:?}")]
    BadCharacter(char),
}

impl WakePhraseText {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for WakePhraseText {
    type Error = InvalidWakePhrase;

    fn try_from(phrase: String) -> Result<Self, Self::Error> {
        if let Some(bad) = phrase
            .chars()
            .find(|&ch| !(ch.is_alphabetic() || ch.is_whitespace() || ch == '\'' || ch == '’'))
        {
            return Err(InvalidWakePhrase::BadCharacter(bad));
        }
        let words: Vec<&str> = phrase.split_whitespace().collect();
        if words
            .iter()
            .all(|word| !word.chars().any(char::is_alphabetic))
        {
            return Err(InvalidWakePhrase::Empty);
        }
        if words.len() > MAX_WAKE_PHRASE_WORDS {
            return Err(InvalidWakePhrase::TooManyWords);
        }
        Ok(Self(words.join(" ")))
    }
}

impl From<WakePhraseText> for String {
    fn from(phrase: WakePhraseText) -> Self {
        phrase.0
    }
}

/// One wake-word operation, carried by `Request::WakeWord`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum WakeWordOp {
    /// Report the saved profile. Emits `WakeWordStatus`, then `Done`.
    Status,
    /// Open an enrollment session on this connection, which streams
    /// `WakeWordEnrollment` events until it is finished, cancelled, or the
    /// client disconnects.
    EnrollStart { phrase: WakePhraseText },
    /// Save the open enrollment. Errors before enough samples are accepted.
    EnrollFinish,
    /// Abandon the open enrollment without saving.
    EnrollCancel,
    /// Open a test session on this connection, which streams `WakeWordTest`
    /// events against the saved phrase until stopped or disconnected.
    TestStart,
    /// End the open test session.
    TestStop,
    /// Delete the saved profile and return to the built-in phrase.
    Reset,
}

/// Why an enrollment sample was not accepted.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SampleRejection {
    TooShort,
    TooLong,
    /// The recording hit full scale; the mic gain is too high.
    Clipped,
    /// The transcript is too far from the phrase to be a mis-hearing of it.
    DidNotMatch,
}

/// Progress of an enrollment session, carried by `Event::WakeWordEnrollment`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EnrollEvent {
    /// Say the phrase again. `consistent` once the last few samples taught
    /// nothing new; saving needs `accepted >= minimum`.
    Prompt {
        accepted: u8,
        minimum: u8,
        consistent: bool,
    },
    /// Sample `index` (from 1) was kept; `heard` is Whisper's transcript.
    SampleAccepted { index: u8, heard: String },
    /// `heard` is `None` when the audio was refused before transcription.
    SampleRejected {
        reason: SampleRejection,
        heard: Option<String>,
    },
    /// The profile was written and the listener now uses it.
    Saved {
        phrase: String,
        variants: Vec<String>,
    },
}
