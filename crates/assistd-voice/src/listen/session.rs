//! What a wake-word test session hears from the listener's capture stack.

/// One VAD-cut utterance, 16 kHz mono.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct CapturedUtterance {
    pub(super) pcm: Vec<i16>,
}

/// What a wake-word session's capture reports from the mic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum SessionAudio {
    /// An utterance started (`true`) or ended (`false`), kept or not.
    Speaking(bool),
    Utterance(CapturedUtterance),
}
