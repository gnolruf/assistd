//! Piper text-to-speech. Each utterance is its own `piper --output-raw`
//! subprocess: piper writes raw PCM with no in-band frame delimiter, so
//! per-utterance EOF on stdout is the only reliable end marker. The
//! 50-250 ms model-load cost overlaps with generation of the next
//! sentence. A circuit breaker in [`service`] stops repeated failures
//! from spamming the logs.

pub mod cache;
pub mod config;
pub mod error;
pub mod playback;
pub mod service;
pub mod synth;

pub use error::PiperError;
pub use service::PiperVoiceOutput;
