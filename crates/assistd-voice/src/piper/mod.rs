//! Piper text-to-speech via managed per-utterance subprocess.
//!
//! Each `speak()` spawns a fresh `piper --output-raw` child, writes the
//! sentence to stdin, drops stdin to signal EOF, drains raw 16-bit LE
//! PCM from stdout, and queues it on the rodio playback worker. EOF on
//! stdout is the unambiguous end-of-utterance marker — no in-band
//! protocol or stderr-marker race. Crashes only fail the in-flight
//! utterance; a circuit breaker disables further calls after repeated
//! failures so a missing binary or bad audio device doesn't spam the
//! logs forever.

pub mod cache;
pub mod config;
pub mod error;
pub mod playback;
pub mod service;
pub mod synth;

pub use error::PiperError;
pub use service::{PiperVoiceOutput, ReadyState};
