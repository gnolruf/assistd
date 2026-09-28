//! Piper text-to-speech: one `piper --output-raw` subprocess per
//! utterance, played back through rodio.

pub mod cache;
pub mod config;
pub mod error;
pub mod playback;
pub mod service;
pub mod synth;

pub use error::PiperError;
pub use service::PiperVoiceOutput;
