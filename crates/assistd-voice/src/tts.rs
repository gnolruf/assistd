//! Text-to-speech: a sherpa-onnx voice loaded in-process, played back
//! through rodio.

pub mod engine;
pub mod error;
pub mod phonemize;
pub mod playback;
pub mod service;
pub mod voice_files;

pub use error::SpeechError;
pub use service::SherpaVoiceOutput;
