//! Text-to-speech: a sherpa-onnx voice loaded in-process, played back
//! through rodio.

mod engine;
mod error;
mod phonemize;
mod playback;
mod service;
mod voice_files;

pub use error::SpeechError;
pub(crate) use service::SherpaVoiceOutput;
