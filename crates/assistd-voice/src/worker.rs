//! The voice worker: a child process that owns sherpa-onnx, cpal and rodio,
//! and the daemon-side proxies and supervisor that drive it over its pipes.

mod link;
mod protocol;
mod proxy;
mod serve;
mod stack;
mod supervisor;

pub use protocol::ProtocolError;
pub use serve::{ServeError, serve};
pub use supervisor::{VoiceWorkerSpec, supervise};
