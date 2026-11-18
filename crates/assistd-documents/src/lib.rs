//! Document text extraction in a child process: the protocol, the worker side the hidden
//! `assistd extract-worker` subcommand runs, and the daemon-side handle and supervisor.

mod extract;
mod extractor;
mod protocol;
mod serve;
mod supervisor;

pub use extractor::{Extractor, ExtractorError};
pub use protocol::{Extraction, MAX_TEXT_BYTES, SkipReason};
pub use serve::{ServeError, serve};
pub use supervisor::{WorkerCommand, supervise};
