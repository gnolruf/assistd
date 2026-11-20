//! Document text extraction in a child process: the protocol, the worker side the hidden
//! `assistd extract-worker` subcommand runs, and the daemon-side handle and supervisor.

mod archive;
mod epub;
mod extract;
mod extractor;
mod office;
mod opendocument;
mod pdf;
mod protocol;
#[cfg(test)]
mod sample_archive;
#[cfg(test)]
mod sample_pdf;
mod serve;
mod supervisor;
mod text;
mod xml;

pub use extractor::{Extractor, ExtractorError};
pub use protocol::{
    Extraction, MAX_PAGE_IMAGE_BYTES, MAX_PAGES, MAX_TEXT_BYTES, RenderedPage, SkipReason,
};
pub use serve::{ServeError, serve};
pub use supervisor::{WorkerCommand, supervise};
