//! Runs the extraction worker under the shared pipe-worker supervisor and
//! keeps the [`Extractor`] in step with it.

use std::io;
use std::sync::Arc;
use std::time::Duration;

use assistd_utils::pipe_worker::{self, PipeWorker};
use assistd_utils::readiness::NotReady;
use tokio::process::Command;
use tokio::sync::{mpsc, oneshot, watch};
use tracing::warn;

use crate::extractor::Extractor;
use crate::protocol::{FromWorker, MAX_REPLY_LINE_BYTES, Request, decode, encode};

/// How long the worker has to say it is ready, its first line, once spawned.
const STARTUP_DEADLINE: Duration = Duration::from_secs(30);

/// Builds the command one worker process is spawned from. Called on the
/// blocking pool before every start, so it may open files.
pub type WorkerCommand = Arc<dyn Fn() -> io::Result<Command> + Send + Sync>;

struct ExtractionWorker {
    command: WorkerCommand,
    extractor: Arc<Extractor>,
}

impl PipeWorker for ExtractionWorker {
    type Outgoing = Request;

    const NAME: &'static str = "extraction worker";

    const MAX_LINE_BYTES: usize = MAX_REPLY_LINE_BYTES;

    const STARTUP_DEADLINE: Option<Duration> = Some(STARTUP_DEADLINE);

    fn command(&self) -> io::Result<Command> {
        (self.command)()
    }

    fn connect(&self, outgoing: mpsc::UnboundedSender<Request>) -> oneshot::Receiver<()> {
        self.extractor.connect(outgoing)
    }

    fn encode(message: &Request) -> String {
        encode(message)
    }

    fn receive(&self, line: &[u8]) {
        match decode::<FromWorker>(line) {
            Ok(message) => self.extractor.receive(message),
            Err(e) => warn!(target: "assistd::documents::worker", "{e}; line ignored"),
        }
    }

    fn disconnect(&self) {
        self.extractor.disconnect();
    }

    fn set_not_ready(&self, not_ready: NotReady) {
        self.extractor.set_readiness(Err(not_ready));
    }
}

/// Run the extraction worker `command` builds until `shutdown` flips, restarting
/// it with backoff when it dies. `extractor` is starting until the worker says
/// it is ready, and unavailable between the slow retries of a worker that keeps dying.
pub async fn supervise(
    command: WorkerCommand,
    extractor: Arc<Extractor>,
    shutdown: watch::Receiver<bool>,
) {
    let worker = Arc::new(ExtractionWorker { command, extractor });
    pipe_worker::supervise(worker, shutdown).await;
}
