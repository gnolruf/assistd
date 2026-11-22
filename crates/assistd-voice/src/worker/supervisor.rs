//! Runs the voice worker under the shared pipe-worker supervisor and keeps
//! the [`VoiceManager`] in step with it.

use std::ffi::OsString;
use std::io;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use assistd_config::VoiceConfig;
use assistd_utils::pipe_worker::{self, PipeWorker};
use assistd_utils::readiness::{NotReady, Readiness};
use tokio::process::Command;
use tokio::sync::{mpsc, oneshot, watch};
use tokio::task::JoinSet;
use tracing::{info, warn};

use super::link::{Link, StartupReport};
use super::protocol::{MAX_REPLY_LINE_BYTES, Startup, ToWorker, decode, encode};
use super::proxy::{WorkerListener, WorkerVoiceInput, WorkerVoiceOutput};
use crate::{VoiceCapture, VoiceManager, VoiceOutput};

/// How to launch the voice worker, and the settings it builds its stack from.
#[derive(Debug, Clone)]
pub struct VoiceWorkerSpec {
    pub program: PathBuf,
    pub args: Vec<OsString>,
    pub config: VoiceConfig,
}

/// The proxies handed out while a worker is up; the same ones across restarts.
#[derive(Debug)]
struct Proxies {
    capture: VoiceCapture,
    output: Arc<dyn VoiceOutput>,
}

/// The voice worker's process and the daemon's end of its pipes.
struct VoiceWorker {
    spec: VoiceWorkerSpec,
    link: Arc<Link>,
    voice: Arc<VoiceManager>,
    proxies: Proxies,
    /// Calls resuming listening on the running worker; aborted when it goes.
    resumes: parking_lot::Mutex<JoinSet<()>>,
}

impl VoiceWorker {
    /// Hand out the proxies for a component that came up, resuming listening
    /// if it was on when capture went away.
    fn record_startup(&self, report: StartupReport) {
        match report {
            StartupReport::Capture(startup) => {
                let ready = startup == Startup::Ready;
                self.voice
                    .set_capture(readiness_of(startup, self.proxies.capture.clone()));
                if ready && self.link.listen_wanted.load(Ordering::SeqCst) {
                    let listener = Arc::clone(&self.proxies.capture.listener);
                    self.resumes.lock().spawn(resume_listening(listener));
                }
            }
            StartupReport::Speech(startup) => self
                .voice
                .set_speech(readiness_of(startup, Arc::clone(&self.proxies.output))),
        }
    }
}

impl PipeWorker for VoiceWorker {
    type Outgoing = ToWorker;

    const NAME: &'static str = "voice worker";

    const MAX_LINE_BYTES: usize = MAX_REPLY_LINE_BYTES;

    const STARTUP_DEADLINE: Option<Duration> = None;

    fn command(&self) -> io::Result<Command> {
        let mut command = Command::new(&self.spec.program);
        command.args(&self.spec.args);
        Ok(command)
    }

    fn connect(&self, outgoing: mpsc::UnboundedSender<ToWorker>) -> oneshot::Receiver<()> {
        let _ = outgoing.send(ToWorker::Configure {
            config: Box::new(self.spec.config.clone()),
        });
        self.link.connect(outgoing)
    }

    fn encode(message: &ToWorker) -> String {
        encode(message)
    }

    fn receive(&self, line: &[u8]) {
        match decode(line) {
            Ok(message) => {
                if let Some(report) = self.link.receive(message) {
                    self.record_startup(report);
                }
            }
            Err(e) => warn!(target: "assistd::voice::worker", "{e}; line ignored"),
        }
    }

    fn disconnect(&self) {
        self.link.disconnect();
        *self.resumes.lock() = JoinSet::new();
    }

    fn set_not_ready(&self, not_ready: NotReady) {
        self.voice.set_capture(readiness_without(&not_ready));
        self.voice.set_speech(readiness_without(&not_ready));
    }
}

/// Run the voice worker `spec` describes until `shutdown` flips, restarting
/// it with backoff when it dies. `voice` reports each component `Starting`
/// until the worker says how it came up, and `Unavailable` between the slow
/// retries of a worker that keeps dying.
pub async fn supervise(
    spec: VoiceWorkerSpec,
    voice: Arc<VoiceManager>,
    shutdown: watch::Receiver<bool>,
) {
    let link = Arc::new(Link::new());
    let proxies = Proxies {
        capture: VoiceCapture {
            input: Arc::new(WorkerVoiceInput(Arc::clone(&link))),
            listener: Arc::new(WorkerListener(Arc::clone(&link))),
        },
        output: Arc::new(WorkerVoiceOutput(Arc::clone(&link))),
    };
    let worker = Arc::new(VoiceWorker {
        spec,
        link,
        voice,
        proxies,
        resumes: parking_lot::Mutex::new(JoinSet::new()),
    });
    pipe_worker::supervise(worker, shutdown).await;
}

fn readiness_of<T>(startup: Startup, ready: T) -> Readiness<T> {
    match startup {
        Startup::Ready => Readiness::Ready(ready),
        Startup::Unavailable { reason } => Readiness::Unavailable(reason.into()),
    }
}

fn readiness_without<T>(not_ready: &NotReady) -> Readiness<T> {
    match not_ready {
        NotReady::Starting => Readiness::Starting,
        NotReady::Unavailable(reason) => Readiness::Unavailable(Arc::clone(reason)),
    }
}

async fn resume_listening(listener: Arc<dyn crate::ContinuousListener>) {
    match listener.start().await {
        Ok(()) => info!(target: "assistd::voice::worker", "listening resumed"),
        Err(e) => warn!(target: "assistd::voice::worker", "failed to resume listening: {e}"),
    }
}
