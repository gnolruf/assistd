//! Runs the voice worker as a child in its own process group, restarts it
//! with backoff when it dies, and keeps the [`VoiceManager`] in step with it.

use std::ffi::OsString;
use std::io;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use assistd_config::VoiceConfig;
use assistd_utils::backoff::{RestartDecision, RestartPolicy};
use assistd_utils::process_group::{ProcessGroup, set_parent_death_signal};
use assistd_utils::readiness::Readiness;
use rustix::process::Signal;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, Lines};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::{mpsc, watch};
use tokio::task::JoinSet;
use tracing::{info, warn};

use super::link::{Link, StartupReport};
use super::protocol::{Startup, ToWorker, decode, encode};
use super::proxy::{WorkerListener, WorkerVoiceInput, WorkerVoiceOutput};
use crate::{BusyProbe, VoiceCapture, VoiceManager, VoiceOutput};

/// How often GPU availability is checked for a change to send on.
const GPU_POLL: Duration = Duration::from_millis(250);

/// How long a worker whose stdin closed has to exit before SIGTERM, and
/// again before SIGKILL.
const EXIT_GRACE: Duration = Duration::from_secs(2);

/// How to launch the voice worker, and the settings it builds its stack from.
#[derive(Debug, Clone)]
pub struct VoiceWorkerSpec {
    pub program: PathBuf,
    pub args: Vec<OsString>,
    pub config: VoiceConfig,
}

/// How a worker's session ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SessionEnd {
    Died,
    Shutdown,
}

/// The proxies handed out while a worker is up; the same ones across restarts.
#[derive(Debug)]
struct Proxies {
    capture: VoiceCapture,
    output: Arc<dyn VoiceOutput>,
}

/// A spawned worker and its pipes.
struct RunningWorker {
    child: Child,
    group: ProcessGroup,
    stdin: ChildStdin,
    stdout: Lines<BufReader<ChildStdout>>,
}

impl RunningWorker {
    fn spawn(spec: &VoiceWorkerSpec) -> io::Result<Self> {
        let mut command = Command::new(&spec.program);
        command
            .args(&spec.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .kill_on_drop(true)
            .process_group(0);
        set_parent_death_signal(&mut command);
        let mut child = command.spawn()?;
        let group = ProcessGroup::led_by(&child)
            .ok_or_else(|| io::Error::other("spawned voice worker reported no pid"))?;
        let stdin = child.stdin.take().expect("stdin piped");
        let stdout = BufReader::new(child.stdout.take().expect("stdout piped")).lines();
        Ok(Self {
            child,
            group,
            stdin,
            stdout,
        })
    }
}

/// One worker's run, from spawn to exit.
struct Session<'a> {
    link: &'a Link,
    voice: &'a VoiceManager,
    proxies: &'a Proxies,
    gpu: &'a dyn BusyProbe,
}

impl Session<'_> {
    async fn run(
        &self,
        worker: RunningWorker,
        config: &VoiceConfig,
        shutdown: &mut watch::Receiver<bool>,
    ) -> SessionEnd {
        let RunningWorker {
            mut child,
            mut group,
            stdin,
            mut stdout,
        } = worker;
        let (outgoing, outgoing_rx) = mpsc::unbounded_channel();
        let _ = outgoing.send(ToWorker::Configure {
            config: Box::new(config.clone()),
        });
        let unresponsive = self.link.connect(outgoing);
        let mut resumes = JoinSet::new();
        let ended = tokio::select! {
            Ok(()) = unresponsive => {
                warn!(target: "assistd::voice::worker", "killing the unresponsive voice worker");
                SessionEnd::Died
            }
            () = write_messages(stdin, outgoing_rx) => SessionEnd::Died,
            () = self.read_messages(&mut stdout, &mut resumes) => SessionEnd::Died,
            () = self.forward_gpu() => SessionEnd::Died,
            status = group.reap_leader(&mut child) => {
                warn!(target: "assistd::voice::worker", "voice worker exited: {status:?}");
                SessionEnd::Died
            }
            _ = shutdown.wait_for(|stop| *stop) => SessionEnd::Shutdown,
        };
        match ended {
            SessionEnd::Shutdown => stop_gracefully(child, group).await,
            SessionEnd::Died => {
                group.signal(Signal::KILL);
                let _ = group.reap_leader(&mut child).await;
            }
        }
        ended
    }

    /// Mirror the worker's messages until its stdout closes.
    async fn read_messages(
        &self,
        stdout: &mut Lines<BufReader<ChildStdout>>,
        resumes: &mut JoinSet<()>,
    ) {
        loop {
            let line = match stdout.next_line().await {
                Ok(Some(line)) => line,
                Ok(None) => return,
                Err(e) => {
                    warn!(target: "assistd::voice::worker", "voice worker stdout: {e}");
                    return;
                }
            };
            match decode(&line) {
                Ok(message) => {
                    if let Some(report) = self.link.receive(message) {
                        self.record_startup(report, resumes);
                    }
                }
                Err(e) => warn!(target: "assistd::voice::worker", "{e}; line ignored"),
            }
        }
    }

    /// Hand out the proxies for a component that came up, resuming listening
    /// if it was on before the worker restarted.
    fn record_startup(&self, report: StartupReport, resumes: &mut JoinSet<()>) {
        match report {
            StartupReport::Capture(startup) => {
                let ready = startup == Startup::Ready;
                self.voice
                    .set_capture(readiness_of(startup, self.proxies.capture.clone()));
                if ready && self.link.listen_wanted.load(Ordering::SeqCst) {
                    resumes.spawn(resume_listening(Arc::clone(&self.proxies.capture.listener)));
                }
            }
            StartupReport::Speech(startup) => self
                .voice
                .set_speech(readiness_of(startup, Arc::clone(&self.proxies.output))),
        }
    }

    /// Send GPU availability now and on every change; never returns.
    async fn forward_gpu(&self) {
        let mut reported = None;
        let mut ticks = tokio::time::interval(GPU_POLL);
        loop {
            ticks.tick().await;
            let current = (self.gpu.presence_active(), self.gpu.foreign_gpu_busy());
            if reported != Some(current) {
                let (presence_active, foreign_busy) = current;
                self.link.notify(ToWorker::Gpu {
                    presence_active,
                    foreign_busy,
                });
                reported = Some(current);
            }
        }
    }
}

/// Run the voice worker `spec` describes until `shutdown` flips, restarting
/// it with backoff when it dies. `voice` reports each component `Starting`
/// until the worker says how it came up, and `Unavailable` once restarts stop.
pub async fn supervise(
    spec: VoiceWorkerSpec,
    voice: Arc<VoiceManager>,
    gpu: Arc<dyn BusyProbe>,
    mut shutdown: watch::Receiver<bool>,
) {
    let link = Arc::new(Link::new());
    let proxies = Proxies {
        capture: VoiceCapture {
            input: Arc::new(WorkerVoiceInput(Arc::clone(&link))),
            listener: Arc::new(WorkerListener(Arc::clone(&link))),
        },
        output: Arc::new(WorkerVoiceOutput(Arc::clone(&link))),
    };
    let session = Session {
        link: &link,
        voice: &voice,
        proxies: &proxies,
        gpu: gpu.as_ref(),
    };
    let mut policy = RestartPolicy::default();
    loop {
        let started = std::time::Instant::now();
        let ended = match RunningWorker::spawn(&spec) {
            Ok(worker) => {
                let ended = session.run(worker, &spec.config, &mut shutdown).await;
                policy.record_session_end(started.elapsed());
                ended
            }
            Err(e) => {
                warn!(target: "assistd::voice::worker", "failed to spawn the voice worker: {e}");
                policy.record_spawn_failure();
                SessionEnd::Died
            }
        };
        link.disconnect();
        if ended == SessionEnd::Shutdown {
            return;
        }
        voice.set_capture(Readiness::Starting);
        voice.set_speech(Readiness::Starting);
        if !wait_to_restart(&mut policy, &voice, &mut shutdown).await {
            return;
        }
    }
}

/// Sleep out the policy's backoff; false when shutdown flips first or the
/// policy gives up, in which case `voice` is marked unavailable.
async fn wait_to_restart(
    policy: &mut RestartPolicy,
    voice: &VoiceManager,
    shutdown: &mut watch::Receiver<bool>,
) -> bool {
    let reason = match policy.next_restart(std::time::Instant::now()) {
        RestartDecision::Backoff { delay, failures } => {
            info!(
                target: "assistd::voice::worker",
                "restarting the voice worker in {delay:?} ({failures} failed in a row)"
            );
            return tokio::select! {
                () = tokio::time::sleep(delay) => true,
                _ = shutdown.wait_for(|stop| *stop) => false,
            };
        }
        RestartDecision::ConsecutiveCapReached { failures } => {
            format!("the voice worker failed {failures} times in a row")
        }
        RestartDecision::WindowCapReached { restarts } => {
            format!("the voice worker restarted {restarts} times in ten minutes")
        }
    };
    warn!(target: "assistd::voice::worker", "{reason}; voice is off until the daemon restarts");
    voice.set_capture(Readiness::Unavailable(reason.as_str().into()));
    voice.set_speech(Readiness::Unavailable(reason.into()));
    false
}

fn readiness_of<T>(startup: Startup, ready: T) -> Readiness<T> {
    match startup {
        Startup::Ready => Readiness::Ready(ready),
        Startup::Unavailable { reason } => Readiness::Unavailable(reason.into()),
    }
}

async fn resume_listening(listener: Arc<dyn crate::ContinuousListener>) {
    match listener.start().await {
        Ok(()) => info!(target: "assistd::voice::worker", "listening resumed after a restart"),
        Err(e) => warn!(target: "assistd::voice::worker", "failed to resume listening: {e}"),
    }
}

/// Write each message to the worker's stdin; returns once the pipe fails.
async fn write_messages(mut stdin: ChildStdin, mut messages: mpsc::UnboundedReceiver<ToWorker>) {
    while let Some(message) = messages.recv().await {
        let line = encode(&message);
        let written = async {
            stdin.write_all(line.as_bytes()).await?;
            stdin.flush().await
        };
        if let Err(e) = written.await {
            warn!(target: "assistd::voice::worker", "voice worker stdin: {e}");
            return;
        }
    }
}

/// Give a worker whose stdin has closed time to exit, then SIGTERM and
/// finally SIGKILL its process group.
async fn stop_gracefully(mut child: Child, mut group: ProcessGroup) {
    if tokio::time::timeout(EXIT_GRACE, group.reap_leader(&mut child))
        .await
        .is_ok()
    {
        return;
    }
    warn!(target: "assistd::voice::worker", "voice worker still running; sending SIGTERM");
    group.signal(Signal::TERM);
    if tokio::time::timeout(EXIT_GRACE, group.reap_leader(&mut child))
        .await
        .is_err()
    {
        group.signal(Signal::KILL);
        let _ = group.reap_leader(&mut child).await;
    }
}
