//! Page transcription: while the daemon is `Drowsy`, the OCR model borrows the idle
//! llama-server and reads the pages that wait for it, one rendered image at a time.

use std::fmt::Display;
use std::sync::Arc;
use std::time::Duration;

use assistd_documents::{Extractor, ExtractorError, RenderedPage};
use assistd_ipc::PresenceState;
use assistd_llm::{LlamaServerControl, PageTranscriber, TranscribeError, Transcription};
use assistd_memory::{DocumentStore, MemoryError, PageReading, WaitingPage};
use assistd_utils::readiness::NotReady;
use parking_lot::Mutex;
use thiserror::Error;
use tokio::sync::{Notify, watch};
use tracing::{debug, info, warn};

use super::indexer::DocumentIndex;
use crate::presence::{BorrowedModel, PresenceError, PresenceManager};

/// How long a run that stopped early is left alone, the daemon still `Drowsy`,
/// before the next one starts.
const RETRY_DELAY: Duration = Duration::from_secs(300);

/// Times the extraction worker may exit over one page before that page fails.
const WORKER_EXITS_PER_PAGE: u32 = 2;

/// Requests in a row the loaded model may refuse before the run stops: that many
/// are taken as the model's fault, not the pages'.
const REFUSALS_IN_A_ROW: u32 = 3;

/// How the last run of page transcription went.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum TranscriptionRun {
    /// None has started, or the last one ended with no page left or with a wake.
    #[default]
    Idle,
    Running,
    /// The last one stopped early; holds why.
    Stopped(String),
}

/// Page transcription as the rest of the daemon sees it: where it is told that
/// pages may wait, and how its last run went.
#[derive(Debug, Default)]
pub struct PageTranscription {
    pages_may_wait: Notify,
    last_run: Mutex<TranscriptionRun>,
}

impl PageTranscription {
    pub fn last_run(&self) -> TranscriptionRun {
        self.last_run.lock().clone()
    }

    /// Have transcription look for waiting pages once the daemon is `Drowsy`.
    pub(super) fn pages_may_wait(&self) {
        self.pages_may_wait.notify_one();
    }

    fn record(&self, run: TranscriptionRun) {
        *self.last_run.lock() = run;
    }
}

/// Why a run stopped with pages still waiting.
#[derive(Debug, Error)]
enum RunStopped {
    #[error(transparent)]
    Presence(#[from] PresenceError),
    #[error(transparent)]
    Store(#[from] MemoryError),
    #[error("document extraction is {0}")]
    Worker(NotReady),
    #[error("the OCR model stopped answering: {0}")]
    Model(TranscribeError),
    #[error("the OCR model refused {REFUSALS_IN_A_ROW} pages in a row: {0}")]
    Refusing(TranscribeError),
}

/// How many pages a run read and failed on before it ended, and how many
/// requests the model has just refused one after another.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct RunCounts {
    read: u64,
    failed: u64,
    refused_in_a_row: u32,
}

/// What page transcription reads from and writes to, and the OCR model it reads with.
#[derive(Debug)]
pub struct TranscriptionDeps {
    pub store: Arc<dyn DocumentStore>,
    pub extractor: Arc<Extractor>,
    pub presence: Arc<PresenceManager>,
    /// Wakes the embedding worker after a document leaves chunks to embed.
    pub embed_wake: Arc<Notify>,
    pub model: BorrowedModel,
    pub transcriber: PageTranscriber,
    /// Says whether the server still has the OCR model loaded.
    pub control: LlamaServerControl,
    /// The largest file a page is rendered from, in bytes.
    pub max_file_bytes: u64,
}

impl TranscriptionDeps {
    /// One run: nothing when no page waits, else the OCR model on the drowsy server
    /// until none does, the daemon leaves `Drowsy` or `shutdown` flips.
    async fn run_once(
        &self,
        status: &PageTranscription,
        shutdown: &watch::Receiver<bool>,
    ) -> Result<(), RunStopped> {
        if self.store.next_waiting_page().await?.is_none() {
            return Ok(());
        }
        let mut counts = RunCounts::default();
        let reading = self.read_waiting_pages(status, &mut counts, shutdown);
        let lent = self.presence.lend_while_drowsy(&self.model, reading).await;
        if counts.read + counts.failed > 0 {
            info!(
                target: "assistd::documents",
                "page transcription read {} page(s) and failed on {}",
                counts.read, counts.failed
            );
        }
        match lent {
            Ok(Some(read)) => read,
            Ok(None) | Err(PresenceError::NotDrowsy) => Ok(()),
            Err(error) => Err(RunStopped::Presence(error)),
        }
    }

    /// Read page after page with the model now loaded, reporting the run as running.
    async fn read_waiting_pages(
        &self,
        status: &PageTranscription,
        counts: &mut RunCounts,
        shutdown: &watch::Receiver<bool>,
    ) -> Result<(), RunStopped> {
        status.record(TranscriptionRun::Running);
        while !*shutdown.borrow() {
            let Some(page) = self.store.next_waiting_page().await? else {
                return Ok(());
            };
            let reading = self.read(&page, counts).await?;
            match reading {
                PageReading::Text(_) => counts.read += 1,
                PageReading::Failed => counts.failed += 1,
            }
            let added = self.store.store_page_reading(page, reading).await?;
            if !added.is_empty() {
                self.embed_wake.notify_one();
            }
        }
        Ok(())
    }

    /// What `page` reads as: nothing when it is blank, and failed when it cannot be
    /// rendered or the model's reply is not its text.
    async fn read(
        &self,
        page: &WaitingPage,
        counts: &mut RunCounts,
    ) -> Result<PageReading, RunStopped> {
        match self.render(page).await? {
            Some(RenderedPage::Blank) => Ok(PageReading::Text(String::new())),
            Some(RenderedPage::Image { png_base64 }) => {
                self.transcribe(page, &png_base64, counts).await
            }
            Some(RenderedPage::Failed { reason }) => Ok(failed(page, reason)),
            None => Ok(failed(page, "the extraction worker exited over it")),
        }
    }

    /// `page` as the worker renders it. A worker that exits is waited for and asked
    /// again; `None` once it has exited over the page [`WORKER_EXITS_PER_PAGE`] times.
    async fn render(&self, page: &WaitingPage) -> Result<Option<RenderedPage>, RunStopped> {
        let number = u32::try_from(page.page).unwrap_or(u32::MAX);
        let mut worker = self.extractor.subscribe_readiness();
        let mut worker_exits = 0;
        while worker_exits < WORKER_EXITS_PER_PAGE {
            worker_started(&mut worker).await?;
            let rendering = self
                .extractor
                .render_page(&page.path, self.max_file_bytes, number);
            match rendering.await {
                Ok(rendered) => return Ok(Some(rendered)),
                Err(ExtractorError::NotReady(_)) => {}
                Err(ExtractorError::Exited) => worker_exits += 1,
            }
        }
        Ok(None)
    }

    /// The model's reading of `page`. A request that fails with the model still
    /// loaded fails the page, until [`REFUSALS_IN_A_ROW`] have; that one, and one
    /// that fails with the model gone, stops the run with the page still waiting.
    async fn transcribe(
        &self,
        page: &WaitingPage,
        png_base64: &str,
        counts: &mut RunCounts,
    ) -> Result<PageReading, RunStopped> {
        let answered = self.transcriber.transcribe(png_base64).await;
        if answered.is_ok() {
            counts.refused_in_a_row = 0;
        }
        let error = match answered {
            Ok(Transcription::Text(text)) => return Ok(PageReading::Text(text)),
            Ok(Transcription::Unusable(why)) => return Ok(failed(page, why)),
            Err(error) => error,
        };
        let model = self.model.served_as.as_str();
        if !self.control.model_is_loaded(model).await.unwrap_or(false) {
            return Err(RunStopped::Model(error));
        }
        counts.refused_in_a_row += 1;
        if counts.refused_in_a_row == REFUSALS_IN_A_ROW {
            return Err(RunStopped::Refusing(error));
        }
        Ok(failed(page, error))
    }
}

/// Until `shutdown` flips, read the waiting pages with the OCR model whenever the
/// daemon is `Drowsy`. Pages that failed before wait again, and a run that stops
/// early is retried in five minutes. The store's database must stay open meanwhile.
pub async fn transcribe_while_drowsy(
    deps: TranscriptionDeps,
    index: &DocumentIndex,
    mut shutdown: watch::Receiver<bool>,
) {
    if let Err(error) = deps.store.retry_failed_transcriptions().await {
        warn!(target: "assistd::documents", "failed pages are not tried again: {error}");
    }
    let status = index.transcription();
    let mut presence = deps.presence.subscribe();
    loop {
        tokio::select! {
            biased;
            _ = shutdown.wait_for(|stop| *stop) => return,
            drowsy = presence.wait_for(|state| *state == PresenceState::Drowsy) => {
                if drowsy.is_err() {
                    return;
                }
            }
        }
        let ended = deps.run_once(status, &shutdown).await;
        let stopped_early = ended.is_err();
        status.record(match ended {
            Ok(()) => TranscriptionRun::Idle,
            Err(stopped) => {
                warn!(target: "assistd::documents", "page transcription stopped: {stopped}");
                TranscriptionRun::Stopped(stopped.to_string())
            }
        });
        tokio::select! {
            biased;
            _ = shutdown.wait_for(|stop| *stop) => return,
            _ = presence.changed() => {}
            () = status.pages_may_wait.notified(), if !stopped_early => {}
            () = tokio::time::sleep(RETRY_DELAY), if stopped_early => {}
        }
    }
}

/// Wait out a worker that is starting; errors when no worker is coming.
async fn worker_started(
    worker: &mut watch::Receiver<Result<(), NotReady>>,
) -> Result<(), RunStopped> {
    let started = worker
        .wait_for(|readiness| *readiness != Err(NotReady::Starting))
        .await;
    match started.as_deref() {
        Ok(Ok(())) => Ok(()),
        Ok(Err(not_ready)) => Err(RunStopped::Worker(not_ready.clone())),
        Err(_) => Err(RunStopped::Worker(NotReady::Starting)),
    }
}

/// `page` as failed, saying `why` in the log.
fn failed(page: &WaitingPage, why: impl Display) -> PageReading {
    debug!(
        target: "assistd::documents",
        "page {} of {} was not transcribed: {why}",
        page.page + 1,
        page.path.display()
    );
    PageReading::Failed
}
