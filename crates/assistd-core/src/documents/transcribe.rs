//! Page transcription: while its permit holds, the OCR model borrows the drowsy
//! llama-server and reads the pages that wait for it, one rendered image at a time.

use std::fmt::Display;
use std::sync::Arc;
use std::time::Duration;

use assistd_documents::{Extractor, ExtractorError, RenderedPage};
use assistd_llm::{LlamaServerControl, PageTranscriber, TranscribeError, Transcription};
use assistd_memory::{
    DocumentId, DocumentStore, MemoryError, PageReader, PageReading, WaitingPage,
};
use assistd_utils::backlog::{
    Backlog, DrainEnd, Page, Pass, Signals, drain, keep_drained, permitted,
};
use assistd_utils::readiness::NotReady;
use parking_lot::Mutex;
use thiserror::Error;
use tokio::sync::watch;
use tracing::{debug, info, warn};

use super::indexer::DocumentIndex;
use crate::presence::{BorrowedModel, PresenceError, PresenceManager};

/// How long a run that stopped early is left alone before the next one starts,
/// since each loads a model.
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

/// Page transcription as the rest of the daemon sees it: how its last run went.
#[derive(Debug, Default)]
pub struct PageTranscription {
    last_run: Mutex<TranscriptionRun>,
}

impl PageTranscription {
    pub fn last_run(&self) -> TranscriptionRun {
        self.last_run.lock().clone()
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
    /// Whether pages may be transcribed now.
    pub permit: watch::Receiver<bool>,
    pub model: BorrowedModel,
    /// The model and prompt the readings kept from now on are by.
    pub reader: PageReader,
    pub transcriber: PageTranscriber,
    /// Says whether the server still has the OCR model loaded.
    pub control: LlamaServerControl,
    /// The largest file a page is rendered from, in bytes.
    pub max_file_bytes: u64,
}

impl TranscriptionDeps {
    /// Have the store keep only readings by this model and prompt. Run it before the
    /// first index pass, which gives pages the readings kept for them.
    pub async fn adopt_reader(&self) {
        match self.store.adopt_page_reader(&self.reader).await {
            Ok(None) => {}
            Ok(Some(replaced)) => info!(
                target: "assistd::documents",
                "dropped every kept page reading; they were read by model={}, prompt={:?}",
                replaced.model, replaced.prompt
            ),
            Err(error) => warn!(
                target: "assistd::documents",
                "cannot check which model the kept page readings are by: {error}"
            ),
        }
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

/// The pages waiting for transcription, read with the OCR model on loan. A page done
/// no longer waits, so the first one waiting is always the next.
struct WaitingPages<'a> {
    deps: &'a TranscriptionDeps,
    permit: watch::Receiver<bool>,
    counts: RunCounts,
}

impl Backlog for WaitingPages<'_> {
    type RowId = (DocumentId, usize);
    type Row = WaitingPage;
    type Error = RunStopped;

    fn permit(&mut self) -> Option<&mut watch::Receiver<bool>> {
        Some(&mut self.permit)
    }

    async fn page_after(&self, _after: Option<Self::RowId>) -> Result<Page<Self>, RunStopped> {
        let next = self.deps.store.next_waiting_page().await?;
        Ok(next
            .map(|page| ((page.document_id, page.page), page))
            .into_iter()
            .collect())
    }

    async fn do_page(&mut self, page: Page<Self>) -> Result<usize, RunStopped> {
        for (_, page) in page {
            let reading = self.deps.read(&page, &mut self.counts).await?;
            match reading {
                PageReading::Text(_) => self.counts.read += 1,
                PageReading::Failed => self.counts.failed += 1,
            }
            self.deps.store.store_page_reading(page, reading).await?;
        }
        Ok(0)
    }
}

/// One run of page transcription: the waiting pages drained on a loan of the server.
struct TranscriptionPass<'a> {
    deps: TranscriptionDeps,
    status: &'a PageTranscription,
}

impl TranscriptionPass<'_> {
    /// Borrow the server for the OCR model and drain the waiting pages, without the
    /// loan when none waits. `None` when the loan was cut short or the daemon is not
    /// `Drowsy` after all.
    async fn drain_on_loan(&self, signals: &Signals) -> Result<Option<DrainEnd>, RunStopped> {
        let deps = &self.deps;
        if deps.store.next_waiting_page().await?.is_none() {
            return Ok(Some(DrainEnd::Emptied { rows_left: 0 }));
        }
        let mut pages = WaitingPages {
            deps,
            permit: deps.permit.clone(),
            counts: RunCounts::default(),
        };
        let draining = async {
            self.status.record(TranscriptionRun::Running);
            drain_through_wakes(&mut pages, signals).await
        };
        let lent = deps.presence.lend_while_drowsy(&deps.model, draining).await;
        let RunCounts { read, failed, .. } = pages.counts;
        if read + failed > 0 {
            info!(
                target: "assistd::documents",
                "page transcription read {read} page(s) and failed on {failed}"
            );
        }
        match lent {
            Ok(Some(ended)) => ended.map(Some),
            Ok(None) | Err(PresenceError::NotDrowsy) => Ok(None),
            Err(error) => Err(RunStopped::Presence(error)),
        }
    }
}

impl Pass for TranscriptionPass<'_> {
    type Error = RunStopped;

    async fn run(&mut self, signals: &Signals) -> Result<DrainEnd, RunStopped> {
        let mut shutdown = signals.shutdown.clone();
        tokio::select! {
            biased;
            _ = shutdown.wait_for(|stop| *stop) => return Ok(DrainEnd::Shutdown),
            () = signals.wake.notified() => return Ok(DrainEnd::Woken),
            () = permitted(Some(&mut self.deps.permit)) => {}
        }
        let ended = self.drain_on_loan(signals).await;
        self.status.record(match &ended {
            Ok(_) => TranscriptionRun::Idle,
            Err(stopped) => TranscriptionRun::Stopped(stopped.to_string()),
        });
        if let Some(ended) = ended? {
            return Ok(ended);
        }
        tokio::select! {
            biased;
            _ = shutdown.wait_for(|stop| *stop) => Ok(DrainEnd::Shutdown),
            () = signals.wake.notified() => Ok(DrainEnd::Woken),
            permit = self.deps.permit.changed() => Ok(match permit {
                Ok(()) => DrainEnd::Woken,
                Err(_) => DrainEnd::Shutdown,
            }),
        }
    }
}

/// Until `shutdown` flips, read the waiting pages with the OCR model whenever the
/// permit holds and again when the store signals that more wait. Pages that failed
/// before wait again, and a run that stops early is retried in five minutes. The
/// store's database must stay open meanwhile.
pub async fn transcribe_while_permitted(
    deps: TranscriptionDeps,
    index: &DocumentIndex,
    shutdown: watch::Receiver<bool>,
) {
    if let Err(error) = deps.store.retry_failed_transcriptions().await {
        warn!(target: "assistd::documents", "failed pages are not tried again: {error}");
    }
    let signals = Signals {
        wake: deps.store.pages_to_transcribe(),
        shutdown,
    };
    let pass = TranscriptionPass {
        deps,
        status: index.transcription(),
    };
    keep_drained("page transcription", &signals, |_| RETRY_DELAY, pass).await;
}

/// Drain `pages` to the end, listing again after a wake so the loan is kept.
async fn drain_through_wakes(
    pages: &mut WaitingPages<'_>,
    signals: &Signals,
) -> Result<DrainEnd, RunStopped> {
    loop {
        match drain(pages, signals).await? {
            DrainEnd::Woken => {}
            ended => return Ok(ended),
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
