//! Stored backlogs: work that is whichever stored rows lack their result, and the
//! loop that drains it a page at a time each time it is woken.

use std::fmt::Display;
use std::future::{Future, pending};
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{Notify, watch};

/// Rows of backlog `B` in listing order, each with the id that names it.
pub type Page<B> = Vec<(<B as Backlog>::RowId, <B as Backlog>::Row)>;

/// One kind of stored work: the rows that lack their result.
pub trait Backlog: Send + Sync {
    /// Names a row, so that the page after it can be listed.
    type RowId: Copy + Send;
    type Row: Send;
    /// What the backlog needs being away; never the fault of a row.
    type Error: Send;

    /// Reads `true` while pages may be done; `None` when they always may.
    fn permit(&mut self) -> Option<&mut watch::Receiver<bool>> {
        None
    }

    /// The next page of rows still waiting after `after`, from the first when `None`,
    /// in order; empty when none is left.
    fn page_after(
        &self,
        after: Option<Self::RowId>,
    ) -> impl Future<Output = Result<Page<Self>, Self::Error>> + Send;

    /// Do `page`, recording each row's outcome where the row is stored: its result, or
    /// that it cannot be done. Returns how many of its rows are left for a later pass.
    fn do_page(
        &mut self,
        page: Page<Self>,
    ) -> impl Future<Output = Result<usize, Self::Error>> + Send;
}

/// One pass over the stored work [`keep_drained`] keeps drained.
pub trait Pass: Send {
    /// Why a pass stopped early.
    type Error: Display + Send;

    /// Drain what waits, with [`drain`], ending early on a wake or shutdown.
    fn run(
        &mut self,
        signals: &Signals,
    ) -> impl Future<Output = Result<DrainEnd, Self::Error>> + Send;
}

/// What a backlog's loop runs on.
#[derive(Debug, Clone)]
pub struct Signals {
    /// Signalled when rows may have been added.
    pub wake: Arc<Notify>,
    /// Reads `true` once the loop should stop.
    pub shutdown: watch::Receiver<bool>,
}

/// How [`drain`] ended without the backlog failing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrainEnd {
    /// The listing ran out; `rows_left` rows of the pages done wait for a later pass.
    Emptied { rows_left: usize },
    /// A wake arrived between pages.
    Woken,
    /// Shutdown was signalled between pages.
    Shutdown,
}

/// Page through `backlog` until it is empty, a wake arrives or shutdown flips, waiting
/// before each page for its permit. A row is met once, whatever became of it. Errors
/// as soon as the backlog does.
pub async fn drain<B: Backlog>(backlog: &mut B, signals: &Signals) -> Result<DrainEnd, B::Error> {
    let mut shutdown = signals.shutdown.clone();
    let mut after = None;
    let mut rows_left = 0usize;
    loop {
        tokio::select! {
            biased;
            _ = shutdown.wait_for(|stop| *stop) => return Ok(DrainEnd::Shutdown),
            () = signals.wake.notified() => return Ok(DrainEnd::Woken),
            () = permitted(backlog.permit()) => {}
        }
        let page = backlog.page_after(after).await?;
        let Some(&(last_row, _)) = page.last() else {
            return Ok(DrainEnd::Emptied { rows_left });
        };
        after = Some(last_row);
        rows_left = rows_left.saturating_add(backlog.do_page(page).await?);
    }
}

/// Run `pass` now and after every wake until shutdown. A pass that failed or left rows
/// is run again after `retry_delay` of how many passes in a row have ended so.
pub async fn keep_drained<P: Pass>(
    name: &'static str,
    signals: &Signals,
    retry_delay: fn(u32) -> Duration,
    mut pass: P,
) {
    let mut shutdown = signals.shutdown.clone();
    let mut unfinished_passes = 0u32;
    loop {
        let retry_in = match pass.run(signals).await {
            Ok(DrainEnd::Shutdown) => break,
            Ok(DrainEnd::Woken) => continue,
            Ok(DrainEnd::Emptied { rows_left: 0 }) => {
                unfinished_passes = 0;
                None
            }
            Ok(DrainEnd::Emptied { rows_left }) => {
                tracing::debug!(
                    target: "assistd::backlog",
                    backlog = name,
                    rows_left,
                    "pass left rows for a retry"
                );
                Some(retry_delay(unfinished_passes))
            }
            Err(err) => {
                log_stopped_pass(name, &err, unfinished_passes);
                Some(retry_delay(unfinished_passes))
            }
        };
        if retry_in.is_some() {
            unfinished_passes = unfinished_passes.saturating_add(1);
        }
        tokio::select! {
            biased;
            _ = shutdown.wait_for(|stop| *stop) => break,
            () = signals.wake.notified() => {}
            () = tokio::time::sleep(retry_in.unwrap_or_default()), if retry_in.is_some() => {}
        }
    }
}

/// Wait until `permit` reads `true`; never returns once it can no longer do so.
async fn permitted(permit: Option<&mut watch::Receiver<bool>>) {
    if let Some(permit) = permit
        && permit.wait_for(|open| *open).await.is_err()
    {
        pending::<()>().await;
    }
}

fn log_stopped_pass(name: &'static str, err: &impl Display, earlier_unfinished: u32) {
    if earlier_unfinished == 0 {
        tracing::warn!(
            target: "assistd::backlog",
            backlog = name,
            error = %err,
            "pass stopped early; its rows wait until a retry succeeds"
        );
    } else {
        tracing::debug!(
            target: "assistd::backlog",
            backlog = name,
            error = %err,
            earlier_unfinished,
            "pass stopped early again"
        );
    }
}
