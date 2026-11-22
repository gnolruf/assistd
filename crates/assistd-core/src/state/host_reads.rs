//! `BranchHostReads`: what the history of the user's active branch has read from this machine.

use tokio::sync::Mutex;
use tracing::warn;

use assistd_memory::{BranchId, HistoryRow, HistoryScope, MessageKind, TranscriptStore};
use assistd_tools::HostReads;

/// The reads of the branch the last turn ran on, shared by its turns. Another branch's are
/// found again in its saved rows.
#[derive(Debug, Default)]
pub(super) struct BranchHostReads {
    tracked: Mutex<Option<(Option<BranchId>, HostReads)>>,
}

impl BranchHostReads {
    /// The reads of `branch`, `None` being a session with nothing saved: those the last turn
    /// recorded if it ran on `branch`, else what its saved tool results recorded.
    pub(super) async fn of(
        &self,
        branch: Option<BranchId>,
        transcripts: &dyn TranscriptStore,
    ) -> HostReads {
        let mut tracked = self.tracked.lock().await;
        if let Some((of, host_reads)) = tracked.as_ref()
            && *of == branch
        {
            return host_reads.clone();
        }
        let host_reads = match branch {
            Some(branch) => saved_host_reads(branch, transcripts).await,
            None => HostReads::default(),
        };
        *tracked = Some((branch, host_reads.clone()));
        host_reads
    }
}

/// A branch that cannot be loaded counts as having read this machine.
async fn saved_host_reads(branch: BranchId, transcripts: &dyn TranscriptStore) -> HostReads {
    let seen = match transcripts.load_branch(branch, HistoryScope::Whole).await {
        Ok(history) => rows_read_host(&history.rows),
        Err(e) => {
            warn!(
                target: "assistd::policy",
                error = %e,
                "load_branch failed; treating the branch as having read this machine"
            );
            true
        }
    };
    HostReads::known(seen)
}

fn rows_read_host(rows: &[HistoryRow]) -> bool {
    rows.iter().any(|row| {
        matches!(
            row.message.kind,
            MessageKind::Tool {
                read_host: true,
                ..
            }
        )
    })
}
