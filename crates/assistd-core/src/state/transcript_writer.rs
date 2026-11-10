//! The single task that writes transcripts, in the order the writes were queued.

use std::sync::{Arc, OnceLock};

use tokio::sync::{Notify, mpsc, oneshot};
use tracing::{debug, warn};

use assistd_llm::CompactionCut;
use assistd_memory::{
    AppendedMessage, BranchId, BranchSummary, HistoryScope, MemoryError, PersistedMessage,
    PersistedRelayStep, TranscriptStore, TurnId, TurnStats,
};

use super::seq_ledger::{RowRole, SeqLedger, SharedLedger};

/// Writes that may wait on the store before a turn queueing another waits too.
pub(super) const WRITE_QUEUE_CAPACITY: usize = 256;

/// Where a turn's messages are written: the branch current when the turn
/// opened, and the turn's row once the write opening it has landed.
#[derive(Debug, Clone)]
pub(super) struct PersistTarget {
    pub(super) branch: BranchId,
    turn: Arc<OnceLock<TurnId>>,
    ledger: SharedLedger,
}

impl PersistTarget {
    /// A target on `branch` for a turn of the conversation `ledger` follows.
    pub(super) fn new(branch: BranchId, ledger: SharedLedger) -> Self {
        Self {
            branch,
            turn: Arc::default(),
            ledger,
        }
    }
}

/// One step for the transcript writer.
#[derive(Debug)]
pub(super) enum WriteCommand {
    /// Open the target's turn with `prompt` as its first message.
    OpenTurn {
        target: PersistTarget,
        prompt: PersistedMessage,
    },
    /// Append `msg` to the target's turn, with the sub-agent steps behind a `relay` result.
    Message {
        target: PersistTarget,
        msg: PersistedMessage,
        relay_steps: Option<Vec<PersistedRelayStep>>,
    },
    TurnStats {
        target: PersistTarget,
        stats: TurnStats,
    },
    /// Record `summary` as standing in for what `cut` leaves out of the target's conversation.
    Summary {
        target: PersistTarget,
        summary: String,
        cut: CompactionCut,
        duration_ms: u64,
    },
    /// Signal once every command queued before this one has been carried out.
    Barrier(oneshot::Sender<()>),
}

/// Carries out [`WriteCommand`]s one at a time, since the store assigns `seq`
/// in arrival order.
#[derive(Debug)]
pub(super) struct TranscriptWriter {
    transcripts: Arc<dyn TranscriptStore>,
    /// The embedding worker's wake signal, when embedding is enabled.
    embed_wake: Option<Arc<Notify>>,
    failed_writes: u64,
}

impl TranscriptWriter {
    pub(super) fn new(
        transcripts: Arc<dyn TranscriptStore>,
        embed_wake: Option<Arc<Notify>>,
    ) -> Self {
        Self {
            transcripts,
            embed_wake,
            failed_writes: 0,
        }
    }

    /// Carry out `commands` until every sender is gone.
    pub(super) async fn run(mut self, mut commands: mpsc::Receiver<WriteCommand>) {
        while let Some(command) = commands.recv().await {
            match command {
                WriteCommand::OpenTurn { target, prompt } => self.open_turn(&target, prompt).await,
                WriteCommand::Message {
                    target,
                    msg,
                    relay_steps,
                } => self.append_message(&target, msg, relay_steps).await,
                WriteCommand::TurnStats { target, stats } => {
                    self.record_turn_stats(&target, stats).await;
                }
                WriteCommand::Summary {
                    target,
                    summary,
                    cut,
                    duration_ms,
                } => {
                    self.record_summary(&target, summary, cut, duration_ms)
                        .await;
                }
                WriteCommand::Barrier(landed) => {
                    let _ = landed.send(());
                }
            }
        }
    }

    async fn open_turn(&mut self, target: &PersistTarget, prompt: PersistedMessage) {
        self.read_ledger(target).await;
        match self.transcripts.begin_turn(target.branch, prompt).await {
            Ok(opened) => {
                let _ = target.turn.set(opened.turn);
                self.landed(target, RowRole::Question, &opened.message);
            }
            Err(e) => {
                target.ledger.record(RowRole::Question, None);
                self.failed_writes += 1;
                warn!(
                    target: "assistd::memory",
                    error = %e,
                    failed_writes = self.failed_writes,
                    "failed to open the turn; none of it will be persisted (continuing)"
                );
            }
        }
    }

    /// Build the target's ledger from its branch when the conversation's
    /// history was replaced since it was last built.
    async fn read_ledger(&self, target: &PersistTarget) {
        if target.ledger.is_read() {
            return;
        }
        match self
            .transcripts
            .load_branch(target.branch, HistoryScope::Unsummarized)
            .await
        {
            Ok(history) => target
                .ledger
                .replace(SeqLedger::from_history(&history.rows)),
            Err(e) => warn!(
                target: "assistd::memory",
                error = %e,
                "failed to read the branch; a compaction this turn will not be persisted"
            ),
        }
    }

    async fn append_message(
        &mut self,
        target: &PersistTarget,
        msg: PersistedMessage,
        relay_steps: Option<Vec<PersistedRelayStep>>,
    ) {
        let role = RowRole::of(&msg);
        let Some(&turn) = target.turn.get() else {
            target.ledger.record(role, None);
            self.failed_writes += 1;
            debug!(
                target: "assistd::memory",
                failed_writes = self.failed_writes,
                "turn was never opened; dropping message"
            );
            return;
        };
        let appended = match relay_steps {
            Some(steps) => {
                self.transcripts
                    .append_relay_result(target.branch, turn, msg, steps)
                    .await
            }
            None => {
                self.transcripts
                    .append_message_to_branch(target.branch, turn, msg)
                    .await
            }
        };
        match appended {
            Ok(appended) => self.landed(target, role, &appended),
            Err(e) => {
                target.ledger.record(role, None);
                self.mark_turn_incomplete(turn, &e).await;
            }
        }
    }

    /// Enter the stored `message` in the target's ledger and wake the
    /// embedding worker for the chunks cut from it.
    fn landed(&self, target: &PersistTarget, role: RowRole, message: &AppendedMessage) {
        target.ledger.record(role, Some(message.seq));
        if let Some(embed_wake) = &self.embed_wake
            && !message.chunks.is_empty()
        {
            embed_wake.notify_one();
        }
    }

    /// Count the message of `turn` lost to `error` and flag the turn.
    async fn mark_turn_incomplete(&mut self, turn: TurnId, error: &MemoryError) {
        self.failed_writes += 1;
        warn!(
            target: "assistd::memory",
            %error,
            turn = turn.0,
            failed_writes = self.failed_writes,
            "failed to persist message; marking its turn incomplete (continuing)"
        );
        if let Err(e) = self.transcripts.mark_turn_incomplete(turn).await {
            warn!(
                target: "assistd::memory",
                error = %e,
                turn = turn.0,
                "failed to mark the turn incomplete"
            );
        }
    }

    async fn record_turn_stats(&mut self, target: &PersistTarget, stats: TurnStats) {
        let Some(&turn) = target.turn.get() else {
            return;
        };
        if let Err(e) = self.transcripts.record_turn_stats(turn, stats).await {
            self.failed_writes += 1;
            warn!(
                target: "assistd::memory",
                error = %e,
                failed_writes = self.failed_writes,
                "failed to persist turn stats (continuing)"
            );
        }
    }

    async fn record_summary(
        &mut self,
        target: &PersistTarget,
        content: String,
        cut: CompactionCut,
        duration_ms: u64,
    ) {
        let Some(seqs) = target.ledger.cut_at(cut) else {
            self.failed_writes += 1;
            warn!(
                target: "assistd::memory",
                failed_writes = self.failed_writes,
                "the compaction keeps from a message that was never stored; \
                 keeping the branch's earlier summary"
            );
            return;
        };
        let summary = BranchSummary {
            kept_from_seq: seqs.kept_from,
            steps_folded_through: seqs.steps_folded_through,
            compacted_after_seq: seqs.compacted_after,
            content,
            duration_ms,
        };
        if let Err(e) = self
            .transcripts
            .record_branch_summary(target.branch, summary)
            .await
        {
            self.failed_writes += 1;
            warn!(
                target: "assistd::memory",
                error = %e,
                failed_writes = self.failed_writes,
                "failed to persist compaction summary (continuing)"
            );
        }
    }
}
