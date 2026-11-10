//! Branch seqs of the rows behind a conversation's messages, which place a
//! compaction cut on the branch.

use std::sync::Arc;

use parking_lot::Mutex;

use assistd_llm::CompactionCut;
use assistd_memory::{HistoryRow, MessageKind, PersistedMessage, Seq};

/// What a message is to a compaction cut.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RowRole {
    /// The user prompt that opens a turn.
    Question,
    /// An assistant message that calls tools.
    ToolStep,
    Other,
}

impl RowRole {
    pub(super) fn of(msg: &PersistedMessage) -> Self {
        match &msg.kind {
            MessageKind::User => RowRole::Question,
            MessageKind::Assistant { tool_calls, .. } if !tool_calls.is_empty() => {
                RowRole::ToolStep
            }
            MessageKind::Assistant { .. } | MessageKind::Tool { .. } => RowRole::Other,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct LedgerRow {
    role: RowRole,
    /// `None` when the message's write failed.
    seq: Option<Seq>,
}

/// Where a compaction cut falls on a branch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct CutSeqs {
    /// The user message opening the history kept word for word.
    pub(super) kept_from: Seq,
    /// Last row of that message's turn before its kept tool steps.
    pub(super) steps_folded_through: Option<Seq>,
    /// The newest stored message.
    pub(super) compacted_after: Seq,
}

/// One row per message the model holds, oldest first, including those whose
/// write failed.
#[derive(Debug, Default)]
pub(super) struct SeqLedger {
    rows: Vec<LedgerRow>,
}

impl SeqLedger {
    /// The ledger of a conversation replayed from `rows`.
    pub(super) fn from_history(rows: &[HistoryRow]) -> Self {
        let rows = rows
            .iter()
            .map(|row| LedgerRow {
                role: RowRole::of(&row.message),
                seq: Some(row.seq),
            })
            .collect();
        Self { rows }
    }

    pub(super) fn record(&mut self, role: RowRole, seq: Option<Seq>) {
        self.rows.push(LedgerRow { role, seq });
    }

    /// Drop the rows `cut` summarises and place it on the branch; `None` when
    /// the turn it keeps from is not held or its question was never stored.
    pub(super) fn cut_at(&mut self, cut: CompactionCut) -> Option<CutSeqs> {
        let question = self
            .rows
            .iter()
            .enumerate()
            .rev()
            .filter(|(_, row)| row.role == RowRole::Question)
            .nth(cut.kept_turns.get() - 1)
            .map(|(index, _)| index)?;
        self.rows.drain(..question);
        let (question, later) = self.rows.split_first()?;
        let kept_from = question.seq?;
        let turn_len = later
            .iter()
            .position(|row| row.role == RowRole::Question)
            .unwrap_or(later.len());
        let steps_folded_through = cut.kept_steps.and_then(|kept_steps| {
            let first_kept_step = later[..turn_len]
                .iter()
                .enumerate()
                .rev()
                .filter(|(_, row)| row.role == RowRole::ToolStep)
                .nth(kept_steps.get() - 1)
                .map(|(index, _)| index)?;
            later[..first_kept_step]
                .iter()
                .rev()
                .find_map(|row| row.seq)
        });
        let compacted_after = later
            .iter()
            .rev()
            .find_map(|row| row.seq)
            .unwrap_or(kept_from);
        Some(CutSeqs {
            kept_from,
            steps_folded_through,
            compacted_after,
        })
    }

    /// Drop the newest turn. `false` when that turn was never stored, so an
    /// undo on the branch removed an older one.
    fn forget_last_turn(&mut self) -> bool {
        let Some(question) = self
            .rows
            .iter()
            .rposition(|row| row.role == RowRole::Question)
        else {
            return true;
        };
        let stored = self.rows[question].seq.is_some();
        self.rows.truncate(question);
        stored
    }
}

/// A conversation's [`SeqLedger`], shared with the transcript writer. Unread
/// until the writer builds it from the branch at the conversation's next turn.
#[derive(Debug, Clone, Default)]
pub(super) struct SharedLedger(Arc<Mutex<Option<SeqLedger>>>);

impl SharedLedger {
    /// The ledger of a conversation starting on an empty branch.
    pub(super) fn of_new_branch() -> Self {
        Self(Arc::new(Mutex::new(Some(SeqLedger::default()))))
    }

    pub(super) fn is_read(&self) -> bool {
        self.0.lock().is_some()
    }

    pub(super) fn replace(&self, ledger: SeqLedger) {
        *self.0.lock() = Some(ledger);
    }

    /// Forget the rows, for a conversation whose history was replaced from the store.
    pub(super) fn invalidate(&self) {
        *self.0.lock() = None;
    }

    pub(super) fn record(&self, role: RowRole, seq: Option<Seq>) {
        if let Some(ledger) = self.0.lock().as_mut() {
            ledger.record(role, seq);
        }
    }

    pub(super) fn cut_at(&self, cut: CompactionCut) -> Option<CutSeqs> {
        self.0.lock().as_mut()?.cut_at(cut)
    }

    /// Follow an undo of the conversation's newest turn.
    pub(super) fn forget_last_turn(&self) {
        let mut ledger = self.0.lock();
        if ledger.as_mut().is_some_and(|l| !l.forget_last_turn()) {
            *ledger = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use super::RowRole::{Other, Question, ToolStep};
    use super::*;

    fn cut(kept_turns: usize, kept_steps: Option<usize>) -> CompactionCut {
        CompactionCut {
            kept_turns: NonZeroUsize::new(kept_turns).unwrap(),
            kept_steps: kept_steps.and_then(NonZeroUsize::new),
        }
    }

    fn ledger(rows: &[(RowRole, Option<i64>)]) -> SeqLedger {
        let mut ledger = SeqLedger::default();
        for (role, seq) in rows {
            ledger.record(*role, seq.map(Seq));
        }
        ledger
    }

    #[test]
    fn a_cut_counts_messages_whose_write_failed() {
        let rows = [
            (Question, Some(0)),
            (Other, Some(1)),
            (Question, None),
            (Other, None),
            (Question, Some(2)),
            (ToolStep, Some(3)),
            (Other, Some(4)),
            (ToolStep, None),
            (Other, Some(5)),
            (ToolStep, Some(6)),
            (Other, Some(7)),
        ];

        assert_eq!(
            ledger(&rows).cut_at(cut(1, Some(2))),
            Some(CutSeqs {
                kept_from: Seq(2),
                steps_folded_through: Some(Seq(4)),
                compacted_after: Seq(7),
            }),
            "the unsaved step is one of the two kept"
        );
        assert_eq!(
            ledger(&rows).cut_at(cut(2, None)),
            None,
            "the second newest turn has no stored question to keep from"
        );
        assert_eq!(
            ledger(&rows).cut_at(cut(3, None)).map(|c| c.kept_from),
            Some(Seq(0)),
            "the unsaved turn is one of the three kept"
        );
    }
}
