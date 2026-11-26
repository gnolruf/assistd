//! The sub-agent runs of the tool call in flight, recorded from what the call and its
//! sub-agents stream into the runs stored with the call's tool result.

use std::time::Instant;

use assistd_ipc::{Access, AgentId, LlmEvent, RelayStepEnd, RelayStepPlan};
use assistd_memory::{
    AgentRunEnd, AgentRunRecord, BranchSummary, PersistedAgentRun, PersistedMessage, RunMessage,
    Seq,
};
use assistd_utils::time::saturating_millis;

use super::messages::{StepDraft, tool_result_message};
use super::seq_ledger::{RowRole, SeqLedger};

/// How a run still going when its tool call ended is stored.
const STOPPED_MESSAGE: &str = "stopped before it reported";

/// The runs the tool call in flight planned and what each has done so far.
pub(crate) struct AgentRunsRecorder {
    runs: Vec<RunRecorder>,
}

struct RunRecorder {
    task: String,
    agent: String,
    progress: RunProgress,
    tokens: Option<u32>,
    step: StepDraft,
    messages: Vec<RunMessage>,
    /// The seq of each of `messages`, which is its index.
    ledger: SeqLedger,
    summary: Option<BranchSummary>,
}

enum RunProgress {
    Pending,
    /// Sub-agent `agent` is on it.
    Running {
        agent: AgentId,
        started_at: Instant,
    },
    Ended {
        end: AgentRunEnd,
        duration_ms: u64,
    },
}

impl AgentRunsRecorder {
    pub(crate) fn planned(plan: &[RelayStepPlan]) -> Self {
        Self {
            runs: plan.iter().map(RunRecorder::pending).collect(),
        }
    }

    /// Hand run `step`, counted from 1, to sub-agent `agent`; an unknown run is ignored.
    pub(crate) fn start(&mut self, step: u32, agent: AgentId) {
        if let Some(run) = self.run_mut(step) {
            run.progress = RunProgress::Running {
                agent,
                started_at: Instant::now(),
            };
        }
    }

    /// End run `step`, counted from 1, if it is going.
    pub(crate) fn end(&mut self, step: u32, end: &RelayStepEnd, duration_ms: u64) {
        let Some(run) = self.run_mut(step) else {
            return;
        };
        if let RunProgress::Running { .. } = run.progress {
            run.progress = RunProgress::Ended {
                end: stored_end(end),
                duration_ms,
            };
        }
    }

    /// Record `event` in the history of the run sub-agent `agent` is on, if any.
    pub(crate) fn record(&mut self, agent: AgentId, event: &LlmEvent) {
        let running = self.runs.iter_mut().find(|run| {
            matches!(run.progress, RunProgress::Running { agent: runner, .. } if runner == agent)
        });
        if let Some(run) = running {
            run.record(event);
        }
    }

    fn run_mut(&mut self, step: u32) -> Option<&mut RunRecorder> {
        let index = usize::try_from(step).ok()?.checked_sub(1)?;
        self.runs.get_mut(index)
    }

    /// The runs as stored; one still going is stored as stopped.
    pub(crate) fn finish(self) -> Vec<PersistedAgentRun> {
        self.runs.into_iter().map(RunRecorder::finish).collect()
    }
}

impl RunRecorder {
    fn pending(plan: &RelayStepPlan) -> Self {
        Self {
            task: plan.task.clone(),
            agent: plan.agent.clone(),
            progress: RunProgress::Pending,
            tokens: None,
            step: StepDraft::default(),
            messages: Vec::new(),
            ledger: SeqLedger::default(),
            summary: None,
        }
    }

    fn record(&mut self, event: &LlmEvent) {
        match event {
            LlmEvent::Task { text } => self.push(PersistedMessage::user(text.clone())),
            LlmEvent::Delta { text } => self.step.push_text(text),
            LlmEvent::ReasoningDelta { text } => self.step.push_reasoning(text),
            LlmEvent::StepDiscarded => self.step.discard(),
            LlmEvent::ContextSize { total, .. } => self.tokens = Some(*total),
            LlmEvent::ToolCallsRequested { calls } => {
                let message = self.step.requesting(calls);
                self.push(message);
            }
            LlmEvent::ToolResult {
                id,
                name,
                result,
                read_host,
                accesses,
            } => {
                let message = tool_result_message(id, name, result, *read_host);
                self.push_touching(message, accesses.clone());
            }
            LlmEvent::HistoryCompacted {
                summary,
                cut,
                elapsed,
            } => {
                let cut = self.ledger.cut_at(*cut);
                self.summary =
                    cut.map(|seqs| seqs.summary(summary.clone(), saturating_millis(*elapsed)));
            }
            LlmEvent::Interrupted { reply } => {
                let message = self.step.interrupted(reply.clone());
                self.push(message);
            }
            LlmEvent::Done => {
                if let Some(message) = self.step.reply() {
                    self.push(message);
                }
            }
            LlmEvent::StepStarted
            | LlmEvent::ToolCall { .. }
            | LlmEvent::RelayPlanned { .. }
            | LlmEvent::RelayStepStarted { .. }
            | LlmEvent::RelayStepEnded { .. }
            | LlmEvent::FromAgent { .. }
            | LlmEvent::Status { .. } => {}
        }
    }

    fn push(&mut self, message: PersistedMessage) {
        self.push_touching(message, Vec::new());
    }

    /// Append `message`, whose call touched `accesses`, to the run's history.
    fn push_touching(&mut self, message: PersistedMessage, accesses: Vec<Access>) {
        let seq = Seq(i64::try_from(self.messages.len()).unwrap_or(i64::MAX));
        self.ledger.record(RowRole::of(&message), Some(seq));
        self.messages.push(RunMessage { message, accesses });
    }

    fn finish(self) -> PersistedAgentRun {
        let ended = match self.progress {
            RunProgress::Pending => None,
            RunProgress::Running { started_at, .. } => Some((
                AgentRunEnd::Failed {
                    message: STOPPED_MESSAGE.into(),
                },
                saturating_millis(started_at.elapsed()),
            )),
            RunProgress::Ended { end, duration_ms } => Some((end, duration_ms)),
        };
        PersistedAgentRun {
            task: self.task,
            agent: self.agent,
            record: ended.map(|(end, duration_ms)| AgentRunRecord {
                end,
                duration_ms,
                tokens: self.tokens,
                messages: self.messages,
                summary: self.summary,
            }),
        }
    }
}

fn stored_end(end: &RelayStepEnd) -> AgentRunEnd {
    match end {
        RelayStepEnd::Answered { answer_file } => AgentRunEnd::Answered {
            answer_file: answer_file.clone(),
        },
        RelayStepEnd::Failed { message } => AgentRunEnd::Failed {
            message: message.clone(),
        },
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;
    use std::time::Duration;

    use assistd_ipc::{CompactionCut, ToolCall, ToolReport};
    use assistd_memory::PersistedRole;
    use serde_json::json;

    use super::*;

    fn plan() -> Vec<RelayStepPlan> {
        ["fetch the notes", "compare them", "summarise"]
            .into_iter()
            .map(|task| RelayStepPlan {
                task: task.into(),
                agent: "file_review".into(),
            })
            .collect()
    }

    fn delta(text: &str) -> LlmEvent {
        LlmEvent::Delta { text: text.into() }
    }

    fn calls_requested(id: &str) -> LlmEvent {
        LlmEvent::ToolCallsRequested {
            calls: vec![ToolCall {
                id: id.into(),
                name: "run".into(),
                arguments: json!({ "command": "grep -rn x /in" }),
            }],
        }
    }

    fn tool_result(id: &str) -> LlmEvent {
        LlmEvent::ToolResult {
            id: id.into(),
            name: "run".into(),
            result: Box::new(ToolReport {
                output: "none".into(),
                exit_code: 1,
                duration_ms: 12,
                ..ToolReport::default()
            }),
            read_host: true,
            accesses: vec![Access::Read("/in/a".into())],
        }
    }

    #[test]
    fn a_run_is_its_messages_without_a_discarded_step_cut_where_it_compacted_and_stopped_if_unfinished()
     {
        let mut runs = AgentRunsRecorder::planned(&plan());
        let (first, second) = (AgentId(1), AgentId(2));
        let first_events = [
            LlmEvent::Task {
                text: "fetch the notes, handed over".into(),
            },
            LlmEvent::ReasoningDelta {
                text: "grep first".into(),
            },
            calls_requested("c-1"),
            tool_result("c-1"),
            delta("Nothi"),
            LlmEvent::StepDiscarded,
            calls_requested("c-2"),
            tool_result("c-2"),
            LlmEvent::HistoryCompacted {
                summary: "grep found nothing".into(),
                cut: CompactionCut {
                    kept_turns: NonZeroUsize::MIN,
                    kept_steps: Some(NonZeroUsize::MIN),
                },
                elapsed: Duration::from_millis(900),
            },
            delta("Nothing "),
            delta("found."),
            LlmEvent::ContextSize {
                tokens: 3000,
                total: 4100,
            },
            LlmEvent::Done,
        ];
        runs.start(1, first);
        first_events
            .iter()
            .for_each(|event| runs.record(first, event));
        let answered = RelayStepEnd::Answered {
            answer_file: "/data/relay-1.txt".into(),
        };
        runs.end(1, &answered, 41_000);
        runs.record(first, &delta("after its end"));
        runs.start(2, second);
        runs.record(second, &calls_requested("c-3"));
        let stored = runs.finish();

        let first = stored[0].record.as_ref().expect("run 1 ran");
        assert_eq!((first.duration_ms, first.tokens), (41_000, Some(4100)));
        let history: Vec<_> = first
            .messages
            .iter()
            .map(|m| {
                (
                    m.message.role(),
                    m.message.content.as_str(),
                    m.accesses.len(),
                )
            })
            .collect();
        assert_eq!(
            history,
            [
                (PersistedRole::User, "fetch the notes, handed over", 0),
                (PersistedRole::Assistant, "", 0),
                (PersistedRole::Tool, "none", 1),
                (PersistedRole::Assistant, "", 0),
                (PersistedRole::Tool, "none", 1),
                (PersistedRole::Assistant, "Nothing found.", 0),
            ]
        );
        assert_eq!(
            first.summary,
            Some(BranchSummary {
                kept_from_seq: Seq(0),
                steps_folded_through: Some(Seq(2)),
                compacted_after_seq: Seq(4),
                content: "grep found nothing".into(),
                duration_ms: 900,
            })
        );

        let second = stored[1].record.as_ref().expect("run 2 started");
        assert_eq!(
            second.end,
            AgentRunEnd::Failed {
                message: STOPPED_MESSAGE.into()
            }
        );
        assert_eq!(second.messages.len(), 1);
        assert_eq!(stored[2].record, None, "run 3 never started");
    }
}
