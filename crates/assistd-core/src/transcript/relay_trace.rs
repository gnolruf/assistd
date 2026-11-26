//! A relay's sub-agent trace: recorded from what its tool call and its sub-agents stream,
//! stored with the call's tool result, and replayed with branch history.

use std::time::Instant;

use assistd_ipc::{
    AgentId, LlmEvent, RelayStepEnd, RelayStepPlan, RelayStepRecord, RelayStepRun,
    RelayToolOutcome, RelayTraceEntry, StatusKind, ToolReport,
};
use assistd_memory::{
    PersistedRelayEnd, PersistedRelayEntry, PersistedRelayRun, PersistedRelayStep, ToolOutcome,
};
use assistd_utils::time::saturating_millis;

/// How a step still running when its relay's tool call ended is stored.
const STOPPED_MESSAGE: &str = "stopped before it reported";

/// The steps the running relay planned and what each has done so far.
pub(crate) struct RelayTraceBuilder {
    steps: Vec<StepBuilder>,
}

struct StepBuilder {
    task: String,
    agent: String,
    progress: StepProgress,
    tokens: Option<u32>,
    entries: Vec<PersistedRelayEntry>,
    /// When the reasoning entry being streamed began.
    reasoning_since: Option<Instant>,
    /// When the compaction entry still unfinished began.
    compacting_since: Option<Instant>,
}

enum StepProgress {
    Pending,
    /// Sub-agent `agent` is running it.
    Running {
        agent: AgentId,
        started_at: Instant,
    },
    Ended {
        end: PersistedRelayEnd,
        duration_ms: u64,
    },
}

impl RelayTraceBuilder {
    pub(crate) fn planned(plan: &[RelayStepPlan]) -> Self {
        Self {
            steps: plan.iter().map(StepBuilder::pending).collect(),
        }
    }

    /// Hand step `step`, counted from 1, to sub-agent `agent`; an unknown step is ignored.
    pub(crate) fn start(&mut self, step: u32, agent: AgentId) {
        if let Some(step) = self.step_mut(step) {
            step.progress = StepProgress::Running {
                agent,
                started_at: Instant::now(),
            };
        }
    }

    /// End step `step`, counted from 1, if it is running.
    pub(crate) fn end(&mut self, step: u32, end: &RelayStepEnd) {
        if let Some(step) = self.step_mut(step) {
            step.end(match end {
                RelayStepEnd::Answered { answer_file } => PersistedRelayEnd::Answered {
                    answer_file: answer_file.clone(),
                },
                RelayStepEnd::Failed { message } => PersistedRelayEnd::Failed {
                    message: message.clone(),
                },
            });
        }
    }

    /// Record `event` against the step sub-agent `agent` is running, if any.
    pub(crate) fn record(&mut self, agent: AgentId, event: &LlmEvent) {
        let running = self.steps.iter_mut().find(|step| {
            matches!(step.progress, StepProgress::Running { agent: runner, .. } if runner == agent)
        });
        if let Some(step) = running {
            step.record(event);
        }
    }

    fn step_mut(&mut self, step: u32) -> Option<&mut StepBuilder> {
        let index = usize::try_from(step).ok()?.checked_sub(1)?;
        self.steps.get_mut(index)
    }

    /// The steps as stored; one still running is stored as stopped.
    pub(crate) fn finish(self) -> Vec<PersistedRelayStep> {
        self.steps.into_iter().map(StepBuilder::finish).collect()
    }
}

impl StepBuilder {
    fn pending(plan: &RelayStepPlan) -> Self {
        Self {
            task: plan.task.clone(),
            agent: plan.agent.clone(),
            progress: StepProgress::Pending,
            tokens: None,
            entries: Vec::new(),
            reasoning_since: None,
            compacting_since: None,
        }
    }

    fn record(&mut self, event: &LlmEvent) {
        match event {
            LlmEvent::Delta { text } => self.push_text(text),
            LlmEvent::ReasoningDelta { text } => self.push_reasoning(text),
            LlmEvent::StepDiscarded => self.discard_step(),
            LlmEvent::ContextSize { total, .. } => self.tokens = Some(*total),
            LlmEvent::ToolCall {
                name, arguments, ..
            } => self.entries.push(PersistedRelayEntry::Tool {
                name: name.clone(),
                arguments: arguments.clone(),
                output: String::new(),
                outcome: None,
            }),
            LlmEvent::ToolResult { result, .. } => self.finish_tool(result),
            LlmEvent::Status {
                event: StatusKind::CompactingHistory,
                ..
            } => {
                self.compacting_since = Some(Instant::now());
                self.entries
                    .push(PersistedRelayEntry::Compaction { duration_ms: None });
            }
            LlmEvent::Status {
                event: StatusKind::HistoryCompacted,
                ..
            } => self.finish_compaction(),
            _ => {}
        }
    }

    /// Drop the text and reasoning of the step in flight, which follow every other entry.
    fn discard_step(&mut self) {
        while let Some(PersistedRelayEntry::Text(_) | PersistedRelayEntry::Reasoning { .. }) =
            self.entries.last()
        {
            self.entries.pop();
        }
    }

    /// Extend the trailing text entry, or start one.
    fn push_text(&mut self, delta: &str) {
        if delta.is_empty() {
            return;
        }
        match self.entries.last_mut() {
            Some(PersistedRelayEntry::Text(text)) => text.push_str(delta),
            _ => self
                .entries
                .push(PersistedRelayEntry::Text(delta.to_owned())),
        }
    }

    /// Extend the trailing reasoning entry and its duration, or start one.
    fn push_reasoning(&mut self, delta: &str) {
        if delta.is_empty() {
            return;
        }
        if let Some(PersistedRelayEntry::Reasoning { text, duration_ms }) = self.entries.last_mut()
            && let Some(since) = self.reasoning_since
        {
            text.push_str(delta);
            *duration_ms = saturating_millis(since.elapsed());
            return;
        }
        self.reasoning_since = Some(Instant::now());
        self.entries.push(PersistedRelayEntry::Reasoning {
            text: delta.to_owned(),
            duration_ms: 0,
        });
    }

    /// Fill in the latest unfinished tool call from `result`.
    fn finish_tool(&mut self, result: &ToolReport) {
        let unfinished = self.entries.iter_mut().rev().find_map(|entry| match entry {
            PersistedRelayEntry::Tool {
                output, outcome, ..
            } if outcome.is_none() => Some((output, outcome)),
            _ => None,
        });
        if let Some((output, outcome)) = unfinished {
            output.clone_from(&result.output);
            *outcome = Some(ToolOutcome {
                exit_code: i64::from(result.exit_code),
                duration_ms: result.duration_ms,
            });
        }
    }

    /// Time the latest unfinished compaction from its announcement.
    fn finish_compaction(&mut self) {
        let unfinished = self.entries.iter_mut().rev().find_map(|entry| match entry {
            PersistedRelayEntry::Compaction { duration_ms } if duration_ms.is_none() => {
                Some(duration_ms)
            }
            _ => None,
        });
        if let (Some(duration_ms), Some(since)) = (unfinished, self.compacting_since.take()) {
            *duration_ms = Some(saturating_millis(since.elapsed()));
        }
    }

    fn end(&mut self, end: PersistedRelayEnd) {
        if let StepProgress::Running { started_at, .. } = self.progress {
            self.progress = StepProgress::Ended {
                end,
                duration_ms: saturating_millis(started_at.elapsed()),
            };
        }
    }

    fn finish(self) -> PersistedRelayStep {
        let ended = match self.progress {
            StepProgress::Pending => None,
            StepProgress::Running { started_at, .. } => Some((
                PersistedRelayEnd::Failed {
                    message: STOPPED_MESSAGE.into(),
                },
                saturating_millis(started_at.elapsed()),
            )),
            StepProgress::Ended { end, duration_ms } => Some((end, duration_ms)),
        };
        PersistedRelayStep {
            task: self.task,
            agent: self.agent,
            run: ended.map(|(end, duration_ms)| PersistedRelayRun {
                end,
                duration_ms,
                tokens: self.tokens,
                entries: self.entries,
            }),
        }
    }
}

/// `steps` as replayed to a client with their `relay` tool row.
pub(crate) fn wire_relay_steps(steps: Vec<PersistedRelayStep>) -> Vec<RelayStepRecord> {
    steps
        .into_iter()
        .map(|step| RelayStepRecord {
            task: step.task,
            agent: step.agent,
            run: step.run.map(wire_relay_run),
        })
        .collect()
}

fn wire_relay_run(run: PersistedRelayRun) -> RelayStepRun {
    RelayStepRun {
        end: match run.end {
            PersistedRelayEnd::Answered { answer_file } => RelayStepEnd::Answered { answer_file },
            PersistedRelayEnd::Failed { message } => RelayStepEnd::Failed { message },
        },
        duration_ms: run.duration_ms,
        tokens: run.tokens,
        entries: run.entries.into_iter().map(wire_relay_entry).collect(),
    }
}

fn wire_relay_entry(entry: PersistedRelayEntry) -> RelayTraceEntry {
    match entry {
        PersistedRelayEntry::Text(text) => RelayTraceEntry::Text { text },
        PersistedRelayEntry::Reasoning { text, duration_ms } => {
            RelayTraceEntry::Reasoning { text, duration_ms }
        }
        PersistedRelayEntry::Tool {
            name,
            arguments,
            output,
            outcome,
        } => RelayTraceEntry::Tool {
            name,
            args: arguments,
            output,
            outcome: outcome.map(|o| RelayToolOutcome {
                exit_code: o.exit_code,
                duration_ms: o.duration_ms,
            }),
        },
        PersistedRelayEntry::Compaction { duration_ms } => {
            RelayTraceEntry::Compaction { duration_ms }
        }
    }
}

#[cfg(test)]
mod tests {
    use assistd_ipc::{Component, StatusSeverity};
    use serde_json::json;

    use super::*;

    fn plan() -> Vec<RelayStepPlan> {
        ["fetch the notes", "compare them", "summarise"]
            .into_iter()
            .enumerate()
            .map(|(index, task)| RelayStepPlan {
                task: task.into(),
                agent: if index == 0 {
                    "web_research"
                } else {
                    "file_review"
                }
                .into(),
            })
            .collect()
    }

    fn delta(text: &str) -> LlmEvent {
        LlmEvent::Delta { text: text.into() }
    }

    fn reasoning(text: &str) -> LlmEvent {
        LlmEvent::ReasoningDelta { text: text.into() }
    }

    fn tool_call(command: &str) -> LlmEvent {
        LlmEvent::ToolCall {
            id: "c".into(),
            name: "run".into(),
            arguments: json!({ "command": command }),
        }
    }

    fn compacting() -> LlmEvent {
        LlmEvent::Status {
            severity: StatusSeverity::Info,
            component: Component::Llm,
            event: StatusKind::CompactingHistory,
            message: String::new(),
        }
    }

    #[test]
    fn trace_merges_deltas_pairs_results_drops_a_discarded_step_and_stores_an_unfinished_step_as_stopped()
     {
        let mut trace = RelayTraceBuilder::planned(&plan());
        let (first, second) = (AgentId(1), AgentId(2));
        let first_events = [
            reasoning("grep "),
            reasoning("first"),
            tool_call("grep -rn x /in"),
            LlmEvent::ToolResult {
                id: "c".into(),
                name: "run".into(),
                result: Box::new(ToolReport {
                    output: "none".into(),
                    exit_code: 1,
                    duration_ms: 12,
                    ..ToolReport::default()
                }),
                read_host: false,
                accesses: Vec::new(),
            },
            reasoning("cut off"),
            delta("Nothi"),
            LlmEvent::StepDiscarded,
            delta("Nothing "),
            delta("found."),
            LlmEvent::ContextSize {
                tokens: 3000,
                total: 4100,
            },
        ];
        trace.start(1, first);
        first_events
            .iter()
            .for_each(|event| trace.record(first, event));
        trace.end(
            1,
            &RelayStepEnd::Answered {
                answer_file: "/data/relay-1.txt".into(),
            },
        );
        trace.record(first, &delta("after its end"));
        trace.start(2, second);
        trace.record(second, &tool_call("cat /in/a"));
        trace.record(second, &compacting());
        let steps = trace.finish();

        let first = steps[0].run.as_ref().expect("step 1 ran");
        assert_eq!(
            first.end,
            PersistedRelayEnd::Answered {
                answer_file: "/data/relay-1.txt".into()
            }
        );
        assert_eq!(first.tokens, Some(4100));
        let PersistedRelayEntry::Reasoning { text, .. } = &first.entries[0] else {
            panic!("reasoning first: {:?}", first.entries);
        };
        assert_eq!(text, "grep first");
        assert_eq!(
            first.entries[1..],
            [
                PersistedRelayEntry::Tool {
                    name: "run".into(),
                    arguments: json!({"command": "grep -rn x /in"}),
                    output: "none".into(),
                    outcome: Some(ToolOutcome {
                        exit_code: 1,
                        duration_ms: 12,
                    }),
                },
                PersistedRelayEntry::Text("Nothing found.".into()),
            ]
        );

        let second = steps[1].run.as_ref().expect("step 2 started");
        assert_eq!(
            second.end,
            PersistedRelayEnd::Failed {
                message: STOPPED_MESSAGE.into()
            }
        );
        assert!(matches!(
            second.entries[..],
            [
                PersistedRelayEntry::Tool { outcome: None, .. },
                PersistedRelayEntry::Compaction { duration_ms: None },
            ]
        ));
        assert_eq!(steps[2].run, None, "step 3 never ran");
    }
}
