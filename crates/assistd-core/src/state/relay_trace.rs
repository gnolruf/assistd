//! A relay's sub-agent trace: recorded from the progress its tool call streams,
//! stored with the call's tool result, and replayed with branch history.

use std::time::{Duration, Instant};

use serde_json::Value;

use assistd_ipc::{
    RelayEvent, RelayProfile, RelayStepEnd, RelayStepPlan, RelayStepRecord, RelayStepRun,
    RelayToolOutcome, RelayTraceEntry,
};
use assistd_memory::{
    PersistedRelayEnd, PersistedRelayEntry, PersistedRelayProfile, PersistedRelayRun,
    PersistedRelayStep, ToolOutcome,
};

/// How a step still running when its relay's tool call ended is stored.
const STOPPED_MESSAGE: &str = "stopped before it reported";

/// The steps the running relay planned and what each has done so far.
pub(super) struct RelayTraceBuilder {
    steps: Vec<StepBuilder>,
}

struct StepBuilder {
    task: String,
    profile: PersistedRelayProfile,
    progress: StepProgress,
    tokens: Option<u32>,
    entries: Vec<PersistedRelayEntry>,
    /// When the reasoning entry being streamed began.
    reasoning_since: Option<Instant>,
}

enum StepProgress {
    Pending,
    Running {
        started_at: Instant,
    },
    Ended {
        end: PersistedRelayEnd,
        duration_ms: u64,
    },
}

impl RelayTraceBuilder {
    pub(super) fn planned(plan: &[RelayStepPlan]) -> Self {
        Self {
            steps: plan.iter().map(StepBuilder::pending).collect(),
        }
    }

    /// Record `event` against the step it belongs to; a new plan starts over.
    pub(super) fn record(&mut self, event: &RelayEvent) {
        match event {
            RelayEvent::Planned { steps } => *self = Self::planned(steps),
            RelayEvent::Started { step } => {
                let index = usize::try_from(*step).ok().and_then(|s| s.checked_sub(1));
                if let Some(step) = index.and_then(|i| self.steps.get_mut(i)) {
                    step.progress = StepProgress::Running {
                        started_at: Instant::now(),
                    };
                }
            }
            event => {
                let running = self
                    .steps
                    .iter_mut()
                    .find(|step| matches!(step.progress, StepProgress::Running { .. }));
                if let Some(step) = running {
                    step.record(event);
                }
            }
        }
    }

    /// The steps as stored; one still running is stored as stopped.
    pub(super) fn finish(self) -> Vec<PersistedRelayStep> {
        self.steps.into_iter().map(StepBuilder::finish).collect()
    }
}

impl StepBuilder {
    fn pending(plan: &RelayStepPlan) -> Self {
        Self {
            task: plan.task.clone(),
            profile: match plan.profile {
                RelayProfile::Read => PersistedRelayProfile::Read,
                RelayProfile::Fetch => PersistedRelayProfile::Fetch,
            },
            progress: StepProgress::Pending,
            tokens: None,
            entries: Vec::new(),
            reasoning_since: None,
        }
    }

    fn record(&mut self, event: &RelayEvent) {
        match event {
            RelayEvent::Planned { .. } | RelayEvent::Started { .. } => {}
            RelayEvent::Delta { text } => self.push_text(text),
            RelayEvent::ReasoningDelta { text } => self.push_reasoning(text),
            RelayEvent::ContextSize { total } => self.tokens = Some(*total),
            RelayEvent::ToolCall { name, args } => self.entries.push(PersistedRelayEntry::Tool {
                name: name.clone(),
                arguments: args.clone(),
                output: String::new(),
                outcome: None,
            }),
            RelayEvent::ToolResult { result, .. } => self.finish_tool(result),
            RelayEvent::Compacting => self
                .entries
                .push(PersistedRelayEntry::Compaction { duration_ms: None }),
            RelayEvent::Compacted { duration_ms } => self.finish_compaction(*duration_ms),
            RelayEvent::Answered { truncated, .. } => self.end(PersistedRelayEnd::Answered {
                truncated: *truncated,
            }),
            RelayEvent::Failed { message } => self.end(PersistedRelayEnd::Failed {
                message: message.clone(),
            }),
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
            *duration_ms = millis(since.elapsed());
            return;
        }
        self.reasoning_since = Some(Instant::now());
        self.entries.push(PersistedRelayEntry::Reasoning {
            text: delta.to_owned(),
            duration_ms: 0,
        });
    }

    /// Fill in the latest unfinished tool call from `result`, shaped as
    /// [`assistd_tools::ToolOutput::to_json`] makes it.
    fn finish_tool(&mut self, result: &Value) {
        let unfinished = self.entries.iter_mut().rev().find_map(|entry| match entry {
            PersistedRelayEntry::Tool {
                output, outcome, ..
            } if outcome.is_none() => Some((output, outcome)),
            _ => None,
        });
        if let Some((output, outcome)) = unfinished {
            result
                .get("output")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .clone_into(output);
            *outcome = Some(ToolOutcome {
                exit_code: result.get("exit_code").and_then(Value::as_i64).unwrap_or(0),
                duration_ms: result
                    .get("duration_ms")
                    .and_then(Value::as_u64)
                    .unwrap_or(0),
            });
        }
    }

    fn finish_compaction(&mut self, took_ms: u64) {
        let unfinished = self.entries.iter_mut().rev().find_map(|entry| match entry {
            PersistedRelayEntry::Compaction { duration_ms } if duration_ms.is_none() => {
                Some(duration_ms)
            }
            _ => None,
        });
        if let Some(duration_ms) = unfinished {
            *duration_ms = Some(took_ms);
        }
    }

    fn end(&mut self, end: PersistedRelayEnd) {
        if let StepProgress::Running { started_at } = self.progress {
            self.progress = StepProgress::Ended {
                end,
                duration_ms: millis(started_at.elapsed()),
            };
        }
    }

    fn finish(self) -> PersistedRelayStep {
        let ended = match self.progress {
            StepProgress::Pending => None,
            StepProgress::Running { started_at } => Some((
                PersistedRelayEnd::Failed {
                    message: STOPPED_MESSAGE.into(),
                },
                millis(started_at.elapsed()),
            )),
            StepProgress::Ended { end, duration_ms } => Some((end, duration_ms)),
        };
        PersistedRelayStep {
            task: self.task,
            profile: self.profile,
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
pub(super) fn wire_relay_steps(steps: Vec<PersistedRelayStep>) -> Vec<RelayStepRecord> {
    steps
        .into_iter()
        .map(|step| RelayStepRecord {
            task: step.task,
            profile: match step.profile {
                PersistedRelayProfile::Read => RelayProfile::Read,
                PersistedRelayProfile::Fetch => RelayProfile::Fetch,
            },
            run: step.run.map(wire_relay_run),
        })
        .collect()
}

fn wire_relay_run(run: PersistedRelayRun) -> RelayStepRun {
    RelayStepRun {
        end: match run.end {
            PersistedRelayEnd::Answered { truncated } => RelayStepEnd::Answered { truncated },
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

fn millis(elapsed: Duration) -> u64 {
    u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests;
