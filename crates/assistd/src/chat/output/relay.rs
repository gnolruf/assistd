//! A relay's sub-agent steps, drawn inside the `relay` tool block that ran them.

use std::time::{Duration, Instant};

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};

use assistd_ipc::RelayProfile;

use super::{elapsed_text, push_barred, tool_call_style, tool_result_style, tool_running_style};
use crate::client::terminal_text::escape_controls_single_line;

/// Characters of a step's task shown in its header.
const TASK_PREVIEW_CHARS: usize = 160;

/// One plan step: the sub-agent it started and what that sub-agent did.
#[derive(Debug, Clone)]
pub(in crate::chat) struct RelayStepTrace {
    task: String,
    profile: RelayProfile,
    step: u32,
    steps: u32,
    entries: Vec<RelayEntry>,
    outcome: Option<RelayStepOutcome>,
}

/// One row of what a sub-agent did, in the order it happened.
#[derive(Debug, Clone)]
enum RelayEntry {
    /// A tool call; `result` is its exit code and duration once it finished.
    Call {
        command: String,
        result: Option<(i32, u64)>,
    },
    /// A compaction of its history, with how long it took once it finished.
    Compaction {
        started_at: Instant,
        took: Option<Duration>,
    },
}

/// How a relay step ended.
#[derive(Debug, Clone)]
pub(in crate::chat) enum RelayStepOutcome {
    Answered {
        answer_chars: u32,
        tool_steps: u32,
        truncated: bool,
    },
    Failed {
        message: String,
    },
}

impl RelayStepTrace {
    pub(super) fn new(task: &str, profile: RelayProfile, step: u32, steps: u32) -> Self {
        Self {
            task: preview(task),
            profile,
            step,
            steps,
            entries: Vec::new(),
            outcome: None,
        }
    }

    pub(super) fn begin_call(&mut self, command: &str) {
        self.entries.push(RelayEntry::Call {
            command: escape_controls_single_line(command).into_owned(),
            result: None,
        });
    }

    /// Finish the latest unanswered call; a result with none is dropped.
    pub(super) fn finish_call(&mut self, exit_code: i32, duration_ms: u64) {
        let unanswered = self.entries.iter_mut().rev().find_map(|entry| match entry {
            RelayEntry::Call { result, .. } if result.is_none() => Some(result),
            _ => None,
        });
        if let Some(result) = unanswered {
            *result = Some((exit_code, duration_ms));
        }
    }

    pub(super) fn begin_compaction(&mut self) {
        self.entries.push(RelayEntry::Compaction {
            started_at: Instant::now(),
            took: None,
        });
    }

    /// Finish the latest unfinished compaction; with none, it is dropped.
    pub(super) fn finish_compaction(&mut self, duration_ms: u64) {
        let unfinished = self.entries.iter_mut().rev().find_map(|entry| match entry {
            RelayEntry::Compaction { took, .. } if took.is_none() => Some(took),
            _ => None,
        });
        if let Some(took) = unfinished {
            *took = Some(Duration::from_millis(duration_ms));
        }
    }

    fn calls(&self) -> usize {
        self.entries
            .iter()
            .filter(|entry| matches!(entry, RelayEntry::Call { .. }))
            .count()
    }

    pub(super) fn finish(&mut self, outcome: RelayStepOutcome) {
        self.outcome = Some(outcome);
    }
}

/// Draw `steps` behind `bar`; `collapsed` hides each step's rows behind a count of its calls.
pub(super) fn push_relay_steps(
    out: &mut Vec<Line<'static>>,
    bar: &Span<'static>,
    steps: &[RelayStepTrace],
    collapsed: bool,
    inner_w: u16,
) {
    for step in steps {
        let header = format!(
            "⇢ sub-agent {}/{} · {} · {}",
            step.step,
            step.steps,
            step.profile.as_str(),
            step.task
        );
        push_barred(out, bar, &header, relay_style(), inner_w);
        if collapsed && !step.entries.is_empty() {
            let calls = match step.calls() {
                1 => "1 tool call".to_string(),
                count => format!("{count} tool calls"),
            };
            let count = format!("  {calls} (Tab to expand)");
            push_barred(out, bar, &count, hidden_style(), inner_w);
        } else {
            for entry in &step.entries {
                let (row, style) = entry_row(entry);
                push_barred(out, bar, &row, style, inner_w);
            }
        }
        if let Some(outcome) = &step.outcome {
            let (text, style) = outcome_row(outcome);
            push_barred(out, bar, &text, style, inner_w);
        }
    }
}

fn entry_row(entry: &RelayEntry) -> (String, Style) {
    match entry {
        RelayEntry::Call {
            command,
            result: None,
        } => (format!("  $ {command} …"), tool_running_style()),
        RelayEntry::Call {
            command,
            result: Some((exit_code, duration_ms)),
        } => (
            format!("  $ {command}  [exit:{exit_code} | {duration_ms}ms]"),
            if *exit_code == 0 {
                tool_result_style()
            } else {
                Style::default().fg(Color::Red)
            },
        ),
        RelayEntry::Compaction {
            started_at,
            took: None,
        } => (
            format!(
                "  compacting history… ({})",
                elapsed_text(started_at.elapsed())
            ),
            tool_running_style(),
        ),
        RelayEntry::Compaction {
            took: Some(took), ..
        } => (
            format!("  compacted history in {}", elapsed_text(*took)),
            hidden_style(),
        ),
    }
}

fn outcome_row(outcome: &RelayStepOutcome) -> (String, Style) {
    match outcome {
        RelayStepOutcome::Answered {
            answer_chars,
            tool_steps,
            truncated,
        } => {
            let cut = if *truncated { ", cut to fit" } else { "" };
            (
                format!("⇠ reported {answer_chars} chars after {tool_steps} tool steps{cut}"),
                tool_call_style(),
            )
        }
        RelayStepOutcome::Failed { message } => (
            format!("✗ {}", escape_controls_single_line(message)),
            Style::default().fg(Color::Red),
        ),
    }
}

/// `task` on one line, cut to [`TASK_PREVIEW_CHARS`].
fn preview(task: &str) -> String {
    let line = escape_controls_single_line(task);
    match line.char_indices().nth(TASK_PREVIEW_CHARS) {
        Some((cut, _)) => format!("{}…", &line[..cut]),
        None => line.into_owned(),
    }
}

fn relay_style() -> Style {
    Style::default()
        .fg(Color::Magenta)
        .add_modifier(Modifier::BOLD)
}

fn hidden_style() -> Style {
    Style::default()
        .fg(Color::DarkGray)
        .add_modifier(Modifier::ITALIC)
}
