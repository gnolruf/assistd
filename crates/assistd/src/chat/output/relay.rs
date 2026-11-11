//! A relay's sub-agents: the compact block drawn in place of the `relay` tool
//! block, and each sub-agent's own transcript.

use std::time::{Duration, Instant};

use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use textwrap::core::display_width;

use assistd_ipc::{
    RelayEvent, RelayStepEnd, RelayStepPlan, RelayStepRecord, RelayStepRun, RelayTraceEntry,
};

use super::{
    OutputPane, ToolState, elapsed_text, exit_color, split_body, tool_block_label, tool_outcome,
    tool_output,
};
use crate::chat::text::ellipsize;
use crate::chat::theme::Theme;
use crate::chat::ui::compact_count;
use crate::client::terminal_text::{escape_controls, escape_controls_single_line};

/// Latest calls a running sub-agent shows inline; the rest are counted.
const TAIL_CALLS: usize = 3;
/// Columns of a task shown on one row.
const TASK_PREVIEW_COLS: usize = 120;
/// Columns a task keeps in a block row however narrow the pane.
const MIN_TASK_COLS: usize = 16;
const HEADER_BULLET: &str = "● ";
const HEADER_INDENT: &str = "  ";
const RESULT_FIRST: &str = "  ⎿ ";
const RESULT_REST: &str = "    ";
const STOPPED_MESSAGE: &str = "stopped before it reported";

/// Every sub-agent one `relay` call planned, in plan order.
pub(in crate::chat) struct RelayRun {
    steps: Vec<RelayStepTrace>,
}

impl RelayRun {
    pub(super) fn planned(plan: Vec<RelayStepPlan>) -> Self {
        let steps = plan.len();
        Self {
            steps: plan
                .into_iter()
                .enumerate()
                .map(|(index, planned)| RelayStepTrace::pending(planned, index + 1, steps))
                .collect(),
        }
    }

    /// A finished run rebuilt from its stored `records`.
    pub(super) fn replayed(records: Vec<RelayStepRecord>) -> Self {
        let steps = records.len();
        Self {
            steps: records
                .into_iter()
                .enumerate()
                .map(|(index, record)| RelayStepTrace::replayed(record, index + 1, steps))
                .collect(),
        }
    }

    /// Start step `step`, counted from 1; an unknown step is ignored.
    fn start(&mut self, step: u32) {
        let index = usize::try_from(step).unwrap_or(usize::MAX);
        if let Some(trace) = index.checked_sub(1).and_then(|i| self.steps.get_mut(i)) {
            trace.start();
        }
    }

    /// Start the step `event` names, or apply `event` to the running step;
    /// with none running, it is dropped.
    pub(super) fn apply(&mut self, event: RelayEvent) {
        match event {
            RelayEvent::Started { step } => self.start(step),
            event => {
                if let Some(step) = self.steps.iter_mut().find(|step| step.is_running()) {
                    step.apply(event);
                }
            }
        }
    }

    /// Fail a step still running when the relay's tool call ends.
    pub(super) fn end(&mut self) {
        for step in self.steps.iter_mut().filter(|step| step.is_running()) {
            step.finish(RelayStepEnd::Failed {
                message: STOPPED_MESSAGE.into(),
            });
        }
    }

    /// Steps that have started, in plan order.
    pub(super) fn started(&self) -> impl Iterator<Item = &RelayStepTrace> {
        self.steps.iter().filter(|step| step.has_started())
    }

    pub(super) fn started_mut(&mut self) -> impl Iterator<Item = &mut RelayStepTrace> {
        self.steps.iter_mut().filter(|step| step.has_started())
    }

    fn tool_uses(&self) -> usize {
        self.steps.iter().map(RelayStepTrace::tool_uses).sum()
    }

    fn tokens(&self) -> Option<u32> {
        self.steps
            .iter()
            .filter_map(|step| step.tokens)
            .reduce(u32::saturating_add)
    }

    fn failure(&self) -> Option<&str> {
        self.steps.iter().find_map(|step| match &step.state {
            StepState::Finished {
                outcome: RelayStepEnd::Failed { message },
                ..
            } => Some(message.as_str()),
            _ => None,
        })
    }
}

/// One plan step: its task, where it stands, and what its sub-agent did.
pub(in crate::chat) struct RelayStepTrace {
    task: String,
    agent: String,
    /// Position in the plan, counted from 1, and the plan's length.
    step: usize,
    steps: usize,
    state: StepState,
    calls: Vec<RelayEntry>,
    /// Tokens its conversation has held, once reported.
    tokens: Option<u32>,
    transcript: OutputPane,
}

impl RelayStepTrace {
    fn pending(planned: RelayStepPlan, step: usize, steps: usize) -> Self {
        Self {
            task: planned.task,
            agent: planned.agent,
            step,
            steps,
            state: StepState::Pending,
            calls: Vec::new(),
            tokens: None,
            transcript: OutputPane::new(),
        }
    }

    fn replayed(record: RelayStepRecord, step: usize, steps: usize) -> Self {
        let planned = RelayStepPlan {
            task: record.task,
            agent: record.agent,
        };
        let mut trace = Self::pending(planned, step, steps);
        if let Some(run) = record.run {
            trace.replay_run(run);
        }
        trace
    }

    fn start(&mut self) {
        self.transcript.push_user(&escape_controls(&self.task));
        self.state = StepState::Running {
            started_at: Instant::now(),
        };
    }

    /// Rebuild what a finished step did, and how it ended, from its stored `run`.
    fn replay_run(&mut self, run: RelayStepRun) {
        self.transcript.push_user(&escape_controls(&self.task));
        for entry in run.entries {
            self.replay_entry(entry);
        }
        self.transcript.finish_assistant();
        if let RelayStepEnd::Failed { message } = &run.end {
            self.transcript.push_error(&escape_controls(message));
        }
        self.tokens = run.tokens;
        self.state = StepState::Finished {
            elapsed: Duration::from_millis(run.duration_ms),
            outcome: run.end,
        };
    }

    fn replay_entry(&mut self, entry: RelayTraceEntry) {
        match entry {
            RelayTraceEntry::Text { text } => self.transcript.append_assistant(&text),
            RelayTraceEntry::Reasoning { text, duration_ms } => self
                .transcript
                .push_replayed_thinking(text, Duration::from_millis(duration_ms)),
            RelayTraceEntry::Tool {
                name,
                args,
                output,
                outcome,
            } => {
                let command = tool_block_label(name, Some(&args));
                let result = outcome.map(|outcome| {
                    let exit_code = i32::try_from(outcome.exit_code).unwrap_or(i32::MIN);
                    (exit_code, outcome.duration_ms)
                });
                self.calls.push(RelayEntry::Call {
                    command: escape_controls_single_line(&command).into_owned(),
                    result,
                });
                match result {
                    Some((exit_code, duration_ms)) => {
                        self.transcript
                            .push_tool_block(command, output, exit_code, duration_ms);
                    }
                    None => {
                        self.transcript.begin_tool_block(command);
                        self.transcript.abandon_running_tool();
                    }
                }
            }
            RelayTraceEntry::Compaction { duration_ms } => {
                let took = Duration::from_millis(duration_ms.unwrap_or(0));
                self.calls.push(RelayEntry::Compaction {
                    started_at: Instant::now(),
                    took: Some(took),
                });
                self.transcript.push_replayed_compaction(took);
            }
        }
    }

    fn apply(&mut self, event: RelayEvent) {
        match event {
            RelayEvent::Planned { .. } | RelayEvent::Started { .. } => {}
            RelayEvent::Delta { text } => {
                self.transcript.finish_thinking();
                self.transcript.append_assistant(&text);
            }
            RelayEvent::ReasoningDelta { text } => self.transcript.append_thinking(&text),
            RelayEvent::ContextSize { total } => self.tokens = Some(total),
            RelayEvent::ToolCall { name, args } => {
                let command = tool_block_label(name, Some(&args));
                self.calls.push(RelayEntry::Call {
                    command: escape_controls_single_line(&command).into_owned(),
                    result: None,
                });
                self.transcript.finish_thinking();
                self.transcript.begin_tool_block(command);
            }
            RelayEvent::ToolResult { result, .. } => {
                let (exit_code, duration_ms) = tool_outcome(&result);
                self.finish_call(exit_code, duration_ms);
                self.transcript
                    .finish_tool_block(tool_output(&result), exit_code, duration_ms);
            }
            RelayEvent::Compacting => {
                self.calls.push(RelayEntry::Compaction {
                    started_at: Instant::now(),
                    took: None,
                });
                self.transcript.begin_compaction();
            }
            RelayEvent::Compacted { duration_ms } => {
                self.finish_compaction(duration_ms);
                self.transcript.finish_compaction();
            }
            RelayEvent::Answered { .. } => self.finish(RelayStepEnd::Answered),
            RelayEvent::Failed { message } => self.finish(RelayStepEnd::Failed { message }),
        }
    }

    /// Finish the latest unanswered call; a result with none is dropped.
    fn finish_call(&mut self, exit_code: i32, duration_ms: u64) {
        let unanswered = self.calls.iter_mut().rev().find_map(|entry| match entry {
            RelayEntry::Call { result, .. } if result.is_none() => Some(result),
            _ => None,
        });
        if let Some(result) = unanswered {
            *result = Some((exit_code, duration_ms));
        }
    }

    /// Finish the latest unfinished compaction; with none, it is dropped.
    fn finish_compaction(&mut self, duration_ms: u64) {
        let unfinished = self.calls.iter_mut().rev().find_map(|entry| match entry {
            RelayEntry::Compaction { took, .. } if took.is_none() => Some(took),
            _ => None,
        });
        if let Some(took) = unfinished {
            *took = Some(Duration::from_millis(duration_ms));
        }
    }

    fn finish(&mut self, outcome: RelayStepEnd) {
        let StepState::Running { started_at } = self.state else {
            return;
        };
        self.transcript.finish_thinking();
        self.transcript.finish_assistant();
        self.transcript.abandon_running_tool();
        if let RelayStepEnd::Failed { message } = &outcome {
            self.transcript.push_error(&escape_controls(message));
        }
        self.state = StepState::Finished {
            elapsed: started_at.elapsed(),
            outcome,
        };
    }

    fn is_running(&self) -> bool {
        matches!(self.state, StepState::Running { .. })
    }

    fn has_started(&self) -> bool {
        !matches!(self.state, StepState::Pending)
    }

    fn tool_uses(&self) -> usize {
        self.calls
            .iter()
            .filter(|entry| matches!(entry, RelayEntry::Call { .. }))
            .count()
    }

    /// `● file_review · step 2/3 · <task>`, on one row.
    pub(in crate::chat) fn label(&self) -> String {
        let position = if self.steps > 1 {
            format!(" · step {}/{}", self.step, self.steps)
        } else {
            String::new()
        };
        format!(
            "{} {}{position} · {}",
            self.icon(),
            self.agent,
            ellipsize(&escape_controls_single_line(&self.task), TASK_PREVIEW_COLS)
        )
    }

    /// `Running… (3 tool uses · 12.1k tokens · 41s)`, `Done (…)` or `Failed (…)`.
    pub(in crate::chat) fn status(&self) -> String {
        let (verb, elapsed) = match &self.state {
            StepState::Pending => return "Waiting".into(),
            StepState::Running { started_at } => ("Running…", started_at.elapsed()),
            StepState::Finished {
                elapsed,
                outcome: RelayStepEnd::Answered,
            } => ("Done", *elapsed),
            StepState::Finished {
                elapsed,
                outcome: RelayStepEnd::Failed { .. },
            } => ("Failed", *elapsed),
        };
        format!("{verb} ({})", stats(self.tool_uses(), self.tokens, elapsed))
    }

    /// The style its label and status are drawn in.
    pub(in crate::chat) fn style(&self, theme: &Theme) -> Style {
        match &self.state {
            StepState::Pending => theme.meta_style().add_modifier(Modifier::DIM),
            StepState::Running { .. } => theme.meta_style(),
            StepState::Finished {
                outcome: RelayStepEnd::Answered,
                ..
            } => theme.success_style(),
            StepState::Finished {
                outcome: RelayStepEnd::Failed { .. },
                ..
            } => theme.error_style(),
        }
    }

    pub(in crate::chat) fn transcript(&self) -> &OutputPane {
        &self.transcript
    }

    pub(in crate::chat) fn transcript_mut(&mut self) -> &mut OutputPane {
        &mut self.transcript
    }

    fn icon(&self) -> char {
        match &self.state {
            StepState::Pending => '○',
            StepState::Running { .. } => '●',
            StepState::Finished {
                outcome: RelayStepEnd::Answered,
                ..
            } => '✓',
            StepState::Finished {
                outcome: RelayStepEnd::Failed { .. },
                ..
            } => '✗',
        }
    }

    /// Its calls, cut to the latest few while running unless `all`, then
    /// how it failed.
    fn detail_rows(&self, all: bool, theme: &Theme) -> Vec<(String, Style)> {
        let shown = match (&self.state, all) {
            (_, true) => self.calls.len(),
            (StepState::Running { .. }, false) => TAIL_CALLS.min(self.calls.len()),
            _ => 0,
        };
        let hidden = self.calls.len().saturating_sub(shown);
        let mut rows: Vec<(String, Style)> = self.calls[hidden..]
            .iter()
            .map(|entry| entry_row(entry, theme))
            .collect();
        if hidden > 0 && shown > 0 {
            rows.push((format!("+{hidden} more tool uses"), theme.aside_style()));
        }
        if let StepState::Finished {
            outcome: RelayStepEnd::Failed { message },
            ..
        } = &self.state
        {
            rows.push((
                format!("✗ {}", escape_controls_single_line(message)),
                theme.error_style(),
            ));
        }
        rows
    }
}

enum StepState {
    Pending,
    Running {
        started_at: Instant,
    },
    Finished {
        elapsed: Duration,
        outcome: RelayStepEnd,
    },
}

/// One row of what a sub-agent did, in the order it happened.
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

/// The first row's prefix and the one every later row is indented by.
#[derive(Clone, Copy)]
struct Gutter<'a> {
    first: &'a str,
    rest: &'a str,
}

/// Draw `run` as one compact block: a header, each step's latest calls while
/// it runs, and a summary. `collapsed` hides finished steps' calls and the report.
pub(super) fn render_relay_block(
    out: &mut Vec<Line<'static>>,
    run: &RelayRun,
    tool: &ToolState,
    collapsed: bool,
    width: u16,
    theme: &Theme,
) {
    let (bullet, elapsed, report) = match tool {
        ToolState::Running { started_at } => (theme.status_text, started_at.elapsed(), None),
        ToolState::Finished {
            output,
            exit_code,
            duration_ms,
        } => (
            exit_color(*exit_code, theme),
            Duration::from_millis(*duration_ms),
            Some(output.as_str()),
        ),
    };
    let header_gutter = Gutter {
        first: HEADER_BULLET,
        rest: HEADER_INDENT,
    };
    let bullet_style = Style::default().fg(bullet).add_modifier(Modifier::BOLD);
    match run.steps.as_slice() {
        [only] => {
            let lead = format!("{} · ", only.agent);
            let header = fitted_row(&lead, &only.task, "", block_cols(width, HEADER_BULLET));
            push_gutter_row(
                out,
                header_gutter,
                bullet_style,
                &header,
                theme.title_style(),
                width,
            );
            push_rows(
                out,
                &only.detail_rows(!collapsed, theme),
                result_gutter(),
                width,
                theme,
            );
        }
        steps => {
            let header = format!("plan of {} steps", steps.len());
            push_gutter_row(
                out,
                header_gutter,
                bullet_style,
                &header,
                theme.title_style(),
                width,
            );
            push_plan_steps(out, steps, !collapsed, width, theme);
        }
    }
    let summary = run_summary(run, report.is_some(), elapsed, collapsed, theme);
    push_rows(out, &[summary], result_gutter(), width, theme);
    if let Some(report) = report.filter(|_| !collapsed) {
        let rows: Vec<(String, Style)> = split_body(report)
            .into_iter()
            .map(|line| (line, theme.code_style()))
            .collect();
        let indent = Gutter {
            first: RESULT_REST,
            rest: RESULT_REST,
        };
        push_rows(out, &rows, indent, width, theme);
    }
    out.push(Line::from(""));
}

/// Each step on a branch of a tree, with its detail rows beneath it.
fn push_plan_steps(
    out: &mut Vec<Line<'static>>,
    steps: &[RelayStepTrace],
    all: bool,
    width: u16,
    theme: &Theme,
) {
    for (index, step) in steps.iter().enumerate() {
        let last = index + 1 == steps.len();
        let (branch, stem) = if last {
            ("  └─ ", "     ")
        } else {
            ("  ├─ ", "  │  ")
        };
        let lead = format!("{} {} · ", step.icon(), step.agent);
        let tail = if step.has_started() {
            format!(" · {}", step.status())
        } else {
            String::new()
        };
        let row = fitted_row(&lead, &step.task, &tail, block_cols(width, branch));
        let gutter = Gutter {
            first: branch,
            rest: stem,
        };
        push_gutter_row(
            out,
            gutter,
            theme.rule_style(),
            &row,
            step.style(theme),
            width,
        );
        let detail_first = format!("{stem}⎿ ");
        let detail_rest = format!("{stem}  ");
        let detail = Gutter {
            first: &detail_first,
            rest: &detail_rest,
        };
        push_rows(out, &step.detail_rows(all, theme), detail, width, theme);
    }
}

/// `Running… (…)`, `Done (…)` or `Failed (…)` over every step, with how to
/// see the rest when `collapsed`.
fn run_summary(
    run: &RelayRun,
    finished: bool,
    elapsed: Duration,
    collapsed: bool,
    theme: &Theme,
) -> (String, Style) {
    let stats = stats(run.tool_uses(), run.tokens(), elapsed);
    let hint = if collapsed { " (Tab to expand)" } else { "" };
    match (finished, run.failure()) {
        (false, _) => (format!("Running… ({stats})"), theme.aside_style()),
        (true, None) => (format!("Done ({stats}){hint}"), theme.meta_style()),
        (true, Some(_)) => (format!("Failed ({stats}){hint}"), theme.error_style()),
    }
}

/// `3 tool uses · 12.1k tokens · 41s`, leaving out tokens not yet reported.
fn stats(tool_uses: usize, tokens: Option<u32>, elapsed: Duration) -> String {
    let uses = match tool_uses {
        1 => "1 tool use".to_string(),
        count => format!("{count} tool uses"),
    };
    match tokens {
        Some(tokens) => format!(
            "{uses} · {} tokens · {}",
            compact_count(u64::from(tokens)),
            elapsed_text(elapsed)
        ),
        None => format!("{uses} · {}", elapsed_text(elapsed)),
    }
}

/// `rows` behind `gutter`: its first prefix before the first row only.
fn push_rows(
    out: &mut Vec<Line<'static>>,
    rows: &[(String, Style)],
    gutter: Gutter<'_>,
    width: u16,
    theme: &Theme,
) {
    for (index, (text, style)) in rows.iter().enumerate() {
        let row_gutter = if index == 0 {
            gutter
        } else {
            Gutter {
                first: gutter.rest,
                rest: gutter.rest,
            }
        };
        push_gutter_row(out, row_gutter, theme.rule_style(), text, *style, width);
    }
}

/// `text` wrapped beside `gutter`, whose prefixes are drawn in `gutter_style`.
fn push_gutter_row(
    out: &mut Vec<Line<'static>>,
    gutter: Gutter<'_>,
    gutter_style: Style,
    text: &str,
    style: Style,
    width: u16,
) {
    let inner = usize::from(width)
        .saturating_sub(display_width(gutter.first))
        .max(1);
    let wrapped = textwrap::wrap(text, inner);
    if wrapped.is_empty() {
        out.push(Line::from(Span::styled(
            gutter.first.to_string(),
            gutter_style,
        )));
        return;
    }
    for (index, chunk) in wrapped.into_iter().enumerate() {
        let prefix = if index == 0 {
            gutter.first
        } else {
            gutter.rest
        };
        out.push(Line::from(vec![
            Span::styled(prefix.to_string(), gutter_style),
            Span::styled(chunk.into_owned(), style),
        ]));
    }
}

fn result_gutter() -> Gutter<'static> {
    Gutter {
        first: RESULT_FIRST,
        rest: RESULT_REST,
    }
}

fn entry_row(entry: &RelayEntry, theme: &Theme) -> (String, Style) {
    match entry {
        RelayEntry::Call {
            command,
            result: None,
        } => (format!("$ {command} …"), theme.aside_style()),
        RelayEntry::Call {
            command,
            result: Some((exit_code, _)),
        } if *exit_code != 0 => (
            format!("$ {command}  [exit:{exit_code}]"),
            theme.error_style(),
        ),
        RelayEntry::Call { command, .. } => (format!("$ {command}"), theme.code_style()),
        RelayEntry::Compaction {
            started_at,
            took: None,
        } => (
            format!(
                "compacting history… ({})",
                elapsed_text(started_at.elapsed())
            ),
            theme.aside_style(),
        ),
        RelayEntry::Compaction {
            took: Some(took), ..
        } => (
            format!("compacted history in {}", elapsed_text(*took)),
            theme.aside_style(),
        ),
    }
}

/// `{lead}{task}{tail}` in `cols` columns, the task cut to fit but kept to at
/// least [`MIN_TASK_COLS`].
fn fitted_row(lead: &str, task: &str, tail: &str, cols: usize) -> String {
    let room = cols
        .saturating_sub(display_width(lead) + display_width(tail))
        .max(MIN_TASK_COLS);
    format!(
        "{lead}{}{tail}",
        ellipsize(&escape_controls_single_line(task), room)
    )
}

/// Columns left for text in a `width`-wide pane behind `gutter`.
fn block_cols(width: u16, gutter: &str) -> usize {
    usize::from(width).saturating_sub(display_width(gutter))
}
