//! The `/agents` panel: the list of sub-agents, and one's transcript drawn
//! like the chat's.

use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::Paragraph;

use super::super::app::{AgentPanel, App};
use super::super::output::RelayStepTrace;
use super::{picker_scroll_start, render_rule, render_transcript};

/// Rows each listed sub-agent takes: its label and its status.
const AGENT_ENTRY_ROWS: usize = 2;
const STATUS_INDENT: &str = "    ";
const LIST_HINT: &str = "↑/↓ select · Enter open · Esc close";
const TRANSCRIPT_HINT: &str = "Esc back · ↑/↓ PgUp/PgDn scroll · Tab expand · ctrl+o verbose";

/// The list, or the selected sub-agent's transcript once opened.
pub(super) fn render_agent_panel(
    frame: &mut Frame<'_>,
    area: Rect,
    app: &mut App,
    panel: AgentPanel,
) {
    if panel.open {
        render_agent_transcript(frame, area, app, panel.selected);
    } else {
        render_agent_list(frame, area, &app.output.agents(), panel.selected);
    }
}

/// The keys that work in `panel`'s view, for the status line.
pub(super) fn agent_panel_hint(panel: AgentPanel) -> &'static str {
    if panel.open {
        TRANSCRIPT_HINT
    } else {
        LIST_HINT
    }
}

/// A title, then every sub-agent with its status.
fn render_agent_list(
    frame: &mut Frame<'_>,
    area: Rect,
    agents: &[&RelayStepTrace],
    selected: usize,
) {
    let [title_area, _, list_area] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Length(1),
        Constraint::Min(0),
    ])
    .areas(area);
    let title = Line::from(vec![
        Span::styled("Sub-agents", Style::default().add_modifier(Modifier::BOLD)),
        Span::styled(format!(" ({})", agents.len()), hint_style()),
    ]);
    frame.render_widget(Paragraph::new(title), title_area);
    let visible = (usize::from(list_area.height) / AGENT_ENTRY_ROWS).max(1);
    let start = picker_scroll_start(selected, agents.len(), visible);
    let lines: Vec<Line<'static>> = agents
        .iter()
        .enumerate()
        .skip(start)
        .take(visible)
        .flat_map(|(index, agent)| agent_entry_rows(agent, index == selected))
        .collect();
    frame.render_widget(Paragraph::new(Text::from(lines)), list_area);
}

fn agent_entry_rows(agent: &RelayStepTrace, selected: bool) -> [Line<'static>; 2] {
    let (pointer, style) = if selected {
        ("❯ ", agent.style().add_modifier(Modifier::BOLD))
    } else {
        ("  ", agent.style())
    };
    [
        Line::from(Span::styled(format!("{pointer}{}", agent.label()), style)),
        Line::from(Span::styled(
            format!("{STATUS_INDENT}{}", agent.status()),
            hint_style(),
        )),
    ]
}

/// The agent's label and status above its transcript, scrolled like the chat's.
fn render_agent_transcript(frame: &mut Frame<'_>, area: Rect, app: &mut App, selected: usize) {
    let verbose = app.verbose;
    let [label_area, status_area, rule_area, transcript_area] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Length(1),
        Constraint::Length(1),
        Constraint::Min(0),
    ])
    .areas(area);
    app.set_output_height(transcript_area.height);
    let Some(agent) = app.output.agent_mut(selected) else {
        return;
    };
    let label = Span::styled(agent.label(), agent.style().add_modifier(Modifier::BOLD));
    frame.render_widget(Paragraph::new(label), label_area);
    let status = Span::styled(agent.status(), agent.style());
    frame.render_widget(Paragraph::new(status), status_area);
    render_rule(frame, rule_area);
    let transcript = agent.transcript_mut();
    transcript.set_verbose(verbose);
    if transcript_area.width > 0 && transcript_area.height > 0 {
        render_transcript(frame, transcript_area, transcript);
    }
}

fn hint_style() -> Style {
    Style::default().fg(Color::Gray).add_modifier(Modifier::DIM)
}
