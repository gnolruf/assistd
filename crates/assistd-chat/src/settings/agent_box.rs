//! ratatui render for the read-only list of sub-agents and of the agent files refused.

use assistd_config::{AgentCatalog, AgentFileError, AgentOrigin, AgentSpec};
use assistd_utils::terminal_text::escape_controls_single_line;
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use super::super::selection::list_window;
use super::super::text::ellipsize;
use super::super::theme::Theme;
use super::render::{NO_POINTER, SCROLL_MARGIN, pointed_line, titled_box};

pub(super) const AGENTS_HINT: &str = "↑/↓ move · Esc done";

/// Columns an agent's name is padded to in the list.
const NAME_WIDTH: usize = 24;
const TITLE: &str = "Sub-agents";

/// The agents of `catalog`, then its refused files, in a box, with entry `selected`
/// pointed at and kept in view.
pub(super) fn render_agent_list(
    frame: &mut Frame<'_>,
    area: Rect,
    catalog: &AgentCatalog,
    selected: usize,
    theme: &Theme,
) {
    let block = titled_box(TITLE, theme);
    let inner = block.inner(area);
    frame.render_widget(block, area);
    let width = usize::from(inner.width).saturating_sub(NO_POINTER.len());
    let agents = catalog.agents.iter().map(|agent| agent_spans(agent, theme));
    let refused = catalog
        .refused
        .iter()
        .map(|refused| refused_spans(refused, width, theme));
    let mut lines: Vec<Line<'static>> = agents
        .chain(refused)
        .enumerate()
        .map(|(index, spans)| pointed_line(index == selected, theme, spans))
        .collect();
    let visible = usize::from(inner.height).max(1);
    let window = list_window(selected, lines.len(), visible, SCROLL_MARGIN);
    let lines: Vec<Line<'static>> = lines.drain(window).collect();
    frame.render_widget(Paragraph::new(lines), inner);
}

/// What the detail line says of entry `selected` of `catalog`: an agent's description, or
/// why a file was refused.
pub(super) fn agent_detail(catalog: &AgentCatalog, selected: usize) -> String {
    let refused = selected.checked_sub(catalog.agents.len());
    match (catalog.agents.get(selected), refused) {
        (Some(agent), _) => format!("{} : {}", agent.name.as_str(), agent.description),
        (None, Some(index)) => catalog
            .refused
            .get(index)
            .map_or_default(ToString::to_string),
        (None, None) => String::new(),
    }
}

/// The agent's name, then where it comes from and what it may reach.
fn agent_spans(agent: &AgentSpec, theme: &Theme) -> Vec<Span<'static>> {
    let name = format!("{:NAME_WIDTH$} ", agent.name.as_str());
    let origin = match agent.origin {
        AgentOrigin::BuiltIn => "built-in",
        AgentOrigin::File(_) => "file",
    };
    let mut reach = agent.capabilities.names().join(", ");
    for server in &agent.capabilities.mcp_servers {
        reach.push_str(&format!(", mcp:{}", server.as_str()));
    }
    let reach = reach.trim_start_matches(", ");
    let reach = if reach.is_empty() { "nothing" } else { reach };
    vec![
        Span::raw(name),
        Span::styled(format!("{origin:9} reaches {reach}"), theme.meta_style()),
    ]
}

/// The refused file's name and why, cut to `width` columns.
fn refused_spans(refused: &AgentFileError, width: usize, theme: &Theme) -> Vec<Span<'static>> {
    let file = refused
        .path
        .file_name()
        .map_or_default(|name| name.to_string_lossy().into_owned());
    let text = format!("! {file}: {}", refused.issue);
    vec![Span::styled(
        ellipsize(&escape_controls_single_line(&text), width),
        theme.warning_style(),
    )]
}
