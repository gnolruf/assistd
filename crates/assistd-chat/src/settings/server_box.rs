//! ratatui render for the MCP server list, or the form its open server is written in.

use assistd_utils::terminal_text::escape_controls_single_line;
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use super::super::selection::list_window;
use super::super::text::ellipsize;
use super::super::theme::Theme;
use super::form_box::{form_hint, render_form};
use super::render::{NO_POINTER, SCROLL_MARGIN, confirm_delete_line, pointed_line, titled_box};
use super::server_list::{ServerList, ServerListState, server_name, server_target};

/// Columns a server's name is padded to in the list.
const NAME_WIDTH: usize = 16;
const BROWSE_HINT: &str = "↑/↓ move · Enter edit · a add · d delete · Esc done";
const CONFIRM_DELETE_HINT: &str = "y delete · n keep";

/// The `servers` then an add row in a box titled `title`, or in its place the open form,
/// or the list box open over one of the form's fields.
pub(super) fn render_server_list(
    frame: &mut Frame<'_>,
    area: Rect,
    list: &ServerList,
    servers: &[toml::Value],
    title: &str,
    theme: &Theme,
) {
    let Some(form) = list.form() else {
        return render_servers(frame, area, list, servers, title, theme);
    };
    let title = if list.selected() < servers.len() {
        "Edit server"
    } else {
        "Add server"
    };
    render_form(frame, area, form, title, theme);
}

/// The keys the server list takes in its state.
pub(super) fn server_hint(list: &ServerList) -> &'static str {
    match list.state() {
        ServerListState::Browse => BROWSE_HINT,
        ServerListState::ConfirmDelete => CONFIRM_DELETE_HINT,
        ServerListState::Form(form) => form_hint(form),
    }
}

fn render_servers(
    frame: &mut Frame<'_>,
    area: Rect,
    list: &ServerList,
    servers: &[toml::Value],
    title: &str,
    theme: &Theme,
) {
    let block = titled_box(title, theme);
    let inner = block.inner(area);
    frame.render_widget(block, area);
    let width = usize::from(inner.width).saturating_sub(NO_POINTER.len());
    let deleting = matches!(list.state(), ServerListState::ConfirmDelete);
    let server_lines = servers.iter().enumerate().map(|(index, server)| {
        let selected = index == list.selected();
        if selected && deleting {
            confirm_delete_line(&toml::Value::from(server_name(server)), width, theme)
        } else {
            pointed_line(selected, theme, server_spans(server, width, theme))
        }
    });
    let add_line = pointed_line(
        list.selected() == servers.len(),
        theme,
        vec![Span::styled("+ add server", theme.meta_style())],
    );
    let mut lines: Vec<Line<'static>> = server_lines.chain([add_line]).collect();
    let visible = usize::from(inner.height).max(1);
    let window = list_window(list.selected(), lines.len(), visible, SCROLL_MARGIN);
    let lines: Vec<Line<'static>> = lines.drain(window).collect();
    frame.render_widget(Paragraph::new(lines), inner);
}

/// The server's name, then what it reaches cut to the rest of `width` columns.
fn server_spans(server: &toml::Value, width: usize, theme: &Theme) -> Vec<Span<'static>> {
    let name = format!("{:NAME_WIDTH$} ", server_name(server));
    let target = server_target(server);
    let target = ellipsize(
        &escape_controls_single_line(&target),
        width.saturating_sub(name.chars().count()),
    );
    vec![Span::raw(name), Span::styled(target, theme.meta_style())]
}
