//! ratatui render for the MCP server list and the form one server is written in.

use assistd_utils::terminal_text::escape_controls_single_line;
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use super::super::selection::list_window;
use super::super::text::ellipsize;
use super::super::theme::Theme;
use super::render::{
    NO_POINTER, SCROLL_MARGIN, choice_span, confirm_delete_line, editor_window, list_hint,
    place_cursor, pointed_line, render_list_editor, titled_box,
};
use super::server_form::{FieldValue, ServerField, ServerForm};
use super::server_list::{ServerList, ServerListState, server_name, server_target};

/// Columns a server's name is padded to in the list.
const NAME_WIDTH: usize = 16;
/// Columns the form's labels are padded to.
const LABEL_WIDTH: usize = 15;
/// Stands for a client secret while its field is not focused.
const MASKED: &str = "••••••••";
const BROWSE_HINT: &str = "↑/↓ move · Enter edit · a add · d delete · Esc done";
const CONFIRM_DELETE_HINT: &str = "y delete · n keep";
const TEXT_HINT: &str = "↑/↓ field · Enter apply · Esc cancel · Ctrl+U clear";
const CHOICE_HINT: &str = "↑/↓ field · ←/→ choose · Enter apply · Esc cancel";
const LIST_HINT: &str = "↑/↓ field · Enter edit entries · Esc cancel";

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
    match form.open_list() {
        Some(editor) => {
            let entries = form.list_entries();
            render_list_editor(frame, area, editor, &entries, form.focus().label(), theme);
        }
        None if list.selected() < servers.len() => {
            render_form(frame, area, form, "Edit server", theme);
        }
        None => render_form(frame, area, form, "Add server", theme),
    }
}

/// The keys the server list takes in its state.
pub(super) fn server_hint(list: &ServerList) -> &'static str {
    let form = match list.state() {
        ServerListState::Browse => return BROWSE_HINT,
        ServerListState::ConfirmDelete => return CONFIRM_DELETE_HINT,
        ServerListState::Form(form) => form,
    };
    match (form.open_list(), form.value(form.focus())) {
        (Some(editor), _) => list_hint(editor.state()),
        (None, FieldValue::Text(_)) => TEXT_HINT,
        (None, FieldValue::Choice(_)) => CHOICE_HINT,
        (None, FieldValue::List(_)) => LIST_HINT,
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

/// The form's fields in a box titled `title`, scrolled to keep the focused one in view,
/// with the cursor in it when it is text.
fn render_form(frame: &mut Frame<'_>, area: Rect, form: &ServerForm, title: &str, theme: &Theme) {
    let block = titled_box(title, theme);
    let inner = block.inner(area);
    frame.render_widget(block, area);
    let value_column = NO_POINTER.len() + LABEL_WIDTH;
    let width = usize::from(inner.width).saturating_sub(value_column);
    let fields = form.fields();
    let focused_row = fields
        .iter()
        .position(|field| *field == form.focus())
        .unwrap_or(0);
    let mut cursor_column = None;
    let mut lines = Vec::with_capacity(fields.len());
    for field in fields {
        let focused = field == form.focus();
        let value = match form.value(field) {
            FieldValue::Text(editor) if focused => {
                let (text, column) = editor_window(editor, width);
                cursor_column = Some(value_column + column);
                Span::raw(text)
            }
            FieldValue::Text(editor) => unfocused_text(field, editor.text(), width),
            FieldValue::Choice(text) => choice_span(text, focused, theme),
            FieldValue::List(entries) if entries.is_empty() => {
                Span::styled("(none)", theme.meta_style())
            }
            FieldValue::List(entries) => {
                Span::raw(ellipsize(&escape_controls_single_line(&entries), width))
            }
        };
        let label = Span::raw(format!("{:LABEL_WIDTH$}", field.label()));
        lines.push(pointed_line(focused, theme, vec![label, value]));
    }
    let visible = usize::from(inner.height).max(1);
    let window = list_window(focused_row, lines.len(), visible, 0);
    let start = window.start;
    let lines: Vec<Line<'static>> = lines.drain(window).collect();
    frame.render_widget(Paragraph::new(lines), inner);
    if let Some(column) = cursor_column {
        place_cursor(frame, inner, focused_row - start, column);
    }
}

/// `text` cut to `width` columns, or a mask when it is a client secret.
fn unfocused_text(field: ServerField, text: &str, width: usize) -> Span<'static> {
    if field == ServerField::ClientSecret && !text.is_empty() {
        return Span::raw(MASKED);
    }
    Span::raw(ellipsize(&escape_controls_single_line(text), width))
}
