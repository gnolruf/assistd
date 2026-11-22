//! ratatui render for a form and the list box open over one of its fields.

use assistd_utils::terminal_text::escape_controls_single_line;
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use super::super::selection::list_window;
use super::super::text::ellipsize;
use super::super::theme::Theme;
use super::form::{FieldValue, Form, FormFields};
use super::render::{
    NO_POINTER, choice_span, editor_window, list_hint, place_cursor, pointed_line,
    render_list_editor, titled_box,
};

/// Columns the labels are padded to.
const LABEL_WIDTH: usize = 15;
/// Stands for a secret while its field is not focused.
const MASKED: &str = "••••••••";
const TEXT_HINT: &str = "↑/↓ field · Enter apply · Esc cancel · Ctrl+U clear";
const CHOICE_HINT: &str = "↑/↓ field · ←/→ choose · Enter apply · Esc cancel";
const LIST_HINT: &str = "↑/↓ field · Enter edit entries · Esc cancel";

/// The form's fields in a box titled `title`, scrolled to keep the focused one in view,
/// with the cursor in it when it is text; or in the box's place the list box open over a
/// field.
pub(crate) fn render_form<F: FormFields>(
    frame: &mut Frame<'_>,
    area: Rect,
    form: &Form<F>,
    title: &str,
    theme: &Theme,
) {
    if let Some(editor) = form.open_list() {
        let entries = form.list_entries();
        let title = F::label(form.focus());
        return render_list_editor(frame, area, editor, &entries, title, theme);
    }
    let block = titled_box(title, theme);
    let inner = block.inner(area);
    frame.render_widget(block, area);
    let value_column = NO_POINTER.len() + LABEL_WIDTH;
    let width = usize::from(inner.width).saturating_sub(value_column);
    let fields = form.fields().fields();
    let focused_row = fields
        .iter()
        .position(|field| *field == form.focus())
        .unwrap_or(0);
    let mut cursor_column = None;
    let mut lines = Vec::with_capacity(fields.len());
    for field in fields {
        let focused = field == form.focus();
        let value = match form.fields().value(field) {
            FieldValue::Text(editor) | FieldValue::Secret(editor) if focused => {
                let (text, column) = editor_window(editor, width);
                cursor_column = Some(value_column + column);
                Span::raw(text)
            }
            FieldValue::Secret(editor) if !editor.text().is_empty() => Span::raw(MASKED),
            FieldValue::Text(editor) | FieldValue::Secret(editor) => cut(editor.text(), width),
            FieldValue::Choice(text) => choice_span(text, focused, theme),
            FieldValue::List { entries: [], .. } => Span::styled("(none)", theme.meta_style()),
            FieldValue::List { summary, .. } => cut(&summary, width),
        };
        let label = Span::raw(format!("{:LABEL_WIDTH$}", F::label(field)));
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

/// The keys the form takes on its focused field, or its open list box in its state.
pub(crate) fn form_hint<F: FormFields>(form: &Form<F>) -> &'static str {
    match (form.open_list(), form.fields().value(form.focus())) {
        (Some(editor), _) => list_hint(editor.state()),
        (None, FieldValue::Text(_) | FieldValue::Secret(_)) => TEXT_HINT,
        (None, FieldValue::Choice(_)) => CHOICE_HINT,
        (None, FieldValue::List { .. }) => LIST_HINT,
    }
}

/// `text` cut to `width` columns.
fn cut(text: &str, width: usize) -> Span<'static> {
    Span::raw(ellipsize(&escape_controls_single_line(text), width))
}
