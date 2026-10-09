//! Escapes control characters in daemon-supplied text so a terminal prints
//! them instead of acting on them (clipboard writes, cursor moves, erases).

use std::borrow::Cow;

/// `text` with every control character except newline and tab escaped.
pub(super) fn escape_controls(text: &str) -> Cow<'_, str> {
    escape_where(text, |c| c.is_control() && c != '\n' && c != '\t')
}

/// `text` with every control character escaped, newlines and tabs included,
/// so it always renders as a single line.
pub(super) fn escape_controls_single_line(text: &str) -> Cow<'_, str> {
    escape_where(text, char::is_control)
}

fn escape_where(text: &str, needs_escape: impl Fn(char) -> bool) -> Cow<'_, str> {
    if !text.chars().any(&needs_escape) {
        return Cow::Borrowed(text);
    }
    let mut escaped = String::with_capacity(text.len());
    for c in text.chars() {
        if needs_escape(c) {
            escaped.extend(c.escape_default());
        } else {
            escaped.push(c);
        }
    }
    Cow::Owned(escaped)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn osc52_clipboard_write_is_escaped() {
        assert_eq!(
            escape_controls("hi\x1b]52;c;cm0gLXJm\x07"),
            r"hi\u{1b}]52;c;cm0gLXJm\u{7}"
        );
    }

    #[test]
    fn single_line_escapes_newlines_and_tabs() {
        assert_eq!(
            escape_controls_single_line("ls\n[tool result: bash exit:0]\tx"),
            r"ls\n[tool result: bash exit:0]\tx"
        );
    }
}
