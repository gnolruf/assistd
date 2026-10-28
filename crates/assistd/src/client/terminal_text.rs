//! Escapes control and invisible formatting characters in daemon-supplied
//! text so a terminal prints them instead of acting on them.

use std::borrow::Cow;

/// `text` with every control character except newline and tab escaped, and
/// the bidi controls and line separators that would reorder or split it.
pub(crate) fn escape_controls(text: &str) -> Cow<'_, str> {
    escape_where(text, |c| {
        (c.is_control() && c != '\n' && c != '\t') || is_bidi_control(c) || is_separator(c)
    })
}

/// `text` with every control and invisible formatting character escaped,
/// newlines and tabs included, so it renders as a single line exactly as
/// written.
pub(crate) fn escape_controls_single_line(text: &str) -> Cow<'_, str> {
    escape_where(text, |c| c.is_control() || is_format(c) || is_separator(c))
}

/// `text` with only the bidi controls that would reorder it escaped.
#[cfg(feature = "chat")]
pub(crate) fn escape_bidi_controls(text: &str) -> Cow<'_, str> {
    escape_where(text, is_bidi_control)
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

/// Characters that reorder the text around them on bidi-aware terminals.
fn is_bidi_control(c: char) -> bool {
    matches!(
        c,
        '\u{061C}' | '\u{200E}' | '\u{200F}' | '\u{202A}'..='\u{202E}' | '\u{2066}'..='\u{2069}'
    )
}

/// Unicode general category Cf: invisible characters that format the text
/// around them.
fn is_format(c: char) -> bool {
    matches!(
        c,
        '\u{00AD}'
            | '\u{0600}'..='\u{0605}'
            | '\u{061C}'
            | '\u{06DD}'
            | '\u{070F}'
            | '\u{0890}'..='\u{0891}'
            | '\u{08E2}'
            | '\u{180E}'
            | '\u{200B}'..='\u{200F}'
            | '\u{202A}'..='\u{202E}'
            | '\u{2060}'..='\u{2064}'
            | '\u{2066}'..='\u{206F}'
            | '\u{FEFF}'
            | '\u{FFF9}'..='\u{FFFB}'
            | '\u{110BD}'
            | '\u{110CD}'
            | '\u{13430}'..='\u{1343F}'
            | '\u{1BCA0}'..='\u{1BCA3}'
            | '\u{1D173}'..='\u{1D17A}'
            | '\u{E0001}'
            | '\u{E0020}'..='\u{E007F}'
    )
}

/// The Unicode line and paragraph separators.
fn is_separator(c: char) -> bool {
    matches!(c, '\u{2028}' | '\u{2029}')
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

    #[test]
    fn bidi_overrides_and_invisible_characters_are_escaped() {
        assert_eq!(
            escape_controls_single_line("rm -rf \u{202E}fdp.\u{2066}x\u{200B}\u{2028}y"),
            r"rm -rf \u{202e}fdp.\u{2066}x\u{200b}\u{2028}y"
        );
        assert_eq!(
            escape_controls("a\u{202E}b\u{2029}c\u{200D}d"),
            "a\\u{202e}b\\u{2029}c\u{200D}d"
        );
    }

    #[cfg(feature = "chat")]
    #[test]
    fn bidi_only_escape_keeps_other_controls() {
        assert_eq!(
            escape_bidi_controls("\x1b[1mfdp.\u{202E}exe\u{200D}\u{2066}"),
            "\x1b[1mfdp.\\u{202e}exe\u{200D}\\u{2066}"
        );
    }
}
