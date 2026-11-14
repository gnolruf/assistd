//! Text fitting shared by the chat views: cutting to a width with an
//! ellipsis, expanding tabs, hard wrapping, and the row layout of editable text.

use textwrap::core::display_width;

/// Terminal columns `c` takes.
pub(crate) fn char_width(c: char) -> usize {
    display_width(c.encode_utf8(&mut [0; 4]))
}

/// `text` in at most `cols` display columns, ending in `…` when cut.
pub(crate) fn ellipsize(text: &str, cols: usize) -> String {
    if display_width(text) <= cols {
        return text.to_string();
    }
    let mut used = 0;
    let mut cut: String = text
        .chars()
        .take_while(|&c| {
            used += char_width(c);
            used < cols
        })
        .collect();
    cut.push('…');
    cut
}

/// `text` with its tabs and carriage returns expanded as [`append_expanded`] does, a final
/// carriage return dropped.
pub(crate) fn expanded(text: String, tab_stop: usize) -> String {
    if !text.contains(['\t', '\r']) {
        return text;
    }
    let mut expanded = String::with_capacity(text.len());
    append_expanded(&mut expanded, &text, tab_stop);
    if expanded.ends_with('\r') {
        expanded.pop();
    }
    expanded
}

/// Append `delta` to `text`, widening each tab to the next multiple of `tab_stop` columns
/// and turning each carriage return, with any line feed after it, into one line break. A
/// carriage return ending `delta` stays last in `text` until the next call resolves it.
pub(crate) fn append_expanded(text: &mut String, delta: &str, tab_stop: usize) {
    let mut after_return = text.ends_with('\r');
    if after_return {
        text.pop();
    }
    let mut column = display_width(text.rsplit('\n').next().unwrap_or_default());
    for c in delta.chars() {
        if std::mem::take(&mut after_return) {
            text.push('\n');
            column = 0;
            if c == '\n' {
                continue;
            }
        }
        match c {
            '\r' => after_return = true,
            '\n' => {
                text.push('\n');
                column = 0;
            }
            '\t' => {
                let fill = tab_stop - column % tab_stop;
                text.extend(std::iter::repeat_n(' ', fill));
                column += fill;
            }
            c => {
                text.push(c);
                column += char_width(c);
            }
        }
    }
    if after_return {
        text.push('\r');
    }
}

/// Split `text` into rows of at most `cols` display columns, breaking
/// anywhere so whitespace is kept verbatim. Always yields at least one row.
pub(crate) fn hard_wrap(text: &str, cols: usize) -> Vec<String> {
    let cols = cols.max(1);
    let mut rows = Vec::new();
    let mut row = String::new();
    let mut row_cols = 0;
    for c in text.chars() {
        let char_cols = char_width(c);
        if row_cols + char_cols > cols && row_cols > 0 {
            rows.push(std::mem::take(&mut row));
            row_cols = 0;
        }
        row.push(c);
        row_cols += char_cols;
    }
    rows.push(row);
    rows
}

/// `(start, end)` char ranges of each input row: a new row at each line
/// break, otherwise after the last whitespace that fits past the prompt. A
/// final row that fills the width is followed by an empty one for the cursor.
pub(crate) fn wrap_input(buf: &[char], prompt_w: u16, width: u16) -> Vec<(usize, usize)> {
    let n = buf.len();
    if width == 0 {
        return vec![(0, n)];
    }
    let cap = width.saturating_sub(prompt_w).max(1) as usize;
    let mut rows: Vec<(usize, usize)> = Vec::new();
    let line_ends = buf
        .iter()
        .enumerate()
        .filter_map(|(i, &c)| (c == '\n').then_some(i))
        .chain([n]);
    let mut start = 0usize;
    for end in line_ends {
        wrap_input_line(buf, (start, end), cap, &mut rows);
        start = end + 1;
    }
    let (ls, le) = *rows.last().expect("rows non-empty");
    if le - ls >= cap {
        rows.push((n, n));
    }
    rows
}

/// Append the rows, at most `cap` chars each, of the break-free span
/// `(start, end)` to `rows`, always at least one.
fn wrap_input_line(
    buf: &[char],
    (mut start, end): (usize, usize),
    cap: usize,
    rows: &mut Vec<(usize, usize)>,
) {
    while end - start > cap {
        let end_max = start + cap;
        let row_end = buf[start..end_max]
            .iter()
            .rposition(|c| c.is_whitespace())
            .map_or(end_max, |i| start + i + 1);
        rows.push((start, row_end));
        start = row_end;
    }
    rows.push((start, end));
}

/// Row and column of char offset `cursor`. At a soft wrap the cursor moves
/// to the next row; before a line break it stays at the end of its row.
pub(crate) fn locate_cursor(rows: &[(usize, usize)], cursor: usize) -> (usize, usize) {
    for (idx, &(s, e)) in rows.iter().enumerate() {
        if cursor < e {
            return (idx, cursor - s);
        }
        if cursor == e {
            if rows.get(idx + 1).is_some_and(|&(next, _)| next == e) {
                return (idx + 1, 0);
            }
            return (idx, cursor - s);
        }
    }
    let last = rows.len().saturating_sub(1);
    let (s, e) = rows.get(last).copied().unwrap_or((0, 0));
    (last, cursor.saturating_sub(s).min(e - s))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chars(s: &str) -> Vec<char> {
        s.chars().collect()
    }

    #[test]
    fn ellipsize_fits_the_width_counting_wide_chars() {
        assert_eq!(ellipsize("hello", 5), "hello");
        assert_eq!(ellipsize("hello!", 5), "hell…");
        assert_eq!(ellipsize("漢字漢字", 5), "漢字…");
    }

    #[test]
    fn hard_wrap_breaks_before_a_wide_char_that_would_overflow() {
        assert_eq!(hard_wrap("ab漢", 3), vec!["ab", "漢"]);
        assert_eq!(hard_wrap("", 3), vec![""]);
    }

    #[test]
    fn wrap_input_hard_breaks_when_word_is_longer_than_row() {
        let rows = wrap_input(&chars("hellotherefriend"), 2, 10);
        assert_eq!(rows, vec![(0, 8), (8, 16), (16, 16)]);
    }

    #[test]
    fn locate_cursor_jumps_to_next_row_on_boundary() {
        let buf = chars("hello world");
        let rows = wrap_input(&buf, 2, 10);
        assert_eq!(rows, [(0, 6), (6, 11)]);
        assert_eq!(locate_cursor(&rows, 0), (0, 0));
        assert_eq!(locate_cursor(&rows, 5), (0, 5));
        assert_eq!(locate_cursor(&rows, 6), (1, 0));
        assert_eq!(locate_cursor(&rows, 11), (1, 5));
    }

    #[test]
    fn line_breaks_start_rows_and_place_cursor_around_them() {
        let buf = chars("ab\n\ncd\n");
        let rows = wrap_input(&buf, 2, 10);
        assert_eq!(rows, [(0, 2), (3, 3), (4, 6), (7, 7)]);
        assert_eq!(locate_cursor(&rows, 2), (0, 2));
        assert_eq!(locate_cursor(&rows, 3), (1, 0));
        assert_eq!(locate_cursor(&rows, 6), (2, 2));
        assert_eq!(locate_cursor(&rows, 7), (3, 0));
    }

    #[test]
    fn locate_cursor_lands_on_phantom_row_after_full_row() {
        let buf = chars("aaaaaaaa");
        let rows = wrap_input(&buf, 2, 10);
        assert_eq!(locate_cursor(&rows, 8), (1, 0));
    }
}
