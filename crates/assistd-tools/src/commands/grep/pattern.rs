//! Compiles `grep`'s PATTERN arguments in the dialect its flags select.

use fancy_regex::{Regex, RegexBuilder};

/// How PATTERN is read.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum Dialect {
    /// POSIX basic regex: `\|`, `\(`, `\+`, `\?`, `\{` are operators and
    /// their bare spellings are literal.
    #[default]
    Basic,
    /// Extended regex with Perl escapes, lookaround and backreferences.
    Extended,
    /// A literal string.
    Fixed,
}

/// How the compiled matcher treats each PATTERN.
#[derive(Default)]
pub(super) struct MatchOptions {
    pub(super) dialect: Dialect,
    pub(super) case_insensitive: bool,
    pub(super) whole_word: bool,
}

/// One matcher that hits a line when any of `patterns` does.
pub(super) fn compile(patterns: &[String], options: &MatchOptions) -> Result<Regex, String> {
    let alternatives: Vec<String> = patterns
        .iter()
        .map(|pattern| format!("(?:{})", translate(pattern, options.dialect)))
        .collect();
    let mut source = alternatives.join("|");
    if options.whole_word {
        source = format!(r"\b{{start-half}}(?:{source})\b{{end-half}}");
    }
    RegexBuilder::new(&source)
        .case_insensitive(options.case_insensitive)
        .build()
        .map_err(|error| error.to_string())
}

/// Whether a basic-regex `pattern` spells alternation as a bare `|`,
/// which matches a literal pipe there.
pub(super) fn has_bare_pipe(pattern: &str) -> bool {
    pattern
        .match_indices('|')
        .any(|(index, _)| !pattern[..index].ends_with('\\'))
}

fn translate(pattern: &str, dialect: Dialect) -> String {
    match dialect {
        Dialect::Basic => translate_basic(pattern),
        Dialect::Extended => pattern.to_owned(),
        Dialect::Fixed => fancy_regex::escape(pattern).into_owned(),
    }
}

/// Rewrite a POSIX basic regex into extended syntax. Escapes that are not
/// BRE operators, such as `\d`, `\w` and `\1`, pass through unchanged.
fn translate_basic(pattern: &str) -> String {
    let mut out = String::with_capacity(pattern.len() + 8);
    let mut rest = pattern;
    let mut at_expression_start = true;
    while let Some(ch) = rest.chars().next() {
        rest = &rest[ch.len_utf8()..];
        let mut starts_expression = false;
        match ch {
            '\\' => {
                let Some(escaped) = rest.chars().next() else {
                    out.push('\\');
                    break;
                };
                rest = &rest[escaped.len_utf8()..];
                starts_expression = matches!(escaped, '|' | '(');
                push_basic_escape(escaped, &mut out);
            }
            '[' => {
                let length = bracket_length(rest);
                out.push('[');
                out.push_str(&rest[..length]);
                rest = &rest[length..];
            }
            '^' if at_expression_start => {
                out.push('^');
                starts_expression = true;
            }
            '*' if at_expression_start => out.push_str(r"\*"),
            '$' if ends_expression(rest) => out.push('$'),
            '|' | '(' | ')' | '{' | '}' | '+' | '?' | '^' | '$' => {
                out.push('\\');
                out.push(ch);
            }
            _ => out.push(ch),
        }
        at_expression_start = starts_expression;
    }
    out
}

fn push_basic_escape(escaped: char, out: &mut String) {
    if !matches!(escaped, '|' | '(' | ')' | '{' | '}' | '+' | '?') {
        out.push('\\');
    }
    out.push(escaped);
}

fn ends_expression(rest: &str) -> bool {
    rest.is_empty() || rest.starts_with(r"\)") || rest.starts_with(r"\|")
}

/// Byte length of a bracket expression's body after its opening `[`,
/// through the closing `]`; all of `body` when it never closes.
fn bracket_length(body: &str) -> usize {
    let bytes = body.as_bytes();
    let mut index = usize::from(bytes.first() == Some(&b'^'));
    index += usize::from(bytes.get(index) == Some(&b']'));
    while let Some(&byte) = bytes.get(index) {
        match (byte, bytes.get(index + 1)) {
            (b']', _) => return index + 1,
            (b'\\', Some(_)) => index += 2,
            (b'[', Some(&class @ (b':' | b'=' | b'.'))) => {
                index = bytes[index + 2..]
                    .windows(2)
                    .position(|pair| pair == [class, b']'])
                    .map_or(bytes.len(), |end| index + 2 + end + 2);
            }
            _ => index += 1,
        }
    }
    bytes.len()
}
