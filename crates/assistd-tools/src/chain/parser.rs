//! Tokenizer and recursive-descent parser producing a [`Chain`].

use std::fmt;
use std::iter::Peekable;
use std::vec::IntoIter;

use thiserror::Error;

use super::{Chain, Word};

/// Most `|`, `&&`, `||`, and `;` operators one command line may contain.
pub const MAX_OPERATORS: usize = 128;

/// Error returned by [`parse_chain`] when the input cannot be parsed.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum ParseError {
    #[error("empty command")]
    Empty,
    #[error("unterminated quoted string")]
    UnterminatedQuote,
    #[error("unexpected operator '{0}' at start of expression")]
    UnexpectedOperator(&'static str),
    #[error("trailing operator '{0}'")]
    TrailingOperator(&'static str),
    #[error("empty command between operators")]
    EmptyCommand,
    #[error("{0}")]
    Unsupported(&'static str),
    #[error("{0} is not supported")]
    Redirection(Redirection),
    #[error(
        "'\\|' outside quotes: the '|' opened a pipeline and the '\\' stayed \
         on the previous word"
    )]
    UnquotedAlternation,
    #[error("more than {MAX_OPERATORS} operators in one command line")]
    TooManyOperators,
}

/// Which redirection the line asked for; each shape has a different
/// alternative to suggest.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Redirection {
    Output,
    Append,
    Input,
    HereDoc,
    Stderr,
}

impl fmt::Display for Redirection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Redirection::Output => "output redirection ('>')",
            Redirection::Append => "append redirection ('>>')",
            Redirection::Input => "input redirection ('<')",
            Redirection::HereDoc => "here-document / here-string ('<<', '<<<')",
            Redirection::Stderr => "stderr redirection ('2>', '2>&1', '&>')",
        };
        f.write_str(s)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Token {
    Word(Word),
    Op(Op),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Op {
    Pipe,
    Or,
    And,
    Seq,
}

impl Op {
    fn as_str(self) -> &'static str {
        match self {
            Op::Pipe => "|",
            Op::Or => "||",
            Op::And => "&&",
            Op::Seq => ";",
        }
    }
}

/// Parse a shell-style command line into a [`Chain`] AST.
///
/// Grammar, lowest precedence first; all operators are left-associative,
/// as in bash:
///
/// ```text
/// seq     := andor ( ';'            andor? )*     (trailing ';' allowed)
/// andor   := pipe  ( ('&&' | '||')  pipe   )*
/// pipe    := cmd   ( '|'            cmd    )*
/// cmd     := WORD+
/// ```
///
/// Fails with [`ParseError::TooManyOperators`] past [`MAX_OPERATORS`].
pub fn parse_chain(input: &str) -> Result<Chain, ParseError> {
    let tokens = tokenize(input)?;
    if tokens.is_empty() {
        return Err(ParseError::Empty);
    }
    if count_operators(&tokens) > MAX_OPERATORS {
        return Err(ParseError::TooManyOperators);
    }
    Parser {
        tokens: tokens.into_iter().peekable(),
    }
    .parse_seq()
}

fn tokenize(input: &str) -> Result<Vec<Token>, ParseError> {
    let bytes = input.as_bytes();
    let mut i = 0;
    let mut tokens = Vec::new();

    while i < bytes.len() {
        let c = bytes[i];
        if c.is_ascii_whitespace() {
            i += 1;
            continue;
        }
        match c {
            b'|' => {
                if bytes.get(i + 1) == Some(&b'|') {
                    tokens.push(Token::Op(Op::Or));
                    i += 2;
                } else if follows_unquoted_backslash(&tokens) {
                    return Err(ParseError::UnquotedAlternation);
                } else {
                    tokens.push(Token::Op(Op::Pipe));
                    i += 1;
                }
            }
            b'&' => {
                if bytes.get(i + 1) == Some(&b'&') {
                    tokens.push(Token::Op(Op::And));
                    i += 2;
                } else if bytes.get(i + 1) == Some(&b'>') {
                    return Err(ParseError::Redirection(Redirection::Stderr));
                } else {
                    return Err(ParseError::Unsupported(
                        "'&' (background) not supported; only '&&' is",
                    ));
                }
            }
            b';' => {
                tokens.push(Token::Op(Op::Seq));
                i += 1;
            }
            b'>' | b'<' => {
                let Some(next) = discarded_stderr_end(&tokens, input, i) else {
                    return Err(ParseError::Redirection(redirection_kind(&tokens, bytes, i)));
                };
                tokens.pop();
                i = next;
            }
            _ => {
                let (word, next) = read_word(input, i)?;
                tokens.push(Token::Word(word));
                i = next;
            }
        }
    }

    Ok(tokens)
}

fn count_operators(tokens: &[Token]) -> usize {
    tokens.iter().filter(|t| matches!(t, Token::Op(_))).count()
}

/// An unquoted `a\|b` is a BRE alternation that the pipe would silently
/// split into two commands.
fn follows_unquoted_backslash(tokens: &[Token]) -> bool {
    matches!(tokens.last(), Some(Token::Word(w)) if !w.quoted && w.text.ends_with('\\'))
}

/// The offset just past a `2>/dev/null` whose `>` is at `i`. It is dropped
/// rather than rejected: a command's stderr is reported either way.
fn discarded_stderr_end(tokens: &[Token], input: &str, i: usize) -> Option<usize> {
    let bytes = input.as_bytes();
    let fd_adjacent = bytes[..i].last() == Some(&b'2')
        && matches!(tokens.last(), Some(Token::Word(w)) if !w.quoted && w.text == "2");
    if bytes[i] != b'>' || !fd_adjacent {
        return None;
    }
    let target_start = bytes[i + 1..]
        .iter()
        .position(|b| !b.is_ascii_whitespace())
        .map(|offset| i + 1 + offset)?;
    let (target, next) = read_word(input, target_start).ok()?;
    (target.text == "/dev/null").then_some(next)
}

/// Classify the redirection starting at `i`. An unquoted `1` or `2` word
/// just before `>` is a file-descriptor prefix, making it a stderr redirect.
fn redirection_kind(tokens: &[Token], bytes: &[u8], i: usize) -> Redirection {
    let fd_prefixed = matches!(
        tokens.last(),
        Some(Token::Word(w)) if !w.quoted && matches!(w.text.as_str(), "1" | "2")
    );
    match bytes[i] {
        b'>' if fd_prefixed => Redirection::Stderr,
        b'>' if bytes.get(i + 1) == Some(&b'>') => Redirection::Append,
        b'>' => Redirection::Output,
        _ if bytes.get(i + 1) == Some(&b'<') => Redirection::HereDoc,
        _ => Redirection::Input,
    }
}

/// Read one word starting at `start`, returning it with quotes resolved and
/// the offset just past it. Single quotes are fully literal; in double quotes
/// `\` escapes only `"` and `\`; unquoted, `\` is literal.
fn read_word(input: &str, start: usize) -> Result<(Word, usize), ParseError> {
    let bytes = input.as_bytes();
    let mut text = String::new();
    let mut quoted = false;
    let mut i = start;

    while i < bytes.len() {
        match bytes[i] {
            b'\'' => {
                quoted = true;
                i = read_single_quoted(input, i + 1, &mut text)?;
            }
            b'"' => {
                quoted = true;
                i = read_double_quoted(input, i + 1, &mut text)?;
            }
            c if c.is_ascii_whitespace() => break,
            b'|' | b'&' | b';' | b'>' | b'<' => break,
            _ => i = push_char(input, i, &mut text),
        }
    }

    Ok((Word { text, quoted }, i))
}

/// Append the single-quoted body starting at `begin`; returns the offset
/// past the closing quote.
fn read_single_quoted(input: &str, begin: usize, text: &mut String) -> Result<usize, ParseError> {
    let bytes = input.as_bytes();
    let mut i = begin;
    while i < bytes.len() && bytes[i] != b'\'' {
        i += 1;
    }
    if i >= bytes.len() {
        return Err(ParseError::UnterminatedQuote);
    }
    text.push_str(&input[begin..i]);
    Ok(i + 1)
}

/// Append the double-quoted body starting at `i`; returns the offset past
/// the closing quote.
fn read_double_quoted(input: &str, mut i: usize, text: &mut String) -> Result<usize, ParseError> {
    let bytes = input.as_bytes();
    while i < bytes.len() && bytes[i] != b'"' {
        if bytes[i] == b'\\' && matches!(bytes.get(i + 1), Some(b'"' | b'\\')) {
            text.push(bytes[i + 1] as char);
            i += 2;
        } else {
            i = push_char(input, i, text);
        }
    }
    if i >= bytes.len() {
        return Err(ParseError::UnterminatedQuote);
    }
    Ok(i + 1)
}

fn push_char(input: &str, i: usize, text: &mut String) -> usize {
    let ch = input[i..].chars().next().unwrap();
    text.push(ch);
    i + ch.len_utf8()
}

type ParseFn = fn(&mut Parser) -> Result<Chain, ParseError>;

struct Parser {
    tokens: Peekable<IntoIter<Token>>,
}

impl Parser {
    fn peek_op(&mut self) -> Option<Op> {
        match self.tokens.peek() {
            Some(Token::Op(op)) => Some(*op),
            _ => None,
        }
    }

    fn parse_seq(&mut self) -> Result<Chain, ParseError> {
        let mut left = self.parse_andor()?;
        while self.peek_op() == Some(Op::Seq) {
            self.tokens.next();
            if self.tokens.peek().is_none() {
                break;
            }
            let right = self.operand(Op::Seq, Self::parse_andor)?;
            left = Chain::Seq(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_andor(&mut self) -> Result<Chain, ParseError> {
        let mut left = self.parse_pipe()?;
        while let Some(op @ (Op::And | Op::Or)) = self.peek_op() {
            self.tokens.next();
            let right = self.operand(op, Self::parse_pipe)?;
            let join = if op == Op::And { Chain::And } else { Chain::Or };
            left = join(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_pipe(&mut self) -> Result<Chain, ParseError> {
        let mut left = self.parse_cmd()?;
        while self.peek_op() == Some(Op::Pipe) {
            self.tokens.next();
            let right = self.operand(Op::Pipe, Self::parse_cmd)?;
            left = Chain::Pipe(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    /// Parse the right-hand side of `op`, which must begin with a word.
    fn operand(&mut self, op: Op, parse: ParseFn) -> Result<Chain, ParseError> {
        match self.tokens.peek() {
            None => Err(ParseError::TrailingOperator(op.as_str())),
            Some(Token::Op(_)) => Err(ParseError::EmptyCommand),
            Some(Token::Word(_)) => parse(self),
        }
    }

    fn parse_cmd(&mut self) -> Result<Chain, ParseError> {
        let mut argv = Vec::new();
        while let Some(Token::Word(w)) = self.tokens.next_if(|t| matches!(t, Token::Word(_))) {
            argv.push(w);
        }
        if argv.is_empty() {
            return Err(match self.peek_op() {
                Some(op) => ParseError::UnexpectedOperator(op.as_str()),
                None => ParseError::Empty,
            });
        }
        Ok(Chain::Command(argv))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cmd(args: &[&str]) -> Chain {
        Chain::Command(args.iter().copied().map(Word::bare).collect())
    }

    fn pipe(l: Chain, r: Chain) -> Chain {
        Chain::Pipe(Box::new(l), Box::new(r))
    }

    fn and(l: Chain, r: Chain) -> Chain {
        Chain::And(Box::new(l), Box::new(r))
    }

    fn or(l: Chain, r: Chain) -> Chain {
        Chain::Or(Box::new(l), Box::new(r))
    }

    fn seq(l: Chain, r: Chain) -> Chain {
        Chain::Seq(Box::new(l), Box::new(r))
    }

    fn bare(text: &str) -> Token {
        Token::Word(Word::bare(text))
    }

    fn quoted(text: &str) -> Token {
        Token::Word(Word::quoted(text))
    }

    fn op(op: Op) -> Token {
        Token::Op(op)
    }

    #[test]
    fn tokenize_splits_words_and_prefers_two_char_operators() {
        for (line, expected) in [
            ("cat log.txt", vec![bare("cat"), bare("log.txt")]),
            (
                "a && b || c",
                vec![bare("a"), op(Op::And), bare("b"), op(Op::Or), bare("c")],
            ),
            (
                "a | b || c",
                vec![bare("a"), op(Op::Pipe), bare("b"), op(Op::Or), bare("c")],
            ),
        ] {
            assert_eq!(tokenize(line), Ok(expected), "{line:?}");
        }
    }

    #[test]
    fn tokenize_strips_quotes_and_marks_the_word_quoted() {
        for (line, expected) in [
            ("echo 'a|b'", vec![bare("echo"), quoted("a|b")]),
            ("echo \"a|b\"", vec![bare("echo"), quoted("a|b")]),
            (
                "echo \"he said \\\"hi\\\"\"",
                vec![bare("echo"), quoted("he said \"hi\"")],
            ),
            (r#"echo "a\\b""#, vec![bare("echo"), quoted(r"a\b")]),
            (
                "grep '.*ERROR' log.txt",
                vec![bare("grep"), quoted(".*ERROR"), bare("log.txt")],
            ),
        ] {
            assert_eq!(tokenize(line), Ok(expected), "{line:?}");
        }
    }

    #[test]
    fn tokenize_rejects_unterminated_quotes() {
        for line in ["echo 'abc", "echo \"abc"] {
            assert_eq!(
                tokenize(line),
                Err(ParseError::UnterminatedQuote),
                "{line:?}"
            );
        }
    }

    #[test]
    fn tokenize_rejects_single_ampersand() {
        assert!(matches!(tokenize("a & b"), Err(ParseError::Unsupported(_))));
    }

    #[test]
    fn tokenize_catches_unquoted_bre_alternation() {
        assert_eq!(
            tokenize(r"grep -r TODO\|FIXME AGENTS.md"),
            Err(ParseError::UnquotedAlternation)
        );
        assert_eq!(
            tokenize(r#"grep -r "TODO\|FIXME" AGENTS.md"#),
            Ok(vec![
                bare("grep"),
                bare("-r"),
                quoted(r"TODO\|FIXME"),
                bare("AGENTS.md")
            ])
        );
    }

    #[test]
    fn tokenize_rejects_redirection() {
        for (line, expected) in [
            ("echo hi > out", Redirection::Output),
            ("echo hi >> out", Redirection::Append),
            ("cat < in", Redirection::Input),
            ("cat <<< text", Redirection::HereDoc),
            ("wm list 2>&1", Redirection::Stderr),
            ("grep x f 2>err.log", Redirection::Stderr),
            ("grep x f 2>>/dev/null", Redirection::Stderr),
            ("echo 2 >/dev/null", Redirection::Stderr),
            ("ls &> out", Redirection::Stderr),
        ] {
            assert_eq!(
                tokenize(line),
                Err(ParseError::Redirection(expected)),
                "wrong classification for {line:?}"
            );
        }
    }

    #[test]
    fn tokenize_drops_stderr_sent_to_dev_null() {
        for (line, expected) in [
            (
                "grep x f 2>/dev/null",
                vec![bare("grep"), bare("x"), bare("f")],
            ),
            (
                "grep x f 2> /dev/null",
                vec![bare("grep"), bare("x"), bare("f")],
            ),
            (
                "grep x f 2>/dev/null | head",
                vec![
                    bare("grep"),
                    bare("x"),
                    bare("f"),
                    op(Op::Pipe),
                    bare("head"),
                ],
            ),
            (
                "ls 2>/dev/null; cat f",
                vec![bare("ls"), op(Op::Seq), bare("cat"), bare("f")],
            ),
        ] {
            assert_eq!(tokenize(line), Ok(expected), "{line:?}");
        }
    }

    #[test]
    fn parse_builds_the_precedence_tree() {
        for (line, expected) in [
            ("cat notes.md", cmd(&["cat", "notes.md"])),
            (
                "cat a | grep b | wc -l",
                pipe(
                    pipe(cmd(&["cat", "a"]), cmd(&["grep", "b"])),
                    cmd(&["wc", "-l"]),
                ),
            ),
            (
                "a | b && c",
                and(pipe(cmd(&["a"]), cmd(&["b"])), cmd(&["c"])),
            ),
            (
                "a && b | c",
                and(cmd(&["a"]), pipe(cmd(&["b"]), cmd(&["c"]))),
            ),
            (
                "a && b && c",
                and(and(cmd(&["a"]), cmd(&["b"])), cmd(&["c"])),
            ),
            (
                "a && b || c",
                or(and(cmd(&["a"]), cmd(&["b"])), cmd(&["c"])),
            ),
            (
                "a ; b && c",
                seq(cmd(&["a"]), and(cmd(&["b"]), cmd(&["c"]))),
            ),
            ("echo a;", cmd(&["echo", "a"])),
        ] {
            assert_eq!(parse_chain(line), Ok(expected), "{line:?}");
        }
    }

    #[test]
    fn parse_rejects_malformed_chains() {
        for (line, expected) in [
            ("", ParseError::Empty),
            ("   ", ParseError::Empty),
            ("a |", ParseError::TrailingOperator("|")),
            ("a &&", ParseError::TrailingOperator("&&")),
            ("a ||", ParseError::TrailingOperator("||")),
            ("| a", ParseError::UnexpectedOperator("|")),
            ("a ;; b", ParseError::EmptyCommand),
            ("a | | b", ParseError::EmptyCommand),
            ("a && | b", ParseError::EmptyCommand),
        ] {
            assert_eq!(parse_chain(line), Err(expected), "{line:?}");
        }
    }

    #[test]
    fn parse_caps_the_operator_count() {
        let at_limit = "echo; ".repeat(MAX_OPERATORS);
        assert!(parse_chain(&at_limit).is_ok());
        let past_limit = "echo && ".repeat(MAX_OPERATORS + 1) + "echo";
        assert_eq!(parse_chain(&past_limit), Err(ParseError::TooManyOperators));
    }
}
