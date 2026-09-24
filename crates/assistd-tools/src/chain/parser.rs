//! Command-line tokenizer and recursive-descent parser that produces a
//! [`super::Chain`] AST.

use std::iter::Peekable;
use std::vec::IntoIter;

use super::{Chain, Word};
use thiserror::Error;

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
}

/// Which redirection the line asked for. Kept apart from
/// [`ParseError::Unsupported`] because the way out differs per shape:
/// output goes through `write`, input through a pipe, and stderr is
/// already part of every result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Redirection {
    Output,
    Append,
    Input,
    HereDoc,
    Stderr,
}

impl std::fmt::Display for Redirection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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
pub fn parse_chain(input: &str) -> Result<Chain, ParseError> {
    let tokens = tokenize(input)?;
    if tokens.is_empty() {
        return Err(ParseError::Empty);
    }
    Parser {
        tokens: tokens.into_iter().peekable(),
    }
    .parse_seq()
}

fn tokenize(input: &str) -> Result<Vec<Token>, ParseError> {
    let bytes = input.as_bytes();
    let mut i = 0;
    let mut out = Vec::new();

    while i < bytes.len() {
        let c = bytes[i];
        if c.is_ascii_whitespace() {
            i += 1;
            continue;
        }
        match c {
            b'|' => {
                if bytes.get(i + 1) == Some(&b'|') {
                    out.push(Token::Op(Op::Or));
                    i += 2;
                } else {
                    // `a\|b` unquoted: the word keeps the backslash and
                    // the pipe splits the line, so a BRE-style
                    // alternation silently becomes two commands. Catch
                    // it here rather than let the second half surface as
                    // `unknown command`.
                    if matches!(out.last(), Some(Token::Word(w)) if !w.quoted && w.text.ends_with('\\'))
                    {
                        return Err(ParseError::UnquotedAlternation);
                    }
                    out.push(Token::Op(Op::Pipe));
                    i += 1;
                }
            }
            b'&' => {
                if bytes.get(i + 1) == Some(&b'&') {
                    out.push(Token::Op(Op::And));
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
                out.push(Token::Op(Op::Seq));
                i += 1;
            }
            b'>' | b'<' => {
                return Err(ParseError::Redirection(redirection_kind(&out, bytes, i)));
            }
            _ => {
                let (word, next) = read_word(input, i)?;
                out.push(Token::Word(word));
                i = next;
            }
        }
    }

    Ok(out)
}

/// Classify the redirection starting at `i`. A bare `2` (or `1`) token
/// immediately before `>` is the file-descriptor prefix of a stderr
/// redirect — the tokenizer has already pushed it as a word by the time
/// the operator is seen, so the lookback happens here.
fn redirection_kind(out: &[Token], bytes: &[u8], i: usize) -> Redirection {
    let fd_prefixed = matches!(
        out.last(),
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

/// Read a single shell-style word starting at byte offset `start`.
/// Returns the assembled word (with quotes stripped / escapes resolved)
/// and the byte offset just past the word's end.
///
/// Rules:
/// - Single quotes `'…'`: everything up to the next `'` is literal. No
///   escapes (bash-compatible).
/// - Double quotes `"…"`: everything up to the next unescaped `"` is
///   literal. `\` escapes only `"` and `\`; before anything else it is
///   itself literal, as in bash.
/// - Unquoted chars: stop on whitespace or the start of an operator
///   (`|`, `&`, `;`). Backslash outside quotes is treated as literal.
fn read_word(input: &str, start: usize) -> Result<(Word, usize), ParseError> {
    let bytes = input.as_bytes();
    let mut buf = String::new();
    let mut quoted = false;
    let mut i = start;

    while i < bytes.len() {
        let c = bytes[i];
        match c {
            b'\'' => {
                quoted = true;
                i += 1;
                let begin = i;
                while i < bytes.len() && bytes[i] != b'\'' {
                    i += 1;
                }
                if i >= bytes.len() {
                    return Err(ParseError::UnterminatedQuote);
                }
                buf.push_str(&input[begin..i]);
                i += 1;
            }
            b'"' => {
                quoted = true;
                i += 1;
                while i < bytes.len() && bytes[i] != b'"' {
                    // A backslash only guards a quote or another
                    // backslash. Consuming it before anything else
                    // would silently turn the regex "\d+" into "d+".
                    if bytes[i] == b'\\' && matches!(bytes.get(i + 1), Some(b'"' | b'\\')) {
                        buf.push(bytes[i + 1] as char);
                        i += 2;
                        continue;
                    }
                    let ch = input[i..].chars().next().unwrap();
                    buf.push(ch);
                    i += ch.len_utf8();
                }
                if i >= bytes.len() {
                    return Err(ParseError::UnterminatedQuote);
                }
                i += 1;
            }
            _ if c.is_ascii_whitespace() => break,
            b'|' | b'&' | b';' | b'>' | b'<' => break,
            _ => {
                let ch = input[i..].chars().next().unwrap();
                buf.push(ch);
                i += ch.len_utf8();
            }
        }
    }

    Ok((Word { text: buf, quoted }, i))
}

struct Parser {
    tokens: Peekable<IntoIter<Token>>,
}

type ParseFn = fn(&mut Parser) -> Result<Chain, ParseError>;

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
            // A backslash before anything but `"` or `\` is literal, so
            // the regex the model writes reaches grep intact.
            (r#"grep "\d+\s""#, vec![bare("grep"), quoted(r"\d+\s")]),
            (
                "grep '.*ERROR' log.txt",
                vec![bare("grep"), quoted(".*ERROR"), bare("log.txt")],
            ),
            // An unquoted prefix glued to a quoted tail is treated as
            // quoted as a whole: the safe reading for expansion.
            ("echo pre\"fix\"", vec![bare("echo"), quoted("prefix")]),
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
            ("grep x f 2>/dev/null", Redirection::Stderr),
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
    fn a_quoted_fd_prefix_is_not_a_stderr_redirect() {
        assert_eq!(
            tokenize(r#"echo "2" > out"#),
            Err(ParseError::Redirection(Redirection::Output))
        );
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
}
