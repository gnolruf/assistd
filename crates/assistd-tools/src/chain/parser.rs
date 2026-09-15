//! Command-line tokenizer and recursive-descent parser. Produces a
//! [`super::Chain`] AST from a user-supplied string.
//!
//! Grammar (lowest precedence first; all operators left-associative,
//! matching bash semantics):
//!
//! ```text
//! seq     := andor ( ';'            andor? )*     (trailing ';' allowed)
//! andor   := pipe  ( ('&&' | '||')  pipe   )*
//! pipe    := cmd   ( '|'            cmd    )*
//! cmd     := WORD+
//! ```

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
/// already in the result the caller gets back.
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
/// # Errors
///
/// Returns [`ParseError`] when the input is empty, contains an unterminated
/// quote, an unexpected or trailing operator, or an unsupported shell feature.
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
                i += 1; // consume closing '
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
                i += 1; // consume closing "
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

    fn bare(text: &str) -> Token {
        Token::Word(Word::bare(text))
    }

    fn quoted(text: &str) -> Token {
        Token::Word(Word::quoted(text))
    }

    fn op(op: Op) -> Token {
        Token::Op(op)
    }

    // -- tokenizer ----------------------------------------------------------

    #[test]
    fn tokenize_plain_words() {
        let t = tokenize("cat log.txt").unwrap();
        assert_eq!(t, vec![bare("cat"), bare("log.txt")]);
    }

    #[test]
    fn tokenize_two_char_ops_preferred() {
        let t = tokenize("a && b || c").unwrap();
        assert_eq!(
            t,
            vec![bare("a"), op(Op::And), bare("b"), op(Op::Or), bare("c")]
        );
    }

    #[test]
    fn tokenize_single_pipe_vs_double_pipe() {
        let t = tokenize("a | b || c").unwrap();
        assert_eq!(
            t,
            vec![bare("a"), op(Op::Pipe), bare("b"), op(Op::Or), bare("c")]
        );
    }

    #[test]
    fn tokenize_single_quotes_preserve_operators() {
        let t = tokenize("echo 'a|b'").unwrap();
        assert_eq!(t, vec![bare("echo"), quoted("a|b")]);
    }

    #[test]
    fn tokenize_double_quotes_preserve_operators() {
        let t = tokenize("echo \"a|b\"").unwrap();
        assert_eq!(t, vec![bare("echo"), quoted("a|b")]);
    }

    #[test]
    fn tokenize_double_quote_escape() {
        let t = tokenize("echo \"he said \\\"hi\\\"\"").unwrap();
        assert_eq!(t, vec![bare("echo"), quoted("he said \"hi\"")]);
    }

    #[test]
    fn parse_keeps_a_quoted_pipe_inside_one_command() {
        let chain = parse_chain(r#"grep "Command|Tool" docs/tools.md | wc -l"#).unwrap();
        assert_eq!(
            chain,
            Chain::Pipe(
                Box::new(Chain::Command(vec![
                    Word::bare("grep"),
                    Word::quoted("Command|Tool"),
                    Word::bare("docs/tools.md"),
                ])),
                Box::new(cmd(&["wc", "-l"])),
            )
        );
    }

    #[test]
    fn tokenize_double_quotes_keep_regex_backslashes() {
        // The pattern the model actually writes must survive intact;
        // eating the backslash here silently changes what grep matches.
        let t = tokenize(r#"grep "\d+\s""#).unwrap();
        assert_eq!(t, vec![bare("grep"), quoted(r"\d+\s")]);
    }

    #[test]
    fn tokenize_double_quote_escapes_only_quote_and_backslash() {
        let t = tokenize(r#"echo "a\\b""#).unwrap();
        assert_eq!(t, vec![bare("echo"), quoted(r"a\b")]);
    }

    #[test]
    fn tokenize_unterminated_single_quote() {
        assert_eq!(tokenize("echo 'abc"), Err(ParseError::UnterminatedQuote));
    }

    #[test]
    fn tokenize_unterminated_double_quote() {
        assert_eq!(tokenize("echo \"abc"), Err(ParseError::UnterminatedQuote));
    }

    #[test]
    fn tokenize_empty_is_empty() {
        assert!(tokenize("").unwrap().is_empty());
        assert!(tokenize("   ").unwrap().is_empty());
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
        // Quoted patterns are the caller's business, pipe and all.
        assert!(tokenize(r#"grep -r "TODO\|FIXME" AGENTS.md"#).is_ok());
        assert!(tokenize(r"ls | grep x").is_ok());
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

    // -- parser -------------------------------------------------------------

    #[test]
    fn parse_single_command() {
        let c = parse_chain("cat notes.md").unwrap();
        assert_eq!(c, cmd(&["cat", "notes.md"]));
    }

    #[test]
    fn parse_pipeline() {
        let c = parse_chain("cat a | grep b | wc -l").unwrap();
        // Left-associative: ((cat | grep) | wc)
        assert_eq!(
            c,
            Chain::Pipe(
                Box::new(Chain::Pipe(
                    Box::new(cmd(&["cat", "a"])),
                    Box::new(cmd(&["grep", "b"])),
                )),
                Box::new(cmd(&["wc", "-l"])),
            )
        );
    }

    #[test]
    fn parse_pipe_binds_tighter_than_andor() {
        // `a | b && c` → And(Pipe(a, b), c)
        let c = parse_chain("a | b && c").unwrap();
        assert_eq!(
            c,
            Chain::And(
                Box::new(Chain::Pipe(Box::new(cmd(&["a"])), Box::new(cmd(&["b"])))),
                Box::new(cmd(&["c"])),
            )
        );
    }

    #[test]
    fn parse_andor_binds_on_right_of_pipe() {
        // `a && b | c` → And(a, Pipe(b, c))
        let c = parse_chain("a && b | c").unwrap();
        assert_eq!(
            c,
            Chain::And(
                Box::new(cmd(&["a"])),
                Box::new(Chain::Pipe(Box::new(cmd(&["b"])), Box::new(cmd(&["c"]))))
            )
        );
    }

    #[test]
    fn parse_seq_binds_loosest() {
        // `a ; b && c` → Seq(a, And(b, c))
        let c = parse_chain("a ; b && c").unwrap();
        assert_eq!(
            c,
            Chain::Seq(
                Box::new(cmd(&["a"])),
                Box::new(Chain::And(Box::new(cmd(&["b"])), Box::new(cmd(&["c"])))),
            )
        );
    }

    #[test]
    fn parse_left_associative_andand() {
        let c = parse_chain("a && b && c").unwrap();
        assert_eq!(
            c,
            Chain::And(
                Box::new(Chain::And(Box::new(cmd(&["a"])), Box::new(cmd(&["b"])))),
                Box::new(cmd(&["c"])),
            )
        );
    }

    #[test]
    fn parse_trailing_semicolon_accepted() {
        let c = parse_chain("echo a;").unwrap();
        assert_eq!(c, cmd(&["echo", "a"]));
    }

    #[test]
    fn parse_empty_is_error() {
        assert_eq!(parse_chain(""), Err(ParseError::Empty));
        assert_eq!(parse_chain("   "), Err(ParseError::Empty));
    }

    #[test]
    fn parse_trailing_operator_is_error() {
        assert_eq!(parse_chain("a |"), Err(ParseError::TrailingOperator("|")));
        assert_eq!(parse_chain("a &&"), Err(ParseError::TrailingOperator("&&")));
        assert_eq!(parse_chain("a ||"), Err(ParseError::TrailingOperator("||")));
    }

    #[test]
    fn parse_leading_operator_is_error() {
        assert!(matches!(
            parse_chain("| a"),
            Err(ParseError::UnexpectedOperator(_))
        ));
    }

    #[test]
    fn parse_empty_inter_op_is_error() {
        assert_eq!(parse_chain("a ;; b"), Err(ParseError::EmptyCommand));
        assert_eq!(parse_chain("a | | b"), Err(ParseError::EmptyCommand));
        assert_eq!(parse_chain("a && | b"), Err(ParseError::EmptyCommand));
    }

    // -- snapshot of acceptance-criteria AST shape --------------------------

    #[test]
    fn quoting_is_recorded_on_the_word() {
        // The executor reads this flag to decide whether a word is a
        // glob to expand or a literal (a regex, a path with a `*`).
        let c = parse_chain("grep '.*ERROR' log.txt").unwrap();
        assert_eq!(
            c,
            Chain::Command(vec![
                Word::bare("grep"),
                Word::quoted(".*ERROR"),
                Word::bare("log.txt"),
            ])
        );
    }

    #[test]
    fn partially_quoted_word_counts_as_quoted() {
        // `--flag="a b"` glues an unquoted prefix onto a quoted tail;
        // treating the whole word as quoted is the safe reading.
        let c = parse_chain("echo pre\"fix\"").unwrap();
        assert_eq!(
            c,
            Chain::Command(vec![Word::bare("echo"), Word::quoted("prefix")])
        );
    }

    #[test]
    fn snapshot_acceptance_strings() {
        assert_eq!(
            parse_chain("cat notes.md").unwrap(),
            cmd(&["cat", "notes.md"])
        );
        let piped = format!(
            "{:?}",
            parse_chain("cat log.txt | grep ERROR | wc -l").unwrap()
        );
        assert!(piped.starts_with("Pipe(Pipe("), "got {piped}");
        let or = format!(
            "{:?}",
            parse_chain("cat missing.txt || echo 'not found'").unwrap()
        );
        assert!(or.starts_with("Or("), "got {or}");
        let and = format!("{:?}", parse_chain("ls /tmp && echo done").unwrap());
        assert!(and.starts_with("And("), "got {and}");
        let seq = format!("{:?}", parse_chain("echo hello ; echo world").unwrap());
        assert!(seq.starts_with("Seq("), "got {seq}");
    }
}
