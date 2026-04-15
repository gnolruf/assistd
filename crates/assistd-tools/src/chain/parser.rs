//! Command-line tokenizer and recursive-descent parser. Produces a
//! [`super::Chain`] AST from a user-supplied string.
//!
//! Grammar (lowest precedence first; all operators left-associative —
//! matches bash semantics):
//!
//! ```text
//! seq     := andor ( ';'            andor? )*     (trailing ';' allowed)
//! andor   := pipe  ( ('&&' | '||')  pipe   )*
//! pipe    := cmd   ( '|'            cmd    )*
//! cmd     := WORD+
//! ```

use super::Chain;
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ParseError {
    #[error("empty command")]
    Empty,
    #[error("unterminated quoted string")]
    UnterminatedQuote,
    #[error("unexpected operator '{0}' at start of expression")]
    UnexpectedOperator(String),
    #[error("trailing operator '{0}'")]
    TrailingOperator(String),
    #[error("empty command between operators")]
    EmptyCommand,
    #[error("{0}")]
    Unsupported(&'static str),
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Token {
    Word(String),
    Pipe, // |
    Or,   // ||
    And,  // &&
    Seq,  // ;
}

impl Token {
    fn op_str(&self) -> Option<&'static str> {
        match self {
            Token::Pipe => Some("|"),
            Token::Or => Some("||"),
            Token::And => Some("&&"),
            Token::Seq => Some(";"),
            Token::Word(_) => None,
        }
    }
}

pub fn parse_chain(input: &str) -> Result<Chain, ParseError> {
    let tokens = tokenize(input)?;
    if tokens.is_empty() {
        return Err(ParseError::Empty);
    }
    let mut p = Parser { tokens, pos: 0 };
    let chain = p.parse_seq()?;
    if p.pos != p.tokens.len() {
        // Should be unreachable given the grammar, but belt-and-suspenders.
        return Err(ParseError::Empty);
    }
    Ok(chain)
}

// ---- tokenizer ----------------------------------------------------------

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
                if i + 1 < bytes.len() && bytes[i + 1] == b'|' {
                    out.push(Token::Or);
                    i += 2;
                } else {
                    out.push(Token::Pipe);
                    i += 1;
                }
            }
            b'&' => {
                if i + 1 < bytes.len() && bytes[i + 1] == b'&' {
                    out.push(Token::And);
                    i += 2;
                } else {
                    return Err(ParseError::Unsupported(
                        "'&' (background) not supported; only '&&' is",
                    ));
                }
            }
            b';' => {
                out.push(Token::Seq);
                i += 1;
            }
            b'>' | b'<' => {
                return Err(ParseError::Unsupported(
                    "redirection ('>', '<') not supported; use 'write' for output",
                ));
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

/// Read a single shell-style word starting at byte offset `start`.
/// Returns the assembled word (with quotes stripped / escapes resolved)
/// and the byte offset just past the word's end.
///
/// Rules:
/// - Single quotes `'…'`: everything up to the next `'` is literal. No
///   escapes (bash-compatible).
/// - Double quotes `"…"`: everything up to the next unescaped `"` is
///   literal; `\` escapes the following character.
/// - Unquoted chars: stop on whitespace or the start of an operator
///   (`|`, `&`, `;`). Backslash outside quotes is treated as literal.
fn read_word(input: &str, start: usize) -> Result<(String, usize), ParseError> {
    let bytes = input.as_bytes();
    let mut buf = String::new();
    let mut i = start;

    while i < bytes.len() {
        let c = bytes[i];
        match c {
            b'\'' => {
                i += 1;
                let begin = i;
                while i < bytes.len() && bytes[i] != b'\'' {
                    i += 1;
                }
                if i >= bytes.len() {
                    return Err(ParseError::UnterminatedQuote);
                }
                buf.push_str(std::str::from_utf8(&bytes[begin..i]).unwrap_or(""));
                i += 1; // consume closing '
            }
            b'"' => {
                i += 1;
                while i < bytes.len() && bytes[i] != b'"' {
                    if bytes[i] == b'\\' && i + 1 < bytes.len() {
                        // Escape the next byte verbatim.
                        let next = bytes[i + 1] as char;
                        buf.push(next);
                        i += 2;
                    } else {
                        let ch = input[i..].chars().next().unwrap();
                        buf.push(ch);
                        i += ch.len_utf8();
                    }
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

    Ok((buf, i))
}

// ---- parser --------------------------------------------------------------

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn bump(&mut self) -> Option<Token> {
        let t = self.tokens.get(self.pos).cloned();
        if t.is_some() {
            self.pos += 1;
        }
        t
    }

    fn parse_seq(&mut self) -> Result<Chain, ParseError> {
        let mut left = self.parse_andor()?;
        while let Some(Token::Seq) = self.peek() {
            self.bump();
            // Trailing ';' is permissible: if nothing (or another
            // terminator) follows, just return what we have so far.
            if self.peek().is_none() {
                return Ok(left);
            }
            if matches!(
                self.peek(),
                Some(Token::Pipe | Token::Or | Token::And | Token::Seq)
            ) {
                return Err(ParseError::EmptyCommand);
            }
            let right = self.parse_andor()?;
            left = Chain::Seq(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_andor(&mut self) -> Result<Chain, ParseError> {
        let mut left = self.parse_pipe()?;
        loop {
            match self.peek() {
                Some(Token::And) => {
                    self.bump();
                    let right = self.parse_pipe_require("&&")?;
                    left = Chain::And(Box::new(left), Box::new(right));
                }
                Some(Token::Or) => {
                    self.bump();
                    let right = self.parse_pipe_require("||")?;
                    left = Chain::Or(Box::new(left), Box::new(right));
                }
                _ => break,
            }
        }
        Ok(left)
    }

    fn parse_pipe(&mut self) -> Result<Chain, ParseError> {
        let mut left = self.parse_cmd()?;
        while let Some(Token::Pipe) = self.peek() {
            self.bump();
            let right = self.parse_cmd_require("|")?;
            left = Chain::Pipe(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    /// Like [`parse_pipe`], but treat missing RHS as `TrailingOperator`.
    fn parse_pipe_require(&mut self, op: &str) -> Result<Chain, ParseError> {
        match self.peek() {
            None => Err(ParseError::TrailingOperator(op.into())),
            Some(tok) if tok.op_str().is_some() => Err(ParseError::EmptyCommand),
            _ => self.parse_pipe(),
        }
    }

    fn parse_cmd(&mut self) -> Result<Chain, ParseError> {
        let mut argv: Vec<String> = Vec::new();
        while let Some(Token::Word(_)) = self.peek() {
            if let Some(Token::Word(w)) = self.bump() {
                argv.push(w);
            }
        }
        if argv.is_empty() {
            match self.peek() {
                Some(tok) if tok.op_str().is_some() => {
                    return Err(ParseError::UnexpectedOperator(
                        tok.op_str().unwrap().to_string(),
                    ));
                }
                _ => return Err(ParseError::Empty),
            }
        }
        Ok(Chain::Command(argv))
    }

    fn parse_cmd_require(&mut self, op: &str) -> Result<Chain, ParseError> {
        match self.peek() {
            None => Err(ParseError::TrailingOperator(op.into())),
            Some(tok) if tok.op_str().is_some() => Err(ParseError::EmptyCommand),
            _ => self.parse_cmd(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cmd(args: &[&str]) -> Chain {
        Chain::Command(args.iter().map(|s| s.to_string()).collect())
    }

    // -- tokenizer ----------------------------------------------------------

    #[test]
    fn tokenize_plain_words() {
        let t = tokenize("cat log.txt").unwrap();
        assert_eq!(
            t,
            vec![Token::Word("cat".into()), Token::Word("log.txt".into())]
        );
    }

    #[test]
    fn tokenize_two_char_ops_preferred() {
        let t = tokenize("a && b || c").unwrap();
        assert_eq!(
            t,
            vec![
                Token::Word("a".into()),
                Token::And,
                Token::Word("b".into()),
                Token::Or,
                Token::Word("c".into()),
            ]
        );
    }

    #[test]
    fn tokenize_single_pipe_vs_double_pipe() {
        let t = tokenize("a | b || c").unwrap();
        assert_eq!(
            t,
            vec![
                Token::Word("a".into()),
                Token::Pipe,
                Token::Word("b".into()),
                Token::Or,
                Token::Word("c".into()),
            ]
        );
    }

    #[test]
    fn tokenize_single_quotes_preserve_operators() {
        let t = tokenize("echo 'a|b'").unwrap();
        assert_eq!(
            t,
            vec![Token::Word("echo".into()), Token::Word("a|b".into())]
        );
    }

    #[test]
    fn tokenize_double_quotes_preserve_operators() {
        let t = tokenize("echo \"a|b\"").unwrap();
        assert_eq!(
            t,
            vec![Token::Word("echo".into()), Token::Word("a|b".into())]
        );
    }

    #[test]
    fn tokenize_double_quote_escape() {
        let t = tokenize("echo \"he said \\\"hi\\\"\"").unwrap();
        assert_eq!(
            t,
            vec![
                Token::Word("echo".into()),
                Token::Word("he said \"hi\"".into())
            ]
        );
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
    fn tokenize_rejects_redirection() {
        assert!(matches!(
            tokenize("echo hi > out"),
            Err(ParseError::Unsupported(_))
        ));
        assert!(matches!(
            tokenize("cat < in"),
            Err(ParseError::Unsupported(_))
        ));
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
        assert_eq!(
            parse_chain("a |"),
            Err(ParseError::TrailingOperator("|".into()))
        );
        assert_eq!(
            parse_chain("a &&"),
            Err(ParseError::TrailingOperator("&&".into()))
        );
        assert_eq!(
            parse_chain("a ||"),
            Err(ParseError::TrailingOperator("||".into()))
        );
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
    fn snapshot_acceptance_strings() {
        assert_eq!(
            format!("{:?}", parse_chain("cat notes.md").unwrap()),
            r#"Command(["cat", "notes.md"])"#
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
