//! The command-line AST ([`Chain`]) with its parser, word expander, and
//! executor.

use std::fmt;

pub mod executor;
pub mod expand;
pub mod parser;

pub use executor::{OUTPUT_MAX, execute};
pub use expand::expand_args;
pub use parser::{MAX_OPERATORS, ParseError, Redirection, parse_chain};

/// One argv entry as written, plus whether any part of it was quoted.
/// Quoting suppresses tilde and glob expansion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Word {
    pub text: String,
    pub quoted: bool,
}

impl Word {
    /// An unquoted word, subject to tilde and glob expansion.
    pub fn bare(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            quoted: false,
        }
    }

    /// A word that carried quotes, passed through expansion verbatim.
    pub fn quoted(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            quoted: true,
        }
    }

    /// The texts of `words` joined by single spaces.
    pub fn joined_by_spaces(words: &[Self]) -> String {
        let texts: Vec<&str> = words.iter().map(Self::as_str).collect();
        texts.join(" ")
    }

    /// The word's text, without the quotes it was written in.
    pub fn as_str(&self) -> &str {
        &self.text
    }
}

impl PartialEq<str> for Word {
    fn eq(&self, text: &str) -> bool {
        self.text == text
    }
}

impl fmt::Display for Word {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.text)
    }
}

/// A parsed command line. Precedence, lowest first: `;` < `&&`/`||` < `|`;
/// all operators are left-associative, as in bash.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Chain {
    /// `argv[0]` is the command name; `argv[1..]` are its positional args.
    Command(Vec<Word>),
    /// The left side's stdout is the stdin of the command on the right.
    Pipe(Box<Chain>, Vec<Word>),
    And(Box<Chain>, Box<Chain>),
    Or(Box<Chain>, Box<Chain>),
    Seq(Box<Chain>, Box<Chain>),
}
