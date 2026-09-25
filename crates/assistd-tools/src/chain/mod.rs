//! The command-line AST ([`Chain`]) with its parser, word expander, and
//! executor.

pub mod executor;
pub mod expand;
pub mod parser;

pub use executor::{PIPE_BUF_MAX, execute};
pub use expand::expand_args;
pub use parser::{ParseError, Redirection, parse_chain};

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
}

/// A parsed command line. Precedence, lowest first: `;` < `&&`/`||` < `|`;
/// all operators are left-associative, as in bash.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Chain {
    /// `argv[0]` is the command name; `argv[1..]` are its positional args.
    Command(Vec<Word>),
    Pipe(Box<Chain>, Box<Chain>),
    And(Box<Chain>, Box<Chain>),
    Or(Box<Chain>, Box<Chain>),
    Seq(Box<Chain>, Box<Chain>),
}
