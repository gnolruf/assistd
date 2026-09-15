//! Parsed-command AST (`Chain`), the parser that builds it, the word
//! expander, and the executor that walks it. The parser and executor are
//! kept as separate modules so the AST is the single shared contract
//! between them; that makes it easy to unit-test either side in
//! isolation.

pub mod executor;
pub mod expand;
pub mod parser;

pub use executor::{PIPE_BUF_MAX, execute};
pub use expand::expand_args;
pub use parser::{ParseError, Redirection, parse_chain};

/// One argv entry as written on the command line, plus whether any part
/// of it was quoted.
///
/// Quoting rides along in the AST because it is what suppresses
/// expansion: `cat *.toml` names every TOML file in the directory,
/// while `grep '.*ERROR'` is a regex that must reach `grep` untouched.
/// Without the flag the executor cannot tell the two apart.
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

/// A parsed command line. The tree is right-skewed per precedence level:
/// looser operators sit closer to the root so a post-order walk runs the
/// leftmost stage first. Operator precedence (lowest → highest):
/// `;` < `&&`/`||` < `|`. All operators are left-associative, matching
/// bash.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Chain {
    /// `argv[0]` is the command name; `argv[1..]` are its positional args.
    Command(Vec<Word>),
    Pipe(Box<Chain>, Box<Chain>),
    And(Box<Chain>, Box<Chain>),
    Or(Box<Chain>, Box<Chain>),
    Seq(Box<Chain>, Box<Chain>),
}
