//! Parsed-command AST (`Chain`), the parser that builds it, and the
//! executor that walks it. The parser and executor are kept as separate
//! modules so the AST is the single shared contract between them —
//! makes it easy to unit-test either side in isolation.

pub mod executor;
pub mod parser;

pub use executor::{PIPE_BUF_MAX, execute};
pub use parser::{ParseError, parse_chain};

/// A parsed command line. The tree is right-skewed per precedence level:
/// looser operators sit closer to the root so a post-order walk runs the
/// leftmost stage first. Operator precedence (lowest → highest):
/// `;` < `&&`/`||` < `|`. All operators are left-associative, matching
/// bash.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Chain {
    /// `argv[0]` is the command name; `argv[1..]` are its positional args.
    Command(Vec<String>),
    Pipe(Box<Chain>, Box<Chain>),
    And(Box<Chain>, Box<Chain>),
    Or(Box<Chain>, Box<Chain>),
    Seq(Box<Chain>, Box<Chain>),
}
