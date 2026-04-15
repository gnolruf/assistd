//! Built-in commands dispatched by the chain executor. Each command is
//! a small in-process Rust handler; `bash` is the escape hatch to the
//! real shell.

pub mod bash;
pub mod cat;
pub mod echo;
pub mod grep;
pub mod ls;
pub mod see;
pub mod wc;
pub mod web;
pub mod write;

pub use bash::BashCommand;
pub use cat::CatCommand;
pub use echo::EchoCommand;
pub use grep::GrepCommand;
pub use ls::LsCommand;
pub use see::SeeCommand;
pub use wc::WcCommand;
pub use web::WebCommand;
pub use write::WriteCommand;
