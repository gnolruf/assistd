//! Built-in commands dispatched by the chain executor. Each command is
//! a small in-process Rust handler; `bash` is the escape hatch to the
//! real shell.

pub mod bash;
pub mod cat;
pub mod echo;
pub mod grep;
pub mod head_tail;
pub mod ls;
pub mod screenshot;
pub mod see;
pub mod sort;
pub mod uniq;
pub mod wc;
pub mod web;
pub mod wm;
pub mod write;

pub use bash::{BashCommand, BashPolicyCfg};
pub use cat::CatCommand;
pub use echo::EchoCommand;
pub use grep::GrepCommand;
pub use head_tail::{HeadCommand, TailCommand};
pub use ls::LsCommand;
pub use screenshot::{Backend as ScreenshotBackendKind, ScreenshotCommand, ScreenshotPolicyCfg};
pub use see::SeeCommand;
pub use sort::SortCommand;
pub use uniq::UniqCommand;
pub use wc::WcCommand;
pub use web::WebCommand;
pub use wm::WmCommand;
pub use write::{WriteCommand, WritePolicyCfg};

/// The daemon's production command set, built with test-only policy
/// stand-ins, for tests that need every name to flow through.
#[cfg(test)]
pub(crate) fn test_registry() -> crate::command::CommandRegistry {
    use assistd_wm::NoWindowManager;
    use std::sync::Arc;

    let mut r = crate::command::CommandRegistry::new();
    r.register(CatCommand);
    r.register(LsCommand);
    r.register(GrepCommand);
    r.register(WcCommand);
    r.register(HeadCommand);
    r.register(TailCommand);
    r.register(SortCommand);
    r.register(UniqCommand);
    r.register(EchoCommand);
    r.register(WriteCommand::permissive_for_tests());
    r.register(SeeCommand::default());
    r.register(ScreenshotCommand::default());
    r.register(WebCommand::new());
    r.register(BashCommand::default());
    r.register(WmCommand::for_test(Arc::new(NoWindowManager)));
    r
}
