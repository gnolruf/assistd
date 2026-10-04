//! Helpers shared across the workspace: backoff and restart accounting,
//! path, file and text helpers, XDG lookups, and the supervised child-server
//! stack.

pub mod backoff;
#[cfg(feature = "child-server")]
pub mod child_server;
pub mod fs;
#[cfg(feature = "process")]
pub mod log_lines;
pub mod path;
#[cfg(feature = "process")]
pub mod process_group;
#[cfg(feature = "process")]
pub mod procfs;
pub mod text;
#[cfg(feature = "tracing-init")]
pub mod tracing_init;
pub mod xdg;
