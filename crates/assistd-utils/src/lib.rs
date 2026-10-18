//! Helpers shared across the workspace: backoff and restart accounting,
//! path, file, program-lookup, capped HTTP body and text helpers, per-turn
//! token counting, Hugging Face file IDs and downloads, subsystem readiness,
//! and the supervised child-server stack.

pub mod backoff;
pub mod child_server;
pub mod fs;
pub mod hf;
pub mod http;
pub mod log_lines;
pub mod path;
pub mod process_group;
pub mod program;
pub mod readiness;
pub mod text;
pub mod token_counter;
pub mod tracing_init;
