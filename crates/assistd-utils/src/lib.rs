//! Helpers shared across the workspace: backoff and restarts, path, file, HTTP, text, binary sniffing,
//! terminal-escaping and duration utilities, token counting, Hugging Face file IDs, readiness, and child-server and
//! pipe-worker supervision.

pub mod backoff;
pub mod binary;
pub mod child_server;
pub mod fs;
pub mod hf;
pub mod http;
pub mod log_lines;
pub mod net;
pub mod path;
pub mod pipe_worker;
pub mod process_group;
pub mod program;
pub mod readiness;
pub mod terminal_text;
pub mod text;
pub mod time;
pub mod token_counter;
