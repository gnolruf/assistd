//! Web search backend: the launch spec for a daemon-supervised SearXNG
//! instance and the settings file it runs with.

mod error;
mod server;
mod settings;

pub use error::SearchServerError;
pub use server::SearxngServerSpec;
