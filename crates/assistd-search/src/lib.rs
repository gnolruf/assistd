//! Web search backend: the launch spec and settings file for a
//! daemon-supervised SearXNG instance, and a typed client for its JSON API.

mod client;
mod error;
mod server;
mod settings;

pub use client::{SearchHit, SearchQuery, SearchResults, SearxngClient};
pub use error::{SearchError, SearchServerError};
pub use server::SearxngServerSpec;
