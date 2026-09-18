//! Streaming chat client for the locally-managed llama-server process:
//! one conversation, wire types for `/v1/chat/completions`, and the SSE
//! and `<think>` decoders the stream runs through.

pub mod client;
pub mod conversation;
pub mod error;
pub mod sse;
pub mod think_splitter;
pub mod wire;

pub use client::LlamaChatClient;
pub use error::ChatClientError;
