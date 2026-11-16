//! Streaming client for the locally-managed llama-server: one message history, wire types
//! for `/v1/chat/completions`, and the SSE and `<think>` decoders the stream runs through.

pub mod client;
pub mod error;
pub mod history;
pub mod sse;
pub mod think_splitter;
pub mod wire;

pub use client::LlamaClient;
pub use error::LlamaClientError;
