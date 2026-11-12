//! What a finished tool call reports to clients.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::ImageMime;

/// The outcome of one tool call, as clients are shown it.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolReport {
    /// The text the model read.
    pub output: String,
    /// Zero on success; `-1` when the call itself failed rather than the command it ran.
    pub exit_code: i32,
    /// Whether `output` or a raw stream was cut to the output cap.
    pub truncated: bool,
    pub duration_ms: u64,
    /// Where the full `output` was spilled when it was cut.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub overflow_file: Option<PathBuf>,
    /// A `run` chain's uncut streams.
    #[serde(flatten)]
    pub streams: Option<ToolStreams>,
    /// What the call showed the model beside `output`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub attachments: Vec<ToolAttachment>,
}

impl ToolReport {
    /// A copy that carries no `attachments`.
    pub fn without_attachments(&self) -> Self {
        Self {
            output: self.output.clone(),
            exit_code: self.exit_code,
            truncated: self.truncated,
            duration_ms: self.duration_ms,
            overflow_file: self.overflow_file.clone(),
            streams: self.streams.clone(),
            attachments: Vec::new(),
        }
    }
}

/// The stdout and stderr heads of a command chain, before footer and framing.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolStreams {
    pub stdout: String,
    pub stderr: String,
    /// Where the full stderr was spilled when it overflowed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stderr_overflow_file: Option<PathBuf>,
}

/// A payload a tool call produced beside its text.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolAttachment {
    /// An image; `data` is standard padded base64.
    Image { mime: ImageMime, data: String },
}
