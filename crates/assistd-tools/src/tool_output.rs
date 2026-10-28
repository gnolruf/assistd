//! The typed result of a [`Tool`](crate::Tool) call and its JSON report form.

use std::path::PathBuf;
use std::time::Duration;

use base64::Engine;
use base64::engine::general_purpose::STANDARD as B64;
use serde_json::{Value, json};

use crate::command::Attachment;

/// What a tool call produced: the text the model reads plus the outcome
/// clients are shown.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ToolOutput {
    /// The model-facing body.
    pub output: String,
    /// Zero on success; `-1` when the call itself failed rather than the
    /// command it ran.
    pub exit_code: i32,
    pub duration: Duration,
    /// Whether `output` (or a raw stream) was cut to the output cap.
    pub truncated: bool,
    /// Where the full `output` was spilled when it was cut.
    pub overflow_file: Option<PathBuf>,
    /// Images shown to the model on its next turn.
    pub attachments: Vec<Attachment>,
    /// A `run` chain's uncut streams, reported to clients only.
    pub streams: Option<RawStreams>,
}

/// The stdout and stderr heads of a command chain, before footer and framing.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RawStreams {
    pub stdout: String,
    pub stderr: String,
    /// Set when stderr overflowed and its spill file was written.
    pub stderr_overflow_file: Option<PathBuf>,
}

impl ToolOutput {
    /// A successful result whose body is `output`.
    pub fn text(output: impl Into<String>, duration: Duration) -> Self {
        Self {
            output: output.into(),
            duration,
            ..Self::default()
        }
    }

    /// A failed call (`exit_code` `-1`) whose body is `output`.
    pub fn failed(output: impl Into<String>, duration: Duration) -> Self {
        Self {
            exit_code: -1,
            ..Self::text(output, duration)
        }
    }

    /// Whole milliseconds in `duration`, saturating.
    pub fn duration_ms(&self) -> u64 {
        u64::try_from(self.duration.as_millis()).unwrap_or(u64::MAX)
    }

    /// The JSON object sent to IPC clients, with attachments base64-encoded.
    pub fn to_json(&self) -> Value {
        let mut report = json!({
            "output":      self.output,
            "exit_code":   self.exit_code,
            "truncated":   self.truncated,
            "duration_ms": self.duration_ms(),
        });
        if let Some(path) = &self.overflow_file {
            report["overflow_file"] = json!(path.to_string_lossy());
        }
        if let Some(streams) = &self.streams {
            report["stdout"] = json!(streams.stdout);
            report["stderr"] = json!(streams.stderr);
            if let Some(path) = &streams.stderr_overflow_file {
                report["stderr_overflow_file"] = json!(path.to_string_lossy());
            }
        }
        if !self.attachments.is_empty() {
            report["attachments"] = self.attachments.iter().map(attachment_json).collect();
        }
        report
    }
}

fn attachment_json(attachment: &Attachment) -> Value {
    match attachment {
        Attachment::Image { mime, bytes } => json!({
            "type": "image",
            "mime": mime,
            "data": B64.encode(bytes),
        }),
    }
}
