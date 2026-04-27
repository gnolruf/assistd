//! `see PATH` — read an image file and attach it as an
//! [`crate::Attachment::Image`] on the command output. The chain
//! executor threads the attachment through pipes, and `RunTool` surfaces
//! it in the JSON tool result; the chat loop (separate ticket) is
//! responsible for turning that into a vision input on the model's
//! next turn.

use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;

use crate::attachment::{LoadImageError, load_image_attachment};
use crate::command::{Attachment, Command, CommandInput, CommandOutput, error_line, io_error_nav};
use crate::commands::cat::human_size;
use crate::vision::VisionGate;

pub struct SeeCommand {
    /// Shared, runtime-mutable vision flag. Read on every `run()` so a
    /// model swap on the running llama-server (revalidated by the
    /// daemon) flips the gate without rebuilding the registry.
    gate: Arc<VisionGate>,
}

impl SeeCommand {
    pub fn new(gate: Arc<VisionGate>) -> Self {
        Self { gate }
    }
}

#[cfg(test)]
impl Default for SeeCommand {
    /// Test-only default: vision enabled. Lets the
    /// convention-compliance harness in `command.rs` and the
    /// per-command tests construct an instance without rethreading the
    /// flag through every call site.
    fn default() -> Self {
        Self::new(VisionGate::new(true))
    }
}

#[async_trait]
impl Command for SeeCommand {
    fn name(&self) -> &str {
        "see"
    }

    fn summary(&self) -> &'static str {
        if self.gate.supported() {
            "attach an image file as a vision input for the next LLM turn"
        } else {
            "(unavailable: model has no vision encoder)"
        }
    }

    fn help(&self) -> String {
        "usage: see PATH\n\
         \n\
         Read the image file at PATH and attach it to the tool result as a \
         vision input. The chat loop surfaces the attachment on the model's \
         next turn. Attachments flow through pipes untouched (e.g. \
         `see img.png | wc` still surfaces the image).\n\
         \n\
         Exit 1 if the file is missing or not a recognized image format.\n"
            .to_string()
    }

    async fn run(&self, input: CommandInput) -> Result<CommandOutput> {
        if !self.gate.supported() {
            return Ok(CommandOutput::failed(
                1,
                error_line(
                    "see",
                    "vision not available: model does not support images",
                    "Use",
                    "a model with mmproj loaded",
                )
                .into_bytes(),
            ));
        }
        if input.args.is_empty() {
            return Ok(CommandOutput {
                stdout: self.help().into_bytes(),
                stderr: Vec::new(),
                exit_code: 2,
                attachments: Vec::new(),
            });
        }
        if input.args.len() != 1 {
            return Ok(CommandOutput::failed(
                2,
                error_line(
                    "see",
                    "expects exactly one path argument",
                    "Use",
                    "see <PATH>",
                )
                .into_bytes(),
            ));
        }
        let path = &input.args[0];
        match load_image_attachment(Path::new(path)).await {
            Ok((attachment, size)) => {
                let mime = match &attachment {
                    Attachment::Image { mime, .. } => mime.clone(),
                };
                let stdout = format!("attached {mime} ({}) from {path}\n", human_size(size));
                Ok(CommandOutput {
                    stdout: stdout.into_bytes(),
                    stderr: Vec::new(),
                    exit_code: 0,
                    attachments: vec![attachment],
                })
            }
            Err(LoadImageError::Io { source, .. }) => Ok(CommandOutput::failed(
                1,
                io_error_nav("see", path, &source).into_bytes(),
            )),
            Err(e @ LoadImageError::TooLarge { .. }) => Ok(CommandOutput::failed(
                1,
                error_line(
                    "see",
                    e.user_message(),
                    "Use",
                    "a smaller image (resize or crop)",
                )
                .into_bytes(),
            )),
            Err(LoadImageError::Unrecognized { .. }) => Ok(CommandOutput::failed(
                1,
                error_line(
                    "see",
                    format_args!("not an image file: {path}"),
                    "Use",
                    format_args!("cat {path}"),
                )
                .into_bytes(),
            )),
            Err(LoadImageError::NotAnImage { detected, .. }) => Ok(CommandOutput::failed(
                1,
                error_line(
                    "see",
                    format_args!("not an image file: {path} (detected {detected})"),
                    "Use",
                    format_args!("cat {path}"),
                )
                .into_bytes(),
            )),
            Err(LoadImageError::UnsupportedFormat { mime, .. }) => Ok(CommandOutput::failed(
                1,
                error_line(
                    "see",
                    format_args!("unsupported image format: {path} ({mime})"),
                    "Use",
                    "PNG, JPEG, or WebP",
                )
                .into_bytes(),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    const PNG_BYTES: &[u8] = &[
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44,
        0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x06, 0x00, 0x00, 0x00, 0x1F,
        0x15, 0xC4, 0x89, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9C, 0x63, 0x00,
        0x01, 0x00, 0x00, 0x05, 0x00, 0x01, 0x0D, 0x0A, 0x2D, 0xB4, 0x00, 0x00, 0x00, 0x00, 0x49,
        0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
    ];

    #[tokio::test]
    async fn attaches_png_image() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("shot.png");
        tokio::fs::write(&path, PNG_BYTES).await.unwrap();
        let out = SeeCommand::default()
            .run(CommandInput {
                args: vec![path.to_string_lossy().into_owned()],
                stdin: Vec::new(),
            })
            .await
            .unwrap();
        assert_eq!(out.exit_code, 0);
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(stdout.contains("attached image/png"), "{stdout}");
        assert_eq!(out.attachments.len(), 1);
        match &out.attachments[0] {
            Attachment::Image { mime, bytes } => {
                assert_eq!(mime, "image/png");
                assert_eq!(bytes.as_slice(), PNG_BYTES);
            }
        }
    }

    #[tokio::test]
    async fn rejects_non_image() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("notes.txt");
        tokio::fs::write(&path, b"not an image").await.unwrap();
        let out = SeeCommand::default()
            .run(CommandInput {
                args: vec![path.to_string_lossy().into_owned()],
                stdin: Vec::new(),
            })
            .await
            .unwrap();
        assert_eq!(out.exit_code, 1);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            stderr.contains("[error] see: not an image file:"),
            "{stderr}"
        );
        assert!(stderr.contains("Use: cat "), "{stderr}");
        assert!(out.attachments.is_empty());
    }

    #[tokio::test]
    async fn missing_file_exits_1() {
        let out = SeeCommand::default()
            .run(CommandInput {
                args: vec!["/nonexistent/image.png".into()],
                stdin: Vec::new(),
            })
            .await
            .unwrap();
        assert_eq!(out.exit_code, 1);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            stderr.contains("[error] see: file not found: /nonexistent/image.png"),
            "{stderr}"
        );
        assert!(stderr.contains("Use: ls to check the path"), "{stderr}");
        assert!(out.attachments.is_empty());
    }

    #[tokio::test]
    async fn no_args_errors() {
        let out = SeeCommand::default()
            .run(CommandInput {
                args: Vec::new(),
                stdin: Vec::new(),
            })
            .await
            .unwrap();
        assert_eq!(out.exit_code, 2);
    }

    #[tokio::test]
    async fn too_many_args_errors() {
        let out = SeeCommand::default()
            .run(CommandInput {
                args: vec!["a.png".into(), "b.png".into()],
                stdin: Vec::new(),
            })
            .await
            .unwrap();
        assert_eq!(out.exit_code, 2);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            stderr.contains("[error] see: expects exactly one path argument"),
            "{stderr}"
        );
        assert!(stderr.contains("Use: see <PATH>"), "{stderr}");
    }

    /// AC #3: when vision is disabled, `see` short-circuits with the
    /// exact wording "vision not available: model does not support
    /// images" and never touches the filesystem (so a real path
    /// argument is irrelevant; we still pass one to mirror normal
    /// usage).
    #[tokio::test]
    async fn vision_disabled_returns_exact_error() {
        let out = SeeCommand::new(VisionGate::new(false))
            .run(CommandInput {
                args: vec!["/tmp/some-image.png".into()],
                stdin: Vec::new(),
            })
            .await
            .unwrap();
        assert_eq!(out.exit_code, 1);
        assert!(out.stdout.is_empty());
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            stderr.contains("[error] see: vision not available: model does not support images"),
            "{stderr}"
        );
        assert!(
            stderr.contains("Use: a model with mmproj loaded"),
            "{stderr}"
        );
        assert!(out.attachments.is_empty());
    }

    #[test]
    fn summary_changes_when_vision_disabled() {
        assert!(
            SeeCommand::new(VisionGate::new(true))
                .summary()
                .contains("attach an image")
        );
        assert!(
            SeeCommand::new(VisionGate::new(false))
                .summary()
                .contains("unavailable")
        );
    }

    #[test]
    fn gate_flip_changes_summary_dynamically() {
        // The whole point of VisionGate: a single Arc shared with the
        // daemon's revalidation path can flip a long-lived command from
        // "available" to "unavailable" mid-process without rebuilding.
        let gate = VisionGate::new(true);
        let cmd = SeeCommand::new(gate.clone());
        assert!(cmd.summary().contains("attach an image"));
        gate.set(false);
        assert!(cmd.summary().contains("unavailable"));
        gate.set(true);
        assert!(cmd.summary().contains("attach an image"));
    }
}
