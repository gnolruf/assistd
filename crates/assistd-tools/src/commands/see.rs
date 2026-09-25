use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;

use crate::attachment::{LoadImageError, load_image_attachment};
use crate::command::{
    Attachment, Command, CommandInput, CommandOutput, Hint, error_line, io_error_nav,
};
use crate::commands::cat::human_size;
use crate::vision::VisionGate;

/// `see PATH`: read an image file and attach it as a vision input.
pub struct SeeCommand {
    gate: Arc<VisionGate>,
}

impl SeeCommand {
    /// A `see` command that refuses to run while `gate` reports no
    /// vision support.
    pub fn new(gate: Arc<VisionGate>) -> Self {
        Self { gate }
    }
}

#[cfg(test)]
impl Default for SeeCommand {
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

    async fn run(&self, input: CommandInput) -> CommandOutput {
        if !self.gate.supported() {
            return CommandOutput::failed(
                1,
                error_line(
                    "see",
                    "vision not available: model does not support images",
                    Hint::Use,
                    "a model with mmproj loaded",
                )
                .into_bytes(),
            );
        }
        if input.args.is_empty() {
            return CommandOutput::usage(self.help());
        }
        if input.args.len() != 1 {
            return CommandOutput::usage_error(
                "see",
                "expects exactly one path argument",
                "see <PATH>",
            );
        }
        let path = &input.args[0];
        match load_image_attachment(Path::new(path)).await {
            Ok((attachment, size)) => attached(attachment, size, path),
            Err(e) => load_failed(e, path),
        }
    }
}

fn attached(attachment: Attachment, size: usize, path: &str) -> CommandOutput {
    let mime = match &attachment {
        Attachment::Image { mime, .. } => mime.clone(),
    };
    let stdout = format!("attached {mime} ({}) from {path}\n", human_size(size));
    CommandOutput {
        stdout: stdout.into_bytes(),
        stderr: Vec::new(),
        exit_code: 0,
        attachments: vec![attachment],
    }
}

fn load_failed(err: LoadImageError, path: &str) -> CommandOutput {
    let line = match err {
        LoadImageError::Io { source, .. } => io_error_nav("see", path, &source),
        e @ LoadImageError::TooLarge { .. } => error_line(
            "see",
            e.user_message(),
            Hint::Use,
            "a smaller image (resize or crop)",
        ),
        LoadImageError::Unrecognized { .. } => error_line(
            "see",
            format_args!("not an image file: {path}"),
            Hint::Use,
            format_args!("cat {path}"),
        ),
        LoadImageError::NotAnImage { detected, .. } => error_line(
            "see",
            format_args!("not an image file: {path} (detected {detected})"),
            Hint::Use,
            format_args!("cat {path}"),
        ),
        LoadImageError::UnsupportedFormat { mime, .. } => error_line(
            "see",
            format_args!("unsupported image format: {path} ({mime})"),
            Hint::Use,
            "PNG, JPEG, or WebP",
        ),
    };
    CommandOutput::failed(1, line.into_bytes())
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::*;
    use crate::fixtures::PNG_BYTES;

    async fn run_see(cmd: &SeeCommand, args: &[&str]) -> CommandOutput {
        cmd.run(CommandInput {
            args: args.iter().map(|s| s.to_string()).collect(),
            stdin: None,
        })
        .await
    }

    #[tokio::test]
    async fn attaches_png_image() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("shot.png");
        std::fs::write(&path, PNG_BYTES).unwrap();
        let path = path.to_string_lossy().into_owned();
        let out = run_see(&SeeCommand::default(), &[&path]).await;
        assert_eq!(out.exit_code, 0);
        assert_eq!(
            String::from_utf8_lossy(&out.stdout),
            format!("attached image/png ({}B) from {path}\n", PNG_BYTES.len())
        );
        let [Attachment::Image { mime, bytes }] = out.attachments.as_slice() else {
            panic!("expected one attachment: {:?}", out.attachments);
        };
        assert_eq!(mime, "image/png");
        assert_eq!(bytes.as_slice(), PNG_BYTES);
    }

    #[tokio::test]
    async fn rejects_non_image() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("notes.txt");
        std::fs::write(&path, b"not an image").unwrap();
        let path = path.to_string_lossy().into_owned();
        let out = run_see(&SeeCommand::default(), &[&path]).await;
        assert_eq!(out.exit_code, 1);
        assert_eq!(
            String::from_utf8_lossy(&out.stderr),
            format!("[error] see: not an image file: {path}. Use: cat {path}\n")
        );
        assert!(out.attachments.is_empty());
    }

    #[tokio::test]
    async fn missing_file_exits_1() {
        let out = run_see(&SeeCommand::default(), &["/nonexistent/image.png"]).await;
        assert_eq!(out.exit_code, 1);
        assert_eq!(
            String::from_utf8_lossy(&out.stderr),
            "[error] see: file not found: /nonexistent/image.png. \
             Use: ls /nonexistent to see what is there\n"
        );
        assert!(out.attachments.is_empty());
    }

    #[tokio::test]
    async fn no_args_emits_usage() {
        let out = run_see(&SeeCommand::default(), &[]).await;
        assert_eq!(out.exit_code, 2);
        assert!(out.stdout.starts_with(b"usage: see"), "{out:?}");
    }

    #[tokio::test]
    async fn too_many_args_errors() {
        let out = run_see(&SeeCommand::default(), &["a.png", "b.png"]).await;
        assert_eq!(out.exit_code, 2);
        assert_eq!(
            String::from_utf8_lossy(&out.stderr),
            "[error] see: expects exactly one path argument. Use: see <PATH>\n"
        );
    }

    #[tokio::test]
    async fn vision_disabled_returns_exact_error() {
        let cmd = SeeCommand::new(VisionGate::new(false));
        let out = run_see(&cmd, &["/tmp/some-image.png"]).await;
        assert_eq!(out.exit_code, 1);
        assert!(out.stdout.is_empty());
        assert_eq!(
            String::from_utf8_lossy(&out.stderr),
            "[error] see: vision not available: model does not support images. \
             Use: a model with mmproj loaded\n"
        );
        assert!(out.attachments.is_empty());
    }

    #[test]
    fn gate_flip_changes_summary_dynamically() {
        let gate = VisionGate::new(true);
        let cmd = SeeCommand::new(gate.clone());
        assert!(cmd.summary().contains("attach an image"));
        gate.set(false);
        assert!(cmd.summary().contains("unavailable"));
        gate.set(true);
        assert!(cmd.summary().contains("attach an image"));
    }
}
