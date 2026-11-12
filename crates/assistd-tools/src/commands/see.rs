//! `see`: attach an image file as vision input for the next model turn.

use std::path::Path;
use std::sync::Arc;

use assistd_utils::text::human_size;
use async_trait::async_trait;

use crate::attachment::{LoadImageError, load_image_attachment};
use crate::chain::Word;
use crate::command::{Attachment, Command, CommandError, CommandInput, CommandOutput, Hint, Usage};
use crate::vision::VisionGate;

/// `see PATH`: read an image file and attach it as a vision input.
#[derive(Debug)]
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

#[async_trait]
impl Command for SeeCommand {
    fn name(&self) -> &'static str {
        "see"
    }

    fn summary(&self) -> &'static str {
        if self.gate.supported() {
            "attach an image file as a vision input for the next LLM turn"
        } else {
            "(unavailable: model has no vision encoder)"
        }
    }

    fn usage(&self) -> Usage {
        Usage::new(
            "PATH",
            "Read the image file at PATH and attach it to the tool result as a \
         vision input. The chat loop surfaces the attachment on the model's \
         next turn. Attachments flow through pipes untouched (e.g. \
         `see img.png | wc` still surfaces the image).\n\
         \n\
         Exit 1 if the file is missing or not a recognized image format.\n",
        )
    }

    async fn run(&self, input: CommandInput) -> Result<CommandOutput, CommandError> {
        if !self.gate.supported() {
            return Err(CommandError::failed(
                "vision not available: model does not support images",
                Hint::Use,
                "a model with mmproj loaded",
            ));
        }
        let path = match input.args.as_slice() {
            [] => return Ok(CommandOutput::usage(self.help())),
            [path] => path,
            _ => {
                return Err(CommandError::usage(
                    "expects exactly one path argument",
                    "see <PATH>",
                ));
            }
        };
        let (attachment, size) = load_image_attachment(Path::new(path.as_str()))
            .await
            .map_err(|e| load_failed(e, path))?;
        Ok(attached(attachment, size, path))
    }
}

fn attached(attachment: Attachment, size: usize, path: &Word) -> CommandOutput {
    let mime = match &attachment {
        Attachment::Image { mime, .. } => *mime,
    };
    let stdout = format!(
        "attached {mime} ({}) from {path}\n",
        human_size(size as u64)
    );
    CommandOutput {
        stdout: stdout.into_bytes(),
        stderr: Vec::new(),
        exit_code: 0,
        attachments: vec![attachment],
    }
}

fn load_failed(err: LoadImageError, path: &Word) -> CommandError {
    match err {
        LoadImageError::Io { source, .. } => CommandError::io(path, &source),
        e @ LoadImageError::TooLarge { .. } => CommandError::failed(
            e.user_message(),
            Hint::Use,
            "a smaller image (resize or crop)",
        ),
        LoadImageError::Unrecognized { .. } => CommandError::failed(
            format_args!("not an image file: {path}"),
            Hint::Use,
            format_args!("cat {path}"),
        ),
        LoadImageError::NotAnImage { detected, .. } => CommandError::failed(
            format_args!("not an image file: {path} (detected {detected})"),
            Hint::Use,
            format_args!("cat {path}"),
        ),
        LoadImageError::UnsupportedFormat { mime, .. } => CommandError::failed(
            format_args!("unsupported image format: {path} ({mime})"),
            Hint::Use,
            "PNG, JPEG, or WebP",
        ),
    }
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use assistd_ipc::ImageMime;

    use super::*;
    use crate::TurnContext;
    use crate::command::run_rendered;
    use crate::commands::test_see;
    use crate::fixtures::PNG_BYTES;

    async fn run_see(cmd: &SeeCommand, args: &[&str]) -> CommandOutput {
        let input = CommandInput {
            args: args.iter().copied().map(Word::bare).collect(),
            stdin: None,
            scope: crate::FileScope::Host,
            turn: TurnContext::default(),
        };
        run_rendered(cmd, input).await
    }

    #[tokio::test]
    async fn attaches_png_image() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("shot.png");
        std::fs::write(&path, PNG_BYTES).unwrap();
        let path = path.to_string_lossy().into_owned();
        let out = run_see(&test_see(), &[&path]).await;
        assert_eq!(out.exit_code, 0);
        assert_eq!(
            String::from_utf8_lossy(&out.stdout),
            format!("attached image/png ({}B) from {path}\n", PNG_BYTES.len())
        );
        let [Attachment::Image { mime, bytes }] = out.attachments.as_slice() else {
            panic!("expected one attachment: {:?}", out.attachments);
        };
        assert_eq!(*mime, ImageMime::Png);
        assert_eq!(bytes.as_slice(), PNG_BYTES);
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
}
