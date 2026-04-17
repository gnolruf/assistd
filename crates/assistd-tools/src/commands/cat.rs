use anyhow::Result;
use async_trait::async_trait;

use crate::command::{Command, CommandInput, CommandOutput, error_line, io_error_nav};

/// `cat [FILE]...` — concatenate files, or echo stdin if no files given.
/// Binary files are rejected so their raw bytes don't pollute the model's
/// context window; pair with `see` (images) or `cat -b` (metadata-only)
/// to inspect them safely.
pub struct CatCommand;

#[async_trait]
impl Command for CatCommand {
    fn name(&self) -> &str {
        "cat"
    }

    fn summary(&self) -> &'static str {
        "read text files or stdin; binary rejected (see `see`, `cat -b`)"
    }

    fn help(&self) -> String {
        "usage: cat [-b] [FILE]...\n\
         \n\
         Concatenate files, or echo stdin if no files given. Binary files \
         are rejected so their raw bytes don't pollute the model's context.\n\
         \n\
         Flags:\n  \
           -b  print metadata (mime, size) instead of content — safe for binary files\n\
         \n\
         For image files, use `see PATH` to attach them as a vision input.\n"
            .to_string()
    }

    async fn run(&self, input: CommandInput) -> Result<CommandOutput> {
        let (metadata_only, files) = partition_flags(&input.args);

        if files.is_empty() {
            // No files → operate on stdin.
            if metadata_only {
                return Ok(CommandOutput::ok(describe(&input.stdin, None)));
            }
            return Ok(CommandOutput::ok(input.stdin));
        }

        let mut out = Vec::new();
        for path in &files {
            let bytes = match tokio::fs::read(path).await {
                Ok(b) => b,
                Err(e) => {
                    return Ok(CommandOutput::failed(
                        1,
                        io_error_nav("cat", path, &e).into_bytes(),
                    ));
                }
            };

            if metadata_only {
                out.extend_from_slice(&describe(&bytes, Some(path)));
                continue;
            }

            if let Some(mime) = sniff_binary(&bytes) {
                let size = human_size(bytes.len());
                let msg = if mime.starts_with("image/") {
                    error_line(
                        "cat",
                        format_args!("binary image file ({size}): {path}"),
                        "Use",
                        format_args!("see {path}"),
                    )
                } else {
                    error_line(
                        "cat",
                        format_args!("binary {mime} file ({size}): {path}"),
                        "Use",
                        format_args!("cat -b {path}"),
                    )
                };
                return Ok(CommandOutput::failed(1, msg.into_bytes()));
            }
            out.extend_from_slice(&bytes);
        }
        Ok(CommandOutput::ok(out))
    }
}

/// Split `argv` into `(metadata_only, paths)`. `-b` is the only flag
/// recognized; anything else starting with `-` is treated as a path to
/// stay consistent with how the chain executor quotes arguments.
fn partition_flags(argv: &[String]) -> (bool, Vec<String>) {
    let mut metadata_only = false;
    let mut files = Vec::with_capacity(argv.len());
    for a in argv {
        if a == "-b" {
            metadata_only = true;
        } else {
            files.push(a.clone());
        }
    }
    (metadata_only, files)
}

/// `Some(mime)` if the bytes look binary, `None` if they're plausibly
/// text. First asks the `infer` crate (magic-byte sniff); on unknown,
/// falls back to a NUL-byte scan of the first 8 KB — the same heuristic
/// GNU grep uses.
pub(crate) fn sniff_binary(bytes: &[u8]) -> Option<String> {
    if let Some(t) = infer::get(bytes) {
        let mime = t.mime_type();
        if !mime.starts_with("text/") {
            return Some(mime.to_string());
        }
    }
    let sniff_len = bytes.len().min(8192);
    if bytes[..sniff_len].contains(&0u8) {
        return Some("application/octet-stream".to_string());
    }
    None
}

/// Render `<mime>\n<size> bytes\n` (with optional path prefix for
/// multi-file metadata listings). Used by `cat -b` and by the binary
/// error message via `human_size`.
fn describe(bytes: &[u8], path: Option<&str>) -> Vec<u8> {
    let mime = infer::get(bytes)
        .map(|t| t.mime_type().to_string())
        .unwrap_or_else(|| {
            if sniff_binary(bytes).is_some() {
                "application/octet-stream".into()
            } else {
                "text/plain".into()
            }
        });
    let prefix = path.map(|p| format!("{p}: ")).unwrap_or_default();
    format!("{prefix}{mime}\n{prefix}{} bytes\n", bytes.len()).into_bytes()
}

pub(crate) fn human_size(n: usize) -> String {
    const KB: usize = 1024;
    const MB: usize = KB * 1024;
    const GB: usize = MB * 1024;
    if n >= GB {
        format!("{:.1}GB", n as f64 / GB as f64)
    } else if n >= MB {
        format!("{:.1}MB", n as f64 / MB as f64)
    } else if n >= KB {
        format!("{}KB", n / KB)
    } else {
        format!("{n}B")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    // Minimal valid 1x1 PNG — `infer` recognizes this as `image/png`.
    const PNG_BYTES: &[u8] = &[
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44,
        0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x06, 0x00, 0x00, 0x00, 0x1F,
        0x15, 0xC4, 0x89, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9C, 0x63, 0x00,
        0x01, 0x00, 0x00, 0x05, 0x00, 0x01, 0x0D, 0x0A, 0x2D, 0xB4, 0x00, 0x00, 0x00, 0x00, 0x49,
        0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
    ];

    #[tokio::test]
    async fn cat_reads_text_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("hello.txt");
        tokio::fs::write(&path, b"hello world\n").await.unwrap();
        let out = CatCommand
            .run(CommandInput {
                args: vec![path.to_string_lossy().into_owned()],
                stdin: Vec::new(),
            })
            .await
            .unwrap();
        assert_eq!(out.stdout, b"hello world\n");
        assert_eq!(out.exit_code, 0);
    }

    #[tokio::test]
    async fn cat_concatenates_multiple_text_files() {
        let dir = tempdir().unwrap();
        let a = dir.path().join("a.txt");
        let b = dir.path().join("b.txt");
        tokio::fs::write(&a, b"A").await.unwrap();
        tokio::fs::write(&b, b"B").await.unwrap();
        let out = CatCommand
            .run(CommandInput {
                args: vec![
                    a.to_string_lossy().into_owned(),
                    b.to_string_lossy().into_owned(),
                ],
                stdin: Vec::new(),
            })
            .await
            .unwrap();
        assert_eq!(out.stdout, b"AB");
    }

    #[tokio::test]
    async fn cat_missing_file_exits_1() {
        let out = CatCommand
            .run(CommandInput {
                args: vec!["/nonexistent/path/xyz".into()],
                stdin: Vec::new(),
            })
            .await
            .unwrap();
        assert_eq!(out.exit_code, 1);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            stderr.contains("[error] cat: file not found: /nonexistent/path/xyz"),
            "{stderr}"
        );
        assert!(stderr.contains("Use: ls to check the path"), "{stderr}");
    }

    #[tokio::test]
    async fn cat_no_args_echoes_stdin() {
        let out = CatCommand
            .run(CommandInput {
                args: Vec::new(),
                stdin: b"from stdin".to_vec(),
            })
            .await
            .unwrap();
        assert_eq!(out.stdout, b"from stdin");
    }

    #[tokio::test]
    async fn cat_rejects_binary_image_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("photo.png");
        tokio::fs::write(&path, PNG_BYTES).await.unwrap();
        let out = CatCommand
            .run(CommandInput {
                args: vec![path.to_string_lossy().into_owned()],
                stdin: Vec::new(),
            })
            .await
            .unwrap();
        assert_eq!(out.exit_code, 1);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            stderr.contains("[error] cat: binary image file"),
            "{stderr}"
        );
        assert!(stderr.contains("Use: see "), "{stderr}");
        assert!(out.stdout.is_empty(), "must not leak bytes on rejection");
    }

    #[tokio::test]
    async fn cat_rejects_binary_with_nul_bytes() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("garbled.bin");
        let mut bytes = vec![b'x'; 200];
        bytes[100] = 0;
        tokio::fs::write(&path, &bytes).await.unwrap();
        let out = CatCommand
            .run(CommandInput {
                args: vec![path.to_string_lossy().into_owned()],
                stdin: Vec::new(),
            })
            .await
            .unwrap();
        assert_eq!(out.exit_code, 1);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            stderr.contains("[error] cat: binary application/octet-stream"),
            "{stderr}"
        );
        assert!(stderr.contains("Use: cat -b "), "{stderr}");
    }

    #[tokio::test]
    async fn cat_b_prints_metadata_for_binary() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("photo.png");
        tokio::fs::write(&path, PNG_BYTES).await.unwrap();
        let out = CatCommand
            .run(CommandInput {
                args: vec!["-b".into(), path.to_string_lossy().into_owned()],
                stdin: Vec::new(),
            })
            .await
            .unwrap();
        assert_eq!(out.exit_code, 0);
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(stdout.contains("image/png"), "{stdout}");
        assert!(
            stdout.contains(&format!("{} bytes", PNG_BYTES.len())),
            "{stdout}"
        );
    }

    #[tokio::test]
    async fn cat_b_works_on_text_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("notes.txt");
        tokio::fs::write(&path, b"some text\n").await.unwrap();
        let out = CatCommand
            .run(CommandInput {
                args: vec!["-b".into(), path.to_string_lossy().into_owned()],
                stdin: Vec::new(),
            })
            .await
            .unwrap();
        assert_eq!(out.exit_code, 0);
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(stdout.contains("text/plain"), "{stdout}");
        assert!(stdout.contains("10 bytes"), "{stdout}");
    }

    #[test]
    fn human_size_formats_expected_ranges() {
        assert_eq!(human_size(0), "0B");
        assert_eq!(human_size(500), "500B");
        assert_eq!(human_size(2048), "2KB");
        assert_eq!(human_size(1024 * 1024 * 3), "3.0MB");
    }
}
