use anyhow::Result;
use async_trait::async_trait;

use crate::command::{Command, CommandInput, CommandOutput, Hint, error_line, io_error_nav};

/// `cat [-bn] [FILE]...`: concatenate files, or echo stdin if no files
/// given. Binary files are rejected so their raw bytes don't pollute the
/// model's context window; pair with `see` (images) or `cat -b`
/// (metadata-only) to inspect them safely.
///
/// Flags:
/// - `-b` print metadata (mime, size) instead of content
/// - `-n` prefix each output line with its 1-based number
pub struct CatCommand;

#[derive(Default)]
struct Flags {
    metadata_only: bool,
    number_lines: bool,
}

#[async_trait]
impl Command for CatCommand {
    fn name(&self) -> &str {
        "cat"
    }

    fn summary(&self) -> &'static str {
        "read text files or stdin (-n numbers lines); binary rejected"
    }

    fn help(&self) -> String {
        "usage: cat [-bn] [FILE]...\n\
         \n\
         Concatenate files, or echo stdin if no files given. Binary files \
         are rejected so their raw bytes don't pollute the model's context.\n\
         \n\
         Flags:\n  \
           -b  print metadata (mime, size) instead of content (safe for binary files)\n  \
           -n  prefix each output line with its 1-based line number\n\
         \n\
         For image files, use `see PATH` to attach them as a vision input.\n"
            .to_string()
    }

    async fn run(&self, input: CommandInput) -> Result<CommandOutput> {
        let (flags, files) = match parse_flags(&input.args) {
            Ok(v) => v,
            Err(msg) => {
                return Ok(CommandOutput::usage_error(
                    "cat",
                    msg,
                    "cat -b FILE or cat -n FILE",
                ));
            }
        };

        if files.is_empty() {
            let Some(stdin) = input.stdin else {
                return Ok(CommandOutput::usage(self.help()));
            };
            if flags.metadata_only {
                return Ok(CommandOutput::ok(describe(&stdin, None)));
            }
            return Ok(CommandOutput::ok(number_if(stdin, &flags)));
        }

        let mut out = Vec::new();
        for path in &files {
            let bytes = match super::read_regular_file(path).await {
                Ok(b) => b,
                Err(e) => {
                    return Ok(CommandOutput::failed(
                        1,
                        io_error_nav("cat", path, &e).into_bytes(),
                    ));
                }
            };

            if flags.metadata_only {
                out.extend_from_slice(&describe(&bytes, Some(path)));
                continue;
            }

            if let Some(mime) = sniff_binary(&bytes) {
                let size = human_size(bytes.len());
                let msg = if mime.starts_with("image/") {
                    error_line(
                        "cat",
                        format_args!("binary image file ({size}): {path}"),
                        Hint::Use,
                        format_args!("see {path}"),
                    )
                } else {
                    error_line(
                        "cat",
                        format_args!("binary {mime} file ({size}): {path}"),
                        Hint::Use,
                        format_args!("cat -b {path}"),
                    )
                };
                return Ok(CommandOutput::failed(1, msg.into_bytes()));
            }
            out.extend_from_slice(&bytes);
        }
        Ok(CommandOutput::ok(number_if(out, &flags)))
    }
}

fn number_if(bytes: Vec<u8>, flags: &Flags) -> Vec<u8> {
    if !flags.number_lines {
        return bytes;
    }
    let mut out = Vec::with_capacity(bytes.len() + bytes.len() / 16);
    for (i, line) in bytes.split_inclusive(|b| *b == b'\n').enumerate() {
        out.extend_from_slice(format!("{}\t", i + 1).as_bytes());
        out.extend_from_slice(line);
    }
    out
}

fn parse_flags(argv: &[String]) -> Result<(Flags, Vec<String>), String> {
    let mut flags = Flags::default();
    let mut files = Vec::with_capacity(argv.len());
    for arg in argv {
        let Some(letters) = arg.strip_prefix('-').filter(|l| !l.is_empty()) else {
            files.push(arg.clone());
            continue;
        };
        for ch in letters.chars() {
            match ch {
                'b' => flags.metadata_only = true,
                'n' => flags.number_lines = true,
                other => return Err(format!("unknown flag '-{other}'")),
            }
        }
    }
    Ok((flags, files))
}

/// `Some(mime)` if the bytes look binary: a recognised non-text magic
/// number, or a NUL byte in the first 8 KB (GNU grep's heuristic).
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
mod tests;
