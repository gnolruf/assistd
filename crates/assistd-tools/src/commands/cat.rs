//! `cat`: print text files or stdin, refusing binary files.

use assistd_utils::text::human_size;
use async_trait::async_trait;

use crate::chain::OUTPUT_MAX;
use crate::chain::Word;
use crate::command::{Command, CommandError, CommandInput, CommandOutput, Hint, Usage};
use crate::commands::{read_regular_file, read_regular_head};
use crate::policy::FileScope;

/// How much of a file's head [`sniff_binary`] looks at for a NUL byte.
pub(crate) const SNIFF_LEN: usize = 8192;

/// `cat [-bn] [FILE]...`: concatenate files, or echo stdin if none are
/// given. Binary files are rejected so their bytes stay out of the context.
#[derive(Debug)]
pub struct CatCommand;

#[derive(Default)]
struct Flags {
    metadata_only: bool,
    number_lines: bool,
}

#[async_trait]
impl Command for CatCommand {
    fn name(&self) -> &'static str {
        "cat"
    }

    fn summary(&self) -> &'static str {
        "read text files or stdin (-n numbers lines); binary rejected"
    }

    fn usage(&self) -> Usage {
        Usage::new(
            "[-bn] [FILE]...",
            "Concatenate files, or echo stdin if no files given. Binary files \
         are rejected so their raw bytes don't pollute the model's context.\n\
         \n\
         Flags:\n  \
           -b  print metadata (mime, size) instead of content (safe for binary files)\n  \
           -n  prefix each output line with its 1-based line number\n\
         \n\
         For image files, use `see PATH` to attach them as a vision input.\n",
        )
    }

    async fn run(&self, input: CommandInput) -> Result<CommandOutput, CommandError> {
        let (flags, files) = parse_flags(&input.args)
            .map_err(|what| CommandError::usage(what, "cat -b FILE or cat -n FILE"))?;

        if files.is_empty() {
            let Some(stdin) = input.stdin else {
                return Ok(CommandOutput::usage(self.help()));
            };
            if flags.metadata_only {
                return Ok(CommandOutput::ok(describe(
                    &stdin,
                    stdin.len() as u64,
                    None,
                )));
            }
            return Ok(CommandOutput::ok(number_if(stdin, &flags)));
        }

        let mut out = Vec::new();
        for path in files {
            let chunk = if flags.metadata_only {
                read_metadata(&input.scope, path).await?
            } else {
                read_text(&input.scope, path).await?
            };
            out.extend_from_slice(&chunk);
            if out.len() > OUTPUT_MAX {
                break;
            }
        }
        Ok(CommandOutput::ok(number_if(out, &flags)))
    }
}

async fn read_metadata(scope: &FileScope, path: &Word) -> Result<Vec<u8>, CommandError> {
    let (head, size) = read_regular_head(scope, path.as_str(), SNIFF_LEN as u64)
        .await
        .map_err(|e| CommandError::io(path, &e))?;
    Ok(describe(&head, size, Some(path.as_str())))
}

async fn read_text(scope: &FileScope, path: &Word) -> Result<Vec<u8>, CommandError> {
    let bytes = read_regular_file(scope, path.as_str())
        .await
        .map_err(|e| CommandError::io(path, &e))?;
    let Some(mime) = sniff_binary(&bytes) else {
        return Ok(bytes);
    };
    let size = human_size(bytes.len() as u64);
    Err(if mime.starts_with("image/") {
        CommandError::failed(
            format_args!("binary image file ({size}): {path}"),
            Hint::Use,
            format_args!("see {path}"),
        )
    } else {
        CommandError::failed(
            format_args!("binary {mime} file ({size}): {path}"),
            Hint::Use,
            format_args!("cat -b {path}"),
        )
    })
}

fn number_if(bytes: Vec<u8>, flags: &Flags) -> Vec<u8> {
    if !flags.number_lines {
        return bytes;
    }
    let mut out = Vec::with_capacity(bytes.len() + bytes.len() / 16);
    for (index, line) in bytes.split_inclusive(|b| *b == b'\n').enumerate() {
        out.extend_from_slice(format!("{}\t", index + 1).as_bytes());
        out.extend_from_slice(line);
    }
    out
}

fn parse_flags(argv: &[Word]) -> Result<(Flags, Vec<&Word>), String> {
    let mut flags = Flags::default();
    let mut files = Vec::with_capacity(argv.len());
    for arg in argv {
        let Some(letters) = arg.text.strip_prefix('-').filter(|l| !l.is_empty()) else {
            files.push(arg);
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
    if bytes[..bytes.len().min(SNIFF_LEN)].contains(&0u8) {
        return Some("application/octet-stream".to_string());
    }
    None
}

fn describe(head: &[u8], size: u64, path: Option<&str>) -> Vec<u8> {
    let mime = infer::get(head).map_or_else(
        || {
            if sniff_binary(head).is_some() {
                "application/octet-stream".into()
            } else {
                "text/plain".into()
            }
        },
        |t| t.mime_type().to_string(),
    );
    let prefix = path.map(|p| format!("{p}: ")).unwrap_or_default();
    format!("{prefix}{mime}\n{prefix}{size} bytes\n").into_bytes()
}

#[cfg(test)]
mod tests;
