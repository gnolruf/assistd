//! Built-in commands dispatched by the chain executor.

use std::io::{self, ErrorKind};
use std::path::Path;

use assistd_utils::text::human_size;
use tokio::io::AsyncReadExt;

use crate::chain::OUTPUT_MAX;
use crate::command::{CommandOutput, Hint, error_line, io_error_nav};

pub mod bash;
pub mod cat;
pub mod echo;
pub mod grep;
pub mod head_tail;
pub mod ls;
pub mod screenshot;
pub mod see;
pub mod sort;
pub mod uniq;
pub mod wc;
pub mod web;
pub mod wm;
pub mod write;

#[cfg(test)]
mod test_support;

pub use crate::policy::BashPolicyCfg;
pub use bash::BashCommand;
pub use cat::CatCommand;
pub use echo::EchoCommand;
pub use grep::GrepCommand;
pub use head_tail::{HeadCommand, TailCommand};
pub use ls::LsCommand;
pub use screenshot::{Backend as ScreenshotBackendKind, ScreenshotCommand, ScreenshotPolicyCfg};
pub use see::SeeCommand;
pub use sort::SortCommand;
pub use uniq::UniqCommand;
pub use wc::WcCommand;
pub use web::WebCommand;
pub use wm::WmCommand;
pub use write::{WriteCommand, WritePolicyCfg};

#[cfg(test)]
pub(crate) use test_support::{
    RecordingGate, hold_fifo_open, make_fifo, test_patterns, test_registry,
};

/// Largest file a command reads into memory; a bigger one would overflow
/// a pipeline stage anyway.
pub(crate) const FILE_READ_MAX: u64 = OUTPUT_MAX as u64;

/// Read `path` whole, refusing anything that is not a regular file or is
/// (or grows while being read to be) larger than [`FILE_READ_MAX`].
pub(crate) async fn read_regular_file(path: impl AsRef<Path>) -> io::Result<Vec<u8>> {
    let too_large = || {
        io::Error::new(
            ErrorKind::FileTooLarge,
            format!(
                "file exceeds the {} read limit",
                human_size(OUTPUT_MAX as u64)
            ),
        )
    };
    let (file, size) = open_regular(path.as_ref()).await?;
    if size > FILE_READ_MAX {
        return Err(too_large());
    }
    let mut bytes = Vec::new();
    file.take(FILE_READ_MAX + 1).read_to_end(&mut bytes).await?;
    if bytes.len() as u64 > FILE_READ_MAX {
        return Err(too_large());
    }
    Ok(bytes)
}

/// Read at most `limit` bytes from the start of `path` with the same
/// file-type check as [`read_regular_file`]; returns them and the file size.
pub(crate) async fn read_regular_head(
    path: impl AsRef<Path>,
    limit: u64,
) -> io::Result<(Vec<u8>, u64)> {
    let (file, size) = open_regular(path.as_ref()).await?;
    let mut head = Vec::new();
    file.take(limit).read_to_end(&mut head).await?;
    Ok((head, size))
}

/// Open `path` with its size on the blocking pool, refusing devices, FIFOs
/// and sockets without opening or blocking on them. Directories fall through
/// to `EISDIR`.
async fn open_regular(path: &Path) -> io::Result<(tokio::fs::File, u64)> {
    let path = path.to_owned();
    let (file, size) =
        tokio::task::spawn_blocking(move || assistd_utils::fs::open_regular_or_dir(&path))
            .await
            .map_err(io::Error::other)??;
    Ok((tokio::fs::File::from_std(file), size))
}

/// Input for a stdin-or-files command: named files (concatenated, binary
/// refused, at most [`OUTPUT_MAX`] in all) win over stdin. `Ok(None)` means
/// neither was supplied.
pub(crate) async fn collect_input(
    cmd: &str,
    files: &[String],
    stdin: Option<Vec<u8>>,
) -> Result<Option<Vec<u8>>, CommandOutput> {
    if files.is_empty() {
        return Ok(stdin);
    }
    let mut out = Vec::new();
    for path in files {
        let bytes = read_regular_file(path)
            .await
            .map_err(|e| CommandOutput::failed(1, io_error_nav(cmd, path, &e).into_bytes()))?;
        if let Some(mime) = cat::sniff_binary(&bytes) {
            let size = human_size(bytes.len() as u64);
            return Err(CommandOutput::failed(
                1,
                error_line(
                    cmd,
                    format_args!("binary {mime} file ({size}): {path}"),
                    Hint::Use,
                    format_args!("cat -b {path}"),
                )
                .into_bytes(),
            ));
        }
        out.extend_from_slice(&bytes);
        if out.len() > OUTPUT_MAX {
            return Err(CommandOutput::failed(
                1,
                error_line(
                    cmd,
                    format_args!(
                        "files together exceed the {} read limit",
                        human_size(OUTPUT_MAX as u64)
                    ),
                    Hint::Try,
                    format_args!("fewer files per {cmd}"),
                )
                .into_bytes(),
            ));
        }
    }
    Ok(Some(out))
}
