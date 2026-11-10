//! Built-in commands dispatched by the chain executor.

use std::io::{self, ErrorKind};
use std::path::Path;
use std::sync::Arc;

use assistd_utils::text::human_size;
use tokio::io::AsyncReadExt;

use crate::chain::OUTPUT_MAX;
use crate::command::CommandRegistry;
use crate::policy::{ConfirmationGate, FileScope, SandboxInfo};

pub mod bash;
pub mod cat;
pub mod echo;
pub mod ls;
pub mod program;
pub mod screenshot;
pub mod see;
pub mod web;
pub mod wm;
pub mod write;

#[cfg(test)]
pub(crate) mod test_support;

pub use crate::policy::BashPolicyCfg;
pub use bash::BashCommand;
pub use cat::CatCommand;
pub use echo::EchoCommand;
pub use ls::LsCommand;
pub use program::ProgramCommand;
pub use screenshot::{ScreenshotCommand, ScreenshotPolicyCfg};
pub use see::SeeCommand;
pub use web::WebCommand;
pub use wm::WmCommand;
pub use write::{WriteCommand, WritePolicyCfg};

#[cfg(test)]
pub(crate) use test_support::{
    RecordingGate, hold_fifo_open, make_fifo, permissive_write, register_test_readers, test_bash,
    test_registry, test_screenshot, test_see, test_wm,
};

/// Largest file a command reads into memory; a bigger one would overflow
/// a pipeline stage anyway.
pub(crate) const FILE_READ_MAX: u64 = OUTPUT_MAX as u64;

/// Read `path` in `scope` whole, refusing anything that is not a regular file
/// or is (or grows while being read to be) larger than [`FILE_READ_MAX`].
pub(crate) async fn read_regular_file(
    scope: &FileScope,
    path: impl AsRef<Path>,
) -> io::Result<Vec<u8>> {
    let too_large = || {
        io::Error::new(
            ErrorKind::FileTooLarge,
            format!(
                "file exceeds the {} read limit",
                human_size(OUTPUT_MAX as u64)
            ),
        )
    };
    let (file, size) = open_regular(scope, path.as_ref()).await?;
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

/// Read at most `limit` bytes from the start of `path` in `scope` with the same
/// file-type check as [`read_regular_file`]; returns them and the file size.
pub(crate) async fn read_regular_head(
    scope: &FileScope,
    path: impl AsRef<Path>,
    limit: u64,
) -> io::Result<(Vec<u8>, u64)> {
    let (file, size) = open_regular(scope, path.as_ref()).await?;
    let mut head = Vec::new();
    file.take(limit).read_to_end(&mut head).await?;
    Ok((head, size))
}

/// Open `path` with its size on the blocking pool, refusing devices, FIFOs
/// and sockets without opening or blocking on them. Directories fall through
/// to `EISDIR`.
async fn open_regular(scope: &FileScope, path: &Path) -> io::Result<(tokio::fs::File, u64)> {
    let path = path.to_owned();
    let scope = scope.clone();
    let (file, size) = tokio::task::spawn_blocking(move || scope.open_regular_or_dir(&path))
        .await
        .map_err(io::Error::other)??;
    Ok((tokio::fs::File::from_std(file), size))
}

/// Register the file and text commands: `cat`, `ls` and `echo` in-process, and every
/// [`ProgramCommand`] under `cfg`, inside `sandbox`, asking `gate`.
pub fn register_readers(
    registry: &mut CommandRegistry,
    cfg: &Arc<BashPolicyCfg>,
    sandbox: &Arc<SandboxInfo>,
    gate: &Arc<dyn ConfirmationGate>,
) {
    registry.register(CatCommand);
    registry.register(LsCommand);
    registry.register(EchoCommand);
    for program in ProgramCommand::all(cfg, sandbox, gate) {
        registry.register(program);
    }
}
