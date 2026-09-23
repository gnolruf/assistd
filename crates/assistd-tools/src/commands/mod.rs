//! Built-in commands dispatched by the chain executor.

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

use std::path::Path;

use tokio::io::AsyncReadExt;

use crate::chain::PIPE_BUF_MAX;
use crate::command::{CommandOutput, Hint, error_line, io_error_nav};

/// Largest file a command reads into memory. A bigger one would
/// overflow a pipeline stage anyway.
pub(crate) const FILE_READ_MAX: u64 = PIPE_BUF_MAX as u64;

/// Read `path` whole, refusing anything that is not a regular file or
/// is larger than [`FILE_READ_MAX`]. A file that grows past the cap
/// while being read is refused too.
pub(crate) async fn read_regular_file(path: impl AsRef<Path>) -> std::io::Result<Vec<u8>> {
    let too_large = || {
        std::io::Error::new(
            std::io::ErrorKind::FileTooLarge,
            format!(
                "file exceeds the {} read limit",
                cat::human_size(PIPE_BUF_MAX)
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

/// Read at most `limit` bytes from the start of `path`, with the same
/// file-type check as [`read_regular_file`] but no size cap. Returns
/// those bytes and the file's size.
pub(crate) async fn read_regular_head(
    path: impl AsRef<Path>,
    limit: u64,
) -> std::io::Result<(Vec<u8>, u64)> {
    let (file, size) = open_regular(path.as_ref()).await?;
    let mut head = Vec::new();
    file.take(limit).read_to_end(&mut head).await?;
    Ok((head, size))
}

/// Open `path` only if it is a regular file, returning it with its
/// size. A device, FIFO or socket (including `/dev/stdin`, the daemon's
/// own terminal) can block the open or read forever or never end, so
/// anything that is neither a file nor a directory is refused before it
/// is opened. Directories fall through so the read fails with the usual
/// `EISDIR` message.
async fn open_regular(path: &Path) -> std::io::Result<(tokio::fs::File, u64)> {
    let meta = tokio::fs::metadata(path).await?;
    if !meta.is_file() && !meta.is_dir() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "not a regular file (device, pipe, or socket)",
        ));
    }
    Ok((tokio::fs::File::open(path).await?, meta.len()))
}

/// Gather what a stdin-or-files command should operate on. Files named
/// on the command line win over stdin (as in coreutils), and several of
/// them concatenate exactly as `cat FILE... | <cmd>` would. Binary files
/// are refused here for the reason `cat` refuses them: their bytes would
/// land in the model's context window.
///
/// `Ok(None)` means neither a file nor stdin was supplied, which every
/// caller answers with its own usage text.
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
        let bytes = match read_regular_file(path).await {
            Ok(b) => b,
            Err(e) => {
                return Err(CommandOutput::failed(
                    1,
                    io_error_nav(cmd, path, &e).into_bytes(),
                ));
            }
        };
        if let Some(mime) = cat::sniff_binary(&bytes) {
            let size = cat::human_size(bytes.len());
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
    }
    Ok(Some(out))
}

#[cfg(test)]
pub(crate) fn test_registry() -> crate::command::CommandRegistry {
    use assistd_wm::NoWindowManager;
    use std::sync::Arc;

    let mut r = crate::command::CommandRegistry::new();
    r.register(CatCommand);
    r.register(LsCommand);
    r.register(GrepCommand);
    r.register(WcCommand);
    r.register(HeadCommand);
    r.register(TailCommand);
    r.register(SortCommand);
    r.register(UniqCommand);
    r.register(EchoCommand);
    r.register(WriteCommand::permissive_for_tests());
    r.register(SeeCommand::default());
    r.register(ScreenshotCommand::default());
    r.register(WebCommand::new());
    r.register(BashCommand::default());
    r.register(WmCommand::for_test(Arc::new(NoWindowManager)));
    r
}

/// Confirmation gate that answers every prompt the same way and records
/// each prompt's `(tool, script, matched_pattern)`.
#[cfg(test)]
pub(crate) struct RecordingGate {
    approve: bool,
    prompts: parking_lot::Mutex<Vec<(String, String, String)>>,
}

#[cfg(test)]
impl RecordingGate {
    pub(crate) fn new(approve: bool) -> std::sync::Arc<Self> {
        std::sync::Arc::new(Self {
            approve,
            prompts: parking_lot::Mutex::default(),
        })
    }

    pub(crate) fn prompts(&self) -> Vec<(String, String, String)> {
        self.prompts.lock().clone()
    }
}

#[cfg(test)]
#[async_trait::async_trait]
impl crate::policy::ConfirmationGate for RecordingGate {
    async fn confirm(&self, req: crate::policy::ConfirmationRequest) -> bool {
        self.prompts
            .lock()
            .push((req.tool, req.script, req.matched_pattern));
        self.approve
    }
}
