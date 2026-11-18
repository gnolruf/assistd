//! Launching the document extraction worker: the daemon's own program in a reader
//! sandbox over the index roots.

use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use assistd_config::DocumentsConfig;
use assistd_documents::WorkerCommand;
use assistd_tools::{ReaderProgram, ReaderSandbox, ReaderSandboxError, SandboxInfo, ToolsDisabled};

/// Where the daemon's program looks for its own shared libraries, under the
/// directory that holds the one it is in.
const PRIVATE_LIBRARY_DIR: &str = "lib/assistd";

/// Why no extraction worker runs, so no document is indexed.
#[derive(Debug, thiserror::Error)]
pub enum DocumentsUnavailable {
    #[error("disabled in config (documents.enabled = false)")]
    Disabled,
    #[error("no folder to index (documents.roots is empty)")]
    NoRoots,
    #[error("files are read only in the tool sandbox, and {0}")]
    NoSandbox(ToolsDisabled),
    #[error(transparent)]
    IdlePriority(#[from] ReaderSandboxError),
    #[error("cannot resolve the worker's program {}: {source}", executable.display())]
    Program {
        executable: PathBuf,
        #[source]
        source: io::Error,
    },
}

/// The command that starts `executable` with `subcommand` as an extraction worker:
/// in a reader sandbox showing `documents.roots` and the program's own directories.
/// Errors with why document indexing is unavailable. Blocks on the filesystem.
pub fn extract_worker_command(
    documents: &DocumentsConfig,
    sandbox: Result<&SandboxInfo, &ToolsDisabled>,
    executable: &Path,
    subcommand: &'static str,
) -> Result<WorkerCommand, DocumentsUnavailable> {
    if !documents.enabled {
        return Err(DocumentsUnavailable::Disabled);
    }
    if documents.roots.is_empty() {
        return Err(DocumentsUnavailable::NoRoots);
    }
    let sandbox = sandbox.map_err(|disabled| DocumentsUnavailable::NoSandbox(disabled.clone()))?;
    let path =
        std::fs::canonicalize(executable).map_err(|source| DocumentsUnavailable::Program {
            executable: executable.to_path_buf(),
            source,
        })?;
    let read_only = documents
        .roots
        .iter()
        .map(|root| root.as_path().to_path_buf())
        .chain(program_dirs(&path))
        .collect();
    let reader = ReaderSandbox::over(sandbox, read_only)?;
    let program = ReaderProgram {
        image: executable.to_path_buf(),
        path,
    };
    Ok(Arc::new(move || reader.command(&program, [subcommand])))
}

/// The directories the program at `path` loads its shared libraries from: its
/// own, and [`PRIVATE_LIBRARY_DIR`] beside it.
fn program_dirs(path: &Path) -> Vec<PathBuf> {
    let bin_dir = path.parent();
    let library_dir = bin_dir
        .and_then(Path::parent)
        .map(|prefix| prefix.join(PRIVATE_LIBRARY_DIR));
    bin_dir
        .map(Path::to_path_buf)
        .into_iter()
        .chain(library_dir)
        .collect()
}
