//! Document indexing: the extraction worker's launch in a reader sandbox over the
//! index roots, the passes that walk those roots into the index, the watches that
//! have changed files indexed between them, and the gate on embedding what they store.

use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use assistd_config::{Config, DocumentsConfig};
use assistd_documents::WorkerCommand;
use assistd_tools::{ReaderProgram, ReaderSandbox, ReaderSandboxError, SandboxInfo, ToolsDisabled};

pub(crate) mod embed_gate;
mod indexer;
mod pass;
mod walk;
mod watch;

pub(crate) use walk::root_unavailable;

pub use indexer::{DocumentIndex, IndexPasses, IndexProgress, PassProgress, remove_dropped_roots};
pub use pass::{IndexDeps, MAX_DOCUMENT_CHUNKS, PassCounts, PassEnd, RootPass, SkipCause};

/// Where the daemon's program looks for its own shared libraries, under the
/// directory that holds the one it is in.
const PRIVATE_LIBRARY_DIR: &str = "lib/assistd";

/// What the system prompt says of document search while `documents.enabled`.
const SEARCH_NOTE: &str = "The user's indexed documents can be searched by meaning with the \
     `run` command `docs search <words>`, and one that a hit names read with `docs read <path>`.";

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
    #[error("documents.exclude has a pattern that is not a valid glob: {0}")]
    Exclude(#[source] ignore::Error),
}

/// `chat.system_prompt` as it is sent: followed, unless it is blank or `documents.enabled`
/// is off, by a line saying the indexed documents can be searched and read with `docs`.
pub fn system_prompt(config: &Config) -> String {
    let written = &config.chat.system_prompt;
    if config.documents.enabled && !written.trim().is_empty() {
        format!("{written}\n\n{SEARCH_NOTE}")
    } else {
        written.clone()
    }
}

/// What indexing `documents.roots` runs: the command that starts `executable` with
/// `subcommand` as an extraction worker in a reader sandbox showing the roots and the
/// program's own directories, and the passes that feed it. Blocks on the filesystem.
pub fn document_indexing(
    documents: &DocumentsConfig,
    sandbox: Result<&Arc<SandboxInfo>, &ToolsDisabled>,
    executable: &Path,
    subcommand: &'static str,
) -> Result<(WorkerCommand, IndexPasses), DocumentsUnavailable> {
    if !documents.enabled {
        return Err(DocumentsUnavailable::Disabled);
    }
    if documents.roots.is_empty() {
        return Err(DocumentsUnavailable::NoRoots);
    }
    let sandbox = sandbox.map_err(|disabled| DocumentsUnavailable::NoSandbox(disabled.clone()))?;
    let passes = IndexPasses::new(documents, Arc::clone(sandbox))?;
    let worker = extract_worker_command(documents, sandbox, executable, subcommand)?;
    Ok((worker, passes))
}

/// The command that starts `executable` with `subcommand` as an extraction worker
/// in a reader sandbox over `documents.roots`. Blocks on the filesystem.
fn extract_worker_command(
    documents: &DocumentsConfig,
    sandbox: &SandboxInfo,
    executable: &Path,
    subcommand: &'static str,
) -> Result<WorkerCommand, DocumentsUnavailable> {
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
