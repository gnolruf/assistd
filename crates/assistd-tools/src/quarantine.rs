//! A sub-agent's disposable world: one fresh directory per relay, holding
//! copies of its inputs, and the only part of the host its commands can see.

use std::collections::HashSet;
use std::ffi::OsString;
use std::fs::{File, Permissions};
use std::io::{self, ErrorKind};
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use tracing::warn;

use crate::command::CommandRegistry;
use crate::commands::{
    BashCommand, BashPolicyCfg, CatCommand, EchoCommand, GrepCommand, HeadCommand, LsCommand,
    SortCommand, TailCommand, UniqCommand, WcCommand, WriteCommand, WritePolicyCfg,
};
use crate::policy::{ConfinedRoot, ConfirmationGate, SandboxInfo, on_blocking_pool};

const INPUT_DIR: &str = "input";
const HOME_DIR: &str = "home";
const OUTPUT_DIR: &str = "output";

/// Why a quarantine could not be set up.
#[derive(Debug, thiserror::Error)]
pub enum QuarantineError {
    #[error("a quarantine needs the bwrap sandbox, and this configuration does not wrap")]
    Unsandboxed,
    #[error("cannot create a quarantine directory under {}: {source}", parent.display())]
    Create {
        parent: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("input {} has no file name to copy it under", .0.display())]
    UnnamedInput(PathBuf),
    #[error("more than one input is named {0:?}")]
    DuplicateInput(OsString),
    #[error("cannot copy input {}: {source}", path.display())]
    CopyInput {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
}

/// A relay's directory and the sandbox over it: copied inputs read-only, a
/// writable home, and a directory for spilled output, all at their host
/// paths; no network. Dropping it, on cancel or panic too, removes it all.
#[derive(Debug)]
pub struct Quarantine {
    dir: RelayDir,
    root: Arc<ConfinedRoot>,
    sandbox: Arc<SandboxInfo>,
}

impl Quarantine {
    /// A new quarantine under `parent` wrapped as `sandbox` wraps, with each of `inputs` copied
    /// into [`Quarantine::input_dir`] under its file name. Fails when `sandbox` does not wrap.
    pub async fn create(
        parent: &Path,
        sandbox: &SandboxInfo,
        inputs: Vec<PathBuf>,
    ) -> Result<Self, QuarantineError> {
        let parent = parent.to_path_buf();
        let (dir, root) = on_blocking_pool(move || {
            let (dir, root) = RelayDir::create(&parent)?;
            copy_inputs(&dir.path.join(INPUT_DIR), &inputs)?;
            Ok::<_, QuarantineError>((dir, root))
        })
        .await?;
        let sandbox = sandbox
            .quarantined(
                dir.path.join(HOME_DIR),
                vec![dir.path.join(INPUT_DIR), dir.path.join(OUTPUT_DIR)],
            )
            .ok_or(QuarantineError::Unsandboxed)?;
        Ok(Self {
            dir,
            root: Arc::new(root),
            sandbox: Arc::new(sandbox),
        })
    }

    /// Where the inputs are, read-only.
    pub fn input_dir(&self) -> PathBuf {
        self.dir.path.join(INPUT_DIR)
    }

    /// The writable `$HOME` and working directory of every command.
    pub fn home_dir(&self) -> PathBuf {
        self.dir.path.join(HOME_DIR)
    }

    /// Where truncated output is spilled to, readable by every command.
    pub fn output_dir(&self) -> PathBuf {
        self.dir.path.join(OUTPUT_DIR)
    }

    /// The file commands, confined to this directory, plus `echo`, `write` limited to the home,
    /// and `bash` in the quarantine sandbox under `bash`'s rules, asking `gate` as usual.
    pub fn commands(
        &self,
        bash: &BashPolicyCfg,
        gate: Arc<dyn ConfirmationGate>,
    ) -> CommandRegistry {
        let home = self.home_dir();
        let bash_cfg = BashPolicyCfg {
            scratch: Some(home.clone()),
            ..bash.clone()
        };
        let mut commands = CommandRegistry::confined(Arc::clone(&self.root));
        commands.register(CatCommand);
        commands.register(LsCommand);
        commands.register(GrepCommand);
        commands.register(WcCommand);
        commands.register(HeadCommand);
        commands.register(TailCommand);
        commands.register(SortCommand);
        commands.register(UniqCommand);
        commands.register(EchoCommand);
        commands.register(WriteCommand::new(
            Arc::new(WritePolicyCfg::within(home)),
            Arc::clone(&gate),
            Arc::clone(&self.sandbox),
        ));
        commands.register(BashCommand::new(
            Arc::new(bash_cfg),
            Arc::clone(&self.sandbox),
            gate,
        ));
        commands
    }
}

/// A canonical directory removed with everything in it when dropped, even
/// what a sandboxed command made unwritable.
#[derive(Debug)]
struct RelayDir {
    path: PathBuf,
}

impl RelayDir {
    /// A new directory under `parent` with empty input, home and output
    /// subdirectories, and the root that confines paths to it.
    fn create(parent: &Path) -> Result<(Self, ConfinedRoot), QuarantineError> {
        let create_error = |source| QuarantineError::Create {
            parent: parent.to_path_buf(),
            source,
        };
        let path = tempfile::Builder::new()
            .prefix("relay-")
            .tempdir_in(parent)
            .map_err(create_error)?
            .keep();
        let dir = Self {
            path: std::fs::canonicalize(&path).map_err(create_error)?,
        };
        for name in [INPUT_DIR, HOME_DIR, OUTPUT_DIR] {
            std::fs::create_dir(dir.path.join(name)).map_err(create_error)?;
        }
        let root =
            ConfinedRoot::open(dir.path.clone(), dir.path.join(HOME_DIR)).map_err(create_error)?;
        Ok((dir, root))
    }
}

impl Drop for RelayDir {
    fn drop(&mut self) {
        if std::fs::remove_dir_all(&self.path).is_ok() {
            return;
        }
        make_dirs_writable(&self.path);
        if let Err(error) = std::fs::remove_dir_all(&self.path) {
            warn!(
                target: "assistd::policy",
                path = %self.path.display(),
                %error,
                "cannot remove a relay's quarantine directory"
            );
        }
    }
}

/// Give the owner full access to `dir` and every directory under it, so
/// their entries can be removed. Symlinks are not followed.
fn make_dirs_writable(dir: &Path) {
    if !std::fs::symlink_metadata(dir).is_ok_and(|meta| meta.is_dir()) {
        return;
    }
    let _ = std::fs::set_permissions(dir, Permissions::from_mode(0o700));
    for entry in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        make_dirs_writable(&entry.path());
    }
}

fn copy_inputs(input_dir: &Path, inputs: &[PathBuf]) -> Result<(), QuarantineError> {
    let mut names = HashSet::new();
    for input in inputs {
        let name = input
            .file_name()
            .ok_or_else(|| QuarantineError::UnnamedInput(input.clone()))?;
        if !names.insert(name) {
            return Err(QuarantineError::DuplicateInput(name.to_os_string()));
        }
        copy_input(input, &input_dir.join(name)).map_err(|source| QuarantineError::CopyInput {
            path: input.clone(),
            source,
        })?;
    }
    Ok(())
}

/// Copy the regular file or directory tree at `source` to `dest`. Only
/// `source` itself may be a symlink; links and special files inside a
/// directory are left out.
fn copy_input(source: &Path, dest: &Path) -> io::Result<()> {
    let meta = std::fs::metadata(source)?;
    if meta.is_dir() {
        copy_tree(source, dest)
    } else if meta.is_file() {
        copy_file(source, dest)
    } else {
        Err(io::Error::new(
            ErrorKind::InvalidInput,
            "not a regular file or directory",
        ))
    }
}

fn copy_tree(source: &Path, dest: &Path) -> io::Result<()> {
    std::fs::create_dir(dest)?;
    for entry in std::fs::read_dir(source)? {
        let entry = entry?;
        let kind = entry.file_type()?;
        let target = dest.join(entry.file_name());
        if kind.is_dir() {
            copy_tree(&entry.path(), &target)?;
        } else if kind.is_file() {
            copy_file(&entry.path(), &target)?;
        }
    }
    Ok(())
}

fn copy_file(source: &Path, dest: &Path) -> io::Result<()> {
    let (mut file, _) = assistd_utils::fs::open_regular(source)?;
    io::copy(&mut file, &mut File::create_new(dest)?)?;
    Ok(())
}
