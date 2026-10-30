//! A sub-agent's disposable world: one fresh directory per relay, showing its
//! inputs read-only in place, and the only part of the host its commands can see.
//! Its only way to the network, if any, is the relay's egress proxy.

use std::collections::HashSet;
use std::ffi::OsString;
use std::fs::{File, Permissions};
use std::io::{self, ErrorKind};
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use rustix::fs::{FileType, Mode, OFlags};
use tracing::warn;

use assistd_search::SearxngClient;

use crate::command::CommandRegistry;
use crate::commands::{
    BashCommand, BashPolicyCfg, CatCommand, EchoCommand, GrepCommand, HeadCommand, LsCommand,
    SortCommand, TailCommand, UniqCommand, WcCommand, WebCommand, WriteCommand, WritePolicyCfg,
};
use crate::policy::{
    ApprovalGate, ConfinedRoot, ConfirmationGate, EgressPolicy, EgressRoute, InputBind,
    SandboxInfo, on_blocking_pool,
};

mod egress;

use egress::EgressProxy;

const INPUT_DIR: &str = "input";
const HOME_DIR: &str = "home";
const OUTPUT_DIR: &str = "output";
const EGRESS_SOCKET: &str = "egress.sock";

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
    #[error("input {} has no file name to mount it under", .0.display())]
    UnnamedInput(PathBuf),
    #[error("more than one input is named {0:?}")]
    DuplicateInput(OsString),
    #[error("cannot hand over input {}: {source}", path.display())]
    Input {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("quarantine wrapper {} is not a file", .0.display())]
    MissingExec(PathBuf),
    #[error("cannot start the relay's egress proxy: {0}")]
    Proxy(#[source] io::Error),
}

/// How a quarantine reaches the web: through a proxy that asks `hosts`
/// about each host, and through `search` for `web search` when it is enabled.
#[derive(Debug, Clone)]
pub struct Egress {
    pub hosts: ApprovalGate,
    pub search: Option<SearxngClient>,
}

/// A relay's directory and the sandbox over it: its inputs read-only, a
/// writable home, and a directory for spilled output, all at their host
/// paths; no network but its egress proxy, if it has one. Dropping it, on
/// cancel or panic too, closes the proxy and removes it all.
#[derive(Debug)]
pub struct Quarantine {
    egress: Option<RelayEgress>,
    dir: RelayDir,
    root: Arc<ConfinedRoot>,
    sandbox: Arc<SandboxInfo>,
}

impl Quarantine {
    /// A new quarantine under `parent` wrapped as `sandbox` wraps, showing each of `inputs`
    /// read-only in [`Quarantine::input_dir`] under its file name, and with `egress` its proxied
    /// way to the web. Each command runs under `exec`, the `assistd-quarantine-exec` program,
    /// which keeps it off every Unix socket made outside the sandbox but the proxy's. Fails
    /// when `sandbox` does not wrap.
    pub async fn create(
        parent: &Path,
        sandbox: &SandboxInfo,
        exec: PathBuf,
        inputs: Vec<PathBuf>,
        egress: Option<Egress>,
    ) -> Result<Self, QuarantineError> {
        let parent = parent.to_path_buf();
        let (dir, root, binds, exec) = on_blocking_pool(move || {
            if !exec.is_file() {
                return Err(QuarantineError::MissingExec(exec));
            }
            let (dir, mut root) = RelayDir::create(&parent)?;
            let binds = mount_inputs(&mut root, &dir.path.join(INPUT_DIR), &inputs)?;
            Ok((dir, root, binds, exec))
        })
        .await?;
        let egress = egress
            .map(|egress| RelayEgress::start(&dir.path.join(EGRESS_SOCKET), egress))
            .transpose()?;
        let sandbox = sandbox
            .quarantined(
                dir.path.join(HOME_DIR),
                vec![dir.path.join(INPUT_DIR), dir.path.join(OUTPUT_DIR)],
                binds,
                exec,
                egress.as_ref().map(|egress| egress.route.clone()),
            )
            .ok_or(QuarantineError::Unsandboxed)?;
        Ok(Self {
            egress,
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
    /// and `bash` in the quarantine sandbox under `bash`'s rules, asking `gate` as usual. With
    /// egress, also `web`, whose fetches are held to the same rule as the proxy.
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
        if let Some(egress) = &self.egress {
            commands.register(WebCommand::under(
                egress.policy.clone(),
                egress.search.clone(),
            ));
        }
        commands
    }
}

/// A relay's running egress proxy, the rule it holds tunnels to, how the
/// sandbox reaches it, and the relay's web search.
#[derive(Debug)]
struct RelayEgress {
    _proxy: EgressProxy,
    policy: EgressPolicy,
    route: EgressRoute,
    search: Option<SearxngClient>,
}

impl RelayEgress {
    fn start(socket: &Path, egress: Egress) -> Result<Self, QuarantineError> {
        let policy = EgressPolicy::https_only(egress.hosts);
        let proxy = EgressProxy::bind(socket, policy.clone()).map_err(QuarantineError::Proxy)?;
        Ok(Self {
            _proxy: proxy,
            policy,
            route: EgressRoute {
                socket: socket.to_path_buf(),
            },
            search: egress.search,
        })
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

/// Mount every input under its file name in `input_dir`, both for `root`'s
/// readers and, through the returned binds, for the sandbox.
fn mount_inputs(
    root: &mut ConfinedRoot,
    input_dir: &Path,
    inputs: &[PathBuf],
) -> Result<Vec<InputBind>, QuarantineError> {
    let mut names = HashSet::new();
    inputs
        .iter()
        .map(|input| {
            let name = input
                .file_name()
                .ok_or_else(|| QuarantineError::UnnamedInput(input.clone()))?;
            if !names.insert(name) {
                return Err(QuarantineError::DuplicateInput(name.to_os_string()));
            }
            mount_input(root, input, input_dir.join(name)).map_err(|source| {
                QuarantineError::Input {
                    path: input.clone(),
                    source,
                }
            })
        })
        .collect()
}

/// Open the regular file or directory at `source`, through a symlink if it is
/// one, make an empty mount point of its kind at `at`, and mount it there.
fn mount_input(root: &mut ConfinedRoot, source: &Path, at: PathBuf) -> io::Result<InputBind> {
    let source = std::fs::canonicalize(source)?;
    let fd = rustix::fs::open(&source, OFlags::PATH | OFlags::CLOEXEC, Mode::empty())?;
    match FileType::from_raw_mode(rustix::fs::fstat(&fd)?.st_mode) {
        FileType::Directory => std::fs::create_dir(&at)?,
        FileType::RegularFile => {
            File::create_new(&at)?;
        }
        _ => {
            return Err(io::Error::new(
                ErrorKind::InvalidInput,
                "not a regular file or directory",
            ));
        }
    }
    root.mount(at.clone(), fd)?;
    Ok(InputBind { source, at })
}
