use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

use assistd_utils::path::tilde_remainder;
use async_trait::async_trait;
use rustix::fs::{FileType, Mode, OFlags};
use rustix::io::Errno;
use tokio::io::AsyncWriteExt;

use crate::command::{Command, CommandInput, CommandOutput, Hint, error_line, io_error_nav};
use crate::exec::POLICY_DENIED_EXIT;
use crate::policy::{Approval, ConfirmationGate, ConfirmationRequest, SandboxInfo};

/// Characters of the content shown when asking to confirm a write.
const PREVIEW_MAX_CHARS: usize = 4096;

/// Writable-path allowlist of canonical prefixes, non-empty by construction.
#[derive(Debug, Clone)]
pub struct WritePolicyCfg {
    first: PathBuf,
    rest: Vec<PathBuf>,
    protected: Vec<PathBuf>,
}

impl WritePolicyCfg {
    /// A policy over canonicalized writable prefixes, or `None` if none
    /// were given.
    pub fn new(writable_paths: Vec<PathBuf>) -> Option<Self> {
        let mut prefixes = writable_paths.into_iter();
        let first = prefixes.next()?;
        Some(Self {
            first,
            rest: prefixes.collect(),
            protected: Vec::new(),
        })
    }

    /// Also refuse anything under the canonical `dirs`, even inside a
    /// writable prefix.
    pub fn protecting(mut self, dirs: Vec<PathBuf>) -> Self {
        self.protected = dirs;
        self
    }

    /// The permitted path prefixes, in configuration order.
    pub fn prefixes(&self) -> impl Iterator<Item = &PathBuf> {
        std::iter::once(&self.first).chain(&self.rest)
    }

    /// Resolve `raw` to the path that will be written, refusing anything
    /// outside the allowlist, a symlink, or hidden at any depth below a prefix.
    fn resolve(&self, raw: &str, home: Option<&str>) -> Result<PathBuf, PathResolveError> {
        let resolved = resolve_for_allowlist(raw, home)?;
        if self.protected.iter().any(|dir| resolved.starts_with(dir)) {
            return Err(PathResolveError::Protected);
        }
        if resolved.is_symlink() {
            return Err(PathResolveError::Symlink);
        }
        let mut covering = self
            .prefixes()
            .filter_map(|prefix| resolved.strip_prefix(prefix).ok())
            .peekable();
        if covering.peek().is_none() {
            Err(PathResolveError::NotAllowlisted)
        } else if covering.any(|below| !has_hidden_component(below)) {
            Ok(resolved)
        } else {
            Err(PathResolveError::Hidden)
        }
    }
}

/// `write PATH [CONTENT...]`: write the joined args (or else stdin) to an
/// absolute, allowlisted PATH, asking first unless it is under the
/// sandbox's shared scratch directory.
/// Policy refusals exit 126.
#[derive(Debug)]
pub struct WriteCommand {
    cfg: Arc<WritePolicyCfg>,
    gate: Arc<dyn ConfirmationGate>,
    sandbox: Arc<SandboxInfo>,
}

impl WriteCommand {
    /// A `write` command confined to `cfg`'s allowlist, asking `gate`
    /// before writing outside `sandbox`'s shared scratch directory, and
    /// noting a file `sandbox` hides from `bash`.
    pub fn new(
        cfg: Arc<WritePolicyCfg>,
        gate: Arc<dyn ConfirmationGate>,
        sandbox: Arc<SandboxInfo>,
    ) -> Self {
        Self { cfg, gate, sandbox }
    }

    /// A command whose allowlist is `/`, permitting any absolute path
    /// without asking.
    #[cfg(test)]
    pub fn permissive_for_tests() -> Self {
        Self::new(
            Arc::new(WritePolicyCfg::new(vec![PathBuf::from("/")]).expect("non-empty allowlist")),
            Arc::new(crate::policy::AlwaysAllowGate),
            SandboxInfo::none(),
        )
    }

    async fn confirmed(&self, target: &Path, content: &[u8]) -> bool {
        if self.scratch().is_some_and(|dir| target.starts_with(dir)) {
            return true;
        }
        let approval = self
            .gate
            .confirm(ConfirmationRequest {
                tool: "write".to_string(),
                script: confirmation_script(target, content),
                matched_pattern: "writes a file outside the scratch directory".to_string(),
                always_allow: Vec::new(),
            })
            .await;
        approval != Approval::Deny
    }

    fn scratch(&self) -> Option<&Path> {
        self.sandbox.shared.scratch.as_deref()
    }

    fn declined_recovery(&self) -> String {
        match self.scratch() {
            Some(scratch) => format!(
                "a path under {}, or ask the user to make this change",
                scratch.display()
            ),
            None => "asking the user to make this change".to_string(),
        }
    }
}

#[async_trait]
impl Command for WriteCommand {
    fn name(&self) -> &'static str {
        "write"
    }

    fn summary(&self) -> &'static str {
        "write a file (allowlist-gated) from stdin or inline args"
    }

    fn help(&self) -> String {
        let sharing_hint = self.sandbox.sharing_hint();
        format!(
            "usage: write PATH [CONTENT...]\n\
         \n\
         Write bytes to PATH. Two shapes:\n  \
           `echo \"hi\" | write PATH`   - stdin is the file content (pipeline form)\n  \
           `write PATH hello world`   - args beyond PATH are joined by spaces and written\n\
         \n\
         PATH must be absolute (relative paths are rejected) and must fall \
         under one of the prefixes in `[tools.write] writable_paths`. \
         Symlinks, devices, pipes, sockets, and hidden (dot) entries at any \
         depth below a prefix, are refused. Writes outside the shared scratch \
         directory ask the user first. A file under /tmp \
         or /run is not visible to `bash` scripts, which get empty ones of \
         their own; for a file they must see, use {sharing_hint}. \
         Tilde expansion is supported. Exit 126 on policy denial or a \
         declined confirmation, 1 on write failure (permissions, \
         no-such-dir, etc.).\n"
        )
    }

    async fn run(&self, input: CommandInput) -> CommandOutput {
        if input.args.is_empty() {
            return CommandOutput::usage(self.help());
        }
        let raw_path = input.args[0].clone();
        let content: Vec<u8> = if input.args.len() > 1 {
            input.args[1..].join(" ").into_bytes()
        } else {
            input.stdin.unwrap_or_default()
        };

        let home = std::env::var("HOME").ok();
        let write_target = match self.cfg.resolve(&raw_path, home.as_deref()) {
            Ok(path) => path,
            Err(e) => {
                return CommandOutput::failed(
                    POLICY_DENIED_EXIT,
                    e.error_line(&raw_path, self.scratch()).into_bytes(),
                );
            }
        };

        if !self.confirmed(&write_target, &content).await {
            return CommandOutput::failed(
                POLICY_DENIED_EXIT,
                error_line(
                    "write",
                    format_args!("{raw_path}: write cancelled by user"),
                    Hint::Try,
                    self.declined_recovery(),
                )
                .into_bytes(),
            );
        }

        let hidden_by = self.sandbox.private_dir_hiding(&write_target);
        match write_without_symlinks(write_target, content).await {
            Ok(()) => CommandOutput {
                stderr: hidden_by
                    .map(|dir| {
                        hidden_from_bash_note(&raw_path, &dir, &self.sandbox.sharing_hint())
                            .into_bytes()
                    })
                    .unwrap_or_default(),
                ..CommandOutput::default()
            },
            Err(failure) => failure.output(&raw_path),
        }
    }
}

#[derive(Debug)]
enum PathResolveError {
    Relative,
    HomeNotSet,
    AnchorMissing(String),
    NotAllowlisted,
    Protected,
    Symlink,
    Hidden,
}

impl PathResolveError {
    fn error_line(&self, raw_path: &str, scratch: Option<&Path>) -> String {
        if let (Self::NotAllowlisted, Some(scratch)) = (self, scratch) {
            return error_line(
                "write",
                format_args!("{raw_path}: path not in writable allowlist"),
                Hint::Try,
                format_args!("a path under {}", scratch.display()),
            );
        }
        let (what, hint, recovery) = match self {
            Self::Relative => (
                format!("{raw_path}: relative paths not permitted"),
                Hint::Try,
                "an absolute path under an allowlisted directory",
            ),
            Self::HomeNotSet => (
                format!("{raw_path}: cannot expand ~ ($HOME not set)"),
                Hint::Try,
                "writing an explicit absolute path instead of ~",
            ),
            Self::AnchorMissing(anchor) => (
                format!("{raw_path}: cannot resolve ancestor {anchor}"),
                Hint::Check,
                "that the directory exists or widen [tools.write] writable_paths",
            ),
            Self::NotAllowlisted => (
                format!("{raw_path}: path not in writable allowlist"),
                Hint::Check,
                "[tools.write] writable_paths in config",
            ),
            Self::Protected => (
                format!("{raw_path}: assistd's own configuration is not writable"),
                Hint::Try,
                "asking the user to make this change themselves",
            ),
            Self::Symlink => (
                format!("{raw_path}: is a symlink"),
                Hint::Try,
                "writing to the file it points at",
            ),
            Self::Hidden => (
                format!("{raw_path}: hidden files and directories are not writable"),
                Hint::Try,
                "asking the user to make this change themselves",
            ),
        };
        error_line("write", what, hint, recovery)
    }
}

#[derive(Debug)]
enum WriteFailure {
    NotRegularFile,
    Io(std::io::Error),
}

impl WriteFailure {
    fn output(&self, raw_path: &str) -> CommandOutput {
        match self {
            Self::NotRegularFile => CommandOutput::failed(
                POLICY_DENIED_EXIT,
                error_line(
                    "write",
                    format_args!("{raw_path}: not a regular file (device, pipe, or socket)"),
                    Hint::Try,
                    "a path to a regular file, or to one that does not exist yet",
                )
                .into_bytes(),
            ),
            Self::Io(e) => {
                CommandOutput::failed(1, io_error_nav("write", raw_path, e).into_bytes())
            }
        }
    }
}

impl From<std::io::Error> for WriteFailure {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<Errno> for WriteFailure {
    fn from(errno: Errno) -> Self {
        Self::Io(errno.into())
    }
}

async fn write_without_symlinks(path: PathBuf, content: Vec<u8>) -> Result<(), WriteFailure> {
    let open = tokio::task::spawn_blocking(move || open_without_symlinks(&path));
    let mut file = tokio::fs::File::from_std(open.await.map_err(std::io::Error::other)??);
    file.write_all(&content).await?;
    Ok(file.flush().await?)
}

/// Open `path` for writing without blocking, refusing anything but a
/// regular file; a FIFO with no reader fails the open with `ENXIO`.
fn open_without_symlinks(path: &Path) -> Result<std::fs::File, WriteFailure> {
    let fd = rustix::fs::openat2(
        rustix::fs::CWD,
        path,
        OFlags::WRONLY
            | OFlags::CREATE
            | OFlags::TRUNC
            | OFlags::CLOEXEC
            | OFlags::NONBLOCK
            | OFlags::NOCTTY,
        Mode::from_raw_mode(0o666),
        rustix::fs::ResolveFlags::NO_SYMLINKS,
    )
    .map_err(|errno| {
        if errno == Errno::NXIO {
            WriteFailure::NotRegularFile
        } else {
            WriteFailure::from(errno)
        }
    })?;
    if !FileType::from_raw_mode(rustix::fs::fstat(&fd)?.st_mode).is_file() {
        return Err(WriteFailure::NotRegularFile);
    }
    rustix::fs::fcntl_setfl(
        &fd,
        rustix::fs::fcntl_getfl(&fd)?.difference(OFlags::NONBLOCK),
    )?;
    Ok(std::fs::File::from(fd))
}

/// Resolve `raw` to an absolute path whose ancestors are canonical and
/// whose final component is kept as written.
fn resolve_for_allowlist(raw: &str, home: Option<&str>) -> Result<PathBuf, PathResolveError> {
    let expanded = expand_tilde(raw, home)?;
    if !expanded.is_absolute() {
        return Err(PathResolveError::Relative);
    }
    let cleaned = lexical_clean(&expanded);
    match (cleaned.parent(), cleaned.file_name()) {
        (Some(parent), Some(file_name)) => {
            Ok(canonicalize_existing_prefix(parent)?.join(file_name))
        }
        _ => canonicalize_existing_prefix(&cleaned),
    }
}

fn canonicalize_existing_prefix(path: &Path) -> Result<PathBuf, PathResolveError> {
    let (anchor, tail) = split_at_existing(path);
    let canonical_anchor = std::fs::canonicalize(&anchor)
        .map_err(|_| PathResolveError::AnchorMissing(anchor.to_string_lossy().into_owned()))?;
    if tail.as_os_str().is_empty() {
        Ok(canonical_anchor)
    } else {
        Ok(canonical_anchor.join(tail))
    }
}

fn expand_tilde(raw: &str, home: Option<&str>) -> Result<PathBuf, PathResolveError> {
    if tilde_remainder(raw).is_none() {
        return Ok(PathBuf::from(raw));
    }
    let home = home.ok_or(PathResolveError::HomeNotSet)?;
    Ok(assistd_utils::path::expand_tilde(raw, Path::new(home)))
}

fn has_hidden_component(path: &Path) -> bool {
    path.components()
        .any(|component| component.as_os_str().as_encoded_bytes().starts_with(b"."))
}

fn hidden_from_bash_note(raw_path: &str, dir: &Path, sharing_hint: &str) -> String {
    format!(
        "[note] write: {raw_path} is written, but bash scripts cannot see it: the sandbox \
         gives each bash call its own empty {}. {}: {sharing_hint}\n",
        dir.display(),
        Hint::Use,
    )
}

fn confirmation_script(target: &Path, content: &[u8]) -> String {
    let text = String::from_utf8_lossy(content);
    let mut script = format!("write {}\n", target.display());
    script.extend(text.chars().take(PREVIEW_MAX_CHARS));
    if text.chars().nth(PREVIEW_MAX_CHARS).is_some() {
        script.push_str("\n…");
    }
    script
}

/// Collapse `.` and `..` components without touching disk. A leading
/// `/..` is discarded, as the kernel does.
pub(crate) fn lexical_clean(path: &Path) -> PathBuf {
    let mut out: Vec<Component<'_>> = Vec::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => match out.last() {
                Some(Component::Normal(_)) => {
                    out.pop();
                }
                Some(Component::RootDir) => {}
                _ => out.push(component),
            },
            _ => out.push(component),
        }
    }
    let mut result = PathBuf::new();
    for component in out {
        result.push(component.as_os_str());
    }
    result
}

/// Split `path` into its deepest existing ancestor and the remainder.
fn split_at_existing(path: &Path) -> (PathBuf, PathBuf) {
    let mut anchor = path.to_path_buf();
    let mut tail = PathBuf::new();
    loop {
        if anchor.symlink_metadata().is_ok() {
            return (anchor, tail);
        }
        let Some(parent) = anchor.parent() else {
            return (path.to_path_buf(), PathBuf::new());
        };
        let Some(file_name) = anchor.file_name() else {
            return (anchor, tail);
        };
        let mut new_tail = PathBuf::from(file_name);
        if !tail.as_os_str().is_empty() {
            new_tail.push(&tail);
        }
        tail = new_tail;
        anchor = parent.to_path_buf();
    }
}

#[cfg(test)]
mod tests;
