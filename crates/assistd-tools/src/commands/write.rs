use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use rustix::fs::{Mode, OFlags};
use tokio::io::AsyncWriteExt;

use crate::command::{Command, CommandInput, CommandOutput, Hint, error_line, io_error_nav};
use crate::exec::POLICY_DENIED_EXIT;

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
    /// outside the allowlist, a symlink, or inside a hidden entry of a prefix.
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
        } else if covering.any(|below| !starts_with_hidden_entry(below)) {
            Ok(resolved)
        } else {
            Err(PathResolveError::Hidden)
        }
    }
}

/// `write PATH [CONTENT...]`: write the joined args (or else stdin) to an
/// absolute, allowlisted PATH. Policy refusals exit 126.
pub struct WriteCommand {
    cfg: Arc<WritePolicyCfg>,
}

impl WriteCommand {
    /// A `write` command confined to `cfg`'s allowlist.
    pub fn new(cfg: Arc<WritePolicyCfg>) -> Self {
        Self { cfg }
    }

    /// A command whose allowlist is `/`, permitting any absolute path.
    #[cfg(test)]
    pub fn permissive_for_tests() -> Self {
        Self {
            cfg: Arc::new(
                WritePolicyCfg::new(vec![PathBuf::from("/")]).expect("non-empty allowlist"),
            ),
        }
    }
}

#[async_trait]
impl Command for WriteCommand {
    fn name(&self) -> &str {
        "write"
    }

    fn summary(&self) -> &'static str {
        "write a file (allowlist-gated) from stdin or inline args"
    }

    fn help(&self) -> String {
        "usage: write PATH [CONTENT...]\n\
         \n\
         Write bytes to PATH. Two shapes:\n  \
           `echo \"hi\" | write /tmp/x`   - stdin is the file content (pipeline form)\n  \
           `write /tmp/x hello world`   - args beyond PATH are joined by spaces and written\n\
         \n\
         PATH must be absolute (relative paths are rejected) and must fall \
         under one of the prefixes in `[tools.write] writable_paths`. \
         Symlinks, and hidden (dot) entries directly inside a prefix, are \
         refused. \
         Tilde expansion is supported. Exit 126 on policy denial, 1 on \
         write failure (permissions, no-such-dir, etc.).\n"
            .to_string()
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
                    e.error_line(&raw_path).into_bytes(),
                );
            }
        };

        match write_without_symlinks(write_target, content).await {
            Ok(()) => CommandOutput::ok(Vec::new()),
            Err(e) => CommandOutput::failed(1, io_error_nav("write", &raw_path, &e).into_bytes()),
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
    fn error_line(&self, raw_path: &str) -> String {
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

async fn write_without_symlinks(path: PathBuf, content: Vec<u8>) -> std::io::Result<()> {
    let open = tokio::task::spawn_blocking(move || open_without_symlinks(&path));
    let mut file = tokio::fs::File::from_std(open.await.map_err(std::io::Error::other)??);
    file.write_all(&content).await?;
    file.flush().await
}

#[cfg(target_os = "linux")]
fn open_without_symlinks(path: &Path) -> std::io::Result<std::fs::File> {
    let fd = rustix::fs::openat2(
        rustix::fs::CWD,
        path,
        OFlags::WRONLY | OFlags::CREATE | OFlags::TRUNC | OFlags::CLOEXEC,
        Mode::from_raw_mode(0o666),
        rustix::fs::ResolveFlags::NO_SYMLINKS,
    )?;
    Ok(std::fs::File::from(fd))
}

#[cfg(target_os = "macos")]
fn open_without_symlinks(path: &Path) -> std::io::Result<std::fs::File> {
    let fd = rustix::fs::open(
        path,
        OFlags::WRONLY | OFlags::CREATE | OFlags::TRUNC | OFlags::CLOEXEC | OFlags::NOFOLLOW_ANY,
        Mode::from_raw_mode(0o666),
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
    if let Some(rest) = raw.strip_prefix("~/") {
        let home = home.ok_or(PathResolveError::HomeNotSet)?;
        Ok(PathBuf::from(home).join(rest))
    } else if raw == "~" {
        let home = home.ok_or(PathResolveError::HomeNotSet)?;
        Ok(PathBuf::from(home))
    } else {
        Ok(PathBuf::from(raw))
    }
}

fn starts_with_hidden_entry(path: &Path) -> bool {
    path.components()
        .next()
        .is_some_and(|first| first.as_os_str().as_encoded_bytes().starts_with(b"."))
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
