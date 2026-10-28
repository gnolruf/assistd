//! The programs a command may run without the user's confirmation.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::Metadata;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::{fmt, iter};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use super::approvals::{read_store, write_store};
use super::review::program;

/// File, beside the config file, that keeps "always allow" approvals.
pub const APPROVALS_FILE: &str = "allowed_programs.toml";

/// Shells, interpreters, and programs that run any command they are given
/// (as root or elsewhere), named without a trailing version; "always
/// allow" never covers them.
const RUNS_ANY_CODE: &[&str] = &[
    "ash",
    "awk",
    "bash",
    "bun",
    "busybox",
    "chroot",
    "csh",
    "dash",
    "deno",
    "doas",
    "elixir",
    "erl",
    "fish",
    "flock",
    "gawk",
    "guile",
    "irb",
    "julia",
    "ksh",
    "lua",
    "luajit",
    "mawk",
    "mksh",
    "nawk",
    "node",
    "nodejs",
    "nsenter",
    "parallel",
    "perl",
    "php",
    "pkexec",
    "pwsh",
    "pypy",
    "python",
    "r",
    "rscript",
    "ruby",
    "run0",
    "setsid",
    "sh",
    "ssh",
    "strace",
    "su",
    "sudo",
    "systemd-run",
    "tclsh",
    "tcsh",
    "unshare",
    "watch",
    "wish",
    "zsh",
];

/// The system's list of installed shells, one path per line, each matched
/// like an entry of [`RUNS_ANY_CODE`].
const SHELLS_FILE: &str = "/etc/shells";

const APPROVALS_HEADER: &str = "\
# Programs approved with \"always allow\". Each runs without confirmation
# while its name still resolves to `path` on the command's PATH: with
# `subcommand` as its first argument, or with only options when there is
# none. Delete an entry to be asked again.
";

/// Directories bare command names are looked up in, as the command sees
/// them.
#[derive(Debug, Clone)]
pub struct SearchPath {
    /// Absolute directories, in lookup order.
    pub dirs: Vec<PathBuf>,
    /// Commands cannot modify anything in `dirs`, so ownership is not
    /// checked.
    pub read_only: bool,
}

/// Why approvals could not be loaded or saved.
#[derive(Debug, thiserror::Error)]
pub enum AllowlistError {
    #[error("failed to read {path}: {source}")]
    Read {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("failed to parse {path}: {source}")]
    Parse {
        path: PathBuf,
        source: toml::de::Error,
    },
    #[error("failed to save {path}: {source}")]
    Write {
        path: PathBuf,
        source: std::io::Error,
    },
}

/// One "always allow" approval: a program and the runs of it it covers.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct ProgramApproval {
    pub(super) name: String,
    pub(super) scope: ApprovalScope,
}

impl fmt::Display for ProgramApproval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.scope {
            ApprovalScope::OptionsOnly => write!(f, "{} [options]", self.name),
            ApprovalScope::Subcommand(subcommand) => write!(f, "{} {subcommand} …", self.name),
        }
    }
}

/// Which runs of a program one approval covers.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum ApprovalScope {
    /// No arguments, or only options.
    OptionsOnly,
    /// This first argument, whatever follows it.
    Subcommand(String),
}

/// How the allowlist treats one command word.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Verdict {
    Allowed,
    /// `approvable` when "always allow" can add it for the scope asked about.
    Unlisted {
        approvable: bool,
    },
    /// A bare name that resolves to no program.
    Missing,
}

/// The programs a command may run without confirmation: the configured
/// ones, which count only where the user cannot modify them, and those
/// approved with "always allow", each pinned to the file it resolved to.
#[derive(Debug)]
pub struct Allowlist {
    configured: BTreeSet<String>,
    approved: RwLock<BTreeMap<ProgramApproval, PathBuf>>,
    search_path: SearchPath,
    /// The user cannot add programs to the search path.
    search_path_fixed: bool,
    /// File names of the shells in [`SHELLS_FILE`] and of what they link to.
    shells: BTreeSet<String>,
    store: Option<PathBuf>,
    saving: tokio::sync::Mutex<()>,
}

impl Allowlist {
    /// An allowlist of `configured` names and absolute paths, with approvals loaded from and
    /// saved to `store`. Fails when `store` exists but cannot be read or parsed.
    pub fn load(
        configured: impl IntoIterator<Item = String>,
        search_path: SearchPath,
        store: PathBuf,
    ) -> Result<Self, AllowlistError> {
        let mut allowlist = Self::unsaved(configured, search_path);
        let approved = load_approvals(&store)?
            .into_iter()
            .filter(|(approval, path)| !allowlist.runs_any_code(&approval.name, path))
            .collect();
        allowlist.approved = RwLock::new(approved);
        allowlist.store = Some(store);
        Ok(allowlist)
    }

    /// An allowlist whose approvals last only as long as it does.
    pub fn unsaved(configured: impl IntoIterator<Item = String>, search_path: SearchPath) -> Self {
        let search_path_fixed = search_path.read_only
            || search_path
                .dirs
                .iter()
                .all(|dir| std::fs::metadata(dir).is_ok_and(|m| !user_can_modify(&m)));
        Self {
            configured: configured.into_iter().collect(),
            approved: RwLock::default(),
            search_path,
            search_path_fixed,
            shells: installed_shells(),
            store: None,
            saving: tokio::sync::Mutex::new(()),
        }
    }

    /// Keep `approvals` for good, pinning each program to the file it resolves to now (skipping
    /// any that resolve to nothing), and save. On [`AllowlistError::Write`] the approvals still
    /// hold until the daemon exits.
    pub async fn approve(&self, approvals: &[ProgramApproval]) -> Result<(), AllowlistError> {
        if approvals.is_empty() {
            return Ok(());
        }
        let _saving = self.saving.lock().await;
        let approved = {
            let mut approved = self.approved.write();
            for approval in approvals {
                if let Some(path) = self.resolve(&approval.name) {
                    approved.insert(approval.clone(), path);
                }
            }
            approved.clone()
        };
        match &self.store {
            Some(store) => save(store, approved).await,
            None => Ok(()),
        }
    }

    /// Whether a name that resolves to no program now will still resolve
    /// to none when the command runs.
    pub(super) fn search_path_fixed(&self) -> bool {
        self.search_path_fixed
    }

    /// How the allowlist treats `word` run with arguments in `scope`, which
    /// is `None` when no approval could cover them.
    pub(super) fn verdict(&self, word: &str, scope: Option<&ApprovalScope>) -> Verdict {
        if word.contains('/') {
            self.path_verdict(word, scope)
        } else {
            self.name_verdict(word, scope)
        }
    }

    /// A configured path, the exact file an approval in `scope` is pinned
    /// to, or an unmodifiable absolute path whose file name is configured.
    fn path_verdict(&self, word: &str, scope: Option<&ApprovalScope>) -> Verdict {
        let path = Path::new(word);
        let allowed = self.configured.contains(word)
            || self
                .approved
                .read()
                .iter()
                .any(|(approval, pinned)| pinned == path && scope == Some(&approval.scope))
            || (path.is_absolute()
                && path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| self.configured.contains(name))
                && self.cannot_be_modified(path));
        if allowed {
            Verdict::Allowed
        } else {
            Verdict::Unlisted { approvable: false }
        }
    }

    fn name_verdict(&self, word: &str, scope: Option<&ApprovalScope>) -> Verdict {
        let Some(found) = self.resolve(word) else {
            return Verdict::Missing;
        };
        let configured = (self.configured.contains(word) && self.cannot_be_modified(&found))
            || found
                .to_str()
                .is_some_and(|path| self.configured.contains(path));
        let approved = self.approved.read().iter().any(|(approval, pinned)| {
            approval.name == word && scope == Some(&approval.scope) && *pinned == found
        });
        if configured || approved {
            Verdict::Allowed
        } else {
            Verdict::Unlisted {
                approvable: scope.is_some() && !self.runs_any_code(word, &found),
            }
        }
    }

    /// Whether `name`, or the file `path` finally links to, is a shell or
    /// one of [`RUNS_ANY_CODE`], with or without a version (`python3.12`).
    fn runs_any_code(&self, name: &str, path: &Path) -> bool {
        let real = std::fs::canonicalize(path).ok();
        let real_name = real
            .as_deref()
            .and_then(Path::file_name)
            .and_then(|name| name.to_str());
        iter::once(name).chain(real_name).any(|name| {
            self.shells.contains(name)
                || [name.to_ascii_lowercase(), program(name)]
                    .iter()
                    .any(|name| RUNS_ANY_CODE.contains(&name.as_str()))
        })
    }

    fn resolve(&self, name: &str) -> Option<PathBuf> {
        self.search_path
            .dirs
            .iter()
            .map(|dir| dir.join(name))
            .find(|candidate| {
                std::fs::metadata(candidate).is_ok_and(|m| m.is_file() && m.mode() & 0o111 != 0)
            })
    }

    /// Neither the file (after following symlinks) nor its directory can be
    /// modified by the user, or it sits directly in a read-only search path.
    fn cannot_be_modified(&self, path: &Path) -> bool {
        if self.search_path.read_only
            && path
                .parent()
                .is_some_and(|dir| self.search_path.dirs.iter().any(|d| d == dir))
        {
            return true;
        }
        let Ok(real) = std::fs::canonicalize(path) else {
            return false;
        };
        [Some(real.as_path()), real.parent()]
            .into_iter()
            .flatten()
            .all(|p| std::fs::metadata(p).is_ok_and(|m| !user_can_modify(&m)))
    }
}

/// The file names of every shell [`SHELLS_FILE`] lists and of the files
/// they link to; none when it cannot be read.
fn installed_shells() -> BTreeSet<String> {
    let listed = std::fs::read_to_string(SHELLS_FILE).unwrap_or_default();
    listed
        .lines()
        .map(str::trim)
        .filter(|line| line.starts_with('/'))
        .flat_map(|line| {
            let path = PathBuf::from(line);
            let real = std::fs::canonicalize(&path).ok();
            [Some(path), real]
        })
        .flatten()
        .filter_map(|path| path.file_name()?.to_str().map(str::to_string))
        .collect()
}

/// Whether the daemon's user may write to a file with this metadata.
pub(super) fn user_can_modify(meta: &Metadata) -> bool {
    let uid = rustix::process::geteuid();
    if uid.is_root() {
        return true;
    }
    let mode = meta.mode();
    let in_group = meta.gid() == rustix::process::getegid().as_raw()
        || rustix::process::getgroups().map_or(true, |groups| {
            groups.iter().any(|group| group.as_raw() == meta.gid())
        });
    (meta.uid() == uid.as_raw() && mode & 0o200 != 0)
        || (in_group && mode & 0o020 != 0)
        || mode & 0o002 != 0
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct Stored {
    #[serde(default, rename = "program")]
    programs: Vec<StoredProgram>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct StoredProgram {
    name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    subcommand: Option<String>,
    path: PathBuf,
}

/// The approvals saved in `store`; none when it does not exist.
fn load_approvals(store: &Path) -> Result<BTreeMap<ProgramApproval, PathBuf>, AllowlistError> {
    let Some(text) = read_store(store)? else {
        return Ok(BTreeMap::new());
    };
    let stored: Stored = toml::from_str(&text).map_err(|source| AllowlistError::Parse {
        path: store.to_path_buf(),
        source,
    })?;
    Ok(stored
        .programs
        .into_iter()
        .map(|program| {
            let scope = program
                .subcommand
                .map_or(ApprovalScope::OptionsOnly, ApprovalScope::Subcommand);
            let approval = ProgramApproval {
                name: program.name,
                scope,
            };
            (approval, program.path)
        })
        .collect())
}

async fn save(
    store: &Path,
    approved: BTreeMap<ProgramApproval, PathBuf>,
) -> Result<(), AllowlistError> {
    let stored = Stored {
        programs: approved
            .into_iter()
            .map(|(approval, path)| StoredProgram {
                name: approval.name,
                subcommand: match approval.scope {
                    ApprovalScope::OptionsOnly => None,
                    ApprovalScope::Subcommand(subcommand) => Some(subcommand),
                },
                path,
            })
            .collect(),
    };
    let body = toml::to_string(&stored).map_err(|e| AllowlistError::Write {
        path: store.to_path_buf(),
        source: std::io::Error::new(std::io::ErrorKind::InvalidData, e),
    })?;
    write_store(store, APPROVALS_HEADER, &body).await
}

#[cfg(test)]
mod tests;
