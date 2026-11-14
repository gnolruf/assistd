//! The "always allow" store: programs on neither of the classifier's lists that the user
//! approved for good.

use std::collections::{BTreeMap, BTreeSet};
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
    "bwrap",
    "capsh",
    "cgexec",
    "chroot",
    "chrt",
    "csh",
    "dash",
    "dbus-run-session",
    "deno",
    "doas",
    "elixir",
    "erl",
    "expect",
    "fakeroot",
    "firejail",
    "fish",
    "flatpak-spawn",
    "flock",
    "gawk",
    "gdb",
    "guile",
    "ionice",
    "irb",
    "julia",
    "ksh",
    "ltrace",
    "lua",
    "luajit",
    "mawk",
    "mksh",
    "nawk",
    "node",
    "nodejs",
    "nsenter",
    "numactl",
    "parallel",
    "perl",
    "php",
    "pkexec",
    "prlimit",
    "proot",
    "pwsh",
    "pypy",
    "python",
    "r",
    "rscript",
    "ruby",
    "run0",
    "runuser",
    "script",
    "setpriv",
    "setsid",
    "sg",
    "sh",
    "ssh",
    "strace",
    "su",
    "sudo",
    "systemd-run",
    "taskset",
    "tclsh",
    "tcsh",
    "time",
    "unshare",
    "valgrind",
    "watch",
    "wish",
    "xvfb-run",
    "zsh",
];

/// Build tools, compilers, package managers and version control, named without a trailing
/// version. Each runs code named by the project, config or options it is pointed at, so
/// "always allow" never covers them.
const RUNS_PROJECT_CODE: &[&str] = &[
    "ant",
    "bazel",
    "bazelisk",
    "bmake",
    "buck",
    "bundle",
    "bundler",
    "c++",
    "cabal",
    "cargo",
    "cc",
    "clang",
    "clang++",
    "cmake",
    "composer",
    "conda",
    "cpan",
    "cpanm",
    "ctest",
    "dart",
    "dotnet",
    "dune",
    "flutter",
    "g++",
    "gcc",
    "gem",
    "gh",
    "ghc",
    "git",
    "gmake",
    "go",
    "go-task",
    "gradle",
    "hatch",
    "hg",
    "java",
    "javac",
    "jj",
    "just",
    "lein",
    "luarocks",
    "make",
    "makepkg",
    "mamba",
    "meson",
    "mix",
    "mvn",
    "nimble",
    "ninja",
    "nix",
    "nix-build",
    "nix-shell",
    "nox",
    "npm",
    "npx",
    "opam",
    "paru",
    "pdm",
    "pip",
    "pipenv",
    "pipx",
    "pixi",
    "pnpm",
    "pnpx",
    "poetry",
    "pre-commit",
    "pytest",
    "qmake",
    "rake",
    "rebar",
    "rustc",
    "rustup",
    "sbt",
    "scons",
    "stack",
    "svn",
    "swift",
    "tox",
    "uv",
    "uvx",
    "xmake",
    "yarn",
    "yay",
    "zig",
];

/// Bash builtins the classifier leaves unvetted. Bash runs them in place of any program of the
/// same name, so an approval pinned to that program would not cover what runs.
const UNVETTED_BUILTINS: &[&str] = &[
    "bind",
    "compgen",
    "complete",
    "enable",
    "fc",
    "history",
    "kill",
    "mapfile",
    "readarray",
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

/// How the approvals treat one program run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Verdict {
    Approved,
    /// `approvable` when "always allow" can add it for the scope asked about.
    Unapproved {
        approvable: bool,
    },
}

/// The programs approved with "always allow", each pinned to the file its name resolved to and
/// to the runs of it the approval covers.
#[derive(Debug)]
pub struct Allowlist {
    approved: RwLock<BTreeMap<ProgramApproval, PathBuf>>,
    search_path: SearchPath,
    /// File names of the shells in [`SHELLS_FILE`] and of what they link to.
    shells: BTreeSet<String>,
    store: Option<PathBuf>,
    saving: tokio::sync::Mutex<()>,
}

impl Allowlist {
    /// Approvals loaded from and saved to `store`. Fails when `store` exists but cannot be read
    /// or parsed.
    pub fn load(search_path: SearchPath, store: PathBuf) -> Result<Self, AllowlistError> {
        let mut allowlist = Self::unsaved(search_path);
        let approved = load_approvals(&store)?
            .into_iter()
            .filter(|(approval, path)| !allowlist.never_approvable(&approval.name, path))
            .collect();
        allowlist.approved = RwLock::new(approved);
        allowlist.store = Some(store);
        Ok(allowlist)
    }

    /// Approvals that last only as long as the allowlist does.
    pub fn unsaved(search_path: SearchPath) -> Self {
        Self {
            approved: RwLock::default(),
            search_path,
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

    /// How the approvals treat the bare program `name` run with arguments in `scope`, which is
    /// `None` when no approval could cover them.
    pub(super) fn verdict(&self, name: &str, scope: Option<&ApprovalScope>) -> Verdict {
        let Some(found) = self.resolve(name) else {
            return Verdict::Unapproved { approvable: false };
        };
        let approved = self.approved.read().iter().any(|(approval, pinned)| {
            approval.name == name && scope == Some(&approval.scope) && *pinned == found
        });
        if approved {
            Verdict::Approved
        } else {
            Verdict::Unapproved {
                approvable: scope.is_some() && !self.never_approvable(name, &found),
            }
        }
    }

    /// Whether `name`, or the file `path` finally links to, is a shell, one of
    /// [`RUNS_ANY_CODE`] or [`RUNS_PROJECT_CODE`] with or without a version (`python3.12`), or
    /// one of [`UNVETTED_BUILTINS`].
    fn never_approvable(&self, name: &str, path: &Path) -> bool {
        if UNVETTED_BUILTINS.contains(&name) {
            return true;
        }
        let real = std::fs::canonicalize(path).ok();
        let real_name = real
            .as_deref()
            .and_then(Path::file_name)
            .and_then(|name| name.to_str());
        iter::once(name).chain(real_name).any(|name| {
            self.shells.contains(name)
                || [name.to_ascii_lowercase(), program(name)]
                    .iter()
                    .any(|name| runs_code(name))
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
}

fn runs_code(name: &str) -> bool {
    RUNS_ANY_CODE.contains(&name) || RUNS_PROJECT_CODE.contains(&name)
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
