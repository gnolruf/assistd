//! The programs a command may run without the user's confirmation.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::Metadata;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

/// Name of the file, beside the config file, that keeps the programs
/// approved with "always allow".
pub const APPROVALS_FILE: &str = "allowed_programs.toml";

const APPROVALS_HEADER: &str = "\
# Programs approved with \"always allow\". Each runs without confirmation
# while its name still resolves to `path` on the command's PATH. Delete an
# entry to be asked again.
";

/// Directories bare command names are looked up in, as the command will
/// see them.
#[derive(Debug, Clone)]
pub struct SearchPath {
    /// Absolute directories, in lookup order.
    pub dirs: Vec<PathBuf>,
    /// Commands cannot modify anything in `dirs` (they run in a sandbox
    /// that mounts them read-only), so no ownership checks are needed.
    pub read_only: bool,
}

/// Why approvals could not be loaded or saved.
#[derive(Debug, thiserror::Error)]
pub enum AllowlistError {
    /// The approvals file exists but could not be read.
    #[error("failed to read {path}: {source}")]
    Read {
        path: PathBuf,
        source: std::io::Error,
    },
    /// The approvals file is not valid.
    #[error("failed to parse {path}: {source}")]
    Parse {
        path: PathBuf,
        source: toml::de::Error,
    },
    /// The approvals could not be saved.
    #[error("failed to save {path}: {source}")]
    Write {
        path: PathBuf,
        source: std::io::Error,
    },
}

/// How the allowlist treats one command word.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Verdict {
    Allowed,
    /// Not allowed. `approvable` when "always allow" can add it: a bare
    /// name that resolves to a program.
    Unlisted {
        approvable: bool,
    },
    /// A bare name that resolves to no program.
    Missing,
}

/// The programs a command may run without confirmation: the configured
/// ones, and those approved with "always allow". A configured bare name
/// counts only when it resolves to a file the user cannot modify, so a
/// copy planted earlier on the search path does not inherit its trust.
/// An approval is pinned to the file its name resolved to when given.
#[derive(Debug)]
pub struct Allowlist {
    configured: BTreeSet<String>,
    approved: RwLock<BTreeMap<String, PathBuf>>,
    search_path: SearchPath,
    /// Nothing on the search path can be created or replaced by the
    /// user, so a name that resolves to nothing will still resolve to
    /// nothing when the command runs.
    search_path_fixed: bool,
    store: Option<PathBuf>,
    saving: tokio::sync::Mutex<()>,
}

impl Allowlist {
    /// An allowlist of `configured` names and absolute paths, with the
    /// approvals kept in `store` (loaded now, and saved on every
    /// approval).
    ///
    /// # Errors
    ///
    /// [`AllowlistError`] when `store` exists but cannot be read or
    /// parsed.
    pub fn load(
        configured: impl IntoIterator<Item = String>,
        search_path: SearchPath,
        store: PathBuf,
    ) -> Result<Self, AllowlistError> {
        let approved = match std::fs::read_to_string(&store) {
            Ok(text) => {
                let stored: Stored =
                    toml::from_str(&text).map_err(|source| AllowlistError::Parse {
                        path: store.clone(),
                        source,
                    })?;
                stored
                    .programs
                    .into_iter()
                    .map(|p| (p.name, p.path))
                    .collect()
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => BTreeMap::new(),
            Err(source) => {
                return Err(AllowlistError::Read {
                    path: store,
                    source,
                });
            }
        };
        let mut allowlist = Self::unsaved(configured, search_path);
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
            store: None,
            saving: tokio::sync::Mutex::new(()),
        }
    }

    /// Approve `names` for good, pinning each to the file it resolves to
    /// now, and save the approvals. Names that resolve to nothing are
    /// skipped.
    ///
    /// # Errors
    ///
    /// [`AllowlistError::Write`] when the approvals cannot be saved; they
    /// still hold until the daemon exits.
    pub async fn approve(&self, names: &[String]) -> Result<(), AllowlistError> {
        let _saving = self.saving.lock().await;
        let approved = {
            let mut approved = self.approved.write();
            for name in names {
                if let Some(path) = self.resolve(name) {
                    approved.insert(name.clone(), path);
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

    pub(super) fn verdict(&self, word: &str) -> Verdict {
        if word.contains('/') {
            let path = Path::new(word);
            let allowed = self.configured.contains(word)
                || (path.is_absolute()
                    && path
                        .file_name()
                        .and_then(|name| name.to_str())
                        .is_some_and(|name| self.configured.contains(name))
                    && self.cannot_be_modified(path));
            return if allowed {
                Verdict::Allowed
            } else {
                Verdict::Unlisted { approvable: false }
            };
        }
        let Some(found) = self.resolve(word) else {
            return Verdict::Missing;
        };
        let configured = (self.configured.contains(word) && self.cannot_be_modified(&found))
            || found
                .to_str()
                .is_some_and(|path| self.configured.contains(path));
        if configured || self.approved.read().get(word) == Some(&found) {
            Verdict::Allowed
        } else {
            Verdict::Unlisted { approvable: true }
        }
    }

    /// Whether `word` names a program the user cannot modify: a bare name
    /// that resolves to one, or an absolute path to one.
    pub(super) fn trusted(&self, word: &str) -> bool {
        if word.contains('/') {
            let path = Path::new(word);
            path.is_absolute()
                && std::fs::metadata(path).is_ok_and(|m| m.is_file())
                && self.cannot_be_modified(path)
        } else {
            self.resolve(word)
                .is_some_and(|found| self.cannot_be_modified(&found))
        }
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

    /// Neither the file nor its directory can be modified by the user,
    /// following symlinks to the file they point at. Anything directly in
    /// a read-only search path qualifies.
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

/// Whether the daemon's user may write to a file with metadata `m`.
fn user_can_modify(m: &Metadata) -> bool {
    let uid = rustix::process::geteuid();
    if uid.is_root() {
        return true;
    }
    let mode = m.mode();
    let in_group = m.gid() == rustix::process::getegid().as_raw()
        || rustix::process::getgroups()
            .map_or(true, |groups| groups.iter().any(|g| g.as_raw() == m.gid()));
    (m.uid() == uid.as_raw() && mode & 0o200 != 0)
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
    path: PathBuf,
}

/// Write the approvals to a sibling temporary file, then rename it over
/// `store`, so a crash never leaves a half-written file.
async fn save(store: &Path, approved: BTreeMap<String, PathBuf>) -> Result<(), AllowlistError> {
    let write_err = |source| AllowlistError::Write {
        path: store.to_path_buf(),
        source,
    };
    let stored = Stored {
        programs: approved
            .into_iter()
            .map(|(name, path)| StoredProgram { name, path })
            .collect(),
    };
    let body = toml::to_string(&stored)
        .map_err(|e| write_err(std::io::Error::new(std::io::ErrorKind::InvalidData, e)))?;
    if let Some(dir) = store.parent() {
        tokio::fs::create_dir_all(dir).await.map_err(write_err)?;
    }
    let tmp = store.with_extension("toml.tmp");
    tokio::fs::write(&tmp, format!("{APPROVALS_HEADER}\n{body}"))
        .await
        .map_err(write_err)?;
    tokio::fs::rename(&tmp, store).await.map_err(write_err)
}

#[cfg(test)]
mod tests;
