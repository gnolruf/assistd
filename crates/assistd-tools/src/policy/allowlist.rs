//! The programs a command may run without the user's confirmation.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::Metadata;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

/// File, beside the config file, that keeps "always allow" approvals.
pub const APPROVALS_FILE: &str = "allowed_programs.toml";

const APPROVALS_HEADER: &str = "\
# Programs approved with \"always allow\". Each runs without confirmation
# while its name still resolves to `path` on the command's PATH. Delete an
# entry to be asked again.
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

/// How the allowlist treats one command word.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Verdict {
    Allowed,
    /// `approvable` when "always allow" can add it.
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
    approved: RwLock<BTreeMap<String, PathBuf>>,
    search_path: SearchPath,
    /// The user cannot add programs to the search path.
    search_path_fixed: bool,
    store: Option<PathBuf>,
    saving: tokio::sync::Mutex<()>,
}

impl Allowlist {
    /// An allowlist of `configured` names and absolute paths, with
    /// approvals loaded from and saved to `store`.
    ///
    /// # Errors
    /// [`AllowlistError`] when `store` exists but cannot be read or parsed.
    pub fn load(
        configured: impl IntoIterator<Item = String>,
        search_path: SearchPath,
        store: PathBuf,
    ) -> Result<Self, AllowlistError> {
        let approved = load_approvals(&store)?;
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
    /// now, and save. Names that resolve to nothing are skipped.
    ///
    /// # Errors
    /// [`AllowlistError::Write`] when saving fails; the approvals still hold
    /// until the daemon exits.
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
            self.path_verdict(word)
        } else {
            self.name_verdict(word)
        }
    }

    fn path_verdict(&self, word: &str) -> Verdict {
        let path = Path::new(word);
        let allowed = self.configured.contains(word)
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

    fn name_verdict(&self, word: &str) -> Verdict {
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

/// Whether the daemon's user may write to a file with this metadata.
fn user_can_modify(meta: &Metadata) -> bool {
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
    path: PathBuf,
}

/// The approvals saved in `store`; none when it does not exist.
fn load_approvals(store: &Path) -> Result<BTreeMap<String, PathBuf>, AllowlistError> {
    let text = match std::fs::read_to_string(store) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(BTreeMap::new()),
        Err(source) => {
            return Err(AllowlistError::Read {
                path: store.to_path_buf(),
                source,
            });
        }
    };
    let stored: Stored = toml::from_str(&text).map_err(|source| AllowlistError::Parse {
        path: store.to_path_buf(),
        source,
    })?;
    Ok(stored
        .programs
        .into_iter()
        .map(|program| (program.name, program.path))
        .collect())
}

/// Write the approvals to a sibling file, then rename it over `store`, so
/// a crash never leaves a half-written file.
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
    let staged = store.with_extension("toml.tmp");
    tokio::fs::write(&staged, format!("{APPROVALS_HEADER}\n{body}"))
        .await
        .map_err(write_err)?;
    tokio::fs::rename(&staged, store).await.map_err(write_err)
}

#[cfg(test)]
mod tests;
