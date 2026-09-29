//! Names approved with "always allow" for gates outside the program
//! allowlist, such as web hosts and MCP tools, kept in a TOML file.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tracing::warn;

use super::allowlist::AllowlistError;
use super::confirm::{Approval, ConfirmationGate, ConfirmationRequest};

/// File, beside the config file, that keeps the web hosts `web` may fetch
/// from without confirmation.
pub const APPROVED_HOSTS_FILE: &str = "approved_hosts.toml";

/// File, beside the config file, that keeps the MCP tools that run without
/// confirmation.
pub const APPROVED_MCP_TOOLS_FILE: &str = "approved_mcp_tools.toml";

const APPROVALS_HEADER: &str = "\
# Approved with \"always allow\". Delete an entry to be asked again.
";

/// A set of names approved for good, saved after every approval.
#[derive(Debug)]
pub struct Approvals {
    approved: RwLock<BTreeSet<String>>,
    store: Option<PathBuf>,
    saving: tokio::sync::Mutex<()>,
}

impl Approvals {
    /// The approvals saved in `store`, and saved back there; none yet when
    /// it does not exist.
    ///
    /// # Errors
    /// [`AllowlistError`] when `store` exists but cannot be read or parsed.
    pub fn load(store: PathBuf) -> Result<Self, AllowlistError> {
        let approved = match read_store(&store)? {
            Some(text) => {
                let stored: Stored =
                    toml::from_str(&text).map_err(|source| AllowlistError::Parse {
                        path: store.clone(),
                        source,
                    })?;
                stored.approved
            }
            None => BTreeSet::new(),
        };
        Ok(Self {
            approved: RwLock::new(approved),
            store: Some(store),
            saving: tokio::sync::Mutex::new(()),
        })
    }

    /// Approvals that last only as long as this value.
    pub fn unsaved() -> Self {
        Self {
            approved: RwLock::default(),
            store: None,
            saving: tokio::sync::Mutex::new(()),
        }
    }

    /// Whether `name` was approved.
    pub fn contains(&self, name: &str) -> bool {
        self.approved.read().contains(name)
    }

    /// Whether a use of `name` may go ahead: it was approved for good, or
    /// `gate` approves the `request` built now. An "always" answer approves
    /// `name` for good.
    pub async fn confirm(
        &self,
        name: &str,
        gate: &dyn ConfirmationGate,
        request: impl FnOnce() -> ConfirmationRequest,
    ) -> bool {
        if self.contains(name) {
            return true;
        }
        let approval = gate.confirm(request()).await;
        if approval == Approval::Always
            && let Err(e) = self.approve(name).await
        {
            warn!(
                target: "assistd::policy",
                error = %e,
                "approval holds until the daemon exits but was not saved"
            );
        }
        approval != Approval::Deny
    }

    /// Approve `name` for good and save.
    ///
    /// # Errors
    /// [`AllowlistError::Write`] when saving fails; the approval still holds
    /// until the daemon exits.
    pub async fn approve(&self, name: &str) -> Result<(), AllowlistError> {
        let _saving = self.saving.lock().await;
        let approved = {
            let mut approved = self.approved.write();
            approved.insert(name.to_string());
            approved.clone()
        };
        let Some(store) = &self.store else {
            return Ok(());
        };
        let body = toml::to_string(&Stored { approved }).map_err(|e| AllowlistError::Write {
            path: store.clone(),
            source: std::io::Error::new(std::io::ErrorKind::InvalidData, e),
        })?;
        write_store(store, APPROVALS_HEADER, &body).await
    }
}

#[derive(Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Stored {
    #[serde(default)]
    approved: BTreeSet<String>,
}

/// The text of `store`, or `None` when it does not exist.
pub(super) fn read_store(store: &Path) -> Result<Option<String>, AllowlistError> {
    match std::fs::read_to_string(store) {
        Ok(text) => Ok(Some(text)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(source) => Err(AllowlistError::Read {
            path: store.to_path_buf(),
            source,
        }),
    }
}

/// Write `header` and `body` to a sibling file, then rename it over
/// `store`, so a crash never leaves a half-written file.
pub(super) async fn write_store(
    store: &Path,
    header: &str,
    body: &str,
) -> Result<(), AllowlistError> {
    let write_err = |source| AllowlistError::Write {
        path: store.to_path_buf(),
        source,
    };
    if let Some(dir) = store.parent() {
        tokio::fs::create_dir_all(dir).await.map_err(write_err)?;
    }
    let staged = store.with_extension("toml.tmp");
    tokio::fs::write(&staged, format!("{header}\n{body}"))
        .await
        .map_err(write_err)?;
    tokio::fs::rename(&staged, store).await.map_err(write_err)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn approvals_survive_a_reload() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = dir.path().join(APPROVED_HOSTS_FILE);
        let approvals = Approvals::load(store.clone()).expect("missing store loads empty");
        assert!(!approvals.contains("example.com"));

        approvals.approve("example.com").await.expect("saved");
        assert!(approvals.contains("example.com"));

        let reloaded = Approvals::load(store).expect("saved store loads");
        assert!(reloaded.contains("example.com"));
        assert!(!reloaded.contains("example.org"));
    }

    #[test]
    fn unparseable_store_is_an_error() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = dir.path().join(APPROVED_MCP_TOOLS_FILE);
        std::fs::write(&store, "approved = 3\n").expect("write");
        assert!(matches!(
            Approvals::load(store),
            Err(AllowlistError::Parse { .. })
        ));
    }
}
