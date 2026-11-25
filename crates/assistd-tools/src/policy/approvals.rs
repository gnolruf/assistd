//! Names approved with "always allow" for gates outside the program allowlist (web hosts, MCP
//! tools, launched applications), kept in a TOML file; an MCP tool with the hash of its definition.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use assistd_ipc::DefinitionHash;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tracing::warn;

use super::allowlist::AllowlistError;
use super::confirm::{Approval, ConfirmationGate, ConfirmationRequest};
use crate::TurnContext;

/// File, beside the config file, that keeps the web hosts `web` may fetch
/// from without confirmation.
pub const APPROVED_HOSTS_FILE: &str = "approved_hosts.toml";

/// File, beside the config file, that keeps the desktop entries `wm open`
/// launches without confirmation, each with the program it runs.
pub const APPROVED_APPS_FILE: &str = "approved_apps.toml";

/// File, beside the config file, that keeps the MCP tools that run without
/// confirmation.
pub const APPROVED_MCP_TOOLS_FILE: &str = "approved_mcp_tools.toml";

const APPROVALS_HEADER: &str = "\
# Approved with \"always allow\". Delete an entry to be asked again.
";

const PINNED_APPROVALS_HEADER: &str = "\
# Approved with \"always allow\", each for the definition with this SHA-256.
# Delete an entry to be asked again.
";

/// A set of names approved for good, saved after every approval.
#[derive(Debug)]
pub struct Approvals {
    approved: RwLock<BTreeSet<String>>,
    store: Option<PathBuf>,
    saving: tokio::sync::Mutex<()>,
}

impl Approvals {
    /// The approvals saved in `store`, and saved back there; none when it does not exist.
    /// Fails when `store` exists but cannot be read or parsed.
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

    /// Approve `name` for good and save. On [`AllowlistError::Write`] the approval still holds
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

/// A [`ConfirmationGate`] that is skipped for names in its [`Approvals`].
#[derive(Debug, Clone)]
pub struct ApprovalGate {
    gate: Arc<dyn ConfirmationGate>,
    approvals: Arc<Approvals>,
}

impl ApprovalGate {
    /// Asks `gate` about names missing from `approvals`.
    pub fn new(gate: Arc<dyn ConfirmationGate>, approvals: Arc<Approvals>) -> Self {
        Self { gate, approvals }
    }

    /// The same approvals, asking `gate` about the rest.
    pub fn asking(&self, gate: Arc<dyn ConfirmationGate>) -> Self {
        Self::new(gate, Arc::clone(&self.approvals))
    }

    /// Whether `name` was approved for good.
    pub fn contains(&self, name: &str) -> bool {
        self.approvals.contains(name)
    }

    /// Whether a use of `name` may go ahead: it was approved for good, or
    /// `turn`'s gate approves the `request` built now. An "always" answer
    /// approves `name` for good.
    pub async fn confirm(
        &self,
        turn: &TurnContext,
        name: &str,
        request: impl FnOnce() -> ConfirmationRequest,
    ) -> bool {
        if self.contains(name) {
            return true;
        }
        confirm_remembering(
            self.gate.as_ref(),
            turn,
            request(),
            self.approvals.approve(name),
        )
        .await
    }

    /// Whether the gate approves `request`, asked even about approved names;
    /// an "always" answer counts only for this use.
    pub async fn confirm_once(&self, turn: &TurnContext, request: ConfirmationRequest) -> bool {
        self.gate.confirm(turn, request).await != Approval::Deny
    }
}

#[derive(Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Stored {
    #[serde(default)]
    approved: BTreeSet<String>,
}

/// How a definition stands against the one approved for good under its name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Standing {
    Approved,
    /// The name was approved for a definition that hashed differently.
    Changed,
    Unapproved,
}

/// Names approved for good, each for the one definition whose hash is kept with it; saved
/// after every approval.
#[derive(Debug)]
pub struct PinnedApprovals {
    approved: RwLock<BTreeMap<String, DefinitionHash>>,
    store: Option<PathBuf>,
    saving: tokio::sync::Mutex<()>,
}

impl PinnedApprovals {
    /// The approvals saved in `store`, and saved back there; none when it does not exist or
    /// lists names without hashes. Fails when `store` exists but cannot be read or parsed.
    pub fn load(store: PathBuf) -> Result<Self, AllowlistError> {
        let approved = match read_store(&store)? {
            Some(text) => {
                let stored: StoredPins =
                    toml::from_str(&text).map_err(|source| AllowlistError::Parse {
                        path: store.clone(),
                        source,
                    })?;
                stored.approved.pinned(&store)
            }
            None => BTreeMap::new(),
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

    /// How `definition` stands against the one `name` was approved for.
    pub fn standing(&self, name: &str, definition: &DefinitionHash) -> Standing {
        match self.approved.read().get(name) {
            Some(approved) if approved == definition => Standing::Approved,
            Some(_) => Standing::Changed,
            None => Standing::Unapproved,
        }
    }

    /// Approve `name` for good while it hashes to `definition`, in place of any earlier
    /// approval, and save. On [`AllowlistError::Write`] the approval still holds until the
    /// daemon exits.
    pub async fn approve(
        &self,
        name: &str,
        definition: &DefinitionHash,
    ) -> Result<(), AllowlistError> {
        let _saving = self.saving.lock().await;
        let approved = {
            let mut approved = self.approved.write();
            approved.insert(name.to_string(), definition.clone());
            approved.clone()
        };
        let Some(store) = &self.store else {
            return Ok(());
        };
        let stored = StoredPins {
            approved: StoredApproved::Pinned(approved),
        };
        let body = toml::to_string(&stored).map_err(|e| AllowlistError::Write {
            path: store.clone(),
            source: std::io::Error::new(std::io::ErrorKind::InvalidData, e),
        })?;
        write_store(store, PINNED_APPROVALS_HEADER, &body).await
    }
}

/// A [`ConfirmationGate`] that is skipped for definitions its [`PinnedApprovals`] hold.
#[derive(Debug, Clone)]
pub struct PinnedApprovalGate {
    gate: Arc<dyn ConfirmationGate>,
    approvals: Arc<PinnedApprovals>,
}

impl PinnedApprovalGate {
    /// Asks `gate` about definitions missing from `approvals`.
    pub fn new(gate: Arc<dyn ConfirmationGate>, approvals: Arc<PinnedApprovals>) -> Self {
        Self { gate, approvals }
    }

    /// How `definition` stands against the one `name` was approved for.
    pub fn standing(&self, name: &str, definition: &DefinitionHash) -> Standing {
        self.approvals.standing(name, definition)
    }

    /// Whether the gate approves `request`; an "always" answer approves `name` for good
    /// while it hashes to `definition`.
    pub async fn confirm(
        &self,
        turn: &TurnContext,
        name: &str,
        definition: &DefinitionHash,
        request: ConfirmationRequest,
    ) -> bool {
        confirm_remembering(
            self.gate.as_ref(),
            turn,
            request,
            self.approvals.approve(name, definition),
        )
        .await
    }

    /// Whether the gate approves `request`; an "always" answer counts only for this use.
    pub async fn confirm_once(&self, turn: &TurnContext, request: ConfirmationRequest) -> bool {
        self.gate.confirm(turn, request).await != Approval::Deny
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct StoredPins {
    approved: StoredApproved,
}

/// The `approved` key: a table of name to hash, or the list of bare names it once was.
#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
enum StoredApproved {
    Pinned(BTreeMap<String, DefinitionHash>),
    Unpinned(BTreeSet<String>),
}

impl StoredApproved {
    /// The approvals that name a definition; bare names from `store` are dropped with a warning.
    fn pinned(self, store: &Path) -> BTreeMap<String, DefinitionHash> {
        match self {
            Self::Pinned(approved) => approved,
            Self::Unpinned(names) => {
                if !names.is_empty() {
                    warn!(
                        target: "assistd::policy",
                        store = %store.display(),
                        dropped = names.len(),
                        "approvals without a definition hash no longer apply; their tools ask again"
                    );
                }
                BTreeMap::new()
            }
        }
    }
}

/// Whether `gate` lets the use in `request` go ahead; an "always" answer
/// also runs `remember`, whose failure to save is logged.
pub(super) async fn confirm_remembering(
    gate: &dyn ConfirmationGate,
    turn: &TurnContext,
    request: ConfirmationRequest,
    remember: impl Future<Output = Result<(), AllowlistError>>,
) -> bool {
    let approval = gate.confirm(turn, request).await;
    if approval == Approval::Always
        && let Err(e) = remember.await
    {
        warn!(
            target: "assistd::policy",
            error = %e,
            "approval holds until the daemon exits but was not saved"
        );
    }
    approval != Approval::Deny
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

    #[tokio::test]
    async fn pinned_approval_holds_only_for_the_definition_approved() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = dir.path().join(APPROVED_MCP_TOOLS_FILE);
        let (first, second) = (DefinitionHash("aa".into()), DefinitionHash("bb".into()));

        std::fs::write(&store, "approved = [\"mcp__files__delete\"]\n").expect("old format");
        let approvals = PinnedApprovals::load(store.clone()).expect("old format loads");
        assert_eq!(
            approvals.standing("mcp__files__delete", &first),
            Standing::Unapproved
        );

        approvals
            .approve("mcp__files__delete", &first)
            .await
            .expect("saved");
        let reloaded = PinnedApprovals::load(store).expect("saved store loads");
        assert_eq!(
            reloaded.standing("mcp__files__delete", &first),
            Standing::Approved
        );
        assert_eq!(
            reloaded.standing("mcp__files__delete", &second),
            Standing::Changed
        );
        assert_eq!(
            reloaded.standing("mcp__files__read", &first),
            Standing::Unapproved
        );
    }
}
