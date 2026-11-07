//! The token file: every OAuth server's tokens in one owner-only JSON file, replaced
//! atomically on each save.

use std::fs::DirBuilder;
use std::io::ErrorKind;
use std::os::unix::fs::DirBuilderExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use rmcp::transport::auth::{AuthError, CredentialStore, StoredCredentials};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use url::Url;

use assistd_utils::fs::replace_private_file;

/// Mode of the directory the token file is created in.
const TOKEN_DIR_MODE: u32 = 0o700;

/// OAuth tokens for every server, kept in one JSON file shared by all of them.
#[derive(Debug, Clone)]
pub struct TokenStore {
    file: Arc<TokenFile>,
}

#[derive(Debug)]
struct TokenFile {
    path: PathBuf,
    writes: Mutex<()>,
}

/// What a stored token was issued for. A token is only loaded for a server whose name, URL
/// and configured client ID all match, so changing any of them never sends it elsewhere.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct TokenKey {
    pub(crate) server: String,
    pub(crate) url: String,
    /// The configured client ID; `None` for a dynamically registered client.
    pub(crate) client_id: Option<String>,
}

/// One server's slot in the [`TokenStore`].
pub(crate) struct ServerTokens {
    file: Arc<TokenFile>,
    key: TokenKey,
}

#[derive(Default, Serialize, Deserialize)]
struct TokenFileContents {
    entries: Vec<TokenEntry>,
}

#[derive(Serialize, Deserialize)]
struct TokenEntry {
    #[serde(flatten)]
    key: TokenKey,
    credentials: StoredCredentials,
}

impl TokenStore {
    /// A store kept at `path`, created mode `0600` in a mode-`0700` directory on first save.
    pub fn new(path: PathBuf) -> Self {
        Self {
            file: Arc::new(TokenFile {
                path,
                writes: Mutex::new(()),
            }),
        }
    }

    pub(crate) fn for_server(
        &self,
        server: &str,
        url: &Url,
        client_id: Option<&str>,
    ) -> ServerTokens {
        ServerTokens {
            file: self.file.clone(),
            key: TokenKey {
                server: server.to_owned(),
                url: url.as_str().to_owned(),
                client_id: client_id.map(str::to_owned),
            },
        }
    }
}

impl TokenFile {
    async fn read(&self) -> Result<TokenFileContents, AuthError> {
        let path = self.path.clone();
        run_blocking(move || read_contents(&path)).await
    }

    /// Rewrite the file with `edit` applied, one writer at a time.
    async fn update(
        &self,
        edit: impl FnOnce(&mut Vec<TokenEntry>) + Send + 'static,
    ) -> Result<(), AuthError> {
        let _writing = self.writes.lock().await;
        let path = self.path.clone();
        run_blocking(move || {
            let mut contents = read_contents(&path)?;
            edit(&mut contents.entries);
            write_contents(&path, &contents)
        })
        .await
    }
}

#[async_trait]
impl CredentialStore for ServerTokens {
    async fn load(&self) -> Result<Option<StoredCredentials>, AuthError> {
        let contents = self.file.read().await?;
        Ok(contents
            .entries
            .into_iter()
            .find(|entry| entry.key == self.key)
            .map(|entry| entry.credentials))
    }

    /// Replace every entry for this server, including ones for an old URL or client ID.
    async fn save(&self, credentials: StoredCredentials) -> Result<(), AuthError> {
        let key = self.key.clone();
        self.file
            .update(move |entries| {
                entries.retain(|entry| entry.key.server != key.server);
                entries.push(TokenEntry { key, credentials });
            })
            .await
    }

    async fn clear(&self) -> Result<(), AuthError> {
        let server = self.key.server.clone();
        self.file
            .update(move |entries| entries.retain(|entry| entry.key.server != server))
            .await
    }
}

async fn run_blocking<T: Send + 'static>(
    work: impl FnOnce() -> Result<T, AuthError> + Send + 'static,
) -> Result<T, AuthError> {
    tokio::task::spawn_blocking(work)
        .await
        .map_err(|e| store_error(format!("token file task failed: {e}")))?
}

/// The file's entries, or none when it does not exist yet. A parse failure reports only
/// where it happened, since serde's message can quote a token.
fn read_contents(path: &Path) -> Result<TokenFileContents, AuthError> {
    let text = match std::fs::read(path) {
        Ok(text) => text,
        Err(e) if e.kind() == ErrorKind::NotFound => return Ok(TokenFileContents::default()),
        Err(e) => return Err(store_error(format!("reading {}: {e}", path.display()))),
    };
    serde_json::from_slice(&text).map_err(|e| {
        store_error(format!(
            "{} is not a valid token file (line {}, column {})",
            path.display(),
            e.line(),
            e.column()
        ))
    })
}

fn write_contents(path: &Path, contents: &TokenFileContents) -> Result<(), AuthError> {
    let write_error = |e: std::io::Error| store_error(format!("writing {}: {e}", path.display()));
    if let Some(parent) = path.parent() {
        DirBuilder::new()
            .recursive(true)
            .mode(TOKEN_DIR_MODE)
            .create(parent)
            .map_err(write_error)?;
    }
    let json = serde_json::to_vec_pretty(contents)
        .map_err(|e| store_error(format!("encoding the token file: {e}")))?;
    replace_private_file(path, &json).map_err(write_error)
}

fn store_error(detail: String) -> AuthError {
    AuthError::CredentialStoreError(detail)
}
