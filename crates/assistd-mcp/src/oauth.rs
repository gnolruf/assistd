//! OAuth 2.1 for Streamable HTTP servers: discovering the authorization server, signing in and
//! out, picking up the stored token, and classifying OAuth failures without their secrets.

use std::fmt;

use rmcp::transport::auth::{AuthError, AuthorizationManager, OAuthClientConfig};
use url::Url;

use crate::error::McpError;

mod login;
mod revoke;
mod store;

pub use login::{LOGIN_TIMEOUT, PendingLogin, begin_login};
pub use revoke::{Revocation, sign_out};
pub use store::TokenStore;

/// The OAuth client one server is authorized as, and where its tokens live. `Debug` never
/// shows the secret.
#[derive(Clone)]
pub struct OAuthConfig {
    /// A pre-registered client ID; `None` registers one dynamically.
    pub client_id: Option<String>,
    pub client_secret: Option<String>,
    pub scopes: Vec<String>,
    pub tokens: TokenStore,
}

impl fmt::Debug for OAuthConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OAuthConfig")
            .field("client_id", &self.client_id)
            .field(
                "client_secret",
                &self.client_secret.as_ref().map(|_| "[REDACTED]"),
            )
            .field("scopes", &self.scopes)
            .field("tokens", &self.tokens)
            .finish()
    }
}

/// An authorization manager for server `label` at `url`, ready to attach and refresh its
/// stored token. Fails with [`McpError::NeedsAuthorization`] when no token is stored, and
/// with [`McpError::RegistrationUnsupported`] when the client must be registered
/// dynamically but the authorization server does not offer it.
pub(crate) async fn authorization_manager(
    label: &str,
    url: &Url,
    oauth: &OAuthConfig,
) -> Result<AuthorizationManager, McpError> {
    let mut manager = discovered_manager(url, oauth).await?;
    manager.set_credential_store(
        oauth
            .tokens
            .for_server(label, url, oauth.client_id.as_deref()),
    );
    if !manager
        .initialize_from_store()
        .await
        .map_err(|e| auth_failure(&e))?
    {
        return Err(McpError::NeedsAuthorization { scopes: Vec::new() });
    }
    let (client_id, _) = manager
        .get_credentials()
        .await
        .map_err(|e| auth_failure(&e))?;
    let mut client =
        OAuthClientConfig::new(client_id, url.as_str()).with_scopes(oauth.scopes.clone());
    if let Some(secret) = &oauth.client_secret {
        client = client.with_client_secret(secret.clone());
    }
    manager
        .configure_client(client)
        .map_err(|e| auth_failure(&e))?;
    Ok(manager)
}

/// An authorization manager for the server at `url` with its authorization server's metadata
/// resolved. Fails with [`McpError::RegistrationUnsupported`] when the client must be
/// registered dynamically but the authorization server does not offer it.
async fn discovered_manager(
    url: &Url,
    oauth: &OAuthConfig,
) -> Result<AuthorizationManager, McpError> {
    let mut manager = AuthorizationManager::new(url.as_str())
        .await
        .map_err(|e| auth_failure(&e))?;
    let metadata = manager
        .resolve_metadata()
        .await
        .map_err(|e| auth_failure(&e))?
        .metadata;
    if oauth.client_id.is_none() && metadata.registration_endpoint.is_none() {
        return Err(McpError::RegistrationUnsupported);
    }
    manager.set_metadata(metadata);
    Ok(manager)
}

/// The [`McpError`] for an OAuth failure. Only failures that never involve a token, code or
/// secret keep their detail.
pub(crate) fn auth_failure(error: &AuthError) -> McpError {
    match error {
        AuthError::AuthorizationRequired
        | AuthError::TokenRefreshRejected(_)
        | AuthError::TokenExpired => McpError::NeedsAuthorization { scopes: Vec::new() },
        AuthError::InsufficientScope { required_scope, .. } => McpError::NeedsAuthorization {
            scopes: split_scopes(required_scope),
        },
        AuthError::TokenRefreshFailed(_) => {
            McpError::OAuth("could not refresh the access token".into())
        }
        AuthError::HttpError(_) => {
            McpError::OAuth("could not reach the authorization server".into())
        }
        AuthError::CredentialStoreError(detail) => {
            McpError::OAuth(format!("token store: {detail}"))
        }
        AuthError::RegistrationFailed(detail) => {
            McpError::OAuth(format!("client registration failed: {detail}"))
        }
        AuthError::TokenExchangeFailed(_) => {
            McpError::OAuth("the authorization server would not issue a token for the code".into())
        }
        error @ (AuthError::MetadataError(_)
        | AuthError::NoAuthorizationSupport
        | AuthError::PkceUnsupported
        | AuthError::UrlError(_)
        | AuthError::OAuthError(_)
        | AuthError::InternalError(_)
        | AuthError::AuthorizationServerMismatch { .. }
        | AuthError::AuthorizationServerMissingIssuer { .. }) => McpError::OAuth(error.to_string()),
        _ => McpError::OAuth("the authorization server refused the request".into()),
    }
}

/// The scopes in a space-separated `scope` value.
pub(crate) fn split_scopes(scope: &str) -> Vec<String> {
    scope.split_whitespace().map(str::to_owned).collect()
}
