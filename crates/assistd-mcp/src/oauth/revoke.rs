//! Signing a server out: revoking its stored token at the authorization server (RFC 7009)
//! where that is offered, then deleting it.

use std::time::Duration;

use http::header::CONTENT_TYPE;
use oauth2::TokenResponse;
use reqwest::redirect;
use rmcp::transport::auth::CredentialStore;
use serde_json::Value;
use url::{Host, Url, form_urlencoded};

use assistd_utils::http::remote_client_builder;

use super::{OAuthConfig, auth_failure, discover};
use crate::error::McpError;

/// What signing out did to the token at the authorization server.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Revocation {
    Revoked,
    /// The authorization server advertises no revocation endpoint.
    NotOffered,
    /// No token was stored for the server.
    NothingStored,
    /// Revoking failed; the token was still deleted. Never carries the token.
    Failed(String),
}

/// Revoke server `label`'s stored token where its authorization server offers revocation, then
/// delete every token stored for `label`. The token is deleted even when revoking fails; errs
/// only when the token file cannot be read or rewritten. Revoking is bounded by
/// `request_timeout`.
pub async fn sign_out(
    label: &str,
    url: &Url,
    oauth: &OAuthConfig,
    request_timeout: Duration,
) -> Result<Revocation, McpError> {
    let tokens = oauth
        .tokens
        .for_server(label, url, oauth.client_id.as_deref());
    let stored = tokens.load().await.map_err(|e| auth_failure(&e))?;
    let revocation = match stored.and_then(|credentials| credentials.token_response) {
        None => Revocation::NothingStored,
        Some(token) => {
            let (secret, kind) = match token.refresh_token() {
                Some(refresh) => (refresh.secret(), "refresh_token"),
                None => (token.access_token().secret(), "access_token"),
            };
            let revoking = revoke(url, oauth, secret, kind);
            tokio::time::timeout(request_timeout, revoking)
                .await
                .unwrap_or_else(|_| {
                    Revocation::Failed(format!("timed out after {}s", request_timeout.as_secs()))
                })
        }
    };
    tokens.clear().await.map_err(|e| auth_failure(&e))?;
    Ok(revocation)
}

/// Ask the authorization server of the server at `url` to revoke `token`, a token of type
/// `kind`.
async fn revoke(url: &Url, oauth: &OAuthConfig, token: &str, kind: &str) -> Revocation {
    let endpoint = match revocation_endpoint(url).await {
        Ok(Some(endpoint)) => endpoint,
        Ok(None) => return Revocation::NotOffered,
        Err(reason) => return Revocation::Failed(reason),
    };
    match post_revocation(&endpoint, oauth, token, kind).await {
        Ok(()) => Revocation::Revoked,
        Err(reason) => Revocation::Failed(reason),
    }
}

/// The revocation endpoint the authorization server advertises, if any. Fails when discovery
/// does, or the endpoint is not `https` (or `http` to a loopback host).
async fn revocation_endpoint(url: &Url) -> Result<Option<Url>, String> {
    let (_, metadata) = discover(url).await.map_err(|e| e.to_string())?;
    let Some(endpoint) = metadata
        .additional_fields
        .get("revocation_endpoint")
        .and_then(Value::as_str)
    else {
        return Ok(None);
    };
    let endpoint = Url::parse(endpoint).map_err(|e| format!("invalid revocation endpoint: {e}"))?;
    let loopback = match endpoint.host() {
        Some(Host::Ipv4(ip)) => ip.is_loopback(),
        Some(Host::Ipv6(ip)) => ip.is_loopback(),
        Some(Host::Domain(domain)) => domain == "localhost",
        None => false,
    };
    match endpoint.scheme() {
        "https" => Ok(Some(endpoint)),
        "http" if loopback => Ok(Some(endpoint)),
        _ => Err("the revocation endpoint is not https".into()),
    }
}

async fn post_revocation(
    endpoint: &Url,
    oauth: &OAuthConfig,
    token: &str,
    kind: &str,
) -> Result<(), String> {
    let http = remote_client_builder()
        .redirect(redirect::Policy::none())
        .build()
        .map_err(|e| format!("could not build an HTTP client: {}", e.without_url()))?;
    let mut request = http
        .post(endpoint.as_str())
        .header(CONTENT_TYPE, "application/x-www-form-urlencoded")
        .body(revocation_form(oauth, token, kind));
    if let (Some(client_id), Some(secret)) = (&oauth.client_id, &oauth.client_secret) {
        request = request.basic_auth(client_id, Some(secret));
    }
    let response = request.send().await.map_err(|e| {
        format!(
            "could not reach the revocation endpoint: {}",
            e.without_url()
        )
    })?;
    let status = response.status();
    if status.is_success() {
        Ok(())
    } else {
        Err(format!(
            "the revocation endpoint answered HTTP {}",
            status.as_u16()
        ))
    }
}

/// The form body revoking `token`; a public client names itself in it, a confidential one
/// authenticates with HTTP Basic instead.
fn revocation_form(oauth: &OAuthConfig, token: &str, kind: &str) -> String {
    let mut form = form_urlencoded::Serializer::new(String::new());
    form.append_pair("token", token)
        .append_pair("token_type_hint", kind);
    if let (Some(client_id), None) = (&oauth.client_id, &oauth.client_secret) {
        form.append_pair("client_id", client_id);
    }
    form.finish()
}
