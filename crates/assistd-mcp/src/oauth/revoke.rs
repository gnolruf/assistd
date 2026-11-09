//! Signing a server out: revoking its stored token at the authorization server (RFC 7009)
//! where that is offered, then deleting it.

use std::net::IpAddr;
use std::time::Duration;

use http::header::CONTENT_TYPE;
use oauth2::TokenResponse;
use reqwest::redirect;
use rmcp::transport::auth::{CredentialStore, StoredCredentials};
use serde_json::Value;
use url::{Host, Url, form_urlencoded};

use assistd_utils::http::remote_client_builder;
use assistd_utils::net::{FilteredResolver, is_non_public, is_public};

use super::{OAuthConfig, auth_failure, discover, is_web_endpoint};
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
    let revocation = match stored {
        Some(StoredCredentials {
            client_id: registered,
            token_response: Some(token),
            ..
        }) => {
            let (secret, kind) = match token.refresh_token() {
                Some(refresh) => (refresh.secret(), "refresh_token"),
                None => (token.access_token().secret(), "access_token"),
            };
            let client_id = Some(registered.as_str())
                .filter(|id| !id.is_empty())
                .or(oauth.client_id.as_deref());
            let revoking = revoke(url, oauth, client_id, secret, kind);
            tokio::time::timeout(request_timeout, revoking)
                .await
                .unwrap_or_else(|_| {
                    Revocation::Failed(format!("timed out after {}s", request_timeout.as_secs()))
                })
        }
        _ => Revocation::NothingStored,
    };
    tokens.clear().await.map_err(|e| auth_failure(&e))?;
    Ok(revocation)
}

/// Ask the authorization server of the server at `url` to revoke `token`, a token of type
/// `kind` issued to `client_id`.
async fn revoke(
    url: &Url,
    oauth: &OAuthConfig,
    client_id: Option<&str>,
    token: &str,
    kind: &str,
) -> Revocation {
    let endpoint = match revocation_endpoint(url).await {
        Ok(Some(endpoint)) => endpoint,
        Ok(None) => return Revocation::NotOffered,
        Err(reason) => return Revocation::Failed(reason),
    };
    let reachable = reachable_from(url);
    if literal_ip(&endpoint).is_some_and(|ip| !reachable(ip)) {
        return Revocation::Failed("the revocation endpoint is not a public address".into());
    }
    match post_revocation(&endpoint, reachable, oauth, client_id, token, kind).await {
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
    if is_web_endpoint(&endpoint) {
        Ok(Some(endpoint))
    } else {
        Err("the revocation endpoint is not https".into())
    }
}

/// POST the revocation form to `endpoint`, connecting only to addresses `reachable` accepts.
async fn post_revocation(
    endpoint: &Url,
    reachable: fn(IpAddr) -> bool,
    oauth: &OAuthConfig,
    client_id: Option<&str>,
    token: &str,
    kind: &str,
) -> Result<(), String> {
    let http = remote_client_builder()
        .no_proxy()
        .dns_resolver(FilteredResolver { reachable })
        .redirect(redirect::Policy::none())
        .build()
        .map_err(|e| format!("could not build an HTTP client: {}", e.without_url()))?;
    let mut request = http
        .post(endpoint.as_str())
        .header(CONTENT_TYPE, "application/x-www-form-urlencoded")
        .body(revocation_form(
            client_id,
            oauth.client_secret.is_some(),
            token,
            kind,
        ));
    if let (Some(client_id), Some(secret)) = (client_id, &oauth.client_secret) {
        request = request.basic_auth(client_id, Some(secret));
    }
    let response = request.send().await.map_err(|e| {
        if is_non_public(&e) {
            "the revocation endpoint resolves only to non-public addresses".to_string()
        } else {
            format!(
                "could not reach the revocation endpoint: {}",
                e.without_url()
            )
        }
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

/// The addresses revocation for the server at `url` may reach: any when the server is itself on
/// a non-public address or `localhost`, otherwise only public ones.
fn reachable_from(url: &Url) -> fn(IpAddr) -> bool {
    let server_is_local = match url.host() {
        Some(Host::Domain(domain)) => {
            let domain = domain.trim_end_matches('.');
            domain == "localhost" || domain.ends_with(".localhost")
        }
        _ => literal_ip(url).is_some_and(|ip| !is_public(ip)),
    };
    if server_is_local { |_| true } else { is_public }
}

fn literal_ip(url: &Url) -> Option<IpAddr> {
    match url.host()? {
        Host::Ipv4(ip) => Some(IpAddr::V4(ip)),
        Host::Ipv6(ip) => Some(IpAddr::V6(ip)),
        Host::Domain(_) => None,
    }
}

/// The form body revoking `token`; a public client names itself in it, a `confidential` one
/// authenticates with HTTP Basic instead.
fn revocation_form(client_id: Option<&str>, confidential: bool, token: &str, kind: &str) -> String {
    let mut form = form_urlencoded::Serializer::new(String::new());
    form.append_pair("token", token)
        .append_pair("token_type_hint", kind);
    if let Some(client_id) = client_id.filter(|_| !confidential) {
        form.append_pair("client_id", client_id);
    }
    form.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_a_local_server_may_revoke_at_non_public_addresses() {
        let cases: [(&str, &str, bool); 7] = [
            ("https://mcp.example.com/mcp", "93.184.216.34", true),
            ("https://mcp.example.com/mcp", "10.0.0.1", false),
            ("https://mcp.example.com/mcp", "169.254.169.254", false),
            ("https://93.184.216.34/mcp", "127.0.0.1", false),
            ("http://127.0.0.1:8080/mcp", "127.0.0.1", true),
            ("http://localhost:8080/mcp", "10.0.0.1", true),
            ("https://192.168.1.5/mcp", "192.168.1.6", true),
        ];
        for (server, endpoint, allowed) in cases {
            let reachable = reachable_from(&Url::parse(server).unwrap());
            assert_eq!(
                reachable(endpoint.parse().unwrap()),
                allowed,
                "{server} → {endpoint}"
            );
        }
    }
}
