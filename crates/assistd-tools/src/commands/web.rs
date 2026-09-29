use std::fmt::Display;
use std::time::Duration;

use async_trait::async_trait;
use reqwest::redirect::{Attempt, Policy};
use reqwest::{Response, Url};

use crate::command::{Command, CommandInput, CommandOutput, Hint, error_line};
use crate::exec::POLICY_DENIED_EXIT;
use crate::policy::{ApprovalGate, ConfirmationRequest};

/// Hard cap on response bytes.
pub const BODY_MAX: usize = 10 * 1024 * 1024;

/// Redirects followed before a fetch fails.
const MAX_REDIRECTS: usize = 10;

const UNREACHABLE: &str = "a different URL or check the endpoint is reachable";

/// `web URL`: HTTP GET a URL and return the response body as stdout. Hosts
/// not yet approved for good are fetched only once the user confirms.
#[derive(Debug)]
pub struct WebCommand {
    client: reqwest::Client,
    hosts: ApprovalGate,
}

impl WebCommand {
    /// A command with a 30-second request timeout that asks before fetching
    /// from a host `hosts` has not approved.
    pub fn new(hosts: ApprovalGate) -> Self {
        Self::with_timeout(hosts, Duration::from_secs(30))
    }

    /// [`WebCommand::new`] with requests timing out after `timeout`, and
    /// connecting capped at 10 seconds.
    pub fn with_timeout(hosts: ApprovalGate, timeout: Duration) -> Self {
        let redirect_hosts = hosts.clone();
        let client = reqwest::Client::builder()
            .no_proxy()
            .connect_timeout(Duration::from_secs(10))
            .timeout(timeout)
            .redirect(Policy::custom(move |attempt| {
                follow_redirect(attempt, &redirect_hosts)
            }))
            .build()
            .expect("reqwest client builds with valid config");
        Self { client, hosts }
    }

    async fn confirmed(&self, url: &Url) -> bool {
        let host = url.host_str().unwrap_or_default();
        self.hosts
            .confirm(host, || ConfirmationRequest {
                tool: "web".to_string(),
                script: format!("web {url}"),
                matched_pattern: format!("fetches from {host}, which is not yet approved"),
                always_allow: vec![host.to_string()],
            })
            .await
    }

    async fn fetch(&self, url: Url) -> CommandOutput {
        let response = match self.client.get(url.clone()).send().await {
            Ok(r) => r,
            Err(e) => {
                return fetch_failed(
                    format_args!("transport error: {url}: {}", error_chain(&e)),
                    UNREACHABLE,
                );
            }
        };
        let status = response.status();
        if !status.is_success() {
            return fetch_failed(
                format_args!(
                    "HTTP {} {}: {url}",
                    status.as_u16(),
                    status.canonical_reason().unwrap_or("")
                ),
                UNREACHABLE,
            );
        }
        match read_capped(response).await {
            Ok(body) => CommandOutput::ok(body),
            Err(BodyError::TooLarge) => fetch_failed(
                format_args!("response body exceeded {BODY_MAX} bytes: {url}"),
                "a URL path that returns less content",
            ),
            Err(BodyError::Read(e)) => fetch_failed(
                format_args!("body read failed: {url}: {e}"),
                "re-running or a different URL",
            ),
        }
    }
}

#[async_trait]
impl Command for WebCommand {
    fn name(&self) -> &'static str {
        "web"
    }

    fn summary(&self) -> &'static str {
        "HTTP GET a URL and return the body as stdout (http/https only)"
    }

    fn help(&self) -> String {
        "usage: web URL\n\
         \n\
         HTTP GET the URL (http or https only) and return the response body \
         as stdout. 30-second timeout; response body capped at 10 MiB. The \
         user is asked before fetching from a host they have not approved, \
         and redirects are followed only to the same or an approved host.\n\
         \n\
         Non-2xx statuses exit 1 so `||` fallbacks fire; transport errors \
         also exit 1. Exit 2 on usage errors (wrong arg count, non-http scheme), \
         126 when the user declines.\n"
            .to_string()
    }

    async fn run(&self, input: CommandInput) -> CommandOutput {
        if input.args.is_empty() {
            return CommandOutput::usage(self.help());
        }
        if input.args.len() != 1 {
            return CommandOutput::usage_error(
                "web",
                "expects exactly one URL argument",
                "web <URL>",
            );
        }
        let raw = &input.args[0];
        let url = match Url::parse(raw) {
            Ok(url) if matches!(url.scheme(), "http" | "https") && url.has_host() => url,
            _ => {
                return CommandOutput::usage_error(
                    "web",
                    format_args!("only http(s):// URLs are allowed: {raw}"),
                    "web https://... or web http://...",
                );
            }
        };
        if !self.confirmed(&url).await {
            return CommandOutput::failed(
                POLICY_DENIED_EXIT,
                error_line(
                    "web",
                    format_args!("fetch cancelled by user: {url}"),
                    Hint::Try,
                    "answering without this page",
                )
                .into_bytes(),
            );
        }
        self.fetch(url).await
    }
}

/// Why a response body was not read in full.
#[derive(Debug)]
enum BodyError {
    TooLarge,
    Read(reqwest::Error),
}

/// A redirect was refused because it leaves for an unapproved host.
#[derive(Debug, thiserror::Error)]
#[error("redirect to unapproved host {0}; fetch it directly to be asked")]
struct UnapprovedRedirect(String);

fn follow_redirect(attempt: Attempt<'_>, hosts: &ApprovalGate) -> reqwest::redirect::Action {
    if attempt.previous().len() > MAX_REDIRECTS {
        return attempt.error("too many redirects");
    }
    let target = attempt.url().host_str().unwrap_or_default().to_string();
    let original = attempt.previous().first().and_then(Url::host_str);
    if original == Some(target.as_str()) || hosts.contains(&target) {
        attempt.follow()
    } else {
        attempt.error(UnapprovedRedirect(target))
    }
}

async fn read_capped(mut response: Response) -> Result<Vec<u8>, BodyError> {
    if response
        .content_length()
        .is_some_and(|len| usize::try_from(len).map_or(true, |len| len > BODY_MAX))
    {
        return Err(BodyError::TooLarge);
    }
    let mut body = Vec::new();
    while let Some(chunk) = response.chunk().await.map_err(BodyError::Read)? {
        if body.len() + chunk.len() > BODY_MAX {
            return Err(BodyError::TooLarge);
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

fn error_chain(err: &dyn std::error::Error) -> String {
    std::iter::successors(Some(err), |e| e.source())
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(": ")
}

fn fetch_failed(what: impl Display, recovery: &str) -> CommandOutput {
    CommandOutput::failed(1, error_line("web", what, Hint::Try, recovery).into_bytes())
}

#[cfg(test)]
mod tests;
