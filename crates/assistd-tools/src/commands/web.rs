//! `web <search|fetch>`: search the web through the supervised SearXNG
//! instance, or HTTP GET a public URL.

use std::fmt::Display;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::time::Duration;

use async_trait::async_trait;
use reqwest::Url;
use reqwest::dns::{Addrs, Name, Resolve, Resolving};
use reqwest::header::CONTENT_TYPE;
use reqwest::redirect::{Attempt, Policy};
use url::Host;

use assistd_search::SearxngClient;
use assistd_utils::http::{BodyError, read_capped};

use crate::command::{Command, CommandInput, CommandOutput, Hint, error_line};
use crate::exec::POLICY_DENIED_EXIT;
use crate::policy::{ApprovalGate, ConfirmationRequest, Risk, on_blocking_pool};

mod search;

/// Hard cap on response bytes.
pub const BODY_MAX: usize = 10 * 1024 * 1024;

/// Redirects followed before a fetch fails.
const MAX_REDIRECTS: usize = 10;

const NAME: &str = "web";

/// Columns HTML pages are wrapped to once rendered as text.
const TEXT_WIDTH: usize = 100;

const UNREACHABLE: &str = "a different URL or check the endpoint is reachable";

const FETCH_USAGE: &str = "web fetch <URL>";

const FETCH_HELP: &str = "usage: web fetch <URL>\n\
    \n\
    HTTP GET the URL (http or https only) and return the response body \
    as stdout. 30-second timeout; response body capped at 10 MiB. The \
    user is asked before fetching from a host they have not approved, \
    and redirects are followed only to the same or an approved host. \
    Loopback, private, link-local, and other non-public addresses are \
    never fetched.\n\
    \n\
    HTML pages come back as plain text: headings marked with #, tables \
    as aligned columns, and each link as [n] with its URL listed at the \
    end. Other content types come back as sent.\n\
    \n\
    Non-2xx statuses exit 1 so `||` fallbacks fire; transport errors, \
    including a host resolving only to non-public addresses, also exit 1. \
    Exit 2 on usage errors (wrong arg count, non-http scheme), 126 when \
    the URL names a non-public address or the user declines.\n";

/// IPv4 ranges that are not globally routable unicast, as (network, prefix length).
const NON_PUBLIC_V4: [(Ipv4Addr, u32); 13] = [
    (Ipv4Addr::new(0, 0, 0, 0), 8),
    (Ipv4Addr::new(10, 0, 0, 0), 8),
    (Ipv4Addr::new(100, 64, 0, 0), 10),
    (Ipv4Addr::new(127, 0, 0, 0), 8),
    (Ipv4Addr::new(169, 254, 0, 0), 16),
    (Ipv4Addr::new(172, 16, 0, 0), 12),
    (Ipv4Addr::new(192, 0, 0, 0), 24),
    (Ipv4Addr::new(192, 0, 2, 0), 24),
    (Ipv4Addr::new(192, 168, 0, 0), 16),
    (Ipv4Addr::new(198, 18, 0, 0), 15),
    (Ipv4Addr::new(198, 51, 100, 0), 24),
    (Ipv4Addr::new(203, 0, 113, 0), 24),
    (Ipv4Addr::new(224, 0, 0, 0), 3),
];

const GLOBAL_UNICAST_V6: (Ipv6Addr, u32) = (Ipv6Addr::new(0x2000, 0, 0, 0, 0, 0, 0, 0), 3);

const DOCUMENTATION_V6: (Ipv6Addr, u32) = (Ipv6Addr::new(0x2001, 0xdb8, 0, 0, 0, 0, 0, 0), 32);

/// Teredo tunnels reach an obfuscated IPv4 client behind a relay, so the real target is unknowable.
const TEREDO_V6: (Ipv6Addr, u32) = (Ipv6Addr::new(0x2001, 0, 0, 0, 0, 0, 0, 0), 32);

/// 6to4 addresses carry the IPv4 endpoint in bits 16..48.
const SIX_TO_FOUR_V6: (Ipv6Addr, u32) = (Ipv6Addr::new(0x2002, 0, 0, 0, 0, 0, 0, 0), 16);

/// `web <search|fetch>`: search through SearXNG, or HTTP GET a URL and
/// return the body as stdout. Hosts not yet approved for good are fetched
/// only once the user confirms, and only public addresses are connected to.
#[derive(Debug)]
pub struct WebCommand {
    client: reqwest::Client,
    hosts: ApprovalGate,
    reachable: fn(IpAddr) -> bool,
    search: Option<SearxngClient>,
}

impl WebCommand {
    /// A command with a 30-second fetch timeout that asks before fetching
    /// from a host `hosts` has not approved; `search` is `None` when web
    /// search is disabled.
    pub fn new(hosts: ApprovalGate, search: Option<SearxngClient>) -> Self {
        Self {
            search,
            ..Self::connecting_to(is_public, hosts, Duration::from_secs(30))
        }
    }

    fn connecting_to(
        reachable: fn(IpAddr) -> bool,
        hosts: ApprovalGate,
        timeout: Duration,
    ) -> Self {
        let redirect_hosts = hosts.clone();
        let client = reqwest::Client::builder()
            .no_proxy()
            .dns_resolver(FilteredResolver { reachable })
            .connect_timeout(Duration::from_secs(10))
            .timeout(timeout)
            .redirect(Policy::custom(move |attempt| {
                follow_redirect(attempt, &redirect_hosts, reachable)
            }))
            .build()
            .expect("reqwest client builds with valid config");
        Self {
            client,
            hosts,
            reachable,
            search: None,
        }
    }

    async fn confirmed(&self, url: &Url) -> bool {
        let host = url.host_str().unwrap_or_default();
        self.hosts
            .confirm(host, || ConfirmationRequest {
                tool: NAME.to_string(),
                script: format!("web fetch {url}"),
                matched_pattern: format!("fetches from {host}, which is not yet approved"),
                always_allow: vec![host.to_string()],
                risk: Risk::Elevated,
            })
            .await
    }

    async fn fetch_command(&self, args: &[String]) -> CommandOutput {
        let raw = match args {
            [] => return CommandOutput::usage(FETCH_HELP.to_string()),
            [raw] => raw,
            _ => {
                return CommandOutput::usage_error(
                    NAME,
                    "fetch expects exactly one URL argument",
                    FETCH_USAGE,
                );
            }
        };
        let Some(url) = web_url(raw) else {
            return CommandOutput::usage_error(
                NAME,
                format_args!("only http(s):// URLs are allowed: {raw}"),
                "web fetch https://... or web fetch http://...",
            );
        };
        if let Some(refusal) = unreachable_literal(&url, self.reachable) {
            return CommandOutput::failed(
                POLICY_DENIED_EXIT,
                error_line(
                    NAME,
                    format_args!("fetch denied by policy: {refusal}"),
                    Hint::Try,
                    "a URL on the public internet",
                )
                .into_bytes(),
            );
        }
        if !self.confirmed(&url).await {
            return CommandOutput::failed(
                POLICY_DENIED_EXIT,
                error_line(
                    NAME,
                    format_args!("fetch cancelled by user: {url}"),
                    Hint::Try,
                    "answering without this page",
                )
                .into_bytes(),
            );
        }
        self.fetch(url).await
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
        let is_html = is_html(&response);
        match read_capped(response, BODY_MAX).await {
            Ok(body) if is_html => CommandOutput::ok(on_blocking_pool(|| html_as_text(body)).await),
            Ok(body) => CommandOutput::ok(body),
            Err(BodyError::TooLarge { .. }) => fetch_failed(
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
        NAME
    }

    fn summary(&self) -> &'static str {
        "search the web, or HTTP GET a URL and return the body (http/https only)"
    }

    fn help(&self) -> String {
        "usage: web <search|fetch> [args...]\n\
         \n\
         Subcommands:\n  \
           search [-n N] [--lang <tag>] <query...>\n          \
         search the web; TSV: rank, title, url, snippet\n  \
           fetch <URL>\n          \
         HTTP GET a public http(s) URL and return the body as stdout\n\
         \n\
         When you do not have a URL, search first, then `web fetch` the most \
         relevant result. Prefer `-n 5` for simple questions.\n\
         \n\
         Call either subcommand with no arguments to see its parameter help.\n"
            .to_string()
    }

    /// URLs and search queries never name files, so `?` and `*` stay literal.
    fn expands_args(&self) -> bool {
        false
    }

    async fn run(&self, input: CommandInput) -> CommandOutput {
        let Some((subcommand, rest)) = input.args.split_first() else {
            return CommandOutput::usage(self.help());
        };
        match subcommand.as_str() {
            "search" => search::run(self.search.as_ref(), rest).await,
            "fetch" => self.fetch_command(rest).await,
            bare if rest.is_empty() && web_url(bare).is_some() => {
                self.fetch_command(&input.args).await
            }
            other => CommandOutput::usage_error(
                NAME,
                format_args!("unknown subcommand '{other}'"),
                "web search <words...> to search, or web fetch <URL> to read a page",
            ),
        }
    }
}

/// Resolves names with the system resolver, keeping only addresses `reachable` accepts.
#[derive(Debug)]
struct FilteredResolver {
    reachable: fn(IpAddr) -> bool,
}

impl Resolve for FilteredResolver {
    fn resolve(&self, name: Name) -> Resolving {
        let reachable = self.reachable;
        Box::pin(async move {
            let host = name.as_str();
            let addrs: Vec<SocketAddr> = tokio::net::lookup_host((host, 0))
                .await?
                .filter(|addr| reachable(addr.ip()))
                .collect();
            if addrs.is_empty() {
                return Err(NonPublic::Name(host.to_string()).into());
            }
            Ok(Box::new(addrs.into_iter()) as Addrs)
        })
    }
}

/// A destination was refused because it is not on the public internet.
#[derive(Debug, thiserror::Error)]
enum NonPublic {
    #[error("{0} is not a public address")]
    Address(IpAddr),
    #[error("{0} resolves only to non-public addresses")]
    Name(String),
}

/// A redirect was refused because it leaves for an unapproved host.
#[derive(Debug, thiserror::Error)]
#[error("redirect to unapproved host {0}; fetch it directly to be asked")]
struct UnapprovedRedirect(String);

fn follow_redirect(
    attempt: Attempt<'_>,
    hosts: &ApprovalGate,
    reachable: fn(IpAddr) -> bool,
) -> reqwest::redirect::Action {
    if attempt.previous().len() > MAX_REDIRECTS {
        return attempt.error("too many redirects");
    }
    if let Some(refusal) = unreachable_literal(attempt.url(), reachable) {
        return attempt.error(refusal);
    }
    let target = attempt.url().host_str().unwrap_or_default().to_string();
    let original = attempt.previous().first().and_then(Url::host_str);
    if original == Some(target.as_str()) || hosts.contains(&target) {
        attempt.follow()
    } else {
        attempt.error(UnapprovedRedirect(target))
    }
}

fn is_html(response: &reqwest::Response) -> bool {
    response
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.split(';').next())
        .is_some_and(|mime| {
            let mime = mime.trim();
            mime.eq_ignore_ascii_case("text/html")
                || mime.eq_ignore_ascii_case("application/xhtml+xml")
        })
}

fn html_as_text(html: Vec<u8>) -> Vec<u8> {
    html2text::config::plain()
        .allow_width_overflow()
        .no_link_wrapping()
        .no_table_borders()
        .string_from_read(html.as_slice(), TEXT_WIDTH)
        .map_or(html, String::into_bytes)
}

fn web_url(raw: &str) -> Option<Url> {
    Url::parse(raw)
        .ok()
        .filter(|url| matches!(url.scheme(), "http" | "https") && url.has_host())
}

fn unreachable_literal(url: &Url, reachable: fn(IpAddr) -> bool) -> Option<NonPublic> {
    let ip = match url.host()? {
        Host::Ipv4(ip) => IpAddr::V4(ip),
        Host::Ipv6(ip) => IpAddr::V6(ip),
        Host::Domain(_) => return None,
    };
    (!reachable(ip)).then_some(NonPublic::Address(ip))
}

fn is_public(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => is_public_v4(v4),
        IpAddr::V6(v6) => embedded_v4(v6).map_or_else(|| is_public_v6(v6), is_public_v4),
    }
}

fn is_public_v4(ip: Ipv4Addr) -> bool {
    !NON_PUBLIC_V4.iter().any(|&range| in_v4_prefix(ip, range))
}

fn in_v4_prefix(ip: Ipv4Addr, (network, prefix_len): (Ipv4Addr, u32)) -> bool {
    let mask = u32::MAX << (32 - prefix_len);
    u32::from(ip) & mask == u32::from(network)
}

fn embedded_v4(ip: Ipv6Addr) -> Option<Ipv4Addr> {
    ip.to_ipv4_mapped().or_else(|| {
        let [_, _, a, b, c, d, ..] = ip.octets();
        in_v6_prefix(ip, SIX_TO_FOUR_V6).then(|| Ipv4Addr::new(a, b, c, d))
    })
}

fn is_public_v6(ip: Ipv6Addr) -> bool {
    in_v6_prefix(ip, GLOBAL_UNICAST_V6)
        && !in_v6_prefix(ip, DOCUMENTATION_V6)
        && !in_v6_prefix(ip, TEREDO_V6)
}

fn in_v6_prefix(ip: Ipv6Addr, (network, prefix_len): (Ipv6Addr, u32)) -> bool {
    let mask = u128::MAX << (128 - prefix_len);
    u128::from(ip) & mask == u128::from(network)
}

fn error_chain(err: &dyn std::error::Error) -> String {
    std::iter::successors(Some(err), |e| e.source())
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(": ")
}

fn fetch_failed(what: impl Display, recovery: &str) -> CommandOutput {
    CommandOutput::failed(1, error_line(NAME, what, Hint::Try, recovery).into_bytes())
}

#[cfg(test)]
mod tests;
