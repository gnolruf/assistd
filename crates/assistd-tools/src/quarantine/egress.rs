//! The proxy a relay's commands reach the web through: HTTP `CONNECT`
//! tunnels over a Unix socket, each held to the relay's [`EgressPolicy`].

use std::error::Error;
use std::io;
use std::net::IpAddr;
use std::path::Path;
use std::time::Duration;

use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpStream, UnixListener, UnixStream};
use tokio::task::JoinSet;
use tokio_util::task::AbortOnDropHandle;
use tracing::{debug, warn};
use url::Host;

use assistd_ipc::ConfirmSource;

use crate::policy::{ConfirmationRequest, EgressPolicy, NonPublic, Risk};

/// Bytes of request head read before a request is refused as malformed.
const MAX_HEAD: u64 = 8 * 1024;

/// How long a client has to send its request head.
const HEAD_TIMEOUT: Duration = Duration::from_secs(10);

/// How long connecting to the destination may take.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

const ESTABLISHED: &[u8] = b"HTTP/1.1 200 Connection established\r\n\r\n";

/// A proxy serving one relay. Dropping it stops the listener and closes
/// every tunnel through it.
#[derive(Debug)]
pub(super) struct EgressProxy {
    _serving: AbortOnDropHandle<()>,
}

impl EgressProxy {
    /// A proxy listening on a new socket at `socket`, holding every tunnel
    /// to `policy`.
    pub(super) fn bind(socket: &Path, policy: EgressPolicy) -> io::Result<Self> {
        let listener = UnixListener::bind(socket)?;
        Ok(Self {
            _serving: AbortOnDropHandle::new(tokio::spawn(serve(listener, policy))),
        })
    }
}

/// Why a tunnel was not opened, answered to the client as an HTTP status.
#[derive(Debug, thiserror::Error)]
enum Refusal {
    #[error("malformed proxy request")]
    Malformed,
    #[error("only CONNECT tunnels are proxied; use an https:// URL")]
    NotConnect,
    #[error("port {0} is not one a sub-agent may connect to; only HTTPS on 443 is proxied")]
    Port(u16),
    #[error(transparent)]
    NonPublic(NonPublic),
    #[error("the user did not approve {0}")]
    Declined(String),
    #[error("cannot resolve {host}: {source}")]
    Resolve {
        host: String,
        #[source]
        source: io::Error,
    },
    #[error("cannot connect to {host}: {source}")]
    Connect {
        host: String,
        #[source]
        source: io::Error,
    },
}

impl Refusal {
    fn status_line(&self) -> &'static str {
        match self {
            Self::Malformed => "400 Bad Request",
            Self::NotConnect => "405 Method Not Allowed",
            Self::Port(_) | Self::NonPublic(_) | Self::Declined(_) => "403 Forbidden",
            Self::Resolve { source, .. } if is_non_public(source) => "403 Forbidden",
            Self::Resolve { .. } | Self::Connect { .. } => "502 Bad Gateway",
        }
    }

    fn response(&self) -> String {
        let body = format!("{self}\n");
        format!(
            "HTTP/1.1 {}\r\nContent-Type: text/plain\r\nContent-Length: {}\r\n\
             Connection: close\r\n\r\n{body}",
            self.status_line(),
            body.len()
        )
    }
}

/// A `CONNECT` request's destination.
#[derive(Debug)]
struct Target {
    host: Host,
    port: u16,
}

impl Target {
    /// The destination of `request_line`, which must be
    /// `CONNECT host:port HTTP/1.x`.
    fn parse(request_line: &str) -> Result<Self, Refusal> {
        let mut parts = request_line.split_ascii_whitespace();
        let (Some(method), Some(authority), Some(version), None) =
            (parts.next(), parts.next(), parts.next(), parts.next())
        else {
            return Err(Refusal::Malformed);
        };
        if !version.starts_with("HTTP/1.") {
            return Err(Refusal::Malformed);
        }
        if method != "CONNECT" {
            return Err(Refusal::NotConnect);
        }
        let (host, port) = authority.rsplit_once(':').ok_or(Refusal::Malformed)?;
        Ok(Self {
            host: Host::parse(host).map_err(|_| Refusal::Malformed)?,
            port: port.parse().map_err(|_| Refusal::Malformed)?,
        })
    }

    fn literal_ip(&self) -> Option<IpAddr> {
        match self.host {
            Host::Ipv4(ip) => Some(IpAddr::V4(ip)),
            Host::Ipv6(ip) => Some(IpAddr::V6(ip)),
            Host::Domain(_) => None,
        }
    }
}

async fn serve(listener: UnixListener, policy: EgressPolicy) {
    let mut tunnels = JoinSet::new();
    loop {
        tokio::select! {
            accepted = listener.accept() => match accepted {
                Ok((client, _)) => {
                    tunnels.spawn(tunnel(client, policy.clone()));
                }
                Err(error) => {
                    warn!(target: "assistd::policy", %error, "egress proxy stopped accepting");
                    return;
                }
            },
            Some(_) = tunnels.join_next() => {}
        }
    }
}

/// Open the tunnel `client` asks for and relay bytes both ways until either
/// side closes, or answer why not.
async fn tunnel(client: UnixStream, policy: EgressPolicy) {
    let mut client = BufReader::new(client);
    let mut upstream = match open_upstream(&mut client, &policy).await {
        Ok(upstream) => upstream,
        Err(refusal) => {
            debug!(target: "assistd::policy", %refusal, "egress proxy refused a tunnel");
            let _ = client.write_all(refusal.response().as_bytes()).await;
            return;
        }
    };
    if client.write_all(ESTABLISHED).await.is_ok() {
        let _ = tokio::io::copy_bidirectional(&mut client, &mut upstream).await;
    }
}

async fn open_upstream(
    client: &mut BufReader<UnixStream>,
    policy: &EgressPolicy,
) -> Result<TcpStream, Refusal> {
    let request_line = tokio::time::timeout(HEAD_TIMEOUT, read_head(client))
        .await
        .map_err(|_| Refusal::Malformed)??;
    let target = Target::parse(&request_line)?;
    admit(&target, policy).await?;
    connect(&target, policy).await
}

/// The request line of the head `client` sends, after reading the rest of
/// the head up to its blank line.
async fn read_head(client: &mut BufReader<UnixStream>) -> Result<String, Refusal> {
    let mut head = client.take(MAX_HEAD);
    let mut request_line = String::new();
    head.read_line(&mut request_line)
        .await
        .map_err(|_| Refusal::Malformed)?;
    let mut line = Vec::new();
    loop {
        line.clear();
        let read = head
            .read_until(b'\n', &mut line)
            .await
            .map_err(|_| Refusal::Malformed)?;
        match line.as_slice() {
            _ if read == 0 || !line.ends_with(b"\n") => return Err(Refusal::Malformed),
            b"\r\n" | b"\n" => return Ok(request_line),
            _ => {}
        }
    }
}

/// Refuse `target` unless `policy` allows its port, its address if it is
/// one, and the user approves its host.
async fn admit(target: &Target, policy: &EgressPolicy) -> Result<(), Refusal> {
    if !policy.allows_port(target.port) {
        return Err(Refusal::Port(target.port));
    }
    if let Some(ip) = target.literal_ip().filter(|&ip| !(policy.reachable)(ip)) {
        return Err(Refusal::NonPublic(NonPublic::Address(ip)));
    }
    let host = target.host.to_string();
    let approved = policy
        .hosts
        .confirm(&host, || ConfirmationRequest {
            tool: "web".to_string(),
            script: format!("CONNECT {host}:{}", target.port),
            matched_pattern: format!("a sub-agent connects to {host}, which is not yet approved"),
            always_allow: vec![host.clone()],
            risk: Risk::Elevated,
            source: ConfirmSource::Main,
        })
        .await;
    if approved {
        Ok(())
    } else {
        Err(Refusal::Declined(host))
    }
}

async fn connect(target: &Target, policy: &EgressPolicy) -> Result<TcpStream, Refusal> {
    let host = target.host.to_string();
    let addrs = match target.literal_ip() {
        Some(ip) => vec![(ip, target.port).into()],
        None => policy
            .resolve(&host, target.port)
            .await
            .map_err(|source| Refusal::Resolve {
                host: host.clone(),
                source,
            })?,
    };
    let connect_error = |source| Refusal::Connect {
        host: host.clone(),
        source,
    };
    tokio::time::timeout(CONNECT_TIMEOUT, TcpStream::connect(addrs.as_slice()))
        .await
        .map_err(|_| connect_error(io::ErrorKind::TimedOut.into()))?
        .map_err(connect_error)
}

fn is_non_public(error: &io::Error) -> bool {
    error
        .get_ref()
        .is_some_and(<dyn Error + Send + Sync>::is::<NonPublic>)
}
