//! The rule outgoing web traffic is held to: approved hosts, public
//! addresses only, and for sub-agents only HTTPS.

use std::io;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};

use super::approvals::ApprovalGate;

const HTTPS_PORT: u16 = 443;

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

/// What traffic an [`EgressPolicy`] lets through.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Transport {
    /// Either scheme, to any port.
    Any,
    /// Only HTTPS, to port 443, so nothing on the path can read or alter it.
    Https,
}

/// Where outgoing web traffic may go: hosts `hosts` approves, at addresses
/// `reachable` accepts, over the transport the policy allows.
#[derive(Debug, Clone)]
pub(crate) struct EgressPolicy {
    pub(crate) hosts: ApprovalGate,
    pub(crate) reachable: fn(IpAddr) -> bool,
    transport: Transport,
}

impl EgressPolicy {
    /// HTTP or HTTPS to any port of a host reachable as `reachable` says.
    pub(crate) fn any_port(hosts: ApprovalGate, reachable: fn(IpAddr) -> bool) -> Self {
        Self {
            hosts,
            reachable,
            transport: Transport::Any,
        }
    }

    /// Only HTTPS to port 443 of public addresses: a sub-agent's rule.
    pub(crate) fn https_only(hosts: ApprovalGate) -> Self {
        Self {
            hosts,
            reachable: is_public,
            transport: Transport::Https,
        }
    }

    pub(crate) fn allows_port(&self, port: u16) -> bool {
        self.transport == Transport::Any || port == HTTPS_PORT
    }

    pub(crate) fn allows_scheme(&self, scheme: &str) -> bool {
        self.transport == Transport::Any || scheme == "https"
    }

    /// The addresses `host` resolves to that this policy may reach, with
    /// `port`. Fails with [`NonPublic::Name`] when there are none.
    pub(crate) async fn resolve(&self, host: &str, port: u16) -> io::Result<Vec<SocketAddr>> {
        reachable_addrs(host, port, self.reachable).await
    }
}

/// A destination was refused because it is not on the public internet.
#[derive(Debug, thiserror::Error)]
pub(crate) enum NonPublic {
    #[error("{0} is not a public address")]
    Address(IpAddr),
    #[error("{0} resolves only to non-public addresses")]
    Name(String),
}

/// The addresses `host` resolves to with the system resolver that
/// `reachable` accepts, with `port`; [`NonPublic::Name`] when there are none.
pub(crate) async fn reachable_addrs(
    host: &str,
    port: u16,
    reachable: fn(IpAddr) -> bool,
) -> io::Result<Vec<SocketAddr>> {
    let addrs: Vec<SocketAddr> = tokio::net::lookup_host((host, port))
        .await?
        .filter(|addr| reachable(addr.ip()))
        .collect();
    if addrs.is_empty() {
        return Err(io::Error::other(NonPublic::Name(host.to_string())));
    }
    Ok(addrs)
}

/// Whether `ip` is globally routable unicast, judging an IPv4 address
/// embedded in an IPv6 one by the IPv4 rules.
pub(crate) fn is_public(ip: IpAddr) -> bool {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn public_addresses_are_told_from_non_public_ones() {
        let cases: [(&str, bool); 26] = [
            ("1.1.1.1", true),
            ("8.8.8.8", true),
            ("100.63.255.255", true),
            ("172.32.0.1", true),
            ("0.0.0.0", false),
            ("127.0.0.1", false),
            ("10.1.2.3", false),
            ("100.64.0.1", false),
            ("169.254.169.254", false),
            ("172.31.255.255", false),
            ("192.168.1.1", false),
            ("198.18.0.1", false),
            ("224.0.0.1", false),
            ("255.255.255.255", false),
            ("2606:4700:4700::1111", true),
            ("::ffff:8.8.8.8", true),
            ("::1", false),
            ("::", false),
            ("::ffff:127.0.0.1", false),
            ("fd00::1", false),
            ("fe80::1", false),
            ("2001:db8::1", false),
            ("2002:808:808::1", true),
            ("2002:7f00:1::", false),
            ("2002:a9fe:a9fe::1", false),
            ("2001:0:4136:e378:8000:63bf:3fff:fdd2", false),
        ];
        for (addr, public) in cases {
            assert_eq!(is_public(addr.parse().unwrap()), public, "{addr}");
        }
    }
}
