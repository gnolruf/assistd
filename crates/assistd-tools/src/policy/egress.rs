//! The rule outgoing web traffic is held to: approved hosts, public
//! addresses only, HTTPS first for the main agent and only HTTPS for sub-agents.

use std::net::IpAddr;

use assistd_utils::net::is_public;

use super::approvals::ApprovalGate;

const HTTPS_PORT: u16 = 443;

/// What traffic an [`EgressPolicy`] lets through.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Transport {
    /// HTTPS to any port; plain HTTP only for a single fetch the user confirms.
    PreferHttps,
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
    /// HTTPS, or plain HTTP once the user confirms it for that fetch, to any
    /// port of a host reachable as `reachable` says.
    pub(crate) fn any_port(hosts: ApprovalGate, reachable: fn(IpAddr) -> bool) -> Self {
        Self {
            hosts,
            reachable,
            transport: Transport::PreferHttps,
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
        self.transport == Transport::PreferHttps || port == HTTPS_PORT
    }

    /// Whether a plain-HTTP fetch may go ahead once the user confirms it.
    pub(crate) fn allows_plaintext(&self) -> bool {
        self.transport == Transport::PreferHttps
    }
}
