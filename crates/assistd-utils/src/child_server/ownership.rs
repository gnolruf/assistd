//! Whether a process group owns a TCP listening socket, read from `/proc`.

use std::collections::HashSet;
use std::io;
use std::net::SocketAddr;

use procfs::net::TcpState;
use procfs::process::{FDTarget, Process, all_processes};
use rustix::process::Pid;

/// Whether a process in process group `group` holds the socket listening on
/// exactly `addr`. Errors if `/proc` cannot be read.
pub(super) fn is_listening_in_group(addr: SocketAddr, group: Pid) -> io::Result<bool> {
    let inodes = listening_socket_inodes(addr).map_err(io::Error::other)?;
    if inodes.is_empty() {
        return Ok(false);
    }
    let group = group.as_raw_nonzero().get();
    let owned = all_processes()
        .map_err(io::Error::other)?
        .flatten()
        .filter(|process| process.stat().is_ok_and(|stat| stat.pgrp == group))
        .any(|process| holds_any_socket(&process, &inodes));
    Ok(owned)
}

fn listening_socket_inodes(addr: SocketAddr) -> procfs::ProcResult<HashSet<u64>> {
    let table = if addr.is_ipv4() {
        procfs::net::tcp()?
    } else {
        procfs::net::tcp6()?
    };
    Ok(table
        .into_iter()
        .filter(|entry| entry.state == TcpState::Listen && entry.local_address == addr)
        .map(|entry| entry.inode)
        .collect())
}

fn holds_any_socket(process: &Process, inodes: &HashSet<u64>) -> bool {
    process.fd().is_ok_and(|fds| {
        fds.flatten()
            .any(|fd| matches!(fd.target, FDTarget::Socket(inode) if inodes.contains(&inode)))
    })
}

#[cfg(test)]
mod tests {
    use std::net::{Ipv4Addr, TcpListener};
    use std::os::unix::process::CommandExt;
    use std::process::{Command, Stdio};

    use rustix::process::getpgid;

    use super::*;

    fn own_process_group() -> Pid {
        getpgid(None).expect("own process group")
    }

    #[test]
    fn listener_in_own_group_is_found() {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).expect("bind");
        let addr = listener.local_addr().expect("local addr");
        assert!(is_listening_in_group(addr, own_process_group()).expect("read /proc"));
    }

    #[test]
    fn listener_outside_group_is_rejected() {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).expect("bind");
        let addr = listener.local_addr().expect("local addr");
        let mut other = Command::new("sleep")
            .arg("30")
            .stdin(Stdio::null())
            .process_group(0)
            .spawn()
            .expect("spawn sleep");
        let other_group = i32::try_from(other.id())
            .ok()
            .and_then(Pid::from_raw)
            .expect("valid pid");
        let owned = is_listening_in_group(addr, other_group);
        let _ = other.kill();
        let _ = other.wait();
        assert!(!owned.expect("read /proc"));
    }

    #[test]
    fn unbound_address_is_not_owned() {
        let addr = TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
            .and_then(|listener| listener.local_addr())
            .expect("reserve port");
        assert!(!is_listening_in_group(addr, own_process_group()).expect("read /proc"));
    }
}
