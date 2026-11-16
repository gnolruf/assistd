//! Whether a process group owns every TCP socket listening on an address, read from `/proc`.

use std::collections::HashSet;
use std::io;
use std::net::SocketAddr;

use procfs::net::TcpState;
use procfs::process::{FDTarget, Process, all_processes};
use rustix::process::Pid;

/// Whether process group `group` holds every socket listening on exactly
/// `addr`, and there is at least one. Errors if `/proc` cannot be read.
pub(super) fn group_holds_every_listener(addr: SocketAddr, group: Pid) -> io::Result<bool> {
    let listening = listening_socket_inodes(addr).map_err(io::Error::other)?;
    if listening.is_empty() {
        return Ok(false);
    }
    let group = group.as_raw_nonzero().get();
    let held_in_group: HashSet<u64> = all_processes()
        .map_err(io::Error::other)?
        .flatten()
        .filter(|process| process.stat().is_ok_and(|stat| stat.pgrp == group))
        .flat_map(|process| socket_inodes(&process))
        .collect();
    Ok(listening.is_subset(&held_in_group))
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

fn socket_inodes(process: &Process) -> impl Iterator<Item = u64> + use<> {
    process
        .fd()
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|fd| match fd.target {
            FDTarget::Socket(inode) => Some(inode),
            _ => None,
        })
}

#[cfg(test)]
mod tests {
    use std::net::{Ipv4Addr, TcpListener};
    use std::os::fd::OwnedFd;
    use std::os::unix::process::CommandExt;
    use std::process::{Child, Command, Stdio};

    use rustix::process::getpgid;
    use tokio::net::TcpSocket;

    use super::*;

    fn own_process_group() -> Pid {
        getpgid(None).expect("own process group")
    }

    fn sleeper_in_new_group(stdin: Stdio) -> (Child, Pid) {
        let sleeper = Command::new("sleep")
            .arg("30")
            .stdin(stdin)
            .process_group(0)
            .spawn()
            .expect("spawn sleep");
        let group = i32::try_from(sleeper.id())
            .ok()
            .and_then(Pid::from_raw)
            .expect("valid pid");
        (sleeper, group)
    }

    fn reuseport_listener(addr: SocketAddr) -> TcpListener {
        let socket = TcpSocket::new_v4().expect("socket");
        socket.set_reuseport(true).expect("set SO_REUSEPORT");
        socket.bind(addr).expect("bind");
        socket
            .listen(1)
            .and_then(tokio::net::TcpListener::into_std)
            .expect("listen")
    }

    #[test]
    fn listener_in_own_group_is_found() {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).expect("bind");
        let addr = listener.local_addr().expect("local addr");
        assert!(group_holds_every_listener(addr, own_process_group()).expect("read /proc"));
    }

    #[test]
    fn listener_outside_group_is_rejected() {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).expect("bind");
        let addr = listener.local_addr().expect("local addr");
        let (mut other, other_group) = sleeper_in_new_group(Stdio::null());
        let owned = group_holds_every_listener(addr, other_group);
        let _ = other.kill();
        let _ = other.wait();
        assert!(!owned.expect("read /proc"));
    }

    #[tokio::test]
    async fn address_shared_with_a_listener_outside_the_group_is_rejected() {
        let listener = reuseport_listener((Ipv4Addr::LOCALHOST, 0).into());
        let addr = listener.local_addr().expect("local addr");
        let second_listener = OwnedFd::from(reuseport_listener(addr));
        let (mut other, other_group) = sleeper_in_new_group(Stdio::from(second_listener));
        let owned_by_either = [own_process_group(), other_group]
            .map(|group| group_holds_every_listener(addr, group).expect("read /proc"));
        let _ = other.kill();
        let _ = other.wait();
        assert_eq!(owned_by_either, [false, false]);
    }
}
