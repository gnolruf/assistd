//! Reads `/proc` to learn about processes: fields of a `stat` line and
//! which process group owns a TCP listening socket.

use std::collections::HashSet;
use std::fs;
use std::io;
use std::net::{IpAddr, SocketAddr};

use rustix::process::Pid;

const TCP_LISTEN_STATE: &str = "0A";

/// Whether a process in process group `group` holds the socket listening on
/// exactly `addr`. Errors if `/proc` cannot be read.
pub fn is_listening_in_group(addr: SocketAddr, group: Pid) -> io::Result<bool> {
    let inodes = listening_socket_inodes(addr)?;
    if inodes.is_empty() {
        return Ok(false);
    }
    let group = group.as_raw_nonzero().get();
    let owned = fs::read_dir("/proc")?
        .flatten()
        .filter_map(|entry| entry.file_name().to_str()?.parse().ok())
        .filter(|&pid| process_group_of(pid) == Some(group))
        .any(|pid| holds_any_socket(pid, &inodes));
    Ok(owned)
}

fn listening_socket_inodes(addr: SocketAddr) -> io::Result<HashSet<u64>> {
    let table = match addr.ip() {
        IpAddr::V4(_) => "/proc/net/tcp",
        IpAddr::V6(_) => "/proc/net/tcp6",
    };
    let contents = fs::read_to_string(table)?;
    Ok(parse_listening_inodes(&contents, &proc_net_address(addr)))
}

/// Inodes of `LISTEN` rows in a `/proc/net/tcp*` table whose local address
/// is `local_address`.
fn parse_listening_inodes(table: &str, local_address: &str) -> HashSet<u64> {
    table
        .lines()
        .skip(1)
        .filter_map(|line| {
            let fields: Vec<&str> = line.split_whitespace().collect();
            match fields[..] {
                [_, local, _, state, _, _, _, _, _, inode, ..]
                    if local == local_address && state == TCP_LISTEN_STATE =>
                {
                    inode.parse().ok()
                }
                _ => None,
            }
        })
        .collect()
}

/// `addr` as `/proc/net/tcp*` prints it: each 32-bit word of the IP in
/// native byte order as hex, then the port as hex.
fn proc_net_address(addr: SocketAddr) -> String {
    let octets = match addr.ip() {
        IpAddr::V4(ip) => ip.octets().to_vec(),
        IpAddr::V6(ip) => ip.octets().to_vec(),
    };
    let ip_hex: String = octets
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&word| format!("{:08X}", u32::from_ne_bytes(word)))
        .collect();
    format!("{ip_hex}:{:04X}", addr.port())
}

/// Process group of `pid`, read from `/proc` because kernel threads report
/// group 0, which `getpgid` wrappers reject.
fn process_group_of(pid: i32) -> Option<i32> {
    let stat = fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    parse_stat_process_group(&stat)
}

/// Field `index` of a `/proc/<pid>/stat` line, counted from the first field
/// after the parenthesised command name: 0 is the state, 1 the parent pid,
/// 2 the process group. The command name may hold spaces and parentheses.
pub fn proc_stat_field(stat: &str, index: usize) -> Option<&str> {
    let (_, after_command) = stat.rsplit_once(')')?;
    after_command.split_whitespace().nth(index)
}

fn parse_stat_process_group(stat: &str) -> Option<i32> {
    proc_stat_field(stat, 2)?.parse().ok()
}

fn holds_any_socket(pid: i32, inodes: &HashSet<u64>) -> bool {
    let Ok(fds) = fs::read_dir(format!("/proc/{pid}/fd")) else {
        return false;
    };
    fds.flatten()
        .filter_map(|fd| fs::read_link(fd.path()).ok())
        .filter_map(|target| socket_inode(target.to_str()?))
        .any(|inode| inodes.contains(&inode))
}

fn socket_inode(fd_target: &str) -> Option<u64> {
    fd_target
        .strip_prefix("socket:[")?
        .strip_suffix(']')?
        .parse()
        .ok()
}

#[cfg(test)]
mod tests {
    use std::net::{Ipv4Addr, Ipv6Addr, TcpListener};
    use std::os::unix::process::CommandExt;
    use std::process::{Command, Stdio};

    use rustix::process::getpgid;

    use super::*;

    const TCP_TABLE: &str = "\
  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode
   0: 0100007F:20C1 00000000:0000 0A 00000000:00000000 00:00000000 00000000  1000        0 4242 1 0000000000000000 100 0 0 10 0
   1: 0100007F:20C1 0100007F:9C40 01 00000000:00000000 00:00000000 00000000  1000        0 4343 1 0000000000000000 20 4 30 10 -1
   2: 00000000:20C2 00000000:0000 0A 00000000:00000000 00:00000000 00000000  1000        0 4444 1 0000000000000000 100 0 0 10 0
";

    fn own_process_group() -> Pid {
        getpgid(None).expect("own process group")
    }

    #[test]
    fn parse_keeps_only_listen_rows_on_the_exact_address() {
        let inodes = parse_listening_inodes(TCP_TABLE, "0100007F:20C1");
        assert_eq!(inodes, HashSet::from([4242]));
    }

    #[test]
    fn stat_process_group_counts_fields_after_the_command_name() {
        let user = "4242 (llama) server) S 1 4240 4240 0 -1 4194560";
        assert_eq!(parse_stat_process_group(user), Some(4240));
        let kernel_thread = "2 (kthreadd) S 0 0 0 0 -1 2129984";
        assert_eq!(parse_stat_process_group(kernel_thread), Some(0));
    }

    #[test]
    fn socket_inode_rejects_non_socket_targets() {
        assert_eq!(socket_inode("socket:[4242]"), Some(4242));
        assert_eq!(socket_inode("pipe:[4242]"), None);
        assert_eq!(socket_inode("/dev/null"), None);
    }

    #[cfg(target_endian = "little")]
    #[test]
    fn proc_net_address_matches_kernel_format() {
        let v4 = SocketAddr::from((Ipv4Addr::LOCALHOST, 8385));
        assert_eq!(proc_net_address(v4), "0100007F:20C1");
        let v6 = SocketAddr::from((Ipv6Addr::LOCALHOST, 8385));
        assert_eq!(
            proc_net_address(v6),
            "00000000000000000000000001000000:20C1"
        );
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
        let other_group = Pid::from_raw(other.id() as i32).expect("nonzero pid");
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
