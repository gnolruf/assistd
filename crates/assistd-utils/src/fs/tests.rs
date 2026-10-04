use std::io::Read;
use std::os::unix::net::UnixListener;

use super::*;

fn assert_not_regular(err: &io::Error) {
    assert_eq!(err.kind(), ErrorKind::InvalidInput, "{err}");
    assert_eq!(
        err.to_string(),
        "not a regular file (device, pipe, or socket)"
    );
}

#[test]
fn reads_a_regular_file_with_its_size() {
    for open in [open_regular, open_regular_or_dir] {
        let (mut file, size) = open(Path::new("Cargo.toml")).unwrap();
        let mut text = String::new();
        file.read_to_string(&mut text).unwrap();
        assert_eq!(size, text.len() as u64);
        assert!(text.contains("assistd-utils"));
    }
}

#[test]
fn directory_is_refused_or_fails_on_read() {
    assert_not_regular(&open_regular(Path::new("src")).unwrap_err());
    let (mut dir, _) = open_regular_or_dir(Path::new("src")).unwrap();
    let err = dir.read_to_end(&mut Vec::new()).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::IsADirectory);
}

#[test]
fn char_device_is_refused() {
    for open in [open_regular, open_regular_or_dir] {
        assert_not_regular(&open(Path::new("/dev/null")).unwrap_err());
    }
}

#[test]
fn socket_is_refused_before_it_is_opened() {
    let socket = std::env::temp_dir().join(format!("assistd-utils-fs-{}.sock", std::process::id()));
    let listener = UnixListener::bind(&socket).unwrap();
    let opened = rustix::fs::open(&socket, OFlags::RDONLY | OFlags::NONBLOCK, Mode::empty());
    let refused = open_regular_or_dir(&socket);
    drop(listener);
    std::fs::remove_file(&socket).unwrap();
    assert_eq!(opened.unwrap_err(), rustix::io::Errno::NXIO);
    assert_not_regular(&refused.unwrap_err());
}
