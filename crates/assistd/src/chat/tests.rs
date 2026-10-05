use super::InstanceLock;

#[test]
fn instance_lock_refuses_second_chat_until_first_drops() {
    let temp = tempfile::tempdir().unwrap();
    let socket_path = temp.path().join("assistd.sock");

    let first = InstanceLock::acquire(&socket_path).unwrap();
    let err = InstanceLock::acquire(&socket_path)
        .err()
        .expect("second acquire must fail");
    assert!(err.to_string().contains("already open"), "{err:#}");

    drop(first);
    InstanceLock::acquire(&socket_path).expect("lock must be free once the guard drops");
}
