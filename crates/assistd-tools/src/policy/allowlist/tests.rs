use std::os::unix::fs::PermissionsExt;

use super::*;

const OPTIONS_ONLY: Option<&ApprovalScope> = Some(&ApprovalScope::OptionsOnly);

fn options_only(name: &str) -> ProgramApproval {
    ProgramApproval {
        name: name.to_string(),
        scope: ApprovalScope::OptionsOnly,
    }
}

fn executable(dir: &Path, name: &str) -> PathBuf {
    let path = dir.join(name);
    std::fs::write(&path, b"#!/bin/sh\n").expect("write program");
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).expect("chmod");
    path
}

/// A read-only search path over `dirs`, as a sandbox would give: file
/// ownership does not matter, so these tests hold under root too.
fn sandboxed(dirs: &[&Path]) -> SearchPath {
    SearchPath {
        dirs: dirs.iter().map(|d| d.to_path_buf()).collect(),
        read_only: true,
    }
}

#[test]
fn configured_names_resolve_on_the_search_path() {
    let bin = tempfile::tempdir().expect("tempdir");
    executable(bin.path(), "ls");
    executable(bin.path(), "cargo");
    let allowlist = Allowlist::unsaved(["ls".to_string()], sandboxed(&[bin.path()]));
    assert_eq!(allowlist.verdict("ls", OPTIONS_ONLY), Verdict::Allowed);
    assert_eq!(
        allowlist.verdict("cargo", OPTIONS_ONLY),
        Verdict::Unlisted { approvable: true }
    );
    assert_eq!(allowlist.verdict("nope", OPTIONS_ONLY), Verdict::Missing);
    assert!(allowlist.search_path_fixed());
}

#[test]
fn paths_match_only_as_configured_or_as_an_unmodifiable_named_program() {
    let bin = tempfile::tempdir().expect("tempdir");
    let elsewhere = tempfile::tempdir().expect("tempdir");
    let ls = executable(bin.path(), "ls");
    let copy = executable(elsewhere.path(), "ls");
    let tool = executable(elsewhere.path(), "tool");
    let allowlist = Allowlist::unsaved(
        ["ls".to_string(), tool.to_string_lossy().into_owned()],
        sandboxed(&[bin.path()]),
    );
    let verdict = |path: &Path| allowlist.verdict(&path.to_string_lossy(), OPTIONS_ONLY);
    assert_eq!(verdict(&ls), Verdict::Allowed);
    assert_eq!(verdict(&tool), Verdict::Allowed);
    assert_eq!(
        verdict(&copy),
        Verdict::Unlisted { approvable: false },
        "a copy outside the read-only search path is not trusted by name"
    );
    assert_eq!(
        allowlist.verdict("./ls", OPTIONS_ONLY),
        Verdict::Unlisted { approvable: false }
    );
}

#[tokio::test]
async fn an_approval_is_pinned_to_where_the_name_resolved() {
    let early = tempfile::tempdir().expect("tempdir");
    let late = tempfile::tempdir().expect("tempdir");
    executable(late.path(), "tool");
    let allowlist = Allowlist::unsaved(Vec::new(), sandboxed(&[early.path(), late.path()]));
    allowlist
        .approve(&[options_only("tool")])
        .await
        .expect("approve");
    assert_eq!(allowlist.verdict("tool", OPTIONS_ONLY), Verdict::Allowed);

    executable(early.path(), "tool");
    assert_eq!(
        allowlist.verdict("tool", OPTIONS_ONLY),
        Verdict::Unlisted { approvable: true },
        "a program planted earlier on the path does not inherit the approval"
    );
}

#[tokio::test]
async fn approvals_are_saved_and_loaded() {
    let bin = tempfile::tempdir().expect("tempdir");
    let config = tempfile::tempdir().expect("tempdir");
    executable(bin.path(), "tool");
    let store = config.path().join("nested").join(APPROVALS_FILE);

    let allowlist =
        Allowlist::load(Vec::new(), sandboxed(&[bin.path()]), store.clone()).expect("load");
    allowlist
        .approve(&[options_only("tool"), options_only("missing")])
        .await
        .expect("approve");
    let saved = std::fs::read_to_string(&store).expect("saved");
    assert!(saved.contains("name = \"tool\""), "{saved}");
    assert!(!saved.contains("missing"), "{saved}");

    let reloaded = Allowlist::load(Vec::new(), sandboxed(&[bin.path()]), store).expect("reload");
    assert_eq!(reloaded.verdict("tool", OPTIONS_ONLY), Verdict::Allowed);
}

#[test]
fn a_program_that_runs_any_code_is_never_approved() {
    let bin = tempfile::tempdir().expect("tempdir");
    let config = tempfile::tempdir().expect("tempdir");
    let python = executable(bin.path(), "python3.12");
    std::os::unix::fs::symlink(&python, bin.path().join("py")).expect("symlink");
    executable(bin.path(), "tool");
    let store = config.path().join(APPROVALS_FILE);
    let stored = format!(
        "[[program]]\nname = \"py\"\npath = \"{}\"\n",
        python.display()
    );
    std::fs::write(&store, stored).expect("write approvals");
    let allowlist = Allowlist::load(Vec::new(), sandboxed(&[bin.path()]), store).expect("load");
    for name in ["python3.12", "py"] {
        assert_eq!(
            allowlist.verdict(name, OPTIONS_ONLY),
            Verdict::Unlisted { approvable: false },
            "{name}"
        );
    }
    assert_eq!(
        allowlist.verdict("tool", OPTIONS_ONLY),
        Verdict::Unlisted { approvable: true }
    );
}

#[tokio::test]
async fn an_approval_covers_only_its_subcommand() {
    let bin = tempfile::tempdir().expect("tempdir");
    let config = tempfile::tempdir().expect("tempdir");
    executable(bin.path(), "cargo");
    let store = config.path().join(APPROVALS_FILE);
    let subcommand = |name: &str| ApprovalScope::Subcommand(name.to_string());
    let allowlist =
        Allowlist::load(Vec::new(), sandboxed(&[bin.path()]), store.clone()).expect("load");
    allowlist
        .approve(&[ProgramApproval {
            name: "cargo".to_string(),
            scope: subcommand("build"),
        }])
        .await
        .expect("approve");

    let reloaded = Allowlist::load(Vec::new(), sandboxed(&[bin.path()]), store).expect("reload");
    for allowlist in [&allowlist, &reloaded] {
        assert_eq!(
            allowlist.verdict("cargo", Some(&subcommand("build"))),
            Verdict::Allowed
        );
        for other in [Some(&subcommand("publish")), OPTIONS_ONLY] {
            assert_eq!(
                allowlist.verdict("cargo", other),
                Verdict::Unlisted { approvable: true },
                "{other:?}"
            );
        }
        assert_eq!(
            allowlist.verdict("cargo", None),
            Verdict::Unlisted { approvable: false }
        );
    }
}
