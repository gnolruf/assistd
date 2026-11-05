use std::fs;
use std::os::unix::fs::{PermissionsExt, symlink};

use tempfile::TempDir;

use super::*;
use crate::policy::{
    Allowlist, GuardRoots, PathGuards, Review, SearchPath, analyze, guarded_review,
};

/// A home with credentials, a `work` directory commands run from holding `escape`, a symlink
/// to `outside`, and a `bin` directory holding the program `deploy`.
struct Fixture {
    dir: TempDir,
}

impl Fixture {
    fn new() -> Self {
        let dir = TempDir::new().expect("tempdir");
        for sub in ["home/.ssh", "home/work", "outside", "bin"] {
            fs::create_dir_all(dir.path().join(sub)).expect("mkdir");
        }
        symlink(
            dir.path().join("outside"),
            dir.path().join("home/work/escape"),
        )
        .expect("symlink");
        let deploy = dir.path().join("bin/deploy");
        fs::write(&deploy, "#!/bin/sh\n").expect("seed program");
        fs::set_permissions(&deploy, fs::Permissions::from_mode(0o755)).expect("chmod");
        Self { dir }
    }

    fn path(&self, sub: &str) -> PathBuf {
        fs::canonicalize(self.dir.path())
            .expect("canonical tempdir")
            .join(sub)
    }

    /// What grants must cover for `script`, run in `work`, to run unasked.
    fn grantable(&self, script: &str) -> Grantable {
        let guards = PathGuards::read(GuardRoots {
            home: Some(self.path("home")),
            cwd: Some(self.path("home/work")),
            ..GuardRoots::default()
        });
        let allowlist = Allowlist::unsaved(SearchPath {
            dirs: vec![self.path("bin")],
        });
        match guarded_review(&guards, analyze(script, None, Some(&allowlist)), &[], &[]) {
            Review::Ask(_, _, grantable) => grantable,
            other => panic!("{script:?} must ask, got {other:?}"),
        }
    }
}

#[test]
fn no_grant_approves_outside_its_scope() {
    let fixture = Fixture::new();
    let grants = Grants::new(vec![
        TaskGrant::Write {
            under: fixture.path("home"),
        },
        TaskGrant::Program {
            program: "deploy".into(),
            subcommand: Some("push".into()),
        },
        TaskGrant::WebHost {
            host: "example.com".into(),
        },
        TaskGrant::McpTool {
            name: "mcp__fs__read".into(),
        },
    ]);
    let web = |host: &str| Grantable::needs(vec![GrantNeed::WebHost(host.into())]);
    let mcp = |name: &str| Grantable::needs(vec![GrantNeed::McpTool(name.into())]);
    let cases = [
        (
            "a write under the directory",
            fixture.grantable("touch notes.txt"),
            true,
        ),
        (
            "a write in a new subdirectory",
            fixture.grantable("touch new/notes.txt"),
            true,
        ),
        (
            "the granted subcommand",
            fixture.grantable("deploy push"),
            true,
        ),
        (
            "another subcommand",
            fixture.grantable("deploy rollback"),
            false,
        ),
        (
            "the program with only options",
            fixture.grantable("deploy --dry-run"),
            false,
        ),
        (
            "a write escaping through ..",
            fixture.grantable("touch ../../outside/x"),
            false,
        ),
        (
            "a write escaping through a symlink",
            fixture.grantable("touch escape/x"),
            false,
        ),
        (
            "the granted directory itself",
            fixture.grantable("touch ~"),
            false,
        ),
        (
            "a granted run writing outside",
            fixture.grantable("deploy push > ../../outside/log"),
            false,
        ),
        (
            "a credential, which always asks",
            fixture.grantable("touch ~/.ssh/config"),
            false,
        ),
        ("the exact host", web("example.com"), true),
        ("the host in another case", web("Example.COM"), true),
        ("a host sharing a suffix", web("evil-example.com"), false),
        ("a subdomain", web("api.example.com"), false),
        ("the exact tool", mcp("mcp__fs__read"), true),
        ("another tool", mcp("mcp__fs__write"), false),
        ("an always-ask request", Grantable::Never, false),
    ];
    for (case, grantable, covered) in cases {
        assert_eq!(grants.covering(&grantable).is_some(), covered, "{case}");
    }
}
