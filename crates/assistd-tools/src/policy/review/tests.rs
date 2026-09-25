use std::os::unix::fs::PermissionsExt;
use std::path::Path;

use super::*;
use crate::policy::SearchPath;

fn patterns() -> Vec<DestructivePattern> {
    [
        "shutdown",
        "rm -r|--recursive",
        "dd of=",
        "git push -f|--force|--force-with-lease",
        "git reset --hard",
        "find -delete",
        "kill -1",
        "systemctl poweroff|reboot|halt",
    ]
    .into_iter()
    .map(|p| DestructivePattern::new(p.split_whitespace()).expect("valid pattern"))
    .collect()
}

/// An allowlist over an empty, read-only search path: no program is on
/// it, and a name that resolves to nothing is harmless, so only paths,
/// script files and risky builtins come back unlisted.
fn no_programs() -> Allowlist {
    Allowlist::unsaved(
        Vec::new(),
        SearchPath {
            dirs: Vec::new(),
            read_only: true,
        },
    )
}

/// `Some("rm -r|--recursive")` for a pattern match, `Some("?")` for an
/// unverifiable command, and `None` otherwise: these tests are about
/// patterns and what the review can see, not the allowlist.
fn check(script: &str) -> Option<String> {
    let patterns = patterns();
    let allowlist = no_programs();
    let rules = Rules {
        patterns: &patterns,
        allowlist: &allowlist,
        protected: &[],
    };
    match check_script(script, &rules)? {
        Confirmation::Pattern(pattern) => Some(pattern),
        Confirmation::Unverifiable(_) => Some("?".to_string()),
        Confirmation::Unlisted { .. } => None,
    }
}

fn assert_all(cases: &[(&str, Option<&str>)]) {
    for &(script, expected) in cases {
        assert_eq!(check(script).as_deref(), expected, "{script:?}");
    }
}

const RM: Option<&str> = Some("rm -r|--recursive");
const UNVERIFIABLE: Option<&str> = Some("?");

#[test]
fn default_patterns_all_parse() {
    for pattern in assistd_config::defaults::default_bash_destructive_patterns() {
        assert!(
            DestructivePattern::new(pattern.split_whitespace()).is_some(),
            "{pattern:?}"
        );
    }
}

#[test]
fn pattern_words_must_have_alternatives() {
    assert!(DestructivePattern::new(Vec::<&str>::new()).is_none());
    assert!(DestructivePattern::new(["|"]).is_none());
    assert!(DestructivePattern::new(["rm", "||"]).is_none());
    assert_eq!(
        DestructivePattern::new(["rm", "-r|--recursive"])
            .expect("valid")
            .to_string(),
        "rm -r|--recursive"
    );
}

#[test]
fn arguments_match_flags_in_any_spelling_and_order() {
    assert_all(&[
        ("rm -rf foo", RM),
        ("RM -RF foo", RM),
        ("rm -R foo", RM),
        ("rm -fr foo", RM),
        ("rm -rfv foo", RM),
        ("rm -v -r foo", RM),
        ("rm foo -r", RM),
        ("rm --recursive foo", RM),
        ("rm --rec foo", RM),
        ("rm -f foo", None),
        ("rm --force foo", None),
        ("dd if=/dev/zero of=/dev/sda bs=1M", Some("dd of=")),
        ("dd of=/dev/sda if=/dev/zero", Some("dd of=")),
        ("dd if=in.img", None),
        (
            "git push origin main --force",
            Some("git push -f|--force|--force-with-lease"),
        ),
        (
            "git push -uf origin main",
            Some("git push -f|--force|--force-with-lease"),
        ),
        (
            "git push --force-with-lease",
            Some("git push -f|--force|--force-with-lease"),
        ),
        ("git push origin main", None),
        ("git push --follow-tags", None),
        ("git reset --hard HEAD~1", Some("git reset --hard")),
        ("git reset --soft HEAD~1", None),
        ("find . -name '*.tmp' -delete", Some("find -delete")),
        ("kill -9 -1", Some("kill -1")),
        ("kill -s KILL -- -1", Some("kill -1")),
        ("kill -9 1234", None),
        ("systemctl reboot", Some("systemctl poweroff|reboot|halt")),
        ("systemctl restart nginx", None),
    ]);
}

#[test]
fn commands_are_found_wherever_bash_runs_one() {
    assert_all(&[
        ("touch foo && rm -rf bar", RM),
        ("false || rm -rf bar", RM),
        ("echo hi ; rm -rf bar", RM),
        ("ls | rm -rf foo", RM),
        ("true;rm -rf ~", RM),
        ("true&&rm -rf ~", RM),
        ("ls|rm -rf ~", RM),
        ("sleep 1&rm -rf ~", RM),
        ("ls |& rm -rf ~", RM),
        ("true\nrm -rf ~", RM),
        ("cd /tmp\n\n  rm -rf ~\n", RM),
        ("true # it's fine\nrm -rf ~", RM),
        ("rm -rf ~;# it's fine", RM),
        ("(rm -rf ~)", RM),
        ("(cd /tmp && rm -rf ~)", RM),
        ("{ rm -rf ~; }", RM),
        ("! rm -rf ~", RM),
        ("if true; then rm -rf ~; fi", RM),
        ("while true; do rm -rf ~; done", RM),
        ("f() { rm -rf ~; }; f", RM),
        ("function f { rm -rf ~; }", RM),
        ("FOO=1 BAR=2 rm -rf ~", RM),
        ("arr[0]=x rm -rf ~", RM),
        (">/dev/null rm -rf ~", RM),
        ("2>&1 rm -rf ~", RM),
        ("</dev/null rm -rf ~", RM),
        ("rm -rf ~ >/dev/null 2>&1", RM),
        ("/bin/rm -rf ~", RM),
        ("\\rm -rf ~", RM),
        ("\"rm\" -rf ~", RM),
    ]);
}

#[test]
fn substitutions_are_searched_in_every_quoting_context() {
    assert_all(&[
        ("echo $(rm -rf ~)", RM),
        ("echo \"$(rm -rf ~)\"", RM),
        ("echo `rm -rf ~`", RM),
        ("echo \"`rm -rf ~`\"", RM),
        ("echo `echo \\`rm -rf ~\\``", RM),
        ("echo ${x:-$(rm -rf ~)}", RM),
        ("echo \"${x:-$(rm -rf ~)}\"", RM),
        ("x=$(rm -rf ~)", RM),
        ("diff <(rm -rf ~) b", RM),
        ("echo $(( $(rm -rf ~) + 1 ))", RM),
        ("cat <<EOF\n$(rm -rf ~)\nEOF", RM),
        ("cat <<EOF\n`rm -rf ~`\nEOF", RM),
    ]);
}

#[test]
fn text_that_is_not_run_does_not_match() {
    assert_all(&[
        ("ls -l /tmp", None),
        ("echo \"rm -rf\"", None),
        ("echo 'shutdown'", None),
        ("echo rm -rf ~", None),
        ("sudo true ; echo rm -rf ~", None),
        ("echo \"a\nrm -rf ~\"", None),
        ("echo 'a\nrm -rf ~'", None),
        ("echo \\\nrm -rf ~", None),
        ("echo 'true;rm -rf ~'", None),
        ("echo \"(rm -rf ~)\"", None),
        ("echo 'x=$(rm -rf ~)'", None),
        ("echo a\\;rm -rf ~", None),
        ("true;# rm -rf ~", None),
        ("ls /bin/rm -rf", None),
        ("echo sh -c 'rm -rf ~'", None),
        ("cat <<'EOF'\n$(rm -rf ~)\nrm -rf ~\nEOF", None),
        ("cat <<EOF\nrm -rf ~\nEOF", None),
        ("git rm -r --cached foo", None),
        ("find . -name shutdown", None),
        ("command -v shutdown", None),
        ("grep -r shutdown src", None),
        ("echo $((1 << 2))\ntrue", None),
        ("[ -f x ] && echo yes", None),
    ]);
}

#[test]
fn arithmetic_shifts_are_not_heredocs() {
    assert_all(&[
        ("echo $((1 << 2))\nrm -rf ~", RM),
        ("(( x <<= 2 ))\nrm -rf ~", RM),
        ("echo $[1<<2]\nrm -rf ~", RM),
        ("n=$[1<<4]\nrm -rf ~", RM),
        ("echo $((1<<2))\nrm -rf ~", RM),
        ("echo $[ 1 << 2 ]\nshutdown", Some("shutdown")),
        ("a[1<<2]=x\nrm -rf ~", RM),
        ("echo ${a[1<<2]}\nrm -rf ~", RM),
        ("echo $[$(rm -rf ~)]", RM),
    ]);
}

#[test]
fn a_here_document_never_closed_is_unverifiable() {
    assert_all(&[
        ("cat <<EOF\nhello", UNVERIFIABLE),
        ("cat <<EOF\nhello\nEOF\n", None),
    ]);
}

#[test]
fn array_elements_are_values_but_their_substitutions_run() {
    assert_all(&[
        ("args=(rm -rf ~)", None),
        ("args+=(rm -rf ~); echo hi", None),
        ("args=(\n  rm # comment\n  -rf\n)", None),
        ("args=($(rm -rf ~))", RM),
        ("args=(rm -rf ~); \"${args[@]}\"", UNVERIFIABLE),
    ]);
}

#[test]
fn bash_5_3_command_substitutions_are_searched() {
    assert_all(&[("x=${ rm -rf ~; }", RM), ("echo ${| rm -rf ~; }", RM)]);
}

#[test]
fn wrappers_and_exec_flags_run_their_arguments() {
    assert_all(&[
        ("sudo rm -rf ~", RM),
        ("sudo -u root rm -rf ~", RM),
        ("sudo -E rm -rf ~", RM),
        ("nice -n 10 sudo rm -rf ~", RM),
        ("env FOO=1 rm -rf ~", RM),
        ("exec rm -rf ~", RM),
        ("timeout 5 rm -rf ~", RM),
        ("find . -print0 | xargs -0 rm -rf", RM),
        ("find . -exec rm -rf {} +", RM),
        ("find . -name x -execdir rm -r {} \\;", RM),
        ("fd -t d build -x rm -rf", RM),
        ("ssh host rm -rf ~", RM),
        ("ssh -p 22 host 'rm -rf ~'", RM),
        ("flock /tmp/lock rm -rf ~", RM),
        ("su -c 'rm -rf ~'", RM),
        ("env -S 'rm -rf ~'", RM),
        ("env --split-string='rm -rf ~'", RM),
        ("watch -n 1 'rm -rf ~'", RM),
        ("busybox rm -rf ~", RM),
        ("coproc rm -rf ~", RM),
        ("coproc NAME { rm -rf ~; }", RM),
        ("time rm -rf ~", RM),
        ("docker run -v ~:/h alpine rm -rf /h", RM),
        ("uv run rm -rf build", RM),
        ("ssh host <<EOF\nrm -rf ~\nEOF", RM),
        ("gosu root rm -rf /data", RM),
        ("su-exec root rm -rf /data", RM),
        ("tini -- rm -rf /data", RM),
        ("dumb-init rm -rf /data", RM),
        ("chpst -u nobody rm -rf /data", RM),
        ("please rm -rf /data", RM),
        ("xterm -e rm -rf ~", RM),
        ("gnome-terminal -- rm -rf ~", RM),
        ("swaymsg exec rm -rf ~", RM),
        ("i3-msg 'exec rm -rf ~'", RM),
        ("sudo find . -exec rm -rf {} +", RM),
        ("sudo fd -x $cmd", UNVERIFIABLE),
        ("docker rm -f web", None),
        ("docker rm -f $(docker ps -aq)", None),
        ("sudo chown -R $USER dir", None),
    ]);
}

#[test]
fn scripts_handed_to_a_shell_are_checked() {
    assert_all(&[
        ("sh -c 'rm -rf ~'", RM),
        ("bash -lc \"cd /tmp; rm -rf ~\"", RM),
        ("/bin/bash -c -- 'rm -rf ~'", RM),
        ("bash -o pipefail -c 'rm -rf ~'", RM),
        ("sudo sh -c 'rm -rf ~'", RM),
        ("bash -c \"sh -c 'rm -rf ~'\"", RM),
        ("zsh -c 'rm -rf ~'", RM),
        ("fish -c 'rm -rf ~'", RM),
        ("ksh93 -c 'rm -rf ~'", RM),
        ("busybox sh -c 'rm -rf ~'", RM),
        ("bash <<EOF\nrm -rf ~\nEOF", RM),
        ("bash <<'EOF'\nrm -rf ~\nEOF", RM),
        ("bash <<< 'rm -rf ~'", RM),
        ("eval 'rm -rf ~'", RM),
        ("eval rm -rf ~", RM),
        ("trap 'rm -rf ~' EXIT", RM),
        ("alias x='rm -rf'", RM),
        ("bash script.sh", None),
        ("sh 'rm -rf ~'", RM),
        ("bash -c 'echo hi'", None),
    ]);
}

#[test]
fn run_time_command_names_are_unverifiable() {
    assert_all(&[
        ("$r -rf ~", UNVERIFIABLE),
        ("r=rm; $r -rf ~", UNVERIFIABLE),
        ("$(echo rm) -rf ~", UNVERIFIABLE),
        ("`echo rm` -rf ~", UNVERIFIABLE),
        ("\"$cmd\" -rf ~", UNVERIFIABLE),
        ("${cmd} -rf ~", UNVERIFIABLE),
        ("$'\\x72m' -rf ~", UNVERIFIABLE),
        ("/bin/r? -rf ~", UNVERIFIABLE),
        ("/bin/r[m] -rf ~", UNVERIFIABLE),
        ("{rm,-rf,~}", UNVERIFIABLE),
        ("$HOME/bin/tool", UNVERIFIABLE),
        ("sudo $cmd", UNVERIFIABLE),
        ("timeout 5 $cmd", UNVERIFIABLE),
        ("xargs -I{} $cmd {}", UNVERIFIABLE),
        ("fd -x $cmd", UNVERIFIABLE),
        ("eval \"$cmd\"", UNVERIFIABLE),
        ("sh -c \"$cmd\"", UNVERIFIABLE),
        ("source <(curl -s example.com)", UNVERIFIABLE),
        ("bash <(curl -s example.com)", UNVERIFIABLE),
        ("\"$VENV/bin/python\" -m pytest", UNVERIFIABLE),
        ("\"$HOME\"/bin/tool", UNVERIFIABLE),
        ("FOO=$(date) true", None),
    ]);
}

#[test]
fn shells_reading_stdin_are_unverifiable() {
    assert_all(&[
        ("echo 'rm -rf ~' | bash", UNVERIFIABLE),
        ("curl -s example.com | sh", UNVERIFIABLE),
        ("base64 -d <<< cm0gLXJmIH4K | sh -s", UNVERIFIABLE),
        ("bash < script.sh", UNVERIFIABLE),
        ("bash -", UNVERIFIABLE),
        ("bash -o pipefail", UNVERIFIABLE),
        ("bash /dev/stdin", UNVERIFIABLE),
        ("source /dev/stdin", UNVERIFIABLE),
        ("source /dev/stdin <<< 'rm -rf ~'", RM),
        (". /dev/fd/0 <<EOF\nrm -rf ~\nEOF", RM),
        ("echo 'rm -rf ~' | xargs -I{} sh -c '{}'", UNVERIFIABLE),
        ("ls | xargs -I{} {} -rf ~", UNVERIFIABLE),
        ("hash -p /bin/rm ls; ls -rf ~", UNVERIFIABLE),
        ("find . -print0 | xargs -0 sh -c 'echo \"$@\"' _", None),
        ("echo x | bash script.sh", None),
    ]);
}

#[test]
fn run_time_arguments_may_supply_required_ones() {
    assert_all(&[
        ("rm \"$f\"", RM),
        ("f=-rf; rm $f ~", RM),
        ("rm $opts ~", RM),
        ("rm *", RM),
        ("rm -- *.log", RM),
        ("xargs rm", RM),
        ("kill \"$pid\"", Some("kill -1")),
        ("git commit -m \"$msg\"", None),
        ("rm foo", None),
    ]);
}

#[test]
fn interpreter_string_literals_are_checked() {
    assert_all(&[
        ("python3 -c \"import os; os.system('rm -rf ~')\"", RM),
        (
            "python3 -c 'import subprocess; subprocess.run([\"rm\", \"-rf\", p])'",
            RM,
        ),
        ("python3.12 -c 'import os; os.system(\"rm -rf \" + p)'", RM),
        ("perl -e 'system \"rm -rf ~\"'", RM),
        ("ruby -e '`rm -rf ~`'", RM),
        (
            "node -e \"require('child_process').execSync('rm -rf ~')\"",
            RM,
        ),
        ("awk 'BEGIN { system(\"rm -rf ~\") }'", RM),
        ("python3 <<'EOF'\nimport os\nos.system('rm -rf ~')\nEOF", RM),
        ("python3 -c 'print(\"hello\")'", None),
        ("perl -e 'print \"$x\\n\"'", None),
        ("python3 -m http.server 8000", None),
    ]);
}

#[test]
fn broken_or_deeply_nested_scripts_are_unverifiable() {
    assert_all(&[
        ("echo \"oops", UNVERIFIABLE),
        ("echo $(true", UNVERIFIABLE),
        ("rm -rf \"unterminated", RM),
        ("rm -rf ~\necho \"oops", RM),
    ]);
    let eval_chain = |depth: usize| format!("{}rm -rf ~", "eval ".repeat(depth));
    assert_eq!(check(&eval_chain(MAX_NESTED_SCRIPTS)).as_deref(), RM);
    assert_eq!(
        check(&eval_chain(MAX_NESTED_SCRIPTS + 2)).as_deref(),
        UNVERIFIABLE
    );
    assert_eq!(check(&eval_chain(10_000)).as_deref(), UNVERIFIABLE);
    let substitutions = format!("{}rm -rf ~{}", "$(".repeat(10_000), ")".repeat(10_000));
    assert_eq!(check(&substitutions).as_deref(), UNVERIFIABLE);
}

#[test]
fn argv_is_matched_without_reparsing_it() {
    let patterns = patterns();
    let allowlist = no_programs();
    let rules = Rules {
        patterns: &patterns,
        allowlist: &allowlist,
        protected: &[],
    };
    let argv = |args: &[&str]| -> Vec<String> { args.iter().map(|a| a.to_string()).collect() };
    let check_argv = |args: &[&str]| check_argv(&argv(args), &rules).map(|found| found.to_string());
    assert_eq!(
        check_argv(&["rm", "-rf", "/tmp/x"]).as_deref(),
        Some("rm -r|--recursive")
    );
    assert_eq!(
        check_argv(&["bash", "-c", "rm -rf ~"]).as_deref(),
        Some("rm -r|--recursive")
    );
    assert_eq!(
        check_argv(&["xterm", "-e", "rm -rf ~"]).as_deref(),
        Some("rm -r|--recursive")
    );
    assert_eq!(
        check_argv(&["firefox", "https://example.com/?q=$x&a=(b)"]),
        None
    );
}

/// A search path holding a program for each of `names`, read-only as in
/// the sandbox, with `allowed` on the allowlist.
struct Programs {
    dir: tempfile::TempDir,
    allowlist: Allowlist,
}

impl Programs {
    fn new(names: &[&str], allowed: &[&str]) -> Self {
        let dir = tempfile::tempdir().expect("tempdir");
        for name in names {
            let path = dir.path().join(name);
            std::fs::write(&path, b"#!/bin/sh\n").expect("write program");
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).expect("chmod");
        }
        let allowlist = Allowlist::unsaved(
            allowed.iter().map(|s| s.to_string()),
            SearchPath {
                dirs: vec![dir.path().to_path_buf()],
                read_only: true,
            },
        );
        Self { dir, allowlist }
    }

    fn review(&self, script: &str, protected: &[PathBuf]) -> Option<Confirmation> {
        let patterns = patterns();
        check_script(
            script,
            &Rules {
                patterns: &patterns,
                allowlist: &self.allowlist,
                protected,
            },
        )
    }

    fn path(&self) -> &Path {
        self.dir.path()
    }
}

fn unlisted(programs: &[&str], approvable: bool) -> Option<Confirmation> {
    Some(Confirmation::Unlisted {
        programs: programs.iter().map(|p| p.to_string()).collect(),
        approvable,
    })
}

fn dev_machine() -> Programs {
    Programs::new(
        &[
            "cat", "env", "xargs", "nice", "bash", "sh", "cargo", "git", "sudo", "su", "python3",
        ],
        &["cat", "env", "xargs", "nice", "bash", "sh"],
    )
}

#[test]
fn every_program_a_script_runs_must_be_allowed() {
    let machine = dev_machine();
    for (script, expected) in [
        ("cat x", None),
        ("cat x | sort", None),
        ("nonexistent-tool", None),
        ("cargo build", unlisted(&["cargo"], true)),
        (
            "cat x | cargo build && git status",
            unlisted(&["cargo", "git"], true),
        ),
        ("echo $(cargo --version)", unlisted(&["cargo"], true)),
        ("sudo cat x", unlisted(&["sudo"], true)),
        ("env FOO=1 cargo build", unlisted(&["cargo"], true)),
        ("nice -n 10 cat x", None),
        ("find . | xargs -n 1 cat", None),
        ("bash -c 'cargo build'", unlisted(&["cargo"], true)),
        ("bash -c 'cat x'", None),
        ("env -S 'cargo build'", unlisted(&["cargo"], true)),
        ("su -c 'cargo build'", unlisted(&["su", "cargo"], true)),
        ("python3 -c 'print(1)'", unlisted(&["python3"], true)),
        ("f() { cat x; }; f", None),
        ("cargo() { cat x; }; cargo", None),
        ("cargo; cargo() { :; }", unlisted(&["cargo"], true)),
        ("./build.sh", unlisted(&["./build.sh"], false)),
        ("bash build.sh", unlisted(&["bash build.sh"], false)),
        (
            "source venv/bin/activate",
            unlisted(&["source venv/bin/activate"], false),
        ),
        ("enable -f x.so y", unlisted(&["enable"], false)),
        ("cargo build; ./x", unlisted(&["cargo", "./x"], false)),
    ] {
        assert_eq!(machine.review(script, &[]), expected, "{script:?}");
    }
}

#[test]
fn always_allow_is_offered_only_for_unlisted_named_programs() {
    let machine = dev_machine();
    let offered = |script: &str| {
        machine
            .review(script, &[])
            .map(|c| c.always_allow().to_vec())
            .unwrap_or_default()
    };
    assert_eq!(offered("cargo build && git status"), ["cargo", "git"]);
    assert!(
        offered("cargo build; rm -rf ~").is_empty(),
        "a pattern match"
    );
    assert!(
        offered("cargo build; $cmd").is_empty(),
        "an unverifiable command"
    );
    assert!(offered("cargo build; ./x").is_empty(), "a path");
}

#[test]
fn a_program_on_a_path_is_allowed_only_where_it_is_trusted() {
    let machine = dev_machine();
    let cat = machine.path().join("cat");
    assert_eq!(machine.review(&format!("{} x", cat.display()), &[]), None);
    let copy = tempfile::tempdir().expect("tempdir");
    let planted = copy.path().join("cat");
    std::fs::copy(&cat, &planted).expect("copy");
    assert_eq!(
        machine.review(&format!("{} x", planted.display()), &[]),
        unlisted(&[&planted.to_string_lossy()], false)
    );
}

#[test]
fn changing_the_search_path_or_loader_makes_names_unverifiable() {
    let machine = dev_machine();
    for script in [
        "PATH=/tmp:$PATH cat x",
        "export PATH=\"$HOME/bin:$PATH\"; cat x",
        "read -r PATH <<< /tmp; cat x",
        "for PATH in /tmp; do cat x; done",
        "LD_PRELOAD=/tmp/x.so cat x",
        "export LD_LIBRARY_PATH=/tmp",
        "env BASH_ENV=/tmp/x bash -c 'cat x'",
        "declare -n ref=PATH; ref=/tmp",
        ": ${PATH:=/tmp}",
    ] {
        assert!(
            matches!(
                machine.review(script, &[]),
                Some(Confirmation::Unverifiable(_))
            ),
            "{script:?}"
        );
    }
    for script in [
        "echo $PATH",
        "echo \"$LD_LIBRARY_PATH\"",
        "PYTHONPATH=x cat y",
    ] {
        assert_eq!(machine.review(script, &[]), None, "{script:?}");
    }
}

#[test]
fn naming_a_protected_directory_is_unverifiable() {
    let machine = dev_machine();
    let config = tempfile::tempdir().expect("tempdir");
    let protected = [config.path().to_path_buf()];
    let file = config.path().join("config.toml");
    for script in [
        format!("cat {}", file.display()),
        format!("echo x > {}", file.display()),
        format!("cat < {}", file.display()),
    ] {
        assert!(
            matches!(
                machine.review(&script, &protected),
                Some(Confirmation::Unverifiable(_))
            ),
            "{script:?}"
        );
    }
    assert_eq!(machine.review("cat /etc/hostname", &protected), None);
}
