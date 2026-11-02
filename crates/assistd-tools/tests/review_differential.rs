//! Differential tests of the script classifier: scripts run under real bash in bwrap, where each
//! program is a shim logging its name. Returns early when `bwrap` cannot run.

use std::collections::BTreeSet;
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::process::{Command, Stdio};
use std::thread;

use tempfile::TempDir;

use assistd_tools::policy::{Class, classify};

/// Scripts each run in every context.
const CORPUS: &[&str] = &[
    "ls -la",
    "git status --short && git diff --stat",
    r#"for f in *.txt; do
  wc -l "$f"
done"#,
    "find . -name '*.o' -exec rm {} +",
    "find . -type f -newer a -print0 | xargs -0 grep -l TODO",
    r"find . -maxdepth 1 -execdir chmod 644 {} \;",
    "ls | xargs -I{} mv {} {}.bak",
    r"printf '%s\n' a b | xargs -n1 -P2 rm -f",
    "cat a | sort | uniq -c | sort -rn | head -5",
    r#"x=$(rm -v a)
echo "$x""#,
    r#"echo "today is $(date +%F), cleaned $(rm -f b && echo b)""#,
    "echo `curl -s example.com`",
    "diff <(sort a) <(curl -s example.com)",
    "cat > /tmp/out.txt <<EOF
rm -rf /
$(rm -f a)
EOF",
    "cat <<'EOF'
rm -rf / literal
EOF",
    "bash <<EOF
rm a
EOF",
    "source /dev/stdin <<< 'shred a'",
    r#"case "$1" in
  clean) rm -rf build ;;
  *) echo usage ;;
esac"#,
    "if [ -f a ]; then
  mv a b
elif [[ -d b ]]; then
  echo dir
fi",
    r#"while read -r line; do
  echo "$line"
done < a"#,
    "until true; do dd if=/dev/zero of=a; done",
    "cleanup() { rm -f /tmp/lock; }
trap cleanup EXIT
echo working",
    "trap 'rm -f a' INT TERM",
    r#"f() { chmod +x "$1"; }
echo no call"#,
    "alias ll='rm -i'
ll a",
    r#"eval "rm a""#,
    "cmd=rm
$cmd a",
    r#""${EDITOR:-rm}" a"#,
    "env FOO=1 rm a",
    "env -u HOME -C . rm a",
    "nice -n 5 nohup timeout 1 stdbuf -oL mv a b",
    "command -v rm",
    "command rm a",
    "builtin eval rm a",
    "exec rm a",
    "coproc rm a",
    "time rm a",
    "! rm a",
    "{ rm a; } 2>/dev/null",
    "(cd /tmp && rm -f a)",
    r"r\m a",
    "'r''m' a",
    "/usr/bin/rm a",
    "/bin/rm a",
    "./rm a",
    "r{m,} a",
    "{rm,a}",
    "echo hi | tee /dev/null | wc -c",
    "a=1 b=2 rm a",
    r#"declare -a arr=(one two)
echo "${arr[@]}""#,
    r#"for i in $(seq 1 3); do echo "$i"; done"#,
    r#"select x in a b; do rm "$x"; break; done < /dev/null"#,
    "[[ $(curl -s example.com) == ok ]] && echo up",
    r#"test -n "$(wget -qO- example.com)""#,
    "echo $((1 + $(rm a; echo 1)))",
    "arr=([0]=$(rm a))",
    r#": "${x:=$(rm a)}""#,
    "echo ${x:-$(rm a)}",
    r#"printf -v out '%s' "$(rm a)""#,
    r#"git log --oneline | head -n 3 | while read -r sha msg; do echo "$sha"; done"#,
    "cargo build --release 2>&1 | tail -n 20",
    "grep -rn 'fn main' . | sort | head",
    "mkdir -p /tmp/build && cd /tmp/build && cmake .. && make -j4",
    "python3 -c 'print(1)'",
    r#"ls -1 | while IFS= read -r name; do echo "${name%.txt}"; done"#,
    "xargs -a a rm",
    r"find . -name x -ok rm {} \;",
    r"find . -name a -print -exec echo {} \; -exec rm {} \;",
    "timeout -s KILL 1 rm a",
    "stdbuf -o0 shred a",
    "nice -- rm a",
    "env -- rm a",
    "xargs -- rm < /dev/null",
    "function g { rm a; }
g",
    "g() (rm a)
g",
    "if ! command -v jq >/dev/null; then echo missing; fi",
    "echo 'rm a' | bash",
    "echo rm a | xargs",
    "mkdir -p out && touch out/x && cp a out/y && ln -s a l",
    "tee c < a > /dev/null",
    "sort -o b a",
    "ls | xargs sort",
    "cat a | sort | uniq -c",
    "find . -name '*.txt' -exec grep -l line {} +",
];

/// Programs the classifier may call read-only; bash builtins are never shimmed.
const READ_ONLY: &[&str] = &[
    "b2sum",
    "base64",
    "basename",
    "cat",
    "cksum",
    "cmp",
    "column",
    "comm",
    "cut",
    "date",
    "df",
    "diff",
    "dirname",
    "du",
    "egrep",
    "expand",
    "fgrep",
    "find",
    "fold",
    "grep",
    "head",
    "hexdump",
    "id",
    "join",
    "ls",
    "md5sum",
    "nl",
    "od",
    "paste",
    "printenv",
    "readlink",
    "realpath",
    "rev",
    "seq",
    "sha1sum",
    "sha256sum",
    "sha512sum",
    "sleep",
    "sort",
    "stat",
    "strings",
    "tac",
    "tail",
    "tr",
    "uname",
    "unexpand",
    "wc",
    "which",
    "whoami",
];

/// Programs that write or delete the paths they name.
const EDIT: &[&str] = &[
    "chgrp", "chmod", "chown", "cp", "ln", "mkdir", "mv", "rm", "tee", "touch",
];

/// Programs that run a command; their shims run the real program.
const REAL_WRAPPERS: &[&str] = &["env", "find", "nice", "nohup", "stdbuf", "timeout", "xargs"];

/// Where each corpus script is placed, at `@@`, besides on its own.
const CONTEXTS: &[&str] = &[
    "@@",
    "( @@\n)",
    "{ @@\n}",
    "if true; then\n@@\nfi",
    "for i in 1; do\n@@\ndone",
    "while :; do\n@@\nbreak\ndone",
    "q() {\n@@\n}\nq",
    "x=$(\n@@\n)",
    "echo \"$(\n@@\n)\"",
    "true && {\n@@\n}",
    "false || {\n@@\n}",
    "case x in x)\n@@\n;; esac",
    "{\n@@\n} &\nwait",
    "{\n@@\n} | cat",
    "cat <(\n@@\n)",
    "env @@",
    "nice @@",
    "nohup @@",
    "timeout 5 @@",
    "command @@",
    "exec @@",
    "echo a | xargs @@",
    "find . -name a -exec @@ \\;",
];

/// Contexts that single-quote the script for a builtin to evaluate.
const QUOTED_CONTEXTS: &[&str] = &["eval '@@'", "trap '@@' EXIT", "command eval '@@'"];

const PLACEHOLDER: &str = "@@";

/// Files in the directory each script runs in, so `find` and globs see something.
const WORK_FILES: &[&str] = &["a", "b", "c.txt", "d.o"];

const SANDBOX_REAL_BIN: &str = "/tmp/real";

const SANDBOX_BASH: &str = "/tmp/real/bash";

const SANDBOX_RUN: &str = "/tmp/run";

const SANDBOX_WORK: &str = "/tmp/run/work";

const RUN_SECONDS: &str = "10";

/// A shim directory and per-run scratch directories.
struct Harness {
    root: TempDir,
}

impl Harness {
    fn new(names: &BTreeSet<String>) -> Self {
        let root = tempfile::tempdir().expect("tempdir");
        let bin = root.path().join("bin");
        fs::create_dir(&bin).expect("bin dir");
        let shim = root.path().join("shim");
        fs::write(&shim, shim_script()).expect("write shim");
        fs::set_permissions(&shim, fs::Permissions::from_mode(0o755)).expect("chmod shim");
        for name in names {
            fs::hard_link(&shim, bin.join(name)).expect("link shim");
        }
        Self { root }
    }

    /// The shimmed programs bash started, directly or through a wrapper, running `script`.
    fn runs(&self, id: usize, script: &str) -> Vec<String> {
        let run = self.root.path().join(format!("run{id}"));
        let work = run.join("work");
        fs::create_dir_all(&work).expect("work dir");
        for file in WORK_FILES {
            fs::write(work.join(file), "line\n").expect("work file");
        }
        self.sandboxed_bash(&run, script)
            .status()
            .expect("run bwrap");
        fs::read_to_string(run.join("log"))
            .unwrap_or_default()
            .lines()
            .map(str::to_string)
            .collect()
    }

    /// Bash running `script` with the shims over `/usr/bin`, the real programs at
    /// [`SANDBOX_REAL_BIN`], and `run` writable at [`SANDBOX_RUN`], on a read-only root.
    fn sandboxed_bash(&self, run: &Path, script: &str) -> Command {
        let mut command = Command::new("timeout");
        command
            .args(["-s", "KILL", RUN_SECONDS, "bwrap", "--ro-bind", "/", "/"])
            .args(["--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp"])
            .args(["--ro-bind", "/usr/bin", SANDBOX_REAL_BIN, "--ro-bind"])
            .args([
                self.root.path().join("bin").as_path(),
                Path::new("/usr/bin"),
            ])
            .arg("--bind")
            .args([run, Path::new(SANDBOX_RUN)])
            .args(["--unshare-all", "--die-with-parent", "--clearenv"])
            .args([
                "--setenv",
                "PATH",
                "/usr/bin",
                "--setenv",
                "HOME",
                SANDBOX_WORK,
            ])
            .args([
                "--chdir",
                SANDBOX_WORK,
                "--",
                SANDBOX_BASH,
                "--norc",
                "--noprofile",
            ])
            .args(["-c", script])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        command
    }
}

#[test]
fn classified_scripts_run_only_the_programs_their_class_allows() {
    if !bwrap_runs() {
        return;
    }
    let scripts: Vec<String> = CORPUS.iter().flat_map(|script| variants(script)).collect();
    let harness = Harness::new(&shim_names(&scripts));
    let classes: Vec<(&str, Class)> = scripts
        .iter()
        .map(|script| (script.as_str(), classify(script, None)))
        .collect();
    let of_class = |wanted: fn(&Class) -> bool| -> Vec<&str> {
        classes
            .iter()
            .filter(|(_, class)| wanted(class))
            .map(|(script, _)| *script)
            .collect()
    };
    let read_only = of_class(|class| *class == Class::ReadOnly);
    let edits = of_class(|class| matches!(class, Class::Edit { .. }));
    assert!(
        !read_only.is_empty() && !edits.is_empty(),
        "the classifier found no read-only or no edit script"
    );
    let not_read_only = |name: &str| !READ_ONLY.contains(&name) && !REAL_WRAPPERS.contains(&name);
    let not_edit = |name: &str| not_read_only(name) && !EDIT.contains(&name);
    let mismatches: Vec<_> = run_all(&harness, &read_only, &not_read_only)
        .into_iter()
        .chain(run_all(&harness, &edits, &not_edit))
        .collect();
    assert!(
        mismatches.is_empty(),
        "classified scripts ran programs their class does not allow:\n{}",
        mismatches
            .iter()
            .map(|(script, ran)| format!("{ran:?} from:\n{script}\n"))
            .collect::<String>()
    );
}

/// Run every script on a pool of threads, returning those that started a `disallowed` program,
/// with the programs.
fn run_all<'s>(
    harness: &Harness,
    scripts: &[&'s str],
    disallowed: &(dyn Fn(&str) -> bool + Sync),
) -> Vec<(&'s str, Vec<String>)> {
    let threads = thread::available_parallelism().map_or(4, usize::from);
    let chunk = scripts.len().div_ceil(threads).max(1);
    thread::scope(|scope| {
        let workers: Vec<_> = scripts
            .chunks(chunk)
            .enumerate()
            .map(|(worker, batch)| {
                scope.spawn(move || {
                    batch
                        .iter()
                        .enumerate()
                        .map(|(at, script)| {
                            let mut ran = harness.runs(worker * chunk + at, script);
                            ran.retain(|name| disallowed(name));
                            (*script, ran)
                        })
                        .filter(|(_, ran)| !ran.is_empty())
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        workers
            .into_iter()
            .flat_map(|worker| worker.join().expect("worker"))
            .collect()
    })
}

fn bwrap_runs() -> bool {
    Command::new("bwrap")
        .args(["--ro-bind", "/", "/", "--unshare-all", "true"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

fn variants(script: &str) -> impl Iterator<Item = String> + '_ {
    let quoted = script.replace('\'', r"'\''");
    CONTEXTS
        .iter()
        .map(|context| context.replace(PLACEHOLDER, script))
        .chain(
            QUOTED_CONTEXTS
                .iter()
                .map(move |context| context.replace(PLACEHOLDER, &quoted)),
        )
}

/// Every word in `scripts` that could name a program, plus the fixed sets, minus bash's
/// builtins and keywords, which bash never looks up on `PATH`.
fn shim_names(scripts: &[String]) -> BTreeSet<String> {
    let builtins = bash_builtins();
    scripts
        .iter()
        .flat_map(|script| script.split(|c: char| !is_name_char(c)))
        .chain(REAL_WRAPPERS.iter().copied())
        .filter(|name| !name.is_empty() && !name.starts_with(['.', '-', '+']))
        .filter(|name| !builtins.contains(*name))
        .map(str::to_string)
        .collect()
}

fn is_name_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-' | '+')
}

fn bash_builtins() -> BTreeSet<String> {
    let output = Command::new("bash")
        .args(["--norc", "--noprofile", "-c", "compgen -b -k"])
        .output()
        .expect("run bash");
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(str::to_string)
        .collect()
}

/// A shim that appends its name to the run's log and execs the real program when it is a wrapper.
fn shim_script() -> String {
    format!(
        "#!{SANDBOX_BASH}\n\
         name=${{0##*/}}\n\
         printf '%s\\n' \"$name\" >> {SANDBOX_RUN}/log\n\
         case $name in\n\
         {}) exec \"{SANDBOX_REAL_BIN}/$name\" \"$@\" ;;\n\
         esac\n",
        REAL_WRAPPERS.join("|")
    )
}
