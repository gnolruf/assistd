//! Differential tests of the script classifier: scripts run under real bash in bwrap, with each
//! program a shim logging its name, or with the real programs and the files they change or open
//! compared. Returns early when `bwrap` cannot run.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::SystemTime;

use tempfile::TempDir;

use assistd_tools::policy::{Action, Class, Effect, analyze, classify};

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
    "case a in *) shred a ;; esac",
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
    "timeout -v 1 rm a",
    "timeout --foreground 1 shred a",
    "timeout -k 1 --preserve-status 1 rm a",
    "timeout -- 1 rm a",
    "nice -5 rm a",
    "nice --adjustment 5 shred a",
    "stdbuf --output=L -e0 rm a",
    "env -i -u HOME rm a",
    "env -S 'rm a'",
    "env --ignore-environment --chdir . shred a",
    "xargs -r -n1 -a a rm",
    "command -p rm a",
    "time -p rm a",
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
const EDIT: &[&str] = &["chmod", "cp", "mkdir", "mv", "rm", "tee", "touch"];

/// Edit programs in the option and operand forms they take, run for the paths they change.
const EDIT_FORMS: &[&str] = &[
    "cp a copy",
    "cp a sub",
    "cp a b sub/",
    "cp -t sub a b",
    "cp --target-directory=sub a",
    "cp --target-directory sub a",
    "cp -b a b",
    "cp --backup a b",
    "cp --backup=numbered a b",
    "cp -S .bak -b a b",
    "cp --suffix=.bak a b",
    "cp -r sub copy",
    "cp -R sub/ copy",
    "cp sub copy -a -v",
    "cp -pf -- a b",
    "cp --no-clobber --verbose a b",
    "cp --verb a b",
    "cp -l a hard",
    "cp -s a soft",
    "cp a ~/home_copy",
    "cp a $HOME/home_copy",
    r#"cp a "${HOME}/home copy""#,
    r#"cp a ~/"home copy""#,
    r#"cp a "~""#,
    r"mkdir \~ && touch '~'/in_literal ~/in_home",
    r#"rm ~"/a""#,
    "mv a moved",
    "mv a sub",
    "mv sub moved",
    "mv -t sub a b",
    "mv --target-directory=sub a",
    "mv -b a b",
    "mv a b -fv",
    "mv a a2 && mv a2 a3",
    "rm a",
    "rm -- a b",
    "rm a -f",
    "rm -rfv sub",
    "rm -r --force sub/",
    "rm -d sub",
    "rm *",
    "rm sub/*",
    "rm -rf ./*",
    "rm ~/a",
    "mkdir made",
    "mkdir -p x/y/z",
    "mkdir -m 700 made",
    "mkdir -pm700 x/y",
    "mkdir -vp x/y sub/z",
    "mkdir --mode=700 made",
    "touch new",
    "touch new -c",
    "touch -c a",
    "touch -r a b",
    "touch --reference=a b",
    "touch -d yesterday a",
    "touch -dyesterday a",
    "touch -t 202001010000 a new",
    "touch -am a",
    "chmod 600 a",
    "chmod 600 a -v",
    "chmod -R 700 sub",
    "chmod -x a",
    "chmod a-w -- a",
    "chmod u+x,g-r a b",
    "chmod --reference=a b",
    "tee out < a",
    "tee -a b out < a",
    "tee --append b < a",
    "tee -- -x < a",
    "tee --output-error=exit out < a",
    "echo hi | tee out | wc -c",
    "ln a hard",
    "ln -s a soft",
    "mkdir -p out && touch out/x && cp a out/y && rm b",
    "cd sub && touch x",
    "cd sub; rm e",
    "(cd sub && rm e); touch y",
];

/// Read-only programs given the options that make them write, in each spelling.
const READ_ONLY_FORMS: &[&str] = &[
    "sort a",
    "sort -o out a",
    "sort -oout a",
    "sort --output=out a",
    "sort --output out a",
    "sort --out=out a",
    "sort a -o out",
    "sort -u -o out a",
    "sort -uo out a",
    "sort -k1,1 -t: a",
    "find . -name a",
    "find . -name a -delete",
    "find sub -delete",
    "find . -name a -fprint found",
    "find . -fprint0 found",
    "find . -fprintf found %p",
    "find sub -type f -fls listing",
    "find . -name a -exec cat {} +",
    "find . -not -name -exec -delete",
    r"find . -name a -exec echo + -delete \;",
    "find . -name a -exec echo {} + -delete",
    "find -- . -delete",
    "sort -S 1M -T . a",
    "sort -ko a",
    "uniq a out",
    "split a",
    "dd if=a of=out",
    "head -c1 a",
    "grep -r line . --include=a",
    "diff a b",
    "wc -l a b sub/e",
    "cat a b | sort | uniq -c",
    "base64 a | base64 -d",
    "history -w dumped",
    "umask 077; cat a",
];

/// Redirections in the forms bash writes them.
const REDIRECT_FORMS: &[&str] = &[
    "cat a > out",
    "cat a >out",
    "cat a 1> out",
    "cat a >> b",
    "cat a >| b",
    "cat a &> both",
    "cat a &>> both",
    "cat a >& both",
    "cat a > out 2>&1",
    "cat a 2> errors",
    "cat a 2>errors >out",
    "cat a > out > other",
    "cat <> opened",
    "cat a 3<> opened",
    "> bare",
    ": > truncated",
    "echo hi > 'quoted name'",
    "echo hi > ~/home_out",
    "echo hi > $HOME/home_out",
    r#"echo hi > "$HOME/home_out""#,
    r#"echo hi > ~/"home out""#,
    "echo hi > '~'",
    "echo hi > sub/../escaped",
    "echo hi > ./sub/e",
    "echo hi > /dev/null",
    "echo hi > /dev/stdout",
    "echo hi >&2",
    "echo hi 2>&-",
    "{ echo hi; } > grouped",
    "( echo hi ) > subshelled",
    r#"for f in a b; do cat "$f"; done > looped"#,
    "while read -r l; do :; done < a > while_out",
    "if true; then echo hi; fi > if_out",
    "f() { echo hi; } > fn_out
f",
    "cat << EOF > heredoc_out
hi
EOF",
    "cat <<< hi > herestring_out",
    "exec > replaced
echo hi",
    "exec 3> descriptor",
    "echo hi |& tee both",
    "cat a > >(tee substituted)",
    "grep -r line . > found",
    "mapfile -t lines < a > mapped",
    "read -r x < a > read_out",
];

/// Read-only programs, `xargs` and redirections in the forms that read a file or open a network
/// connection, run for what they open.
const READ_FORMS: &[&str] = &[
    "cat guarded",
    "cat < guarded",
    "exec 3< guarded",
    "while read -r line; do :; done < guarded",
    "grep -r line .",
    "grep -r -e line -e other",
    "grep -r -A 1 line",
    "grep -r --regexp line",
    "grep -r -e -- -e line",
    "grep -r line sub",
    "grep line *",
    "ls -l guarded",
    "stat guarded",
    "du -sh *",
    "find . -name guarded -newer a",
    "date -f guarded",
    "date -fguarded",
    "date -uf guarded",
    "date --file=guarded",
    "date --file guarded",
    "date --fi=guarded",
    "du --files0-from=guarded",
    "du --files0-from guarded",
    "du --files0=guarded",
    "du -X guarded",
    "du -Xguarded",
    "du --exclude-from=guarded",
    "du --time-style -- --files0-from=guarded",
    "o=--files0-from=guarded; du $o",
    r#"o=--files0-from=guarded; du "$o""#,
    "o='-s --files0-from=guarded'; du ./$o",
    r#"d=sub; du -s "./$d""#,
    "xargs -a guarded",
    "xargs -a guarded echo",
    r"xargs -aguarded printf '%s\n'",
    "xargs -0a guarded echo",
    "xargs --arg-file=guarded echo",
    "xargs --arg-file guarded true",
    "xargs --arg=guarded echo",
    "xargs -a guarded -I{} echo {}",
    "grep -f guarded a",
    "grep -fguarded a",
    "grep -ifguarded a",
    "grep -e -- -fguarded a",
    "grep -ef -fguarded a",
    "fgrep -fguarded a",
    "hexdump -f guarded a",
    "hexdump -fguarded a",
    "diff -X guarded a b",
    "diff -Xguarded a b",
    "find -files0-from guarded",
    "find -files0-from guarded -name a",
    "find -files0-from guarded -exec cat {} +",
    "wc --files0-from=guarded",
    "wc --files0-from{=,=}guarded",
    "n='1 guarded'; head -$n",
    "n='1 guarded'; head -n$n",
    "f=guarded; cat $f",
    "echo guarded > /tmp/list; strings @/tmp/list",
    "HOME=/tmp/run; cat ~/work/guarded",
    "HOME=/tmp/run/work/sub; cat ~/../guarded",
    "HOME=/tmp/run; cd; cat work/guarded",
    "mkdir -- -P; cd -- -P && cat ../guarded",
    "cat < /dev/tcp/127.0.0.1/9",
    "exec 3< /dev/tcp/127.0.0.1/9",
    r#": < "/dev/tcp/127.0.0.1/9""#,
    "h=127.0.0.1; : < /dev/tcp/$h/9",
    "grep -e line --exclude-from guarded a",
    "diff --from-file=guarded a",
    "diff --to-file guarded a",
    "sort --random-source=guarded -R a",
    "sort --files0-from=guarded",
    "wc --files0-from guarded",
    "hexdump --format-file=guarded a",
    r#"p=-fguarded; grep "$p" a"#,
    r#"p=--file=guarded; grep -- "$p" a"#,
    r#"p=guarded; grep -e "$p" a"#,
    r#"n=guarded; head -n "$n" a"#,
    r#"d=--files0-from=guarded; du -- "$d""#,
    r#"s='x --files0-from=guarded'; du --time-style="$s""#,
    "s='x --files0-from=guarded'; du --time-style=$s",
    r#"o=-fguarded; date -d "$o""#,
    r#"o=-fguarded; date "$o""#,
];

/// Programs that run a command; their shims run the real program.
const REAL_WRAPPERS: &[&str] = &["env", "find", "nice", "nohup", "stdbuf", "timeout", "xargs"];

/// Where each corpus script is placed, at `@@`, besides on its own; the last ones define a
/// function or change directory where it never takes effect or stops holding, then go on.
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
    "false && q() {\n@@\n}\nq",
    "(q() {\n@@\n})\nq",
    "q() {\n@@\n} | cat\nq",
    "q() {\n@@\n}\nenv q",
    "(cd /tmp)\n@@",
    "false && cd /tmp\n@@",
    "env cd /tmp\n@@",
    "q() {\n@@\n} && true &\nwait\nq",
    "cd / && true &\nwait\n@@",
    "q() {\n@@\n}\nunset -f q\nq",
    "cd /nonexistent\n@@",
    "! cd /nonexistent && {\n@@\n}",
    "cd /\nq() {\n@@\n}\ncd ~\nq",
    "cd /\nfor i in 1 2; do\n@@\ncd ~\ndone",
];

/// Contexts that single-quote the script for a builtin to evaluate.
const QUOTED_CONTEXTS: &[&str] = &[
    "eval '@@'",
    "trap '@@' EXIT",
    "command eval '@@'",
    "eval() { :; }; set -o posix; eval '@@'",
    "cd /; trap '@@' EXIT; cd ~",
];

const PLACEHOLDER: &str = "@@";

/// Files in the directory each script runs in, so `find` and globs see something.
const WORK_FILES: &[&str] = &["a", "b", "c.txt", "d.o", "sub/e"];

/// A file in the work directory that no one may read, so that opening it is an error.
const GUARDED_FILE: &str = "guarded";

/// What a program prints on being refused [`GUARDED_FILE`].
const OPEN_REFUSED: &str = "Permission denied";

/// What bash prints when a connection it opened for a redirection fails.
const CONNECT_FAILED: &str = "connect: ";

const SANDBOX_REAL_BIN: &str = "/tmp/real";

const SANDBOX_BASH: &str = "/tmp/real/bash";

const SANDBOX_RUN: &str = "/tmp/run";

const SANDBOX_WORK: &str = "/tmp/run/work";

const RUN_SECONDS: &str = "10";

/// Per-run scratch directories, and the shim directory put over `/usr/bin` when `shimmed`.
struct Harness {
    root: TempDir,
    shimmed: bool,
}

impl Harness {
    /// A harness whose scripts run a shim in place of each of `names`.
    fn shimmed(names: &BTreeSet<String>) -> Self {
        let root = tempfile::tempdir().expect("tempdir");
        let bin = root.path().join("bin");
        fs::create_dir(&bin).expect("bin dir");
        let shim = root.path().join("shim");
        fs::write(&shim, shim_script()).expect("write shim");
        fs::set_permissions(&shim, fs::Permissions::from_mode(0o755)).expect("chmod shim");
        for name in names {
            fs::hard_link(&shim, bin.join(name)).expect("link shim");
        }
        Self {
            root,
            shimmed: true,
        }
    }

    /// A harness whose scripts run the real programs.
    fn real() -> Self {
        Self {
            root: tempfile::tempdir().expect("tempdir"),
            shimmed: false,
        }
    }

    /// Run `id`'s directory, with a work directory holding the [`WORK_FILES`].
    fn seeded_run(&self, id: usize) -> PathBuf {
        let run = self.root.path().join(format!("run{id}"));
        let work = run.join("work");
        for file in WORK_FILES {
            let path = work.join(file);
            fs::create_dir_all(path.parent().expect("work file parent")).expect("work dir");
            fs::write(path, format!("{file} line\n")).expect("work file");
        }
        run
    }

    /// The shimmed programs bash started, directly or through a wrapper, running `script`.
    fn runs(&self, id: usize, script: &str) -> Vec<String> {
        let run = self.seeded_run(id);
        self.sandboxed_bash(&run, script)
            .status()
            .expect("run bwrap");
        fs::read_to_string(run.join("log"))
            .unwrap_or_default()
            .lines()
            .map(str::to_string)
            .collect()
    }

    /// The paths below the work directory that running `script` changed outside `declared`,
    /// other than a directory made on the way to one of `declared`.
    fn undeclared_changes(&self, id: usize, script: &str, declared: &[PathBuf]) -> Vec<String> {
        let run = self.seeded_run(id);
        let work = run.join("work");
        let before = tree(&work);
        self.sandboxed_bash(&run, script)
            .status()
            .expect("run bwrap");
        let after = tree(&work);
        let is_declared = |path: &Path| {
            let made_dir =
                !before.contains_key(path) && matches!(after.get(path), Some(Entry::Dir { .. }));
            declared
                .iter()
                .any(|target| path.starts_with(target) || (made_dir && target.starts_with(path)))
        };
        before
            .keys()
            .chain(after.keys().filter(|path| !before.contains_key(*path)))
            .filter(|path| before.get(*path) != after.get(*path) && !is_declared(path))
            .map(|path| path.display().to_string())
            .collect()
    }

    /// What bash and the programs it started printed to stderr running `script` beside
    /// [`GUARDED_FILE`].
    fn stderr_beside_guarded_file(&self, id: usize, script: &str) -> String {
        let run = self.seeded_run(id);
        let guarded = run.join("work").join(GUARDED_FILE);
        fs::write(&guarded, "guarded line\n").expect("guarded file");
        fs::set_permissions(&guarded, fs::Permissions::from_mode(0o000)).expect("chmod guarded");
        let output = self
            .sandboxed_bash(&run, script)
            .stderr(Stdio::piped())
            .output()
            .expect("run bwrap");
        String::from_utf8_lossy(&output.stderr).into_owned()
    }

    /// Bash running `script` with any shims over `/usr/bin`, the real programs at
    /// [`SANDBOX_REAL_BIN`], and `run` writable at [`SANDBOX_RUN`], on a read-only root.
    fn sandboxed_bash(&self, run: &Path, script: &str) -> Command {
        let mut command = Command::new("timeout");
        command
            .args(["-s", "KILL", RUN_SECONDS, "bwrap", "--ro-bind", "/", "/"])
            .args(["--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp"])
            .args(["--ro-bind", "/usr/bin", SANDBOX_REAL_BIN]);
        if self.shimmed {
            command.arg("--ro-bind").args([
                self.root.path().join("bin").as_path(),
                Path::new("/usr/bin"),
            ]);
        }
        command
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

/// What a tree holds at one path.
#[derive(PartialEq)]
enum Entry {
    Dir {
        mode: u32,
    },
    File {
        mode: u32,
        modified: SystemTime,
        contents: Vec<u8>,
    },
    Link(PathBuf),
}

#[test]
fn classified_scripts_run_only_the_programs_their_class_allows() {
    if !bwrap_runs() {
        return;
    }
    let scripts: Vec<String> = CORPUS.iter().flat_map(|script| variants(script)).collect();
    let harness = Harness::shimmed(&shim_names(&scripts));
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
    let edits = of_class(|class| *class == Class::Edit);
    assert!(
        !read_only.is_empty() && !edits.is_empty(),
        "the classifier found no read-only or no edit script"
    );
    let not_read_only = |name: &str| !READ_ONLY.contains(&name) && !REAL_WRAPPERS.contains(&name);
    let not_edit = |name: &str| not_read_only(name) && !EDIT.contains(&name);
    let mismatches: Vec<_> = run_all(&read_only, &|id, script| {
        disallowed_runs(&harness, id, script, &not_read_only)
    })
    .into_iter()
    .chain(run_all(&edits, &|id, script| {
        disallowed_runs(&harness, id, script, &not_edit)
    }))
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

#[test]
fn classified_scripts_change_only_the_paths_their_effects_declare() {
    if !bwrap_runs() {
        return;
    }
    let scripts: Vec<String> = EDIT_FORMS
        .iter()
        .chain(READ_ONLY_FORMS)
        .chain(REDIRECT_FORMS)
        .flat_map(|script| variants(script))
        .collect();
    let vetted: Vec<(&str, Vec<PathBuf>)> = scripts
        .iter()
        .filter_map(|script| {
            let analysis = analyze(script, None, None);
            (!matches!(analysis.class, Class::Unvetted { .. }))
                .then(|| (script.as_str(), declared_changes(&analysis.effects)))
        })
        .collect();
    assert!(
        vetted.iter().any(|(_, declared)| declared.is_empty())
            && vetted.iter().any(|(_, declared)| !declared.is_empty()),
        "the classifier found no read-only or no edit script"
    );
    let harness = Harness::real();
    let mismatches = run_all(&vetted, &|id, (script, declared)| {
        harness.undeclared_changes(id, script, declared)
    });
    assert!(
        mismatches.is_empty(),
        "classified scripts changed paths their effects do not declare:\n{}",
        mismatches
            .iter()
            .map(|((script, declared), changed)| format!(
                "{changed:?}, declaring {declared:?}, from:\n{script}\n"
            ))
            .collect::<String>()
    );
}

#[test]
fn classified_scripts_open_only_the_files_their_effects_declare() {
    if !bwrap_runs() {
        return;
    }
    let scripts: Vec<String> = READ_FORMS
        .iter()
        .flat_map(|script| variants(script))
        .collect();
    let vetted: Vec<(&str, bool)> = scripts
        .iter()
        .filter_map(|script| {
            let analysis = analyze(script, None, None);
            (!matches!(analysis.class, Class::Unvetted { .. })).then(|| {
                let declared = declares_read(&analysis.effects, Path::new(GUARDED_FILE));
                (script.as_str(), declared)
            })
        })
        .collect();
    assert!(
        vetted.iter().any(|(_, declared)| *declared)
            && vetted.iter().any(|(_, declared)| !declared),
        "the classifier found no script that reads the guarded file or none that does not"
    );
    let harness = Harness::real();
    let seen = undeclared_opens(
        &harness,
        vetted.len(),
        "cat guarded; : < /dev/tcp/127.0.0.1/9",
        false,
    );
    assert_eq!(
        seen.len(),
        2,
        "an open or a connection went unseen: {seen:?}"
    );
    let mismatches = run_all(&vetted, &|id, (script, declared)| {
        undeclared_opens(&harness, id, script, *declared)
    });
    assert!(
        mismatches.is_empty(),
        "classified scripts opened what their effects do not declare:\n{}",
        mismatches
            .iter()
            .map(|((script, _), opened)| format!("{opened:?} from:\n{script}\n"))
            .collect::<String>()
    );
}

/// What running `script` opened that a vetted script may not: [`GUARDED_FILE`] unless it
/// `declares_guarded`, or a network connection.
fn undeclared_opens(
    harness: &Harness,
    id: usize,
    script: &str,
    declares_guarded: bool,
) -> Vec<String> {
    let stderr = harness.stderr_beside_guarded_file(id, script);
    let opened_guarded = stderr
        .lines()
        .any(|line| line.contains(GUARDED_FILE) && line.contains(OPEN_REFUSED));
    let undeclared = (opened_guarded && !declares_guarded).then(|| GUARDED_FILE.to_string());
    let connection = stderr
        .contains(CONNECT_FAILED)
        .then(|| "a network connection".to_string());
    undeclared.into_iter().chain(connection).collect()
}

/// Whether `effects` read `file`, below the work directory, or all of a directory holding it.
fn declares_read(effects: &[Effect], file: &Path) -> bool {
    effects
        .iter()
        .filter(|effect| effect.action == Action::Read)
        .any(|effect| {
            below_work(&effect.path)
                .is_some_and(|path| path == file || (effect.recursive && file.starts_with(&path)))
        })
}

/// The programs `script` started that are `disallowed`.
fn disallowed_runs(
    harness: &Harness,
    id: usize,
    script: &str,
    disallowed: &dyn Fn(&str) -> bool,
) -> Vec<String> {
    let mut ran = harness.runs(id, script);
    ran.retain(|name| disallowed(name));
    ran
}

/// Run `offences` for every item on a pool of threads, each with an index of its own, returning
/// the items that have any, with them.
fn run_all<'s, T: Sync>(
    items: &'s [T],
    offences: &(dyn Fn(usize, &'s T) -> Vec<String> + Sync),
) -> Vec<(&'s T, Vec<String>)> {
    let threads = thread::available_parallelism().map_or(4, usize::from);
    let chunk = items.len().div_ceil(threads).max(1);
    thread::scope(|scope| {
        let workers: Vec<_> = items
            .chunks(chunk)
            .enumerate()
            .map(|(worker, batch)| {
                scope.spawn(move || {
                    batch
                        .iter()
                        .enumerate()
                        .map(|(at, item)| (item, offences(worker * chunk + at, item)))
                        .filter(|(_, found)| !found.is_empty())
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

/// Where below the work directory `effects` write, delete or change the mode of.
fn declared_changes(effects: &[Effect]) -> Vec<PathBuf> {
    effects
        .iter()
        .filter(|effect| effect.action != Action::Read)
        .filter_map(|effect| below_work(&effect.path))
        .collect()
}

/// `path`, as an effect spells it, relative to the work directory, which is also `$HOME`;
/// `None` when it lies elsewhere.
fn below_work(path: &str) -> Option<PathBuf> {
    let below = match path.strip_prefix('~') {
        Some(below_home) => below_home,
        None if path.starts_with('/') => path.strip_prefix(SANDBOX_WORK)?,
        None => path,
    };
    let mut cleaned = PathBuf::new();
    for component in below.split('/') {
        match component {
            "" | "." => {}
            ".." => {
                if !cleaned.pop() {
                    return None;
                }
            }
            name => cleaned.push(name),
        }
    }
    Some(cleaned)
}

/// Every entry below `root`, by its path relative to it.
fn tree(root: &Path) -> BTreeMap<PathBuf, Entry> {
    let mut entries = BTreeMap::new();
    let mut pending = vec![root.to_path_buf()];
    while let Some(dir) = pending.pop() {
        for child in fs::read_dir(&dir).into_iter().flatten().flatten() {
            let path = child.path();
            let Ok(meta) = path.symlink_metadata() else {
                continue;
            };
            let mode = meta.permissions().mode();
            let entry = if meta.is_dir() {
                pending.push(path.clone());
                Entry::Dir { mode }
            } else if meta.is_symlink() {
                Entry::Link(fs::read_link(&path).unwrap_or_default())
            } else {
                Entry::File {
                    mode,
                    modified: meta.modified().expect("modification time"),
                    contents: fs::read(&path).unwrap_or_default(),
                }
            };
            let below = path.strip_prefix(root).expect("below the root");
            entries.insert(below.to_path_buf(), entry);
        }
    }
    entries
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
