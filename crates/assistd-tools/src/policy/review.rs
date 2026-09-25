//! Command review: whether a script or argv may run without the user's
//! confirmation. It may only when every program it can run is allowed and
//! nothing in it matches a destructive pattern or escapes the check.

use std::path::PathBuf;
use std::{fmt, iter};

use super::allowlist::{Allowlist, Verdict};
use super::shell::{self, Script, SimpleCommand, Word};

/// How many scripts deep (`sh -c`, `eval`, here-documents, …) the matcher
/// looks. A script nested deeper counts as unverifiable.
const MAX_NESTED_SCRIPTS: usize = 16;

/// Reserved words that leave the word after them in command position.
const COMMAND_PREFIX_WORDS: &[&str] = &[
    "!", "{", "if", "then", "else", "elif", "do", "while", "until",
];

/// Shells, which run the argument of `-c` as a script, or a script read
/// from stdin when given no script file.
const SHELLS: &[&str] = &[
    "ash",
    "bash",
    "bosh",
    "csh",
    "dash",
    "elvish",
    "es",
    "fish",
    "hush",
    "ion",
    "jsh",
    "ksh",
    "lksh",
    "loksh",
    "mksh",
    "msh",
    "murex",
    "nu",
    "nushell",
    "oil",
    "oksh",
    "osh",
    "pdksh",
    "posh",
    "powershell",
    "pwsh",
    "rbash",
    "rc",
    "rksh",
    "rzsh",
    "sash",
    "sh",
    "tcsh",
    "xonsh",
    "yash",
    "ysh",
    "zsh",
];

/// Interpreters whose inline or stdin code may call out to a shell. Their
/// string literals are checked as commands.
const INTERPRETERS: &[&str] = &[
    "awk",
    "bun",
    "clisp",
    "deno",
    "elixir",
    "erl",
    "escript",
    "expect",
    "gawk",
    "groovy",
    "guile",
    "irb",
    "jruby",
    "jshell",
    "julia",
    "jython",
    "kotlin",
    "lua",
    "luajit",
    "mawk",
    "nawk",
    "node",
    "nodejs",
    "ocaml",
    "osascript",
    "perl",
    "php",
    "pypy",
    "python",
    "r",
    "racket",
    "rscript",
    "ruby",
    "runghc",
    "runhaskell",
    "sbcl",
    "scala",
    "swift",
    "tclsh",
    "wish",
];

/// Programs that run a command given as their arguments, with the number
/// of operands (a host, a lock file, a subcommand and its target) before
/// that command. Every argument is a potential command name, since a
/// wrapper's options cannot be told from its command without knowing its
/// flags; arguments with spaces or shell syntax are also checked as
/// scripts (`su -c '…'`, `ssh host '…'`, `env -S '…'`).
const WRAPPERS: &[(&str, usize)] = &[
    ("alacritty", 0),
    ("arch", 0),
    ("asdf", 1),
    ("at", 1),
    ("autossh", 1),
    ("batch", 0),
    ("builtin", 0),
    ("bundle", 1),
    ("bunx", 0),
    ("busybox", 0),
    ("bwrap", 0),
    ("caffeinate", 0),
    ("capsh", 0),
    ("catatonit", 0),
    ("catchsegv", 0),
    ("cgexec", 0),
    ("chpst", 0),
    ("chronic", 0),
    ("chroot", 1),
    ("chrt", 1),
    ("command", 0),
    ("concurrently", 0),
    ("conda", 1),
    ("coproc", 0),
    ("cpulimit", 0),
    ("daemonize", 0),
    ("dbus-launch", 0),
    ("dbus-run-session", 0),
    ("direnv", 2),
    ("distrobox", 2),
    ("distrobox-host-exec", 0),
    ("doas", 0),
    ("docker", 2),
    ("dumb-init", 0),
    ("eatmydata", 0),
    ("entr", 0),
    ("env", 0),
    ("envdir", 1),
    ("envuidgid", 1),
    ("exec", 0),
    ("fakechroot", 0),
    ("fakeroot", 0),
    ("faketime", 1),
    ("firejail", 0),
    ("flatpak", 2),
    ("flatpak-spawn", 0),
    ("flock", 1),
    ("foot", 0),
    ("footclient", 0),
    ("gamemoderun", 0),
    ("gdb", 0),
    ("ghostty", 0),
    ("gnome-terminal", 0),
    ("gosu", 1),
    ("gtimeout", 1),
    ("guake", 0),
    ("host-spawn", 0),
    ("hyperfine", 0),
    ("hyprctl", 0),
    ("i3-msg", 0),
    ("incus", 2),
    ("ionice", 0),
    ("kgx", 0),
    ("kitty", 0),
    ("konsole", 0),
    ("kubectl", 2),
    ("linux", 0),
    ("ltrace", 0),
    ("lxc", 2),
    ("lxc-attach", 0),
    ("lxterminal", 0),
    ("machinectl", 2),
    ("mangohud", 0),
    ("mate-terminal", 0),
    ("mise", 1),
    ("mosh", 1),
    ("nice", 0),
    ("nix", 2),
    ("nix-shell", 0),
    ("nocache", 0),
    ("nodemon", 0),
    ("nohup", 0),
    ("npm", 1),
    ("npx", 0),
    ("nq", 0),
    ("nsenter", 0),
    ("numactl", 0),
    ("optirun", 0),
    ("parallel", 0),
    ("perf", 1),
    ("pipenv", 1),
    ("pixi", 1),
    ("pkexec", 0),
    ("please", 0),
    ("pnpm", 1),
    ("pnpx", 0),
    ("podman", 2),
    ("poetry", 1),
    ("prime-run", 0),
    ("primusrun", 0),
    ("prlimit", 0),
    ("proot", 0),
    ("proxychains", 0),
    ("pueue", 1),
    ("pyenv", 1),
    ("qterminal", 0),
    ("rbenv", 1),
    ("riverctl", 0),
    ("rlwrap", 0),
    ("rr", 0),
    ("rsh", 1),
    ("run", 0),
    ("runuser", 0),
    ("rxvt", 0),
    ("s6-setuidgid", 1),
    ("sakura", 0),
    ("sandbox-exec", 0),
    ("schroot", 0),
    ("screen", 0),
    ("script", 0),
    ("setarch", 1),
    ("setpriv", 0),
    ("setsid", 0),
    ("setuidgid", 1),
    ("sg", 1),
    ("softlimit", 0),
    ("ssh", 1),
    ("sshpass", 0),
    ("st", 0),
    ("stdbuf", 0),
    ("strace", 0),
    ("su", 1),
    ("su-exec", 1),
    ("sudo", 0),
    ("sudo-rs", 0),
    ("swaymsg", 0),
    ("systemd-nspawn", 0),
    ("systemd-run", 0),
    ("taskset", 1),
    ("terminator", 0),
    ("terminology", 0),
    ("tilix", 0),
    ("time", 0),
    ("timeout", 1),
    ("tini", 0),
    ("tmux", 1),
    ("toolbox", 1),
    ("torsocks", 0),
    ("toybox", 0),
    ("trickle", 0),
    ("tsocks", 0),
    ("tsp", 0),
    ("unbuffer", 0),
    ("unshare", 0),
    ("urxvt", 0),
    ("uv", 1),
    ("uxterm", 0),
    ("vagrant", 1),
    ("valgrind", 0),
    ("vglrun", 0),
    ("watch", 0),
    ("watchexec", 0),
    ("wezterm", 1),
    ("x-terminal-emulator", 0),
    ("xargs", 0),
    ("xfce4-terminal", 0),
    ("xterm", 0),
    ("xvfb-run", 0),
    ("yarn", 1),
];

/// Wrappers whose first operand is a subcommand (`docker rm`), which is
/// not itself a command.
const SUBCOMMAND_WRAPPERS: &[&str] = &[
    "asdf",
    "bundle",
    "conda",
    "direnv",
    "distrobox",
    "docker",
    "flatpak",
    "incus",
    "kubectl",
    "lxc",
    "machinectl",
    "mise",
    "nix",
    "npm",
    "perf",
    "pipenv",
    "pixi",
    "pnpm",
    "podman",
    "poetry",
    "pueue",
    "pyenv",
    "rbenv",
    "tmux",
    "toolbox",
    "uv",
    "vagrant",
    "wezterm",
    "yarn",
];

/// Reserved words, which run nothing themselves when unquoted.
const KEYWORDS: &[&str] = &[
    "!", "[[", "]]", "{", "}", "case", "do", "done", "elif", "else", "esac", "fi", "for",
    "function", "if", "in", "select", "then", "until", "while",
];

/// Builtins that run no code and reach nothing outside the shell.
const SAFE_BUILTINS: &[&str] = &[
    ":", "[", "bg", "break", "caller", "cd", "compopt", "continue", "declare", "dirs", "disown",
    "echo", "exit", "export", "false", "fg", "getopts", "hash", "help", "history", "jobs", "let",
    "local", "logout", "popd", "printf", "pushd", "pwd", "read", "readonly", "return", "set",
    "shift", "shopt", "suspend", "test", "times", "true", "type", "typeset", "ulimit", "umask",
    "unalias", "unset", "wait",
];

/// Builtins and keywords that run their arguments, which are checked in
/// their place.
const RUNS_ITS_ARGUMENTS: &[&str] = &[
    ".", "alias", "builtin", "command", "coproc", "eval", "exec", "source", "time", "trap",
];

/// Builtins that can run or load code no check can see (`enable -f`,
/// `mapfile -C`, `fc`), so they always ask.
const RISKY_BUILTINS: &[&str] = &[
    "bind",
    "compgen",
    "complete",
    "enable",
    "fc",
    "mapfile",
    "readarray",
];

/// Characters that make one word a command line rather than a program
/// name (`su -c 'rm -rf ~'`).
const SHELL_SYNTAX: &str = ";&|()<>`$";

/// Options of the wrappers allowed by default that always take the next
/// word as their value, so that word is never the command. Listing an
/// option that is really a flag would hide the command after it, so only
/// documented value options belong here.
const VALUE_OPTIONS: &[(&str, &[&str])] = &[
    ("env", &["-u", "--unset", "-C", "--chdir"]),
    ("nice", &["-n", "--adjustment"]),
    (
        "stdbuf",
        &["-i", "-o", "-e", "--input", "--output", "--error"],
    ),
    ("timeout", &["-s", "--signal", "-k", "--kill-after"]),
    (
        "xargs",
        &[
            "-a",
            "--arg-file",
            "-d",
            "--delimiter",
            "-E",
            "-I",
            "-L",
            "-n",
            "--max-args",
            "-P",
            "--max-procs",
            "-s",
            "--max-chars",
            "--process-slot-var",
        ],
    ),
];

/// Wrappers that append arguments read from stdin to the command they run.
const FEEDS_ARGUMENTS: &[&str] = &["parallel", "xargs"];

/// Where those wrappers put an input item inside the command instead.
const INPUT_PLACEHOLDER: &str = "{}";

/// Programs that run a command given after one of these flags.
const EXEC_FLAGS: &[(&str, &[&str])] = &[
    ("fd", &["-x", "-X", "--exec", "--exec-batch"]),
    ("fdfind", &["-x", "-X", "--exec", "--exec-batch"]),
    ("find", &["-exec", "-execdir", "-ok", "-okdir"]),
    ("gfind", &["-exec", "-execdir", "-ok", "-okdir"]),
];

/// A configured destructive command: a command name followed by arguments
/// that must all be present, in any order.
///
/// Any word may list alternatives separated by `|`. The name matches a
/// command word or the last component of its path. An argument matches:
/// - `-rf`: short options, each of which may sit in any option cluster
///   (`-r -f`, `-vfr`);
/// - `--force`: the long option, an abbreviation of it (`--forc`), or
///   `--force=…`;
/// - `of=`: any argument starting with it;
/// - anything else: that exact argument.
///
/// All comparisons ignore ASCII case. An argument whose value is only
/// known at run time (`"$f"`) may stand in for any one required argument,
/// and one that may split into several words (`$opts`, `*`) for all of
/// them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DestructivePattern {
    display: String,
    names: Vec<String>,
    args: Vec<Vec<ArgSpec>>,
}

impl DestructivePattern {
    /// Build a pattern from its words. `None` when there are no words or a
    /// word has no alternatives.
    pub fn new<I, S>(words: I) -> Option<Self>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let words: Vec<String> = words.into_iter().map(|w| w.as_ref().to_string()).collect();
        let (name, args) = words.split_first()?;
        let names: Vec<String> = alternatives(name).map(str::to_ascii_lowercase).collect();
        if names.is_empty() {
            return None;
        }
        let args = args
            .iter()
            .map(|arg| {
                let specs: Vec<ArgSpec> = alternatives(arg).map(ArgSpec::parse).collect();
                (!specs.is_empty()).then_some(specs)
            })
            .collect::<Option<Vec<_>>>()?;
        Some(Self {
            display: words.join(" "),
            names,
            args,
        })
    }

    fn invoked_by(&self, command: &Word, args: &[Word], fed: bool) -> bool {
        let program = basename(&command.text);
        if !self
            .names
            .iter()
            .any(|n| command.text.eq_ignore_ascii_case(n) || program.eq_ignore_ascii_case(n))
        {
            return false;
        }
        if fed || args.iter().any(|a| a.splits) {
            return true;
        }
        let unknown = args.iter().filter(|a| a.dynamic).count();
        let missing = self
            .args
            .iter()
            .filter(|alts| !alts.iter().any(|spec| spec.satisfied_by(args)))
            .count();
        missing <= unknown
    }
}

impl fmt::Display for DestructivePattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.display)
    }
}

fn alternatives(word: &str) -> impl Iterator<Item = &str> {
    word.split('|').filter(|alt| !alt.is_empty())
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ArgSpec {
    ShortFlags(String),
    Long(String),
    Prefix(String),
    Exact(String),
}

impl ArgSpec {
    fn parse(alt: &str) -> Self {
        let alt = alt.to_ascii_lowercase();
        if alt.ends_with('=') {
            return Self::Prefix(alt);
        }
        if let Some(long) = alt.strip_prefix("--").filter(|l| !l.is_empty()) {
            return Self::Long(long.to_string());
        }
        match alt.strip_prefix('-') {
            Some(flags) if !flags.is_empty() && flags.chars().all(|c| c.is_ascii_alphabetic()) => {
                Self::ShortFlags(flags.to_string())
            }
            _ => Self::Exact(alt),
        }
    }

    fn satisfied_by(&self, args: &[Word]) -> bool {
        match self {
            Self::ShortFlags(flags) => flags.chars().all(|flag| {
                args.iter().any(|a| {
                    short_options(&a.text)
                        .is_some_and(|opts| opts.chars().any(|o| o.eq_ignore_ascii_case(&flag)))
                })
            }),
            Self::Long(name) => args.iter().any(|a| {
                a.text
                    .strip_prefix("--")
                    .map(|opt| opt.split_once('=').map_or(opt, |(opt, _)| opt))
                    .is_some_and(|opt| {
                        !opt.is_empty()
                            && name
                                .get(..opt.len())
                                .is_some_and(|abbrev| abbrev.eq_ignore_ascii_case(opt))
                    })
            }),
            Self::Prefix(prefix) => args.iter().any(|a| {
                a.text
                    .get(..prefix.len())
                    .is_some_and(|head| head.eq_ignore_ascii_case(prefix))
            }),
            Self::Exact(exact) => args.iter().any(|a| a.text.eq_ignore_ascii_case(exact)),
        }
    }
}

fn short_options(arg: &str) -> Option<&str> {
    arg.strip_prefix('-')
        .filter(|opts| !opts.is_empty() && !opts.starts_with('-'))
}

/// Why a command needs the user's confirmation before it runs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Confirmation {
    /// It runs a command matching this destructive pattern.
    Pattern(String),
    /// It may run something no check can see: a command named only at
    /// run time, a script read from stdin, text too broken or deeply
    /// nested to follow, anything after a change to `PATH` or the dynamic
    /// loader, or a touch of assistd's own configuration. Holds a
    /// description for the prompt.
    Unverifiable(String),
    /// It runs programs that are not on the allowlist. `approvable` when
    /// every one of them is a bare name "always allow" can add.
    Unlisted {
        programs: Vec<String>,
        approvable: bool,
    },
}

impl Confirmation {
    /// The programs an "always allow" answer would add; empty when the
    /// prompt cannot be settled that way.
    pub fn always_allow(&self) -> &[String] {
        match self {
            Self::Unlisted {
                programs,
                approvable: true,
            } => programs,
            _ => &[],
        }
    }
}

impl fmt::Display for Confirmation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pattern(pattern) => f.write_str(pattern),
            Self::Unverifiable(why) => f.write_str(why),
            Self::Unlisted { programs, .. } => {
                write!(f, "not on the allowlist: {}", programs.join(", "))
            }
        }
    }
}

/// What commands are checked against.
#[derive(Debug, Clone, Copy)]
pub struct Rules<'a> {
    /// Commands that need confirmation even when every program they run
    /// is allowed.
    pub patterns: &'a [DestructivePattern],
    /// Programs that run without confirmation.
    pub allowlist: &'a Allowlist,
    /// Directories a command may not name without confirmation.
    pub protected: &'a [PathBuf],
}

/// Whether `script` needs confirmation before it runs, and why. It does
/// when it can run a program not on the allowlist, matches a destructive
/// pattern, or can run something no check can see. A pattern match is
/// reported first, then an unverifiable command, then unlisted programs.
///
/// Commands are found wherever bash would run them: after `;`, `&&`, `|`
/// or a newline, in subshells and `$(…)`/backquote substitutions (also
/// inside double quotes and here-documents), after reserved words such as
/// `if` or `{`, after `NAME=value` assignments and redirections, after
/// wrappers (`sudo`, `env`, `xargs`, `ssh`, …) and `find -exec`, and inside
/// scripts handed to a shell (`sh -c`, `bash <<EOF`), `eval`, `trap` or
/// `alias`. Each must be a shell builtin that runs no code, a function the
/// script defined earlier, a known shell whose script is checked in turn,
/// or on the allowlist. A script file (`./x.sh`, `bash x.sh`, `source x`)
/// is never on it, since the file can change. String literals in
/// interpreter code (`python -c`, `perl -e`) are checked against the
/// patterns, as commands and as argv lists. Quoted text elsewhere stays
/// one word, so `echo "rm -rf"` does not match.
///
/// Unverifiable: a command whose name is only known at run time (`$cmd`,
/// `$(echo rm)`, `/bin/r?`), a shell reading its script from stdin, a
/// script ending inside a quote or substitution, nesting beyond what the
/// matcher follows, a change to `PATH`, `BASH_ENV` or the dynamic loader
/// (`LD_*`), and any word naming a protected directory.
pub fn check_script(script: &str, rules: &Rules<'_>) -> Option<Confirmation> {
    let matcher = Matcher::new(rules);
    let mut findings = Findings::default();
    matcher.script(script, 0, &mut findings);
    findings.into_confirmation()
}

/// [`check_script`] for an argv that is executed directly rather than
/// through a shell. Each argument is also checked on its own as a script
/// against the patterns, for one that smuggles a command to the program
/// it is handed to (`xterm -e "rm -rf ~"`).
pub fn check_argv(argv: &[String], rules: &Rules<'_>) -> Option<Confirmation> {
    let matcher = Matcher::new(rules);
    let mut findings = Findings::default();
    let command = SimpleCommand {
        words: argv.iter().map(|arg| Word::literal(arg)).collect(),
        ..SimpleCommand::default()
    };
    let script = Script::default();
    matcher.protected_words(&command, &mut findings);
    matcher.command(&command, 0, &script, 0, &mut findings);
    for arg in argv {
        if findings.settled() {
            break;
        }
        if let Some(pattern) = matcher.patterns_in(arg, 1) {
            findings.pattern(pattern);
        }
    }
    findings.into_confirmation()
}

/// Everything found in a script so far.
#[derive(Default)]
struct Findings<'a> {
    pattern: Option<&'a DestructivePattern>,
    unverifiable: Option<String>,
    unlisted: Vec<String>,
    unapprovable: bool,
}

impl<'a> Findings<'a> {
    fn pattern(&mut self, pattern: &'a DestructivePattern) {
        if self.pattern.is_none() {
            self.pattern = Some(pattern);
        }
    }

    fn unverifiable(&mut self, why: impl FnOnce() -> String) {
        if self.unverifiable.is_none() {
            self.unverifiable = Some(why());
        }
    }

    fn unlisted(&mut self, program: &str, approvable: bool) {
        self.unapprovable |= !approvable;
        if !self.unlisted.iter().any(|p| p == program) {
            self.unlisted.push(program.to_string());
        }
    }

    /// A pattern match is reported ahead of anything else, so nothing
    /// found after it would change the prompt.
    fn settled(&self) -> bool {
        self.pattern.is_some()
    }

    fn into_confirmation(self) -> Option<Confirmation> {
        if let Some(pattern) = self.pattern {
            return Some(Confirmation::Pattern(pattern.to_string()));
        }
        if let Some(why) = self.unverifiable {
            return Some(Confirmation::Unverifiable(why));
        }
        (!self.unlisted.is_empty()).then_some(Confirmation::Unlisted {
            programs: self.unlisted,
            approvable: !self.unapprovable,
        })
    }
}

/// The command a word sits in, and the script around it.
#[derive(Clone, Copy)]
struct Site<'s> {
    cmd: &'s SimpleCommand,
    script: &'s Script,
    depth: usize,
}

struct Matcher<'a> {
    patterns: &'a [DestructivePattern],
    allowlist: &'a Allowlist,
    /// The protected directories as a script may spell them.
    protected: Vec<String>,
}

impl<'a> Matcher<'a> {
    fn new(rules: &Rules<'a>) -> Self {
        let home = std::env::var("HOME").ok();
        let protected =
            rules
                .protected
                .iter()
                .filter_map(|dir| dir.to_str())
                .flat_map(|dir| {
                    let under_home = home
                        .as_deref()
                        .and_then(|home| dir.strip_prefix(home))
                        .filter(|rest| rest.starts_with('/'));
                    iter::once(dir.to_string()).chain(under_home.into_iter().flat_map(|rest| {
                        ["~", "$HOME", "${HOME}"].map(|home| format!("{home}{rest}"))
                    }))
                })
                .collect();
        Self {
            patterns: rules.patterns,
            allowlist: rules.allowlist,
            protected,
        }
    }

    fn script(&self, src: &str, depth: usize, out: &mut Findings<'a>) {
        let too_deep = || "the script nests too deeply to check".to_string();
        if depth > MAX_NESTED_SCRIPTS {
            out.unverifiable(too_deep);
            return;
        }
        let Ok(script) = shell::parse(src) else {
            out.unverifiable(too_deep);
            return;
        };
        if script.incomplete {
            out.unverifiable(|| "the script ends inside a quote or substitution".into());
        }
        if let Some(variable) = changed_sensitive_variable(&script) {
            out.unverifiable(|| {
                format!("the script changes {variable}, so the programs it names cannot be checked")
            });
        }
        for (index, cmd) in script.commands.iter().enumerate() {
            if out.settled() {
                return;
            }
            self.protected_words(cmd, out);
            self.command(cmd, index, &script, depth, out);
        }
    }

    /// Only the pattern matches in `src`, for text that may not be a
    /// script at all.
    fn patterns_in(&self, src: &str, depth: usize) -> Option<&'a DestructivePattern> {
        let mut scratch = Findings::default();
        self.script(src, depth, &mut scratch);
        scratch.pattern
    }

    fn protected_words(&self, cmd: &SimpleCommand, out: &mut Findings<'a>) {
        if let Some(word) = cmd
            .words
            .iter()
            .chain(&cmd.redirects)
            .find(|w| self.protected.iter().any(|p| w.text.contains(p.as_str())))
        {
            out.unverifiable(|| format!("`{}` touches assistd's own configuration", word.text));
        }
    }

    fn command(
        &self,
        cmd: &SimpleCommand,
        index: usize,
        script: &Script,
        depth: usize,
        out: &mut Findings<'a>,
    ) {
        let words = &cmd.words;
        let Some(start) = command_start(words) else {
            return;
        };
        let mut fed = false;
        for Candidate { at, may_be_command } in candidates(words, start) {
            if out.settled() {
                return;
            }
            let word = &words[at];
            if word.dynamic_name || (fed && word.text.contains(INPUT_PLACEHOLDER)) {
                if may_be_command {
                    out.unverifiable(|| {
                        format!("command `{}` is only known at run time", word.text)
                    });
                }
                continue;
            }
            let args = &words[at + 1..];
            if let Some(pattern) = self.patterns.iter().find(|p| p.invoked_by(word, args, fed)) {
                out.pattern(pattern);
                return;
            }
            if may_be_command {
                self.allowed(word, index, script, depth, out);
                self.runs(word, args, Site { cmd, script, depth }, fed, out);
                fed |= FEEDS_ARGUMENTS.contains(&program(&word.text).as_str());
            }
        }
    }

    /// Record `word`, in command position, when what it runs is not
    /// allowed.
    fn allowed(
        &self,
        word: &Word,
        index: usize,
        script: &Script,
        depth: usize,
        out: &mut Findings<'a>,
    ) {
        let text = word.text.as_str();
        if word.dynamic {
            out.unverifiable(|| format!("program `{text}` is only known at run time"));
            return;
        }
        if is_command_line(word) {
            self.script(text, depth + 1, out);
            return;
        }
        if (!word.quoted && KEYWORDS.contains(&text))
            || SAFE_BUILTINS.contains(&text)
            || RUNS_ITS_ARGUMENTS.contains(&text)
            || script
                .functions
                .iter()
                .any(|(name, defined)| name == text && *defined <= index)
            || (SHELLS.contains(&basename(text)) && self.allowlist.trusted(text))
        {
            return;
        }
        if RISKY_BUILTINS.contains(&text) {
            out.unlisted(text, false);
            return;
        }
        match self.allowlist.verdict(text) {
            Verdict::Allowed => {}
            Verdict::Missing if self.allowlist.search_path_fixed() => {}
            Verdict::Missing => out.unlisted(text, false),
            Verdict::Unlisted { approvable } => out.unlisted(text, approvable),
        }
    }

    /// What `word`, run with `args`, runs in turn: a script handed to a
    /// shell, code handed to an interpreter, or commands embedded in a
    /// wrapper's arguments. `fed` says an earlier wrapper fills in
    /// arguments from its input.
    fn runs(&self, word: &Word, args: &[Word], site: Site<'_>, fed: bool, out: &mut Findings<'a>) {
        let Site { cmd, script, depth } = site;
        let program = program(&word.text);
        let program = program.as_str();
        let inputs = script.inputs_of(cmd);
        if SHELLS.contains(&program) {
            self.shell(&word.text, args, inputs, fed, depth, out);
            return;
        }
        if INTERPRETERS.contains(&program) {
            if let Some(pattern) = args
                .iter()
                .map(|a| a.text.as_str())
                .chain(inputs)
                .find_map(|code| self.code(code, depth + 1))
            {
                out.pattern(pattern);
            }
            return;
        }
        match program {
            "eval" => {
                let joined: Vec<&str> = args.iter().map(|a| a.text.as_str()).collect();
                self.script(&joined.join(" "), depth + 1, out);
            }
            "source" | "." => match args.first() {
                Some(file) if file.dynamic => out.unverifiable(|| {
                    format!(
                        "`{} {}` runs a script only known at run time",
                        word.text, file.text
                    )
                }),
                Some(file) if is_stdin_path(&file.text) => {
                    self.stdin_script(&word.text, inputs, depth, out);
                }
                Some(file) => out.unlisted(&format!("{} {}", word.text, file.text), false),
                None => {}
            },
            "hash"
                if args
                    .iter()
                    .any(|a| short_options(&a.text).is_some_and(|o| o.contains('p'))) =>
            {
                out.unverifiable(|| "`hash -p` makes a command name run another program".into());
            }
            "trap" => {
                if let Some(action) = args.iter().find(|a| !a.text.starts_with('-')) {
                    self.script(&action.text, depth + 1, out);
                }
            }
            "alias" => {
                for (_, body) in args.iter().filter_map(|a| a.text.split_once('=')) {
                    self.script(body, depth + 1, out);
                }
            }
            _ if WRAPPERS.iter().any(|&(w, _)| w == program) => {
                if let Some(pattern) = args
                    .iter()
                    .flat_map(|a| embedded_scripts(&a.text))
                    .chain(inputs)
                    .find_map(|s| self.patterns_in(s, depth + 1))
                {
                    out.pattern(pattern);
                }
            }
            _ => {}
        }
    }

    fn shell<'i>(
        &self,
        name: &str,
        args: &[Word],
        inputs: impl Iterator<Item = &'i str>,
        fed: bool,
        depth: usize,
        out: &mut Findings<'a>,
    ) {
        let mut command_string = false;
        let mut reads_stdin = false;
        let mut operands = Vec::new();
        let mut value_next = false;
        for arg in args {
            if std::mem::take(&mut value_next) {
                continue;
            }
            let text = arg.text.as_str();
            if arg.dynamic || text == "-" || !text.starts_with(['-', '+']) {
                reads_stdin |= is_stdin_path(text);
                operands.push(arg);
            } else if let Some(long) = text.strip_prefix("--") {
                value_next = matches!(long, "rcfile" | "init-file");
            } else {
                let flags = &text[1..];
                if text.starts_with('-') {
                    command_string |= flags.contains('c');
                    reads_stdin |= flags.contains(['s', 'i']);
                }
                value_next = flags.ends_with(['o', 'O']);
            }
        }
        let filled_in = |op: &Word| fed && op.text.contains(INPUT_PLACEHOLDER);
        let unknown =
            |op: &Word| format!("`{name}` runs a script only known at run time: {}", op.text);
        if command_string {
            for op in operands {
                if filled_in(op) {
                    out.unverifiable(|| unknown(op));
                } else {
                    self.script(&op.text, depth + 1, out);
                }
            }
            return;
        }
        match operands.iter().find(|op| !is_stdin_path(&op.text)) {
            Some(file) if !reads_stdin => {
                if file.dynamic || filled_in(file) {
                    out.unverifiable(|| unknown(file));
                } else {
                    out.unlisted(&format!("{name} {}", file.text), false);
                }
            }
            _ => self.stdin_script(name, inputs, depth, out),
        }
    }

    /// A script `name` reads from stdin: the here-documents and
    /// here-strings the script spells out, or unverifiable when stdin is
    /// a pipe or a file.
    fn stdin_script<'i>(
        &self,
        name: &str,
        inputs: impl Iterator<Item = &'i str>,
        depth: usize,
        out: &mut Findings<'a>,
    ) {
        let mut inputs = inputs.peekable();
        if inputs.peek().is_none() {
            out.unverifiable(|| format!("`{name}` runs a script read from stdin"));
        }
        for input in inputs {
            self.script(input, depth + 1, out);
        }
    }

    /// Pattern matches in interpreter code: each string literal checked
    /// as a script and for literals of its own, then the literals in
    /// order as an argv (`["rm", "-rf", path]`).
    fn code(&self, code: &str, depth: usize) -> Option<&'a DestructivePattern> {
        if depth > MAX_NESTED_SCRIPTS {
            return None;
        }
        let literals = string_literals(code);
        literals
            .iter()
            .find_map(|lit| {
                self.patterns_in(lit, depth + 1)
                    .or_else(|| self.code(lit, depth + 1))
            })
            .or_else(|| {
                let words: Vec<Word> = literals.iter().map(|lit| Word::literal(lit)).collect();
                (0..words.len()).find_map(|at| {
                    self.patterns
                        .iter()
                        .find(|p| p.invoked_by(&words[at], &words[at + 1..], false))
                })
            })
    }
}

/// The first variable the script may change whose change the allowlist
/// cannot see past: the search path, what the dynamic loader injects, or
/// what bash sources at startup.
fn changed_sensitive_variable(script: &Script) -> Option<&str> {
    script
        .commands
        .iter()
        .flat_map(|cmd| &cmd.words)
        .flat_map(|word| changed_variables(&word.text))
        .find(|name| matches!(*name, "PATH" | "BASH_ENV" | "GCONV_PATH") || name.starts_with("LD_"))
}

/// Variables a word may assign, wherever it sits: `NAME=…`, a bare
/// `NAME` (`read NAME`, `for NAME in`, `export NAME`), a nameref target
/// (`declare -n ref=NAME`), or a `${NAME:=…}` default.
fn changed_variables(text: &str) -> impl Iterator<Item = &str> {
    let whole = shell::is_name(text).then_some(text);
    let (assigned, value) = match text.split_once('=') {
        Some((name, value)) => {
            let name = name.strip_suffix('+').unwrap_or(name);
            let name = name.split_once('[').map_or(name, |(base, _)| base);
            (
                shell::is_name(name).then_some(name),
                shell::is_name(value).then_some(value),
            )
        }
        None => (None, None),
    };
    let defaults = text.match_indices("${").filter_map(|(at, _)| {
        let rest = &text[at + 2..];
        let end = rest.find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))?;
        let (name, tail) = rest.split_at(end);
        (tail.starts_with(":=") || tail.starts_with('=')).then_some(name)
    });
    whole
        .into_iter()
        .chain(assigned)
        .chain(value)
        .chain(defaults)
}

fn command_start(words: &[Word]) -> Option<usize> {
    let mut at = 0;
    while let Some(word) = words.get(at) {
        let bare = !word.quoted && !word.dynamic;
        if is_assignment(&word.text) || (bare && COMMAND_PREFIX_WORDS.contains(&word.text.as_str()))
        {
            at += 1;
        } else if bare && word.text == "function" {
            at += 2;
        } else {
            return Some(at);
        }
    }
    None
}

#[derive(Debug, Clone, Copy, Default)]
struct Wrapping {
    operands: usize,
    subcommand: bool,
}

fn wrapping(word: &Word, next: Option<&Word>) -> Option<Wrapping> {
    if word.dynamic_name {
        return None;
    }
    let program = program(&word.text);
    let program = program.as_str();
    if program == "command" && next.is_some_and(|n| matches!(n.text.as_str(), "-v" | "-V")) {
        return None;
    }
    if SHELLS.contains(&program) || INTERPRETERS.contains(&program) {
        return Some(Wrapping {
            operands: 0,
            subcommand: false,
        });
    }
    WRAPPERS
        .iter()
        .find(|&&(w, _)| w == program)
        .map(|&(_, operands)| Wrapping {
            operands,
            subcommand: SUBCOMMAND_WRAPPERS.contains(&program),
        })
}

struct Candidate {
    at: usize,
    may_be_command: bool,
}

/// Where [`candidates`] is within a command word's arguments.
#[derive(Default)]
struct Scan {
    /// The command word runs a command still to come, after this many
    /// more operands.
    pending: Option<Wrapping>,
    /// Its next operand is a subcommand, not a command.
    subcommand: bool,
    /// Flags after which it runs a command (`find -exec`).
    exec_flags: Option<&'static [&'static str]>,
    /// No exec flag has been seen yet, so its arguments are its own.
    before_exec: bool,
    /// Its options known to take a value.
    value_options: &'static [&'static str],
    /// The previous word was one of those, so this one is its value.
    value_next: bool,
    /// The previous word was an option that may take this one as its
    /// value.
    after_option: bool,
}

impl Scan {
    fn enter(&mut self, word: &Word, next: Option<&Word>) {
        self.pending = wrapping(word, next);
        self.subcommand = self.pending.is_some_and(|p| p.subcommand);
        self.value_options = (!word.dynamic_name)
            .then(|| program(&word.text))
            .and_then(|program| VALUE_OPTIONS.iter().find(|&&(p, _)| p == program))
            .map_or(&[], |&(_, options)| options);
        if let Some(flags) = exec_flags(word) {
            self.exec_flags = Some(flags);
            self.before_exec = true;
        }
    }
}

/// The words of a simple command that may name a command it runs:
/// `start`, and the words after a wrapper or `find -exec`. A word "may be
/// the command" while a wrapper is still reading its options and
/// operands; later words are that command's arguments, still checked
/// against patterns since a wrapper's flags are not known.
fn candidates(words: &[Word], start: usize) -> Vec<Candidate> {
    let mut out = vec![Candidate {
        at: start,
        may_be_command: true,
    }];
    let mut scan = Scan::default();
    scan.enter(&words[start], words.get(start + 1));
    if scan.pending.is_none() && scan.exec_flags.is_none() {
        return out;
    }
    for (at, word) in words.iter().enumerate().skip(start + 1) {
        if scan
            .exec_flags
            .is_some_and(|flags| flags.contains(&word.text.as_str()))
        {
            scan.before_exec = false;
            scan.pending = Some(Wrapping::default());
            scan.after_option = false;
            continue;
        }
        if scan.before_exec || std::mem::take(&mut scan.value_next) {
            continue;
        }
        if !word.quoted && word.text == "--" {
            scan.pending = scan.pending.map(|p| Wrapping { operands: 0, ..p });
            scan.after_option = false;
            continue;
        }
        let option = !word.dynamic && word.text.starts_with('-');
        let operand = !option && !scan.after_option && !is_assignment(&word.text);
        match scan.pending {
            Some(p) if operand && p.operands > 0 => {
                if !std::mem::take(&mut scan.subcommand) {
                    out.push(Candidate {
                        at,
                        may_be_command: false,
                    });
                }
                scan.pending = Some(Wrapping {
                    operands: p.operands - 1,
                    ..p
                });
            }
            Some(_) if operand => {
                out.push(Candidate {
                    at,
                    may_be_command: true,
                });
                scan.enter(word, words.get(at + 1));
            }
            _ => out.push(Candidate {
                at,
                may_be_command: scan.pending.is_some_and(|p| {
                    p.operands == 0 || (scan.after_option && is_command_line(word))
                }) && !option
                    && !is_assignment(&word.text),
            }),
        }
        scan.value_next = option && scan.value_options.contains(&word.text.as_str());
        scan.after_option = option
            && !scan.value_next
            && !word.text.contains('=')
            && (word.text.len() == 2 || word.text.starts_with("--"));
    }
    out
}

/// A literal word holding a whole command line (`su -c 'rm -rf ~'`),
/// not a program name.
fn is_command_line(word: &Word) -> bool {
    !word.dynamic
        && word
            .text
            .contains(|c: char| c.is_whitespace() || SHELL_SYNTAX.contains(c))
}

fn exec_flags(word: &Word) -> Option<&'static [&'static str]> {
    if word.dynamic_name {
        return None;
    }
    let program = program(&word.text);
    EXEC_FLAGS
        .iter()
        .find(|&&(p, _)| p == program)
        .map(|&(_, flags)| flags)
}

/// Scripts an argument to a wrapper may carry: the argument itself, or
/// the value of an option (`--split-string=…`, `-c…`), when it holds
/// spaces or shell syntax.
fn embedded_scripts(arg: &str) -> impl Iterator<Item = &str> {
    let value = if arg.starts_with("--") {
        arg.split_once('=').map(|(_, value)| value)
    } else if arg.starts_with('-') {
        arg.get(2..)
    } else {
        None
    };
    [Some(arg), value]
        .into_iter()
        .flatten()
        .filter(|s| s.contains(|c: char| c.is_whitespace() || ";&|()<>`$".contains(c)))
}

/// The contents of every `'…'`, `"…"` and `` `…` `` literal in `code`, as
/// written.
fn string_literals(code: &str) -> Vec<&str> {
    let mut literals = Vec::new();
    let mut chars = code.char_indices();
    while let Some((_, quote)) = chars.next() {
        if !matches!(quote, '\'' | '"' | '`') {
            continue;
        }
        let start = chars.offset();
        let mut end = code.len();
        while let Some((at, c)) = chars.next() {
            if c == '\\' {
                chars.next();
            } else if c == quote {
                end = at;
                break;
            }
        }
        literals.push(&code[start..end]);
    }
    literals
}

fn is_assignment(word: &str) -> bool {
    word.split_once('=').is_some_and(|(name, _)| {
        let name = name.strip_suffix('+').unwrap_or(name);
        let name = name.split_once('[').map_or(name, |(base, _)| base);
        name.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_')
            && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
    })
}

fn is_stdin_path(path: &str) -> bool {
    matches!(path, "-" | "/dev/stdin")
        || path.starts_with("/dev/fd/")
        || path.starts_with("/proc/self/fd/")
}

fn basename(word: &str) -> &str {
    word.rsplit_once('/').map_or(word, |(_, name)| name)
}

/// The program a command word runs, normalized for the lists above: its
/// last path component, lowercased, without a trailing version
/// (`python3.12` is `python`, `ksh93` is `ksh`).
fn program(word: &str) -> String {
    let name = basename(word).to_ascii_lowercase();
    match name.trim_end_matches(|c: char| c.is_ascii_digit() || c == '.' || c == '-') {
        "" => name,
        trimmed => trimmed.to_string(),
    }
}

#[cfg(test)]
mod tests;
