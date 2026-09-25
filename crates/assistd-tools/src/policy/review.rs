//! Command review: whether a script or argv may run without the user's
//! confirmation.

use std::path::PathBuf;
use std::{fmt, iter, mem};

use super::allowlist::{Allowlist, Verdict};
use super::shell::{self, Script, SimpleCommand, Word};

/// How many scripts deep (`eval`, `trap`, …) the matcher looks before
/// calling a script unverifiable.
const MAX_NESTED_SCRIPTS: usize = 16;

/// Reserved words that leave the word after them in command position.
const COMMAND_PREFIX_WORDS: &[&str] = &[
    "!", "{", "if", "then", "else", "elif", "do", "while", "until",
];

/// Builtins and default-allowed wrappers that run a command given as their
/// arguments, with the number of operands before that command.
const WRAPPERS: &[(&str, usize)] = &[
    ("builtin", 0),
    ("command", 0),
    ("coproc", 0),
    ("env", 0),
    ("exec", 0),
    ("nice", 0),
    ("nohup", 0),
    ("stdbuf", 0),
    ("time", 0),
    ("timeout", 1),
    ("xargs", 0),
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

/// Builtins and keywords whose arguments are checked in their place.
const RUNS_ITS_ARGUMENTS: &[&str] = &[
    ".", "alias", "builtin", "command", "coproc", "eval", "exec", "source", "time", "trap",
];

/// Builtins that can run or load code no check can see, so they always ask.
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
/// name (`env -S 'rm -rf ~'`).
const SHELL_SYNTAX: &str = ";&|()<>`$";

/// Wrapper options that take the next word as their value. Only documented
/// value options belong here: a flag listed by mistake hides the command.
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

/// The wrapper that appends arguments read from stdin to its command.
const FEEDS_ARGUMENTS: &str = "xargs";

/// Where [`FEEDS_ARGUMENTS`] puts an input item inside its command.
const INPUT_PLACEHOLDER: &str = "{}";

/// `find` flags after which it runs a command.
const FIND_EXEC_FLAGS: &[&str] = &["-exec", "-execdir", "-ok", "-okdir"];

/// A destructive command: a name followed by arguments that must all be
/// present, in any order, each word listing `|`-separated alternatives.
/// `-rf` matches those short options in any cluster, `--force` the option
/// or an abbreviation, `of=` any argument with that prefix, anything else
/// itself, all ignoring ASCII case. A run-time argument (`"$f"`) stands in
/// for one required argument; one that may split (`$opts`, `*`) for all.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DestructivePattern {
    display: String,
    names: Vec<String>,
    args: Vec<Vec<ArgSpec>>,
}

impl DestructivePattern {
    /// `None` when there are no words or a word has no alternatives.
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

/// Why a command needs the user's confirmation before it runs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Confirmation {
    /// It runs a command matching this destructive pattern.
    Pattern(String),
    /// It may run something no check can see; holds why, for the prompt.
    Unverifiable(String),
    /// It runs programs not on the allowlist. `approvable` when "always
    /// allow" can add every one of them.
    Unlisted {
        programs: Vec<String>,
        approvable: bool,
    },
}

impl Confirmation {
    /// The programs an "always allow" answer would add; possibly empty.
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
    /// Commands that need confirmation even when every program is allowed.
    pub patterns: &'a [DestructivePattern],
    pub allowlist: &'a Allowlist,
    /// Directories a command may not name without confirmation.
    pub protected: &'a [PathBuf],
}

/// Why `script` needs confirmation, if it does: it matches a destructive
/// pattern, can run something no check can see, or runs a program not on
/// the allowlist (reported in that order). Every command bash would run is
/// checked, including those behind builtins and default-allowed wrappers;
/// what an allowed program runs in turn (`sh -c`, `sudo`) is trusted.
pub fn check_script(script: &str, rules: &Rules<'_>) -> Option<Confirmation> {
    let matcher = Matcher::new(rules);
    let mut findings = Findings::default();
    matcher.script(script, 0, &mut findings);
    findings.into_confirmation()
}

/// [`check_script`] for an argv executed without a shell. Each argument is
/// also matched against the patterns as a script (`xterm -e "rm -rf ~"`).
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

    /// A pattern match outranks anything found after it.
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

#[derive(Clone, Copy)]
struct Site<'s> {
    cmd: &'s SimpleCommand,
    script: &'s Script,
    depth: usize,
}

struct Matcher<'a> {
    patterns: &'a [DestructivePattern],
    allowlist: &'a Allowlist,
    /// Every spelling of the protected directories.
    protected: Vec<String>,
}

impl<'a> Matcher<'a> {
    fn new(rules: &Rules<'a>) -> Self {
        let home = std::env::var("HOME").ok();
        let protected = rules
            .protected
            .iter()
            .filter_map(|dir| dir.to_str())
            .flat_map(|dir| spellings(dir, home.as_deref()))
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

    /// The pattern match in `src`, for text that may not be a script.
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
                self.runs(word, args, Site { cmd, script, depth }, out);
                fed |= program(&word.text) == FEEDS_ARGUMENTS;
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
        if needs_no_allowlist(word, index, script) {
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
    /// builtin, or a command line hidden in an `env` option.
    fn runs(&self, word: &Word, args: &[Word], site: Site<'_>, out: &mut Findings<'a>) {
        let Site { cmd, script, depth } = site;
        let program = program(&word.text);
        match program.as_str() {
            "eval" => {
                let joined: Vec<&str> = args.iter().map(|a| a.text.as_str()).collect();
                self.script(&joined.join(" "), depth + 1, out);
            }
            "source" | "." => {
                if let Some(file) = args.first() {
                    self.sourced(word, file, script.inputs_of(cmd), depth, out);
                }
            }
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
            "env" => {
                if let Some(option) = env_attached_option(args) {
                    out.unverifiable(|| {
                        format!("`env {option}` may run a command no check can see")
                    });
                }
            }
            _ => {}
        }
    }

    /// Check `word file`, where `word` is `source` or `.`.
    fn sourced<'i>(
        &self,
        word: &Word,
        file: &Word,
        inputs: impl Iterator<Item = &'i str>,
        depth: usize,
        out: &mut Findings<'a>,
    ) {
        if file.dynamic {
            out.unverifiable(|| {
                format!(
                    "`{} {}` runs a script only known at run time",
                    word.text, file.text
                )
            });
        } else if is_stdin_path(&file.text) {
            self.stdin_script(&word.text, inputs, depth, out);
        } else {
            out.unlisted(&format!("{} {}", word.text, file.text), false);
        }
    }

    /// A script `name` reads from stdin: the here-documents and
    /// here-strings spelled out for it, or unverifiable when there are none.
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
}

struct Candidate {
    at: usize,
    may_be_command: bool,
}

/// Where [`candidates`] is within a command word's arguments.
#[derive(Default)]
struct Scan {
    /// The command word runs a command after this many more operands.
    pending: Option<usize>,
    /// Flags after which it runs a command (`find -exec`).
    exec_flags: Option<&'static [&'static str]>,
    /// No exec flag has been seen yet, so its arguments are its own.
    before_exec: bool,
    value_options: &'static [&'static str],
    /// The previous word was one of `value_options`.
    value_next: bool,
    /// The previous word was an option that may take this one as its value.
    after_option: bool,
}

impl Scan {
    fn enter(&mut self, word: &Word, next: Option<&Word>) {
        self.pending = wrapper_operands(word, next);
        self.value_options = if word.dynamic_name {
            &[]
        } else {
            value_options(&program(&word.text))
        };
        if let Some(flags) = exec_flags(word) {
            self.exec_flags = Some(flags);
            self.before_exec = true;
        }
    }

    /// Classify the argument `word` at `at`; `None` when it can be neither
    /// a command nor one of its arguments.
    fn classify(&mut self, at: usize, word: &Word, next: Option<&Word>) -> Option<Candidate> {
        if self
            .exec_flags
            .is_some_and(|flags| flags.contains(&word.text.as_str()))
        {
            self.before_exec = false;
            self.pending = Some(0);
            self.after_option = false;
            return None;
        }
        if self.before_exec || mem::take(&mut self.value_next) {
            return None;
        }
        if !word.quoted && word.text == "--" {
            self.pending = self.pending.map(|_| 0);
            self.after_option = false;
            return None;
        }
        let option = !word.dynamic && word.text.starts_with('-');
        let operand = !option && !self.after_option && !is_assignment(&word.text);
        let may_be_command = match self.pending {
            Some(operands) if operand && operands > 0 => {
                self.pending = Some(operands - 1);
                false
            }
            Some(_) if operand => {
                self.enter(word, next);
                true
            }
            _ => {
                self.pending.is_some_and(|operands| {
                    operands == 0 || (self.after_option && is_command_line(word))
                }) && !option
                    && !is_assignment(&word.text)
            }
        };
        self.value_next = option && self.value_options.contains(&word.text.as_str());
        self.after_option = option
            && !self.value_next
            && !word.text.contains('=')
            && (word.text.len() == 2 || word.text.starts_with("--"));
        Some(Candidate { at, may_be_command })
    }
}

/// The words of a simple command that may name a command it runs:
/// `start`, and the words after a wrapper or `find -exec`. Later words are
/// that command's arguments, still checked against patterns.
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
        out.extend(scan.classify(at, word, words.get(at + 1)));
    }
    out
}

/// `dir` as a script may spell it: as is, and via `~` or `$HOME` when it
/// lies under `home`.
fn spellings(dir: &str, home: Option<&str>) -> impl Iterator<Item = String> {
    let under_home = home
        .and_then(|home| dir.strip_prefix(home))
        .filter(|rest| rest.starts_with('/'));
    iter::once(dir.to_string()).chain(
        under_home
            .into_iter()
            .flat_map(|rest| ["~", "$HOME", "${HOME}"].map(|home| format!("{home}{rest}"))),
    )
}

/// Keywords, builtins, and functions defined before command `index`,
/// none of which need the allowlist.
fn needs_no_allowlist(word: &Word, index: usize, script: &Script) -> bool {
    let text = word.text.as_str();
    (!word.quoted && KEYWORDS.contains(&text))
        || SAFE_BUILTINS.contains(&text)
        || RUNS_ITS_ARGUMENTS.contains(&text)
        || script
            .functions
            .iter()
            .any(|(name, defined)| name == text && *defined <= index)
}

/// The first variable the script may change that the allowlist cannot see
/// past: the search path, the dynamic loader's, or bash's startup file.
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

fn alternatives(word: &str) -> impl Iterator<Item = &str> {
    word.split('|').filter(|alt| !alt.is_empty())
}

fn short_options(arg: &str) -> Option<&str> {
    arg.strip_prefix('-')
        .filter(|opts| !opts.is_empty() && !opts.starts_with('-'))
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

/// How many operands `word` reads before the command it runs, if it is a
/// wrapper.
fn wrapper_operands(word: &Word, next: Option<&Word>) -> Option<usize> {
    if word.dynamic_name {
        return None;
    }
    let program = program(&word.text);
    let program = program.as_str();
    if program == "command" && next.is_some_and(|n| matches!(n.text.as_str(), "-v" | "-V")) {
        return None;
    }
    WRAPPERS
        .iter()
        .find(|&&(w, _)| w == program)
        .map(|&(_, operands)| operands)
}

/// A literal word holding a whole command line (`env -S 'rm -rf ~'`),
/// not a program name.
fn is_command_line(word: &Word) -> bool {
    !word.dynamic
        && word
            .text
            .contains(|c: char| c.is_whitespace() || SHELL_SYNTAX.contains(c))
}

fn exec_flags(word: &Word) -> Option<&'static [&'static str]> {
    (!word.dynamic_name && program(&word.text) == "find").then_some(FIND_EXEC_FLAGS)
}

fn value_options(program: &str) -> &'static [&'static str] {
    VALUE_OPTIONS
        .iter()
        .find(|&&(p, _)| p == program)
        .map_or(&[], |&(_, options)| options)
}

/// The first of `env`'s options with a value attached (`-Sx`, `-iS`,
/// `--split-string=x`), any of which may hide an `-S` command line.
fn env_attached_option(args: &[Word]) -> Option<&str> {
    let value_options = value_options("env");
    let mut value_next = false;
    for arg in args {
        let text = arg.text.as_str();
        if mem::take(&mut value_next) {
            continue;
        }
        if !text.starts_with('-') || text == "--" {
            return None;
        }
        if text.contains('=') || (!text.starts_with("--") && text.len() > 2) {
            return Some(text);
        }
        value_next = value_options.contains(&text);
    }
    None
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

/// A command word's last path component, lowercased, without a trailing
/// version (`python3.12` is `python`).
fn program(word: &str) -> String {
    let name = basename(word).to_ascii_lowercase();
    match name.trim_end_matches(|c: char| c.is_ascii_digit() || c == '.' || c == '-') {
        "" => name,
        trimmed => trimmed.to_string(),
    }
}

#[cfg(test)]
mod tests;
