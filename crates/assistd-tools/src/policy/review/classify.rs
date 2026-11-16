//! Sorts a script into read-only, edit or unvetted by the programs it runs and the files its
//! redirections write, using short lists of programs whose effects are known and the user's
//! "always allow" approvals, and lists the paths those programs read, write, delete or change
//! the mode or owner of, and the paths an unvetted command names.

use std::mem;
use std::path::Path;

use super::placeholder::{self, Placeholder};
use super::{
    Candidate, FEEDS_ARGUMENTS, FIND_EXEC_FLAGS, KEYWORDS, MAX_NESTED_SCRIPTS, SAFE_BUILTINS,
    WRAPPERS, basename, candidates, changed_sensitive_variable, changed_variables, command_start,
    env_attached_option, evaluated, exec_flags, is_assignment, is_command_line, is_harmless_target,
    is_stdin_path, is_subcommand, program, short_options, wrapper_operands,
};
use crate::policy::allowlist::{Allowlist, ApprovalScope, ProgramApproval, Verdict};
use crate::policy::shell::{self, Script, SimpleCommand, Word};

/// Builtins that change the working directory, after which relative targets are unknown.
const CHANGES_DIRECTORY: &[&str] = &["cd", "popd", "pushd"];

/// Wrappers that are reserved words, so quoting one runs a program of that name instead.
const WRAPPER_KEYWORDS: &[&str] = &["coproc", "time"];

/// Words after which `find` runs a command up to a `;` or `+`.
const FIND_COMMAND_ENDS: &[&str] = &[";", "+"];

/// Characters after which the text of a word is only known at run time.
const DYNAMIC_CHARS: &[char] = &['$', '`', '*', '?', '[', '{', '('];

/// The spellings of `$HOME` a path may start with.
const HOME_VARIABLES: &[&str] = &["${HOME}", "$HOME"];

/// Variables a script may export without leaving it unvetted, besides `LC_*`: the locale,
/// time zone, terminal and colour settings, which run nothing.
const HARMLESS_EXPORTS: &[&str] = &["LANG", "LANGUAGE", "NO_COLOR", "TERM", "TZ"];

/// Builtins that export the names they declare when given `-x`.
const EXPORTING_DECLARATIONS: &[&str] = &["declare", "local", "readonly", "typeset"];

/// Characters that start an expansion whose value may name any path, `..` and absolute paths
/// included; a glob only matches existing entries below its directory.
const ANY_PATH_CHARS: &[char] = &['$', '`', '{'];

/// Programs that only read, each with the options that make it write a file or run a program.
const READ_ONLY_PROGRAMS: &[(&str, &[Flag])] = &[
    ("b2sum", &[]),
    ("base64", &[]),
    ("basename", &[]),
    ("cat", &[]),
    ("cksum", &[]),
    ("cmp", &[]),
    ("column", &[]),
    ("comm", &[]),
    ("cut", &[]),
    ("date", &[Flag::Short('s'), Flag::Long("set")]),
    ("df", &[]),
    ("diff", &[]),
    ("dirname", &[]),
    ("du", &[]),
    ("egrep", &[]),
    ("expand", &[]),
    ("fgrep", &[]),
    (
        "find",
        &[
            Flag::Primary("-delete"),
            Flag::Primary("-fls"),
            Flag::Primary("-fprint"),
            Flag::Primary("-fprint0"),
            Flag::Primary("-fprintf"),
        ],
    ),
    ("fold", &[]),
    ("grep", &[]),
    ("head", &[]),
    ("hexdump", &[]),
    ("id", &[]),
    ("join", &[]),
    ("ls", &[]),
    ("md5sum", &[]),
    ("nl", &[]),
    ("od", &[]),
    ("paste", &[]),
    ("printenv", &[]),
    ("readlink", &[]),
    ("realpath", &[]),
    ("rev", &[]),
    ("seq", &[]),
    ("sha1sum", &[]),
    ("sha256sum", &[]),
    ("sha512sum", &[]),
    ("sleep", &[]),
    (
        "sort",
        &[
            Flag::Short('o'),
            Flag::Long("output"),
            Flag::Long("compress-program"),
        ],
    ),
    ("stat", &[]),
    ("strings", &[]),
    ("tac", &[]),
    ("tail", &[]),
    ("tr", &[]),
    ("uname", &[]),
    ("unexpand", &[]),
    ("wc", &[]),
    ("which", &[]),
    ("whoami", &[]),
];

/// Read-only programs whose operands are not files whose contents they read.
const READS_NO_CONTENTS: &[&str] = &[
    "basename", "date", "df", "dirname", "du", "id", "ls", "printenv", "readlink", "realpath",
    "seq", "sleep", "stat", "uname", "which", "whoami",
];

/// `grep`'s options that make it read every file below a directory.
const GREP_RECURSIVE: &[Flag] = &[
    Flag::Short('d'),
    Flag::Short('r'),
    Flag::Short('R'),
    Flag::Long("dereference-recursive"),
    Flag::Long("directories"),
    Flag::Long("recursive"),
];

/// Read-only programs that read every file below a directory operand given one of these options.
const RECURSIVE_READERS: &[(&str, &[Flag])] = &[
    ("diff", &[Flag::Short('r'), Flag::Long("recursive")]),
    ("egrep", GREP_RECURSIVE),
    ("fgrep", GREP_RECURSIVE),
    ("grep", GREP_RECURSIVE),
];

/// The programs that take `grep`'s options.
const GREP_PROGRAMS: &[&str] = &["egrep", "fgrep", "grep"];

/// `grep`'s options that take a value, which may be the next word.
const GREP_VALUE_FLAGS: &[Flag] = &[
    Flag::Short('A'),
    Flag::Short('B'),
    Flag::Short('C'),
    Flag::Short('D'),
    Flag::Short('X'),
    Flag::Short('d'),
    Flag::Short('e'),
    Flag::Short('f'),
    Flag::Short('m'),
    Flag::Long("after-context"),
    Flag::Long("before-context"),
    Flag::Long("binary-files"),
    Flag::Long("context"),
    Flag::Long("devices"),
    Flag::Long("directories"),
    Flag::Long("exclude"),
    Flag::Long("exclude-dir"),
    Flag::Long("exclude-from"),
    Flag::Long("file"),
    Flag::Long("group-separator"),
    Flag::Long("include"),
    Flag::Long("label"),
    Flag::Long("max-count"),
    Flag::Long("regexp"),
];

/// `grep`'s options that give its pattern, leaving every operand a file to search.
const GREP_PATTERN_FLAGS: &[Flag] = &[
    Flag::Short('e'),
    Flag::Short('f'),
    Flag::Long("file"),
    Flag::Long("regexp"),
];

/// `find` options that come before its starting points.
const FIND_LEADING_OPTIONS: &[&str] = &["-H", "-L", "-P"];

/// `grep`'s option naming a file of patterns.
const GREP_PATTERN_FILE: &[Flag] = &[Flag::Short('f')];

/// Read-only programs' options naming a file they read, where its value would go unseen:
/// attached to a short option, or given to a program whose operands are not read.
const FILE_OPTIONS: &[(&str, &[Flag])] = &[
    ("date", &[Flag::Short('f'), Flag::Long("file")]),
    ("diff", &[Flag::Short('X')]),
    (
        "du",
        &[
            Flag::Short('X'),
            Flag::Long("exclude-from"),
            Flag::Long("files0-from"),
        ],
    ),
    ("egrep", GREP_PATTERN_FILE),
    ("fgrep", GREP_PATTERN_FILE),
    ("find", &[Flag::Primary("-files0-from")]),
    ("grep", GREP_PATTERN_FILE),
    ("hexdump", &[Flag::Short('f')]),
];

/// Shells that run the script after `-c`, or on their stdin when given no operand.
const SHELLS: &[&str] = &["ash", "bash", "dash", "ksh", "mksh", "sh", "zsh"];

/// Programs that replace an `@file` argument with the arguments that file holds.
const READS_ARGUMENT_FILES: &[&str] = &["strings"];

/// Paths a redirection opens as a network connection to the host and port below them.
const NETWORK_PATHS: &[&str] = &["/dev/tcp/", "/dev/udp/"];

/// Programs that write or delete the literal paths they name, with the options vetted for each.
const EDIT_PROGRAMS: &[EditProgram] = &[
    EditProgram {
        name: "chmod",
        flags: "cfvR",
        value_flags: "",
        long_flags: &[
            "changes",
            "no-preserve-root",
            "preserve-root",
            "quiet",
            "recursive",
            "silent",
            "verbose",
        ],
        targets: Targets::ChangesAllButFirst,
    },
    EditProgram {
        name: "cp",
        flags: "adfHiLnPprRTuvx",
        value_flags: "",
        long_flags: &[
            "archive",
            "dereference",
            "force",
            "interactive",
            "no-clobber",
            "no-dereference",
            "no-target-directory",
            "one-file-system",
            "recursive",
            "update",
            "verbose",
        ],
        targets: Targets::WritesLast,
    },
    EditProgram {
        name: "mkdir",
        flags: "pv",
        value_flags: "m",
        long_flags: &["parents", "verbose"],
        targets: Targets::WritesAll,
    },
    EditProgram {
        name: "mv",
        flags: "finTuv",
        value_flags: "",
        long_flags: &[
            "force",
            "interactive",
            "no-clobber",
            "no-target-directory",
            "update",
            "verbose",
        ],
        targets: Targets::Moves,
    },
    EditProgram {
        name: "rm",
        flags: "dfIirRv",
        value_flags: "",
        long_flags: &[
            "dir",
            "force",
            "interactive",
            "no-preserve-root",
            "one-file-system",
            "preserve-root",
            "recursive",
            "verbose",
        ],
        targets: Targets::Removes,
    },
    EditProgram {
        name: "tee",
        flags: "aip",
        value_flags: "",
        long_flags: &["append", "ignore-interrupts"],
        targets: Targets::WritesAll,
    },
    EditProgram {
        name: "touch",
        flags: "achm",
        value_flags: "dt",
        long_flags: &["no-create", "no-dereference"],
        targets: Targets::WritesAll,
    },
];

/// What a script may do, as far as the read-only and edit lists can tell.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Class {
    /// It runs only read-only programs, and redirects only to standard streams, `/dev/null`,
    /// `/tmp` or the scratch directory.
    ReadOnly,
    /// It also runs edit programs or redirects elsewhere; targets are unresolved, as written or
    /// as where a copy or move lands.
    Edit {
        writes: Vec<String>,
        deletes: Vec<Delete>,
    },
    /// It may do something neither list nor an approval vets; `why` names the first such thing.
    Unvetted { why: String },
}

/// A path a script deletes, as written.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Delete {
    pub path: String,
    /// The whole tree under it may go (`rm -r`, or `mv` of what may be a directory).
    pub recursive: bool,
}

/// A script's class and the file operations found in it, including those of commands that
/// left it unvetted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Analysis {
    pub class: Class,
    pub effects: Vec<Effect>,
    /// The sources of its copies and moves, spelled as [`Effect::path`] is: each lands
    /// elsewhere as it is, a symlink still a link and a tree with every entry it holds.
    pub carried: Vec<String>,
    /// What "always allow" would approve so that an unvetted script is vetted next time; empty
    /// when it is vetted, or when something other than a program approvals can cover leaves it
    /// unvetted.
    pub always_allow: Vec<ProgramApproval>,
}

/// A file operation a script performs on `path`: as written, with `$HOME` spelled `~`, or
/// joined to the directory an earlier `cd` entered. `~` is left unexpanded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Effect {
    pub action: Action,
    pub path: String,
    /// It reaches every entry below `path`, as `rm -r` or `grep -r` do.
    pub recursive: bool,
}

/// What an [`Effect`] does to its path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Read,
    Write,
    Delete,
    /// `chmod`.
    ChangeAttributes,
    /// Whatever a command no list vets does to a path it names.
    Unknown,
}

/// An option a program may be given.
#[derive(Debug, Clone, Copy)]
enum Flag {
    /// A short option, alone or in a cluster.
    Short(char),
    /// A GNU long option, which may be abbreviated.
    Long(&'static str),
    /// A `find` primary.
    Primary(&'static str),
}

impl Flag {
    fn matched_by(self, option: &str) -> bool {
        self.value_in(option).is_some()
    }

    /// Where `option` puts this flag's value, if it gives the flag.
    fn value_in(self, option: &str) -> Option<OptionValue<'_>> {
        let value = match self {
            Self::Short(flag) => {
                let (_, attached) = short_options(option)?.split_once(flag)?;
                Some(attached).filter(|attached| !attached.is_empty())
            }
            Self::Long(name) => {
                let given = option.strip_prefix("--")?;
                let (given, attached) = match given.split_once('=') {
                    Some((given, attached)) => (given, Some(attached)),
                    None => (given, None),
                };
                if given.is_empty() || !name.starts_with(given) {
                    return None;
                }
                attached
            }
            Self::Primary(name) => {
                if option != name {
                    return None;
                }
                None
            }
        };
        Some(value.map_or(OptionValue::NextWord, OptionValue::Attached))
    }
}

/// Where an option's value is, for a [`Flag`] that takes one.
#[derive(Debug, Clone, Copy)]
enum OptionValue<'o> {
    /// In the option's own word, after the flag or its `=`.
    Attached(&'o str),
    NextWord,
}

/// A program that writes or deletes the paths it names.
struct EditProgram {
    name: &'static str,
    /// Short flags it may be given.
    flags: &'static str,
    /// Short options that take a value, attached or in the next word.
    value_flags: &'static str,
    /// Long options it may be given, spelled in full.
    long_flags: &'static [&'static str],
    targets: Targets,
}

/// Which operands an [`EditProgram`] writes or deletes.
#[derive(Clone, Copy)]
enum Targets {
    /// Deletes every operand.
    Removes,
    /// Writes every operand.
    WritesAll,
    /// Reads all but its last operand, which it writes.
    WritesLast,
    /// Deletes all but its last operand, which it writes.
    Moves,
    /// Changes the mode of every operand after the first, which names the change.
    ChangesAllButFirst,
}

/// An [`EditProgram`]'s arguments.
#[derive(Default)]
struct EditArgs {
    /// Its operands that are paths known before it runs.
    operands: Vec<String>,
    recursive: bool,
    /// Why its first option not vetted for it, or operand only known at run time, is neither.
    unvetted: Option<String>,
}

/// How the program being classified is run.
#[derive(Default)]
struct Invocation<'w> {
    /// `xargs` appends arguments it reads from stdin.
    fed: bool,
    /// `find` or `xargs` runs it once per item, where or with what it chooses.
    per_item: bool,
    placeholders: Vec<Placeholder<'w>>,
}

impl Invocation<'_> {
    fn fills(&self, word: &Word) -> bool {
        self.placeholders.iter().any(|p| p.fills(word))
    }
}

/// Where a simple command sits in the script being classified.
#[derive(Clone, Copy)]
struct Position<'s> {
    cmd: &'s SimpleCommand,
    script: &'s Script,
    index: usize,
    depth: usize,
}

/// The directory relative paths are resolved against, as far as the script shows.
#[derive(Debug, Default)]
enum Directory {
    /// The one the script started in.
    #[default]
    Start,
    /// One a literal `cd` entered, as written, joined to any earlier one.
    At(String),
    Unknown,
}

impl Directory {
    fn entered(&self, dir: String) -> Self {
        match self {
            _ if dir.starts_with(['/', '~']) => Self::At(dir),
            Self::Start => Self::At(dir),
            Self::At(current) => Self::At(joined_path(current, &dir)),
            Self::Unknown => Self::Unknown,
        }
    }
}

#[derive(Default)]
struct Classifier<'a> {
    scratch: Option<&'a str>,
    approvals: Option<&'a Allowlist>,
    effects: Vec<Effect>,
    carried: Vec<String>,
    runs_edit_program: bool,
    directory: Directory,
    /// The script may change `HOME`, so `$HOME` is not `~`.
    home_changed: bool,
    /// The script may change `CDPATH`, so where `cd` goes is unknown.
    cdpath_changed: bool,
    /// The first relative target named once the directory may have changed.
    relative_after_cd: Option<String>,
    /// Why the first command neither list vets is unvetted.
    unvetted: Option<String>,
    /// How many times something has left the script unvetted.
    unvetted_findings: usize,
    /// Approvals that would vet the programs on neither list.
    always_allow: Vec<ProgramApproval>,
    /// Something approvals cannot cover leaves the script unvetted.
    unapprovable: bool,
}

impl Classifier<'_> {
    fn script(&mut self, src: &str, depth: usize) -> Result<(), String> {
        let too_deep = || "the script nests too deeply to vet".to_string();
        if depth > MAX_NESTED_SCRIPTS {
            return Err(too_deep());
        }
        let script = shell::parse(src).map_err(|_| too_deep())?;
        if script.incomplete {
            return Err("the script ends inside a quote or substitution".into());
        }
        self.vet(unvettable_throughout(&script));
        self.home_changed |= changes(&script, "HOME");
        self.cdpath_changed |= changes(&script, "CDPATH");
        for (index, cmd) in script.commands.iter().enumerate() {
            let redirected = self.redirects(cmd);
            self.vet(redirected);
            let position = Position {
                cmd,
                script: &script,
                index,
                depth,
            };
            let findings = self.unvetted_findings;
            let ran = self.command(position);
            self.vet(ran);
            if self.unvetted_findings > findings {
                self.handed(position);
            }
        }
        Ok(())
    }

    /// Keep the first reason a command is unvetted, and go on collecting effects.
    fn vet(&mut self, outcome: Result<(), String>) {
        if let Err(why) = outcome {
            self.unvetted.get_or_insert(why);
            self.unvetted_findings += 1;
            self.unapprovable = true;
        }
    }

    /// Vet `name`, a program on neither list, when an approval covers this run of it; else
    /// leave the script unvetted and note the approval that would.
    fn unlisted(&mut self, name: &str, args: &[Word], invocation: &Invocation<'_>) {
        let scope = approval_scope(args, invocation);
        let verdict = self
            .approvals
            .map(|approvals| approvals.verdict(name, scope.as_ref()));
        if verdict == Some(Verdict::Approved) {
            return;
        }
        self.unvetted_findings += 1;
        self.unvetted.get_or_insert_with(|| {
            format!("`{name}` is on neither the read-only nor the edit list")
        });
        let approvable = verdict == Some(Verdict::Unapproved { approvable: true });
        match scope.filter(|_| approvable) {
            Some(scope) => {
                let approval = ProgramApproval {
                    name: name.to_string(),
                    scope,
                };
                if !self.always_allow.contains(&approval) {
                    self.always_allow.push(approval);
                }
            }
            None => self.unapprovable = true,
        }
    }

    fn redirects(&mut self, cmd: &SimpleCommand) -> Result<(), String> {
        for redirect in &cmd.redirects {
            if !redirect.writes() {
                if redirect.operator == "<" {
                    self.opened_for_reading(&redirect.target);
                }
                continue;
            }
            if is_harmless_target(&redirect.target, self.scratch) {
                continue;
            }
            if redirect.target.dynamic {
                return Err(format!(
                    "`{}` writes to `{}`, only known at run time",
                    redirect.operator, redirect.target.text
                ));
            }
            let target = spelled(&redirect.target, self.home_changed);
            self.target(Action::Write, target, false);
        }
        Ok(())
    }

    /// Record each path the unvetted command at `position` names as handed to it, and classify
    /// the scripts a shell it runs is given, for the effects in them.
    fn handed(&mut self, position: Position<'_>) {
        let words = &position.cmd.words;
        let Some(start) = command_start(words) else {
            return;
        };
        for word in words[start + 1..].iter().filter_map(operand_or_long_value) {
            let named = static_path(&word, self.home_changed)
                .or_else(|| every_entry_of(&spelled(&word, self.home_changed)));
            if let Some(path) = named {
                self.target(Action::Unknown, path, true);
            }
        }
        let scripts = candidates(words, start)
            .into_iter()
            .filter(|candidate| candidate.may_be_command)
            .flat_map(|candidate| shell_scripts(&words[candidate.at..], position));
        for script in scripts {
            let nested = self.script(script, position.depth + 1);
            self.vet(nested);
        }
    }

    /// Record the read a `<` of `target` is; one bash opens as a network connection leaves the
    /// script unvetted.
    fn opened_for_reading(&mut self, target: &Word) {
        if NETWORK_PATHS
            .iter()
            .any(|prefix| target.text.starts_with(prefix))
        {
            self.vet(Err(format!(
                "`< {}` opens a network connection",
                target.text
            )));
        } else {
            self.read(target, false);
        }
    }

    /// Classify every program `position`'s command runs, following wrappers to what they run.
    fn command(&mut self, position: Position<'_>) -> Result<(), String> {
        let words = &position.cmd.words;
        let Some(start) = command_start(words) else {
            return Ok(());
        };
        let mut invocation = Invocation::default();
        let mut fed_wrapper = None;
        for Candidate { at, .. } in candidates(words, start)
            .into_iter()
            .filter(|candidate| candidate.may_be_command)
        {
            let word = &words[at];
            let args = &words[at + 1..];
            if word.dynamic || word.dynamic_name || invocation.fills(word) {
                return Err(format!("command `{}` is only known at run time", word.text));
            }
            let ran = self.program(word, args, &invocation, position);
            self.vet(ran);
            fed_wrapper =
                (invocation.fed && wrapper_operands(word, args.first()).is_some()).then_some(word);
            let name = program(&word.text);
            invocation.fed |= name == FEEDS_ARGUMENTS;
            invocation.per_item |= name == FEEDS_ARGUMENTS || exec_flags(word).is_some();
            invocation
                .placeholders
                .extend(placeholder::placeholders(word, args));
        }
        fed_wrapper.map_or(Ok(()), |wrapper| {
            Err(format!(
                "`{}` runs a command xargs reads from stdin",
                wrapper.text
            ))
        })
    }

    /// Classify `word` run with `args`: a keyword, function or builtin, or a listed program.
    fn program(
        &mut self,
        word: &Word,
        args: &[Word],
        invocation: &Invocation<'_>,
        position: Position<'_>,
    ) -> Result<(), String> {
        let name = word.text.as_str();
        if is_command_line(word) {
            return self.script(name, position.depth + 1);
        }
        if (!word.quoted && KEYWORDS.contains(&name)) || defines(position, name) {
            return Ok(());
        }
        if let Some(why) = evaluated::by_builtin(name, args) {
            return Err(why);
        }
        match name {
            "eval" => self.script(&joined(args), position.depth + 1),
            "trap" => args
                .iter()
                .find(|arg| !arg.text.starts_with('-'))
                .map_or(Ok(()), |action| {
                    self.script(&action.text, position.depth + 1)
                }),
            "source" | "." => self.sourced(word, args.first(), position),
            "alias" if !args.is_empty() => {
                Err("`alias` defines commands the lists cannot see".into())
            }
            "hash" if args.iter().any(|arg| sets_hashed_path(&arg.text)) => {
                Err("`hash -p` makes a command name run another program".into())
            }
            _ if CHANGES_DIRECTORY.contains(&name) => {
                self.change_directory(name, args);
                Ok(())
            }
            _ if SAFE_BUILTINS.contains(&name) => Ok(()),
            _ => self.listed(word, args, invocation),
        }
    }

    /// Follow `cd DIR` when `DIR` is literal; any other change leaves the directory unknown.
    fn change_directory(&mut self, name: &str, args: &[Word]) {
        let operand = args
            .iter()
            .find(|arg| arg.text == "-" || !arg.text.starts_with('-'));
        let entered = match (name, operand) {
            ("cd", None) => Some("~".to_string()),
            ("cd", Some(dir)) if dir.text != "-" => static_path(dir, self.home_changed)
                .filter(|dir| !self.cdpath_changed || dir.starts_with(['/', '~', '.'])),
            _ => None,
        };
        self.directory = match entered {
            Some(dir) => self.directory.entered(dir),
            None => Directory::Unknown,
        };
    }

    /// Classify `word file`, where `word` is `source` or `.`.
    fn sourced(
        &mut self,
        word: &Word,
        file: Option<&Word>,
        position: Position<'_>,
    ) -> Result<(), String> {
        let Some(file) = file else {
            return Ok(());
        };
        if file.dynamic || !is_stdin_path(&file.text) {
            return Err(format!(
                "`{} {}` runs a script the lists cannot see",
                word.text, file.text
            ));
        }
        let inputs: Vec<&str> = position.script.inputs_of(position.cmd).collect();
        if inputs.is_empty() {
            return Err(format!("`{}` runs a script read from stdin", word.text));
        }
        inputs
            .into_iter()
            .try_for_each(|input| self.script(input, position.depth + 1))
    }

    /// Classify a program found on `PATH`: a wrapper, or one on the read-only or edit list.
    fn listed(
        &mut self,
        word: &Word,
        args: &[Word],
        invocation: &Invocation<'_>,
    ) -> Result<(), String> {
        let name = word.text.as_str();
        if name.contains('/') {
            return Err(format!("`{name}` runs a program by its path"));
        }
        if is_wrapper(word) {
            return self.wrapper(name, args);
        }
        if let Some(&(_, flags)) = READ_ONLY_PROGRAMS.iter().find(|(n, _)| *n == name) {
            read_only(name, args, flags, invocation)?;
            return self.reads(name, args, invocation);
        }
        match EDIT_PROGRAMS.iter().find(|edit| edit.name == name) {
            Some(edit) => self.edit(edit, args, invocation),
            None => {
                self.unlisted(name, args, invocation);
                Ok(())
            }
        }
    }

    /// Vet a wrapper's own options and record the file `xargs` reads its items from; the
    /// command it runs is classified on its own.
    fn wrapper(&mut self, name: &str, args: &[Word]) -> Result<(), String> {
        if name == FEEDS_ARGUMENTS {
            for file in placeholder::xargs_arg_files(args) {
                self.read(&file, false);
            }
        }
        if name != "env" {
            return Ok(());
        }
        if let Some(option) = env_attached_option(args) {
            return Err(format!(
                "`env {option}` may run a command the lists cannot see"
            ));
        }
        if args
            .iter()
            .take_while(|arg| arg.text.starts_with('-'))
            .any(|arg| matches!(arg.text.as_str(), "-C" | "--chdir"))
        {
            self.directory = Directory::Unknown;
        }
        Ok(())
    }

    /// Record the files a read-only program reads the contents of: its operands unless it is
    /// one of [`READS_NO_CONTENTS`], and the values of its [`FILE_OPTIONS`].
    fn reads(
        &mut self,
        name: &str,
        args: &[Word],
        invocation: &Invocation<'_>,
    ) -> Result<(), String> {
        let file_flags = file_options(name);
        let reads_operands = !READS_NO_CONTENTS.contains(&name);
        if !reads_operands && file_flags.is_empty() {
            return Ok(());
        }
        if invocation.fed {
            return Err(format!(
                "`{name}` reads files xargs names from stdin, only known at run time"
            ));
        }
        if name == "find" {
            return self.find_reads(args, file_flags);
        }
        if let Some(arg) = args
            .iter()
            .find(|arg| self.may_become_unseen(arg, reads_operands && !invocation.fills(arg)))
        {
            return Err(format!(
                "`{name} {}` may become an option or a file only known at run time",
                arg.text
            ));
        }
        let recursive = RECURSIVE_READERS
            .iter()
            .find(|(reader, _)| *reader == name)
            .is_some_and(|(_, flags)| {
                options(args).any(|arg| flags.iter().any(|flag| flag.matched_by(&arg.text)))
            });
        let files: Vec<Word> = read_words(args, file_flags, reads_operands)
            .into_iter()
            .filter(|word| !invocation.fills(word))
            .collect();
        for file in &files {
            self.read(file, recursive);
        }
        let names_no_file = if GREP_PROGRAMS.contains(&name) {
            grep_file_operands(args) == 0
        } else {
            files.len() <= 1
        };
        if recursive && names_no_file {
            self.target(Action::Read, ".".to_string(), true);
        }
        match args.iter().find_map(|arg| argument_file(name, arg)) {
            Some(list) => {
                self.read(&list, false);
                Err(format!(
                    "`{name} @{}` takes its arguments from a file, only known at run time",
                    list.text
                ))
            }
            None => Ok(()),
        }
    }

    /// Record the file `find` takes its starting points from, and its starting points as read
    /// throughout when it runs a command on what it finds.
    fn find_reads(&mut self, args: &[Word], file_flags: &[Flag]) -> Result<(), String> {
        let start_lists = read_words(find_own_args(args), file_flags, false);
        for list in &start_lists {
            self.read(list, false);
        }
        if !args
            .iter()
            .any(|arg| FIND_EXEC_FLAGS.contains(&arg.text.as_str()))
        {
            return Ok(());
        }
        if let Some(list) = start_lists.first() {
            return Err(format!(
                "`find` runs a command on the files `{}` names, only known at run time",
                list.text
            ));
        }
        let starts: Vec<&Word> = args
            .iter()
            .skip_while(|arg| FIND_LEADING_OPTIONS.contains(&arg.text.as_str()))
            .take_while(|arg| !arg.text.starts_with(['-', '(', '!', ',']))
            .collect();
        if starts.is_empty() {
            self.target(Action::Read, ".".to_string(), true);
        }
        for start in starts {
            self.read(start, true);
        }
        Ok(())
    }

    /// Whether `arg` may become at run time an option or further words that nothing vets: it
    /// is an option holding an expansion outside a `--name=value` value that is `read_as_file`,
    /// or a word not `read_as_file` that an expansion starts or may split.
    fn may_become_unseen(&self, arg: &Word, read_as_file: bool) -> bool {
        let text = spelled(arg, self.home_changed);
        if !arg.dynamic || !text.contains(ANY_PATH_CHARS) {
            return false;
        }
        match text.strip_prefix('-') {
            Some(option) => {
                let names_long_option = option
                    .strip_prefix('-')
                    .and_then(|long| long.split_once('='))
                    .is_some_and(|(name, _)| !name.contains(ANY_PATH_CHARS));
                !(read_as_file && names_long_option)
            }
            None => !read_as_file && (arg.splits || text.starts_with(ANY_PATH_CHARS)),
        }
    }

    /// Record a read of `word`; one only known at run time reads all of its static directory,
    /// and one an expansion may send anywhere leaves the script unvetted.
    fn read(&mut self, word: &Word, recursive: bool) {
        match static_path(word, self.home_changed) {
            Some(path) => self.target(Action::Read, path, recursive),
            None => {
                let spelled = spelled(word, self.home_changed);
                if spelled.contains(ANY_PATH_CHARS) {
                    self.vet(Err(format!(
                        "`{}` reads a path only known at run time, which may lie anywhere",
                        word.text
                    )));
                }
                self.target(Action::Read, static_prefix(&spelled), true);
            }
        }
    }

    /// Record what an edit program does to its operands; with an argument that is not vetted,
    /// only what it deletes or changes the mode of, whichever operand is which.
    fn edit(
        &mut self,
        edit: &EditProgram,
        args: &[Word],
        invocation: &Invocation<'_>,
    ) -> Result<(), String> {
        let run_by_find = invocation.per_item && !invocation.fed;
        let args = if run_by_find {
            before_find_command_end(args)
        } else {
            args
        };
        let EditArgs {
            operands,
            recursive,
            unvetted,
        } = edit_args(edit, args, self.home_changed);
        self.runs_edit_program = true;
        match (edit.targets, operands.split_last()) {
            (_, None) => {}
            (Targets::Removes, Some(_)) => self.each(Action::Delete, &operands, recursive),
            (Targets::ChangesAllButFirst, Some(_)) => {
                self.each(Action::ChangeAttributes, &operands[1..], recursive);
            }
            _ if unvetted.is_some() => {}
            (Targets::WritesAll, Some(_)) => self.each(Action::Write, &operands, false),
            (Targets::WritesLast, Some((last, sources))) if !sources.is_empty() => {
                self.carry(Action::Read, sources, recursive);
                self.copied_into(last, sources, recursive);
            }
            (Targets::Moves, Some((last, sources))) if !sources.is_empty() => {
                self.carry(Action::Delete, sources, true);
                self.copied_into(last, sources, false);
            }
            (Targets::WritesLast | Targets::Moves, Some(_)) => {}
        }
        let per_item = invocation
            .per_item
            .then(|| format!("`{}` runs once per item find or xargs supplies", edit.name));
        per_item.or(unvetted).map_or(Ok(()), Err)
    }

    fn each(&mut self, action: Action, paths: &[String], recursive: bool) {
        for path in paths {
            self.target(action, path.clone(), recursive);
        }
    }

    /// Record `action` on each of a copy's or move's `sources`, and that it lands elsewhere.
    fn carry(&mut self, action: Action, sources: &[String], recursive: bool) {
        for source in sources {
            if let Some(path) = self.placed(action, source.clone()) {
                self.carried.push(path.clone());
                self.effects.push(Effect {
                    action,
                    path,
                    recursive,
                });
            }
        }
    }

    /// Record writes of `destination` and of where each source lands if it is a directory.
    fn copied_into(&mut self, destination: &str, sources: &[String], recursive: bool) {
        self.target(Action::Write, destination.to_string(), recursive);
        for name in sources
            .iter()
            .map(|source| basename(source))
            .filter(|name| !matches!(*name, "" | "." | ".."))
        {
            self.target(Action::Write, joined_path(destination, name), recursive);
        }
    }

    /// Record an effect on `path` where the script's directory places it.
    fn target(&mut self, action: Action, path: String, recursive: bool) {
        if let Some(path) = self.placed(action, path) {
            self.effects.push(Effect {
                action,
                path,
                recursive,
            });
        }
    }

    /// `path`, joined to the directory a `cd` entered when relative; `None`, noting the path,
    /// when that directory is unknown.
    fn placed(&mut self, action: Action, path: String) -> Option<String> {
        let relative = is_relative(&path);
        let moved = !matches!(self.directory, Directory::Start);
        let unknown = matches!(self.directory, Directory::Unknown);
        let vetted = action != Action::Unknown;
        if vetted && relative && (unknown || (moved && action != Action::Read)) {
            self.relative_after_cd.get_or_insert_with(|| path.clone());
        }
        match &self.directory {
            _ if !relative => Some(path),
            Directory::Start => Some(path),
            Directory::At(dir) => Some(joined_path(dir, &path)),
            Directory::Unknown => None,
        }
    }

    fn into_analysis(self, outcome: Result<(), String>) -> Analysis {
        let below_changed_home = self
            .effects
            .iter()
            .find(|effect| self.home_changed && effect.path.starts_with('~'))
            .map(|effect| changed_home_why(&effect.path));
        let approvable = !self.unapprovable
            && outcome.is_ok()
            && self.relative_after_cd.is_none()
            && below_changed_home.is_none();
        let why = self
            .unvetted
            .or(outcome.err())
            .or_else(|| self.relative_after_cd.as_deref().map(changed_directory_why))
            .or(below_changed_home);
        let class = match why {
            Some(why) => Class::Unvetted { why },
            None => edit_class(&self.effects, self.runs_edit_program),
        };
        Analysis {
            class,
            effects: self.effects,
            carried: self.carried,
            always_allow: if approvable {
                self.always_allow
            } else {
                Vec::new()
            },
        }
    }
}

/// Classify `script` as read-only, an edit of the paths it names, or unvetted, with nothing
/// approved. Redirections into `scratch`, like those into `/tmp`, are not edits.
pub fn classify(script: &str, scratch: Option<&Path>) -> Class {
    analyze(script, scratch, None).class
}

/// [`classify`] `script`, counting a program on neither list as vetted where `approvals`
/// cover it, and list the file operations its commands perform.
pub fn analyze(script: &str, scratch: Option<&Path>, approvals: Option<&Allowlist>) -> Analysis {
    let mut classifier = Classifier {
        scratch: scratch.and_then(Path::to_str),
        approvals,
        ..Classifier::default()
    };
    let outcome = classifier.script(script, 0);
    classifier.into_analysis(outcome)
}

fn changed_directory_why(path: &str) -> String {
    format!("the script changes directory, so where `{path}` is cannot be known")
}

fn changed_home_why(path: &str) -> String {
    format!("the script may change HOME, so where `{path}` is cannot be known")
}

/// Which runs an "always allow" for a program run with `args` covers: those with the same
/// subcommand as its first argument, or those with only options; `None` when the arguments
/// make neither certain.
fn approval_scope(args: &[Word], invocation: &Invocation<'_>) -> Option<ApprovalScope> {
    let known = |arg: &Word| !arg.dynamic && !arg.splits && !invocation.fills(arg);
    match args.first() {
        Some(first) if known(first) && is_subcommand(&first.text) => {
            Some(ApprovalScope::Subcommand(first.text.clone()))
        }
        _ => {
            let options_only = !invocation.fed
                && args
                    .iter()
                    .all(|arg| known(arg) && arg.text.starts_with('-'));
            options_only.then_some(ApprovalScope::OptionsOnly)
        }
    }
}

/// The class of a vetted script with `effects`.
fn edit_class(effects: &[Effect], runs_edit_program: bool) -> Class {
    let writes: Vec<String> = effects
        .iter()
        .filter(|effect| matches!(effect.action, Action::Write | Action::ChangeAttributes))
        .map(|effect| effect.path.clone())
        .collect();
    if !runs_edit_program && writes.is_empty() {
        return Class::ReadOnly;
    }
    let deletes = effects
        .iter()
        .filter(|effect| effect.action == Action::Delete)
        .map(|effect| Delete {
            path: effect.path.clone(),
            recursive: effect.recursive,
        })
        .collect();
    Class::Edit { writes, deletes }
}

/// Vet a read-only program's own options against its `disqualifiers`.
fn read_only(
    name: &str,
    args: &[Word],
    disqualifiers: &[Flag],
    invocation: &Invocation<'_>,
) -> Result<(), String> {
    if disqualifiers.is_empty() {
        return Ok(());
    }
    if invocation.fed {
        return Err(format!(
            "`{name}` is handed arguments xargs reads from stdin, which may be options that write"
        ));
    }
    let options: Vec<&Word> = if name == "find" {
        find_own_args(args).collect()
    } else {
        args.iter()
            .take_while(|arg| arg.dynamic || arg.text != "--")
            .collect()
    };
    for arg in options {
        if arg.dynamic || arg.splits || invocation.fills(arg) {
            return Err(format!(
                "`{name} {}` may become an option that writes files or runs a program",
                arg.text
            ));
        }
        if disqualifiers.iter().any(|d| d.matched_by(&arg.text)) {
            return Err(format!(
                "`{name} {}` writes files or runs a program",
                arg.text
            ));
        }
    }
    Ok(())
}

/// `find`'s arguments other than the commands its exec primaries run.
fn find_own_args(args: &[Word]) -> impl Iterator<Item = &Word> {
    let mut in_command = false;
    args.iter().filter(move |arg| {
        let text = arg.text.as_str();
        if in_command {
            in_command = !FIND_COMMAND_ENDS.contains(&text);
            return false;
        }
        in_command = FIND_EXEC_FLAGS.contains(&text);
        !in_command
    })
}

/// The arguments before any `--`.
fn options(args: &[Word]) -> impl Iterator<Item = &Word> {
    args.iter().take_while(|arg| arg.text != "--")
}

/// The options naming a file that `name` reads, as [`FILE_OPTIONS`] lists them.
fn file_options(name: &str) -> &'static [Flag] {
    FILE_OPTIONS
        .iter()
        .find(|(program, _)| *program == name)
        .map_or(&[], |&(_, flags)| flags)
}

/// The words a read-only program may read as files: the values of its `file_flags`, wherever
/// they are given, and when it `reads_operands`, its operands and the values of its
/// `--name=value` options.
fn read_words<'w>(
    args: impl IntoIterator<Item = &'w Word>,
    file_flags: &[Flag],
    reads_operands: bool,
) -> Vec<Word> {
    let mut files = Vec::new();
    let mut after_options = false;
    let mut is_flag_value = false;
    for arg in args {
        if mem::take(&mut is_flag_value) {
            files.push(arg.clone());
        }
        match file_flags.iter().find_map(|flag| flag.value_in(&arg.text)) {
            Some(OptionValue::Attached(value)) => files.push(arg.part(value)),
            Some(OptionValue::NextWord) => is_flag_value = true,
            None => {}
        }
        if !reads_operands {
            continue;
        }
        if after_options {
            files.push(arg.clone());
        } else if arg.text == "--" {
            after_options = true;
        } else {
            files.extend(operand_or_long_value(arg));
        }
    }
    files.dedup();
    files
}

/// How many of `grep`'s `args` are files to search: its operands, less the pattern unless an
/// option gives it. Errs toward fewer where a word may be an option's value.
fn grep_file_operands(args: &[Word]) -> usize {
    let mut operands: usize = 0;
    let mut pattern_given = false;
    let mut after_options = false;
    let mut is_option_value = false;
    for arg in args {
        let text = arg.text.as_str();
        if mem::take(&mut is_option_value) {
            continue;
        }
        if after_options || text == "-" || !text.starts_with('-') {
            operands += 1;
        } else if text == "--" {
            after_options = true;
        } else {
            pattern_given |= GREP_PATTERN_FLAGS.iter().any(|flag| flag.matched_by(text));
            is_option_value = GREP_VALUE_FLAGS
                .iter()
                .any(|flag| matches!(flag.value_in(text), Some(OptionValue::NextWord)));
        }
    }
    if pattern_given {
        operands
    } else {
        operands.saturating_sub(1)
    }
}

/// The file `name` takes more arguments from when `arg` is `@file`.
fn argument_file(name: &str, arg: &Word) -> Option<Word> {
    let file = arg
        .text
        .strip_prefix('@')
        .filter(|_| READS_ARGUMENT_FILES.contains(&name))?;
    Some(arg.part(file))
}

/// `arg` when it is an operand, or the value of the `--name=value` option it is.
fn operand_or_long_value(arg: &Word) -> Option<Word> {
    if let Some((_, value)) = arg.text.strip_prefix("--").and_then(|o| o.split_once('=')) {
        return Some(arg.part(value));
    }
    (arg.text == "-" || !arg.text.starts_with('-')).then(|| arg.clone())
}

/// `args` split into operands and options, noting the first option not vetted for `edit` and
/// the first operand only known at run time, other than a glob for every entry of a directory.
fn edit_args(edit: &EditProgram, args: &[Word], home_changed: bool) -> EditArgs {
    let mut parsed = EditArgs::default();
    let mut options = true;
    let mut words = args.iter();
    while let Some(arg) = words.next() {
        let text = arg.text.as_str();
        let vetted = if options && text == "--" {
            options = false;
            Ok(())
        } else if let Some(long) = text.strip_prefix("--").filter(|_| options && !arg.dynamic) {
            parsed.recursive |= "recursive".starts_with(long);
            if edit.long_flags.contains(&long) {
                Ok(())
            } else {
                Err(format!("`{} {text}` is not a vetted option", edit.name))
            }
        } else if let Some(cluster) = short_options(text).filter(|_| options && !arg.dynamic) {
            parsed.recursive |= cluster.contains(['r', 'R']);
            edit_flags(edit, cluster).map(|value_is_next_word| {
                if value_is_next_word {
                    words.next();
                }
            })
        } else {
            edit_operand(edit, arg, home_changed).map(|operand| parsed.operands.push(operand))
        };
        if let Err(why) = vetted {
            parsed.unvetted.get_or_insert(why);
        }
    }
    parsed
}

/// `args` up to the `;` or `+` that ends the command `find` runs.
fn before_find_command_end(args: &[Word]) -> &[Word] {
    let end = args
        .iter()
        .position(|arg| FIND_COMMAND_ENDS.contains(&arg.text.as_str()))
        .unwrap_or(args.len());
    &args[..end]
}

/// The scripts `command` gives the shell it runs, if it runs one: the word after `-c`, else
/// its inputs when it has no operand.
fn shell_scripts<'s>(command: &'s [Word], position: Position<'s>) -> Vec<&'s str> {
    let Some((shell, args)) = command.split_first() else {
        return Vec::new();
    };
    if !SHELLS.contains(&program(&shell.text).as_str()) {
        return Vec::new();
    }
    let command_string = args
        .iter()
        .position(|arg| short_options(&arg.text).is_some_and(|cluster| cluster.contains('c')))
        .and_then(|option| args.get(option + 1));
    match command_string {
        Some(script) => vec![script.text.as_str()],
        None if args.iter().all(|arg| arg.text.starts_with('-')) => {
            position.script.inputs_of(position.cmd).collect()
        }
        None => Vec::new(),
    }
}

/// Why nothing `script` runs can be vetted: it changes a variable that decides which programs
/// run, exports one a program may act on, or has bash evaluate text the classifier cannot see.
fn unvettable_throughout(script: &Script) -> Result<(), String> {
    if let Some(variable) =
        changed_sensitive_variable(script).or_else(|| changed_git_variable(script))
    {
        return Err(format!(
            "the script changes {variable}, so what its programs do cannot be vetted"
        ));
    }
    if let Some(variable) = exported_variable(script) {
        return Err(format!(
            "the script exports {variable}, which the programs it runs may act on"
        ));
    }
    evaluated::in_script(script).map_or(Ok(()), Err)
}

/// Whether `path` resolves against a working directory.
fn is_relative(path: &str) -> bool {
    !path.starts_with(['/', '~'])
}

/// `arg` as a path, or the directory a glob for its every entry lists.
fn edit_operand(edit: &EditProgram, arg: &Word, home_changed: bool) -> Result<String, String> {
    if let Some(path) = static_path(arg, home_changed) {
        return Ok(path);
    }
    every_entry_of(&spelled(arg, home_changed)).ok_or_else(|| {
        format!(
            "`{} {}` names a path only known at run time",
            edit.name, arg.text
        )
    })
}

/// Vet a cluster of short flags, returning whether its value is the next word.
fn edit_flags(edit: &EditProgram, cluster: &str) -> Result<bool, String> {
    for (at, flag) in cluster.char_indices() {
        if edit.value_flags.contains(flag) {
            return Ok(at + flag.len_utf8() == cluster.len());
        }
        if !edit.flags.contains(flag) {
            return Err(format!("`{} -{flag}` is not a vetted option", edit.name));
        }
    }
    Ok(false)
}

/// `word` as a path known before it runs, spelled as [`spelled`] does; `None` when it is only
/// known at run time.
fn static_path(word: &Word, home_changed: bool) -> Option<String> {
    if !word.dynamic {
        return Some(spelled(word, home_changed));
    }
    home_as_tilde(&word.text, home_changed).filter(|path| !path.contains(DYNAMIC_CHARS))
}

/// `word`'s text with a leading `$HOME` spelled `~` when it may be, and a leading `~` that
/// bash leaves as written spelled `./~`.
fn spelled(word: &Word, home_changed: bool) -> String {
    if word.literal_tilde {
        return format!("./{}", word.text);
    }
    word.dynamic
        .then(|| home_as_tilde(&word.text, home_changed))
        .flatten()
        .unwrap_or_else(|| word.text.clone())
}

fn home_as_tilde(text: &str, home_changed: bool) -> Option<String> {
    if home_changed {
        return None;
    }
    HOME_VARIABLES
        .iter()
        .find_map(|variable| text.strip_prefix(variable))
        .filter(|rest| rest.is_empty() || rest.starts_with('/'))
        .map(|rest| format!("~{rest}"))
}

/// The directory a path only known at run time lies in, from its text before the first part
/// only known at run time.
fn static_prefix(text: &str) -> String {
    let known = text.find(DYNAMIC_CHARS).map_or(text, |end| &text[..end]);
    match known.rfind('/') {
        Some(0) => "/".to_string(),
        Some(end) => known[..end].to_string(),
        None => ".".to_string(),
    }
}

/// The directory `text` lists every entry of, as `dir/*` or `dir/{*,.*}` do.
fn every_entry_of(text: &str) -> Option<String> {
    let (dir, name) = text.rsplit_once('/').unwrap_or((".", text));
    let selects_all = |pattern: &str| !pattern.is_empty() && pattern.chars().all(|c| c == '*');
    let all = selects_all(name)
        || name
            .strip_prefix('{')
            .and_then(|list| list.strip_suffix('}'))
            .is_some_and(|list| list.split(',').any(selects_all));
    let dir = if dir.is_empty() { "/" } else { dir };
    (all && !dir.contains(DYNAMIC_CHARS)).then(|| dir.to_string())
}

/// `name` inside `dir`, with one `/` between them.
fn joined_path(dir: &str, name: &str) -> String {
    format!("{}/{name}", dir.trim_end_matches('/'))
}

/// A wrapper run as such: a reserved-word wrapper only when unquoted.
fn is_wrapper(word: &Word) -> bool {
    let name = word.text.as_str();
    WRAPPERS.iter().any(|&(wrapper, _)| wrapper == name)
        && !(word.quoted && WRAPPER_KEYWORDS.contains(&name))
}

/// Whether `position`'s script defined function `name` before reaching its command.
fn defines(position: Position<'_>, name: &str) -> bool {
    position
        .script
        .functions
        .iter()
        .any(|(function, defined)| function == name && *defined <= position.index)
}

fn joined(args: &[Word]) -> String {
    args.iter()
        .map(|arg| arg.text.as_str())
        .collect::<Vec<_>>()
        .join(" ")
}

fn sets_hashed_path(arg: &str) -> bool {
    short_options(arg).is_some_and(|cluster| cluster.contains('p'))
}

/// The first `GIT_*` variable the script may change, which can make `git` run any program.
fn changed_git_variable(script: &Script) -> Option<&str> {
    script
        .commands
        .iter()
        .flat_map(|cmd| &cmd.words)
        .flat_map(|word| changed_variables(&word.text))
        .find(|name| name.starts_with("GIT_"))
}

/// The first variable the script passes to the programs it runs, other than the
/// [`HARMLESS_EXPORTS`] and `LC_*`: a program may read one as a command to run (`EDITOR`,
/// `PAGER`, `CARGO_TARGET_*_RUNNER`), which no list or approval vets.
fn exported_variable(script: &Script) -> Option<&str> {
    script
        .commands
        .iter()
        .flat_map(|cmd| exported_names(&cmd.words))
        .find(|name| !(HARMLESS_EXPORTS.contains(name) || name.starts_with("LC_")))
}

/// The names one command exports: assignments before a program or after `env`, the operands
/// of `export` or a `-x` declaration, and every variable after `set -a`.
fn exported_names(words: &[Word]) -> Vec<&str> {
    let Some(start) = command_start(words) else {
        return Vec::new();
    };
    let operands = &words[start + 1..];
    let has_option = |letter: char| {
        operands
            .iter()
            .any(|word| short_options(&word.text).is_some_and(|cluster| cluster.contains(letter)))
    };
    let runs_program = !SAFE_BUILTINS.contains(&words[start].text.as_str());
    let mut names: Vec<&str> = words[..start]
        .iter()
        .filter(|word| runs_program && is_assignment(&word.text))
        .map(exported_name)
        .collect();
    let declared = match words[start].text.as_str() {
        "export" => true,
        command => EXPORTING_DECLARATIONS.contains(&command) && has_option('x'),
    };
    if declared {
        names.extend(
            operands
                .iter()
                .filter(|word| !word.text.starts_with('-'))
                .map(exported_name),
        );
    }
    if words[start].text == "set"
        && (has_option('a') || operands.iter().any(|word| word.text == "allexport"))
    {
        names.push("all its variables");
    }
    if let Some(env) = words.iter().position(|word| word.text == "env") {
        names.extend(
            words[env + 1..]
                .iter()
                .filter(|word| is_assignment(&word.text))
                .map(exported_name),
        );
    }
    names
}

/// The variable name `word` assigns or declares.
fn exported_name(word: &Word) -> &str {
    let name = word
        .text
        .split_once('=')
        .map_or(word.text.as_str(), |(name, _)| name);
    name.strip_suffix('+').unwrap_or(name)
}

/// Whether the script may change variable `name`.
fn changes(script: &Script, name: &str) -> bool {
    script
        .commands
        .iter()
        .flat_map(|cmd| &cmd.words)
        .flat_map(|word| changed_variables(&word.text))
        .any(|changed| changed == name)
}

#[cfg(test)]
mod tests;
