//! Sorts a script into read-only, edit or unvetted by the programs it runs and the files its
//! redirections write, using short lists of programs whose effects are known, and lists the
//! paths those programs read, write, delete or change the mode or owner of.

use std::path::Path;

use super::placeholder::{self, Placeholder};
use super::{
    Candidate, FEEDS_ARGUMENTS, FIND_EXEC_FLAGS, KEYWORDS, MAX_NESTED_SCRIPTS, SAFE_BUILTINS,
    WRAPPERS, basename, candidates, changed_sensitive_variable, changed_variables, command_start,
    env_attached_option, evaluated, exec_flags, is_command_line, is_harmless_target, is_stdin_path,
    program, short_options, wrapper_operands,
};
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

/// `find` options that come before its starting points.
const FIND_LEADING_OPTIONS: &[&str] = &["-H", "-L", "-P"];

/// Programs that write or delete the literal paths they name, with the options vetted for each.
const EDIT_PROGRAMS: &[EditProgram] = &[
    EditProgram {
        name: "chgrp",
        flags: "cfhvHLPR",
        value_flags: "",
        long_flags: CHOWN_LONG_FLAGS,
        targets: Targets::ChangesAllButFirst,
    },
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
        name: "chown",
        flags: "cfhvHLPR",
        value_flags: "",
        long_flags: CHOWN_LONG_FLAGS,
        targets: Targets::ChangesAllButFirst,
    },
    EditProgram {
        name: "cp",
        flags: "adfHilLnPprRsTuvx",
        value_flags: "",
        long_flags: &[
            "archive",
            "dereference",
            "force",
            "interactive",
            "link",
            "no-clobber",
            "no-dereference",
            "no-target-directory",
            "one-file-system",
            "recursive",
            "symbolic-link",
            "update",
            "verbose",
        ],
        targets: Targets::WritesLast,
    },
    EditProgram {
        name: "ln",
        flags: "fnrsTv",
        value_flags: "",
        long_flags: &[
            "force",
            "no-dereference",
            "no-target-directory",
            "relative",
            "symbolic",
            "verbose",
        ],
        targets: Targets::Links,
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

/// The long options vetted for `chown` and `chgrp`.
const CHOWN_LONG_FLAGS: &[&str] = &[
    "changes",
    "dereference",
    "no-dereference",
    "no-preserve-root",
    "preserve-root",
    "quiet",
    "recursive",
    "silent",
    "verbose",
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
    /// It may do something neither list vets; `why` names the first such thing.
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
    /// `chmod`, `chown` or `chgrp`.
    ChangeAttributes,
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
        match self {
            Self::Short(flag) => {
                short_options(option).is_some_and(|cluster| cluster.contains(flag))
            }
            Self::Long(name) => option
                .strip_prefix("--")
                .map(|given| given.split_once('=').map_or(given, |(given, _)| given))
                .is_some_and(|given| !given.is_empty() && name.starts_with(given)),
            Self::Primary(name) => option == name,
        }
    }
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
    /// Writes its last operand, or the sole operand's name in the working directory.
    Links,
    /// Changes the mode or owner of every operand after the first, which names the change.
    ChangesAllButFirst,
}

/// An [`EditProgram`]'s arguments, once its options are vetted.
struct EditArgs {
    operands: Vec<String>,
    recursive: bool,
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
    effects: Vec<Effect>,
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
        if let Some(variable) =
            changed_sensitive_variable(&script).or_else(|| changed_git_variable(&script))
        {
            return Err(format!(
                "the script changes {variable}, so what its programs do cannot be vetted"
            ));
        }
        if let Some(why) = evaluated::in_script(&script) {
            return Err(why);
        }
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
            let ran = self.command(position);
            self.vet(ran);
        }
        Ok(())
    }

    /// Keep the first reason a command is unvetted, and go on collecting effects.
    fn vet(&mut self, outcome: Result<(), String>) {
        if let Err(why) = outcome {
            self.unvetted.get_or_insert(why);
        }
    }

    fn redirects(&mut self, cmd: &SimpleCommand) -> Result<(), String> {
        for redirect in &cmd.redirects {
            if !redirect.writes() {
                if redirect.operator == "<" {
                    self.read(&redirect.target, false);
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
            self.target(Action::Write, redirect.target.text.clone(), false);
        }
        Ok(())
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
            None => Err(format!(
                "`{name}` is on neither the read-only nor the edit list"
            )),
        }
    }

    /// Vet a wrapper's own options; the command it runs is classified on its own.
    fn wrapper(&mut self, name: &str, args: &[Word]) -> Result<(), String> {
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

    /// Record the files a read-only program reads the contents of.
    fn reads(
        &mut self,
        name: &str,
        args: &[Word],
        invocation: &Invocation<'_>,
    ) -> Result<(), String> {
        if READS_NO_CONTENTS.contains(&name) {
            return Ok(());
        }
        if invocation.fed {
            return Err(format!(
                "`{name}` reads files xargs names from stdin, only known at run time"
            ));
        }
        if name == "find" {
            self.find_reads(args);
            return Ok(());
        }
        let recursive = RECURSIVE_READERS
            .iter()
            .find(|(reader, _)| *reader == name)
            .is_some_and(|(_, flags)| {
                options(args).any(|arg| flags.iter().any(|flag| flag.matched_by(&arg.text)))
            });
        let operands: Vec<Word> = read_operands(args)
            .filter(|word| !invocation.fills(word))
            .collect();
        for operand in &operands {
            self.read(operand, recursive);
        }
        if recursive && operands.len() <= 1 {
            self.target(Action::Read, ".".to_string(), true);
        }
        Ok(())
    }

    /// Record `find`'s starting points as read throughout when it runs a command on what it finds.
    fn find_reads(&mut self, args: &[Word]) {
        if !args
            .iter()
            .any(|arg| FIND_EXEC_FLAGS.contains(&arg.text.as_str()))
        {
            return;
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
    }

    /// Record a read of `word`; one only known at run time reads all of its static directory.
    fn read(&mut self, word: &Word, recursive: bool) {
        match static_path(word, self.home_changed) {
            Some(path) => self.target(Action::Read, path, recursive),
            None => {
                let prefix = static_prefix(&spelled(word, self.home_changed));
                self.target(Action::Read, prefix, true);
            }
        }
    }

    fn edit(
        &mut self,
        edit: &EditProgram,
        args: &[Word],
        invocation: &Invocation<'_>,
    ) -> Result<(), String> {
        if invocation.per_item {
            return Err(format!(
                "`{}` runs once per item find or xargs supplies",
                edit.name
            ));
        }
        let EditArgs {
            operands,
            recursive,
        } = edit_args(edit, args, self.home_changed)?;
        self.runs_edit_program = true;
        match (edit.targets, operands.split_last()) {
            (_, None) => {}
            (Targets::Removes, Some(_)) => self.each(Action::Delete, &operands, recursive),
            (Targets::WritesAll, Some(_)) => self.each(Action::Write, &operands, false),
            (Targets::ChangesAllButFirst, Some(_)) => {
                self.each(Action::ChangeAttributes, &operands[1..], recursive);
            }
            (Targets::WritesLast, Some((last, sources))) if !sources.is_empty() => {
                self.each(Action::Read, sources, recursive);
                self.copied_into(last, sources, recursive);
            }
            (Targets::Moves, Some((last, sources))) if !sources.is_empty() => {
                self.each(Action::Delete, sources, true);
                self.copied_into(last, sources, false);
            }
            (Targets::Links, Some((last, []))) => {
                self.target(Action::Read, last.clone(), false);
                self.target(Action::Write, basename(last).to_string(), false);
            }
            (Targets::Links, Some((last, sources))) => {
                self.each(Action::Read, sources, false);
                self.copied_into(last, sources, false);
            }
            (Targets::WritesLast | Targets::Moves, Some(_)) => {}
        }
        Ok(())
    }

    fn each(&mut self, action: Action, paths: &[String], recursive: bool) {
        for path in paths {
            self.target(action, path.clone(), recursive);
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

    /// Record an effect on `path`, joined to the directory a `cd` entered when relative.
    fn target(&mut self, action: Action, path: String, recursive: bool) {
        let relative = !path.starts_with(['/', '~']);
        let moved = !matches!(self.directory, Directory::Start);
        let unknown = matches!(self.directory, Directory::Unknown);
        if relative && (unknown || (moved && action != Action::Read)) {
            self.relative_after_cd.get_or_insert_with(|| path.clone());
        }
        let path = match &self.directory {
            _ if !relative => path,
            Directory::Start => path,
            Directory::At(dir) => joined_path(dir, &path),
            Directory::Unknown => return,
        };
        self.effects.push(Effect {
            action,
            path,
            recursive,
        });
    }

    fn into_analysis(self, outcome: Result<(), String>) -> Analysis {
        let why = self
            .unvetted
            .or(outcome.err())
            .or_else(|| self.relative_after_cd.as_deref().map(changed_directory_why));
        let class = match why {
            Some(why) => Class::Unvetted { why },
            None => edit_class(&self.effects, self.runs_edit_program),
        };
        Analysis {
            class,
            effects: self.effects,
        }
    }
}

/// Classify `script` as read-only, an edit of the paths it names, or unvetted. Redirections
/// into `scratch`, like those into `/tmp`, are not edits.
pub fn classify(script: &str, scratch: Option<&Path>) -> Class {
    analyze(script, scratch).class
}

/// [`classify`] `script`, and list the file operations its commands perform.
pub fn analyze(script: &str, scratch: Option<&Path>) -> Analysis {
    let mut classifier = Classifier {
        scratch: scratch.and_then(Path::to_str),
        ..Classifier::default()
    };
    let outcome = classifier.script(script, 0);
    classifier.into_analysis(outcome)
}

fn changed_directory_why(path: &str) -> String {
    format!("the script changes directory, so where `{path}` is cannot be known")
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

/// The words a read-only program may read as files: its operands and the values of its
/// `--name=value` options. Values of short options are not seen.
fn read_operands(args: &[Word]) -> impl Iterator<Item = Word> + '_ {
    let mut after_options = false;
    args.iter().filter_map(move |arg| {
        if after_options {
            return Some(arg.clone());
        }
        if arg.text == "--" {
            after_options = true;
            return None;
        }
        if let Some((_, value)) = arg.text.strip_prefix("--").and_then(|o| o.split_once('=')) {
            return Some(Word {
                text: value.to_string(),
                ..arg.clone()
            });
        }
        (arg.text == "-" || !arg.text.starts_with('-')).then(|| arg.clone())
    })
}

/// `args` split into operands and vetted options; fails on an option not vetted for `edit` or
/// an argument only known at run time, other than a glob for every entry of a directory.
fn edit_args(edit: &EditProgram, args: &[Word], home_changed: bool) -> Result<EditArgs, String> {
    let mut parsed = EditArgs {
        operands: Vec::new(),
        recursive: false,
    };
    let mut options = true;
    let mut words = args.iter();
    while let Some(arg) = words.next() {
        let text = arg.text.as_str();
        if options && text == "--" {
            options = false;
        } else if let Some(long) = text.strip_prefix("--").filter(|_| options && !arg.dynamic) {
            if !edit.long_flags.contains(&long) {
                return Err(format!("`{} {text}` is not a vetted option", edit.name));
            }
            parsed.recursive |= long == "recursive";
        } else if let Some(cluster) = short_options(text).filter(|_| options && !arg.dynamic) {
            if edit_flags(edit, cluster)? {
                words.next();
            }
            parsed.recursive |= cluster.contains(['r', 'R']);
        } else {
            parsed.operands.push(edit_operand(edit, arg, home_changed)?);
        }
    }
    Ok(parsed)
}

/// `arg` as a path, or the directory a glob for its every entry lists.
fn edit_operand(edit: &EditProgram, arg: &Word, home_changed: bool) -> Result<String, String> {
    if let Some(path) = static_path(arg, home_changed) {
        return Ok(path);
    }
    if arg.quoted && arg.text.starts_with('~') {
        return Err(format!(
            "`{} {}` may quote its `~`, so where it points cannot be known",
            edit.name, arg.text
        ));
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

/// `word` as a path known before it runs, with a leading `$HOME` spelled `~`; `None` when it
/// is only known at run time or starts with a `~` that may be quoted.
fn static_path(word: &Word, home_changed: bool) -> Option<String> {
    if word.quoted && word.text.starts_with('~') {
        return None;
    }
    if !word.dynamic {
        return Some(word.text.clone());
    }
    home_as_tilde(&word.text, home_changed).filter(|path| !path.contains(DYNAMIC_CHARS))
}

/// `word`'s text with a leading `$HOME` spelled `~` when it may be.
fn spelled(word: &Word, home_changed: bool) -> String {
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
