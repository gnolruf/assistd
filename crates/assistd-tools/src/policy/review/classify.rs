//! Sorts a script into read-only, edit or unvetted by the programs it runs and the files its
//! redirections write, using short lists of programs whose effects are known.

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

/// Programs that only read, each with the options that make it write a file or run a program.
const READ_ONLY_PROGRAMS: &[(&str, &[Disqualifier])] = &[
    ("b2sum", &[]),
    ("base64", &[]),
    ("basename", &[]),
    ("cat", &[]),
    ("cksum", &[]),
    ("cmp", &[]),
    ("column", &[]),
    ("comm", &[]),
    ("cut", &[]),
    (
        "date",
        &[Disqualifier::Short('s'), Disqualifier::Long("set")],
    ),
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
            Disqualifier::Primary("-delete"),
            Disqualifier::Primary("-fls"),
            Disqualifier::Primary("-fprint"),
            Disqualifier::Primary("-fprint0"),
            Disqualifier::Primary("-fprintf"),
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
            Disqualifier::Short('o'),
            Disqualifier::Long("output"),
            Disqualifier::Long("compress-program"),
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

/// Programs that write or delete the literal paths they name, with the options vetted for each.
const EDIT_PROGRAMS: &[EditProgram] = &[
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
    /// It also runs edit programs or redirects elsewhere; targets are as written, unresolved.
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

/// An option that makes a read-only program write a file or run another program.
#[derive(Debug, Clone, Copy)]
enum Disqualifier {
    /// A short option, alone or in a cluster.
    Short(char),
    /// A GNU long option, which may be abbreviated.
    Long(&'static str),
    /// A `find` primary.
    Primary(&'static str),
}

impl Disqualifier {
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
    /// Writes its last operand.
    WritesLast,
    /// Deletes all but its last operand, which it writes.
    Moves,
    /// Writes its last operand, or the sole operand's name in the working directory.
    Links,
}

/// An [`EditProgram`]'s arguments, once its options are vetted.
struct EditArgs<'w> {
    operands: Vec<&'w str>,
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

#[derive(Default)]
struct Classifier<'a> {
    scratch: Option<&'a str>,
    writes: Vec<String>,
    deletes: Vec<Delete>,
    runs_edit_program: bool,
    changes_directory: bool,
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
        for (index, cmd) in script.commands.iter().enumerate() {
            self.redirects(cmd)?;
            let position = Position {
                cmd,
                script: &script,
                index,
                depth,
            };
            self.command(position)?;
        }
        Ok(())
    }

    fn redirects(&mut self, cmd: &SimpleCommand) -> Result<(), String> {
        for redirect in cmd
            .redirects
            .iter()
            .filter(|r| r.writes() && !is_harmless_target(&r.target, self.scratch))
        {
            if redirect.target.dynamic {
                return Err(format!(
                    "`{}` writes to `{}`, only known at run time",
                    redirect.operator, redirect.target.text
                ));
            }
            self.writes.push(redirect.target.text.clone());
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
            self.program(word, args, &invocation, position)?;
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
                self.changes_directory = true;
                Ok(())
            }
            _ if SAFE_BUILTINS.contains(&name) => Ok(()),
            _ => self.listed(word, args, invocation),
        }
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
        if let Some(&(_, disqualifiers)) = READ_ONLY_PROGRAMS.iter().find(|(n, _)| *n == name) {
            return read_only(name, args, disqualifiers, invocation);
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
        self.changes_directory |= args
            .iter()
            .take_while(|arg| arg.text.starts_with('-'))
            .any(|arg| matches!(arg.text.as_str(), "-C" | "--chdir"));
        Ok(())
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
        } = edit_args(edit, args)?;
        self.runs_edit_program = true;
        let owned = |path: &&str| path.to_string();
        match (edit.targets, operands.split_last()) {
            (_, None) => {}
            (Targets::Removes, Some(_)) => self.delete(&operands, recursive),
            (Targets::WritesAll, Some(_)) => self.writes.extend(operands.iter().map(owned)),
            (Targets::WritesLast, Some((last, sources))) if !sources.is_empty() => {
                self.writes.push(owned(last));
            }
            (Targets::Moves, Some((last, sources))) if !sources.is_empty() => {
                self.delete(sources, true);
                self.writes.push(owned(last));
            }
            (Targets::Links, Some((last, []))) => self.writes.push(basename(last).to_string()),
            (Targets::Links, Some((last, _))) => self.writes.push(owned(last)),
            (Targets::WritesLast | Targets::Moves, Some(_)) => {}
        }
        Ok(())
    }

    fn delete(&mut self, paths: &[&str], recursive: bool) {
        self.deletes.extend(paths.iter().map(|path| Delete {
            path: path.to_string(),
            recursive,
        }));
    }

    /// Fail when the script changes directory and names a relative target, whose directory is
    /// then only known at run time.
    fn relative_targets(&self) -> Result<(), String> {
        if !self.changes_directory {
            return Ok(());
        }
        self.writes
            .iter()
            .chain(self.deletes.iter().map(|delete| &delete.path))
            .find(|path| !path.starts_with(['/', '~']))
            .map_or(Ok(()), |path| {
                Err(format!(
                    "the script changes directory, so where `{path}` is cannot be known"
                ))
            })
    }

    fn into_class(self) -> Class {
        if self.runs_edit_program || !self.writes.is_empty() {
            Class::Edit {
                writes: self.writes,
                deletes: self.deletes,
            }
        } else {
            Class::ReadOnly
        }
    }
}

/// Classify `script` as read-only, an edit of the paths it names, or unvetted. Redirections
/// into `scratch`, like those into `/tmp`, are not edits.
pub fn classify(script: &str, scratch: Option<&Path>) -> Class {
    let mut classifier = Classifier {
        scratch: scratch.and_then(Path::to_str),
        ..Classifier::default()
    };
    match classifier
        .script(script, 0)
        .and_then(|()| classifier.relative_targets())
    {
        Ok(()) => classifier.into_class(),
        Err(why) => Class::Unvetted { why },
    }
}

/// Vet a read-only program's own options against its `disqualifiers`.
fn read_only(
    name: &str,
    args: &[Word],
    disqualifiers: &[Disqualifier],
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

/// `args` split into operands and vetted options; fails on an option not vetted for `edit` or
/// an argument only known at run time.
fn edit_args<'w>(edit: &EditProgram, args: &'w [Word]) -> Result<EditArgs<'w>, String> {
    let mut parsed = EditArgs {
        operands: Vec::new(),
        recursive: false,
    };
    let mut options = true;
    let mut words = args.iter();
    while let Some(arg) = words.next() {
        let text = arg.text.as_str();
        if arg.dynamic || arg.splits {
            return Err(format!(
                "`{} {text}` names a path only known at run time",
                edit.name
            ));
        }
        if options && text == "--" {
            options = false;
        } else if let Some(long) = text.strip_prefix("--").filter(|_| options) {
            if !edit.long_flags.contains(&long) {
                return Err(format!("`{} {text}` is not a vetted option", edit.name));
            }
            parsed.recursive |= long == "recursive";
        } else if let Some(cluster) = short_options(text).filter(|_| options) {
            if edit_flags(edit, cluster)? {
                words.next();
            }
            parsed.recursive |= cluster.contains(['r', 'R']);
        } else {
            parsed.operands.push(text);
        }
    }
    Ok(parsed)
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

#[cfg(test)]
mod tests;
