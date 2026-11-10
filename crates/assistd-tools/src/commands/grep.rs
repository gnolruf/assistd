//! `grep`: filter lines of files, directories or stdin by a GNU-style regex.

use std::borrow::Cow;
use std::fs::Metadata;
use std::io::{self, ErrorKind};
use std::path::{Path, PathBuf};

use assistd_utils::text::human_size;
use async_trait::async_trait;
use fancy_regex::Regex;

use crate::chain::OUTPUT_MAX;
use crate::command::{Command, CommandInput, CommandOutput, Hint, error_line, io_error_nav};
use crate::commands::{FILE_READ_MAX, read_regular_file};
use crate::policy::{FileScope, on_blocking_pool};

use self::pattern::{Dialect, MatchOptions, has_bare_pipe};

mod pattern;

const STDIN_LABEL: &str = "(standard input)";
/// Directory entries `-r` may visit before giving up on the walk.
const WALK_ENTRY_MAX: usize = 50_000;

/// `grep [-cilnrvwEFGP] [-e PATTERN] [-A N] [-B N] [-C N] PATTERN [FILE|DIR]...`: print lines
/// from the named files or stdin matching `PATTERN`. Exits 0 on a match, 1 on none, 2 on errors.
#[derive(Debug)]
pub struct GrepCommand;

#[derive(Default)]
struct Flags {
    matching: MatchOptions,
    /// Patterns given with `-e`; when empty, the first operand is PATTERN.
    patterns: Vec<String>,
    invert: bool,
    count_only: bool,
    line_numbers: bool,
    recursive: bool,
    files_only: bool,
    before: usize,
    after: usize,
}

impl Flags {
    fn prints_lines(&self) -> bool {
        !self.count_only && !self.files_only
    }

    fn has_context(&self) -> bool {
        self.before > 0 || self.after > 0
    }

    /// Apply a flag that takes a value: `-e PATTERN` or a context count.
    fn set_value(&mut self, flag: char, value: Option<&str>) -> Result<(), String> {
        if flag == 'e' {
            let pattern = value.ok_or("flag '-e' needs a pattern, as in -e 'a|b'")?;
            self.patterns.push(pattern.to_owned());
            return Ok(());
        }
        let lines = parse_context_lines(flag, value)?;
        if flag != 'A' {
            self.before = lines;
        }
        if flag != 'B' {
            self.after = lines;
        }
        Ok(())
    }
}

struct OutputLine<'a> {
    label: Option<&'a str>,
    number: Option<usize>,
    matched: bool,
    text: &'a str,
}

impl OutputLine<'_> {
    fn write_to(&self, out: &mut Vec<u8>) {
        let separator = if self.matched { ':' } else { '-' };
        if let Some(label) = self.label {
            out.extend_from_slice(format!("{label}{separator}").as_bytes());
        }
        if let Some(number) = self.number {
            out.extend_from_slice(format!("{number}{separator}").as_bytes());
        }
        out.extend_from_slice(self.text.as_bytes());
        if !self.text.ends_with('\n') {
            out.push(b'\n');
        }
    }
}

/// A file to search, tagged with how it was reached.
enum Target {
    /// Named on the command line; failing to read it is an error.
    Named(PathBuf),
    /// Found while descending; skipped when unreadable.
    Found(PathBuf),
}

impl Target {
    fn path(&self) -> &Path {
        match self {
            Self::Named(path) | Self::Found(path) => path,
        }
    }
}

/// A path named on the command line, classified before any search.
enum Root<'a> {
    File(&'a Path),
    /// A directory to walk under `-r`.
    Tree(&'a Path),
}

/// One input line, decoded lossily; unprintable when its file holds a
/// NUL byte or the line itself is not valid UTF-8.
struct Line<'a> {
    text: Cow<'a, str>,
    printable: bool,
}

/// Matched-line count of one input, and whether a selected line was
/// withheld for being binary.
struct ScanResult {
    count: usize,
    suppressed: bool,
}

/// Writes selected lines to `out`, withholding unprintable ones and
/// emitting a group separator before the next printed line when pending.
struct Printer<'a> {
    lines: &'a [Line<'a>],
    label: Option<&'a str>,
    line_numbers: bool,
    out: &'a mut Vec<u8>,
    pending_separator: bool,
    suppressed: bool,
}

impl Printer<'_> {
    fn is_full(&self) -> bool {
        self.out.len() > OUTPUT_MAX
    }

    fn print(&mut self, index: usize, matched: bool) {
        let line = &self.lines[index];
        if !line.printable {
            self.suppressed = true;
            return;
        }
        if std::mem::take(&mut self.pending_separator) && !self.out.is_empty() {
            self.out.extend_from_slice(b"--\n");
        }
        OutputLine {
            label: self.label,
            number: self.line_numbers.then_some(index + 1),
            matched,
            text: &line.text,
        }
        .write_to(self.out);
    }
}

/// Running state of one `grep` over files: accumulated output and
/// binary-match notices, match total, oversized files skipped, and how
/// many directory entries the walk may still visit.
struct Search<'a> {
    scope: &'a FileScope,
    re: &'a Regex,
    flags: &'a Flags,
    label_lines: bool,
    out: Vec<u8>,
    notices: Vec<u8>,
    total: usize,
    too_large_skipped: usize,
    entries_left: usize,
}

impl Search<'_> {
    fn is_full(&self) -> bool {
        self.out.len() > OUTPUT_MAX
    }

    async fn search_target(&mut self, target: &Target) -> Result<(), TargetError> {
        let Some(bytes) = self.read_bytes(target).await? else {
            return Ok(());
        };
        let display = target.path().to_string_lossy();
        let label = self.label_lines.then_some(display.as_ref());
        let result = scan(self.re, self.flags, &bytes, label, &mut self.out)
            .map_err(TargetError::MatchFailed)?;
        if result.suppressed {
            self.notices
                .extend(binary_match_line(&display).into_bytes());
        }
        if self.flags.files_only {
            self.out.extend(matching_file_line(&display, result.count));
        } else if self.flags.count_only && self.label_lines {
            self.out
                .extend_from_slice(format!("{display}:{}\n", result.count).as_bytes());
        }
        self.total += result.count;
        Ok(())
    }

    /// Bytes of `target`, or `None` for an unreadable [`Target::Found`],
    /// counting the ones over the read limit.
    async fn read_bytes(&mut self, target: &Target) -> Result<Option<Vec<u8>>, TargetError> {
        match (read_regular_file(self.scope, target.path()).await, target) {
            (Ok(bytes), _) => Ok(Some(bytes)),
            (Err(source), Target::Found(_)) => {
                self.too_large_skipped += usize::from(source.kind() == ErrorKind::FileTooLarge);
                Ok(None)
            }
            (Err(source), Target::Named(path)) => Err(TargetError::Unreadable {
                path: path.to_string_lossy().into_owned(),
                source,
            }),
        }
    }

    /// Search every file under `root` in sorted depth-first order while
    /// walking, until the output or the entry budget is used up.
    async fn search_tree(&mut self, root: &Path) -> Result<(), TargetError> {
        let mut stack = vec![root.to_path_buf()];
        while let Some(dir) = stack.pop() {
            let (files, dirs) = list_dir(self.scope, &dir, &mut self.entries_left).await;
            for file in files {
                self.search_target(&Target::Found(file)).await?;
                if self.is_full() {
                    return Ok(());
                }
            }
            if self.entries_left == 0 {
                return Ok(());
            }
            stack.extend(dirs.into_iter().rev());
        }
        Ok(())
    }

    fn into_output(self) -> CommandOutput {
        let stdout = if self.flags.count_only && !self.flags.files_only && !self.label_lines {
            format!("{}\n", self.total).into_bytes()
        } else {
            self.out
        };
        let mut output = outcome(self.total, stdout);
        output.stderr = self.notices;
        if let Some(note) = too_large_note(self.too_large_skipped) {
            output.stderr.extend_from_slice(note.as_bytes());
        }
        if self.entries_left == 0 {
            output.stderr.extend(
                error_line(
                    "grep",
                    format_args!("stopped after visiting {WALK_ENTRY_MAX} directory entries"),
                    Hint::Try,
                    "a narrower directory",
                )
                .into_bytes(),
            );
        }
        output
    }
}

enum TargetError {
    Unreadable { path: String, source: io::Error },
    DirectoryWithoutRecursion { path: String },
    MatchFailed(fancy_regex::Error),
}

impl TargetError {
    fn error_line(&self) -> String {
        match self {
            Self::Unreadable { path, source } => io_error_nav("grep", path, source),
            Self::DirectoryWithoutRecursion { path } => error_line(
                "grep",
                format_args!("{path} is a directory"),
                Hint::Use,
                format_args!("grep -r PATTERN {path}"),
            ),
            Self::MatchFailed(source) => match_failed_line(source),
        }
    }
}

#[async_trait]
impl Command for GrepCommand {
    fn name(&self) -> &'static str {
        "grep"
    }

    fn summary(&self) -> &'static str {
        "filter lines by GNU grep regex: BRE, -E/-P extended, -F fixed; -icnrvlw -e -ABC"
    }

    fn help(&self) -> String {
        "usage: grep [-cilnrvwEFGP] [-e PATTERN]... [-A N] [-B N] [-C N] PATTERN [FILE|DIR]...\n\
         \n\
         Print lines matching the regex PATTERN, read from the named \
         paths or from stdin when none are given.\n\
         \n\
         PATTERN follows GNU grep. By default it is a basic regex: \
         alternation is `a\\|b`, groups `\\(a\\)`, repeats `a\\+`, \
         `a\\?`, `a\\{2\\}`, and the bare `|`, `(`, `+`, `?`, `{` match \
         themselves. With -E or -P it is an extended regex: `a|b`, \
         `(a)`, `a+`. Both dialects also accept `\\d`, `\\w`, `\\s`, \
         `\\b`, `\\<`, `\\>` and backreferences like `\\1`; -E and -P \
         add lookaround such as `(?=x)` and `(?<!x)`. Quote the pattern \
         so `|` cannot start a pipeline.\n\
         \n\
         Flags:\n  \
           -i  case-insensitive\n  \
           -v  invert match (print non-matching lines)\n  \
           -c  print the count instead of the matching lines\n  \
           -n  prefix each line with its 1-based line number\n  \
           -r  descend into directory arguments\n  \
           -l  print only the names of files with a match\n  \
           -w  match only whole words\n  \
           -G  PATTERN is a basic regex (the default)\n  \
           -E, -P  PATTERN is an extended regex\n  \
           -F  PATTERN is a literal string\n  \
           -e PATTERN  use PATTERN; repeat to match any of several\n  \
           -A N  also print N lines after each match\n  \
           -B N  also print N lines before each match\n  \
           -C N  also print N lines before and after each match\n\
         \n\
         Flags can be combined (e.g. `-rn`, `-nC2`) and must come \
         before PATTERN. With -e, every operand is a path. Context lines \
         use `-` where a matching line has `:`, and `--` separates \
         groups that are not adjacent. Output lines carry a `PATH:` \
         prefix whenever more than one file is searched. As in GNU \
         grep, an input with a NUL byte prints none of its lines, and a \
         line that is not valid UTF-8 is never printed; when such a line \
         is selected, stderr says `grep: PATH: binary file matches`. \
         Symlinks, unreadable entries and files over the read limit found \
         while descending are skipped, with a count of the oversized ones \
         on stderr; a path named on the command line that cannot be read \
         is an error.\n\
         \n\
         Exit 0 if any line matched (or the count is non-zero under \
         `-c`), 1 if no matches, 2 on usage/input errors.\n"
            .to_string()
    }

    async fn run(&self, input: CommandInput) -> CommandOutput {
        if input.args.is_empty() {
            return CommandOutput::usage(self.help());
        }
        let (flags, operands) = match parse_flags(&input.args) {
            Ok(v) => v,
            Err(msg) => {
                return CommandOutput::usage_error(
                    "grep",
                    msg,
                    "grep (no args) for supported flags",
                );
            }
        };
        let Some((patterns, paths)) = split_patterns(&flags, operands) else {
            return CommandOutput::usage(self.help());
        };
        let re = match pattern::compile(patterns, &flags.matching) {
            Ok(re) => re,
            Err(message) => return bad_pattern(&message),
        };

        let output = if paths.is_empty() {
            match input.stdin {
                Some(stdin) => search_stdin(&re, &flags, &stdin),
                None => return CommandOutput::usage(self.help()),
            }
        } else {
            match search_paths(&input.scope, &re, &flags, paths).await {
                Ok(out) => out,
                Err(e) => return CommandOutput::failed(2, e.error_line().into_bytes()),
            }
        };
        annotate_dialect(output, flags.matching.dialect, patterns)
    }
}

fn parse_flags(argv: &[String]) -> Result<(Flags, &[String]), String> {
    let mut flags = Flags::default();
    let mut pos = 0;
    while pos < argv.len() {
        let arg = &argv[pos];
        if arg == "--" {
            pos += 1;
            break;
        }
        if let Some(rest) = arg.strip_prefix('-') {
            if rest.is_empty() {
                break;
            }
            for (offset, ch) in rest.char_indices() {
                match ch {
                    'i' => flags.matching.case_insensitive = true,
                    'v' => flags.invert = true,
                    'c' => flags.count_only = true,
                    'n' => flags.line_numbers = true,
                    'r' => flags.recursive = true,
                    'l' => flags.files_only = true,
                    'w' => flags.matching.whole_word = true,
                    'G' => flags.matching.dialect = Dialect::Basic,
                    'E' | 'P' => flags.matching.dialect = Dialect::Extended,
                    'F' => flags.matching.dialect = Dialect::Fixed,
                    'A' | 'B' | 'C' | 'e' => {
                        let attached = &rest[offset + 1..];
                        let value = if attached.is_empty() {
                            pos += 1;
                            argv.get(pos).map(String::as_str)
                        } else {
                            Some(attached)
                        };
                        flags.set_value(ch, value)?;
                        break;
                    }
                    other => return Err(format!("unknown flag '-{other}'")),
                }
            }
            pos += 1;
        } else {
            break;
        }
    }
    Ok((flags, &argv[pos..]))
}

/// The patterns to match and the paths to search: the `-e` patterns and
/// every operand, or else the first operand and the rest.
fn split_patterns<'a>(
    flags: &'a Flags,
    operands: &'a [String],
) -> Option<(&'a [String], &'a [String])> {
    if flags.patterns.is_empty() {
        operands
            .split_first()
            .map(|(pattern, paths)| (std::slice::from_ref(pattern), paths))
    } else {
        Some((&flags.patterns, operands))
    }
}

fn bad_pattern(message: &str) -> CommandOutput {
    CommandOutput::failed(
        2,
        error_line(
            "grep",
            format_args!("bad regex pattern: {message}"),
            Hint::Check,
            "escape regex metachars, or use -F for a literal string; run grep for usage",
        )
        .into_bytes(),
    )
}

fn match_failed_line(source: &fancy_regex::Error) -> String {
    error_line(
        "grep",
        source,
        Hint::Try,
        "a pattern without nested repeats or lookaround",
    )
}

fn parse_context_lines(flag: char, value: Option<&str>) -> Result<usize, String> {
    value
        .and_then(|v| v.parse().ok())
        .ok_or_else(|| format!("flag '-{flag}' needs a line count, as in -{flag} 3"))
}

fn search_stdin(re: &Regex, flags: &Flags, stdin: &[u8]) -> CommandOutput {
    let mut out = Vec::new();
    let result = match scan(re, flags, stdin, None, &mut out) {
        Ok(result) => result,
        Err(source) => return CommandOutput::failed(2, match_failed_line(&source).into_bytes()),
    };
    let stdout = if flags.files_only {
        matching_file_line(STDIN_LABEL, result.count)
    } else if flags.count_only {
        format!("{}\n", result.count).into_bytes()
    } else {
        out
    };
    let mut output = outcome(result.count, stdout);
    if result.suppressed {
        output.stderr = binary_match_line(STDIN_LABEL).into_bytes();
    }
    output
}

async fn search_paths(
    scope: &FileScope,
    re: &Regex,
    flags: &Flags,
    paths: &[String],
) -> Result<CommandOutput, TargetError> {
    let roots = resolve_roots(scope, paths, flags.recursive).await?;
    let mut search = Search {
        scope,
        re,
        flags,
        label_lines: roots.len() > 1 || flags.recursive,
        out: Vec::new(),
        notices: Vec::new(),
        total: 0,
        too_large_skipped: 0,
        entries_left: WALK_ENTRY_MAX,
    };
    for root in &roots {
        match root {
            Root::File(path) => {
                search
                    .search_target(&Target::Named(path.to_path_buf()))
                    .await?;
            }
            Root::Tree(path) => search.search_tree(path).await?,
        }
        if search.is_full() || search.entries_left == 0 {
            break;
        }
    }
    Ok(search.into_output())
}

fn files_noun(count: usize) -> &'static str {
    if count == 1 { "file" } else { "files" }
}

fn too_large_note(count: usize) -> Option<String> {
    (count > 0).then(|| {
        error_line(
            "grep",
            format_args!(
                "skipped {count} {} over the {} read limit",
                files_noun(count),
                human_size(FILE_READ_MAX)
            ),
            Hint::Try,
            "a narrower directory, or head/tail on a large file",
        )
    })
}

/// GNU grep's stderr notice for a matching input whose lines were withheld.
fn binary_match_line(name: &str) -> String {
    format!("grep: {name}: binary file matches\n")
}

/// Lines of `bytes`, all unprintable when it holds a NUL byte.
fn split_lines(bytes: &[u8]) -> Vec<Line<'_>> {
    let has_nul = bytes.contains(&0);
    bytes
        .split_inclusive(|&byte| byte == b'\n')
        .map(|raw| {
            let text = String::from_utf8_lossy(raw);
            let printable = !has_nul && matches!(text, Cow::Borrowed(_));
            Line { text, printable }
        })
        .collect()
}

/// Write the printable lines of `bytes` that `flags` selects to `out` and
/// count the matches; fails when the matcher exceeds its backtracking limit.
fn scan(
    re: &Regex,
    flags: &Flags,
    bytes: &[u8],
    label: Option<&str>,
    out: &mut Vec<u8>,
) -> Result<ScanResult, fancy_regex::Error> {
    let lines = split_lines(bytes);
    let mut printer = Printer {
        lines: &lines,
        label,
        line_numbers: flags.line_numbers,
        out,
        pending_separator: false,
        suppressed: false,
    };
    let mut count = 0;
    let mut next_unprinted = 0;
    let mut trailing = 0;
    for (index, line) in lines.iter().enumerate() {
        let content = line.text.strip_suffix('\n').unwrap_or(&line.text);
        let matched = re.is_match(content)? ^ flags.invert;
        count += usize::from(matched);
        if !flags.prints_lines() {
            continue;
        }
        if matched {
            let first = index.saturating_sub(flags.before).max(next_unprinted);
            let starts_group = next_unprinted == 0 || first > next_unprinted;
            printer.pending_separator |= flags.has_context() && starts_group;
            for before in first..index {
                printer.print(before, false);
                if printer.is_full() {
                    break;
                }
            }
            printer.print(index, true);
            trailing = flags.after;
        } else if trailing > 0 {
            printer.print(index, false);
            trailing -= 1;
        } else {
            continue;
        }
        next_unprinted = index + 1;
        if printer.is_full() {
            break;
        }
    }
    Ok(ScanResult {
        count,
        suppressed: printer.suppressed,
    })
}

fn matching_file_line(name: &str, count: usize) -> Vec<u8> {
    if count == 0 {
        Vec::new()
    } else {
        format!("{name}\n").into_bytes()
    }
}

/// A zero-match result is the only moment a dialect mismatch is
/// visible, so that is where the explanation goes.
fn annotate_dialect(
    mut out: CommandOutput,
    dialect: Dialect,
    patterns: &[String],
) -> CommandOutput {
    let piped = patterns.iter().any(|pattern| has_bare_pipe(pattern));
    if out.exit_code != 1 || !out.stderr.is_empty() || dialect != Dialect::Basic || !piped {
        return out;
    }
    out.stderr = error_line(
        "grep",
        "no matches; a bare `|` matches itself in a basic regex",
        Hint::Use,
        "grep -E \"a|b\" FILE or grep \"a\\|b\" FILE for alternation",
    )
    .into_bytes();
    out
}

fn outcome(count: usize, stdout: Vec<u8>) -> CommandOutput {
    CommandOutput {
        stdout,
        stderr: Vec::new(),
        exit_code: i32::from(count == 0),
        attachments: Vec::new(),
    }
}

async fn resolve_roots<'p>(
    scope: &FileScope,
    paths: &'p [String],
    recursive: bool,
) -> Result<Vec<Root<'p>>, TargetError> {
    let mut roots = Vec::with_capacity(paths.len());
    for raw in paths {
        let path = Path::new(raw);
        let meta =
            symlink_metadata(scope, path)
                .await
                .map_err(|source| TargetError::Unreadable {
                    path: raw.clone(),
                    source,
                })?;
        if !meta.is_dir() {
            roots.push(Root::File(path));
        } else if recursive {
            roots.push(Root::Tree(path));
        } else {
            return Err(TargetError::DirectoryWithoutRecursion { path: raw.clone() });
        }
    }
    Ok(roots)
}

async fn symlink_metadata(scope: &FileScope, path: &Path) -> io::Result<Metadata> {
    let scope = scope.clone();
    let path = path.to_path_buf();
    on_blocking_pool(move || scope.symlink_metadata(&path)).await
}

/// Sorted file and subdirectory paths of `dir`, skipping symlinks and
/// unreadable entries; reads at most `*entries_left` entries and
/// subtracts what it read.
async fn list_dir(
    scope: &FileScope,
    dir: &Path,
    entries_left: &mut usize,
) -> (Vec<PathBuf>, Vec<PathBuf>) {
    let scope = scope.clone();
    let dir = dir.to_path_buf();
    let mut left = *entries_left;
    let (listing, left) = on_blocking_pool(move || {
        let listing = read_dir_sorted(&scope, &dir, &mut left);
        (listing, left)
    })
    .await;
    *entries_left = left;
    listing
}

fn read_dir_sorted(
    scope: &FileScope,
    dir: &Path,
    entries_left: &mut usize,
) -> (Vec<PathBuf>, Vec<PathBuf>) {
    let mut files = Vec::new();
    let mut dirs = Vec::new();
    let Ok(mut reader) = scope.read_dir(dir) else {
        return (files, dirs);
    };
    while *entries_left > 0 {
        let Some(Ok(entry)) = reader.next() else {
            break;
        };
        *entries_left -= 1;
        let Ok(kind) = entry.file_type() else {
            continue;
        };
        let path = dir.join(entry.file_name());
        if kind.is_dir() {
            dirs.push(path);
        } else if !kind.is_symlink() {
            files.push(path);
        }
    }
    files.sort();
    dirs.sort();
    (files, dirs)
}

#[cfg(test)]
mod tests {
    use tempfile::{TempDir, tempdir};

    use super::*;

    /// `root/top.txt`, `root/sub/deep.txt`, `root/sub/notes.bin`.
    fn tree() -> TempDir {
        let dir = tempdir().expect("tempdir");
        std::fs::write(dir.path().join("top.txt"), b"alpha ERROR\nbeta\n").expect("top");
        std::fs::create_dir(dir.path().join("sub")).expect("mkdir");
        std::fs::write(dir.path().join("sub/deep.txt"), b"gamma\ndelta ERROR\n").expect("deep");
        std::fs::write(dir.path().join("sub/notes.bin"), b"ERROR\0binary\n").expect("bin");
        dir
    }

    fn write_oversized(path: &Path) {
        std::fs::File::create(path)
            .expect("create")
            .set_len(FILE_READ_MAX + 1)
            .expect("set_len");
    }

    async fn run_grep(args: &[&str], stdin: &[u8]) -> CommandOutput {
        GrepCommand
            .run(CommandInput {
                args: args.iter().map(ToString::to_string).collect(),
                stdin: Some(stdin.to_vec()),
                scope: FileScope::Host,
            })
            .await
    }

    #[tokio::test]
    async fn filters_stdin_per_flags() {
        let cases: [(&[&str], &[u8], i32, &str); 4] = [
            (
                &["-E", "Command|Tool"],
                b"a Command here\nnothing\na Tool there\n",
                0,
                "a Command here\na Tool there\n",
            ),
            (
                &["-n", "ERROR"],
                b"ok\nERROR one\nok\nERROR two\n",
                0,
                "2:ERROR one\n4:ERROR two\n",
            ),
            (&["-ivc", "error"], b"ERROR\ninfo\nError\nok\n", 0, "2\n"),
            (&["-c", "ZZZ"], b"a\nb\n", 1, "0\n"),
        ];
        for (args, stdin, exit_code, stdout) in cases {
            let out = run_grep(args, stdin).await;
            assert_eq!(out.exit_code, exit_code, "{args:?}");
            assert_eq!(String::from_utf8_lossy(&out.stdout), stdout, "{args:?}");
            assert!(out.stderr.is_empty(), "{args:?}: {out:?}");
        }
    }

    async fn assert_matches(cases: &[(&[&str], &str, &str)]) {
        for (args, stdin, stdout) in cases {
            let out = run_grep(args, stdin.as_bytes()).await;
            assert_eq!(String::from_utf8_lossy(&out.stdout), *stdout, "{args:?}");
            assert!(out.exit_code < 2, "{args:?}: {out:?}");
        }
    }

    #[tokio::test]
    async fn basic_regex_reads_gnu_operators_and_literals() {
        assert_matches(&[
            (
                &[r"Command\|Tool"],
                "a Command\nnone\na Tool\n",
                "a Command\na Tool\n",
            ),
            (&[r"\(ab\)\+$"], "abab\nab c\n", "abab\n"),
            (&[r"x\{2\}"], "x\nxx\n", "xx\n"),
            (&[r"colou\?r"], "color\ncolour\n", "color\ncolour\n"),
            (&[r"\(ab\)\1"], "abab\nab\n", "abab\n"),
            (&[r"v\d\+"], "v12\nvx\n", "v12\n"),
            (&[r"\<cat\>"], "cat\nconcat\n", "cat\n"),
            (&["fn main("], "fn main() {\nmain\n", "fn main() {\n"),
            (&["a+b"], "a+b\naab\n", "a+b\n"),
            (&["cost $5"], "cost $5\ncost 5\n", "cost $5\n"),
            (&["*.rs"], "src/*.rs\nmain.rs\n", "src/*.rs\n"),
            (&["^#[^!]"], "#[test]\n#!x\n", "#[test]\n"),
            (&[r"[\.|(]x"], "(x\n.x\nax\n", "(x\n.x\n"),
            (&["[[:digit:]]]"], "1]\n1\n", "1]\n"),
        ])
        .await;
    }

    #[tokio::test]
    async fn dialect_flags_match_as_intended() {
        assert_matches(&[
            (&["-E", r"(ab)+$"], "abab\nab c\n", "abab\n"),
            (&["-E", "a+b"], "a+b\naab\n", "aab\n"),
            (&["-P", r"foo(?=bar)"], "foobar\nfoobaz\n", "foobar\n"),
            (
                &["-P", r"(?<!un)safe"],
                "unsafe\nsafe code\n",
                "safe code\n",
            ),
            (&["-E", r"(\w+) \1"], "the the\nthe cat\n", "the the\n"),
            (&["-F", "a.b(c"], "a.b(c\naxb(c\n", "a.b(c\n"),
            (&["-w", "id"], "id = 1\nidle\nuser_id\n", "id = 1\n"),
            (&["-wF", "a.b"], "x a.b y\nxa.b\n", "x a.b y\n"),
            (
                &["-e", "foo", "-e", "-bar"],
                "foo\nbaz\n-bar\n",
                "foo\n-bar\n",
            ),
            (&["-ie", r"x\|Y"], "X\ny\nz\n", "X\ny\n"),
        ])
        .await;
    }

    #[tokio::test]
    async fn bare_pipe_that_matches_nothing_explains_the_dialect() {
        let out = run_grep(&["Command|Tool"], b"a Command here\na Tool there\n").await;
        assert_eq!(out.exit_code, 1);
        assert_eq!(
            String::from_utf8_lossy(&out.stderr),
            "[error] grep: no matches; a bare `|` matches itself in a basic regex. \
             Use: grep -E \"a|b\" FILE or grep \"a\\|b\" FILE for alternation\n"
        );
    }

    #[tokio::test]
    async fn multiple_files_are_path_prefixed() {
        let dir = tree();
        let a = dir.path().join("top.txt").to_string_lossy().into_owned();
        let b = dir
            .path()
            .join("sub/deep.txt")
            .to_string_lossy()
            .into_owned();
        let out = run_grep(&["ERROR", &a, &b], b"").await;
        assert_eq!(out.exit_code, 0);
        assert_eq!(
            String::from_utf8_lossy(&out.stdout),
            format!("{a}:alpha ERROR\n{b}:delta ERROR\n")
        );
    }

    #[tokio::test]
    async fn r_flag_descends_in_order_and_reports_binary_matches() {
        let dir = tree();
        let root = dir.path().to_string_lossy().into_owned();
        let out = run_grep(&["-rn", "ERROR", &root], b"").await;
        assert_eq!(out.exit_code, 0);
        assert_eq!(
            String::from_utf8_lossy(&out.stdout),
            format!("{root}/top.txt:1:alpha ERROR\n{root}/sub/deep.txt:2:delta ERROR\n")
        );
        assert_eq!(
            String::from_utf8_lossy(&out.stderr),
            format!("grep: {root}/sub/notes.bin: binary file matches\n")
        );
    }

    #[tokio::test]
    async fn binary_lines_are_withheld_as_gnu_grep_does() {
        let dir = tree();
        let latin1 = dir.path().join("app.log");
        std::fs::write(&latin1, b"ok one\ncaf\xe9 ok\nok three\n").expect("latin1");
        let latin1 = latin1.to_string_lossy().into_owned();
        let binary = dir
            .path()
            .join("sub/notes.bin")
            .to_string_lossy()
            .into_owned();
        let cases: [(&[&str], &[u8], &str, String); 5] = [
            (&["ERROR", &latin1], b"", "", String::new()),
            (
                &["-n", "ok", &latin1],
                b"",
                "1:ok one\n3:ok three\n",
                format!("grep: {latin1}: binary file matches\n"),
            ),
            (
                &["ERROR", &binary],
                b"",
                "",
                format!("grep: {binary}: binary file matches\n"),
            ),
            (&["-c", "ERROR", &binary], b"", "1\n", String::new()),
            (
                &["x"],
                b"x\0y\n",
                "",
                "grep: (standard input): binary file matches\n".to_owned(),
            ),
        ];
        for (args, stdin, stdout, stderr) in cases {
            let out = run_grep(args, stdin).await;
            assert_eq!(String::from_utf8_lossy(&out.stdout), stdout, "{args:?}");
            assert_eq!(String::from_utf8_lossy(&out.stderr), stderr, "{args:?}");
        }
    }

    #[tokio::test]
    async fn context_flags_print_neighbouring_lines() {
        let stdin = b"a1\nb2\nHIT3\nc4\nd5\ne6\nf7\nHIT8\ng9\n";
        let cases: [(&[&str], &str); 3] = [
            (
                &["-nC1", "HIT"],
                "2-b2\n3:HIT3\n4-c4\n--\n7-f7\n8:HIT8\n9-g9\n",
            ),
            (
                &["-C", "2", "HIT"],
                "a1\nb2\nHIT3\nc4\nd5\ne6\nf7\nHIT8\ng9\n",
            ),
            (&["-A", "9", "HIT8"], "HIT8\ng9\n"),
        ];
        for (args, stdout) in cases {
            let out = run_grep(args, stdin).await;
            assert_eq!(out.exit_code, 0, "{args:?}");
            assert_eq!(String::from_utf8_lossy(&out.stdout), stdout, "{args:?}");
        }
    }

    #[tokio::test]
    async fn output_stops_growing_once_past_output_max() {
        let dir = tempdir().expect("tempdir");
        let line = format!("{}\n", "x".repeat(1024 * 1024));
        let path_of = |name: &str| dir.path().join(name).to_string_lossy().into_owned();
        let (first, second, never_read) = (path_of("a.txt"), path_of("b.txt"), path_of("c.txt"));
        std::fs::write(&first, line.repeat(6)).expect("a");
        std::fs::write(&second, line.repeat(6)).expect("b");
        std::fs::write(&never_read, &line).expect("c");
        let out = run_grep(&["x", &first, &second, &never_read], b"").await;
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(stdout.len() <= OUTPUT_MAX + second.len() + line.len() + 1);
        assert!(
            !stdout.contains(&never_read),
            "third file must not be searched"
        );
    }

    #[tokio::test]
    async fn unreadable_file_found_while_descending_is_skipped() {
        let dir = tree();
        write_oversized(&dir.path().join("sub/huge.log"));
        let root = dir.path().to_string_lossy().into_owned();
        let out = run_grep(&["-rl", "ERROR", &root], b"").await;
        assert_eq!(out.exit_code, 0);
        assert!(
            String::from_utf8_lossy(&out.stderr).contains("skipped 1 file over the"),
            "{out:?}"
        );
        assert_eq!(
            String::from_utf8_lossy(&out.stdout),
            format!("{root}/top.txt\n{root}/sub/deep.txt\n{root}/sub/notes.bin\n")
        );
    }
}
