use std::path::{Path, PathBuf};

use async_trait::async_trait;
use regex::{Regex, RegexBuilder};

use crate::command::{Command, CommandInput, CommandOutput, Hint, error_line, io_error_nav};
use crate::commands::cat::sniff_binary;
use crate::commands::read_regular_file;

/// BRE metacharacters spelled with a backslash, which the regex crate
/// reads as the literal character.
const BRE_ESCAPES: [&str; 7] = [r"\|", r"\(", r"\)", r"\{", r"\}", r"\+", r"\?"];

/// `grep [-icnrv] PATTERN [FILE|DIR]...`: print lines from the named files
/// or stdin matching `PATTERN`. Exits 0 on a match, 1 on none, 2 on errors.
pub struct GrepCommand;

#[derive(Default)]
struct Flags {
    case_insensitive: bool,
    invert: bool,
    count_only: bool,
    line_numbers: bool,
    recursive: bool,
}

enum TargetError {
    Unreadable {
        path: String,
        source: std::io::Error,
    },
    DirectoryWithoutRecursion {
        path: String,
    },
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
        }
    }
}

#[async_trait]
impl Command for GrepCommand {
    fn name(&self) -> &str {
        "grep"
    }

    fn summary(&self) -> &'static str {
        "filter lines matching an ERE regex (quote it: \"a|b\"); -i, -v, -c, -n, -r"
    }

    fn help(&self) -> String {
        "usage: grep [-icnrv] PATTERN [FILE|DIR]...\n\
         \n\
         Print lines matching the regex PATTERN, read from the named \
         paths or from stdin when none are given.\n\
         \n\
         PATTERN is Rust/ERE regex, not BRE: alternation is `a|b` and \
         groups are `(a)b`, so quote the pattern to keep `|` from \
         starting a pipeline (`grep \"Command|Tool\" FILE`). The BRE \
         spellings `\\|`, `\\(`, `\\+` match those characters \
         literally here.\n\
         \n\
         Flags:\n  \
           -i  case-insensitive\n  \
           -v  invert match (print non-matching lines)\n  \
           -c  print the count instead of the matching lines\n  \
           -n  prefix each line with its 1-based line number\n  \
           -r  descend into directory arguments\n\
         \n\
         Flags can be combined (e.g. `-rn`). Output lines carry a \
         `PATH:` prefix whenever more than one file is searched. Binary \
         files, symlinks and unreadable entries found while descending \
         are skipped; a path named on the command line that cannot be \
         read is an error.\n\
         \n\
         Exit 0 if any line matched (or the count is non-zero under \
         `-c`), 1 if no matches, 2 on usage/input errors.\n"
            .to_string()
    }

    async fn run(&self, input: CommandInput) -> CommandOutput {
        if input.args.is_empty() {
            return CommandOutput::usage(self.help());
        }
        let (flags, positional) = match parse_flags(&input.args) {
            Ok(v) => v,
            Err(msg) => {
                return CommandOutput::usage_error(
                    "grep",
                    msg,
                    "grep (no args) for supported flags",
                );
            }
        };
        if positional.is_empty() {
            return CommandOutput::usage(self.help());
        }

        let pattern = &positional[0];
        let re = match RegexBuilder::new(pattern)
            .case_insensitive(flags.case_insensitive)
            .build()
        {
            Ok(r) => r,
            Err(e) => {
                return CommandOutput::failed(
                    2,
                    error_line(
                        "grep",
                        format_args!("bad regex pattern: {e}"),
                        Hint::Check,
                        "escape regex metachars; run grep for usage",
                    )
                    .into_bytes(),
                );
            }
        };

        let paths = &positional[1..];
        if paths.is_empty() {
            return match input.stdin {
                Some(stdin) => annotate_dialect(search_stdin(&re, &flags, stdin), pattern),
                None => CommandOutput::usage(self.help()),
            };
        }

        let targets = match collect_targets(paths, flags.recursive).await {
            Ok(t) => t,
            Err(e) => return CommandOutput::failed(2, e.error_line().into_bytes()),
        };
        annotate_dialect(search_files(&re, &flags, &targets).await, pattern)
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
            for ch in rest.chars() {
                match ch {
                    'i' => flags.case_insensitive = true,
                    'v' => flags.invert = true,
                    'c' => flags.count_only = true,
                    'n' => flags.line_numbers = true,
                    'r' => flags.recursive = true,
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

fn search_stdin(re: &Regex, flags: &Flags, stdin: Vec<u8>) -> CommandOutput {
    let Ok(text) = std::str::from_utf8(&stdin) else {
        return CommandOutput::failed(
            2,
            error_line(
                "grep",
                "input is not valid UTF-8",
                Hint::Try,
                "grep on a text file or pipe from cat",
            )
            .into_bytes(),
        );
    };
    let mut out = Vec::new();
    let count = scan(re, flags, text, None, &mut out);
    let stdout = if flags.count_only {
        format!("{count}\n").into_bytes()
    } else {
        out
    };
    outcome(count, stdout)
}

async fn search_files(re: &Regex, flags: &Flags, targets: &[PathBuf]) -> CommandOutput {
    let label_lines = targets.len() > 1 || flags.recursive;
    let mut out = Vec::new();
    let mut total = 0usize;
    for path in targets {
        let Ok(bytes) = read_regular_file(path).await else {
            continue;
        };
        if sniff_binary(&bytes).is_some() {
            continue;
        }
        let Ok(text) = std::str::from_utf8(&bytes) else {
            continue;
        };
        let display = path.to_string_lossy();
        let label = label_lines.then_some(display.as_ref());
        let count = scan(re, flags, text, label, &mut out);
        if flags.count_only && label_lines {
            out.extend_from_slice(format!("{display}:{count}\n").as_bytes());
        }
        total += count;
    }
    let stdout = if flags.count_only && !label_lines {
        format!("{total}\n").into_bytes()
    } else {
        out
    };
    outcome(total, stdout)
}

fn scan(re: &Regex, flags: &Flags, text: &str, label: Option<&str>, out: &mut Vec<u8>) -> usize {
    let mut count = 0;
    for (index, line) in text.split_inclusive('\n').enumerate() {
        if !(re.is_match(line) ^ flags.invert) {
            continue;
        }
        count += 1;
        if flags.count_only {
            continue;
        }
        if let Some(label) = label {
            out.extend_from_slice(label.as_bytes());
            out.push(b':');
        }
        if flags.line_numbers {
            out.extend_from_slice(format!("{}:", index + 1).as_bytes());
        }
        out.extend_from_slice(line.as_bytes());
    }
    count
}

/// A zero-match result is the only moment a dialect mismatch is
/// visible, so that is where the explanation goes.
fn annotate_dialect(mut out: CommandOutput, pattern: &str) -> CommandOutput {
    if out.exit_code != 1 || !out.stderr.is_empty() {
        return out;
    }
    let Some(found) = BRE_ESCAPES.iter().find(|e| pattern.contains(**e)) else {
        return out;
    };
    out.stderr = error_line(
        "grep",
        format_args!(
            "no matches; `{found}` matches those characters literally here \
             (PATTERN is Rust/ERE regex, not BRE)"
        ),
        Hint::Use,
        "unescaped ERE metachars in a quoted pattern, e.g. grep \"a|b\" FILE",
    )
    .into_bytes();
    out
}

fn outcome(count: usize, stdout: Vec<u8>) -> CommandOutput {
    CommandOutput {
        stdout,
        stderr: Vec::new(),
        exit_code: if count > 0 { 0 } else { 1 },
        attachments: Vec::new(),
    }
}

async fn collect_targets(paths: &[String], recursive: bool) -> Result<Vec<PathBuf>, TargetError> {
    let mut targets = Vec::with_capacity(paths.len());
    for raw in paths {
        let path = Path::new(raw);
        let meta =
            tokio::fs::symlink_metadata(path)
                .await
                .map_err(|source| TargetError::Unreadable {
                    path: raw.clone(),
                    source,
                })?;
        if !meta.is_dir() {
            targets.push(path.to_path_buf());
            continue;
        }
        if !recursive {
            return Err(TargetError::DirectoryWithoutRecursion { path: raw.clone() });
        }
        descend(path, &mut targets).await;
    }
    Ok(targets)
}

/// Append every non-directory entry under `root` in sorted depth-first
/// order, skipping symlinks and anything unreadable.
async fn descend(root: &Path, targets: &mut Vec<PathBuf>) {
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(mut reader) = tokio::fs::read_dir(&dir).await else {
            continue;
        };
        let mut files = Vec::new();
        let mut dirs = Vec::new();
        while let Ok(Some(entry)) = reader.next_entry().await {
            let Ok(kind) = entry.file_type().await else {
                continue;
            };
            if kind.is_symlink() {
                continue;
            }
            if kind.is_dir() {
                dirs.push(entry.path());
            } else {
                files.push(entry.path());
            }
        }
        files.sort();
        dirs.sort();
        targets.extend(files);
        stack.extend(dirs.into_iter().rev());
    }
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

    async fn run_grep(args: &[&str], stdin: &[u8]) -> CommandOutput {
        GrepCommand
            .run(CommandInput {
                args: args.iter().map(|s| s.to_string()).collect(),
                stdin: Some(stdin.to_vec()),
            })
            .await
    }

    #[tokio::test]
    async fn filters_stdin_per_flags() {
        let cases: [(&[&str], &[u8], i32, &str); 9] = [
            (
                &["ERROR"],
                b"INFO ok\nERROR boom\nINFO also\n",
                0,
                "ERROR boom\n",
            ),
            (&["ZZZ"], b"nothing here\n", 1, ""),
            (
                &["Command|Tool"],
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
            (&["-i", "error"], b"ERROR boom\nnope\n", 0, "ERROR boom\n"),
            (&["-v", "ERROR"], b"INFO ok\nERROR boom\n", 0, "INFO ok\n"),
            (&["-c", "ERROR"], b"ERROR a\nINFO\nERROR b\n", 0, "2\n"),
            (&["-c", "ZZZ"], b"a\nb\n", 1, "0\n"),
            (&["-ivc", "error"], b"ERROR\ninfo\nError\nok\n", 0, "2\n"),
        ];
        for (args, stdin, exit_code, stdout) in cases {
            let out = run_grep(args, stdin).await;
            assert_eq!(out.exit_code, exit_code, "{args:?}");
            assert_eq!(String::from_utf8_lossy(&out.stdout), stdout, "{args:?}");
            assert!(out.stderr.is_empty(), "{args:?}: {out:?}");
        }
    }

    #[tokio::test]
    async fn bre_escape_that_matches_nothing_explains_the_dialect() {
        let out = run_grep(&[r"Command\|Tool"], b"a Command here\na Tool there\n").await;
        assert_eq!(out.exit_code, 1);
        assert_eq!(
            String::from_utf8_lossy(&out.stderr),
            "[error] grep: no matches; `\\|` matches those characters literally here \
             (PATTERN is Rust/ERE regex, not BRE). \
             Use: unescaped ERE metachars in a quoted pattern, e.g. grep \"a|b\" FILE\n"
        );
    }

    #[tokio::test]
    async fn bre_escape_that_does_match_stays_quiet() {
        let out = run_grep(&[r"a\|b"], b"literal a|b line\n").await;
        assert_eq!(out.exit_code, 0);
        assert!(out.stderr.is_empty(), "{:?}", out.stderr);
    }

    #[tokio::test]
    async fn missing_pattern_emits_usage() {
        let out = run_grep(&[], b"").await;
        assert_eq!(out.exit_code, 2);
        assert!(out.stdout.starts_with(b"usage: grep"), "{out:?}");
    }

    #[tokio::test]
    async fn pattern_without_files_or_stdin_emits_usage() {
        let out = GrepCommand
            .run(CommandInput {
                args: vec!["ERROR".into()],
                stdin: None,
            })
            .await;
        assert_eq!(out.exit_code, 2);
        assert!(out.stdout.starts_with(b"usage: grep"), "{out:?}");
    }

    #[tokio::test]
    async fn single_file_is_not_path_prefixed() {
        let dir = tree();
        let path = dir.path().join("top.txt");
        let out = run_grep(&["ERROR", &path.to_string_lossy()], b"").await;
        assert_eq!(out.exit_code, 0);
        assert_eq!(out.stdout, b"alpha ERROR\n");
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
    async fn r_flag_descends_in_order_and_skips_binary_files() {
        let dir = tree();
        let root = dir.path().to_string_lossy().into_owned();
        let out = run_grep(&["-rn", "ERROR", &root], b"").await;
        assert_eq!(out.exit_code, 0);
        assert_eq!(
            String::from_utf8_lossy(&out.stdout),
            format!("{root}/top.txt:1:alpha ERROR\n{root}/sub/deep.txt:2:delta ERROR\n")
        );
    }

    #[tokio::test]
    async fn rc_reports_a_count_per_file() {
        let dir = tree();
        let root = dir.path().to_string_lossy().into_owned();
        let out = run_grep(&["-rc", "ERROR", &root], b"").await;
        assert_eq!(out.exit_code, 0);
        assert_eq!(
            String::from_utf8_lossy(&out.stdout),
            format!("{root}/top.txt:1\n{root}/sub/deep.txt:1\n")
        );
    }

    #[tokio::test]
    async fn r_flag_with_no_matches_exits_1() {
        let dir = tree();
        let out = run_grep(&["-r", "nothing-here", &dir.path().to_string_lossy()], b"").await;
        assert_eq!(out.exit_code, 1);
        assert!(out.stdout.is_empty(), "{out:?}");
    }

    #[tokio::test]
    async fn directory_without_r_points_at_the_flag() {
        let dir = tree();
        let root = dir.path().to_string_lossy().into_owned();
        let out = run_grep(&["ERROR", &root], b"").await;
        assert_eq!(out.exit_code, 2);
        assert_eq!(
            String::from_utf8_lossy(&out.stderr),
            format!("[error] grep: {root} is a directory. Use: grep -r PATTERN {root}\n")
        );
    }

    #[tokio::test]
    async fn missing_file_reports_navigation_error() {
        let out = run_grep(&["ERROR", "/definitely/not/here.txt"], b"").await;
        assert_eq!(out.exit_code, 2);
        assert_eq!(
            String::from_utf8_lossy(&out.stderr),
            "[error] grep: file not found: /definitely/not/here.txt. \
             Use: ls /definitely/not to see what is there\n"
        );
    }

    #[tokio::test]
    async fn unknown_flag_errors() {
        let out = run_grep(&["-x", "foo"], b"").await;
        assert_eq!(out.exit_code, 2);
        assert_eq!(
            String::from_utf8_lossy(&out.stderr),
            "[error] grep: unknown flag '-x'. Use: grep (no args) for supported flags\n"
        );
    }
}
