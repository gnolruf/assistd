use std::path::{Path, PathBuf};

use anyhow::Result;
use async_trait::async_trait;
use regex::{Regex, RegexBuilder};

use crate::command::{Command, CommandInput, CommandOutput, error_line, io_error_nav};
use crate::commands::cat::sniff_binary;

/// `grep [-icnrv] PATTERN [FILE|DIR]...`: print lines from the named
/// files (or stdin) that match `PATTERN`.
///
/// Flags:
/// - `-i` case-insensitive
/// - `-v` invert match
/// - `-c` print the count instead of the matching lines
/// - `-n` prefix each line with its 1-based line number
/// - `-r` descend into directory arguments
///
/// Flags can be combined (`-rn`). Exit 0 if any line matched (or the
/// count is non-zero under `-c`), 1 otherwise, 2 on usage/input errors.
pub struct GrepCommand;

#[derive(Default)]
struct Flags {
    case_insensitive: bool,
    invert: bool,
    count_only: bool,
    line_numbers: bool,
    recursive: bool,
}

fn parse_flags(argv: &[String]) -> Result<(Flags, &[String]), String> {
    let mut flags = Flags::default();
    let mut i = 0;
    while i < argv.len() {
        let a = &argv[i];
        if a == "--" {
            i += 1;
            break;
        }
        if let Some(rest) = a.strip_prefix('-') {
            if rest.is_empty() {
                break; // bare `-` = stdin sentinel; treat as positional
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
            i += 1;
        } else {
            break;
        }
    }
    Ok((flags, &argv[i..]))
}

#[async_trait]
impl Command for GrepCommand {
    fn name(&self) -> &str {
        "grep"
    }

    fn summary(&self) -> &'static str {
        "filter lines matching a pattern (supports -i, -v, -c, -n, -r)"
    }

    fn help(&self) -> String {
        "usage: grep [-icnrv] PATTERN [FILE|DIR]...\n\
         \n\
         Print lines matching the regex PATTERN, read from the named \
         paths or from stdin when none are given.\n\
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

    async fn run(&self, input: CommandInput) -> Result<CommandOutput> {
        if input.args.is_empty() {
            return Ok(CommandOutput {
                stdout: self.help().into_bytes(),
                stderr: Vec::new(),
                exit_code: 2,
                attachments: Vec::new(),
            });
        }
        let (flags, positional) = match parse_flags(&input.args) {
            Ok(v) => v,
            Err(msg) => {
                return Ok(CommandOutput::failed(
                    2,
                    error_line("grep", msg, "Use", "grep (no args) for supported flags")
                        .into_bytes(),
                ));
            }
        };
        if positional.is_empty() {
            return Ok(CommandOutput {
                stdout: self.help().into_bytes(),
                stderr: Vec::new(),
                exit_code: 2,
                attachments: Vec::new(),
            });
        }

        let pattern = &positional[0];
        let re = match RegexBuilder::new(pattern)
            .case_insensitive(flags.case_insensitive)
            .build()
        {
            Ok(r) => r,
            Err(e) => {
                return Ok(CommandOutput::failed(
                    2,
                    error_line(
                        "grep",
                        format_args!("bad regex pattern: {e}"),
                        "Check",
                        "escape regex metachars; run grep for usage",
                    )
                    .into_bytes(),
                ));
            }
        };

        let paths = &positional[1..];
        if paths.is_empty() {
            return Ok(search_stdin(&re, &flags, input.stdin));
        }

        let targets = match collect_targets(paths, flags.recursive).await {
            Ok(t) => t,
            Err(e) => return Ok(CommandOutput::failed(2, e.error_line().into_bytes())),
        };
        Ok(search_files(&re, &flags, &targets).await)
    }
}

fn search_stdin(re: &Regex, flags: &Flags, stdin: Vec<u8>) -> CommandOutput {
    let Ok(text) = std::str::from_utf8(&stdin) else {
        return CommandOutput::failed(
            2,
            error_line(
                "grep",
                "input is not valid UTF-8",
                "Try",
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
        let Ok(bytes) = tokio::fs::read(path).await else {
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
    for (i, line) in text.split_inclusive('\n').enumerate() {
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
            out.extend_from_slice(format!("{}:", i + 1).as_bytes());
        }
        out.extend_from_slice(line.as_bytes());
    }
    count
}

fn outcome(count: usize, stdout: Vec<u8>) -> CommandOutput {
    CommandOutput {
        stdout,
        stderr: Vec::new(),
        exit_code: if count > 0 { 0 } else { 1 },
        attachments: Vec::new(),
    }
}

/// Why a path named on the command line could not be searched.
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
                "Use",
                format_args!("grep -r PATTERN {path}"),
            ),
        }
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
        // Reversed so the pop order matches the sorted order.
        stack.extend(dirs.into_iter().rev());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::{TempDir, tempdir};

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
                stdin: stdin.to_vec(),
            })
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn n_flag_numbers_matching_lines() {
        let out = run_grep(&["-n", "ERROR"], b"ok\nERROR one\nok\nERROR two\n").await;
        assert_eq!(out.exit_code, 0);
        assert_eq!(out.stdout, b"2:ERROR one\n4:ERROR two\n");
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
    async fn r_flag_descends_and_prefixes_paths() {
        let dir = tree();
        let root = dir.path().to_string_lossy().into_owned();
        let out = run_grep(&["-rn", "ERROR", &root], b"").await;
        assert_eq!(out.exit_code, 0);
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(
            stdout.contains(&format!("{root}/top.txt:1:alpha ERROR\n")),
            "{stdout}"
        );
        assert!(
            stdout.contains(&format!("{root}/sub/deep.txt:2:delta ERROR\n")),
            "{stdout}"
        );
    }

    #[tokio::test]
    async fn r_flag_skips_binary_files() {
        let dir = tree();
        let out = run_grep(&["-r", "ERROR", &dir.path().to_string_lossy()], b"").await;
        assert!(
            !String::from_utf8_lossy(&out.stdout).contains("notes.bin"),
            "{out:?}"
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
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(stdout.contains(&format!("{a}:alpha ERROR\n")), "{stdout}");
        assert!(stdout.contains(&format!("{b}:delta ERROR\n")), "{stdout}");
    }

    #[tokio::test]
    async fn rc_reports_a_count_per_file() {
        let dir = tree();
        let out = run_grep(&["-rc", "ERROR", &dir.path().to_string_lossy()], b"").await;
        assert_eq!(out.exit_code, 0);
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(stdout.contains("top.txt:1\n"), "{stdout}");
        assert!(stdout.contains("deep.txt:1\n"), "{stdout}");
    }

    #[tokio::test]
    async fn directory_without_r_points_at_the_flag() {
        let dir = tree();
        let root = dir.path().to_string_lossy().into_owned();
        let out = run_grep(&["ERROR", &root], b"").await;
        assert_eq!(out.exit_code, 2);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            stderr.contains(&format!("[error] grep: {root} is a directory")),
            "{stderr}"
        );
        assert!(
            stderr.contains(&format!("Use: grep -r PATTERN {root}")),
            "{stderr}"
        );
    }

    #[tokio::test]
    async fn missing_file_reports_navigation_error() {
        let out = run_grep(&["ERROR", "/definitely/not/here.txt"], b"").await;
        assert_eq!(out.exit_code, 2);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            stderr.contains("[error] grep: file not found: /definitely/not/here.txt"),
            "{stderr}"
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
    async fn matches_from_stdin() {
        let out = run_grep(&["ERROR"], b"INFO ok\nERROR boom\nINFO also\n").await;
        assert_eq!(out.exit_code, 0);
        assert_eq!(out.stdout, b"ERROR boom\n");
    }

    #[tokio::test]
    async fn no_match_exits_1() {
        let out = run_grep(&["ZZZ"], b"nothing here\n").await;
        assert_eq!(out.exit_code, 1);
        assert!(out.stdout.is_empty());
    }

    #[tokio::test]
    async fn missing_pattern_errors() {
        let out = run_grep(&[], b"").await;
        assert_eq!(out.exit_code, 2);
    }

    #[tokio::test]
    async fn i_flag_matches_case_insensitively() {
        let out = run_grep(&["-i", "error"], b"ERROR boom\nnope\n").await;
        assert_eq!(out.exit_code, 0);
        assert_eq!(out.stdout, b"ERROR boom\n");
    }

    #[tokio::test]
    async fn v_flag_inverts_match() {
        let out = run_grep(&["-v", "ERROR"], b"INFO ok\nERROR boom\n").await;
        assert_eq!(out.exit_code, 0);
        assert_eq!(out.stdout, b"INFO ok\n");
    }

    #[tokio::test]
    async fn c_flag_returns_count() {
        let out = run_grep(&["-c", "ERROR"], b"ERROR a\nINFO\nERROR b\n").await;
        assert_eq!(out.exit_code, 0);
        assert_eq!(out.stdout, b"2\n");
    }

    #[tokio::test]
    async fn ic_combined_case_insensitive_count() {
        let out = run_grep(&["-ic", "error"], b"ERROR a\ninfo\nError b\nnothing\n").await;
        assert_eq!(out.exit_code, 0);
        assert_eq!(out.stdout, b"2\n");
    }

    #[tokio::test]
    async fn ivc_all_three_flags_together() {
        let out = run_grep(&["-ivc", "error"], b"ERROR\ninfo\nError\nok\n").await;
        assert_eq!(out.exit_code, 0);
        assert_eq!(out.stdout, b"2\n");
    }

    #[tokio::test]
    async fn c_with_no_matches_exits_1_but_prints_zero() {
        let out = run_grep(&["-c", "ZZZ"], b"a\nb\n").await;
        assert_eq!(out.exit_code, 1);
        assert_eq!(out.stdout, b"0\n");
    }

    #[tokio::test]
    async fn unknown_flag_errors() {
        let out = run_grep(&["-x", "foo"], b"").await;
        assert_eq!(out.exit_code, 2);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            stderr.contains("[error] grep: unknown flag '-x'"),
            "{stderr}"
        );
        assert!(
            stderr.contains("Use: grep (no args) for supported flags"),
            "{stderr}"
        );
    }
}
