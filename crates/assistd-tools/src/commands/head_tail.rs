//! `head` and `tail`: take lines from one end of the named files or stdin.

use std::io::{self, SeekFrom};
use std::path::Path;

use assistd_utils::text::human_size;
use async_trait::async_trait;
use tokio::fs::File;
use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncReadExt, AsyncSeekExt, BufReader};

use crate::chain::OUTPUT_MAX;
use crate::command::{Command, CommandInput, CommandOutput, Hint, error_line, io_error_nav};
use crate::commands::cat::{SNIFF_LEN, sniff_binary};
use crate::commands::{FILE_READ_MAX, binary_refusal, open_regular};
use crate::policy::FileScope;

/// Lines emitted when no count flag is given, matching coreutils.
const DEFAULT_LINES: usize = 10;

/// Bytes `tail` reads per step while seeking back from the end of a file.
const TAIL_BLOCK: u64 = 64 * 1024;

/// `head [-n N] [FILE]...`: emit the first `N` lines of the named
/// files, or of stdin when none are given.
#[derive(Debug)]
pub struct HeadCommand;

#[async_trait]
impl Command for HeadCommand {
    fn name(&self) -> &'static str {
        "head"
    }

    fn summary(&self) -> &'static str {
        "emit the first N lines of FILE or stdin (default 10; -n N to change)"
    }

    fn help(&self) -> String {
        format!(
            "usage: head [-n N] [FILE]...\n\
             \n\
             Emit the first N lines of the named files, or of stdin when \
             none are given, {DEFAULT_LINES} by default. `-n N`, `-nN` \
             and `-N` are all accepted.\n\
             \n\
             Several files concatenate, exactly as `cat FILE... | head` \
             would; there are no `==> FILE <==` banners. Binary files are \
             refused — use `cat -b FILE` for those.\n"
        )
    }

    async fn run(&self, input: CommandInput) -> CommandOutput {
        let (count, files) = match parse_flags("head", &input.args) {
            Ok(v) => v,
            Err(e) => return count_error("head", e),
        };
        if !files.is_empty() {
            return finish(head_of_files(&input.scope, &files, count.amount()).await);
        }
        match input.stdin {
            Some(data) => CommandOutput::ok(first_lines(&data, count.amount())),
            None => CommandOutput::usage(self.help()),
        }
    }
}

/// `tail [-n N] [FILE]...`: emit the last `N` lines of the named
/// files, or of stdin when none are given.
#[derive(Debug)]
pub struct TailCommand;

#[async_trait]
impl Command for TailCommand {
    fn name(&self) -> &'static str {
        "tail"
    }

    fn summary(&self) -> &'static str {
        "emit the last N lines of FILE or stdin (default 10; -n N, or -n +N from line N)"
    }

    fn help(&self) -> String {
        format!(
            "usage: tail [-n N] [FILE]...\n\
             \n\
             Emit the last N lines of the named files, or of stdin when \
             none are given, {DEFAULT_LINES} by default. `-n N`, `-nN` \
             and `-N` are all accepted. `-n +N` instead emits everything \
             from line N onward.\n\
             \n\
             Several files concatenate, exactly as `cat FILE... | tail` \
             would; there are no `==> FILE <==` banners. Binary files are \
             refused — use `cat -b FILE` for those.\n"
        )
    }

    async fn run(&self, input: CommandInput) -> CommandOutput {
        let (count, files) = match parse_flags("tail", &input.args) {
            Ok(v) => v,
            Err(e) => return count_error("tail", e),
        };
        if !files.is_empty() {
            return finish(match count {
                LineCount::Last(count) => tail_of_files(&input.scope, &files, count).await,
                LineCount::FromLine(line) => files_from_line(&input.scope, &files, line).await,
            });
        }
        let Some(data) = input.stdin else {
            return CommandOutput::usage(self.help());
        };
        CommandOutput::ok(match count {
            LineCount::Last(count) => last_lines(&data, count),
            LineCount::FromLine(line) => lines_from(&data, line),
        })
    }
}

/// A parsed `-n` value: a plain count, or a `+`-prefixed 1-based line to
/// start from (`tail` only; `head` reads the number as a count, as coreutils
/// does).
#[derive(Debug, Clone, Copy)]
enum LineCount {
    Last(usize),
    FromLine(usize),
}

impl LineCount {
    fn amount(self) -> usize {
        match self {
            Self::Last(n) | Self::FromLine(n) => n,
        }
    }
}

/// Where a backward search for the start of the last lines stands.
#[derive(Debug, PartialEq, Eq)]
enum ScanProgress {
    Searching,
    Found,
    Overflow,
}

/// Blocks kept while reading a stream backward until the start of its last
/// `lines` lines is found.
#[derive(Debug)]
struct BackwardScan {
    lines: usize,
    newlines_owed: Option<usize>,
    blocks: Vec<Vec<u8>>,
    kept: usize,
}

impl BackwardScan {
    fn new(lines: usize) -> Self {
        Self {
            lines,
            newlines_owed: None,
            blocks: Vec::new(),
            kept: 0,
        }
    }

    /// Take the block just before every block taken so far. `lines` must be
    /// non-zero; a trailing newline at the very end of the stream ends the
    /// last line rather than starting a new one.
    fn take_block(&mut self, mut block: Vec<u8>) -> ScanProgress {
        if block.is_empty() {
            return ScanProgress::Searching;
        }
        let owed = *self
            .newlines_owed
            .get_or_insert(self.lines + usize::from(block.ends_with(b"\n")));
        let line_start = nth_newline_from_end(&block, owed - 1).map(|newline| newline + 1);
        let progress = match line_start {
            Some(start) => {
                block.drain(..start);
                ScanProgress::Found
            }
            None => {
                self.newlines_owed = Some(owed - count_newlines(&block));
                ScanProgress::Searching
            }
        };
        self.kept += block.len();
        self.blocks.push(block);
        if self.kept > OUTPUT_MAX {
            ScanProgress::Overflow
        } else {
            progress
        }
    }

    fn into_bytes(mut self) -> Vec<u8> {
        self.blocks.reverse();
        self.blocks.concat()
    }
}

/// Why an argument list could not be read as a line count.
struct CountError {
    what: String,
    recovery: String,
}

/// Split argv into the line count and the files to read. A bare `-` is a
/// file (stdin, as in coreutils), not a flag.
fn parse_flags(cmd: &str, argv: &[String]) -> Result<(LineCount, Vec<String>), CountError> {
    let mut count = LineCount::Last(DEFAULT_LINES);
    let mut files = Vec::new();
    let mut pos = 0;
    while pos < argv.len() {
        let arg = &argv[pos];
        let Some(rest) = arg.strip_prefix('-').filter(|r| !r.is_empty()) else {
            files.push(arg.clone());
            pos += 1;
            continue;
        };
        let digits = match rest.strip_prefix('n') {
            Some("") => {
                pos += 1;
                argv.get(pos)
                    .map(String::as_str)
                    .ok_or_else(|| CountError {
                        what: "-n needs a line count".to_string(),
                        recovery: format!("{cmd} -n 20"),
                    })?
            }
            Some(glued) => glued,
            None => rest,
        };
        count = parse_count(digits).ok_or_else(|| CountError {
            what: format!("not a line count: '-{digits}'"),
            recovery: format!("{cmd} -n 20"),
        })?;
        pos += 1;
    }
    Ok((count, files))
}

fn parse_count(digits: &str) -> Option<LineCount> {
    let (from_line, digits) = match digits.strip_prefix('+') {
        Some(rest) => (true, rest),
        None => (false, digits),
    };
    if !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let n = digits.parse().ok()?;
    Some(if from_line {
        LineCount::FromLine(n)
    } else {
        LineCount::Last(n)
    })
}

fn count_error(cmd: &str, e: CountError) -> CommandOutput {
    CommandOutput::usage_error(cmd, e.what, e.recovery)
}

fn finish(result: Result<Vec<u8>, CommandOutput>) -> CommandOutput {
    result.map_or_else(|failure| failure, CommandOutput::ok)
}

/// The first `count` lines of the named files read as one stream, opening
/// each only once the files before it ran out of lines.
async fn head_of_files(
    scope: &FileScope,
    files: &[String],
    count: usize,
) -> Result<Vec<u8>, CommandOutput> {
    let mut out = Vec::new();
    let mut lines_owed = count;
    for path in files {
        if lines_owed == 0 {
            break;
        }
        let (file, _) = open_text("head", scope, path).await?;
        let mut reader = BufReader::new(file.take(output_budget(&out)));
        lines_owed = pass_lines(&mut reader, lines_owed, Some(&mut out))
            .await
            .map_err(|e| read_failure("head", path, &e))?;
        if out.len() > OUTPUT_MAX {
            return Err(output_too_large("head"));
        }
    }
    Ok(out)
}

/// The named files read as one stream from 1-based `line` onward.
async fn files_from_line(
    scope: &FileScope,
    files: &[String],
    line: usize,
) -> Result<Vec<u8>, CommandOutput> {
    let mut out = Vec::new();
    let mut lines_to_skip = line.saturating_sub(1);
    for path in files {
        let (file, _) = open_text("tail", scope, path).await?;
        let mut reader = BufReader::new(file);
        lines_to_skip = pass_lines(&mut reader, lines_to_skip, None)
            .await
            .map_err(|e| read_failure("tail", path, &e))?;
        reader
            .take(output_budget(&out))
            .read_to_end(&mut out)
            .await
            .map_err(|e| read_failure("tail", path, &e))?;
        if out.len() > OUTPUT_MAX {
            return Err(output_too_large("tail"));
        }
    }
    Ok(out)
}

/// The last `count` lines of the named files read as one stream. Every file
/// is opened and checked first; then they are read backward from the end.
async fn tail_of_files(
    scope: &FileScope,
    files: &[String],
    count: usize,
) -> Result<Vec<u8>, CommandOutput> {
    let mut opened = Vec::with_capacity(files.len());
    for path in files {
        opened.push((path, open_text("tail", scope, path).await?));
    }
    if count == 0 {
        return Ok(Vec::new());
    }
    let mut scan = BackwardScan::new(count);
    for (path, (mut file, size)) in opened.into_iter().rev() {
        let progress = scan_backward(&mut file, size, &mut scan)
            .await
            .map_err(|e| read_failure("tail", path, &e))?;
        match progress {
            ScanProgress::Searching => {}
            ScanProgress::Found => break,
            ScanProgress::Overflow => return Err(output_too_large("tail")),
        }
    }
    Ok(scan.into_bytes())
}

/// Feed `file` to `scan` block by block from its end. A file reporting size
/// zero may be a pseudo-file whose content only shows on read, so it is read
/// forward whole instead.
async fn scan_backward(
    file: &mut File,
    size: u64,
    scan: &mut BackwardScan,
) -> io::Result<ScanProgress> {
    if size == 0 {
        let mut whole = Vec::new();
        file.take(FILE_READ_MAX + 1).read_to_end(&mut whole).await?;
        return Ok(scan.take_block(whole));
    }
    let mut end = size;
    while end > 0 {
        let start = end.saturating_sub(TAIL_BLOCK);
        file.seek(SeekFrom::Start(start)).await?;
        let mut block = vec![0; usize::try_from(end - start).map_err(io::Error::other)?];
        file.read_exact(&mut block).await?;
        match scan.take_block(block) {
            ScanProgress::Searching => end = start,
            done => return Ok(done),
        }
    }
    Ok(ScanProgress::Searching)
}

/// Open `path` and refuse it when its head sniffs as binary; the file is
/// returned rewound, with its size.
async fn open_text(cmd: &str, scope: &FileScope, path: &str) -> Result<(File, u64), CommandOutput> {
    let failure = |e: io::Error| read_failure(cmd, path, &e);
    let (mut file, size) = open_regular(scope, Path::new(path))
        .await
        .map_err(failure)?;
    let mut head = Vec::new();
    (&mut file)
        .take(SNIFF_LEN as u64)
        .read_to_end(&mut head)
        .await
        .map_err(failure)?;
    if let Some(mime) = sniff_binary(&head) {
        return Err(binary_refusal(cmd, &mime, size, path));
    }
    file.seek(SeekFrom::Start(0)).await.map_err(failure)?;
    Ok((file, size))
}

/// Move `reader` past `lines` newlines, appending the bytes passed to `out`
/// when given; returns the newlines still owed when the reader ran dry.
async fn pass_lines(
    reader: &mut (impl AsyncBufRead + Unpin),
    mut lines: usize,
    mut out: Option<&mut Vec<u8>>,
) -> io::Result<usize> {
    while lines > 0 {
        let buf = reader.fill_buf().await?;
        if buf.is_empty() {
            break;
        }
        let (taken, passed) = match nth_newline(buf, lines - 1) {
            Some(newline) => (newline + 1, lines),
            None => (buf.len(), count_newlines(buf)),
        };
        if let Some(out) = out.as_deref_mut() {
            out.extend_from_slice(&buf[..taken]);
        }
        reader.consume(taken);
        lines -= passed;
    }
    Ok(lines)
}

fn nth_newline(bytes: &[u8], n: usize) -> Option<usize> {
    newline_positions(bytes).nth(n)
}

fn nth_newline_from_end(bytes: &[u8], n: usize) -> Option<usize> {
    newline_positions(bytes).rev().nth(n)
}

fn newline_positions(bytes: &[u8]) -> impl DoubleEndedIterator<Item = usize> + '_ {
    bytes
        .iter()
        .enumerate()
        .filter_map(|(at, byte)| (*byte == b'\n').then_some(at))
}

fn count_newlines(bytes: &[u8]) -> usize {
    bytes.iter().filter(|byte| **byte == b'\n').count()
}

/// Bytes still readable before `out` passes [`OUTPUT_MAX`] by one.
fn output_budget(out: &[u8]) -> u64 {
    (OUTPUT_MAX + 1).saturating_sub(out.len()) as u64
}

fn read_failure(cmd: &str, path: &str, e: &io::Error) -> CommandOutput {
    CommandOutput::failed(1, io_error_nav(cmd, path, e).into_bytes())
}

fn output_too_large(cmd: &str) -> CommandOutput {
    CommandOutput::failed(
        1,
        error_line(
            cmd,
            format_args!("output exceeds the {} limit", human_size(OUTPUT_MAX as u64)),
            Hint::Try,
            format_args!("a smaller -n, or grep FILE for the lines you need"),
        )
        .into_bytes(),
    )
}

fn first_lines(text: &[u8], count: usize) -> Vec<u8> {
    let end: usize = text
        .split_inclusive(|b| *b == b'\n')
        .take(count)
        .map(<[u8]>::len)
        .sum();
    text[..end].to_vec()
}

fn last_lines(text: &[u8], count: usize) -> Vec<u8> {
    let lines: Vec<&[u8]> = text.split_inclusive(|b| *b == b'\n').collect();
    lines[lines.len().saturating_sub(count)..].concat()
}

/// Lines from 1-based `line` onward; `0` and `1` both mean the whole text.
fn lines_from(text: &[u8], line: usize) -> Vec<u8> {
    text.split_inclusive(|b| *b == b'\n')
        .skip(line.saturating_sub(1))
        .flatten()
        .copied()
        .collect()
}

#[cfg(test)]
mod tests {

    use super::*;

    async fn run(cmd: &dyn Command, args: &[&str], stdin: &[u8]) -> CommandOutput {
        cmd.run(CommandInput {
            args: args.iter().map(ToString::to_string).collect(),
            stdin: Some(stdin.to_vec()),
            scope: FileScope::Host,
        })
        .await
    }

    const FIVE: &[u8] = b"one\ntwo\nthree\nfour\nfive\n";

    #[tokio::test]
    async fn head_accepts_every_count_spelling() {
        for args in [vec!["-n", "2"], vec!["-n2"], vec!["-2"]] {
            let out = run(&HeadCommand, &args, FIVE).await;
            assert_eq!(out.stdout, b"one\ntwo\n", "args={args:?}");
        }
    }

    #[tokio::test]
    async fn takes_lines_from_the_right_end() {
        const FIVE_TEXT: &str = "one\ntwo\nthree\nfour\nfive\n";
        let cases: [(&dyn Command, &[&str], &str, &str); 5] = [
            (&TailCommand, &["-2"], FIVE_TEXT, "four\nfive\n"),
            (&TailCommand, &["-n", "+4"], FIVE_TEXT, "four\nfive\n"),
            (
                &TailCommand,
                &["-n+2"],
                FIVE_TEXT,
                "two\nthree\nfour\nfive\n",
            ),
            (&TailCommand, &["-99"], FIVE_TEXT, FIVE_TEXT),
            (&TailCommand, &["-1"], "a\nb", "b"),
        ];
        for (cmd, args, stdin, expected) in cases {
            let label = format!("{} {args:?} {stdin:?}", cmd.name());
            let out = run(cmd, args, stdin.as_bytes()).await;
            assert_eq!(out.exit_code, 0, "{label}");
            assert_eq!(String::from_utf8_lossy(&out.stdout), expected, "{label}");
        }
    }

    #[tokio::test]
    async fn reads_both_ends_of_a_file_over_the_read_limit() {
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("big.log");
        let mut body = b"first\n".repeat(OUTPUT_MAX / 6 + 1);
        body.extend_from_slice(b"penultimate\nlast\n");
        std::fs::write(&log, &body).unwrap();
        let log = log.to_str().unwrap();
        let tail = run(&TailCommand, &["-n", "2", log], b"").await;
        assert_eq!(tail.stdout, b"penultimate\nlast\n", "{tail:?}");
        let head = run(&HeadCommand, &["-n", "1", log], b"").await;
        assert_eq!(head.stdout, b"first\n", "{head:?}");
    }

    #[tokio::test]
    async fn binary_file_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let bin = dir.path().join("blob.bin");
        std::fs::write(&bin, b"\x00\x01\x02binary\x00").unwrap();
        let out = run(&TailCommand, &[bin.to_str().unwrap()], b"").await;
        assert_eq!(out.exit_code, 1);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(stderr.contains("[error] tail: binary"), "{stderr}");
        assert!(stderr.contains("Use: cat -b "), "{stderr}");
    }
}
