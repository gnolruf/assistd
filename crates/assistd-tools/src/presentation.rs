//! The per-result output budget every tool shares, and rendering a chain's [`CommandOutput`]
//! within it: binary refused, long streams cut and spilled to a file, an `[exit:N | Mms]` footer.

use std::env;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use assistd_config::ToolsOutputConfig;
use assistd_ipc::Attachment;
use assistd_utils::text::human_size;

use self::spill::{prune_spills, write_overflow_file};
use crate::binary::binary_label;
use crate::command::CommandOutput;

mod spill;

pub use spill::is_spill_file_name;

/// Stem of the spill files `run` writes.
const SPILL_STEM: &str = "cmd";

const STDERR_PREFIX: &str = "[stderr] ";

/// Limits and destinations for output rendering.
#[derive(Debug, Clone)]
pub struct PresentSpec {
    /// Max lines of a stream surfaced before truncation.
    pub max_lines: usize,
    /// Max bytes of the truncated head, bounding a single huge line too.
    pub max_bytes: usize,
    /// Directory where full overflow output is spilled as `cmd-<n>.txt`.
    pub overflow_dir: PathBuf,
}

impl PresentSpec {
    /// The caps from `output`, spilling into `overflow_dir`.
    pub fn from_config(output: &ToolsOutputConfig, overflow_dir: PathBuf) -> Self {
        Self {
            max_lines: output.max_lines.get() as usize,
            max_bytes: output.max_bytes(),
            overflow_dir,
        }
    }
}

impl Default for PresentSpec {
    /// The default caps, spilling into `assistd/output` under the system
    /// temporary directory.
    fn default() -> Self {
        let overflow_dir = env::temp_dir().join("assistd").join("output");
        Self::from_config(&ToolsOutputConfig::default(), overflow_dir)
    }
}

/// The output budget of one tool result. Text past it is cut to a head and
/// a truncation notice that together fit the budget, with the whole text
/// spilled as `<stem>-<n>.txt`, numbered across every stem.
#[derive(Debug)]
pub struct OutputCap {
    spec: PresentSpec,
    counter: AtomicU64,
}

/// A text body after [`OutputCap::cap`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TruncatedText {
    /// The visible head, followed by the truncation notice when cut.
    pub text: String,
    pub truncated: bool,
    /// Set when the body overflowed and the spill file was written.
    pub overflow_file: Option<PathBuf>,
}

impl OutputCap {
    pub fn new(spec: PresentSpec) -> Self {
        Self {
            spec,
            counter: AtomicU64::new(0),
        }
    }

    pub fn spec(&self) -> &PresentSpec {
        &self.spec
    }

    /// Return `text` unchanged when it fits, else its head plus the
    /// truncation notice, with the whole body spilled under `stem`.
    pub fn cap(&self, stem: &str, text: String) -> TruncatedText {
        let cut = self.cut_stream(text, "output", self.budget(), stem);
        TruncatedText {
            text: cut.visible(),
            truncated: cut.truncated(),
            overflow_file: cut.overflow_file,
        }
    }

    fn budget(&self) -> Budget {
        Budget {
            bytes: self.spec.max_bytes,
            lines: self.spec.max_lines,
        }
    }

    /// Cut `text` so its banner, head, line terminator, and notice together
    /// fit `budget`, spilling it in full when it overflows; `label` names the
    /// stream in the banner and notice.
    fn cut_stream(&self, text: String, label: &str, budget: Budget, stem: &str) -> StreamCut {
        let full = Budget::of(&text);
        if full.fits_in(budget) {
            return StreamCut {
                head: text,
                truncation: None,
                overflow_file: None,
            };
        }
        let overflow_file = self.spill(text.as_bytes(), stem);
        let spill = overflow_file.as_deref();
        let widest = Truncation::new(label, full.lines, full, spill);
        let room = budget
            .saturating_sub(widest.cost())
            .saturating_sub(Budget::NEWLINE);
        let head = truncate_lines_bytes(&text, room.lines, room.bytes);
        let truncation = Truncation::new(label, count_lines(&head), full, spill);
        StreamCut {
            head,
            truncation: Some(truncation),
            overflow_file,
        }
    }

    /// Write `raw` to the next numbered `<stem>-<n>.txt` spill file,
    /// logging and returning `None` when that fails.
    fn spill(&self, raw: &[u8], stem: &str) -> Option<PathBuf> {
        let n = self.counter.fetch_add(1, Ordering::Relaxed) + 1;
        let file_name = format!("{stem}-{n}.txt");
        match write_overflow_file(raw, &self.spec.overflow_dir.join(&file_name)) {
            Ok(path) => {
                prune_spills(&self.spec.overflow_dir, &path, raw.len() as u64);
                Some(path)
            }
            Err(e) => {
                tracing::warn!(
                    "failed to write overflow file {file_name} to {}: {e}",
                    self.spec.overflow_dir.display()
                );
                None
            }
        }
    }
}

/// Bytes and lines of text, or the room left for them.
#[derive(Debug, Clone, Copy)]
struct Budget {
    bytes: usize,
    lines: usize,
}

impl Budget {
    const NEWLINE: Self = Self { bytes: 1, lines: 0 };

    fn of(text: &str) -> Self {
        Self {
            bytes: text.len(),
            lines: count_lines(text),
        }
    }

    fn fits_in(self, budget: Self) -> bool {
        self.bytes <= budget.bytes && self.lines <= budget.lines
    }

    fn saturating_sub(self, used: Self) -> Self {
        Self {
            bytes: self.bytes.saturating_sub(used.bytes),
            lines: self.lines.saturating_sub(used.lines),
        }
    }

    fn max(self, other: Self) -> Self {
        Self {
            bytes: self.bytes.max(other.bytes),
            lines: self.lines.max(other.lines),
        }
    }

    fn half(self) -> Self {
        Self {
            bytes: self.bytes / 2,
            lines: self.lines / 2,
        }
    }
}

/// One stream cut to a [`PresentSpec`]: its visible head and, when cut,
/// the banner and notice that say so.
#[derive(Debug)]
struct StreamCut {
    head: String,
    truncation: Option<Truncation>,
    overflow_file: Option<PathBuf>,
}

impl StreamCut {
    fn truncated(&self) -> bool {
        self.truncation.is_some()
    }

    /// The head as the model sees it: between the banner and the notice
    /// when the stream was cut.
    fn visible(&self) -> String {
        let Some(truncation) = &self.truncation else {
            return self.head.clone();
        };
        let mut text = truncation.banner.clone();
        text.push_str(&self.head);
        terminate_line(&mut text);
        text.push_str(&truncation.notice);
        text
    }
}

/// What a cut stream says about itself: a banner line above the head and a
/// notice below it naming the lines left out and how to read them.
#[derive(Debug)]
struct Truncation {
    banner: String,
    notice: String,
}

impl Truncation {
    /// Texts for a stream of `full` size cut to its first `shown` lines,
    /// spilled to `spill` unless writing it failed. Never longer for a
    /// smaller `shown`, so sizing with `shown` at its maximum is safe.
    fn new(label: &str, shown: usize, full: Budget, spill: Option<&Path>) -> Self {
        let tag = format!("[{} TRUNCATED]", label.to_uppercase());
        let total = full.lines;
        let size = human_size(full.bytes as u64);
        let banner = match shown {
            0 => format!("{tag} None of its {total} lines ({size}) fit here.\n"),
            _ => format!(
                "{tag} Showing only lines 1-{shown} of {total} ({size}); the rest is NOT shown.\n"
            ),
        };
        let mut notice = format!(
            "{tag} Lines {}-{total} were left out. Do not treat the lines above as the whole {label}.\n",
            shown + 1
        );
        match spill {
            Some(path) => {
                let path = path.display();
                notice.push_str(&format!("Full {label}: {path}\n"));
                if shown > 0 {
                    notice.push_str(&format!(
                        "Next page: cat {path} | head -n {} | tail -n {shown}\n",
                        2 * shown
                    ));
                }
                notice.push_str(&format!("Search it: cat {path} | grep PATTERN\n"));
            }
            None => notice.push_str(&format!("The full {label} could not be saved.\n")),
        }
        Self { banner, notice }
    }

    fn cost(&self) -> Budget {
        let banner = Budget::of(&self.banner);
        let notice = Budget::of(&self.notice);
        Budget {
            bytes: banner.bytes + notice.bytes,
            lines: banner.lines + notice.lines,
        }
    }
}

/// A rendered chain result.
#[derive(Debug, Clone)]
pub struct PresentResult {
    /// Full model-facing body, ending with the `[exit:N | Mms]` footer.
    pub output: String,
    /// Lossy-decoded stdout head; empty when the binary guard fired.
    pub stdout_raw: String,
    /// Lossy-decoded stderr head with per-stage `[name]\t` prefixes.
    pub stderr_raw: String,
    pub exit_code: i32,
    pub duration_ms: u128,
    /// Whether either stream was cut.
    pub truncated: bool,
    /// Set when stdout overflowed and its spill file was written.
    pub overflow_file: Option<PathBuf>,
    /// Set when stderr overflowed and its spill file was written.
    pub stderr_overflow_file: Option<PathBuf>,
    pub attachments: Vec<Attachment>,
}

/// Render a completed chain's output within `cap`, footer included;
/// `duration` is the elapsed time shown in the footer. Stderr keeps at least
/// half the budget when both streams overflow it.
pub fn present(out: CommandOutput, cap: &OutputCap, duration: Duration) -> PresentResult {
    let duration_ms = duration.as_millis();
    let footer = format!("[exit:{} | {}ms]", out.exit_code, duration_ms);
    let room = cap.budget().saturating_sub(Budget::of(&footer));
    let stderr_text = String::from_utf8_lossy(&out.stderr).into_owned();

    if let Some(label) = binary_label(&out.stdout) {
        let notice = format!(
            "[error] binary output ({}, {}). Use: cat -b <path>\n",
            label,
            human_size(out.stdout.len() as u64),
        );
        let stderr = cut_stderr(cap, stderr_text, room.saturating_sub(Budget::of(&notice)));
        return present_binary(out, notice, &footer, stderr, duration_ms);
    }

    let stdout_text = String::from_utf8_lossy(&out.stdout).into_owned();
    let stderr_room = room
        .saturating_sub(Budget::of(&stdout_text))
        .saturating_sub(Budget::NEWLINE)
        .max(room.half());
    let stderr = cut_stderr(cap, stderr_text, stderr_room);
    let mut stderr_block = String::new();
    push_stderr(&mut stderr_block, &stderr);
    let stdout_room = room
        .saturating_sub(Budget::of(&stderr_block))
        .saturating_sub(Budget::NEWLINE);
    let stdout = cap.cut_stream(stdout_text, "output", stdout_room, SPILL_STEM);

    let mut body = stdout.visible();
    terminate_line(&mut body);
    body.push_str(&stderr_block);
    body.push_str(&footer);

    PresentResult {
        output: body,
        truncated: stdout.truncated() || stderr.truncated(),
        stdout_raw: stdout.head,
        stderr_raw: stderr.head,
        exit_code: out.exit_code,
        duration_ms,
        overflow_file: stdout.overflow_file,
        stderr_overflow_file: stderr.overflow_file,
        attachments: out.attachments,
    }
}

/// Cut stderr to `room`, less the `[stderr] ` prefix and closing newline
/// [`push_stderr`] adds.
fn cut_stderr(cap: &OutputCap, text: String, room: Budget) -> StreamCut {
    let frame = Budget {
        bytes: STDERR_PREFIX.len() + 1,
        lines: 0,
    };
    cap.cut_stream(text, "stderr", room.saturating_sub(frame), SPILL_STEM)
}

fn present_binary(
    out: CommandOutput,
    notice: String,
    footer: &str,
    stderr: StreamCut,
    duration_ms: u128,
) -> PresentResult {
    let mut body = notice;
    push_stderr(&mut body, &stderr);
    body.push_str(footer);
    PresentResult {
        output: body,
        stdout_raw: String::new(),
        truncated: stderr.truncated(),
        stderr_raw: stderr.head,
        exit_code: out.exit_code,
        duration_ms,
        overflow_file: None,
        stderr_overflow_file: stderr.overflow_file,
        attachments: out.attachments,
    }
}

fn push_stderr(body: &mut String, stderr: &StreamCut) {
    if stderr.head.is_empty() && !stderr.truncated() {
        return;
    }
    body.push_str(STDERR_PREFIX);
    body.push_str(stderr.visible().trim_end_matches('\n'));
    body.push('\n');
}

/// End `text` with a newline unless it is empty or already does.
fn terminate_line(text: &mut String) {
    if !text.is_empty() && !text.ends_with('\n') {
        text.push('\n');
    }
}

/// Count lines, where an unterminated final line still counts.
pub(crate) fn count_lines(s: &str) -> usize {
    if s.is_empty() {
        return 0;
    }
    let newlines = s.bytes().filter(|b| *b == b'\n').count();
    if s.ends_with('\n') {
        newlines
    } else {
        newlines + 1
    }
}

/// Truncate `s` to at most `max_lines` lines and `max_bytes` bytes,
/// never splitting a UTF-8 character.
pub(crate) fn truncate_lines_bytes(s: &str, max_lines: usize, max_bytes: usize) -> String {
    if max_lines == 0 || max_bytes == 0 {
        return String::new();
    }

    let mut cut = s.len();
    let mut seen = 0usize;
    for (i, b) in s.bytes().enumerate() {
        if b == b'\n' {
            seen += 1;
            if seen == max_lines {
                cut = i + 1;
                break;
            }
        }
    }

    if cut > max_bytes {
        cut = s.floor_char_boundary(max_bytes);
    }

    s[..cut].to_string()
}

#[cfg(test)]
mod tests {
    use std::os::unix::fs::PermissionsExt;

    use tempfile::tempdir;

    use assistd_ipc::ImageMime;

    use super::spill::OVERFLOW_FILE_MODE;
    use super::*;
    use crate::fixtures::PNG_BYTES;

    fn spec_in(dir: &Path) -> PresentSpec {
        PresentSpec {
            max_lines: 200,
            max_bytes: 50 * 1024,
            overflow_dir: dir.to_path_buf(),
        }
    }

    fn output(stdout: &[u8], stderr: &[u8], exit_code: i32) -> CommandOutput {
        CommandOutput {
            stdout: stdout.to_vec(),
            stderr: stderr.to_vec(),
            exit_code,
            attachments: Vec::new(),
        }
    }

    fn present_ms(out: CommandOutput, spec: &PresentSpec, ms: u64) -> PresentResult {
        present(
            out,
            &OutputCap::new(spec.clone()),
            Duration::from_millis(ms),
        )
    }

    #[test]
    fn truncate_lines_bytes_applies_both_caps() {
        for (s, max_lines, max_bytes, expected) in [
            ("a\nb\nc\nd\ne\n", 3, 1024, "a\nb\nc\n"),
            ("abcdefghij\n", 100, 5, "abcde"),
            ("日本", 100, 4, "日"),
            ("a\nb\nc\n", 100, 1024, "a\nb\nc\n"),
        ] {
            assert_eq!(
                truncate_lines_bytes(s, max_lines, max_bytes),
                expected,
                "{s:?} lines={max_lines} bytes={max_bytes}"
            );
        }
    }

    #[test]
    fn present_appends_footer_on_success() {
        let dir = tempdir().unwrap();
        let r = present_ms(
            CommandOutput::ok(b"hello\n".to_vec()),
            &spec_in(dir.path()),
            7,
        );
        assert_eq!(r.output, "hello\n[exit:0 | 7ms]");
        assert_eq!(r.stdout_raw, "hello\n");
        assert_eq!(r.exit_code, 0);
        assert_eq!(r.duration_ms, 7);
        assert!(!r.truncated);
        assert!(r.overflow_file.is_none());
    }

    #[test]
    fn present_shows_stderr_on_zero_exit() {
        let dir = tempdir().unwrap();
        let r = present_ms(
            output(
                b"",
                b"[error] unknown command: find. Available: cat, ls\n",
                0,
            ),
            &spec_in(dir.path()),
            1,
        );
        assert_eq!(
            r.output,
            "[stderr] [error] unknown command: find. Available: cat, ls\n[exit:0 | 1ms]"
        );
    }

    #[test]
    fn present_overflow_writes_temp_file_and_exposes_path() {
        let dir = tempdir().unwrap();
        let big: Vec<u8> = (1..=5000)
            .flat_map(|i| format!("line {i}\n").into_bytes())
            .collect();
        let r = present_ms(CommandOutput::ok(big.clone()), &spec_in(dir.path()), 9);

        assert!(r.truncated);
        let path = r.overflow_file.as_ref().expect("overflow path");
        assert_eq!(path, &dir.path().join("cmd-1.txt"));
        assert_eq!(std::fs::read(path).unwrap(), big);
        let mode = std::fs::metadata(path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, OVERFLOW_FILE_MODE);

        let head: String = (1..=194).map(|i| format!("line {i}\n")).collect();
        let p = path.display();
        let size = human_size(big.len() as u64);
        assert_eq!(
            r.output,
            format!(
                "[OUTPUT TRUNCATED] Showing only lines 1-194 of 5000 ({size}); the rest is NOT shown.\n\
                 {head}\
                 [OUTPUT TRUNCATED] Lines 195-5000 were left out. Do not treat the lines above as \
                 the whole output.\n\
                 Full output: {p}\n\
                 Next page: cat {p} | head -n 388 | tail -n 194\n\
                 Search it: cat {p} | grep PATTERN\n\
                 [exit:0 | 9ms]",
            )
        );
        assert_eq!(r.stdout_raw, head);
    }

    #[test]
    fn present_binary_guard_suppresses_stdout_preserves_attachments() {
        let dir = tempdir().unwrap();
        let out = CommandOutput {
            attachments: vec![Attachment::Image {
                mime: ImageMime::Png,
                bytes: PNG_BYTES.to_vec(),
            }],
            ..output(PNG_BYTES, b"", 0)
        };
        let r = present_ms(out, &spec_in(dir.path()), 2);
        assert_eq!(
            r.output,
            format!(
                "[error] binary output (image/png, {}). Use: cat -b <path>\n[exit:0 | 2ms]",
                human_size(PNG_BYTES.len() as u64)
            )
        );
        assert_eq!(r.stdout_raw, "");
        assert!(!r.truncated);
        assert!(r.overflow_file.is_none());
        assert_eq!(r.attachments.len(), 1);
    }

    #[test]
    fn output_cap_cuts_within_its_budget_and_spills_under_the_stem() {
        let dir = tempdir().unwrap();
        let spec = PresentSpec {
            max_lines: 8,
            max_bytes: 1024,
            overflow_dir: dir.path().to_path_buf(),
        };
        let cap = OutputCap::new(spec);
        let body: String = (1..=10).map(|i| format!("line {i}\n")).collect();

        for expected_file in ["mcp__web-1.txt", "reminisce-2.txt"] {
            let stem = expected_file.split_once('-').unwrap().0;
            let cut = cap.cap(stem, body.clone());
            let spilled = dir.path().join(expected_file);
            assert_eq!(cut.overflow_file.as_deref(), Some(spilled.as_path()));
            assert!(cut.truncated);
            assert_eq!(std::fs::read_to_string(&spilled).unwrap(), body);
            assert!(
                cut.text.starts_with(
                    "[OUTPUT TRUNCATED] Showing only lines 1-3 of 10 (71B); the rest is NOT shown.\n\
                     line 1\nline 2\nline 3\n[OUTPUT TRUNCATED] Lines 4-10 were left out."
                ),
                "{}",
                cut.text
            );
            assert_eq!(count_lines(&cut.text), 8);
        }
    }

    #[test]
    fn present_fits_the_cap_when_both_streams_overflow() {
        let dir = tempdir().unwrap();
        let spec = PresentSpec {
            max_lines: 200,
            max_bytes: 4096,
            overflow_dir: dir.path().to_path_buf(),
        };
        let stdout: Vec<u8> = (1..=3000)
            .flat_map(|i| format!("out {i}\n").into_bytes())
            .collect();
        let stderr: Vec<u8> = (1..=3000)
            .flat_map(|i| format!("err {i}\n").into_bytes())
            .collect();
        let r = present_ms(output(&stdout, &stderr, 1), &spec, 5);

        assert!(r.output.len() <= spec.max_bytes, "{}", r.output.len());
        assert!(count_lines(&r.output) <= spec.max_lines);
        assert!(
            r.output
                .contains("[stderr] [STDERR TRUNCATED] Showing only lines 1-")
        );
        assert!(r.output.contains("\nerr 1\n"));
        assert!(r.output.ends_with("[exit:1 | 5ms]"));
        assert_eq!(std::fs::read(r.overflow_file.unwrap()).unwrap(), stdout);
        assert_eq!(
            std::fs::read(r.stderr_overflow_file.unwrap()).unwrap(),
            stderr
        );
    }

    #[test]
    fn present_overflow_truncates_stderr_and_spills_it() {
        let dir = tempdir().unwrap();
        let big: Vec<u8> = (1..=5000)
            .flat_map(|i| format!("[find]\terr {i}\n").into_bytes())
            .collect();
        let r = present_ms(output(b"ok\n", &big, 1), &spec_in(dir.path()), 3);

        assert!(r.truncated);
        assert!(r.overflow_file.is_none());
        let path = r.stderr_overflow_file.as_ref().expect("stderr spill path");
        assert_eq!(path, &dir.path().join("cmd-1.txt"));
        assert_eq!(std::fs::read(path).unwrap(), big);

        let head: String = (1..=193).map(|i| format!("[find]\terr {i}\n")).collect();
        let p = path.display();
        let size = human_size(big.len() as u64);
        assert_eq!(
            r.output,
            format!(
                "ok\n[stderr] [STDERR TRUNCATED] Showing only lines 1-193 of 5000 ({size}); the rest \
                 is NOT shown.\n\
                 {head}\
                 [STDERR TRUNCATED] Lines 194-5000 were left out. Do not treat the lines above as \
                 the whole stderr.\n\
                 Full stderr: {p}\n\
                 Next page: cat {p} | head -n 386 | tail -n 193\n\
                 Search it: cat {p} | grep PATTERN\n\
                 [exit:1 | 3ms]",
            )
        );
        assert_eq!(r.stdout_raw, "ok\n");
        assert_eq!(r.stderr_raw, head);
    }
}
