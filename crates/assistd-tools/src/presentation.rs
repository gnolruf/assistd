//! Renders a completed chain's [`CommandOutput`] for the model: refuses
//! binary, truncates long stdout and stderr (spilling each to a file),
//! and ends with an `[exit:N | Mms]` footer.

use std::fs::OpenOptions;
use std::io::Write;
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use assistd_config::ToolsOutputConfig;
use assistd_config::defaults::default_tools_overflow_dir;

use crate::command::{Attachment, CommandOutput};
use crate::commands::cat::{human_size, sniff_binary};

/// Spill files hold raw tool output, so only the daemon's user may read them.
const OVERFLOW_FILE_MODE: u32 = 0o600;

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
    fn default() -> Self {
        Self {
            max_lines: 200,
            max_bytes: 50 * 1024,
            overflow_dir: default_tools_overflow_dir(),
        }
    }
}

/// Cuts text bodies to one [`PresentSpec`], spilling each overflow in
/// full as `<stem>-<n>.txt` in the spec's overflow directory.
#[derive(Debug)]
pub struct TextTruncator {
    spec: PresentSpec,
    stem: String,
    counter: AtomicU64,
}

/// A text body after [`TextTruncator::truncate`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TruncatedText {
    /// The visible head, followed by the truncation notice when cut.
    pub text: String,
    pub truncated: bool,
    /// Set when the body overflowed and the spill file was written.
    pub overflow_file: Option<PathBuf>,
}

impl TextTruncator {
    /// A truncator whose spill files are named after `stem`.
    pub fn new(spec: PresentSpec, stem: impl Into<String>) -> Self {
        Self {
            spec,
            stem: stem.into(),
            counter: AtomicU64::new(0),
        }
    }

    /// Return `text` unchanged when it fits, else its head plus the
    /// truncation notice, with the whole body spilled to a file.
    pub fn truncate(&self, text: String) -> TruncatedText {
        let cut = cut_stream(text, "output", &self.spec, &self.stem, &self.counter);
        let truncated = cut.truncated();
        let mut text = cut.head;
        if let Some(notice) = &cut.notice {
            terminate_line(&mut text);
            text.push_str(notice);
        }
        TruncatedText {
            text,
            truncated,
            overflow_file: cut.overflow_file,
        }
    }
}

/// One stream cut to a [`PresentSpec`]: its visible head and, when cut,
/// the notice pointing at the spill file.
#[derive(Debug)]
struct StreamCut {
    head: String,
    notice: Option<String>,
    overflow_file: Option<PathBuf>,
}

impl StreamCut {
    fn truncated(&self) -> bool {
        self.notice.is_some()
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

/// Render a completed chain's output. `counter` numbers the spill files;
/// `duration` is the elapsed time shown in the footer.
pub fn present(
    out: CommandOutput,
    spec: &PresentSpec,
    counter: &AtomicU64,
    duration: Duration,
) -> PresentResult {
    let duration_ms = duration.as_millis();
    let footer = format!("[exit:{} | {}ms]", out.exit_code, duration_ms);
    let stderr_text = String::from_utf8_lossy(&out.stderr).into_owned();

    if let Some(label) = binary_label(&out.stdout) {
        let stderr = cut_stream(stderr_text, "stderr", spec, "cmd", counter);
        return present_binary(out, &label, &footer, stderr, duration_ms);
    }

    let stdout_text = String::from_utf8_lossy(&out.stdout).into_owned();
    let stdout = cut_stream(stdout_text, "output", spec, "cmd", counter);
    let stderr = cut_stream(stderr_text, "stderr", spec, "cmd", counter);

    let mut body = String::new();
    body.push_str(&stdout.head);
    terminate_line(&mut body);
    if let Some(notice) = &stdout.notice {
        body.push_str(notice);
    }
    push_stderr(&mut body, &stderr);
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

fn present_binary(
    out: CommandOutput,
    label: &str,
    footer: &str,
    stderr: StreamCut,
    duration_ms: u128,
) -> PresentResult {
    let mut body = format!(
        "[error] binary output ({}, {}). Use: cat -b <path>\n",
        label,
        human_size(out.stdout.len()),
    );
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

/// Cut `text` to `spec`, spilling it in full as `<stem>-<n>.txt` and
/// building a notice that names the stream `label` when it overflows.
fn cut_stream(
    text: String,
    label: &str,
    spec: &PresentSpec,
    stem: &str,
    counter: &AtomicU64,
) -> StreamCut {
    let line_count = count_lines(&text);
    let byte_count = text.len();
    if line_count <= spec.max_lines && byte_count <= spec.max_bytes {
        return StreamCut {
            head: text,
            notice: None,
            overflow_file: None,
        };
    }
    let overflow_file = spill_overflow(text.as_bytes(), spec, stem, counter);
    let head = truncate_lines_bytes(&text, spec.max_lines, spec.max_bytes);
    let notice = truncation_notice(label, line_count, byte_count, overflow_file.as_deref());
    StreamCut {
        head,
        notice: Some(notice),
        overflow_file,
    }
}

/// Write the full body to the next numbered `<stem>-<n>.txt` spill
/// file, logging and returning `None` when that fails.
fn spill_overflow(
    raw: &[u8],
    spec: &PresentSpec,
    stem: &str,
    counter: &AtomicU64,
) -> Option<PathBuf> {
    let n = counter.fetch_add(1, Ordering::Relaxed) + 1;
    let file_name = format!("{stem}-{n}.txt");
    match write_overflow_file(raw, &spec.overflow_dir.join(&file_name)) {
        Ok(path) => Some(path),
        Err(e) => {
            tracing::warn!(
                "failed to write overflow file {file_name} to {}: {e}",
                spec.overflow_dir.display()
            );
            None
        }
    }
}

fn truncation_notice(
    label: &str,
    line_count: usize,
    byte_count: usize,
    overflow_file: Option<&Path>,
) -> String {
    let mut notice = format!(
        "--- {label} truncated ({} lines, {}) ---\n",
        line_count,
        human_size(byte_count),
    );
    if let Some(path) = overflow_file {
        let display = path.display();
        notice.push_str(&format!("Full {label}: {display}\n"));
        notice.push_str(&format!("Explore: cat {display} | grep\n"));
        notice.push_str(&format!("cat {display} | tail -n 100\n"));
    }
    notice
}

fn push_stderr(body: &mut String, stderr: &StreamCut) {
    if stderr.head.is_empty() && stderr.notice.is_none() {
        return;
    }
    body.push_str("[stderr] ");
    body.push_str(stderr.head.trim_end_matches('\n'));
    body.push('\n');
    if let Some(notice) = &stderr.notice {
        body.push_str(notice);
    }
}

/// End `text` with a newline unless it is empty or already does.
fn terminate_line(text: &mut String) {
    if !text.is_empty() && !text.ends_with('\n') {
        text.push('\n');
    }
}

/// Why `raw` must not reach the model: a sniffed MIME type (or
/// `application/octet-stream`) for NUL bytes, `invalid-utf8`, or
/// `control-chars` above 10% controls. `None` for plain text.
pub(crate) fn binary_label(raw: &[u8]) -> Option<String> {
    if raw.is_empty() {
        return None;
    }

    if raw.contains(&0u8) {
        return Some(sniff_binary(raw).unwrap_or_else(|| "application/octet-stream".into()));
    }

    let Ok(s) = std::str::from_utf8(raw) else {
        return Some("invalid-utf8".into());
    };

    let total = s.chars().count();
    if total == 0 {
        return None;
    }
    let controls = s.chars().filter(|c| is_suspicious_control(*c)).count();
    if controls * 10 > total {
        return Some("control-chars".into());
    }

    None
}

fn is_suspicious_control(c: char) -> bool {
    let cp = c as u32;
    if cp == 0x09 || cp == 0x0A || cp == 0x0D {
        return false;
    }
    cp <= 0x1F || cp == 0x7F
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
        cut = max_bytes;
        while cut > 0 && !s.is_char_boundary(cut) {
            cut -= 1;
        }
    }

    s[..cut].to_string()
}

fn write_overflow_file(raw: &[u8], path: &Path) -> std::io::Result<PathBuf> {
    OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(OVERFLOW_FILE_MODE)
        .open(path)?
        .write_all(raw)?;
    Ok(path.to_path_buf())
}

#[cfg(test)]
mod tests;
