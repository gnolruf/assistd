//! Renders a completed chain's [`CommandOutput`] for the model: refuses
//! binary bytes, truncates long output while spilling the full text to
//! a file, attaches stderr whenever any stage wrote to it (not only on
//! failure, since `find . | head` reports `head`'s exit code), and ends
//! with an `[exit:N | Mms]` footer.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use crate::command::{Attachment, CommandOutput};
use crate::commands::cat::{human_size, sniff_binary};

/// Limits and destinations for output rendering.
#[derive(Debug, Clone)]
pub struct PresentSpec {
    /// Max lines of stdout surfaced before truncation.
    pub max_lines: usize,
    /// Max bytes of the truncated head, so a single huge line is also
    /// bounded.
    pub max_bytes: usize,
    /// Directory where full overflow output is spilled as `cmd-<n>.txt`.
    pub overflow_dir: PathBuf,
}

impl Default for PresentSpec {
    fn default() -> Self {
        Self {
            max_lines: 200,
            max_bytes: 50 * 1024,
            overflow_dir: PathBuf::from("/tmp/assistd-output"),
        }
    }
}

/// A rendered chain result.
#[derive(Debug, Clone)]
pub struct PresentResult {
    /// Full LLM-facing body: head or binary-guard error, optional
    /// `[stderr]` block, and the `[exit:N | Mms]` footer.
    pub output: String,
    /// Lossy-decoded stdout head; empty when the binary guard fired.
    pub stdout_raw: String,
    /// Lossy-decoded full stderr with per-stage `[name]\t` prefixes.
    pub stderr_raw: String,
    pub exit_code: i32,
    pub duration_ms: u128,
    pub truncated: bool,
    /// Set when output overflowed and the spill file was written.
    pub overflow_file: Option<PathBuf>,
    pub attachments: Vec<Attachment>,
}

/// Render a completed chain's output. `duration` is whatever the caller
/// measured around execution.
pub fn present(
    out: CommandOutput,
    spec: &PresentSpec,
    counter: &AtomicU64,
    duration: Duration,
) -> PresentResult {
    let duration_ms = duration.as_millis();
    let footer = format!("[exit:{} | {}ms]", out.exit_code, duration_ms);
    let stderr_raw = String::from_utf8_lossy(&out.stderr).into_owned();

    if let Some(label) = binary_label(&out.stdout) {
        let mut body = format!(
            "[error] binary output ({}, {}). Use: cat -b <path>",
            label,
            human_size(out.stdout.len()),
        );
        if !stderr_raw.is_empty() {
            body.push('\n');
            body.push_str("[stderr] ");
            body.push_str(stderr_raw.trim_end_matches('\n'));
        }
        body.push('\n');
        body.push_str(&footer);
        return PresentResult {
            output: body,
            stdout_raw: String::new(),
            stderr_raw,
            exit_code: out.exit_code,
            duration_ms,
            truncated: false,
            overflow_file: None,
            attachments: out.attachments,
        };
    }

    let stdout_str = String::from_utf8_lossy(&out.stdout).into_owned();
    let line_count = count_lines(&stdout_str);
    let byte_count = stdout_str.len();

    let overflow = line_count > spec.max_lines || byte_count > spec.max_bytes;
    let (visible_head, overflow_file) = if overflow {
        let n = counter.fetch_add(1, Ordering::Relaxed) + 1;
        let write_result = write_overflow_file(&out.stdout, &spec.overflow_dir, n);
        let head = truncate_lines_bytes(&stdout_str, spec.max_lines, spec.max_bytes);
        let path = match write_result {
            Ok(p) => Some(p),
            Err(e) => {
                tracing::warn!(
                    "failed to write overflow file cmd-{n}.txt to {}: {e}",
                    spec.overflow_dir.display()
                );
                None
            }
        };
        (head, path)
    } else {
        (stdout_str.clone(), None)
    };

    let mut body = String::new();
    if !visible_head.is_empty() {
        body.push_str(&visible_head);
        if !body.ends_with('\n') {
            body.push('\n');
        }
    }
    if overflow {
        body.push_str(&format!(
            "--- output truncated ({} lines, {}) ---\n",
            line_count,
            human_size(byte_count),
        ));
        if let Some(p) = &overflow_file {
            let display = p.display();
            body.push_str(&format!("Full output: {display}\n"));
            body.push_str(&format!("Explore: cat {display} | grep\n"));
            body.push_str(&format!("cat {display} | tail -n 100\n"));
        }
    }
    if !stderr_raw.is_empty() {
        body.push_str("[stderr] ");
        body.push_str(stderr_raw.trim_end_matches('\n'));
        body.push('\n');
    }
    body.push_str(&footer);

    PresentResult {
        output: body,
        stdout_raw: visible_head,
        stderr_raw,
        exit_code: out.exit_code,
        duration_ms,
        truncated: overflow,
        overflow_file,
        attachments: out.attachments,
    }
}

/// Why `raw` must not reach the model verbatim: a sniffed MIME type or
/// `application/octet-stream` for NUL bytes, `invalid-utf8`, or
/// `control-chars` when over 10% of characters are non-whitespace
/// controls. `None` when it is plain text (or empty).
pub(crate) fn binary_label(raw: &[u8]) -> Option<String> {
    if raw.is_empty() {
        return None;
    }

    if raw.contains(&0u8) {
        return Some(sniff_binary(raw).unwrap_or_else(|| "application/octet-stream".into()));
    }

    let s = match std::str::from_utf8(raw) {
        Ok(s) => s,
        Err(_) => return Some("invalid-utf8".into()),
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

fn write_overflow_file(raw: &[u8], dir: &Path, n: u64) -> std::io::Result<PathBuf> {
    let path = dir.join(format!("cmd-{n}.txt"));
    std::fs::write(&path, raw)?;
    Ok(path)
}

#[cfg(test)]
mod tests;
