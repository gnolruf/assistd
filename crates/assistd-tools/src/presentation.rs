//! Layer 2 — the LLM presentation layer. Runs once on the final
//! [`CommandOutput`] of a completed chain (Layer 1's output). Responsible for:
//!
//! - Binary guarding: refuses to surface null-byte / non-UTF-8 / control-heavy
//!   bytes to the model.
//! - Overflow spill: line/byte-truncates the body, writes the raw stdout to a
//!   temp file under `overflow_dir`, and appends exploration hints.
//! - `[stderr]` attachment on non-zero exit so the model sees *why* a command
//!   failed.
//! - `[exit:N | Mms]` metadata footer on every successful presentation so the
//!   model can distinguish a cache hit from a timeout.
//!
//! This layer is deliberately kept out of the chain executor: Layer 1 (pipes,
//! sequencing, and-or) threads raw bytes between stages without any of these
//! transforms, so `cat bigfile | grep foo | wc -l` sees the full cat output
//! flow into grep — truncation only kicks in on the final `wc -l` result.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use crate::command::{Attachment, CommandOutput};
use crate::commands::cat::{human_size, sniff_binary};

/// Limits and destinations for Layer 2 output rendering. Built from
/// `config.tools.output` in `daemon.rs`.
#[derive(Debug, Clone)]
pub struct PresentSpec {
    /// Max lines of stdout surfaced to the LLM before truncation. Config:
    /// `tools.output.max_lines` (default 200).
    pub max_lines: usize,
    /// Max bytes of the truncated head. Defense-in-depth against a
    /// single-line command emitting megabytes. Config: `tools.output.max_kb`
    /// × 1024 (default 50 KB = 51200 bytes).
    pub max_bytes: usize,
    /// Directory where full overflow output is spilled as `cmd-<n>.txt`.
    /// Cleared + recreated by the daemon on startup.
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

/// Structured output of a single Layer 2 presentation. `RunTool` projects
/// this into the JSON surface the LLM consumes.
#[derive(Debug, Clone)]
pub struct PresentResult {
    /// Full LLM-facing body: truncated head (or binary-guard error) + optional
    /// `[stderr] ...` block + `[exit:N | Mms]` footer as the final line.
    pub output: String,
    /// Lossy-decoded stdout head surfaced to programmatic consumers. Equals
    /// the truncated head in overflow mode (full content lives in
    /// `overflow_file`). Empty when `binary_guard` suppressed stdout.
    pub stdout_raw: String,
    /// Lossy-decoded full stderr (preserving the chain executor's per-stage
    /// `[name]\t` prefix).
    pub stderr_raw: String,
    pub exit_code: i32,
    pub duration_ms: u128,
    /// `true` iff line or byte threshold was hit and the body contains a
    /// `--- output truncated ---` banner.
    pub truncated: bool,
    /// `Some(path)` iff overflow fired AND the write succeeded. `None` on a
    /// degraded-write failure (the body still shows the head + banner, but
    /// omits the `Full output:` / `Explore:` lines).
    pub overflow_file: Option<PathBuf>,
    pub attachments: Vec<Attachment>,
}

/// Render a completed chain's `CommandOutput` into an LLM-facing
/// `PresentResult`. Measures nothing itself — `duration` is whatever the
/// caller timed around their `execute()` call.
pub fn present(
    out: CommandOutput,
    spec: &PresentSpec,
    counter: &AtomicU64,
    duration: Duration,
) -> PresentResult {
    let duration_ms = duration.as_millis();
    let footer = format!("[exit:{} | {}ms]", out.exit_code, duration_ms);
    let stderr_raw = String::from_utf8_lossy(&out.stderr).into_owned();

    // 1. Binary guard — stdout might be an image dump, a stray `/dev/urandom`
    //    read, or Latin-1 text. Refuse to splat it into the model's context.
    if let Some(label) = binary_guard(&out.stdout) {
        let mut body = format!(
            "[error] binary output ({}, {}). Use: cat -b <path>",
            label,
            human_size(out.stdout.len()),
        );
        if out.exit_code != 0 && !stderr_raw.is_empty() {
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

    // 2. Decode stdout and measure.
    let stdout_str = String::from_utf8_lossy(&out.stdout).into_owned();
    let line_count = count_lines(&stdout_str);
    let byte_count = stdout_str.len();

    // 3. Overflow spill.
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

    // 4. Assemble body: head (+ trailing newline) → banner (+ hints) → stderr → footer.
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
            body.push_str(&format!("cat {display} | tail 100\n"));
        }
    }
    if out.exit_code != 0 && !stderr_raw.is_empty() {
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

/// Return `Some(label)` if `raw` should not be surfaced verbatim to the LLM:
/// any NUL byte, invalid UTF-8, or >10% non-whitespace control characters.
/// Returns `None` for empty input (empty is not binary).
///
/// Label is either a MIME type (from magic-byte sniff), or one of the
/// synthetic tags `"application/octet-stream"`, `"invalid-utf8"`,
/// `"control-chars"`.
pub(crate) fn binary_guard(raw: &[u8]) -> Option<String> {
    if raw.is_empty() {
        return None;
    }

    // Rule 1: any NUL byte → binary. `sniff_binary` gives a nicer MIME label
    //  when the magic bytes are recognized (PNG, ZIP, …).
    if raw.contains(&0u8) {
        return Some(sniff_binary(raw).unwrap_or_else(|| "application/octet-stream".into()));
    }

    // Rule 2: non-UTF-8 bytes (e.g. Latin-1 text with 0xE9) → binary.
    let s = match std::str::from_utf8(raw) {
        Ok(s) => s,
        Err(_) => return Some("invalid-utf8".into()),
    };

    // Rule 3: more than 10% control chars (excluding \t, \n, \r) → binary.
    //  Counting over chars (not bytes) so multibyte UTF-8 runes aren't
    //  double-counted.
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

/// Count lines in `s`. A trailing non-newline-terminated line counts as a
/// line — `"a"` is 1 line, `"a\n"` is 1, `"a\nb"` is 2, `""` is 0.
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

/// Truncate `s` to at most `max_lines` lines and at most `max_bytes` bytes.
/// The byte clamp respects UTF-8 char boundaries so multibyte runes are never
/// split. Returns an owned `String`.
pub(crate) fn truncate_lines_bytes(s: &str, max_lines: usize, max_bytes: usize) -> String {
    if max_lines == 0 || max_bytes == 0 {
        return String::new();
    }

    // Line clamp: cut after the Nth '\n' (inclusive). If fewer than N
    // newlines exist, keep everything through end-of-string.
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

    // Byte clamp (UTF-8 boundary safe). Walk backward if the clamp lands
    // inside a multibyte codepoint.
    if cut > max_bytes {
        cut = max_bytes;
        while cut > 0 && !s.is_char_boundary(cut) {
            cut -= 1;
        }
    }

    s[..cut].to_string()
}

/// Write `raw` to `<dir>/cmd-<n>.txt`. Caller is responsible for ensuring
/// `dir` exists — the daemon creates it at startup. Degraded-write failure
/// (disk full, bad perms) is handled one level up in `present`.
fn write_overflow_file(raw: &[u8], dir: &Path, n: u64) -> std::io::Result<PathBuf> {
    let path = dir.join(format!("cmd-{n}.txt"));
    std::fs::write(&path, raw)?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::command::CommandOutput;
    use tempfile::tempdir;

    // Minimal valid 1x1 PNG — contains NUL bytes in IHDR.
    const PNG_BYTES: &[u8] = &[
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44,
        0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x06, 0x00, 0x00, 0x00, 0x1F,
        0x15, 0xC4, 0x89, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9C, 0x63, 0x00,
        0x01, 0x00, 0x00, 0x05, 0x00, 0x01, 0x0D, 0x0A, 0x2D, 0xB4, 0x00, 0x00, 0x00, 0x00, 0x49,
        0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
    ];

    fn spec_in(dir: &Path) -> PresentSpec {
        PresentSpec {
            max_lines: 200,
            max_bytes: 50 * 1024,
            overflow_dir: dir.to_path_buf(),
        }
    }

    fn tiny_spec_in(dir: &Path, max_lines: usize, max_bytes: usize) -> PresentSpec {
        PresentSpec {
            max_lines,
            max_bytes,
            overflow_dir: dir.to_path_buf(),
        }
    }

    // --- binary_guard ------------------------------------------------------

    #[test]
    fn binary_guard_rejects_nul_byte_with_mime_label() {
        let label = binary_guard(PNG_BYTES).expect("png should be flagged");
        assert_eq!(label, "image/png");
    }

    #[test]
    fn binary_guard_rejects_nul_without_magic_match() {
        let mut bytes = b"plain text".to_vec();
        bytes.push(0);
        let label = binary_guard(&bytes).expect("NUL should be flagged");
        assert_eq!(label, "application/octet-stream");
    }

    #[test]
    fn binary_guard_rejects_invalid_utf8() {
        let bytes: &[u8] = &[0xC3, 0x28, b' ', b'h', b'i']; // 0xC3 0x28 is invalid
        let label = binary_guard(bytes).expect("invalid utf-8 should be flagged");
        assert_eq!(label, "invalid-utf8");
    }

    #[test]
    fn binary_guard_rejects_high_control_ratio() {
        // 20 chars: 2 non-whitespace control chars (\x01, \x02) = 10%. Must
        // exceed 10% to reject, so add one more (15%).
        let bytes = b"abcdef\x01\x02\x03ghijklmnopq".to_vec();
        let label = binary_guard(&bytes).expect("control ratio should trip");
        assert_eq!(label, "control-chars");
    }

    #[test]
    fn binary_guard_accepts_tabs_and_newlines() {
        // 50% whitespace-controls — none are "suspicious".
        let bytes = b"a\tb\nc\td\ne\tf\n".to_vec();
        assert!(binary_guard(&bytes).is_none());
    }

    #[test]
    fn binary_guard_accepts_empty_input() {
        assert!(binary_guard(&[]).is_none());
    }

    #[test]
    fn binary_guard_accepts_normal_text() {
        let bytes = b"hello world\nthis is fine\n".to_vec();
        assert!(binary_guard(&bytes).is_none());
    }

    #[test]
    fn binary_guard_accepts_utf8_multibyte() {
        let bytes = "héllo wörld ñ 日本語\n".as_bytes().to_vec();
        assert!(binary_guard(&bytes).is_none());
    }

    #[test]
    fn binary_guard_accepts_exactly_10_percent_controls() {
        // 20 chars, exactly 2 controls (10%). Rule is strict >10% → accept.
        let bytes = b"abcdefgh\x01\x02ijklmnopqr".to_vec();
        assert_eq!(bytes.len(), 20);
        assert!(binary_guard(&bytes).is_none());
    }

    // --- count_lines -------------------------------------------------------

    #[test]
    fn count_lines_empty_is_zero() {
        assert_eq!(count_lines(""), 0);
    }

    #[test]
    fn count_lines_no_trailing_newline_counts_partial() {
        assert_eq!(count_lines("one"), 1);
        assert_eq!(count_lines("one\ntwo"), 2);
    }

    #[test]
    fn count_lines_trailing_newline_exact() {
        assert_eq!(count_lines("one\n"), 1);
        assert_eq!(count_lines("one\ntwo\n"), 2);
    }

    // --- truncate_lines_bytes ---------------------------------------------

    #[test]
    fn truncate_lines_bytes_line_cap_first() {
        let s = "a\nb\nc\nd\ne\n";
        let t = truncate_lines_bytes(s, 3, 1024);
        assert_eq!(t, "a\nb\nc\n");
    }

    #[test]
    fn truncate_lines_bytes_byte_cap_clamps_head() {
        let s = "abcdefghij\n"; // 11 bytes, 1 line
        let t = truncate_lines_bytes(s, 100, 5);
        assert_eq!(t, "abcde");
    }

    #[test]
    fn truncate_lines_bytes_utf8_boundary_safe() {
        // "日" is 3 bytes: 0xE6 0x97 0xA5.
        let s = "日本"; // 6 bytes total
        let t = truncate_lines_bytes(s, 100, 4); // clamp inside 2nd rune
        assert_eq!(t, "日"); // 3 bytes — must not return partial rune
    }

    #[test]
    fn truncate_lines_bytes_returns_full_when_under_caps() {
        let s = "a\nb\nc\n";
        let t = truncate_lines_bytes(s, 100, 1024);
        assert_eq!(t, s);
    }

    // --- present: footer on every path ------------------------------------

    #[test]
    fn present_appends_footer_on_success() {
        let dir = tempdir().unwrap();
        let out = CommandOutput::ok(b"hello\n".to_vec());
        let counter = AtomicU64::new(0);
        let r = present(
            out,
            &spec_in(dir.path()),
            &counter,
            Duration::from_millis(7),
        );
        assert!(r.output.ends_with("[exit:0 | 7ms]"));
        assert!(r.output.starts_with("hello\n"));
        assert_eq!(r.exit_code, 0);
        assert_eq!(r.duration_ms, 7);
        assert!(!r.truncated);
        assert!(r.overflow_file.is_none());
    }

    #[test]
    fn present_footer_on_zero_stdout_zero_exit() {
        let dir = tempdir().unwrap();
        let out = CommandOutput::ok(Vec::new());
        let counter = AtomicU64::new(0);
        let r = present(
            out,
            &spec_in(dir.path()),
            &counter,
            Duration::from_millis(2),
        );
        assert_eq!(r.output, "[exit:0 | 2ms]");
    }

    // --- present: stderr attachment ---------------------------------------

    #[test]
    fn present_appends_stderr_marker_when_nonzero_and_nonempty() {
        let dir = tempdir().unwrap();
        let out = CommandOutput {
            stdout: b"ok\n".to_vec(),
            stderr: b"[cat]\tboom\n".to_vec(),
            exit_code: 1,
            attachments: Vec::new(),
        };
        let counter = AtomicU64::new(0);
        let r = present(
            out,
            &spec_in(dir.path()),
            &counter,
            Duration::from_millis(5),
        );
        // Body preserves the executor's per-stage prefix inside the marker.
        assert!(
            r.output
                .contains("ok\n[stderr] [cat]\tboom\n[exit:1 | 5ms]")
        );
    }

    #[test]
    fn present_skips_stderr_marker_on_zero_exit_even_if_stderr_present() {
        let dir = tempdir().unwrap();
        let out = CommandOutput {
            stdout: b"ok\n".to_vec(),
            stderr: b"warning\n".to_vec(),
            exit_code: 0,
            attachments: Vec::new(),
        };
        let counter = AtomicU64::new(0);
        let r = present(
            out,
            &spec_in(dir.path()),
            &counter,
            Duration::from_millis(1),
        );
        assert!(!r.output.contains("[stderr]"));
        assert!(r.output.ends_with("[exit:0 | 1ms]"));
    }

    #[test]
    fn present_stderr_survives_with_nonempty_stdout() {
        // Acceptance: stderr is never silently dropped even when stdout is
        // non-empty.
        let dir = tempdir().unwrap();
        let out = CommandOutput {
            stdout: b"stdout content\n".to_vec(),
            stderr: b"stderr content\n".to_vec(),
            exit_code: 127,
            attachments: Vec::new(),
        };
        let counter = AtomicU64::new(0);
        let r = present(
            out,
            &spec_in(dir.path()),
            &counter,
            Duration::from_millis(3),
        );
        assert!(r.output.contains("stdout content"));
        assert!(r.output.contains("[stderr] stderr content"));
        assert!(r.output.ends_with("[exit:127 | 3ms]"));
    }

    // --- present: overflow -----------------------------------------------

    #[test]
    fn present_overflow_writes_temp_file_and_exposes_path() {
        let dir = tempdir().unwrap();
        let big: Vec<u8> = (1..=5000)
            .flat_map(|i| format!("line {i}\n").into_bytes())
            .collect();
        let expected = big.clone();
        let out = CommandOutput::ok(big);
        let counter = AtomicU64::new(0);
        let r = present(
            out,
            &spec_in(dir.path()),
            &counter,
            Duration::from_millis(9),
        );

        assert!(r.truncated);
        let overflow_path = r.overflow_file.as_ref().expect("overflow path");
        assert_eq!(overflow_path, &dir.path().join("cmd-1.txt"));
        assert_eq!(std::fs::read(overflow_path).unwrap(), expected);

        // Body: first 200 lines, banner, full path, explore hints, footer.
        assert!(r.output.contains("line 1\n"));
        assert!(r.output.contains("line 200\n"));
        assert!(!r.output.contains("line 201\n"));
        assert!(r.output.contains("--- output truncated (5000 lines,"));
        assert!(
            r.output
                .contains(&format!("Full output: {}", overflow_path.display()))
        );
        assert!(r.output.contains("Explore: cat "));
        assert!(r.output.contains("| tail 100"));
        assert!(r.output.ends_with("[exit:0 | 9ms]"));
    }

    #[test]
    fn present_overflow_byte_threshold_trips_independently_of_lines() {
        let dir = tempdir().unwrap();
        // One long line — no newline, so count_lines = 1 but bytes > max_bytes.
        let big = vec![b'x'; 10_000];
        let out = CommandOutput::ok(big);
        let counter = AtomicU64::new(0);
        let r = present(
            out,
            &tiny_spec_in(dir.path(), 200, 1024),
            &counter,
            Duration::from_millis(1),
        );
        assert!(r.truncated);
        assert!(r.overflow_file.is_some());
    }

    #[test]
    fn present_overflow_write_failure_degrades() {
        // overflow_dir points at a path that doesn't exist — write fails,
        // presentation still produces a body with the head + banner, but
        // NOT the "Full output:" / "Explore:" lines.
        let bad_dir = PathBuf::from("/nonexistent-assistd-test-dir/nope");
        let spec = PresentSpec {
            max_lines: 2,
            max_bytes: 1024,
            overflow_dir: bad_dir,
        };
        let out = CommandOutput::ok(b"a\nb\nc\nd\n".to_vec());
        let counter = AtomicU64::new(0);
        let r = present(out, &spec, &counter, Duration::from_millis(1));
        assert!(r.truncated);
        assert!(r.overflow_file.is_none(), "write should have failed");
        assert!(r.output.contains("--- output truncated (4 lines,"));
        assert!(
            !r.output.contains("Full output:"),
            "degraded path must omit Full output line"
        );
        assert!(
            !r.output.contains("Explore:"),
            "degraded path must omit Explore hints"
        );
        assert!(r.output.ends_with("[exit:0 | 1ms]"));
    }

    #[test]
    fn present_counter_increments_across_calls() {
        let dir = tempdir().unwrap();
        let counter = AtomicU64::new(0);
        let spec = spec_in(dir.path());
        let mk = || {
            CommandOutput::ok(
                (1..=250)
                    .flat_map(|i| format!("{i}\n").into_bytes())
                    .collect(),
            )
        };
        let r1 = present(mk(), &spec, &counter, Duration::from_millis(1));
        let r2 = present(mk(), &spec, &counter, Duration::from_millis(1));
        let r3 = present(mk(), &spec, &counter, Duration::from_millis(1));
        assert_eq!(r1.overflow_file.unwrap(), dir.path().join("cmd-1.txt"));
        assert_eq!(r2.overflow_file.unwrap(), dir.path().join("cmd-2.txt"));
        assert_eq!(r3.overflow_file.unwrap(), dir.path().join("cmd-3.txt"));
    }

    // --- present: binary guard end-to-end ---------------------------------

    #[test]
    fn present_binary_guard_suppresses_stdout_preserves_attachments() {
        let dir = tempdir().unwrap();
        let out = CommandOutput {
            stdout: PNG_BYTES.to_vec(),
            stderr: Vec::new(),
            exit_code: 0,
            attachments: vec![Attachment::Image {
                mime: "image/png".into(),
                bytes: PNG_BYTES.to_vec(),
            }],
        };
        let counter = AtomicU64::new(0);
        let r = present(
            out,
            &spec_in(dir.path()),
            &counter,
            Duration::from_millis(2),
        );
        assert!(r.output.starts_with("[error] binary output (image/png, "));
        assert!(r.output.contains(". Use: cat -b <path>"));
        assert!(r.output.ends_with("[exit:0 | 2ms]"));
        assert_eq!(r.stdout_raw, "");
        assert!(!r.truncated);
        assert!(r.overflow_file.is_none());
        assert_eq!(r.attachments.len(), 1);
    }

    #[test]
    fn present_binary_guard_with_stderr_on_failure() {
        let dir = tempdir().unwrap();
        let out = CommandOutput {
            stdout: PNG_BYTES.to_vec(),
            stderr: b"something went wrong\n".to_vec(),
            exit_code: 1,
            attachments: Vec::new(),
        };
        let counter = AtomicU64::new(0);
        let r = present(
            out,
            &spec_in(dir.path()),
            &counter,
            Duration::from_millis(4),
        );
        assert!(r.output.starts_with("[error] binary output (image/png"));
        assert!(r.output.contains("\n[stderr] something went wrong\n"));
        assert!(r.output.ends_with("[exit:1 | 4ms]"));
    }

    // --- pipe integrity ---------------------------------------------------
    //
    // Layer 1 integrity (pipes never truncate mid-chain) is covered by the
    // chain::executor tests. Here we confirm that when a final stage emits
    // tiny output after a huge upstream stage, Layer 2 does NOT truncate.
    #[test]
    fn present_does_not_truncate_small_final_output() {
        let dir = tempdir().unwrap();
        let out = CommandOutput::ok(b"5000\n".to_vec()); // final count after big pipe
        let counter = AtomicU64::new(0);
        let r = present(
            out,
            &spec_in(dir.path()),
            &counter,
            Duration::from_millis(1),
        );
        assert!(!r.truncated);
        assert!(r.overflow_file.is_none());
        assert_eq!(r.output, "5000\n[exit:0 | 1ms]");
    }
}
