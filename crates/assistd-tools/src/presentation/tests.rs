use super::*;
use crate::command::CommandOutput;
use crate::fixtures::PNG_BYTES;
use tempfile::tempdir;

// Minimal valid 1x1 PNG; contains NUL bytes in IHDR.

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
    let label = binary_label(PNG_BYTES).expect("png should be flagged");
    assert_eq!(label, "image/png");
}

#[test]
fn binary_guard_rejects_nul_without_magic_match() {
    let mut bytes = b"plain text".to_vec();
    bytes.push(0);
    let label = binary_label(&bytes).expect("NUL should be flagged");
    assert_eq!(label, "application/octet-stream");
}

#[test]
fn binary_guard_rejects_invalid_utf8() {
    let bytes: &[u8] = &[0xC3, 0x28, b' ', b'h', b'i']; // 0xC3 0x28 is invalid
    let label = binary_label(bytes).expect("invalid utf-8 should be flagged");
    assert_eq!(label, "invalid-utf8");
}

#[test]
fn binary_guard_rejects_high_control_ratio() {
    // 20 chars: 2 non-whitespace control chars (\x01, \x02) = 10%. Must
    // exceed 10% to reject, so add one more (15%).
    let bytes = b"abcdef\x01\x02\x03ghijklmnopq".to_vec();
    let label = binary_label(&bytes).expect("control ratio should trip");
    assert_eq!(label, "control-chars");
}

#[test]
fn binary_guard_accepts_tabs_and_newlines() {
    // 50% whitespace-controls; none are "suspicious".
    let bytes = b"a\tb\nc\td\ne\tf\n".to_vec();
    assert!(binary_label(&bytes).is_none());
}

#[test]
fn binary_guard_accepts_empty_input() {
    assert!(binary_label(&[]).is_none());
}

#[test]
fn binary_guard_accepts_normal_text() {
    let bytes = b"hello world\nthis is fine\n".to_vec();
    assert!(binary_label(&bytes).is_none());
}

#[test]
fn binary_guard_accepts_utf8_multibyte() {
    let bytes = "héllo wörld ñ 日本語\n".as_bytes().to_vec();
    assert!(binary_label(&bytes).is_none());
}

#[test]
fn binary_guard_accepts_exactly_10_percent_controls() {
    // 20 chars, exactly 2 controls (10%). Rule is strict >10% → accept.
    let bytes = b"abcdefgh\x01\x02ijklmnopqr".to_vec();
    assert_eq!(bytes.len(), 20);
    assert!(binary_label(&bytes).is_none());
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
    assert_eq!(t, "日"); // 3 bytes; must not return partial rune
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
fn present_shows_stderr_on_zero_exit() {
    let dir = tempdir().unwrap();
    let out = CommandOutput {
        stdout: Vec::new(),
        stderr: b"[error] unknown command: find. Available: cat, ls\n".to_vec(),
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
    assert!(
        r.output.contains("[stderr] [error] unknown command: find"),
        "{}",
        r.output
    );
    assert!(r.output.ends_with("[exit:0 | 1ms]"), "{}", r.output);
}

#[test]
fn present_stderr_survives_with_nonempty_stdout() {
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
    assert!(r.output.contains("| tail -n 100"));
    assert!(r.output.ends_with("[exit:0 | 9ms]"));
}

#[test]
fn present_overflow_byte_threshold_trips_independently_of_lines() {
    let dir = tempdir().unwrap();
    // One long line: no newline, so count_lines = 1 but bytes > max_bytes.
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
    // overflow_dir points at a path that doesn't exist (write fails);
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
