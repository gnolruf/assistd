use tempfile::tempdir;

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
    present(out, spec, &AtomicU64::new(0), Duration::from_millis(ms))
}

#[test]
fn binary_label_flags_bytes_the_model_should_not_see() {
    let mut nul_text = b"plain text".to_vec();
    nul_text.push(0);
    let cases: [(&str, &[u8], Option<&str>); 8] = [
        ("png magic", PNG_BYTES, Some("image/png")),
        (
            "NUL without magic",
            &nul_text[..],
            Some("application/octet-stream"),
        ),
        (
            "invalid utf-8",
            &[0xC3, 0x28, b' ', b'h', b'i'][..],
            Some("invalid-utf8"),
        ),
        (
            "15% controls",
            b"abcdef\x01\x02\x03ghijklmnopq",
            Some("control-chars"),
        ),
        (
            "exactly 10% controls is still text",
            b"abcdefgh\x01\x02ijklmnopqr",
            None,
        ),
        ("tabs and newlines", b"a\tb\nc\td\ne\tf\n", None),
        ("empty", b"", None),
        ("multibyte utf-8", "héllo wörld ñ 日本語\n".as_bytes(), None),
    ];
    for (case, bytes, expected) in cases {
        assert_eq!(binary_label(bytes).as_deref(), expected, "{case}");
    }
}

#[test]
fn count_lines_counts_an_unterminated_last_line() {
    for (s, expected) in [
        ("", 0),
        ("one", 1),
        ("one\ntwo", 2),
        ("one\n", 1),
        ("one\ntwo\n", 2),
    ] {
        assert_eq!(count_lines(s), expected, "{s:?}");
    }
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
fn present_footer_on_zero_stdout_zero_exit() {
    let dir = tempdir().unwrap();
    let r = present_ms(CommandOutput::ok(Vec::new()), &spec_in(dir.path()), 2);
    assert_eq!(r.output, "[exit:0 | 2ms]");
}

#[test]
fn present_appends_stderr_after_stdout() {
    let dir = tempdir().unwrap();
    let r = present_ms(
        output(b"ok\n", b"[cat]\tboom\n", 1),
        &spec_in(dir.path()),
        5,
    );
    assert_eq!(r.output, "ok\n[stderr] [cat]\tboom\n[exit:1 | 5ms]");
}

/// A pipeline reports its last stage's exit code, so an earlier stage's
/// failure is only visible through stderr.
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

    let head: String = (1..=200).map(|i| format!("line {i}\n")).collect();
    let p = path.display();
    assert_eq!(
        r.output,
        format!(
            "{head}--- output truncated (5000 lines, {}) ---\n\
             Full output: {p}\n\
             Explore: cat {p} | grep\n\
             cat {p} | tail -n 100\n\
             [exit:0 | 9ms]",
            human_size(big.len()),
        )
    );
    assert_eq!(r.stdout_raw, head);
}

#[test]
fn present_overflow_byte_threshold_trips_independently_of_lines() {
    let dir = tempdir().unwrap();
    let spec = PresentSpec {
        max_bytes: 1024,
        ..spec_in(dir.path())
    };
    let r = present_ms(CommandOutput::ok(vec![b'x'; 10_000]), &spec, 1);
    assert!(r.truncated);
    assert_eq!(r.stdout_raw, "x".repeat(1024));
    assert_eq!(r.overflow_file, Some(dir.path().join("cmd-1.txt")));
}

/// An unwritable overflow dir still yields the head and banner, just
/// without pointing at a file that does not exist.
#[test]
fn present_overflow_write_failure_degrades() {
    let spec = PresentSpec {
        max_lines: 2,
        max_bytes: 1024,
        overflow_dir: PathBuf::from("/nonexistent-assistd-test-dir/nope"),
    };
    let r = present_ms(CommandOutput::ok(b"a\nb\nc\nd\n".to_vec()), &spec, 1);
    assert!(r.truncated);
    assert!(r.overflow_file.is_none());
    assert_eq!(
        r.output,
        "a\nb\n--- output truncated (4 lines, 8B) ---\n[exit:0 | 1ms]"
    );
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
    for n in 1..=3 {
        let r = present(mk(), &spec, &counter, Duration::from_millis(1));
        assert_eq!(
            r.overflow_file,
            Some(dir.path().join(format!("cmd-{n}.txt")))
        );
    }
}

#[test]
fn present_binary_guard_suppresses_stdout_preserves_attachments() {
    let dir = tempdir().unwrap();
    let out = CommandOutput {
        attachments: vec![Attachment::Image {
            mime: "image/png".into(),
            bytes: PNG_BYTES.to_vec(),
        }],
        ..output(PNG_BYTES, b"", 0)
    };
    let r = present_ms(out, &spec_in(dir.path()), 2);
    assert_eq!(
        r.output,
        format!(
            "[error] binary output (image/png, {}). Use: cat -b <path>\n[exit:0 | 2ms]",
            human_size(PNG_BYTES.len())
        )
    );
    assert_eq!(r.stdout_raw, "");
    assert!(!r.truncated);
    assert!(r.overflow_file.is_none());
    assert_eq!(r.attachments.len(), 1);
}

#[test]
fn present_binary_guard_keeps_stderr() {
    let dir = tempdir().unwrap();
    let r = present_ms(
        output(PNG_BYTES, b"something went wrong\n", 1),
        &spec_in(dir.path()),
        4,
    );
    assert_eq!(
        r.output,
        format!(
            "[error] binary output (image/png, {}). Use: cat -b <path>\n\
             [stderr] something went wrong\n[exit:1 | 4ms]",
            human_size(PNG_BYTES.len())
        )
    );
}
