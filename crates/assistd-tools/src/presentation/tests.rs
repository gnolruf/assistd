use std::os::unix::fs::PermissionsExt;

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

fn sorted_file_names(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().into_string().unwrap())
        .collect();
    names.sort();
    names
}

fn write_file_aged(dir: &Path, name: &str, len: usize, age_secs: u64) -> PathBuf {
    let path = dir.join(name);
    std::fs::write(&path, vec![b'x'; len]).unwrap();
    let modified = SystemTime::now() - Duration::from_secs(age_secs);
    std::fs::File::options()
        .write(true)
        .open(&path)
        .unwrap()
        .set_modified(modified)
        .unwrap();
    path
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
    let mode = std::fs::metadata(path).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode, OVERFLOW_FILE_MODE);

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
            human_size(big.len() as u64),
        )
    );
    assert_eq!(r.stdout_raw, head);
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
            human_size(PNG_BYTES.len() as u64)
        )
    );
    assert_eq!(r.stdout_raw, "");
    assert!(!r.truncated);
    assert!(r.overflow_file.is_none());
    assert_eq!(r.attachments.len(), 1);
}

#[test]
fn text_truncator_cuts_and_spills_under_its_own_stem() {
    let dir = tempdir().unwrap();
    let spec = PresentSpec {
        max_lines: 2,
        max_bytes: 1024,
        overflow_dir: dir.path().to_path_buf(),
    };
    let truncator = TextTruncator::new(spec, "mcp__web");

    for expected_file in ["mcp__web-1.txt", "mcp__web-2.txt"] {
        let body = "line 1\nline 2\nline 3\nline 4";
        let cut = truncator.truncate(body.into());
        let spilled = dir.path().join(expected_file);
        assert_eq!(cut.overflow_file.as_deref(), Some(spilled.as_path()));
        assert!(cut.truncated);
        assert_eq!(std::fs::read_to_string(&spilled).unwrap(), body);
        assert!(
            cut.text
                .starts_with("line 1\nline 2\n--- output truncated (4 lines, ")
        );
        assert!(
            cut.text
                .contains(&format!("Full output: {}\n", spilled.display())),
            "{}",
            cut.text
        );
        assert!(!cut.text.contains("line 3"));
    }
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

    let head: String = (1..=200).map(|i| format!("[find]\terr {i}\n")).collect();
    let p = path.display();
    assert_eq!(
        r.output,
        format!(
            "ok\n[stderr] {}\n--- stderr truncated (5000 lines, {}) ---\n\
             Full stderr: {p}\n\
             Explore: cat {p} | grep\n\
             cat {p} | tail -n 100\n\
             [exit:1 | 3ms]",
            head.trim_end_matches('\n'),
            human_size(big.len() as u64),
        )
    );
    assert_eq!(r.stdout_raw, "ok\n");
    assert_eq!(r.stderr_raw, head);
}

#[test]
fn spilling_past_the_file_cap_keeps_only_the_newest_spills() {
    let dir = tempdir().unwrap();
    let spec = PresentSpec {
        max_lines: 1,
        ..spec_in(dir.path())
    };
    let truncator = TextTruncator::new(spec, "cmd");
    let total = MAX_SPILL_FILES + 5;

    for _ in 0..total {
        let cut = truncator.truncate("a\nb\n".into());
        assert!(cut.overflow_file.as_deref().is_some_and(Path::exists));
    }

    let mut expected: Vec<String> = (6..=total).map(|n| format!("cmd-{n}.txt")).collect();
    expected.sort();
    assert_eq!(sorted_file_names(dir.path()), expected);
}

#[test]
fn is_spill_file_name_accepts_only_stem_dash_counter_dot_txt() {
    let cases = [
        ("cmd-1.txt", true),
        ("mcp-files-12.txt", true),
        ("cmd-007.txt", true),
        ("cmd-18446744073709551615.txt", true),
        ("cmd-18446744073709551616.txt", false),
        ("notes.txt", false),
        ("cmd1.txt", false),
        ("cmd-.txt", false),
        ("-3.txt", false),
        ("cmd-+1.txt", false),
        ("cmd-x.txt", false),
        ("cmd-1.md", false),
        ("cmd-1.txt.bak", false),
    ];
    for (name, expected) in cases {
        assert_eq!(is_spill_file_name(name), expected, "{name}");
    }
}

/// Age decides across stems whose counters are unrelated, and files not
/// named like spills are never touched.
#[test]
fn prune_spills_evicts_the_oldest_until_under_the_byte_cap() {
    let dir = tempdir().unwrap();
    write_file_aged(dir.path(), "notes.txt", 500, 50);
    write_file_aged(dir.path(), "mcp-web-9.txt", 100, 40);
    write_file_aged(dir.path(), "cmd-1.txt", 100, 30);
    write_file_aged(dir.path(), "mcp-web-10.txt", 100, 20);
    let newest = write_file_aged(dir.path(), "cmd-2.txt", 100, 0);
    let limits = SpillLimits {
        max_files: 10,
        max_bytes: 300,
    };

    prune_spills(dir.path(), &newest, 100, limits);

    assert_eq!(
        sorted_file_names(dir.path()),
        ["cmd-1.txt", "cmd-2.txt", "mcp-web-10.txt", "notes.txt"]
    );
}

#[test]
fn prune_spills_keeps_the_new_spill_even_when_it_alone_exceeds_the_caps() {
    let dir = tempdir().unwrap();
    write_file_aged(dir.path(), "cmd-1.txt", 10, 0);
    let newest = write_file_aged(dir.path(), "cmd-2.txt", 1000, 60);
    let limits = SpillLimits {
        max_files: 1,
        max_bytes: 100,
    };

    prune_spills(dir.path(), &newest, 1000, limits);

    assert_eq!(sorted_file_names(dir.path()), ["cmd-2.txt"]);
}
