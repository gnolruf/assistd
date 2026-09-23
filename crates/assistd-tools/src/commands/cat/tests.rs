use super::*;
use crate::fixtures::PNG_BYTES;
use tempfile::tempdir;

async fn run_cat(args: &[&str], stdin: Option<&[u8]>) -> CommandOutput {
    CatCommand
        .run(CommandInput {
            args: args.iter().map(|s| s.to_string()).collect(),
            stdin: stdin.map(<[u8]>::to_vec),
        })
        .await
}

fn write_file(dir: &tempfile::TempDir, name: &str, bytes: &[u8]) -> String {
    let path = dir.path().join(name);
    std::fs::write(&path, bytes).unwrap();
    path.to_string_lossy().into_owned()
}

#[tokio::test]
async fn cat_no_args_and_no_stdin_emits_usage() {
    let out = run_cat(&[], None).await;
    assert_eq!(out.exit_code, 2);
    assert!(out.stdout.starts_with(b"usage: cat"), "{out:?}");
}

#[tokio::test]
async fn cat_unknown_flag_errors() {
    let out = run_cat(&["-q", "notes.md"], None).await;
    assert_eq!(out.exit_code, 2);
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        "[error] cat: unknown flag '-q'. Use: cat -b FILE or cat -n FILE\n"
    );
}

#[tokio::test]
async fn cat_reads_text_file() {
    let dir = tempdir().unwrap();
    let path = write_file(&dir, "hello.txt", b"hello world\n");
    let out = run_cat(&[&path], None).await;
    assert_eq!(out.exit_code, 0);
    assert_eq!(out.stdout, b"hello world\n");
}

#[tokio::test]
async fn cat_concatenates_multiple_text_files() {
    let dir = tempdir().unwrap();
    let a = write_file(&dir, "a.txt", b"A");
    let b = write_file(&dir, "b.txt", b"B");
    let out = run_cat(&[&a, &b], None).await;
    assert_eq!(out.stdout, b"AB");
}

#[tokio::test]
async fn cat_missing_file_exits_1() {
    let out = run_cat(&["/nonexistent/path/xyz"], None).await;
    assert_eq!(out.exit_code, 1);
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        "[error] cat: file not found: /nonexistent/path/xyz. Use: ls /nonexistent/path to see what is there\n"
    );
}

#[tokio::test]
async fn cat_n_numbers_lines_across_files() {
    let dir = tempdir().unwrap();
    let a = write_file(&dir, "a.txt", b"one\ntwo\n");
    let b = write_file(&dir, "b.txt", b"three\n");
    let out = run_cat(&["-n", &a, &b], None).await;
    assert_eq!(out.exit_code, 0);
    assert_eq!(out.stdout, b"1\tone\n2\ttwo\n3\tthree\n");
}

#[tokio::test]
async fn cat_n_numbers_stdin() {
    let out = run_cat(&["-n"], Some(b"alpha\nbeta\n")).await;
    assert_eq!(out.stdout, b"1\talpha\n2\tbeta\n");
}

#[tokio::test]
async fn cat_no_args_echoes_stdin() {
    let out = run_cat(&[], Some(b"from stdin")).await;
    assert_eq!(out.stdout, b"from stdin");
}

#[tokio::test]
async fn cat_rejects_binary_image_file() {
    let dir = tempdir().unwrap();
    let path = write_file(&dir, "photo.png", PNG_BYTES);
    let out = run_cat(&[&path], None).await;
    assert_eq!(out.exit_code, 1);
    assert!(out.stdout.is_empty(), "must not leak bytes on rejection");
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        format!(
            "[error] cat: binary image file ({}B): {path}. Use: see {path}\n",
            PNG_BYTES.len()
        )
    );
}

#[tokio::test]
async fn cat_rejects_binary_with_nul_bytes() {
    let dir = tempdir().unwrap();
    let mut bytes = vec![b'x'; 200];
    bytes[100] = 0;
    let path = write_file(&dir, "garbled.bin", &bytes);
    let out = run_cat(&[&path], None).await;
    assert_eq!(out.exit_code, 1);
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        format!(
            "[error] cat: binary application/octet-stream file (200B): {path}. Use: cat -b {path}\n"
        )
    );
}

#[tokio::test]
async fn cat_b_prints_metadata_for_binary() {
    let dir = tempdir().unwrap();
    let path = write_file(&dir, "photo.png", PNG_BYTES);
    let out = run_cat(&["-b", &path], None).await;
    assert_eq!(out.exit_code, 0);
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        format!("{path}: image/png\n{path}: {} bytes\n", PNG_BYTES.len())
    );
}

#[tokio::test]
async fn cat_b_works_on_text_file() {
    let dir = tempdir().unwrap();
    let path = write_file(&dir, "notes.txt", b"some text\n");
    let out = run_cat(&["-b", &path], None).await;
    assert_eq!(out.exit_code, 0);
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        format!("{path}: text/plain\n{path}: 10 bytes\n")
    );
}

#[test]
fn human_size_formats_expected_ranges() {
    assert_eq!(human_size(0), "0B");
    assert_eq!(human_size(500), "500B");
    assert_eq!(human_size(2048), "2KB");
    assert_eq!(human_size(1024 * 1024 * 3), "3.0MB");
}

#[tokio::test]
async fn cat_refuses_a_device_file() {
    let out = run_cat(&["/dev/null"], None).await;
    assert_eq!(out.exit_code, 1);
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        "[error] cat: /dev/null: not a regular file (device, pipe, or socket). Check: ls -l /dev/null\n"
    );
}

fn oversized_file() -> (tempfile::TempDir, String) {
    let dir = tempdir().unwrap();
    let path = dir.path().join("huge.log");
    std::fs::File::create(&path)
        .unwrap()
        .set_len(crate::commands::FILE_READ_MAX + 1)
        .unwrap();
    let path = path.to_string_lossy().into_owned();
    (dir, path)
}

#[tokio::test]
async fn cat_refuses_a_file_over_the_read_limit() {
    let (_dir, path) = oversized_file();
    let out = run_cat(&[&path], None).await;
    assert_eq!(out.exit_code, 1);
    assert!(out.stdout.is_empty());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("read limit"), "{stderr}");
    assert!(stderr.contains(&format!("tail -n 200 {path}")), "{stderr}");
}

#[tokio::test]
async fn cat_b_reports_the_size_of_a_file_over_the_read_limit() {
    let (_dir, path) = oversized_file();
    let out = run_cat(&["-b", &path], None).await;
    assert_eq!(out.exit_code, 0);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let expected = format!("{} bytes", crate::commands::FILE_READ_MAX + 1);
    assert!(stdout.contains(&expected), "{stdout}");
}
