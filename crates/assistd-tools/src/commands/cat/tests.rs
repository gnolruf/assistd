use std::time::Duration;

use tempfile::{TempDir, tempdir};

use super::*;
use crate::commands::{FILE_READ_MAX, hold_fifo_open, make_fifo};
use crate::fixtures::PNG_BYTES;

async fn run_cat(args: &[&str], stdin: Option<&[u8]>) -> CommandOutput {
    CatCommand
        .run(CommandInput {
            args: args.iter().map(ToString::to_string).collect(),
            stdin: stdin.map(<[u8]>::to_vec),
        })
        .await
}

fn write_file(dir: &TempDir, name: &str, bytes: &[u8]) -> String {
    let path = dir.path().join(name);
    std::fs::write(&path, bytes).unwrap();
    path.to_string_lossy().into_owned()
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
async fn cat_refuses_a_device_file() {
    let out = run_cat(&["/dev/null"], None).await;
    assert_eq!(out.exit_code, 1);
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        "[error] cat: /dev/null: not a regular file (device, pipe, or socket). Check: ls -l /dev/null\n"
    );
}

async fn assert_cat_refuses_fifo_promptly(fifo: &str) {
    for args in [vec![fifo], vec!["-b", fifo]] {
        let out = tokio::time::timeout(Duration::from_secs(5), run_cat(&args, None))
            .await
            .unwrap_or_else(|_| panic!("cat {args:?} blocked on a FIFO"));
        assert_eq!(out.exit_code, 1, "{args:?}");
        assert_eq!(
            String::from_utf8_lossy(&out.stderr),
            format!(
                "[error] cat: {fifo}: not a regular file (device, pipe, or socket). \
                 Check: ls -l {fifo}\n"
            ),
            "{args:?}"
        );
    }
}

#[tokio::test]
async fn cat_refuses_a_fifo_with_no_writer() {
    let dir = tempdir().unwrap();
    let fifo = make_fifo(dir.path());
    assert_cat_refuses_fifo_promptly(&fifo).await;
}

#[tokio::test]
async fn cat_refuses_a_fifo_held_open_by_a_writer() {
    let dir = tempdir().unwrap();
    let fifo = make_fifo(dir.path());
    let _ends = hold_fifo_open(&fifo);
    assert_cat_refuses_fifo_promptly(&fifo).await;
}

fn oversized_file() -> (TempDir, String) {
    let dir = tempdir().unwrap();
    let path = dir.path().join("huge.log");
    std::fs::File::create(&path)
        .unwrap()
        .set_len(FILE_READ_MAX + 1)
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
async fn cat_stops_reading_files_once_output_passes_output_max() {
    let dir = tempdir().unwrap();
    let half = vec![b'a'; OUTPUT_MAX / 2 + 1];
    let first = write_file(&dir, "first.txt", &half);
    let second = write_file(&dir, "second.txt", &half);
    let never_read = dir.path().join("missing.txt");
    let out = run_cat(&[&first, &second, &never_read.to_string_lossy()], None).await;
    assert_eq!(out.exit_code, 0, "{}", String::from_utf8_lossy(&out.stderr));
    assert_eq!(out.stdout.len(), 2 * half.len());
}
