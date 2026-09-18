use super::*;
use tempfile::tempdir;

// Minimal valid 1x1 PNG; `infer` recognizes this as `image/png`.
const PNG_BYTES: &[u8] = &[
    0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
    0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x06, 0x00, 0x00, 0x00, 0x1F, 0x15, 0xC4,
    0x89, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9C, 0x63, 0x00, 0x01, 0x00, 0x00,
    0x05, 0x00, 0x01, 0x0D, 0x0A, 0x2D, 0xB4, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE,
    0x42, 0x60, 0x82,
];

#[tokio::test]
async fn cat_no_args_and_no_stdin_emits_usage() {
    let out = CatCommand
        .run(CommandInput {
            args: Vec::new(),
            stdin: None,
        })
        .await
        .unwrap();
    assert_eq!(out.exit_code, 2);
    assert!(out.stdout.starts_with(b"usage: cat"), "{out:?}");
}

#[tokio::test]
async fn cat_unknown_flag_errors() {
    let out = CatCommand
        .run(CommandInput {
            args: vec!["-q".into(), "notes.md".into()],
            stdin: None,
        })
        .await
        .unwrap();
    assert_eq!(out.exit_code, 2);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("[error] cat: unknown flag '-q'"),
        "{stderr}"
    );
    assert!(stderr.contains("Use: "), "{stderr}");
}

#[tokio::test]
async fn cat_reads_text_file() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("hello.txt");
    tokio::fs::write(&path, b"hello world\n").await.unwrap();
    let out = CatCommand
        .run(CommandInput {
            args: vec![path.to_string_lossy().into_owned()],
            stdin: None,
        })
        .await
        .unwrap();
    assert_eq!(out.stdout, b"hello world\n");
    assert_eq!(out.exit_code, 0);
}

#[tokio::test]
async fn cat_concatenates_multiple_text_files() {
    let dir = tempdir().unwrap();
    let a = dir.path().join("a.txt");
    let b = dir.path().join("b.txt");
    tokio::fs::write(&a, b"A").await.unwrap();
    tokio::fs::write(&b, b"B").await.unwrap();
    let out = CatCommand
        .run(CommandInput {
            args: vec![
                a.to_string_lossy().into_owned(),
                b.to_string_lossy().into_owned(),
            ],
            stdin: None,
        })
        .await
        .unwrap();
    assert_eq!(out.stdout, b"AB");
}

#[tokio::test]
async fn cat_missing_file_exits_1() {
    let out = CatCommand
        .run(CommandInput {
            args: vec!["/nonexistent/path/xyz".into()],
            stdin: None,
        })
        .await
        .unwrap();
    assert_eq!(out.exit_code, 1);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("[error] cat: file not found: /nonexistent/path/xyz"),
        "{stderr}"
    );
    assert!(stderr.contains("Use: ls to check the path"), "{stderr}");
}

#[tokio::test]
async fn cat_n_numbers_lines_across_files() {
    let dir = tempdir().unwrap();
    let a = dir.path().join("a.txt");
    let b = dir.path().join("b.txt");
    std::fs::write(&a, b"one\ntwo\n").unwrap();
    std::fs::write(&b, b"three\n").unwrap();
    let out = CatCommand
        .run(CommandInput {
            args: vec![
                "-n".into(),
                a.to_string_lossy().into_owned(),
                b.to_string_lossy().into_owned(),
            ],
            stdin: None,
        })
        .await
        .unwrap();
    assert_eq!(out.exit_code, 0);
    assert_eq!(out.stdout, b"1\tone\n2\ttwo\n3\tthree\n");
}

#[tokio::test]
async fn cat_n_numbers_stdin() {
    let out = CatCommand
        .run(CommandInput {
            args: vec!["-n".into()],
            stdin: Some(b"alpha\nbeta\n".to_vec()),
        })
        .await
        .unwrap();
    assert_eq!(out.stdout, b"1\talpha\n2\tbeta\n");
}

#[tokio::test]
async fn cat_no_args_echoes_stdin() {
    let out = CatCommand
        .run(CommandInput {
            args: Vec::new(),
            stdin: Some(b"from stdin".to_vec()),
        })
        .await
        .unwrap();
    assert_eq!(out.stdout, b"from stdin");
}

#[tokio::test]
async fn cat_rejects_binary_image_file() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("photo.png");
    tokio::fs::write(&path, PNG_BYTES).await.unwrap();
    let out = CatCommand
        .run(CommandInput {
            args: vec![path.to_string_lossy().into_owned()],
            stdin: None,
        })
        .await
        .unwrap();
    assert_eq!(out.exit_code, 1);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("[error] cat: binary image file"),
        "{stderr}"
    );
    assert!(stderr.contains("Use: see "), "{stderr}");
    assert!(out.stdout.is_empty(), "must not leak bytes on rejection");
}

#[tokio::test]
async fn cat_rejects_binary_with_nul_bytes() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("garbled.bin");
    let mut bytes = vec![b'x'; 200];
    bytes[100] = 0;
    tokio::fs::write(&path, &bytes).await.unwrap();
    let out = CatCommand
        .run(CommandInput {
            args: vec![path.to_string_lossy().into_owned()],
            stdin: None,
        })
        .await
        .unwrap();
    assert_eq!(out.exit_code, 1);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("[error] cat: binary application/octet-stream"),
        "{stderr}"
    );
    assert!(stderr.contains("Use: cat -b "), "{stderr}");
}

#[tokio::test]
async fn cat_b_prints_metadata_for_binary() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("photo.png");
    tokio::fs::write(&path, PNG_BYTES).await.unwrap();
    let out = CatCommand
        .run(CommandInput {
            args: vec!["-b".into(), path.to_string_lossy().into_owned()],
            stdin: None,
        })
        .await
        .unwrap();
    assert_eq!(out.exit_code, 0);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("image/png"), "{stdout}");
    assert!(
        stdout.contains(&format!("{} bytes", PNG_BYTES.len())),
        "{stdout}"
    );
}

#[tokio::test]
async fn cat_b_works_on_text_file() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("notes.txt");
    tokio::fs::write(&path, b"some text\n").await.unwrap();
    let out = CatCommand
        .run(CommandInput {
            args: vec!["-b".into(), path.to_string_lossy().into_owned()],
            stdin: None,
        })
        .await
        .unwrap();
    assert_eq!(out.exit_code, 0);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("text/plain"), "{stdout}");
    assert!(stdout.contains("10 bytes"), "{stdout}");
}

#[test]
fn human_size_formats_expected_ranges() {
    assert_eq!(human_size(0), "0B");
    assert_eq!(human_size(500), "500B");
    assert_eq!(human_size(2048), "2KB");
    assert_eq!(human_size(1024 * 1024 * 3), "3.0MB");
}
