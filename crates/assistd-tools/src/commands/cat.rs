//! `cat`: print text files or stdin, refusing binary files and asking before sensitive ones.

use std::sync::Arc;

use assistd_utils::binary::{SNIFF_LEN, sniff_binary};
use assistd_utils::text::human_size;
use async_trait::async_trait;

use crate::chain::OUTPUT_MAX;
use crate::chain::Word;
use crate::command::{Command, CommandError, CommandInput, CommandOutput, Hint, Usage};
use crate::commands::{read_regular_file, read_regular_head, reads_of};
use crate::policy::{ConfirmationGate, FileScope, ReadPolicy, SandboxInfo};

/// `cat [-bn] [FILE]...`: concatenate files, or echo stdin if none are
/// given. Binary files are rejected so their bytes stay out of the context,
/// and a sensitive file's content is read only once the user agrees.
#[derive(Debug)]
pub struct CatCommand {
    policy: ReadPolicy,
}

impl CatCommand {
    /// A `cat` command that asks `gate` before reading a file `sandbox`'s path guards
    /// call sensitive.
    pub fn new(sandbox: Arc<SandboxInfo>, gate: Arc<dyn ConfirmationGate>) -> Self {
        Self {
            policy: ReadPolicy { sandbox, gate },
        }
    }
}

#[derive(Default)]
struct Flags {
    metadata_only: bool,
    number_lines: bool,
}

#[async_trait]
impl Command for CatCommand {
    fn name(&self) -> &'static str {
        "cat"
    }

    fn summary(&self) -> &'static str {
        "read text files or stdin (-n numbers lines); binary rejected"
    }

    fn usage(&self) -> Usage {
        Usage::new(
            "[-bn] [FILE]...",
            "Concatenate files, or echo stdin if no files given. Binary files \
         are rejected so their raw bytes don't pollute the model's context. \
         Credentials and files in $HOME only their owner may read ask the user \
         first; a refusal exits 126.\n\
         \n\
         Flags:\n  \
           -b  print metadata (mime, size) instead of content (safe for binary files)\n  \
           -n  prefix each output line with its 1-based line number\n\
         \n\
         For image files, use `see PATH` to attach them as a vision input.\n",
        )
    }

    async fn run(&self, input: CommandInput) -> Result<CommandOutput, CommandError> {
        let (flags, files) = parse_flags(&input.args)
            .map_err(|what| CommandError::usage(what, "cat -b FILE or cat -n FILE"))?;

        if files.is_empty() {
            let Some(stdin) = input.stdin else {
                return Ok(CommandOutput::usage(self.help()));
            };
            if flags.metadata_only {
                return Ok(CommandOutput::ok(describe(
                    &stdin,
                    stdin.len() as u64,
                    None,
                )));
            }
            return Ok(CommandOutput::ok(number_if(stdin, &flags)));
        }

        if !flags.metadata_only {
            self.policy
                .authorize(&input.turn, "cat", &input.scope, &files)
                .await?;
        }
        let mut out = Vec::new();
        let mut read = Vec::new();
        for path in files {
            let chunk = if flags.metadata_only {
                read_metadata(&input.scope, path).await?
            } else {
                read.push(path.as_str());
                read_text(&input.scope, path).await?
            };
            out.extend_from_slice(&chunk);
            if out.len() > OUTPUT_MAX {
                break;
            }
        }
        let accesses = reads_of(&input.scope, read).await;
        Ok(CommandOutput::ok(number_if(out, &flags)).touching(accesses))
    }
}

async fn read_metadata(scope: &FileScope, path: &Word) -> Result<Vec<u8>, CommandError> {
    let (head, size) = read_regular_head(scope, path.as_str(), SNIFF_LEN as u64)
        .await
        .map_err(|e| CommandError::io(path, &e))?;
    Ok(describe(&head, size, Some(path.as_str())))
}

async fn read_text(scope: &FileScope, path: &Word) -> Result<Vec<u8>, CommandError> {
    let bytes = read_regular_file(scope, path.as_str())
        .await
        .map_err(|e| CommandError::io(path, &e))?;
    let Some(mime) = sniff_binary(&bytes) else {
        return Ok(bytes);
    };
    let size = human_size(bytes.len() as u64);
    Err(if mime.starts_with("image/") {
        CommandError::failed(
            format_args!("binary image file ({size}): {path}"),
            Hint::Use,
            format_args!("see {path}"),
        )
    } else {
        CommandError::failed(
            format_args!("binary {mime} file ({size}): {path}"),
            Hint::Use,
            format_args!("cat -b {path}"),
        )
    })
}

fn number_if(bytes: Vec<u8>, flags: &Flags) -> Vec<u8> {
    if !flags.number_lines {
        return bytes;
    }
    let mut out = Vec::with_capacity(bytes.len() + bytes.len() / 16);
    for (index, line) in bytes.split_inclusive(|b| *b == b'\n').enumerate() {
        out.extend_from_slice(format!("{}\t", index + 1).as_bytes());
        out.extend_from_slice(line);
    }
    out
}

fn parse_flags(argv: &[Word]) -> Result<(Flags, Vec<&Word>), String> {
    let mut flags = Flags::default();
    let mut files = Vec::with_capacity(argv.len());
    for arg in argv {
        let Some(letters) = arg.text.strip_prefix('-').filter(|l| !l.is_empty()) else {
            files.push(arg);
            continue;
        };
        for ch in letters.chars() {
            match ch {
                'b' => flags.metadata_only = true,
                'n' => flags.number_lines = true,
                other => return Err(format!("unknown flag '-{other}'")),
            }
        }
    }
    Ok((flags, files))
}

fn describe(head: &[u8], size: u64, path: Option<&str>) -> Vec<u8> {
    let mime = infer::get(head).map_or_else(
        || {
            if sniff_binary(head).is_some() {
                "application/octet-stream".into()
            } else {
                "text/plain".into()
            }
        },
        |t| t.mime_type().to_string(),
    );
    let prefix = path.map_or_default(|p| format!("{p}: "));
    format!("{prefix}{mime}\n{prefix}{size} bytes\n").into_bytes()
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use tempfile::{TempDir, tempdir};

    use super::*;
    use crate::TurnContext;
    use crate::Unattended;
    use crate::command::run_rendered;
    use crate::commands::{
        FILE_READ_MAX, RecordingGate, hold_fifo_open, make_fifo, test_cat, test_sandbox,
    };
    use crate::fixtures::PNG_BYTES;

    async fn run_cat(args: &[&str], stdin: Option<&[u8]>) -> CommandOutput {
        run_cat_as(&test_cat(), args, stdin).await
    }

    async fn run_cat_as(cat: &CatCommand, args: &[&str], stdin: Option<&[u8]>) -> CommandOutput {
        run_rendered(
            cat,
            CommandInput {
                args: args.iter().copied().map(Word::bare).collect(),
                stdin: stdin.map(<[u8]>::to_vec),
                scope: FileScope::Host,
                turn: TurnContext::default(),
            },
        )
        .await
    }

    fn write_file(dir: &TempDir, name: &str, bytes: &[u8]) -> String {
        let path = dir.path().join(name);
        std::fs::write(&path, bytes).unwrap();
        path.to_string_lossy().into_owned()
    }

    #[tokio::test]
    async fn cat_asks_before_a_credentials_path_and_a_refusal_reads_nothing() {
        let home = std::env::var("HOME").expect("HOME is set");
        let key = format!("{home}/.ssh/assistd-test-key-that-does-not-exist");
        let dir = tempdir().unwrap();
        let ordinary = write_file(&dir, "notes.txt", b"notes\n");
        let gate = RecordingGate::new(false);
        let cat = CatCommand::new(test_sandbox(), gate.clone());

        let read = run_cat_as(&cat, &[&ordinary], None).await;
        assert_eq!(read.stdout, b"notes\n");
        assert!(gate.prompts().is_empty(), "{:?}", gate.prompts());

        let refused = run_cat_as(&cat, &[&ordinary, &key], None).await;
        assert_eq!(refused.exit_code, 126, "{refused:?}");
        assert!(refused.stdout.is_empty(), "{refused:?}");
        let requests = gate.requests();
        let [request] = requests.as_slice() else {
            panic!("expected one prompt: {requests:?}");
        };
        assert_eq!(request.tool, "cat");
        assert_eq!(request.script, format!("cat {ordinary:?} {key:?}"));
        assert_eq!(request.unattended, Unattended::Never);

        let described = run_cat_as(&cat, &["-b", &key], None).await;
        assert_eq!(described.exit_code, 1, "{described:?}");
        assert_eq!(gate.prompts().len(), 1, "metadata alone does not ask");
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
}
