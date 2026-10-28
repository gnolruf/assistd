//! Text to IPA phonemes for Piper voices, running `espeak-ng` as a separate program once per
//! clause, split the way piper-phonemize splits text.

use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, ChildStdout, Command, Output, Stdio};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::thread;
use std::time::Duration;

use assistd_utils::program::find_program;
use parking_lot::Mutex;

const PROGRAM: &str = "espeak-ng";

/// How long one clause may take before `espeak-ng` is killed.
const CLAUSE_DEADLINE: Duration = Duration::from_secs(5);

/// Punctuation that ends a clause when followed by whitespace or the end.
const CLAUSE_PUNCTUATION: &[char] = &[',', '.', ';', ':', '!', '?'];

/// Runs `espeak-ng` for one voice, such as `en-us`.
#[derive(Debug)]
pub struct Phonemizer {
    program: PathBuf,
    voice: String,
}

/// A clause's text and the punctuation that closed it.
struct Clause<'a> {
    text: &'a str,
    end: ClauseEnd,
}

enum ClauseEnd {
    /// `,`, `;` or `:`; the sentence continues after it.
    Pause(char),
    /// `.`, `!` or `?`, kept as the sentence's last phoneme.
    Stop(char),
    /// An ellipsis or the end of the text, which add no phoneme.
    Silent,
}

impl Phonemizer {
    /// Find `espeak-ng` on `$PATH` for `voice`.
    pub fn new(voice: String) -> Result<Self, String> {
        let path_env = std::env::var_os("PATH").unwrap_or_default();
        let program = find_program(PROGRAM.as_ref(), &path_env)
            .map_err(|reason| format!("cannot run {PROGRAM} ({reason}); install espeak-ng"))?;
        Ok(Self { program, voice })
    }

    /// Phonemes of `text`, one list of codepoints per sentence, with the
    /// clause punctuation Piper voices were trained on.
    pub fn sentences(&self, text: &str) -> Result<Vec<Vec<i32>>, String> {
        let mut sentences = Vec::new();
        let mut current = String::new();
        for clause in clauses(text) {
            current.push_str(&self.clause_phonemes(clause.text)?);
            match clause.end {
                ClauseEnd::Pause(mark) => {
                    current.push(mark);
                    current.push(' ');
                }
                ClauseEnd::Stop(mark) => {
                    current.push(mark);
                    push_sentence(&mut sentences, &mut current);
                }
                ClauseEnd::Silent => push_sentence(&mut sentences, &mut current),
            }
        }
        Ok(sentences)
    }

    /// espeak-ng prints a clause's phonemes one line per inner break.
    fn clause_phonemes(&self, clause: &str) -> Result<String, String> {
        if clause.trim().is_empty() {
            return Ok(String::new());
        }
        let child = Command::new(&self.program)
            .args(["-q", "--ipa", "-v", &self.voice])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|err| format!("spawn {}: {err}", self.program.display()))?;
        let output = run_with_deadline(child, clause.replace("[[", "").as_bytes())?;
        if !output.status.success() {
            return Err(format!(
                "{PROGRAM} exited with {}: {}",
                output.status,
                String::from_utf8_lossy(&output.stderr).trim()
            ));
        }
        let phonemes = String::from_utf8_lossy(&output.stdout);
        Ok(phonemes
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>()
            .join(" "))
    }
}

/// Feed `input` to `child` and collect its output, killing it if it runs past
/// [`CLAUSE_DEADLINE`]. The child is always reaped.
fn run_with_deadline(mut child: Child, input: &[u8]) -> Result<Output, String> {
    let stdin = child.stdin.take().expect("stdin piped");
    let stdout = child.stdout.take().expect("stdout piped");
    let mut stderr = child.stderr.take().expect("stderr piped");
    let child = Mutex::new(child);
    let (finished_tx, finished_rx) = mpsc::channel::<()>();
    let (exchanged, stderr_bytes, timed_out) = thread::scope(|scope| {
        let child = &child;
        let watchdog = scope.spawn(move || kill_after_deadline(child, &finished_rx));
        let stderr_reader = scope.spawn(move || {
            let mut bytes = Vec::new();
            let _ = stderr.read_to_end(&mut bytes);
            bytes
        });
        let exchanged = exchange(stdin, stdout, input);
        let stderr_bytes = stderr_reader.join().unwrap_or_default();
        drop(finished_tx);
        (exchanged, stderr_bytes, watchdog.join().unwrap_or(true))
    });
    let status = child
        .into_inner()
        .wait()
        .map_err(|err| format!("wait for {PROGRAM}: {err}"))?;
    if timed_out {
        return Err(format!(
            "{PROGRAM} did not finish within {}s and was killed",
            CLAUSE_DEADLINE.as_secs()
        ));
    }
    Ok(Output {
        status,
        stdout: exchanged?,
        stderr: stderr_bytes,
    })
}

/// Write `input` and close stdin, then read stdout to its end.
fn exchange(
    mut stdin: ChildStdin,
    mut stdout: ChildStdout,
    input: &[u8],
) -> Result<Vec<u8>, String> {
    let written = stdin.write_all(input);
    drop(stdin);
    let mut bytes = Vec::new();
    let read = stdout.read_to_end(&mut bytes);
    written.map_err(|err| format!("write to {PROGRAM}: {err}"))?;
    read.map_err(|err| format!("read from {PROGRAM}: {err}"))?;
    Ok(bytes)
}

/// Kill `child` unless `finished` hangs up within [`CLAUSE_DEADLINE`]; whether
/// it was killed.
fn kill_after_deadline(child: &Mutex<Child>, finished: &mpsc::Receiver<()>) -> bool {
    if !matches!(
        finished.recv_timeout(CLAUSE_DEADLINE),
        Err(RecvTimeoutError::Timeout)
    ) {
        return false;
    }
    let _ = child.lock().kill();
    true
}

fn push_sentence(sentences: &mut Vec<Vec<i32>>, current: &mut String) {
    let sentence = std::mem::take(current);
    let sentence = sentence.trim();
    if !sentence.is_empty() {
        sentences.push(sentence.chars().map(codepoint).collect());
    }
}

fn codepoint(phoneme: char) -> i32 {
    i32::try_from(u32::from(phoneme)).unwrap_or(0)
}

/// Split `text` at runs of clause punctuation followed by whitespace or the
/// end, so `3.50` and `e.g.x` stay whole.
fn clauses(text: &str) -> Vec<Clause<'_>> {
    let mut clauses = Vec::new();
    let mut start = 0;
    let mut chars = text.char_indices().peekable();
    while let Some((at, c)) = chars.next() {
        if !CLAUSE_PUNCTUATION.contains(&c) {
            continue;
        }
        let mut run_end = at + c.len_utf8();
        while let Some(&(next_at, next)) = chars.peek()
            && CLAUSE_PUNCTUATION.contains(&next)
        {
            run_end = next_at + next.len_utf8();
            chars.next();
        }
        if chars.peek().is_none_or(|&(_, next)| next.is_whitespace()) {
            clauses.push(Clause {
                text: &text[start..at],
                end: clause_end(&text[at..run_end]),
            });
            start = run_end;
        }
    }
    clauses.push(Clause {
        text: &text[start..],
        end: ClauseEnd::Silent,
    });
    clauses
}

fn clause_end(run: &str) -> ClauseEnd {
    if run.contains('?') {
        ClauseEnd::Stop('?')
    } else if run.contains('!') {
        ClauseEnd::Stop('!')
    } else if run.starts_with("..") {
        ClauseEnd::Silent
    } else {
        match run.chars().next() {
            Some('.') => ClauseEnd::Stop('.'),
            Some(mark) => ClauseEnd::Pause(mark),
            None => ClauseEnd::Silent,
        }
    }
}
