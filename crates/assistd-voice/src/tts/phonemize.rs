//! Text to IPA phonemes for Piper voices, by running `espeak-ng` once per
//! clause the way piper-phonemize splits text, so espeak-ng stays a separate
//! program.

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

use assistd_utils::program::find_program;

const PROGRAM: &str = "espeak-ng";

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
        let mut child = Command::new(&self.program)
            .args(["-q", "--ipa", "-v", &self.voice])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|err| format!("spawn {}: {err}", self.program.display()))?;
        let mut stdin = child.stdin.take().expect("stdin piped");
        stdin
            .write_all(clause.as_bytes())
            .map_err(|err| format!("write to {PROGRAM}: {err}"))?;
        drop(stdin);
        let output = child
            .wait_with_output()
            .map_err(|err| format!("wait for {PROGRAM}: {err}"))?;
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
