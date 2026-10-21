//! The wake phrase and the prefix match that decides whether a transcript
//! is addressed to assistd.

use assistd_config::UnitInterval;

/// Phrase used until the user enrolls their own.
pub const DEFAULT_WAKE_PHRASE: &str = "hey computer";

/// The wake phrase plus the spellings Whisper has been heard to produce
/// for it. Every word of a transcript's leading words must reach
/// `min_similarity` against the phrase or a variant for it to match.
#[derive(Debug, Clone)]
pub struct WakePhrase {
    spellings: Vec<Vec<String>>,
    phrase: String,
    min_similarity: f32,
}

impl WakePhrase {
    /// Spellings that normalize to no words are ignored.
    pub fn new(phrase: &str, variants: &[String], min_similarity: UnitInterval) -> Self {
        let spellings = std::iter::once(phrase)
            .chain(variants.iter().map(String::as_str))
            .map(normalized_words)
            .filter(|words| !words.is_empty())
            .collect();
        Self {
            spellings,
            phrase: phrase.to_string(),
            min_similarity: min_similarity.get(),
        }
    }

    /// [`DEFAULT_WAKE_PHRASE`] with no learned variants.
    pub fn builtin(min_similarity: UnitInterval) -> Self {
        Self::new(DEFAULT_WAKE_PHRASE, &[], min_similarity)
    }

    /// The phrase as the user enrolled it, for biasing Whisper.
    pub fn phrase(&self) -> &str {
        &self.phrase
    }

    /// When `transcript` opens with the phrase or a variant, the text
    /// after it, trimmed of leading punctuation and whitespace; empty when
    /// the phrase was said alone. Case and punctuation are ignored.
    pub fn match_prefix<'a>(&self, transcript: &'a str) -> Option<&'a str> {
        let words = word_spans(transcript);
        let end = self
            .spellings
            .iter()
            .filter(|spelling| words.len() >= spelling.len())
            .find(|spelling| {
                spelling.iter().zip(&words).all(|(expected, (heard, _))| {
                    similarity(expected, heard) >= self.min_similarity
                })
            })
            .map(|spelling| words[spelling.len() - 1].1)?;
        Some(transcript[end..].trim_start_matches(|c: char| !c.is_alphanumeric()))
    }
}

/// Lowercase alphanumeric words, apostrophes dropped, other punctuation
/// treated as a separator.
pub(super) fn normalized_words(text: &str) -> Vec<String> {
    word_spans(text).into_iter().map(|(word, _)| word).collect()
}

/// Each normalized word with the byte offset just past it in `text`.
fn word_spans(text: &str) -> Vec<(String, usize)> {
    let mut spans = Vec::new();
    let mut word = String::new();
    for (offset, ch) in text.char_indices() {
        if ch.is_alphanumeric() {
            word.extend(ch.to_lowercase());
        } else if ch != '\'' && ch != '’' && !word.is_empty() {
            spans.push((std::mem::take(&mut word), offset));
        }
    }
    if !word.is_empty() {
        spans.push((word, text.len()));
    }
    spans
}

/// `1 - levenshtein / longer length`, in `0.0..=1.0`.
pub(super) fn similarity(expected: &str, heard: &str) -> f32 {
    let expected: Vec<char> = expected.chars().collect();
    let heard: Vec<char> = heard.chars().collect();
    let longer = expected.len().max(heard.len());
    if longer == 0 {
        return 1.0;
    }
    let distance = levenshtein(&expected, &heard);
    1.0 - distance as f32 / longer as f32
}

fn levenshtein(left: &[char], right: &[char]) -> usize {
    let mut previous: Vec<usize> = (0..=right.len()).collect();
    let mut current = vec![0; right.len() + 1];
    for (i, left_char) in left.iter().enumerate() {
        current[0] = i + 1;
        for (j, right_char) in right.iter().enumerate() {
            let substitution = previous[j] + usize::from(left_char != right_char);
            current[j + 1] = substitution.min(previous[j + 1] + 1).min(current[j] + 1);
        }
        std::mem::swap(&mut previous, &mut current);
    }
    previous[right.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_a_leading_phrase_or_variant_matches() {
        let wake = WakePhrase::new(
            DEFAULT_WAKE_PHRASE,
            &["a computer".to_string()],
            UnitInterval::new(0.8).unwrap(),
        );
        let cases = [
            ("Hey, computer. What's the time?", Some("What's the time?")),
            ("hey computer", Some("")),
            ("Hey Computer!", Some("")),
            ("a computer what's the weather", Some("what's the weather")),
            ("Hey, computers, open firefox", Some("open firefox")),
            ("my computer is slow", None),
            ("and then I said hey computer", None),
            ("hey", None),
            ("", None),
        ];
        for (transcript, expected) in cases {
            assert_eq!(wake.match_prefix(transcript), expected, "{transcript:?}");
        }
    }
}
