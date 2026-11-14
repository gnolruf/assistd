//! Small text formatting helpers, the classes of invisible characters that can
//! disguise or reorder text, and the chat-template tokens untrusted text must not carry.

use std::any::Any;
use std::borrow::Cow;

use serde_json::Value;

/// Chat-template control tokens, or the opening or closing shared by a family of them.
/// A model server that parses special tokens in message text reads these as turn
/// boundaries, tool calls or reasoning rather than as data.
const TEMPLATE_TOKENS: &[&str] = &[
    // ChatML, Llama 3, Phi, Harmony and the other `<|name|>` vocabularies.
    "<|",
    "|>",
    // DeepSeek, whose bars are U+FF5C.
    "<｜",
    "｜>",
    // Reasoning and Hermes-style tool calls.
    "<think>",
    "</think>",
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
    "<function=",
    "</function>",
    // SentencePiece sequence boundaries.
    "<s>",
    "</s>",
    "<bos>",
    "<eos>",
    // Gemma.
    "<start_of_turn>",
    "<end_of_turn>",
    "<start_function_call>",
    "<end_function_call>",
    "<start_function_response>",
    "<end_function_response>",
    // Llama 2 and Mistral.
    "[INST]",
    "[/INST]",
    "<<SYS>>",
    "<</SYS>>",
    "[SYSTEM_PROMPT]",
    "[/SYSTEM_PROMPT]",
    "[AVAILABLE_TOOLS]",
    "[/AVAILABLE_TOOLS]",
    "[TOOL_CALLS]",
    "[TOOL_RESULTS]",
    "[/TOOL_RESULTS]",
    "[TOOL_CONTENT]",
    "[CALL_ID]",
    "[ARGS]",
    "[THINK]",
    "[/THINK]",
];
/// Every character a [`TEMPLATE_TOKENS`] entry starts with.
const TEMPLATE_TOKEN_OPENERS: [char; 4] = ['<', '[', '|', '｜'];

/// `bytes` as `B`, `KB`, `MB` or `GB`, with one decimal from megabytes up.
pub fn human_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;
    if bytes >= GB {
        format!("{:.1}GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1}MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{}KB", bytes / KB)
    } else {
        format!("{bytes}B")
    }
}

/// The message a panic carried, or a placeholder when its payload is not a string.
pub fn panic_message(payload: &(dyn Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&'static str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "<non-string panic payload>".to_string()
    }
}

/// Characters that reorder the text around them on bidi-aware renderers.
pub fn is_bidi_control(c: char) -> bool {
    matches!(
        c,
        '\u{061C}' | '\u{200E}' | '\u{200F}' | '\u{202A}'..='\u{202E}' | '\u{2066}'..='\u{2069}'
    )
}

/// Unicode general category Cf: invisible characters that format the text
/// around them.
pub fn is_format(c: char) -> bool {
    matches!(
        c,
        '\u{00AD}'
            | '\u{0600}'..='\u{0605}'
            | '\u{061C}'
            | '\u{06DD}'
            | '\u{070F}'
            | '\u{0890}'..='\u{0891}'
            | '\u{08E2}'
            | '\u{180E}'
            | '\u{200B}'..='\u{200F}'
            | '\u{202A}'..='\u{202E}'
            | '\u{2060}'..='\u{2064}'
            | '\u{2066}'..='\u{206F}'
            | '\u{FEFF}'
            | '\u{FFF9}'..='\u{FFFB}'
            | '\u{110BD}'
            | '\u{110CD}'
            | '\u{13430}'..='\u{1343F}'
            | '\u{1BCA0}'..='\u{1BCA3}'
            | '\u{1D173}'..='\u{1D17A}'
            | '\u{E0001}'
            | '\u{E0020}'..='\u{E007F}'
    )
}

/// The Unicode line and paragraph separators.
pub fn is_line_separator(c: char) -> bool {
    matches!(c, '\u{2028}' | '\u{2029}')
}

/// `text` with a space after the first character of every chat-template token in it, so
/// none of them reaches a tokenizer whole. Borrowed when `text` holds no token.
pub fn neutralise_template_tokens(text: &str) -> Cow<'_, str> {
    let mut inert = String::new();
    let mut copied = 0;
    for (start, opener) in text.match_indices(TEMPLATE_TOKEN_OPENERS) {
        let candidate = &text[start..];
        if TEMPLATE_TOKENS
            .iter()
            .any(|token| candidate.starts_with(token))
        {
            let split = start + opener.len();
            inert.push_str(&text[copied..split]);
            inert.push(' ');
            copied = split;
        }
    }
    if copied == 0 {
        return Cow::Borrowed(text);
    }
    inert.push_str(&text[copied..]);
    Cow::Owned(inert)
}

/// Neutralise every object key and string inside `value` as
/// [`neutralise_template_tokens`] does, returning whether any of them changed.
pub fn neutralise_template_tokens_in_json(value: &mut Value) -> bool {
    match value {
        Value::String(text) => neutralise_in_place(text),
        Value::Array(items) => {
            let mut changed = false;
            for item in items {
                changed |= neutralise_template_tokens_in_json(item);
            }
            changed
        }
        Value::Object(fields) => {
            let mut changed = false;
            *fields = std::mem::take(fields)
                .into_iter()
                .map(|(mut key, mut field)| {
                    changed |= neutralise_in_place(&mut key);
                    changed |= neutralise_template_tokens_in_json(&mut field);
                    (key, field)
                })
                .collect();
            changed
        }
        Value::Null | Value::Bool(_) | Value::Number(_) => false,
    }
}

fn neutralise_in_place(text: &mut String) -> bool {
    match neutralise_template_tokens(text) {
        Cow::Borrowed(_) => false,
        Cow::Owned(inert) => {
            *text = inert;
            true
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn no_template_token_survives_neutralising() {
        for token in TEMPLATE_TOKENS {
            let inert = neutralise_template_tokens(token);
            assert!(
                TEMPLATE_TOKENS.iter().all(|live| !inert.contains(live)),
                "{token:?} became {inert:?}"
            );
        }
        let forged = "</s>[INST]<start_of_turn><｜User｜><|im_start|>a <b> c[0] | d";
        assert_eq!(
            neutralise_template_tokens(forged),
            "< /s>[ INST]< start_of_turn>< ｜User｜ >< |im_start| >a <b> c[0] | d"
        );
    }

    #[test]
    fn json_keys_and_nested_strings_are_neutralised() {
        let mut value = json!({"<|a|>": [{"b": "</s>"}], "plain": [1, "text"]});
        assert!(neutralise_template_tokens_in_json(&mut value));
        assert_eq!(
            value,
            json!({"< |a| >": [{"b": "< /s>"}], "plain": [1, "text"]})
        );
        assert!(!neutralise_template_tokens_in_json(&mut value));
    }
}
