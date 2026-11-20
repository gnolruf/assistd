//! One request per page image to the OCR model: the image and a prompt in, the
//! page's text out, with a reply that is not the page's text told apart.

use std::borrow::Cow;
use std::net::SocketAddr;
use std::time::Duration;

use assistd_utils::child_server::ApiKey;
use assistd_utils::http::{BodyError, local_client_builder, read_capped};
use serde::Deserialize;
use thiserror::Error;

use crate::completion::wire::{
    ChatMessage, ChatRequest, ChatTemplateKwargs, ContentBody, ContentPart, ImageUrl,
};
use crate::llama_server::OCR_MODEL_ALIAS;

/// Most tokens of one page's transcription; a reply that reaches it is not kept.
/// A dense printed page measured 1,400.
pub(crate) const MAX_TRANSCRIPTION_TOKENS: u32 = 4_096;

/// How long the model has to answer for one page.
const PAGE_DEADLINE: Duration = Duration::from_secs(300);

const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// Largest reply body read; a transcription of [`MAX_TRANSCRIPTION_TOKENS`] is far smaller.
const REPLY_MAX_BYTES: usize = 1024 * 1024;

/// Characters of an error body kept in [`TranscribeError::Server`].
const ERROR_BODY_CHARS: usize = 512;

/// The `finish_reason` of a reply cut off at `max_tokens`.
const FINISHED_AT_TOKEN_CAP: &str = "length";

/// Longest unit, in characters, looked for repeating at the end of a transcription.
const MAX_LOOP_UNIT_CHARS: usize = 256;

/// Times in a row a unit must end a transcription to count as a loop.
const LOOP_REPEATS: usize = 8;

/// Fewest characters the repeats of a unit must cover to count as a loop.
const LOOP_SPAN_CHARS: usize = 64;

/// A page request the model did not answer with a reply.
#[derive(Debug, Error)]
pub enum TranscribeError {
    #[error("HTTP transport error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("llama-server returned {status}: {body}")]
    Server { status: u16, body: String },

    #[error("llama-server reply: {0}")]
    Body(#[from] BodyError),

    #[error("parse llama-server reply: {0}")]
    Json(#[from] serde_json::Error),

    #[error("llama-server replied with no message")]
    NoMessage,
}

/// What the model's reply to one page image is taken as.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Transcription {
    /// The page's text, trimmed; empty when the model read none.
    Text(String),
    /// Not the page's text, so nothing of it is kept.
    Unusable(UnusableTranscription),
}

impl Transcription {
    fn of_reply(text: &str, finish_reason: Option<&str>) -> Self {
        let text = text.trim();
        if finish_reason == Some(FINISHED_AT_TOKEN_CAP) {
            Self::Unusable(UnusableTranscription::HitTokenCap)
        } else if ends_in_loop(text) {
            Self::Unusable(UnusableTranscription::Loops)
        } else {
            Self::Text(text.to_owned())
        }
    }
}

/// Why a reply is not kept as a page's text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum UnusableTranscription {
    #[error("the transcription reached the limit of {MAX_TRANSCRIPTION_TOKENS} tokens")]
    HitTokenCap,
    #[error("the transcription ends in a repeating loop")]
    Loops,
}

/// Sends page images to the OCR model on the router, which must already have
/// it loaded: a request never loads it.
#[derive(Debug)]
pub struct PageTranscriber {
    client: reqwest::Client,
    url: String,
    prompt: String,
}

impl PageTranscriber {
    /// A transcriber for the router at `addr` that sends `prompt` after each
    /// image, or the image alone when it is empty.
    pub fn new(
        addr: SocketAddr,
        api_key: Option<&ApiKey>,
        prompt: String,
    ) -> Result<Self, TranscribeError> {
        let client = local_client_builder()
            .default_headers(api_key.map_or_default(ApiKey::authorization_headers))
            .connect_timeout(CONNECT_TIMEOUT)
            .timeout(PAGE_DEADLINE)
            .build()?;
        Ok(Self {
            client,
            url: format!("http://{addr}/v1/chat/completions?autoload=false"),
            prompt,
        })
    }

    /// Transcribe the page in `png_base64`, a PNG in standard base64, with one
    /// request: no history, no tools, thinking off, greedy sampling.
    pub async fn transcribe(&self, png_base64: &str) -> Result<Transcription, TranscribeError> {
        let image_url = format!("data:image/png;base64,{png_base64}");
        let response = self
            .client
            .post(&self.url)
            .json(&self.request(&image_url))
            .send()
            .await?;
        let status = response.status();
        let body = read_capped(response, REPLY_MAX_BYTES).await?;
        if !status.is_success() {
            return Err(TranscribeError::Server {
                status: status.as_u16(),
                body: String::from_utf8_lossy(&body)
                    .chars()
                    .take(ERROR_BODY_CHARS)
                    .collect(),
            });
        }
        let reply: ChatCompletion = serde_json::from_slice(&body)?;
        let choice = reply
            .choices
            .into_iter()
            .next()
            .ok_or(TranscribeError::NoMessage)?;
        let text = choice.message.content.unwrap_or_default();
        Ok(Transcription::of_reply(
            &text,
            choice.finish_reason.as_deref(),
        ))
    }

    fn request<'a>(&'a self, image_url: &'a str) -> ChatRequest<'a> {
        let image = ContentPart::ImageUrl {
            image_url: ImageUrl { url: image_url },
        };
        let prompt = (!self.prompt.is_empty()).then_some(ContentPart::Text {
            text: Cow::Borrowed(self.prompt.as_str()),
        });
        let parts = std::iter::once(image).chain(prompt).collect();
        ChatRequest {
            model: OCR_MODEL_ALIAS,
            messages: vec![ChatMessage::plain("user", ContentBody::Parts(parts))],
            stream: false,
            stream_options: None,
            temperature: 0.0,
            max_tokens: MAX_TRANSCRIPTION_TOKENS,
            top_p: None,
            top_k: None,
            min_p: None,
            presence_penalty: None,
            tools: None,
            tool_choice: None,
            reasoning_budget_tokens: None,
            reasoning_budget_message: None,
            chat_template_kwargs: Some(ChatTemplateKwargs {
                enable_thinking: Some(false),
                reasoning_effort: None,
            }),
            response_format: None,
            id_slot: None,
        }
    }
}

/// A non-streamed `/v1/chat/completions` reply.
#[derive(Deserialize)]
struct ChatCompletion {
    #[serde(default)]
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    message: ReplyMessage,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct ReplyMessage {
    #[serde(default)]
    content: Option<String>,
}

/// Whether `text` ends with one unit of up to [`MAX_LOOP_UNIT_CHARS`] characters
/// at least [`LOOP_REPEATS`] times in a row over [`LOOP_SPAN_CHARS`] or more. A unit
/// with no letter or digit, such as a ruled line or an empty table row, is not one.
fn ends_in_loop(text: &str) -> bool {
    let chars: Vec<char> = text.chars().collect();
    (1..=MAX_LOOP_UNIT_CHARS).any(|unit| {
        let repeats = LOOP_REPEATS.max(LOOP_SPAN_CHARS.div_ceil(unit));
        let Some(start) = chars.len().checked_sub(unit * repeats) else {
            return false;
        };
        let tail = &chars[start..];
        tail[..unit].iter().any(|c| c.is_alphanumeric())
            && tail.iter().zip(&tail[unit..]).all(|(a, b)| a == b)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_reply_that_hits_the_token_cap_or_ends_in_a_loop_is_unusable_and_any_other_is_kept() {
        let page = "# Minutes\n\nPresent: Ann, Bo and Cy.\n\n| Item | Owner |\n|---|---|\n";
        let rows = "| Budget | Ann |\n".repeat(3);
        let ruled = format!("{page}{}", "_".repeat(200));
        let empty_rows = format!("{page}{}", "|  |  |\n".repeat(40));
        let note = "Note: The handwritten text contains errors. ";
        let kept = [
            format!("  {page}{rows}\n"),
            ruled,
            empty_rows,
            String::new(),
        ];
        for text in kept {
            assert_eq!(
                Transcription::of_reply(&text, Some("stop")),
                Transcription::Text(text.trim().to_owned()),
            );
        }
        let looping = [
            format!("{page}{}", note.repeat(LOOP_REPEATS + 1)),
            format!("{page}{}", "the the ".repeat(40)),
            format!("{page}{}", "0".repeat(LOOP_SPAN_CHARS)),
        ];
        for text in looping {
            assert_eq!(
                Transcription::of_reply(&text, Some("stop")),
                Transcription::Unusable(UnusableTranscription::Loops),
                "{text}"
            );
        }
        assert_eq!(
            Transcription::of_reply(page, Some("length")),
            Transcription::Unusable(UnusableTranscription::HitTokenCap),
        );
    }
}
