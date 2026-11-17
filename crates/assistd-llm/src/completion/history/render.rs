//! Renders history messages to the wire. Text the model, a tool or a context
//! block supplied has its chat-template tokens neutralised here.

use std::borrow::Cow;

use assistd_utils::text::{neutralise_template_tokens, neutralise_template_tokens_in_json};
use serde_json::Value;

use super::{AssistantMessage, ImageDataUri, Message, Role, UserMessage, step_log, truncate_utf8};
use crate::completion::wire;

pub(super) const SUMMARY_PREFIX: &str = "[Conversation summary] ";
/// Longest quote of a message the summary instruction uses to name it.
const SUMMARY_QUOTE_BYTES: usize = 120;
/// Delimiters around the tool name that heads an image-carrying tool result.
pub(super) const TOOL_RESULT_OPEN: &str = "[tool:";
pub(super) const TOOL_RESULT_CLOSE: &str = "]\n";
/// Plain-text delimiters around a context block folded into its user turn;
/// templates mishandle system messages anywhere but the head of the list.
pub(super) const CONTEXT_OPEN: &str = "[Context: added automatically, not written by the user]\n";
pub(super) const CONTEXT_CLOSE: &str = "\n[End of context]\n\n";

pub(super) fn wire_message(message: &Message) -> wire::ChatMessage<'_> {
    let role = message.role().as_wire();
    match message {
        Message::Summary(text) => wire::ChatMessage::plain(
            role,
            wire::ContentBody::Text(Cow::Owned(summary_text(text))),
        ),
        Message::User(user) => {
            wire::ChatMessage::plain(role, content_body(wire_text(user), &user.attachments))
        }
        Message::Assistant(step) if !step.tool_calls.is_empty() => tool_calls_message(step),
        Message::Assistant(reply) => wire::ChatMessage::plain(role, defanged(&reply.content)),
        Message::ToolResult { call_id, content } => wire::ChatMessage {
            tool_call_id: Some(neutralise_template_tokens(call_id)),
            ..wire::ChatMessage::plain(role, defanged(content))
        },
        Message::ToolImageResult {
            tool_name,
            content,
            attachments,
        } => wire::ChatMessage::plain(
            role,
            content_body(
                Cow::Owned(image_result_text(tool_name, content)),
                attachments,
            ),
        ),
    }
}

fn defanged(text: &str) -> wire::ContentBody<'_> {
    wire::ContentBody::Text(neutralise_template_tokens(text))
}

pub(super) fn summary_text(summary: &str) -> String {
    format!("{SUMMARY_PREFIX}{}", neutralise_template_tokens(summary))
}

fn image_result_text(tool_name: &str, content: &str) -> String {
    format!(
        "{TOOL_RESULT_OPEN}{}{TOOL_RESULT_CLOSE}{}",
        neutralise_template_tokens(tool_name),
        neutralise_template_tokens(content)
    )
}

fn tool_calls_message(message: &AssistantMessage) -> wire::ChatMessage<'_> {
    let specs = message
        .tool_calls
        .iter()
        .map(|call| wire::ToolCallSpec {
            id: neutralise_template_tokens(&call.id),
            kind: "function",
            function: wire::FunctionCallSpec {
                name: neutralise_template_tokens(&call.name),
                arguments: replayed_arguments(&call.arguments),
            },
        })
        .collect();
    wire::ChatMessage {
        role: Role::Assistant.as_wire(),
        content: (!message.content.is_empty()).then(|| defanged(&message.content)),
        tool_calls: Some(specs),
        tool_call_id: None,
        reasoning_content: (!message.reasoning.is_empty())
            .then(|| neutralise_template_tokens(&message.reasoning)),
    }
}

/// `arguments` with every key and string neutralised after JSON decoding, so no
/// escape sequence carries a template token past; verbatim when it holds none.
fn replayed_arguments(arguments: &str) -> Cow<'_, str> {
    let Ok(mut decoded) = serde_json::from_str::<Value>(arguments) else {
        return neutralise_template_tokens(arguments);
    };
    if neutralise_template_tokens_in_json(&mut decoded) {
        Cow::Owned(decoded.to_string())
    } else {
        Cow::Borrowed(arguments)
    }
}

fn content_body<'a>(text: Cow<'a, str>, attachments: &'a [ImageDataUri]) -> wire::ContentBody<'a> {
    if attachments.is_empty() {
        return wire::ContentBody::Text(text);
    }
    let mut parts = Vec::with_capacity(attachments.len() + 1);
    parts.push(wire::ContentPart::Text { text });
    parts.extend(attachments.iter().map(|image| wire::ContentPart::ImageUrl {
        image_url: wire::ImageUrl {
            url: image.as_str(),
        },
    }));
    wire::ContentBody::Parts(parts)
}

fn wire_text(user: &UserMessage) -> Cow<'_, str> {
    let text = text_with_context(user);
    if user.earlier_steps.is_empty() {
        return text;
    }
    let log = step_log::render(&user.earlier_steps);
    Cow::Owned(format!("{text}\n\n{}", neutralise_template_tokens(&log)))
}

fn text_with_context(user: &UserMessage) -> Cow<'_, str> {
    match &user.context {
        Some(ctx) => {
            let ctx = neutralise_context_markers(ctx.trim_end());
            Cow::Owned(format!(
                "{CONTEXT_OPEN}{}{CONTEXT_CLOSE}{}",
                neutralise_template_tokens(&ctx),
                user.content
            ))
        }
        None => Cow::Borrowed(&user.content),
    }
}

fn neutralise_context_markers(ctx: &str) -> Cow<'_, str> {
    let open = CONTEXT_OPEN.trim();
    let close = CONTEXT_CLOSE.trim();
    if !ctx.contains(open) && !ctx.contains(close) {
        return Cow::Borrowed(ctx);
    }
    let defang = |marker: &str| marker.replace('[', "(").replace(']', ")");
    Cow::Owned(
        ctx.replace(open, &defang(open))
            .replace(close, &defang(close)),
    )
}

/// The opening of `text`'s first non-empty line, defanged, for naming a
/// message inside the summary instruction.
pub(super) fn quote(text: &str) -> String {
    let line = text
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty())
        .unwrap_or("");
    truncate_utf8(&neutralise_template_tokens(line), SUMMARY_QUOTE_BYTES)
}

/// `step`'s first tool call as the instruction names it.
pub(super) fn quote_first_call(step: &Message) -> String {
    step.tool_calls().first().map_or_else(String::new, |call| {
        format!(
            "`{}` with arguments `{}`",
            neutralise_template_tokens(&call.name),
            quote(&call.arguments)
        )
    })
}
