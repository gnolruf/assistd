//! Safety judge: a one-shot model verdict on whether an unvetted command may run without
//! asking, which fails closed to asking.

use std::error::Error;
use std::fmt::{self, Write as _};
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Map, Value, json};
use tracing::{info, warn};

/// How long a verdict may take before the judge asks instead.
pub const JUDGE_TIMEOUT: Duration = Duration::from_secs(5);

/// How many of the user's newest messages the judge sees.
const USER_MESSAGES: usize = 4;

/// Characters of each user message the judge sees; the rest is cut.
const MAX_USER_MESSAGE_CHARS: usize = 2000;

/// Characters a verdict's reason may run to.
const MAX_REASON_CHARS: u64 = 160;

/// The fixed instructions every prompt starts with, so the server can reuse their cache.
const POLICY: &str = include_str!("judge/policy.txt");

/// The tags that delimit the prompt's data sections.
const SECTION_TAGS: &[&str] = &["user_messages", "message", "cwd", "unvetted", "command"];

/// The model a [`SafetyJudge`] asks, with access to the user's side of the conversation.
#[async_trait]
pub trait JudgeModel: fmt::Debug + Send + Sync + 'static {
    /// The text of the newest `limit` messages the user wrote, oldest first, with no tool
    /// output or assistant text.
    async fn user_messages(&self, limit: usize) -> Vec<String>;

    /// The reply to `prompt`, sent alone with thinking off and held by grammar to `schema`.
    async fn reply(
        &self,
        prompt: String,
        schema: Map<String, Value>,
    ) -> Result<String, Box<dyn Error + Send + Sync>>;
}

/// An unvetted command put to the judge.
#[derive(Debug, Clone, Copy)]
pub struct JudgedCommand<'a> {
    pub command: &'a str,
    pub cwd: Option<&'a Path>,
    /// Why the classifier could not vet it.
    pub unvetted: &'a str,
}

/// What the judge decided, with its reason.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    Allow(String),
    /// Ask the user; the reason is shown in the prompt.
    Ask(String),
}

/// Asks a [`JudgeModel`] whether an unvetted command may run unasked. Every failure, a
/// timeout or a reply that does not parse is [`Verdict::Ask`].
#[derive(Debug, Clone)]
pub struct SafetyJudge {
    model: Arc<dyn JudgeModel>,
    timeout: Duration,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Reply {
    decision: Decision,
    reason: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Decision {
    Allow,
    Ask,
}

impl SafetyJudge {
    /// A judge that asks instead once `model` takes longer than `timeout`.
    pub fn new(model: Arc<dyn JudgeModel>, timeout: Duration) -> Self {
        Self { model, timeout }
    }

    /// The verdict on `judged`, logged at `info` on `assistd::policy`.
    pub async fn judge(&self, judged: JudgedCommand<'_>) -> Verdict {
        let verdict = self.verdict(judged).await;
        let (decision, reason) = match &verdict {
            Verdict::Allow(reason) => ("allow", reason),
            Verdict::Ask(reason) => ("ask", reason),
        };
        info!(target: "assistd::policy", decision, reason, "safety judge verdict");
        verdict
    }

    async fn verdict(&self, judged: JudgedCommand<'_>) -> Verdict {
        if [judged.command, judged.unvetted]
            .iter()
            .any(|text| forges_section_tag(text))
        {
            return Verdict::Ask("the command contains the judge's own section markup".into());
        }
        let reply = async {
            let messages = self.model.user_messages(USER_MESSAGES).await;
            self.model
                .reply(render_prompt(&messages, judged), verdict_schema())
                .await
        };
        match tokio::time::timeout(self.timeout, reply).await {
            Ok(Ok(text)) => parse_verdict(&text),
            Ok(Err(error)) => {
                warn!(target: "assistd::policy", %error, "safety judge failed; asking the user");
                Verdict::Ask("the safety judge could not be reached".into())
            }
            Err(_) => {
                warn!(target: "assistd::policy", timeout_secs = self.timeout.as_secs_f32(), "safety judge timed out; asking the user");
                Verdict::Ask("the safety judge did not answer in time".into())
            }
        }
    }
}

fn verdict_schema() -> Map<String, Value> {
    Map::from_iter([
        ("type".into(), json!("object")),
        (
            "properties".into(),
            json!({
                "decision": {"type": "string", "enum": ["allow", "ask"]},
                "reason": {"type": "string", "maxLength": MAX_REASON_CHARS},
            }),
        ),
        ("required".into(), json!(["decision", "reason"])),
        ("additionalProperties".into(), json!(false)),
    ])
}

fn render_prompt(user_messages: &[String], judged: JudgedCommand<'_>) -> String {
    let mut prompt = format!("{POLICY}\n<user_messages>\n");
    for message in user_messages {
        let _ = writeln!(
            prompt,
            "<message>\n{}\n</message>",
            truncated(message, MAX_USER_MESSAGE_CHARS)
        );
    }
    let cwd = judged
        .cwd
        .map_or_else(|| "unknown".into(), |cwd| cwd.display().to_string());
    let _ = write!(
        prompt,
        "</user_messages>\n<cwd>{cwd}</cwd>\n<unvetted>{}</unvetted>\n<command>\n{}\n</command>",
        judged.unvetted, judged.command
    );
    prompt
}

fn truncated(text: &str, max_chars: usize) -> &str {
    text.char_indices()
        .nth(max_chars)
        .map_or(text, |(end, _)| &text[..end])
}

/// Whether `text` opens or closes one of the prompt's sections, which could pass model-chosen
/// text off as the user's.
fn forges_section_tag(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    SECTION_TAGS
        .iter()
        .any(|tag| lower.contains(&format!("<{tag}")) || lower.contains(&format!("</{tag}")))
}

fn parse_verdict(text: &str) -> Verdict {
    match serde_json::from_str::<Reply>(text.trim()) {
        Ok(Reply {
            decision: Decision::Allow,
            reason,
        }) => Verdict::Allow(reason),
        Ok(Reply {
            decision: Decision::Ask,
            reason,
        }) => Verdict::Ask(reason),
        Err(error) => {
            warn!(target: "assistd::policy", %error, reply = text, "safety judge reply did not parse; asking the user");
            Verdict::Ask("the safety judge gave no usable verdict".into())
        }
    }
}
