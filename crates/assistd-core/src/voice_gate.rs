//! The voice confirmation gate: asks a headless turn's prompts out loud and reads the
//! spoken answer.

use std::future;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use parking_lot::Mutex;
use tokio::sync::mpsc;
use tokio::time::{Instant, sleep_until, timeout};
use tracing::{debug, info, warn};

use assistd_llm::{LlmBackend, OneShotReply};
use assistd_tools::{CONFIRM_TIMEOUT, ConfirmationRequest, SpokenAnswer, Summary, VoiceGate};
use assistd_voice::{VoiceManager, VoiceOutput};

/// How long after a question ends its answer may begin.
const ANSWER_WAIT: Duration = Duration::from_secs(10);

/// How long past [`ANSWER_WAIT`] an answer begun in time may take to end and be transcribed.
const ANSWER_GRACE: Duration = Duration::from_secs(5);

/// How long the model may take to read an answer before it counts as a refusal.
const READING_TIMEOUT: Duration = Duration::from_secs(5);

/// Names a plan's question in the log, where a prompt's is named by its tool.
const PLAN_ASKER: &str = "plan";

const CONFIRM: &str = "Confirm";
const DENY: &str = "Deny";
const INSTRUCT: &str = "Instruct";

/// What reaches a question being asked from the mic.
#[derive(Debug)]
enum Reply {
    /// A push-to-talk press began recording the answer.
    Recording,
    /// The answer as transcribed, empty when nothing was heard.
    Said(String),
}

/// Speaks each prompt of a headless turn as one short question and has the chat model read
/// the next thing the user says as a yes, a no or words for the agent. Silence, muted or unavailable speech and a
/// failed reading all refuse.
#[derive(Debug)]
pub(crate) struct VoiceConfirmationGate {
    voice: Arc<VoiceManager>,
    llm: Arc<dyn LlmBackend>,
    /// Where the question being asked takes its answer; `None` while none is.
    asking: Mutex<Option<mpsc::UnboundedSender<Reply>>>,
}

impl VoiceConfirmationGate {
    pub(crate) fn new(voice: Arc<VoiceManager>, llm: Arc<dyn LlmBackend>) -> Self {
        Self {
            voice,
            llm,
            asking: Mutex::new(None),
        }
    }

    /// Whether a question is being asked or awaits its answer.
    pub(crate) fn asking(&self) -> bool {
        self.asking.lock().is_some()
    }

    /// Note that a push-to-talk press is recording the answer, which then has as long as the
    /// press takes; `false` when no question awaits one.
    pub(crate) fn recording_answer(&self) -> bool {
        self.send(Reply::Recording)
    }

    /// Hand `said` to the question awaiting an answer; `false` when none does.
    pub(crate) fn answer(&self, said: String) -> bool {
        self.send(Reply::Said(said))
    }

    fn send(&self, reply: Reply) -> bool {
        self.asking
            .lock()
            .as_ref()
            .is_some_and(|question| question.send(reply).is_ok())
    }

    /// Ask `question` and wait for what the user says to it; `None` when nothing was said.
    async fn heard_answer(&self, output: &dyn VoiceOutput, question: String) -> Option<String> {
        let (replies_tx, mut replies) = mpsc::unbounded_channel();
        let _asking = Asking::begin(self, replies_tx);
        let asked = say(output, question);
        tokio::pin!(asked);
        let mut asked_whole = false;
        let mut recording = false;
        let mut deadline = None;
        loop {
            tokio::select! {
                reply = replies.recv() => match reply? {
                    Reply::Recording => {
                        recording = true;
                        deadline = None;
                    }
                    Reply::Said(said) => return Some(said).filter(|said| !said.trim().is_empty()),
                },
                () = &mut asked, if !asked_whole => {
                    asked_whole = true;
                    if recording {
                        self.voice.speech().hush().await;
                    } else {
                        self.open_answer_window();
                        deadline = Some(Instant::now() + ANSWER_WAIT + ANSWER_GRACE);
                    }
                }
                () = until(deadline) => return None,
            }
        }
    }

    fn open_answer_window(&self) {
        if let Ok(capture) = self.voice.capture() {
            capture.listener.await_answer(ANSWER_WAIT);
        }
    }

    /// The chat model's reading of `said`, with nothing of the turn's history; anything but a
    /// clear yes refuses, handing `said` to the agent when it asks for something else.
    async fn read_intent(&self, said: String) -> SpokenAnswer {
        let reading = self.llm.complete_oneshot(
            intent_prompt(&said),
            OneShotReply::OneOf(&[CONFIRM, DENY, INSTRUCT]),
        );
        match timeout(READING_TIMEOUT, reading).await {
            Ok(Ok(intent)) if intent.trim() == CONFIRM => SpokenAnswer::Confirm,
            Ok(Ok(intent)) if intent.trim() == INSTRUCT => SpokenAnswer::Message(said),
            Ok(Ok(_)) => SpokenAnswer::Deny,
            Ok(Err(error)) => {
                warn!(target: "assistd::policy", %error, "spoken answer could not be read");
                SpokenAnswer::Deny
            }
            Err(_) => {
                warn!(target: "assistd::policy", "spoken answer was not read in time");
                SpokenAnswer::Deny
            }
        }
    }

    /// Speak `question`, asked for `asker`, and read what the user says to it.
    async fn asked(&self, asker: &str, question: String) -> SpokenAnswer {
        let speech = self.voice.speech();
        let output = match speech.output() {
            Ok(output) if speech.enabled() => output,
            Ok(_) => {
                warn!(target: "assistd::policy", %asker, "not asked: speech is muted");
                return SpokenAnswer::Unanswered;
            }
            Err(why) => {
                warn!(target: "assistd::policy", %asker, %why, "not asked: no speech");
                return SpokenAnswer::Unanswered;
            }
        };
        info!(target: "assistd::policy", %asker, %question, "asking out loud");
        let heard = self.heard_answer(output.as_ref(), question);
        match timeout(CONFIRM_TIMEOUT, heard).await {
            Ok(Some(said)) => {
                debug!(target: "assistd::policy", %said, "spoken answer heard");
                self.read_intent(said).await
            }
            Ok(None) | Err(_) => SpokenAnswer::Unanswered,
        }
    }
}

#[async_trait]
impl VoiceGate for VoiceConfirmationGate {
    async fn ask_aloud(&self, req: &ConfirmationRequest) -> SpokenAnswer {
        self.asked(&req.tool, spoken_question(&req.summary)).await
    }

    async fn propose_aloud(&self, proposal: &str) -> SpokenAnswer {
        self.asked(PLAN_ASKER, speakable(proposal)).await
    }
}

/// A question taking its answer through the gate, until dropped.
struct Asking<'gate>(&'gate VoiceConfirmationGate);

impl<'gate> Asking<'gate> {
    fn begin(gate: &'gate VoiceConfirmationGate, replies: mpsc::UnboundedSender<Reply>) -> Self {
        *gate.asking.lock() = Some(replies);
        Self(gate)
    }
}

impl Drop for Asking<'_> {
    fn drop(&mut self) {
        *self.0.asking.lock() = None;
        if let Ok(capture) = self.0.voice.capture() {
            capture.listener.answer_settled();
        }
    }
}

/// Speak `question` once the reply so far has played, and wait until it has played too.
async fn say(output: &dyn VoiceOutput, question: String) {
    let spoken = async {
        output.wait_idle().await?;
        output.speak(question).await?;
        output.wait_idle().await
    };
    if let Err(error) = spoken.await {
        warn!(target: "assistd::policy", %error, "the question may not have been spoken whole");
    }
}

async fn until(deadline: Option<Instant>) {
    match deadline {
        Some(deadline) => sleep_until(deadline).await,
        None => future::pending().await,
    }
}

/// The question that asks for `summary` out loud.
fn spoken_question(summary: &Summary) -> String {
    speakable(&match &summary.why {
        Some(why) => format!("Allow {}? {why}", summary.what),
        None => format!("Allow {}?", summary.what),
    })
}

/// `text` as speech can say it: no backticks, and nothing in brackets.
fn speakable(text: &str) -> String {
    let mut bracket_depth = 0_usize;
    let mut spoken = String::with_capacity(text.len());
    for character in text.chars() {
        match character {
            '(' => {
                bracket_depth += 1;
                spoken.truncate(spoken.trim_end().len());
            }
            ')' => bracket_depth = bracket_depth.saturating_sub(1),
            '`' => {}
            _ if bracket_depth == 0 => spoken.push(character),
            _ => {}
        }
    }
    spoken
}

fn intent_prompt(said: &str) -> String {
    format!(
        "Infer the intent of the following phrase from the user. It can be classified into one \
         of the following options:\n{CONFIRM}\n{DENY}\n{INSTRUCT}\n\n{INSTRUCT} is for a phrase \
         that asks for something different or adds a condition, even when it begins with a yes or \
         a no. If it is at all unclear what the intent is, you should assume {DENY} by \
         default\n\nThe phrase: {said}"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn question_names_the_request_and_its_reason_without_what_speech_cannot_say() {
        let fetch = Summary::of("a fetch from example.com");
        assert_eq!(spoken_question(&fetch), "Allow a fetch from example.com?");

        let read = Summary::of("a read of config in .ssh").because(
            "It reads a tree holding SSH keys (`~/.ssh/config`, /home/u/.ssh/config), a \
             sensitive path.",
        );
        assert_eq!(
            spoken_question(&read),
            "Allow a read of config in .ssh? It reads a tree holding SSH keys, a sensitive path."
        );
    }
}
