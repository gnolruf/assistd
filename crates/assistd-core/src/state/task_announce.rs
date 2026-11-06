//! Announcing a task run's final reply: broadcast for notifications, then spoken once the
//! user's conversation has gone quiet, a chat turn starting meanwhile cutting it off.

use tokio::sync::mpsc;

use assistd_ipc::{Event, TaskName, TaskRunId};
use assistd_voice::SentenceBuffer;

use super::AppState;
use super::conversation::ConversationState;
use super::query::run_speech_worker;

/// A finished run's reply to announce, and the request id its events carry.
pub(super) struct TaskReply {
    pub(super) id: String,
    pub(super) run: TaskRunId,
    pub(super) task_name: TaskName,
    pub(super) reply: String,
}

impl AppState {
    pub(super) async fn announce_task_reply(&self, reply: TaskReply) {
        let TaskReply {
            id,
            run,
            task_name,
            reply,
        } = reply;
        let sentences = self.reply_sentences(&reply);
        self.runtime.publish(&Event::TaskReply {
            id: id.clone(),
            run,
            task_name,
            reply,
        });
        self.wait_for_quiet_chat().await;
        let voice_output = self.subsystems.voice.speech().clone();
        let start_epoch = voice_output.current_epoch();
        let (speech_tx, speech_rx) = mpsc::channel(sentences.len().max(1));
        for sentence in sentences {
            let _ = speech_tx.try_send(sentence);
        }
        drop(speech_tx);
        run_speech_worker(
            voice_output,
            self.runtime.events_bus().clone(),
            id,
            start_epoch,
            speech_rx,
        )
        .await;
    }

    /// Return once no user turn is in flight and nothing is being spoken.
    async fn wait_for_quiet_chat(&self) {
        let mut conversation = self.runtime.conversation.subscribe();
        let mut speaking = self.subsystems.voice.speech().subscribe_speaking();
        loop {
            let talking = matches!(
                *conversation.borrow_and_update(),
                ConversationState::Talking { .. }
            );
            if !talking && !*speaking.borrow_and_update() {
                return;
            }
            let changed = tokio::select! {
                changed = conversation.changed() => changed,
                changed = speaking.changed() => changed,
            };
            if changed.is_err() {
                return;
            }
        }
    }

    fn reply_sentences(&self, reply: &str) -> Vec<String> {
        let max_chars = self.config.voice.synthesis.max_sentence_chars.get() as usize;
        let mut sentences = SentenceBuffer::new(max_chars);
        let mut split = sentences.push(reply);
        split.extend(sentences.finish());
        split
    }
}
