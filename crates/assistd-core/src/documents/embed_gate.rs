//! The gate on embedding document chunks: shut while the user's interaction, a task
//! run or a GPU sleep lasts, unless `[documents]` lets embedding go on through it.

use std::sync::Arc;

use tokio::sync::watch;

use assistd_config::Config;

use crate::admission::TurnAdmission;
use crate::admission::interaction::InteractionState;
use crate::admission::lanes::changed_or_pending;
use crate::presence::PresenceManager;

/// Keep `gate` at whether document chunks may be embedded now, until its last
/// receiver is dropped.
pub(crate) async fn keep_document_embed_gate(
    config: Arc<Config>,
    admission: Arc<TurnAdmission>,
    presence: Arc<PresenceManager>,
    gate: watch::Sender<bool>,
) {
    let documents = &config.documents;
    let mut interaction = admission.interaction.subscribe();
    let mut lanes = admission.lanes.watch();
    let mut settled = presence.subscribe_settled();
    loop {
        let interaction_open = *interaction.borrow_and_update() != InteractionState::Idle;
        let asleep_for_gpu = settled.borrow_and_update().asleep_for_gpu();
        let open = (documents.embed_during_interaction || !interaction_open)
            && (documents.embed_during_tasks || !admission.lanes.held_by_run())
            && (documents.embed_during_gpu_sleep || !asleep_for_gpu);
        gate.send_if_modified(|current| {
            let changed = *current != open;
            *current = open;
            changed
        });
        tokio::select! {
            () = gate.closed() => return,
            () = changed_or_pending(&mut interaction) => {}
            () = lanes.changed() => {}
            () = changed_or_pending(&mut settled) => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;
    use std::time::Duration;

    use assistd_config::DocumentsConfig;
    use assistd_ipc::{PresenceTarget, TaskRunId};

    use super::*;
    use crate::presence::SleepCause;
    use crate::presence::tests::{settle_stub_active, stub_presence};

    /// Long enough for the gate's task to act on a change, in paused time.
    const SETTLE: Duration = Duration::from_millis(10);

    /// The gate of a daemon with one model lane, and what its states are read from.
    struct Harness {
        admission: Arc<TurnAdmission>,
        presence: Arc<PresenceManager>,
        gate: watch::Receiver<bool>,
    }

    impl Harness {
        fn new(documents: DocumentsConfig) -> Self {
            let config = Arc::new(Config {
                documents,
                ..Config::default()
            });
            let slots = watch::channel(NonZeroU32::MIN).1;
            let admission = Arc::new(TurnAdmission::new(Arc::clone(&config), slots));
            let presence = stub_presence(PresenceTarget::Active);
            let (gate_tx, gate) = watch::channel(false);
            tokio::spawn(keep_document_embed_gate(
                config,
                Arc::clone(&admission),
                Arc::clone(&presence),
                gate_tx,
            ));
            Self {
                admission,
                presence,
                gate,
            }
        }

        async fn open(&self) -> bool {
            tokio::time::sleep(SETTLE).await;
            *self.gate.borrow()
        }

        /// Whether the gate is open during the user's interaction, the quiet time
        /// after its turn included, a task run, and a GPU sleep that a wake ends.
        async fn open_during_each_state(&self) -> [bool; 3] {
            assert!(self.open().await, "shut while idle");
            let turn = self
                .admission
                .interaction
                .begin_turn(Duration::from_secs(60));
            let during_turn = self.open().await;
            drop(turn);
            let during_interaction = self.open().await;
            assert_eq!(during_turn, during_interaction, "the quiet time differs");
            self.admission.interaction.end();
            assert!(self.open().await, "shut after the interaction");

            let run = self.admission.lanes.admit(TaskRunId(1)).await;
            let during_task = self.open().await;
            drop(run);
            assert!(self.open().await, "shut after the task run");

            self.presence
                .sleep(SleepCause::GpuBusy)
                .await
                .expect("stub sleep");
            let during_gpu_sleep = self.open().await;
            settle_stub_active(&self.presence);
            assert!(self.open().await, "shut after the wake from the GPU sleep");
            [during_interaction, during_task, during_gpu_sleep]
        }
    }

    #[tokio::test(start_paused = true)]
    async fn gate_shuts_in_each_busy_state_except_the_ones_documents_config_embeds_during() {
        let defaults = DocumentsConfig::default();
        let cases = [
            (defaults.clone(), [false, false, false]),
            (
                DocumentsConfig {
                    embed_during_interaction: true,
                    ..defaults.clone()
                },
                [true, false, false],
            ),
            (
                DocumentsConfig {
                    embed_during_tasks: true,
                    ..defaults.clone()
                },
                [false, true, false],
            ),
            (
                DocumentsConfig {
                    embed_during_gpu_sleep: true,
                    ..defaults
                },
                [false, false, true],
            ),
        ];
        for (documents, open) in cases {
            let harness = Harness::new(documents.clone());
            assert_eq!(
                harness.open_during_each_state().await,
                open,
                "{documents:?}"
            );
        }
    }
}
