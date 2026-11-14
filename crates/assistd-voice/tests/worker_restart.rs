//! The voice-worker supervisor against a stand-in worker spawned through the
//! real path: killing it mid-speech fails the speech, reports voice starting,
//! and the restarted worker brings voice back.

use std::ffi::OsString;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use assistd_config::VoiceConfig;
use assistd_ipc::ComponentReadiness;
use assistd_voice::worker::{VoiceWorkerSpec, supervise};
use assistd_voice::{NullBusyProbe, VoiceManager, VoiceOutputError, VoiceWorkerError};
use rustix::process::{Pid, Signal, kill_process};
use tokio::sync::watch;

/// Records its pid, reports both components ready, and answers every call
/// except `speak`, which it leaves in flight.
const STAND_IN_WORKER: &str = r#"
echo $$ >> "$1"
read -r _configure
echo '{"type":"capture_readiness","startup":{"status":"ready"}}'
echo '{"type":"speech_readiness","startup":{"status":"ready"}}'
while read -r line; do
  case "$line" in
    *'"op":"speak"'*) ;;
    *'"type":"call"'*)
      id=${line#*\"id\":}
      echo "{\"type\":\"reply\",\"id\":${id%%,*},\"outcome\":{\"status\":\"done\"}}" ;;
  esac
done
"#;

const SETTLE_TIMEOUT: Duration = Duration::from_secs(10);

async fn wait_for_readiness(voice: &VoiceManager, expected: &ComponentReadiness) {
    let mut changes = voice.subscribe_readiness();
    let settled = async {
        while !voice.readiness().iter().all(|(_, state)| state == expected) {
            changes.changed().await.expect("voice manager alive");
        }
    };
    tokio::time::timeout(SETTLE_TIMEOUT, settled)
        .await
        .unwrap_or_else(|_| panic!("voice never became {expected}: {:?}", voice.readiness()));
}

fn spawned_pids(path: &Path) -> Vec<i32> {
    std::fs::read_to_string(path)
        .unwrap_or_default()
        .lines()
        .map(|pid| pid.parse().expect("pid"))
        .collect()
}

#[tokio::test]
async fn a_worker_killed_mid_speech_fails_the_speech_and_restarts_ready() {
    let dir = tempfile::tempdir().unwrap();
    let pid_file = dir.path().join("pids");
    let spec = VoiceWorkerSpec {
        program: "/bin/sh".into(),
        args: ["-c", STAND_IN_WORKER, "stand-in-worker"]
            .into_iter()
            .map(OsString::from)
            .chain([pid_file.clone().into_os_string()])
            .collect(),
        config: VoiceConfig::default(),
    };
    let voice = VoiceManager::new(true);
    let (shutdown_tx, shutdown_rx) = watch::channel(false);
    let supervisor = tokio::spawn(supervise(
        spec,
        Arc::clone(&voice),
        Arc::new(NullBusyProbe),
        shutdown_rx,
    ));
    wait_for_readiness(&voice, &ComponentReadiness::Ready).await;

    let output = voice.speech().output().expect("speech is ready");
    let speech = tokio::spawn(async move { output.speak("Hello there.".into()).await });
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert!(
        !speech.is_finished(),
        "the stand-in never finishes speaking"
    );
    let first = Pid::from_raw(spawned_pids(&pid_file)[0]).expect("nonzero pid");
    kill_process(first, Signal::KILL).unwrap();

    let failed = speech.await.unwrap();
    assert!(
        matches!(
            failed,
            Err(VoiceOutputError::Worker(VoiceWorkerError::Exited))
        ),
        "{failed:?}"
    );
    wait_for_readiness(&voice, &ComponentReadiness::Starting).await;
    wait_for_readiness(&voice, &ComponentReadiness::Ready).await;
    assert_eq!(spawned_pids(&pid_file).len(), 2, "restarted once");

    shutdown_tx.send_replace(true);
    supervisor.await.unwrap();
}
