use std::ffi::OsString;
use std::time::Duration;

use assistd_config::VoiceConfig;
use assistd_voice::NullBusyProbe;
use assistd_voice::worker::{VoiceWorkerSpec, supervise};

use super::*;

/// Reports the mic missing, then capture ready once the file named by its
/// first argument exists.
const STAND_IN_WORKER: &str = r#"
read -r _configure
echo '{"type":"capture_readiness","startup":{"status":"unavailable","reason":"mic not found"}}'
echo '{"type":"speech_readiness","startup":{"status":"ready"}}'
while [ ! -e "$1" ]; do sleep 0.05; done
echo '{"type":"capture_readiness","startup":{"status":"ready"}}'
while read -r _line; do :; done
"#;

#[tokio::test]
async fn capture_is_awaited_through_a_mic_missing_at_startup() {
    let dir = tempfile::tempdir().unwrap();
    let plugged_in = dir.path().join("plugged-in");
    let spec = VoiceWorkerSpec {
        program: "/bin/sh".into(),
        args: ["-c", STAND_IN_WORKER, "stand-in-worker"]
            .into_iter()
            .map(OsString::from)
            .chain([plugged_in.clone().into_os_string()])
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
    let mut readiness = voice.subscribe_readiness();
    while voice.speech().output().is_err() {
        readiness.changed().await.unwrap();
    }

    let capture = started_capture(&voice);
    tokio::pin!(capture);
    let while_missing = tokio::time::timeout(Duration::from_millis(200), &mut capture).await;
    assert!(
        while_missing.is_err(),
        "a missing mic must not end the wait"
    );

    std::fs::write(&plugged_in, "").unwrap();
    let plugged = tokio::time::timeout(Duration::from_secs(10), capture).await;
    assert!(matches!(plugged, Ok(Some(_))), "capture never came up");

    shutdown_tx.send_replace(true);
    supervisor.await.unwrap();
}
