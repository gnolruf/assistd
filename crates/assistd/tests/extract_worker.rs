//! The extraction worker in its sandbox: what it reads and what it cannot.
//! Returns early when the bwrap sandbox cannot start.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use assistd_config::DocumentsConfig;
use assistd_documents::{Extraction, Extractor, SkipReason, supervise};
use assistd_tools::{
    Protected, SandboxInfo, SandboxRequest, SharedDirs, ToolSandbox, probe_sandbox,
};
use tokio::sync::watch;

const WORKER: &str = env!("CARGO_BIN_EXE_assistd");
const WORKER_SUBCOMMAND: &str = "extract-worker";
const MAX_FILE_BYTES: u64 = 1024 * 1024;
const STARTUP_LIMIT: Duration = Duration::from_secs(60);

fn bwrap_sandbox() -> Option<Arc<SandboxInfo>> {
    match probe_sandbox(
        SandboxRequest::Bwrap,
        Vec::new(),
        Protected::default(),
        SharedDirs::default(),
    ) {
        Ok(ToolSandbox::Bwrap(sandbox)) => Some(sandbox),
        _ => None,
    }
}

#[tokio::test]
async fn the_worker_reads_files_under_its_roots_and_none_outside_them() {
    let Some(sandbox) = bwrap_sandbox() else {
        return;
    };
    let root = tempfile::tempdir().expect("index root");
    let outside = tempfile::tempdir().expect("directory outside the roots");
    std::fs::write(root.path().join("notes.md"), "# Indexed\n\nvisible\n").expect("document");
    std::fs::write(outside.path().join("secret.txt"), "hidden").expect("outside file");
    std::os::unix::fs::symlink(
        outside.path().join("secret.txt"),
        root.path().join("link.txt"),
    )
    .expect("symlink out of the root");
    let documents = DocumentsConfig {
        enabled: true,
        roots: vec![root.path().to_str().expect("utf-8").parse().expect("root")],
        ..DocumentsConfig::default()
    };
    let command = assistd_core::extract_worker_command(
        &documents,
        Ok(&sandbox),
        Path::new(WORKER),
        WORKER_SUBCOMMAND,
    )
    .expect("worker command");
    let extractor = Arc::new(Extractor::starting());
    let (stop, shutdown) = watch::channel(false);
    let supervisor = tokio::spawn(supervise(command, Arc::clone(&extractor), shutdown));
    let mut readiness = extractor.subscribe_readiness();
    tokio::time::timeout(STARTUP_LIMIT, readiness.wait_for(Result::is_ok))
        .await
        .expect("the worker starts in its sandbox")
        .expect("the supervisor is running");

    let indexed = extractor
        .extract(&root.path().join("notes.md"), MAX_FILE_BYTES)
        .await;
    assert_eq!(
        indexed,
        Ok(Extraction::Text {
            text: "# Indexed\n\nvisible\n".into(),
            truncated: false,
        })
    );
    for hidden in [
        outside.path().join("secret.txt"),
        root.path().join("link.txt"),
        PathBuf::from("/etc/passwd"),
    ] {
        let extraction = extractor.extract(&hidden, MAX_FILE_BYTES).await;
        assert!(
            matches!(
                extraction,
                Ok(Extraction::Skipped {
                    reason: SkipReason::Unreadable { .. }
                })
            ),
            "{}: {extraction:?}",
            hidden.display()
        );
    }

    stop.send_replace(true);
    supervisor.await.expect("supervisor stops");
}
