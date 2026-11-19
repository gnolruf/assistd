//! The extraction worker in its sandbox: what it reads, what it cannot, and the index
//! passes and watch through it. Returns early when the bwrap sandbox cannot start.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use assistd_config::DocumentsConfig;
use assistd_core::{DocumentIndex, IndexDeps, PassEnd, RootPass, SkipCause};
use assistd_documents::{Extraction, Extractor, SkipReason, supervise};
use assistd_memory::{SemanticStore, SqliteHandle, SqliteSemanticStore};
use assistd_tools::{
    Protected, SandboxInfo, SandboxRequest, SharedDirs, ToolSandbox, probe_sandbox,
};
use tokio::sync::watch;

const WORKER: &str = env!("CARGO_BIN_EXE_assistd");
const WORKER_SUBCOMMAND: &str = "extract-worker";
const MAX_FILE_BYTES: u64 = 1024 * 1024;
const STARTUP_LIMIT: Duration = Duration::from_secs(60);
const POLL_INTERVAL: Duration = Duration::from_millis(20);

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

/// Ask for a round of passes and wait for a pass that ended after `earlier`.
async fn pass_after(index: &DocumentIndex, earlier: Option<&RootPass>) -> RootPass {
    index.request_pass();
    let ended = async {
        loop {
            match index.last_passes().pop() {
                Some(last) if Some(&last) != earlier => return last,
                _ => tokio::time::sleep(POLL_INTERVAL).await,
            }
        }
    };
    tokio::time::timeout(STARTUP_LIMIT, ended)
        .await
        .expect("the pass ends")
}

async fn is_stored(store: &SqliteSemanticStore, path: &Path) -> bool {
    let first_byte = store.document_text(path, 0, 0..1).await;
    first_byte.expect("the store answers").is_some()
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
    let (command, _passes) = assistd_core::document_indexing(
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
        Ok(Extraction::Pages {
            pages: vec!["# Indexed\n\nvisible\n".into()],
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

#[tokio::test]
async fn passes_and_the_watch_between_them_index_changed_files_and_drop_deleted_ones() {
    let Some(sandbox) = bwrap_sandbox() else {
        return;
    };
    let temp = tempfile::tempdir().expect("tempdir");
    let root = temp.path().join("notes");
    let plan = root.join("work/plan.md");
    std::fs::create_dir_all(root.join("work")).expect("index root");
    std::fs::write(&plan, "# Plan\n\nfirst draft\n").expect("document");
    std::fs::write(root.join("old.md"), "to be deleted\n").expect("document");
    std::fs::write(root.join("image.png"), b"\x89PNG\r\n\x1a\n\0\0\0\rIHDR").expect("binary");
    let documents = DocumentsConfig {
        enabled: true,
        roots: vec![root.to_str().expect("utf-8").parse().expect("root")],
        ..DocumentsConfig::default()
    };
    let (command, passes) = assistd_core::document_indexing(
        &documents,
        Ok(&sandbox),
        Path::new(WORKER),
        WORKER_SUBCOMMAND,
    )
    .expect("document indexing");
    let handle = SqliteHandle::open(&temp.path().join("memory.db"))
        .await
        .expect("database");
    let store = Arc::new(SqliteSemanticStore::new(handle));
    let extractor = Arc::new(Extractor::starting());
    let index = Arc::new(DocumentIndex::default());
    let (stop, shutdown) = watch::channel(false);
    let supervisor = tokio::spawn(supervise(command, Arc::clone(&extractor), shutdown.clone()));
    let deps = IndexDeps {
        store: store.clone(),
        extractor,
        embed_wake: Arc::default(),
    };
    let runner = {
        let index = Arc::clone(&index);
        tokio::spawn(async move { passes.run(deps, &index, shutdown).await })
    };

    let first = pass_after(&index, None).await;
    assert_eq!(first.end, PassEnd::Finished);
    assert_eq!((first.counts.indexed, first.counts.unchanged), (2, 0));
    assert_eq!(first.counts.skipped.get(&SkipCause::Unsupported), Some(&1));
    assert_eq!(store.count_document_chunks().await.unwrap(), 2);

    std::fs::remove_file(root.join("old.md")).expect("deleted");
    std::fs::write(&plan, "# Plan\n\nsecond draft, longer\n").expect("edited");
    std::fs::write(root.join("new.md"), "a new note\n").expect("added");
    let second = pass_after(&index, Some(&first)).await;
    assert_eq!(second.end, PassEnd::Finished);
    assert_eq!(
        (
            second.counts.indexed,
            second.counts.unchanged,
            second.counts.removed
        ),
        (2, 0, 1)
    );
    let third = pass_after(&index, Some(&second)).await;
    assert_eq!(
        (
            third.counts.indexed,
            third.counts.unchanged,
            third.counts.removed
        ),
        (0, 2, 0)
    );

    let written = root.join("written.md");
    std::fs::write(&written, "written between passes\n").expect("added");
    std::fs::rename(root.join("work"), temp.path().join("moved-away")).expect("moved");
    let watched = async {
        while !is_stored(&store, &written).await || is_stored(&store, &plan).await {
            tokio::time::sleep(POLL_INTERVAL).await;
        }
    };
    tokio::time::timeout(STARTUP_LIMIT, watched)
        .await
        .expect("the watch has the changes indexed");
    assert!(is_stored(&store, &root.join("new.md")).await);
    assert_eq!(index.last_passes(), [third], "no pass over the whole root");

    stop.send_replace(true);
    runner.await.expect("passes stop");
    supervisor.await.expect("supervisor stops");
}
