//! Persistence survives closing and reopening the store at the same path.

use assistd_memory::{
    BranchId, MemoryStore, PersistedMessage, PersistedRole, SessionId, SqliteHandle,
    SqliteMemoryStore, SqliteTranscriptStore, TranscriptStore,
};
use tokio::sync::watch;

#[tokio::test]
async fn turn_persists_across_store_reopen() {
    let temp = tempfile::Builder::new().suffix(".db").tempfile().unwrap();
    let path = temp.path().to_path_buf();

    let branch: BranchId;

    {
        let (_tx, rx) = watch::channel(false);
        let (handle, writer) = SqliteHandle::open(&path, rx).await.unwrap();
        let convs = SqliteTranscriptStore::new(handle.clone());
        let mems = SqliteMemoryStore::new(handle);

        let session = SessionId::new();
        let main = convs
            .begin_session_with_main_branch(&session, std::process::id())
            .await
            .unwrap();
        branch = main;

        let turn = convs.begin_turn(&session).await.unwrap();

        convs
            .append_message_to_branch(
                &session,
                main,
                Some(turn),
                PersistedMessage::user("what is rust?"),
            )
            .await
            .unwrap();

        convs
            .append_message_to_branch(
                &session,
                main,
                Some(turn),
                PersistedMessage::assistant_text(
                    "Rust is a systems programming language with a strong type system.",
                    String::new(),
                ),
            )
            .await
            .unwrap();

        convs.end_session(&session).await.unwrap();

        mems.save("fact:lang", "rust".into()).await.unwrap();

        // Awaiting the writer once every sender is dropped flushes the DB before reopening.
        drop(convs);
        drop(mems);
        writer.await.unwrap();
    }

    {
        let (_tx, rx) = watch::channel(false);
        let (handle, writer) = SqliteHandle::open(&path, rx).await.unwrap();
        let convs = SqliteTranscriptStore::new(handle.clone());
        let mems = SqliteMemoryStore::new(handle);

        let history = convs.load_branch_history(branch).await.unwrap();
        let messages: Vec<(PersistedRole, &str)> = history
            .iter()
            .map(|r| (r.message.role(), r.message.content.as_str()))
            .collect();
        assert_eq!(
            messages,
            [
                (PersistedRole::User, "what is rust?"),
                (
                    PersistedRole::Assistant,
                    "Rust is a systems programming language with a strong type system."
                ),
            ]
        );

        assert_eq!(
            mems.load("fact:lang").await.unwrap().as_deref(),
            Some("rust")
        );

        drop(convs);
        drop(mems);
        writer.await.unwrap();
    }
}

#[tokio::test]
async fn writer_drains_op_enqueued_immediately_after_shutdown_signal() {
    let temp = tempfile::Builder::new().suffix(".db").tempfile().unwrap();
    let path = temp.path().to_path_buf();
    let branch: BranchId;

    {
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let (handle, writer) = SqliteHandle::open(&path, shutdown_rx).await.unwrap();
        let convs = SqliteTranscriptStore::new(handle.clone());

        let session = SessionId::new();
        let main = convs
            .begin_session_with_main_branch(&session, std::process::id())
            .await
            .unwrap();
        branch = main;
        let turn = convs.begin_turn(&session).await.unwrap();

        shutdown_tx.send(true).unwrap();

        convs
            .append_message_to_branch(
                &session,
                main,
                Some(turn),
                PersistedMessage::user("drain race"),
            )
            .await
            .unwrap();
        convs
            .append_message_to_branch(
                &session,
                main,
                Some(turn),
                PersistedMessage::assistant_text("survived the drain", String::new()),
            )
            .await
            .unwrap();
        convs.end_session(&session).await.unwrap();

        drop(convs);
        writer.await.unwrap();
    }

    let (_tx, rx) = watch::channel(false);
    let (handle, writer) = SqliteHandle::open(&path, rx).await.unwrap();
    let convs = SqliteTranscriptStore::new(handle);
    let history: Vec<String> = convs
        .load_branch_history(branch)
        .await
        .unwrap()
        .into_iter()
        .map(|r| r.message.content)
        .collect();
    assert_eq!(history, ["drain race", "survived the drain"]);
    drop(convs);
    writer.await.unwrap();
}
