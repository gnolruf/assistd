//! Persistence survives closing and reopening the store at the same path.

use assistd_memory::{
    BranchId, ConversationStore, MemoryStore, PersistedMessage, PersistedRole,
    SqliteConversationStore, SqliteHandle, SqliteMemoryStore,
};
use std::sync::Arc;
use tokio::sync::watch;

#[tokio::test]
async fn turn_persists_across_store_reopen() {
    let temp = tempfile::Builder::new().suffix(".db").tempfile().unwrap();
    let path = temp.path().to_path_buf();

    let branch: BranchId;

    // ─── Session 1 ────────────────────────────────────────────────
    {
        let (_tx, rx) = watch::channel(false);
        let (handle, writer) = SqliteHandle::open(&path, rx).await.unwrap();
        let handle = Arc::new(handle);
        let convs = SqliteConversationStore::new(handle.clone());
        let mems = SqliteMemoryStore::new(handle);

        let (session, main) = convs
            .begin_session_with_main_branch(std::process::id())
            .await
            .unwrap();
        branch = main;

        let turn = convs.begin_turn(&session, "what is rust?").await.unwrap();

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
                ),
            )
            .await
            .unwrap();

        convs.end_turn(turn).await.unwrap();
        convs.end_session(&session).await.unwrap();

        mems.save("fact:lang", "rust".into()).await.unwrap();

        // Dropping the last sender lets the writer exit; awaiting it
        // flushes the DB before the reopen.
        drop(convs);
        drop(mems);
        writer.await.unwrap();
    }

    // ─── Session 2 (simulates a daemon restart) ──────────────────
    {
        let (_tx, rx) = watch::channel(false);
        let (handle, writer) = SqliteHandle::open(&path, rx).await.unwrap();
        let handle = Arc::new(handle);
        let convs = SqliteConversationStore::new(handle.clone());
        let mems = SqliteMemoryStore::new(handle);

        let history = convs.load_branch_history(branch).await.unwrap();
        let roles: Vec<PersistedRole> = history.iter().map(|r| r.role).collect();
        assert_eq!(roles, [PersistedRole::User, PersistedRole::Assistant]);
        assert!(history[1].content.contains("systems programming"));

        assert_eq!(
            mems.load("fact:lang").await.unwrap().as_deref(),
            Some("rust")
        );

        let turns = convs.recent_turns(5).await.unwrap();
        assert!(!turns.is_empty());
        assert_eq!(turns[0].user_text, "what is rust?");

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
        let handle = Arc::new(handle);
        let convs = SqliteConversationStore::new(handle.clone());

        let (session, main) = convs
            .begin_session_with_main_branch(std::process::id())
            .await
            .unwrap();
        branch = main;
        let turn = convs.begin_turn(&session, "drain race").await.unwrap();

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
                PersistedMessage::assistant_text("survived the drain"),
            )
            .await
            .unwrap();
        convs.end_turn(turn).await.unwrap();
        convs.end_session(&session).await.unwrap();

        drop(convs);
        writer.await.unwrap();
    }

    let (_tx, rx) = watch::channel(false);
    let (handle, writer) = SqliteHandle::open(&path, rx).await.unwrap();
    let handle = Arc::new(handle);
    let convs = SqliteConversationStore::new(handle);
    let history = convs.load_branch_history(branch).await.unwrap();
    assert!(
        history.iter().any(|r| r.content == "survived the drain"),
        "post-shutdown append did not persist: {history:#?}"
    );
    drop(convs);
    writer.await.unwrap();
}
