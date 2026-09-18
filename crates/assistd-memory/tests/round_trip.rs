//! Persistence survives closing and reopening the store at the same path.

use assistd_memory::{
    ConversationStore, MemoryStore, PersistedMessage, PersistedRole, SqliteConversationStore,
    SqliteHandle, SqliteMemoryStore,
};
use std::sync::Arc;
use tokio::sync::watch;

#[tokio::test]
async fn turn_persists_across_store_reopen() {
    let temp = tempfile::Builder::new().suffix(".db").tempfile().unwrap();
    let path = temp.path().to_path_buf();

    let session_id_text: String;

    // ─── Session 1 ────────────────────────────────────────────────
    {
        let (_tx, rx) = watch::channel(false);
        let (handle, writer) = SqliteHandle::open(&path, rx).await.unwrap();
        let handle = Arc::new(handle);
        let convs = SqliteConversationStore::new(handle.clone());
        let mems = SqliteMemoryStore::new(handle);

        let session = convs.begin_session(std::process::id()).await.unwrap();
        session_id_text = session.0.clone();

        let turn = convs.begin_turn(&session, "what is rust?").await.unwrap();

        convs
            .append_message(
                &session,
                Some(turn),
                PersistedMessage::user("what is rust?"),
            )
            .await
            .unwrap();

        convs
            .append_message(
                &session,
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

        let hits = convs.search("rust", 10).await.unwrap();
        assert!(
            hits.iter()
                .any(|h| h.snippet.to_lowercase().contains("systems programming")),
            "expected an assistant hit mentioning 'systems programming': {hits:#?}"
        );
        // The session id from session 1 must round-trip via the hit.
        assert!(
            hits.iter().any(|h| h.session_id == session_id_text),
            "expected at least one hit from session {session_id_text}: {hits:#?}"
        );
        // Roles include both User (the prompt) and Assistant (the answer).
        let roles: std::collections::HashSet<PersistedRole> = hits.iter().map(|h| h.role).collect();
        assert!(roles.contains(&PersistedRole::User));
        assert!(roles.contains(&PersistedRole::Assistant));

        // KV memory persisted too.
        assert_eq!(
            mems.load("fact:lang").await.unwrap().as_deref(),
            Some("rust")
        );

        // recent_turns reflects the prior session.
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
    let session_text: String;

    {
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let (handle, writer) = SqliteHandle::open(&path, shutdown_rx).await.unwrap();
        let handle = Arc::new(handle);
        let convs = SqliteConversationStore::new(handle.clone());

        let session = convs.begin_session(std::process::id()).await.unwrap();
        session_text = session.0.clone();
        let turn = convs.begin_turn(&session, "drain race").await.unwrap();

        // Fire the shutdown signal, then immediately enqueue more
        // writes. Without the fix these would silently disappear.
        shutdown_tx.send(true).unwrap();

        convs
            .append_message(&session, Some(turn), PersistedMessage::user("drain race"))
            .await
            .unwrap();
        convs
            .append_message(
                &session,
                Some(turn),
                PersistedMessage::assistant_text("survived the drain"),
            )
            .await
            .unwrap();
        convs.end_turn(turn).await.unwrap();
        convs.end_session(&session).await.unwrap();

        drop(convs);
        // Writer must process the post-shutdown ops before exiting.
        writer.await.unwrap();
    }

    // Reopen and confirm the row landed.
    let (_tx, rx) = watch::channel(false);
    let (handle, writer) = SqliteHandle::open(&path, rx).await.unwrap();
    let handle = Arc::new(handle);
    let convs = SqliteConversationStore::new(handle);
    let hits = convs.search("survived the drain", 5).await.unwrap();
    assert!(
        hits.iter().any(|h| h.session_id == session_text),
        "post-shutdown append_message did not persist: {hits:#?}"
    );
    drop(convs);
    writer.await.unwrap();
}
