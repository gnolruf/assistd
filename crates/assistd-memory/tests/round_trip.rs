//! Persistence survives closing and reopening the store at the same path.

use assistd_memory::{
    BranchId, MemoryStore, PersistedMessage, PersistedRole, SessionId, SqliteHandle,
    SqliteMemoryStore, SqliteTranscriptStore, TranscriptStore,
};

#[tokio::test]
async fn turn_persists_across_store_reopen() {
    let temp = tempfile::Builder::new().suffix(".db").tempfile().unwrap();
    let path = temp.path().to_path_buf();

    let branch: BranchId;

    {
        let handle = SqliteHandle::open(&path).await.unwrap();
        let convs = SqliteTranscriptStore::new(handle.clone());
        let mems = SqliteMemoryStore::new(handle.clone());

        let session = SessionId::new();
        let main = convs
            .begin_session_with_main_branch(session, std::process::id())
            .await
            .unwrap();
        branch = main;

        let turn = convs.begin_turn(session).await.unwrap();

        convs
            .append_message_to_branch(main, turn, PersistedMessage::user("what is rust?"))
            .await
            .unwrap();

        convs
            .append_message_to_branch(
                main,
                turn,
                PersistedMessage::assistant_text(
                    "Rust is a systems programming language with a strong type system.",
                    String::new(),
                ),
            )
            .await
            .unwrap();

        convs.end_session(session).await.unwrap();

        mems.save("fact:lang", "rust".into()).await.unwrap();

        handle.close().await.unwrap();
    }

    {
        let handle = SqliteHandle::open(&path).await.unwrap();
        let convs = SqliteTranscriptStore::new(handle.clone());
        let mems = SqliteMemoryStore::new(handle.clone());

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

        handle.close().await.unwrap();
    }
}
