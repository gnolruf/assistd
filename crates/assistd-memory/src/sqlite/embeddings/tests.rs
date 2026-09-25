use std::sync::Arc;

use tokio::sync::watch;

use super::*;
use crate::sqlite::{ConversationStore, SqliteConversationStore, SqliteHandle, SqliteMemoryStore};
use crate::{MemoryStore, PersistedMessage};

/// The returned guard keeps the temp dir and the writer's shutdown sender alive.
async fn fresh() -> (
    Arc<SqliteHandle>,
    SqliteSemanticStore,
    (tempfile::TempDir, watch::Sender<bool>),
) {
    let temp = tempfile::tempdir().unwrap();
    let (tx, rx) = watch::channel(false);
    let (handle, _writer) = SqliteHandle::open(&temp.path().join("memory.db"), rx)
        .await
        .unwrap();
    let handle = Arc::new(handle);
    let store = SqliteSemanticStore::new(handle.clone());
    (handle, store, (temp, tx))
}

async fn seed_conversation(handle: &Arc<SqliteHandle>, msg: PersistedMessage) -> (SessionId, i64) {
    let store = SqliteConversationStore::new(handle.clone());
    let (session, branch) = store.begin_session_with_main_branch(0).await.unwrap();
    let conv_id = store
        .append_message_to_branch(&session, branch, None, msg)
        .await
        .unwrap();
    (session, conv_id)
}

/// 2-d unit vector at `angle` radians.
fn unit_vec(angle: f32) -> Vec<f32> {
    vec![angle.cos(), angle.sin()]
}

async fn insert_chunk_with_vec(
    handle: &SqliteHandle,
    store: &SqliteSemanticStore,
    conv_id: i64,
    chunk_index: i64,
    vector: &[f32],
    model: &str,
) -> i64 {
    let chunk_id = handle
        .store_chunk(conv_id, chunk_index, format!("chunk{chunk_index}"), None)
        .await
        .unwrap();
    store
        .store_chunk_embedding(chunk_id, model.into(), 2, vector_to_blob(vector))
        .await
        .unwrap();
    chunk_id
}

async fn save_memory(handle: &Arc<SqliteHandle>, key: &str, value: &str) -> i64 {
    SqliteMemoryStore::new(handle.clone())
        .save(key, value.into())
        .await
        .unwrap()
}

async fn embed_memory(store: &SqliteSemanticStore, memory_id: i64, vector: &[f32], model: &str) {
    store
        .store_memory_embedding(memory_id, model.into(), 2, vector_to_blob(vector))
        .await
        .unwrap();
}

#[tokio::test]
async fn nearest_chunks_ranks_by_similarity_and_returns_parent_message() {
    let (handle, store, _guard) = fresh().await;
    let (session, conv_id) =
        seed_conversation(&handle, PersistedMessage::user("hello world")).await;

    let c1 = insert_chunk_with_vec(&handle, &store, conv_id, 0, &unit_vec(0.0), "m").await;
    let c2 = insert_chunk_with_vec(&handle, &store, conv_id, 1, &unit_vec(0.3), "m").await;
    let c3 = insert_chunk_with_vec(&handle, &store, conv_id, 2, &unit_vec(1.5), "m").await;

    let hits = store
        .nearest_chunks(unit_vec(0.0), 3, "m", None)
        .await
        .unwrap();
    let ids: Vec<i64> = hits.iter().map(|h| h.chunk_id).collect();
    assert_eq!(ids, [c1, c2, c3]);
    assert!(hits[0].similarity > hits[1].similarity);
    assert!(hits[1].similarity > hits[2].similarity);
    assert!((hits[0].similarity - 1.0).abs() < 1e-4);
    for hit in &hits {
        assert_eq!(hit.conversation_id, conv_id);
        assert_eq!(hit.session_id, session.0);
        assert_eq!(hit.role, PersistedRole::User);
        assert_eq!(hit.content, "hello world");
    }
}

#[tokio::test]
async fn nearest_chunks_keeps_the_best_top_k() {
    let (handle, store, _guard) = fresh().await;
    let (_, conv_id) = seed_conversation(&handle, PersistedMessage::user("x")).await;
    let mut ids = Vec::new();
    for i in 0..10 {
        ids.push(
            insert_chunk_with_vec(
                &handle,
                &store,
                conv_id,
                i,
                &unit_vec((i as f32) * 0.1),
                "m",
            )
            .await,
        );
    }

    for (top_k, expected) in [(3, &ids[..3]), (usize::MAX, &ids[..])] {
        let hits = store
            .nearest_chunks(unit_vec(0.0), top_k, "m", None)
            .await
            .unwrap();
        let got: Vec<i64> = hits.iter().map(|h| h.chunk_id).collect();
        assert_eq!(got, expected, "top_k {top_k}");
    }
}

#[tokio::test]
async fn nearest_chunks_can_exclude_one_session() {
    let (handle, store, _guard) = fresh().await;
    let (past, past_conv) = seed_conversation(&handle, PersistedMessage::user("past")).await;
    let (current, current_conv) =
        seed_conversation(&handle, PersistedMessage::user("current")).await;
    let past_chunk =
        insert_chunk_with_vec(&handle, &store, past_conv, 0, &unit_vec(0.4), "m").await;
    let current_chunk =
        insert_chunk_with_vec(&handle, &store, current_conv, 0, &unit_vec(0.0), "m").await;

    let all = store
        .nearest_chunks(unit_vec(0.0), 5, "m", None)
        .await
        .unwrap();
    let all_ids: Vec<i64> = all.iter().map(|h| h.chunk_id).collect();
    assert_eq!(
        all_ids,
        [current_chunk, past_chunk],
        "the excluded session must hold the closer match"
    );

    let others = store
        .nearest_chunks(unit_vec(0.0), 5, "m", Some(&current))
        .await
        .unwrap();
    let other_ids: Vec<(i64, &str)> = others
        .iter()
        .map(|h| (h.chunk_id, h.session_id.as_str()))
        .collect();
    assert_eq!(other_ids, [(past_chunk, past.as_str())]);
}

#[tokio::test]
async fn nearest_chunks_filters_by_model() {
    let (handle, store, _guard) = fresh().await;
    let (_, conv_id) = seed_conversation(&handle, PersistedMessage::user("x")).await;
    insert_chunk_with_vec(&handle, &store, conv_id, 0, &unit_vec(0.0), "old-model").await;
    let hits = store
        .nearest_chunks(unit_vec(0.0), 5, "new-model", None)
        .await
        .unwrap();
    assert_eq!(hits, Vec::<EmbeddingHit>::new());
}

#[tokio::test]
async fn nearest_memories_round_trips() {
    let (handle, store, _guard) = fresh().await;
    let mem_id = save_memory(&handle, "editor", "vim").await;
    embed_memory(&store, mem_id, &unit_vec(0.0), "m").await;

    let hits = store.nearest_memories(unit_vec(0.0), 5, "m").await.unwrap();
    let [hit] = hits.as_slice() else {
        panic!("expected one hit, got {hits:?}");
    };
    assert_eq!(
        (hit.memory_id, hit.key.as_str(), hit.value.as_str()),
        (mem_id, "editor", "vim")
    );
    assert!((hit.similarity - 1.0).abs() < 1e-4);
}

#[tokio::test]
async fn upsert_replaces_memory_embedding_in_place() {
    let (handle, store, _guard) = fresh().await;
    let mem_id = save_memory(&handle, "k", "v").await;
    embed_memory(&store, mem_id, &unit_vec(0.0), "m").await;
    embed_memory(&store, mem_id, &unit_vec(std::f32::consts::FRAC_PI_2), "m").await;

    assert_eq!(store.count_for_model("m").await.unwrap(), (0, 1));
    let hits = store.nearest_memories(unit_vec(0.0), 5, "m").await.unwrap();
    let [hit] = hits.as_slice() else {
        panic!("expected one hit, got {hits:?}");
    };
    assert!(
        hit.similarity.abs() < 1e-4,
        "stale vector survived: {}",
        hit.similarity
    );
}

#[tokio::test]
async fn missing_embedding_lists_only_unindexed_rows_for_current_model() {
    let (handle, store, _guard) = fresh().await;
    let (_, conv_id) = seed_conversation(&handle, PersistedMessage::user("x")).await;

    insert_chunk_with_vec(&handle, &store, conv_id, 0, &unit_vec(0.0), "new").await;
    let old_chunk = insert_chunk_with_vec(&handle, &store, conv_id, 1, &unit_vec(0.5), "old").await;
    let naked_chunk = handle
        .store_chunk(conv_id, 2, "naked-chunk".into(), None)
        .await
        .unwrap();

    let indexed_mem = save_memory(&handle, "indexed", "v1").await;
    embed_memory(&store, indexed_mem, &unit_vec(0.0), "new").await;
    let bare_mem = save_memory(&handle, "bare", "v2").await;

    assert_eq!(
        store.chunks_missing_embedding("new").await.unwrap(),
        [
            (old_chunk, "chunk1".to_string()),
            (naked_chunk, "naked-chunk".to_string())
        ]
    );
    assert_eq!(
        store.memories_missing_embedding("new").await.unwrap(),
        [(bare_mem, "v2".to_string())]
    );

    embed_memory(&store, bare_mem, &unit_vec(0.0), "new").await;
    assert_eq!(
        store.memories_missing_embedding("new").await.unwrap(),
        Vec::<(i64, String)>::new()
    );
}

#[tokio::test]
async fn count_stale_aggregates_across_chunks_and_memories() {
    let (handle, store, _guard) = fresh().await;
    let (_, conv_id) = seed_conversation(&handle, PersistedMessage::user("x")).await;

    insert_chunk_with_vec(&handle, &store, conv_id, 0, &unit_vec(0.0), "old-A").await;
    insert_chunk_with_vec(&handle, &store, conv_id, 1, &unit_vec(0.5), "old-A").await;
    insert_chunk_with_vec(&handle, &store, conv_id, 2, &unit_vec(1.0), "old-B").await;
    insert_chunk_with_vec(&handle, &store, conv_id, 3, &unit_vec(1.5), "new").await;
    let mem_id = save_memory(&handle, "k", "v").await;
    embed_memory(&store, mem_id, &unit_vec(0.0), "old-A").await;

    assert_eq!(
        store.count_stale("new").await.unwrap(),
        (4, vec!["old-A".to_string(), "old-B".to_string()]),
        "2 old-A chunks + 1 old-B chunk + 1 old-A memory"
    );
    assert_eq!(
        store.count_stale("old-A").await.unwrap(),
        (2, vec!["new".to_string(), "old-B".to_string()]),
        "the old-B chunk + the new chunk"
    );
}

#[test]
fn vector_to_blob_round_trips() {
    let vector = vec![0.5f32, -0.25, 0.125, 1e-6];
    let blob = vector_to_blob(&vector);
    let (words, _) = blob.as_chunks::<4>();
    let decoded: Vec<f32> = words.iter().copied().map(f32::from_le_bytes).collect();
    assert_eq!(decoded, vector);
}

#[test]
fn score_against_rejects_malformed_blobs() {
    let q = [1.0f32, 2.0];
    assert_eq!(score_against(&q, &vector_to_blob(&[3.0, 4.0])), Some(11.0));
    assert_eq!(score_against(&q, &vector_to_blob(&[1.0, 0.0, 0.0])), None);
    assert_eq!(score_against(&q, &[0u8; 7]), None);
}
