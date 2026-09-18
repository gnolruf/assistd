use super::*;
use crate::PersistedMessage;
use crate::sqlite::SqliteHandle;
use crate::sqlite::writer::WriteOp;
use std::sync::Arc;
use tokio::sync::{oneshot, watch};

async fn fresh() -> (Arc<SqliteHandle>, tokio::task::JoinHandle<()>) {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("memory.db");
    std::mem::forget(temp);
    let (_tx, rx) = watch::channel(false);
    let (handle, writer) = SqliteHandle::open(&path, rx).await.unwrap();
    (Arc::new(handle), writer)
}

fn unit_vec(angle: f32) -> Vec<f32> {
    // 2-d unit vector at the given angle.
    vec![angle.cos(), angle.sin()]
}

async fn insert_chunk_with_vec(
    handle: &SqliteHandle,
    conv_id: i64,
    chunk_index: i64,
    v: &[f32],
    model: &str,
) -> i64 {
    // store_chunk
    let (tx, rx) = oneshot::channel();
    handle
        .writer()
        .send(WriteOp::StoreChunk {
            conversation_id: conv_id,
            chunk_index,
            content: format!("chunk{chunk_index}"),
            token_count: None,
            ack: tx,
        })
        .await
        .unwrap();
    let chunk_id = rx.await.unwrap().unwrap();
    // store embedding
    let (tx, rx) = oneshot::channel();
    handle
        .writer()
        .send(WriteOp::StoreChunkEmbedding {
            chunk_id,
            model: model.into(),
            dim: v.len() as i64,
            vector: vector_to_blob(v),
            ack: tx,
        })
        .await
        .unwrap();
    rx.await.unwrap().unwrap();
    chunk_id
}

#[tokio::test]
async fn nearest_chunks_empty_store_returns_empty() {
    let (handle, _w) = fresh().await;
    let s = SqliteSemanticStore::new(handle);
    let hits = s
        .nearest_chunks(unit_vec(0.0), 5, "test-model", None)
        .await
        .unwrap();
    assert!(hits.is_empty());
}

#[tokio::test]
async fn nearest_chunks_ranks_by_similarity() {
    let (handle, _w) = fresh().await;
    // Create one conversation row to FK the chunks against.
    let (tx, rx) = oneshot::channel();
    handle
        .writer()
        .send(WriteOp::BeginSession {
            session_id: "sess-1".into(),
            daemon_pid: 1,
            ack: tx,
        })
        .await
        .unwrap();
    rx.await.unwrap().unwrap();
    let (tx, rx) = oneshot::channel();
    handle
        .writer()
        .send(WriteOp::AppendMessage {
            session_id: "sess-1".into(),
            turn_id: None,
            msg: PersistedMessage::user("hello world"),
            ack: tx,
        })
        .await
        .unwrap();
    let conv_id = rx.await.unwrap().unwrap();

    // Insert three chunks with vectors at different angles.
    // Query points at 0; expect chunk at angle 0 to win, then 0.3, then 1.5.
    let _c1 = insert_chunk_with_vec(&handle, conv_id, 0, &unit_vec(0.0), "m").await;
    let _c2 = insert_chunk_with_vec(&handle, conv_id, 1, &unit_vec(0.3), "m").await;
    let _c3 = insert_chunk_with_vec(&handle, conv_id, 2, &unit_vec(1.5), "m").await;

    let s = SqliteSemanticStore::new(handle);
    let hits = s.nearest_chunks(unit_vec(0.0), 3, "m", None).await.unwrap();
    assert_eq!(hits.len(), 3);
    // Best-first.
    assert!(hits[0].similarity > hits[1].similarity);
    assert!(hits[1].similarity > hits[2].similarity);
    // First hit should be ~1.0.
    assert!((hits[0].similarity - 1.0).abs() < 1e-4);
}

#[tokio::test]
async fn nearest_chunks_top_k_caps_results() {
    let (handle, _w) = fresh().await;
    let (tx, rx) = oneshot::channel();
    handle
        .writer()
        .send(WriteOp::BeginSession {
            session_id: "s".into(),
            daemon_pid: 0,
            ack: tx,
        })
        .await
        .unwrap();
    rx.await.unwrap().unwrap();
    let (tx, rx) = oneshot::channel();
    handle
        .writer()
        .send(WriteOp::AppendMessage {
            session_id: "s".into(),
            turn_id: None,
            msg: PersistedMessage::user("x"),
            ack: tx,
        })
        .await
        .unwrap();
    let conv_id = rx.await.unwrap().unwrap();
    for i in 0..10 {
        insert_chunk_with_vec(&handle, conv_id, i, &unit_vec((i as f32) * 0.1), "m").await;
    }
    let s = SqliteSemanticStore::new(handle);
    let hits = s.nearest_chunks(unit_vec(0.0), 3, "m", None).await.unwrap();
    assert_eq!(hits.len(), 3);
}

#[tokio::test]
async fn nearest_chunks_can_exclude_one_session() {
    let (handle, _w) = fresh().await;
    let mut conv_ids = Vec::new();
    for session in ["past", "current"] {
        let (tx, rx) = oneshot::channel();
        handle
            .writer()
            .send(WriteOp::BeginSession {
                session_id: session.into(),
                daemon_pid: 0,
                ack: tx,
            })
            .await
            .unwrap();
        rx.await.unwrap().unwrap();
        let (tx, rx) = oneshot::channel();
        handle
            .writer()
            .send(WriteOp::AppendMessage {
                session_id: session.into(),
                turn_id: None,
                msg: PersistedMessage::user(session),
                ack: tx,
            })
            .await
            .unwrap();
        conv_ids.push(rx.await.unwrap().unwrap());
    }
    // The current session holds the closer match, so excluding it
    // has to change the result rather than just trim the tail.
    insert_chunk_with_vec(&handle, conv_ids[0], 0, &unit_vec(0.4), "m").await;
    insert_chunk_with_vec(&handle, conv_ids[1], 0, &unit_vec(0.0), "m").await;

    let s = SqliteSemanticStore::new(handle);
    let all = s.nearest_chunks(unit_vec(0.0), 5, "m", None).await.unwrap();
    assert_eq!(all.len(), 2);

    let current = SessionId("current".into());
    let others = s
        .nearest_chunks(unit_vec(0.0), 5, "m", Some(&current))
        .await
        .unwrap();
    assert_eq!(others.len(), 1);
    assert_eq!(others[0].session_id, "past");
}

#[tokio::test]
async fn nearest_chunks_filters_by_model() {
    let (handle, _w) = fresh().await;
    let (tx, rx) = oneshot::channel();
    handle
        .writer()
        .send(WriteOp::BeginSession {
            session_id: "s".into(),
            daemon_pid: 0,
            ack: tx,
        })
        .await
        .unwrap();
    rx.await.unwrap().unwrap();
    let (tx, rx) = oneshot::channel();
    handle
        .writer()
        .send(WriteOp::AppendMessage {
            session_id: "s".into(),
            turn_id: None,
            msg: PersistedMessage::user("x"),
            ack: tx,
        })
        .await
        .unwrap();
    let conv_id = rx.await.unwrap().unwrap();
    insert_chunk_with_vec(&handle, conv_id, 0, &unit_vec(0.0), "old-model").await;
    let s = SqliteSemanticStore::new(handle);
    // Query with the new model name; old-model rows must not appear.
    let hits = s
        .nearest_chunks(unit_vec(0.0), 5, "new-model", None)
        .await
        .unwrap();
    assert!(hits.is_empty());
}

#[tokio::test]
async fn nearest_memories_round_trips() {
    let (handle, _w) = fresh().await;
    // Save a memory; capture its row id from the writer ack.
    let (tx, rx) = oneshot::channel();
    handle
        .writer()
        .send(WriteOp::SaveMemory {
            key: "editor".into(),
            value: "vim".into(),
            source_conversation_id: None,
            ack: tx,
        })
        .await
        .unwrap();
    let mem_id = rx.await.unwrap().unwrap();
    // Embed it.
    let (tx, rx) = oneshot::channel();
    let v = unit_vec(0.0);
    handle
        .writer()
        .send(WriteOp::StoreMemoryEmbedding {
            memory_id: mem_id,
            model: "m".into(),
            dim: v.len() as i64,
            vector: vector_to_blob(&v),
            ack: tx,
        })
        .await
        .unwrap();
    rx.await.unwrap().unwrap();
    // Retrieve.
    let s = SqliteSemanticStore::new(handle);
    let hits = s.nearest_memories(unit_vec(0.0), 5, "m").await.unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].key, "editor");
    assert_eq!(hits[0].value, "vim");
    assert!((hits[0].similarity - 1.0).abs() < 1e-4);
}

#[tokio::test]
async fn upsert_replaces_memory_embedding_in_place() {
    let (handle, _w) = fresh().await;
    let (tx, rx) = oneshot::channel();
    handle
        .writer()
        .send(WriteOp::SaveMemory {
            key: "k".into(),
            value: "v1".into(),
            source_conversation_id: None,
            ack: tx,
        })
        .await
        .unwrap();
    let mem_id_1 = rx.await.unwrap().unwrap();
    // Re-save under same key; id should be stable.
    let (tx, rx) = oneshot::channel();
    handle
        .writer()
        .send(WriteOp::SaveMemory {
            key: "k".into(),
            value: "v2".into(),
            source_conversation_id: None,
            ack: tx,
        })
        .await
        .unwrap();
    let mem_id_2 = rx.await.unwrap().unwrap();
    assert_eq!(mem_id_1, mem_id_2, "UPSERT must keep same row id");
}

#[tokio::test]
async fn no_semantic_store_returns_empty() {
    let s = NoSemanticStore;
    assert!(
        s.nearest_chunks(vec![1.0], 5, "m", None)
            .await
            .unwrap()
            .is_empty()
    );
    assert!(
        s.nearest_memories(vec![1.0], 5, "m")
            .await
            .unwrap()
            .is_empty()
    );
    assert_eq!(s.count_for_model("m").await.unwrap(), (0, 0));
    let (n, models) = s.count_stale("m").await.unwrap();
    assert_eq!(n, 0);
    assert!(models.is_empty());
}

#[tokio::test]
async fn missing_embedding_lists_only_unindexed_rows_for_current_model() {
    let (handle, _w) = fresh().await;
    let (tx, rx) = oneshot::channel();
    handle
        .writer()
        .send(WriteOp::BeginSession {
            session_id: "sess-mx".into(),
            daemon_pid: 1,
            ack: tx,
        })
        .await
        .unwrap();
    rx.await.unwrap().unwrap();
    let (tx, rx) = oneshot::channel();
    handle
        .writer()
        .send(WriteOp::AppendMessage {
            session_id: "sess-mx".into(),
            turn_id: None,
            msg: PersistedMessage::user("x"),
            ack: tx,
        })
        .await
        .unwrap();
    let conv_id = rx.await.unwrap().unwrap();

    // Two chunks: one indexed under "new", one indexed under "old".
    let _ = insert_chunk_with_vec(&handle, conv_id, 0, &unit_vec(0.0), "new").await;
    let _ = insert_chunk_with_vec(&handle, conv_id, 1, &unit_vec(0.5), "old").await;
    // One unindexed chunk (no embedding row at all).
    let (tx, rx) = oneshot::channel();
    handle
        .writer()
        .send(WriteOp::StoreChunk {
            conversation_id: conv_id,
            chunk_index: 2,
            content: "naked-chunk".into(),
            token_count: None,
            ack: tx,
        })
        .await
        .unwrap();
    let naked_chunk = rx.await.unwrap().unwrap();

    // Two memories: one indexed under "new", one bare.
    let (tx, rx) = oneshot::channel();
    handle
        .writer()
        .send(WriteOp::SaveMemory {
            key: "indexed".into(),
            value: "v1".into(),
            source_conversation_id: None,
            ack: tx,
        })
        .await
        .unwrap();
    let indexed_mem = rx.await.unwrap().unwrap();
    let (tx, rx) = oneshot::channel();
    handle
        .writer()
        .send(WriteOp::StoreMemoryEmbedding {
            memory_id: indexed_mem,
            model: "new".into(),
            dim: 2,
            vector: vector_to_blob(&unit_vec(0.0)),
            ack: tx,
        })
        .await
        .unwrap();
    rx.await.unwrap().unwrap();
    let (tx, rx) = oneshot::channel();
    handle
        .writer()
        .send(WriteOp::SaveMemory {
            key: "bare".into(),
            value: "v2".into(),
            source_conversation_id: None,
            ack: tx,
        })
        .await
        .unwrap();
    let bare_mem = rx.await.unwrap().unwrap();

    let s = SqliteSemanticStore::new(handle);
    // Under current = "new":
    // - Chunks missing: the "old"-indexed chunk + the naked one.
    // - Memories missing: just the bare memory.
    let chunks = s.chunks_missing_embedding("new").await.unwrap();
    assert_eq!(chunks.len(), 2);
    let chunk_contents: Vec<&str> = chunks.iter().map(|(_, t)| t.as_str()).collect();
    assert!(chunk_contents.contains(&"chunk1")); // old-indexed
    assert!(chunk_contents.contains(&"naked-chunk"));
    assert!(chunks.iter().any(|(id, _)| *id == naked_chunk));

    let memories = s.memories_missing_embedding("new").await.unwrap();
    assert_eq!(memories.len(), 1);
    assert_eq!(memories[0].0, bare_mem);
    assert_eq!(memories[0].1, "v2");

    // store_*_embedding should be idempotent: write under "new"
    // and the row drops out of the missing list.
    s.store_memory_embedding(
        bare_mem,
        "new".to_string(),
        2,
        vector_to_blob(&unit_vec(0.0)),
    )
    .await
    .unwrap();
    let memories = s.memories_missing_embedding("new").await.unwrap();
    assert!(memories.is_empty());
}

#[tokio::test]
async fn count_stale_aggregates_across_chunks_and_memories() {
    let (handle, _w) = fresh().await;
    // Create a conversation row to FK chunk inserts.
    let (tx, rx) = oneshot::channel();
    handle
        .writer()
        .send(WriteOp::BeginSession {
            session_id: "sess-stale".into(),
            daemon_pid: 1,
            ack: tx,
        })
        .await
        .unwrap();
    rx.await.unwrap().unwrap();
    let (tx, rx) = oneshot::channel();
    handle
        .writer()
        .send(WriteOp::AppendMessage {
            session_id: "sess-stale".into(),
            turn_id: None,
            msg: PersistedMessage::user("x"),
            ack: tx,
        })
        .await
        .unwrap();
    let conv_id = rx.await.unwrap().unwrap();

    // Two chunks under "old-A", one under "old-B", one under "new".
    let _ = insert_chunk_with_vec(&handle, conv_id, 0, &unit_vec(0.0), "old-A").await;
    let _ = insert_chunk_with_vec(&handle, conv_id, 1, &unit_vec(0.5), "old-A").await;
    let _ = insert_chunk_with_vec(&handle, conv_id, 2, &unit_vec(1.0), "old-B").await;
    let _ = insert_chunk_with_vec(&handle, conv_id, 3, &unit_vec(1.5), "new").await;

    // One memory embedding under "old-A".
    let (tx, rx) = oneshot::channel();
    handle
        .writer()
        .send(WriteOp::SaveMemory {
            key: "k".into(),
            value: "v".into(),
            source_conversation_id: None,
            ack: tx,
        })
        .await
        .unwrap();
    let mem_id = rx.await.unwrap().unwrap();
    let (tx, rx) = oneshot::channel();
    handle
        .writer()
        .send(WriteOp::StoreMemoryEmbedding {
            memory_id: mem_id,
            model: "old-A".into(),
            dim: 2,
            vector: vector_to_blob(&unit_vec(0.0)),
            ack: tx,
        })
        .await
        .unwrap();
    rx.await.unwrap().unwrap();

    let s = SqliteSemanticStore::new(handle);
    // Current = "new" → 2 chunk rows under old-A + 1 chunk under old-B
    // + 1 memory under old-A = 4 stale rows, two distinct models.
    let (n, models) = s.count_stale("new").await.unwrap();
    assert_eq!(n, 4);
    assert_eq!(models, vec!["old-A".to_string(), "old-B".to_string()]);

    // Switching current to "old-A" should leave only the "old-B"
    // chunk + the "new" chunk as stale = 2 rows, two models.
    let (n, models) = s.count_stale("old-A").await.unwrap();
    assert_eq!(n, 2);
    assert_eq!(models, vec!["new".to_string(), "old-B".to_string()]);
}

#[test]
fn vector_to_blob_round_trips() {
    let v = vec![0.5f32, -0.25, 0.125, 1e-6];
    let b = vector_to_blob(&v);
    let (words, _) = b.as_chunks::<4>();
    let decoded: Vec<f32> = words.iter().copied().map(f32::from_le_bytes).collect();
    assert_eq!(decoded, v);
}

#[test]
fn score_against_dim_mismatch_returns_none() {
    let q = vec![1.0f32, 0.0];
    let v_3d = vector_to_blob(&[1.0, 0.0, 0.0]);
    assert!(score_against(&q, &v_3d).is_none());
}
