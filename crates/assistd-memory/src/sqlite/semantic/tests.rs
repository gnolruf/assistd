use super::*;
use crate::sqlite::{SqliteMemoryStore, SqliteTranscriptStore};
use crate::{BranchId, MemoryStore, MessageId, PersistedMessage, PersistedRole, TranscriptStore};

/// The returned temp dir must outlive the stores.
async fn fresh() -> (SqliteHandle, SqliteSemanticStore, tempfile::TempDir) {
    let temp = tempfile::tempdir().unwrap();
    let handle = SqliteHandle::open(&temp.path().join("memory.db"))
        .await
        .unwrap();
    let store = SqliteSemanticStore::new(handle.clone());
    (handle, store, temp)
}

/// One saved session's main branch, to append user messages to.
struct Transcript {
    store: SqliteTranscriptStore,
    session: SessionId,
    branch: BranchId,
}

impl Transcript {
    async fn new(handle: &SqliteHandle) -> Self {
        let store = SqliteTranscriptStore::new(handle.clone());
        let session = SessionId::new();
        let branch = store
            .begin_session_with_main_branch(session, 0)
            .await
            .unwrap();
        Self {
            store,
            session,
            branch,
        }
    }

    /// Append user message `text`, returning its id and stored chunks.
    async fn append(&self, text: &str) -> (MessageId, Vec<(ChunkId, String)>) {
        let turn = self.store.begin_turn(self.session).await.unwrap();
        let appended = self
            .store
            .append_message_to_branch(self.branch, turn, PersistedMessage::user(text))
            .await
            .unwrap();
        (appended.id, appended.chunks)
    }

    /// Append user message `text`, short enough to be one chunk, and embed that chunk.
    async fn embedded_chunk(
        &self,
        store: &SqliteSemanticStore,
        text: &str,
        vector: &[f32],
        model: &str,
    ) -> ChunkId {
        let (_, chunks) = self.append(text).await;
        let [chunk] = &chunks[..] else {
            panic!("{text:?} should be one chunk, got {chunks:?}");
        };
        embed_chunk(store, chunk, vector, model).await;
        chunk.0
    }
}

/// 2-d unit vector at `angle` radians.
fn unit_vec(angle: f32) -> Vec<f32> {
    vec![angle.cos(), angle.sin()]
}

async fn embed_chunk(
    store: &SqliteSemanticStore,
    (chunk_id, text): &(ChunkId, String),
    vector: &[f32],
    model: &str,
) {
    store
        .store_embeddings(
            model,
            vec![Embedding {
                target: EmbeddingTarget::Chunk(*chunk_id),
                text: text.clone(),
                vector: vector.to_vec(),
            }],
        )
        .await
        .unwrap();
}

async fn save_memory(handle: &SqliteHandle, key: &str, value: &str) -> MemoryId {
    SqliteMemoryStore::new(handle.clone())
        .save(key, value.into())
        .await
        .unwrap()
        .id
}

async fn embed_memory(
    store: &SqliteSemanticStore,
    memory_id: MemoryId,
    value: &str,
    vector: &[f32],
    model: &str,
) {
    store
        .store_embeddings(
            model,
            vec![Embedding {
                target: EmbeddingTarget::Memory(memory_id),
                text: value.to_string(),
                vector: vector.to_vec(),
            }],
        )
        .await
        .unwrap();
}

#[tokio::test]
async fn nearest_chunks_ranks_by_similarity_and_returns_parent_message() {
    let (handle, store, _temp) = fresh().await;
    let transcript = Transcript::new(&handle).await;
    let long_text = "word ".repeat(150);
    let (long_message, long_chunks) = transcript.append(&long_text).await;
    let [first, second] = &long_chunks[..] else {
        panic!("expected two chunks, got {}", long_chunks.len());
    };
    embed_chunk(&store, first, &unit_vec(0.0), "m").await;
    embed_chunk(&store, second, &unit_vec(0.3), "m").await;
    let short = transcript
        .embedded_chunk(&store, "short", &unit_vec(1.5), "m")
        .await;

    let hits = store
        .nearest_chunks(unit_vec(0.0), 3, "m", None)
        .await
        .unwrap();
    let ids: Vec<ChunkId> = hits.iter().map(|h| h.chunk_id).collect();
    assert_eq!(ids, [first.0, second.0, short]);
    assert!(hits[0].similarity > hits[1].similarity);
    assert!(hits[1].similarity > hits[2].similarity);
    assert!((hits[0].similarity - 1.0).abs() < 1e-4);
    for hit in &hits[..2] {
        assert_eq!(hit.message_id, long_message);
        assert_eq!(hit.content, long_text);
    }
    assert_eq!(hits[2].content, "short");
    for hit in &hits {
        assert_eq!(hit.session_id, transcript.session);
        assert_eq!(hit.role, PersistedRole::User);
    }
}

#[tokio::test]
async fn nearest_chunks_keeps_the_best_top_k() {
    let (handle, store, _temp) = fresh().await;
    let transcript = Transcript::new(&handle).await;
    let mut ids = Vec::new();
    for i in 0..10 {
        let angle = (i as f32) * 0.1;
        ids.push(
            transcript
                .embedded_chunk(&store, &format!("m{i}"), &unit_vec(angle), "m")
                .await,
        );
    }

    for (top_k, expected) in [(3, &ids[..3]), (usize::MAX, &ids[..])] {
        let hits = store
            .nearest_chunks(unit_vec(0.0), top_k, "m", None)
            .await
            .unwrap();
        let got: Vec<ChunkId> = hits.iter().map(|h| h.chunk_id).collect();
        assert_eq!(got, expected, "top_k {top_k}");
    }
}

#[tokio::test]
async fn nearest_chunks_can_exclude_one_session() {
    let (handle, store, _temp) = fresh().await;
    let past = Transcript::new(&handle).await;
    let current = Transcript::new(&handle).await;
    let past_chunk = past
        .embedded_chunk(&store, "past", &unit_vec(0.4), "m")
        .await;
    let current_chunk = current
        .embedded_chunk(&store, "current", &unit_vec(0.0), "m")
        .await;

    let all = store
        .nearest_chunks(unit_vec(0.0), 5, "m", None)
        .await
        .unwrap();
    let all_ids: Vec<ChunkId> = all.iter().map(|h| h.chunk_id).collect();
    assert_eq!(
        all_ids,
        [current_chunk, past_chunk],
        "the excluded session must hold the closer match"
    );

    let others = store
        .nearest_chunks(unit_vec(0.0), 5, "m", Some(current.session))
        .await
        .unwrap();
    let other_ids: Vec<(ChunkId, SessionId)> =
        others.iter().map(|h| (h.chunk_id, h.session_id)).collect();
    assert_eq!(other_ids, [(past_chunk, past.session)]);
}

#[tokio::test]
async fn upsert_replaces_memory_embedding_in_place() {
    let (handle, store, _temp) = fresh().await;
    let mem_id = save_memory(&handle, "k", "v").await;
    embed_memory(&store, mem_id, "v", &unit_vec(0.0), "m").await;
    embed_memory(
        &store,
        mem_id,
        "v",
        &unit_vec(std::f32::consts::FRAC_PI_2),
        "m",
    )
    .await;

    assert_eq!(
        store.count_missing("m").await.unwrap(),
        UnindexedRows::default()
    );
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
async fn overwriting_a_memory_value_drops_its_stale_embedding() {
    let (handle, store, _temp) = fresh().await;
    let mem_id = save_memory(&handle, "editor", "vim").await;
    embed_memory(&store, mem_id, "vim", &unit_vec(0.0), "m").await;

    assert_eq!(save_memory(&handle, "editor", "emacs").await, mem_id);
    embed_memory(&store, mem_id, "vim", &unit_vec(0.0), "m").await;

    assert_eq!(
        store.count_missing("m").await.unwrap(),
        UnindexedRows {
            chunks: 0,
            memories: 1
        },
        "neither the trigger nor a late embedding of the old value may leave it indexed"
    );
    assert_eq!(
        store.nearest_memories(unit_vec(0.0), 5, "m").await.unwrap(),
        Vec::<MemoryHit>::new()
    );
    assert_eq!(
        store
            .memories_missing_embedding("m", None, 100)
            .await
            .unwrap(),
        [(mem_id, "emacs".to_string())]
    );
}

#[tokio::test]
async fn non_finite_vectors_never_enter_top_k() {
    let (handle, store, _temp) = fresh().await;
    let transcript = Transcript::new(&handle).await;
    transcript
        .embedded_chunk(&store, "nan", &[f32::NAN, 0.0], "m")
        .await;
    transcript
        .embedded_chunk(&store, "inf", &[f32::INFINITY, 0.0], "m")
        .await;
    let far = transcript
        .embedded_chunk(&store, "far", &unit_vec(1.5), "m")
        .await;
    let near = transcript
        .embedded_chunk(&store, "near", &unit_vec(0.1), "m")
        .await;

    let hits = store
        .nearest_chunks(unit_vec(0.0), 2, "m", None)
        .await
        .unwrap();
    let got: Vec<ChunkId> = hits.iter().map(|h| h.chunk_id).collect();
    assert_eq!(got, [near, far]);
    assert!(hits.iter().all(|h| h.similarity.is_finite()));
}

#[tokio::test]
async fn missing_embedding_lists_only_unindexed_rows_for_current_model() {
    let (handle, store, _temp) = fresh().await;
    let transcript = Transcript::new(&handle).await;

    transcript
        .embedded_chunk(&store, "x", &unit_vec(0.0), "new")
        .await;
    let old_chunk = transcript
        .embedded_chunk(&store, "y", &unit_vec(0.5), "old")
        .await;
    let (_, naked) = transcript.append("z").await;
    let naked_chunk = naked[0].0;

    let indexed_mem = save_memory(&handle, "indexed", "v1").await;
    embed_memory(&store, indexed_mem, "v1", &unit_vec(0.0), "new").await;
    let bare_mem = save_memory(&handle, "bare", "v2").await;

    assert_eq!(
        store
            .chunks_missing_embedding("new", None, 100)
            .await
            .unwrap(),
        [(old_chunk, "y".to_string()), (naked_chunk, "z".to_string())]
    );
    assert_eq!(
        store
            .chunks_missing_embedding("new", None, 1)
            .await
            .unwrap(),
        [(old_chunk, "y".to_string())]
    );
    assert_eq!(
        store
            .chunks_missing_embedding("new", Some(old_chunk), 100)
            .await
            .unwrap(),
        [(naked_chunk, "z".to_string())]
    );
    assert_eq!(
        store
            .memories_missing_embedding("new", None, 100)
            .await
            .unwrap(),
        [(bare_mem, "v2".to_string())]
    );

    embed_memory(&store, bare_mem, "v2", &unit_vec(0.0), "new").await;
    assert_eq!(
        store
            .memories_missing_embedding("new", None, 100)
            .await
            .unwrap(),
        Vec::<(MemoryId, String)>::new()
    );
}

#[tokio::test]
async fn count_stale_aggregates_across_chunks_and_memories() {
    let (handle, store, _temp) = fresh().await;
    let transcript = Transcript::new(&handle).await;

    for (text, angle, model) in [
        ("a", 0.0, "old-A"),
        ("b", 0.5, "old-A"),
        ("c", 1.0, "old-B"),
        ("d", 1.5, "new"),
    ] {
        transcript
            .embedded_chunk(&store, text, &unit_vec(angle), model)
            .await;
    }
    let mem_id = save_memory(&handle, "k", "v").await;
    embed_memory(&store, mem_id, "v", &unit_vec(0.0), "old-A").await;

    assert_eq!(
        store.count_stale("new").await.unwrap(),
        StaleEmbeddings {
            rows: 4,
            models: vec!["old-A".to_string(), "old-B".to_string()]
        },
        "2 old-A chunks + 1 old-B chunk + 1 old-A memory"
    );
    assert_eq!(
        store.count_stale("old-A").await.unwrap(),
        StaleEmbeddings {
            rows: 2,
            models: vec!["new".to_string(), "old-B".to_string()]
        },
        "the old-B chunk + the new chunk"
    );
}

#[tokio::test]
async fn batch_skips_targets_deleted_before_it_lands() {
    let (handle, store, _temp) = fresh().await;
    let kept = save_memory(&handle, "kept", "v").await;
    let deleted = save_memory(&handle, "deleted", "v").await;
    SqliteMemoryStore::new(handle.clone())
        .delete("deleted")
        .await
        .unwrap();

    let embeddings = [deleted, kept]
        .map(|memory_id| Embedding {
            target: EmbeddingTarget::Memory(memory_id),
            text: "v".to_string(),
            vector: unit_vec(0.0),
        })
        .to_vec();
    store.store_embeddings("m", embeddings).await.unwrap();

    let hits = store.nearest_memories(unit_vec(0.0), 5, "m").await.unwrap();
    let hit_ids: Vec<MemoryId> = hits.iter().map(|hit| hit.memory_id).collect();
    assert_eq!(hit_ids, [kept]);
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
