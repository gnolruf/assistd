//! Document search: the documents' ranking by vector and their ranking by keyword,
//! fused by reciprocal rank.

use std::collections::HashMap;
use std::ffi::OsString;
use std::os::unix::ffi::OsStringExt;
use std::path::PathBuf;

use rusqlite::{Connection, Row};

use crate::sqlite::timestamp::timestamp_at;
use crate::{ChunkId, DocumentHit, DocumentId};

use super::{
    DOCUMENT_CHUNK_COLUMNS, DOCUMENT_CHUNK_OWNERS, DOCUMENT_CHUNK_VECTORS, json_id_array,
    scan_best, score_against_quantised,
};

/// Documents each ranking brings to the fusion.
const CANDIDATES_PER_RANKING: usize = 30;

/// Added to a rank before it is inverted into a score. The smaller it is, the more a
/// top rank in one ranking outweighs middling ranks in both.
const RANK_OFFSET: f32 = 10.0;

/// Where [`DOCUMENT_CHUNK_COLUMNS`] is followed by the chunk's vector.
const VECTOR_COLUMN: usize = 9;

/// A document as one ranking places it: through the chunk that ranks best there.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Candidate {
    document: DocumentId,
    chunk: ChunkId,
}

/// A document's standing across the rankings.
struct Fused {
    document: DocumentId,
    /// The chunk from the ranking that places the document highest.
    chunk: ChunkId,
    chunk_rank: u16,
    score: f32,
}

/// The `top_k` documents for `query`, best first: those whose chunks hold its words
/// and those whose chunks are nearest `query_vector`, fused by reciprocal rank.
pub(super) fn search(
    conn: &Connection,
    query: &str,
    query_vector: &[f32],
    top_k: usize,
) -> rusqlite::Result<Vec<DocumentHit>> {
    let by_keyword = keyword_ranking(conn, query)?;
    let by_vector = vector_ranking(conn, query_vector)?;
    let mut chunks = fuse([&by_keyword, &by_vector]);
    chunks.truncate(top_k);
    document_hits(conn, query_vector, &chunks)
}

/// The FTS5 query for a chunk holding any word of `query`, or `None` when it has no
/// word. Each word is quoted, which makes it a phrase of its own tokens and keeps
/// FTS5 from reading any of it as syntax.
fn match_expression(query: &str) -> Option<String> {
    let phrases: Vec<String> = query
        .split_whitespace()
        .filter(|word| word.chars().any(char::is_alphanumeric))
        .map(|word| format!("\"{}\"", word.replace('"', "\"\"")))
        .collect();
    (!phrases.is_empty()).then(|| phrases.join(" OR "))
}

/// The documents whose chunks hold words of `query`, by their best chunk's BM25.
fn keyword_ranking(conn: &Connection, query: &str) -> rusqlite::Result<Vec<Candidate>> {
    let Some(expression) = match_expression(query) else {
        return Ok(Vec::new());
    };
    conn.prepare_cached(
        "SELECT cc.document_id, f.rowid, min(f.rank) AS best
         FROM document_chunk_fts f JOIN chunks cc ON cc.id = f.rowid
         WHERE document_chunk_fts MATCH ?1
         GROUP BY cc.document_id
         ORDER BY best
         LIMIT ?2",
    )?
    .query_map(
        rusqlite::params![expression, CANDIDATES_PER_RANKING],
        |row| {
            Ok(Candidate {
                document: row.get(0)?,
                chunk: row.get(1)?,
            })
        },
    )?
    .collect()
}

/// The documents nearest `query`, by their best chunk's cosine.
fn vector_ranking(conn: &Connection, query: &[f32]) -> rusqlite::Result<Vec<Candidate>> {
    let nearest = scan_best::<DocumentId, ChunkId>(
        conn,
        DOCUMENT_CHUNK_VECTORS,
        [],
        query,
        score_against_quantised,
        CANDIDATES_PER_RANKING,
    )?;
    Ok(nearest
        .into_best_first()
        .into_iter()
        .map(|ranked| Candidate {
            document: ranked.group,
            chunk: ranked.id,
        })
        .collect())
}

/// One chunk per document in `rankings`, best first by the sum of the document's
/// reciprocal ranks: the chunk from the ranking that places it highest. An earlier
/// ranking wins a tie, both for the order and for the chunk.
fn fuse(rankings: [&[Candidate]; 2]) -> Vec<ChunkId> {
    let mut fused: Vec<Fused> = Vec::new();
    for ranking in rankings {
        for (rank, candidate) in (1u16..).zip(ranking) {
            let score = 1.0 / (RANK_OFFSET + f32::from(rank));
            match fused
                .iter_mut()
                .find(|entry| entry.document == candidate.document)
            {
                Some(entry) => {
                    entry.score += score;
                    if rank < entry.chunk_rank {
                        entry.chunk = candidate.chunk;
                        entry.chunk_rank = rank;
                    }
                }
                None => fused.push(Fused {
                    document: candidate.document,
                    chunk: candidate.chunk,
                    chunk_rank: rank,
                    score,
                }),
            }
        }
    }
    fused.sort_by(|a, b| b.score.total_cmp(&a.score));
    fused.into_iter().map(|entry| entry.chunk).collect()
}

/// A hit for each of `chunks`, in their order, with its similarity to `query` when
/// it has a vector; chunks deleted since they were ranked are skipped.
fn document_hits(
    conn: &Connection,
    query: &[f32],
    chunks: &[ChunkId],
) -> rusqlite::Result<Vec<DocumentHit>> {
    if chunks.is_empty() {
        return Ok(Vec::new());
    }
    let select = format!(
        "SELECT {DOCUMENT_CHUNK_COLUMNS}, e.vector
         FROM chunks cc {DOCUMENT_CHUNK_OWNERS}
         LEFT JOIN document_chunk_embeddings e ON e.chunk_id = cc.id
         WHERE cc.id IN (SELECT value FROM json_each(?1))"
    );
    let mut found = conn
        .prepare_cached(&select)?
        .query_map(rusqlite::params![json_id_array(chunks)], |row| {
            let hit = document_hit(row, query)?;
            Ok((hit.chunk_id, hit))
        })?
        .collect::<rusqlite::Result<HashMap<_, _>>>()?;
    Ok(chunks
        .iter()
        .filter_map(|chunk| found.remove(chunk))
        .collect())
}

fn document_hit(row: &Row<'_>, query: &[f32]) -> rusqlite::Result<DocumentHit> {
    let vector = row.get_ref(VECTOR_COLUMN)?.as_blob_or_null()?;
    Ok(DocumentHit {
        chunk_id: row.get(0)?,
        document_id: row.get(1)?,
        path: PathBuf::from(OsString::from_vec(row.get(2)?)),
        page: row.get(3)?,
        page_count: row.get(4)?,
        bytes: row.get(5)?..row.get(6)?,
        indexed_at: timestamp_at(row, 7)?,
        text: row.get(8)?,
        similarity: vector.and_then(|blob| score_against_quantised(query, blob)),
    })
}
