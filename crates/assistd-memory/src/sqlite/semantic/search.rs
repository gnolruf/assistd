//! The one search over chunks of every kind: their sources ranked by vector and by
//! keyword, the rankings fused by reciprocal rank, and a text several sources hold
//! shown once. The accesses to one file or page are one source.

use std::collections::{HashMap, HashSet};

use rusqlite::{Connection, Row};

use crate::chunking::TextHash;
use crate::sqlite::chunks::{CHUNK_SOURCES, kind_code, vector_table};
use crate::{
    ChunkId, ChunkSearch, ChunkSource, ContextChunk, SearchHit, SearchRanking, SessionId,
    SessionScope,
};

use super::{
    ACCESS_CHUNK_OWNERS, DOCUMENT_CHUNK_COLUMNS, DOCUMENT_CHUNK_OWNERS, MEMORY_CHUNK_COLUMNS,
    MESSAGE_CHUNK_COLUMNS, access_chunk_columns, access_context_chunk, document_context_chunk,
    json_id_array, memory_context_chunk, message_context_chunk, scan_top_k,
    score_against_quantised,
};

/// Sources each ranking brings to the fusion, when the search asks for no more hits
/// than that.
const CANDIDATES_PER_RANKING: usize = 30;

/// Texts the vector ranking takes from a scan of the vectors alone, for each document
/// or access it is to find among them. When the document texts sit in too few
/// documents, it scans by chunk.
const NEAREST_TEXTS_PER_CANDIDATE: usize = 10;

/// Sources the keyword ranking reads for each one it is to place, so that sources
/// repeating one text do not use up its places.
const KEYWORD_SOURCES_PER_CANDIDATE: usize = 4;

/// The `chunks.kind` code of a memory's chunk, which is never folded into another hit.
const MEMORY_KIND: i64 = kind_code(ChunkSource::Memory);

/// The `chunks.kind` code of an access's chunk.
const ACCESS_KIND: i64 = kind_code(ChunkSource::Access);

/// Added to a rank before it is inverted into a score. The smaller it is, the more a
/// top rank in one ranking outweighs middling ranks in both.
const RANK_OFFSET: f32 = 10.0;

/// The ids of the messages of session `?1`.
const SESSION_MESSAGES: &str = "
    SELECT m.id FROM sessions s JOIN messages m ON m.session_id = s.id
    WHERE s.uuid = ?1";

/// The ids of the messages of session `?1` in its live context: all but those its
/// current branch's summary folded.
const SESSION_CONTEXT_MESSAGES: &str = "
    SELECT m.id FROM sessions s JOIN messages m ON m.session_id = s.id
    WHERE s.uuid = ?1
      AND m.id NOT IN (
          SELECT bm.message_id
          FROM branch_messages bm
          JOIN branch_summaries bs ON bs.branch_id = bm.branch_id
          WHERE bm.branch_id = s.current_branch_id AND bm.seq < bs.kept_from_seq)";

/// The message an access chunk `cc` is stored with.
const ACCESS_CHUNK_MESSAGE: &str = "(SELECT a.message_id FROM accesses a WHERE a.id = cc.source)";

/// The vector of every text the document chunks hold, each text its own group.
const DOCUMENT_TEXT_VECTORS: &str =
    "SELECT hash, hash, vector FROM document_vectors WHERE vector IS NOT NULL";

/// The vector of every text the access chunks hold, each text its own group.
const ACCESS_TEXT_VECTORS: &str =
    "SELECT hash, hash, vector FROM access_vectors WHERE vector IS NOT NULL";

/// Which messages a search held to one session, or away from it, keeps.
#[derive(Clone, Copy)]
enum SessionTest {
    /// Only the session's messages.
    Inside,
    /// Only the messages of other sessions.
    Outside,
    /// Every message but those in the session's live context.
    OutsideContext,
}

impl SessionTest {
    /// The SQL test that `message`, an expression for a message's id, passes when the
    /// search keeps that message; the session is bound to `?1`.
    fn kept(self, message: &str) -> String {
        match self {
            Self::Inside => format!("{message} IN ({SESSION_MESSAGES})"),
            Self::Outside => format!("{message} NOT IN ({SESSION_MESSAGES})"),
            Self::OutsideContext => format!("{message} NOT IN ({SESSION_CONTEXT_MESSAGES})"),
        }
    }
}

/// The chunks a search may return: those of its kinds, the message and access chunks
/// among them held to its sessions.
struct Searched {
    /// In the order of their `chunks.kind` codes, each once.
    kinds: Vec<ChunkSource>,
    /// The test the message of a message or access chunk must pass and the session it
    /// binds to `?1`.
    sessions: Option<(SessionTest, SessionId)>,
}

impl Searched {
    fn of(search: &ChunkSearch) -> Self {
        let sessions = match search.sessions {
            SessionScope::All => None,
            SessionScope::AllExcept(session) => Some((SessionTest::Outside, session)),
            SessionScope::AllExceptInContext(session) => {
                Some((SessionTest::OutsideContext, session))
            }
            SessionScope::Only(session) => Some((SessionTest::Inside, session)),
        };
        Self {
            kinds: CHUNK_SOURCES
                .into_iter()
                .filter(|kind| search.kinds.contains(kind))
                .collect(),
            sessions,
        }
    }

    /// The session statements holding [`Self::chunk_test`] bind to `?1`.
    fn session(&self) -> Option<SessionId> {
        self.sessions.map(|(_, session)| session)
    }

    /// The SQL test a chunk `cc` of `kind` passes when its session is searched; `None`
    /// when every chunk of that kind is.
    fn session_test(&self, kind: ChunkSource) -> Option<String> {
        let (test, _) = self.sessions?;
        match kind {
            ChunkSource::Message => Some(test.kept("cc.source")),
            ChunkSource::Access => Some(test.kept(ACCESS_CHUNK_MESSAGE)),
            ChunkSource::Memory | ChunkSource::Document => None,
        }
    }

    /// The SQL test a chunk `cc` passes when it is of `kind` and the search may
    /// return it.
    fn kind_test(&self, kind: ChunkSource) -> String {
        let code = kind_code(kind);
        match self.session_test(kind) {
            Some(in_sessions) => format!("(cc.kind = {code} AND {in_sessions})"),
            None => format!("cc.kind = {code}"),
        }
    }

    /// The SQL test a chunk `cc` passes when the search may return it.
    fn chunk_test(&self) -> String {
        let tests: Vec<String> = self
            .kinds
            .iter()
            .map(|kind| self.kind_test(*kind))
            .collect();
        format!("({})", tests.join(" OR "))
    }
}

/// What a search places once, however many of its chunks match.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Source {
    /// A message, a memory or a document: a `chunks.kind` code and a `chunks.source`.
    Row { kind: i64, id: i64 },
    /// Every access to one file or page, by the hash of the text they all hold.
    Target(TextHash),
}

/// A source as one ranking places it: through the chunk that ranks best there.
struct Candidate {
    source: Source,
    chunk: ChunkId,
}

/// A candidate of the vector ranking with its chunk's cosine to the query.
struct Near {
    candidate: Candidate,
    similarity: f32,
}

/// A source's standing across the rankings.
struct Fused {
    /// The chunk from the ranking that places the source highest.
    chunk: ChunkId,
    chunk_rank: usize,
    score: f32,
}

/// A chunk a search shows, with how many other sources hold its text.
struct Shown {
    chunk: ChunkId,
    held_elsewhere: usize,
}

/// The hits of `search`, best first.
pub(super) fn search(conn: &Connection, search: &ChunkSearch) -> rusqlite::Result<Vec<SearchHit>> {
    let searched = Searched::of(search);
    if searched.kinds.is_empty() {
        return Ok(Vec::new());
    }
    let candidates = search.top_k.max(CANDIDATES_PER_RANKING);
    let by_keyword = match search.ranking {
        SearchRanking::VectorAndKeyword => {
            keyword_ranking(conn, &searched, &search.query, candidates)?
        }
        SearchRanking::Vector => Vec::new(),
    };
    let by_vector = vector_ranking(conn, &searched, &search.query_vector, candidates)?;
    let fused = fuse([&by_keyword, &by_vector]);
    let shown = distinct_texts(conn, &searched, &fused, search.top_k)?;
    hits(conn, &searched, &search.query_vector, &shown)
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

/// The first `limit` sources whose chunks hold words of `query`, by their best
/// chunk's BM25. A text the best chunks of several sources hold counts for the
/// first of them alone, among [`KEYWORD_SOURCES_PER_CANDIDATE`] times `limit` sources;
/// every memory counts, and the accesses to one target are placed through the first.
fn keyword_ranking(
    conn: &Connection,
    searched: &Searched,
    query: &str,
    limit: usize,
) -> rusqlite::Result<Vec<Candidate>> {
    let Some(expression) = match_expression(query) else {
        return Ok(Vec::new());
    };
    let searched_chunk = searched.chunk_test();
    let mut ranked = conn.prepare_cached(&format!(
        "SELECT cc.kind, cc.source, f.rowid AS chunk, cc.hash, min(f.rank) AS best
         FROM chunk_words f JOIN chunks cc ON cc.id = f.rowid
         WHERE chunk_words MATCH ?2 AND {searched_chunk}
         GROUP BY cc.kind, CASE cc.kind WHEN {ACCESS_KIND} THEN cc.hash ELSE cc.source END
         ORDER BY best, chunk
         LIMIT ?3"
    ))?;
    let sources = limit.saturating_mul(KEYWORD_SOURCES_PER_CANDIDATE);
    let mut rows = ranked.query(rusqlite::params![
        searched.session(),
        expression,
        sql_limit(sources)
    ])?;
    let mut texts: HashSet<TextHash> = HashSet::new();
    let mut candidates = Vec::new();
    while let Some(row) = rows.next()? {
        if candidates.len() == limit {
            break;
        }
        let kind: i64 = row.get(0)?;
        let text: TextHash = row.get(3)?;
        if kind == ACCESS_KIND {
            if let Some(chunk) = first_access_holding(conn, searched, text)? {
                candidates.push(Candidate {
                    source: Source::Target(text),
                    chunk,
                });
            }
        } else if kind == MEMORY_KIND || texts.insert(text) {
            candidates.push(Candidate {
                source: Source::Row {
                    kind,
                    id: row.get(1)?,
                },
                chunk: row.get(2)?,
            });
        }
    }
    Ok(candidates)
}

/// The first-stored access chunk the search may return that holds `text`.
fn first_access_holding(
    conn: &Connection,
    searched: &Searched,
    text: TextHash,
) -> rusqlite::Result<Option<ChunkId>> {
    let searched_access = searched.kind_test(ChunkSource::Access);
    conn.prepare_cached(&format!(
        "SELECT min(cc.id) FROM chunks cc WHERE cc.hash = ?2 AND {searched_access}"
    ))?
    .query_row(rusqlite::params![searched.session(), text], |row| {
        row.get(0)
    })
}

/// `limit` as a `LIMIT` operand.
fn sql_limit(limit: usize) -> i64 {
    i64::try_from(limit).unwrap_or(i64::MAX)
}

/// The first `limit` sources nearest `query` across the searched kinds, by their best
/// chunk's cosine; of chunks equally near, the earlier stored goes first.
fn vector_ranking(
    conn: &Connection,
    searched: &Searched,
    query: &[f32],
    limit: usize,
) -> rusqlite::Result<Vec<Candidate>> {
    if query.is_empty() {
        return Ok(Vec::new());
    }
    let mut nearest = Vec::new();
    for kind in &searched.kinds {
        nearest.extend(match kind {
            ChunkSource::Message | ChunkSource::Memory => {
                nearest_by_chunk(conn, searched, *kind, query, limit)?
            }
            ChunkSource::Document => nearest_documents(conn, searched, query, limit)?,
            ChunkSource::Access => nearest_accesses(conn, searched, query, limit)?,
        });
    }
    nearest.sort_by(|a, b| {
        b.similarity
            .total_cmp(&a.similarity)
            .then(a.candidate.chunk.cmp(&b.candidate.chunk))
    });
    nearest.truncate(limit);
    Ok(nearest.into_iter().map(|near| near.candidate).collect())
}

/// The `limit` messages, memories or documents nearest `query`, from a scan of the
/// vector of every chunk of `kind` the search may return, which keeps one chunk a
/// source however many near texts one source holds.
fn nearest_by_chunk(
    conn: &Connection,
    searched: &Searched,
    kind: ChunkSource,
    query: &[f32],
    limit: usize,
) -> rusqlite::Result<Vec<Near>> {
    let code = kind_code(kind);
    let chunk_vectors = format!(
        "SELECT cc.source, cc.id, v.vector
         FROM chunks cc JOIN {vectors} v ON v.hash = cc.hash
         WHERE {searched_kind} AND v.vector IS NOT NULL",
        vectors = vector_table(kind),
        searched_kind = searched.kind_test(kind),
    );
    let ranked = if searched.session_test(kind).is_some() {
        scan_top_k::<i64, ChunkId>(conn, &chunk_vectors, [searched.session()], query, limit)?
    } else {
        scan_top_k::<i64, ChunkId>(conn, &chunk_vectors, [], query, limit)?
    };
    let near = |ranked: super::Ranked<i64, ChunkId>| Near {
        candidate: Candidate {
            source: Source::Row {
                kind: code,
                id: ranked.group,
            },
            chunk: ranked.id,
        },
        similarity: ranked.similarity,
    };
    Ok(ranked.into_iter().map(near).collect())
}

/// The `limit` files and pages nearest `query`, each through the first access to it
/// the search may return.
fn nearest_accesses(
    conn: &Connection,
    searched: &Searched,
    query: &[f32],
    limit: usize,
) -> rusqlite::Result<Vec<Near>> {
    let texts = limit.saturating_mul(NEAREST_TEXTS_PER_CANDIDATE);
    let nearest = scan_top_k::<TextHash, TextHash>(conn, ACCESS_TEXT_VECTORS, [], query, texts)?;
    let mut accesses = Vec::new();
    for text in nearest {
        if accesses.len() == limit {
            break;
        }
        if let Some(chunk) = first_access_holding(conn, searched, text.id)? {
            accesses.push(Near {
                candidate: Candidate {
                    source: Source::Target(text.id),
                    chunk,
                },
                similarity: text.similarity,
            });
        }
    }
    Ok(accesses)
}

/// The `limit` documents nearest `query`.
fn nearest_documents(
    conn: &Connection,
    searched: &Searched,
    query: &[f32],
    limit: usize,
) -> rusqlite::Result<Vec<Near>> {
    let texts = limit.saturating_mul(NEAREST_TEXTS_PER_CANDIDATE);
    let nearest = scan_top_k::<TextHash, TextHash>(conn, DOCUMENT_TEXT_VECTORS, [], query, texts)?;
    let more_texts_unscanned = nearest.len() == texts;
    let holders = documents_holding(conn, nearest, limit)?;
    if holders.len() < limit && more_texts_unscanned {
        return nearest_by_chunk(conn, searched, ChunkSource::Document, query, limit);
    }
    Ok(holders)
}

/// For each of `texts` in their order, the first document holding it that holds none
/// before it, through its first chunk of that text; `limit` documents at most.
fn documents_holding(
    conn: &Connection,
    texts: Vec<super::Ranked<TextHash, TextHash>>,
    limit: usize,
) -> rusqlite::Result<Vec<Near>> {
    let mut holders = conn.prepare_cached(
        "SELECT source, id FROM chunks
         WHERE hash = ?1 AND kind = 2
         ORDER BY id",
    )?;
    let mut placed: HashSet<i64> = HashSet::new();
    let mut nearest = Vec::new();
    for text in texts {
        if nearest.len() == limit {
            break;
        }
        let mut rows = holders.query(rusqlite::params![text.id])?;
        while let Some(row) = rows.next()? {
            let document: i64 = row.get(0)?;
            if placed.insert(document) {
                nearest.push(Near {
                    candidate: Candidate {
                        source: Source::Row {
                            kind: kind_code(ChunkSource::Document),
                            id: document,
                        },
                        chunk: row.get(1)?,
                    },
                    similarity: text.similarity,
                });
                break;
            }
        }
    }
    Ok(nearest)
}

/// One chunk per source in `rankings`, best first by the sum of the source's
/// reciprocal ranks: the chunk from the ranking that places it highest. An earlier
/// ranking wins a tie, both for the order and for the chunk.
fn fuse(rankings: [&[Candidate]; 2]) -> Vec<ChunkId> {
    let mut fused: Vec<Fused> = Vec::new();
    let mut placed: HashMap<Source, usize> = HashMap::new();
    for ranking in rankings {
        for (rank, candidate) in (1usize..).zip(ranking) {
            let score = reciprocal(rank);
            match placed.get(&candidate.source) {
                Some(&index) => {
                    let entry = &mut fused[index];
                    entry.score += score;
                    if rank < entry.chunk_rank {
                        entry.chunk = candidate.chunk;
                        entry.chunk_rank = rank;
                    }
                }
                None => {
                    placed.insert(candidate.source, fused.len());
                    fused.push(Fused {
                        chunk: candidate.chunk,
                        chunk_rank: rank,
                        score,
                    });
                }
            }
        }
    }
    fused.sort_by(|a, b| b.score.total_cmp(&a.score));
    fused.into_iter().map(|entry| entry.chunk).collect()
}

/// What a source at `rank` in one ranking adds to its fused score.
fn reciprocal(rank: usize) -> f32 {
    let rank = u16::try_from(rank).unwrap_or(u16::MAX);
    1.0 / (RANK_OFFSET + f32::from(rank))
}

/// The first `top_k` of `chunks`, in their order, that hold a text none before them
/// holds, each with how many other searched sources hold its text, which for an access
/// is how many other searched sessions touched its target; chunks deleted since they
/// were ranked are skipped, and so are those whose message, memory or access is gone.
/// A memory is always shown and never counted.
fn distinct_texts(
    conn: &Connection,
    searched: &Searched,
    chunks: &[ChunkId],
    top_k: usize,
) -> rusqlite::Result<Vec<Shown>> {
    let texts = conn
        .prepare_cached(
            "SELECT cc.id, cc.kind, cc.hash FROM chunks cc
             WHERE cc.id IN (SELECT value FROM json_each(?1))
               AND CASE cc.kind
                   WHEN 0 THEN EXISTS (SELECT 1 FROM messages m WHERE m.id = cc.source)
                   WHEN 1 THEN EXISTS (SELECT 1 FROM memories mem WHERE mem.id = cc.source)
                   WHEN 3 THEN EXISTS (SELECT 1 FROM accesses a WHERE a.id = cc.source)
                   ELSE 1 END",
        )?
        .query_map(rusqlite::params![json_id_array(chunks)], |row| {
            Ok((row.get(0)?, (row.get(1)?, row.get(2)?)))
        })?
        .collect::<rusqlite::Result<HashMap<ChunkId, (i64, TextHash)>>>()?;
    let mut seen: HashSet<TextHash> = HashSet::new();
    let mut shown = Vec::new();
    for chunk in chunks {
        if shown.len() == top_k {
            break;
        }
        let Some((kind, text)) = texts.get(chunk) else {
            continue;
        };
        if *kind == MEMORY_KIND {
            shown.push(Shown {
                chunk: *chunk,
                held_elsewhere: 0,
            });
        } else if seen.insert(*text) {
            let holding = if *kind == ACCESS_KIND {
                sessions_touching(conn, searched, *text)?
            } else {
                sources_holding(conn, searched, *text)?
            };
            shown.push(Shown {
                chunk: *chunk,
                held_elsewhere: holding.saturating_sub(1),
            });
        }
    }
    Ok(shown)
}

/// How many searched messages and documents hold `text`.
fn sources_holding(
    conn: &Connection,
    searched: &Searched,
    text: TextHash,
) -> rusqlite::Result<usize> {
    conn.prepare_cached(&format!(
        "SELECT count(*) FROM (SELECT DISTINCT cc.kind, cc.source FROM chunks cc
                               WHERE cc.hash = ?2
                                 AND cc.kind NOT IN ({MEMORY_KIND}, {ACCESS_KIND})
                                 AND {})",
        searched.chunk_test()
    ))?
    .query_row(rusqlite::params![searched.session(), text], |r| r.get(0))
}

/// How many sessions hold a searched access whose text is `text`.
fn sessions_touching(
    conn: &Connection,
    searched: &Searched,
    text: TextHash,
) -> rusqlite::Result<usize> {
    conn.prepare_cached(&format!(
        "SELECT count(DISTINCT m.session_id)
         FROM chunks cc
         JOIN accesses a ON a.id = cc.source
         JOIN messages m ON m.id = a.message_id
         WHERE cc.hash = ?2 AND {}",
        searched.kind_test(ChunkSource::Access)
    ))?
    .query_row(rusqlite::params![searched.session(), text], |r| r.get(0))
}

/// A hit for each of `shown`, in their order, with its similarity to `query` when
/// its text has a vector.
fn hits(
    conn: &Connection,
    searched: &Searched,
    query: &[f32],
    shown: &[Shown],
) -> rusqlite::Result<Vec<SearchHit>> {
    if shown.is_empty() {
        return Ok(Vec::new());
    }
    let chunks = json_id_array(shown.iter().map(|shown| shown.chunk));
    let mut found: HashMap<ChunkId, (ContextChunk, Option<f32>)> = HashMap::new();
    for kind in &searched.kinds {
        let (select, vector_column, to_chunk) = chunk_rows(*kind);
        let mut rows = conn.prepare_cached(&select)?;
        let mut rows = rows.query(rusqlite::params![chunks])?;
        while let Some(row) = rows.next()? {
            let chunk = to_chunk(row)?;
            let vector = row.get_ref(vector_column)?.as_blob_or_null()?;
            let similarity = vector.and_then(|blob| score_against_quantised(query, blob));
            found.insert(chunk.chunk_id, (chunk, similarity));
        }
    }
    let hit = |shown: &Shown| {
        let (chunk, similarity) = found.remove(&shown.chunk)?;
        Some(SearchHit {
            chunk_id: chunk.chunk_id,
            locator: chunk.locator,
            text: chunk.text,
            similarity,
            held_elsewhere: shown.held_elsewhere,
        })
    };
    Ok(shown.iter().filter_map(hit).collect())
}

type ToContextChunk = fn(&Row<'_>) -> rusqlite::Result<ContextChunk>;

/// The statement reading the chunks of `kind` among the ids bound to `?1` as a JSON
/// array, each row followed by its text's vector; that vector's column; and the
/// reader of the columns before it.
fn chunk_rows(kind: ChunkSource) -> (String, usize, ToContextChunk) {
    let access_columns = access_chunk_columns();
    let (columns, owners, vector_column, to_chunk): (_, _, _, ToContextChunk) = match kind {
        ChunkSource::Access => (
            access_columns.as_str(),
            ACCESS_CHUNK_OWNERS,
            8,
            access_context_chunk,
        ),
        ChunkSource::Message => (
            MESSAGE_CHUNK_COLUMNS,
            "JOIN messages m ON m.id = cc.source JOIN sessions s ON s.id = m.session_id",
            8,
            message_context_chunk,
        ),
        ChunkSource::Memory => (
            MEMORY_CHUNK_COLUMNS,
            "JOIN memories mem ON mem.id = cc.source",
            4,
            memory_context_chunk,
        ),
        ChunkSource::Document => (
            DOCUMENT_CHUNK_COLUMNS,
            DOCUMENT_CHUNK_OWNERS,
            9,
            document_context_chunk,
        ),
    };
    let select = format!(
        "SELECT {columns}, v.vector
         FROM chunks cc {owners}
         LEFT JOIN {vectors} v ON v.hash = cc.hash
         WHERE cc.kind = {code} AND cc.id IN (SELECT value FROM json_each(?1))",
        vectors = vector_table(kind),
        code = kind_code(kind),
    );
    (select, vector_column, to_chunk)
}
