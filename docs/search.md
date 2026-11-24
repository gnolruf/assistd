# Search

The daemon has one search over everything it stores. It ranks four
kinds of text together, by meaning and by the query's exact words:

| Kind        | What is ranked                                                         | Searched with       |
|-------------|------------------------------------------------------------------------|---------------------|
| `sessions`  | The user and assistant messages of saved sessions.                     | `memory.enabled`    |
| `memories`  | The values saved by `remember` and `assistd memory`.                   | `memory.enabled`    |
| `accesses`  | The files that sessions' tool calls read or wrote and the web pages they fetched, by path, or by title and URL. | `memory.enabled`    |
| `documents` | The files in the folders `documents.roots` lists ([document search](documents.md)). | `documents.enabled` |

Every kind needs the embedding server (`[embedding] enabled`, on by
default): a search embeds its query, and reports why when it cannot.

## The commands

The model searches with one command behind its `run` tool, registered
when either setting above is on:

```text
search [-n N] [--in <kinds>] <query...>       # TSV hits: kind, place, similarity, passage
search [-n N] --session <id> <query...>       # the hits of one past session, each in its exchange
```

The words after the options are one query, so nothing is quoted. `-n`
takes 1 to 20 hits and defaults to `embedding.top_k`. `--in` holds the
search to the kinds named, separated by commas; naming a kind whose
setting is off is an error that names the setting. The system prompt
says in one line what `search` covers.

The same search runs from a terminal:

| Command                            | Kinds                                    |
|------------------------------------|------------------------------------------|
| `assistd search <words>`           | every kind turned on, or those `--in` names |
| `assistd memory reminisce <words>` | `sessions`                               |
| `assistd documents search <words>` | `documents`                              |

They print what the model's command prints, without the header, and
exit 1 when nothing matched. `assistd search` ranks every saved
session. The model's `search` leaves out the messages of the session
in progress that are still in its context, and ranks those a compaction
summary replaced.

Neither asks the user first. Both count as reading this machine, so a
later `web fetch` in the same session asks before it is sent
([web search](web-search.md)).

## What a hit shows

Output starts with a header marking everything below it as untrusted
data, not instructions. Then comes one line per hit, best first, as
tab-separated fields: the kind, where the passage is, its similarity
of meaning, and the passage with its whitespace collapsed.

```text
session   3f9c2a1e "Lease questions" 2026-11-19T09:14:02Z assistant   sim=81%   The renewal clause gives sixty days…
memory    landlord_contact   sim=74%   Mara Lindqvist, 555-0142
document  "/home/me/papers/lease.pdf" page 3 of 12 bytes 512..1024 indexed 2026-11-19   sim=n/a   also in 1 other place   INV-2024-0113 …
access    3f9c2a1e "Lease questions" 2026-11-19T09:15:40Z fetched   sim=68%   Tenancy renewal rules https://example.org/renewal
```

| Kind       | The place                                               | Reading on                                              |
|------------|---------------------------------------------------------|---------------------------------------------------------|
| `session`  | The session's short id and title, the message's time and role. | `search --session <id> <words>`                  |
| `memory`   | The key.                                                | The passage is the whole value.                         |
| `document` | The path, the page of the passage, its byte range within the page and the date the file was indexed. | `docs read --page P --offset START "<path>"` |
| `access`   | The session that touched the file or page, the call's time, and `read`, `wrote` or `fetched`. | The passage is the path, or the page's title and URL. |

- `sim=n/a` marks a passage found by its words that has no vector yet,
  or whose text the embedding model refused.
- `also in N other places` precedes a passage that other messages or
  documents hold too. For an access the count is of other sessions.
- A message of the session in progress has `this one, before its
  summary` for its place.
- All of a search's output fits 8 KiB, or one tool result when that is
  less. Hits that would overflow it are dropped from the bottom and
  counted in a closing line.

`search --session` prints the session's id and title, then each hit
between the chunks before and after it, as lines of `[time role
sim=N%] text`. Hits that share a neighbour are merged into one
exchange. It takes the id a session hit gave, refuses the session in
progress, and is recorded as an access to that session.

## How hits are ranked

Embeddings match meaning and miss exact strings: an identifier, an
error message, a name, an invoice number. So the search ranks its
sources twice and fuses the two rankings. A source is one message, one
memory or one document, or one file or page however many accesses
touched it.

- **By vector**: each source by its chunk nearest the query vector.
  Every vector of each kind searched is scanned; there is no vector
  index ([database.md](database.md#scans)).
- **By keyword**: each source by its chunk with the best BM25 score,
  among the chunks of the kinds searched that hold any word of the
  query. Words are split at every character that is not a letter or a
  digit and compared without case or accents. Each
  whitespace-separated word of the query is matched as a phrase of its
  own tokens, so `INV-2024-0113` finds only that number and no
  character of a query is read as search syntax.
- **Fused**: the 30 best sources of each ranking, or as many as the
  hits asked for when that is more, are scored by reciprocal rank
  fusion, the sum of `1 / (10 + rank)` over the rankings that hold the
  source. That needs no common scale between cosine and BM25. A source
  is shown through its chunk from the ranking that places it higher. At
  equal score or rank the keyword ranking wins.

The kinds share one embedding space and one stored form of vector, so
their similarities compare, and a memory can outrank a document.

A chunk needs no vector to be found by its words, so a message or a
document can be searched for an exact string as soon as it is stored.
Of a memory only the value has its words indexed, not the key.

### One hit per text

- A text several messages or documents hold is one hit: the
  best-ranked of them, with the count of the others among the kinds
  and sessions searched. Four copies of a file are one line.
- Memories are never folded: two keys saved with one value are two
  hits.
- A file or a page is one hit however often it was touched, shown at
  the first access to it among the sessions searched, with the count of
  the other sessions that touched it. Reading one file five times in a
  session adds nothing to that count.
- A long document is one hit, through its best passage, so it never
  crowds out the rest.

### Measured

Keywords are matched per chunk, not per page or per document: the
chunk that matched is then the passage to show, and BM25 compares
texts of one size. Measured on an index of 3,050 files in 151,597
chunks (this repository, Rust's books and error index, system
documentation and a novel) embedded with EmbeddingGemma 2, over 69
queries. Each cell is the share of queries whose first relevant
document came first, and within the first five:

| Queries                                                     | Vectors only | Fused        |
|-------------------------------------------------------------|--------------|--------------|
| 33 exact strings: identifiers, error messages, codes, names | 64%, 88%     | 94%, 100%    |
| 12 questions that hold one exact string                     | 75%, 100%    | 100%, 100%   |
| 24 paraphrases sharing no rare word with their answer       | 67%, 92%     | 75%, 92%     |

For an exact string, one of the first five hits showed the passage
holding it in 76% of queries by vector alone and in all of them fused.
Matching keywords per page or per document instead put the right
document first for 82% of the exact strings and showed the passage for
91%. With the fusion's usual constant of 60 in place of 10, nearly any
document found by both rankings outranks the best of either, and the
paraphrases' share within the first five fell to 83%. The keyword
query took 10 ms at the median and 58 ms for a question made of common
words, which most chunks match.

These figures are for documents. Messages, memories and accesses are
fused with the same constants, which have not been measured on them.

## Recalled context

The first prompt of a new session is also searched for, unasked, when
it is at least three characters long and `memory.enabled` is on. The
saved memories nearest it, and the other sessions holding the messages
nearest it, go into that turn's context: the memories under a header
marking them as untrusted, the sessions as ids for `search --session`.

Recall ranks by vector alone and keeps only what reaches
`embedding.min_similarity`, at most `embedding.top_k` of each. A match
by words has no similarity to hold to that floor. Documents and
accesses are never recalled; the model searches for them.

## When text can be found

| Text                | By its words                                   | By meaning                                    |
|---------------------|------------------------------------------------|-----------------------------------------------|
| A message           | Once the message chunker has cut it, right after it is saved. | Once embedded, which is not held back. |
| A memory            | Once its save has committed.                   | Once embedded, which is not held back.        |
| A file or page touched | With its tool result's message.             | Once embedded, which is not held back.        |
| A document          | Once an index pass has stored it.              | Once embedded, which by default waits until the model is not in use ([when](documents.md#when-indexing-and-embedding-run)). |

Undoing a turn removes its messages and accesses from every later
search, and deleting a memory removes it; their chunks and vectors go
with them ([database.md](database.md#what-ties-the-files-together)).
With `embedding.enabled = false` messages are saved but not cut, and
are cut at the next start with embedding on.

## What is kept of the web

The daemon keeps no copy of the web to search. Page text changes too
much to be worth caching, and a page is never stored as a document.

- **A fetched page leaves its address and title.** `web fetch` reports
  the URL the response came from, after redirects, and the page's
  title on one line of at most 200 characters. They are saved with the
  tool result as an access, and `search` finds the page by the words of
  either, with the session that fetched it.
- **The page's text stays in the transcript only.** What `web fetch`
  printed is the tool result, saved in `memory.db` like any other and
  shown again when the session is resumed. Tool results are not cut
  into chunks, so no search ranks that text. What the model wrote about
  the page in its reply is a message, and is ranked.
- **A web search leaves nothing to find.** `web search` reports no
  access: a list of results is not a visit. Its query and its results
  are in the transcript as the call and its result.
- **A fetch the daemon cannot see is not recorded.** A `bash` script
  or an MCP tool that reaches the network reports nothing.

A URL and a title are the page's own choice, so both are untrusted
text and print under the header like the rest. Undoing the turn
deletes the access. To keep what a page said, have the model write it
to a file in an indexed folder, or save it as a memory.

## Troubleshooting

**`cannot search: embedding is still starting`.** The embedding server
is not up yet. The model is told to try again later.

**`<kind> are not searched while <setting> is false`.** `--in` named a
kind that is off. Set `memory.enabled` or `documents.enabled`.

**Nothing matches, or only exact words do.** Text still waiting for a
vector is found by its words alone. `assistd memory reindex` prints
what is left to embed, and `assistd documents status` what is indexed
and waiting ([document search](documents.md#troubleshooting)).

**A file the model read is not among the accesses.** Only `cat`, `see`,
`docs read`, `write`, `web fetch` and the text commands (`grep`,
`head`, `tail`, `sort`, `uniq`, `wc`) report what they touched. A
script's reads are not known.
