# Document search

With `[documents]` enabled, the daemon keeps an index of the files in
folders you choose, and the model gets two commands: `search`, which
finds a file, and `docs`, which reads what it found:

```text
search [-n N] [--in <kinds>] <query...>      # TSV hits: kind, place, similarity, passage
docs read [--page P] [--offset N] <path>     # the text stored for one indexed file
```

A search ranks files by meaning and by the query's exact words, so both
"what did the landlord say about renewing" and `INV-2024-0113` find
their file. Text files, HTML, PDFs, office files and e-books are read.
Everything runs on this machine.

## Setup

1. **Keep embedding on.** Document search needs the embedding server
   (`[embedding] enabled`, on by default).

2. **Name the folders** in `~/.config/assistd/config.toml`:

   ```toml
   [documents]
   enabled = true
   roots = ["~/Documents/notes", "~/papers"]
   ```

3. **Restart the daemon.** The first pass over the folders starts once
   the embedding server is up. Follow it with `assistd documents status`.

There is nothing else to install. Files are read by a sandboxed worker
that needs what the model's tools need (`bubblewrap`, `util-linux` and a
systemd user session), plus `chrt` and `ionice`, which `util-linux` also
provides. PDFs and office files are parsed by the worker itself, without
poppler, LibreOffice or pandoc.

The [sample config](../config/config.sample.toml) lists every
`[documents]` key, and `/settings` in the chat has them under
"Documents".

## Choosing folders

Index the folders you would ask about by content: notes, papers,
manuals, contracts, a project's documentation. Each entry of `roots` is
an absolute path or starts with `~/`.

Don't index your whole home directory. The config refuses it, along
with `/`, because no folder may hold assistd's own config, data, state
or cache directory. Narrow folders are also better for other reasons:

- **The index has a ceiling** of 500,000 chunks (a chunk is a passage
  of up to 512 characters). Once it is full, new files are skipped. As a
  scale, 3,050 files of documentation, source code and books came to
  about 150,000 chunks.
- **Everything indexed is copied** into the daemon's database (see
  [Privacy](#privacy)). That index took about 210 MB there.
- **Every chunk has to be embedded** before a search by meaning finds
  it, and by default that only happens while you are not using the
  model.
- **Noise ranks.** Downloads, build output and vendored code compete
  with the files you meant to find.
- **Each directory takes an inotify watch**, and a user has a limited
  number of them (`fs.inotify.max_user_watches`).

A few more rules:

- No folder may sit inside another one in the list.
- Symlinks are not followed. To index what a link points at, list the
  target as its own folder.
- A folder that is missing or empty when a pass reaches it is left
  alone and keeps its documents, so an unmounted drive loses nothing.
  Once it is back, run `assistd documents reindex`.
- Removing a folder from `roots` deletes its documents from the index
  at the next daemon start.

## Which files are indexed

A pass leaves these out without reading them:

- hidden files and directories (names starting with `.`),
- anything a `.gitignore` or `.ignore` at or below the folder names,
  whether or not the folder is a git repository,
- anything a `documents.exclude` pattern matches,
- symlinks, and anything that is not a regular file,
- files over `documents.max_file_mb` (20 by default),
- files the model would have to ask before reading with `cat`: anything
  under `~/.ssh` or `~/.gnupg`, `~/.netrc`, and any file in your home
  directory that no one but you may read or write (mode `600` or
  `400`). Such a file is indexed once its group or others may read it
  (mode `640` or `644`).

`exclude` takes `.gitignore`-style patterns, matched against paths
relative to each folder:

```toml
[documents]
exclude = ["drafts/**", "*.log", "node_modules/"]
```

The rest are read by format, which the file's extension decides:

| Files                              | Read as                                                         |
|------------------------------------|-----------------------------------------------------------------|
| `.html`, `.htm`, `.xhtml`          | Rendered to text, headings kept.                                |
| `.pdf`                             | Each page converted to Markdown. Scanned pages need an [OCR model](#scanned-pdfs). |
| `.docx`, `.odt`                    | One page of Markdown.                                           |
| `.pptx`, `.odp`                    | One page per slide.                                             |
| `.xlsx`, `.ods`                    | One page per sheet, cells as the workbook shows them.           |
| `.epub`                            | One page per chapter.                                           |
| Anything else                      | Text, kept as it is: plain text, Markdown, source code, CSV, …  |

A file in the last row must be UTF-8, or start with a byte-order mark
naming its encoding. Images, archives, the old binary office formats
(`.doc`, `.xls`, `.ppt`), RTF and other binary files are skipped. So are
password-protected files, except a PDF that is only restricted against
printing or copying.

At most 4 MiB of text is kept per file and the rest is cut off. A PDF
is read up to 20,000 pages. A file that takes more than two minutes to
extract is skipped.

`assistd documents status` counts what the last pass skipped, by reason:

| Reason                  | What it means                                                           |
|-------------------------|-------------------------------------------------------------------------|
| `sensitive`             | Credentials, or a file in your home that only you may read.             |
| `too large`             | Over `max_file_mb`, or an office file or e-book that expands too far.    |
| `unsupported type`      | A binary format that is not read, such as an image or an archive.        |
| `binary`                | Other binary content.                                                   |
| `not decodable as text` | Text that is not valid UTF-8.                                           |
| `password protected`    | The file needs a password to open.                                      |
| `no text layer`         | No page has any text: a scanned PDF with no OCR model set, or an empty file. |
| `unreadable`            | The daemon may not read the file or its directory.                      |
| `extraction failed`     | The file could not be parsed, or crashed the worker twice.              |
| `extraction timed out`  | Extraction took more than two minutes.                                  |
| `index full`            | The index already holds 500,000 chunks.                                 |

Hidden, ignored and excluded files and symlinks are not counted.

A file skipped for any reason from `unsupported type` down to
`extraction timed out` is remembered with its modification time and
size. Later passes count it under the same reason without reading it,
until the file changes or you run `assistd documents reindex`, which
reads every remembered file again. Run it after an upgrade that reads
a new format, or when a file timed out only because the machine was
busy.

## Scanned PDFs

A scanned page is an image with no text. By default such pages are left
empty, and a PDF with no text on any page is skipped. To have them
read, set an OCR model:

```toml
[documents.ocr]
model = "ggml-org/GLM-OCR-GGUF:Q8_0"
```

`llama-server` downloads the model the first time it is used. It must
be a vision model; one without a vision projector refuses every page.

Page transcription waits for a permit that is open only while the
daemon is **drowsy**: the main model is unloaded then, so the OCR model
takes its place on the GPU and never competes with a conversation. The
daemon drowses after
`sleep.idle_to_drowsy_mins` idle minutes (30 by default), or at once
with `assistd drowse`. Transcription stops when the daemon wakes, and
when it goes on to sleep (`sleep.idle_to_sleep_mins`, 120 by default).
With `idle_to_drowsy_mins = 0` it never drowses on its own, so nothing
is transcribed until you run `assistd drowse`.

Each page's text is stored as it is read, so waking the daemon loses at
most the page in progress. A document's transcribed pages become
searchable once its last waiting page is done. A page the model could
not read is tried again when its file changes or the daemon restarts.

`prompt` is the text sent with each page image. The default suits the
model above; another model's card gives its own.

## When indexing and embedding run

**Indexing** (reading files and storing their text) happens:

- at daemon start, once the embedding server is up: a pass over every
  folder that reads only new and changed files,
- as files change: each folder is watched, and a change is picked up
  about two seconds later,
- when you ask: `assistd documents reindex` walks every folder now,
  reads again the files earlier passes skipped, and prints what each
  pass did.

The worker runs at idle CPU and disk priority, so indexing only uses
time nothing else wants. An edited `.gitignore` takes effect at the
next whole pass, so run `reindex` after changing one.

**Embedding** (computing the vectors a search by meaning uses) follows
in the background and takes longer. Each batch of document text waits
for a permit, which by default is closed while embedding could slow
something down:

| The permit is closed while…                                  | To embed anyway, set         |
|--------------------------------------------------------------|------------------------------|
| you are talking to the model, and for `tasks.resume_after_idle_secs` (120 by default) after its last reply | `embed_during_interaction`   |
| a task is running                                            | `embed_during_tasks`         |
| the daemon is asleep because another program needs the GPU   | `embed_during_gpu_sleep`     |

The permit stays open while the daemon is drowsy or asleep from
idleness. It is read between batches of 32 chunks, so embedding stops
within one batch of your message. A file can be found by its exact
words as soon as it is indexed, before it has been embedded.

**Page transcription** waits for its own permit, open only while the
daemon is drowsy ([scanned PDFs](#scanned-pdfs)).

Both are backlogs: the work is whatever stored text still lacks its
result, so a restart loses nothing.
[architecture.md](architecture.md#who-runs-when) has the rules.

### Checking progress

The chat's status line shows `indexing documents` during a pass and
then the number of chunks left to embed. For the detail:

```text
$ assistd documents status
indexing: ready
embedding: ready
chunks: 9310 of 500000
page transcription: runs while the daemon is drowsy
pages without text: 12 waiting for transcription, 0 failed

/home/me/Documents/notes
  folder: available
  stored: 412 documents in 9310 chunks, 1240 waiting for a vector
  last pass: finished 2026-11-19 09:14 +01:00: 3 indexed, 409 unchanged, 7 skipped, 0 removed
  skipped: 4 too large, 3 unsupported type
```

`waiting for a vector` is the embedding backlog. `last pass` covers
passes since the daemon started.

## How the model uses it

While `documents.enabled` is on, the system prompt tells the model that
indexed documents can be searched, and `search` and `docs` are among
the commands behind its `run` tool. It usually searches, then reads
from the best hit:

```text
search lease renewal terms
docs read --page 3 --offset 512 "/home/me/Documents/notes/lease.pdf"
```

`search` is one search over everything stored: indexed documents and,
with `[memory]` enabled, saved memories and past sessions, ranked
together. `search --in documents <words>` holds it to documents. It
prints one line per hit, best first, for `embedding.top_k` hits (or
`-n`, at most 20). A document's line gives `document`, then the quoted
path with the page of the best passage, that passage's byte range
within the page and the date the file was indexed, then the similarity
of meaning, and the passage.
`sim=n/a` marks a passage found by its words before it was embedded.
A passage several files hold is one line, for the first of them, with
`also in N other places` ahead of the passage. A memory's line gives
its key, and a session's the short id that `search --session <id>
<words>` takes to show the hits of that one session, each between the
messages around it.

`docs read` prints the stored text of one indexed file from a page and
byte offset, as much as fits one tool result, and says how to continue.
This is how the model reads a PDF or an office file at all, since `cat`
refuses binary files. A file that is not indexed is read with `cat` as
usual.

The model reads the text as it was when the file was indexed, and each
hit carries that date. Document text reaches the model under a header
marking it as untrusted data, not instructions, as is everything
`search` prints. Neither command asks
you first: listing a folder in `roots` is the permission. Both count as
reading files on this machine, so a later `web fetch` in the same
conversation asks before it is sent ([web search](web-search.md)).

To see what the model would get for a query, run the same search
yourself:

```sh
assistd documents search lease renewal terms
```

`assistd search` runs it over past sessions and saved memories as well;
`--in` holds it to the kinds named.

## Privacy

- **The text of every indexed file is copied** into
  `~/.local/share/assistd/index.db`, with its path, and stays there
  until the file is deleted, excluded, or its folder is removed from
  `roots`. The data directory is only readable by you (`0700`, the
  database `0600`), but a backup or a copy of that file contains your
  documents. `memory.db`, beside it, holds your sessions and memories
  and none of the indexed text, and `index.db` is built again when it is
  missing, so a backup needs only `memory.db`. Passwords and keys are safer kept out of indexed folders
  than left to the `sensitive` rule.
- **Nothing leaves the machine.** The worker that parses files has no
  network, and of your home directory it sees only the indexed folders,
  read-only. Embedding and OCR are done by the daemon's own local model
  servers.
- **What the model reads becomes part of the conversation**, which is
  saved in the same database like any other tool result.
- **Turning it off keeps the index.** `enabled = false` stops indexing
  and takes `docs` away from the model, but what was stored stays. To
  delete it, keep `enabled = true`, set `roots = []`, and restart the
  daemon: every stored folder that `roots` no longer lists is removed
  with its documents.

The index is a cache of files on disk, so nothing is lost by deleting
it. [database.md](database.md#embeddings) describes the tables.

## Troubleshooting

Start with `assistd documents status`, and the daemon log
(`journalctl --user -u assistd`) for lines about document indexing.

**Config error `documents.enabled needs embedding.enabled`.**
Set `[embedding] enabled = true`, or turn `[documents]` off.

**Config error `documents.roots lists …`.** One folder sits inside
another, or a folder holds one of assistd's own directories, as `~`
does. Name narrower folders.

**`indexing: unavailable (no folder to index (documents.roots is empty))`.**
Add at least one folder to `roots`.

**`indexing: unavailable (files are read only in the tool sandbox, and tools are disabled: …)`.**
The worker has no unsandboxed mode. Fix what the message names, usually
a missing `bubblewrap`; the model has no tools until then either.

**`indexing: unavailable (cannot run `chrt` …)`** or **`ionice`.**
Install `util-linux`.

**`folder: unavailable (…)`.** The folder is missing, unreadable or
empty. Its documents are kept. Once it is back, run
`assistd documents reindex`; a folder that was missing when the daemon
started is not watched until the next restart.

**`changes: wait for the next pass (…)`.** The folder could not be
watched, most often because you are out of inotify watches. Raise
`fs.inotify.max_user_watches`, restart the daemon, and until then run
`assistd documents reindex` after changing files.

**`the index is full: files not yet stored are skipped`.** Index less:
remove a folder or add `exclude` patterns, then restart the daemon.

**A search finds nothing, or misses a file you know is there.** Check
that the file is not among the skipped ones, and that `waiting for a
vector` has reached zero: until then only exact words find a file.

**A file is found by its exact words but never by meaning, with
nothing waiting for a vector.** Some of its chunks are refused rows. A
row is refused when the embedding server rejects its text, usually as
longer than the model's input limit, or when it fails three times on
its own while the server embeds other text. The daemon log names each
with `row cannot be embedded`. A refused row is not counted as waiting
and is not tried again until you run `assistd memory reindex`, which
retries every refused row and prints what is left to embed until
nothing is. It helps after a server fault; text the model rejects is
refused again.

**`page transcription: never runs …`.** `sleep.idle_to_drowsy_mins` is
`0`. Set it, or run `assistd drowse` when you want pages read.

**`the last transcription run stopped early: …`.** The reason follows.
A model without a vision projector refuses every page; one that does
not fit on the GPU cannot be loaded. The next run starts after five
minutes, or when the daemon next drowses.
