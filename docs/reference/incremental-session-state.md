# Incremental session state

The durable session store (`exp/runtime/sessions/`) keeps each candidate world as an immutable
event log plus a content-addressed projection, not as one complete JSON document rewritten on
every commit. This page states the storage contract, the retention and collection rules, the
measured effect, and the hard cutover from the previous full-state layout.

## Storage shape

- **Nodes** (`state_nodes`): every stored object or array that is a document root, or whose
  declared JSON text is longer than 128 characters, is one node. A node's `template` is its
  exact declared text with each nested node replaced by one placeholder, and its `children` are
  the digests of those nested nodes in textual order. Scalars and short nested containers stay
  inline, so declared key order, string, number and boolean spellings, null versus missing, and
  empty containers survive byte for byte. A node's digest covers its template and its children's
  digests, so equal subtrees share one row across revisions, sessions and forks in the same file.
- **Events** (`state_events`): revision 0 records the genesis state; every later revision records
  the ordered structural edits of one operation and the resulting root digest. Events are never
  updated or deleted, and one operation commits at most once. `SessionStore.verify_history`
  rebuilds every root from the events alone, with no model, tool, clock or stored node.
- **Heads** (`sessions`): the current revision and root. A `SessionHead` carries a revision-bound
  `StateRef(store_id, session_id, revision, root_sha256)`; it holds no JSON and never triggers
  hidden reads. `read_state(ref, path)` materializes one path of one revision;
  `read_session` still materializes the whole state where a caller needs it.
- **Pins** (`state_pins`) and the **collection cursor** (`state_collection`), described below.

## Structural edits

The runtime owns four primitives (`exp/runtime/sessions/edits.py`): `set` (an empty path replaces
the document), `delete`, `append` and `copy`. One interpreter applies them to a plain document and
to a stored draft whose untouched containers remain unloaded references, so the state a commit
publishes is the state the coordinator validated. The simulation generic-delta reducer
(`exp/simulation/engines/text/generic_delta.py`) resolves its model-facing ops into these
primitives while computing the next state; `ProposedTransition.state_edits` returns both. The
world-model coordinator commits its envelope as: append the transition record, copy the business
state into `before`, apply the proposal's edits, copy the business state into `after`, append the
history entry. Both copies are structural, so they share every unchanged subtree with the state
they copy.

`complete_step(claim, edits, response)` loads only the containers the edits touch, inserts only
nodes the store does not already hold, records the event and updates the head in the one
transaction that already checks the claim token, the saved proposal, the expected revision and
the deadline. An edit that does not apply publishes nothing and leaves the step open with its
proposal intact. Replay, rejection, ambiguity, recovery and draining are unchanged.

## Connections and durability

The store reuses one connection per thread and database file through
`exp.common.sqlite.connection.persistent_connection`, bounded to sixteen idle connections per
thread and forgotten in a forked child. Every transaction still begins with `BEGIN IMMEDIATE` and
commits under `synchronous = FULL` before the call returns; reuse removes the connection setup
and the WAL checkpoint-on-close that each operation paid before, not a sync.

## Retention and collection

A session head always keeps its current revision. Older revisions stay materialized while a
**retention pin** owns them: `SessionStore.pin_revisions(owner_id, session_id, start, end)`
registers a durable named owner of the half-open range, refused when the owner already holds a
different range, when the range exceeds the head, or when a revision in it was already reclaimed.
`release_pin` is the only release; a stopped process releases nothing. Capturing a world checkpoint
pins the source session's revisions `[0, prefix + 1)` under the checkpoint's artifact identity.

`collect_garbage(batch_size, maximum_batches)` marks every node reachable from heads and pins in
one read snapshot, then deletes unmarked nodes in bounded write transactions behind a durable
cursor, so an interrupted run resumes. Node rows are numbered monotonically; rows created after
the mark are never candidates, and each batch re-checks live roots and the children of newer rows
so a commit that reuses a candidate by digest keeps it. Every read runs in one SQLite snapshot and
sees a complete revision during a sweep. Events are never reclaimed. The report counts reclaimed
rows and bytes; freed pages are reused inside the file, which does not shrink without a vacuum.

## Checkpoints and forks

A `WorldCheckpoint` artifact carries its projected prefix states, so it restores in a fresh store
without the original database. A fork opens a new session whose genesis is the checkpoint's
current state; in the same database file its nodes are shared with the source by digest, and each
sibling stores only its own changes. Parent, child and sibling edits never leak into each other.

## Measured effect

Offline benchmark (`exp/runtime/sessions/tests/state_benchmark_test.py`, deterministic worlds,
no provider): bytes the store inserts per commit versus the complete envelope a full-state store
rewrites.

| Workload | Commits | Incremental bytes | Full-state bytes | Ratio |
| --- | --- | --- | --- | --- |
| sparse leaf edits, small world (8x10 rows) | 60 | 246,513 | 19,267,550 | 78x |
| sparse leaf edits, medium world (40x25) | 60 | 367,152 | 242,557,312 | 661x |
| sparse leaf edits, large world (120x40) | 60 | 580,273 | 1,172,958,610 | 2,021x |
| dense replacement of 10% of rows, medium | 20 | 2,399,856 | 27,078,886 | 11x |
| deletes from a 2,000-element array | 30 | 684,899 | 26,085,429 | 38x |
| five forks of one large prefix, 10 edits each | 50 | 375,380 | 189,134,535 | 504x |

Per-operation latency on the same disk (`begin_step`, `mark_provider_started`,
`record_proposal`, `complete_step`) fell from 287, 468, 308 and 292 ms to 9, 8, 11 and 9 ms,
almost entirely from connection reuse: a fresh connection per operation paid open, pragma, WAL
setup and a checkpoint on close around each durable sync.

Full prompt assembly still reads the whole business state and history for the predictor; that
cost is proportional to world size and is unchanged.

## Hard cutover

Schema 4 replaces schema 3. A schema 3 database kept one complete state per session and recorded
no replayable genesis or effects, so it cannot be converted without inventing history. Opening one
is refused with a read-only check that leaves the file byte for byte untouched and an instruction
to preserve it for the release that wrote it and point the run at a fresh session database path.
There is no migration command and no dual-format reader.
