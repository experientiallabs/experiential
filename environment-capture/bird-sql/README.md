# bird-sql

Text-to-SQL over real SQLite databases. The environment is a workspace holding a fresh COPY of
the task's database as `database.db` plus its DDL as `schema.sql`; the agent explores with the
`sqlite3` CLI and submits a single SQLite `SELECT`/`WITH` query as its answer. Scoring is
deterministic EXECUTION MATCH — the predicted and gold SQL are each run against a pristine
read-only copy of the database and their result rows compared as an order-insensitive multiset
(order-sensitive when the question implies ordering) — see
`environment_capture/benchmarks/bird_sql.py`.

## Contents

- `data/train.jsonl` (52 tasks) / `data/test.jsonl` (20 tasks) — agent-visible tasks
  (question + folded-in evidence hint + `db_name`). Splits are disjoint and drawn from every
  database.
- `schemas/<db>.sql` — DDL only (tables/indexes/views), staged into the workspace as `schema.sql`.
- `gold/<task_id>.json` — gold SQL (`gold_sql`), never staged into the agent workspace.
- `databases/<db>.sqlite` — the real SQLite databases (gitignored; re-materialize with
  `fetch_data.py`). The adapter and grader need these present locally.
- `traces.otel.jsonl` — the trace corpus: **519 traces / 1012 real transitions**, fresh REAL
  Bedrock captures over the **train split only** (the hidden test split is never captured so the
  world model can't absorb its dynamics); waves r1–r5 across opus-4-8/-4-7 with run-suffixed
  task ids.
- `fetch_data.py` — materializes the real upstream data into the shape above.
- `capture.py` — fresh real-run capture against this adapter (Bedrock bash/sqlite agent).

## Databases (slice)

Four databases from BIRD mini-dev, chosen for schema variety and manageable size:
`superhero`, `toxicology`, `student_club`, `california_schools`. Up to 18 questions per database,
seeded (seed 7), split ~70/30 into train/test.

## Results (2026-07-02, corpus as committed)

- **Open-loop fidelity** (suite `bird-sql/default`, seed 0, Opus 4.8 target + rubric judge, run
  via `uv run wmh eval run bird-sql/default --examples-root environment-capture`): mean fidelity
  **0.864**, error-flag accuracy **1.000**, n=21 held-out steps (re-measured post-hygiene;
  corpus unchanged, drift is judge noise). Structured sqlite output
  reconstructs far better than document-excerpt observations (financebench: 0.581).

## Provenance

- **Dataset**: BIRD **mini-dev** (v2, SQLite dialect) — 500 curated text-to-SQL instances over 11
  real end-user databases. Questions, evidence hints, gold SQL, and the SQLite databases are the
  real upstream release; all adapter/grader/materialization code here is fresh.
- **Materialization**: `fetch_data.py` converts an unzipped MINIDEV directory into this on-disk
  shape — question + evidence → `prompt`, `SQL` → `gold/*.json` sidecar, real `.sqlite` files
  copied into `databases/`, DDL dumped into `schemas/`. Task ids are `bird-{split}-{i}`.
- **Traces**: captured fresh with `capture.py` — a Bedrock bash/sqlite agent (models
  `us.anthropic.claude-opus-4-8` / `-4-7` / `-4-6-v1`) exploring each database for real and
  submitting SQL, graded by this adapter's execution-match grader. Each trace's task id is
  run-suffixed (`bird-train-3#opus48-r1`) so trace ids never collide across models/runs; the base
  task id and reward ride in the trace metadata. Observations are never synthesized.

## Getting the databases

BIRD mini-dev ships the SQLite databases only inside a single zip on the project's Google Drive
(there is no direct HTTP endpoint for the `.sqlite` files). Fetch and unzip it once, then
materialize:

```bash
pip install gdown
gdown 13VLWIwpw5E3d5DUkMvzw7hvHE67a4XkG -O minidev.zip   # BIRD mini-dev package
unzip minidev.zip
uv run python environment-capture/bird-sql/fetch_data.py --minidev-root minidev/MINIDEV
```

## License — read before redistributing

BIRD (BIg Bench for Large-scale Database Grounded Text-to-SQL) is published under
**CC BY-SA 4.0** (attribution, share-alike). The task data, gold SQL, and schemas redistributed
here, and any traces embedding database contents, are provided under the same **CC BY-SA 4.0**
terms, with attribution to the BIRD-bench authors (https://bird-bench.github.io/). Derivatives
must be shared alike.
