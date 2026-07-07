# bird-sql

Text-to-SQL over real SQLite databases. The environment is a workspace holding a fresh COPY of
the task's database as `database.db` plus its DDL as `schema.sql`; the agent explores with the
`sqlite3` CLI and submits a single SQLite `SELECT`/`WITH` query as its answer. Scoring is
deterministic EXECUTION MATCH — the predicted and gold SQL are each run against a pristine
read-only copy of the database and their result rows compared as an order-insensitive multiset
(order-sensitive when the question implies ordering) — see
`environment_capture/benchmarks/bird_sql.py`.

## Contents

- `data/train.jsonl` (222 tasks) / `data/test.jsonl` (20 tasks) — agent-visible tasks
  (question + folded-in evidence hint + `db_name`). Splits are disjoint and drawn from every
  database; no two tasks share an upstream `question_id`.
- `schemas/<db>.sql` — DDL only (tables/indexes/views), staged into the workspace as `schema.sql`.
- `gold/<task_id>.json` — gold SQL (`gold_sql`), never staged into the agent workspace.
- `databases/<db>.sqlite` — the real SQLite databases (gitignored; re-materialize with
  `fetch_data.py`). The adapter and grader need these present locally.
- `traces.otel.jsonl` — the trace corpus: **1993 traces / 4168 real transitions**, fresh REAL
  Bedrock captures over the **train split only** (the hidden test split is never captured so the
  world model can't absorb its dynamics); the original 4-db waves r1–r5 plus one r1 wave across
  opus-4-8/-4-7 over the 170 expansion tasks, all with run-suffixed task ids.
- `fetch_data.py` — materializes the real upstream data into the shape above; `--expand` grows the
  train split from more of the upstream pool without touching the test split (see below).
- `capture.py` — fresh real-run capture against this adapter (Bedrock bash/sqlite agent).

## Databases

All 11 databases of BIRD mini-dev, for schema variety. The original split drew from four
(`superhero`, `toxicology`, `student_club`, `california_schools`) at up to 18 questions/db, seeded
(seed 7), split ~70/30 into train/test. The train split was then **expanded** (`fetch_data.py
--expand`) to up to 22 questions/db across all 11 databases (the seven added:
`formula_1`, `card_games`, `european_football_2`, `thrombosis_prediction`, `codebase_community`,
`financial`, `debit_card_specializing`), appending 170 new train tasks. The same seed keeps the
first 18 questions/db identical to the original split, so expansion only adds new tail questions and
the test split is untouched.

## Results (2026-07-02, corpus as committed)

- **Open-loop fidelity** (Opus 4.8 target + rubric judge, seed 0; final 1993-trace / 222-task
  corpus): mean fidelity **0.944**, error-flag accuracy **0.997**, n=1258 held-out steps.
  Task-set expansion did NOT dent fidelity (0.943 on the resampled 415-trace snapshot →
  0.944 task-diverse), so the cross-run-overlap caveat (DECISIONS D33) is immaterial here:
  structured sqlite output genuinely reconstructs at ~0.94, far above document-excerpt
  observations (financebench 0.586).

## Provenance

- **Dataset**: BIRD **mini-dev** (v2, SQLite dialect) — 500 curated text-to-SQL instances over 11
  real end-user databases. Questions, evidence hints, gold SQL, and the SQLite databases are the
  real upstream release; all adapter/grader/materialization code here is fresh.
- **Materialization**: `fetch_data.py` converts an unzipped MINIDEV directory into this on-disk
  shape — question + evidence → `prompt`, `SQL` → `gold/*.json` sidecar, real `.sqlite` files
  copied into `databases/`, DDL dumped into `schemas/`. Task ids are `bird-{split}-{i}`. `--expand`
  re-reads the committed splits and appends only questions whose upstream `question_id` is not
  already present, with fresh sequential train ids counting past the last one
  (`environment_capture.plan_appended_tasks` enforces the no-duplicate / test-untouched invariant).
- **Traces**: captured fresh with `capture.py` — a Bedrock bash/sqlite agent (models
  `us.anthropic.claude-opus-4-8` / `-4-7` / `-4-6-v1`) exploring each database for real and
  submitting SQL, graded by this adapter's execution-match grader. Each trace's task id is
  run-suffixed (`bird-train-3#opus48-r1`) so trace ids never collide across models/runs; the base
  task id and reward ride in the trace metadata. Observations are never synthesized. The 170
  expansion tasks were captured in one r1 wave across opus-4-8/-4-7 (mean reward 0.624, 106/170
  solved).

## Getting the databases

BIRD mini-dev ships the SQLite databases only inside a single zip on the project's Google Drive
(there is no direct HTTP endpoint for the `.sqlite` files). Fetch and unzip it once, then
materialize:

```bash
pip install gdown
gdown 13VLWIwpw5E3d5DUkMvzw7hvHE67a4XkG -O minidev.zip   # BIRD mini-dev package
unzip minidev.zip
# base split (4 dbs, 52 train / 20 test) — reproduces the original materialization
uv run python packages/environment-capture/bird-sql/fetch_data.py --minidev-root minidev/MINIDEV
# expansion (all 11 dbs) — appends the 170 extra train tasks, test split untouched
uv run python packages/environment-capture/bird-sql/fetch_data.py --minidev-root minidev/MINIDEV --expand
```

## License — read before redistributing

BIRD (BIg Bench for Large-scale Database Grounded Text-to-SQL) is published under
**CC BY-SA 4.0** (attribution, share-alike). The task data, gold SQL, and schemas redistributed
here, and any traces embedding database contents, are provided under the same **CC BY-SA 4.0**
terms, with attribution to the BIRD-bench authors (https://bird-bench.github.io/). Derivatives
must be shared alike.
