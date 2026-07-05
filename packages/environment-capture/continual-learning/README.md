# continual-learning

Database-exploration QA over a large, deliberately obfuscated SQLite database. The environment is
a workspace holding one shared `products.db` (~400 MB of Amazon product/review data with cryptic
table/column names, prices in integer cents, timestamps in epoch milliseconds, and drifted/corrupt
values); the agent explores it with real `sqlite3`/`python3` shell commands and submits a final
answer. Scoring is deterministic and LLM-free — numeric match within the gold's absolute tolerance,
else normalized text exact-match or containment — see
`environment_capture/benchmarks/continual_learning.py`.

## Contents

- `data/train.jsonl` (25 tasks) / `data/test.jsonl` (25 tasks) — agent-visible tasks (raw
  question + `db_file`/`db_name` + difficulty; every question is upstream-labeled `hard`).
- `gold/<task_id>.json` — gold answers (`answer`, `answer_type`, `tolerance`, optional `numeric`),
  never staged into the agent workspace.
- `traces.otel.jsonl` — the trace corpus (train split only; the hidden test split is never captured
  so the world model can't absorb its dynamics): **286 traces / 2071 real transitions** (2 trajectories with host-escape content dropped whole by the hygiene audit — see `environment_capture/hygiene.py`) — 25
  converted real runs plus 25 fresh Bedrock captures, together covering every train task.
- `fetch_data.py` — fetches the shared `products.db` from the upstream HuggingFace dataset (the
  ~400 MB db is gitignored; questions + gold are committed, so a fresh clone only needs this).
- `convert_cache.py` — converts a frozen baseline cache of real runs into the corpus.
- `capture.py` — fresh real-run capture against this adapter (Bedrock agent), sharded across models.

## The 400 MB shared database

The db is far too large to copy per task, so `open_env` stages it **read-only** by symlinking it
into each task workspace as `database.db` (never a per-task copy). Read-only permissions make
concurrent capture safe — no WAL/journal writes, no cross-task corruption — and match the tasks'
read-only nature. It is fetched on demand into `datafiles/` (gitignored) by `fetch_data.py`. Tests
never touch it: the adapter also accepts a hermetic inline `schema_sql` task that builds a tiny db
in-process, which is all the offline suite uses.

## Results (2026-07-02, corpus as committed)

- **Open-loop fidelity** (suite `continual-learning/default`, seed 0, Opus 4.8 target + rubric
  judge): mean fidelity **0.894**, error-flag accuracy **0.975**, n=239 held-out steps (snapshot
  @139 traces). Snapshot evals on multi-run corpora carry a caveat: tasks are resampled across capture waves, and the whole-trace split lets a held-out step retrieve the same task's other runs — fidelity partly reflects cross-run overlap, not pure generalization (DECISIONS.md D33).

## Provenance

- **Dataset**: [Continual Learning Bench](https://continual-learning-bench.com/) (UC Berkeley Sky
  Computing Lab; arXiv 2606.05661), HuggingFace
  [`continual-learning-benchmark/continual-learning-bench-data`](https://huggingface.co/datasets/continual-learning-benchmark/continual-learning-bench-data),
  `database_exploration` subset. We import it as **independent single-shot tasks** (one question
  graded on its own), not the benchmark's native cross-episode "Gain" loop — a faithful import of
  its questions, database, and gold answers that measures single-shot DB-exploration accuracy. Task
  ids, the disjoint train/test split, and gold sidecars come from a prior materialization of the
  upstream dataset, reused as data; all adapter/grader code here is fresh. The underlying rows
  derive from the public Amazon Product Reviews dataset (Office Products, Electronics, Musical
  Instruments), obfuscated by upstream.
- **Traces**: two real sources, no synthesized observations.
  - *Converted*: 25 traces from a frozen baseline cache of real `gpt-5.4` runs over the full train
    split (mean reward ≈ 0.28 — these hard tasks leave real headroom). `convert_cache.py` re-emits
    the real commands/outputs and keeps each run's reward in metadata. The reference harness's
    submission-sentinel keyword is renamed to a neutral `SUBMIT` (a harness protocol token, not
    environment content — no query result, schema, or number is altered); conversion asserts no
    source-project reference survives.
  - *Fresh*: 25 real captures via `capture.py` (Bedrock, `us.anthropic.claude-opus-4-8` / `-4-7` /
    `-4-6-v1`, sharded to beat throttling; mean reward ≈ 0.16 — these models don't max out the hard
    tasks either), graded by this adapter's grader (documented thresholds) and carrying their own
    model id. Each fresh run gets a run-suffixed task id (`clb-train-3#opus48-r1`) so trace ids
    never collide with the converted traces or with each other.

## License — read before redistributing

Continual Learning Bench is published under **CC BY 4.0** (attribution). The questions, gold
answers, and any traces embedding query results are redistributed here **with attribution to
Continual Learning Bench** (and, transitively, the public Amazon Product Reviews dataset the rows
derive from). CC BY 4.0 permits redistribution and derivative use, including commercially, provided
attribution is preserved.
